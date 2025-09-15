import base64
import binascii
import dataclasses
import datetime as dt
import hashlib
import json
import logging
import os
import re
import tarfile
import tempfile
import time
import typing
import unicodedata
import uuid
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Dict, List, Union
from urllib.parse import unquote, urlparse

import boto3
import botocore
import requests
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from osgeo import gdal, ogr
from te_schemas import jobs, results

from te_algorithms.gdal.util import combine_all_bands_into_vrt

logger = logging.getLogger(__name__)

SDG_RAW_JOB_NAME = ("sdg-15-3-1-sub-indicators",)
SDG_SUMMARY_RAW_JOB_NAME = ("sdg-15-3-1-summary",)

# Set the desired multipart threshold value (5GB)
MB = 1024
MULTIPART_THRESHOLD = 100 * MB


def get_s3_client(access_key_id=None, secret_access_key=None, no_sign=False):
    if no_sign:
        config = Config(signature_version=botocore.UNSIGNED)
    else:
        config = Config()
    try:
        client = boto3.client(
            "s3",
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=config,
        )
    except OSError:
        logger.warning(
            "AWS credentials file not found. Credentials must be in "
            "environment variable in default AWS credentials location, "
            "or (if on the cloude) an instance role must be attached)"
        )
        client = boto3.client("s3", config=config)

    return client


def set_s3_obj_acl(client, obj, prefix, bucket, acl="public-read"):
    client.put_object_acl(
        ACL="bucket-owner-full-control", Key=f"{prefix}/{obj}", Bucket=bucket
    )
    client.copy_object(
        CopySource={"Bucket": bucket, "Key": f"{prefix}/{obj}"},
        StorageClass="STANDARD",
        Key=f"{prefix}/{obj}",
        Bucket=bucket,
    )


def put_to_s3(
    filename: Path,
    bucket: str,
    prefix: str,
    s3_extra_args={},
    aws_access_key_id=None,
    aws_secret_access_key=None,
):
    client = get_s3_client(
        access_key_id=aws_access_key_id, secret_access_key=aws_secret_access_key
    )
    try:
        up_to_date = etag_compare(
            filename, get_s3_etag(key=f"{prefix}/{filename.name}", bucket=bucket)
        )
    except botocore.exceptions.ClientError:
        # File is not there
        up_to_date = False

    if up_to_date:
        logger.info(f"Skipping upload of {filename} to s3 - already up to date")
    else:
        key = f"{prefix}/{filename.name}"
        # Get file size for progress tracking
        file_size = filename.stat().st_size
        file_size_mb = file_size / (1024 * 1024)

        logger.info(
            f"Uploading {filename} (size: {file_size_mb:.1f} MB) to s3 at {key}"
        )

        # Create a progress callback for large files using a closure to track progress
        last_logged_percent = [0]  # Use list to make it mutable in closure

        def progress_callback(bytes_transferred):
            percent = (bytes_transferred / file_size) * 100
            mb_transferred = bytes_transferred / (1024 * 1024)
            # Log progress every 10% for large files (>100MB)
            if file_size_mb > 100 and int(percent) % 10 == 0 and int(percent) > 0:
                if int(percent) > last_logged_percent[0]:
                    logger.info(
                        f"Upload progress for {filename.name}: "
                        f"{mb_transferred:.1f}/{file_size_mb:.1f} MB "
                        f"({percent:.1f}%)"
                    )
                    last_logged_percent[0] = int(percent)

        config = TransferConfig(multipart_threshold=MULTIPART_THRESHOLD)

        start_time = time.time()
        client.upload_file(
            str(filename),
            bucket,
            key,
            Config=config,
            ExtraArgs=s3_extra_args,
            Callback=progress_callback,
        )

        upload_time = time.time() - start_time
        upload_speed_mbps = file_size_mb / upload_time if upload_time > 0 else 0
        logger.info(
            f"Upload completed for {filename.name} in {upload_time:.1f} seconds "
            f"(average speed: {upload_speed_mbps:.1f} MB/s)"
        )


def get_from_s3(
    obj, out_path, bucket, prefix, aws_access_key_id=None, aws_secret_access_key=None
):
    client = get_s3_client(
        access_key_id=aws_access_key_id, secret_access_key=aws_secret_access_key
    )
    logger.info(f"Downloading {obj} from s3 to {out_path}")
    client.download_file(bucket, f"{prefix}/{obj}", out_path)


def md5_checksum(filename):
    m = hashlib.md5()
    file_size = Path(filename).stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    bytes_read = 0

    logger.debug(
        f"Computing MD5 checksum for {Path(filename).name} ({file_size_mb:.1f} MB)"
    )
    start_time = time.time()

    # Use larger read buffer for better performance on large files
    buffer_size = 1024 * 1024  # 1MB buffer
    last_logged_percent = 0  # Track last logged percentage to prevent spam

    with open(filename, "rb") as f:
        while True:
            data = f.read(buffer_size)
            if not data:
                break
            m.update(data)
            bytes_read += len(data)

            # Log progress every 25% for files larger than 50MB, but less
            # frequently for large files
            if file_size_mb > 50:
                percent = (bytes_read / file_size) * 100
                # For very large files (>1GB), log every 10% to reduce log spam
                log_interval = 10 if file_size_mb > 1024 else 25

                # Only log if we've crossed the next threshold
                if (
                    int(percent) >= last_logged_percent + log_interval
                    and int(percent) > 0
                ):
                    mb_read = bytes_read / (1024 * 1024)
                    elapsed = time.time() - start_time
                    rate_mbps = mb_read / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"MD5 checksum progress: {mb_read:.1f}/{file_size_mb:.1f} MB "
                        f"({percent:.1f}%) - {rate_mbps:.1f} MB/s"
                    )
                    last_logged_percent = int(percent)

    checksum_time = time.time() - start_time
    rate_mbps = file_size_mb / checksum_time if checksum_time > 0 else 0
    logger.debug(
        f"MD5 checksum completed in {checksum_time:.1f} seconds for "
        f"{Path(filename).name} ({rate_mbps:.1f} MB/s)"
    )
    return m.hexdigest()


def etag_checksum_multipart(filename, chunk_size=8 * 1024 * 1024):
    file_size = Path(filename).stat().st_size
    file_size_mb = file_size / (1024 * 1024)

    logger.debug(
        f"Computing multipart ETag checksum for {Path(filename).name} "
        f"({file_size_mb:.1f} MB)"
    )
    start_time = time.time()

    # For small files, use single MD5 instead of multipart
    if file_size <= chunk_size:
        logger.debug("File is small enough for single-part ETag, using optimized MD5")
        return md5_checksum(filename)

    # Estimate number of parts to optimize memory usage
    estimated_parts = (file_size + chunk_size - 1) // chunk_size
    logger.debug(f"Estimated {estimated_parts} parts for multipart ETag")

    # Use a more memory-efficient approach with streaming MD5
    final_md5 = hashlib.md5()
    bytes_read = 0
    part_count = 0
    last_logged_percent = 0  # Track last logged percentage to prevent spam

    # Read and process file in chunks
    with open(filename, "rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break

            # Compute MD5 for this chunk and add to final hash immediately
            chunk_md5 = hashlib.md5(data).digest()
            final_md5.update(chunk_md5)

            part_count += 1
            bytes_read += len(data)

            # Log progress every 25% for files larger than 50MB,
            # but less frequently for large files
            if file_size_mb > 50:
                percent = (bytes_read / file_size) * 100
                # For very large files (>1GB), log every 10% to reduce log spam
                log_interval = 10 if file_size_mb > 1024 else 25

                # Only log if we've crossed the next threshold
                if (
                    int(percent) >= last_logged_percent + log_interval
                    and int(percent) > 0
                ):
                    mb_read = bytes_read / (1024 * 1024)
                    elapsed = time.time() - start_time
                    rate_mbps = mb_read / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"ETag checksum progress: {mb_read:.1f}/{file_size_mb:.1f} MB "
                        f"({percent:.1f}%) - {rate_mbps:.1f} MB/s"
                    )
                    last_logged_percent = int(percent)

    checksum_time = time.time() - start_time
    rate_mbps = file_size_mb / checksum_time if checksum_time > 0 else 0
    logger.info(
        f"Multipart ETag checksum completed in {checksum_time:.1f} seconds for "
        f"{Path(filename).name} ({rate_mbps:.1f} MB/s)"
    )

    return f"{final_md5.hexdigest()}-{part_count}"


def etag_compare(filename, et):
    logger.debug(
        f"Comparing file checksum for {Path(filename).name} to verify "
        f"if S3 upload is needed"
    )
    try:
        if "-" in et and et == etag_checksum_multipart(filename):
            # AWS S3 multipart etags

            return True
        elif "-" not in et and et == md5_checksum(filename):
            # AWS S3 single part etags

            return True
        elif binascii.hexlify(base64.b64decode(et)).decode() == md5_checksum(filename):
            # Google cloud etas

            return True
    except (FileNotFoundError, binascii.Error):
        return False

    return False


def get_s3_etag(key, bucket, aws_access_key_id=None, aws_secret_access_key=None):
    client = get_s3_client(
        access_key_id=aws_access_key_id, secret_access_key=aws_secret_access_key
    )
    resp = client.head_object(Bucket=bucket, Key=key)

    return resp["ETag"].strip('"')


def get_etag(key, bucket, aws_access_key_id=None, aws_secret_access_key=None):
    """Gets etag for a file as a te_schemas.results.Etag instance"""
    # TODO: need to support GCS as well as S3

    et = get_s3_etag(key, bucket, aws_access_key_id, aws_secret_access_key)

    if "-" in et:
        et_type = results.EtagType.AWS_MULTIPART
    else:
        et_type = results.EtagType.AWS_MD5

    return results.Etag(hash=et, type=et_type)


def write_to_cog(in_file, out_file, nodata_value, max_processing_time_hours=6):
    logger.info(f"Converting {in_file} to COG format: {out_file}")
    start_time = time.time()
    estimated_size_mb = 0  # Initialize for later use in compression selection

    # Set timeout based on estimated processing time
    max_processing_seconds = max_processing_time_hours * 3600
    logger.info(f"Maximum processing time allowed: {max_processing_time_hours} hours")

    # Get input file info for diagnostics
    try:
        if Path(in_file).exists():
            file_size = Path(in_file).stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            logger.info(f"Input file size: {file_size_mb:.1f} MB")
        else:
            logger.info(f"Converting VRT file: {in_file}")
            # For VRT files, try to get some basic info
            try:
                ds = gdal.Open(str(in_file))
                if ds:
                    logger.info(
                        f"VRT dimensions: {ds.RasterXSize} x {ds.RasterYSize}, "
                        f"bands: {ds.RasterCount}"
                    )

                    # Estimate actual data size based on dimensions and data type
                    total_pixels = ds.RasterXSize * ds.RasterYSize * ds.RasterCount

                    # Check first band for data type
                    band = ds.GetRasterBand(1)
                    if band:
                        data_type = band.DataType
                        type_name = gdal.GetDataTypeName(data_type)
                        logger.info(f"Band 1 datatype: {type_name}")

                        # Estimate size based on data type
                        bytes_per_pixel = gdal.GetDataTypeSize(data_type) // 8
                        estimated_size_bytes = total_pixels * bytes_per_pixel
                        estimated_size_mb = estimated_size_bytes / (1024 * 1024)
                        logger.info(
                            f"Estimated data size: {estimated_size_mb:.1f} MB "
                            f"({total_pixels:,} pixels)"
                        )

                        # Check if this is an unreasonably large operation
                        if estimated_size_mb > 10000:  # 10GB
                            logger.warning(
                                f"Very large data size: {estimated_size_mb:.1f} MB - "
                                f"this may take a very long time"
                            )
                        elif estimated_size_mb > 1000:  # 1GB
                            logger.warning(
                                f"Large estimated data size: {estimated_size_mb:.1f} "
                                f"MB - expect slow processing"
                            )
                    else:
                        estimated_size_mb = 0  # Default value if band analysis fails

                    # Check VRT sources to see if they're accessible
                    for i in range(min(3, ds.RasterCount)):
                        band = ds.GetRasterBand(i + 1)
                        if band:
                            # Try to read a small sample to test accessibility
                            try:
                                sample = band.ReadAsArray(
                                    0,
                                    0,
                                    min(100, ds.RasterXSize),
                                    min(100, ds.RasterYSize),
                                )
                                if sample is not None:
                                    logger.info(
                                        f"Band {i + 1} appears accessible "
                                        f"(sample read successful)"
                                    )
                                else:
                                    logger.warning(
                                        f"Band {i + 1} returned no "
                                        f"data (sample read failed)"
                                    )
                            except Exception as e:
                                logger.warning(f"Band {i + 1} read error: {e}")

                    ds = None
                else:
                    logger.warning(f"Could not open VRT file: {in_file}")
            except Exception as e:
                logger.warning(f"Error reading VRT info: {e}")
    except Exception as e:
        logger.warning(f"Error getting file info: {e}")

    gdal.UseExceptions()

    # Create a progress callback to track COG conversion progress with timeout
    last_percent = [0]  # Use list to make it mutable in closure
    last_log_time = [time.time()]  # Track when we last logged
    timed_out = [False]  # Flag to track timeout

    def progress_callback(complete, message, cb_data):
        current_time = time.time()
        percent = int(complete * 100)

        # Check for timeout
        elapsed = current_time - start_time
        if elapsed > max_processing_seconds:
            logger.error(
                f"COG conversion timed out after {elapsed:.1f} "
                f"seconds (max: {max_processing_seconds})"
            )
            timed_out[0] = True
            return 0  # Return 0 to abort processing

        should_log = False

        if percent >= last_percent[0] + 10 and percent > 0:
            # Standard 10% logging
            should_log = True
        elif elapsed > 300 and percent >= last_percent[0] + 2 and percent > 0:
            # After 5 minutes, log every 2% to track slow progress
            should_log = True
        elif current_time - last_log_time[0] > 300:
            # Every 5 minutes regardless for very slow operations
            should_log = True

        if should_log:
            rate = percent / elapsed * 60 if elapsed > 0 else 0  # percent per minute
            estimated_total_time = elapsed / (percent / 100) if percent > 0 else 0
            remaining_time = estimated_total_time - elapsed if percent > 0 else 0

            logger.info(
                f"COG conversion progress: {percent}% complete "
                f"(elapsed: {elapsed:.1f}s, "
                f"rate: {rate:.2f}%/min, est. remaining: {remaining_time:.0f}s)"
            )
            if message:
                logger.info(f"GDAL message: {message}")
            last_percent[0] = percent
            last_log_time[0] = current_time
        return 1  # Return 1 to continue, 0 to abort

    logger.debug("Starting GDAL Translate operation...")

    # Advanced GDAL settings for maximum performance
    gdal.SetConfigOption("GDAL_CACHEMAX", "2048")  # Increased to 2GB cache
    gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
    gdal.SetConfigOption("VSI_CACHE", "TRUE")
    gdal.SetConfigOption("VSI_CACHE_SIZE", "100000000")  # 100MB VSI cache

    # Additional performance optimizations
    gdal.SetConfigOption(
        "GDAL_SWATH_SIZE", "100000000"
    )  # 100MB swath for large datasets
    gdal.SetConfigOption("GDAL_MAX_DATASET_POOL_SIZE", "1000")  # Increase dataset pool
    gdal.SetConfigOption(
        "GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR"
    )  # Skip unnecessary directory reads
    gdal.SetConfigOption(
        "CPL_VSIL_CURL_CACHE_SIZE", "200000000"
    )  # 200MB for remote files
    gdal.SetConfigOption("GDAL_HTTP_TIMEOUT", "300")  # 5 minutes for HTTP requests
    gdal.SetConfigOption(
        "GDAL_HTTP_CONNECTTIMEOUT", "60"
    )  # 1 minute connection timeout

    # Memory optimization for large datasets
    gdal.SetConfigOption(
        "GDAL_FORCE_CACHING", "NO"
    )  # Disable caching for very large files
    gdal.SetConfigOption("GDAL_ONE_BIG_READ", "YES")  # Read entire blocks at once

    # Optimize for specific data types and sizes
    if estimated_size_mb > 5000:  # For very large datasets (>5GB)
        logger.info("Applying optimizations for very large dataset")
        gdal.SetConfigOption("GDAL_CACHEMAX", "4096")  # 4GB cache for huge datasets
        gdal.SetConfigOption("GDAL_SWATH_SIZE", "200000000")  # 200MB swath
        block_size = "1024"  # Larger blocks for better I/O
        overview_count = "3"  # Fewer overviews to speed up processing
    elif estimated_size_mb > 1000:  # For large datasets (>1GB)
        logger.info("Applying optimizations for large dataset")
        block_size = "512"
        overview_count = "4"
    else:
        block_size = "512"
        overview_count = "5"

    creation_options = [
        "BIGTIFF=YES",
        "NUM_THREADS=ALL_CPUS",
        f"BLOCKSIZE={block_size}",
        "COMPRESS=LZW",
        "PREDICTOR=2",
        "OVERVIEW_RESAMPLING=NEAREST",
        f"OVERVIEW_COUNT={overview_count}",
        "SPARSE_OK=TRUE",  # Allow sparse files to save space
    ]

    # Add quality vs speed trade-offs for different dataset sizes
    if estimated_size_mb > 5000:
        creation_options.extend(["OVERVIEW_QUALITY=1"])
        # Set overview block size as config option instead
        gdal.SetConfigOption("GDAL_TIFF_OVR_BLOCKSIZE", "1024")

    logger.info(f"Using creation options: {creation_options}")

    try:
        gdal.Translate(
            out_file,
            in_file,
            format="COG",
            creationOptions=creation_options,
            noData=nodata_value,
            callback=progress_callback,
        )

        if timed_out[0]:
            raise TimeoutError(
                f"COG conversion timed out after {max_processing_seconds} seconds"
            )

        conversion_time = time.time() - start_time
        logger.info(
            f"COG conversion completed in {conversion_time:.1f} seconds for {out_file}"
        )

        # Check output file size
        try:
            if Path(out_file).exists():
                out_size = Path(out_file).stat().st_size
                out_size_mb = out_size / (1024 * 1024)
                logger.debug(f"Output file size: {out_size_mb:.1f} MB")
            else:
                logger.error(f"Output file was not created: {out_file}")
        except Exception as e:
            logger.warning(f"Could not check output file size: {e}")

    except Exception as e:
        conversion_time = time.time() - start_time
        if timed_out[0]:
            logger.error(
                f"COG conversion timed out after {max_processing_seconds} seconds: {e}"
            )
            raise TimeoutError(
                f"COG conversion timed out after {max_processing_seconds} seconds"
            )
        else:
            logger.error(
                f"COG conversion failed after {conversion_time:.1f} seconds: {e}"
            )
        raise
    finally:
        # Reset GDAL config options to avoid affecting other operations
        gdal.SetConfigOption("GDAL_CACHEMAX", None)
        gdal.SetConfigOption("GDAL_SWATH_SIZE", None)
        gdal.SetConfigOption("GDAL_MAX_DATASET_POOL_SIZE", None)
        gdal.SetConfigOption("CPL_VSIL_CURL_CACHE_SIZE", None)
        gdal.SetConfigOption("GDAL_TIFF_OVR_BLOCKSIZE", None)


def write_to_cog_chunked(
    in_file, out_file, nodata_value, chunk_size_mb=1000, max_processing_time_hours=8
):
    """
    Alternative COG conversion that processes data in chunks for very large datasets.
    This can be faster for datasets larger than available RAM.
    """
    logger.info(
        f"Converting {in_file} to COG format using chunked processing: {out_file}"
    )

    try:
        # Open input dataset to get dimensions
        ds = gdal.Open(str(in_file))
        if not ds:
            raise Exception(f"Could not open input file: {in_file}")

        width = ds.RasterXSize
        height = ds.RasterYSize
        bands = ds.RasterCount

        # Calculate optimal chunk size based on available memory
        band = ds.GetRasterBand(1)
        data_type = band.DataType
        bytes_per_pixel = gdal.GetDataTypeSize(data_type) // 8

        # Calculate chunk dimensions to fit within memory limit
        pixels_per_chunk = (chunk_size_mb * 1024 * 1024) // (bytes_per_pixel * bands)
        chunk_height = min(height, int((pixels_per_chunk / width) ** 0.5))
        chunk_width = min(width, pixels_per_chunk // chunk_height)

        logger.info(
            f"Processing {width}x{height} image in "
            f"chunks of {chunk_width}x{chunk_height}"
        )

        # For very large datasets, consider tiling approach
        if width * height * bands * bytes_per_pixel > (
            chunk_size_mb * 1024 * 1024 * 10
        ):
            logger.warning(
                "Dataset is extremely large - consider"
                "using write_to_cog_tiled() instead"
            )

        # Use standard conversion but with optimized settings for chunked processing
        gdal.SetConfigOption("GDAL_CACHEMAX", str(chunk_size_mb))
        gdal.SetConfigOption("GDAL_SWATH_SIZE", str(chunk_size_mb * 1024 * 1024))

        return write_to_cog(in_file, out_file, nodata_value, max_processing_time_hours)

    except Exception as e:
        logger.error(f"Chunked COG conversion failed: {e}")
        raise
    finally:
        gdal.SetConfigOption("GDAL_CACHEMAX", None)
        gdal.SetConfigOption("GDAL_SWATH_SIZE", None)


def write_to_cog_tiled(
    in_file, out_file, nodata_value, tile_size=10000, max_processing_time_hours=12
):
    """
    Alternative COG conversion that splits very large datasets into tiles first.
    Best for datasets that are too large to process as a single file.
    """
    logger.info(f"Converting {in_file} to tiled COG format: {out_file}")

    try:
        ds = gdal.Open(str(in_file))
        if not ds:
            raise Exception(f"Could not open input file: {in_file}")

        width = ds.RasterXSize
        height = ds.RasterYSize

        # If dataset is small enough, use standard conversion
        if width <= tile_size and height <= tile_size:
            logger.info("Dataset is small enough for standard conversion")
            return write_to_cog(
                in_file, out_file, nodata_value, max_processing_time_hours
            )

        # Calculate number of tiles needed
        tiles_x = (width + tile_size - 1) // tile_size
        tiles_y = (height + tile_size - 1) // tile_size
        total_tiles = tiles_x * tiles_y

        logger.info(
            f"Splitting {width}x{height} dataset into "
            f"{total_tiles} tiles ({tiles_x}x{tiles_y})"
        )

        if total_tiles > 100:
            logger.warning(
                f"Very large number of tiles ({total_tiles}) - this may be slow"
            )

        # Create temporary directory for tiles
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            tile_files = []

            # Process each tile
            for y in range(tiles_y):
                for x in range(tiles_x):
                    tile_num = y * tiles_x + x + 1
                    logger.info(f"Processing tile {tile_num}/{total_tiles}")

                    # Calculate tile bounds
                    x_off = x * tile_size
                    y_off = y * tile_size
                    x_size = min(tile_size, width - x_off)
                    y_size = min(tile_size, height - y_off)

                    # Extract tile as a temporary file using gdal.Translate
                    tile_tif = temp_path / f"tile_{x}_{y}.tif"
                    gdal.Translate(
                        str(tile_tif),
                        str(in_file),
                        srcWin=[x_off, y_off, x_size, y_size],
                    )

                    # Convert tile to COG (in-place or to another file if needed)
                    cog_tile = temp_path / f"cog_tile_{x}_{y}.tif"
                    write_to_cog(
                        str(tile_tif),
                        str(cog_tile),
                        nodata_value,
                        max_processing_time_hours / total_tiles,
                    )
                    tile_files.append(str(cog_tile))

            # Build final VRT from all tiles
            logger.info("Combining tiles into final COG")
            final_vrt = temp_path / "combined.vrt"
            gdal.BuildVRT(str(final_vrt), tile_files)

            # Convert final VRT to COG
            write_to_cog(
                str(final_vrt), out_file, nodata_value, max_processing_time_hours
            )

        logger.info("Tiled COG conversion completed successfully")

    except Exception as e:
        logger.error(f"Tiled COG conversion failed: {e}")
        raise


def choose_optimal_cog_conversion(
    in_file, out_file, nodata_value, max_processing_time_hours=6
):
    """
    Automatically choose the best COG conversion strategy.

    Args:
        in_file: Input file path
        out_file: Output COG file path
        nodata_value: NoData value
        max_processing_time_hours: Maximum time to allow for processing

    Returns:
        Path to the created COG file
    """
    try:
        # Analyze dataset to choose optimal strategy
        file_size_mb = 0

        # Check if this is a VRT file first (regardless of whether it exists)
        if str(in_file).lower().endswith(".vrt"):
            logger.info(f"Converting VRT file: {in_file}")
            try:
                ds = gdal.Open(str(in_file))
                if ds:
                    logger.info(
                        f"VRT dimensions: {ds.RasterXSize} x {ds.RasterYSize}, "
                        f"bands: {ds.RasterCount}"
                    )
                    total_pixels = ds.RasterXSize * ds.RasterYSize * ds.RasterCount
                    band = ds.GetRasterBand(1)
                    if band:
                        data_type = band.DataType
                        type_name = gdal.GetDataTypeName(data_type)
                        logger.info(f"Band 1 datatype: {type_name}")
                        bytes_per_pixel = gdal.GetDataTypeSize(data_type) // 8
                        estimated_size_bytes = total_pixels * bytes_per_pixel
                        file_size_mb = estimated_size_bytes / (1024 * 1024)
                        logger.info(
                            f"Estimated uncompressed data size: {file_size_mb:.1f} MB "
                            f"({total_pixels:,} pixels)"
                        )
                        if file_size_mb > 10000:
                            logger.warning(
                                f"Very large estimated data size: "
                                f"{file_size_mb:.1f} MB - this may take a very "
                                f"long time"
                            )
                        elif file_size_mb > 1000:
                            logger.warning(
                                f"Large estimated data size: "
                                f"{file_size_mb:.1f} MB - expect slow processing"
                            )
                    else:
                        file_size_mb = 0
                        logger.warning(
                            "Could not determine band data type for VRT size "
                            "estimation."
                        )
                    # Optionally, check VRT sources for accessibility
                    for i in range(min(3, ds.RasterCount)):
                        band = ds.GetRasterBand(i + 1)
                        if band:
                            try:
                                sample = band.ReadAsArray(
                                    0,
                                    0,
                                    min(100, ds.RasterXSize),
                                    min(100, ds.RasterYSize),
                                )
                                if sample is not None:
                                    logger.info(
                                        f"Band {i + 1} appears accessible "
                                        f"(sample read successful)"
                                    )
                                else:
                                    logger.warning(
                                        f"Band {i + 1} returned no data "
                                        f"(sample read failed)"
                                    )
                            except Exception as e:
                                logger.warning(f"Band {i + 1} read error: {e}")
                    ds = None
                else:
                    file_size_mb = 0
                    logger.warning(f"Could not open VRT file: {in_file}")
            except Exception as e:
                file_size_mb = 0
                logger.warning(f"Error reading VRT info: {e}")
        else:
            # Handle regular files (non-VRT)
            if Path(in_file).exists():
                file_size_mb = Path(in_file).stat().st_size / (1024 * 1024)
                logger.info(f"Input file size: {file_size_mb:.1f} MB")
            else:
                file_size_mb = 0
                logger.warning(f"Input file does not exist: {in_file}")

        logger.info(f"Estimated dataset size: {file_size_mb:.1f} MB")

        # Choose strategy based on size and complexity
        if file_size_mb < 500:  # Small files - standard conversion
            logger.info("Using standard COG conversion for small dataset")
            return write_to_cog(
                in_file, out_file, nodata_value, max_processing_time_hours
            )

        elif file_size_mb < 5000:  # Medium files - chunked conversion
            logger.info("Using chunked COG conversion for medium dataset")
            return write_to_cog_chunked(
                in_file,
                out_file,
                nodata_value,
                chunk_size_mb=min(1000, int(file_size_mb // 4)),
                max_processing_time_hours=max_processing_time_hours,
            )

        else:  # Large files - tiled conversion
            logger.info("Using tiled COG conversion for large dataset")
            tile_size = 8000 if file_size_mb > 20000 else 10000
            timeout = max(max_processing_time_hours, 12)
            return write_to_cog_tiled(
                in_file,
                out_file,
                nodata_value,
                tile_size=tile_size,
                max_processing_time_hours=timeout,
            )

    except Exception as e:
        logger.error(
            f"Optimal COG conversion failed, falling back to standard conversion: {e}"
        )
        return write_to_cog(
            in_file, out_file, nodata_value, max_processing_time_hours * 2
        )


def push_cog_to_s3(local_path, obj, s3_prefix, s3_bucket, s3_extra_args={}):
    try:
        cog_up_to_date = etag_compare(
            local_path, get_s3_etag(key=f"{s3_prefix}/{obj}", bucket=s3_bucket)
        )
    except botocore.exceptions.ClientError:
        cog_up_to_date = False

    if cog_up_to_date:
        logger.info(f"COG at {local_path} already up to date on S3")
    else:
        put_to_s3(
            local_path, bucket=s3_bucket, prefix=s3_prefix, s3_extra_args=s3_extra_args
        )


def get_job_json_from_s3(
    s3_prefix,
    s3_bucket,
    aws_access_key_id=None,
    aws_secret_access_key=None,
    substr_regexs=None,
):
    # Returns most recent job JSON from s3
    client = get_s3_client(
        access_key_id=aws_access_key_id, secret_access_key=aws_secret_access_key
    )
    objects = client.list_objects_v2(
        Bucket=s3_bucket,
        Prefix=s3_prefix,
    )
    logger.debug(f"Looking in bucket {s3_bucket} for key prefix: {s3_prefix}")

    if objects.get("Contents", []) == []:
        logger.exception(f"No objects found for {s3_prefix}")

        return None
    else:
        objects = objects["Contents"]
        logger.debug(f"len(objects) = {len(objects)}")
        # Want most recent key
        objects.sort(key=lambda o: o["LastModified"], reverse=True)
        # Only want JSON files
        objects = [o for o in objects if bool(re.search(".json$", o["Key"]))]
        logger.debug(f"found objects pre-grep: {[o['Key'] for o in objects]}")

        if substr_regexs:
            for substr_regex in substr_regexs:
                logger.debug(f"grepping for {substr_regex}")
                objects = [
                    o for o in objects if bool(re.search(substr_regex, o["Key"]))
                ]
            logger.debug(f"found objects post-grep: {[o['Key'] for o in objects]}")
            if objects == []:
                return None
    jobfile = tempfile.NamedTemporaryFile(suffix=".json").name
    client.download_file(s3_bucket, objects[0]["Key"], jobfile)
    with open(jobfile) as f:
        job_json = json.load(f)

    return jobs.Job.Schema().load(job_json)


def _write_subregion_cogs(
    data_path,
    cog_vsi_base,
    bounding_boxes,
    temp_dir,
    s3_prefix,
    s3_bucket,
    nodata_value,
    s3_extra_args,
    suffix="",
):
    cog_vsi_paths = []
    total_regions = len(bounding_boxes)

    logger.info(
        f"Converting {total_regions} subregion(s) to COG format and uploading to S3"
    )

    for index, bounding_box in enumerate(bounding_boxes):
        region_num = index + 1
        logger.info(f"Processing subregion {region_num}/{total_regions}")

        this_suffix = (
            suffix + f"{('_' + str(index)) if len(bounding_boxes) > 1 else ''}"
        )

        temp_vrt_local_path = Path(temp_dir) / (
            cog_vsi_base.name + f"{this_suffix}.vrt"
        )
        gdal.BuildVRT(
            str(temp_vrt_local_path), str(data_path), outputBounds=bounding_box
        )

        cog_local_path = Path(temp_dir) / (cog_vsi_base.name + f"{this_suffix}.tif")
        cog_vsi_path = Path(str(cog_vsi_base) + f"{this_suffix}.tif")
        cog_vsi_paths.append(cog_vsi_path)
        choose_optimal_cog_conversion(
            str(temp_vrt_local_path), str(cog_local_path), nodata_value
        )
        push_cog_to_s3(
            cog_local_path, cog_vsi_path.name, s3_prefix, s3_bucket, s3_extra_args
        )

        logger.info(f"Completed subregion {region_num}/{total_regions}")

    logger.info(f"All {total_regions} subregion(s) processed successfully")
    return cog_vsi_paths


def get_vsis3_path(filename, s3_prefix, s3_bucket):
    return Path(f"/vsis3/{s3_bucket}/{s3_prefix}/{filename}")


def get_s3_url(filename, s3_prefix, s3_bucket, s3_region="us-east-1"):
    if s3_region == "us-east-1":
        subdomain = "s3"
    else:
        subdomain = f"s3-{s3_region}"
    s3_url_base = f"https://{subdomain}.amazonaws.com/{s3_bucket}"

    return f"{s3_url_base}/{s3_prefix}/{filename}"


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]

    return text  # or whatever


def write_output_to_s3_cog(
    data_path,
    aoi,
    filename_base,
    s3_prefix,
    s3_bucket,
    s3_region,
    aws_access_key_id=None,
    aws_secret_access_key=None,
    extra_suffix="",
    nodata_value=-32768,
    s3_extra_args={},
):
    bounding_boxes = aoi.get_aligned_output_bounds(str(data_path))
    gdal.SetConfigOption(
        "AWS_ACCESS_KEY_ID",
        aws_access_key_id,
    )
    gdal.SetConfigOption("AWS_SECRET_ACCESS_KEY", aws_secret_access_key)

    cog_vsi_base = get_vsis3_path(filename_base, s3_prefix, s3_bucket)
    with tempfile.TemporaryDirectory() as temp_dir:
        cog_vsi_paths = _write_subregion_cogs(
            data_path,
            cog_vsi_base,
            bounding_boxes,
            temp_dir,
            s3_prefix,
            s3_bucket,
            nodata_value,
            s3_extra_args,
        )

        if len(bounding_boxes) > 1:
            vrt_file = Path(temp_dir) / (cog_vsi_base.name + ".vrt")
            gdal.BuildVRT(
                str(vrt_file), [str(cog_vsi_path) for cog_vsi_path in cog_vsi_paths]
            )
            put_to_s3(
                vrt_file,
                bucket=s3_bucket,
                prefix=s3_prefix,
                s3_extra_args=s3_extra_args,
            )
            data_path = cog_vsi_base.with_suffix(".vrt")
        else:
            data_path = cog_vsi_base.with_suffix(".tif")

        # Need to use path-style addressing to avoid SSL errors due to the dot
        # in the (trends.earth) bucket name
        # s3_url_base = f'https://{s3_bucket}.s3.{s3_region}.amazonaws.com'

        if s3_region == "us-east-1":
            subdomain = "s3"
        else:
            subdomain = f"s3-{s3_region}"
        s3_url_base = f"https://{subdomain}.amazonaws.com/{s3_bucket}"
        keys = [
            remove_prefix(str(cog_vsi_path), f"/vsis3/{s3_bucket}/")
            for cog_vsi_path in cog_vsi_paths
        ]
        urls = [
            jobs.JobUrl(
                url=s3_url_base + "/" + k, md5_hash=get_s3_etag(k, bucket=s3_bucket)
            )
            for k in keys
        ]

    return data_path, urls


def _get_main_raster_uri(raster, aws_access_key_id=None, aws_secret_access_key=None):
    if raster.type == results.RasterType.TILED_RASTER:
        vrt_file = tempfile.NamedTemporaryFile(suffix=".vrt").name
        _get_raster_vrt(
            tiles=[str(uri.uri) for uri in raster.tile_uris],
            out_file=vrt_file,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        raster.uri = results.URI(uri=vrt_file)
    elif raster.type == results.RasterType.ONE_FILE_RASTER:
        pass
    else:
        raise Exception(f"Unknown raster type {raster.type}")

    return raster.uri


def write_results_to_s3_cog(
    res,
    aoi,
    filename_base,
    s3_prefix,
    s3_bucket,
    aws_access_key_id=None,
    aws_secret_access_key=None,
    nodata_value=-32768,
    s3_extra_args={},
):
    bounding_boxes = aoi.get_aligned_output_bounds(str(res.uri.uri))

    gdal.SetConfigOption(
        "AWS_ACCESS_KEY_ID",
        aws_access_key_id,
    )
    gdal.SetConfigOption("AWS_SECRET_ACCESS_KEY", aws_secret_access_key)
    gdal.UseExceptions()
    cog_vsi_base = get_vsis3_path(filename_base, s3_prefix, s3_bucket)
    with tempfile.TemporaryDirectory() as temp_dir:
        for key, raster in res.rasters.items():
            # Fill in  main uri field (linking to all rasters making up this
            # raster

            main_uri = _get_main_raster_uri(
                raster,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )

            suffix = f"_{key}"
            cog_vsi_paths = _write_subregion_cogs(
                main_uri.uri,
                cog_vsi_base,
                bounding_boxes,
                temp_dir,
                s3_prefix,
                s3_bucket,
                nodata_value,
                s3_extra_args,
                suffix,
            )

            if len(bounding_boxes) > 1:
                vrt_file = Path(temp_dir) / (cog_vsi_base.name + suffix + ".vrt")
                vrt_file_vsi_path = f"/vsis3/{s3_bucket}/{s3_prefix}/{vrt_file.name}"
                gdal.BuildVRT(
                    str(vrt_file), [str(cog_vsi_path) for cog_vsi_path in cog_vsi_paths]
                )
                put_to_s3(
                    vrt_file,
                    bucket=s3_bucket,
                    prefix=s3_prefix,
                    s3_extra_args=s3_extra_args,
                )
                raster.uri = results.URI(
                    uri=get_vsis3_path(vrt_file, s3_prefix, s3_bucket),
                )
            else:
                raster.uri = results.URI(
                    uri=get_vsis3_path(
                        cog_vsi_base.with_suffix(".tif").name, s3_prefix, s3_bucket
                    )
                )

            # All COG outputs are single tile rasters UNLESS there are multiple
            # polygons within the AOI. So write everything as Raster regardless
            # of what came in, unless there are multiple regions in which case
            # need a TiledRaster

            out_uris = [
                results.URI(
                    uri=cog_vsi_path,
                    etag=get_etag(
                        f"{s3_prefix}/{cog_vsi_path.name}",
                        s3_bucket,
                        aws_access_key_id,
                        aws_secret_access_key,
                    ),
                )
                for cog_vsi_path in cog_vsi_paths
            ]

            if len(out_uris) > 1:
                logger.debug(f"out_uris are: {out_uris}")
                # Write a TiledRaster
                res.rasters[key] = results.TiledRaster(
                    tile_uris=out_uris,
                    bands=raster.bands,
                    datatype=raster.datatype,
                    filetype=raster.filetype,
                    uri=results.URI(uri=vrt_file_vsi_path),
                )

            else:
                # Write a Raster
                res.rasters[key] = results.Raster(
                    uri=out_uris[0],
                    bands=raster.bands,
                    datatype=raster.datatype,
                    filetype=raster.filetype,
                )

        if len(res.rasters) > 1 or res.has_tiled_raster():
            main_vrt_file = Path(temp_dir) / f"{filename_base}.vrt"
            logger.info("Saving main vrt file to %s", main_vrt_file)
            main_uris = [uri.uri for uri in res.get_main_uris()]
            combine_all_bands_into_vrt(
                main_uris,
                main_vrt_file,
                is_relative=False,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )
            put_to_s3(
                main_vrt_file,
                bucket=s3_bucket,
                prefix=s3_prefix,
                s3_extra_args=s3_extra_args,
            )
            res.uri = results.URI(
                uri=get_vsis3_path(main_vrt_file.name, s3_prefix, s3_bucket),
                etag=get_etag(
                    f"{s3_prefix}/{main_vrt_file.name}",
                    s3_bucket,
                    aws_access_key_id,
                    aws_secret_access_key,
                ),
            )
        else:
            res.uri = [*res.rasters.values()][0].uri
            res.uri.type = "cloud"
    return res


def write_job_to_s3_cog(
    job,
    aoi,
    filename_base,
    s3_prefix,
    s3_bucket,
    aws_access_key_id=None,
    aws_secret_access_key=None,
    nodata_value=-32768,
    s3_extra_args={},
):
    gdal.UseExceptions()
    write_results_to_s3_cog(
        job.results,
        aoi,
        filename_base,
        s3_prefix,
        s3_bucket,
        aws_access_key_id,
        aws_secret_access_key,
        nodata_value,
        s3_extra_args={},
    )
    job.status = jobs.JobStatus.DOWNLOADED

    write_job_json_to_s3(
        job, filename_base + ".json", s3_prefix, s3_bucket, s3_extra_args
    )

    return job


def write_job_json_to_s3(job, filename, s3_prefix, s3_bucket, s3_extra_args=None):
    with tempfile.TemporaryDirectory() as temp_dir:
        job_file_path = Path(temp_dir) / filename
        with job_file_path.open(mode="w", encoding="utf-8") as fh:
            raw_job = jobs.Job.Schema().dump(job)
            json.dump(raw_job, fh, indent=2)

        put_to_s3(
            job_file_path,
            bucket=s3_bucket,
            prefix=s3_prefix,
            s3_extra_args=s3_extra_args,
        )


def slugify(value: Union[int, float, complex, str], allow_unicode=False):
    """
    Create an ASCII or Unicode slug

    Taken from https://github.com/django/django/blob/master/django/utils/text.py.
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)

    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())

    return re.sub(r"[-\s]+", "-", value).strip("-_")


def _get_download_size(url):
    resp = requests.get(url, stream=True)
    total_size = int(resp.headers["Content-length"])

    if total_size < 1e5:
        total_size_pretty = "{:.2f} KB".format(round(total_size / 1024, 2))
    else:
        total_size_pretty = "{:.2f} MB".format(round(total_size * 1e-6, 2))

    return total_size_pretty


def _download_file(url, out_file):
    download_size = _get_download_size(url)
    logger.info(f"Downloading {url} ({download_size}) to {out_file}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    return out_file


def _get_raster_tile(
    uri: results.URI, out_file: Path, aws_access_key_id=None, aws_secret_access_key=None
) -> typing.Optional[Path]:
    path_exists = out_file.is_file()
    if uri.etag:
        hash_matches = etag_compare(out_file, uri.etag.hash)
    else:
        hash_matches = False

    if path_exists and hash_matches:
        logger.debug(f"No download necessary, result already present in {out_file}")
        result = out_file
    else:
        if "vsis3" not in str(uri.uri) and "amazonaws.com" not in str(uri.uri):
            try:
                _download_file(uri.uri, out_file)
            except requests.exceptions.HTTPError:
                if "https://www.googleapis.com/download/storage" in str(uri.uri):
                    # If the download fails, try stripping the generation
                    # parameter from the URL, which is sometimes present in
                    # Google Cloud Storage URLs.
                    uri_stripped = re.sub(
                        r"([&?])generation=\d+(&|$)",
                        lambda m: "?"
                        if m.group(1) == "?" and m.group(2) == "&"
                        else m.group(2),
                        str(uri.uri),
                    )
                    _download_file(uri_stripped, out_file)
                else:
                    raise
        else:
            try:
                # Below will fail if uri is not a URL but a path
                uri_path = PurePosixPath(unquote(urlparse(uri.uri).path))
                obj = uri_path.parts[-1]
                bucket = uri_path.parts[1]
                prefix = PurePosixPath(*uri_path.parts[2:-1])
            except AttributeError:
                obj = uri.uri.parts[-1]
                # Below will ignore the "vsis3" portion of the path (the first
                # component)
                bucket = uri.uri.parts[2]
                prefix = PurePosixPath(*uri.uri.parts[3:-1])
            logger.debug("bucket is %s", bucket)
            logger.debug("prefix is %s", prefix)
            logger.debug("obj is %s", obj)
            # Use get_from_s3 in case file is not public and need to use a key
            # to access
            get_from_s3(
                obj=obj,
                out_path=str(out_file),
                bucket=bucket,
                prefix=prefix,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )

        downloaded_hash_matches = etag_compare(out_file, uri.etag.hash)
        result = out_file if downloaded_hash_matches else None

    return result


def _get_raster_vrt(
    tiles: List[Path],
    out_file: Path,
    aws_access_key_id: str,
    aws_secret_access_key: str,
):
    gdal.UseExceptions()
    gdal.SetConfigOption("AWS_ACCESS_KEY_ID", aws_access_key_id)
    gdal.SetConfigOption("AWS_SECRET_ACCESS_KEY", aws_secret_access_key)
    gdal.BuildVRT(str(out_file), [str(tile) for tile in tiles])


def _get_job_basename(job: jobs.Job):
    separator = "_"
    name_fragments = []
    task_name = slugify(job.task_name)

    if task_name != "":
        name_fragments.append(task_name)

    if job.script:
        name_fragments.append(slugify(job.script.name))
    name_fragments.extend([slugify(job.local_context.area_of_interest_name)])

    return separator.join(name_fragments)


def download_cloud_results(
    job: jobs.Job,
    download_path: Path,
    aws_access_key_id: Union[str, None] = None,
    aws_secret_access_key: Union[str, None] = None,
) -> typing.Optional[Path]:
    base_output_path = download_path / f"{_get_job_basename(job)}"

    out_rasters = []

    for key, raster in job.results.rasters.items():
        file_out_base = f"{base_output_path.name}_{key}"

        if raster.type == results.RasterType.TILED_RASTER:
            tile_uris = []

            for uri_number, uri in enumerate(raster.tile_uris):
                out_file = base_output_path.parent / f"{file_out_base}_{uri_number}.tif"
                _get_raster_tile(
                    uri=uri,
                    out_file=base_output_path.parent
                    / f"{file_out_base}_{uri_number}.tif",
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                )
                tile_uris.append(results.URI(uri=out_file))

            raster.tile_uris = tile_uris

            vrt_file = base_output_path.parent / f"{file_out_base}.vrt"
            logger.debug("Saving vrt file to %s", vrt_file)
            _get_raster_vrt(
                tiles=[str(uri.uri) for uri in tile_uris],
                out_file=vrt_file,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )
            out_rasters.append(
                results.TiledRaster(
                    tile_uris=tile_uris,
                    bands=raster.bands,
                    datatype=raster.datatype,
                    filetype=raster.filetype,
                    uri=results.URI(uri=vrt_file),
                    type=results.RasterType.TILED_RASTER,
                )
            )
        else:
            out_file = base_output_path.parent / f"{file_out_base}.tif"
            _get_raster_tile(
                uri=raster.uri,
                out_file=out_file,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )
            raster_uri = results.URI(uri=out_file)
            raster.uri = raster_uri
            out_rasters.append(
                results.Raster(
                    uri=raster_uri,
                    bands=raster.bands,
                    datatype=raster.datatype,
                    filetype=raster.filetype,
                    type=results.RasterType.ONE_FILE_RASTER,
                )
            )

    # Setup the main data_path. This could be a vrt (if job.results.rasters has
    # more than one Raster or if it has one or more TiledRasters

    if len(job.results.rasters) > 1 or (
        len(job.results.rasters) == 1
        and list(job.results.rasters.values())[0].type
        == results.RasterType.TILED_RASTER
    ):
        vrt_file = base_output_path.parent / f"{base_output_path.name}.vrt"
        logger.debug("Saving vrt file to %s", vrt_file)
        main_raster_file_paths = [raster.uri.uri for raster in out_rasters]
        combine_all_bands_into_vrt(main_raster_file_paths, vrt_file)
        job.results.uri = results.URI(uri=vrt_file)
    else:
        job.results.uri = list(job.results.rasters.values())[0].uri


def get_cloud_results_vrt(job: jobs.Job) -> typing.Optional[Path]:
    """Returns a vrt pointing to COGs in the cloud"""
    vrt_file_path = None
    vsicurl_paths = []

    if len(job.results.urls) > 0:
        vsicurl_paths = [f"/vsicurl/{url.url}" for url in job.results.urls]
        vrt_file_path = tempfile.NamedTemporaryFile(suffix=".vrt").name
        gdal.BuildVRT(
            str(vrt_file_path), [str(vsicurl_path) for vsicurl_path in vsicurl_paths]
        )
    else:
        logger.info(f"job {job} does not have downloadable results")

    return Path(vrt_file_path)


@dataclasses.dataclass
class BandData:
    band: results.Band
    band_number: int


def get_bands_by_name(
    job: jobs.Job, band_name: str, sort_property: str = "year"
) -> typing.List[BandData]:
    bands = job.results.get_bands()

    bands_and_indices = [
        (band, index)
        for index, band in enumerate(bands, start=1)
        if band.name == band_name
    ]

    sorted_bands_and_indices = sorted(
        bands_and_indices, key=lambda b: b[0].metadata[sort_property]
    )

    return [BandData(b[0], b[1]) for b in sorted_bands_and_indices]


# This version drops the sort_property in favor of filtering down to a single band based
# on metadata
def get_band_by_name(
    job: jobs.Job,
    band_name: str,
    filters: Union[None, List[Dict]] = None,
    return_band: bool = False,
) -> typing.Union[BandData, typing.Tuple[BandData, results.Band]]:
    bands = job.results.get_bands()

    if filters:
        bands_and_indices = [
            (band, index)
            for index, band in enumerate(bands, start=1)
            if (
                band.name == band_name
                and all(
                    [
                        str(band.metadata[filter["field"]]) == str(filter["value"])
                        for filter in filters
                    ]
                )
            )
        ]
    else:
        bands_and_indices = [
            (band, index)
            for index, band in enumerate(bands, start=1)
            if band.name == band_name
        ]

    if len(bands_and_indices) == 0:
        raise Exception(
            f"no bands found when filtering job {job.id} "
            f"for {band_name} with {filters} "
        )
    elif len(bands_and_indices) > 1:
        raise Exception(
            f"multiple bands found when filtering job {job.id} "
            f"for {band_name} with {filters} "
        )
    else:
        band, index = bands_and_indices[0]
        band_data = BandData(band, index)
        if return_band:
            return band_data, band
        else:
            return band_data


def make_job(params: Dict, script):
    final_params = params.copy()
    task_name = final_params.pop("task_name")
    task_notes = final_params.pop("task_notes")
    local_context = final_params.pop("local_context")

    return jobs.Job(
        id=uuid.uuid4(),
        params=final_params,
        progress=0,
        start_date=dt.datetime.now(dt.timezone.utc),
        status=jobs.JobStatus.PENDING,
        local_context=local_context,
        task_name=task_name,
        task_notes=task_notes,
        results=results.RasterResults(name=script.name, rasters=[], uri=None),
        script=script,
    )


def tar_gz_folder(path: Path, out_tar_gz, max_file_size_mb=None):
    paths = [p for p in path.rglob("*.*")]
    _make_tar_gz(out_tar_gz, paths, max_file_size_mb)


def _make_tar_gz(out_tar_gz, in_files, max_file_size_mb=None):
    with tarfile.open(out_tar_gz, "w:gz") as tar:
        for in_file in in_files:
            if max_file_size_mb:
                if (
                    in_file.is_file()
                    and os.path.getsize(in_file) <= max_file_size_mb * 1024 * 1024
                ):
                    tar.add(in_file, arcname=in_file.name)
            else:
                tar.add(in_file, arcname=in_file.name)


def write_cloud_job_metadata_file(job: jobs.Job):
    output_path = job.results.data_path.with_suffix(".json")
    # Ensure local paths aren't embedded in the output job file - but don't
    # modify the original job, so the full path can be used still on the server
    job_copy = deepcopy(job)
    job_copy.results.data_path = job_copy.results.data_path.name
    job_copy.results.other_paths = [c.name for c in job_copy.results.other_paths]
    with output_path.open("w", encoding="utf-8") as fh:
        raw_job = jobs.Job.Schema().dump(job_copy)
        json.dump(raw_job, fh, indent=2)

    return output_path


def write_json_job_metadata_file(job: jobs.Job, output_path):
    with output_path.open("w", encoding="utf-8") as fh:
        raw_job = jobs.Job.Schema().dump(job)
        json.dump(raw_job, fh, indent=2)

    return output_path


def buffer_feat(geojson, dist=0.000001):
    return ogr.CreateGeometryFromJson(geojson).Buffer(dist).ExportToJson()


def download_job(job, download_path):
    dl_output_path = download_cloud_results(job, download_path)

    if dl_output_path is not None:
        job.status = jobs.JobStatus.DOWNLOADED
        logger.info(f"Downloaded job to {dl_output_path}")
