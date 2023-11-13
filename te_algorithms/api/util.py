import base64
import binascii
import dataclasses
import datetime as dt
import hashlib
import json
import logging
import re
import tarfile
import tempfile
import typing
import unicodedata
import uuid
from copy import deepcopy
from pathlib import Path
from pathlib import PurePosixPath
from typing import Dict
from typing import List
from urllib.parse import unquote
from urllib.parse import urlparse

import boto3
import botocore
import requests
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from osgeo import gdal
from osgeo import ogr
from te_schemas import jobs
from te_schemas import results

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
        logger.info(f"Uploading {filename} to s3 at {key}")
        config = TransferConfig(multipart_threshold=MULTIPART_THRESHOLD)
        client.upload_file(
            str(filename), bucket, key, Config=config, ExtraArgs=s3_extra_args
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
    with open(filename, "rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            m.update(data)

    return m.hexdigest()


def etag_checksum_multipart(filename, chunk_size=8 * 1024 * 1024):
    md5s = []
    with open(filename, "rb") as f:
        for data in iter(lambda: f.read(chunk_size), b""):
            md5s.append(hashlib.md5(data).digest())
    m = hashlib.md5(b"".join(md5s))

    return "{}-{}".format(m.hexdigest(), len(md5s))


def etag_compare(filename, et):
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


def write_to_cog(in_file, out_file, nodata_value):
    gdal.UseExceptions()
    gdal.Translate(
        out_file,
        in_file,
        format="COG",
        creationOptions=[
            "COMPRESS=LZW",
            "BIGTIFF=YES",
            "NUM_THREADS=ALL_CPUS",
        ],
        noData=nodata_value,
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
    logger.info(f"Looking in bucket {s3_bucket} " f"for key prefix: {s3_prefix}")

    logger.debug(f"objects {objects}")

    if len(objects["Contents"]) == 0:
        logger.exception(f"No objects found for {s3_prefix}")

        return None
    else:
        objects = objects["Contents"]
        # Want most recent key
        objects.sort(key=lambda o: o["LastModified"], reverse=True)
        # Only want JSON files
        objects = [o for o in objects if bool(re.search(".json$", o["Key"]))]
        logger.debug(f"found objects pre-grep: {[o['Key'] for o in objects]}")

        for substr_regex in substr_regexs:
            logger.debug(f"grepping for {substr_regex}")
            objects = [o for o in objects if bool(re.search(substr_regex, o["Key"]))]
        logger.debug(f"found objects post-grep: {[o['Key'] for o in objects]}")
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
    # Label pieces of polygon as east/west when there are two
    piece_labels = ["E", "W"]

    cog_vsi_paths = []

    for index, bounding_box in enumerate(bounding_boxes):
        this_suffix = (
            suffix + f'{("_" + piece_labels[index]) if len(bounding_boxes) > 1 else ""}'
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
        write_to_cog(str(temp_vrt_local_path), str(cog_local_path), nodata_value)
        push_cog_to_s3(
            cog_local_path, cog_vsi_path.name, s3_prefix, s3_bucket, s3_extra_args
        )

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
            # (East and West) subregions. So write everything as Raster
            # regardless of what came in, unless there are E/W in which case
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


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
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
    hash_matches = etag_compare(out_file, uri.etag.hash)

    if path_exists and hash_matches:
        logger.info(f"No download necessary, result already present in {out_file}")
        result = out_file
    else:
        if "vsis3" not in str(uri.uri) and "amazonaws.com" not in str(uri.uri):
            _download_file(uri.uri, out_file)
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
    job: jobs.Job, download_path, aws_access_key_id=None, aws_secret_access_key=None
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
            logger.info("Saving vrt file to %s", vrt_file)
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
            raster_uri = (results.URI(uri=out_file),)
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
        len(job.results.rasters == 1)
        and job.results.rasters[0].type == results.RasterType.TILED_RASTER
    ):
        vrt_file = base_output_path.parent / f"{base_output_path.name}.vrt"
        logger.info("Saving vrt file to %s", vrt_file)
        main_raster_file_paths = [raster.uri.uri for raster in out_rasters]
        combine_all_bands_into_vrt(main_raster_file_paths, vrt_file)
        job.results.uri = results.URI(uri=vrt_file)
    else:
        job.results.uri = job.results.rasters[0].uri


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
    job, band_name: str, sort_property: str = "year"
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


# This version drops the sort_property in favor fr filtering down to a single band based
# on metadata
def get_band_by_name(
    job, band_name: str, filters: List[Dict] = None
) -> typing.List[BandData]:
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
            f"no bands found when filtering for {band_name} with {filters} "
        )
    elif len(bands_and_indices) > 1:
        raise Exception(
            f"multiple bands found when filtering for {band_name} with {filters} "
        )
    else:
        return BandData(*bands_and_indices[0])


def make_job(params, script):
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


def tar_gz_folder(path: Path, out_tar_gz):
    paths = [p for p in path.rglob("*.*")]
    _make_tar_gz(out_tar_gz, paths)


def _make_tar_gz(out_tar_gz, in_files):
    with tarfile.open(out_tar_gz, "w:gz") as tar:
        for in_file in in_files:
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
