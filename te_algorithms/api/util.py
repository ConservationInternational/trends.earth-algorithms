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
from urllib.parse import unquote
from urllib.parse import urlparse

import boto3
import botocore
import requests
from botocore.config import Config
from osgeo import gdal
from osgeo import ogr
from te_schemas import jobs

logger = logging.getLogger(__name__)

SDG_RAW_JOB_NAME = 'sdg-15-3-1-sub-indicators',
SDG_SUMMARY_RAW_JOB_NAME = 'sdg-15-3-1-summary',


def get_s3_client(access_key_id=None, secret_access_key=None, no_sign=False):
    if no_sign:
        config = Config(signature_version=botocore.UNSIGNED)
    else:
        config = Config()
    try:
        client = boto3.client(
            's3',
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=config
        )
    except IOError:
        logging.warning(
            "AWS credentials file not found. Credentials must be in "
            "environment variable in default AWS credentials location, "
            "or (if on the cloude) an instance role must be attached)"
        )
        client = boto3.client('s3', config=config)

    return client


def set_s3_obj_acl(client, obj, prefix, bucket, acl='public-read'):
    client.put_object_acl(
        ACL='bucket-owner-full-control', Key=f'{prefix}/{obj}', Bucket=bucket
    )
    client.copy_object(
        CopySource={
            'Bucket': bucket,
            'Key': f'{prefix}/{obj}'
        },
        StorageClass='STANDARD',
        Key=f'{prefix}/{obj}',
        Bucket=bucket
    )


def put_to_s3(
    filename: Path,
    bucket: str,
    prefix: str,
    extra_args={},
    aws_access_key_id=None,
    aws_secret_access_key=None
):

    client = get_s3_client(
        access_key_id=aws_access_key_id,
        secret_access_key=aws_secret_access_key
    )
    try:
        up_to_date = etag_compare(
            filename,
            get_s3_etag(key=f'{prefix}/{filename.name}', bucket=bucket)
        )
    except botocore.exceptions.ClientError:
        # File is not there
        up_to_date = False

    if up_to_date:
        logging.info(
            f'Skipping upload of {filename} to s3 - already up to date'
        )
    else:
        logging.info(f'Uploading {filename} to s3')
        client.upload_file(
            str(filename),
            bucket,
            f'{prefix}/{filename.name}',
            ExtraArgs=extra_args
        )
    #expected_etag = hashlib.md5(filename.read_bytes()).hexdigest()
    # waiter = client.get_waiter('object_exists')
    # waiter.wait(
    #     Bucket=bucket,
    #     Key=f'{prefix}/{filename.name}',
    #     IfMatch=expected_etag,
    #     WaiterConfig={
    #         'Delay': 1,
    #         'MaxAttempts': 20
    #     }
    # )


def get_from_s3(
    obj,
    out_path,
    bucket,
    prefix,
    aws_access_key_id=None,
    aws_secret_access_key=None
):
    client = get_s3_client(
        access_key_id=aws_access_key_id,
        secret_access_key=aws_secret_access_key
    )
    logging.info(f'Downloading {obj} from s3 to {out_path}')
    client.download_file(bucket, f'{prefix}/{obj}', out_path)


def md5_checksum(filename):
    m = hashlib.md5()
    with open(filename, 'rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            m.update(data)

    return m.hexdigest()


def etag_checksum_multipart(filename, chunk_size=8 * 1024 * 1024):
    md5s = []
    with open(filename, 'rb') as f:
        for data in iter(lambda: f.read(chunk_size), b''):
            md5s.append(hashlib.md5(data).digest())
    m = hashlib.md5(b"".join(md5s))

    return '{}-{}'.format(m.hexdigest(), len(md5s))


def etag_compare(filename, et):
    try:
        if '-' in et and et == etag_checksum_multipart(filename):
            # AWS S3 multipart etags

            return True
        elif '-' not in et and et == md5_checksum(filename):
            # AWS S3 single part etags

            return True
        elif binascii.hexlify(base64.b64decode(et)
                              ).decode() == md5_checksum(filename):
            # Google cloud etas

            return True
    except (FileNotFoundError, binascii.Error):
        return False

    return False


def get_s3_etag(
    key, bucket, aws_access_key_id=None, aws_secret_access_key=None
):
    client = get_s3_client(
        access_key_id=aws_access_key_id,
        secret_access_key=aws_secret_access_key
    )
    resp = client.head_object(Bucket=bucket, Key=key)

    return resp['ETag'].strip('"')


def write_to_cog(in_file, out_file):
    gdal.UseExceptions()
    gdal.Translate(
        out_file,
        in_file,
        format='COG',
        creationOptions=[
            'COMPRESS=LZW',
            'BIGTIFF=YES',
            'NUM_THREADS=ALL_CPUS',
        ]
    )


def push_cog_to_s3(local_path, obj, s3_prefix, s3_bucket, extra_args={}):
    try:
        cog_up_to_date = etag_compare(
            local_path,
            get_s3_etag(key=f"{s3_prefix}/{obj}", bucket=s3_bucket)
        )
    except botocore.exceptions.ClientError:
        cog_up_to_date = False

    if cog_up_to_date:
        logging.info(f'COG at {local_path} already up to date on S3')
    else:
        put_to_s3(
            local_path,
            bucket=s3_bucket,
            prefix=s3_prefix,
            extra_args=extra_args
        )


def get_job_json_from_s3(
    s3_prefix,
    s3_bucket,
    aws_access_key_id=None,
    aws_secret_access_key=None,
    substr_re=None
):
    # Returns most recent job JSON from s3
    client = get_s3_client(
        access_key_id=aws_access_key_id,
        secret_access_key=aws_secret_access_key
    )
    objects = client.list_objects_v2(
        Bucket=s3_bucket,
        Prefix=s3_prefix,
    )
    logging.info(
        f"Looking in bucket {s3_bucket} "
        f"for key prefix: {s3_prefix}"
    )

    if len(objects['Contents']) == 0:
        logging.exception(f"No objects found for {s3_prefix}")

        return None
    else:
        objects = objects['Contents']
        # Want most recent key
        objects.sort(key=lambda o: o['LastModified'], reverse=True)
        # Only want JSON files
        objects = [o for o in objects if bool(re.search('.json$', o['Key']))]
        # Need correct LPD choice if this is an SDG job

        logging.info(f'substr_re: {substr_re}')

        if substr_re:
            objects = [
                o for o in objects if bool(re.search(substr_re, o['Key']))
            ]
    jobfile = tempfile.NamedTemporaryFile(suffix='.json').name
    client.download_file(s3_bucket, objects[0]['Key'], jobfile)
    with open(jobfile, 'r') as f:
        job_json = json.load(f)

    return jobs.Job.Schema().load(job_json)


def _write_subregion_cogs(
    job, cog_vsi_base, bounding_boxes, temp_dir, s3_prefix, s3_bucket,
    extra_args
):
    # Label pieces of polygon as east/west when there are two
    piece_labels = ['E', 'W']

    cog_vsi_paths = []

    for index, bounding_box in enumerate(bounding_boxes):
        suffix = f'{("_" + piece_labels[index]) if len(bounding_boxes) > 1 else ""}'

        temp_vrt_local_path = Path(temp_dir
                                   ) / (cog_vsi_base.name + f'{suffix}.vrt')
        gdal.BuildVRT(
            str(temp_vrt_local_path),
            str(job.results.data_path),
            outputBounds=bounding_box
        )

        cog_local_path = Path(temp_dir) / (cog_vsi_base.name + f'{suffix}.tif')
        cog_vsi_path = Path(str(cog_vsi_base) + f'{suffix}.tif')
        cog_vsi_paths.append(cog_vsi_path)
        write_to_cog(str(temp_vrt_local_path), str(cog_local_path))
        push_cog_to_s3(
            cog_local_path, cog_vsi_path.name, s3_prefix, s3_bucket, extra_args
        )

    return cog_vsi_paths


def get_job_cog_vsi_prefix(job, filename_base, s3_prefix, s3_bucket):

    return Path(f'/vsis3/{s3_bucket}/{s3_prefix}/{filename_base}')


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]

    return text  # or whatever


def write_job_to_s3_cog(
    job,
    aoi,
    filename_base,
    s3_prefix,
    s3_bucket,
    s3_region,
    aws_access_key_id=None,
    aws_secret_access_key=None,
    extra_suffix='',
    extra_args={}
):
    if job.results.type == jobs.JobResultType.CLOUD_RESULTS:
        bounding_boxes = aoi.get_aligned_output_bounds(
            str(job.results.data_path)
        )

        cog_vsi_base = get_job_cog_vsi_prefix(
            job, filename_base, s3_prefix, s3_bucket
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            cog_vsi_paths = _write_subregion_cogs(
                job, cog_vsi_base, bounding_boxes, temp_dir, s3_prefix,
                s3_bucket, extra_args
            )

            # Need to temporarily set data_path to S3 vsi path so jobfile on s3
            # will point to correct location, so save the local data_path for later
            data_path_local = job.results.data_path

            logging.debug(f'cog_vsi_paths: {cog_vsi_paths}')

            if len(bounding_boxes) > 1:
                vrt_file = Path(temp_dir) / (cog_vsi_base.name + '.vrt')
                logging.info(f'vrt_file: {vrt_file}')
                gdal.SetConfigOption(
                    "AWS_ACCESS_KEY_ID",
                    aws_access_key_id,
                )
                gdal.SetConfigOption(
                    "AWS_SECRET_ACCESS_KEY", aws_secret_access_key
                )
                gdal.BuildVRT(
                    str(vrt_file),
                    [str(cog_vsi_path) for cog_vsi_path in cog_vsi_paths]
                )
                put_to_s3(
                    vrt_file,
                    bucket=s3_bucket,
                    prefix=s3_prefix,
                    extra_args=extra_args
                )
                job.results.data_path = cog_vsi_base.with_suffix('.vrt')
            else:
                job.results.data_path = cog_vsi_base.with_suffix('.tif')

            # Need to use path-style addressing to avoid SSL errors due to the dot
            # in the (trends.earth) bucket name
            #s3_url_base = f'https://{s3_bucket}.s3.{s3_region}.amazonaws.com'

            if s3_region == 'us-east-1':
                subdomain = 's3'
            else:
                subdomain = f's3-{s3_region}'
            s3_url_base = f'https://{subdomain}.amazonaws.com/{s3_bucket}'
            keys = [
                remove_prefix(str(cog_vsi_path), f'/vsis3/{s3_bucket}/')
                for cog_vsi_path in cog_vsi_paths
            ]
            job.results.urls = [
                jobs.JobUrl(
                    url=s3_url_base + '/' + k,
                    md5_hash=get_s3_etag(k, bucket=s3_bucket)
                ) for k in keys
            ]
            job.status = jobs.JobStatus.DOWNLOADED
            job.results.type = jobs.JobResultType.CLOUD_RESULTS

            write_job_json_to_s3(
                job, filename_base + '.json', s3_prefix, s3_bucket, extra_args
            )

            job.results.data_path = data_path_local

    else:
        write_job_json_to_s3(job, filename_base + '.json', extra_args)


def write_job_json_to_s3(job, filename, s3_prefix, s3_bucket, extra_args):
    with tempfile.TemporaryDirectory() as temp_dir:
        job_file_path = Path(temp_dir) / filename
        with job_file_path.open(mode="w", encoding="utf-8") as fh:
            raw_job = jobs.Job.Schema().dump(job)
            json.dump(raw_job, fh, indent=2)

        put_to_s3(
            job_file_path,
            bucket=s3_bucket,
            prefix=s3_prefix,
            extra_args=extra_args
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
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD',
                                      value).encode('ascii',
                                                    'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())

    return re.sub(r'[-\s]+', '-', value).strip('-_')


def _get_download_size(url):
    resp = requests.get(url, stream=True)
    total_size = int(resp.headers['Content-length'])

    if total_size < 1e5:
        total_size_pretty = '{:.2f} KB'.format(round(total_size / 1024, 2))
    else:
        total_size_pretty = '{:.2f} MB'.format(round(total_size * 1e-6, 2))

    return total_size_pretty


def _download_file(url, out_path):
    download_size = _get_download_size(url)
    logger.info(f"Downloading {url} ({download_size}) to {out_path}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    return out_path


def _get_single_cloud_result(
    url: jobs.JobUrl,
    output_path: Path,
    aws_access_key_id=None,
    aws_secret_access_key=None
) -> typing.Optional[Path]:
    path_exists = output_path.is_file()
    hash_matches = etag_compare(output_path, url.md5_hash)

    if path_exists and hash_matches:
        logger.info(
            f"No download necessary, result already present in {output_path}"
        )
        result = output_path
    else:
        if 'amazonaws.com' not in url.url:
            _download_result(url.url, output_path)
        else:
            url_path = PurePosixPath(unquote(urlparse(url.url).path))
            # Use get_from_s3 in case file is not public and need to use a key
            # to access
            get_from_s3(
                obj=url_path.parts[-1],
                out_path=str(output_path),
                bucket=url_path.parts[1],
                prefix=PurePosixPath(*url_path.parts[2:-1]),
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
            )

        downloaded_hash_matches = etag_compare(output_path, url.md5_hash)
        result = output_path if downloaded_hash_matches else None

    return result


def _download_result(url: str, output_path: Path) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _download_file(url, str(output_path))

    return output_path


def _get_multiple_cloud_results(
    job: jobs.Job,
    base_output_path: Path,
    aws_access_key_id=None,
    aws_secret_access_key=None
) -> Path:
    vrt_tiles = []

    for index, url in enumerate(job.results.urls):
        output_path = (
            base_output_path.parent / f"{base_output_path.name}_{index}.tif"
        )
        tile_path = _get_single_cloud_result(
            url, output_path, aws_access_key_id, aws_secret_access_key
        )
        logging.info(f'tile_path {tile_path}')

        if tile_path is not None:
            vrt_tiles.append(tile_path)
    vrt_file_path = base_output_path.parent / f"{base_output_path.name}.vrt"
    logging.info(f'vrt_tiles: {vrt_tiles}')
    gdal.SetConfigOption("AWS_ACCESS_KEY_ID", aws_access_key_id)
    gdal.SetConfigOption("AWS_SECRET_ACCESS_KEY", aws_secret_access_key)
    gdal.BuildVRT(
        str(vrt_file_path), [str(vrt_tile) for vrt_tile in vrt_tiles]
    )

    return vrt_file_path


def _get_job_basename(job: jobs.Job):
    separator = "_"
    name_fragments = []
    task_name = slugify(job.task_name)

    if task_name != "":
        name_fragments.append(task_name)

    if job.script:
        name_fragments.append(job.script.name)
    name_fragments.extend([job.local_context.area_of_interest_name])

    return separator.join(name_fragments)


def download_cloud_results(
    job: jobs.Job,
    download_path,
    aws_access_key_id=None,
    aws_secret_access_key=None
) -> typing.Optional[Path]:
    base_output_path = download_path / f"{_get_job_basename(job)}"
    output_path = None

    if len(job.results.urls) > 0:
        if len(job.results.urls) == 1:
            final_output_path = (
                base_output_path.parent / f"{base_output_path.name}.tif"
            )
            output_path = _get_single_cloud_result(
                job.results.urls[0], final_output_path, aws_access_key_id,
                aws_secret_access_key
            )
        else:  # multiple files, download them then save VRT
            output_path = _get_multiple_cloud_results(
                job, base_output_path, aws_access_key_id, aws_secret_access_key
            )
    else:
        logger.info(f"job {job} does not have downloadable results")

    return output_path


def get_cloud_results_vrt(job: jobs.Job) -> typing.Optional[Path]:
    vrt_file_path = None
    vsicurl_paths = []

    if len(job.results.urls) > 0:
        vsicurl_paths = [f'/vsicurl/{url.url}' for url in job.results.urls]
        vrt_file_path = tempfile.NamedTemporaryFile(suffix='.vrt').name
        res = gdal.BuildVRT(
            str(vrt_file_path),
            [str(vsicurl_path) for vsicurl_path in vsicurl_paths]
        )
    else:
        logger.info(f"job {job} does not have downloadable results")

    return Path(vrt_file_path)


@dataclasses.dataclass
class BandData:
    band: jobs.JobBand
    band_number: int


def get_bands_by_name(
    job,
    band_name: str,
    sort_property: str = "year"
) -> typing.List[jobs.JobBand]:

    bands = job.results.bands

    bands_and_indices = [
        (band, index) for index, band in enumerate(bands, start=1)
        if band.name == band_name
    ]

    sorted_bands_and_indices = sorted(
        bands_and_indices, key=lambda b: b[0].metadata[sort_property]
    )

    return [BandData(b[0], b[1]) for b in sorted_bands_and_indices]


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
        results=jobs.JobCloudResults(
            name=script.name,
            bands=[],
            urls=[],
            data_path=None,
            other_paths=[]
        ),
        script=script
    )


def tar_gz_folder(path: Path, out_tar_gz):
    paths = [p for p in path.rglob('*.*')]
    _make_tar_gz(out_tar_gz, paths)


def _make_tar_gz(out_tar_gz, in_files):
    with tarfile.open(out_tar_gz, "w:gz") as tar:
        for in_file in in_files:
            tar.add(in_file, arcname=in_file.name)


def write_cloud_job_metadata_file(job: jobs.Job):
    output_path = job.results.data_path.with_suffix('.json')
    # Ensure local paths aren't embedded in the output job file - but don't
    # modify the original job, so the full path can be used still on the server
    job_copy = deepcopy(job)
    job_copy.results.data_path = job_copy.results.data_path.name
    job_copy.results.other_paths = [
        c.name for c in job_copy.results.other_paths
    ]
    with output_path.open("w", encoding="utf-8") as fh:
        raw_job = jobs.Job.Schema().dump(job_copy)
        json.dump(raw_job, fh, indent=2)

    return output_path


def write_json_job_metadata_file(job: jobs.Job, output_path):
    with output_path.open("w", encoding="utf-8") as fh:
        raw_job = jobs.Job.Schema().dump(job)
        json.dump(raw_job, fh, indent=2)

    return output_path


def buffer_feat(geojson, dist=.000001):
    return ogr.CreateGeometryFromJson(geojson).Buffer(dist).ExportToJson()


def download_job(job, download_path):
    dl_output_path = download_cloud_results(job, download_path)

    if dl_output_path is not None:
        job.results.data_path = dl_output_path
        job.status = jobs.JobStatus.DOWNLOADED
        logging.info(f'Downloaded job to {dl_output_path}')
