"""MinIO-backed blob store for tile bytes - the claim-check pattern.

Before this module existed, tile image bytes were base64-encoded and
embedded directly in Kafka task/result messages (~700KB per message,
requiring message.max.bytes raised to 50MB as a band-aid). Kafka is a log,
not a blob store - that pattern works but doesn't scale and bloats the
broker's disk and network usage for no reason. Now the master/worker only
ever exchange a short key (~50 bytes) through Kafka; the actual bytes live
here and are fetched only by whichever process needs them.
"""
import logging
from io import BytesIO

from minio import Minio
from minio.deleteobjects import DeleteObject
from minio.error import S3Error

from app.config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET, MINIO_SECURE

logger = logging.getLogger(__name__)

_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE,
)


def put_tile(job_id: str, tile_id: int, kind: str, data: bytes) -> str:
    """Store tile bytes, return the key to publish through Kafka.

    kind is 'task' (master's input tile) or 'result' (worker's processed
    tile) - keeps the two namespaced under the same job so both can be
    swept in one delete_job_tiles() call after reconstruction.
    """
    key = f"{job_id}/{kind}/{tile_id}.jpg"
    _client.put_object(MINIO_BUCKET, key, BytesIO(data), length=len(data), content_type='image/jpeg')
    return key


def get_tile(key: str) -> bytes:
    """Fetch tile bytes by key. Raises minio.error.S3Error if missing."""
    response = _client.get_object(MINIO_BUCKET, key)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def tile_exists(key: str) -> bool:
    """True if a blob is still present. Used by the reaper before requeuing a
    task: the tile's original bytes were staged at dispatch time and are only
    swept after reconstruction, so a missing blob means there is nothing left
    to reprocess and requeuing would just fail three more times."""
    try:
        _client.stat_object(MINIO_BUCKET, key)
        return True
    except S3Error:
        return False


def delete_job_tiles(job_id: str) -> int:
    """Delete every blob (task and result tiles) for a job, called after
    reconstruction - mirrors the original code's Redis tile-data cleanup,
    now extended to MinIO so blobs don't accumulate unboundedly."""
    try:
        objects = _client.list_objects(MINIO_BUCKET, prefix=f"{job_id}/", recursive=True)
        to_delete = [DeleteObject(obj.object_name) for obj in objects]
        if not to_delete:
            return 0
        errors = list(_client.remove_objects(MINIO_BUCKET, to_delete))
        for err in errors:
            logger.error(f"Failed to delete blob {err.object_name}: {err}")
        return len(to_delete) - len(errors)
    except S3Error as e:
        logger.error(f"Error cleaning up blobs for job {job_id}: {e}")
        return 0
