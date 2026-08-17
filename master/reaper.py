"""Requeues tiles that never came back, and gives up in a bounded way.

At-least-once is the honest delivery semantic here: a worker commits its
Kafka offset only after publishing a tile's result, so an abrupt death
(`docker kill`, OOM, node loss) mid-tile leaves that tile's offset
uncommitted. Kafka redelivers it once the consumer group rebalances - but
only for partitions the dead worker actually still owned. Anything already
committed-but-unpublished, or dropped by a worker that exhausted its
retries and parked the task on the DLQ, is gone as far as Kafka is
concerned. The job then sits at 'processing' forever, because
expected_tiles can never be met.

This closes that hole from the master's side. Every scan interval it looks
for jobs still 'processing' past a grace window, works out which tile_ids
are genuinely missing, and republishes just those tasks:

- Geometry is recomputed, not remembered: app.imaging.tile_geometries() is
  the same loop split_image_into_tiles() uses, fed with the job hash's
  original_width/original_height, so tile_ids line up exactly with what was
  dispatched. Nothing needs the source image.
- The tile's bytes are already in MinIO from the original dispatch and are
  only swept after reconstruction, so the requeued task can point at the
  existing blob rather than re-tiling anything. Two pipelines, two
  conventions: OpenCV tasks (`master/publisher.py`) at
  `{job_id}/task/{tile_id}.jpg` on `tasks`; ML tasks
  (`master/ml_publisher.py`) at `{job_id}/ml_task/{tile_id}.jpg` on
  `ml_tasks`, keyed by `job_data['operation'] == 'object_detection'` -
  `_republish_tile()` branches on this rather than assuming one convention.
  (Earlier versions of this reaper only knew the OpenCV convention and
  skipped ML jobs outright - full recovery is the fix, not just the skip.)
- Attempts are capped (MAX_TILE_ATTEMPTS) and counted in Redis, not in
  memory, so the cap survives a master restart. Past the cap the job is
  marked 'degraded' - a terminal status meaning "as complete as this is
  going to get" - instead of being retried forever or hanging silently.
"""
import json
import logging
import time

from confluent_kafka import Producer

from app.blobstore import tile_exists, delete_job_tiles
from app.leader import is_leader
from app.metrics import tiles_requeued_total, jobs_finished_total
from app.config import (
    TASK_TOPIC, ML_TASK_TOPIC, MAX_TILE_ATTEMPTS, REAPER_SCAN_INTERVAL,
    TILE_OUTSTANDING_TIMEOUT_SECONDS, configure_logging,
)
from app.imaging import tile_geometries
from app.jobstore import (
    get_job, get_tile_results, get_last_reap_time, incr_tile_attempts,
    scan_job_ids, set_job_status, set_last_reap_time,
    add_abandoned_tile, get_abandoned_tiles,
)
from app.kafkaio import master_producer_conf

logger = logging.getLogger(__name__)

# Statuses the reaper will act on. Everything else - 'ready_for_reconstruction',
# 'completed', 'degraded' - is either finished or has already given up, and
# must not be requeued by later scans.
REAPABLE_STATUS = 'processing'


def _missing_tile_ids(job_id, expected_tiles):
    received = set(get_tile_results(job_id).keys())
    return sorted(set(range(expected_tiles)) - received)


def _republish_tile(producer, job_id, operation, geom, attempt):
    """Republish one tile's task, reusing the blob staged at dispatch time.

    Two pipelines, two topics and blob-key conventions - master/publisher.py
    stages OpenCV tasks at "{job_id}/task/{tile_id}.jpg" on `tasks`;
    master/ml_publisher.py stages ML tasks at "{job_id}/ml_task/{tile_id}.jpg"
    on `ml_tasks`, with no `operation` field (there's exactly one thing an ML
    task means). This used to only know about the OpenCV convention and
    skipped ML jobs entirely (see the git history / docs/ARCHITECTURE.md for
    why that was the safer interim choice) - now branches on job_data's
    operation instead of assuming.
    """
    is_ml = operation == 'object_detection'
    tile_id = geom['tile_id']
    blob_key = f"{job_id}/{'ml_task' if is_ml else 'task'}/{tile_id}.jpg"

    if not tile_exists(blob_key):
        logger.error(f"Cannot requeue job {job_id[:8]} tile {tile_id}: blob {blob_key} is gone")
        return False

    task = {
        'job_id': job_id,
        'tile_id': tile_id,
        'blob_key': blob_key,
        'x': geom['x'],
        'y': geom['y'],
        'width': geom['width'],
        'height': geom['height'],
        'timestamp': time.time(),
        # Carried through to the DLQ record if this attempt also fails, so
        # the number of times a poison tile has been tried is visible there.
        # inference/main.py ignores unknown fields, so it's harmless to
        # include this on ML tasks too rather than branching the dict shape.
        'attempt': attempt,
    }
    if not is_ml:
        task['operation'] = operation

    producer.produce(
        ML_TASK_TOPIC if is_ml else TASK_TOPIC,
        key=f"{job_id}:{tile_id}",  # same keying as master/publisher.py
        value=json.dumps(task).encode('utf-8'),
    )
    producer.poll(0)
    logger.warning(f"Requeued job {job_id[:8]} tile {tile_id} (attempt {attempt})")
    return True


def reap_once(producer):
    """One sweep over all jobs. Returns the number of tiles requeued.

    Leadership is re-checked before EACH job, not just once before the loop
    starts - a sweep over many jobs (each doing a MinIO stat_object call, a
    producer.flush(), Redis writes) can outlive a leadership handoff, and
    without this a former leader could keep doing reaper work - including
    incr_tile_attempts(), which two concurrent reapers touching the same
    tile is exactly the correctness bug leader election exists to prevent
    (see app/leader.py's module docstring) - for a real, non-negligible
    window after actually losing the lock.
    """
    now = time.time()
    requeued = 0

    for job_id in scan_job_ids():
        if not is_leader():
            logger.warning("Lost leadership mid-sweep - stopping rather than "
                            "risk reaping under a lock we may no longer hold")
            break

        try:
            job_data = get_job(job_id)
            if not job_data or job_data['status'] != REAPABLE_STATUS:
                continue

            # Clock from whichever came later: the job's creation or its last
            # reap. Without the latter, a job past the timeout would be
            # requeued on every scan, never giving the requeued task the same
            # grace window the original got.
            last_activity = max(job_data['timestamp'], get_last_reap_time(job_id))
            if now - last_activity < TILE_OUTSTANDING_TIMEOUT_SECONDS:
                continue

            missing = _missing_tile_ids(job_id, job_data['expected_tiles'])
            if not missing:
                # Every tile is in Redis but the status hasn't flipped - the
                # results consumer will catch it on the next message; nothing
                # for the reaper to do.
                continue

            abandoned = get_abandoned_tiles(job_id)
            retryable = [t for t in missing if t not in abandoned]

            if not retryable:
                # Every currently-missing tile has already been permanently
                # given up on in a PRIOR pass - nothing left that could ever
                # complete this job, so only now is it actually correct to
                # call it degraded. This used to flip to degraded the
                # instant ANY tile exceeded its attempt cap in a SINGLE
                # pass, even when other tiles from that same pass were
                # simultaneously being successfully re-queued - discarding
                # their in-flight recovery for no reason, since a degraded
                # job's tiles are excluded from all future reaper scans and
                # from the results consumer's completion check.
                set_job_status(job_id, 'degraded')
                pipeline = 'ml' if job_data['operation'] == 'object_detection' else 'opencv'
                jobs_finished_total.labels(pipeline=pipeline, status='degraded').inc()
                logger.error(f"Job {job_id[:8]} marked DEGRADED - all {len(missing)} outstanding "
                             f"tile(s) {sorted(missing)} are permanently unrecoverable")

                # A degraded job never reaches /reconstruct (OpenCV) or the
                # results-consumer completion branch (ML) - both are the
                # only other places that sweep MinIO blobs, so without this
                # a degraded job's blobs (task/result for OpenCV, ml_task
                # for ML) leaked forever. Safe to sweep everything, not
                # just the abandoned tiles' blobs: no code path produces
                # partial reconstruction for a degraded job, so a
                # successfully-processed tile's blob is exactly as
                # unreachable as an abandoned one once the job is terminal.
                delete_job_tiles(job_id)
                continue

            # Job's own tile_size, not the process's current TILE_SIZE
            # default - they can differ (TILE_SIZE is runtime-configurable,
            # see app/config.py) and using the wrong one silently produces a
            # different tile_id layout than what was actually dispatched.
            geometries = {
                g['tile_id']: g for g in
                tile_geometries(job_data['original_width'], job_data['original_height'],
                                 tile_size=job_data['tile_size'])
            }

            logger.warning(f"Job {job_id[:8]} has {len(retryable)} tile(s) outstanding after "
                           f"{now - last_activity:.0f}s: {retryable[:10]}")

            for tile_id in retryable:
                attempt = incr_tile_attempts(job_id, tile_id) + 1  # original dispatch was attempt 1
                if attempt > MAX_TILE_ATTEMPTS:
                    add_abandoned_tile(job_id, tile_id)
                    logger.error(f"Job {job_id[:8]} tile {tile_id} exceeded "
                                 f"{MAX_TILE_ATTEMPTS} attempts - abandoned")
                    continue

                geom = geometries.get(tile_id)
                if geom is None:
                    # expected_tiles and the recomputed layout disagree -
                    # would mean the stored tile_size and original
                    # dimensions no longer produce the same tile count,
                    # which shouldn't happen now that tile_size is
                    # job-specific, but fail safe rather than crash.
                    logger.error(f"Job {job_id[:8]} tile {tile_id} has no geometry - abandoned")
                    add_abandoned_tile(job_id, tile_id)
                    continue

                if _republish_tile(producer, job_id, job_data['operation'], geom, attempt):
                    requeued += 1
                    tiles_requeued_total.inc()
                else:
                    add_abandoned_tile(job_id, tile_id)

            producer.flush(timeout=10)
            set_last_reap_time(job_id, now)

        except Exception as e:
            logger.error(f"Reaper error on job {job_id[:8]}: {str(e)}", exc_info=True)

    return requeued


def run_reaper():
    """Background thread: sweep for outstanding tiles forever.

    Runs on every master replica, but only does anything while is_leader()
    - reap_once() increments a per-tile attempt counter, and N replicas
    reaping the same stalled job in the same window would inflate that
    counter N times for one real outage, hitting MAX_TILE_ATTEMPTS after a
    single genuine failure instead of the intended retry budget. See
    app/leader.py.
    """
    logger.info(f"Reaper started (scan every {REAPER_SCAN_INTERVAL}s, "
                f"tile timeout {TILE_OUTSTANDING_TIMEOUT_SECONDS}s, "
                f"max {MAX_TILE_ATTEMPTS} attempts/tile)")

    producer = Producer(master_producer_conf())

    while True:
        if is_leader():
            try:
                reap_once(producer)
            except Exception as e:
                logger.error(f"Error in reaper loop: {str(e)}", exc_info=True)
        time.sleep(REAPER_SCAN_INTERVAL)


if __name__ == '__main__':
    # Runnable standalone (`python -m master.reaper`) for testing without a
    # full master; in normal operation master/app.py starts run_reaper() as a
    # daemon thread alongside the results consumer and heartbeat monitor.
    configure_logging('reaper')
    run_reaper()
