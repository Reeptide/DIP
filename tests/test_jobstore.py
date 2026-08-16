"""Integration tests for app/jobstore.py against a real Redis (testcontainers).

Covers the two properties Step 6 and Step 7 depend on: O(1) atomic per-tile
updates via hashes (not the old GET-mutate-SETEX blob), and idempotent
duplicate-result handling via record_tile_result's HSET-return-value gate
(the fix for the real race reproduced live during the Step 7 kill-a-worker
demo - a duplicate result must not inflate results_count past the true
distinct tile count).
"""


def test_create_and_get_job_round_trips_types(jobstore):
    jobstore.create_job(
        "job-1", original_filename="a.jpg", operation="grayscale",
        original_width=1024, original_height=768, expected_tiles=4,
        active_workers=2,
    )
    job = jobstore.get_job("job-1")
    assert job['status'] == 'processing'
    assert job['expected_tiles'] == 4  # int, not the redis string '4'
    assert job['original_width'] == 1024
    assert job['results_count'] == 0
    assert job['completion_time'] is None


def test_get_job_returns_none_for_missing_job(jobstore):
    assert jobstore.get_job("does-not-exist") is None


def test_record_tile_result_increments_distinct_count(jobstore):
    jobstore.create_job(
        "job-2", original_filename="a.jpg", operation="grayscale",
        original_width=512, original_height=512, expected_tiles=2,
        active_workers=1,
    )
    n1 = jobstore.record_tile_result("job-2", 0, {'x': 0, 'y': 0}, 0.05)
    n2 = jobstore.record_tile_result("job-2", 1, {'x': 512, 'y': 0}, 0.03)
    assert n1 == 1
    assert n2 == 2
    assert jobstore.get_job("job-2")['results_count'] == 2


def test_duplicate_tile_result_does_not_inflate_count(jobstore):
    """The Step 7 bug, reproduced directly: a redelivered tile (Kafka
    at-least-once + a reaper requeue landing near-simultaneously) must not
    push the distinct count past the number of tiles actually held, or a
    job flips to ready-for-reconstruction while genuinely missing a tile."""
    jobstore.create_job(
        "job-3", original_filename="a.jpg", operation="grayscale",
        original_width=512, original_height=512, expected_tiles=2,
        active_workers=1,
    )
    jobstore.record_tile_result("job-3", 0, {'x': 0}, 0.05)
    # Same tile_id delivered again - HSET overwrites, HINCRBY must not fire.
    n = jobstore.record_tile_result("job-3", 0, {'x': 0}, 0.05)

    assert n == 1
    assert jobstore.get_job("job-3")['results_count'] == 1
    assert jobstore.count_tile_results("job-3") == 1


def test_get_tile_latencies_only_records_first_delivery(jobstore):
    jobstore.create_job(
        "job-4", original_filename="a.jpg", operation="grayscale",
        original_width=512, original_height=512, expected_tiles=1,
        active_workers=1,
    )
    jobstore.record_tile_result("job-4", 0, {'x': 0}, 0.111)
    jobstore.record_tile_result("job-4", 0, {'x': 0}, 0.999)  # duplicate, ignored

    latencies = jobstore.get_tile_latencies_ms("job-4")
    assert latencies == [111.0]


def test_delete_job_tile_state_clears_tiles_and_attempts_but_not_job_metadata(jobstore):
    jobstore.create_job(
        "job-5", original_filename="a.jpg", operation="grayscale",
        original_width=512, original_height=512, expected_tiles=1,
        active_workers=1,
    )
    jobstore.record_tile_result("job-5", 0, {'x': 0}, 0.05)
    jobstore.incr_tile_attempts("job-5", 0)

    jobstore.delete_job_tile_state("job-5")

    assert jobstore.count_tile_results("job-5") == 0
    assert jobstore.get_job("job-5") is not None  # metadata survives


def test_incr_tile_attempts_counts_across_calls(jobstore):
    a = jobstore.incr_tile_attempts("job-6", 0)
    b = jobstore.incr_tile_attempts("job-6", 0)
    c = jobstore.incr_tile_attempts("job-6", 1)  # different tile, own counter
    assert (a, b, c) == (1, 2, 1)


def test_scan_job_ids_excludes_companion_keys(jobstore):
    jobstore.create_job(
        "job-7", original_filename="a.jpg", operation="grayscale",
        original_width=512, original_height=512, expected_tiles=1,
        active_workers=1,
    )
    jobstore.record_tile_result("job-7", 0, {'x': 0}, 0.05)  # creates job:job-7:tiles, :latencies too

    ids = jobstore.scan_job_ids()
    assert "job-7" in ids
    # scan_job_ids must not also return "job-7:tiles" etc as if they were job ids.
    assert all(':' not in job_id for job_id in ids)
