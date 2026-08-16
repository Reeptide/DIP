"""Session-scoped Redis testcontainer for the jobstore integration test.

A real container (not fakeredis) because app/jobstore.py's correctness
claims - atomic HSET-gated counting, HLEN as the distinct-tile source of
truth - are about real Redis semantics under concurrent access (see Step 7's
idempotency-race writeup). A pure-Python Redis reimplementation is exactly
the kind of thing that could quietly diverge from those semantics and mask
a bug this test exists to catch.
"""
import pytest
from testcontainers.redis import RedisContainer

import app.jobstore as jobstore_module
from app.jobstore import RedisClient


@pytest.fixture(scope="session")
def redis_container():
    with RedisContainer("redis:7-alpine") as container:
        yield container


@pytest.fixture
def jobstore(redis_container, monkeypatch):
    """Point app.jobstore's module-level redis_client at the testcontainer
    for the duration of one test, and flush it after - tests must not leak
    state into each other."""
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(6379)
    client = RedisClient(host, int(port))
    monkeypatch.setattr(jobstore_module, "redis_client", client)
    yield jobstore_module
    client.client.flushdb()
