# Distributed Image Processing Pipeline

A distributed image-processing system built on Kafka and Redis. A master node tiles
uploaded images and distributes the tiles as tasks; worker nodes consume tiles, apply
OpenCV operations, and publish results back for reassembly. A Flask web UI handles
upload and monitoring.

## Architecture

| Node | File | Role |
|---|---|---|
| Master (#1) | `master.py` | Flask web UI, image tiling, task distribution, result reconstruction, worker heartbeat monitoring |
| Worker (#3) | `worker1.py` | Consumes `tasks`, applies OpenCV ops, publishes to `results` + `heartbeats` |
| Worker (#4) | `worker2.py` | Same as worker1, second instance |

Kafka topics: `tasks`, `results`, `heartbeats`.
Redis holds job state on the master.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then edit for your machine
```

Kafka and Redis must be running first — see `broker_setup/BROKER_SETUP_STEPS.md`.

## Running

Start each node in its own terminal:

```bash
python master.py     # web UI on http://localhost:5000
python worker1.py
python worker2.py
```

## Repo conventions

- `uploads/` and `results/` are runtime artifacts and are **git-ignored**. Do not commit
  images — the previous repo grew to 54 MB of test JPEGs this way.
- Configuration comes from environment variables (see `.env.example`), never hardcoded.

## Contributing

1. Branch off `main`: `git checkout -b your-name/short-description`
2. Commit, push, and open a pull request.
3. `main` is the integration branch — avoid committing to it directly.
