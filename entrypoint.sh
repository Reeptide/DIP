#!/bin/sh
# Dispatches to the right process based on the first arg / $ROLE.
# Usage: entrypoint.sh {master|worker|inference}
# Lets docker-compose.yml run one image as three different services.
set -e

ROLE="${1:-${ROLE:-worker}}"

case "$ROLE" in
  master)
    exec python -m master.app
    ;;
  worker)
    exec python -m worker.main
    ;;
  inference)
    exec python -m inference.main
    ;;
  *)
    echo "Unknown ROLE '$ROLE' - expected master, worker, or inference" >&2
    exit 1
    ;;
esac
