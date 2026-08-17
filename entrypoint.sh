#!/bin/sh
# Dispatches to the right process based on the first arg / $ROLE.
# Usage: entrypoint.sh {master|worker|inference}
# Lets docker-compose.yml run one image as three different services.
set -e

ROLE="${1:-${ROLE:-worker}}"

# The container starts as root (see Dockerfile - no USER directive) so this
# can chown the bind-mounted ./uploads and ./results (master only) to
# dipapp before dropping privileges. Without this, those directories keep
# whatever uid the HOST user who ran `docker compose up` has - if that's
# not 1000, file.save() in master/app.py's /upload EPERMs on every upload.
# `|| true`: these paths don't exist for worker/inference (no such bind
# mount), and a chown that can't do anything useful shouldn't block startup.
chown -R dipapp:dipapp /srv/uploads /srv/results 2>/dev/null || true

case "$ROLE" in
  master)
    exec gosu dipapp python -m master.app
    ;;
  worker)
    exec gosu dipapp python -m worker.main
    ;;
  inference)
    exec gosu dipapp python -m inference.main
    ;;
  *)
    echo "Unknown ROLE '$ROLE' - expected master, worker, or inference" >&2
    exit 1
    ;;
esac
