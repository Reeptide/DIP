# One image, three roles (master / worker / inference), selected at runtime
# via the ROLE env var (see entrypoint.sh). Keeps the build simple and
# guarantees all three services share identical dependency versions.
#
FROM python:3.11-slim AS base

# opencv needs these at runtime even with opencv-python-headless.
# gosu: see entrypoint.sh - lets the container start as root just long
# enough to fix bind-mount ownership, then drop to dipapp for the actual
# process, without gosu's own quirks (su/sudo need a TTY or mess up
# signal forwarding; gosu is the standard tool official Docker images use
# for exactly this handoff).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    gosu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# Default false: build with plain CPU onnxruntime (requirements.txt),
# portable to any arch. docker-compose.gpu.yml sets this to "true" for the
# `inference` service only, layering in requirements-gpu.txt's
# onnxruntime-gpu + CUDA 13 nvidia-* wheels - see requirements.txt for why
# these used to be unconditional (broke arm64 builds, bloated master/worker
# images with CUDA they never touch).
ARG INSTALL_GPU=false

COPY requirements.txt requirements-gpu.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    if [ "$INSTALL_GPU" = "true" ]; then pip install --no-cache-dir -r requirements-gpu.txt; fi

COPY . .

# "dip" collides with Debian's built-in dial-up group of the same name -
# hence "dipapp" instead.
#
# No USER directive here (unlike before) - the container now starts as
# root and entrypoint.sh drops to dipapp itself via gosu, after fixing up
# ownership on the bind-mounted ./uploads and ./results (master service
# only, see docker-compose.yml). Found during a full-project review: uid
# 1000 only happens to be writable because it matches this dev host's own
# user - on any host where the user who ran `docker compose up` isn't uid
# 1000, /srv/uploads and /srv/results keep THEIR uid's ownership (bind
# mounts pass the host directory through as-is, a Dockerfile chown can't
# touch it), and file.save() in /upload EPERMs on every single upload.
RUN useradd --create-home --uid 1000 dipapp && chown -R dipapp:dipapp /srv

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/srv \
    # pip-installed CUDA libs (nvidia-cublas-cu12/nvidia-cudnn-cu12, see
    # requirements.txt) land inside site-packages, not on the system linker
    # path - nvidia-container-toolkit's GPU passthrough injects the driver
    # (libcuda.so) but not these, so onnxruntime-gpu can't find
    # libcublasLt.so/libcudnn.so without this. Only load-bearing for the
    # inference role; harmless no-op for master/worker, which never touch CUDA.
    LD_LIBRARY_PATH="/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/curand/lib:/usr/local/lib/python3.11/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib"

ENTRYPOINT ["/srv/entrypoint.sh"]
CMD ["worker"]
