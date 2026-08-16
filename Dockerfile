# One image, three roles (master / worker / inference), selected at runtime
# via the ROLE env var (see entrypoint.sh). Keeps the build simple and
# guarantees all three services share identical dependency versions.
#
FROM python:3.11-slim AS base

# opencv needs these at runtime even with opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# "dip" collides with Debian's built-in dial-up group of the same name -
# hence "dipapp" instead.
RUN useradd --create-home --uid 1000 dipapp && chown -R dipapp:dipapp /srv
USER dipapp

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
