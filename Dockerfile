FROM pytorch/pytorch@sha256:a22a1fca37f8361c8a1e859cd6eb6bd9d1fb384f9c0dcb2cfc691a178eb03d17

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-gpu-lock.txt /tmp/requirements-gpu-lock.txt

RUN python -m pip install --no-cache-dir -r /tmp/requirements-gpu-lock.txt \
    && python -m pip install --no-cache-dir --no-build-isolation flash-attn==2.8.3

WORKDIR /artifact

CMD ["bash"]
