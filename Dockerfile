FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Python deps (install before copying code for cache efficiency)
COPY pyproject.toml ./
RUN pip install --no-cache-dir .

# Copy source
COPY . .

# Install taxembed package
RUN pip install --no-cache-dir -e .

# Default entrypoint: taxembed CLI
ENTRYPOINT ["python", "-m", "taxembed.cli.main"]
