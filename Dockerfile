FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Copy source. (We can't split deps and source into separate layers the usual
# way: hatchling's metadata validation reads README.md and the wheel build
# needs src/taxembed, so `pip install .` against pyproject.toml alone fails
# during metadata prep. The base image already has torch + cuda, so the
# remaining pip install is fast anyway.)
COPY . .

# Install taxembed (editable for in-place dev when bind-mounting source).
RUN pip install --no-cache-dir -e .

# Default entrypoint: taxembed CLI
ENTRYPOINT ["python", "-m", "taxembed.cli.main"]
