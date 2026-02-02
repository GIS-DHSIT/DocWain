
# ---------- Builder: build wheels ----------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Build deps for compiling wheels when needed
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better caching
COPY requirements.txt /build/requirements.txt

# Upgrade build tooling (prevents many wheel build issues)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip setuptools wheel

# Build wheels (cache downloads on the builder host; wheels only output)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --wheel-dir /wheelhouse -r /build/requirements.txt


# ---------- Runtime ----------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Runtime deps (OCR + PDF tools + basic runtime libs + DNS tools)
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr \
      poppler-utils \
      libglib2.0-0 \
      libgl1 \
      ca-certificates \
      curl \
      dnsutils \
    && rm -rf /var/lib/apt/lists/*

# Install deps from wheelhouse using requirements.txt (more deterministic)
COPY --from=builder /wheelhouse /wheelhouse
COPY requirements.txt /app/requirements.txt
RUN pip install --no-index --find-links=/wheelhouse -r /app/requirements.txt \
    && rm -rf /wheelhouse

# Copy application code last (keeps dependency layers cached)
COPY . /app

EXPOSE 8000

# Optional: better shutdown handling + faster logs
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
