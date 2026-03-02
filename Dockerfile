# ---------- Builder: build wheels ----------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Build deps needed to compile certain wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (maximizes docker layer caching)
COPY requirements.txt /build/requirements.txt

# Build wheels into a clean wheelhouse directory (ONLY .whl files live here)
RUN pip wheel --no-cache-dir --wheel-dir /wheelhouse -r /build/requirements.txt


# ---------- Runtime: minimal image ----------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Runtime deps for OCR + PDF/image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libglib2.0-0 \
    libgl1 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps from wheelhouse (ONLY wheels)
COPY --from=builder /wheelhouse /wheelhouse
RUN pip install --no-cache-dir /wheelhouse/*.whl && rm -rf /wheelhouse

# Copy application code last
COPY . /app

EXPOSE 8000

# Start FastAPI
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
