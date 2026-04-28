# Stage 1: Builder
FROM python:3.11-slim AS builder

ARG MODEL_ID=Salesforce/blip-image-captioning-base
ARG FINETUNED_MODEL_PATH=
ARG PREFETCH_MODEL=true

WORKDIR /app

ENV HF_HOME=/opt/huggingface
ENV TRANSFORMERS_CACHE=/opt/huggingface
ENV MODEL_ID=${MODEL_ID}
ENV FINETUNED_MODEL_PATH=${FINETUNED_MODEL_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.11.0 && \
    grep -v '^torch==' requirements.txt > requirements-render.txt && \
    pip install --user --no-cache-dir -r requirements-render.txt && \
    rm -f requirements-render.txt
COPY prefetch_model.py .

# Pre-download the configured model during the image build so runtime startup is faster.
# Prefer a fine-tuned BLIP checkpoint when FINETUNED_MODEL_PATH is provided.
RUN if [ "$PREFETCH_MODEL" = "true" ]; then python prefetch_model.py; fi

# Stage 2: Runtime
FROM python:3.11-slim

ARG MODEL_ID=Salesforce/blip-image-captioning-base
ARG FINETUNED_MODEL_PATH=

WORKDIR /app

ENV HF_HOME=/opt/huggingface
ENV TRANSFORMERS_CACHE=/opt/huggingface
ENV PRELOAD_MODEL_ON_STARTUP=false
ENV RENDER_OPTIMIZED=true
ENV TORCH_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false
ENV MODEL_ID=${MODEL_ID}
ENV FINETUNED_MODEL_PATH=${FINETUNED_MODEL_PATH}

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /opt/huggingface /opt/huggingface
ENV PATH="/root/.local/bin:${PATH}"

# Copy application code
COPY main.py .

# Expose port (Gradio default)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860').read()" || exit 1

# Run the app
CMD ["python", "-u", "main.py"]
