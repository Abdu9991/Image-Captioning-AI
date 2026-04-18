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
RUN pip install --user --no-cache-dir -r requirements.txt

# Pre-download the configured model during the image build so runtime startup is faster.
# Prefer a fine-tuned BLIP checkpoint when FINETUNED_MODEL_PATH is provided.
RUN if [ "$PREFETCH_MODEL" = "true" ]; then \
        cat > /tmp/prefetch_model.py <<'PY'
import os

from transformers import (
    AutoProcessor,
    BlipForConditionalGeneration,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    BlipProcessor,
)

model_source = os.environ.get("FINETUNED_MODEL_PATH") or os.environ["MODEL_ID"]
normalized = model_source.lower()

if "qwen2.5-vl" in normalized or "qwen-vl" in normalized:
    AutoProcessor.from_pretrained(model_source)
elif "blip2" in normalized:
    Blip2Processor.from_pretrained(model_source)
    Blip2ForConditionalGeneration.from_pretrained(model_source, low_cpu_mem_usage=True)
else:
    BlipProcessor.from_pretrained(model_source)
    BlipForConditionalGeneration.from_pretrained(model_source, low_cpu_mem_usage=True)
PY
        python /tmp/prefetch_model.py; \
        rm -f /tmp/prefetch_model.py; \
    fi

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
