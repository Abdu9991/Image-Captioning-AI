# Stage 1: Builder
FROM python:3.11-slim as builder

ARG MODEL_ID=Salesforce/blip-image-captioning-base
ARG PREFETCH_MODEL=true

WORKDIR /app

ENV HF_HOME=/opt/huggingface
ENV TRANSFORMERS_CACHE=/opt/huggingface
ENV MODEL_ID=${MODEL_ID}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Pre-download the default model during the image build so runtime startup on Render is faster.
RUN if [ "$PREFETCH_MODEL" = "true" ]; then python -c "from transformers import BlipForConditionalGeneration, BlipProcessor; import os; model_id = os.environ['MODEL_ID']; BlipProcessor.from_pretrained(model_id); BlipForConditionalGeneration.from_pretrained(model_id, low_cpu_mem_usage=True)"; fi

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

ENV HF_HOME=/opt/huggingface
ENV TRANSFORMERS_CACHE=/opt/huggingface
ENV PRELOAD_MODEL_ON_STARTUP=false
ENV RENDER_OPTIMIZED=true
ENV TORCH_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false

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
