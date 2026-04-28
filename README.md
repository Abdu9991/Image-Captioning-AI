# Image-Captioning-AI

An AI-powered image captioning web app built with Gradio, PyTorch, and Hugging Face vision-language models including Salesforce BLIP, BLIP-2, and Qwen 2.5 VL. Upload an image and the app generates a detailed natural-language description.

## Project Overview

This project provides:
- A local web UI for image upload and caption generation.
- BLIP, BLIP-2, and Qwen 2.5 VL captioning with tuned generation settings for cleaner and richer output.
- `Caption Level` options (`All`, `Brief`, `Detailed`, `Highly Detailed`) selectable from a dropdown.
- Structured `Highly Detailed` caption format with `Short Summary` and `Highly Detailed Description` sections.
- Post-processing cleanup to remove repeated fragments and truncated ending artifacts.
- A second-pass refinement step to reduce obvious child/animal confusion in captions.
- Docker support for containerized execution.
- Terraform infrastructure for Azure Container Apps deployment.

## How It Works

The app defaults to `Salesforce/blip-image-captioning-base` (lower memory) and can switch to BLIP-2, Qwen 2.5 VL, or a fine-tuned checkpoint via environment variables. It performs:
1. Image preprocessing with PIL.
2. Model-specific preprocessing via `BlipProcessor`, `Blip2Processor`, or `AutoProcessor`.
3. Caption generation with beam search and length controls.
4. Text decoding and display in the Gradio UI.

Current default generation base configuration in `main.py`:
- `max_new_tokens=48`
- `min_new_tokens=10`
- `num_beams=3`
- `repetition_penalty=1.15`
- `no_repeat_ngram_size=3`

The web UI accepts an uploaded image and returns the selected caption output in a single result panel on the right-hand side. You can choose `All`, `Brief`, `Detailed`, or `Highly Detailed` from the `Caption Level` dropdown before generating.

Caption rules:

- Brief caption: one very short sentence focused only on the main subject.
- Detailed caption: 1-2 sentences with subject, basic attributes, and setting.
- Highly Detailed caption: a two-part structured format:
  - `Short Summary` (1-2 sentences)
  - `Highly Detailed Description` (full scene breakdown)

Example output style:

- Brief: `A dog runs freely.`
- Detailed: `A cheerful brown dog runs across a sunny park, enjoying the open grassy space.`
- Highly Detailed: `A lively brown dog dashes across a vibrant green park, its paws barely touching the ground as sunlight filters through the trees. The gentle breeze moves the leaves, and the open space feels bright, fresh, and full of energy. The moment unfolds like a small scene from a story, filled with movement, freedom, and joy.`

`Detailed` is tuned for a clean, human-readable caption, while `Highly Detailed` is tuned for structured, richer scene analysis.

## Tech Stack

- Python 3.11+
- Gradio 6.12.0
- PyTorch 2.11.0
- Transformers 5.5.3
- Pillow 12.2.0
- NumPy 2.4.4
- Docker (multi-stage image)
- Terraform + AzureRM provider

## Repository Structure

- `main.py`: Gradio app and BLIP captioning logic.
- `requirements.txt`: Python dependencies.
- `Dockerfile`: Container build/runtime image.
- `.dockerignore`: Build context exclusions.
- `infra/terraform.tf`: Terraform provider and version settings.
- `infra/main.tf`: Azure resources (RG, ACR, Log Analytics, Container App Environment, Container App).
- `infra/variables.tf`: Deployment variables.
- `infra/outputs.tf`: Deployment outputs (URL, IDs).
- `infra/terraform.tfvars`: Default deployment values.
- `DEPLOYMENT.md`: Extended Azure deployment runbook.

## Local Development Setup

### Prerequisites

- Windows PowerShell
- Python installed
- Virtual environment created at `.venv`

### Install Dependencies

Use the venv interpreter directly (works even if PowerShell activation policy blocks scripts):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Optional activation approach:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If activation is blocked by execution policy:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Run the App

```powershell
.\.venv\Scripts\python.exe main.py
```

By default, the app runs on:
- `http://127.0.0.1:10000`

Use a fine-tuned checkpoint (local path or Hugging Face model ID):

```powershell
$env:FINETUNED_MODEL_PATH="your-org/your-finetuned-blip-model"
.\.venv\Scripts\python.exe main.py
```

Use Qwen 2.5 VL:

```powershell
$env:MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"
.\.venv\Scripts\python.exe main.py
```

Notes for Qwen 2.5 VL:
- It requires substantially more memory than the default BLIP model.
- GPU is strongly preferred for acceptable startup and inference speed.
- The first run downloads the full checkpoint from Hugging Face.

Use BLIP-2:

```powershell
$env:MODEL_ID="Salesforce/blip2-opt-2.7b"
.\.venv\Scripts\python.exe main.py
```

Notes for BLIP-2:
- BLIP-2 is supported directly by the app and usually produces stronger captions than the base BLIP model.
- `Salesforce/blip2-opt-2.7b` is a common general-purpose choice, but it needs much more RAM than `Salesforce/blip-image-captioning-base`.
- BLIP-2 is not a good default for low-memory Render deployments unless you substantially increase container memory.

## Render Deployment Notes

- Render instances are CPU-bound, so Qwen 2.5 VL and BLIP-2 are usually not practical defaults there.
- The Docker image prefetches the configured base model or fine-tuned checkpoint during build to reduce first-request latency.
- Render defaults limit preload, threads, and generation settings to reduce memory pressure on CPU.
- For lowest memory usage, keep `MODEL_ID=Salesforce/blip-image-captioning-base` and prefer a fine-tuned BLIP base checkpoint.

Recommended low-memory Render setup:

- Base model for fine-tuning: `Salesforce/blip-image-captioning-base`
- Deploy target: a fine-tuned BLIP checkpoint published as a Hugging Face model repo
- Keep `RENDER_OPTIMIZED=true`
- Keep `PRELOAD_MODEL_ON_STARTUP=false`
- Avoid BLIP-2 and Qwen on small Render instances
- Keep thread env vars at `1` (`TORCH_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`)

### Render 512 MiB Survival Settings

If your Render service is capped at `512 MiB`, keep the deployment conservative:

- Use `MODEL_ID=Salesforce/blip-image-captioning-base`
- If you fine-tune, fine-tune that BLIP base model and deploy the fine-tuned checkpoint instead of switching to a larger architecture
- Keep `PRELOAD_MODEL_ON_STARTUP=false` so the model is not loaded during container boot
- Keep `CPU_QUANTIZE=true`
- Keep `TORCH_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`
- Reduce generation cost with `CAPTION_MAX_NEW_TOKENS=32-40` and `CAPTION_NUM_BEAMS=1-2`

Notes:

- The `UNEXPECTED ... position_ids` log line from Transformers is benign and is not the cause of the memory issue
- Fine-tuning does not materially reduce inference memory on its own; model size and startup behavior are the main levers
- If the service still OOMs, move to a larger Render instance before trying BLIP-2 or Qwen

If you have a fine-tuned Hugging Face checkpoint, set Render environment variables like this:

```text
MODEL_ID=Salesforce/blip-image-captioning-base
FINETUNED_MODEL_PATH=your-org/your-finetuned-blip-base
RENDER_OPTIMIZED=true
PRELOAD_MODEL_ON_STARTUP=false
TORCH_NUM_THREADS=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
TOKENIZERS_PARALLELISM=false
```

If you build the Docker image yourself, you can prefetch the fine-tuned checkpoint during build:

```powershell
docker build `
  --build-arg MODEL_ID="Salesforce/blip-image-captioning-base" `
  --build-arg FINETUNED_MODEL_PATH="your-org/your-finetuned-blip-base" `
  -t image-captioning-ai:latest .
```

Use a Hugging Face model ID for Render. A local fine-tuned path will not exist inside the deployed container unless you explicitly copy that checkpoint into the image.

Optional generation tuning:

```powershell
$env:MODEL_ID="Salesforce/blip-image-captioning-base"
$env:CAPTION_DETAIL_LEVEL="Highly Detailed"
$env:CAPTION_MAX_NEW_TOKENS="40"
$env:CAPTION_NUM_BEAMS="2"
$env:CAPTION_NO_REPEAT_NGRAM_SIZE="3"
```

Then open:
- `http://127.0.0.1:10000`

Notes:
- First run downloads the configured model/checkpoint, so startup can take time.

## Docker

Build image:

```powershell
docker build -t image-captioning-ai:latest .
```

Run container:

```powershell
docker run --rm -e PORT=7860 -p 7860:7860 image-captioning-ai:latest
```

Open:
- `http://127.0.0.1:7860`

## Terraform Deployment (Azure Container Apps)

This repo includes IaC to deploy:
- Resource Group
- Azure Container Registry (ACR)
- Log Analytics Workspace
- Container Apps Environment
- Azure Container App (public ingress on port 7860)

### Configure Variables

Edit `infra/terraform.tfvars`:
- `location`
- `resource_group_name`
- `app_name`
- `registry_name` (must be globally unique, lowercase alphanumeric)
- `container_cpu`, `container_memory`
- `model_id` (defaults to `Salesforce/blip-image-captioning-base`)
- `finetuned_model_path` (optional fine-tuned model path or model ID)
- `min_replicas`, `max_replicas`

### Deploy

```powershell
cd infra
terraform init
terraform validate
terraform plan -out=tfplan
terraform apply tfplan
```

Get URL:

```powershell
terraform output -raw container_app_url
```

Detailed cloud instructions are in `DEPLOYMENT.md`.

## Troubleshooting

- `!pip` not recognized in PowerShell:
  - `!pip` is notebook syntax. Use `python -m pip ...` in terminal.
- `No module named ...`:
  - Install dependencies in the same interpreter used to run app.
- App appears stuck on start:
  - Wait for initial model download to complete.
- TensorFlow install mismatch:
  - This project does not require TensorFlow for current caption pipeline.

## Suggested Improvements

- Add GPU-specific runtime and CUDA image options.
- Add caching/warm-start for model loading.
- Add unit tests for `generate_caption` behavior.
- Add CI pipeline for lint/build/deploy checks.
- Add security scanning for container images.

## License

No license file is currently included. Add a `LICENSE` file if you plan to share publicly.
