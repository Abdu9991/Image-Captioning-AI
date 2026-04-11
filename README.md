# Image-Captioning-AI

An AI-powered image captioning web app built with Gradio, PyTorch, and Salesforce BLIP. Upload an image and the app generates a detailed natural-language description.

## Project Overview

This project provides:
- A local web UI for image upload and caption generation.
- BLIP-based captioning with tuned generation settings for richer output.
- Docker support for containerized execution.
- Terraform infrastructure for Azure Container Apps deployment.

## How It Works

The app loads `Salesforce/blip-image-captioning-large` and performs:
1. Image preprocessing with PIL.
2. BLIP tokenization via `BlipProcessor`.
3. Caption generation with beam search and length controls.
4. Text decoding and display in the Gradio UI.

Current generation configuration in `main.py`:
- Prompt: `"a detailed description of"`
- `max_new_tokens=80`
- `min_new_tokens=20`
- `num_beams=5`
- `repetition_penalty=1.2`

## Tech Stack

- Python 3.14 (local venv currently used)
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

## Prerequisites

- Windows PowerShell
- Python installed
- Virtual environment created at `.venv`

## Install Dependencies

Use the venv interpreter directly (works even if PowerShell activation policy blocks scripts):

```powershell
c:/Users/abdua/Desktop/AM/AI/Image-Captioning-AI/.venv/Scripts/python.exe -m pip install -r requirements.txt
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

## Run the App

```powershell
c:/Users/abdua/Desktop/AM/AI/Image-Captioning-AI/.venv/Scripts/python.exe main.py
```

Then open:
- `http://127.0.0.1:7860`

Notes:
- First run downloads the BLIP model (~1.88 GB), so startup can take time.
- The app uses `share=True`, so Gradio may also generate a public share link.

## Docker

Build image:

```powershell
docker build -t image-captioning-ai:latest .
```

Run container:

```powershell
docker run --rm -p 7860:7860 image-captioning-ai:latest
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
