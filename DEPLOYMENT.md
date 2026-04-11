# Image Captioning App - Azure Deployment Guide

## Overview
This guide walks you through deploying the Image-Captioning AI app to Azure Container Apps using Terraform and Docker.

## Prerequisites

1. **Azure CLI** - [Install](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli)
   ```powershell
   az version
   az login
   ```

2. **Terraform** - [Install](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli)
   ```powershell
   terraform version
   ```

3. **Docker** - [Install Docker Desktop](https://www.docker.com/products/docker-desktop)
   ```powershell
   docker version
   ```

4. **Azure Subscription** - Ensure you have an active subscription
   ```powershell
   az account show
   ```

## Deployment Steps

### Step 1: Build and Push Docker Image

```powershell
# Navigate to project root
cd C:\Users\abdua\Desktop\AM\AI\Image-Captioning-AI

# Set variables
$REGISTRY_NAME = "imagecaptioningacr"
$IMAGE_NAME = "image-captioning-app"
$IMAGE_TAG = "latest"
$LOCATION = "eastus"
$RESOURCE_GROUP = "rg-image-captioning"

# Create resource group
az group create `
  --name $RESOURCE_GROUP `
  --location $LOCATION

# Create Azure Container Registry
az acr create `
  --resource-group $RESOURCE_GROUP `
  --name $REGISTRY_NAME `
  --sku Standard

# Get registry login credentials
$LOGIN_SERVER = az acr show `
  --name $REGISTRY_NAME `
  --query loginServer `
  --output tsv

# Login to registry
az acr login --name $REGISTRY_NAME

# Build and push image
az acr build `
  --registry $REGISTRY_NAME `
  --image "$IMAGE_NAME`:$IMAGE_TAG" `
  .
```

### Step 2: Initialize and Plan Terraform

```powershell
# Navigate to infrastructure directory
cd infra

# Initialize Terraform
terraform init

# Validate configuration
terraform validate

# Plan deployment (review changes)
terraform plan -out=tfplan
```

### Step 3: Apply Terraform Configuration

```powershell
# Apply the plan
terraform apply tfplan

# Or apply directly with auto-approval (not recommended for production)
terraform apply -auto-approve
```

### Step 4: Get App URL and Test

```powershell
# Retrieve the Container App URL
$APP_URL = terraform output -raw container_app_url
Write-Host "App is available at: https://$APP_URL"

# Test the endpoint
Invoke-WebRequest -Uri "https://$APP_URL" -UseBasicParsing
```

## Configuration

### Update Variables

Edit `infra/terraform.tfvars` to customize:

```hcl
location            = "eastus"           # Azure region
app_name            = "my-image-caption-app"
registry_name       = "myregistryname"   # Must be globally unique
container_cpu       = "1.5"              # CPU cores
container_memory    = "3.0"              # GB
min_replicas        = 1                  # Min autoscale
max_replicas        = 3                  # Max autoscale
```

### Advanced: Custom Variables

Create `infra/terraform.auto.tfvars` for local overrides:

```hcl
environment = "prod"
location    = "westus2"
```

## Monitoring & Logs

### View Container Logs

```powershell
# Get recent logs
az containerapp logs show `
  --name image-captioning-app `
  --resource-group rg-image-captioning `
  --follow
```

### Monitor with Log Analytics

```powershell
# View metrics in Azure Portal
az monitor log-analytics workspace show `
  --name image-captioning-app-logs `
  --resource-group rg-image-captioning
```

## Scaling

### Horizontal Autoscaling

The app autoscales between 1-3 replicas based on CPU/memory. Adjust in `terraform.tfvars`:

```hcl
min_replicas = 1
max_replicas = 10
```

### Vertical Scaling

Increase container resources:

```hcl
container_cpu    = "2.0"   # Increase from 1.5
container_memory = "4.0"   # Increase from 3.0
```

## Cleanup

### Destroy All Resources

```powershell
cd infra

# Preview what will be deleted
terraform plan -destroy

# Delete everything
terraform destroy
```

## Troubleshooting

### Image Pull Errors
```powershell
# Verify image exists in registry
az acr repository list --name $REGISTRY_NAME

# Check image details
az acr repository show --name $REGISTRY_NAME --image image-captioning-app:latest
```

### Container App Not Starting
```powershell
# Check app status
az containerapp show `
  --name image-captioning-app `
  --resource-group rg-image-captioning `
  --query properties.provisioningState

# View recent events
az containerapp logs show `
  --name image-captioning-app `
  --resource-group rg-image-captioning
```

### Terraform State Issues
```powershell
# Refresh state
terraform refresh

# View current state
terraform state list

# Remove stuck resource (careful!)
terraform state rm azurerm_container_app.app
```

## Security Best Practices

1. **Enable Private Endpoints** - Restrict registry access
2. **Use Managed Identity** - Remove hardcoded credentials
3. **Enable Container Scanning** - Check for vulnerabilities
4. **Use Secret Management** - Store HuggingFace tokens in Azure Key Vault

## Cost Estimation

- **Container Apps**: ~$40-60/month (1 vCPU, 2 GB memory, 1-3 replicas)
- **Container Registry**: ~$5/month (Standard tier)
- **Log Analytics**: ~$30/month (30-day retention)

**Total**: ~$75-95/month

Optimize costs with:
- Reduce `max_replicas`
- Use "Basic" registry tier
- Enable consumption-based pricing

## Next Steps

- [ ] Build and push Docker image
- [ ] Initialize Terraform
- [ ] Review and apply infrastructure
- [ ] Test deployed endpoint
- [ ] Configure monitoring/alerts
- [ ] Document custom configurations
