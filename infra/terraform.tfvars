# Deployment configuration
location            = "eastus"
environment         = "dev"
resource_group_name = "rg-image-captioning"
app_name            = "image-captioning-app"
registry_name       = "imagecaptioningacr"
registry_sku        = "Standard"

# Container resource allocation
container_cpu    = "1.5"
container_memory = "1.5"

# Model configuration
model_id             = "Salesforce/blip-image-captioning-base"
finetuned_model_path = ""

# Autoscaling
min_replicas = 1
max_replicas = 3
