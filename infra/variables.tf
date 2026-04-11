variable "location" {
  description = "Azure region for resources"
  type        = string
  default     = "eastus"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "resource_group_name" {
  description = "Name of the resource group"
  type        = string
  default     = "rg-image-captioning"
}

variable "app_name" {
  description = "Name of the Container App (must be globally unique in Azure)"
  type        = string
  default     = "image-captioning-app"
}

variable "container_image" {
  description = "Container image URI (e.g., image-captioning-app:latest)"
  type        = string
  default     = "image-captioning-app:latest"
}

variable "registry_name" {
  description = "Name of Azure Container Registry (must be globally unique, alphanumeric only)"
  type        = string
  default     = "imagecaptioningacr"

  validation {
    condition     = can(regex("^[a-z0-9]{5,50}$", var.registry_name))
    error_message = "Registry name must be 5-50 lowercase alphanumeric characters."
  }
}

variable "registry_sku" {
  description = "SKU for container registry (Basic, Standard, Premium)"
  type        = string
  default     = "Standard"
}

variable "container_cpu" {
  description = "CPU allocation for container (cores)"
  type        = string
  default     = "1.0"
}

variable "container_memory" {
  description = "Memory allocation for container (GB)"
  type        = string
  default     = "2.0"
}

variable "min_replicas" {
  description = "Minimum number of replicas"
  type        = number
  default     = 1
}

variable "max_replicas" {
  description = "Maximum number of replicas for autoscaling"
  type        = number
  default     = 3
}
