resource "azurerm_resource_group" "rg" {
  name     = var.resource_group_name
  location = var.location

  tags = {
    environment = var.environment
    project     = "image-captioning"
    managed_by  = "terraform"
  }
}

resource "azurerm_container_registry" "acr" {
  name                = var.registry_name
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  sku                 = var.registry_sku

  tags = azurerm_resource_group.rg.tags
}

resource "azurerm_log_analytics_workspace" "workspace" {
  name                = "${var.app_name}-logs"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  sku                 = "PerGB2018"
  retention_in_days   = 30

  tags = azurerm_resource_group.rg.tags
}

resource "azurerm_container_app_environment" "env" {
  name                           = "${var.app_name}-env"
  location                       = azurerm_resource_group.rg.location
  resource_group_name            = azurerm_resource_group.rg.name
  log_analytics_workspace_id     = azurerm_log_analytics_workspace.workspace.id

  tags = azurerm_resource_group.rg.tags
}

resource "azurerm_container_app" "app" {
  name                         = var.app_name
  container_app_environment_id = azurerm_container_app_environment.env.id
  resource_group_name          = azurerm_resource_group.rg.name
  revision_mode                = "Single"

  ingress {
    allow_insecure_connections = false
    external_enabled           = true
    target_port                = 7860
    transport                  = "http"
    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  template {
    containers {
      name   = var.app_name
      image  = "${azurerm_container_registry.acr.login_server}/${var.container_image}"
      cpu    = var.container_cpu
      memory = var.container_memory

      env {
        name  = "GRADIO_SERVER_NAME"
        value = "0.0.0.0"
      }

      env {
        name  = "GRADIO_SERVER_PORT"
        value = "7860"
      }

      env {
        name  = "HF_HUB_DISABLE_PROGRESS_BARS"
        value = "1"
      }
    }

    min_replicas = var.min_replicas
    max_replicas = var.max_replicas
  }

  registry {
    server               = azurerm_container_registry.acr.login_server
    username             = azurerm_container_registry.acr.admin_username
    password_secret_name = "registry-password"
  }

  secret {
    name  = "registry-password"
    value = azurerm_container_registry.acr.admin_password
  }

  tags = azurerm_resource_group.rg.tags

  depends_on = [azurerm_container_app_environment.env]
}
