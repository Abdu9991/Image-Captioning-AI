output "container_app_url" {
  description = "URL of the deployed Container App"
  value       = azurerm_container_app.app.ingress[0].fqdn
}

output "registry_login_server" {
  description = "Login server of the Azure Container Registry"
  value       = azurerm_container_registry.acr.login_server
}

output "registry_id" {
  description = "ID of the Azure Container Registry"
  value       = azurerm_container_registry.acr.id
}

output "resource_group_id" {
  description = "ID of the resource group"
  value       = azurerm_resource_group.rg.id
}

output "container_app_id" {
  description = "ID of the Container App"
  value       = azurerm_container_app.app.id
}

output "log_analytics_workspace_id" {
  description = "ID of the Log Analytics workspace"
  value       = azurerm_log_analytics_workspace.workspace.id
}
