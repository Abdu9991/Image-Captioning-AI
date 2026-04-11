terraform {
  required_version = ">= 1.5"
  
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.100"
    }
  }

  # Uncomment to use Azure Storage for state (recommended for production)
  # backend "azurerm" {
  #   resource_group_name  = "rg-tfstate"
  #   storage_account_name = "tfstate[randomstring]"
  #   container_name       = "tfstate"
  #   key                  = "image-captioning.tfstate"
  # }
}

provider "azurerm" {
  features {}
}
