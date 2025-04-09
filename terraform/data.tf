data "aws_caller_identity" "current" {}

data "aws_region" "current_region" {}

data "terraform_remote_state" "infrastructure" {
  backend = "s3"

  config = {
    bucket  = "bmin5100-terraform-state"
    key     = "infrastructure/terraform.tfstate"
    region  = "us-east-1"
  }
}
