terraform {
  backend "s3" {
    bucket         = "bmin5100-terraform-state"
    key            = "Rohan.Shah1@Pennmedicine.upenn.edu-template/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
  }
}
