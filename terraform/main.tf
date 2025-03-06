resource "aws_s3_bucket" "template_project_data" {
  bucket = "template-project-data"

  tags = {
    Owner = element(split("/", data.aws_caller_identity.current.arn), 1)
  }
}

resource "aws_s3_bucket_ownership_controls" "template_project_data_ownership_controls" {
  bucket = aws_s3_bucket.template_project_data.id
  rule {
    object_ownership = "BucketOwnerPreferred"
  }
}

resource "aws_s3_bucket_acl" "template_project_data_acl" {
  depends_on = [aws_s3_bucket_ownership_controls.template_project_data_ownership_controls]

  bucket = aws_s3_bucket.template_project_data.id
  acl    = "private"
}

resource "aws_s3_bucket_lifecycle_configuration" "template_project_data_expiration" {
  bucket = aws_s3_bucket.template_project_data.id

  rule {
    id      = "compliance-retention-policy"
    status  = "Enabled"

    expiration {
	  days = 100
    }
  }
}
