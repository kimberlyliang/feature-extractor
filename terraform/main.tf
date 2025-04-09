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

resource "aws_ecr_repository" "template_project_ecr_repository" {
  name = "bmin5100-example"
  force_delete = true
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Owner = element(split("/", data.aws_caller_identity.current.arn), 1)
  }
}

resource "aws_iam_role" "ecs_task_execution_role" {
  name = "ECSTaskExecutionRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_policy_attachment" "aws_ecs_task_execution_policy_attachment" {
  name       = "AWSECSTaskExecutionAttachment"
  roles      = [aws_iam_role.ecs_task_execution_role.name]
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_policy" "ecs_execution_task_policy" {
  name        = "ECSNetworkInterfacePolicy"
  description = "Allows ECS Fargate to manage ENIs and CloudWatch logs"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "ec2:DescribeNetworkInterfaces",
        "ec2:CreateNetworkInterface",
        "ec2:AttachNetworkInterface",
        "ec2:DeleteNetworkInterface",
        "ec2:AssignPrivateIpAddresses",
        "ec2:UnassignPrivateIpAddresses",
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutDestination",
        "logs:PutLogEvents",
        "logs:DescribeLogStreams",
      ]
      Resource = "*"
    }]
  })
}

resource "aws_iam_policy_attachment" "ecs_task_execution_policy_attachment" {
  name       = "ECSTaskExecutionAttachment"
  roles      = [aws_iam_role.ecs_task_execution_role.name]
  policy_arn = aws_iam_policy.ecs_execution_task_policy.arn
}

resource "aws_iam_role" "ecs_task_role" {
  name = "ECSTaskRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_policy" "ecs_task_policy" {
  name        = "ECSTaskPolicy"
  description = "Allows ECS task to access S3"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:*"]
        Resource = [
          "arn:aws:s3:::${aws_s3_bucket.template_project_data.bucket}",
          "arn:aws:s3:::${aws_s3_bucket.template_project_data.bucket}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_policy_attachment" "ecs_task_policy_attachment" {
  name       = "ECSTaskAttachment"
  roles      = [aws_iam_role.ecs_task_role.name]
  policy_arn = aws_iam_policy.ecs_task_policy.arn
}

resource "aws_cloudwatch_log_group" "template_project_ecs_log_group" {
  name              = "/ecs/bmin5100-example"
  retention_in_days = 30  # Optional: Set log retention
}

resource "aws_ecs_task_definition" "template_project_task_definition" {
  family                   = "bmin5100_example_task_definition"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "512"  # Adjust CPU
  memory                   = "1024" # Adjust Memory
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_task_role.arn

  ephemeral_storage {
    size_in_gib = 50 # Increase storage beyond default 20GB
  }

  container_definitions = jsonencode([
    {
      name      = "bmin5100-example-project-container"
      image     = aws_ecr_repository.template_project_ecr_repository.repository_url
      cpu       = 512
      memory    = 1024
      essential = true

      environment = [
        { name = "S3_BUCKET", value = "${aws_s3_bucket.template_project_data.id}" },
        { name = "INPUT_DIR", value = "/data/input" },
        { name = "OUTPUT_DIR", value = "/data/output" },
        { name = "ENVIRONMENT", value = "ECS" }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.template_project_ecs_log_group.name
          awslogs-region        = data.aws_region.current_region.name
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])
}
