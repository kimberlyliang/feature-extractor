output "template_project_ecr_repository_url" {
  value = aws_ecr_repository.template_project_ecr_repository.repository_url
}

output "template_project_ecs_task_definition_arn" {
  value = aws_ecs_task_definition.template_project_task_definition.arn
}

output "template_project_ecs_task_definition_id" {
  value = aws_ecs_task_definition.template_project_task_definition.id
}
