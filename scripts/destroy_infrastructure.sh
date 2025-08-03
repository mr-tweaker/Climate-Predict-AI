#!/bin/bash

# ClimatePredict AI - Infrastructure Destruction Script
# This script safely destroys all AWS resources to avoid charges

set -e  # Exit on any error

echo "🚨 CLIMATEPREDICT AI - INFRASTRUCTURE DESTRUCTION"
echo "=================================================="
echo ""
echo "⚠️  WARNING: This will destroy ALL AWS resources for ClimatePredict AI"
echo "⚠️  This action cannot be undone!"
echo ""
echo "Resources that will be destroyed:"
echo "- ECS Cluster and Service"
echo "- Application Load Balancer"
echo "- VPC and Security Groups"
echo "- S3 Bucket (models and data)"
echo "- ECR Repository"
echo "- CloudWatch Logs and Alarms"
echo "- Secrets Manager secrets"
echo "- SNS Topics"
echo "- IAM Roles and Policies"
echo "- Lambda Functions (if enabled)"
echo ""
read -p "Are you sure you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "❌ Destruction cancelled by user"
    exit 1
fi

echo ""
echo "🔄 Starting infrastructure destruction..."
echo ""

# Change to terraform directory
cd terraform

# Check if terraform is initialized
if [ ! -d ".terraform" ]; then
    echo "🔧 Initializing Terraform..."
    terraform init
fi

# Plan the destruction to see what will be destroyed
echo "📋 Planning destruction..."
terraform plan -destroy -out=tfplan-destroy

echo ""
echo "📋 Resources that will be destroyed:"
terraform show tfplan-destroy | grep -E "^-|^\+" | grep -v "data\." | head -20
echo "..."

read -p "Proceed with destruction? (yes/no): " confirm_destroy

if [ "$confirm_destroy" != "yes" ]; then
    echo "❌ Destruction cancelled by user"
    exit 1
fi

echo ""
echo "🗑️  Destroying infrastructure..."

# Destroy the infrastructure
terraform destroy -auto-approve

echo ""
echo "✅ Infrastructure destruction completed!"
echo ""
echo "📊 Summary of destroyed resources:"
echo "- ECS Cluster: climatepredict-cluster"
echo "- ECS Service: climatepredict-service"
echo "- ALB: climatepredict-alb"
echo "- VPC: climatepredict-vpc"
echo "- S3 Bucket: climatepredict-models-*"
echo "- ECR Repository: climatepredict-ai"
echo "- CloudWatch: Logs and Alarms"
echo "- Secrets Manager: API keys"
echo "- SNS: Alerts topic"
echo "- IAM: Roles and policies"
echo ""
echo "💰 You should no longer incur charges for these resources"
echo ""
echo "💡 To verify, check your AWS Console or run:"
echo "   aws ecs list-clusters"
echo "   aws s3 ls"
echo "   aws ecr describe-repositories"
echo ""
echo "🎉 Infrastructure cleanup completed successfully!" 