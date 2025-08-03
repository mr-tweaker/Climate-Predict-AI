#!/bin/bash

# ClimatePredict AI - Deployment Script
# This script automates the deployment of the ClimatePredict AI application to AWS

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="climatepredict-ai"
AWS_REGION="us-east-1"
ECR_REPOSITORY="climatepredict-ai"
ECS_CLUSTER="climatepredict-cluster"
ECS_SERVICE="climatepredict-service"
ECS_TASK_DEFINITION="climatepredict-task"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check AWS CLI configuration
check_aws_config() {
    print_status "Checking AWS CLI configuration..."
    
    if ! command_exists aws; then
        print_error "AWS CLI is not installed. Please install it first."
        exit 1
    fi
    
    if ! aws sts get-caller-identity >/dev/null 2>&1; then
        print_error "AWS CLI is not configured. Please run 'aws configure' first."
        exit 1
    fi
    
    print_success "AWS CLI is properly configured"
}

# Function to check Docker
check_docker() {
    print_status "Checking Docker installation..."
    
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    # Try to run docker info, if it fails, try with sudo
    if ! docker info >/dev/null 2>&1; then
        print_warning "Docker requires sudo permissions. Using sudo for Docker commands."
        DOCKER_CMD="sudo docker"
    else
        DOCKER_CMD="docker"
    fi
    
    # Test if Docker is working
    if ! $DOCKER_CMD info >/dev/null 2>&1; then
        print_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    print_success "Docker is properly configured (using: $DOCKER_CMD)"
}

# Function to deploy infrastructure with Terraform
deploy_infrastructure() {
    print_status "Deploying infrastructure with Terraform..."
    
    cd terraform
    
    # Initialize Terraform
    print_status "Initializing Terraform..."
    terraform init
    
    # Plan the deployment
    print_status "Planning Terraform deployment..."
    terraform plan -out=tfplan
    
    # Apply the plan
    print_status "Applying Terraform plan..."
    terraform apply tfplan
    
    # Get outputs
    print_status "Getting Terraform outputs..."
    ECR_REPOSITORY_URL=$(terraform output -raw ecr_repository_url)
    ALB_DNS_NAME=$(terraform output -raw alb_dns_name)
    
    cd ..
    
    print_success "Infrastructure deployed successfully"
    print_status "ECR Repository: $ECR_REPOSITORY_URL"
    print_status "ALB DNS: $ALB_DNS_NAME"
}

# Function to build and push Docker image
build_and_push_image() {
    print_status "Building and pushing Docker image..."
    
    # Get ECR repository URL from Terraform output
    if [ -z "$ECR_REPOSITORY_URL" ]; then
        print_status "Getting ECR repository URL from Terraform..."
        ECR_REPOSITORY_URL=$(cd terraform && terraform output -raw ecr_repository_url && cd ..)
        if [ -z "$ECR_REPOSITORY_URL" ]; then
            print_error "Failed to get ECR repository URL from Terraform"
            exit 1
        fi
    fi
    
    # Get ECR login token
    print_status "Logging into ECR..."
    aws ecr get-login-password --region $AWS_REGION | $DOCKER_CMD login --username AWS --password-stdin $ECR_REPOSITORY_URL
    
    # Build Docker image
    print_status "Building Docker image..."
    $DOCKER_CMD build -t $PROJECT_NAME .
    
    # Tag images
    print_status "Tagging Docker images..."
    $DOCKER_CMD tag $PROJECT_NAME:latest $ECR_REPOSITORY_URL:latest
    
    # Generate a simple tag if git is not available
    if git rev-parse --git-dir > /dev/null 2>&1; then
        GIT_TAG=$(git rev-parse --short HEAD)
    else
        GIT_TAG=$(date +%Y%m%d-%H%M%S)
    fi
    
    $DOCKER_CMD tag $PROJECT_NAME:latest $ECR_REPOSITORY_URL:$GIT_TAG
    
    # Push images
    print_status "Pushing Docker images to ECR..."
    $DOCKER_CMD push $ECR_REPOSITORY_URL:latest
    $DOCKER_CMD push $ECR_REPOSITORY_URL:$GIT_TAG
    
    print_success "Docker images pushed successfully"
}

# Function to deploy application to ECS
deploy_application() {
    print_status "Deploying application to ECS..."
    
    # Update ECS service
    print_status "Updating ECS service..."
    aws ecs update-service \
        --cluster $ECS_CLUSTER \
        --service $ECS_SERVICE \
        --force-new-deployment
    
    # Wait for service to be stable
    print_status "Waiting for ECS service to be stable..."
    aws ecs wait services-stable \
        --cluster $ECS_CLUSTER \
        --services $ECS_SERVICE
    
    print_success "Application deployed successfully"
}

# Function to set up API keys in Secrets Manager
setup_secrets() {
    print_status "Setting up API keys in Secrets Manager..."
    
    # Check if secrets exist
    if [ -z "$GOOGLE_API_KEY" ]; then
        print_warning "GOOGLE_API_KEY environment variable not set. Skipping Google API key setup."
    else
        print_status "Setting up Google API key..."
        aws secretsmanager put-secret-value \
            --secret-id climatepredict/google-api-key \
            --secret-string "$GOOGLE_API_KEY" \
            --region $AWS_REGION
    fi
    
    if [ -z "$OPENWEATHER_API_KEY" ]; then
        print_warning "OPENWEATHER_API_KEY environment variable not set. Skipping OpenWeather API key setup."
    else
        print_status "Setting up OpenWeather API key..."
        aws secretsmanager put-secret-value \
            --secret-id climatepredict/openweather-api-key \
            --secret-string "$OPENWEATHER_API_KEY" \
            --region $AWS_REGION
    fi
    
    print_success "API keys configured successfully"
}

# Function to upload models to S3
upload_models() {
    print_status "Uploading models to S3..."
    
    # Get S3 bucket name from Terraform output
    S3_BUCKET=$(cd terraform && terraform output -raw s3_bucket_name && cd ..)
    
    # Upload models directory
    if [ -d "models" ]; then
        print_status "Uploading models directory..."
        aws s3 sync models/ s3://$S3_BUCKET/models/ --region $AWS_REGION
        print_success "Models uploaded successfully"
    else
        print_warning "Models directory not found. Skipping model upload."
    fi
    
    # Upload data directory
    if [ -d "data" ]; then
        print_status "Uploading data directory..."
        aws s3 sync data/ s3://$S3_BUCKET/data/ --region $AWS_REGION
        print_success "Data uploaded successfully"
    else
        print_warning "Data directory not found. Skipping data upload."
    fi
}

# Function to test deployment
test_deployment() {
    print_status "Testing deployment..."
    
    # Get ALB DNS name
    ALB_DNS_NAME=$(cd terraform && terraform output -raw alb_dns_name && cd ..)
    
    # Wait for application to be ready
    print_status "Waiting for application to be ready..."
    sleep 30
    
    # Test health endpoint
    print_status "Testing health endpoint..."
    if curl -f http://$ALB_DNS_NAME/_stcore/health >/dev/null 2>&1; then
        print_success "Application is healthy"
        print_success "🌐 Application URL: http://$ALB_DNS_NAME"
    else
        print_error "Application health check failed"
        exit 1
    fi
}

# Function to show deployment status
show_status() {
    print_status "Showing deployment status..."
    
    # Get outputs
    ALB_DNS_NAME=$(cd terraform && terraform output -raw alb_dns_name && cd ..)
    ECR_REPOSITORY_URL=$(cd terraform && terraform output -raw ecr_repository_url && cd ..)
    S3_BUCKET=$(cd terraform && terraform output -raw s3_bucket_name && cd ..)
    
    echo ""
    echo "🚀 ClimatePredict AI Deployment Status"
    echo "======================================"
    echo "🌐 Application URL: http://$ALB_DNS_NAME"
    echo "🐳 ECR Repository: $ECR_REPOSITORY_URL"
    echo "📦 S3 Bucket: $S3_BUCKET"
    echo "🔧 ECS Cluster: $ECS_CLUSTER"
    echo "⚙️ ECS Service: $ECS_SERVICE"
    echo ""
    
    # Show ECS service status
    print_status "ECS Service Status:"
    aws ecs describe-services \
        --cluster $ECS_CLUSTER \
        --services $ECS_SERVICE \
        --query 'services[0].{Status:status,RunningCount:runningCount,DesiredCount:desiredCount}' \
        --output table
}

# Main deployment function
main() {
    echo "🚀 ClimatePredict AI - Deployment Script"
    echo "========================================"
    echo ""
    
    # Check prerequisites
    check_aws_config
    check_docker
    
    # Parse command line arguments
    case "${1:-deploy}" in
        "infrastructure")
            deploy_infrastructure
            ;;
        "build")
            build_and_push_image
            ;;
        "deploy")
            deploy_infrastructure
            build_and_push_image
            deploy_application
            setup_secrets
            upload_models
            test_deployment
            show_status
            ;;
        "status")
            show_status
            ;;
        "secrets")
            setup_secrets
            ;;
        "models")
            upload_models
            ;;
        "test")
            test_deployment
            ;;
        *)
            echo "Usage: $0 {infrastructure|build|deploy|status|secrets|models|test}"
            echo ""
            echo "Commands:"
            echo "  infrastructure  - Deploy only infrastructure with Terraform"
            echo "  build          - Build and push Docker image only"
            echo "  deploy         - Full deployment (default)"
            echo "  status         - Show deployment status"
            echo "  secrets        - Set up API keys in Secrets Manager"
            echo "  models         - Upload models to S3"
            echo "  test           - Test the deployment"
            exit 1
            ;;
    esac
    
    print_success "Deployment completed successfully!"
}

# Run main function
main "$@" 