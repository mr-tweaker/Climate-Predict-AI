#!/bin/bash

# ClimatePredict AI - Cleanup Remaining Resources
# This script cleans up S3 bucket and ECR repository before destruction

set -e  # Exit on any error

echo "🧹 CLEANING UP REMAINING AWS RESOURCES"
echo "======================================"
echo ""

# S3 Bucket cleanup
echo "🗑️  Cleaning up S3 bucket: climatepredict-models-ge9t4296"
echo "This will delete all your trained models..."
echo ""

# Delete all objects in S3 bucket
aws s3 rm s3://climatepredict-models-ge9t4296 --recursive

echo "✅ S3 bucket cleaned successfully"
echo ""

# ECR Repository cleanup
echo "🗑️  Cleaning up ECR repository: climatepredict-ai"
echo "This will delete all Docker images..."
echo ""

# Get all image digests
IMAGE_DIGESTS=$(aws ecr list-images --repository-name climatepredict-ai --query 'imageIds[*].imageDigest' --output text)

if [ ! -z "$IMAGE_DIGESTS" ]; then
    # Delete all images
    aws ecr batch-delete-image \
        --repository-name climatepredict-ai \
        --image-ids $(echo $IMAGE_DIGESTS | tr ' ' '\n' | sed 's/^/imageDigest=/')
    
    echo "✅ ECR repository cleaned successfully"
else
    echo "ℹ️  No images found in ECR repository"
fi

echo ""
echo "🎉 Cleanup completed! Now you can run terraform destroy"
echo ""
echo "💡 Next steps:"
echo "   1. Run: terraform destroy"
echo "   2. This will now succeed without errors"
echo "" 