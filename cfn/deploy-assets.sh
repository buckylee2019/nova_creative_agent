#!/bin/bash

STACK_NAME="dpa-agent-assets"
TEMPLATE_FILE="assets-infrastructure.yaml"

echo "🚀 Deploying DPA Agent Assets Infrastructure..."

aws cloudformation deploy \
  --template-file $TEMPLATE_FILE \
  --stack-name $STACK_NAME \
  --capabilities CAPABILITY_IAM \
  --no-fail-on-empty-changeset

if [ $? -eq 0 ]; then
    echo "✅ Stack deployed successfully!"
    
    echo "📋 Getting outputs..."
    BUCKET_NAME=$(aws cloudformation describe-stacks --stack-name $STACK_NAME --query 'Stacks[0].Outputs[?OutputKey==`BucketName`].OutputValue' --output text)
    CLOUDFRONT_DOMAIN=$(aws cloudformation describe-stacks --stack-name $STACK_NAME --query 'Stacks[0].Outputs[?OutputKey==`CloudFrontDomain`].OutputValue' --output text)
    DISTRIBUTION_ID=$(aws cloudformation describe-stacks --stack-name $STACK_NAME --query 'Stacks[0].Outputs[?OutputKey==`DistributionId`].OutputValue' --output text)
    
    echo ""
    echo "Configuration:"
    echo "S3_BUCKET_NAME=$BUCKET_NAME"
    echo "CLOUDFRONT_DOMAIN=$CLOUDFRONT_DOMAIN"
    echo "CLOUDFRONT_DISTRIBUTION_ID=$DISTRIBUTION_ID"
    
    # Append to .env file
    echo "" >> .env
    echo "# DPA Agent Assets Configuration" >> .env
    echo "S3_BUCKET_NAME=$BUCKET_NAME" >> .env
    echo "CLOUDFRONT_DOMAIN=$CLOUDFRONT_DOMAIN" >> .env
    echo "CLOUDFRONT_DISTRIBUTION_ID=$DISTRIBUTION_ID" >> .env
    
    echo "✅ Configuration added to .env file"
else
    echo "❌ Deployment failed"
    exit 1
fi
