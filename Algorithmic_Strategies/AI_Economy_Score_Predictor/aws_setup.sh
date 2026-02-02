#!/bin/bash
# AWS Setup Script for Transcript Scoring Infrastructure

set -e

echo "=========================================="
echo "AWS Transcript Scoring Setup"
echo "=========================================="

# Configuration
BUCKET_NAME="${BUCKET_NAME:-transcript-scoring-$(date +%s)}"
REGION="${AWS_REGION:-us-east-1}"
TABLE_NAME="transcript-scoring-jobs"

echo ""
echo "Configuration:"
echo "  Bucket: $BUCKET_NAME"
echo "  Region: $REGION"
echo "  DynamoDB Table: $TABLE_NAME"
echo ""

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "ERROR: AWS CLI is not installed"
    echo "Install it with: pip install awscli"
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo "ERROR: AWS credentials not configured"
    echo "Run: aws configure"
    exit 1
fi

echo "✓ AWS CLI configured"

# Create S3 bucket
echo ""
echo "Creating S3 bucket: $BUCKET_NAME"
if aws s3 ls "s3://$BUCKET_NAME" 2>&1 | grep -q 'NoSuchBucket'; then
    if [ "$REGION" = "us-east-1" ]; then
        aws s3api create-bucket --bucket "$BUCKET_NAME" --region "$REGION"
    else
        aws s3api create-bucket \
            --bucket "$BUCKET_NAME" \
            --region "$REGION" \
            --create-bucket-configuration LocationConstraint="$REGION"
    fi
    echo "✓ Created S3 bucket: $BUCKET_NAME"
else
    echo "✓ S3 bucket already exists: $BUCKET_NAME"
fi

# Create bucket folders
echo ""
echo "Creating S3 folder structure..."
aws s3api put-object --bucket "$BUCKET_NAME" --key "input/batches/"
aws s3api put-object --bucket "$BUCKET_NAME" --key "output/scored/"
aws s3api put-object --bucket "$BUCKET_NAME" --key "code/"
echo "✓ Created folder structure"

# Create DynamoDB table
echo ""
echo "Creating DynamoDB table: $TABLE_NAME"
if ! aws dynamodb describe-table --table-name "$TABLE_NAME" --region "$REGION" &> /dev/null; then
    aws dynamodb create-table \
        --table-name "$TABLE_NAME" \
        --attribute-definitions \
            AttributeName=job_id,AttributeType=S \
        --key-schema \
            AttributeName=job_id,KeyType=HASH \
        --billing-mode PAY_PER_REQUEST \
        --region "$REGION"
    
    echo "Waiting for table to be active..."
    aws dynamodb wait table-exists --table-name "$TABLE_NAME" --region "$REGION"
    echo "✓ Created DynamoDB table: $TABLE_NAME"
else
    echo "✓ DynamoDB table already exists: $TABLE_NAME"
fi

# Create IAM role for EC2 instances
echo ""
echo "Creating IAM role for EC2 instances..."
ROLE_NAME="transcript-scorer-ec2-role"

# Check if role exists
if ! aws iam get-role --role-name "$ROLE_NAME" &> /dev/null; then
    # Create trust policy
    cat > /tmp/trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

    # Create role
    aws iam create-role \
        --role-name "$ROLE_NAME" \
        --assume-role-policy-document file:///tmp/trust-policy.json
    
    # Create policy
    cat > /tmp/role-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::$BUCKET_NAME",
        "arn:aws:s3:::$BUCKET_NAME/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:UpdateItem",
        "dynamodb:Scan",
        "dynamodb:Query"
      ],
      "Resource": "arn:aws:dynamodb:$REGION:*:table/$TABLE_NAME"
    }
  ]
}
EOF

    # Attach policy to role
    aws iam put-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-name "transcript-scorer-policy" \
        --policy-document file:///tmp/role-policy.json
    
    # Create instance profile
    aws iam create-instance-profile --instance-profile-name "$ROLE_NAME-profile"
    aws iam add-role-to-instance-profile \
        --instance-profile-name "$ROLE_NAME-profile" \
        --role-name "$ROLE_NAME"
    
    echo "✓ Created IAM role: $ROLE_NAME"
else
    echo "✓ IAM role already exists: $ROLE_NAME"
fi

# Package and upload worker code
echo ""
echo "Packaging worker code..."
zip -r /tmp/worker-code.zip \
    aws_worker.py \
    llm_scorer.py \
    config.yaml \
    requirements.txt \
    -x "*.pyc" "__pycache__/*" "*.ipynb"

echo "Uploading worker code to S3..."
aws s3 cp /tmp/worker-code.zip "s3://$BUCKET_NAME/code/worker-code.zip"
echo "✓ Uploaded worker code"

# Create .env file with configuration
echo ""
echo "Creating .env file with AWS configuration..."
cat > .env.aws <<EOF
# AWS Configuration for Transcript Scoring
AWS_BUCKET_NAME=$BUCKET_NAME
AWS_REGION=$REGION
AWS_DYNAMODB_TABLE=$TABLE_NAME
AWS_IAM_ROLE=$ROLE_NAME
AWS_INSTANCE_PROFILE=$ROLE_NAME-profile

# Usage:
# source .env.aws
# python aws_job_submitter.py --bucket \$AWS_BUCKET_NAME --input-file transcripts.csv
EOF

echo "✓ Created .env.aws"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Load AWS configuration:"
echo "   source .env.aws"
echo ""
echo "2. Submit a scoring job:"
echo "   python aws_job_submitter.py \\"
echo "     --bucket \$AWS_BUCKET_NAME \\"
echo "     --input-file scoring_transcripts.csv \\"
echo "     --batch-size 50 \\"
echo "     --launch-instances \\"
echo "     --num-instances 2"
echo ""
echo "3. Monitor job status:"
echo "   python aws_monitor.py --bucket \$AWS_BUCKET_NAME"
echo ""
echo "4. Download results:"
echo "   python aws_download_results.py --bucket \$AWS_BUCKET_NAME"
echo ""
echo "Resources created:"
echo "  - S3 Bucket: s3://$BUCKET_NAME"
echo "  - DynamoDB Table: $TABLE_NAME"
echo "  - IAM Role: $ROLE_NAME"
echo ""
