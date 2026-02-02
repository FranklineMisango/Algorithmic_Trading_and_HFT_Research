#!/bin/bash
# Quick progress checker for AWS transcript scoring jobs

BUCKET="transcript-scoring-1770013499"
REGION="us-east-1"

echo "========================================"
echo "AWS Transcript Scoring - Status Check"
echo "========================================"
echo ""

# Check DynamoDB for job counts
echo "Job Status:"
aws dynamodb scan \
    --table-name transcript-scoring-jobs \
    --region $REGION \
    --select COUNT \
    --filter-expression "attribute_exists(#s)" \
    --expression-attribute-names '{"#s":"status"}' \
    --output text | awk '{print "  Total jobs: " $1}'

COMPLETED=$(aws dynamodb scan \
    --table-name transcript-scoring-jobs \
    --region $REGION \
    --select COUNT \
    --filter-expression "#s = :status" \
    --expression-attribute-names '{"#s":"status"}' \
    --expression-attribute-values '{":status":{"S":"completed"}}' \
    --output text | awk '{print $1}')

echo "  Completed: $COMPLETED"

PENDING=$(aws dynamodb scan \
    --table-name transcript-scoring-jobs \
    --region $REGION \
    --select COUNT \
    --filter-expression "#s = :status" \
    --expression-attribute-names '{"#s":"status"}' \
    --expression-attribute-values '{":status":{"S":"pending"}}' \
    --output text | awk '{print $1}')

echo "  Pending: $PENDING"

FAILED=$(aws dynamodb scan \
    --table-name transcript-scoring-jobs \
    --region $REGION \
    --select COUNT \
    --filter-expression "#s = :status" \
    --expression-attribute-names '{"#s":"status"}' \
    --expression-attribute-values '{":status":{"S":"failed"}}' \
    --output text | awk '{print $1}')

echo "  Failed: $FAILED"

# Check S3 for output files
echo ""
echo "Output Files:"
OUTPUT_COUNT=$(aws s3 ls s3://$BUCKET/output/scored/ --region $REGION --recursive | wc -l)
echo "  Files in S3: $OUTPUT_COUNT"

# Check EC2 instances
echo ""
echo "Worker Instances:"
aws ec2 describe-instances \
    --region $REGION \
    --filters "Name=tag:Name,Values=transcript-scorer-worker" "Name=instance-state-name,Values=running,pending" \
    --query 'Reservations[*].Instances[*].[InstanceId,State.Name,InstanceType]' \
    --output table

echo ""
echo "========================================"
if [ "$PENDING" -eq "0" ] && [ "$COMPLETED" -gt "0" ]; then
    echo "All jobs complete! Download results:"
    echo "   python aws_monitor.py --bucket $BUCKET --download all_scored_transcripts_2015_2025.csv"
else
    echo "   Still processing... Check again later"
    echo "   bash check_progress.sh"
fi
echo "========================================"
