#!/bin/bash
# Download completed scoring results from AWS

BUCKET="transcript-scoring-1770013499"
REGION="us-east-1"
OUTPUT_FILE="${1:-all_scored_transcripts_2015_2025.csv}"

echo "========================================"
echo "Downloading Results from AWS"
echo "========================================"
echo ""

# Use Python script for proper downloading and merging
python aws_monitor.py --bucket $BUCKET --download "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "Results saved to: $OUTPUT_FILE"
    echo ""
    echo "Quick stats:"
    wc -l "$OUTPUT_FILE"
    head -n 1 "$OUTPUT_FILE"
    echo "..."
    tail -n 3 "$OUTPUT_FILE"
else
    echo "Download failed. Check if jobs are complete:"
    echo "   bash check_progress.sh"
fi
