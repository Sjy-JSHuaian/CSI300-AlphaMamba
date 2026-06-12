#!/bin/bash
set -e
echo "=== Meta Ranker Prediction ==="
python code/src/test.py
echo "=== Result ==="
cat /app/output/result.csv
