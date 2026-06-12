#!/bin/bash
set -e
echo "=== Phase 6B: Training Bull/NonBull Regime Models ==="
python code/src/train.py
echo "=== Training Complete ==="
