#!/bin/bash
set -e
echo "=== Building Docker Image bdc2026 ==="
docker build -t bdc2026 .
echo ""
echo "=== Exporting Image ==="
docker save bdc2026 -o bdc2026.tar
echo ""
echo "=== Done ==="
echo "Submit file: bdc2026.tar"
echo "Load with: docker load -i bdc2026.tar"
ls -lh bdc2026.tar
