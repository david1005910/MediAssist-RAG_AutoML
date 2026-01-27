#!/bin/bash
# Run all tests with coverage

set -e

echo "Running tests..."

# Run pytest with coverage
pytest tests/ \
    --cov=. \
    --cov-report=term-missing \
    --cov-report=xml \
    --cov-fail-under=80 \
    -v

echo "Tests completed!"
