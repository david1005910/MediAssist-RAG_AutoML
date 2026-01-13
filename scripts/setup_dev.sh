#!/bin/bash
# Development environment setup script

set -e

echo "Setting up MediAssist AI development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
if [[ "$python_version" < "3.11" ]]; then
    echo "Error: Python 3.11+ is required"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov pytest-asyncio
pip install ruff black mypy
pip install pre-commit

# Setup pre-commit hooks
echo "Setting up pre-commit hooks..."
pre-commit install

# Start Docker services
echo "Starting Docker services..."
docker-compose up -d postgres redis chromadb

# Wait for services
echo "Waiting for services to be ready..."
sleep 5

# Run migrations
echo "Running database migrations..."
# alembic upgrade head

echo "Development environment setup complete!"
echo "Activate the virtual environment with: source venv/bin/activate"
