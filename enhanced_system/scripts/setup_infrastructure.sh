#!/bin/bash

# Setup Infrastructure Script for Enhanced Agent System
# This script sets up the necessary infrastructure components

set -e  # Exit on error

echo "=========================================="
echo "Enhanced Agent System - Infrastructure Setup"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if running from correct directory
if [ ! -d "enhanced_system" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Step 1: Check prerequisites
echo "1. Checking prerequisites..."

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_status "Python ${PYTHON_VERSION} found"
else
    print_error "Python 3 is required but not installed"
    exit 1
fi

# Check Docker
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | tr -d ',')
    print_status "Docker ${DOCKER_VERSION} found"
else
    print_warning "Docker not found - containerized deployment will not be available"
fi

# Check Docker Compose
if command -v docker-compose &> /dev/null; then
    print_status "Docker Compose found"
else
    print_warning "Docker Compose not found - use 'docker compose' (v2) instead"
fi

echo ""

# Step 2: Create directories
echo "2. Creating directories..."

mkdir -p enhanced_system/logs
mkdir -p enhanced_system/data
mkdir -p enhanced_system/models
mkdir -p enhanced_system/tests/unit
mkdir -p enhanced_system/tests/integration
mkdir -p enhanced_system/tests/fixtures

print_status "Directories created"
echo ""

# Step 3: Install Python dependencies
echo "3. Installing Python dependencies..."

if [ -f "enhanced_system/requirements.txt" ]; then
    python3 -m pip install -r enhanced_system/requirements.txt
    print_status "Python dependencies installed"
else
    print_warning "requirements.txt not found"
fi

echo ""

# Step 4: Setup configuration
echo "4. Setting up configuration..."

# Copy default config if not exists
if [ ! -f "enhanced_system/config/local.yaml" ]; then
    cp enhanced_system/config/development.yaml enhanced_system/config/local.yaml
    print_status "Local configuration created from development.yaml"
fi

echo ""

# Step 5: Initialize databases
echo "5. Initializing databases..."

# Create error tracking database
python3 -c "
import sqlite3
import os

db_path = 'enhanced_system/data/errors.db'
os.makedirs(os.path.dirname(db_path), exist_ok=True)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS errors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        error_type TEXT,
        error_message TEXT,
        agent TEXT,
        task TEXT,
        timestamp TEXT,
        retry_count INTEGER,
        resolved INTEGER,
        resolution_method TEXT
    )
''')

conn.commit()
conn.close()

print('Error database initialized')
"

print_status "Error tracking database initialized"
echo ""

# Step 6: Start Docker services (optional)
if command -v docker &> /dev/null; then
    echo "6. Docker services setup..."
    echo "   To start services, run:"
    echo "   cd enhanced_system/infrastructure/docker"
    echo "   docker-compose up -d"
    echo ""
    
    read -p "   Start Docker services now? (y/n) " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd enhanced_system/infrastructure/docker
        docker-compose up -d
        cd ../../..
        print_status "Docker services started"
    else
        print_warning "Skipped Docker services"
    fi
fi

echo ""

# Step 7: Run tests
echo "7. Running tests..."
echo "   To run tests, execute:"
echo "   pytest enhanced_system/tests/ -v"
echo ""

read -p "   Run tests now? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    pytest enhanced_system/tests/unit/ -v --tb=short || print_warning "Some tests failed"
else
    print_warning "Skipped tests"
fi

echo ""

# Final summary
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review configuration in enhanced_system/config/"
echo "2. Start services: cd enhanced_system/infrastructure/docker && docker-compose up -d"
echo "3. Run example: python3 enhanced_system/examples/basic_inference.py"
echo "4. View metrics: http://localhost:3000 (Grafana, admin/admin)"
echo "5. View logs: tail -f enhanced_system/logs/*.log"
echo ""
print_status "Enhanced Agent System is ready!"

