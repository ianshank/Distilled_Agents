#!/bin/bash

# Quick Start Script for Enhanced Agent System
# Runs a simple demo to verify the system works

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Enhanced Agent System - Quick Start                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "✗ Python 3 is required but not installed"
    exit 1
fi

echo "✓ Python 3 found"
echo ""

# Install dependencies if needed
echo "Installing dependencies (this may take a moment)..."
python3 -m pip install -q pydantic pyyaml numpy &> /dev/null || true
echo "✓ Dependencies ready"
echo ""

# Run basic demo
echo "Running basic inference demo..."
echo "─────────────────────────────────────────────────────────────"
echo ""

python3 enhanced_system/examples/basic_inference.py

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    Quick Start Complete!                     ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║ What's next?                                                 ║"
echo "║                                                              ║"
echo "║ 1. Full setup: ./enhanced_system/scripts/setup_infrastructure.sh ║"
echo "║ 2. Start services: cd infrastructure/docker && docker-compose up  ║"
echo "║ 3. Run tests: pytest enhanced_system/tests/ -v              ║"
echo "║ 4. Read docs: cat enhanced_system/README.md                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"

