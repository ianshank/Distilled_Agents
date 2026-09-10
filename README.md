# Distilled Agents

A comprehensive system for training, deploying, and managing distilled AI agents with enhanced inference capabilities.

## Project Structure

```
Distilled_Agents/
├── pyproject.toml                 # Installable package, pytest, ruff, coverage
├── tox.ini                        # Root test/lint environments
├── LICENSE
├── .github/                       # CI, Dependabot, CODEOWNERS
├── configs/                       # Hoisted YAML + agent profiles
├── config/                        # SageMaker image requirements
│   └── requirements.txt
├── data/training/                 # Training datasets (.jsonl)
├── docs/                          # Guides and ADRs
│   └── adr/
├── enhanced_system/               # Installable inference library
│   ├── config/
│   ├── core/
│   ├── evaluation/
│   ├── ops/                       # Shared SageMaker launcher + settings
│   └── tests/
├── scripts/                       # Thin CLIs over shared modules
└── tests/                         # Root pytest (training-data smoke tests)
```

## Getting Started

### Installation

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install the package (library + test tools):
```bash
pip install -e ".[dev]"
```

SageMaker training images still use `config/requirements.txt` (ML + AWS only). Copy `.env.example` to `.env` for local overrides. Use an IAM role or `aws login`; do not commit access keys.

CI runs on GitHub Actions (`.github/workflows/ci.yml`). ADRs live in `docs/adr/`. License: MIT.

### Quick Start

#### Running Inference
```bash
python scripts/inference.py
```

#### Training an Agent
```bash
python scripts/training/train_agent_skill.py
```

#### Deploying to SageMaker
```bash
python scripts/deployment/simple_launch_sagemaker.py
```

## Documentation

Detailed documentation can be found in the `docs/` directory:

- [Agent Distillation Guide](docs/README_AGENT_DISTILLATION.md)
- [SageMaker Training Guide](docs/README_SAGEMAKER_TRAINING.md)
- [SageMaker Launcher Guide](docs/README_SAGEMAKER_LAUNCHER.md)
- [Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md)
- [Refactoring Summary](docs/REFACTORING_SUMMARY.md)

## Testing

Run all tests:
```bash
pytest
```

Run specific test suites:
```bash
# Unit tests
pytest enhanced_system/tests/unit/

# Integration tests
pytest enhanced_system/tests/integration/

# Root-level tests
pytest tests/
```

## Key Components

### Enhanced System
The `enhanced_system/` directory contains the core inference system with:
- Adaptive routing
- Intelligent caching
- Consensus inference
- Error handling and retries
- Input validation
- Performance monitoring

### Training Data
Agent training datasets are located in `data/training/` and include specialized agents for:
- Software Engineering (SWE)
- Quality Engineering (SQE)
- Architecture
- DevOps
- Product Management
- VP Product

### Scripts
Organized by function:
- **deployment/**: SageMaker deployment and job management
- **training/**: Model training and distillation
- **evaluation/**: Agent skill evaluation and registration
- **infrastructure/**: Setup, security, and verification

## Contributing

Please ensure all tests pass before submitting changes:
```bash
pytest --tb=short
```

## License

MIT. See [LICENSE](LICENSE).
