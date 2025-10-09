# Distilled Agents

A comprehensive system for training, deploying, and managing distilled AI agents with enhanced inference capabilities.

## Project Structure

```
Distilled_Agents/
├── config/                        # Configuration files
│   └── requirements.txt          # Python dependencies
├── data/                          # Training and test data
│   ├── training/                 # Training datasets (.jsonl files)
│   │   ├── architect_agent.jsonl
│   │   ├── architect_agent_real_data.jsonl
│   │   ├── devops_agent.jsonl
│   │   ├── product_manager_agent_real_data.jsonl
│   │   ├── sample_training_data.jsonl
│   │   ├── sqe_agent.jsonl
│   │   ├── sqe_agent_real_data.jsonl
│   │   ├── swe_agent.jsonl
│   │   ├── swe_agent_real_data.jsonl
│   │   ├── tools_agent.jsonl
│   │   └── vp_product_agent.jsonl
│   └── samples/                  # Sample data for testing
├── docs/                          # Documentation
│   ├── README_AGENT_DISTILLATION.md
│   ├── README_SAGEMAKER_LAUNCHER.md
│   ├── README_SAGEMAKER_TRAINING.md
│   ├── IMPLEMENTATION_SUMMARY.md
│   ├── REFACTORING_SUMMARY.md
│   └── TEST_FIXES_NEEDED.md
├── enhanced_system/               # Core enhanced inference system
│   ├── config/                   # System configuration
│   ├── core/                     # Core components
│   │   ├── adaptive_router.py
│   │   ├── cache_manager.py
│   │   ├── confidence_calibrator.py
│   │   ├── consensus_inference.py
│   │   ├── error_handler.py
│   │   ├── input_validator.py
│   │   └── monitoring.py
│   ├── evaluation/               # Evaluation tools
│   └── tests/                    # Unit and integration tests
│       ├── unit/
│       └── integration/
├── scripts/                       # Operational scripts
│   ├── deployment/               # Deployment scripts
│   │   ├── launch_all_agents_sagemaker.py
│   │   ├── launch_sagemaker_training_jobs.py
│   │   ├── run_sagemaker_training.py
│   │   ├── sagemaker_distillation_job.py
│   │   ├── simple_launch_sagemaker.py
│   │   └── simple_launch_sagemaker_cpu.py
│   ├── evaluation/               # Evaluation scripts
│   │   ├── evaluate_agent_skill.py
│   │   └── register_agent_skill.py
│   ├── infrastructure/           # Infrastructure setup
│   │   ├── notify_failure.py
│   │   ├── security_scan.py
│   │   ├── setup_infrastructure.py
│   │   └── verify_infrastructure.py
│   ├── training/                 # Training scripts
│   │   ├── train_agent_skill.py
│   │   ├── train_distilled_adapter.py
│   │   └── train_software_development_agent.py
│   ├── generate_deployment_summary.py
│   ├── inference.py              # Main inference script
│   ├── package_to_onnx.py
│   └── validate_dataset.py
└── tests/                         # Root-level tests
    ├── quick_test.py
    └── test_software_dev_agent.py

```

## Getting Started

### Installation

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r config/requirements.txt
```

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

[Add your license information here]
