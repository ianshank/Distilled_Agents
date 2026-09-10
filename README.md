# Distilled Agents

A comprehensive system for training, deploying, and managing distilled AI agents with enhanced inference capabilities.

## Project Structure

```
Distilled_Agents/
├── Makefile                       # install, lint, typecheck, test-aqa, validate
├── pyproject.toml                 # Installable package, pytest, ruff, coverage
├── AGENTS.md                      # Which skill/CLI to use
├── CHANGELOG.md
├── .github/                       # CI, composite validate action, CODEOWNERS
├── .cursor/skills/                # mangomas-train/evaluate/launch/scan
├── configs/                       # Hoisted YAML + agent profiles
├── config/                        # SageMaker image requirements
│   └── requirements.txt
├── data/training/                 # Training datasets (.jsonl)
├── docs/                          # Guides, ADRs, C4 architecture
├── enhanced_system/               # Installable inference library
│   ├── core/
│   ├── ops/                       # Shared SageMaker launcher + settings
│   └── tests/
├── scripts/                       # Thin CLIs over shared modules
└── tests/                         # Root pytest + skill/CLI harness
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
make install
# or: pip install -e ".[dev]"
```

SageMaker training images still use `config/requirements.txt` (ML + AWS only). Copy `.env.example` to `.env` for local overrides. Use an IAM role or `aws login`; do not commit access keys.

CI runs `make validate` via `.github/actions/mangomas-validate`. ADRs live in `docs/adr/`. License: MIT.

### Quick Start

#### Running Inference
```bash
python scripts/inference.py
```

#### Training an Agent
```bash
python scripts/training/train_agent_skill.py --help
```

See `.cursor/skills/mangomas-train/SKILL.md` and `AGENTS.md`.

#### Deploying to SageMaker
```bash
python scripts/deployment/simple_launch_sagemaker.py --help
```

## Documentation

- [C4 context](docs/architecture/c4-context.md)
- [C4 container](docs/architecture/c4-container.md)
- [C4 component (core + ops)](docs/architecture/c4-component.md)
- [Next steps](docs/NEXT_STEPS.md)
- [Changelog](CHANGELOG.md)
- [Agent Distillation Guide](docs/README_AGENT_DISTILLATION.md)
- [SageMaker Training Guide](docs/README_SAGEMAKER_TRAINING.md)
- [SageMaker Launcher Guide](docs/README_SAGEMAKER_LAUNCHER.md)

## Testing and pre-PR validation

```bash
make validate
```

That runs ruff, mypy (`enhanced_system/ops` + cache), pytest `-m "unit or integration or regression"` with `--cov-fail-under=65`, bandit, and gitleaks.

```bash
make test-aqa
make lint
make typecheck
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

## License

MIT. See [LICENSE](LICENSE).
