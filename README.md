# Distilled Agents

A comprehensive system for training, deploying, and managing distilled AI agents with enhanced inference capabilities.

## Project Structure

```
Distilled_Agents/
├── Makefile                       # make validate (lint, mypy, pytest AQA, bandit, gitleaks)
├── pyproject.toml                 # Installable package, pytest, ruff, coverage
├── tox.ini                        # Root test/lint/harness environments
├── LICENSE
├── .github/                       # CI jobs + mangomas-validate composite (skills)
├── .cursor/skills/                # Operator Cursor skills
├── configs/                       # Hoisted YAML, agent profiles, harnesses/
├── config/                        # SageMaker image requirements
│   └── requirements.txt
├── data/training/                 # Training datasets (.jsonl)
├── docs/                          # Guides, ADRs 0001–0006, C4
│   ├── adr/
│   └── architecture/
├── enhanced_system/               # Installable inference library
│   ├── config/harnesses/          # Packaged harness YAML (keep in sync with configs/)
│   ├── core/
│   ├── harness/                   # Local runtime H (not SageMaker predict_fn)
│   ├── evaluation/
│   ├── ops/                       # Shared SageMaker launcher + settings
│   └── tests/
├── scripts/                       # Thin CLIs over shared modules
│   └── harness/                   # run_agent, collect_trajectories, tailor_harness
└── tests/                         # Root pytest (skills contract, training-data smoke)
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

CI runs on GitHub Actions (`.github/workflows/ci.yml`): lint, types (mypy), unit+integration@60, harness@90, security (bandit + gitleaks). The composite `.github/actions/mangomas-validate` wraps `make validate` for skills and does not replace those jobs. ADRs live in `docs/adr/` (0004 skills, 0005 gitleaks/mypy, 0006 harness). C4: `docs/architecture/`. License: MIT.

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

## Three surfaces

| Surface | Entry | Tools? |
| --- | --- | --- |
| Distill | `scripts/training/train_distilled_adapter.py` (`--trajectory_mode` local) | N/A (trains LoRA) |
| Serve | SageMaker `predict_fn` in `scripts/inference.py` | No (single-shot) |
| Harness H | `scripts/harness/run_agent.py` | Yes (local loop) |

Collect trusted JSONL with `scripts/harness/collect_trajectories.py` (injection detection off). Interactive tasks use `run_agent` (injection on).

## Documentation

Detailed documentation can be found in the `docs/` directory:

- [Agent Distillation Guide](docs/README_AGENT_DISTILLATION.md)
- [SageMaker Training Guide](docs/README_SAGEMAKER_TRAINING.md)
- [SageMaker Launcher Guide](docs/README_SAGEMAKER_LAUNCHER.md)
- [C4 architecture](docs/architecture/c4-context.md)
- [Next steps](docs/NEXT_STEPS.md)
- [Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md)
- [Refactoring Summary](docs/REFACTORING_SUMMARY.md)

## Testing

Pre-PR (from repo root):

```bash
make validate
pytest -m harness --cov-config=.coveragerc.harness --cov=enhanced_system.harness
ruff check enhanced_system scripts tests
```

Global coverage `fail_under` is 60. Harness package coverage is 90 via `.coveragerc.harness`. Invoke pytest from the repo root so `tests/harness/` is collected.

## Key Components

### Enhanced System
The `enhanced_system/` directory contains the core inference system with:
- Adaptive routing
- Intelligent caching
- Consensus inference
- Error handling and retries
- Input validation
- Performance monitoring
- Local harness runtime (`enhanced_system/harness/`)

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
- **harness/**: Local agent runtime (`run_agent`, collect, tailor)

## Contributing

Please ensure all tests pass before submitting changes:
```bash
make validate
```

## License

MIT. See [LICENSE](LICENSE).
