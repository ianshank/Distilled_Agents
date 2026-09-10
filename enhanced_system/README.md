# Enhanced Agent System

A comprehensive enhancement to the Distilled_Agents system with advanced features for production deployment.

## 🎯 Overview

The Enhanced Agent System provides enterprise-grade improvements to the base agent distillation framework, including:

- **Input Validation & Security**: PII detection, injection prevention, content sanitization
- **Multi-Level Caching**: L1 (memory), L2 (Redis), L3 (S3) with semantic similarity
- **Intelligent Error Handling**: Exponential backoff, fallback strategies, error learning
- **Confidence Calibration**: Historical accuracy-based calibration with explanations
- **Multi-Agent Consensus**: Parallel execution with agreement calculation
- **Adaptive Routing**: Dynamic agent selection based on task complexity
- **Streaming Inference**: Token-by-token streaming with real-time feedback
- **Batch Processing**: Intelligent batching with priority queuing
- **Real-Time Monitoring**: Prometheus metrics and Grafana dashboards
- **A/B Testing**: Statistical experimentation framework

## 📁 Architecture

```
enhanced_system/
├── core/                   # Core enhancement modules
│   ├── input_validator.py
│   ├── cache_manager.py
│   ├── error_handler.py
│   ├── confidence_calibrator.py
│   ├── consensus_inference.py
│   ├── adaptive_router.py
│   ├── streaming_inference.py
│   ├── batch_processor.py
│   └── monitoring.py
├── training/               # Enhanced training
│   ├── training_orchestrator.py
│   ├── data_curator.py
│   └── adaptive_config.py
├── evaluation/             # Evaluation & testing
│   ├── skill_evaluator.py
│   └── ab_testing.py
├── config/                 # Configuration files
│   ├── default.yaml
│   ├── production.yaml
│   ├── development.yaml
│   └── agent_profiles.json
└── infrastructure/         # Infrastructure configs
    ├── docker/
    ├── redis/
    └── monitoring/
```

## 🚀 Quick Start

### Installation

```bash
# Install the installable library plus test tools from the repo root
pip install -e ".[dev]"
```

### Basic Usage

```python
from enhanced_system.core import InputValidator, IntelligentCacheManager
from enhanced_system.config import load_config

# Load configuration
config = load_config('default')

# Input validation
validator = InputValidator(config.input_validation)
result = validator.validate_task_input("Your task here")

if result.is_valid:
    # Use sanitized input
    task = result.sanitized_input
else:
    print(f"Invalid input: {result.error_message}")

# Caching
cache_manager = IntelligentCacheManager(config.caching)
cache_key = cache_manager.compute_cache_key(task, "agent_name")

# Get or compute result
async def compute():
    return await your_agent_function(task)

result = await cache_manager.get_or_compute(cache_key, compute)
```

## 📊 Features

### 1. Input Validation

```python
from enhanced_system.core import InputValidator

validator = InputValidator(config)
result = validator.validate_task_input(task)

# Features:
# - Length validation
# - SQL/Command/Prompt injection detection  
# - PII detection and redaction
# - Content policy enforcement
```

### 2. Multi-Level Caching

```python
from enhanced_system.core import IntelligentCacheManager

cache = IntelligentCacheManager(config)

# Automatic L1 (memory) → L2 (Redis) → L3 (S3) fallback
result = await cache.get_or_compute(key, compute_function)

# Semantic similarity for cache hits
cache_key = cache.compute_cache_key(task, agent, params)
```

### 3. Error Handling

```python
from enhanced_system.core import IntelligentRetryHandler, FallbackManager

retry_handler = IntelligentRetryHandler(config)
fallback_manager = FallbackManager(config)

# Intelligent retry with exponential backoff
result = await retry_handler.execute_with_retry(
    agent_function,
    task,
    agent="my_agent"
)

# Multi-level fallback strategies
result = await fallback_manager.execute_with_fallback(
    primary_function,
    task
)
```

### 4. Confidence Calibration

```python
from enhanced_system.core import ConfidenceCalibrator

calibrator = ConfidenceCalibrator(config)

# Calibrate confidence scores
result = calibrator.calibrate_confidence(
    raw_confidence=0.8,
    agent="my_agent",
    task_type="coding"
)

print(f"Confidence: {result.confidence:.2%}")
print(f"Band: {result.reliability_band.value}")
print(f"Explanation: {result.explanation}")
```

### 5. Multi-Agent Consensus

```python
from enhanced_system.core import ConsensusInference

consensus = ConsensusInference(config)

# Run multiple agents and get consensus
result = await consensus.infer_with_consensus(
    task="Design a microservices architecture",
    agent_funcs=[agent1, agent2, agent3]
)

print(f"Agreement: {result.agreement_score:.2%}")
print(f"Consensus: {result.consensus_response}")
```

### 6. Adaptive Routing

```python
from enhanced_system.core import AdaptiveRouter

router = AdaptiveRouter(config)

# Route task to optimal agents
decision = router.route_task(
    task="Implement a RESTful API",
    constraints={'max_cost': 0.02, 'min_quality': 0.85}
)

print(f"Selected agents: {decision.selected_agents}")
print(f"Reasoning: {decision.reasoning}")
```

### 7. Streaming Inference

```python
from enhanced_system.core import StreamingInference

streamer = StreamingInference(config)

# Stream tokens in real-time
async for token in streamer.stream_response(generate_func, task):
    print(token.token, end='', flush=True)
    if token.confidence and token.confidence < 0.3:
        print("\n[Low confidence detected]")
```

### 8. Batch Processing

```python
from enhanced_system.core import BatchProcessor, Priority

processor = BatchProcessor(config)
await processor.start()

# Submit requests with priority
result = await processor.submit_request(
    task="Generate unit tests",
    priority=Priority.HIGH
)

await processor.stop()
```

### 9. Monitoring

```python
from enhanced_system.core import AgentMonitor

monitor = AgentMonitor(config)

# Track inference
monitor.track_inference(
    agent="swe_agent",
    task=task,
    result={
        'latency_ms': 150,
        'token_count': 500,
        'confidence': 0.92,
        'success': True
    }
)

# Get statistics
stats = monitor.get_agent_stats("swe_agent")
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Avg latency: {stats['avg_latency_ms']:.0f}ms")
```

## 🧪 Training Enhancements

### Training Orchestration

```python
from enhanced_system.training import TrainingOrchestrator

orchestrator = TrainingOrchestrator(config)

# Add training jobs with dependencies
orchestrator.add_training_job(
    job_id="base_agent",
    agent_name="base",
    dataset_path="s3://bucket/base_data.jsonl",
    model_config={'epochs': 5}
)

orchestrator.add_training_job(
    job_id="swe_agent",
    agent_name="swe",
    dataset_path="s3://bucket/swe_data.jsonl",
    model_config={'epochs': 5},
    dependencies=["base_agent"]  # Train after base_agent
)

# Train all agents in parallel
results = await orchestrator.train_all()
```

### Data Curation

```python
from enhanced_system.training import DataCurator

curator = DataCurator(config)

# Curate dataset
curated_data = curator.curate_dataset(raw_data)

# Features:
# - Quality filtering (threshold-based)
# - Data augmentation (paraphrasing)
# - Dataset balancing (oversampling)
# - Diversity analysis
```

### Adaptive Training Config

```python
from enhanced_system.training import AdaptiveTrainingConfig

# Create configuration
config = AdaptiveTrainingConfig.from_dict({
    'batch_size': 16,
    'num_epochs': 5,
    'learning_rate': 5e-5,
    'early_stopping': {
        'enabled': True,
        'patience': 5
    },
    'lr_schedule': {
        'strategy': 'cosine_annealing',
        'warmup_steps': 1000
    }
})

# Get adaptive learning rate
lr = config.calculate_lr(step=500, total_steps=10000)
```

## 📈 Evaluation

### Skill Evaluation

```python
from enhanced_system.evaluation import SkillEvaluator

evaluator = SkillEvaluator()

# Evaluate agent
result = evaluator.evaluate_agent(
    agent_name="swe_agent",
    test_suite=test_cases,
    agent_func=agent.infer
)

# View report
print(evaluator.generate_report(result))

# Detect regressions
regressions = evaluator.detect_regression("swe_agent", result)
```

### A/B Testing

```python
from enhanced_system.evaluation import ABTestFramework

ab_test = ABTestFramework(config)

# Create experiment
experiment_id = ab_test.create_experiment(
    name="Model v2 vs v1",
    control_model="model_v1",
    treatment_model="model_v2",
    traffic_split=0.1  # 10% to treatment
)

# Route requests
variant = ab_test.route_request(experiment_id, user_id)

# Record metrics
ab_test.record_metric(experiment_id, variant, "latency", 150.0)

# Analyze results
analysis = ab_test.analyze_experiment(experiment_id)
print(f"Winner: {analysis['winner']}")
print(f"Recommendation: {analysis['recommendation']}")
```

## ⚙️ Configuration

Configuration files support three environments:

- `default.yaml`: Default settings
- `development.yaml`: Local development
- `production.yaml`: Production deployment

Set environment via:

```bash
export ENHANCED_SYSTEM_ENV=production
```

Or programmatically:

```python
from enhanced_system.config import load_config

config = load_config('production')
```

## 🔧 Infrastructure

### Docker

```bash
# Build and run
cd enhanced_system/infrastructure/docker
docker-compose up -d
```

### Redis

Redis is used for L2 caching. Configure in config YAML:

```yaml
caching:
  l2:
    enabled: true
    host: localhost
    port: 6379
    ttl: 86400
```

### Monitoring

Prometheus metrics exposed on port 9090.
Grafana dashboards available for visualization.

## 🧪 Testing

The system includes comprehensive test coverage (80%+ target):

### Quick Test Commands

```bash
# Run all tests
pytest

# Run unit tests only
pytest -m unit

# Run integration tests only
pytest -m integration

# Run with coverage
pytest --cov=enhanced_system --cov-report=html --cov-report=term

# Run specific test file
pytest tests/unit/test_cache_manager.py -v
```

### Test Organization

- **Unit Tests** (80%+ coverage): Test individual components in isolation
- **Integration Tests** (70%+ coverage): Test component interactions
- **E2E Tests** (60%+ coverage): Test complete workflows
- **Performance Tests**: Benchmark critical paths

### Multi-Environment Testing with Tox

```bash
# Test across Python 3.9, 3.10, 3.11
tox

# Run coverage check
tox -e coverage

# Run linters
tox -e lint
```

### Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

📖 **See [TESTING.md](TESTING.md) for complete testing guide**

## 📚 Documentation

- [Configuration YAML](config/default.yaml)
- [SageMaker training](../docs/README_SAGEMAKER_TRAINING.md)
- [Architecture decisions](../docs/adr/0001-json-cache-serialization.md)
- [Testing](TESTING.md)

## 🤝 Contributing

1. Follow PEP 8 style guidelines
2. Add unit tests for new features
3. Update documentation
4. Run tests: `pytest -m unit`

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

Built on top of the Distilled_Agents system.
Uses PyTorch, Transformers, Redis, Prometheus, and other open-source tools.

---

**Version:** 1.0.0  
**Last Updated:** October 9, 2025

