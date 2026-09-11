# 🧠 MangoMAS Agent Distillation System

## Overview

The MangoMAS Agent Distillation System enables knowledge transfer from large, powerful teacher models to smaller, specialized student models. This approach allows for efficient deployment of AI agents while maintaining high performance through knowledge distillation techniques.

## Trajectory SFT (local)

`python scripts/training/train_distilled_adapter.py --trajectory_mode True` uses `scripts/training/distill/trajectory_collator.py` (observation masking) and sets `distillation_alpha` from `MANGOMAS_TRAJECTORY_DISTILL_ALPHA` (default `0.0`). Collect traces with `scripts/harness/collect_trajectories.py`. SageMaker `create_job_spec` does not take trajectory keys until argparse and the launcher change together.

## 🎯 Key Features

- **Knowledge Distillation**: Transfer knowledge from large models to smaller, efficient ones
- **LoRA Integration**: Efficient fine-tuning with Low-Rank Adaptation
- **SageMaker Compatibility**: Full AWS SageMaker integration for scalable training
- **Multi-Stage Pipelines**: Progressive distillation for optimal results
- **Comprehensive Monitoring**: Weights & Biases integration for experiment tracking
- **Production Ready**: Inference endpoints and deployment automation

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Teacher Model │    │  Distillation   │    │  Student Model  │
│   (Large)       │───▶│   Process       │───▶│   (Small)       │
│                 │    │                 │    │                 │
│ - Mistral-7B    │    │ - KL Divergence │    │ - DialoGPT      │
│ - GPT-4         │    │ - Temperature   │    │ - Custom        │
│ - Claude        │    │ - LoRA          │    │ - Specialized   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 File Structure

```
training/
├── train_distilled_adapter.py      # Main training script
├── sagemaker_distillation_job.py   # SageMaker job manager
├── inference.py                    # Inference script
├── requirements.txt                # Dependencies
├── README_AGENT_DISTILLATION.md    # This file
└── examples/
    ├── local_training.py           # Local training example
    ├── sagemaker_example.py        # SageMaker example
    └── inference_example.py        # Inference example
```

## 🛠️ Installation

### **Prerequisites**
- Python 3.8+
- PyTorch 1.13+
- Transformers 4.26+
- AWS SageMaker access
- HuggingFace account (optional)

### **Setup**
```bash
# Clone the repository
git clone <repository-url>
cd MangoMAS

# Install dependencies
pip install -r training/requirements.txt

# Configure AWS credentials
aws configure
```

## 🚀 Quick Start

### 1. Local Training

```python
from train_distilled_adapter import AgentDistillationTrainer
import argparse

# Setup arguments
args = argparse.Namespace(
    teacher_model_name='mistralai/Mistral-7B-v0.1',
    student_model_name='microsoft/DialoGPT-medium',
    train_file='data/train.jsonl',
    output_dir='./distilled_model',
    num_train_epochs=3,
    distillation_alpha=0.5,
    temperature=2.0,
    use_lora=True,
    use_fp16=True
)

# Initialize trainer and start training
trainer = AgentDistillationTrainer(args)
trainer.train()
```

### 2. SageMaker Training

```python
from sagemaker_distillation_job import AgentDistillationJob

# Initialize job manager
job_manager = AgentDistillationJob(region='us-east-1')

# Create training job
job_name = job_manager.create_training_job(
    job_name="mangomas-distillation-20250730",
    teacher_model='mistralai/Mistral-7B-v0.1',
    student_model='microsoft/DialoGPT-medium',
    train_data_s3="s3://your-bucket/training-data/train.jsonl",
    eval_data_s3="s3://your-bucket/training-data/eval.jsonl",
    instance_type='ml.g5.2xlarge',
    hyperparameters={
        'distillation_alpha': 0.5,
        'temperature': 2.0,
        'use_lora': True
    }
)

# Monitor training
job_manager.monitor_training_job(job_name)
```

### 3. Inference

```python
from inference import DistilledAgentInference

# Initialize inference handler
inference = DistilledAgentInference()
inference.model_fn("/path/to/distilled/model")

# Generate response
input_data = {
    "prompt": "What is the best approach for software architecture?",
    "max_length": 256,
    "temperature": 0.7
}

result = inference.predict_fn(input_data)
print(result["responses"][0])
```

## 📊 Training Configuration

### Model Configuration

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `teacher_model_name` | Large model for knowledge transfer | `mistralai/Mistral-7B-v0.1` | Any HuggingFace model |
| `student_model_name` | Target model for distillation | `microsoft/DialoGPT-medium` | Smaller model |
| `distillation_alpha` | Weight for distillation loss | `0.5` | `0.0 - 1.0` |
| `temperature` | Temperature for knowledge distillation | `2.0` | `0.1 - 10.0` |

### LoRA Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `use_lora` | Enable LoRA fine-tuning | `True` |
| `lora_r` | LoRA rank | `16` |
| `lora_alpha` | LoRA alpha | `32` |
| `lora_dropout` | LoRA dropout | `0.1` |

### Training Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_train_epochs` | Number of training epochs | `3` |
| `per_device_train_batch_size` | Training batch size | `2` |
| `learning_rate` | Learning rate | `5e-5` |
| `gradient_accumulation_steps` | Gradient accumulation | `4` |

## 🔧 Advanced Usage

### Multi-Stage Distillation Pipeline

```python
# Create multi-stage pipeline
pipeline_name = job_manager.create_distillation_pipeline(
    pipeline_name="mangomas-advanced-distillation",
    teacher_model='mistralai/Mistral-7B-v0.1',
    student_model='microsoft/DialoGPT-medium',
    train_data_s3="s3://your-bucket/training-data/train.jsonl",
    eval_data_s3="s3://your-bucket/training-data/eval.jsonl"
)
```

### Custom Loss Functions

```python
# Custom distillation loss
def custom_distillation_loss(student_outputs, teacher_outputs, labels):
    # Task loss
    task_loss = torch.nn.functional.cross_entropy(
        student_outputs.logits.view(-1, student_outputs.logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )
    
    # Custom distillation loss
    distillation_loss = torch.nn.functional.mse_loss(
        student_outputs.logits,
        teacher_outputs.logits
    )
    
    return 0.7 * task_loss + 0.3 * distillation_loss
```

### Experiment Tracking

```python
# Enable Weights & Biases
args.use_wandb = True

# Custom experiment name
wandb.init(
    project="mangomas-agent-distillation",
    name="custom-experiment",
    config={
        "teacher_model": "mistralai/Mistral-7B-v0.1",
        "student_model": "microsoft/DialoGPT-medium",
        "distillation_alpha": 0.5,
        "temperature": 2.0
    }
)
```

## 📈 Performance Optimization

### Memory Optimization

```python
# Enable FP16 training
args.use_fp16 = True

# Use device map for large models
args.use_device_map = True

# Gradient checkpointing
training_args.gradient_checkpointing = True
```

### Speed Optimization

```python
# Use spot instances for cost efficiency
use_spot_instances = True

# Optimize batch size
per_device_train_batch_size = 4
gradient_accumulation_steps = 2

# Use mixed precision
fp16 = True
```

## 🧪 Testing and Validation

### Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run local training test
python train_distilled_adapter.py \
    --teacher_model_name mistralai/Mistral-7B-v0.1 \
    --student_model_name microsoft/DialoGPT-medium \
    --train_file test_data.jsonl \
    --num_train_epochs 1 \
    --use_lora
```

### SageMaker Testing

```bash
# Test SageMaker job creation
python sagemaker_distillation_job.py

# Monitor training jobs
python -c "
from sagemaker_distillation_job import AgentDistillationJob
job_manager = AgentDistillationJob()
job_manager.list_training_jobs('mangomas')
"
```

## 📊 Monitoring and Metrics

### Training Metrics

- **Loss**: Combined task and distillation loss
- **Accuracy**: Task-specific accuracy metrics
- **Perplexity**: Language modeling perplexity
- **BLEU Score**: Text generation quality
- **ROUGE Score**: Summarization quality

### Resource Metrics

- **GPU Utilization**: GPU usage during training
- **Memory Usage**: Peak memory consumption
- **Training Time**: Total training duration
- **Cost**: AWS SageMaker costs

## 🚀 Deployment

### SageMaker Endpoint

```python
# Create inference endpoint
predictor = job_manager.create_inference_endpoint(
    model_name="mangomas-distillation-20250730",
    endpoint_name="mangomas-distilled-agent",
    instance_type="ml.m5.large"
)

# Test endpoint
response = predictor.predict({
    "prompt": "Explain software architecture",
    "max_length": 256,
    "temperature": 0.7
})
```

### Local Deployment

```bash
# Start local inference server
python inference.py

# Test with curl
curl -X POST http://localhost:8080/invocations \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "What is the best approach for software architecture?",
        "max_length": 256,
        "temperature": 0.7
    }'
```

## 🔍 Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or enable gradient checkpointing
2. **Slow Training**: Use spot instances or optimize hyperparameters
3. **Poor Quality**: Adjust distillation alpha and temperature
4. **Model Loading**: Check model compatibility and dependencies

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Verbose training
training_args.logging_steps = 10
training_args.save_steps = 100
```

## 📚 Examples

### **Basic Training Example**
```python
from train_distilled_adapter import AgentDistillationTrainer

# Initialize trainer
trainer = AgentDistillationTrainer(
    teacher_model='mistralai/Mistral-7B-v0.1',
    student_model='microsoft/DialoGPT-medium',
    train_file='training/swe_agent.jsonl'
)

# Start training
trainer.train()
```

### **SageMaker Training Example**
```python
from sagemaker_distillation_job import AgentDistillationJob

# Create training job
job = AgentDistillationJob()
job_name = job.create_training_job(
    agent_type="software_engineer",
    training_data_s3_uri="s3://bucket/training-data.jsonl"
)

# Monitor progress
job.monitor_training_job(job_name)
```

### **Custom Agent Training Example**
```python
# Train custom agent with specific configuration
config = {
    "model_name": "microsoft/DialoGPT-medium",
    "training_data": "custom_training_data.jsonl",
    "epochs": 5,
    "learning_rate": 1e-4
}

python train_distilled_adapter.py --config config.json
```

### **Inference Example**
```python
from inference import DistilledAgentInference

# Load trained model
inference = DistilledAgentInference(
    model_path="s3://bucket/trained-model"
)

# Generate response
response = inference.predict("Write a Python function for sorting")
print(response)
```

## 📚 Best Practices

### Model Selection

- **Teacher**: Choose large, high-quality models (Mistral-7B, GPT-4, Claude)
- **Student**: Select smaller, efficient models (DialoGPT, GPT-2, BERT)
- **Domain**: Match teacher and student domains when possible

### Hyperparameter Tuning

- **Distillation Alpha**: Start with 0.5, tune based on task
- **Temperature**: Higher values (2-4) for softer knowledge transfer
- **Learning Rate**: Lower rates (1e-5 to 5e-5) for stable training

### Data Preparation

- **Quality**: Use high-quality, diverse training data
- **Format**: JSONL format with 'prompt' field
- **Size**: 10K-100K examples for good results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- HuggingFace Transformers team
- AWS SageMaker team
- PEFT (Parameter-Efficient Fine-Tuning) contributors
- Weights & Biases for experiment tracking

---

**MangoMAS Agent Distillation System** - Efficient knowledge transfer for AI agents 