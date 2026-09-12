---
name: mangomas-train
description: Train a MangoMAS agent skill with the existing CLI. Use when asked to train, fine-tune, or produce a LoRA/ALoRA adapter for a role.
cli: scripts/training/train_agent_skill.py
inputs:
  role: Agent role name (CLI --role)
  dataset: Path to JSONL dataset (CLI --dataset)
  model: Base model; default from MANGOMAS_TEACHER_MODEL
  output_dir: Adapter output directory (CLI --output_dir)
  region: AWS region from MANGOMAS_AWS_REGION
  bucket: Training bucket from MANGOMAS_TRAINING_DATA_BUCKET
  table: DynamoDB table from MANGOMAS_DYNAMODB_TABLE (CLI --table)
  instance_type: SageMaker instance from MANGOMAS_GPU_INSTANCE_TYPE
  epochs: MANGOMAS_SKILL_EPOCHS
  batch_size: MANGOMAS_SKILL_BATCH_SIZE
---

# Train a MangoMAS agent skill

Call the checked-in CLI. Do not hardcode AWS account IDs, access keys, or bucket names.

```bash
python scripts/training/train_agent_skill.py \
  --role "$ROLE" \
  --dataset "$DATASET" \
  --model "${MANGOMAS_TEACHER_MODEL}" \
  --output_dir "$OUTPUT_DIR" \
  --region "${MANGOMAS_AWS_REGION}" \
  --bucket "${MANGOMAS_TRAINING_DATA_BUCKET}" \
  --table "${MANGOMAS_DYNAMODB_TABLE}"
```

Settings come from `enhanced_system.ops.settings.get_settings()` (`MANGOMAS_` env prefix). Mock training runs when AWS is unavailable. Never pass `--trust_remote_code` unless operators explicitly set `MANGOMAS_TRUST_REMOTE_CODE=true`.
