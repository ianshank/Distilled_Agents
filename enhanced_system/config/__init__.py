"""Configuration management for enhanced agent system"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field


class Config(BaseModel):
    """Configuration model with validation"""

    class InputValidationConfig(BaseModel):
        max_length: int = 4096
        enable_pii_detection: bool = True
        enable_injection_detection: bool = True
        pii_entities: list = Field(default_factory=list)
        injection_patterns: list = Field(default_factory=list)

    class CachingConfig(BaseModel):
        class L1Config(BaseModel):
            enabled: bool = True
            max_size: int = 1000
            ttl: int = 3600

        class L2Config(BaseModel):
            enabled: bool = True
            host: str = "localhost"
            port: int = 6379
            db: int = 0
            ttl: int = 86400
            pool_size: int = 10
            socket_timeout: int = 5
            socket_connect_timeout: int = 5

        class L3Config(BaseModel):
            enabled: bool = True
            bucket: str = "agent-cache"
            region: str = Field(
                default_factory=lambda: (
                    os.getenv("AWS_REGION") or os.getenv("MANGOMAS_AWS_REGION") or "us-east-1"
                )
            )
            prefix: str = "enhanced-agents/"
            ttl: int = 604800

        class SemanticSimilarityConfig(BaseModel):
            enabled: bool = True
            threshold: float = 0.95
            model: str = "all-MiniLM-L6-v2"

        l1: L1Config = Field(default_factory=L1Config)
        l2: L2Config = Field(default_factory=L2Config)
        l3: L3Config = Field(default_factory=L3Config)
        semantic_similarity: SemanticSimilarityConfig = Field(
            default_factory=SemanticSimilarityConfig
        )

    class ErrorHandlingConfig(BaseModel):
        max_retries: int = 3
        base_delay: float = 1.0
        max_delay: float = 32.0
        exponential_base: int = 2
        enable_fallback: bool = True
        enable_error_learning: bool = True
        error_db_path: str = "./data/errors.db"

    class ConfidenceConfig(BaseModel):
        enabled: bool = True
        calibration_method: str = "isotonic"
        min_samples_for_calibration: int = 100
        reliability_bands: Dict[str, list] = Field(default_factory=dict)

    class ConsensusConfig(BaseModel):
        enabled: bool = False
        min_agents: int = 3
        max_agents: int = 5
        threshold: float = 0.7
        agreement_method: str = "embedding_similarity"
        ensemble_method: str = "confidence_weighted"

    class RoutingConfig(BaseModel):
        enabled: bool = True
        complexity_model: str = "heuristic"
        agent_profiles_path: str = str(Path(__file__).resolve().parent / "agent_profiles.json")
        routing_strategy: str = "cost_optimized"

    class StreamingConfig(BaseModel):
        enabled: bool = True
        buffer_size: int = 10
        confidence_per_token: bool = True
        stop_conditions: Dict[str, Any] = Field(default_factory=dict)

    class BatchProcessingConfig(BaseModel):
        enabled: bool = True
        max_batch_size: int = 32
        batch_timeout_ms: int = 100
        num_workers: int = 4
        queue_size: int = 1000
        priority_levels: int = 3

    class MonitoringConfig(BaseModel):
        enabled: bool = True
        prometheus_port: int = 9090
        metrics_interval: int = 10
        alerting: Dict[str, Any] = Field(default_factory=dict)
        metrics: list = Field(default_factory=list)

    class ABTestingConfig(BaseModel):
        enabled: bool = False
        default_traffic_split: float = 0.1
        min_samples_for_significance: int = 100
        significance_level: float = 0.05
        metrics_to_track: list = Field(default_factory=list)

    class TrainingConfig(BaseModel):
        parallel_training: bool = True
        max_parallel_jobs: int = 3
        dependency_tracking: bool = True
        resource_limits: Dict[str, Any] = Field(default_factory=dict)

    class DataCurationConfig(BaseModel):
        quality_threshold: float = 0.7
        enable_augmentation: bool = True
        augmentation_factor: int = 2
        enable_balancing: bool = True
        diversity_threshold: float = 0.8

    class LoggingConfig(BaseModel):
        level: str = "INFO"
        format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        file: str = "./logs/enhanced_system.log"
        max_bytes: int = 10485760
        backup_count: int = 5

    class ModelsConfig(BaseModel):
        default_model_path: str = "/opt/ml/model"
        teacher_model: str = "mistralai/Mistral-7B-v0.1"
        student_model: str = "microsoft/DialoGPT-medium"
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    class SecurityConfig(BaseModel):
        enable_rate_limiting: bool = True
        rate_limit_per_minute: int = 60
        rate_limit_per_hour: int = 1000
        enable_audit_logging: bool = True
        audit_log_path: str = "./logs/audit.log"

    input_validation: InputValidationConfig = Field(default_factory=InputValidationConfig)
    caching: CachingConfig = Field(default_factory=CachingConfig)
    error_handling: ErrorHandlingConfig = Field(default_factory=ErrorHandlingConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    consensus: ConsensusConfig = Field(default_factory=ConsensusConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    batch_processing: BatchProcessingConfig = Field(default_factory=BatchProcessingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    ab_testing: ABTestingConfig = Field(default_factory=ABTestingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data_curation: DataCurationConfig = Field(default_factory=DataCurationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)


def _config_search_dirs() -> list[Path]:
    """Prefer ops configs/, then the installable package YAML."""
    env_dir = os.getenv("MANGOMAS_CONFIG_DIR") or os.getenv("ENHANCED_SYSTEM_CONFIG_DIR")
    package_dir = Path(__file__).resolve().parent
    repo_configs = package_dir.parents[1] / "configs"
    dirs: list[Path] = []
    if env_dir:
        dirs.append(Path(env_dir))
    dirs.append(repo_configs)
    dirs.append(package_dir)
    return dirs


def load_config(config_name: Optional[str] = None, config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file

    Args:
        config_name: Name of config file (default, production, development)
        config_path: Full path to config file (overrides config_name)

    Returns:
        Config object with validated configuration
    """
    if config_path is not None:
        resolved = Path(config_path)
    else:
        if config_name is None:
            config_name = os.getenv("ENHANCED_SYSTEM_ENV", "default")
        filename = f"{config_name}.yaml"
        resolved = None
        searched: list[str] = []
        for directory in _config_search_dirs():
            candidate = directory / filename
            searched.append(str(candidate))
            if candidate.exists():
                resolved = candidate
                break
        if resolved is None:
            raise FileNotFoundError(
                f"Configuration file not found: {filename} (searched {searched})"
            )

    if not resolved.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved}")

    with open(resolved, encoding="utf-8") as handle:
        config_dict = yaml.safe_load(handle) or {}

    routing = config_dict.get("routing") or {}
    raw_profiles = routing.get("agent_profiles_path")
    if raw_profiles:
        routing["agent_profiles_path"] = _resolve_agent_profiles_path(str(raw_profiles))
        config_dict["routing"] = routing

    return Config(**config_dict)


def _resolve_agent_profiles_path(raw: str) -> str:
    """Prefer a real file; fall back to packaged profiles after install."""
    candidate = Path(raw)
    if candidate.exists():
        return str(candidate)
    packaged = Path(__file__).resolve().parent / "agent_profiles.json"
    if packaged.exists():
        return str(packaged)
    return raw


# Global config instance
_config: Optional[Config] = None


def get_config(reload: bool = False) -> Config:
    """
    Get global configuration instance

    Args:
        reload: Force reload configuration

    Returns:
        Config object
    """
    global _config

    if _config is None or reload:
        _config = load_config()

    return _config


from .builders import ConfigBuilder  # noqa: E402

__all__ = ["Config", "load_config", "get_config", "ConfigBuilder"]
