#!/usr/bin/env python3
"""
MangoMAS Distilled Agent Inference Script
SageMaker-compatible inference script for distilled agent models
"""

import hmac
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistilledAgentInference:
    """Inference handler for distilled agent models"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = "/opt/ml/model"
        self.active_adapter = None
        self.adapters_loaded = []
        self.adapter_root: Path | None = None
        self.available_adapters: dict[str, Path] = {}
        self._adapter_lock = threading.RLock()

    def model_fn(self, model_dir: str):
        """Load the model and tokenizer"""
        logger.info(f"Loading model from: {model_dir}")
        self.model_dir = model_dir
        configured_root = os.getenv("MANGOMAS_ADAPTER_ROOT")
        self.adapter_root = (
            Path(configured_root).resolve() if configured_root else Path(model_dir).resolve()
        )
        self.available_adapters = self._discover_available_adapters()

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)  # nosec B615
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(  # nosec B615
                model_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=os.getenv("MANGOMAS_TRUST_REMOTE_CODE", "false").lower()
                == "true",
            )

            # Check if LoRA adapter is present
            if os.path.exists(os.path.join(model_dir, "adapter_config.json")):
                logger.info("Loading default LoRA adapter...")
                self.model = PeftModel.from_pretrained(
                    self.model, model_dir, adapter_name="default"
                )
                self.adapters_loaded.append("default")
                self.active_adapter = "default"

            self.model.eval()
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _discover_available_adapters(self) -> dict[str, Path]:
        root = self.adapter_root
        if root is None or not root.exists():
            return {}
        available: dict[str, Path] = {}
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            if (entry / "adapter_config.json").is_file():
                available[entry.name] = entry
        return available

    def load_adapter(self, adapter_name: str):
        """Load a new adapter for Blue/Green deployments."""
        assert self.model is not None, "Model not loaded"
        if not hasattr(self.model, "load_adapter"):
            raise ValueError("Base model does not support adapters (not a PeftModel).")
        self.available_adapters = self._discover_available_adapters()
        requested = self.available_adapters.get(adapter_name)
        if requested is None:
            raise ValueError(f"Adapter {adapter_name} not found under {self.adapter_root}")
        with self._adapter_lock:
            logger.info(f"Loading adapter '{adapter_name}' from {requested}")
            self.model.load_adapter(str(requested), adapter_name=adapter_name)
            if adapter_name not in self.adapters_loaded:
                self.adapters_loaded.append(adapter_name)

    def set_adapter(self, adapter_name: str):
        """Switch the active adapter."""
        assert self.model is not None, "Model not loaded"
        if adapter_name not in self.adapters_loaded:
            raise ValueError(f"Adapter {adapter_name} not loaded.")
        with self._adapter_lock:
            logger.info(f"Switching active adapter to '{adapter_name}'")
            self.model.set_adapter(adapter_name)
            self.active_adapter = adapter_name

    def unload_adapter(self, adapter_name: str):
        """Unload an adapter to free memory."""
        assert self.model is not None, "Model not loaded"
        if adapter_name not in self.adapters_loaded:
            raise ValueError(f"Adapter {adapter_name} not loaded.")
        if adapter_name == self.active_adapter:
            raise ValueError(
                f"Cannot unload active adapter '{adapter_name}'. Switch to another adapter first."
            )

        with self._adapter_lock:
            logger.info(f"Unloading adapter '{adapter_name}'")
            if hasattr(self.model, "delete_adapter"):
                self.model.delete_adapter(adapter_name)
            self.adapters_loaded.remove(adapter_name)

    def input_fn(self, request_body: str, request_content_type: str = "application/json"):
        """Parse input data"""
        if request_content_type == "application/json":
            input_data = json.loads(request_body)
            return input_data
        else:
            raise ValueError(f"Unsupported content type: {request_content_type}")

    def predict_fn(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate predictions"""
        assert self.model is not None, "Model not loaded"
        assert self.tokenizer is not None, "Tokenizer not loaded"
        # SEC-003: Validate prompt size to prevent DoS via unbounded tokenizer input
        max_prompt_bytes = int(os.getenv("MANGOMAS_MAX_PROMPT_BYTES", "1048576"))
        try:
            # Extract input parameters
            prompt = input_data.get("prompt", "")
            if len(prompt.encode("utf-8")) > max_prompt_bytes:
                raise ValueError(
                    f"Prompt exceeds maximum size: {len(prompt.encode('utf-8'))} > {max_prompt_bytes} bytes"
                )
            max_length = input_data.get("max_length", 512)
            temperature = input_data.get("temperature", 0.7)
            top_p = input_data.get("top_p", 0.9)
            top_k = input_data.get("top_k", 50)
            do_sample = input_data.get("do_sample", True)
            num_return_sequences = input_data.get("num_return_sequences", 1)

            # Tokenize input
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=max_length, padding=True
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate response
            with self._adapter_lock:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        do_sample=do_sample,
                        num_return_sequences=num_return_sequences,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=input_data.get("repetition_penalty", 1.1),
                    )

            # Decode outputs
            responses = []
            for output in outputs:
                # Remove input tokens from output
                response_tokens = output[inputs["input_ids"].shape[1] :]
                response_text = self.tokenizer.decode(
                    response_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                responses.append(response_text.strip())

            # Prepare result
            result = {
                "responses": responses,
                "input_prompt": prompt,
                "generation_config": {
                    "max_length": max_length,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "do_sample": do_sample,
                    "num_return_sequences": num_return_sequences,
                },
            }

            return result

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise  # SEC-002: Let Flask handler return proper HTTP 500

    def output_fn(self, prediction: Dict[str, Any], content_type: str = "application/json") -> str:
        """Format output"""
        if content_type == "application/json":
            return json.dumps(prediction)
        else:
            raise ValueError(f"Unsupported content type: {content_type}")


# Global inference handler
inference_handler = DistilledAgentInference()


def model_fn(model_dir: str):
    """SageMaker model loading function"""
    return inference_handler.model_fn(model_dir)


def input_fn(request_body: str, request_content_type: str = "application/json"):
    """SageMaker input processing function"""
    return inference_handler.input_fn(request_body, request_content_type)


def predict_fn(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """SageMaker prediction function"""
    return inference_handler.predict_fn(input_data)


def output_fn(prediction: Dict[str, Any], content_type: str = "application/json") -> str:
    """SageMaker output formatting function"""
    return inference_handler.output_fn(prediction, content_type)


# Flask app for local testing
if __name__ == "__main__":
    from flask import Flask, jsonify, request

    try:
        from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

        INFERENCE_REQUESTS = Counter("inference_requests_total", "Total inference requests")
        INFERENCE_ERRORS = Counter("inference_errors_total", "Total inference errors")
        # Tune buckets for LLM generation: 0.1s to 60.0s
        INFERENCE_LATENCY = Histogram(
            "inference_latency_seconds",
            "Inference latency",
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, float("inf")),
        )
        HAS_PROMETHEUS = True
    except ImportError:
        HAS_PROMETHEUS = False
    import time

    app = Flask(__name__)

    # Load model — CFG-001: Use MANGOMAS_ prefix with backwards-compatible fallback
    _legacy_model_dir = os.getenv("MODEL_DIR")
    if _legacy_model_dir and not os.getenv("MANGOMAS_MODEL_DIR"):
        logger.warning("MODEL_DIR is deprecated, use MANGOMAS_MODEL_DIR instead")
    model_dir = os.getenv("MANGOMAS_MODEL_DIR", _legacy_model_dir or "/opt/ml/model")
    inference_handler.model_fn(model_dir)

    @app.route("/ping", methods=["GET"])
    def ping():
        """Health check endpoint"""
        active = inference_handler.active_adapter if inference_handler.active_adapter else "base"
        return jsonify({"status": "healthy", "active_adapter": active})

    if HAS_PROMETHEUS:

        @app.route("/metrics", methods=["GET"])
        def metrics():
            from flask import Response

            return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

    @app.route("/adapter/load", methods=["POST"])
    def load_adapter():
        auth_error = _require_adapter_auth()
        if auth_error:
            return auth_error
        data = request.json or {}
        adapter_name = data.get("adapter_name")
        if not adapter_name:
            return jsonify({"error": "Missing adapter_name"}), 400
        try:
            inference_handler.load_adapter(adapter_name)
            return jsonify({"status": "loaded", "adapters": inference_handler.adapters_loaded})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/adapter/unload", methods=["POST"])
    def unload_adapter():
        auth_error = _require_adapter_auth()
        if auth_error:
            return auth_error
        data = request.json or {}
        adapter_name = data.get("adapter_name")
        if not adapter_name:
            return jsonify({"error": "Missing adapter_name"}), 400
        try:
            inference_handler.unload_adapter(adapter_name)
            return jsonify({"status": "unloaded", "adapters": inference_handler.adapters_loaded})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/adapter/switch", methods=["POST"])
    def switch_adapter():
        auth_error = _require_adapter_auth()
        if auth_error:
            return auth_error
        data = request.json or {}
        adapter_name = data.get("adapter_name")
        if not adapter_name:
            return jsonify({"error": "Missing adapter_name"}), 400
        try:
            inference_handler.set_adapter(adapter_name)
            return jsonify(
                {"status": "switched", "active_adapter": inference_handler.active_adapter}
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/invocations", methods=["POST"])
    def invoke():
        """Inference endpoint"""
        start_time = time.time()
        if HAS_PROMETHEUS:
            INFERENCE_REQUESTS.inc()
        try:
            # Get input data
            input_data = inference_handler.input_fn(
                request.get_data(as_text=True), request.content_type
            )

            # Generate prediction
            prediction = inference_handler.predict_fn(input_data)

            # Format output
            response = inference_handler.output_fn(prediction)

            if HAS_PROMETHEUS:
                INFERENCE_LATENCY.observe(time.time() - start_time)
            return response, 200, {"Content-Type": "application/json"}

        except Exception as e:
            if HAS_PROMETHEUS:
                INFERENCE_ERRORS.inc()
            return jsonify({"error": str(e)}), 500

    # CFG-002: Use MANGOMAS_ prefix with backwards-compatible fallback
    _legacy_port = os.getenv("PORT")
    if _legacy_port and not os.getenv("MANGOMAS_PORT"):
        logger.warning("PORT is deprecated, use MANGOMAS_PORT instead")
    port = int(os.getenv("MANGOMAS_PORT", _legacy_port or "8080"))
    host = os.getenv("MANGOMAS_BIND_HOST", os.getenv("BIND_HOST", "127.0.0.1"))

    def _require_adapter_auth():
        token = os.getenv("MANGOMAS_ADAPTER_ADMIN_TOKEN", "")
        non_local = host not in {"127.0.0.1", "localhost", "::1"}
        if not token and not non_local:
            return None
        if not token:
            return jsonify(
                {"error": "Adapter management requires MANGOMAS_ADAPTER_ADMIN_TOKEN"}
            ), 403
        expected = token
        candidate = request.headers.get("X-Adapter-Token", "")
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            candidate = auth_header[7:]
        if not hmac.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8")):
            return jsonify({"error": "Unauthorized"}), 401
        return None

    app.run(host=host, port=port, debug=False)
