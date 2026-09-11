"""Optional Transformers backend; import is deferred so tests stay CPU-only."""

from __future__ import annotations

from typing import Any, Optional, Sequence


class TransformersBackend:
    """Thin wrapper around a generate callable (injected for tests)."""

    def __init__(
        self,
        generate_fn=None,
        *,
        model_name: Optional[str] = None,
        trust_remote_code: bool = False,
        model_revision: Optional[str] = None,
        max_input_length: int = 512,
        max_new_tokens: int = 128,
    ) -> None:
        self._generate_fn = generate_fn
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._model_revision = model_revision
        self._max_input_length = max_input_length
        self._max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None
        self._device = None

    def _ensure_model(self) -> tuple[Any, Any, Any]:
        if self._generate_fn is not None:
            return self._generate_fn, None, None
        if self._model is not None and self._tokenizer is not None and self._device is not None:
            return self._generate_fn, self._tokenizer, self._device
        if not self._model_name:
            raise ImportError("transformers backend requires an injected generate_fn or model_name")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            trust_remote_code=self._trust_remote_code,
            revision=self._model_revision,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        if torch.cuda.is_available():
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=self._trust_remote_code,
                revision=self._model_revision,
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                torch_dtype=torch.float32,
                trust_remote_code=self._trust_remote_code,
                revision=self._model_revision,
            )
        if not torch.cuda.is_available():
            self._model.to(self._device)
        self._model.eval()
        return self._generate_fn, self._tokenizer, self._device

    def _render_prompt(self, messages: Sequence[dict[str, str]], prefix: Optional[str]) -> str:
        parts = [f"{item.get('role', 'user')}: {item.get('content', '')}" for item in messages]
        tail = "assistant:"
        if prefix:
            tail = f"{tail} {prefix}"
        parts.append(tail)
        return "\n".join(parts)

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        prefix: Optional[str] = None,
        n: int = 1,
        temperature: Optional[float] = None,
    ) -> list[str]:
        generate_fn, tokenizer, device = self._ensure_model()
        if generate_fn is not None:
            return generate_fn(list(messages), prefix=prefix, n=n, temperature=temperature)
        import torch

        prompt = self._render_prompt(messages, prefix)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_input_length,
            padding=True,
        )
        input_device = getattr(self._model, "device", device)
        if input_device is not None:
            inputs = {key: value.to(input_device) for key, value in inputs.items()}
        generate_kwargs = {
            "max_new_tokens": self._max_new_tokens,
            "num_return_sequences": max(n, 1),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if temperature is not None and temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = temperature
        else:
            generate_kwargs["do_sample"] = False
        with torch.no_grad():
            outputs = self._model.generate(**inputs, **generate_kwargs)
        prompt_tokens = inputs["input_ids"].shape[1]
        return [
            tokenizer.decode(
                output[prompt_tokens:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            for output in outputs
        ]
