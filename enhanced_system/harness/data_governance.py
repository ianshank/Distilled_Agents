"""Data governance and PII redaction for agent trajectories."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional, cast

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from presidio_analyzer import AnalyzerEngine as TypedAnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine as TypedAnonymizerEngine
    from presidio_anonymizer.entities import RecognizerResult as AnonymizerResult
else:
    TypedAnalyzerEngine = Any
    TypedAnonymizerEngine = Any
    AnonymizerResult = Any

try:
    from presidio_analyzer import AnalyzerEngine as _AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine as _AnonymizerEngine

    HAS_PRESIDIO = True
except ImportError:
    HAS_PRESIDIO = False
    _AnalyzerEngine = Any  # type: ignore
    _AnonymizerEngine = Any  # type: ignore


class PIIScrubber:
    """Redacts PII from text and complex JSON structures using Presidio.

    Requires the `security` optional dependency group.
    """

    def __init__(self, entities: Optional[List[str]] = None):
        if not HAS_PRESIDIO:
            raise ImportError(
                "Presidio libraries not found. Install with: pip install 'mangomas[security]'"
            )

        # Initialize Presidio
        try:
            self.analyzer: TypedAnalyzerEngine = _AnalyzerEngine()
        except OSError as e:
            raise RuntimeError(
                "Failed to load NLP model for Presidio. Try: python -m spacy download en_core_web_lg"
            ) from e

        self.anonymizer: TypedAnonymizerEngine = _AnonymizerEngine()

        # Default high-risk entities
        self.entities = entities or [
            "EMAIL_ADDRESS",
            "CREDIT_CARD",
            "PHONE_NUMBER",
            "IBAN_CODE",
            "US_SSN",
            "US_BANK_NUMBER",
            "IP_ADDRESS",
            "PERSON",
            "CRYPTO",
        ]

    def redact_text(self, text: str) -> str:
        """Redact PII from a single string."""
        if not text or not isinstance(text, str):
            return text

        results = self.analyzer.analyze(text=text, entities=self.entities, language="en")

        if not results:
            return text

        # presidio-analyzer and presidio-anonymizer have mismatched type hints for RecognizerResult
        anon_results = cast(List[AnonymizerResult], results)
        anonymized = self.anonymizer.anonymize(text=text, analyzer_results=anon_results)
        return str(anonymized.text)

    def redact_object(self, obj: Any) -> Any:
        """Recursively redact PII from JSON-like dictionaries and lists."""
        if isinstance(obj, str):
            return self.redact_text(obj)
        elif isinstance(obj, dict):
            return {k: self.redact_object(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.redact_object(item) for item in obj]
        else:
            return obj
