"""Data governance and PII redaction for agent trajectories."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
except ImportError:
    AnalyzerEngine = None
    AnonymizerEngine = None


class PIIScrubber:
    """Redacts PII from text and complex JSON structures using Presidio.

    Requires the `security` optional dependency group.
    """

    def __init__(self, entities: Optional[List[str]] = None):
        if AnalyzerEngine is None or AnonymizerEngine is None:
            raise ImportError(
                "Presidio libraries not found. Install with: pip install 'mangomas[security]'"
            )

        # Initialize Presidio
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

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

        anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text

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
