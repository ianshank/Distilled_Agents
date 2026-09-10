"""
Input Validation and Sanitization Module
Provides comprehensive input validation, PII detection, and injection prevention
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from enhanced_system.core.base.validator import BaseValidator
from enhanced_system.core.base.validator import ValidationResult as BaseValidationResult

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    logging.warning("Presidio not available. PII detection will be disabled.")


logger = logging.getLogger(__name__)


@dataclass
class ValidationResult(BaseValidationResult):
    """Result of input validation with optional PII details."""

    pii_detected: bool = False
    pii_entities: list = None

    def __post_init__(self):
        super().__post_init__()
        if self.pii_entities is None:
            self.pii_entities = []


class InputValidator(BaseValidator):
    """
    Comprehensive input validation and sanitization

    Features:
    - Length validation
    - SQL injection detection
    - Command injection detection
    - Prompt injection detection
    - PII detection and redaction
    - Special character sanitization
    """

    # Injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\bunion\b.*\bselect\b)",
        r"(\bselect\b.*\bfrom\b)",
        r"(\binsert\b.*\binto\b)",
        r"(\bupdate\b.*\bset\b)",
        r"(\bdelete\b.*\bfrom\b)",
        r"(\bdrop\b.*\btable\b)",
        r"(--|#|\/\*|\*\/)",
        r"(\bor\b.*=.*)",
        r"(\band\b.*=.*)",
        r"(';|\")",
    ]

    COMMAND_INJECTION_PATTERNS = [
        r"(&&|\|\|)",
        r"(\$\(.*\))",
        r"(`.*`)",
        r"(\|)",
        r"(\bwget\b|\bcurl\b)",
        r"(\brm\b.*-rf)",
        r"(\bsudo\b)",
        r"(\bchmod\b|\bchown\b)",
    ]

    PROMPT_INJECTION_PATTERNS = [
        r"(ignore\s+(?:all\s+)?(?:previous|above|all)\s+(?:instructions|rules|prompts?))",
        r"(disregard\s+(?:all\s+)?(?:previous|above|all)\s+(?:instructions|rules|prompts?))",
        r"(forget\s+(?:all\s+)?(?:previous|above|all)\s+(?:instructions|rules|prompts?))",
        r"(system\s*:\s*)",
        r"(you\s+are\s+now\s+)",
        r"(new\s+instructions?\s*:)",
        r"(roleplay\s+as\s+)",
        r"(pretend\s+(you're|you\s+are)\s+)",
    ]

    # Control characters to remove
    CONTROL_CHARS = r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize InputValidator

        Args:
            config: Configuration dictionary with validation settings
        """
        self.config = config or {}
        self.max_length = self.config.get("max_length", 4096)
        self.enable_pii_detection = self.config.get("enable_pii_detection", True)
        self.enable_injection_detection = self.config.get("enable_injection_detection", True)
        self.pii_entities = self.config.get(
            "pii_entities",
            ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "SSN", "IP_ADDRESS"],
        )

        # Initialize PII detector if available
        self.pii_analyzer = None
        if self.enable_pii_detection and PRESIDIO_AVAILABLE:
            try:
                nlp_configuration = {
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
                }
                provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
                self.pii_analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())
                logger.info("PII analyzer initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize PII analyzer: {e}")
                self.pii_analyzer = None

    def validate_task_input(self, task: str) -> ValidationResult:
        """
        Validate task input comprehensively

        Args:
            task: Input task string to validate

        Returns:
            ValidationResult with validation status and details
        """
        warnings = []

        # Check if input is empty
        if not task or not task.strip():
            return ValidationResult(
                is_valid=False, error_message="Task input cannot be empty", warnings=warnings
            )

        # Check length
        if len(task) > self.max_length:
            return ValidationResult(
                is_valid=False,
                error_message=f"Task exceeds maximum length of {self.max_length} characters",
                warnings=warnings,
            )

        # Check for injection attempts
        if self.enable_injection_detection:
            injection_detected, injection_type = self._detect_injection(task)
            if injection_detected:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Potential {injection_type} injection detected",
                    warnings=warnings,
                )

        # Check for PII
        pii_detected = False
        pii_entities = []
        if self.enable_pii_detection:
            pii_detected, pii_entities = self._detect_pii(task)
            if pii_detected:
                warnings.append(f"PII detected: {len(pii_entities)} entities found")

        # Sanitize input
        sanitized = self.sanitize_input(task)

        return ValidationResult(
            is_valid=True,
            sanitized_input=sanitized,
            warnings=warnings,
            pii_detected=pii_detected,
            pii_entities=pii_entities,
        )

    def _detect_injection(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Detect various injection attempts

        Args:
            text: Text to check for injection

        Returns:
            Tuple of (detected, injection_type)
        """
        text_lower = text.lower()

        # SQL injection
        for pattern in self.SQL_INJECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                logger.warning(f"SQL injection pattern detected: {pattern}")
                return True, "SQL"

        # Command injection
        for pattern in self.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, text, re.MULTILINE):
                logger.warning(f"Command injection pattern detected: {pattern}")
                return True, "command"

        # Prompt injection
        for pattern in self.PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                logger.warning(f"Prompt injection pattern detected: {pattern}")
                return True, "prompt"

        return False, None

    def _detect_pii(self, text: str) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Detect PII in text using Presidio

        Args:
            text: Text to check for PII

        Returns:
            Tuple of (detected, entities_list)
        """
        if not self.pii_analyzer:
            return False, []

        try:
            results = self.pii_analyzer.analyze(
                text=text, entities=self.pii_entities, language="en"
            )

            if results:
                entities = [
                    {
                        "type": result.entity_type,
                        "start": result.start,
                        "end": result.end,
                        "score": result.score,
                        "text": text[result.start : result.end],
                    }
                    for result in results
                ]
                return True, entities

            return False, []

        except Exception as e:
            logger.error(f"PII detection error: {e}")
            return False, []

    def sanitize_input(self, text: str) -> str:
        """
        Sanitize input by removing/escaping dangerous characters

        Args:
            text: Text to sanitize

        Returns:
            Sanitized text
        """
        # Remove control characters
        text = re.sub(self.CONTROL_CHARS, "", text)

        # Normalize whitespace
        text = " ".join(text.split())

        # Remove null bytes
        text = text.replace("\x00", "")

        return text.strip()

    def redact_pii(self, text: str) -> str:
        """
        Redact PII from text

        Args:
            text: Text to redact PII from

        Returns:
            Text with PII redacted
        """
        pii_detected, entities = self._detect_pii(text)

        if not pii_detected:
            return text

        # Sort entities by position (reverse order to maintain indices)
        entities_sorted = sorted(entities, key=lambda x: x["start"], reverse=True)

        # Redact each entity
        redacted_text = text
        for entity in entities_sorted:
            redaction = f"[REDACTED_{entity['type']}]"
            redacted_text = (
                redacted_text[: entity["start"]] + redaction + redacted_text[entity["end"] :]
            )

        return redacted_text

    def is_safe_content(self, text: str) -> Tuple[bool, List[str]]:
        """
        Check if content is safe (no prohibited content)

        Args:
            text: Text to check

        Returns:
            Tuple of (is_safe, list_of_issues)
        """
        issues = []

        # Check for extremely long words (potential DoS)
        words = text.split()
        for word in words:
            if len(word) > 100:
                issues.append("Extremely long word detected (potential DoS)")
                break

        # Check for excessive repetition
        if len(set(words)) < len(words) * 0.1 and len(words) > 50:
            issues.append("Excessive repetition detected")

        # Check for binary/encoded content
        if re.search(r"[^\x20-\x7E\s]{50,}", text):
            issues.append("Potential binary or encoded content detected")

        is_safe = len(issues) == 0
        return is_safe, issues

    def validate(self, input_data: str) -> ValidationResult:
        """ABC-compatible validate entry point."""
        return self.validate_task_input(input_data)

    def sanitize(self, input_data: str) -> str:
        """ABC-compatible sanitize entry point."""
        return self.sanitize_input(input_data)


def validate_input(task: str, config: Optional[Dict[str, Any]] = None) -> ValidationResult:
    """
    Convenience function to validate input

    Args:
        task: Task string to validate
        config: Optional configuration

    Returns:
        ValidationResult
    """
    validator = InputValidator(config)
    return validator.validate_task_input(task)
