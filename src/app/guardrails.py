# src/app/guardrails.py
import re
from typing import Tuple, Optional


class CustomGuardrails:
    def __init__(self):
        # 1. Input Validation Rules
        self.injection_keywords = [
            "ignore previous instructions",
            "system prompt",
            "delete database",
            "unrestricted mode",
            "execute command",
        ]

        # Regex for Pakistani PII (CNIC and Phone) + Email
        self.pii_patterns = {
            "CNIC": r"\d{5}-\d{7}-\d{1}",  # Matches 12345-1234567-1
            "Phone": r"(\+92|0)?3\d{9}",  # Matches +923001234567 or 03001234567
            "Email": r"[^@]+@[^@]+\.[^@]+",  # Basic email pattern
        }

        # 2. Output Moderation Rules
        self.banned_words = [
            "cheat",
            "scam",
            "hack",
            "password",
            "kill",
            "suicide",
            "fuck",
            "shit",
            "bitch",  # Toxicity filters
        ]

    def check_input(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validates User Input.
        Returns: (is_safe: bool, reason: str)
        """
        query_lower = query.lower()

        # Rule 1: Prompt Injection
        for keyword in self.injection_keywords:
            if keyword in query_lower:
                return False, f"Prompt Injection Detected: '{keyword}'"

        # Rule 2: PII Detection
        for pii_type, pattern in self.pii_patterns.items():
            if re.search(pattern, query):
                return False, f"PII Detected ({pii_type}) - Request Blocked"

        return True, None

    def check_output(self, response: str) -> Tuple[bool, Optional[str]]:
        """
        Validates Model Output.
        Returns: (is_safe: bool, reason: str)
        """
        response_lower = response.lower()

        # Rule 3: Toxicity / Banned Content
        for word in self.banned_words:
            if f" {word} " in f" {response_lower} ":
                return False, f"Toxic/Banned Content Detected: '{word}'"

        # Rule 4: Hallucination/Empty Check (Basic)
        if not response or len(response.strip()) < 5:
            return False, "Response too short or empty (Potential Error)"

        return True, None
