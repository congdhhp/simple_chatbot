"""Safety filters and PII detection pipeline (Wave 2)."""

import re
import logging
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class SafetyLevel(Enum):
    """Safety levels for content filtering."""
    STRICT = "strict"
    MODERATE = "moderate"
    PERMISSIVE = "permissive"


class FilterAction(Enum):
    """Actions to take when content is flagged."""
    ALLOW = "allow"
    BLOCK = "block"
    REDACT = "redact"
    WARN = "warn"


@dataclass
class FilterResult:
    """Result of content filtering."""
    action: FilterAction
    filtered_content: str
    flags: List[str]
    confidence: float
    metadata: Dict[str, Any]


class PIIDetector:
    """PII detection and redaction."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # PII patterns (basic regex-based detection)
        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),
            'ssn': re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            'credit_card': re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
            'ip_address': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            'api_key': re.compile(r'\b[A-Za-z0-9]{32,}\b'),  # Generic API key pattern
            'jwt_token': re.compile(r'\beyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*\b'),
        }
    
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """Detect PII in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary mapping PII types to found instances
        """
        detected = {}
        
        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                detected[pii_type] = matches
        
        return detected
    
    def redact_pii(self, text: str, redaction_method: str = "hash") -> Tuple[str, Dict[str, List[str]]]:
        """Redact PII from text.
        
        Args:
            text: Text to redact
            redaction_method: Method for redaction ('mask', 'hash', 'remove')
            
        Returns:
            Tuple of (redacted_text, detected_pii)
        """
        detected = self.detect_pii(text)
        redacted_text = text
        
        for pii_type, matches in detected.items():
            for match in matches:
                if redaction_method == "mask":
                    replacement = f"[{pii_type.upper()}_REDACTED]"
                elif redaction_method == "hash":
                    hash_value = hashlib.md5(match.encode()).hexdigest()[:8]
                    replacement = f"[{pii_type.upper()}_{hash_value}]"
                elif redaction_method == "remove":
                    replacement = ""
                else:
                    replacement = f"[{pii_type.upper()}_REDACTED]"
                
                redacted_text = redacted_text.replace(match, replacement)
        
        return redacted_text, detected


class ContentFilter:
    """Content safety filtering."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Harmful content patterns (basic keyword-based)
        self.harmful_patterns = {
            'violence': [
                r'\b(?:kill|murder|assault|attack|violence|weapon|bomb|gun)\b',
                r'\b(?:hurt|harm|damage|destroy|violent)\b'
            ],
            'hate_speech': [
                r'\b(?:hate|racist|discrimination|prejudice)\b',
                # Add more patterns as needed
            ],
            'harassment': [
                r'\b(?:harass|bully|threaten|intimidate|stalk)\b'
            ],
            'explicit': [
                r'\b(?:explicit|nsfw|adult|sexual)\b'
            ],
            'illegal': [
                r'\b(?:illegal|drug|trafficking|fraud|scam)\b'
            ]
        }
    
    def check_content(self, text: str, safety_level: SafetyLevel = SafetyLevel.MODERATE) -> FilterResult:
        """Check content for safety violations.
        
        Args:
            text: Text to check
            safety_level: Safety level to apply
            
        Returns:
            FilterResult with action and details
        """
        flags = []
        confidence = 0.0
        
        # Check for harmful patterns
        for category, patterns in self.harmful_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    flags.append(f"{category}_detected")
                    confidence += 0.2  # Simple confidence scoring
        
        # Determine action based on flags and safety level
        if flags:
            if safety_level == SafetyLevel.STRICT:
                action = FilterAction.BLOCK
            elif safety_level == SafetyLevel.MODERATE:
                action = FilterAction.WARN if confidence < 0.5 else FilterAction.BLOCK
            else:  # PERMISSIVE
                action = FilterAction.WARN
        else:
            action = FilterAction.ALLOW
        
        return FilterResult(
            action=action,
            filtered_content=text,  # Could implement content modification here
            flags=flags,
            confidence=min(confidence, 1.0),
            metadata={
                "safety_level": safety_level.value,
                "patterns_triggered": flags
            }
        )


class SafetyPipeline:
    """Complete safety pipeline combining PII detection and content filtering."""
    
    def __init__(self, safety_level: SafetyLevel = SafetyLevel.MODERATE, enable_pii_detection: bool = True):
        """Initialize safety pipeline.
        
        Args:
            safety_level: Default safety level
            enable_pii_detection: Whether to enable PII detection
        """
        self.safety_level = safety_level
        self.enable_pii_detection = enable_pii_detection
        self.pii_detector = PIIDetector() if enable_pii_detection else None
        self.content_filter = ContentFilter()
        self.logger = logging.getLogger(__name__)
    
    def process_input(self, text: str, user_id: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """Process input text through safety pipeline.
        
        Args:
            text: Input text to process
            user_id: Optional user ID for logging
            
        Returns:
            Tuple of (processed_text, safety_report)
        """
        safety_report = {
            "pii_detected": {},
            "content_flags": [],
            "actions_taken": [],
            "filtered_text": text
        }
        
        processed_text = text
        
        # PII Detection and Redaction
        if self.pii_detector:
            processed_text, pii_detected = self.pii_detector.redact_pii(processed_text)
            safety_report["pii_detected"] = pii_detected
            
            if pii_detected:
                safety_report["actions_taken"].append("pii_redacted")
                self.logger.warning(f"PII detected and redacted for user {user_id}: {list(pii_detected.keys())}")
        
        # Content Safety Filtering
        filter_result = self.content_filter.check_content(processed_text, self.safety_level)
        safety_report["content_flags"] = filter_result.flags
        safety_report["filter_confidence"] = filter_result.confidence
        
        if filter_result.action == FilterAction.BLOCK:
            safety_report["actions_taken"].append("content_blocked")
            safety_report["filtered_text"] = "[CONTENT_BLOCKED: Violates safety guidelines]"
            processed_text = safety_report["filtered_text"]
            
            self.logger.warning(f"Content blocked for user {user_id}: {filter_result.flags}")
            
        elif filter_result.action == FilterAction.WARN:
            safety_report["actions_taken"].append("content_warned")
            self.logger.info(f"Content flagged but allowed for user {user_id}: {filter_result.flags}")
        
        return processed_text, safety_report
    
    def process_output(self, text: str, user_id: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """Process output text through safety pipeline.
        
        Args:
            text: Output text to process
            user_id: Optional user ID for logging
            
        Returns:
            Tuple of (processed_text, safety_report)
        """
        # For output, we might be less strict but still check for PII leakage
        safety_report = {
            "pii_detected": {},
            "content_flags": [],
            "actions_taken": [],
            "filtered_text": text
        }
        
        processed_text = text
        
        # PII Detection (but maybe don't redact, just warn)
        if self.pii_detector:
            pii_detected = self.pii_detector.detect_pii(processed_text)
            safety_report["pii_detected"] = pii_detected
            
            if pii_detected:
                safety_report["actions_taken"].append("pii_detected_in_output")
                self.logger.warning(f"PII detected in model output for user {user_id}: {list(pii_detected.keys())}")
        
        # Content filtering with more permissive level for outputs
        output_safety_level = SafetyLevel.PERMISSIVE if self.safety_level == SafetyLevel.MODERATE else self.safety_level
        filter_result = self.content_filter.check_content(processed_text, output_safety_level)
        safety_report["content_flags"] = filter_result.flags
        safety_report["filter_confidence"] = filter_result.confidence
        
        if filter_result.action == FilterAction.BLOCK:
            safety_report["actions_taken"].append("output_blocked")
            safety_report["filtered_text"] = "[OUTPUT_BLOCKED: Response violates safety guidelines]"
            processed_text = safety_report["filtered_text"]
            
            self.logger.warning(f"Model output blocked for user {user_id}: {filter_result.flags}")
        
        return processed_text, safety_report


# Factory function for easy instantiation
def create_safety_pipeline(config: Optional[Dict[str, Any]] = None) -> SafetyPipeline:
    """Create a safety pipeline with configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured SafetyPipeline instance
    """
    if config is None:
        config = {}
    
    safety_level_str = config.get('safety_level', 'moderate')
    safety_level = SafetyLevel(safety_level_str)
    
    enable_pii = config.get('enable_pii_detection', True)
    
    return SafetyPipeline(safety_level=safety_level, enable_pii_detection=enable_pii)
