"""
Domain-Specific Exception Hierarchy for Continuous Legal Memory.

This module defines custom exceptions tailored to legal memory management, domain logic constraints,
temporal decay invalidations, and context capacity limits. Distinguishing between system error domains
allows downstream legal agent orchestrators to execute target resilience strategies (e.g., triggering
forgetting policies, context compaction, or fallback routing) rather than catching generic errors.
"""

from typing import Any


class LegalMemoryError(Exception):
    """
    Base exception for all domain-specific errors raised within the Continuous Legal Memory engine.

    Rationale:
        Establishes a common ancestor for exception filtering in upstream LLM agent frameworks
        and observability telemetry pipelines.
    """

    def __init__(self, message: str, payload: Any | None = None) -> None:
        """
        Initialize the base LegalMemoryError.

        Args:
            message: Descriptive failure narrative explaining the root operational cause.
            payload: Optional diagnostic context or state payload associated with the error.
        """
        super().__init__(message)
        self.message = message
        self.payload = payload

    def __str__(self) -> str:
        if self.payload is not None:
            return f"{self.message} | Payload: {self.payload}"
        return self.message


class ContextWindowExceededError(LegalMemoryError):
    """
    Raised when active working context memory exceeds pre-allocated token or memory node limits.

    Rationale:
        Legal documents frequently hit context constraints. Raising an explicit error signals the
        orchestrator to run compaction, sliding-window truncation, or episodic eviction algorithms.
    """
    pass


class MemoryContradictionError(LegalMemoryError):
    """
    Raised when newly ingested legal rules or memory records directly contradict active high-priority node rules.

    Rationale:
        In legal reasoning (e.g., lex specialis vs. lex posterior), identifying conflicting statutes
        is critical. Catching this exception allows surprise-weighted replay adaptation or human-in-the-loop audit logs.
    """
    pass


class TemporalInvalidationError(LegalMemoryError):
    """
    Raised when querying memory nodes whose temporal validity range (`valid_from` to `valid_to`) has expired.

    Rationale:
        Enforces strict legal compliance and GDPR Art. 17 right-to-be-forgotten auditability by preventing
        stale or invalid statutes from influencing modern inference sessions.
    """
    pass


class EncoderInferenceError(LegalMemoryError):
    """
    Raised when an underlying embedding or LLM encoder model fails to process text input.

    Rationale:
        Isolates infrastructure-level LLM and API vector extraction failures from internal cognitive neural logic.
    """
    pass


class InvalidMemoryVectorError(LegalMemoryError):
    """
    Raised when key embeddings or action value vectors violate shape, dimension, or numerical assertions.

    Rationale:
        Guarantees input tensor integrity before executing PyTorch neural matrix operations, avoiding silent NaN or shape mismatch errors.
    """
    pass
