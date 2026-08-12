"""Deployment policies for cache precision and memory budgeting."""

from .joint_precision import (
    CAPACITY_BYTES_SEMANTICS,
    NoFeasibleCandidate,
    PolicyInputError,
    canonical_precision_args,
    select_joint_precision,
    validate_joint_precision_profile,
)

__all__ = [
    "CAPACITY_BYTES_SEMANTICS",
    "NoFeasibleCandidate",
    "PolicyInputError",
    "canonical_precision_args",
    "select_joint_precision",
    "validate_joint_precision_profile",
]
