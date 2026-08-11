"""Deployment policies for cache precision and memory budgeting."""

from .joint_precision import NoFeasibleCandidate, select_joint_precision

__all__ = ["NoFeasibleCandidate", "select_joint_precision"]
