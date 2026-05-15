"""Compatibility wrapper for the BJ/oral domain classifier.

The old "trap guard" command remains available, but the canonical terminology
is BJ/oral domain classification.  BJ/oral motion is a valid semantic family.
"""

from __future__ import annotations

from typing import Any

from vam_timeline_ai.semantics.bj_oral_domain_classifier import (
    bj_oral_domain_for_window,
    classify_bj_oral_domain,
)


def audit_bj_oral_trap_guard(*args: Any, **kwargs: Any):
    return classify_bj_oral_domain(*args, **kwargs)


def bj_oral_trap_guard_for_window(*args: Any, **kwargs: Any):
    return bj_oral_domain_for_window(*args, **kwargs)
