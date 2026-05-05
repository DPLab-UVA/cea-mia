"""Helpers for grouping probe families into evaluation rounds."""
from __future__ import annotations


def group_probe_pairs_by_round(probe_pairs):
    """Run each probe pair as its own round."""
    return [[pair] for pair in probe_pairs]
