"""Helpers for grouping probe families into evaluation rounds."""
from __future__ import annotations

from models import ProbeType


def group_probe_pairs_by_round(probe_pairs, max_rounds=None):
    """Keep singleton rounds except for contiguous paraphrase variants."""
    grouped = []
    current_batch = []

    for pair in probe_pairs:
        probe_type = pair[0].probe_type
        if probe_type == ProbeType.PARAPHRASE:
            current_batch.append(pair)
            continue

        if current_batch:
            grouped.append(current_batch)
            current_batch = []
        grouped.append([pair])

    if current_batch:
        grouped.append(current_batch)

    if max_rounds is not None:
        return grouped[:max_rounds]
    return grouped
