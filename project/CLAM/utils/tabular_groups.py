"""Token grouping for the adapted co-attention fusion baseline.

The co-attention operator needs the tabular modality split into a small number of tokens.
Two grouping specifications are supported:

``prefix``
    Group one-hot blocks by column-name prefix (everything before the final underscore),
    which recovers the natural clinicopathology blocks (age, stage, T, N, M, histology).

a path to a signature CSV
    Each column is a gene-set name and its rows are gene symbols; a feature joins every
    group that lists it, matching the overlapping gene sets the co-attention literature uses.

Features matching no group are collected into a final ``unassigned`` token, so the
co-attention arm sees exactly the same feature set as every other arm and only the fusion
operator differs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


MAX_TOKENS = 64


def build_tabular_groups(feature_names, spec: str):
    """Return ``(group_names, group_indices)`` for the given selected feature names."""
    if not feature_names:
        raise ValueError("feature_names is empty; the tabular transform must be fitted first.")

    if spec == "prefix":
        groups: dict[str, list[int]] = {}
        for idx, name in enumerate(feature_names):
            prefix = name.rsplit("_", 1)[0] if "_" in name else name
            groups.setdefault(prefix, []).append(idx)
        # Prefix grouping only makes sense for one-hot blocks. Applied to bare gene symbols
        # every feature becomes its own token, which would build thousands of Linear(1, H)
        # encoders and a multi-thousand-token attention -- slow, useless, and silent.
        if len(groups) > MAX_TOKENS:
            raise ValueError(
                f"'prefix' grouping produced {len(groups)} tokens from {len(feature_names)} "
                f"features, above the limit of {MAX_TOKENS}. This spec is for one-hot blocks "
                "(clinicopathology); for gene expression pass a signature CSV instead."
            )
        names = sorted(groups)
        return names, [groups[name] for name in names]

    csv_path = Path(spec)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"--tabular_group_spec must be 'prefix' or a readable signature CSV; got {spec}"
        )

    signatures = pd.read_csv(csv_path)
    position = {name: idx for idx, name in enumerate(feature_names)}
    names, indices, assigned = [], [], set()
    for column in signatures.columns:
        members = {str(value) for value in signatures[column].dropna()}
        matched = sorted(position[name] for name in members if name in position)
        if not matched:
            continue
        names.append(column)
        indices.append(matched)
        assigned.update(matched)

    # Fail loudly rather than degrade to a single catch-all token. Without this, a signature
    # CSV whose identifiers do not match the feature naming (wrong ID space, numeric column,
    # wrong file) would silently produce one giant 'unassigned' group -- co-attention over a
    # single token, which trains happily and looks like a valid baseline while testing
    # nothing.
    if not names:
        raise ValueError(
            f"No feature matched any group in {csv_path}. Checked {len(feature_names)} "
            f"feature names against {len(signatures.columns)} groups; e.g. "
            f"{feature_names[:3]} vs {[str(v) for v in signatures.iloc[:3, 0].dropna()]}."
        )

    unassigned = sorted(set(range(len(feature_names))) - assigned)
    if unassigned:
        names.append("unassigned")
        indices.append(unassigned)
    return names, indices
