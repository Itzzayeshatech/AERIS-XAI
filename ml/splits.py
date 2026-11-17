"""Flight-level train, validation, and held-out test splits."""

from __future__ import annotations

import numpy as np

from ml.errors import InsufficientFlightsError

MIN_DISTINCT_FLIGHTS = 5


def split_flight_ids(
    flight_ids: list[str],
    seed: int,
    labels: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    """Split distinct flight ids without allowing a flight to cross partitions."""
    unique = sorted(set(flight_ids))
    found = len(unique)
    if found < MIN_DISTINCT_FLIGHTS:
        raise InsufficientFlightsError(found, MIN_DISTINCT_FLIGHTS)

    n = len(unique)
    n_train = max(1, int(round(0.6 * n)))
    n_val = max(1, int(round(0.2 * n)))
    if n_train + n_val > n - 1:
        overflow = n_train + n_val - (n - 1)
        n_train -= overflow
    n_test = n - n_train - n_val

    rng = np.random.default_rng(seed)
    pool = unique.copy()
    rng.shuffle(pool)

    test: list[str] = []
    if labels and n_test >= 2:
        clean = [fid for fid in pool if labels.get(fid) == "clean"]
        anomalous = [fid for fid in pool if labels.get(fid) == "anomalous"]
        if clean and anomalous:
            test.extend([clean[0], anomalous[0]])

    remaining = [fid for fid in pool if fid not in set(test)]
    fill_count = n_test - len(test)
    if fill_count > 0:
        test.extend(remaining[:fill_count])
        remaining = remaining[fill_count:]

    rng.shuffle(remaining)
    n_train = min(n_train, len(remaining) - 1)
    train = remaining[:n_train]
    validation = remaining[n_train:n_train + n_val]

    # Defensive complete coverage if rounding left an unexpected partition size.
    assigned = set(train) | set(validation) | set(test)
    if len(assigned) != n:
        for fid in remaining:
            if fid not in assigned:
                train.append(fid)
                assigned.add(fid)

    split = {"train": train, "validation": validation, "test": test}
    _assert_split(set(unique), split)
    return split


def assert_split_integrity(flight_ids: set[str], split: dict[str, list[str]]) -> None:
    _assert_split(flight_ids, split)


def _assert_split(flight_ids: set[str], split: dict[str, list[str]]) -> None:
    train, validation, test = set(split["train"]), set(split["validation"]), set(split["test"])
    if train & validation or validation & test or train & test:
        raise RuntimeError("A flight_id crossed a split boundary.")
    if train | validation | test != flight_ids:
        raise RuntimeError("The split does not cover every flight exactly once.")
    if not train or not validation or not test:
        raise RuntimeError("A split partition is empty.")
