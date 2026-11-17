"""Domain errors that must stay distinct from a metric value of zero."""

from __future__ import annotations


class InsufficientFlightsError(ValueError):
    def __init__(self, found: int, minimum: int = 5) -> None:
        self.found = found
        self.minimum = minimum
        super().__init__(
            f"Dataset has {found} distinct flights. A three-way split requires at least {minimum} "
            "distinct flights so that train, validation, and held-out test would each receive a flight. "
            "No dataset version was created."
        )


class TrainingError(RuntimeError):
    """A trainer stopped before writing a threshold or a fake zero metric."""
