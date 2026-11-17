"""Feature ablation and temporal perturbation for LSTM Autoencoder sequence explanations."""
from __future__ import annotations

from typing import Any
import numpy as np

from ml.models.lstm_autoencoder import FlightLSTMAutoencoder


class LSTMAblationExplainer:
    """Attribute reconstruction error to individual telemetry channels via systematic ablation."""

    def __init__(self, model_wrapper: FlightLSTMAutoencoder) -> None:
        self.model_wrapper = model_wrapper
        self.feature_columns = model_wrapper.feature_columns

    def explain_sequence(
        self,
        sequence: np.ndarray,
        top_k: int = 8,
    ) -> dict[str, Any]:
        """Explain a sequence of shape (seq_len, num_features).

        Algorithm:
        1. Measure baseline reconstruction error of the full sequence.
        2. For each telemetry channel, zero out or replace with channel mean across sequence (ablation).
        3. Measure the change in reconstruction error: channels causing high reconstruction error
           produce the greatest drop in error when replaced, or highest sensitivity.
        """
        if sequence.ndim == 2:
            seq_3d = sequence[np.newaxis, :, :]
        else:
            seq_3d = sequence

        baseline_error = float(self.model_wrapper.reconstruction_errors(seq_3d)[0])
        num_features = seq_3d.shape[2]

        channel_contributions: dict[str, float] = {}

        # Channel ablation loop
        for f_idx in range(num_features):
            f_name = self.feature_columns[f_idx]
            ablated_seq = seq_3d.copy()
            # Replace with channel temporal mean (removes dynamic excursions)
            ablated_seq[0, :, f_idx] = np.mean(ablated_seq[0, :, f_idx])
            ablated_error = float(self.model_wrapper.reconstruction_errors(ablated_seq)[0])

            # Positive delta = channel was contributing to higher reconstruction error
            error_delta = baseline_error - ablated_error
            channel_contributions[f_name] = round(error_delta, 6)

        # Rank features by contribution
        sorted_feats = sorted(channel_contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)

        top_features: list[dict[str, Any]] = []
        for rank, (fname, delta) in enumerate(sorted_feats):
            idx = self.feature_columns.index(fname)
            current_mean = float(np.mean(seq_3d[0, :, idx]))
            if rank < top_k:
                top_features.append({
                    "feature_name": fname,
                    "attribution": delta,
                    "feature_value": round(current_mean, 4),
                    "direction": "destabilized_reconstruction" if delta > 0 else "neutral_or_stabilizing",
                })

        high_error_channels = [f["feature_name"] for f in top_features if f["attribution"] > 0]
        if high_error_channels:
            summary = (
                f"LSTM reconstruction failure was predominantly triggered by temporal instability in "
                f"{', '.join(high_error_channels[:3])}. "
                f"Replacing these channels with steady-state values significantly restored autoencoder fidelity."
            )
        else:
            summary = "Temporal sequence aligns with nominal aerodynamic trajectories."

        return {
            "method": "feature_ablation_perturbation",
            "base_value": round(baseline_error, 6),
            "top_features": top_features,
            "all_attributions": channel_contributions,
            "summary_text": summary,
        }
