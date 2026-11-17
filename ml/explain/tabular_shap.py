"""SHAP feature attribution for Isolation Forest anomaly detection."""
from __future__ import annotations

from typing import Any
import numpy as np
import shap

from ml.models.isolation_forest import FlightIsolationForest


class IsolationForestShapExplainer:
    """Computes Shapley values for Isolation Forest predictions using TreeExplainer."""

    def __init__(self, model_wrapper: FlightIsolationForest) -> None:
        self.model_wrapper = model_wrapper
        # IsolationForest is a tree-based ensemble supported by TreeExplainer
        self.explainer = shap.TreeExplainer(model_wrapper.model)
        self.feature_names = model_wrapper.feature_names

    def explain_window(
        self,
        X_single: np.ndarray,
        top_k: int = 8,
    ) -> dict[str, Any]:
        """Explain a single feature vector of shape (num_features,) or (1, num_features)."""
        if X_single.ndim == 1:
            X_arr = X_single.reshape(1, -1)
        else:
            X_arr = X_single

        # Compute SHAP values
        shap_values = self.explainer.shap_values(X_arr)
        if isinstance(shap_values, list):
            vals = shap_values[0][0]
        else:
            vals = shap_values[0]

        # For IsolationForest, TreeExplainer outputs attribution toward normality / anomaly
        # Invert or align so positive SHAP indicates contribution toward anomaly detection
        # Note: In scikit-learn, negative score indicates anomaly.
        anomaly_attribution = -vals

        # Sort features by absolute contribution
        sorted_indices = np.argsort(np.abs(anomaly_attribution))[::-1]

        top_features: list[dict[str, Any]] = []
        all_attributions: dict[str, float] = {}

        for rank, idx in enumerate(sorted_indices):
            fname = self.feature_names[idx]
            attr_val = float(anomaly_attribution[idx])
            feat_val = float(X_arr[0, idx])
            all_attributions[fname] = round(attr_val, 6)

            if rank < top_k:
                top_features.append({
                    "feature_name": fname,
                    "attribution": round(attr_val, 6),
                    "feature_value": round(feat_val, 4),
                    "direction": "increases_anomaly_risk" if attr_val > 0 else "supports_normal_profile",
                })

        base_val = float(self.explainer.expected_value) if np.isscalar(self.explainer.expected_value) else float(self.explainer.expected_value[0])

        # Generate human-readable aerospace engineering summary
        top_anomaly_contributors = [f["feature_name"] for f in top_features if f["attribution"] > 0]
        if top_anomaly_contributors:
            narrative = (
                f"Anomaly score was primarily driven by deviations in "
                f"{', '.join(top_anomaly_contributors[:3])}. "
                f"These parameters showed abnormal variance or extreme offsets from the nominal training envelope."
            )
        else:
            narrative = "Observations fall comfortably within nominal multi-dimensional flight parameters."

        return {
            "method": "shap_tree_explainer",
            "base_value": round(-base_val, 4),
            "top_features": top_features,
            "all_attributions": all_attributions,
            "summary_text": narrative,
        }
