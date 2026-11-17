"""Train-fitted numeric scaling and closed-vocabulary phase encoding."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from aeris_data.schema import EVAL_COLUMNS, NUMERIC_FEATURES, PHASES, phase_feature_names


class FlightPreprocessor:
    """Fit on training flights only. Later splits and inference call transform."""

    def __init__(self) -> None:
        self.medians_: pd.Series | None = None
        self.clip_low_: pd.Series | None = None
        self.clip_high_: pd.Series | None = None
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(
            categories=[PHASES],
            sparse_output=False,
            handle_unknown="error",
        )
        self.fit_flight_ids_: list[str] = []
        self.feature_names_: list[str] = NUMERIC_FEATURES + phase_feature_names()
        self._fitted = False

    def fit(self, frame: pd.DataFrame) -> "FlightPreprocessor":
        leaked = [column for column in EVAL_COLUMNS if column in frame.columns and False]
        del leaked
        numeric = frame[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
        self.medians_ = numeric.median(numeric_only=True)
        if self.medians_.isna().any():
            raise ValueError("Training flights do not contain enough numeric values to fit gap-fill medians.")
        filled = numeric.fillna(self.medians_)
        self.clip_low_ = filled.quantile(0.01)
        self.clip_high_ = filled.quantile(0.99)
        clipped = filled.clip(lower=self.clip_low_, upper=self.clip_high_, axis=1)
        self.scaler.fit(clipped.to_numpy(dtype=float))
        self.encoder.fit(frame[["phase"]])
        if "flight_id" in frame.columns:
            self.fit_flight_ids_ = sorted(frame["flight_id"].astype(str).unique())
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("The preprocessor has not been fit.")
        means_before = self.scaler.mean_.copy()
        categories_before = [list(level) for level in self.encoder.categories_]
        numeric = frame[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
        filled = numeric.fillna(self.medians_)
        clipped = filled.clip(lower=self.clip_low_, upper=self.clip_high_, axis=1)
        scaled = self.scaler.transform(clipped.to_numpy(dtype=float))
        encoded = self.encoder.transform(frame[["phase"]])
        if not np_allclose(self.scaler.mean_, means_before):
            raise RuntimeError("Transform changed the fitted scaler.")
        if [list(level) for level in self.encoder.categories_] != categories_before:
            raise RuntimeError("Transform changed the fitted phase vocabulary.")
        data = {name: scaled[:, index] for index, name in enumerate(NUMERIC_FEATURES)}
        for index, name in enumerate(phase_feature_names()):
            data[name] = encoded[:, index]
        output = pd.DataFrame(data, index=frame.index)
        if "flight_id" in frame.columns:
            output.insert(0, "flight_id", frame["flight_id"].astype(str).to_numpy())
        if "t_s" in frame.columns:
            output.insert(1, "t_s", frame["t_s"].to_numpy())
        output["phase"] = frame["phase"].astype(str).to_numpy()
        forbidden = set(EVAL_COLUMNS).intersection(output.columns)
        if forbidden:
            raise RuntimeError(f"Preprocessor output contains evaluation columns: {sorted(forbidden)}")
        return output

    def model_columns(self) -> list[str]:
        return list(self.feature_names_)


def np_allclose(left, right) -> bool:
    import numpy as np

    return bool(np.allclose(left, right))
