"""PyTorch LSTM Autoencoder for temporal aircraft flight behaviour anomaly detection.

Beginner-Friendly Architecture Overview:
----------------------------------------
An Autoencoder is a neural network designed to learn "normal patterns" by compressing
input sequences into a compact code (the "bottleneck") and then attempting to reconstruct
the exact original sequence from that code.

1. The ENCODER: Reads the sequence of aircraft telemetry step-by-step (e.g., 30 seconds
   of pitch, roll, airspeed, and altitude) using LSTM (Long Short-Term Memory) cells.
   It captures how aerodynamic states evolve smoothly over time and summarizes the entire
   window into a compact hidden latent vector.
2. The BOTTLENECK: A compressed representation storing only the essential physics of
   healthy flight behavior.
3. The DECODER: Takes the compressed vector and reconstructs the sequence of 30 timesteps.
4. RECONSTRUCTION ERROR:
   - When fed a NORMAL flight, the model reconstructs it faithfully with LOW error.
   - When an ANOMALY occurs (e.g., sensor glitch, sudden attitude excursion, engine flameout),
     the autoencoder fails to reconstruct the unfamiliar pattern, resulting in a HIGH error.
   If reconstruction error exceeds a calibrated threshold, the window is flagged.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.metrics import detection_metrics
from ml.preprocess import FlightPreprocessor
from ml.windows import iter_windows, sequence_matrix, window_label


class Encoder(nn.Module):
    """LSTM Encoder compressing sequential observations into a bottleneck vector."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 24, num_layers: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)
        # h_n[-1] is the final layer hidden state: (batch_size, hidden_dim)
        latent = self.fc(h_n[-1])
        return latent  # shape: (batch_size, latent_dim)


class Decoder(nn.Module):
    """LSTM Decoder reconstructing sequence from latent bottleneck."""

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, seq_len: int, num_layers: int = 2) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.fc_in = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # latent shape: (batch_size, latent_dim)
        h0 = self.fc_in(latent)  # (batch_size, hidden_dim)
        # Repeat vector along sequence length: (batch_size, seq_len, hidden_dim)
        repeated = h0.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(repeated)
        reconstruction = self.fc_out(out)
        return reconstruction  # (batch_size, seq_len, output_dim)


class LSTMAutoencoderNet(nn.Module):
    """Complete Autoencoder combining Encoder and Decoder."""

    def __init__(
        self,
        input_dim: int,
        seq_len: int = 30,
        hidden_dim: int = 64,
        latent_dim: int = 24,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, num_layers=num_layers)
        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            seq_len=seq_len,
            num_layers=num_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


class FlightLSTMAutoencoder:
    """Wrapper managing LSTM Autoencoder training, thresholding, and evaluation."""

    def __init__(
        self,
        seq_len: int = 30,
        hidden_dim: int = 48,
        latent_dim: int = 16,
        num_layers: int = 1,
        learning_rate: float = 0.002,
        batch_size: int = 16,
        epochs: int = 25,
        patience: int = 5,
        random_seed: int = 42,
    ) -> None:
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_seed = random_seed

        self.input_dim: int | None = None
        self.feature_columns: list[str] = []
        self.model: LSTMAutoencoderNet | None = None
        self.threshold: float = 0.0
        self.training_history: list[dict[str, float]] = []
        self.is_fitted: bool = False

        # Set seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def extract_sequences(
        self,
        frame: pd.DataFrame,
        preprocessor: FlightPreprocessor,
        length: int = 30,
        stride: int = 10,
    ) -> tuple[np.ndarray, list[dict[str, Any]], list[int | None]]:
        """Extract continuous sequences of shape (N, length, features)."""
        transformed = preprocessor.transform(frame)
        self.feature_columns = preprocessor.model_columns()
        self.input_dim = len(self.feature_columns)

        sequences: list[np.ndarray] = []
        meta_list: list[dict[str, Any]] = []
        labels: list[int | None] = []

        for flight_id, window in iter_windows(transformed, length=length, stride=stride):
            seq = sequence_matrix(window, self.feature_columns)
            sequences.append(seq)

            start_t = float(window["t_s"].iloc[0])
            end_t = float(window["t_s"].iloc[-1])
            lbl = window_label(window)

            meta_list.append({
                "flight_id": str(flight_id),
                "start_t_s": start_t,
                "end_t_s": end_t,
                "phase_mode": str(window["phase"].mode(dropna=False).iloc[0]),
            })
            labels.append(lbl)

        if not sequences:
            return np.empty((0, length, self.input_dim)), [], []

        return np.stack(sequences, axis=0), meta_list, labels

    def fit(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        threshold_quantile: float = 0.95,
    ) -> "FlightLSTMAutoencoder":
        """Train the autoencoder on training sequences and calibrate threshold on validation."""
        if len(X_train) == 0:
            raise ValueError("Training set must not be empty.")

        self.input_dim = X_train.shape[2]
        self.seq_len = X_train.shape[1]

        self.model = LSTMAutoencoderNet(
            input_dim=self.input_dim,
            seq_len=self.seq_len,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_layers=self.num_layers,
        )

        device = torch.device("cpu")
        self.model.to(device)

        train_tensor = torch.tensor(X_train, dtype=torch.float32)
        train_loader = DataLoader(
            TensorDataset(train_tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        self.training_history = []

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_losses: list[float] = []

            for (batch_x,) in train_loader:
                batch_x = batch_x.to(device)
                optimizer.zero_grad()
                recon = self.model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = float(np.mean(train_losses))

            # Evaluate on validation
            val_loss = avg_train_loss
            if len(X_val) > 0:
                self.model.eval()
                with torch.no_grad():
                    val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
                    val_recon = self.model(val_t)
                    val_loss = float(criterion(val_recon, val_t).item())

            self.training_history.append({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
            })

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Calibrate detection threshold on validation errors
        calibration_set = X_val if len(X_val) > 0 else X_train
        val_errors = self.reconstruction_errors(calibration_set)
        self.threshold = float(np.quantile(val_errors, threshold_quantile))
        self.is_fitted = True
        return self

    def reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """Compute MSE reconstruction error for each sequence."""
        if self.model is None or not self.is_fitted:
            raise RuntimeError("Model must be fitted before computing reconstruction errors.")
        if len(X) == 0:
            return np.empty((0,), dtype=float)

        self.model.eval()
        with torch.no_grad():
            t_x = torch.tensor(X, dtype=torch.float32)
            recon = self.model(t_x)
            # Per-sequence MSE across all timesteps and features
            diff = (t_x - recon) ** 2
            mse_per_sequence = diff.mean(dim=(1, 2)).numpy()
            return mse_per_sequence

    def predict(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Predict binary anomalies: 1 if error >= threshold, else 0."""
        errors = self.reconstruction_errors(X)
        th = threshold if threshold is not None else self.threshold
        return (errors >= th).astype(int)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: list[int | None],
        test_flight_ids: list[str],
        test_flight_labels: dict[str, str],
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Evaluate performance on test sequences."""
        scores = self.reconstruction_errors(X_test)
        preds = self.predict(X_test, threshold=threshold)

        has_any_label = any(y is not None for y in y_test)
        if has_any_label:
            n_clean = sum(1 for fid, kind in test_flight_labels.items() if kind == "clean")
            n_anom = sum(1 for fid, kind in test_flight_labels.items() if kind == "anomalous")

            valid_indices = [i for i, y in enumerate(y_test) if y is not None]
            y_true_valid = [y_test[i] for i in valid_indices]
            y_score_valid = [float(scores[i]) for i in valid_indices]
            y_pred_valid = [int(preds[i]) for i in valid_indices]

            metrics = detection_metrics(
                y_true=y_true_valid,
                y_score=y_score_valid,
                y_pred=y_pred_valid,
                n_clean_flights=n_clean,
                n_anomalous_flights=n_anom,
            )
            metrics["threshold_applied"] = threshold if threshold is not None else self.threshold
            metrics["total_windows_evaluated"] = len(valid_indices)
            metrics["flagged_windows"] = int(np.sum(preds))
            return metrics

        return {
            "evaluation_scope": "unsupervised_diagnostics",
            "status": "unsupervised",
            "reason": "Test dataset contains no ground truth anomaly labels.",
            "threshold_applied": threshold if threshold is not None else self.threshold,
            "total_windows_evaluated": len(X_test),
            "flagged_windows": int(np.sum(preds)),
            "outlier_ratio": float(np.mean(preds)) if len(preds) > 0 else 0.0,
            "reconstruction_error_mean": float(np.mean(scores)) if len(scores) > 0 else 0.0,
            "reconstruction_error_p50": float(np.quantile(scores, 0.50)) if len(scores) > 0 else 0.0,
            "reconstruction_error_p95": float(np.quantile(scores, 0.95)) if len(scores) > 0 else 0.0,
            "reconstruction_error_max": float(np.max(scores)) if len(scores) > 0 else 0.0,
        }

    def save(self, filepath: Path | str) -> None:
        """Save model weights and hyperparameters."""
        filepath = Path(filepath)
        state = {
            "model_state": self.model.state_dict() if self.model else None,
            "input_dim": self.input_dim,
            "seq_len": self.seq_len,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "num_layers": self.num_layers,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "threshold": self.threshold,
            "feature_columns": self.feature_columns,
            "training_history": self.training_history,
            "is_fitted": self.is_fitted,
        }
        torch.save(state, filepath)

    @classmethod
    def load(cls, filepath: Path | str) -> "FlightLSTMAutoencoder":
        """Load trained autoencoder from disk."""
        state = torch.load(filepath, map_location="cpu", weights_only=False)
        instance = cls(
            seq_len=state["seq_len"],
            hidden_dim=state["hidden_dim"],
            latent_dim=state["latent_dim"],
            num_layers=state["num_layers"],
            learning_rate=state["learning_rate"],
            batch_size=state["batch_size"],
            epochs=state["epochs"],
        )
        instance.input_dim = state["input_dim"]
        instance.feature_columns = state["feature_columns"]
        instance.threshold = state["threshold"]
        instance.training_history = state.get("training_history", [])
        instance.is_fitted = state["is_fitted"]

        instance.model = LSTMAutoencoderNet(
            input_dim=instance.input_dim,
            seq_len=instance.seq_len,
            hidden_dim=instance.hidden_dim,
            latent_dim=instance.latent_dim,
            num_layers=instance.num_layers,
        )
        if state["model_state"] is not None:
            instance.model.load_state_dict(state["model_state"])
        instance.model.eval()
        return instance
