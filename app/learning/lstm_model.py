"""
Pose-based LSTM anomaly detector.

Input:  sequence of pose keypoints [seq_len, 34]  (17 keypoints × 2 coords)
Output: probability that the sequence represents anomalous/suspicious behaviour

Binary classification: 0 = normal, 1 = anomalous
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

SEQ_LEN    = 30    # frames per sequence (≈1 second at 30 fps)
INPUT_SIZE = 34    # 17 keypoints × (x, y)
MODEL_PATH = "models/pose_lstm.pt"


class PoseLSTM(nn.Module):
    def __init__(self, input_size: int = INPUT_SIZE,
                 hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0.3
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_size]
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :]).squeeze(1)  # [batch]


def flatten_keypoints(kp: np.ndarray) -> np.ndarray:
    """[17, 2] → [34] normalised to [0, 1] range."""
    flat = kp.flatten().astype(np.float32)
    # Normalise by image dimensions assumed 1280×720; avoids large raw pixel values
    flat[0::2] /= 1280.0
    flat[1::2] /= 720.0
    return np.clip(flat, 0.0, 1.0)


class LSTMInference:
    """Wraps the trained model for per-track streaming inference."""

    def __init__(self, model_path: str = MODEL_PATH, threshold: float = 0.70):
        self.threshold = threshold
        self._model: Optional[PoseLSTM] = None
        self._model_path = model_path
        self._load()

    def _load(self):
        if os.path.exists(self._model_path):
            self._model = PoseLSTM()
            self._model.load_state_dict(
                torch.load(self._model_path, map_location="cpu")
            )
            self._model.eval()

    @property
    def available(self) -> bool:
        return self._model is not None

    def reload(self):
        self._load()

    def predict(self, sequence: np.ndarray) -> float:
        """
        sequence: [SEQ_LEN, 34] float32
        Returns anomaly probability 0–1.
        """
        if self._model is None or len(sequence) < SEQ_LEN:
            return 0.0
        x = torch.tensor(sequence[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return float(self._model(x).item())
