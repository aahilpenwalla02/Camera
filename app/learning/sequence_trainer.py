"""
Extracts pose keypoint sequences from labelled videos and trains the PoseLSTM.

Workflow:
  1. User uploads a video + picks label (Normal or Anomalous)
  2. extract_sequences() runs YOLO pose on the video, slices 30-frame windows
  3. Sequences are saved to data/sequences/{normal,anomalous}/
  4. train() loads all saved sequences and fits the LSTM
"""

import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from app.learning.lstm_model import (
    INPUT_SIZE,
    MODEL_PATH,
    PoseLSTM,
    SEQ_LEN,
    flatten_keypoints,
)

NORMAL_DIR    = Path("data/sequences/normal")
ANOMALOUS_DIR = Path("data/sequences/anomalous")
STRIDE        = 5   # slide window every 5 frames


def _ensure_dirs():
    NORMAL_DIR.mkdir(parents=True, exist_ok=True)
    ANOMALOUS_DIR.mkdir(parents=True, exist_ok=True)


def count_sequences() -> Tuple[int, int]:
    _ensure_dirs()
    return (
        len(list(NORMAL_DIR.glob("*.npy"))),
        len(list(ANOMALOUS_DIR.glob("*.npy"))),
    )


def extract_sequences(video_path: str, label: str, pose_model) -> int:
    """
    Run YOLO pose on video_path, extract sliding-window sequences.
    label: "normal" or "anomalous"
    Returns number of sequences saved.
    """
    import cv2
    _ensure_dirs()
    out_dir = NORMAL_DIR if label == "normal" else ANOMALOUS_DIR

    cap = cv2.VideoCapture(video_path)
    frame_keypoints: List[np.ndarray] = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = pose_model(frame, verbose=False)
        pr = results[0]
        if pr.keypoints is not None and len(pr.keypoints) > 0:
            # Take the first (most confident) person's keypoints
            kp = pr.keypoints.xy.cpu().numpy()[0]  # [17, 2]
        else:
            kp = np.zeros((17, 2))
        frame_keypoints.append(flatten_keypoints(kp))

    cap.release()

    if len(frame_keypoints) < SEQ_LEN:
        return 0

    saved = 0
    prefix = f"{label}_{Path(video_path).stem}"
    for start in range(0, len(frame_keypoints) - SEQ_LEN + 1, STRIDE):
        seq = np.stack(frame_keypoints[start:start + SEQ_LEN])  # [SEQ_LEN, 34]
        fname = out_dir / f"{prefix}_{start:06d}.npy"
        np.save(str(fname), seq)
        saved += 1

    return saved


class SequenceDataset(Dataset):
    def __init__(self):
        _ensure_dirs()
        self.samples: List[Tuple[Path, int]] = []
        for p in NORMAL_DIR.glob("*.npy"):
            self.samples.append((p, 0))
        for p in ANOMALOUS_DIR.glob("*.npy"):
            self.samples.append((p, 1))
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        seq = np.load(str(path)).astype(np.float32)
        return torch.tensor(seq), torch.tensor(float(label))


def train(epochs: int = 10, progress_callback=None) -> str:
    """
    Train PoseLSTM on all saved sequences.
    progress_callback(epoch, total, loss, val_acc) — optional UI hook.
    Returns path to saved model weights.
    """
    dataset = SequenceDataset()
    if len(dataset) < 10:
        raise ValueError(
            f"Need at least 10 sequences to train (have {len(dataset)}). "
            "Upload more videos."
        )

    val_size  = max(1, int(len(dataset) * 0.15))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=16)

    model     = PoseLSTM(INPUT_SIZE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_val_acc = 0.0
    os.makedirs("models", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for seqs, labels in train_loader:
            optimizer.zero_grad()
            preds = model(seqs)
            loss  = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for seqs, labels in val_loader:
                preds = model(seqs)
                predicted = (preds > 0.5).float()
                correct  += (predicted == labels).sum().item()
                total    += labels.size(0)

        val_acc  = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(train_loader)
        scheduler.step()

        if progress_callback:
            progress_callback(epoch, epochs, avg_loss, val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

    return MODEL_PATH
