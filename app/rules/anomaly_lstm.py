"""
LSTM-based anomaly detection rule.

Maintains a rolling keypoint sequence buffer per tracked person.
Once SEQ_LEN frames are buffered, runs the LSTM and fires an alert
if the anomaly probability exceeds the threshold.
"""

from collections import defaultdict, deque
from typing import Dict, List

import numpy as np

from app.detection.detector import DetectionResult
from app.learning.lstm_model import LSTMInference, SEQ_LEN, flatten_keypoints
from app.storage.database import AlertEvent

COOLDOWN_FRAMES = 90


class LSTMAnomalyRule:
    def __init__(self, lstm: LSTMInference):
        self._lstm = lstm
        # track_id → deque of flattened keypoint vectors [34]
        self._buffers: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=SEQ_LEN)
        )
        self._last_fired: Dict[int, int] = {}

    def reset(self):
        self._buffers.clear()
        self._last_fired.clear()

    @property
    def active(self) -> bool:
        return self._lstm.available

    def evaluate(self, result: DetectionResult, frame_idx: int) -> List[AlertEvent]:
        if not self._lstm.available:
            return []

        person_mask = result.class_ids == 0
        if not person_mask.any():
            return []

        track_ids = result.track_ids[person_mask]
        kps       = result.keypoints_xy  # [M, 17, 2]
        confs     = result.confidences[person_mask]

        alerts = []

        for i, tid in enumerate(track_ids):
            if tid < 0 or float(confs[i]) < 0.5:
                continue

            tid = int(tid)
            kp  = kps[i] if i < len(kps) else np.zeros((17, 2))
            self._buffers[tid].append(flatten_keypoints(kp))

            if len(self._buffers[tid]) < SEQ_LEN:
                continue

            last = self._last_fired.get(tid, -COOLDOWN_FRAMES)
            if frame_idx - last < COOLDOWN_FRAMES:
                continue

            sequence = np.stack(list(self._buffers[tid]))  # [SEQ_LEN, 34]
            prob     = self._lstm.predict(sequence)

            if prob >= self._lstm.threshold:
                self._last_fired[tid] = frame_idx
                alerts.append(AlertEvent(
                    event_type="LSTM_ANOMALY",
                    confidence_score=round(prob, 3),
                    metadata={
                        "track_id": tid,
                        "anomaly_probability": round(prob, 3),
                    },
                    frame_number=frame_idx,
                ))

        return alerts
