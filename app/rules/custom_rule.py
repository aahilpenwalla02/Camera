"""
Generic configurable rule — detects any COCO object class optionally
constrained to a zone, firing an alert after a sustained presence threshold.

Rule config dict keys:
  alert_name        str   — event_type string e.g. "ILLEGAL_PARKING"
  objects           list  — COCO object names e.g. ["car", "truck"]
  duration_seconds  float — seconds object must be present (0 = instant)
  zone_pct          list  — [x1%, y1%, x2%, y2%] (optional; None = full frame)
  description       str   — human-readable description
"""

import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from app.detection.detector import DetectionResult
from app.storage.database import AlertEvent

COCO_NAME_TO_ID: Dict[str, int] = {
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "airplane": 4,
    "bus": 5, "train": 6, "truck": 7, "boat": 8, "traffic light": 9,
    "fire hydrant": 10, "stop sign": 11, "parking meter": 12, "bench": 13,
    "bird": 14, "cat": 15, "dog": 16, "horse": 17, "sheep": 18, "cow": 19,
    "elephant": 20, "bear": 21, "umbrella": 25, "backpack": 24,
    "suitcase": 28, "bottle": 39, "cup": 41, "chair": 56,
    "couch": 57, "potted plant": 58, "cell phone": 67,
}

NOTIFICATION_COOLDOWN = 300  # seconds between repeat alerts for same track_id


class CustomRule:
    def __init__(self, rule_config: dict):
        self._name        = rule_config["alert_name"]
        self._description = rule_config.get("description", "")
        self._duration    = float(rule_config.get("duration_seconds", 30))
        self._zone_pct: Optional[Tuple] = (
            tuple(rule_config["zone_pct"]) if rule_config.get("zone_pct") else None
        )
        self._class_ids: Set[int] = {
            COCO_NAME_TO_ID[obj]
            for obj in rule_config.get("objects", [])
            if obj in COCO_NAME_TO_ID
        }
        self._first_seen: Dict[int, float] = {}
        self._last_fired: Dict[int, float] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def reset(self):
        self._first_seen.clear()
        self._last_fired.clear()

    def evaluate(self, result: DetectionResult,
                 frame_w: int, frame_h: int,
                 frame_idx: int) -> List[AlertEvent]:
        if not self._class_ids:
            return []

        mask = np.isin(result.class_ids, list(self._class_ids))
        if not mask.any():
            self._prune_gone(set())
            return []

        boxes     = result.boxes_xyxy[mask]
        track_ids = result.track_ids[mask]
        confs     = result.confidences[mask]

        # Zone filter
        if self._zone_pct is not None:
            x1z = self._zone_pct[0] / 100 * frame_w
            y1z = self._zone_pct[1] / 100 * frame_h
            x2z = self._zone_pct[2] / 100 * frame_w
            y2z = self._zone_pct[3] / 100 * frame_h
            keep = np.array([
                x1z <= (b[0] + b[2]) / 2 <= x2z and y1z <= (b[1] + b[3]) / 2 <= y2z
                for b in boxes
            ])
            boxes     = boxes[keep]
            track_ids = track_ids[keep]
            confs     = confs[keep]

        now         = time.time()
        current_ids = {int(tid) for tid in track_ids if tid >= 0}
        self._prune_gone(current_ids)

        for tid in current_ids:
            if tid not in self._first_seen:
                self._first_seen[tid] = now

        alerts = []
        for i, tid in enumerate(track_ids):
            tid = int(tid)
            if tid < 0:
                continue
            elapsed = now - self._first_seen.get(tid, now)
            last    = self._last_fired.get(tid, 0)
            if elapsed >= self._duration and (now - last) >= NOTIFICATION_COOLDOWN:
                self._last_fired[tid] = now
                alerts.append(AlertEvent(
                    event_type=self._name,
                    confidence_score=round(float(confs[i]), 3),
                    metadata={
                        "track_id": tid,
                        "duration_seconds": round(elapsed, 1),
                        "rule": self._name,
                        "description": self._description,
                    },
                    frame_number=frame_idx,
                ))

        return alerts

    def _prune_gone(self, current_ids: set):
        for tid in list(self._first_seen):
            if tid not in current_ids:
                del self._first_seen[tid]
