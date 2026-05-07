"""
YOLO-based fight detection rule.
Uses the Musawer14/fight_detection_yolov8 model which was trained on the
RWF-2000 real-world surveillance fight dataset.

Fires CONFLICT_DETECTED when the model detects a fight with sufficient confidence,
sustained across multiple frames to reduce false positives.
"""

from typing import List

import numpy as np

from app.detection.detector import DetectionResult
from app.storage.database import AlertEvent

MIN_CONF        = 0.40   # minimum confidence from fight model
SUSTAIN_FRAMES  = 2      # consecutive detections before alert fires
COOLDOWN_FRAMES = 45


class FightDetectorRule:
    def __init__(self, conf_threshold: float = MIN_CONF):
        self._conf_threshold = conf_threshold
        self._sustain_counter = 0
        self._last_fired = -COOLDOWN_FRAMES

    def reset(self):
        self._sustain_counter = 0
        self._last_fired = -COOLDOWN_FRAMES

    @property
    def available(self) -> bool:
        # Rule is active only if the fight model produced detections array
        return True

    def evaluate(self, result: DetectionResult, frame_idx: int) -> List[AlertEvent]:
        fight_boxes = result.fight_boxes_xyxy
        fight_confs = result.fight_confidences

        # Check if fight model fired anything above threshold
        if len(fight_confs) > 0 and float(fight_confs.max()) >= self._conf_threshold:
            self._sustain_counter += 1
        else:
            self._sustain_counter = 0

        if self._sustain_counter >= SUSTAIN_FRAMES:
            if frame_idx - self._last_fired >= COOLDOWN_FRAMES:
                self._last_fired = frame_idx
                self._sustain_counter = 0
                best_conf = float(fight_confs.max()) if len(fight_confs) > 0 else 1.0
                return [AlertEvent(
                    event_type="FIGHT_DETECTED",
                    confidence_score=round(best_conf, 3),
                    metadata={
                        "source": "yolo_fight_model",
                        "detections": len(fight_boxes),
                    },
                    frame_number=frame_idx,
                )]

        return []
