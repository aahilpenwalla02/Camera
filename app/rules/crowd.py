from typing import List
import numpy as np

from app.detection.detector import DetectionResult
from app.storage.database import AlertEvent
from app.utils.config_loader import AppConfig


class CrowdRule:
    def __init__(self, config: AppConfig):
        self._threshold = config.crowd_threshold

    def evaluate(self, result: DetectionResult) -> List[AlertEvent]:
        person_ids = result.person_track_ids
        unique_count = len(set(int(tid) for tid in person_ids if tid >= 0))

        if unique_count > self._threshold:
            return [AlertEvent(
                event_type="CROWD_DETECTED",
                confidence_score=1.0,
                metadata={"person_count": unique_count, "threshold": self._threshold},
            )]
        return []
