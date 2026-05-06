import time
from typing import Dict, List, Set

import numpy as np
import supervision as sv

from app.detection.detector import DetectionResult
from app.detection.zone_manager import ZoneManager
from app.storage.database import AlertEvent
from app.utils.config_loader import AppConfig, ZoneType

# COCO fallback proxy class IDs (used only when no dedicated trash model is loaded)
COCO_TRASH_PROXY_IDS: Set[int] = {39, 41, 26, 28}  # bottle, cup, backpack, suitcase


class TrashOverflowRule:
    def __init__(self, zone_manager: ZoneManager, config: AppConfig,
                 has_trash_model: bool = False):
        self._zone_manager = zone_manager
        self._zone_names = zone_manager.zones_of_type(ZoneType.TRASH)
        self._stationary_seconds = config.trash_stationary_seconds
        self._has_trash_model = has_trash_model
        self._first_seen: Dict[int, float] = {}

    def reset(self):
        self._first_seen.clear()

    def evaluate(self, result: DetectionResult) -> List[AlertEvent]:
        if self._has_trash_model:
            return self._evaluate_with_trash_model(result)
        return self._evaluate_with_coco_proxy(result)

    def _evaluate_with_trash_model(self, result: DetectionResult) -> List[AlertEvent]:
        if result.trash_sv_detections is None or len(result.trash_sv_detections) == 0:
            return []
        return self._check_zone_and_time(
            result.trash_sv_detections,
            result.trash_confidences,
            result.trash_track_ids,
        )

    def _evaluate_with_coco_proxy(self, result: DetectionResult) -> List[AlertEvent]:
        if result.sv_detections is None or len(result.sv_detections) == 0:
            return []
        trash_mask = np.isin(result.class_ids, list(COCO_TRASH_PROXY_IDS))
        if not trash_mask.any():
            return []
        return self._check_zone_and_time(
            result.sv_detections[trash_mask],
            result.confidences[trash_mask],
            result.track_ids[trash_mask],
        )

    def _check_zone_and_time(
        self,
        detections: sv.Detections,
        confidences: np.ndarray,
        track_ids: np.ndarray,
    ) -> List[AlertEvent]:
        alerts = []
        now = time.time()

        for zone_name in self._zone_names:
            in_zone = self._zone_manager.check_zone(zone_name, detections)
            if not in_zone.any():
                continue

            current_ids = set(
                int(tid) for tid in track_ids[in_zone] if tid >= 0
            )

            for tid in current_ids:
                if tid not in self._first_seen:
                    self._first_seen[tid] = now

            gone = [tid for tid in list(self._first_seen) if tid not in current_ids]
            for tid in gone:
                self._first_seen.pop(tid, None)

            for tid in current_ids:
                elapsed = now - self._first_seen.get(tid, now)
                if elapsed >= self._stationary_seconds:
                    alerts.append(AlertEvent(
                        event_type="TRASH_OVERFLOW",
                        confidence_score=1.0,
                        metadata={
                            "zone": zone_name,
                            "track_id": tid,
                            "stationary_seconds": round(elapsed, 1),
                            "model": "trash_model" if self._has_trash_model else "coco_proxy",
                        },
                    ))
                    self._first_seen[tid] = now  # reset so it doesn't re-fire every frame

        return alerts
