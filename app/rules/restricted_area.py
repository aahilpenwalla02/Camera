from typing import List
import numpy as np
import supervision as sv

from app.detection.detector import DetectionResult
from app.detection.zone_manager import ZoneManager
from app.storage.database import AlertEvent
from app.utils.config_loader import ZoneType


class RestrictedAreaRule:
    def __init__(self, zone_manager: ZoneManager):
        self._zone_manager = zone_manager
        self._zone_names = zone_manager.zones_of_type(ZoneType.RESTRICTED)

    def evaluate(self, result: DetectionResult) -> List[AlertEvent]:
        if result.sv_detections is None or len(result.sv_detections) == 0:
            return []

        # Work with person-only detections
        person_mask = result.class_ids == 0
        if not person_mask.any():
            return []

        person_det = result.sv_detections[person_mask]
        alerts = []

        for zone_name in self._zone_names:
            in_zone = self._zone_manager.check_zone(zone_name, person_det)
            count_in = int(in_zone.sum())
            if count_in > 0:
                track_ids_in = (
                    person_det.tracker_id[in_zone].tolist()
                    if person_det.tracker_id is not None
                    else []
                )
                alerts.append(AlertEvent(
                    event_type="RESTRICTED_AREA_BREACH",
                    confidence_score=float(person_det.confidence[in_zone].mean())
                    if person_det.confidence is not None else 1.0,
                    metadata={"zone": zone_name, "count": count_in, "track_ids": track_ids_in},
                ))

        return alerts
