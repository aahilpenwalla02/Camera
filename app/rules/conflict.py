from typing import List

from app.detection.detector import DetectionResult
from app.detection.pose_analyzer import PoseAnalyzer
from app.storage.database import AlertEvent


class ConflictRule:
    def __init__(self, pose_analyzer: PoseAnalyzer):
        self._analyzer = pose_analyzer

    def evaluate(self, result: DetectionResult) -> List[AlertEvent]:
        person_mask = result.class_ids == 0
        person_track_ids = result.track_ids[person_mask]
        person_boxes = result.boxes_xyxy[person_mask]

        # Update pose history with person keypoints
        self._analyzer.update(person_track_ids, result.keypoints_xy)

        # Check for confirmed conflict pairs
        confirmed = self._analyzer.check_conflict(person_track_ids, person_boxes)

        return [
            AlertEvent(
                event_type="CONFLICT_DETECTED",
                confidence_score=0.85,
                metadata={"persons": list(pair)},
            )
            for pair in confirmed
        ]
