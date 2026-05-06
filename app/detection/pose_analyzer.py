from collections import defaultdict, deque
from typing import Dict, List, Tuple
import numpy as np

from app.utils.config_loader import AppConfig

# COCO wrist keypoint indices
LEFT_WRIST = 9
RIGHT_WRIST = 10


def _box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IOU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class PoseAnalyzer:
    def __init__(self, config: AppConfig):
        self._config = config
        # track_id → deque of keypoint arrays [17, 2]
        self._history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=10))
        # frozenset({id_a, id_b}) → consecutive conflict frame count
        self._conflict_counters: Dict[frozenset, int] = defaultdict(int)

    def reset(self):
        self._history.clear()
        self._conflict_counters.clear()

    def update(self, track_ids: np.ndarray, keypoints_xy: np.ndarray):
        """
        Associate pose keypoints with tracked person IDs by spatial proximity
        (nearest bounding-box centroid heuristic is not available here, so we
        match by detection order — works well when counts are stable).
        """
        n = min(len(track_ids), len(keypoints_xy))
        for i in range(n):
            tid = int(track_ids[i])
            if tid >= 0:
                self._history[tid].append(keypoints_xy[i])

    def _wrist_velocity(self, track_id: int) -> float:
        history = self._history[track_id]
        if len(history) < 2:
            return 0.0
        velocities = []
        frames = list(history)[-4:]  # last 3 diffs from last 4 frames
        for i in range(1, len(frames)):
            for wrist in (LEFT_WRIST, RIGHT_WRIST):
                prev = frames[i - 1][wrist]
                curr = frames[i][wrist]
                if prev[0] > 0 and curr[0] > 0:  # ignore invisible keypoints
                    velocities.append(np.linalg.norm(curr - prev))
        return float(np.mean(velocities)) if velocities else 0.0

    def check_conflict(
        self,
        track_ids: np.ndarray,
        boxes_xyxy: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """
        Returns list of (id_a, id_b) pairs confirmed as conflicting
        (sustained over conflict_frames_required consecutive frames).
        """
        person_ids = [int(tid) for tid in track_ids if tid >= 0]
        n = len(person_ids)
        confirmed = []

        active_pairs = set()
        for i in range(n):
            for j in range(i + 1, n):
                id_a, id_b = person_ids[i], person_ids[j]
                if i >= len(boxes_xyxy) or j >= len(boxes_xyxy):
                    continue

                iou = _box_iou(boxes_xyxy[i], boxes_xyxy[j])
                vel_a = self._wrist_velocity(id_a)
                vel_b = self._wrist_velocity(id_b)

                pair = frozenset({id_a, id_b})
                active_pairs.add(pair)

                if (iou >= self._config.conflict_iou_threshold and
                        vel_a >= self._config.conflict_velocity_threshold and
                        vel_b >= self._config.conflict_velocity_threshold):
                    self._conflict_counters[pair] += 1
                else:
                    self._conflict_counters[pair] = 0

                if self._conflict_counters[pair] >= self._config.conflict_frames_required:
                    confirmed.append((id_a, id_b))
                    self._conflict_counters[pair] = 0  # reset after firing

        # Decay counters for pairs no longer visible
        for pair in list(self._conflict_counters.keys()):
            if pair not in active_pairs:
                del self._conflict_counters[pair]

        return confirmed
