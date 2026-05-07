"""
Trash overflow detection — no manual polygon required.

Two modes (auto-selected):
  Bin-calibrated: user marks the bin location in the sidebar once.
    Counts trash items detected inside/near that bounding box.
  Density fallback: no bin marked.
    Counts trash items anywhere in the frame; fires when 3+ cluster together.

Fires TRASH_OVERFLOW after OVERFLOW_COUNT items are detected for
SUSTAIN_FRAMES consecutive frames, with a COOLDOWN_FRAMES cooldown.
"""

from typing import List, Optional, Set, Tuple

import numpy as np

from app.detection.detector import DetectionResult
from app.storage.database import AlertEvent

COCO_TRASH_PROXY_IDS: Set[int] = {39, 41}  # bottle, cup only — backpack/suitcase removed (false positives)

PERSON_OVERLAP_THRESHOLD = 0.4  # suppress trash if >40% of its area overlaps a person


def _iou_with_person(box: np.ndarray, person_box: np.ndarray) -> float:
    """Return what fraction of box's area overlaps with person_box."""
    ix1 = max(box[0], person_box[0])
    iy1 = max(box[1], person_box[1])
    ix2 = min(box[2], person_box[2])
    iy2 = min(box[3], person_box[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    box_area = max((box[2] - box[0]) * (box[3] - box[1]), 1)
    return inter / box_area


def _overlaps_person(box: np.ndarray, person_boxes: np.ndarray) -> bool:
    return any(_iou_with_person(box, p) >= PERSON_OVERLAP_THRESHOLD for p in person_boxes)

OVERFLOW_COUNT  = 3    # minimum trash items to consider overflow
SUSTAIN_FRAMES  = 5    # consecutive frames before alert fires
COOLDOWN_FRAMES = 150  # frames between alerts (~5 s at 30 fps)
MIN_CONF        = 0.50
BIN_EXPAND_PX   = 120  # pixels to expand bin bbox when searching for nearby trash
CLUSTER_RADIUS  = 300  # pixels — items within this radius are "clustered"


class TrashOverflowRule:
    def __init__(self, has_trash_model: bool = False,
                 bin_zone: Optional[Tuple[int, int, int, int]] = None):
        """
        bin_zone: (x1, y1, x2, y2) pixel coords of the bin bounding box.
                  Pass None to use full-frame density/cluster detection.
        """
        self._has_trash_model = has_trash_model
        self._bin_zone = bin_zone
        self._sustain = 0
        self._last_fired = -COOLDOWN_FRAMES

    def update_bin_zone(self, bin_zone: Optional[Tuple[int, int, int, int]]):
        self._bin_zone = bin_zone

    def reset(self):
        self._sustain = 0
        self._last_fired = -COOLDOWN_FRAMES

    # ------------------------------------------------------------------

    def evaluate(self, result: DetectionResult, frame_idx: int) -> List[AlertEvent]:
        # Choose detection source
        if self._has_trash_model and len(result.trash_boxes_xyxy) > 0:
            boxes = result.trash_boxes_xyxy
            confs = result.trash_confidences
            source = "trash_model"
        else:
            proxy_mask = np.isin(result.class_ids, list(COCO_TRASH_PROXY_IDS))
            boxes = result.boxes_xyxy[proxy_mask]
            confs = result.confidences[proxy_mask]
            source = "coco_proxy"

        if len(boxes) == 0:
            self._sustain = 0
            return []

        conf_mask = confs >= MIN_CONF
        boxes = boxes[conf_mask]
        if len(boxes) == 0:
            self._sustain = 0
            return []

        # Suppress detections that overlap with a person (e.g. backpack on someone)
        person_boxes = result.person_boxes
        if len(person_boxes) > 0:
            boxes = np.array([b for b in boxes if not _overlaps_person(b, person_boxes)])
        if len(boxes) == 0:
            self._sustain = 0
            return []

        count = (
            self._count_near_bin(boxes)
            if self._bin_zone is not None
            else self._count_clustered(boxes)
        )

        if count >= OVERFLOW_COUNT:
            self._sustain += 1
        else:
            self._sustain = 0

        if self._sustain >= SUSTAIN_FRAMES:
            if frame_idx - self._last_fired >= COOLDOWN_FRAMES:
                self._last_fired = frame_idx
                self._sustain = 0
                return [AlertEvent(
                    event_type="TRASH_OVERFLOW",
                    confidence_score=round(min(1.0, count / (OVERFLOW_COUNT + 2)), 3),
                    metadata={
                        "item_count": count,
                        "model": source,
                        "bin_calibrated": self._bin_zone is not None,
                    },
                    frame_number=frame_idx,
                )]

        return []

    # ------------------------------------------------------------------

    def _count_near_bin(self, boxes: np.ndarray) -> int:
        x1b, y1b, x2b, y2b = self._bin_zone
        x1e = x1b - BIN_EXPAND_PX
        y1e = y1b - BIN_EXPAND_PX
        x2e = x2b + BIN_EXPAND_PX
        y2e = y2b + BIN_EXPAND_PX
        count = 0
        for box in boxes:
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            if x1e <= cx <= x2e and y1e <= cy <= y2e:
                count += 1
        return count

    def _count_clustered(self, boxes: np.ndarray) -> int:
        """Return the size of the largest cluster of trash items."""
        if len(boxes) == 0:
            return 0
        centroids = np.array([((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes])
        max_cluster = 1
        for c in centroids:
            dists = np.linalg.norm(centroids - c, axis=1)
            max_cluster = max(max_cluster, int((dists <= CLUSTER_RADIUS).sum()))
        return max_cluster
