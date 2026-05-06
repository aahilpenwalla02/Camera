from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import supervision as sv
from ultralytics import YOLO


@dataclass
class ModelBundle:
    det_model: YOLO
    pose_model: YOLO
    trash_model: Optional[YOLO] = None
    # Class names for the trash model (e.g. Glass, Metal, Paper, Plastic, Waste)
    trash_class_names: list = field(default_factory=list)


@dataclass
class DetectionResult:
    boxes_xyxy: np.ndarray           # [N, 4] all detections
    track_ids: np.ndarray            # [N] ByteTrack IDs
    class_ids: np.ndarray            # [N] COCO class IDs
    confidences: np.ndarray          # [N]
    keypoints_xy: np.ndarray         # [M, 17, 2] from pose model
    sv_detections: Optional[sv.Detections] = field(default=None)

    # Trash detections from dedicated trash model
    trash_boxes_xyxy: np.ndarray = field(default_factory=lambda: np.empty((0, 4)))
    trash_track_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    trash_class_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    trash_confidences: np.ndarray = field(default_factory=lambda: np.empty(0))
    trash_sv_detections: Optional[sv.Detections] = field(default=None)

    @property
    def person_mask(self) -> np.ndarray:
        return self.class_ids == 0

    @property
    def person_track_ids(self) -> np.ndarray:
        return self.track_ids[self.person_mask]

    @property
    def person_boxes(self) -> np.ndarray:
        return self.boxes_xyxy[self.person_mask]


def load_models(det_path: str, pose_path: str,
                trash_model_path: str = "") -> ModelBundle:
    trash_model = None
    trash_class_names = []

    if trash_model_path and __import__("os").path.exists(trash_model_path):
        trash_model = YOLO(trash_model_path)
        trash_class_names = list(trash_model.names.values())
        print(f"Trash model loaded: {trash_class_names}")

    return ModelBundle(
        det_model=YOLO(det_path),
        pose_model=YOLO(pose_path),
        trash_model=trash_model,
        trash_class_names=trash_class_names,
    )


class Detector:
    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle
        self._tracker = sv.ByteTrack()
        self._trash_tracker = sv.ByteTrack()

    def reset(self):
        self._tracker = sv.ByteTrack()
        self._trash_tracker = sv.ByteTrack()

    def run(self, frame: np.ndarray) -> DetectionResult:
        # --- Person / general object detection ---
        det_results = self.bundle.det_model.track(
            frame, persist=True, tracker="bytetrack.yaml", verbose=False
        )
        r = det_results[0]

        if r.boxes is None or len(r.boxes) == 0:
            empty = np.empty((0, 4))
            empty_ids = np.empty(0, dtype=int)
            base = DetectionResult(
                boxes_xyxy=empty,
                track_ids=empty_ids,
                class_ids=empty_ids.copy(),
                confidences=np.empty(0),
                keypoints_xy=np.empty((0, 17, 2)),
                sv_detections=sv.Detections.empty(),
            )
        else:
            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy().astype(int)
            confidences = r.boxes.conf.cpu().numpy()
            track_ids = (
                r.boxes.id.cpu().numpy().astype(int)
                if r.boxes.id is not None
                else np.full(len(boxes_xyxy), -1, dtype=int)
            )
            sv_det = sv.Detections.from_ultralytics(r)
            if sv_det.tracker_id is None:
                sv_det.tracker_id = track_ids

            pose_results = self.bundle.pose_model(frame, verbose=False)
            pr = pose_results[0]
            keypoints_xy = (
                pr.keypoints.xy.cpu().numpy()
                if pr.keypoints is not None and len(pr.keypoints) > 0
                else np.empty((0, 17, 2))
            )

            base = DetectionResult(
                boxes_xyxy=boxes_xyxy,
                track_ids=track_ids,
                class_ids=class_ids,
                confidences=confidences,
                keypoints_xy=keypoints_xy,
                sv_detections=sv_det,
            )

        # --- Dedicated trash model (if loaded) ---
        if self.bundle.trash_model is not None:
            tr = self.bundle.trash_model.track(
                frame, persist=True, tracker="bytetrack.yaml", verbose=False
            )[0]
            if tr.boxes is not None and len(tr.boxes) > 0:
                base.trash_boxes_xyxy = tr.boxes.xyxy.cpu().numpy()
                base.trash_class_ids = tr.boxes.cls.cpu().numpy().astype(int)
                base.trash_confidences = tr.boxes.conf.cpu().numpy()
                base.trash_track_ids = (
                    tr.boxes.id.cpu().numpy().astype(int)
                    if tr.boxes.id is not None
                    else np.full(len(base.trash_boxes_xyxy), -1, dtype=int)
                )
                base.trash_sv_detections = sv.Detections.from_ultralytics(tr)
                if base.trash_sv_detections.tracker_id is None:
                    base.trash_sv_detections.tracker_id = base.trash_track_ids

        return base
