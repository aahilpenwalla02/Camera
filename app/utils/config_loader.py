from dataclasses import dataclass, field
from enum import Enum
from typing import List
import yaml
import numpy as np


class ZoneType(str, Enum):
    RESTRICTED = "RESTRICTED"
    TRASH = "TRASH"


@dataclass
class ZoneConfig:
    name: str
    type: ZoneType
    polygon: np.ndarray  # shape [N, 2], int32


@dataclass
class AppConfig:
    det_model: str = "yolo11n.pt"
    pose_model: str = "yolo11n-pose.pt"
    trash_model: str = "models/trash_model.pt"
    db_path: str = "data/events.db"
    crowd_threshold: int = 10
    trash_stationary_seconds: float = 30.0
    conflict_iou_threshold: float = 0.3
    conflict_velocity_threshold: float = 15.0
    conflict_frames_required: int = 5
    zones: List[ZoneConfig] = field(default_factory=list)
    # Break-in detection
    breakin_loiter_seconds: float = 45.0
    breakin_crouch_threshold: float = 0.75
    breakin_after_hours_start: int = 23
    breakin_after_hours_end: int = 6
    # Self-learning
    self_learning_enabled: bool = True
    self_learning_conf_threshold: float = 0.70
    self_learning_min_samples: int = 50
    fine_tuned_model: str = "models/fine_tuned.pt"


def load_config(path: str = "config.yaml") -> AppConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    zones = []
    for z in raw.get("zones", []):
        polygon = np.array(z["polygon"], dtype=np.int32)
        if len(polygon) < 3:
            raise ValueError(f"Zone '{z['name']}' polygon must have at least 3 points")
        zones.append(ZoneConfig(
            name=z["name"],
            type=ZoneType(z["type"]),
            polygon=polygon,
        ))

    return AppConfig(
        det_model=raw.get("det_model", "yolo11n.pt"),
        pose_model=raw.get("pose_model", "yolo11n-pose.pt"),
        trash_model=raw.get("trash_model", "models/trash_model.pt"),
        db_path=raw.get("db_path", "data/events.db"),
        crowd_threshold=int(raw.get("crowd_threshold", 10)),
        trash_stationary_seconds=float(raw.get("trash_stationary_seconds", 30.0)),
        conflict_iou_threshold=float(raw.get("conflict_iou_threshold", 0.3)),
        conflict_velocity_threshold=float(raw.get("conflict_velocity_threshold", 15.0)),
        conflict_frames_required=int(raw.get("conflict_frames_required", 5)),
        zones=zones,
        breakin_loiter_seconds=float(raw.get("breakin_loiter_seconds", 45.0)),
        breakin_crouch_threshold=float(raw.get("breakin_crouch_threshold", 0.75)),
        breakin_after_hours_start=int(raw.get("breakin_after_hours_start", 23)),
        breakin_after_hours_end=int(raw.get("breakin_after_hours_end", 6)),
        self_learning_enabled=bool(raw.get("self_learning_enabled", True)),
        self_learning_conf_threshold=float(raw.get("self_learning_conf_threshold", 0.70)),
        self_learning_min_samples=int(raw.get("self_learning_min_samples", 50)),
        fine_tuned_model=raw.get("fine_tuned_model", "models/fine_tuned.pt"),
    )
