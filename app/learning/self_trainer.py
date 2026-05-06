"""
Self-learning pipeline: collects high-confidence detections from uploaded videos
as pseudo-labeled training data, then fine-tunes the detection model on them.
"""
import os
import random
import shutil
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import yaml


TRAIN_CONF_THRESHOLD = 0.70   # Only save detections above this confidence
MIN_SAMPLES_TO_TRAIN = 50     # Minimum new images before fine-tuning is offered
TRAIN_VAL_SPLIT = 0.85        # 85% train / 15% val


class SelfTrainer:
    def __init__(
        self,
        base_model_path: str,
        data_dir: str = "data/training",
        output_model_path: str = "models/fine_tuned.pt",
        class_names: Optional[List[str]] = None,
    ):
        self.base_model_path = base_model_path
        self.output_model_path = output_model_path
        self.data_dir = Path(data_dir)
        self.class_names = class_names or ["object"]

        self._images_train = self.data_dir / "images" / "train"
        self._images_val = self.data_dir / "images" / "val"
        self._labels_train = self.data_dir / "labels" / "train"
        self._labels_val = self.data_dir / "labels" / "val"

        for d in [self._images_train, self._images_val,
                  self._labels_train, self._labels_val]:
            d.mkdir(parents=True, exist_ok=True)

        self._session_count = 0   # samples collected this session

    @property
    def total_samples(self) -> int:
        return len(list(self._images_train.glob("*.jpg"))) + \
               len(list(self._images_val.glob("*.jpg")))

    @property
    def session_samples(self) -> int:
        return self._session_count

    @property
    def ready_to_train(self) -> bool:
        return self._session_count >= MIN_SAMPLES_TO_TRAIN

    def collect_sample(
        self,
        frame: np.ndarray,
        boxes_xyxy: np.ndarray,
        class_ids: np.ndarray,
        confidences: np.ndarray,
        frame_idx: int,
    ):
        """Save a frame and its high-confidence detections as a training sample."""
        if len(boxes_xyxy) == 0:
            return

        high_conf = confidences >= TRAIN_CONF_THRESHOLD
        if not high_conf.any():
            return

        h, w = frame.shape[:2]
        label_lines = []

        for box, cls_id, conf in zip(
            boxes_xyxy[high_conf], class_ids[high_conf], confidences[high_conf]
        ):
            x1, y1, x2, y2 = box
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            label_lines.append(f"{int(cls_id)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not label_lines:
            return

        is_val = random.random() > TRAIN_VAL_SPLIT
        img_dir = self._images_val if is_val else self._images_train
        lbl_dir = self._labels_val if is_val else self._labels_train

        stem = f"frame_{frame_idx:07d}_{self._session_count:05d}"
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), frame)
        (lbl_dir / f"{stem}.txt").write_text("\n".join(label_lines))

        self._session_count += 1

    def _write_dataset_yaml(self) -> str:
        yaml_path = str(self.data_dir / "dataset.yaml")
        data = {
            "path": str(self.data_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": len(self.class_names),
            "names": self.class_names,
        }
        with open(yaml_path, "w") as f:
            yaml.dump(data, f)
        return yaml_path

    def fine_tune(self, epochs: int = 5, imgsz: int = 640) -> str:
        """Fine-tune the model on collected data. Returns path to new weights."""
        from ultralytics import YOLO

        # Use fine-tuned weights as base if they exist (continual learning)
        base = self.output_model_path if os.path.exists(self.output_model_path) \
               else self.base_model_path

        model = YOLO(base)
        yaml_path = self._write_dataset_yaml()

        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=8,
            patience=3,           # early stop if no improvement
            save=True,
            project="data/runs",
            name="fine_tune",
            exist_ok=True,
            verbose=False,
        )

        best = Path("data/runs/fine_tune/weights/best.pt")
        if best.exists():
            shutil.copy(str(best), self.output_model_path)
            self._session_count = 0  # reset counter after training
            return self.output_model_path

        return ""

    def reset_session(self):
        self._session_count = 0
