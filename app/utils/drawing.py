from typing import List, Optional
import cv2
import numpy as np

# COCO skeleton connections: pairs of keypoint indices
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # head
    (5, 6),                                  # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),        # arms
    (5, 11), (6, 12), (11, 12),             # torso
    (11, 13), (13, 15), (12, 14), (14, 16), # legs
]


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    cv2.putText(frame, f"FPS: {fps:.1f}", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


def draw_person_count(frame: np.ndarray, count: int) -> np.ndarray:
    cv2.putText(frame, f"Persons: {count}", (12, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    return frame


def draw_alert_banner(frame: np.ndarray, alert_types: List[str]) -> np.ndarray:
    if not alert_types:
        return frame
    h, w = frame.shape[:2]
    label = "ALERT: " + " | ".join(set(alert_types))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, label, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def draw_boxes(frame: np.ndarray, boxes_xyxy: np.ndarray,
               track_ids: Optional[np.ndarray] = None,
               color: tuple = (0, 220, 0)) -> np.ndarray:
    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if track_ids is not None:
            tid = int(track_ids[i]) if track_ids[i] is not None else -1
            cv2.putText(frame, f"ID:{tid}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return frame


def draw_skeleton(frame: np.ndarray, keypoints_xy: np.ndarray) -> np.ndarray:
    """keypoints_xy: shape [N_persons, 17, 2]"""
    for person_kps in keypoints_xy:
        for idx, (x, y) in enumerate(person_kps):
            if x > 0 and y > 0:
                cv2.circle(frame, (int(x), int(y)), 3, (255, 128, 0), -1)
        for a, b in COCO_SKELETON:
            xa, ya = person_kps[a]
            xb, yb = person_kps[b]
            if xa > 0 and ya > 0 and xb > 0 and yb > 0:
                cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)),
                         (255, 128, 0), 1, cv2.LINE_AA)
    return frame


def draw_zones(frame: np.ndarray, zones: dict) -> np.ndarray:
    """zones: dict[name, ZoneConfig]"""
    for zone_cfg in zones.values():
        color = (0, 0, 200) if zone_cfg.type.value == "RESTRICTED" else (0, 165, 255)
        pts = zone_cfg.polygon.reshape((-1, 1, 2))
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        cv2.polylines(frame, [pts], True, color, 2)
        cx, cy = zone_cfg.polygon.mean(axis=0).astype(int)
        cv2.putText(frame, zone_cfg.name, (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame
