"""
Break-in detection rules. No polygon zone required — operates on the full frame.

Rules:
  1. Loitering   — person visible for longer than threshold seconds
  2. Crouching   — hip keypoints low, sustained for 20+ consecutive frames
  3. Hands raised — both wrists above shoulders, sustained for 15+ consecutive frames
  4. After-hours — any person detected outside allowed hours

False-positive guards:
  - Rules 2–3 only fire when the person is ALONE (not a fight/crowd)
  - Rules 2 & 3 require sustained pose across multiple frames before firing
  - Minimum detection confidence of 0.5 required
"""

import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

from app.detection.detector import DetectionResult
from app.storage.database import AlertEvent
from app.utils.config_loader import AppConfig

# COCO keypoint indices
L_SHOULDER = 5
R_SHOULDER = 6
L_HIP      = 11
R_HIP      = 12
L_ANKLE    = 15
R_ANKLE    = 16
L_WRIST    = 9
R_WRIST    = 10

MIN_CONF       = 0.50   # ignore low-confidence detections entirely
CROUCH_FRAMES  = 20     # consecutive frames crouching before alert
HANDS_FRAMES   = 15     # consecutive frames both hands up before alert


def _kp(keypoints: np.ndarray, idx: int) -> Tuple[float, float]:
    if idx < len(keypoints):
        return float(keypoints[idx][0]), float(keypoints[idx][1])
    return 0.0, 0.0


def _visible(x: float, y: float) -> bool:
    return x > 1.0 and y > 1.0



class BreakInRule:
    def __init__(self, config: AppConfig):
        self._config = config

        # Rule 1 — loitering
        self._first_seen: Dict[int, float] = {}

        # Rules 2 & 3 — sustained pose counters: track_id → consecutive frame count
        self._crouch_counter: Dict[int, int] = defaultdict(int)
        self._hands_counter:  Dict[int, int] = defaultdict(int)

        # Cooldown tracker: rule_name → {track_id → last_frame_fired}
        self._last_fired: Dict[str, Dict[int, int]] = defaultdict(dict)

    def reset(self):
        self._first_seen.clear()
        self._crouch_counter.clear()
        self._hands_counter.clear()
        self._last_fired.clear()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def evaluate(self, result: DetectionResult, frame_idx: int) -> List[AlertEvent]:
        alerts = []

        person_mask = result.class_ids == 0
        if not person_mask.any():
            return alerts

        track_ids = result.track_ids[person_mask]
        boxes     = result.boxes_xyxy[person_mask]
        confs     = result.confidences[person_mask]
        kps       = result.keypoints_xy  # [M, 17, 2]

        # Rule 4 — after-hours (frame-level, not person-level)
        alerts += self._check_after_hours(frame_idx)

        # Count persons with sufficient confidence
        valid_person_count = int((confs >= MIN_CONF).sum())

        # If 2+ people are visible, all behavioral rules are suppressed.
        # Break-ins are solo acts. Multiple people = fight, crowd, or normal activity.
        multiple_people = valid_person_count >= 2

        active_ids = set()
        for i, tid in enumerate(track_ids):
            if tid < 0 or i >= len(boxes):
                continue

            tid  = int(tid)
            conf = float(confs[i])
            active_ids.add(tid)

            if conf < MIN_CONF or multiple_people:
                self._crouch_counter[tid] = 0
                self._hands_counter[tid]  = 0
                continue

            kp = kps[i] if i < len(kps) else np.zeros((17, 2))

            # Rule 1 — loitering
            alert = self._check_loitering(tid, conf, frame_idx)
            if alert:
                alerts.append(alert)

            # Rule 2 — crouching (sustained)
            alert = self._check_crouching(tid, kp, conf, frame_idx)
            if alert:
                alerts.append(alert)

            # Rule 3 — hands raised (sustained)
            alert = self._check_hands_raised(tid, kp, conf, frame_idx)
            if alert:
                alerts.append(alert)

        # Clean up state for persons no longer visible
        for tid in list(self._first_seen):
            if tid not in active_ids:
                del self._first_seen[tid]
        for tid in list(self._crouch_counter):
            if tid not in active_ids:
                del self._crouch_counter[tid]
        for tid in list(self._hands_counter):
            if tid not in active_ids:
                del self._hands_counter[tid]

        return alerts

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cooldown_ok(self, rule: str, tid: int, frame_idx: int,
                     cooldown: int = 150) -> bool:
        last = self._last_fired[rule].get(tid, -cooldown)
        if frame_idx - last >= cooldown:
            self._last_fired[rule][tid] = frame_idx
            return True
        return False

    # ------------------------------------------------------------------
    # Individual rules
    # ------------------------------------------------------------------

    def _check_loitering(self, tid: int, conf: float,
                         frame_idx: int) -> AlertEvent | None:
        now = time.time()
        if tid not in self._first_seen:
            self._first_seen[tid] = now
            return None
        elapsed = now - self._first_seen[tid]
        if elapsed >= self._config.breakin_loiter_seconds:
            if self._cooldown_ok("loiter", tid, frame_idx, cooldown=270):
                return AlertEvent(
                    event_type="BREAKIN_LOITERING",
                    confidence_score=round(conf, 3),
                    metadata={"track_id": tid, "loiter_seconds": round(elapsed, 1)},
                )
        return None

    def _check_crouching(self, tid: int, kp: np.ndarray, conf: float,
                         frame_idx: int) -> AlertEvent | None:
        lhx, lhy = _kp(kp, L_HIP);     rhx, rhy = _kp(kp, R_HIP)
        lsx, lsy = _kp(kp, L_SHOULDER); rsx, rsy = _kp(kp, R_SHOULDER)
        lax, lay = _kp(kp, L_ANKLE);   rax, ray = _kp(kp, R_ANKLE)

        hip_y = (lhy + rhy) / 2 if _visible(lhx, lhy) and _visible(rhx, rhy) else 0.0
        sho_y = (lsy + rsy) / 2 if _visible(lsx, lsy) and _visible(rsx, rsy) else 0.0
        ank_y = (lay + ray) / 2 if _visible(lax, lay) and _visible(rax, ray) else 0.0

        if not (hip_y > 0 and sho_y > 0 and ank_y > 0):
            self._crouch_counter[tid] = 0
            return None

        body_height = ank_y - sho_y
        if body_height < 20:
            self._crouch_counter[tid] = 0
            return None

        ratio = (hip_y - sho_y) / body_height
        if ratio >= self._config.breakin_crouch_threshold:
            self._crouch_counter[tid] += 1
        else:
            self._crouch_counter[tid] = 0

        if self._crouch_counter[tid] >= CROUCH_FRAMES:
            if self._cooldown_ok("crouch", tid, frame_idx):
                self._crouch_counter[tid] = 0
                return AlertEvent(
                    event_type="BREAKIN_CROUCHING",
                    confidence_score=round(conf, 3),
                    metadata={"track_id": tid, "crouch_ratio": round(ratio, 3)},
                )
        return None

    def _check_hands_raised(self, tid: int, kp: np.ndarray, conf: float,
                            frame_idx: int) -> AlertEvent | None:
        lwx, lwy = _kp(kp, L_WRIST);   rwx, rwy = _kp(kp, R_WRIST)
        lsx, lsy = _kp(kp, L_SHOULDER); rsx, rsy = _kp(kp, R_SHOULDER)

        l_raised = _visible(lwx, lwy) and _visible(lsx, lsy) and lwy < lsy
        r_raised = _visible(rwx, rwy) and _visible(rsx, rsy) and rwy < rsy

        if l_raised and r_raised:
            self._hands_counter[tid] += 1
        else:
            self._hands_counter[tid] = 0

        if self._hands_counter[tid] >= HANDS_FRAMES:
            if self._cooldown_ok("hands", tid, frame_idx):
                self._hands_counter[tid] = 0
                return AlertEvent(
                    event_type="BREAKIN_HANDS_RAISED",
                    confidence_score=round(conf, 3),
                    metadata={"track_id": tid},
                )
        return None

    def _check_after_hours(self, frame_idx: int) -> List[AlertEvent]:
        start = self._config.breakin_after_hours_start
        end   = self._config.breakin_after_hours_end
        hour  = datetime.now().hour

        in_after_hours = (
            (start > end and (hour >= start or hour < end)) or
            (start < end and start <= hour < end)
        )

        if in_after_hours:
            tid  = -1
            last = self._last_fired["after_hours"].get(tid, -300)
            if frame_idx - last >= 300:
                self._last_fired["after_hours"][tid] = frame_idx
                return [AlertEvent(
                    event_type="BREAKIN_AFTER_HOURS",
                    confidence_score=1.0,
                    metadata={"hour": datetime.now().strftime("%H:%M")},
                )]
        return []
