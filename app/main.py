import os
import sys
import tempfile
import time
from typing import List, Optional

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.detection.detector import Detector, load_models
from app.detection.pose_analyzer import PoseAnalyzer
from app.detection.zone_manager import ZoneManager
from app.learning.self_trainer import SelfTrainer
from app.rules.conflict import ConflictRule
from app.rules.crowd import CrowdRule
from app.rules.restricted_area import RestrictedAreaRule
from app.rules.trash_overflow import TrashOverflowRule
from app.storage.database import AlertEvent, Database
from app.utils.config_loader import load_config
from app.utils.drawing import (
    draw_alert_banner,
    draw_boxes,
    draw_fps,
    draw_person_count,
    draw_skeleton,
    draw_zones,
)
from app.utils.model_downloader import download_trash_model

st.set_page_config(
    page_title="CCTV Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_config():
    return load_config("config.yaml")


@st.cache_resource
def get_models_cached(det_path, pose_path, trash_path):
    return load_models(det_path, pose_path, trash_path)


@st.cache_resource
def get_database(db_path: str):
    return Database(db_path)


def get_trash_model_path(config) -> str:
    """Return fine-tuned model if it exists, else download base trash model."""
    if os.path.exists(config.fine_tuned_model):
        return config.fine_tuned_model
    return download_trash_model()


def build_rules(config, zone_manager, pose_analyzer, has_trash_model: bool):
    return [
        RestrictedAreaRule(zone_manager),
        CrowdRule(config),
        TrashOverflowRule(zone_manager, config, has_trash_model=has_trash_model),
        ConflictRule(pose_analyzer),
    ]


def render_sidebar(config, db: Database):
    st.sidebar.title("Configuration")
    st.sidebar.markdown(f"**Det model:** `{config.det_model}`")
    st.sidebar.markdown(f"**Pose model:** `{config.pose_model}`")
    trash_status = "✅ Loaded" if os.path.exists(config.trash_model) or \
                   os.path.exists(config.fine_tuned_model) else "⬇️ Will download"
    st.sidebar.markdown(f"**Trash model:** {trash_status}")
    fine_tuned = "✅ Active" if os.path.exists(config.fine_tuned_model) else "—"
    st.sidebar.markdown(f"**Fine-tuned model:** {fine_tuned}")
    st.sidebar.markdown(f"**Crowd threshold:** {config.crowd_threshold} persons")
    st.sidebar.markdown(f"**Trash stationary:** {config.trash_stationary_seconds}s")
    st.sidebar.markdown(f"**Conflict IOU:** {config.conflict_iou_threshold}")
    st.sidebar.markdown(f"**Conflict velocity:** {config.conflict_velocity_threshold} px/frame")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Zones**")
    for zone in config.zones:
        color = "🔴" if zone.type.value == "RESTRICTED" else "🟠"
        st.sidebar.markdown(f"{color} `{zone.name}` ({zone.type.value})")
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑 Clear Alert Log", use_container_width=True):
        db.clear()
        st.sidebar.success("Alert log cleared.")
        st.rerun()


def extract_clip(video_path: str, frame_number: int, fps: float,
                 duration: float = 4.0) -> Optional[bytes]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    half = int(fps * duration / 2)
    start_frame = max(0, frame_number - half)
    end_frame = frame_number + half

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_out.close()

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(tmp_out.name, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    cap.release()
    writer.release()

    with open(tmp_out.name, "rb") as f:
        clip_bytes = f.read()
    os.unlink(tmp_out.name)

    return clip_bytes if len(clip_bytes) > 0 else None


def render_event_log(db: Database, video_path: Optional[str], video_fps: float):
    import pandas as pd

    recent = db.get_recent_events(50)
    if not recent:
        st.info("No alerts logged yet.")
        return

    df = pd.DataFrame(recent)
    cols = [c for c in ["id", "timestamp", "event_type", "confidence_score", "frame_number"]
            if c in df.columns]
    df = df[cols]
    if "confidence_score" in df.columns:
        df["confidence_score"] = df["confidence_score"].round(3)

    st.markdown("**Click a row to view the 4-second clip around that alert.**")
    selection = st.dataframe(
        df,
        use_container_width=True,
        height=300,
        on_select="rerun",
        selection_mode="single-row",
        key="alert_table",
    )

    selected_rows = selection.selection.rows
    if selected_rows and video_path and os.path.exists(video_path):
        row = recent[selected_rows[0]]
        frame_num = row.get("frame_number", 0)
        event_type = row["event_type"]
        timestamp = row["timestamp"]

        st.markdown(f"**{event_type}** at frame {frame_num} ({timestamp})")
        with st.spinner("Extracting clip..."):
            clip_bytes = extract_clip(video_path, frame_num, video_fps)

        if clip_bytes:
            st.video(clip_bytes)
        else:
            st.warning("Could not extract clip from video.")
    elif selected_rows and not video_path:
        st.info("Re-upload the video to enable clip playback.")


def render_self_learning_panel(trainer: SelfTrainer, config):
    st.subheader("Self-Learning")
    col1, col2, col3 = st.columns(3)
    col1.metric("Session Samples", trainer.session_samples)
    col2.metric("Total Samples", trainer.total_samples)
    col3.metric("Min to Train", config.self_learning_min_samples)

    if trainer.ready_to_train:
        st.success(f"✅ {trainer.session_samples} new samples collected — model can be fine-tuned.")
        if st.button("🧠 Fine-tune Model Now", type="primary"):
            with st.spinner("Fine-tuning model on collected data... this may take several minutes."):
                result_path = trainer.fine_tune(epochs=5)
            if result_path:
                st.success(f"Fine-tuned model saved to `{result_path}`. Restart the app to use it.")
            else:
                st.error("Fine-tuning failed. Check the terminal for details.")
    else:
        remaining = config.self_learning_min_samples - trainer.session_samples
        st.info(f"Collect {remaining} more high-confidence samples to unlock fine-tuning.")
        if trainer.total_samples > 0:
            st.caption(f"Total samples accumulated across all sessions: {trainer.total_samples}")


def process_video(video_path: str, config, detector: Detector,
                  pose_analyzer: PoseAnalyzer, zone_manager: ZoneManager,
                  db: Database, rules: list, trainer: Optional[SelfTrainer],
                  frame_ph, events_ph, metrics_ph):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Failed to open video file.")
        return None

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_alerts = 0
    frame_idx = 0
    prev_time = time.perf_counter()

    COOLDOWN_FRAMES = 90
    last_alert_frame: dict = {}

    stop_btn = st.button("⏹ Stop", key="stop_btn")
    progress = st.progress(0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    try:
        while cap.isOpened():
            if stop_btn:
                break

            ret, frame = cap.read()
            if not ret:
                break

            result = detector.run(frame)

            # Self-learning: collect samples from general detections
            if trainer and config.self_learning_enabled and len(result.boxes_xyxy) > 0:
                trainer.collect_sample(
                    frame,
                    result.boxes_xyxy,
                    result.class_ids,
                    result.confidences,
                    frame_idx,
                )
                # Also collect from trash model detections
                if len(result.trash_boxes_xyxy) > 0:
                    trainer.collect_sample(
                        frame,
                        result.trash_boxes_xyxy,
                        result.trash_class_ids,
                        result.trash_confidences,
                        frame_idx + 1_000_000,  # offset to avoid filename collision
                    )

            raw_alerts: List[AlertEvent] = []
            for rule in rules:
                raw_alerts.extend(rule.evaluate(result))

            frame_alerts: List[AlertEvent] = []
            for alert in raw_alerts:
                last = last_alert_frame.get(alert.event_type, -COOLDOWN_FRAMES)
                if frame_idx - last >= COOLDOWN_FRAMES:
                    alert.frame_number = frame_idx
                    frame_alerts.append(alert)
                    last_alert_frame[alert.event_type] = frame_idx

            for alert in frame_alerts:
                db.log_event(alert)
            total_alerts += len(frame_alerts)

            now = time.perf_counter()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            person_count = int(result.person_mask.sum())
            alert_types = [a.event_type for a in frame_alerts]

            draw_zones(frame, zone_manager.configs)
            draw_boxes(frame, result.person_boxes,
                       result.person_track_ids if len(result.person_track_ids) else None)
            # Draw trash detections in orange
            if len(result.trash_boxes_xyxy) > 0:
                draw_boxes(frame, result.trash_boxes_xyxy,
                           result.trash_track_ids if len(result.trash_track_ids) else None,
                           color=(0, 165, 255))
            if len(result.keypoints_xy) > 0:
                draw_skeleton(frame, result.keypoints_xy)
            draw_fps(frame, fps)
            draw_person_count(frame, person_count)
            draw_alert_banner(frame, alert_types)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_ph.image(frame_rgb, use_container_width=True)

            with metrics_ph.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("FPS", f"{fps:.1f}")
                c2.metric("Persons", person_count)
                c3.metric("Total Alerts", total_alerts)
                c4.metric("Training Samples", trainer.session_samples if trainer else 0)

            if frame_idx % 5 == 0:
                import pandas as pd
                recent = db.get_recent_events(50)
                if recent:
                    df = pd.DataFrame(recent)
                    cols = [c for c in ["timestamp", "event_type", "confidence_score", "frame_number"]
                            if c in df.columns]
                    events_ph.dataframe(df[cols], use_container_width=True, height=300)

            progress.progress(min(frame_idx / total_frames, 1.0))
            frame_idx += 1

    finally:
        cap.release()
        progress.progress(1.0)
        st.success(f"Analysis complete — {frame_idx} frames processed, {total_alerts} alerts logged.")

    return video_fps


def main():
    config = get_config()

    trash_model_path = get_trash_model_path(config)
    bundle = get_models_cached(config.det_model, config.pose_model, trash_model_path)
    db = get_database(config.db_path)

    detector = Detector(bundle)
    pose_analyzer = PoseAnalyzer(config)
    zone_manager = ZoneManager(config.zones)
    has_trash_model = bundle.trash_model is not None

    # SelfTrainer uses the base trash model (or fine-tuned if exists)
    trainer = SelfTrainer(
        base_model_path=trash_model_path or config.det_model,
        output_model_path=config.fine_tuned_model,
        class_names=bundle.trash_class_names or ["object"],
    ) if config.self_learning_enabled else None

    rules = build_rules(config, zone_manager, pose_analyzer, has_trash_model)

    render_sidebar(config, db)

    if "video_path" not in st.session_state:
        st.session_state.video_path = None
    if "video_fps" not in st.session_state:
        st.session_state.video_fps = 30.0

    st.title("CCTV Security Monitor")
    st.markdown("Upload a video file to run real-time event detection locally.")

    uploaded = st.file_uploader(
        "Upload CCTV video", type=["mp4", "avi", "mov", "mkv"], key="uploader"
    )

    col_video, col_events = st.columns([3, 2])

    with col_video:
        st.subheader("Live Feed")
        frame_ph = st.empty()

    metrics_ph = st.empty()

    if uploaded is not None:
        detector.reset()
        pose_analyzer.reset()
        if trainer:
            trainer.reset_session()

        run = st.button("▶ Run Analysis", type="primary")

        if run:
            if st.session_state.video_path and os.path.exists(st.session_state.video_path):
                os.unlink(st.session_state.video_path)
                st.session_state.video_path = None

            suffix = os.path.splitext(uploaded.name)[1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded.read())
            tmp.close()
            st.session_state.video_path = tmp.name

            with col_events:
                st.subheader("Event Log (live)")
                events_ph = st.empty()

            video_fps = process_video(
                tmp.name, config, detector, pose_analyzer,
                zone_manager, db, rules, trainer,
                frame_ph, events_ph, metrics_ph,
            )
            if video_fps:
                st.session_state.video_fps = video_fps

        # Self-learning panel
        if trainer:
            st.divider()
            render_self_learning_panel(trainer, config)

        # Interactive alert log
        st.divider()
        st.subheader("Alert Log — Click a row to view clip")
        with col_events:
            pass  # col_events used during live run only
        render_event_log(db, st.session_state.video_path, st.session_state.video_fps)

    else:
        frame_ph.info("Waiting for video upload...")
        if st.session_state.video_path:
            if trainer:
                st.divider()
                render_self_learning_panel(trainer, config)
            st.divider()
            st.subheader("Alert Log — Click a row to view clip")
            render_event_log(db, st.session_state.video_path, st.session_state.video_fps)


if __name__ == "__main__":
    main()
