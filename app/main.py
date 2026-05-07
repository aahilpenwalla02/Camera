import json
import os
import queue
import sys
import tempfile
import threading
import time
from typing import List, Optional

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.detection.detector import Detector, load_models
from app.detection.pose_analyzer import PoseAnalyzer
from app.detection.zone_manager import ZoneManager
from app.learning.lstm_model import LSTMInference, MODEL_PATH
from app.learning.self_trainer import SelfTrainer
from app.learning.sequence_trainer import count_sequences, extract_sequences, train as lstm_train
from app.rules.anomaly_lstm import LSTMAnomalyRule
from app.rules.custom_rule import CustomRule, COCO_NAME_TO_ID
from app.rules.fight_detector import FightDetectorRule
from app.rules.conflict import ConflictRule
from app.rules.crowd import CrowdRule
from app.rules.restricted_area import RestrictedAreaRule
from app.rules.trash_overflow import TrashOverflowRule
from app.storage.database import AlertEvent, Database
from app.utils.config_loader import load_config
from app.utils import slm
from app.utils.drawing import (
    draw_alert_banner,
    draw_boxes,
    draw_fps,
    draw_person_count,
    draw_skeleton,
    draw_zones,
)
from app.utils.model_downloader import download_fight_model, download_trash_model

st.set_page_config(
    page_title="SentinelAI — CCTV Monitor",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_config():
    return load_config("config.yaml")


@st.cache_resource
def get_models_cached(det_path, pose_path, trash_path, fight_path):
    return load_models(det_path, pose_path, trash_path, fight_path)


@st.cache_resource
def get_database(db_path: str):
    return Database(db_path)


def get_trash_model_path(config) -> str:
    if os.path.exists(config.fine_tuned_model):
        return config.fine_tuned_model
    return download_trash_model()


CUSTOM_RULES_PATH = "data/custom_rules.json"


def load_custom_rules() -> List[dict]:
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(CUSTOM_RULES_PATH):
        return []
    try:
        with open(CUSTOM_RULES_PATH) as f:
            return json.load(f)
    except Exception:
        return []


def save_custom_rules(rules: List[dict]):
    os.makedirs("data", exist_ok=True)
    with open(CUSTOM_RULES_PATH, "w") as f:
        json.dump(rules, f, indent=2)


def build_custom_rule_objects(rule_configs: List[dict]) -> List[CustomRule]:
    return [CustomRule(cfg) for cfg in rule_configs]


ALERT_COLORS = {
    "FIGHT_DETECTED":         "#ff3b3b",
    "CONFLICT_DETECTED":      "#ff6b35",
    "CROWD_DETECTED":         "#ff9500",
    "RESTRICTED_AREA_BREACH": "#ff2d78",
    "TRASH_OVERFLOW":         "#ffd60a",
    "LSTM_ANOMALY":           "#bf5af2",
}

ALERT_LABELS = {
    "FIGHT_DETECTED":         "Fight",
    "CONFLICT_DETECTED":      "Conflict",
    "CROWD_DETECTED":         "Crowd",
    "RESTRICTED_AREA_BREACH": "Restricted Area",
    "TRASH_OVERFLOW":         "Trash Overflow",
    "LSTM_ANOMALY":           "Anomaly",
}


THEMES = {
    "light": {
        "bg":          "#f6f8fa",
        "sidebar_bg":  "#ffffff",
        "card_bg":     "#ffffff",
        "border":      "#d0d7de",
        "text":        "#24292f",
        "text_muted":  "#57606a",
        "metric_bg":   "#ffffff",
        "header_bg":   "linear-gradient(135deg,#ffffff,#f6f8fa)",
        "pill_bg":     "#f6f8fa",
        "sb_border":   "#d0d7de",
        "sb_label":    "#57606a",
        "sb_value":    "#24292f",
        "empty_bg":    "#f6f8fa",
        "empty_border":"#d0d7de",
        "empty_text":  "#8c959f",
    },
    "dark": {
        "bg":          "#080b12",
        "sidebar_bg":  "#0d1117",
        "card_bg":     "#0d1117",
        "border":      "#1c2333",
        "text":        "#f0f6fc",
        "text_muted":  "#8b949e",
        "metric_bg":   "#0d1117",
        "header_bg":   "linear-gradient(135deg,#0d1117,#161b22)",
        "pill_bg":     "#0d1117",
        "sb_border":   "#1c2333",
        "sb_label":    "#8b949e",
        "sb_value":    "#f0f6fc",
        "empty_bg":    "#0d1117",
        "empty_border":"#1c2333",
        "empty_text":  "#484f58",
    },
}


def inject_css(theme: str = "light"):
    t = THEMES[theme]
    st.markdown(f"""
    <style>
    #MainMenu {{visibility: hidden;}}
    footer     {{visibility: hidden;}}
    header     {{visibility: hidden;}}

    .stApp {{ background-color: {t['bg']}; }}

    [data-testid="stSidebar"] {{
        background-color: {t['sidebar_bg']};
        border-right: 1px solid {t['border']};
    }}

    [data-testid="metric-container"] {{
        background: {t['metric_bg']};
        border: 1px solid {t['border']};
        border-radius: 10px;
        padding: 18px 22px !important;
    }}
    [data-testid="stMetricValue"] {{
        font-size: 30px !important;
        font-weight: 700 !important;
        color: {t['text']} !important;
    }}
    [data-testid="stMetricLabel"] {{
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: {t['text_muted']} !important;
    }}

    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, #1f6feb, #388bfd);
        border: none;
        font-weight: 600;
        border-radius: 8px;
        padding: 10px 24px;
        color: #ffffff !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        background: linear-gradient(135deg, #388bfd, #58a6ff);
    }}

    .section-label {{
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: {t['text_muted']};
        margin-bottom: 10px;
        margin-top: 4px;
    }}

    .header-banner {{
        background: {t['header_bg']};
        border: 1px solid {t['border']};
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}
    .header-title {{
        font-size: 26px;
        font-weight: 700;
        color: {t['text']};
        letter-spacing: -0.5px;
    }}
    .header-sub {{
        font-size: 13px;
        color: {t['text_muted']};
        margin-top: 4px;
    }}
    .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: {t['pill_bg']};
        border: 1px solid {t['border']};
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 12px;
        font-weight: 600;
        color: {t['text_muted']};
    }}
    .dot       {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}
    .dot-green {{ background:#3fb950; box-shadow:0 0 6px #3fb950; }}
    .dot-grey  {{ background:#8c959f; }}

    .alert-badge {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    .card {{
        background: {t['card_bg']};
        border: 1px solid {t['border']};
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 12px;
    }}

    [data-testid="stImage"] img {{
        border-radius: 8px;
        border: 1px solid {t['border']};
    }}

    .stProgress > div > div {{ background: #1f6feb; border-radius: 4px; }}

    hr {{ border-color: {t['border']} !important; }}

    [data-testid="stDataFrame"] {{ border-radius: 8px; overflow: hidden; }}

    [data-testid="stExpander"] {{
        background: {t['card_bg']};
        border: 1px solid {t['border']} !important;
        border-radius: 8px;
    }}

    .sb-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 5px 0;
        border-bottom: 1px solid {t['sb_border']};
        font-size: 13px;
    }}
    .sb-label {{ color: {t['sb_label']}; }}
    .sb-value {{ color: {t['sb_value']}; font-weight: 500; }}
    .sb-ok    {{ color: #1a7f37; font-weight: 600; }}
    .sb-warn  {{ color: #d29922; font-weight: 600; }}
    </style>
    """, unsafe_allow_html=True)


def render_header(is_running: bool = False):
    dot_class = "dot-green" if is_running else "dot-grey"
    status_text = "LIVE" if is_running else "STANDBY"
    st.markdown(f"""
    <div class="header-banner">
        <div>
            <div class="header-title">🎥 SentinelAI</div>
            <div class="header-sub">Property CCTV Security Monitoring System</div>
        </div>
        <div class="status-pill">
            <span class="dot {dot_class}"></span> {status_text}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar(config, db: Database, device: str):
    from app.utils.model_downloader import FIGHT_MODEL_PATH

    st.sidebar.markdown("## SentinelAI")

    # Theme toggle
    current = st.session_state.get("theme", "light")
    toggle_label = "Switch to Dark" if current == "light" else "Switch to Light"
    if st.sidebar.button(toggle_label, use_container_width=True):
        st.session_state.theme = "dark" if current == "light" else "light"
        st.rerun()

    st.sidebar.markdown('<div class="section-label">System</div>', unsafe_allow_html=True)

    trash_ok = os.path.exists(config.trash_model) or os.path.exists(config.fine_tuned_model)
    fight_ok = os.path.exists(FIGHT_MODEL_PATH)
    fine_ok  = os.path.exists(config.fine_tuned_model)

    def sb_row(label, value, ok=True):
        cls = "sb-ok" if ok else "sb-warn"
        st.sidebar.markdown(
            f'<div class="sb-row"><span class="sb-label">{label}</span>'
            f'<span class="{cls}">{value}</span></div>',
            unsafe_allow_html=True,
        )

    sb_row("Device",       device.upper())
    sb_row("Det model",    config.det_model.split("/")[-1])
    sb_row("Pose model",   config.pose_model.split("/")[-1])
    sb_row("Trash model",  "Loaded" if trash_ok else "Downloading...", ok=trash_ok)
    sb_row("Fight model",  "Loaded" if fight_ok else "Downloading...", ok=fight_ok)
    sb_row("Fine-tuned",   "Active" if fine_ok else "—", ok=fine_ok)

    st.sidebar.markdown('<div class="section-label" style="margin-top:16px">Thresholds</div>',
                        unsafe_allow_html=True)
    sb_row("Crowd alert",  f"{config.crowd_threshold} persons")
    sb_row("Conflict IOU", str(config.conflict_iou_threshold))

    if config.zones:
        st.sidebar.markdown('<div class="section-label" style="margin-top:16px">Zones</div>',
                            unsafe_allow_html=True)
        for zone in config.zones:
            icon = "🔴" if zone.type.value == "RESTRICTED" else "🟠"
            st.sidebar.markdown(f"{icon} `{zone.name}`")

    st.sidebar.markdown("---")
    if st.sidebar.button("Clear Alert Log", use_container_width=True):
        db.clear()
        st.sidebar.success("Log cleared.")
        st.rerun()


def extract_clip(video_path: str, frame_number: int, fps: float,
                 duration: float = 4.0) -> Optional[bytes]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    half = int(fps * duration / 2)
    start_frame = max(0, frame_number - half)
    end_frame = frame_number + half

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
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


def get_first_frame(video_path: str) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def render_bin_calibration(video_path: Optional[str], rules: list):
    with st.sidebar.expander("Bin / Dumpster Zone", expanded=False):
        st.markdown("Mark the bin location so overflow detection targets that area.")

        if video_path and os.path.exists(video_path):
            frame = get_first_frame(video_path)
            if frame is not None:
                h, w = frame.shape[:2]
                col_a, col_b = st.columns(2)
                x1_pct = col_a.slider("Left %",   0, 100, st.session_state.get("bin_x1", 20), key="bin_x1")
                x2_pct = col_b.slider("Right %",  0, 100, st.session_state.get("bin_x2", 80), key="bin_x2")
                y1_pct = col_a.slider("Top %",    0, 100, st.session_state.get("bin_y1", 20), key="bin_y1")
                y2_pct = col_b.slider("Bottom %", 0, 100, st.session_state.get("bin_y2", 80), key="bin_y2")
                enabled = st.checkbox("Enable bin zone",
                                      value=st.session_state.get("bin_enabled", False),
                                      key="bin_enabled")

                if enabled and x1_pct < x2_pct and y1_pct < y2_pct:
                    x1 = int(x1_pct / 100 * w); y1 = int(y1_pct / 100 * h)
                    x2 = int(x2_pct / 100 * w); y2 = int(y2_pct / 100 * h)
                    bin_zone = (x1, y1, x2, y2)
                    preview = frame.copy()
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 200, 255), 3)
                    cv2.putText(preview, "BIN ZONE", (x1, max(y1 - 8, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                             use_container_width=True)
                else:
                    bin_zone = None
                    if not enabled:
                        st.caption("Density mode — full frame scan.")

                for rule in rules:
                    if isinstance(rule, TrashOverflowRule):
                        rule.update_bin_zone(bin_zone)
                        break
        else:
            st.caption("Upload a video first.")


def _alert_badge_html(event_type: str) -> str:
    color = ALERT_COLORS.get(event_type, "#8b949e")
    label = ALERT_LABELS.get(event_type, event_type)
    return (
        f'<span class="alert-badge" '
        f'style="background:{color}22;color:{color};border:1px solid {color}66;">'
        f'{label}</span>'
    )


def render_event_log(db: Database, video_path: Optional[str], video_fps: float):
    import pandas as pd

    recent = db.get_recent_events(50)
    if not recent:
        st.markdown('<div class="card" style="color:#8b949e;text-align:center;padding:32px;">No alerts logged yet.</div>',
                    unsafe_allow_html=True)
        return

    df = pd.DataFrame(recent)
    cols = [c for c in ["id", "timestamp", "event_type", "confidence_score", "frame_number"]
            if c in df.columns]
    df_display = df[cols].copy()
    if "confidence_score" in df_display.columns:
        df_display["confidence_score"] = df_display["confidence_score"].round(3)

    def row_style(row):
        color = ALERT_COLORS.get(row.get("event_type", ""), "#1c2333")
        return [f"background-color:{color}18; border-left:3px solid {color}"] * len(row)

    styled = df_display.style.apply(row_style, axis=1)

    st.caption("Click a row to play the 4-second clip around that alert.")
    selection = st.dataframe(
        styled,
        use_container_width=True,
        height=280,
        on_select="rerun",
        selection_mode="single-row",
        key="alert_table",
    )

    selected_rows = selection.selection.rows
    if selected_rows and video_path and os.path.exists(video_path):
        row = recent[selected_rows[0]]
        frame_num   = row.get("frame_number", 0)
        event_type  = row["event_type"]
        color       = ALERT_COLORS.get(event_type, "#8b949e")
        label       = ALERT_LABELS.get(event_type, event_type)
        timestamp   = row["timestamp"]

        st.markdown(
            f'<div style="margin:12px 0 8px;">{_alert_badge_html(event_type)}'
            f' <span style="color:#8b949e;font-size:13px;margin-left:8px;">'
            f'Frame {frame_num} · {timestamp}</span></div>',
            unsafe_allow_html=True,
        )

        # Show AI description if available
        meta = row.get("metadata", {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        ai_desc = meta.get("ai_description", "")
        if ai_desc:
            st.markdown(
                f'<div style="background:#1f6feb18;border-left:3px solid #1f6feb;'
                f'border-radius:0 6px 6px 0;padding:10px 14px;margin-bottom:10px;'
                f'font-size:13px;color:#57606a;">🤖 <em>{ai_desc}</em></div>',
                unsafe_allow_html=True,
            )

        with st.spinner("Extracting clip..."):
            clip_bytes = extract_clip(video_path, frame_num, video_fps)
        if clip_bytes:
            st.video(clip_bytes)
        else:
            st.warning("Could not extract clip.")
    elif selected_rows and not video_path:
        st.info("Re-upload the video to enable clip playback.")


def render_self_learning_panel(trainer: SelfTrainer, config):
    st.markdown('<div class="section-label">Self-Learning</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Session Samples", trainer.session_samples)
    c2.metric("Total Samples",   trainer.total_samples)
    c3.metric("Min to Fine-tune", config.self_learning_min_samples)

    if trainer.ready_to_train:
        st.success(f"{trainer.session_samples} samples collected — model ready to fine-tune.")
        if st.button("Fine-tune Model", type="primary"):
            with st.spinner("Fine-tuning... this may take a few minutes."):
                result_path = trainer.fine_tune(epochs=5)
            if result_path:
                st.success(f"Saved to `{result_path}`. Restart to activate.")
            else:
                st.error("Fine-tuning failed. Check terminal.")
    else:
        remaining = config.self_learning_min_samples - trainer.session_samples
        st.caption(f"{remaining} more samples needed to unlock fine-tuning.")


def render_lstm_panel(pose_model, lstm_inference: LSTMInference):
    st.markdown('<div class="section-label">LSTM Anomaly Detector</div>',
                unsafe_allow_html=True)

    n_normal, n_anomalous = count_sequences()
    status_color = "#3fb950" if lstm_inference.available else "#f85149"
    status_text  = "Trained" if lstm_inference.available else "Not trained"

    c1, c2, c3 = st.columns(3)
    c1.metric("Normal sequences",    n_normal)
    c2.metric("Anomalous sequences", n_anomalous)
    c3.markdown(
        f'<div style="padding-top:8px;font-size:13px;color:#8b949e;">Model status<br>'
        f'<span style="color:{status_color};font-weight:700;">{status_text}</span></div>',
        unsafe_allow_html=True,
    )

    with st.expander("Add training video"):
        train_upload = st.file_uploader(
            "Upload labelled video", type=["mp4", "avi", "mov", "mkv"], key="lstm_upload"
        )
        label = st.radio("Label:", ["normal", "anomalous"], horizontal=True)

        if train_upload and st.button("Extract Sequences", key="extract_btn"):
            suffix = os.path.splitext(train_upload.name)[1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(train_upload.read())
            tmp.close()
            with st.spinner(f"Extracting {label} sequences..."):
                saved = extract_sequences(tmp.name, label, pose_model)
            os.unlink(tmp.name)
            st.success(f"Saved {saved} sequences as **{label}**.")
            st.rerun()

    n_normal, n_anomalous = count_sequences()
    can_train = n_normal >= 5 and n_anomalous >= 5

    if can_train:
        epochs = st.slider("Epochs", 5, 30, 10, key="lstm_epochs")
        if st.button("Train LSTM", type="primary", key="train_lstm_btn"):
            pb = st.progress(0.0)
            status_ph = st.empty()

            def on_progress(epoch, total, loss, val_acc):
                pb.progress(epoch / total)
                status_ph.caption(f"Epoch {epoch}/{total} · Loss {loss:.4f} · Acc {val_acc:.1%}")

            with st.spinner("Training..."):
                result_path = lstm_train(epochs=epochs, progress_callback=on_progress)
            pb.progress(1.0)
            lstm_inference.reload()
            st.success(f"LSTM saved to `{result_path}`.")
    else:
        parts = []
        if n_normal < 5:
            parts.append(f"{5 - n_normal} more normal")
        if n_anomalous < 5:
            parts.append(f"{5 - n_anomalous} more anomalous")
        st.caption(f"Need {' and '.join(parts)} video(s) to train.")


def render_custom_rules_panel(video_path: Optional[str]):
    slm_ok = slm.is_available()
    if not slm_ok:
        st.warning("Ollama is not running. Start it to use custom rules.")

    rule_configs = load_custom_rules()

    # Show existing rules
    if rule_configs:
        st.markdown('<div class="section-label">Active Custom Rules</div>', unsafe_allow_html=True)
        for i, cfg in enumerate(rule_configs):
            col_info, col_del = st.columns([5, 1])
            with col_info:
                color = ALERT_COLORS.get(cfg["alert_name"], "#8b949e")
                st.markdown(
                    f'<div style="background:{color}18;border:1px solid {color}44;'
                    f'border-radius:8px;padding:10px 14px;margin-bottom:6px;">'
                    f'<strong style="color:{color};">{cfg["alert_name"]}</strong> — '
                    f'{cfg.get("description","")}<br>'
                    f'<small>Objects: {", ".join(cfg.get("objects",[]))} · '
                    f'Duration: {cfg.get("duration_seconds",0)}s'
                    f'{" · Zone set" if cfg.get("zone_pct") else " · Full frame"}</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with col_del:
                if st.button("Delete", key=f"del_rule_{i}"):
                    rule_configs.pop(i)
                    save_custom_rules(rule_configs)
                    st.rerun()

    # New rule builder
    st.markdown('<div class="section-label" style="margin-top:16px;">Add New Rule</div>',
                unsafe_allow_html=True)

    rule_text = st.text_area(
        "Describe what to detect",
        placeholder='e.g. "Flag any car parked for more than 2 minutes" or "Alert when a person enters the rooftop"',
        key="rule_text",
        height=80,
    )

    if rule_text and st.button("Interpret Rule with AI", type="primary", key="interpret_btn"):
        if not slm_ok:
            st.error("Ollama is not running.")
        else:
            with st.spinner("Asking phi3 to interpret your rule..."):
                parsed = slm.parse_rule(rule_text)
            if parsed:
                st.session_state["parsed_rule"] = parsed
            else:
                st.error("Could not interpret the rule. Try rephrasing it.")

    parsed = st.session_state.get("parsed_rule")
    if parsed:
        st.markdown("**Interpreted as:**")
        col_a, col_b = st.columns(2)
        col_a.markdown(f"**Alert name:** `{parsed.get('alert_name','')}`")
        col_b.markdown(f"**Duration:** `{parsed.get('duration_seconds',0)}s`")
        st.markdown(f"**Objects:** {', '.join(parsed.get('objects', []))}")
        st.markdown(f"**Description:** {parsed.get('description','')}")

        # Optional zone
        use_zone = st.checkbox("Restrict to a zone in the video", key="use_zone_custom")
        zone_pct = None
        if use_zone and video_path and os.path.exists(video_path):
            frame = get_first_frame(video_path)
            if frame is not None:
                h, w = frame.shape[:2]
                cz1, cz2 = st.columns(2)
                zx1 = cz1.slider("Zone Left %",   0, 100, 10, key="czx1")
                zx2 = cz2.slider("Zone Right %",  0, 100, 90, key="czx2")
                zy1 = cz1.slider("Zone Top %",    0, 100, 10, key="czy1")
                zy2 = cz2.slider("Zone Bottom %", 0, 100, 90, key="czy2")
                if zx1 < zx2 and zy1 < zy2:
                    zone_pct = [zx1, zy1, zx2, zy2]
                    preview = frame.copy()
                    cv2.rectangle(preview,
                                  (int(zx1/100*w), int(zy1/100*h)),
                                  (int(zx2/100*w), int(zy2/100*h)),
                                  (255, 165, 0), 3)
                    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                             caption="Rule zone preview", use_container_width=True)

        if st.button("Save Rule", type="primary", key="save_rule_btn"):
            new_rule = {**parsed}
            if zone_pct:
                new_rule["zone_pct"] = zone_pct
            rule_configs.append(new_rule)
            save_custom_rules(rule_configs)
            st.session_state.pop("parsed_rule", None)
            st.success(f"Rule `{new_rule['alert_name']}` saved.")
            st.rerun()


def build_rules(config, zone_manager, pose_analyzer, has_trash_model: bool,
                lstm_inference: LSTMInference, has_fight_model: bool):
    rules = [
        RestrictedAreaRule(zone_manager),
        CrowdRule(config),
        TrashOverflowRule(has_trash_model=has_trash_model),
        ConflictRule(pose_analyzer),
        LSTMAnomalyRule(lstm_inference),
    ]
    if has_fight_model:
        rules.append(FightDetectorRule())
    return rules


def process_video(video_path: str, config, detector: Detector,
                  pose_analyzer: PoseAnalyzer, zone_manager: ZoneManager,
                  db: Database, rules: list, custom_rules: List[CustomRule],
                  trainer: Optional[SelfTrainer],
                  frame_ph, events_ph, metrics_ph, heavy_every: int = 1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Failed to open video file.")
        return None

    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_alerts = 0
    frame_idx    = 0
    prev_time    = time.perf_counter()
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    COOLDOWN_FRAMES  = 45
    last_alert_frame: dict = {}

    # SLM background worker — sends frames to moondream without blocking
    slm_available = slm.is_available()
    desc_queue: queue.Queue = queue.Queue(maxsize=10)

    def _slm_worker():
        while True:
            item = desc_queue.get()
            if item is None:
                break
            frm, alert_id, event_type = item
            try:
                desc = slm.describe_frame(frm, event_type)
                if desc:
                    db.update_description(alert_id, desc)
            except Exception:
                pass
            desc_queue.task_done()

    slm_thread = threading.Thread(target=_slm_worker, daemon=True)
    slm_thread.start()

    col_stop, col_prog = st.columns([1, 5])
    stop_btn = col_stop.button("Stop", key="stop_btn")
    progress = col_prog.progress(0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    try:
        while cap.isOpened():
            if stop_btn:
                break

            ret, frame = cap.read()
            if not ret:
                break

            run_heavy = (frame_idx % heavy_every == 0)
            result    = detector.run(frame, run_heavy=run_heavy)

            if trainer and config.self_learning_enabled and len(result.boxes_xyxy) > 0:
                trainer.collect_sample(frame, result.boxes_xyxy,
                                       result.class_ids, result.confidences, frame_idx)
                if len(result.trash_boxes_xyxy) > 0:
                    trainer.collect_sample(frame, result.trash_boxes_xyxy,
                                           result.trash_class_ids, result.trash_confidences,
                                           frame_idx + 1_000_000)

            raw_alerts: List[AlertEvent] = []
            for rule in rules:
                if isinstance(rule, (LSTMAnomalyRule, FightDetectorRule, TrashOverflowRule)):
                    raw_alerts.extend(rule.evaluate(result, frame_idx))
                else:
                    raw_alerts.extend(rule.evaluate(result))

            for crule in custom_rules:
                raw_alerts.extend(crule.evaluate(result, frame_w, frame_h, frame_idx))

            frame_alerts: List[AlertEvent] = []
            for alert in raw_alerts:
                last = last_alert_frame.get(alert.event_type, -COOLDOWN_FRAMES)
                if frame_idx - last >= COOLDOWN_FRAMES:
                    alert.frame_number = frame_idx
                    frame_alerts.append(alert)
                    last_alert_frame[alert.event_type] = frame_idx

            for alert in frame_alerts:
                alert_id = db.log_event(alert)
                if slm_available and not desc_queue.full():
                    desc_queue.put_nowait((frame.copy(), alert_id, alert.event_type))

            total_alerts += len(frame_alerts)

            now       = time.perf_counter()
            fps       = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            person_count = int(result.person_mask.sum())
            alert_types  = [a.event_type for a in frame_alerts]

            draw_zones(frame, zone_manager.configs)
            draw_boxes(frame, result.person_boxes,
                       result.person_track_ids if len(result.person_track_ids) else None)
            if len(result.trash_boxes_xyxy) > 0:
                draw_boxes(frame, result.trash_boxes_xyxy,
                           result.trash_track_ids if len(result.trash_track_ids) else None,
                           color=(0, 165, 255))
            if len(result.keypoints_xy) > 0:
                draw_skeleton(frame, result.keypoints_xy)
            draw_fps(frame, fps)
            draw_person_count(frame, person_count)
            draw_alert_banner(frame, alert_types)

            frame_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            with metrics_ph.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Processing FPS",  f"{fps:.1f}")
                c2.metric("Persons Tracked", person_count)
                c3.metric("Total Alerts",    total_alerts)
                c4.metric("Train Samples",   trainer.session_samples if trainer else 0)

            if frame_idx % 5 == 0:
                import pandas as pd
                recent = db.get_recent_events(20)
                if recent:
                    df = pd.DataFrame(recent)
                    cols_live = [c for c in ["timestamp", "event_type", "confidence_score"]
                                 if c in df.columns]
                    events_ph.dataframe(df[cols_live], use_container_width=True, height=220)

            progress.progress(min(frame_idx / total_frames, 1.0))
            frame_idx += 1

    finally:
        desc_queue.put(None)  # stop SLM worker
        cap.release()
        progress.progress(1.0)
        st.success(f"Analysis complete — {frame_idx} frames · {total_alerts} alerts")

    return video_fps


def main():
    if "theme" not in st.session_state:
        st.session_state.theme = "light"
    inject_css(st.session_state.theme)

    config = get_config()

    trash_model_path = get_trash_model_path(config)
    fight_model_path = download_fight_model()
    bundle = get_models_cached(config.det_model, config.pose_model,
                               trash_model_path, fight_model_path)
    db = get_database(config.db_path)

    detector     = Detector(bundle)
    pose_analyzer = PoseAnalyzer(config)
    zone_manager  = ZoneManager(config.zones)
    has_trash_model = bundle.trash_model is not None
    has_fight_model = bundle.fight_model is not None

    trainer = SelfTrainer(
        base_model_path=trash_model_path or config.det_model,
        output_model_path=config.fine_tuned_model,
        class_names=bundle.trash_class_names or ["object"],
    ) if config.self_learning_enabled else None

    lstm_inference = LSTMInference(MODEL_PATH)
    rules = build_rules(config, zone_manager, pose_analyzer,
                        has_trash_model, lstm_inference, has_fight_model)
    custom_rule_configs = load_custom_rules()
    custom_rules = build_custom_rule_objects(custom_rule_configs)

    from app.detection.detector import DEVICE
    render_sidebar(config, db, DEVICE)
    render_bin_calibration(st.session_state.get("video_path"), rules)

    if "video_path" not in st.session_state:
        st.session_state.video_path = None
    if "video_fps" not in st.session_state:
        st.session_state.video_fps = 30.0

    is_running = st.session_state.get("analysis_running", False)
    render_header(is_running)

    # ── Upload + controls ──────────────────────────────────────────────
    upload_col, ctrl_col = st.columns([3, 1])

    with upload_col:
        uploaded = st.file_uploader(
            "Upload CCTV footage", type=["mp4", "avi", "mov", "mkv"], key="uploader",
            label_visibility="collapsed",
        )

    with ctrl_col:
        speed_mode = st.selectbox(
            "Speed",
            ["Accurate", "Balanced", "Fast"],
            index=1,
            help="Accurate: every frame · Balanced: heavy models every 2nd frame · Fast: every 3rd",
        )
    heavy_every = {"Accurate": 1, "Balanced": 2, "Fast": 3}[speed_mode]

    # ── Main layout: video | events ────────────────────────────────────
    col_video, col_events = st.columns([3, 2])

    with col_video:
        st.markdown('<div class="section-label">Camera Feed</div>', unsafe_allow_html=True)
        frame_ph = st.empty()

    metrics_ph = st.empty()

    if uploaded is not None:
        detector.reset()
        pose_analyzer.reset()
        if trainer:
            trainer.reset_session()
        for rule in rules:
            if isinstance(rule, (LSTMAnomalyRule, FightDetectorRule)):
                rule.reset()

        run = st.button("Run Analysis", type="primary", use_container_width=False)

        if run:
            if st.session_state.video_path and os.path.exists(st.session_state.video_path):
                os.unlink(st.session_state.video_path)
                st.session_state.video_path = None

            suffix = os.path.splitext(uploaded.name)[1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded.read())
            tmp.close()
            st.session_state.video_path = tmp.name
            st.session_state.analysis_running = True

            with col_events:
                st.markdown('<div class="section-label">Live Alerts</div>',
                            unsafe_allow_html=True)
                events_ph = st.empty()

            video_fps = process_video(
                tmp.name, config, detector, pose_analyzer,
                zone_manager, db, rules, custom_rules, trainer,
                frame_ph, events_ph, metrics_ph,
                heavy_every=heavy_every,
            )
            st.session_state.analysis_running = False
            if video_fps:
                st.session_state.video_fps = video_fps

        # ── Bottom panels ──────────────────────────────────────────────
        st.divider()
        st.markdown('<div class="section-label">Alert Log</div>', unsafe_allow_html=True)
        render_event_log(db, st.session_state.video_path, st.session_state.video_fps)

        st.divider()
        tab_rules, tab_lstm, tab_learn = st.tabs(["Custom Rules", "LSTM Anomaly Detector", "Self-Learning"])
        with tab_rules:
            render_custom_rules_panel(st.session_state.video_path)
        with tab_lstm:
            render_lstm_panel(bundle.pose_model, lstm_inference)
        with tab_learn:
            if trainer:
                render_self_learning_panel(trainer, config)
            else:
                st.caption("Self-learning disabled in config.yaml.")

    else:
        t = THEMES[st.session_state.get("theme", "light")]
        frame_ph.markdown(
            f'<div style="background:{t["empty_bg"]};border:1px dashed {t["empty_border"]};'
            f'border-radius:8px;padding:60px;text-align:center;color:{t["empty_text"]};">'
            f'No footage uploaded</div>',
            unsafe_allow_html=True,
        )
        st.divider()
        tab_rules, tab_lstm, tab_learn = st.tabs(["Custom Rules", "LSTM Anomaly Detector", "Self-Learning"])
        with tab_rules:
            render_custom_rules_panel(st.session_state.video_path)
        with tab_lstm:
            render_lstm_panel(bundle.pose_model, lstm_inference)
        with tab_learn:
            if trainer:
                render_self_learning_panel(trainer, config)

        if st.session_state.video_path:
            st.divider()
            st.markdown('<div class="section-label">Alert Log</div>', unsafe_allow_html=True)
            render_event_log(db, st.session_state.video_path, st.session_state.video_fps)


if __name__ == "__main__":
    main()
