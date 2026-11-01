"""LangGraph StateGraph: video_input → pose_detector → helmet_detector → evaluator → alarm → alert_logger"""
import os
from typing import Any, TypedDict

import yaml
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
SOUND_DIR = os.path.join(BASE_DIR, "sounds")

REQUIRED_KEYS = [
    "camera_id", "target_fps", "resize_width", "pose_model", "helmet_model", "person_conf", "keypoint_conf",
    "no_helmet_conf", "helmet_detect_conf", "no_helmet_seconds", "danger_seconds", "fall_torso_angle",
    "fall_box_ratio", "blink_interval", "reconnect_interval", "reconnect_max", "log_max_lines",
    "alert_panel_max", "bell_sound", "siren_sound", "log_file",
    "pose_imgsz", "helmet_imgsz", "helmet_every_n", "reset_grace_seconds",
]


class AgentState(TypedDict, total=False):
    # video_input
    frame: Any
    frame_id: int
    timestamp: float
    timestamp_str: str
    video_time: float
    fps: float
    source_status: str
    # pose_detector / helmet_detector
    persons: list
    annotated_frame: Any
    # evaluator (durations·log_marks는 프레임 간 유지)
    person_states: list
    durations: dict
    log_marks: dict
    overall_status: str
    detect_logs: list
    # alarm (active_alerts·released_keys는 프레임 간 유지)
    release_requested: bool
    active_alerts: list
    new_alerts: list
    released_alerts: list
    released_keys: list
    alarm_view: dict


def load_config() -> dict:
    load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)
    with open(os.path.join(BASE_DIR, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"config.yaml 항목 누락: {', '.join(missing)}")
    if os.getenv("CAMERA_ID", "").strip():
        cfg["camera_id"] = os.getenv("CAMERA_ID").strip()
    return cfg


def load_models(cfg: dict, log):
    """(pose_model, helmet_model) 반환. 파일이 없으면 FileNotFoundError(파일명)"""
    from ultralytics import YOLO

    os.makedirs(MODEL_DIR, exist_ok=True)
    helmet_path = os.path.join(MODEL_DIR, cfg["helmet_model"])
    if not os.path.isfile(helmet_path):
        raise FileNotFoundError(cfg["helmet_model"])

    pose_path = os.path.join(MODEL_DIR, cfg["pose_model"])
    if not os.path.isfile(pose_path):
        log(f"{cfg['pose_model']} 없음 → ultralytics 자동 다운로드 시도 (models/ 폴더)")
    try:
        pose = YOLO(pose_path)  # 공식 모델명이면 models/ 경로로 자동 다운로드
    except Exception:
        raise FileNotFoundError(cfg["pose_model"])
    if not os.path.isfile(pose_path):
        raise FileNotFoundError(cfg["pose_model"])
    helmet = YOLO(helmet_path)
    return pose, helmet


def build_graph(video_input, pose_detector, helmet_detector, evaluator, alarm, alert_logger):
    g = StateGraph(AgentState)
    g.add_node("video_input", video_input)
    g.add_node("pose_detector", pose_detector)
    g.add_node("helmet_detector", helmet_detector)
    g.add_node("evaluator", evaluator)
    g.add_node("alarm", alarm)
    g.add_node("alert_logger", alert_logger)

    def after_video(state: AgentState) -> str:
        if state.get("source_status") == "connected":
            return "pose_detector"
        # 프레임을 못 읽으면 분석 노드는 건너뜀 (감독자 해제 요청만 처리)
        return "alarm" if state.get("release_requested") else END

    g.add_edge(START, "video_input")
    g.add_conditional_edges("video_input", after_video, ["pose_detector", "alarm", END])
    g.add_edge("pose_detector", "helmet_detector")
    g.add_edge("helmet_detector", "evaluator")
    g.add_edge("evaluator", "alarm")
    g.add_edge("alarm", "alert_logger")
    g.add_edge("alert_logger", END)
    return g.compile()
