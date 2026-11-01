"""4) evaluator: 작업자별 넘어짐·구조요청·안전모 미착용 지속시간을 누적하여 정상/주의/위험/판정불가 결정"""
import math
from datetime import datetime

from nodes.pose_detector import draw_person_box

DANGER, CAUTION, UNKNOWN, NORMAL = "위험", "주의", "판정불가", "정상"
FALL, HELP, NO_HELMET = "넘어짐", "구조요청", "안전모 미착용"
EVENT_KEYS = {FALL: "fall", HELP: "help", NO_HELMET: "no_helmet"}
# 전체 상태 우선순위: 위험 > 주의 > 정상 > 판정불가 (판정 가능한 사람이 한 명이라도 있으면 회색으로 덮지 않음)
LEVEL_RANK = {DANGER: 3, CAUTION: 2, NORMAL: 1, UNKNOWN: 0}
LEVEL_BGR = {DANGER: (0x35, 0x39, 0xE5), CAUTION: (0x2D, 0xC0, 0xFB)}  # #E53935, #FBC02D


def torso_angle(kp):
    """양어깨 중심 → 양골반 중심 선이 수직선과 이루는 각(도)"""
    sx, sy = (kp[5][0] + kp[6][0]) / 2, (kp[5][1] + kp[6][1]) / 2
    hx, hy = (kp[11][0] + kp[12][0]) / 2, (kp[11][1] + kp[12][1]) / 2
    return math.degrees(math.atan2(abs(hx - sx), abs(hy - sy)))


def judge_fall(p, cfg):
    """True/False, 키포인트 부족 시 None(판정불가)"""
    kp, th = p["keypoints"], float(cfg["keypoint_conf"])
    if not kp or any(kp[i][2] < th for i in (5, 6, 11, 12)):
        return None
    x1, y1, x2, y2 = p["bbox"]
    ratio = (x2 - x1) / max(1, (y2 - y1))
    return torso_angle(kp) >= float(cfg["fall_torso_angle"]) and ratio >= float(cfg["fall_box_ratio"])


def judge_help(p, cfg):
    """양 손목 y < 코 y (화면 좌표상 손목이 코보다 위)"""
    kp, th = p["keypoints"], float(cfg["keypoint_conf"])
    if not kp or any(kp[i][2] < th for i in (0, 9, 10)):
        return None
    return kp[9][1] < kp[0][1] and kp[10][1] < kp[0][1]


class EvaluatorNode:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def __call__(self, state: dict) -> dict:
        cfg = self.cfg
        ts = state["timestamp"]
        persons = state.get("persons", [])
        prev = state.get("durations") or {}
        marks = dict(state.get("log_marks") or {})
        danger_s = float(cfg["danger_seconds"])
        caution_s = float(cfg["no_helmet_seconds"])
        grace = float(cfg["reset_grace_seconds"])
        clock = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

        durations, person_states, logs = {}, [], []
        for p in persons:
            tid = p["track_id"]
            old_meta = prev.get(tid, {}).get("_meta", {})
            helmet_status = p.get("helmet_status", "unknown")
            cond = {  # True: 조건 충족 / False: 조건 해소 확인 / None: 판정불가
                "fall": judge_fall(p, cfg),
                "help": judge_help(p, cfg),
                "no_helmet": {"no_helmet": True, "helmet": False}.get(helmet_status),
            }
            # 지속시간 = 마지막 감지 시각 - 최초 감지 시각 (timestamp 기준).
            # 조건 해소가 확인되면 즉시 0으로 초기화, 판정불가가 grace 초 이내로 잠깐 끼면 유지
            cur, meta = {}, {}
            for key, hit in cond.items():
                m = old_meta.get(key)
                if hit:
                    m = {"start": m["start"] if m else ts, "last": ts}
                elif hit is False or not m or ts - m["last"] > grace:
                    m = None
                if m:
                    meta[key] = m
                    cur[key] = m["last"] - m["start"]
            durations[tid] = {**cur, "_meta": meta}

            events = []
            for ev, key, limit, level in ((FALL, "fall", danger_s, DANGER), (HELP, "help", danger_s, DANGER),
                                          (NO_HELMET, "no_helmet", caution_s, CAUTION)):
                if key not in cur:
                    marks.pop((tid, ev), None)
                    continue
                d = round(cur[key], 1)
                if d >= limit:
                    events.append({"event_type": ev, "level": level, "duration": d})
                    continue
                tick = int(d)  # 같은 작업자·유형의 감지 중 로그는 1초 단위로만 추가
                if marks.get((tid, ev)) != tick:
                    marks[(tid, ev)] = tick
                    logs.append(f"[{clock}] 이상 징후 감지 중: {ev} (ID {tid}, {d:.1f}초 / {limit:g}초)")

            posture_unknown = cond["fall"] is None and cond["help"] is None
            if events:
                top = max(events, key=lambda e: (LEVEL_RANK[e["level"]], e["duration"]))
                status, ev_type, dur = top["level"], top["event_type"], top["duration"]
            elif posture_unknown and p.get("helmet_status", "unknown") == "unknown":
                status, ev_type, dur = UNKNOWN, None, 0.0  # 자세·안전모 모두 판단할 근거가 없음
            else:
                status, ev_type, dur = NORMAL, None, 0.0

            if status == UNKNOWN:
                if marks.get((tid, UNKNOWN)) is None or ts - marks[(tid, UNKNOWN)] >= 30:
                    marks[(tid, UNKNOWN)] = ts
                    logs.append(f"[{clock}] 판정불가: ID {tid} 신체 지점 인식 부족")
            else:
                marks.pop((tid, UNKNOWN), None)

            person_states.append({"track_id": tid, "status": status, "event_type": ev_type,
                                  "duration": dur, "events": events,
                                  "posture_checked": not posture_unknown})

        # 소실된 track_id의 로그 기록 정리
        live = {p["track_id"] for p in persons}
        marks = {k: v for k, v in marks.items() if k[0] in live}

        if not person_states:
            overall = UNKNOWN
        else:
            overall = max((s["status"] for s in person_states), key=LEVEL_RANK.get)

        annotated = state["annotated_frame"]
        by_id = {p["track_id"]: p for p in persons}
        for s in person_states:
            if s["status"] in LEVEL_BGR:
                draw_person_box(annotated, by_id[s["track_id"]], LEVEL_BGR[s["status"]], 3)

        return {"person_states": person_states, "durations": durations, "log_marks": marks,
                "overall_status": overall, "detect_logs": logs, "annotated_frame": annotated}
