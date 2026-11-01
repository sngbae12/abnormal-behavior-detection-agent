"""3) helmet_detector: helmet.pt로 안전모 착용/미착용 판별 후 사람별 머리 영역과 매칭"""
import re

import cv2

from nodes.pose_detector import draw_label

GREEN = (0x47, 0xA0, 0x43)  # #43A047
RED = (0x35, 0x39, 0xE5)    # #E53935

HELMET, NO_HELMET = "Helmet", "No Helmet"

# 정규화된 클래스명(소문자, 영숫자만) → 표준 클래스
_NO_HELMET_ALIASES = {
    "nohelmet", "withouthelmet", "nohardhat", "withouthardhat", "nohat", "nosafetyhelmet",
    "head", "nohelm", "notwearinghelmet", "helmetoff", "bareheaded", "nohelmetperson",
}
_HELMET_ALIASES = {
    "helmet", "hardhat", "withhelmet", "withhardhat", "safetyhelmet", "hat", "helmeton",
    "wearinghelmet", "helmetperson",
}


def map_class_names(names: dict) -> dict:
    """model.names를 {class_id: "Helmet" | "No Helmet"}로 매핑 (해당 없는 클래스는 제외)"""
    mapping = {}
    for cid, name in names.items():
        key = re.sub(r"[^a-z0-9]", "", str(name).lower())
        if key in _NO_HELMET_ALIASES or (key.startswith(("no", "without", "not")) and ("helmet" in key or "hardhat" in key)):
            mapping[int(cid)] = NO_HELMET
        elif key in _HELMET_ALIASES:
            mapping[int(cid)] = HELMET
    return mapping


def head_region(person: dict, kp_conf: float):
    """코·눈·귀(0~4) 중 신뢰도 기준 이상 점들을 감싸는 박스, 없으면 person 박스 상단 25%"""
    pts = [(x, y) for x, y, c in person["keypoints"][:5] if c >= kp_conf] if person["keypoints"] else []
    x1, y1, x2, y2 = person["bbox"]
    if len(pts) >= 2:
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        hx1, hy1, hx2, hy2 = min(xs), min(ys), max(xs), max(ys)
        # 점들만 감싸면 박스가 매우 작으므로 머리 크기만큼 여유를 둠 (점 좌표 자체는 수정하지 않음)
        pad = max(hx2 - hx1, hy2 - hy1, 10) * 0.6
        return [hx1 - pad, hy1 - pad * 1.5, hx2 + pad, hy2 + pad]
    if len(pts) == 1:
        h = (y2 - y1) * 0.25 / 2
        return [pts[0][0] - h, pts[0][1] - h, pts[0][0] + h, pts[0][1] + h]
    return [x1, y1, x2, y1 + (y2 - y1) * 0.25]


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class HelmetDetectorNode:
    def __init__(self, cfg: dict, model, log):
        self.cfg = cfg
        self.model = model
        self.enabled = False
        self.mapping = {}
        self._dets, self._det_frame = [], -1  # helmet_every_n 프레임마다 추론, 사이 프레임은 재사용
        if model is None:
            return
        names = dict(model.names)
        log(f"helmet.pt 클래스: {names}")
        self.mapping = map_class_names(names)
        found = set(self.mapping.values())
        if HELMET in found and NO_HELMET in found:
            self.enabled = True
            if any(names[c] != v for c, v in self.mapping.items()):
                log("helmet.pt 클래스 매핑: " + ", ".join(f"{c}:'{names[c]}' → {v}" for c, v in self.mapping.items()))
        else:
            log(f"helmet.pt에 Helmet / No Helmet 클래스가 없습니다: {names} → 안전모 판별 중지 (자세 감지는 계속)", error=True)

    def __call__(self, state: dict) -> dict:
        persons = state.get("persons", [])
        annotated = state["annotated_frame"]
        if not self.enabled:
            for p in persons:
                p["helmet_status"], p["helmet_conf"] = "unknown", 0.0
            return {"persons": persons}

        fid = state.get("frame_id", 0)
        every = max(1, int(self.cfg["helmet_every_n"]))
        if fid < self._det_frame or fid - self._det_frame >= every:  # 새 감시 시작 시 frame_id가 다시 1부터 시작
            self._dets, self._det_frame = self._infer(state["frame"]), fid
        dets = self._dets

        kp_conf = float(self.cfg["keypoint_conf"])
        nh_conf = float(self.cfg["no_helmet_conf"])
        for p in persons:
            head = head_region(p, kp_conf)
            matched = [d for d in dets if iou(head, d["bbox"]) > 0]
            if not matched:
                p["helmet_status"], p["helmet_conf"] = "unknown", 0.0
                continue
            best = max(matched, key=lambda d: d["conf"])
            p["helmet_conf"] = round(best["conf"], 2)
            # 신뢰도 기준 미만이면 착용/미착용 어느 쪽도 단정하지 않음 (오검출로 지속시간이 초기화되는 것 방지)
            if best["conf"] < nh_conf:
                p["helmet_status"] = "unknown"
            else:
                p["helmet_status"] = "no_helmet" if best["label"] == NO_HELMET else "helmet"

        for d in dets:
            color = GREEN if d["label"] == HELMET else RED
            x1, y1, x2, y2 = (int(v) for v in d["bbox"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            draw_label(annotated, f"{d['label']} {d['conf']:.2f}", x1, y2 + 18, color)
        return {"persons": persons, "annotated_frame": annotated}

    def _infer(self, frame):
        results = self.model.predict(frame, conf=float(self.cfg["helmet_detect_conf"]),
                                     imgsz=int(self.cfg["helmet_imgsz"]), verbose=False)
        dets = []
        r = results[0] if results else None
        if r is not None and r.boxes is not None:
            for box, cls, conf in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.cls.int().cpu().tolist(), r.boxes.conf.cpu().tolist()):
                label = self.mapping.get(int(cls))
                if label:
                    dets.append({"bbox": box, "label": label, "conf": float(conf)})
        return dets
