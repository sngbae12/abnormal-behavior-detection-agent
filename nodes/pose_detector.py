"""2) pose_detector: yolov8n-pose + ByteTrack으로 사람별 17개 키포인트·추적 ID 추출"""
import cv2

GREEN = (0x47, 0xA0, 0x43)  # BGR of #43A047

# COCO 17 키포인트 연결선
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


class PoseDetectorNode:
    def __init__(self, cfg: dict, model):
        self.cfg = cfg
        self.model = model

    def reset_tracker(self):
        """감시 시작마다 추적 ID 초기화"""
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", []) or []:
            tracker.reset()

    def __call__(self, state: dict) -> dict:
        frame = state["frame"]
        person_conf = float(self.cfg["person_conf"])
        kp_conf = float(self.cfg["keypoint_conf"])

        results = self.model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            conf=person_conf, classes=[0], imgsz=int(self.cfg["pose_imgsz"]), verbose=False,
        )
        persons = []
        r = results[0] if results else None
        if r is not None and r.boxes is not None and r.boxes.id is not None and len(r.boxes) > 0:
            ids = r.boxes.id.int().cpu().tolist()
            boxes = r.boxes.xyxy.cpu().tolist()
            confs = r.boxes.conf.cpu().tolist()
            kps = r.keypoints.data.cpu().tolist() if r.keypoints is not None else [[] for _ in ids]
            for tid, box, conf, kp in zip(ids, boxes, confs, kps):
                if conf < person_conf:
                    continue
                persons.append({
                    "track_id": int(tid),
                    "bbox": [int(round(v)) for v in box],
                    "person_conf": round(float(conf), 2),
                    # 키포인트는 모델 출력 그대로 저장 (보정·생성 금지)
                    "keypoints": [[round(float(x), 1), round(float(y), 1), round(float(c), 2)] for x, y, c in kp],
                })

        annotated = state.get("annotated_frame", frame.copy())
        for p in persons:
            draw_skeleton(annotated, p["keypoints"], kp_conf)
            draw_person_box(annotated, p, GREEN, 2)
        return {"persons": persons, "annotated_frame": annotated}


def draw_skeleton(img, keypoints, kp_conf):
    if not keypoints:
        return
    for a, b in SKELETON:
        xa, ya, ca = keypoints[a]
        xb, yb, cb = keypoints[b]
        if ca >= kp_conf and cb >= kp_conf:
            cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), (255, 200, 0), 2, cv2.LINE_AA)
    for x, y, c in keypoints:
        if c >= kp_conf:
            cv2.circle(img, (int(x), int(y)), 3, (0, 255, 255), -1, cv2.LINE_AA)


def draw_person_box(img, p, color, thickness):
    x1, y1, x2, y2 = p["bbox"]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    draw_label(img, f"ID {p['track_id']} person {p['person_conf']:.2f}", x1, y1, color)


def draw_label(img, text, x, y, color):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y_top = max(0, y - th - base - 4)
    cv2.rectangle(img, (x, y_top), (x + tw + 6, y_top + th + base + 4), color, -1)
    cv2.putText(img, text, (x + 3, y_top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
