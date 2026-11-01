"""1) video_input: 영상 소스(RTSP / 웹캠 / mp4)에서 최신 프레임 1장을 수집하여 State로 전달"""
import threading
import time
from collections import deque
from datetime import datetime
from urllib.parse import urlparse

import cv2

SOURCE_LABELS = {"rtsp": "RTSP", "webcam": "웹캠", "file": "영상 파일"}


def mask_rtsp(url: str) -> str:
    """rtsp://user:pw@IP:port/... → rtsp://***@IP (계정·비밀번호·경로 노출 금지)"""
    try:
        host = urlparse(url).hostname or "?"
    except Exception:
        host = "?"
    return f"rtsp://***@{host}"


def describe_source(kind: str, path) -> str:
    if kind == "rtsp":
        return f"RTSP {mask_rtsp(str(path))}"
    if kind == "webcam":
        return f"웹캠 {path}"
    return f"영상 파일 {str(path).replace(chr(92), '/').split('/')[-1]}"


def resize_keep_ratio(frame, width: int):
    h, w = frame.shape[:2]
    if w == width:
        return frame
    return cv2.resize(frame, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


class VideoSource:
    """VideoCapture 래퍼. 라이브 소스는 백그라운드 스레드로 계속 읽어 최신 프레임만 보관(쌓인 프레임 버림),
    mp4는 실제 재생 속도에 맞춰 재생 위치를 진행."""

    def __init__(self, kind: str, path):
        self.kind = kind
        self.path = path
        self.cap = None
        self._lock = threading.Lock()
        self._latest = None
        self._latest_seq = 0
        self._last_frame_time = 0.0
        self._reader = None
        self._running = False
        self._play_start = 0.0
        self.ended = False

    # ---------- 연결 ----------
    def open(self) -> bool:
        self.close()
        if self.kind == "webcam":
            cap = cv2.VideoCapture(int(self.path), cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(int(self.path))
        elif self.kind == "rtsp":
            cap = cv2.VideoCapture(str(self.path), cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            cap.release()
            return False
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            return False
        self.cap = cap
        self.ended = False
        if self.kind == "file":
            self._latest = frame
            self._play_start = time.time()
        else:
            if not self._play_start:
                self._play_start = time.time()
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            with self._lock:
                self._latest = frame
                self._latest_seq += 1
                self._last_frame_time = time.time()
            self._running = True
            self._reader = threading.Thread(target=self._read_loop, daemon=True)
            self._reader.start()
        return True

    def close(self):
        self._running = False
        if self._reader is not None:
            self._reader.join(timeout=2)
            self._reader = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _read_loop(self):
        while self._running and self.cap is not None:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            with self._lock:
                self._latest = frame
                self._latest_seq += 1
                self._last_frame_time = time.time()

    # ---------- 읽기 ----------
    def read(self, stale_timeout: float = 2.0):
        """(frame, video_time) 반환. 실패 시 (None, None)."""
        if self.cap is None:
            return None, None
        if self.kind == "file":
            return self._read_file()
        now = time.time()
        with self._lock:
            if now - self._last_frame_time > stale_timeout:
                return None, None
            frame = self._latest
        return frame.copy(), now - self._play_start

    def _read_file(self):
        target = time.time() - self._play_start  # 실제 경과 시간 = 재생 위치
        frame = None
        while True:
            pos = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if frame is not None and pos >= target:
                break
            if not self.cap.grab():
                self.ended = True
                return None, None
            if self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 >= target:
                ok, frame = self.cap.retrieve()
                if not ok:
                    self.ended = True
                    return None, None
                break
        return frame, self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0


class VideoInputNode:
    def __init__(self, cfg: dict, log, stop_event: threading.Event):
        self.cfg = cfg
        self.log = log
        self.stop_event = stop_event
        self.source: VideoSource | None = None
        self.frame_id = 0
        self.times = deque(maxlen=10)
        self._last_fps_warn = 0.0

    def connect(self, kind: str, path) -> bool:
        """최초 연결 (실패 시 reconnect_interval 간격 reconnect_max회 재시도)"""
        self.log(f"영상 연결 중: {describe_source(kind, path)}")
        self.source = VideoSource(kind, path)
        self.frame_id = 0
        self.times.clear()
        if self.source.open():
            self.log("영상 연결 완료")
            return True
        return self._reconnect()

    def _reconnect(self) -> bool:
        interval = float(self.cfg["reconnect_interval"])
        max_try = int(self.cfg["reconnect_max"])
        for i in range(1, max_try + 1):
            self.log(f"영상 연결 실패: 소스를 확인하세요 (재시도 {i}/{max_try})", error=True)
            if self.stop_event.wait(interval):
                return False
            if self.source.open():
                self.log("영상 연결 완료")
                return True
        self.log("영상 연결 실패: 소스를 확인하세요 (재연결 실패, 감시 중지)", error=True)
        return False

    def release(self):
        if self.source is not None:
            self.source.close()
            self.source = None

    def __call__(self, state: dict) -> dict:
        now = time.time()
        frame, video_time = (None, None) if self.source is None else self.source.read()
        if frame is None:
            if self.source is not None and self.source.ended:
                return {"source_status": "ended", "frame": None}
            ok = self.source is not None and self._reconnect()
            return {"source_status": "disconnected" if ok else "failed", "frame": None}

        frame = resize_keep_ratio(frame, int(self.cfg["resize_width"]))
        self.frame_id += 1
        self.times.append(now)
        fps = (len(self.times) - 1) / (self.times[-1] - self.times[0]) if len(self.times) >= 2 else 0.0

        target_fps = float(self.cfg["target_fps"])
        if len(self.times) == self.times.maxlen and fps < target_fps * 0.95 and now - self._last_fps_warn > 30:  # 30초에 1번만 표시
            self._last_fps_warn = now
            self.log(f"처리 지연: {fps:.1f} FPS", error=True)

        return {
            "frame": frame,
            "frame_id": self.frame_id,
            "timestamp": now,
            "timestamp_str": datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "video_time": round(float(video_time or 0.0), 1),
            "fps": round(fps, 1),
            "source_status": "connected",
            "annotated_frame": frame.copy(),
        }
