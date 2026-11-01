"""5) alarm: 경보 생성·유지·해제, 점멸 정보와 경보음(벨/사이렌) 출력, [선택] Slack 발송"""
import os
import threading
import time
from datetime import datetime

import cv2

try:
    import winsound
except ImportError:  # Windows 이외 환경
    winsound = None

from nodes.evaluator import CAUTION, DANGER, FALL

COLORS = {DANGER: "#E53935", CAUTION: "#FBC02D"}
MESSAGES = {
    CAUTION: "주의 : 안전모 미착용자가 있습니다, 신속하게 조치바랍니다.",
    DANGER: "위험 : 쓰러진 작업자가 있습니다, 즉시 현장 확인바랍니다.",
    "help": "위험 : 구조요청 자세가 감지되었습니다, 즉시 현장 확인바랍니다.",
}


class SoundPlayer:
    """경보음 비동기 재생.
    - .wav: winsound (SND_ASYNC, 사이렌은 SND_LOOP)
    - .mp3/.m4a 등: ffmpeg가 있으면 sounds/_wav/에 wav로 1회 변환 후 winsound, 없으면 Windows MCI로 재생
    - 파일 없음·재생 불가: winsound.Beep(별도 스레드)으로 대체"""

    def __init__(self, cfg: dict, sound_dir: str, log):
        self.log = log
        self.sound_dir = sound_dir
        self.bell = self._resolve(cfg["bell_sound"])
        self.siren = self._resolve(cfg["siren_sound"])
        self.siren_on = False
        self._beep_stop = threading.Event()

    # ---------- 파일 준비 ----------
    def _resolve(self, name: str):
        """(방식, 경로) 반환: ("wav", path) / ("mci", path) / ("beep", None)"""
        path = os.path.join(self.sound_dir, name)
        if not os.path.isfile(path):
            self.log(f"경보음 파일 없음: {name} (비프음으로 대체)", error=True)
            return "beep", None
        if path.lower().endswith(".wav"):
            return "wav", path
        wav = self._convert(path)
        if wav:
            return "wav", wav
        if self._mci(f'open "{path}" type mpegvideo alias chk') == 0:
            self._mci("close chk")
            return "mci", path
        self.log(f"경보음 재생 불가 형식: {name} (wav로 변환하거나 비프음으로 대체)", error=True)
        return "beep", None

    def _convert(self, src: str):
        import shutil
        import subprocess

        dst_dir = os.path.join(self.sound_dir, "_wav")
        dst = os.path.join(dst_dir, os.path.splitext(os.path.basename(src))[0] + ".wav")
        if os.path.isfile(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
            return dst
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return None
        try:
            os.makedirs(dst_dir, exist_ok=True)
            subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-i", src, "-acodec", "pcm_s16le", "-ar", "44100", dst],
                           check=True, timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return dst
        except Exception:
            return None

    @staticmethod
    def _mci(command: str) -> int:
        try:
            import ctypes
            return ctypes.windll.winmm.mciSendStringW(command, None, 0, 0)
        except Exception:
            return -1

    # ---------- 재생 ----------
    def _play(self, sound, loop: bool, alias: str):
        mode, path = sound
        if mode == "wav" and winsound is not None:
            flags = winsound.SND_FILENAME | winsound.SND_ASYNC | (winsound.SND_LOOP if loop else 0)
            winsound.PlaySound(path, flags)
        elif mode == "mci":
            self._mci(f"close {alias}")
            self._mci(f'open "{path}" type mpegvideo alias {alias}')
            self._mci(f"play {alias}" + (" repeat" if loop else ""))
        elif winsound is not None:
            self._beep_stop.clear()
            threading.Thread(target=self._beep_siren if loop else self._beep_bell, daemon=True).start()

    def play_bell(self):
        self._play(self.bell, loop=False, alias="bell_snd")

    def start_siren(self):
        if self.siren_on:
            return
        self.siren_on = True
        self._play(self.siren, loop=True, alias="siren_snd")

    def stop(self):
        self.siren_on = False
        self._beep_stop.set()
        if winsound is not None:
            winsound.PlaySound(None, 0)
        self._mci("close bell_snd")
        self._mci("close siren_snd")

    def _beep_bell(self):
        for freq in (1320, 990, 1320, 990):
            winsound.Beep(freq, 220)

    def _beep_siren(self):
        while not self._beep_stop.is_set():
            for freq in (750, 1500):
                if self._beep_stop.is_set():
                    return
                winsound.Beep(freq, 350)


class SlackNotifier:
    """[선택] Incoming Webhook 텍스트 + Bot API 이미지 첨부. 키 미설정 시 발송만 건너뜀 (키 값은 출력 금지)"""

    def __init__(self, log):
        self.log = log
        self.webhook = os.getenv("SLACK_WEBHOOK_URL", "").strip()
        self.token = os.getenv("SLACK_BOT_TOKEN", "").strip()
        self.channel = os.getenv("SLACK_CHANNEL_ID", "").strip()

    @property
    def enabled(self):
        return bool(self.webhook or (self.token and self.channel))

    def send(self, alert: dict, message: str, frame=None):
        if not self.enabled:
            return
        jpg = None
        if frame is not None and self.token and self.channel:
            ok, buf = cv2.imencode(".jpg", frame)
            jpg = buf.tobytes() if ok else None
        threading.Thread(target=self._send, args=(alert, message, jpg), daemon=True).start()

    def _send(self, alert, message, jpg):
        import requests

        text = (f"[{alert['level']}] {message}\n경보ID {alert['alert_id']} | {alert['camera_id']} | "
                f"ID {alert['track_id']} | {alert['event_type']} | {alert['start_str']}")
        def post_text():
            requests.post(self.webhook, json={"text": text}, timeout=5).raise_for_status()

        def post_image():
            self._upload(requests, jpg, f"{alert['alert_id']}.jpg", f"{alert['alert_id']} 경보 시점 캡처")

        # 텍스트와 이미지를 따로 재시도 (한쪽 실패로 다른 쪽이 중복 발송되지 않게)
        jobs = ([post_text] if self.webhook else []) + ([post_image] if jpg is not None else [])
        for job in jobs:
            for i in range(1, 4):
                try:
                    job()
                    break
                except Exception:
                    self.log(f"Slack 발송 실패 (재시도 {i}/3)", error=True)
                    time.sleep(2)

    def _upload(self, requests, data: bytes, filename: str, text: str):
        headers = {"Authorization": f"Bearer {self.token}"}
        r = requests.post("https://slack.com/api/files.getUploadURLExternal", headers=headers,
                          data={"filename": filename, "length": len(data)}, timeout=10).json()
        if not r.get("ok"):
            raise RuntimeError("getUploadURLExternal")
        requests.post(r["upload_url"], files={"file": (filename, data, "image/jpeg")}, timeout=20).raise_for_status()
        done = requests.post("https://slack.com/api/files.completeUploadExternal", headers=headers, json={
            "files": [{"id": r["file_id"], "title": filename}],
            "channel_id": self.channel, "initial_comment": text,
        }, timeout=10).json()
        if not done.get("ok"):
            raise RuntimeError("completeUploadExternal")


class AlarmNode:
    def __init__(self, cfg: dict, sound_dir: str, log, known_ids):
        """known_ids: 기존 경보ID 목록을 돌려주는 함수 (경보ID 일련번호 중복 방지)"""
        self.cfg = cfg
        self.log = log
        self.known_ids = known_ids
        self.sound = SoundPlayer(cfg, sound_dir, log)
        self.slack = SlackNotifier(log)
        self._seq = {}

    def _next_id(self, ts: float) -> str:
        day = datetime.fromtimestamp(ts).strftime("%Y%m%d")
        if day not in self._seq:
            prefix = f"A-{day}-"
            nums = [int(i[len(prefix):]) for i in self.known_ids() if i.startswith(prefix) and i[len(prefix):].isdigit()]
            self._seq[day] = max(nums, default=0)
        self._seq[day] += 1
        return f"A-{day}-{self._seq[day]:03d}"

    def stop_sound(self):
        self.sound.stop()

    def __call__(self, state: dict) -> dict:
        now = state.get("timestamp") or time.time()
        active = [dict(a) for a in state.get("active_alerts") or []]
        released_keys = {tuple(k) for k in state.get("released_keys") or []}
        logs = list(state.get("detect_logs") or [])
        new_alerts, released = [], []

        # 감독자 [경보 해제]: 진행 중 경보 전체 해제 (자동 해제 없음)
        if state.get("release_requested") and active:
            for a in active:
                a["state"] = "해제"
                a["release"] = now
                released_keys.add((a["track_id"], a["event_type"]))
            released, active = active, []
            self.sound.stop()
            logs.append(f"[{datetime.fromtimestamp(now):%H:%M:%S}] 경보 해제 ({len(released)}건)")

        if "person_states" in state:
            current = {(s["track_id"], e["event_type"]): (s, e)
                       for s in state["person_states"] for e in s.get("events", [])}
            # 해제된 경보 키는 조건이 해소(정상 복귀·추적 소실)된 뒤에만 다시 경보 가능
            released_keys = {k for k in released_keys if k in current}
            active_keys = {(a["track_id"], a["event_type"]) for a in active}
            camera_id = self.cfg["camera_id"]
            for key, (s, e) in current.items():
                if key in released_keys:
                    continue
                if key in active_keys:
                    for a in active:
                        if (a["track_id"], a["event_type"]) == key:
                            a["duration"] = e["duration"]
                    continue
                alert = {
                    "alert_id": self._next_id(now), "track_id": key[0], "event_type": key[1],
                    "level": e["level"], "start": now,
                    "start_str": datetime.fromtimestamp(now).strftime("%H:%M:%S"),
                    "camera_id": camera_id, "state": "진행중", "duration": e["duration"],
                }
                active.append(alert)
                new_alerts.append(alert)
                logs.append(f"[{alert['start_str']}] 경보 발생: {alert['level']} - {alert['event_type']} (ID {key[0]})")

        active.sort(key=lambda a: a["start"])
        dangers = [a for a in active if a["level"] == DANGER]
        if dangers:
            self.sound.start_siren()
            level = DANGER
            msg = MESSAGES[DANGER] if any(a["event_type"] == FALL for a in dangers) else MESSAGES["help"]
        elif active:
            if any(a["level"] == CAUTION for a in new_alerts):
                self.sound.play_bell()
            level, msg = CAUTION, MESSAGES[CAUTION]
        else:
            level, msg = None, ""

        for a in new_alerts:
            m = MESSAGES[CAUTION] if a["level"] == CAUTION else (MESSAGES[DANGER] if a["event_type"] == FALL else MESSAGES["help"])
            self.slack.send(a, m, state.get("annotated_frame") if a["level"] == DANGER else None)

        return {
            "active_alerts": active, "new_alerts": new_alerts, "released_alerts": released,
            "released_keys": sorted(released_keys, key=str), "detect_logs": logs,
            "alarm_view": {"level": level, "color": COLORS.get(level), "message": msg},
        }
