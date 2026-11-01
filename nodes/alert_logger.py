"""6) alert_logger: 경보 발생·해제 이력을 alert_log.csv(UTF-8-BOM)에 경보 사건 단위로 기록"""
import csv
import os
import shutil
from datetime import datetime

COLUMNS = ["경보ID", "발생시각", "카메라번호", "작업자추적ID", "감지유형", "등급", "경보상태", "해제시각", "처리시간(초)"]
FMT = "%Y-%m-%d %H:%M:%S"


class AlertLoggerNode:
    def __init__(self, cfg: dict, base_dir: str, log):
        self.cfg = cfg
        self.log = log
        self.path = os.path.join(base_dir, cfg["log_file"])
        self.rows = {}        # 경보ID → 행(dict), 파일 전체를 메모리에 보관
        self.dirty = False    # 저장 실패 시 True → 다음 기록 때 재시도
        self._load()

    def _load(self):
        if not os.path.isfile(self.path):
            return
        try:
            with open(self.path, encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("경보ID"):
                        self.rows[row["경보ID"]] = {c: row.get(c, "") for c in COLUMNS}
        except Exception as e:
            self.log(f"이력 파일 읽기 실패: {e}", error=True)

    def known_ids(self):
        return list(self.rows.keys())

    def _save(self) -> bool:
        try:
            with open(self.path, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=COLUMNS)
                w.writeheader()
                w.writerows(self.rows.values())
            self.dirty = False
            return True
        except OSError:
            self.dirty = True
            self.log("이력 저장 실패: 파일이 열려 있는지 확인하세요", error=True)
            return False

    def __call__(self, state: dict) -> dict:
        new_alerts = state.get("new_alerts") or []
        released = state.get("released_alerts") or []
        if not new_alerts and not released:  # 저장 실패분은 다음 기록 때 함께 재시도
            return {}
        for a in new_alerts:
            self.rows[a["alert_id"]] = {
                "경보ID": a["alert_id"],
                "발생시각": datetime.fromtimestamp(a["start"]).strftime(FMT),
                "카메라번호": a["camera_id"],
                "작업자추적ID": a["track_id"],
                "감지유형": a["event_type"],
                "등급": a["level"],
                "경보상태": "진행중",
                "해제시각": "",
                "처리시간(초)": "",
            }
        for a in released:
            row = self.rows.get(a["alert_id"])
            if row is None:
                continue
            row["경보상태"] = "해제"
            row["해제시각"] = datetime.fromtimestamp(a["release"]).strftime(FMT)
            row["처리시간(초)"] = f"{a['release'] - a['start']:.1f}"
        self._save()
        return {}

    def export(self, dest: str):
        """[CSV 저장]: 사용자가 고른 위치로 복사"""
        if self.dirty:
            self._save()
        if not os.path.isfile(self.path):
            self._save()
        shutil.copyfile(self.path, dest)
