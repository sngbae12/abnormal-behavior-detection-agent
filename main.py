"""이상 동작 감지 Agent - Tkinter 화면(메인 스레드) + LangGraph 분석 스레드"""
import os
import queue
import re
import threading
import time
import traceback
from datetime import datetime
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

from graph import BASE_DIR, SOUND_DIR, build_graph, load_config, load_models
from nodes.alarm import COLORS, MESSAGES, AlarmNode
from nodes.alert_logger import COLUMNS, AlertLoggerNode
from nodes.evaluator import EvaluatorNode
from nodes.helmet_detector import HelmetDetectorNode
from nodes.pose_detector import PoseDetectorNode
from nodes.video_input import VideoInputNode, mask_rtsp

# ---------- 컬러 (ui.md 상태색 + 시안 레이아웃) ----------
APP_BG, HEADER_BG = "#0F1620", "#121B26"
DARK_CARD, DARK_BORDER, DARK_SUB = "#17212C", "#273444", "#223040"
LIGHT_CARD, LIGHT_BORDER, LIGHT_SUB = "#FFFFFF", "#E2E6EB", "#F4F6F8"
TEXT_DARK, TEXT_MUTED, TEXT_LIGHT, TEXT_DIM = "#222222", "#6B7785", "#FFFFFF", "#9AA6B2"
VIDEO_BG = "#1E1E1E"
C_DANGER, C_CAUTION, C_NORMAL, C_UNKNOWN = "#E53935", "#FBC02D", "#43A047", "#9E9E9E"
STATUS_COLORS = {"위험": C_DANGER, "주의": C_CAUTION, "정상": C_NORMAL, "판정불가": C_UNKNOWN}
STATUS_TINTS = {"위험": "#FDECEA", "주의": "#FFF8E1", "정상": "#E8F5E9", "판정불가": "#EEEEEE"}
BTN_BG, BTN_HOVER, BTN_OFF = "#37474F", "#455A64", "#616161"
BTN_START, BTN_START_HOVER = "#2E7D32", "#388E3C"

FONT = "맑은 고딕"
F_SMALL, F_LOG, F_BTN = (FONT, 10), (FONT, 12), (FONT, 12, "bold")
F_TITLE, F_CARD = (FONT, 17, "bold"), (FONT, 15, "bold")
SYMBOL_FONT = "Segoe UI Symbol"
ICON_FONT = None  # Segoe Fluent Icons / MDL2 Assets (Windows 기본 아이콘 글꼴)
IC_SHIELD, IC_VIDEO, IC_CAMERA, IC_DOC, IC_BELL = "", "", "", "", ""
IC_FOLDER, IC_PLAY, IC_STOP = "", "", ""

SRC_RTSP, SRC_RTSP_OFF, SRC_WEBCAM, SRC_FILE = "RTSP 카메라", "RTSP (미설정)", "웹캠", "영상 파일"
TIME_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s*(.*)$")


def rounded_rect(c: tk.Canvas, x1, y1, x2, y2, r, **kw):
    r = max(0, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
    pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2, x2 - r, y2,
           x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return c.create_polygon(pts, smooth=True, **kw)


# ================= 공통 위젯 =================
class RoundButton(tk.Canvas):
    def __init__(self, parent, text, command, *, icon="", bg=BTN_BG, hover=BTN_HOVER, fg=TEXT_LIGHT,
                 outline=None, width=120, height=38, font=F_BTN, radius=8, disabled_bg="#2A3542",
                 disabled_fg="#7D8894"):
        super().__init__(parent, width=width, height=height, bg=parent["bg"], highlightthickness=0, bd=0,
                         cursor="hand2")
        self.text, self.icon, self.command = text, icon, command
        self.bg, self.hover_bg, self.fg, self.outline = bg, hover, fg, outline
        self.disabled_bg, self.disabled_fg = disabled_bg, disabled_fg
        self.font, self.radius = font, radius
        self.enabled, self._hover = True, False
        self.bind("<Configure>", lambda e: self.redraw())
        self.bind("<Enter>", lambda e: self._set_hover(True))
        self.bind("<Leave>", lambda e: self._set_hover(False))
        self.bind("<ButtonRelease-1>", self._click)

    def _set_hover(self, v):
        self._hover = v
        self.redraw()

    def _click(self, _):
        if self.enabled and self.command:
            self.command()

    def set_enabled(self, enabled: bool):
        if self.enabled != enabled:
            self.enabled = enabled
            self.configure(cursor="hand2" if enabled else "arrow")
            self.redraw()

    def set_colors(self, bg, hover):
        self.bg, self.hover_bg = bg, hover
        self.redraw()

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 4:
            w, h = int(self["width"]), int(self["height"])
        if self.enabled:
            fill = self.hover_bg if self._hover else self.bg
            fg = self.fg
        else:
            fill, fg = self.disabled_bg, self.disabled_fg
        rounded_rect(self, 1, 1, w - 1, h - 1, self.radius, fill=fill, outline=self.outline or fill)
        f = tkfont.Font(font=self.font)
        icon_w = 0
        if self.icon and ICON_FONT:
            icon_f = tkfont.Font(family=ICON_FONT, size=f.actual("size"))
            icon_w = icon_f.measure(self.icon) + 8
        total = icon_w + f.measure(self.text)
        x = (w - total) / 2
        if icon_w:
            self.create_text(x, h / 2 + 1, text=self.icon, font=icon_f, fill=fg, anchor="w")
        self.create_text(x + icon_w, h / 2, text=self.text, font=self.font, fill=fg, anchor="w")


class StatusChip(tk.Canvas):
    SYMBOLS = {"정상": "✓", "주의": "!", "위험": "!", "판정불가": "?"}

    def __init__(self, parent, status):
        super().__init__(parent, height=46, bg=parent["bg"], highlightthickness=0, bd=0)
        self.status, self.active = status, False
        self.bind("<Configure>", lambda e: self.redraw())

    def set_active(self, active: bool):
        if self.active != active:
            self.active = active
            self.redraw()

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        color = STATUS_COLORS[self.status]
        if self.active:
            rounded_rect(self, 2, 2, w - 2, h - 2, 10, fill=STATUS_TINTS[self.status], outline=color, width=2)
            font, fg = (FONT, 15, "bold"), (TEXT_DARK if self.status != "위험" else C_DANGER)
        else:
            rounded_rect(self, 2, 2, w - 2, h - 2, 10, fill=LIGHT_SUB, outline=LIGHT_SUB)
            font, fg = (FONT, 12), TEXT_MUTED
        f = tkfont.Font(font=font)
        r = 13 if self.active else 11
        total = 2 * r + 8 + f.measure(self.status)
        x = (w - total) / 2
        self.create_oval(x, h / 2 - r, x + 2 * r, h / 2 + r, fill=color, outline=color)
        self.create_text(x + r, h / 2, text=self.SYMBOLS[self.status], fill="white",
                         font=(SYMBOL_FONT, 11 if self.active else 9, "bold"))
        self.create_text(x + 2 * r + 8, h / 2, text=self.status, font=font, fill=fg, anchor="w")


class MessageCard(tk.Canvas):
    """경보 메시지 (24pt 굵게, 공간이 부족하면 자동 축소) + 점멸"""

    def __init__(self, parent):
        super().__init__(parent, height=88, bg=parent["bg"], highlightthickness=0, bd=0)
        self.level, self.message, self.blink, self.notice = None, "", False, None
        self.bind("<Configure>", lambda e: self.redraw())

    def set_view(self, level, message, blink, notice=None):
        """notice: 경보 해제 후에도 조건이 계속될 때의 안내 (제목, 설명)"""
        if (level, message, blink, notice) != (self.level, self.message, self.blink, self.notice):
            self.level, self.message, self.blink, self.notice = level, message, blink, notice
            self.redraw()

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if not self.level and self.notice:
            title, sub = self.notice
            rounded_rect(self, 1, 1, w - 1, h - 1, 12, fill="#ECEFF1", outline=C_UNKNOWN, width=2)
            self.create_oval(20, h / 2 - 22, 64, h / 2 + 22, fill="#78909C", outline="#78909C")
            self.create_text(42, h / 2, text="i", fill="white", font=(FONT, 18, "bold"))
            self.create_text(80, h / 2 - 13, text=title, fill=TEXT_DARK, font=(FONT, 15, "bold"), anchor="w")
            self.create_text(80, h / 2 + 16, text=sub, fill="#455A64", font=(FONT, 11), anchor="w",
                             width=max(100, w - 96))
            return
        if not self.level:
            rounded_rect(self, 1, 1, w - 1, h - 1, 12, fill=LIGHT_SUB, outline=LIGHT_BORDER)
            self.create_oval(24, h / 2 - 20, 64, h / 2 + 20, fill=C_NORMAL, outline=C_NORMAL)
            self.create_text(44, h / 2, text="✓", fill="white", font=(SYMBOL_FONT, 16, "bold"))
            self.create_text(80, h / 2, text="진행 중인 경보가 없습니다", fill=TEXT_MUTED, font=(FONT, 15, "bold"), anchor="w")
            return
        color = STATUS_COLORS[self.level]
        if self.blink:  # 점멸: 등급색 ↔ 연한 배경
            fill, title_fg = color, ("white" if self.level == "위험" else TEXT_DARK)
            sub_fg, icon_fill, icon_fg = title_fg, "white", color
        else:
            fill, title_fg, sub_fg = STATUS_TINTS[self.level], TEXT_DARK, "#444444"
            icon_fill, icon_fg = color, "white"
        rounded_rect(self, 1, 1, w - 1, h - 1, 12, fill=fill, outline=color, width=2)
        r = 26
        self.create_oval(20, h / 2 - r, 20 + 2 * r, h / 2 + r, fill=icon_fill, outline=icon_fill)
        self.create_text(20 + r, h / 2, text="!", fill=icon_fg, font=(FONT, 24, "bold"))
        title, _, sub = self.message.partition(", ")
        avail = w - (20 + 2 * r + 18) - 14
        size = 24
        while size > 13 and tkfont.Font(family=FONT, size=size, weight="bold").measure(title) > avail:
            size -= 1
        x = 20 + 2 * r + 18
        self.create_text(x, h / 2 - 13, text=title, fill=title_fg, font=(FONT, size, "bold"), anchor="w")
        self.create_text(x, h / 2 + 22, text=sub, fill=sub_fg, font=(FONT, 12), anchor="w")


def make_card(parent, title, icon, dark=False, action=None):
    """제목줄이 있는 카드 → (외곽 프레임, 본문 프레임, 제목줄 프레임)"""
    bg = DARK_CARD if dark else LIGHT_CARD
    border = DARK_BORDER if dark else LIGHT_BORDER
    fg = TEXT_LIGHT if dark else TEXT_DARK
    card = tk.Frame(parent, bg=bg, highlightthickness=1, highlightbackground=border)
    head = tk.Frame(card, bg=bg)
    head.pack(fill="x", padx=16, pady=(8, 4))
    if ICON_FONT:
        tk.Label(head, text=icon, font=(ICON_FONT, 15), bg=bg, fg=fg).pack(side="left", padx=(0, 10))
    tk.Label(head, text=title, font=F_CARD, bg=bg, fg=fg).pack(side="left")
    if action:
        tk.Button(head, text="전체 보기", command=action, font=F_SMALL, bg=bg, fg=TEXT_MUTED, relief="solid", bd=1,
                  activebackground=LIGHT_SUB, cursor="hand2", padx=8).pack(side="right")
    body = tk.Frame(card, bg=bg)
    body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
    return card, body, head


# ================= 앱 =================
class App:
    def __init__(self, root: tk.Tk):
        global ICON_FONT
        self.root = root
        fams = set(tkfont.families(root))
        ICON_FONT = next((f for f in ("Segoe Fluent Icons", "Segoe MDL2 Assets") if f in fams), None)

        self.q = queue.Queue()
        self.pose_model = self.helmet_model = None
        self.models_ready = False
        self.worker = None
        self.stop_event = threading.Event()
        self.release_event = threading.Event()
        self.persist = {"active_alerts": [], "released_keys": []}  # 진행 중 경보 (해제 전까지 유지)
        self.file_path = None
        self.photo = None
        self.last_frame, self.last_frame_ts = None, ""
        self.blink_on = False
        self.alarm_level, self.alarm_msg, self.alarm_notice = None, "", None
        self.all_logs = []
        self.last_person_count, self.last_person_log = None, 0.0
        try:
            self.overlay_font = ImageFont.truetype("malgun.ttf", 15)
        except Exception:
            self.overlay_font = ImageFont.load_default()

        try:
            self.cfg = load_config()
        except Exception as e:
            messagebox.showerror("설정 오류", f"config.yaml을 읽을 수 없습니다.\n{e}")
            raise SystemExit(1)

        self._build_ui()
        self.logger = AlertLoggerNode(self.cfg, BASE_DIR, self.log)
        self.alarm = AlarmNode(self.cfg, SOUND_DIR, self.log, lambda: self.logger.known_ids())
        self.evaluator = EvaluatorNode(self.cfg)
        self.pose_node = self.helmet_node = None

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self.poll_queue)
        self.root.after(int(float(self.cfg["blink_interval"]) * 1000), self.blink)
        self.tick_clock()
        self.start_model_loading()

    # ================= 화면 구성 =================
    def _build_ui(self):
        r = self.root
        r.title("이상 동작 감지 Agent")
        sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
        w, h = max(1280, min(1600, int(sw * 0.9))), max(720, min(940, int(sh * 0.85)))
        r.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 3)}")
        r.minsize(1280, 720)
        r.configure(bg=APP_BG)
        self._style()

        r.columnconfigure(0, weight=1)
        r.rowconfigure(1, weight=1)
        self._build_header()

        content = tk.Frame(r, bg=APP_BG)
        content.grid(row=1, column=0, sticky="nsew", padx=12, pady=(10, 12))
        content.columnconfigure(0, weight=60, uniform="col")
        content.columnconfigure(1, weight=40, uniform="col")
        content.rowconfigure(0, weight=1)

        # ----- 좌: 실시간 분석 영상 + 컨트롤 바 -----
        left = tk.Frame(content, bg=APP_BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        vcard, vbody, vhead = make_card(left, "실시간 분석 영상", IC_VIDEO, dark=True)
        vcard.grid(row=0, column=0, sticky="nsew")
        self.lbl_src = tk.Label(vhead, text="", bg=DARK_CARD, fg=TEXT_DIM, font=F_SMALL)
        self.lbl_src.pack(side="right")
        self.canvas = tk.Canvas(vbody, bg=VIDEO_BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.render_video())
        self._build_control_bar(left)

        # ----- 우: 감지 로그 + 경보 패널 -----
        right = tk.Frame(content, bg=APP_BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=28, uniform="row")
        right.rowconfigure(1, weight=72, uniform="row")
        self._build_log_card(right)
        self._build_alarm_card(right)

        self.update_source_list()
        self.on_source_change()
        self.set_controls(running=False)
        self.update_status_display(None)
        self.update_alarm_panel({"level": None, "message": ""}, [])

    def _style(self):
        s = ttk.Style(self.root)
        s.theme_use("clam")
        s.configure("Dark.TCombobox", fieldbackground=DARK_SUB, background=DARK_SUB, foreground=TEXT_LIGHT,
                    arrowcolor=TEXT_LIGHT, bordercolor=DARK_BORDER, lightcolor=DARK_SUB, darkcolor=DARK_SUB,
                    padding=6)
        s.map("Dark.TCombobox", fieldbackground=[("readonly", DARK_SUB), ("disabled", "#1B2632")],
              foreground=[("disabled", TEXT_DIM), ("readonly", TEXT_LIGHT)],
              selectbackground=[("readonly", DARK_SUB)], selectforeground=[("readonly", TEXT_LIGHT)])
        self.root.option_add("*TCombobox*Listbox.background", DARK_SUB)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT_LIGHT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", "#3B4B5E")
        self.root.option_add("*TCombobox*Listbox.font", F_LOG)
        s.configure("Vertical.TScrollbar", background="#D5DAE0", troughcolor=LIGHT_CARD, bordercolor=LIGHT_CARD,
                    arrowcolor=TEXT_MUTED, lightcolor="#D5DAE0", darkcolor="#D5DAE0")

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=HEADER_BG, highlightthickness=1, highlightbackground=DARK_BORDER)
        hdr.grid(row=0, column=0, sticky="ew")
        left = tk.Frame(hdr, bg=HEADER_BG)
        left.pack(side="left", padx=16, pady=8)
        if ICON_FONT:
            tk.Label(left, text=IC_SHIELD, font=(ICON_FONT, 20), bg=HEADER_BG, fg="#42A5F5").pack(side="left", padx=(0, 10))
        tk.Label(left, text="이상 동작 감지 Agent", font=F_TITLE, bg=HEADER_BG, fg=TEXT_LIGHT).pack(side="left", padx=(0, 28))

        self.file_menu = tk.Menu(self.root, tearoff=0, font=F_LOG)
        self.file_menu.add_command(label="CSV 저장", command=self.save_csv)
        self.file_menu.add_command(label="설정 다시 읽기", command=self.reload_config)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="종료", command=self.on_close)
        items = [("파일", self.show_file_menu), ("CSV 저장", self.save_csv),
                 ("설정 다시 읽기", self.reload_config), ("종료", self.on_close)]
        for i, (text, cmd) in enumerate(items):
            if i:
                tk.Label(left, text="|", bg=HEADER_BG, fg="#3A4757", font=F_LOG).pack(side="left", padx=4)
            lbl = tk.Label(left, text=text, bg=HEADER_BG, fg="#D7DDE4", font=F_LOG, cursor="hand2", padx=6)
            lbl.pack(side="left")
            lbl.bind("<Button-1>", lambda e, c=cmd, w=lbl: c(w) if c == self.show_file_menu else c())
            lbl.bind("<Enter>", lambda e, w=lbl: w.configure(fg=TEXT_LIGHT))
            lbl.bind("<Leave>", lambda e, w=lbl: w.configure(fg="#D7DDE4"))

        right = tk.Frame(hdr, bg=HEADER_BG)
        right.pack(side="right", padx=16)
        if ICON_FONT:
            tk.Label(right, text=IC_CAMERA, font=(ICON_FONT, 13), bg=HEADER_BG, fg="#D7DDE4").pack(side="left", padx=(0, 6))
        self.lbl_cam = tk.Label(right, text=self.cfg["camera_id"], bg=HEADER_BG, fg="#D7DDE4", font=F_LOG)
        self.lbl_cam.pack(side="left")
        tk.Label(right, text="|", bg=HEADER_BG, fg="#3A4757", font=F_LOG).pack(side="left", padx=12)
        self.lbl_clock = tk.Label(right, text="", bg=HEADER_BG, fg="#D7DDE4", font=F_LOG)
        self.lbl_clock.pack(side="left")
        tk.Label(right, text="|", bg=HEADER_BG, fg="#3A4757", font=F_LOG).pack(side="left", padx=12)
        self.sys_dot = tk.Label(right, text="●", bg=HEADER_BG, fg=C_CAUTION, font=(SYMBOL_FONT, 12))
        self.sys_dot.pack(side="left", padx=(0, 6))
        self.lbl_sys = tk.Label(right, text="모델 로딩 중", bg=HEADER_BG, fg="#D7DDE4", font=F_LOG)
        self.lbl_sys.pack(side="left")

    def _build_control_bar(self, parent):
        bar = tk.Frame(parent, bg=DARK_CARD, highlightthickness=1, highlightbackground=DARK_BORDER)
        bar.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        inner = tk.Frame(bar, bg=DARK_CARD)
        inner.pack(fill="x", padx=14, pady=10)
        self.lbl_fps = tk.Label(inner, text="FPS  -", bg=DARK_CARD, fg=TEXT_LIGHT, font=(FONT, 12, "bold"), width=8, anchor="e")
        self.lbl_fps.pack(side="right")
        tk.Frame(inner, bg=DARK_BORDER, width=1, height=24).pack(side="right", padx=8)
        self.lbl_conn = tk.Label(inner, text="대기", bg=DARK_CARD, fg=TEXT_LIGHT, font=F_LOG)
        self.lbl_conn.pack(side="right")
        self.conn_dot = tk.Label(inner, text="●", bg=DARK_CARD, fg=C_UNKNOWN, font=(SYMBOL_FONT, 12))
        self.conn_dot.pack(side="right", padx=(0, 6))
        tk.Frame(inner, bg=DARK_BORDER, width=1, height=24).pack(side="right", padx=8)

        tk.Label(inner, text="소스 선택", bg=DARK_CARD, fg=TEXT_LIGHT, font=F_LOG).pack(side="left", padx=(0, 8))
        self.src_var = tk.StringVar(value=SRC_WEBCAM)
        self.src_combo = ttk.Combobox(inner, textvariable=self.src_var, state="readonly", width=12, font=F_LOG,
                                      style="Dark.TCombobox")
        self.src_combo.pack(side="left", padx=(0, 10))
        self.src_combo.bind("<<ComboboxSelected>>", self.on_source_change)
        self.btn_open = RoundButton(inner, "파일 열기", self.open_file, icon=IC_FOLDER, bg="#243244", hover="#2E3F55",
                                    outline="#3A4B60", width=104, font=(FONT, 11, "bold"))
        self.btn_open.pack(side="left", padx=4)
        self.btn_start = RoundButton(inner, "감시 시작", self.start_monitoring, icon=IC_PLAY, bg=BTN_START,
                                     hover=BTN_START_HOVER, width=104, font=(FONT, 11, "bold"))
        self.btn_start.pack(side="left", padx=4)
        self.btn_stop = RoundButton(inner, "감시 중지", self.stop_monitoring, icon=IC_STOP, bg=BTN_BG,
                                    hover=BTN_HOVER, width=104, font=(FONT, 11, "bold"))
        self.btn_stop.pack(side="left", padx=4)


    def _build_log_card(self, parent):
        card, body, _ = make_card(parent, "감지 로그", IC_DOC, action=self.show_all_logs)
        card.grid(row=0, column=0, sticky="nsew", padx=6, pady=(0, 0))
        box = tk.Frame(body, bg=LIGHT_CARD, highlightthickness=1, highlightbackground=LIGHT_BORDER)
        box.pack(fill="both", expand=True)
        self.txt_log = tk.Text(box, bg=LIGHT_CARD, fg=TEXT_DARK, font=F_LOG, wrap="word", state="disabled",
                               relief="flat", padx=10, pady=6, spacing1=3, spacing3=3, cursor="arrow")
        sb = ttk.Scrollbar(box, command=self.txt_log.yview, style="Vertical.TScrollbar")
        sb.pack(side="right", fill="y")
        self.txt_log.pack(side="left", fill="both", expand=True)
        self.txt_log.configure(yscrollcommand=sb.set)
        self.txt_log.tag_configure("time", foreground=TEXT_MUTED)
        self.txt_log.tag_configure("error", foreground=C_DANGER)
        for name, color in (("dot_ok", C_NORMAL), ("dot_warn", C_CAUTION), ("dot_err", C_DANGER)):
            self.txt_log.tag_configure(name, foreground=color, font=(SYMBOL_FONT, 10))

    def _build_alarm_card(self, parent):
        # 점멸 테두리(10px): 외곽 프레임 배경을 등급색 ↔ 앱 배경으로 전환
        self.alarm_border = tk.Frame(parent, bg=APP_BG, padx=6, pady=6)
        self.alarm_border.grid(row=1, column=0, sticky="nsew", pady=(0, 0))
        card, body, _ = make_card(self.alarm_border, "경보 패널", IC_BELL)
        card.pack(fill="both", expand=True)
        self.alarm_card = card

        tk.Label(body, text="현재 상태", bg=LIGHT_CARD, fg=TEXT_DARK, font=(FONT, 12, "bold")).pack(anchor="w", padx=4)
        chips = tk.Frame(body, bg=LIGHT_CARD)
        chips.pack(fill="x", pady=(2, 6))
        self.chips = {}
        for i, st in enumerate(("정상", "주의", "위험", "판정불가")):
            chips.columnconfigure(i, weight=1, uniform="chip")
            chip = StatusChip(chips, st)
            chip.grid(row=0, column=i, sticky="ew", padx=3)
            self.chips[st] = chip

        self.msg_card = MessageCard(body)
        self.msg_card.pack(fill="x", pady=(0, 6))

        self.btn_release = RoundButton(body, "경보 해제", self.release_alerts, icon=IC_BELL, bg=C_DANGER,
                                       hover="#C62828", height=46, font=(FONT, 16, "bold"), radius=10,
                                       disabled_bg=BTN_OFF, disabled_fg="#D0D0D0")
        self.btn_release.pack(side="bottom", fill="x", pady=(6, 0))

        thead = tk.Frame(body, bg=LIGHT_CARD)
        thead.pack(fill="x")
        tk.Label(thead, text="진행 중 경보 목록", bg=LIGHT_CARD, fg=TEXT_DARK, font=(FONT, 12, "bold")).pack(side="left", padx=4)
        tk.Button(thead, text="전체 보기", command=self.show_all_alerts, font=F_SMALL, bg=LIGHT_CARD, fg=TEXT_MUTED,
                  relief="solid", bd=1, cursor="hand2", padx=8).pack(side="right")
        table = tk.Frame(body, bg=LIGHT_CARD, highlightthickness=1, highlightbackground=LIGHT_BORDER)
        table.pack(fill="both", expand=True, pady=(4, 0))
        for c, wgt in enumerate((3, 1, 3, 2)):
            table.columnconfigure(c, weight=wgt, uniform="tcol")
        for c, text in enumerate(("발생 시간", "ID", "감지 내용", "상태")):
            tk.Label(table, text=text, bg=LIGHT_SUB, fg=TEXT_MUTED, font=(FONT, 11), anchor="w", padx=12,
                     pady=2).grid(row=0, column=c, sticky="ew")
        self.alert_rows = []
        for i in range(int(self.cfg["alert_panel_max"])):
            cells = []
            for c in range(4):
                if c == 3:
                    holder = tk.Frame(table, bg=LIGHT_CARD)
                    holder.grid(row=i * 2 + 1, column=c, sticky="ew", padx=12, pady=1)
                    badge = tk.Label(holder, text="", bg=LIGHT_CARD, fg="white", font=(FONT, 10, "bold"), padx=10)
                    badge.pack(side="left")
                    cells.append(badge)
                else:
                    lbl = tk.Label(table, text="", bg=LIGHT_CARD, fg=TEXT_DARK, font=(FONT, 11), anchor="w", padx=12)
                    lbl.grid(row=i * 2 + 1, column=c, sticky="ew", pady=1)
                    cells.append(lbl)
            tk.Frame(table, bg=LIGHT_BORDER, height=1).grid(row=i * 2 + 2, column=0, columnspan=4, sticky="ew")
            self.alert_rows.append(cells)

    # ================= 헤더 =================
    def show_file_menu(self, widget):
        self.file_menu.tk_popup(widget.winfo_rootx(), widget.winfo_rooty() + widget.winfo_height())

    def tick_clock(self):
        self.lbl_clock.configure(text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.root.after(1000, self.tick_clock)

    def set_system_status(self, text, color):
        self.lbl_sys.configure(text=text)
        self.sys_dot.configure(fg=color)

    # ================= 로그 =================
    def log(self, msg: str, error: bool = False):
        """모든 스레드에서 호출 가능 (Queue 경유로 화면에 표시)"""
        self.q.put(("log", f"[{datetime.now():%H:%M:%S}] {msg}", error))

    @staticmethod
    def _log_kind(msg: str, error: bool) -> str:
        if error or msg.startswith("경보 발생"):
            return "err"
        if msg.startswith(("이상 징후", "판정불가", "처리 지연", "영상 연결 중", "모델 로딩 중")):
            return "warn"
        return "ok"

    def append_log(self, line: str, error: bool = False):
        m = TIME_RE.match(line)
        stamp, msg = (m.group(1), m.group(2)) if m else (datetime.now().strftime("%H:%M:%S"), line)
        kind = self._log_kind(msg, error)
        self.all_logs.append((stamp, msg, kind))
        del self.all_logs[:-2000]
        t = self.txt_log
        t.configure(state="normal")
        t.insert("end", f"[{stamp}]  ", ("time",))
        t.insert("end", "● ", (f"dot_{kind}",))
        t.insert("end", msg + "\n", ("error",) if kind == "err" else ())
        excess = int(t.index("end-1c").split(".")[0]) - 1 - int(self.cfg["log_max_lines"])
        if excess > 0:
            t.delete("1.0", f"{excess + 1}.0")
        t.configure(state="disabled")
        t.see("end")
        try:
            print(f"[{stamp}] {msg}", flush=True)
        except Exception:
            pass

    def show_all_logs(self):
        win = tk.Toplevel(self.root)
        win.title("감지 로그 전체")
        win.geometry("760x520")
        txt = tk.Text(win, font=F_LOG, wrap="word", padx=10, pady=6)
        sb = ttk.Scrollbar(win, command=txt.yview)
        sb.pack(side="right", fill="y")
        txt.pack(fill="both", expand=True)
        txt.configure(yscrollcommand=sb.set)
        txt.tag_configure("err", foreground=C_DANGER)
        for stamp, msg, kind in self.all_logs:
            txt.insert("end", f"[{stamp}]  {msg}\n", ("err",) if kind == "err" else ())
        txt.configure(state="disabled")
        txt.see("end")

    def show_all_alerts(self):
        win = tk.Toplevel(self.root)
        win.title("경보 이력 전체 (alert_log.csv)")
        win.geometry("980x420")
        tree = ttk.Treeview(win, columns=COLUMNS, show="headings")
        for c in COLUMNS:
            tree.heading(c, text=c)
            tree.column(c, width=150 if "시각" in c else 95, anchor="center")
        for row in reversed(list(self.logger.rows.values())):
            tree.insert("", "end", values=[row[c] for c in COLUMNS])
        sb = ttk.Scrollbar(win, command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        tree.pack(fill="both", expand=True)

    # ================= 모델 로딩 =================
    def start_model_loading(self):
        self.models_ready = False
        self.set_system_status("모델 로딩 중", C_CAUTION)
        self.set_controls(running=False)
        threading.Thread(target=self._load_models, daemon=True).start()

    def _load_models(self):
        self.log("모델 로딩 중...")
        try:
            pose, helmet = load_models(self.cfg, self.log)
            self.log(f"모델 로딩 완료 ({self.cfg['pose_model']}, {self.cfg['helmet_model']})")
            helmet_node = HelmetDetectorNode(self.cfg, helmet, self.log)  # 클래스명 로그 출력·매핑
            dummy = np.zeros((360, int(self.cfg["resize_width"]), 3), dtype=np.uint8)
            pose.predict(dummy, imgsz=int(self.cfg["pose_imgsz"]), verbose=False)  # 첫 프레임 지연 방지용 예열
            helmet.predict(dummy, imgsz=int(self.cfg["helmet_imgsz"]), verbose=False)
            try:
                import lap  # noqa: F401  (ByteTrack 의존 패키지)
            except ImportError:
                self.log("ByteTrack 의존 패키지(lap)가 없습니다. 터미널에서 pip install lap 실행 후 다시 시작하세요", error=True)
            self.q.put(("models", (pose, helmet, helmet_node)))
        except FileNotFoundError as e:
            self.log(f"모델 파일을 찾을 수 없습니다: {e.args[0] if e.args else e}", error=True)
            self.q.put(("models", None))
        except Exception as e:
            self.log(f"모델 로딩 실패: {e}", error=True)
            self.q.put(("models", None))

    # ================= 컨트롤 =================
    def update_source_list(self):
        has_rtsp = bool(os.getenv("RTSP_URL", "").strip())
        self.src_combo["values"] = [SRC_RTSP if has_rtsp else SRC_RTSP_OFF, SRC_WEBCAM, SRC_FILE]
        if self.src_var.get() in (SRC_RTSP, SRC_RTSP_OFF):
            self.src_var.set(SRC_RTSP if has_rtsp else SRC_WEBCAM)

    def on_source_change(self, _=None):
        src = self.src_var.get()
        if src == SRC_RTSP_OFF:
            self.log("RTSP_URL이 .env에 설정되지 않아 RTSP를 사용할 수 없습니다", error=True)
            self.src_var.set(SRC_WEBCAM)
            src = SRC_WEBCAM
        self.btn_open.set_enabled(src == SRC_FILE and not self.running)
        if src == SRC_RTSP:
            self.lbl_src.configure(text=mask_rtsp(os.getenv("RTSP_URL", "")))
        elif src == SRC_FILE:
            self.lbl_src.configure(text=os.path.basename(self.file_path) if self.file_path else "파일을 선택하세요")
        else:
            self.lbl_src.configure(text="웹캠 0")

    def open_file(self):
        path = filedialog.askopenfilename(title="영상 파일 선택", initialdir=os.path.join(BASE_DIR, "samples"),
                                          filetypes=[("MP4 영상", "*.mp4"), ("모든 파일", "*.*")])
        if path:
            self.file_path = path
            self.on_source_change()

    @property
    def running(self):
        return self.worker is not None and self.worker.is_alive()

    def set_controls(self, running: bool):
        self.btn_start.set_enabled(self.models_ready and not running)
        self.btn_stop.set_enabled(running)
        self.src_combo.configure(state="disabled" if running else "readonly")
        self.btn_open.set_enabled(not running and self.src_var.get() == SRC_FILE)

    def set_conn(self, text, color):
        self.lbl_conn.configure(text=text)
        self.conn_dot.configure(fg=color)

    def start_monitoring(self):
        if self.running or not self.models_ready:
            return
        src = self.src_var.get()
        if src == SRC_FILE:
            if not self.file_path or not os.path.isfile(self.file_path):
                self.log("영상 파일을 먼저 선택하세요 ([파일 열기])", error=True)
                return
            kind, path = "file", self.file_path
        elif src == SRC_RTSP:
            kind, path = "rtsp", os.getenv("RTSP_URL", "").strip()
        else:
            kind, path = "webcam", 0
        self.stop_event.clear()
        self.last_person_count = None
        self.worker = threading.Thread(target=self.analysis_loop, args=(kind, path), daemon=True)
        self.worker.start()
        self.set_controls(running=True)
        self.set_conn("연결 중", C_CAUTION)

    def stop_monitoring(self):
        self.stop_event.set()
        self.btn_stop.set_enabled(False)

    # ================= 분석 스레드 =================
    def analysis_loop(self, kind, path):
        reason = "stopped"
        video = VideoInputNode(self.cfg, self.log, self.stop_event)
        try:
            self.pose_node.reset_tracker()
            graph = build_graph(video, self.pose_node, self.helmet_node, self.evaluator, self.alarm, self.logger)
            if not video.connect(kind, path):
                reason = "failed"
                return
            self.q.put(("conn", True))
            mem = {"durations": {}, "log_marks": {}}  # 작업자별 지속시간 (감시 중지 시 초기화)
            interval = 1.0 / float(self.cfg["target_fps"])
            next_t = time.time()
            last_err = ["", 0.0]
            while not self.stop_event.is_set():
                release = self.release_event.is_set()
                if release:
                    self.release_event.clear()
                inp = {**mem, "active_alerts": self.persist["active_alerts"],
                       "released_keys": self.persist["released_keys"], "release_requested": release}
                try:
                    out = graph.invoke(inp)
                except Exception as e:  # 예상하지 못한 오류: 로그 후 다음 프레임 계속
                    msg = f"처리 오류: {type(e).__name__}: {e}"
                    if msg != last_err[0] or time.time() - last_err[1] >= 5:  # 같은 오류는 5초에 1번만 표시
                        last_err[:] = [msg, time.time()]
                        self.log(msg, error=True)
                        traceback.print_exc()
                    if release:
                        self.release_event.set()
                    out = None
                if out is not None:
                    status = out.get("source_status")
                    mem["durations"] = out.get("durations", mem["durations"])
                    mem["log_marks"] = out.get("log_marks", mem["log_marks"])
                    self.persist["active_alerts"] = out.get("active_alerts", self.persist["active_alerts"])
                    self.persist["released_keys"] = out.get("released_keys", self.persist["released_keys"])
                    if status == "ended":
                        self.log("영상 재생 완료")
                        reason = "ended"
                        break
                    if status == "failed":
                        reason = "failed"
                        break
                    connected = status == "connected"
                    self.q.put(("result", {
                        "status": status,
                        "frame": out.get("annotated_frame") if connected else None,
                        "ts": (out.get("timestamp_str") or "")[:19],
                        "fps": out.get("fps"),
                        "persons": len(out.get("persons") or []) if connected else None,
                        "overall": out.get("overall_status") if connected else None,
                        "logs": out.get("detect_logs", []) if (connected or release) else [],
                        "alarm_view": out.get("alarm_view"),
                        "alerts": list(self.persist["active_alerts"]),
                        "released": list(self.persist["released_keys"]),
                    }))
                    if status == "disconnected":
                        mem = {"durations": {}, "log_marks": {}}
                        next_t = time.time()
                next_t += interval
                delay = next_t - time.time()
                if delay > 0:
                    self.stop_event.wait(delay)
                else:
                    next_t = time.time()  # 지연 시 밀린 처리를 버리고 최신 프레임 우선
        except Exception as e:
            self.log(f"분석 스레드 오류: {e}", error=True)
            traceback.print_exc()
            reason = "failed"
        finally:
            video.release()
            self.alarm.stop_sound()
            self.q.put(("stopped", reason))

    # ================= 경보 해제 =================
    def release_alerts(self):
        if not self.persist["active_alerts"]:
            return
        if self.running:
            self.release_event.set()
        else:
            self._release_offline()

    def _release_offline(self):
        """감시 중지 상태에서도 진행 중 경보를 해제·기록 (alarm → alert_logger)"""
        self.release_event.clear()
        out = self.alarm({"release_requested": True, "timestamp": time.time(),
                          "active_alerts": self.persist["active_alerts"], "released_keys": self.persist["released_keys"]})
        self.logger(out)
        self.persist["active_alerts"] = out["active_alerts"]
        self.persist["released_keys"] = []
        for line in out.get("detect_logs", []):
            self.append_log(line)
        self.update_alarm_panel(out["alarm_view"], out["active_alerts"])

    # ================= 화면 갱신 (0.1초) =================
    def poll_queue(self):
        latest = None
        try:
            while True:
                kind, *data = self.q.get_nowait()
                if kind == "log":
                    self.append_log(data[0], data[1])
                elif kind == "result":
                    res = data[0]
                    for line in res["logs"]:
                        self.append_log(line)
                    latest = res
                elif kind == "conn":
                    self.set_conn("연결됨", C_NORMAL)
                elif kind == "models":
                    self.on_models_loaded(data[0])
                elif kind == "stopped":
                    self.on_stopped(data[0])
                    latest = None
        except queue.Empty:
            pass
        if latest is not None:
            self.apply_result(latest)
        self.root.after(100, self.poll_queue)

    def on_models_loaded(self, models):
        if models is None:
            self.models_ready = False
            self.set_system_status("모델 오류 - 감시 불가", C_DANGER)
        else:
            self.pose_model, self.helmet_model, self.helmet_node = models
            self.pose_node = PoseDetectorNode(self.cfg, self.pose_model)
            self.models_ready = True
            self.set_system_status("시스템 정상 가동 중", C_NORMAL)
        self.set_controls(running=self.running)

    def apply_result(self, res):
        if res["status"] == "connected":
            self.set_conn("연결됨", C_NORMAL)
            self.last_frame, self.last_frame_ts = res["frame"], res["ts"]
            self.render_video()
            fps = res["fps"] or 0.0
            ok = fps >= float(self.cfg["target_fps"]) * 0.95
            self.lbl_fps.configure(text=f"FPS  {fps:.1f}", fg=TEXT_LIGHT if ok else C_CAUTION)
            self.update_status_display(res["overall"] or "판정불가")
            n = res["persons"]
            now = time.time()
            if n != self.last_person_count and (now - self.last_person_log >= 2 or self.last_person_count is None):
                self.last_person_count, self.last_person_log = n, now
                self.append_log(f"[{datetime.now():%H:%M:%S}] " + (f"사람 {n}명 감지 중" if n else "사람 미검출"))
        else:
            self.set_conn("연결 끊김", C_DANGER)
        if res["alarm_view"] is not None:
            self.update_alarm_panel(res["alarm_view"], res["alerts"], res["released"])

    def on_stopped(self, reason):
        self.worker = None
        self.set_controls(running=False)
        if reason == "failed":
            self.set_conn("연결 끊김", C_DANGER)
        else:
            self.set_conn("대기", C_UNKNOWN)
        self.lbl_fps.configure(text="FPS  -", fg=TEXT_LIGHT)
        self.update_status_display(None)
        self.last_frame = None
        self.render_video()
        self.log("감시 중지" if reason != "failed" else "감시 중지 (영상 연결 실패)", error=reason == "failed")
        if self.release_event.is_set():
            self._release_offline()
        self.persist["released_keys"] = []
        self.update_alarm_panel(self._view_for(self.persist["active_alerts"]), self.persist["active_alerts"])

    @staticmethod
    def _view_for(alerts):
        dangers = [a for a in alerts if a["level"] == "위험"]
        if dangers:
            msg = MESSAGES["위험"] if any(a["event_type"] == "넘어짐" for a in dangers) else MESSAGES["help"]
            return {"level": "위험", "color": COLORS["위험"], "message": msg}
        if alerts:
            return {"level": "주의", "color": COLORS["주의"], "message": MESSAGES["주의"]}
        return {"level": None, "color": None, "message": ""}

    def update_status_display(self, overall):
        for st, chip in self.chips.items():
            chip.set_active(st == overall)

    @staticmethod
    def _released_notice(released):
        """해제했지만 조건이 계속되는 경보 → (제목, 설명). 관리자가 현장 조치 완료를 확인할 수 있게 안내"""
        if not released:
            return None
        texts = {"안전모 미착용": "안전모 미착용", "넘어짐": "쓰러진 자세", "구조요청": "구조요청 자세"}
        by_event = {}
        for tid, ev in released:
            by_event.setdefault(ev, []).append(str(tid))
        parts = [f"{texts.get(ev, ev)} (ID {', '.join(ids)})" for ev, ids in by_event.items()]
        return "경보 해제됨 - 조치 확인 중", f"{' · '.join(parts)} 상태가 계속되고 있습니다."

    def update_alarm_panel(self, view, alerts, released=None):
        self.alarm_level = view.get("level")
        self.alarm_msg = view.get("message", "")
        self.alarm_notice = self._released_notice(released)
        for i, cells in enumerate(self.alert_rows):
            t_lbl, id_lbl, ev_lbl, badge = cells
            if i < len(alerts):
                a = alerts[i]
                t_lbl.configure(text=a["start_str"], fg=TEXT_DARK)
                id_lbl.configure(text=str(a["track_id"]))
                ev_lbl.configure(text=a["event_type"])
                color = STATUS_COLORS.get(a["level"], C_UNKNOWN)
                badge.configure(text=a["level"], bg=color, fg="white" if a["level"] == "위험" else TEXT_DARK)
            else:
                empty = i == 0 and not alerts
                t_lbl.configure(text="진행 중인 경보 없음" if empty else "", fg=TEXT_MUTED)
                id_lbl.configure(text="")
                ev_lbl.configure(text="")
                badge.configure(text="", bg=LIGHT_CARD)
        self.btn_release.set_enabled(bool(alerts))
        self.paint_blink()

    # ================= 점멸 (0.5초) =================
    def blink(self):
        self.blink_on = not self.blink_on
        self.paint_blink()
        self.root.after(int(float(self.cfg["blink_interval"]) * 1000), self.blink)

    def paint_blink(self):
        color = STATUS_COLORS.get(self.alarm_level)
        on = bool(self.alarm_level) and self.blink_on
        self.alarm_border.configure(bg=color if on else APP_BG)
        self.msg_card.set_view(self.alarm_level, self.alarm_msg, on, self.alarm_notice)

    # ================= 영상 표시 =================
    def render_video(self):
        c = self.canvas
        cw, ch = max(1, c.winfo_width()), max(1, c.winfo_height())
        c.delete("all")
        if self.last_frame is None:
            c.create_text(cw // 2, ch // 2, text="감시 시작 버튼을 눌러주세요", fill=C_UNKNOWN, font=(FONT, 18))
            return
        h, w = self.last_frame.shape[:2]
        scale = min(cw / w, ch / h)  # 창 크기 변경 시 영상 비율 유지
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        rgb = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR)
        # 좌상단 시각·카메라 번호 오버레이
        text = f"{self.last_frame_ts}   ({self.cfg['camera_id']})"
        draw = ImageDraw.Draw(img, "RGBA")
        x1, y1, x2, y2 = draw.textbbox((0, 0), text, font=self.overlay_font)
        draw.rounded_rectangle((12, 12, 12 + (x2 - x1) + 20, 12 + (y2 - y1) + 16), radius=6, fill=(0, 0, 0, 170))
        draw.text((22, 20 - y1), text, font=self.overlay_font, fill=(255, 255, 255, 255))
        self.photo = ImageTk.PhotoImage(img)
        c.create_image(cw // 2, ch // 2, image=self.photo, anchor="center")

    # ================= 메뉴 =================
    def save_csv(self):
        dest = filedialog.asksaveasfilename(title="경보 이력 저장", defaultextension=".csv",
                                            initialfile=os.path.basename(self.logger.path),
                                            filetypes=[("CSV 파일", "*.csv")])
        if not dest:
            return
        try:
            self.logger.export(dest)
            self.log(f"CSV 저장 완료: {os.path.basename(dest)}")
        except Exception:
            self.log("이력 저장 실패: 파일이 열려 있는지 확인하세요", error=True)

    def reload_config(self):
        if self.running:
            messagebox.showinfo("설정 다시 읽기", "감시 중지 상태에서만 설정을 다시 읽을 수 있습니다.")
            return
        try:
            new = load_config()
        except Exception as e:
            self.log(f"설정 읽기 실패: {e}", error=True)
            return
        old = dict(self.cfg)
        self.cfg.clear()
        self.cfg.update(new)  # 모든 노드가 같은 cfg 객체를 참조
        if old["log_file"] != new["log_file"]:
            self.logger = AlertLoggerNode(self.cfg, BASE_DIR, self.log)
        self.alarm.stop_sound()
        self.alarm = AlarmNode(self.cfg, SOUND_DIR, self.log, lambda: self.logger.known_ids())
        self.lbl_cam.configure(text=self.cfg["camera_id"])
        self.update_source_list()
        self.on_source_change()
        self.log("설정 다시 읽기 완료 (config.yaml, .env)")
        if (old["pose_model"], old["helmet_model"]) != (new["pose_model"], new["helmet_model"]) or not self.models_ready:
            self.start_model_loading()

    def on_close(self):
        self.stop_event.set()
        if self.worker is not None:
            self.worker.join(timeout=3)
        self.alarm.stop_sound()
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
