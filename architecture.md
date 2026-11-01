# 1. 주요 패키지
- requirements.txt(pip list) 파일의 패키지 버전을 참조하여 코드 생성
- 언어: Python 3.10 이상
- 화면: tkinter (Python 기본), Pillow (프레임 → Tkinter 이미지 변환)
- 영상 처리: opencv-python
- AI 모델: ultralytics (yolov8n-pose.pt, helmet.pt 추론, ByteTrack 추적), torch(ultralytics 의존)
- Agent 흐름: langgraph (StateGraph로 6개 노드 연결)
- 계산: numpy
- 설정: pyyaml (config.yaml), python-dotenv (.env)
- 이력: csv (Python 기본)
- 경보음: winsound (Windows 기본)
- [선택] Slack: requests (Webhook), slack_sdk (캡처 이미지 업로드)
- 사용하지 않음: LLM·OpenAI API, LangChain, ChromaDB (본 과제는 LLM 없이 수치 규칙으로 판단)

# 2. 동작 순서
- 프레임마다 LangGraph 그래프 1회 실행 (초당 10회)
## 1) video_input
## 2) pose_detector
## 3) helmet_detector
## 4) evaluator
## 5) alarm
## 6) alert_logger
- 사람 판단 지점: alarm 이후 감독자 [경보 해제] 클릭 → alarm(해제 처리) → alert_logger(해제 기록)
- 실행 구조
  - 분석 스레드: video_input → … → alert_logger 반복 실행, 결과를 Queue로 화면에 전달
  - 화면 스레드(Tkinter 메인): 0.1초마다 Queue에서 결과를 꺼내 영상·로그·경보 패널 갱신, 0.5초마다 점멸 전환
  - 경보음: winsound 비동기 재생 (화면 멈춤 없음)

## State 구조 (노드 간 전달)
```python
{
  "frame": ndarray, "frame_id": int, "timestamp": float, "video_time": float,
  "fps": float, "source_status": str,
  "persons": [ {"track_id", "bbox", "person_conf", "keypoints", "helmet_status", "helmet_conf"} ],
  "person_states": [ {"track_id", "status", "event_type", "duration"} ],
  "durations": { track_id: {"fall": float, "help": float, "no_helmet": float} },  # 프레임 간 유지
  "active_alerts": [ ... ], "new_alerts": [ ... ], "released_alerts": [ ... ],
  "overall_status": str, "detect_logs": [str], "annotated_frame": ndarray
}
```

# 3. 사용자 흐름
## 1) 프로그램 실행 → 모델 로딩 완료 확인
## 2) 영상 소스 선택 (RTSP / 웹캠 / 영상 파일)
## 3) [감시 시작] 클릭
## 4) 실시간 영상·감지 로그 확인 (감독자는 화면을 계속 보지 않아도 됨)
## 5) 경보 발생 (점멸 + 벨/사이렌) → 영상으로 상황 확인 → 현장 확인·대응
## 6) [경보 해제] 클릭
## 7) [감시 중지] → 메뉴 > CSV 저장

# 4. 데이터베이스 구성
- 데이터베이스 없음
- 경보 이력: alert_log.csv (UTF-8-BOM)
  - 열: 경보ID, 발생시각, 카메라번호, 작업자추적ID, 감지유형, 등급, 경보상태, 해제시각, 처리시간(초)
- 판정 기준값: config.yaml (prompts.md 개요의 기본값 사용)
- 작업자별 지속시간·진행 중 경보: 메모리(State)에만 유지, 감시 중지 시 초기화

## 폴더 구조
```
project/
├─ main.py              # Tkinter 화면 + 분석 스레드 실행
├─ graph.py             # LangGraph StateGraph (6개 노드 연결)
├─ nodes/
│  ├─ video_input.py
│  ├─ pose_detector.py
│  ├─ helmet_detector.py
│  ├─ evaluator.py
│  ├─ alarm.py
│  └─ alert_logger.py
├─ models/ yolov8n-pose.pt, helmet.pt
├─ sounds/ bell.wav, siren.wav
├─ samples/ sample_normal.mp4, sample_fall.mp4, sample_help.mp4, sample_nohelmet.mp4
├─ config.yaml
├─ .env
├─ requirements.txt
├─ prompts.md, ui.md, architecture.md
└─ alert_log.csv        # 실행 시 생성
```

# 5. 외부 도구
- 외부 API 없음 (모든 추론 로컬 실행)
- yolov8n-pose.pt: 최초 실행 시 ultralytics가 자동 다운로드, 이후 models/ 폴더 파일 사용
- helmet.pt: 교육과정 제공 파일, models/ 폴더에 배치
- [선택] Slack Incoming Webhook: 경보 텍스트 발송
- [선택] Slack Bot API files_upload_v2: 위험 경보 캡처 이미지 첨부

# 6. 주요 환경변수
- .env 파일에 저장, 코드·화면·로그에 값 출력 금지
```
RTSP_URL=rtsp://user:password@192.168.0.10:554/stream1
CAMERA_ID=CAM-01
# [선택] Slack
SLACK_WEBHOOK_URL=
SLACK_BOT_TOKEN=
SLACK_CHANNEL_ID=
```
- 값이 비어 있으면: RTSP는 소스 목록에서 비활성화, Slack은 발송만 건너뜀
