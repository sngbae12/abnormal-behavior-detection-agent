# 1. 개요
- 사무용 PC에서 CCTV(RTSP)·웹캠·mp4 영상을 초당 10프레임으로 분석하여 작업자의 이상 동작(넘어짐·구조요청)과 안전모 미착용을 감지하고, 정상/주의/위험 상태와 주의·위험 2단계 경보(화면 점멸 + 벨/사이렌)를 제공하는 데스크톱 감시 Agent 구현
- 판단은 LLM 없이 키포인트·신뢰도 수치 규칙으로 처리
- AI는 감지·등급 산정·경보·이력 기록까지 수행, 감독자는 경보 확인·현장 대응·경보 해제 수행
- 모든 판정 기준값은 config.yaml에서 읽어서 사용 (코드에 하드코딩 금지)
- 노드 실행 순서 (프레임마다 1회): video_input → pose_detector → helmet_detector → evaluator → alarm → alert_logger
- 노드 간 데이터는 LangGraph State(dict)로 전달, 작업자별 지속시간·경보 상태는 프레임 간 유지

## config.yaml 기본값
```yaml
camera_id: "CAM-01"
target_fps: 10
resize_width: 640
pose_model: "yolov8n-pose.pt"
helmet_model: "helmet.pt"
person_conf: 0.5
keypoint_conf: 0.5
no_helmet_conf: 0.5
no_helmet_seconds: 15        # 주의 판정 지속시간
danger_seconds: 3            # 위험 판정 지속시간
fall_torso_angle: 60         # 몸통 기울기(도)
fall_box_ratio: 1.2          # person 박스 가로/세로
blink_interval: 0.5          # 점멸 주기(초)
reconnect_interval: 5
reconnect_max: 3
log_max_lines: 100
alert_panel_max: 5
bell_sound: "bell.wav"
siren_sound: "siren.wav"
log_file: "alert_log.csv"
```

# 2. 노드구성

## 1) video_input
### 역할
- 선택한 영상 소스에서 프레임을 초당 10장 수집하여 다음 노드로 전달
### 구현 요구사항
- 입력 소스 3종 지원: RTSP URL(.env의 RTSP_URL), 웹캠(장치 번호 0), mp4 파일 1개 선택
- OpenCV VideoCapture 사용, 프레임 가로 640px로 축소(비율 유지)
- 0.1초 간격으로 프레임 처리, 처리 지연 시 쌓인 프레임은 버리고 최신 프레임 1장만 사용
- mp4는 실제 재생 속도에 맞춰 진행 (영상 시각 = 재생 위치)
- 프레임마다 frame_id, 시스템 시각(timestamp), 영상 시각(video_time) 부여
- 최근 10프레임 처리 시간으로 실제 FPS 계산
- 연결 실패 시 5초 간격 재연결 최대 3회
### 입력
- 소스 종류(rtsp / webcam / file), 소스 경로
### 출력
- frame(이미지), frame_id, timestamp, video_time, fps, source_status
### 처리흐름
- 소스 연결 → 프레임 읽기 → 640px 축소 → 시각·ID 부여 → FPS 계산 → State 전달
### 입력 예시 / 출력 예시
- 입력: {"source": "webcam", "path": 0}
- 출력: {"frame_id": 152, "timestamp": "2026-10-08 15:30:12.300", "video_time": 15.2, "fps": 9.8, "source_status": "connected"}
### 제약 조건
- 영상 원본 파일 저장 금지
- RTSP 계정·비밀번호를 화면·로그에 출력 금지 (URL은 "rtsp://***@IP" 형태로 마스킹)
- 프레임을 못 읽으면 다음 노드 실행 없이 source_status="disconnected" 반환

## 2) pose_detector
### 역할
- yolov8n-pose로 사람별 17개 키포인트와 추적 ID 추출
### 구현 요구사항
- Ultralytics `model.track(frame, persist=True, tracker="bytetrack.yaml")` 사용
- person 신뢰도 0.5 이상만 사용
- 사람마다 track_id, bbox(x1,y1,x2,y2), 17개 키포인트 (x, y, conf) 저장
- 키포인트 순서는 COCO 기준 (0 코, 5·6 어깨, 9·10 손목, 11·12 골반 사용)
- 프레임에 person 박스(초록), 스켈레톤, track_id 표시
### 입력
- frame
### 출력
- persons: [{track_id, bbox, person_conf, keypoints[17]}], annotated_frame
### 처리흐름
- frame → pose 추론·추적 → 신뢰도 필터 → persons 리스트 생성 → 박스·스켈레톤 그리기
### 입력 예시 / 출력 예시
- 출력: {"persons": [{"track_id": 3, "bbox": [120, 80, 260, 420], "person_conf": 0.86, "keypoints": [[190, 110, 0.92], ...]}]}
### 제약 조건
- 신뢰도가 낮은 키포인트 좌표를 임의로 보정·생성 금지
- 사람 미검출 시 persons=[] 반환 (오류 아님)

## 3) helmet_detector
### 역할
- helmet.pt로 안전모 착용/미착용을 판별하고 사람별로 매칭
### 구현 요구사항
- 프로그램 시작 시 helmet.pt의 클래스명(model.names)을 로그에 출력하고, Helmet / No Helmet 클래스를 확인
- 같은 프레임에 helmet.pt 추론 적용
- 머리 영역 = 코·눈·귀 키포인트(0~4) 중 신뢰도 0.5 이상 점들을 감싸는 박스 (없으면 person 박스 상단 25%)
- helmet 박스와 머리 영역이 겹치는(IoU > 0) 사람에게 결과 매칭, 여러 개면 신뢰도 가장 높은 것 선택
- helmet_status: "helmet" / "no_helmet"(No Helmet 신뢰도 0.5 이상) / "unknown"(매칭 결과 없음)
- 박스 표시: Helmet 초록, No Helmet 빨강 + 신뢰도
### 입력
- frame, persons
### 출력
- persons에 helmet_status, helmet_conf 추가, annotated_frame
### 처리흐름
- frame → helmet 추론 → 사람별 머리 영역 계산 → 매칭 → helmet_status 기록 → 박스 그리기
### 입력 예시 / 출력 예시
- 출력: {"track_id": 3, "helmet_status": "no_helmet", "helmet_conf": 0.71}
### 제약 조건
- 안전모가 검출되지 않았다는 이유만으로 no_helmet 판정 금지 → "unknown" 처리
- helmet.pt 재학습·수정 금지

## 4) evaluator
### 역할
- 작업자별 동작·안전모 상태와 지속시간을 계산하여 정상/주의/위험/판정불가 결정
### 구현 요구사항
- 넘어짐: 몸통 기울기(양어깨 중심 → 양골반 중심 선이 수직선과 이루는 각) ≥ 60° AND bbox 가로/세로 ≥ 1.2
- 구조요청: 양 손목 y < 코 y (화면 좌표 기준 손목이 코보다 위)
- 판정에 필요한 키포인트(어깨·골반·손목·코) 신뢰도 0.5 미만이면 해당 동작은 판정불가
- 지속시간은 프레임 수가 아니라 timestamp 차이로 계산
- track_id별로 fall / help / no_helmet 지속시간을 따로 누적
- 조건 해소, track_id 변경·소실, 판정불가 시 해당 지속시간 0으로 초기화
- 상태 결정
  - 위험: fall 또는 help 지속시간 ≥ 3초
  - 주의: no_helmet 지속시간 ≥ 15초
  - 판정불가: 필요한 키포인트 부족 또는 사람 미검출
  - 정상: 그 외
- 이상 자세 감지 중(3초 미만)이면 로그 메시지 생성: "이상 징후 감지 중: 넘어짐 (1.0초 / 3초)" (0.1초 단위)
### 입력
- persons(키포인트, helmet_status), timestamp, 이전 State의 지속시간 기록
### 출력
- person_states: [{track_id, status, event_type, duration}], detect_logs, overall_status
### 처리흐름
- 사람별 자세 계산 → 넘어짐·구조요청 판정 → 안전모 상태 확인 → 지속시간 누적·초기화 → 상태 결정 → 전체 상태(위험 > 주의 > 정상) 산출
### 입력 예시 / 출력 예시
- 입력: track_id 3, 몸통 각도 72°, 박스 비율 1.5, 넘어짐 지속 2.9초 → 다음 프레임 3.0초
- 출력: {"track_id": 3, "status": "위험", "event_type": "넘어짐", "duration": 3.0}
### 제약 조건
- 다른 track_id의 지속시간 합산 금지
- 판정불가를 정상으로 처리 금지

## 5) alarm
### 역할
- 상태에 따라 경보를 생성·유지·해제하고 화면 점멸과 경보음 출력
### 구현 요구사항
- 경보 키 = (track_id, event_type), 같은 키는 경보 1건으로 유지 (매 프레임 새로 생성 금지)
- 신규 경보 생성 시 경보ID(A-YYYYMMDD-001 형식), 발생시각, 등급 기록
- 주의: 노란색(#FBC02D) 점멸 + bell.wav 1회 + 메시지 "주의 : 안전모 미착용자가 있습니다, 신속하게 조치바랍니다."
- 위험: 빨간색(#E53935) 점멸 + siren.wav 해제 전까지 반복 + 메시지 "위험 : 쓰러진 작업자가 있습니다, 즉시 현장 확인바랍니다." (구조요청은 "위험 : 구조요청 자세가 감지되었습니다, 즉시 현장 확인바랍니다.")
- 점멸 주기 0.5초
- 위험·주의 동시 진행 시 점멸색·소리는 위험 기준, 패널에는 진행 중 경보를 발생 시각 순으로 최대 5건 표시
- 감독자 [경보 해제] 클릭 시 진행 중 경보 전체 해제: 점멸·소리 중지, 해제시각 기록
- 재경보 보류 없음: 해제된 경보 키는 해당 작업자가 정상으로 돌아간 뒤 조건이 다시 충족될 때만 새 경보 생성
- 경보음은 winsound(Windows)로 비동기 재생 (SND_ASYNC, 사이렌은 SND_LOOP)
- [선택] Slack: 신규 경보 시 별도 스레드로 메시지 발송, 위험은 경보 시점 프레임 캡처(.jpg) 첨부
### 입력
- person_states, overall_status, 감독자 해제 이벤트
### 출력
- active_alerts, new_alerts, released_alerts, alarm_view(점멸색, 메시지)
### 처리흐름
- 상태 확인 → 신규/기존 경보 구분 → 등급별 점멸·소리 출력 → 해제 이벤트 처리 → 결과 전달
### 입력 예시 / 출력 예시
- 출력: {"alert_id": "A-20261008-003", "track_id": 3, "event_type": "넘어짐", "level": "위험", "start": "15:30:12", "state": "진행중"}
### 제약 조건
- 경보 해제는 감독자 버튼으로만 가능 (자동 해제 금지)
- 해제 후에도 영상 분석은 계속 진행

## 6) alert_logger
### 역할
- 경보 발생·해제 이력을 CSV로 기록
### 구현 요구사항
- 파일: alert_log.csv, UTF-8-BOM, 한글 열 이름
- 열: 경보ID, 발생시각, 카메라번호, 작업자추적ID, 감지유형, 등급, 경보상태, 해제시각, 처리시간(초)
- 신규 경보 시 1행 추가(경보상태 "진행중"), 해제 시 같은 경보ID 행의 경보상태 "해제"·해제시각·처리시간 갱신
- 처리시간 = 해제시각 − 발생시각 (초, 소수 1자리)
- 화면 [CSV 저장] 클릭 시 사용자가 고른 위치로 복사
### 입력
- new_alerts, released_alerts
### 출력
- alert_log.csv
### 처리흐름
- 신규 경보 → 행 추가 / 해제 경보 → 행 갱신 → 파일 저장
### 입력 예시 / 출력 예시
- 출력 행: A-20261008-003, 2026-10-08 15:30:12, CAM-01, 3, 넘어짐, 위험, 해제, 2026-10-08 15:31:05, 53.0
### 제약 조건
- 프레임 단위 기록 금지 (경보 사건 단위로만 기록)
- 파일이 열려 있어 저장 실패 시 메모리에 보관 후 다음 기록 때 재시도

# 3. 실패 처리
- 모델 파일(yolov8n-pose.pt, helmet.pt) 없음: "모델 파일을 찾을 수 없습니다: helmet.pt" 표시, [감시 시작] 비활성화
- helmet.pt에 Helmet / No Helmet 클래스 없음: 클래스 목록을 로그에 출력하고 안전모 판별 중지, 자세 감지는 계속
- 경보음 파일 없음: "경보음 파일 없음: siren.wav" 로그, 화면 점멸·메시지 경보는 계속
- 영상 연결 실패: "영상 연결 실패: 소스를 확인하세요" 표시, 5초 간격 3회 재연결 후 실패 시 감시 중지
- 10 FPS 미달: "처리 지연: 7.2 FPS" 로그 표시, 최신 프레임 우선 처리 유지
- 사람 미검출·키포인트 부족: "판정불가" 상태 표시, 경보 생성 안 함
- CSV 저장 실패: "이력 저장 실패: 파일이 열려 있는지 확인하세요" 로그, 메모리 보관 후 재시도
- [선택] Slack 발송 실패: "Slack 발송 실패" 로그 후 3회 재시도, 화면 경보는 유지 / SLACK 키 미설정 시 발송만 건너뜀
- 예상하지 못한 오류: 오류 메시지를 로그에 출력하고 프로그램 종료 없이 다음 프레임 계속 처리
