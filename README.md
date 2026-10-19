# 서울시 출퇴근 교통혼잡 분석 및 AI 기반 교통정책 시뮬레이션

2026 교통문제 해결 데이터 분석 공모전 프로젝트. 보고서: [`reports/final_report.md`](reports/final_report.md)

## 1. 환경 준비

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
brew install libomp            # macOS: XGBoost·LightGBM 실행에 필요
cp .env.example .env           # 아래 키 입력
```

| `.env` 변수 | 발급처 |
|---|---|
| `SEOUL_OPEN_API_KEY` | 서울 열린데이터광장 인증키 — https://data.seoul.go.kr |
| `DATA_GO_KR_SERVICE_KEY` | 공공데이터포털 — https://www.data.go.kr (기상청 ASOS 시간자료, 한국천문연구원 특일 정보 활용신청) |
| `TRAFFIC_DATA_PATH` | t-data에서 받은 1시간 소통정보 CSV 경로 |

수동 다운로드 파일은 `data/raw/manual/`에 둔다: t-data 1시간 소통정보·스마트교차로(`TOPIS_1H_*.zip`), ITS 표준노드링크(`[날짜]NODELINKDATA.zip`), SGIS 경계(`bnd_all_00_2025_2Q.zip`). 출처·링크는 `docs/data_sources.md`.

## 2. 실행 순서

```bash
PY=.venv/bin/python

# 수집 (src/collect 에서 실행, 이미 받은 파일은 건너뜀)
cd src/collect
$PY collect_transit.py --start 202401 --end 202609
$PY collect_weather.py --start 20240101 --end 20261018
$PY collect_holidays.py --years 2024 2025 2026
$PY collect_locations.py
$PY collect_volume.py --dates 20261014 20261015
$PY collect_volume.py --start 20240101 --end 20251231 --hours 07 08 09 12 17 18 19 --out volume_peak  # 수 시간 소요
$PY collect_link_geometry.py "<소통정보 CSV 경로>"
cd ../..

# zip 풀기: data/raw/ext/{smart_intersection,nodelink_20261014,boundary}/  (docs/preprocessing_log.md 참고)

# 전처리 (src/preprocess 에서 실행)
cd src/preprocess
for s in prep_spatial prep_speed prep_volume prep_weather prep_transit; do $PY $s.py; done
cd ../..

# EDA
$PY src/analysis/eda_traffic.py
$PY src/analysis/eda_transit.py
$PY src/analysis/eda_factors.py          # 교통량-속도 관계
$PY src/analysis/eda_factors.py --long   # 장기 교통량: 요일·강수·월별

# AI 예측
$PY src/models/forecast_volume.py
# (코드 검증용 모의 데이터) $PY src/models/make_mock_volume.py && $PY src/models/forecast_volume.py --mock --src data/mock/volume_peak

# 정책 시뮬레이션 (가정값: config/simulation.yaml)
$PY src/simulation/simulate.py
```

## 3. 폴더 구조

| 경로 | 내용 |
|---|---|
| `config/config.yaml` | 첨두 정의, 혼잡 기준, 좌표계 |
| `config/simulation.yaml` | 시뮬레이션 가정값(낮음/중간/높음) |
| `src/collect/` | API 수집 |
| `src/preprocess/` | 전처리 — 규칙은 `docs/preprocessing_log.md` |
| `src/analysis/` | EDA — 요약은 `docs/eda_findings.md` |
| `src/models/` | 첨두 교통량 1시간 전 예측 |
| `src/simulation/` | 시나리오 A~E 시뮬레이션 |
| `src/visualization/style.py` | 그림 공통 스타일·저장 규칙 |
| `outputs/figures/` | 그림(PNG 300dpi, SVG) |
| `outputs/html/` | 인터랙티브 그림(Plotly, Folium) |
| `outputs/tables/` | 그림별 수치(`fig_*`), 품질 점검(`quality_*`), 모델(`model_*`), 시뮬레이션(`sim_*`) |
| `data/mock/` | 코드 검증용 모의 데이터 — 분석에 사용하지 않음 |

## 4. 원칙
- 실측 분석과 가정 기반 시뮬레이션을 구분해 표기한다(보고서 🟦/🟧).
- 결측은 평균 대체하지 않고 원인·처리 방식을 기록한다.
- 모든 그림에는 기간·단위·대상·출처를 표기하고, 수치 표를 함께 저장한다.
