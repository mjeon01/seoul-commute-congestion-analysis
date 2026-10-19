# 서울시 출퇴근 교통혼잡 분석 — 에이전트 인수인계

2026 교통문제 해결 데이터 분석 공모전용 프로젝트. **요구사항 원문은 `docs/project_spec.md`** — 작업 전 반드시 읽을 것.
데이터 확보 현황·출처·호출 제약은 `docs/data_sources.md`.

최종 갱신: 2026-10-19

## 반드시 지킬 원칙 (원문 9장 요약)
- 실제 데이터만 사용. 확보 못 한 데이터로 결과를 만들지 않는다. 모의데이터는 `data/mock/`에서 코드 검증용으로만.
- 실측 분석 결과와 가정 기반 시뮬레이션을 엄격히 구분해 표기.
- 기간이 다른 데이터는 함부로 합치지 않는다(통합 시 공통 기간만).
- API 키는 `.env`에서만 읽는다. 키 값을 출력·로그·커밋하지 않는다.
- 결측은 평균 대체 금지 — 원인·비율 확인 후 처리 방식 기록.
- 그림: 한국어, 300DPI PNG(+SVG), 기간·단위·대상 표기. Plotly HTML도 제작.
- 복잡한 모델이 베이스라인보다 못하면 그대로 보고. SHAP/예측모델을 정책 인과효과로 해석하지 않는다.

## 환경
- Python: 프로젝트 가상환경 `.venv/` (Python 3.14). 실행은 `.venv/bin/python`.
  - `requirements.txt` 설치 완료(pyarrow·pyyaml 포함). libomp 설치됨 → XGBoost·LightGBM 정상.
- `.env` 키: `SEOUL_OPEN_API_KEY`(열린데이터광장), `DATA_GO_KR_SERVICE_KEY`(공공데이터포털), `TRAFFIC_DATA_PATH`(t-data 소통정보 CSV 경로). 형식은 `.env.example` 참고.
- git: `git init`만 됨, 커밋 없음. `data/raw/*`, `data/processed/*`, 루트 `*.csv`, `.env`, `.venv/`는 gitignore.

## 폴더 구조 (원문 8장 기준)
```
src/collect/        수집 코드 (완료)
src/preprocess/     전처리 (완료, 실행 순서·규칙은 docs/preprocessing_log.md)
src/analysis/       EDA (eda_traffic / eda_transit / eda_factors[--long]) — 결과 요약 docs/eda_findings.md
src/models/         forecast_volume.py: 첨두(08·09·18·19시) 지점 교통량 1시간 전 예측 — 코드 완성·모의데이터 검증, 실데이터 실행 대기
src/simulation/     simulate.py: 시나리오 A~E × 가정 낮음/중간/높음 × 속도모형(탄력성/BPR) — 완료, 가정값 config/simulation.yaml
src/visualization/  style.py: 폰트·검증된 팔레트·save()(PNG300+SVG+표)·save_html()
config/  notebooks/  reports/  outputs/{figures,tables,html,models}/
data/raw/  data/processed/  data/mock/
docs/    project_spec.md, data_sources.md, preprocessing_log.md, eda_findings.md
config/config.yaml  첨두 정의·혼잡 기준·좌표계
```
- 최상위 폴더명은 원문의 `traffic_policy_analysis`가 아니라 `교통문제_데이터분석`을 그대로 사용 중(변경 여부 사용자 미정).

## 수집 코드 (`src/collect/`, 해당 폴더에서 실행)
| 스크립트 | 내용 | 출력 |
|---|---|---|
| `common.py` | `.env` 로드, 재시도 요청, 열린데이터광장 페이지 수집(`seoul_all`), JSON/XML 처리 | — |
| `collect_transit.py --start YYYYMM --end YYYYMM` | 지하철 `CardSubwayTime`, 버스 `CardBusTimeNew` 월별 | `data/raw/subway_hourly/`, `bus_hourly/` `YYYYMM.csv` |
| `collect_volume.py --dates ... \| --start --end [--hours]` | 교통량 지점 `SpotInfo` + 이력 `VolInfo` | `data/raw/volume_spots.csv`, `volume_hourly/YYYYMMDD.csv` |
| `collect_weather.py --start YYYYMMDD --end YYYYMMDD` | 기상청 ASOS 시간자료(서울 108) | `data/raw/weather_asos_108/YYYYMM.csv` |
| `collect_holidays.py --years ...` | 공휴일(특일 정보) | `data/raw/holidays.csv` |
| `collect_locations.py` | 버스정류소·지하철역 좌표 | `data/raw/bus_stop_locations.csv`, `subway_station_locations.csv` |
| `collect_volume.py ... --out volume_peak` | 2024~2025 07·08·09·12·17·18·19시 장기 교통량(백그라운드 수집, 약 4시간) | `data/raw/volume_peak/` |
| `collect_link_geometry.py <소통정보 CSV>` | TOPIS 서비스링크 형상 `LinkVerInfo` | `data/raw/service_link_vertices.csv` |

- 모든 스크립트는 이미 있는 파일을 건너뜀(재실행 안전). 부분 수집 파일이 생기면 지우고 재실행.
- 로그: `data/raw/logs/`. CSV는 `utf-8-sig`.

## 데이터 확보 현황 (2026-10-19)
| 데이터 | 상태 |
|---|---|
| 1시간 소통정보(구간별) — t-data 수동 다운로드 | 루트 `1시간 소통정보 (구간별).csv`. **2026-10-14 전체 + 10-15 일부뿐** (168,719행, 링크 5,114개, UTF-8-BOM). 예측·계절 분석엔 부족 → 사용자에게 2024~2025 기간 다운로드 요청함 |
| 교통량 지점 정보 | 확보 (139개 지점) |
| 교통량 이력 | 2026-10-14, 10-15 전 시간 확보 (각 약 2.2만 행). 하루 3,336회 호출 필요 → 소통정보 기간 확정 후 출퇴근 시간대 위주로 추가 수집 예정 |
| 지하철 시간대별 승하차 | 2024-01 ~ 2026-09 확보 (33개월) |
| 버스 시간대별 승하차 | 2024-01 ~ 2026-09 확보 (33개월, 월 약 4.2만 행) |
| 기상 ASOS | 2024-01 ~ 2026-10-18 확보 |
| 공휴일 | 2024~2026 확보 |
| 스마트교차로 교통량 | `data/raw/manual/TOPIS_1H_*.zip` → `data/raw/ext/smart_intersection/`. **30일(2026-09-19~10-18), 노원구 5개 교차로뿐** |
| 표준노드링크 SHP | `manual/[2026-10-14]NODELINKDATA.zip` → `ext/nodelink_20261014/` (구버전 2개 zip은 미사용) |
| 행정구역 경계 SHP | `manual/bnd_all_00_2025_2Q.zip` → `ext/boundary/` (SGIS 2025 2Q) |
| 버스정류소·지하철역 좌표, TOPIS 서비스링크 형상 | 확보 |
| 배출계수 | 공개 API 없음 → 국립환경과학원 공표값을 출처 명시한 가정값으로 사용 예정 |

## API 주의사항
- 열린데이터광장: 1회 최대 1,000행. `SpotInfo`·`VolInfo`·`TrafficInfo`는 **XML만** 지원(JSON 요청 시 ERROR-301). `VolInfo`는 시간(hh) 인자 필수.
- `TrafficInfo`는 현재 시점만 → 분석에 사용하지 않음.
- 공공데이터포털: 서비스별 활용신청 필요. ASOS 1회 최대 999행.

## 기획서 검토 시 합의한 방법론 메모
- 시뮬레이션의 핵심은 실측 기반 구간별 **속도–교통량 관계 추정**(BPR 등). 예측 모델과는 별도 모듈.
- 홀짝제 순응률·예외차량·전환율은 선행 사례(2002 월드컵 2부제, 승용차요일제 등) 범위로 낮음/중간/높음 민감도 분석.
- 원문 출근 08~10시 정의가 실측 첨두와 맞는지 EDA로 확인해 보고.
- Q4(대중교통 수용력)엔 승하차 외 지하철 혼잡도·버스 배차/정원 자료 필요, Q5엔 자치구별 인구·접근성 자료 필요.

## 전처리 산출물 (`data/processed/`)
`speed_hourly.parquet`(소통정보+링크속성+혼잡등급), `volume_hourly.parquet`, `smart_intersection_hourly.parquet`,
`weather_hourly.parquet`, `subway_monthly_hourly.parquet`, `bus_monthly_hourly.parquet`(3,359만 행),
`seoul_gu/dong/links/service_links.gpkg`, `volume_spots.gpkg`, `transit_points.gpkg`(layer subway/bus).
모든 시간은 `hour_start`(1시간 구간 시작 시각) 기준 — 데이터별 변환 규칙은 docs/preprocessing_log.md.

## 핵심 주의
- 소통정보 링크ID는 **TOPIS 서비스링크ID** (국가 표준링크ID 아님). 표준링크 속성은 공간 매칭으로 이전함.
- 소통정보는 **33시간(2026-10-14 01시~10-15 09시)** 뿐. t-data 다운로드가 최근 구간만 주는 것으로 보임 → 장기 속도자료 확보 방안을 사용자와 결정 중.
- 승하차는 월 합계라 소통정보와 기간이 겹치지 않음 → 시간 단위 결합 금지, 패턴 비교만.

## 다음 할 일
1. 장기 교통량(`data/raw/volume_peak/`, 731일) 수집 완료 확인 — 실패 날짜는 로그 `data/raw/logs/volume_peak.log`의 "실패" 확인 후 같은 명령 재실행
2. `eda_factors.py --long` (D02~D04) → 보고서 4.4 갱신
3. `src/models/forecast_volume.py` 실데이터 실행 → 보고서 5절·7.3 작성 (베이스라인보다 못하면 그대로 보고)
4. EDA 노트북 `notebooks/01_EDA.ipynb` (그림·표 불러와 해설)
5. 보고서 최종 검토: 모든 수치를 outputs/tables 와 대조

## 보고서·결론 현황
- 초안 `reports/final_report.md` (5절 AI 결과만 대기). 실행 가이드 `README.md`.
- 시뮬레이션 핵심: 같은 VKT 감소당 속도개선 효율은 B(전역 홀짝제)≈C(선택 관리) → 제한 통행이 적은 C+D(E) 제안. B는 첨두 앞뒤 시간 악화, 높은 가정에서 대중교통 +10.8%.
- 출처 미확인 가정값(선택 관리 감소율 등)은 보고서에 "출처 미확인"으로 명시함 — 출처 없는 수치를 사실처럼 쓰지 말 것.
