# 데이터 출처 및 확보 상태

점검일: 2026-10-19 (실제 API 호출로 확인). 우선 기간: 2024-01 ~ 2025-12, 가능 시 2026년 최신까지 확장.
기간이 서로 다른 데이터는 통합하지 않고, 통합 시 공통 기간만 사용한다.

## 필수 데이터

| 데이터 | 출처 / 서비스명 | 형식 | 확보 상태 | 수집 코드 |
|---|---|---|---|---|
| 1시간 소통정보(구간별) | 교통빅데이터플랫폼 t-data (로그인 후 CSV) | CSV, UTF-8-BOM | **2026-10-14 ~ 15 일부만 확보 (168,719행, 링크 5,114개)** → 기간 확장 필요 | 수동 다운로드 |
| 스마트교차로 1시간 교통량 | t-data (TOPIS_1H_*.zip, `VW_SC_TRF_VOL_1H.csv`) | CSV | **확보** 2026-09-19~10-18, 노원구 5개 교차로 | 수동 다운로드 |
| 교통량 이력 정보 | 열린데이터광장 `VolInfo` (XML만 지원) | API, 지점×일×시 단위 | 2024-09 ~ 2026-09 값 존재 확인. 2026-10-14~15 전 시간 **확보** (각 약 2.2만 행) | `src/collect/collect_volume.py` |
| 지하철 역별 시간대별 승하차 | 열린데이터광장 `CardSubwayTime` | API, 월 단위 | **확보** 2024-01 ~ 2026-09 (33개월) | `src/collect/collect_transit.py` |
| 버스 노선별 정류장별 시간대별 승하차 | 열린데이터광장 `CardBusTimeNew` | API, 월 단위 (월 약 4.2만 행) | **확보** 2024-01 ~ 2026-09 (33개월) | `src/collect/collect_transit.py` |

## 추가 데이터

| 데이터 | 출처 | 형식 | 확보 상태 | 비고 |
|---|---|---|---|---|
| 교통량 지점 정보 | 열린데이터광장 `SpotInfo` | API (XML) | **확보** (139개 지점, GRS80 TM 좌표) | `collect_volume.py` |
| 기상청 ASOS 시간자료 (서울 108) | 공공데이터포털 `AsosHourlyInfoService` | API | **확보** 2024-01 ~ 2026-10-18 | `collect_weather.py` |
| 공휴일(대체·임시 포함) | 공공데이터포털 [한국천문연구원_특일 정보](https://www.data.go.kr/data/15012690/openapi.do) | API | **확보** 2024~2026 | `collect_holidays.py` |
| 지하철 일별 승하차 | 열린데이터광장 `CardSubwayStatsNew` | API, 일 단위 | 호출 가능 확인 (2026-10-01 615행) | 필요 시 수집 |
| 표준노드링크 (도로 공간정보) | [ITS 국가교통정보센터](https://www.its.go.kr/nodelink/nodelinkRef) | SHP | **확보** 2026-10-14판 | 소통정보 링크ID는 서비스링크ID라 직접 결합 불가 → 공간 매칭 |
| 행정구역(자치구·행정동) 경계 | [통계지리정보서비스 SGIS 자료제공](https://sgis.mods.go.kr/view/pss/openDataIntrcn) | SHP | **확보** 2025년 2분기 | |
| 버스정류소 좌표 | 열린데이터광장 `busStopLocationXyInfo` | API | **확보** 11,236개 (서울 정류소만) | `collect_locations.py` |
| 지하철역 좌표 | 열린데이터광장 `subwayStationMaster` | API | **확보** 784개 | `collect_locations.py` |
| TOPIS 서비스링크 형상 | 열린데이터광장 `LinkVerInfo` (XML, 링크ID별) | API | **확보** 5,114개 링크 | `collect_link_geometry.py` |
| 실시간 도로 소통 | 열린데이터광장 `TrafficInfo` | API | 현재 시점만 제공 → 분석 부적합 | 사용 안 함 |
| 차량 유형별 교통량, 배출계수 | 국립환경과학원 등 | 문헌/파일 | 미확보 | 시뮬레이션 CO2 지표용. 가정값으로 사용 시 출처 명시 |

라이선스: 서울 열린데이터광장·공공데이터포털 데이터는 대부분 공공누리 제1유형(출처표시). 데이터셋별로 최종 확인 후 기재.

## 호출 제약 메모
- 열린데이터광장: 1회 최대 1,000행. `SpotInfo`·`VolInfo`·`TrafficInfo`는 JSON 미지원(XML만).
- `VolInfo`는 시간(hh) 인자가 필수 → 하루 139지점×24시간 = 3,336회 호출.
- 공공데이터포털 ASOS: 1회 최대 999행, 서비스별 활용신청 필요.
