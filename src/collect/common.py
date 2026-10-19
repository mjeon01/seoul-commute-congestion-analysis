"""수집 공통 유틸: 인증키 로드, 재시도 요청, 서울 열린데이터광장 페이지 수집."""
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
ENV = dotenv_values(ROOT / ".env")

SEOUL_BASE = "http://openapi.seoul.go.kr:8088"
PAGE = 1000  # 열린데이터광장 1회 최대 행 수


class ApiError(RuntimeError):
    pass


def get(url, params=None, retries=4, timeout=60):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if i == retries - 1:
                raise ApiError(f"요청 실패: {type(e).__name__}") from None
            time.sleep(2 ** i)


def seoul_page(service, start, end, args=(), fmt="json", retries=4):
    """열린데이터광장 1페이지 호출 → (전체건수, 행 리스트). 데이터 없음(INFO-200)은 (0, []).

    서버가 가끔 빈 응답이나 깨진 XML/JSON을 돌려주므로 몇 번 다시 시도한 뒤 ApiError로 넘긴다."""
    for i in range(retries):
        try:
            return _seoul_page(service, start, end, args, fmt)
        except (ET.ParseError, json.JSONDecodeError, ApiError) as e:
            if i == retries - 1:
                raise ApiError(f"{service} {args}: {type(e).__name__} {e}") from None
            time.sleep(2 ** i)


def _seoul_page(service, start, end, args, fmt):
    key = ENV["SEOUL_OPEN_API_KEY"]
    path = "/".join([key, fmt, service, str(start), str(end), *map(str, args)])
    text = get(f"{SEOUL_BASE}/{path}/").text
    if fmt == "json":
        js = json.loads(text)
        body = js.get(service, js)
        code = body.get("RESULT", {}).get("CODE")
        if code == "INFO-200":
            return 0, []
        if code != "INFO-000":
            raise ApiError(f"{service}: {body.get('RESULT')}")
        return body["list_total_count"], body["row"]
    root = ET.fromstring(text)
    code = root.findtext(".//RESULT/CODE")
    if code == "INFO-200":
        return 0, []
    if code != "INFO-000":
        raise ApiError(f"{service}: {code} {root.findtext('.//RESULT/MESSAGE')}")
    rows = [{c.tag: c.text for c in row} for row in root.iter("row")]
    return int(root.findtext("list_total_count")), rows


def seoul_all(service, args=(), fmt="json"):
    """전체 페이지를 순회해 모든 행 반환."""
    total, rows = seoul_page(service, 1, PAGE, args, fmt)
    for s in range(PAGE + 1, total + 1, PAGE):
        rows += seoul_page(service, s, s + PAGE - 1, args, fmt)[1]
    if len(rows) != total:
        raise ApiError(f"{service} {args}: 행 수 불일치 {len(rows)} != {total}")
    return rows


def months(start, end):
    """'YYYYMM' 범위(양끝 포함)."""
    y, m = int(start[:4]), int(start[4:])
    while f"{y}{m:02d}" <= end:
        yield f"{y}{m:02d}"
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
