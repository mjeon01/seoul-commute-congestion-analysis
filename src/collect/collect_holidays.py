"""공휴일(대체·임시공휴일 포함) 수집 — 공공데이터포털 한국천문연구원_특일 정보(getRestDeInfo).

활용신청: https://www.data.go.kr/data/15012690/openapi.do
예) python src/collect/collect_holidays.py --years 2024 2025 2026
출력: data/raw/holidays.csv
"""
import argparse

import pandas as pd

from common import ENV, RAW, ApiError, get

URL = "https://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", default=["2024", "2025", "2026"])
    a = ap.parse_args()

    rows = []
    for y in a.years:
        r = get(URL, dict(serviceKey=ENV["DATA_GO_KR_SERVICE_KEY"], solYear=y, numOfRows=100, _type="json"))
        try:
            body = r.json()["response"]["body"]
        except (ValueError, KeyError):
            raise ApiError(f"특일 정보 응답 오류: {r.text[:300]}") from None
        items = body["items"]["item"] if body["totalCount"] else []
        rows += items if isinstance(items, list) else [items]
        print(f"{y}: 공휴일 {body['totalCount']}건")
    pd.DataFrame(rows).to_csv(RAW / "holidays.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
