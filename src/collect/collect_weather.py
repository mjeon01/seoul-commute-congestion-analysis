"""기상청 ASOS 시간자료(서울 108 지점) 수집 — 공공데이터포털 AsosHourlyInfoService.

예) python src/collect/collect_weather.py --start 20250101 --end 20261018
출력: data/raw/weather_asos_108/YYYYMM.csv (월별)
"""
import argparse

import pandas as pd

from common import ENV, RAW, ApiError, get

URL = "https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
ROWS = 999  # 1회 최대 행 수


def fetch(start, end, stn):
    rows, page = [], 1
    while True:
        params = dict(serviceKey=ENV["DATA_GO_KR_SERVICE_KEY"], pageNo=page, numOfRows=ROWS, dataType="JSON",
                      dataCd="ASOS", dateCd="HR", startDt=start, startHh="00", endDt=end, endHh="23", stnIds=stn)
        r = get(URL, params)
        try:
            js = r.json()["response"]
        except (ValueError, KeyError):
            raise ApiError(f"ASOS 응답 오류: {r.text[:300]}") from None
        if js["header"]["resultCode"] != "00":
            raise ApiError(f"ASOS: {js['header']}")
        body = js["body"]
        rows += body["items"]["item"]
        if len(rows) >= body["totalCount"]:
            return rows
        page += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20250101")
    ap.add_argument("--end", default="20261018")
    ap.add_argument("--stn", default="108")
    a = ap.parse_args()

    out_dir = RAW / f"weather_asos_{a.stn}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in pd.period_range(pd.Timestamp(a.start), pd.Timestamp(a.end), freq="M"):
        out = out_dir / f"{m.strftime('%Y%m')}.csv"
        if out.exists():
            continue
        s = max(m.start_time, pd.Timestamp(a.start)).strftime("%Y%m%d")
        e = min(m.end_time, pd.Timestamp(a.end)).strftime("%Y%m%d")
        rows = fetch(s, e, a.stn)
        pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"weather {m}: {len(rows):,}행")


if __name__ == "__main__":
    main()
