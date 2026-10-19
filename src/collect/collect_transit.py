"""지하철 역별·버스 정류장별 시간대별 승하차 인원(월 단위) 수집.

예) python src/collect/collect_transit.py --start 202501 --end 202609
출력: data/raw/subway_hourly/YYYYMM.csv, data/raw/bus_hourly/YYYYMM.csv (월별 파일, 이미 있으면 건너뜀)
"""
import argparse

import pandas as pd

from common import RAW, months, seoul_all

SERVICES = {"subway_hourly": "CardSubwayTime", "bus_hourly": "CardBusTimeNew"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="202501")
    ap.add_argument("--end", default="202609")
    ap.add_argument("--only", choices=SERVICES)
    a = ap.parse_args()

    for name, service in SERVICES.items():
        if a.only and name != a.only:
            continue
        out_dir = RAW / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for ym in months(a.start, a.end):
            out = out_dir / f"{ym}.csv"
            if out.exists():
                continue
            rows = seoul_all(service, (ym,))
            if not rows:
                print(f"{name} {ym}: 데이터 없음")
                continue
            pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
            print(f"{name} {ym}: {len(rows):,}행")


if __name__ == "__main__":
    main()
