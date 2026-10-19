"""버스 정류소 좌표(busStopLocationXyInfo)·지하철 역사 좌표(subwayStationMaster) 수집 — 열린데이터광장.

예) python src/collect/collect_locations.py
출력: data/raw/bus_stop_locations.csv, data/raw/subway_station_locations.csv
"""
import pandas as pd

from common import RAW, seoul_all

SERVICES = {"bus_stop_locations": "busStopLocationXyInfo", "subway_station_locations": "subwayStationMaster"}


def main():
    for name, service in SERVICES.items():
        rows = seoul_all(service)
        pd.DataFrame(rows).to_csv(RAW / f"{name}.csv", index=False, encoding="utf-8-sig")
        print(f"{name}: {len(rows):,}행")


if __name__ == "__main__":
    main()
