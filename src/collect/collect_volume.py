"""서울시 교통량 지점 정보(SpotInfo) + 지점별 시간대별 교통량 이력(VolInfo) 수집.

VolInfo는 지점×일×시 단위로만 호출되므로(지점 139개 → 하루 3,336회) 날짜를 지정해 수집한다.
예) python src/collect/collect_volume.py --dates 20261014 20261015
    python src/collect/collect_volume.py --start 20240101 --end 20251231 --hours 07 08 09 12 17 18 19 --out volume_peak --workers 4
출력: data/raw/volume_spots.csv, data/raw/volume_hourly/YYYYMMDD.csv
"""
import argparse
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from common import RAW, ApiError, seoul_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="*")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--hours", nargs="*", default=[f"{h:02d}" for h in range(24)])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="volume_hourly", help="data/raw 아래 출력 폴더 (시간대 일부만 받을 땐 다른 폴더 사용)")
    a = ap.parse_args()

    spots_path = RAW / "volume_spots.csv"
    if not spots_path.exists():
        pd.DataFrame(seoul_all("SpotInfo", fmt="xml")).to_csv(spots_path, index=False, encoding="utf-8-sig")
    spots = pd.read_csv(spots_path, dtype=str)["spot_num"].tolist()
    print(f"교통량 지점 {len(spots)}개")

    dates = a.dates or [d.strftime("%Y%m%d") for d in pd.date_range(a.start, a.end)]
    out_dir = RAW / a.out
    out_dir.mkdir(parents=True, exist_ok=True)

    for ymd in dates:
        out = out_dir / f"{ymd}.csv"
        if out.exists():
            continue
        jobs = [(s, ymd, h) for s in spots for h in a.hours]
        try:
            with ThreadPoolExecutor(a.workers) as ex:
                parts = ex.map(lambda j: seoul_all("VolInfo", j, fmt="xml"), jobs)
                rows = [r for p in parts for r in p]
        except ApiError as e:
            # 하루 단위로 건너뛰고 계속 — 재실행 시 이 날짜만 다시 받음
            print(f"{ymd}: 실패, 건너뜀 ({e})")
            continue
        if not rows:
            print(f"{ymd}: 데이터 없음")
            continue
        pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"{ymd}: {len(rows):,}행 ({len(jobs):,}회 호출)")


if __name__ == "__main__":
    main()
