"""기상청 ASOS 시간자료(서울 108) 전처리.

- tm은 관측 시각. 강수량 rn은 직전 1시간 누적 → hour_start = tm - 1시간 으로 맞춰 소통정보와 같은 구간 기준
- rn 결측: ASOS는 무강수 시 강수량을 비워 둠 → 0으로 처리하고 rn_was_blank로 원래 결측 여부 보존
- 기온·풍속·습도 결측은 대체하지 않음

출력: data/processed/weather_hourly.parquet
"""
import pandas as pd

from common import PROC, RAW, add_calendar, add_peak, quality_report

COLS = {"tm": "obs_time", "ta": "temp", "rn": "rain_mm", "ws": "wind", "hm": "humidity", "dsnw": "snow_cm"}


def main():
    w = pd.concat([pd.read_csv(f, dtype=str) for f in sorted((RAW / "weather_asos_108").glob("*.csv"))])
    w = w.rename(columns=COLS)[list(COLS.values())]
    w["obs_time"] = pd.to_datetime(w["obs_time"])
    for c in ["temp", "rain_mm", "wind", "humidity", "snow_cm"]:
        w[c] = pd.to_numeric(w[c], errors="coerce")
    w = w.drop_duplicates("obs_time").sort_values("obs_time")
    expected = pd.date_range(w.obs_time.min(), w.obs_time.max(), freq="h")
    print(f"관측 {len(w):,}시간 / 기대 {len(expected):,}시간, 누락 {len(expected.difference(w.obs_time))}시간")

    w["rn_was_blank"] = w["rain_mm"].isna()
    w["rain_mm"] = w["rain_mm"].fillna(0.0)
    w["rain"] = w["rain_mm"] > 0
    w["datetime"] = w["obs_time"] - pd.Timedelta(hours=1)
    w["date"] = w["datetime"].dt.normalize()
    w["hour_start"] = w["datetime"].dt.hour
    w = add_peak(add_calendar(w))
    quality_report(w, "weather_hourly", key=["datetime"], numeric=["temp", "rain_mm", "wind", "humidity"])
    w.to_parquet(PROC / "weather_hourly.parquet", index=False)


if __name__ == "__main__":
    main()
