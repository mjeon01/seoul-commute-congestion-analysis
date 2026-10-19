"""EDA D: 외부 요인 — 차량 통행량과 속도의 관계, 요일·공휴일, 강수, 월별·계절.

실행: .venv/bin/python src/analysis/eda_factors.py [--long]
- 속도-교통량: VolInfo 지점(2026-10-14·15) ↔ 반경 40m 내 TOPIS 서비스링크 속도(같은 시간). 지점의 방향(io_type)과
  링크 방향의 대응 정보가 없어 양방향 합계 교통량 ↔ 주변 링크 길이가중 평균속도로 결합(한계로 명시).
- --long: 장기 교통량(data/raw/volume_peak, 2024~2025 출퇴근 전후 시간대)으로 요일·공휴일·강수·월별 효과.
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from visualization.style import (GRAY, INK2, SERIES, save, setup, titles)  # noqa: E402
from preprocess.common import add_calendar, add_peak  # noqa: E402

PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
SRC_VOL = "서울 열린데이터광장 서울시 교통량 이력 정보(VolInfo)"
SRC_SPEED = "서울 교통빅데이터플랫폼 1시간 소통정보(구간별)"
SRC_WX = "기상청 ASOS 시간자료(서울 108, 공공데이터포털)"


def spot_link_pairs(radius=40):
    spots = gpd.read_file(PROC / "volume_spots.gpkg")
    links = gpd.read_file(PROC / "seoul_service_links.gpkg")[["link_id", "length_m", "lanes", "geometry"]]
    buf = spots[["spot_num", "geometry"]].copy()
    buf["geometry"] = spots.buffer(radius)
    return gpd.sjoin(links, buf, predicate="intersects")[["spot_num", "link_id", "length_m"]]


def speed_flow_table():
    v = pd.read_parquet(PROC / "volume_hourly.parquet")
    vg = v.groupby(["spot_num", "datetime"]).agg(vol=("vol", "sum"), lanes=("n_lanes", "sum")).reset_index()
    vg["flow_per_lane"] = vg.vol / vg.lanes
    s = pd.read_parquet(PROC / "speed_hourly.parquet", columns=["link_id", "datetime", "speed", "road_class",
                                                                "speed_outlier"])
    s = s[~s.speed_outlier]
    pairs = spot_link_pairs()
    m = pairs.merge(s, on="link_id")
    sp = (m.groupby(["spot_num", "datetime"])
          .apply(lambda g: pd.Series({"speed": np.average(g.speed, weights=g.length_m),
                                      "road_class": g.road_class.mode().iat[0], "n_links": len(g)}),
                 include_groups=False).reset_index())
    t = vg.merge(sp, on=["spot_num", "datetime"])
    t["hour_start"] = t.datetime.dt.hour
    return add_peak(t)


def fig_speed_flow(t):
    classes = [c for c in ["도시고속도로", "주간선도로", "보조간선도로"] if (t.road_class == c).sum() > 50]
    fig, axes = plt.subplots(1, len(classes), figsize=(5.2 * len(classes), 5.4), sharey=True)
    rows = []
    for ax, rc in zip(np.atleast_1d(axes), classes):
        d = t[t.road_class == rc]
        off = d[d.peak == "비첨두"]
        pk = d[d.peak != "비첨두"]
        ax.scatter(off.flow_per_lane, off.speed, s=10, color=GRAY, alpha=0.6, linewidths=0, label="비첨두")
        ax.scatter(pk.flow_per_lane, pk.speed, s=14, color=SERIES[1], alpha=0.8, linewidths=0, label="출퇴근 첨두")
        bins = pd.cut(d.flow_per_lane, np.arange(0, d.flow_per_lane.max() + 200, 200))
        med = d.groupby(bins, observed=True).speed.median()
        x = [b.mid for b in med.index]
        ax.plot(x, med.values, color=SERIES[0], lw=2, label="구간 중앙값")
        r = d[["flow_per_lane", "speed"]].corr(method="spearman").iloc[0, 1]
        # 지점 간 차이(도로 폭·신호 등)가 섞이지 않도록 지점 내부 상관의 중앙값도 계산
        within = d.groupby("spot_num").apply(
            lambda x: x.flow_per_lane.corr(x.speed, method="spearman") if len(x) >= 10 else np.nan,
            include_groups=False).median()
        ax.set_title(f"{rc}  (지점 {d.spot_num.nunique()}개)\n전체 ρ={r:.2f} · 지점 내 ρ 중앙값={within:.2f}", fontsize=11)
        ax.set_xlabel("차로당 교통량 (대/시/차로)")
        ax.set_xlim(0, None)
        rows.append({"도로구분": rc, "지점수": d.spot_num.nunique(), "관측수": len(d), "spearman_전체": round(r, 3),
                     "spearman_지점내_중앙값": round(within, 3)})
    np.atleast_1d(axes)[0].set_ylabel("주변 링크 평균속도 (km/h)")
    np.atleast_1d(axes)[0].legend(loc="lower left", fontsize=9)
    titles(fig, "차량 통행량과 통행속도의 관계",
           "2026-10-14(수)·15(목) 매시 · 교통량 지점 양방향 합계 ↔ 반경 40m 내 TOPIS 링크 길이가중 평균속도 · 상관관계이며 인과 아님")
    fig.tight_layout(rect=(0, 0.02, 1, 0.9))
    save(fig, "D01_교통량_속도_관계", f"{SRC_VOL}; {SRC_SPEED}", pd.DataFrame(rows))
    t.to_parquet(PROC / "speed_flow_spot_hourly.parquet", index=False)


def load_long():
    files = sorted((RAW / "volume_peak").glob("*.csv"))
    v = pd.concat([pd.read_csv(f, dtype=str) for f in files])
    v["vol"] = pd.to_numeric(v.vol, errors="coerce")
    g = v.groupby(["spot_num", "ymd", "hh"]).vol.sum().reset_index()
    g["date"] = pd.to_datetime(g.ymd, format="%Y%m%d")
    g["hour_start"] = g.hh.astype(int)
    g["datetime"] = g.date + pd.to_timedelta(g.hour_start, unit="h")
    # 지점별로 결측일이 다르면 합계가 왜곡되므로 전 기간 관측된 지점만 사용
    n_days = g.date.nunique()
    full = g.groupby("spot_num").date.nunique()
    keep = full[full >= n_days * 0.95].index
    print(f"장기 교통량: {g.date.min():%Y-%m-%d}~{g.date.max():%Y-%m-%d} {n_days}일, 지점 {g.spot_num.nunique()}개 중 "
          f"95% 이상 관측 {len(keep)}개 사용")
    g = g[g.spot_num.isin(keep)]
    return add_peak(add_calendar(g)), len(keep)


def fig_weekday(g, n_spots):
    am = g[g.peak == "출근"].groupby(["date", "weekday", "day_type"]).vol.sum().reset_index()
    order = ["월", "화", "수", "목", "금", "토", "일"]
    t = am[am.day_type != "공휴일"].groupby("weekday").vol.agg(["mean", "std", "count"]).reindex(order)
    hol = am[am.day_type == "공휴일"].vol
    fig, ax = plt.subplots(figsize=(10, 5.4))
    x = np.arange(len(order))
    ax.bar(x, t["mean"] / 1e3, width=0.55, color=[SERIES[0]] * 5 + [GRAY] * 2)
    for xi, (m, n) in enumerate(zip(t["mean"], t["count"])):
        ax.text(xi, m / 1e3 + 2, f"{m / 1e3:,.0f}", ha="center", fontsize=9.5, color=INK2)
    ax.bar(len(order), hol.mean() / 1e3, width=0.55, color=SERIES[1])
    ax.text(len(order), hol.mean() / 1e3 + 2, f"{hol.mean() / 1e3:,.0f}", ha="center", fontsize=9.5, color=INK2)
    ax.set_xticks(list(x) + [len(order)], order + ["공휴일"])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("출근시간 교통량 합계 (천 대, 일평균)")
    titles(fig, "요일·공휴일별 출근시간 교통량",
           f"{g.date.min():%Y-%m-%d} ~ {g.date.max():%Y-%m-%d} · 08~10시 · 교통량 지점 {n_spots}개 합계 · 공휴일 {len(hol)}일은 요일에서 제외")
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    out = t.reset_index().rename(columns={"index": "weekday"})
    out = pd.concat([out, pd.DataFrame([{"weekday": "공휴일", "mean": hol.mean(), "std": hol.std(), "count": len(hol)}])])
    save(fig, "D02_요일공휴일별_출근교통량", SRC_VOL, out.round(0))


def fig_rain(g, n_spots):
    w = pd.read_parquet(PROC / "weather_hourly.parquet", columns=["datetime", "rain_mm"])
    d = g[g.day_type == "평일"].groupby(["datetime", "hour_start", "peak"]).vol.sum().reset_index().merge(w, on="datetime")
    d["강수"] = pd.cut(d.rain_mm, [-0.1, 0, 3, 1000], labels=["무강수", "약한 비(0~3mm]", "비(3mm 초과)"])
    t = d.groupby(["hour_start", "강수"], observed=True).vol.agg(["mean", "count"]).reset_index()
    base = t[t["강수"] == "무강수"].set_index("hour_start")["mean"]
    t["무강수대비_pct"] = (t["mean"] / t.hour_start.map(base) - 1) * 100
    hours = sorted(t.hour_start.unique())
    fig, ax = plt.subplots(figsize=(10, 5.4))
    width = 0.26
    for i, k in enumerate(["무강수", "약한 비(0~3mm]", "비(3mm 초과)"]):
        s = t[t["강수"] == k].set_index("hour_start").reindex(hours)
        ax.bar(np.arange(len(hours)) + (i - 1) * (width + 0.02), s["mean"] / 1e3, width=width, color=SERIES[i],
               label=f"{k}")
        if i:
            for xi, (p, n) in enumerate(zip(s["무강수대비_pct"], s["count"])):
                if pd.notna(p):
                    ax.text(xi + (i - 1) * (width + 0.02), s["mean"].iloc[xi] / 1e3 + 1, f"{p:+.0f}%\n(n={n})",
                            ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(hours)), [f"{h:02d}시" for h in hours])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("시간당 교통량 합계 (천 대, 평균)")
    ax.legend(loc="lower right", ncol=3, fontsize=9)
    titles(fig, "강수 여부별 평일 시간대 교통량",
           f"{g.date.min():%Y-%m-%d} ~ {g.date.max():%Y-%m-%d} 평일 · 교통량 지점 {n_spots}개 합계 · 강수는 같은 1시간 누적(서울 108) · 라벨: 무강수 대비 %")
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    save(fig, "D03_강수여부별_교통량", f"{SRC_VOL}; {SRC_WX}", t.round(1))


def fig_monthly(g, n_spots):
    d = g[(g.day_type == "평일") & g.peak.isin(["출근", "퇴근"])]
    t = d.groupby([d.date.dt.to_period("M"), "peak"]).apply(
        lambda x: x.groupby("date").vol.sum().mean(), include_groups=False).rename("vol").reset_index()
    t["month"] = t.date.dt.to_timestamp()
    fig, ax = plt.subplots(figsize=(11, 5.4))
    for i, p in enumerate(["출근", "퇴근"]):
        s = t[t.peak == p]
        ax.plot(s.month, s.vol / 1e3, color=SERIES[i], marker="o", markersize=4, markeredgecolor="white", label=f"{p} 2시간")
        ax.text(s.month.iloc[-1] + pd.Timedelta(days=12), s.vol.iloc[-1] / 1e3, p, va="center", fontsize=10, color=INK2)
    ax.set_ylim(0, t.vol.max() / 1e3 * 1.15)
    ax.set_ylabel("평일 일평균 교통량 합계 (천 대)")
    ax.legend(loc="lower left", ncol=2)
    titles(fig, "월별 평일 출퇴근 시간 교통량",
           f"{g.date.min():%Y-%m} ~ {g.date.max():%Y-%m} · 출근 08~10시, 퇴근 17~19시 · 교통량 지점 {n_spots}개 합계 · 공휴일 제외")
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    save(fig, "D04_월별_출퇴근_교통량", SRC_VOL, t[["month", "peak", "vol"]].round(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", action="store_true")
    a = ap.parse_args()
    setup()
    if a.long:
        g, n = load_long()
        fig_weekday(g, n)
        fig_rain(g, n)
        fig_monthly(g, n)
    else:
        fig_speed_flow(speed_flow_table())


if __name__ == "__main__":
    main()
