"""EDA C: 대중교통 수요 (지하철·버스 시간대별 승하차, 월 합계 자료).

실행: .venv/bin/python src/analysis/eda_transit.py
- 패턴 분석 기준 연도: 2025년(가장 최근의 완전한 연도). 월별 추이는 2024-01 ~ 2026-09.
- 원자료가 월 합계라 평일/주말 구분 불가 → '일평균'은 월 합계 / 월 일수.
- 서울 밖 역·정류소(자치구 결측)는 자치구 분석에서 제외, 서울 전체 합계에는 포함 여부를 부제에 명시.
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from visualization.style import (AXIS, DIVERGE, GRAY, INK, INK2, SERIES, save, setup, titles)  # noqa: E402

PROC = ROOT / "data" / "processed"
SRC_SUB = "서울 열린데이터광장 지하철 호선별 역별 시간대별 승하차 인원(CardSubwayTime)"
SRC_BUS = "서울 열린데이터광장 버스노선별 정류장별 시간대별 승하차 인원(CardBusTimeNew)"
YEAR = "2025"
AM = [8, 9]


def load():
    sub = pd.read_parquet(PROC / "subway_monthly_hourly.parquet")
    bus = pd.read_parquet(PROC / "bus_monthly_hourly.parquet",
                          columns=["ym", "month", "route", "route_name", "stop_id", "hour_start", "board", "alight",
                                   "bus_type", "SIGUNGU_NM"])
    return sub, bus


def year_daily(df, year=YEAR):
    """해당 연도 월 합계 → 연간 일평균(연 합계 / 연 일수)."""
    d = df[df.ym.str.startswith(year)]
    days = pd.Timestamp(f"{year}-12-31").dayofyear
    return d, days


def fig_subway_hourly(sub):
    d, days = year_daily(sub[sub.SIGUNGU_NM.notna()])
    t = d.groupby("hour_start")[["board", "alight"]].sum().div(days).div(1e4).reset_index()
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for col, c, lbl in [("board", SERIES[0], "승차"), ("alight", SERIES[1], "하차")]:
        ax.plot(t.hour_start, t[col], color=c, label=lbl)
        for hrs in (range(5, 12), range(15, 22)):
            r = t[t.hour_start.isin(hrs)].nlargest(1, col).iloc[0]
            ax.annotate(f"{lbl} {r[col]:.0f}만 ({int(r.hour_start):02d}시)", (r.hour_start, r[col]), xytext=(8, 2),
                        textcoords="offset points", fontsize=9, color=INK2)
    ax.axvspan(AM[0] - 0.5, AM[-1] + 0.5, color="#cde2fb", alpha=0.6, lw=0, zorder=0)
    ax.axvspan(16.5, 18.5, color="#cde2fb", alpha=0.6, lw=0, zorder=0)
    ax.set_ylim(0, t[["board", "alight"]].max().max() * 1.15)
    ax.set_xticks(range(4, 24, 2)) if t.hour_start.min() >= 4 else ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("시각 (1시간 구간 시작)")
    ax.set_ylabel("일평균 인원 (만 명)")
    ax.legend(loc="upper left")
    titles(fig, "지하철 시간대별 승하차 인원",
           f"{YEAR}년 일평균(월 합계 기반, 평일·주말 포함) · 서울 소재 역 {d.station.nunique()}개 · 음영: 출근 08~10시, 퇴근 17~19시 · 단위 만 명")
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    save(fig, "C01_지하철_시간대별_승하차", SRC_SUB, t.round(2))


def fig_monthly_trend(sub, bus):
    s = sub[sub.SIGUNGU_NM.notna()].groupby("month").board.sum()
    b = bus[bus.SIGUNGU_NM.notna()].groupby("month").board.sum()
    t = pd.DataFrame({"지하철": s, "버스": b})
    t = t.div(t.index.days_in_month, axis=0).div(1e4).reset_index()
    fig, ax = plt.subplots(figsize=(11, 5.4))
    for i, col in enumerate(["지하철", "버스"]):
        ax.plot(t.month, t[col], color=SERIES[i], marker="o", markersize=4, markeredgecolor="white", label=col)
        ax.text(t.month.iloc[-1] + pd.Timedelta(days=12), t[col].iloc[-1], col, va="center", fontsize=10, color=INK2)
        for f in (t[col].idxmax(), t[col].idxmin()):
            r = t.loc[f]
            ax.annotate(f"{r[col]:.0f}만\n({r.month:%y.%m})", (r.month, r[col]), xytext=(0, 8 if f == t[col].idxmax() else -24),
                        textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
    ax.set_ylim(0, t[["지하철", "버스"]].max().max() * 1.15)
    ax.set_ylabel("일평균 승차 인원 (만 명)")
    ax.legend(loc="lower left", ncol=2)
    titles(fig, "월별 대중교통 일평균 승차 인원 추이",
           f"2024-01 ~ 2026-09 · 서울 소재 지하철역·버스정류소 승차 · 월 합계 / 월 일수 · 단위 만 명")
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    save(fig, "C02_월별_대중교통_이용추이", f"{SRC_SUB}; {SRC_BUS}", t.round(2))


def fig_bus_routes(bus):
    d, days = year_daily(bus)
    t = d.groupby(["route", "bus_type"]).board.sum().div(days).reset_index().nlargest(15, "board")
    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(t))[::-1]
    types = list(dict.fromkeys(t.bus_type))
    colors = {bt: SERIES[i] for i, bt in enumerate(types)}
    for yi, (_, r) in zip(y, t.iterrows()):
        ax.barh(yi, r.board / 1e4, height=0.6, color=colors[r.bus_type])
        ax.text(r.board / 1e4 + 0.05, yi, f"{r.board / 1e4:.1f}만", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, [f"{r}번" for r in t.route])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("일평균 승차 인원 (만 명)")
    ax.legend(handles=[plt.matplotlib.patches.Patch(color=colors[k], label=k) for k in types], loc="lower right")
    titles(fig, "일평균 승차 인원 상위 15개 버스 노선", f"{YEAR}년 일평균(월 합계 기반) · 전체 정류소(서울 밖 포함) · 단위 만 명")
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    save(fig, "C03_버스노선_승차상위15", SRC_BUS, t.round(0))


def am_gu_flows(sub, bus):
    """출근시간(08~10시) 자치구별 대중교통 일평균 승차·하차."""
    out = []
    for name, df in [("지하철", sub), ("버스", bus)]:
        d, days = year_daily(df[df.SIGUNGU_NM.notna() & df.hour_start.isin(AM)])
        g = d.groupby("SIGUNGU_NM")[["board", "alight"]].sum().div(days)
        out.append(g.add_prefix(f"{name}_"))
    t = pd.concat(out, axis=1).fillna(0)
    t["승차"] = t["지하철_board"] + t["버스_board"]
    t["하차"] = t["지하철_alight"] + t["버스_alight"]
    t["순유입"] = t["하차"] - t["승차"]
    return t.reset_index().rename(columns={"SIGUNGU_NM": "자치구"})


def fig_net_inflow(t):
    gu = gpd.read_file(PROC / "seoul_gu.gpkg")
    m = gu.merge(t, left_on="SIGUNGU_NM", right_on="자치구")
    lim = np.ceil(m["순유입"].abs().max() / 1e4) * 1e4
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("div", [DIVERGE[0], DIVERGE[1], DIVERGE[2]])
    fig, ax = plt.subplots(figsize=(10, 8.6))
    m.plot(column="순유입", ax=ax, cmap=cmap, vmin=-lim, vmax=lim, edgecolor="white", linewidth=0.8)
    for _, r in m.iterrows():
        p = r.geometry.representative_point()
        ax.text(p.x, p.y, f"{r.SIGUNGU_NM}\n{r['순유입'] / 1e4:+.1f}", ha="center", va="center", fontsize=7.5,
                color="white" if abs(r["순유입"]) > lim * 0.55 else INK)
    ax.set_axis_off()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-lim / 1e4, lim / 1e4))
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.035, pad=0.01, aspect=45)
    cb.set_label("출근시간 하차 - 승차 (만 명/일) · 빨강: 유입(업무지) · 파랑: 유출(주거지)", color=INK2)
    cb.outline.set_visible(False)
    titles(fig, "자치구별 출근시간 대중교통 순유입",
           f"{YEAR}년 일평균 · 08~10시 지하철+버스 하차 - 승차 · 서울 소재 역·정류소 · 단위 만 명")
    save(fig, "C04_자치구별_출근시간_대중교통_순유입", f"{SRC_SUB}; {SRC_BUS}; 통계청 SGIS",
         t[["자치구", "승차", "하차", "순유입", "지하철_board", "버스_board"]].round(0))


def fig_speed_vs_transit(t):
    sp = pd.read_csv(ROOT / "outputs" / "tables" / "fig_B01_자치구별_출퇴근_평균속도_지도.csv")
    d = t.merge(sp, on="자치구")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(d["승차"] / 1e4, d["출근_평균속도"], s=60, color=SERIES[0], edgecolor="white", linewidth=2, zorder=3)
    for _, r in d.iterrows():
        ax.annotate(r["자치구"], (r["승차"] / 1e4, r["출근_평균속도"]), xytext=(5, 3), textcoords="offset points",
                    fontsize=8.5, color=INK2)
    rho = d[["승차", "출근_평균속도"]].corr(method="spearman").iloc[0, 1]
    ax.set_xlabel("출근시간(08~10시) 대중교통 일평균 승차 (만 명, 2025년)")
    ax.set_ylabel("출근시간 도로 평균속도 (km/h, 2026-10-14·15)")
    ax.text(0.01, 0.99, f"스피어만 순위상관 ρ = {rho:.2f}\n(n=25, 서로 다른 기간 — 인과 해석 불가)",
            transform=ax.transAxes, ha="left", va="top", fontsize=9, color=INK2)
    titles(fig, "자치구별 출근시간 대중교통 승차량과 도로 속도",
           "대중교통: 2025년 일평균(월 합계 기반) · 도로: 2026-10-14(수)·15(목) 08~10시 · 기간이 달라 패턴 비교만 가능")
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    save(fig, "C05_자치구별_대중교통승차_vs_도로속도", f"{SRC_SUB}; {SRC_BUS}; 서울 교통빅데이터플랫폼 소통정보",
         d[["자치구", "승차", "하차", "출근_평균속도"]].round(1))
    return d


def fig_demand_candidates(d):
    """추가 대중교통 수요 발생 가능 지역 — 탐색 지표.

    도로 혼잡이 심하고(출근 속도 낮음) 출근 유출(주거지 성격)이 커서 차량 수요 관리 시 대중교통 전환 수요가
    발생할 가능성이 큰 자치구를 순위화. 지표 = z(-출근 속도) + z(-순유입). 가중치 1:1은 가정.
    """
    pts = gpd.read_file(PROC / "transit_points.gpkg", layer="bus")
    gu = gpd.read_file(PROC / "seoul_gu.gpkg")
    area = gu.set_index("SIGUNGU_NM").area / 1e6
    stops = pts.groupby("SIGUNGU_NM").size()
    d = d.set_index("자치구")
    d["정류소밀도"] = (stops / area).reindex(d.index)
    z = lambda s: (s - s.mean()) / s.std()  # noqa: E731
    d["지표"] = z(-d["출근_평균속도"]) + z(-d["순유입"])
    d = d.sort_values("지표", ascending=False).reset_index()
    top = d.head(8)
    fig, ax = plt.subplots(figsize=(10, 6.2))
    y = np.arange(len(d))[::-1]
    ax.barh(y, d["지표"], height=0.62, color=[SERIES[0] if i < 8 else GRAY for i in range(len(d))])
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.text(r["지표"] + (0.05 if r["지표"] >= 0 else -0.05), yi,
                f"{r['출근_평균속도']:.0f}km/h · 순유입 {r['순유입'] / 1e4:+.1f}만",
                va="center", ha="left" if r["지표"] >= 0 else "right", fontsize=7.5, color=INK2)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_xlim(d["지표"].min() - 1.9, d["지표"].max() + 1.6)  # 막대 끝 라벨 공간
    ax.set_yticks(y, d["자치구"], fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("탐색 지표 = z(-출근 도로속도) + z(-출근 대중교통 순유입)")
    titles(fig, "추가 대중교통 수요 발생 가능 자치구 (탐색 지표)",
           "도로 혼잡이 심하고 출근 유출이 큰(주거지 성격) 자치구일수록 높음 · 상위 8곳 강조 · 가중치 1:1은 가정 · 인과 아님")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.86, bottom=0.13)
    save(fig, "C06_추가대중교통수요_후보자치구", f"{SRC_SUB}; {SRC_BUS}; 서울 교통빅데이터플랫폼 소통정보",
         d[["자치구", "출근_평균속도", "승차", "하차", "순유입", "정류소밀도", "지표"]].round(2))
    return top


def main():
    setup()
    sub, bus = load()
    fig_subway_hourly(sub)
    fig_monthly_trend(sub, bus)
    fig_bus_routes(bus)
    t = am_gu_flows(sub, bus)
    fig_net_inflow(t)
    d = fig_speed_vs_transit(t)
    fig_demand_candidates(d)


if __name__ == "__main__":
    main()
