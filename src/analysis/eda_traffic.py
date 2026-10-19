"""EDA A·B: 도로 혼잡 현황과 공간 분석 (소통정보 33시간, 교통량 2일, 스마트교차로 30일).

실행: .venv/bin/python src/analysis/eda_traffic.py
주의: 소통정보는 2026-10-14(수) 01시 ~ 10-15(목) 09시 평일 단면 — 모든 그림 부제에 명시.
"""
import sys
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from visualization.style import (AXIS, GRAY, HTML, INK, INK2, MUTED, SEQ, SERIES, save, save_html,  # noqa: E402
                                 setup, titles)

PROC = ROOT / "data" / "processed"
SRC_SPEED = "서울 교통빅데이터플랫폼 1시간 소통정보(구간별), TOPIS 서비스링크 형상(서울 열린데이터광장)"
PERIOD = "2026-10-14(수) 01시 ~ 10-15(목) 09시, 평일"
AM, PM = [8, 9], [17, 18]           # 원문 첨두 정의 (hour_start)
TOPIS_AM, TOPIS_PM = [7, 8], [17, 18]
STATUS = {"정체": "#d03b3b", "서행": "#ec835a", "원활": "#0ca30c"}
ROAD_ORDER = ["도시고속도로", "주간선도로", "보조간선도로", "기타도로"]


def wmean(df, col="speed", w="length_m"):
    return np.average(df[col], weights=df[w])


def load():
    s = pd.read_parquet(PROC / "speed_hourly.parquet")
    s = s[~s.speed_outlier & s.length_m.notna()]
    links = gpd.read_file(PROC / "seoul_service_links.gpkg")
    gu = gpd.read_file(PROC / "seoul_gu.gpkg")
    return s, links, gu


def shade_peaks(ax):
    ax.axvspan(AM[0] - 0.5, AM[-1] + 0.5, color=SEQ[0], alpha=0.6, lw=0, zorder=0)
    ax.axvspan(PM[0] - 0.5, PM[-1] + 0.5, color=SEQ[0], alpha=0.6, lw=0, zorder=0)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 1.08)  # 값 라벨과 첨두 라벨이 겹치지 않도록 여백
    tr = ax.get_xaxis_transform()
    ax.text(np.mean(AM), 1.01, "출근 08~10시", ha="center", va="bottom", fontsize=9, color=INK2, transform=tr)
    ax.text(np.mean(PM), 1.01, "퇴근 17~19시", ha="center", va="bottom", fontsize=9, color=INK2, transform=tr)


def fig_hourly_speed(s):
    rows = []
    for (rc, h), g in s.groupby(["road_class", "hour_start"]):
        rows.append({"도로구분": rc, "시작시각": h, "평균속도_kmh": wmean(g), "관측일수": g.date.nunique()})
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for i, rc in enumerate(ROAD_ORDER):
        d = t[t["도로구분"] == rc]
        ax.plot(d["시작시각"], d["평균속도_kmh"], color=SERIES[i], label=rc)
        ax.text(23.3, d["평균속도_kmh"].iloc[-1], rc, va="center", fontsize=9.5, color=INK2)
    ax.set_xlim(-0.5, 25.5)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("시각 (1시간 구간 시작)")
    ax.set_ylabel("구간길이 가중 평균속도 (km/h)")
    ax.set_ylim(0, None)
    shade_peaks(ax)
    ax.legend(loc="lower left", ncol=4)
    titles(fig, "도로 등급별 시간대 평균 통행속도", f"{PERIOD} · 서울 TOPIS 서비스링크 {s.link_id.nunique():,}개 · 단위 km/h")
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    save(fig, "A01_시간대별_도로등급별_평균속도", SRC_SPEED, t)

    f = go.Figure()
    for i, rc in enumerate(ROAD_ORDER):
        d = t[t["도로구분"] == rc]
        f.add_scatter(x=d["시작시각"], y=d["평균속도_kmh"].round(1), name=rc, line=dict(width=2, color=SERIES[i]),
                      hovertemplate="%{y} km/h")
    f.update_xaxes(title="시각 (1시간 구간 시작)", dtick=2)
    f.update_yaxes(title="평균속도 (km/h)", rangemode="tozero")
    save_html(f, "A01_시간대별_도로등급별_평균속도", "도로 등급별 시간대 평균 통행속도", f"{PERIOD} · 단위 km/h")
    return t


def fig_hourly_jam(s):
    t = (s.assign(jam_len=s.length_m * (s.congestion == "정체"), slow_len=s.length_m * (s.congestion == "서행"))
         .groupby("hour_start")[["jam_len", "slow_len", "length_m"]].sum())
    t["정체비율_pct"] = t.jam_len / t.length_m * 100
    t["서행이상비율_pct"] = (t.jam_len + t.slow_len) / t.length_m * 100
    t = t.reset_index().rename(columns={"hour_start": "시작시각"})[["시작시각", "정체비율_pct", "서행이상비율_pct"]]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.plot(t["시작시각"], t["서행이상비율_pct"], color=SERIES[1], label="서행 이상")
    ax.plot(t["시작시각"], t["정체비율_pct"], color=SERIES[0], label="정체")
    ax.set_ylim(0, t["서행이상비율_pct"].max() * 1.15)
    shade_peaks(ax)
    ax.plot([TOPIS_AM[0] - 0.5, TOPIS_AM[-1] + 0.5], [1.5, 1.5], color=MUTED, lw=1.5)
    ax.text(TOPIS_AM[0] - 0.5, 3, "TOPIS 오전첨두 07~09시", fontsize=8.5, color=MUTED)
    for col, c in [("서행이상비율_pct", SERIES[1]), ("정체비율_pct", SERIES[0])]:
        for hours in (range(5, 13), range(15, 22)):
            r = t[t["시작시각"].isin(hours)].nlargest(1, col).iloc[0]
            ax.annotate(f"{r[col]:.1f}%", (r["시작시각"], r[col]), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=9, color=INK2)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("시각 (1시간 구간 시작)")
    ax.set_ylabel("도로 연장 대비 비율 (%)")
    ax.legend(loc="upper left")
    titles(fig, "시간대별 혼잡 도로 비율", f"{PERIOD} · 정체: 일반도로 15, 도시고속도로 30 km/h 미만 · 서행: 25 / 70 km/h 미만 · 구간길이 기준 %")
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    save(fig, "A02_시간대별_혼잡도로비율", SRC_SPEED, t)
    return t


def fig_gu_maps(s, gu):
    def by_gu(hours):
        g = s[s.hour_start.isin(hours) & s.sigungu_nm.notna()]
        return g.groupby("sigungu_nm").apply(wmean, include_groups=False).rename("speed")
    t = pd.DataFrame({"출근_평균속도": by_gu(AM), "퇴근_평균속도": by_gu(PM)}).reset_index()
    t = t.rename(columns={"sigungu_nm": "자치구"})
    m = gu.merge(t, left_on="SIGUNGU_NM", right_on="자치구")
    vmin, vmax = np.floor(t.iloc[:, 1:].min().min() / 5) * 5, np.ceil(t.iloc[:, 1:].max().max() / 5) * 5
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ[::-1])  # 느릴수록 진하게
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))
    for ax, col, name in [(axes[0], "출근_평균속도", "출근 08~10시"), (axes[1], "퇴근_평균속도", "퇴근 17~19시")]:
        m.plot(column=col, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, edgecolor="white", linewidth=0.8)
        for _, r in m.iterrows():
            p = r.geometry.representative_point()
            ax.text(p.x, p.y, f"{r.SIGUNGU_NM}\n{r[col]:.0f}", ha="center", va="center", fontsize=6.8,
                    color="white" if r[col] < (vmin + vmax) / 2 else INK)
        ax.set_title(name, fontsize=12)
        ax.set_axis_off()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.04, pad=0.02, aspect=50)
    cb.set_label("구간길이 가중 평균속도 (km/h) — 진할수록 느림", color=INK2)
    cb.outline.set_visible(False)
    titles(fig, "자치구별 출퇴근 시간 평균 통행속도", f"{PERIOD} · 자치구 내 TOPIS 서비스링크 기준 · 단위 km/h")
    save(fig, "B01_자치구별_출퇴근_평균속도_지도", SRC_SPEED + ", 통계청 SGIS 행정구역경계(2025 2Q)",
         t.round(1))
    return t


def fig_link_map(s, links, gu, hour=8):
    h = s[s.hour_start == hour].groupby("link_id").agg(speed=("speed", "mean"), congestion=("congestion", "first"),
                                                       road=("road_name", "first"))
    # 2일 평균 속도로 등급 재산정
    h = h.drop(columns="congestion").join(s[["link_id", "road_class"]].drop_duplicates("link_id").set_index("link_id"))
    exp = h.road_class == "도시고속도로"
    h["congestion"] = np.select([h.speed < np.where(exp, 30, 15), h.speed < np.where(exp, 70, 25)],
                                ["정체", "서행"], "원활")
    m = links.merge(h, left_on="link_id", right_index=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    gu.plot(ax=ax, color="#f0efec", edgecolor="white", linewidth=1)
    for k in ["원활", "서행", "정체"]:
        d = m[m.congestion == k]
        d.plot(ax=ax, color=STATUS[k], linewidth=0.6 if k == "원활" else 1.3,
               label=f"{k} ({d.length_m.sum() / m.length_m.sum() * 100:.1f}%)")
    ax.legend(loc="lower right", title="소통 등급 (연장 비율)", fontsize=10)
    ax.set_axis_off()
    titles(fig, f"도로 구간별 소통 등급 — 출근시간 {hour:02d}~{hour + 1:02d}시",
           f"10-14(수)·10-15(목) {hour:02d}시대 2일 평균 · TOPIS 서비스링크 {len(m):,}개 · 정체 15/30, 서행 25/70 km/h 미만(일반/도시고속)")
    save(fig, f"B02_구간별_소통등급_지도_{hour:02d}시", SRC_SPEED + ", 통계청 SGIS",
         m.drop(columns="geometry")[["link_id", "road", "road_class", "SIGUNGU_NM", "speed", "congestion"]], svg=False)

    fm = folium.Map(location=[37.5665, 126.978], zoom_start=11, tiles="cartodbpositron")
    mm = m.to_crs(4326)
    for k in ["원활", "서행", "정체"]:
        layer = folium.FeatureGroup(name=k)
        folium.GeoJson(mm[mm.congestion == k][["link_id", "road", "SIGUNGU_NM", "speed", "geometry"]].round({"speed": 1}),
                       style_function=lambda f, c=STATUS[k]: {"color": c, "weight": 2.5, "opacity": 0.9},
                       tooltip=folium.GeoJsonTooltip(["road", "SIGUNGU_NM", "speed"],
                                                     aliases=["도로", "자치구", "평균속도(km/h)"])).add_to(layer)
        layer.add_to(fm)
    folium.LayerControl(collapsed=False).add_to(fm)
    fm.get_root().html.add_child(folium.Element(
        f"<div style='position:fixed;top:10px;left:50px;z-index:999;background:white;padding:8px 12px;"
        f"font-family:sans-serif;border-radius:6px'><b>도로 구간별 소통 등급 — {hour:02d}~{hour + 1:02d}시</b><br>"
        f"<span style='font-size:12px;color:#52514e'>2026-10-14·15 평균, 출처: 서울 교통빅데이터플랫폼</span></div>"))
    fm.save(HTML / f"B02_구간별_소통등급_지도_{hour:02d}시.html")
    return m


def fig_top20(s):
    a = s[s.hour_start.isin(AM) & (s.length_m >= 300)]
    t = (a.groupby(["link_id", "road_name", "direction", "from_node", "to_node", "road_class", "sigungu_nm"])
         .agg(speed=("speed", "mean"), length_m=("length_m", "first"), n=("speed", "size"))
         .reset_index().nsmallest(20, "speed"))
    t["label"] = t.road_name + " " + t.from_node + "→" + t.to_node
    fig, ax = plt.subplots(figsize=(10, 8.4))
    y = np.arange(len(t))[::-1]
    ax.barh(y, t.speed, height=0.62, color=SERIES[0])
    for yi, (_, r) in zip(y, t.iterrows()):
        ax.text(r.speed + 0.2, yi, f"{r.speed:.1f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, [f"{lbl}  ({g})" for lbl, g in zip(t.label, t.sigungu_nm)], fontsize=9)
    ax.set_xlabel("출근시간 평균속도 (km/h)")
    ax.grid(axis="y", visible=False)
    titles(fig, "출근시간 통행속도 하위 20개 도로 구간",
           f"08~10시, 10-14(수)·10-15(목) 평균 · 길이 300m 이상 구간 · 단위 km/h")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    save(fig, "B03_출근시간_혼잡상위20구간", SRC_SPEED,
         t[["link_id", "road_name", "direction", "from_node", "to_node", "road_class", "sigungu_nm", "length_m", "speed", "n"]])

    f = go.Figure(go.Bar(x=t.speed.round(1)[::-1], y=(t.label + " (" + t.sigungu_nm + ")")[::-1], orientation="h",
                         marker_color=SERIES[0], hovertemplate="%{y}<br>%{x} km/h<extra></extra>"))
    f.update_layout(hovermode="closest", height=650)
    f.update_xaxes(title="출근시간 평균속도 (km/h)")
    save_html(f, "B03_출근시간_혼잡상위20구간", "출근시간 통행속도 하위 20개 도로 구간", "08~10시, 2026-10-14·15 평균, 길이 300m 이상")
    return t


def fig_hotspots(s, links, gu):
    hours = [7, 8, 13, 18]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10.5))
    axes = axes.ravel()
    rows = []
    for ax, h in zip(axes, hours):
        d = s[s.hour_start == h].groupby("link_id").speed.mean()
        jam_ids = s[(s.hour_start == h) & (s.congestion == "정체")].link_id.unique()
        gu.plot(ax=ax, color="#f0efec", edgecolor="white", linewidth=0.8)
        links.plot(ax=ax, color=GRAY, linewidth=0.25)
        j = links[links.link_id.isin(jam_ids)]
        j.plot(ax=ax, color=STATUS["정체"], linewidth=1.2)
        share = j.length_m.sum() / links[links.link_id.isin(d.index)].length_m.sum() * 100
        ax.set_title(f"{h:02d}~{h + 1:02d}시  정체 {share:.1f}%", fontsize=11.5)
        ax.set_axis_off()
        rows.append({"시작시각": h, "정체링크수": len(jam_ids), "정체연장비율_pct": round(share, 2)})
    titles(fig, "시간대별 정체 구간 분포 변화",
           f"{PERIOD} · 빨강: 해당 시간 1회 이상 정체(일반 15, 도시고속 30 km/h 미만) 구간 · % = 도로 연장 대비")
    fig.subplots_adjust(top=0.9, bottom=0.03, wspace=0.02, hspace=0.08)
    save(fig, "B04_시간대별_정체핫스팟", SRC_SPEED + ", 통계청 SGIS", pd.DataFrame(rows), svg=False)


def fig_am_pm_gu(t):
    t = t.sort_values("출근_평균속도")
    fig, ax = plt.subplots(figsize=(9.5, 8.4))
    y = np.arange(len(t))
    ax.hlines(y, t["출근_평균속도"], t["퇴근_평균속도"], color=AXIS, lw=2)
    ax.scatter(t["출근_평균속도"], y, color=SERIES[0], s=55, zorder=3, edgecolor="white", linewidth=2, label="출근 08~10시")
    ax.scatter(t["퇴근_평균속도"], y, color=SERIES[1], s=55, zorder=3, edgecolor="white", linewidth=2, label="퇴근 17~19시")
    ax.set_yticks(y, t["자치구"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("구간길이 가중 평균속도 (km/h)")
    ax.legend(loc="lower right")
    titles(fig, "자치구별 출근·퇴근 시간 평균속도 비교", f"{PERIOD} · 단위 km/h · 출근 속도 낮은 순")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    save(fig, "A03_자치구별_출근퇴근_속도비교", SRC_SPEED, t)


def fig_volume_hourly():
    v = pd.read_parquet(PROC / "volume_hourly.parquet")
    t = v.groupby(["date", "hour_start"]).vol.sum().reset_index()
    t["날짜"] = t.date.dt.strftime("%m-%d(") + t.date.dt.weekday.map(dict(enumerate("월화수목금토일"))) + ")"
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for i, (d, g) in enumerate(t.groupby("날짜")):
        ax.plot(g.hour_start, g.vol / 1000, color=SERIES[i], label=d)
    r = t.loc[t.vol.idxmax()]
    ax.annotate(f"최대 {r.vol / 1000:,.0f}천 대 ({int(r.hour_start):02d}~{int(r.hour_start) + 1:02d}시)",
                (r.hour_start, r.vol / 1000), xytext=(-12, 0), textcoords="offset points", ha="right",
                va="center", fontsize=9, color=INK2)
    ax.set_ylim(0, None)
    shade_peaks(ax)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("시각 (1시간 구간 시작)")
    ax.set_ylabel("교통량 합계 (천 대/시)")
    ax.legend(loc="lower right")
    titles(fig, "시간대별 교통량 (서울시 교통량 조사지점 합계)",
           f"2026-10-14(수)·15(목) · 교통량 지점 {v.spot_num.nunique()}개 양방향 합계 · 단위 천 대/시")
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    save(fig, "A04_시간대별_교통량", "서울 열린데이터광장 서울시 교통량 이력 정보(VolInfo)",
         t[["날짜", "hour_start", "vol"]])


def fig_smart_daytype():
    s = pd.read_parquet(PROC / "smart_intersection_hourly.parquet")
    day = s.groupby(["date", "day_type", "hour_start"]).vol.sum().reset_index()
    t = day.groupby(["day_type", "hour_start"]).vol.mean().reset_index()
    n = day.groupby("day_type").date.nunique()
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for i, dt in enumerate(["평일", "주말", "공휴일"]):
        g = t[t.day_type == dt]
        if g.empty:
            continue
        ax.plot(g.hour_start, g.vol, color=SERIES[i], label=f"{dt} ({n[dt]}일)")
        r = g.loc[g.vol.idxmax()]
        ax.annotate(f"{r.vol:,.0f}", (r.hour_start, r.vol), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=9, color=INK2)
    ax.set_ylim(0, None)
    shade_peaks(ax)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("시각 (1시간 구간 시작)")
    ax.set_ylabel("교차로 통과 교통량 (대/시, 일평균)")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.legend(loc="upper left")
    titles(fig, "평일·주말·공휴일 시간대별 교차로 교통량",
           f"{s.date.min():%Y-%m-%d} ~ {s.date.max():%m-%d} · 노원구 스마트교차로 {s.CROSS_ID.nunique()}곳 합계 · 서울 전역 대표 아님")
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    save(fig, "A05_요일유형별_교차로교통량", "서울 교통빅데이터플랫폼 스마트교차로 1시간 교통량", t)


def main():
    setup()
    s, links, gu = load()
    fig_hourly_speed(s)
    fig_hourly_jam(s)
    t = fig_gu_maps(s, gu)
    fig_am_pm_gu(t)
    fig_link_map(s, links, gu, hour=8)
    fig_top20(s)
    fig_hotspots(s, links, gu)
    fig_volume_hourly()
    fig_smart_daytype()


if __name__ == "__main__":
    main()
