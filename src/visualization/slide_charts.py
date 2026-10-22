"""기획서·발표자료용 그래프(공모전 포스터 색). 수치는 outputs/tables 원본만 사용.

사용: python slide_charts.py [출력폴더] [크기.json]
크기.json = {"이름[@변형]": [가로mm, 세로mm], ...} — 문서의 그림 칸 크기에 맞춰 같은 비율로 그린다.
없으면 기본 크기로 전부 그린다.
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "outputs/tables"
PROC = ROOT / "data/processed"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs/figures/slides"
OUT.mkdir(parents=True, exist_ok=True)
SIZES = json.loads(Path(sys.argv[2]).read_text()) if len(sys.argv) > 2 else {}
K = 1.75  # mm 칸 → 그림 인치 배율(글자 크기 15pt 내외가 문서에서 약 8pt로 보이도록)

BLUE, NAVY, BODY, MUTED = "#125CFC", "#0F2A8C", "#2B3A55", "#4F6285"
LIGHT, PALE, GRID, ORANGE = "#9DBEFD", "#D6E4FF", "#E3ECF8", "#E5531A"

mpl.rcParams.update({
    "font.family": "Apple SD Gothic Neo", "font.size": 15, "axes.unicode_minus": False,
    "text.color": BODY, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": "#B9C8E0", "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
    "grid.linewidth": 1, "xtick.major.size": 0, "ytick.major.size": 0, "xtick.labelsize": 13,
    "ytick.labelsize": 13, "figure.facecolor": "white", "axes.axisbelow": True, "axes.facecolor": "white",
    "legend.frameon": False, "figure.constrained_layout.use": True,
})


def csv(name, **kw):
    return pd.read_csv(T / name, encoding="utf-8-sig", **kw)


def hgrid(ax):
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)


def hourly(fs):
    a = csv("fig_A02_시간대별_혼잡도로비율.csv")
    fig, ax = plt.subplots(figsize=fs)
    x, y = a["시작시각"], a["정체비율_pct"]
    for lo, hi in [(7.5, 9.5), (16.5, 19.5)]:
        ax.axvspan(lo, hi, color="#EEF4FF", zorder=0, lw=0)
    ax.fill_between(x, y, color=BLUE, alpha=0.12, lw=0)
    ax.plot(x, y, color=BLUE, lw=3, marker="o", ms=5)
    for h, lab in [(8, "출근 08시 19.3%"), (18, "퇴근 18시 27.8%")]:
        v = y[x == h].iloc[0]
        ax.plot(h, v, "o", ms=11, color=BLUE, mec="white", mew=2.5, zorder=5)
        ax.annotate(lab, (h, v), xytext=(0, 14), textcoords="offset points", ha="center",
                    fontsize=17, fontweight="bold", color=NAVY)
    ax.set_xticks(range(0, 24, 2), [f"{h}시" for h in range(0, 24, 2)])
    ax.set_ylim(0, 33)
    ax.set_yticks([0, 10, 20, 30], ["0%", "10%", "20%", "30%"])
    ax.set_xlim(-0.5, 23.5)
    return fig


def subway(fs):
    c = csv("fig_C01_지하철_시간대별_승하차.csv")
    fig, ax = plt.subplots(figsize=fs)
    w = 0.4
    ax.bar(c.hour_start - w / 2, c.board, w, color=PALE, label="승차")
    ax.bar(c.hour_start + w / 2, c.alight, w, color=BLUE, label="하차")
    ax.set_xticks(range(0, 24, 3), [f"{h}시" for h in range(0, 24, 3)])
    ax.set_yticks([0, 20, 40, 60], ["0", "20만", "40만", "60만"])
    ax.legend(loc="upper left", fontsize=13, ncol=2)
    ax.set_ylim(0, 72)
    return fig


def weekday(fs):
    d2 = csv("fig_D02_요일공휴일별_출근교통량.csv")
    fig, ax = plt.subplots(figsize=fs)
    ax.bar(d2.weekday, d2["mean"] / 1e4, color=[BLUE] * 5 + [LIGHT] * 3, width=0.66)
    for i, v in enumerate(d2["mean"] / 1e4):
        ax.text(i, v + 1.5, f"{v:.0f}", ha="center", fontsize=13, color=NAVY, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100], ["0", "50만", "100만"])
    ax.tick_params(axis="x", labelsize=14)
    return fig


def monthly(fs):
    d4 = csv("fig_D04_월별_출퇴근_교통량.csv", parse_dates=["month"])
    fig, ax = plt.subplots(figsize=fs)
    for pk, col in [("출근", BLUE), ("퇴근", LIGHT)]:
        s = d4[d4.peak == pk]
        ax.plot(s.month, s.vol / 1e4, color=col, lw=2.6, label=pk)
    ax.set_ylim(75, 95)
    ax.set_yticks([75, 80, 85, 90, 95], ["75만", "80만", "85만", "90만", "95만"])
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%y.%m"))
    ax.xaxis.set_major_locator(mpl.dates.MonthLocator([1, 7]))
    ax.legend(loc="lower left", ncol=2, fontsize=13)
    return fig


def rain(fs):
    d3 = csv("fig_D03_강수여부별_교통량.csv")
    r = d3[d3["강수"] != "무강수"].pivot(index="hour_start", columns="강수", values="무강수대비_pct")
    fig, ax = plt.subplots(figsize=fs)
    xx = np.arange(len(r))
    ax.bar(xx - 0.19, r["약한 비(0~3mm]"], 0.38, color=LIGHT, label="약한 비 0~3mm")
    ax.bar(xx + 0.19, r["비(3mm 초과)"], 0.38, color=BLUE, label="비 3mm 초과")
    ax.set_xticks(xx, [f"{h:02d}시" for h in r.index])
    ax.set_ylim(-15, 0)
    ax.set_yticks([-12, -8, -4, 0], ["-12%", "-8%", "-4%", "0%"])
    ax.legend(loc="lower right", fontsize=12, ncol=2)
    ax.axhline(0, color="#B9C8E0", lw=1)
    return fig


GU = gpd.read_file(PROC / "seoul_gu.gpkg")
NAME = next(cn for cn in GU.columns if GU[cn].astype(str).str.endswith("구").all())
NUDGE = {"중구": (0, 300), "성북구": (-1800, -250), "동대문구": (600, -900), "서대문구": (600, -1200), "종로구": (-200, 700)}


def gu_speed(fs):
    sp = csv("fig_A03_자치구별_출근퇴근_속도비교.csv").set_index("자치구")
    gu = GU.assign(v=GU[NAME].map(sp["출근_평균속도"]))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("b", ["#0F2A8C", "#125CFC", "#9DBEFD", "#EEF4FF"])
    fig, ax = plt.subplots(figsize=fs)
    gu.plot(column="v", cmap=cmap, vmin=20, vmax=29.5, ax=ax, edgecolor="white", linewidth=1.4)
    for _, rw in gu.iterrows():
        p = rw.geometry.representative_point()
        dx, dy = NUDGE.get(rw[NAME], (0, 0))
        dark = rw.v < 23.5 and rw[NAME] != "성북구"
        sep = " " if rw[NAME] == "중구" else "\n"
        ax.text(p.x + dx, p.y + dy, f"{rw[NAME][:-1]}{sep}{rw.v:.0f}", ha="center", va="center", fontsize=10,
                color="white" if dark else NAVY, fontweight="bold", linespacing=1.1)
    ax.set_axis_off()
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(20, 29.5))
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.01, aspect=40)
    cb.set_label("평균속도(km/h) · 진할수록 느림", fontsize=12, color=MUTED)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=11)
    return fig


def inflow(fs):
    ni = csv("fig_C04_자치구별_출근시간_대중교통_순유입.csv").sort_values("순유입")
    fig, ax = plt.subplots(figsize=fs)
    hgrid(ax)
    v = ni["순유입"] / 1e4
    ax.barh(ni["자치구"].str[:-1], v, color=[BLUE if q > 0 else ORANGE for q in v], height=0.7)
    for yi, q in enumerate(v):
        ax.text(q + (0.3 if q > 0 else -0.3), yi, f"{q:+.1f}", va="center", ha="left" if q > 0 else "right",
                fontsize=11, color=BODY)
    ax.set_xlim(-6.5, 18)
    ax.set_ylim(-0.6, len(ni) - 0.4)
    ax.set_xticks([-5, 0, 5, 10, 15], ["-5만", "0", "+5만", "+10만", "+15만"], fontsize=12)
    ax.tick_params(axis="y", labelsize=11.5)
    ax.axvline(0, color="#B9C8E0", lw=1)
    return fig


def candidates(fs):
    full = csv("fig_C06_추가대중교통수요_후보자치구.csv")
    c6 = full.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=fs)
    hgrid(ax)
    top = c6["자치구"].isin(full.head(8)["자치구"])
    ax.barh(c6["자치구"].str[:-1], c6["지표"], color=[BLUE if t else PALE for t in top], height=0.68)
    for yi, (_, rw) in enumerate(c6.iterrows()):
        ax.text(rw["지표"] + 0.05, yi, f"{rw['출근_평균속도']:.0f}km/h · 순유입 {rw['순유입'] / 1e4:+.1f}만",
                va="center", fontsize=11.5, color=BODY)
    ax.set_xlim(0, 4.1)
    ax.set_ylim(-0.6, len(c6) - 0.4)
    ax.set_xticks([0, 1, 2])
    ax.tick_params(axis="y", labelsize=13)
    return fig


def mae(fs):
    e = csv("fig_E01_예측모델_성능비교.csv")
    e = e[e.split == "test"].copy()
    e["lab"] = e.model.str.replace("베이스라인: ", "기준: ")
    fig, ax = plt.subplots(figsize=fs)
    hgrid(ax)
    order = e.iloc[::-1]
    ax.barh(order.lab, order.MAE, color=[PALE if m.startswith("기준") else BLUE for m in order.lab], height=0.64)
    for yi, m in enumerate(order.MAE):
        ax.text(m + 6, yi, f"{m:.0f}", va="center", fontsize=15, fontweight="bold", color=NAVY)
    ax.set_xlim(0, 420)
    ax.set_xlabel("MAE(대/시) · 낮을수록 정확", fontsize=12)
    ax.tick_params(axis="y", labelsize=13.5)
    return fig


def shap(fs):
    s = csv("fig_E02_예측영향요인_SHAP.csv").head(6).iloc[::-1]
    fig, ax = plt.subplots(figsize=fs)
    hgrid(ax)
    ax.barh(s["이름"], s["평균|SHAP|"], color=[LIGHT] * 5 + [BLUE], height=0.64)
    ax.tick_params(axis="y", labelsize=12.5)
    ax.tick_params(axis="x", labelsize=11)
    ax.set_xlabel("평균 |SHAP| (예측 기여도)", fontsize=11.5)
    return fig


def scenario(fs):
    f = csv("fig_F01_시나리오비교_출근.csv")
    f = f[f["지표"] == "속도변화_pct"]
    scen = ["B 전역 홀짝제", "C 혼잡지역 선택 관리", "D 버스 증편", "E 선택 관리+버스"]
    fig, axes = plt.subplots(1, 2, figsize=fs, sharey=True)
    for ax, (mdl, xmax) in zip(axes, [("탄력성", 20), ("BPR", 50)]):
        hgrid(ax)
        g = f[f["속도모형"] == mdl].set_index("시나리오").loc[scen]
        for yi, (sn, rw) in enumerate(g.iloc[::-1].iterrows()):
            col = ORANGE if sn.startswith("B") else BLUE
            ax.plot([rw["낮음"], rw["높음"]], [yi, yi], color=col, lw=10, alpha=0.3, solid_capstyle="round")
            ax.plot(rw["중간"], yi, "o", ms=12, color=col, mec="white", mew=2)
            ax.text(rw["높음"] + xmax * 0.03, yi, f"{rw['낮음']:.1f}~{rw['높음']:.1f}%", va="center",
                    fontsize=13, color=BODY)
        ax.set_yticks(range(4), [s_.split(" ", 1)[1] for s_ in scen[::-1]], fontsize=13.5)
        ax.set_xlim(0, xmax * 1.35)
        ax.set_ylim(-0.6, 3.6)
        ax.set_title(f"{'① 실측 탄력성' if mdl == '탄력성' else '② BPR 함수'}", fontsize=14,
                     color=NAVY, fontweight="bold", loc="left")
        ax.set_xlabel("평균속도 변화(%)", fontsize=12)
    return fig


def link_map(hour):
    def draw(fs):
        spd = pd.read_parquet(PROC / "speed_hourly.parquet")
        spd = spd[~spd.speed_outlier & spd.length_m.notna()]
        links = gpd.read_file(PROC / "seoul_service_links.gpkg")
        stat = {"원활": ("#BFD3F5", 0.5), "서행": ("#F7B377", 0.6), "정체": ("#D7263D", 1.1)}
        h = spd[spd.hour_start == hour].groupby("link_id").agg(speed=("speed", "mean"), road_class=("road_class", "first"))
        exp = h.road_class == "도시고속도로"
        h["st"] = np.select([h.speed < np.where(exp, 30, 15), h.speed < np.where(exp, 70, 25)], ["정체", "서행"], "원활")
        m = links.merge(h, left_on="link_id", right_index=True)
        fig, ax = plt.subplots(figsize=fs)
        GU.plot(ax=ax, color="#F1F5FC", edgecolor="white", linewidth=1.2)
        tot = m.length_m.sum()
        for k in ["원활", "서행", "정체"]:
            d = m[m.st == k]
            d.plot(ax=ax, color=stat[k][0], linewidth=stat[k][1], label=f"{k} {d.length_m.sum() / tot * 100:.0f}%")
        ax.legend(loc="lower right", fontsize=12, handlelength=1.2)
        ax.set_axis_off()
        return fig
    return draw


PM_NUDGE = {"도봉구": (-500, 1200), "강북구": (-1500, -300), "노원구": (900, -600)}  # 북쪽 세 구 이름 겹침 방지


def policy_map(fs):
    tg = csv("sim_targets.csv").set_index("항목")["값"]
    am_t = [g.strip() for g in tg["선택관리 대상 자치구(출근)"].split(",")]
    pm_t = [g.strip() for g in tg["선택관리 대상 자치구(퇴근)"].split(",")]
    bus = [g.strip() for g in tg["버스 증편 대상 자치구"].split(",")]
    stage1 = [g for g in am_t if g in pm_t and g in ("종로구", "중구")]
    stage2 = [g for g in dict.fromkeys(am_t + pm_t) if g not in stage1]
    bus1, bus2 = bus[:4], bus[4:]

    def fill(g):
        if g in stage1:
            return NAVY
        if g in bus1:
            return "#FFD43B"
        if g in bus2:
            return "#FFF0A8"
        if g in stage2:
            return "#BFD3F5"
        return "#F1F5FC"

    fig, ax = plt.subplots(figsize=fs)
    GU.plot(ax=ax, color=[fill(g) for g in GU[NAME]], edgecolor="white", linewidth=1.4)
    GU[GU[NAME].isin(stage2) & GU[NAME].isin(bus)].plot(ax=ax, facecolor="none", edgecolor=BLUE, hatch="///", linewidth=0)
    for _, rw in GU.iterrows():
        g = rw[NAME]
        if g not in stage1 + stage2 + bus:
            continue
        pt = rw.geometry.representative_point()
        dx, dy = {**NUDGE, **PM_NUDGE}.get(g, (0, 0))
        ax.text(pt.x + dx, pt.y + dy, g[:-1], ha="center", va="center", fontsize=9.5, fontweight="bold",
                color="white" if g in stage1 else NAVY,
                bbox=None if g in stage1 else dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    ax.set_axis_off()
    hd = [Patch(color=NAVY, label="1단계 첨두 수요관리"), Patch(color="#FFD43B", label="1단계 버스 재배분 검토"),
          Patch(color="#BFD3F5", label="2단계 관리 확대 후보"), Patch(color="#FFF0A8", label="2단계 버스 재배분 검토"),
          Patch(facecolor="#FFF0A8", edgecolor=BLUE, hatch="///", lw=0, label="2단계 관리·버스 검토")]
    fig.legend(handles=hd, loc="outside right center", fontsize=10.5, ncol=1, handlelength=1.2)
    return fig


def cbd_subway(fs):
    """종로·중구 지하철 시간대별 하차(2025 일평균)와 선택 관리 시 대중교통 증가분(가정 낮음~높음)."""
    s = pd.read_parquet(PROC / "subway_monthly_hourly.parquet", columns=["ym", "hour_start", "daily_alight", "SIGUNGU_NM"])
    s = s[s.ym.astype(str).str.startswith("2025") & s.SIGUNGU_NM.isin(["종로구", "중구"])]
    h = (s.groupby("hour_start").daily_alight.sum() / 12 / 1e4).loc[6:11]
    fig, ax = plt.subplots(figsize=fs)
    ax.bar(h.index, h, color=[BLUE if x in (8, 9) else LIGHT for x in h.index], width=0.64)
    for x in (8, 9):
        ax.bar(x, h[x] * 0.07, bottom=h[x], color=ORANGE, alpha=0.35, width=0.64, hatch="///", edgecolor=ORANGE, lw=0)
    for x, v in h.items():
        ax.text(x, v + (h[x] * 0.07 if x in (8, 9) else 0) + 0.25, f"{v:.1f}", ha="center", fontsize=13, color=NAVY, fontweight="bold")
    ax.annotate("", xy=(7, h[7] + 2.2), xytext=(8, h[8] - 3), arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=2,
                connectionstyle="arc3,rad=0.35"))
    ax.text(7.05, h[7] + 3.2, "시차출근으로\n앞당겨 분산", fontsize=12.5, color=NAVY, fontweight="bold", ha="center")
    ax.set_xticks(h.index, [f"{x:02d}시" for x in h.index])
    ax.set_ylim(0, h.max() * 1.22)
    ax.set_ylabel("하차(만 명/일)", fontsize=12)
    ax.legend(handles=[Patch(color=BLUE, label="출근 첨두 하차"),
                       Patch(facecolor=ORANGE, alpha=0.35, hatch="///", edgecolor=ORANGE, lw=0, label="선택 관리 시 증가분(최대 +7%)")],
              loc="upper right", fontsize=11.5)
    return fig


def sim_map(scen):
    """시나리오별 출근 첨두 구간 속도 변화(중간 가정, BPR 모형) — 가정 기반."""
    def draw(fs):
        d = csv("sim_links_출근_중간.csv", dtype={"link_id": str})
        d = d[(d["시나리오"] == scen) & (d["속도모형"] == "BPR")]
        links = gpd.read_file(PROC / "seoul_service_links.gpkg")[["link_id", "geometry"]].merge(d, on="link_id")
        cmap = mpl.colors.LinearSegmentedColormap.from_list("d", ["#D7263D", "#F3F5F9", "#9DBEFD", "#125CFC", "#0F2A8C"])
        norm = mpl.colors.TwoSlopeNorm(vmin=-5, vcenter=0, vmax=30)
        fig, ax = plt.subplots(figsize=fs)
        GU.plot(ax=ax, color="#F1F5FC", edgecolor="white", linewidth=1.2)
        links.sort_values("chg").plot(ax=ax, column="chg", cmap=cmap, norm=norm, linewidth=1.0)
        ax.set_axis_off()
        cb = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, orientation="horizontal",
                          fraction=0.045, pad=0.01, aspect=35, ticks=[-5, 0, 10, 20, 30])
        cb.ax.set_xticklabels(["-5%", "0", "+10%", "+20%", "+30%↑"], fontsize=11)
        cb.set_label("출근 08~10시 구간 속도 변화 · BPR 모형 · 중간 가정(가정 기반) · 빨강 = 느려짐", fontsize=10.5, color=MUTED)
        cb.outline.set_visible(False)
        return fig
    return draw


def env_bars(fs):
    """승용차 감소·CO2 — B vs E (가정 낮음~높음, 점 = 중간). 산식은 기획서 참고 자료 쪽."""
    r = csv("sim_results.csv")
    occ, shift = {"낮음": 1.2, "중간": 1.3, "높음": 1.4}, {"낮음": 0.5, "중간": 0.6, "높음": 0.7}
    rows = []
    for sc, lab in [("B", "전역 홀짝제"), ("E", "서울형")]:
        d = r[r["시나리오"].str.startswith(sc) & (r["속도모형"] == "탄력성")]
        for lv in ["낮음", "중간", "높음"]:
            add = d[d["가정수준"] == lv]["대중교통_추가수요_인시"].sum()  # 출근+퇴근, 시간당
            cars = 2 * add / shift[lv] / occ[lv]  # 첨두 4시간(출근 2 + 퇴근 2)
            rows.append({"lab": lab, "lv": lv, "cars": cars / 1e4, "co2": cars * 250 * 10 * 0.17 / 1e3 / 1e4})
    t = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=fs)
    for ax, col, title, unit in [(axes[0], "cars", "하루 승용차 운행 감소", "만 대"), (axes[1], "co2", "연간 CO2 감축", "만 톤")]:
        hgrid(ax)
        for yi, lab in enumerate(["서울형", "전역 홀짝제"]):
            g = t[t.lab == lab].set_index("lv")[col]
            c = BLUE if lab == "서울형" else ORANGE
            ax.plot([g["낮음"], g["높음"]], [yi, yi], color=c, lw=14, alpha=0.28, solid_capstyle="round")
            ax.plot(g["중간"], yi, "o", ms=13, color=c, mec="white", mew=2)
            ax.text(g["높음"] * 1.04 + 0.2, yi, f"{g['중간']:.1f}{unit}  ({g['낮음']:.1f}~{g['높음']:.1f})",
                    va="center", fontsize=12.5, color=NAVY, fontweight="bold")
        ax.set_yticks([0, 1], ["서울형", "전역 홀짝제"], fontsize=13.5)
        ax.set_ylim(-0.6, 1.6)
        ax.set_xlim(0, t[col].max() * 1.75)
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color=NAVY)
    return fig


def cf_detect(fs):
    """가짜 효과 주입 실험 — 4주 시범 창에서 방법별 검출률."""
    t = csv("cf_injection.csv")
    x = -t["넣은 효과(%)"]
    fig, ax = plt.subplots(figsize=fs)
    for col, lab, c, lw in [("AI 검출률(%)", "AI 반사실 모델", BLUE, 3.4), ("이중차분(비교 구 대비) 검출률(%)", "이중차분", LIGHT, 2.4),
                            ("단순 전후 비교 검출률(%)", "단순 전후 비교", "#B9C3D6", 2.4)]:
        ax.plot(x, t[col], marker="o", ms=8, lw=lw, color=c, label=lab, mec="white", mew=1.5)
    for xi, yi in zip(x, t["AI 검출률(%)"]):
        ax.text(xi, yi + 5, f"{yi:.0f}%", ha="center", fontsize=12, color=NAVY, fontweight="bold")
    ax.axvspan(7, 16, color="#EEF4FF", zorder=0, lw=0)
    ax.text(11.2, 8, "시뮬레이션 가정 범위\n(관리 구역 교통량 -7~16%)", ha="center", fontsize=11.5, color=BLUE, fontweight="bold")
    ax.set_xticks([v for v in x if v != 3], [f"-{v:.0f}%" for v in x if v != 3])  # -2·-3% 눈금 겹침 방지(점은 모두 표시)
    ax.set_xlabel("넣은 가상 효과(관리 구역 첨두 교통량 감소, 모의실험)", fontsize=12)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100], ["0", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("검출률(4주 창 106개)", fontsize=12)
    ax.legend(loc="upper left", fontsize=11.5)
    return fig


def cf_demo(fs):
    """가상 시범 예시 — 실측 교통량(2025 하반기)에 시범 시작 후 -10%를 적용해 AI가 효과를 읽어내는 과정."""
    d = csv("cf_test_daily.csv", parse_dates=["date"]).set_index("date").iloc[:60]
    pre, post = d.iloc[:20], d.iloc[20:]
    calib = pre.y.sum() / pre.pred.sum()  # 시행 직전 4주로 수준 보정(counterfactual_eval.py와 같은 규칙)
    cf = d.pred * calib / 1e4
    act = pd.concat([pre.y, post.y * 0.9]) / 1e4
    est = (act.iloc[20:].sum() / cf.iloc[20:].sum() - 1) * 100
    fig, ax = plt.subplots(figsize=fs)
    ax.axvspan(post.index[0], post.index[-1], color="#EEF4FF", lw=0, zorder=0)
    ax.fill_between(post.index, act.iloc[20:], cf.iloc[20:], color="#9DBEFD", alpha=0.45, lw=0, zorder=1)
    ax.plot(d.index, cf, color=BLUE, lw=2.4, ls=(0, (4, 2)), label="AI 반사실: 정책이 없었다면", zorder=3)
    ax.plot(d.index, act, color=NAVY, lw=2.4, marker="o", ms=4, label="실측 × 0.9 (가상 효과 적용)", zorder=4)
    ax.text(pre.index[10], ax.get_ylim()[1], "시행 전 4주", ha="center", va="top", fontsize=12, color=MUTED)
    ax.text(post.index[20], ax.get_ylim()[1], "시범 운영", ha="center", va="top", fontsize=12, color=BLUE, fontweight="bold")
    mid = post.index[30]
    ax.annotate(f"차이 = AI 추정 효과 {est:.1f}%\n(넣은 가상 효과 -10%)", xy=(mid, (act[mid] + cf[mid]) / 2), xytext=(post.index[22], act.min() + 0.1),
                fontsize=12.5, color=NAVY, fontweight="bold", arrowprops=dict(arrowstyle="-", color=NAVY, lw=1))
    ax.set_ylabel("관리 구역 첨두 교통량(만 대/일)", fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.set_ylim(act.min() - 0.8, max(act.max(), cf.max()) + 0.8)
    ax.legend(loc="lower left", fontsize=11.5, ncols=2)
    return fig


def cover_map(fs):
    """표지용 — 퇴근 18시대 소통 등급을 포스터 색(파랑·주황)으로, 배경 투명."""
    spd = pd.read_parquet(PROC / "speed_hourly.parquet")
    spd = spd[~spd.speed_outlier & spd.length_m.notna()]
    links = gpd.read_file(PROC / "seoul_service_links.gpkg")
    h = spd[spd.hour_start == 18].groupby("link_id").agg(speed=("speed", "mean"), road_class=("road_class", "first"))
    exp = h.road_class == "도시고속도로"
    h["st"] = np.select([h.speed < np.where(exp, 30, 15), h.speed < np.where(exp, 70, 25)], ["정체", "서행"], "원활")
    m = links.merge(h, left_on="link_id", right_index=True)
    fig, ax = plt.subplots(figsize=fs)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    GU.plot(ax=ax, color="white", edgecolor="#D6E4FA", linewidth=1.2)
    for k, c, lw in [("원활", "#9DBEFD", 0.6), ("서행", "#125CFC", 0.8), ("정체", "#FF7A1A", 1.3)]:
        m[m.st == k].plot(ax=ax, color=c, linewidth=lw)
    ax.set_axis_off()
    return fig


def subway_cong(fs):
    """평일 30분 단위 정원(100%) 초과 역·방향 수 — 서울교통공사 혼잡도(2026-06 기준본)."""
    a = csv("subway_congestion_profile.csv")
    x = range(len(a))
    fig, ax = plt.subplots(figsize=fs)
    col = [BLUE if t in ("7시30분", "8시00분", "8시30분", "18시00분", "18시30분") else PALE for t in a["시각"]]
    ax.bar(x, a["정원초과_구간수"], color=col, width=0.78)
    for t, lab in [("8시00분", "08:00 {n}곳\n최고 {m:.0f}%"), ("18시00분", "18:00 {n}곳\n최고 {m:.0f}%")]:
        i = a.index[a["시각"] == t][0]
        n = a["정원초과_구간수"][i]
        m = a.loc[a["시각"].isin([t, "7시30분", "8시30분"] if t.startswith("8") else [t, "18시30분"]), "최고"].max()
        ax.annotate(lab.format(n=n, m=m), (i, n), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=14,
                    fontweight="bold", color=NAVY)
    tick = [i for i, t in enumerate(a["시각"]) if t.endswith("00분") and int(t.split("시")[0]) % 3 == 0]
    ax.set_xticks(tick, [a["시각"][i].split("시")[0] + "시" for i in tick])
    ax.set_ylim(0, a["정원초과_구간수"].max() * 1.35)
    ax.set_ylabel("정원 초과 역·방향 수 (557곳 중)", fontsize=12.5)
    return fig


def bus_realloc(fs):
    """노선 구분별 노선 수 비중 vs 출근 첨두 승객 비중(같은 단위 %)."""
    g = csv("bus_route_realloc_summary.csv").set_index("구분")
    order = [("증회(업무지구행 고수요)", "업무지구행\n고수요 노선"), ("배차 조정(저수요)", "저수요 노선")]
    fig, ax = plt.subplots(figsize=fs)
    hgrid(ax)
    y = np.arange(len(order))
    for k, (col, lab, c) in enumerate([("노선_비중_pct", "노선 수 비중", PALE), ("승객_비중_pct", "출근 첨두 승객 비중", BLUE)]):
        v = [g.loc[o, col] for o, _ in order]
        ax.barh(y + (k - 0.5) * 0.36, v, height=0.34, color=c, label=lab)
        for yi, vi in zip(y, v):
            ax.text(vi + 0.4, yi + (k - 0.5) * 0.36, f"{vi:.1f}%", va="center", fontsize=14, color=NAVY, fontweight="bold")
    ax.set_yticks(y, [l for _, l in order], fontsize=14)
    ax.invert_yaxis()
    ax.set_xlim(0, 24)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.legend(loc="lower right", fontsize=12.5)
    return fig


CHARTS = {  # 이름: (함수, 기본 크기 inch)
    "c_hourly": (hourly, (10.4, 4.6)), "c_subway": (subway, (6.2, 3.4)), "c_weekday": (weekday, (5.6, 3.9)),
    "c_monthly": (monthly, (5.6, 3.9)), "c_rain": (rain, (5.6, 3.9)), "c_map": (gu_speed, (7.2, 5.6)),
    "c_inflow": (inflow, (6.4, 7.0)), "c_candidates": (candidates, (6.4, 5.6)), "c_mae": (mae, (8.6, 4.4)),
    "c_shap": (shap, (5.4, 3.6)), "c_scenario": (scenario, (10.4, 4.2)), "c_link_am": (link_map(8), (6.4, 5.0)),
    "c_link_pm": (link_map(18), (6.4, 5.0)), "c_cbd_subway": (cbd_subway, (6.4, 4.2)),
    "c_sim_B": (sim_map("B"), (6.4, 5.4)), "c_sim_E": (sim_map("E"), (6.4, 5.4)), "c_env": (env_bars, (10, 3.4)),
    "c_cf_detect": (cf_detect, (7, 4.6)), "c_cf_demo": (cf_demo, (8, 4.2)), "c_cover_map": (cover_map, (6.4, 5.4)), "c_subway_cong": (subway_cong, (8, 4)), "c_bus_realloc": (bus_realloc, (7, 3.6)), "c_policy_map": (policy_map, (7.2, 6.0)),
}

jobs = ({k: (k.split("@")[0], (w / 25.4 * K, h / 25.4 * K)) for k, (w, h) in SIZES.items()} if SIZES
        else {k: (k, fs) for k, (_, fs) in CHARTS.items()})
for out_name, (base, fs) in jobs.items():
    fig = CHARTS[base][0](fs)
    fig.savefig(OUT / f"{out_name.replace('@', '_')}.png", dpi=220, transparent=base == "c_cover_map")
    plt.close(fig)
print("ok", len(jobs))
