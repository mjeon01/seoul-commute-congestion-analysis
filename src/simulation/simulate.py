"""정책 시뮬레이션 (가정 기반 민감도 분석) — 시나리오 A~E.

※ 실측 결과가 아님. 관측 속도(2026-10-14·15)를 기준선(A)으로 두고, config/simulation.yaml의 가정값으로
  링크별 교통량 변화율을 만든 뒤 두 가지 속도 반응 모형으로 속도를 다시 계산한다.

  방법 1 (실측 탄력성): v1 = v0 · m^ε, ε = 지점 내 속도-교통량 로그-로그 탄력성(도로 등급별, EDA D01)
  방법 2 (BPR): t = t0(1 + α x^β). 기준 통행시간비 t/t0로 x0를 역산 → x1 = x0·m → 새 속도
  (m = 링크 교통량 배수, 1보다 작으면 감소)

지표: 첨두 평균속도(공간평균 = 총연장/총통행시간), 통행시간 변화, 정체 연장비율, 주행거리(VKT) 변화율(CO2 대리 지표,
      속도에 따른 배출계수 변화 미반영), 대중교통 추가수요(인/시), 자치구별 증가율·잠재 초과수요, 취약지역 영향.

실행: .venv/bin/python src/simulation/simulate.py
출력: outputs/tables/sim_*.csv, outputs/figures/F*_*.png
"""
import sys
from itertools import product
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from visualization.style import AXIS, DIVERGE, GRAY, INK, INK2, SERIES, save, setup, titles  # noqa: E402

PROC = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
CFG = yaml.safe_load((ROOT / "config" / "simulation.yaml").read_text())
BASE = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
LEVELS = ["low", "mid", "high"]
LEVEL_KO = {"low": "낮음", "mid": "중간", "high": "높음"}
SCEN = {"A": "A 현행 유지", "B": "B 전역 홀짝제", "C": "C 혼잡지역 선택 관리", "D": "D 버스 증편", "E": "E 선택 관리+버스"}
PERIODS = {"출근": CFG["peak_hours"]["am"], "퇴근": CFG["peak_hours"]["pm"]}
SHOULDER = {"출근": {8: 7, 9: 10}, "퇴근": {17: 16, 18: 19}}  # 홀짝제 시간 이동 시 옮겨갈 시각
SRC = "가정 기반 시뮬레이션 — 기준선: 서울 교통빅데이터플랫폼 소통정보(2026-10-14·15), 지하철·버스 승하차(2025), 가정값 config/simulation.yaml"


# ---------------------------------------------------------------- 기준선
def baseline():
    s = pd.read_parquet(PROC / "speed_hourly.parquet")
    s = s[~s.speed_outlier & s.length_m.notna() & s.sigungu_nm.notna()]
    night = s[s.hour_start <= 4].groupby("link_id").speed.max()
    links = s.groupby("link_id").agg(length_m=("length_m", "first"), road_class=("road_class", "first"),
                                     gu=("sigungu_nm", "first"), max_spd=("max_spd", "first"))
    hours = sorted({h for p in PERIODS.values() for h in p} | {h for d in SHOULDER.values() for h in d.values()})
    b = s[s.hour_start.isin(hours)].groupby(["link_id", "hour_start"]).speed.mean().rename("v_base").reset_index()
    b = b.join(links, on="link_id")
    # 자유속도: 야간 최대(제한속도 상한), 기준 속도보다 낮으면 해당 링크 관측 최대속도
    vmax_obs = s.groupby("link_id").speed.max()
    v0 = np.minimum(b.link_id.map(night), b.max_spd.fillna(np.inf))
    b["v_free"] = np.maximum(v0.fillna(b.link_id.map(vmax_obs)), b.v_base)
    exp = b.road_class.eq("도시고속도로")
    c = BASE["congestion"]
    b["jam_thr"] = np.where(exp, c["urban_expressway"]["jam"], c["general"]["jam"])
    return b


def hourly_volume_ratio():
    """시간대 교통량 상대비(교통량 지점 합계, 2026-10-14·15) — 홀짝제 시간 이동량을 앞뒤 시각 교통량 대비로 환산."""
    v = pd.read_parquet(PROC / "volume_hourly.parquet")
    return v.groupby("hour_start").vol.sum() / v.date.nunique()


def transit_by_gu(hours):
    """2025년 일평균, 해당 시각 1시간당 자치구별 대중교통 승차·하차(지하철+버스)."""
    out = []
    for f, cols in [("subway_monthly_hourly.parquet", None), ("bus_monthly_hourly.parquet", None)]:
        d = pd.read_parquet(PROC / f, columns=["ym", "hour_start", "board", "alight", "SIGUNGU_NM"])
        d = d[d.ym.str.startswith("2025") & d.hour_start.isin(hours) & d.SIGUNGU_NM.notna()]
        out.append(d.groupby("SIGUNGU_NM")[["board", "alight"]].sum() / 365 / len(hours))
    return (out[0] + out[1].reindex(out[0].index).fillna(0)).rename_axis("gu")


def target_gus(b, period, n):
    d = b[b.hour_start.isin(PERIODS[period])]
    share = d.assign(j=d.length_m * (d.v_base < d.jam_thr)).groupby("gu").apply(
        lambda g: g.j.sum() / g.length_m.sum(), include_groups=False)
    return list(share.nlargest(n).index), share


def adjacency():
    gu = gpd.read_file(PROC / "seoul_gu.gpkg").set_index("SIGUNGU_NM")
    # 경계 좌표의 미세한 차이로 touches가 빠지는 경우가 있어 20m 버퍼와 교차 여부로 판정
    adj = {g: set(gu.index[gu.intersects(geom.buffer(20))]) - {g} for g, geom in gu.geometry.items()}
    assert all(adj.values()), "인접 자치구가 없는 자치구가 있음"
    return adj


# ---------------------------------------------------------------- 교통량 배수
def multipliers(b, scen, lv, period, ctx):
    """링크·시각별 교통량 배수 m과 감소한 승용차 통행 비율(대중교통 수요 산정용) 반환."""
    m = pd.Series(1.0, index=b.index)
    cs = CFG["car_share"][lv]
    peak = b.hour_start.isin(PERIODS[period])
    info = {}
    if scen == "B":
        oe = CFG["odd_even"]
        restricted = cs * (1 - oe["exempt_share"][lv]) * 0.5 * oe["compliance"][lv]
        m[peak] *= 1 - restricted
        # 시간을 바꾼 차량은 앞뒤 시각 교통량에 더해짐(상대비로 환산)
        vr = ctx["vol_ratio"]
        for h, sh in SHOULDER[period].items():
            add = restricted * oe["retime_share"][lv] * vr[h] / vr[sh]
            m[b.hour_start == sh] *= 1 + add
        info["car_trip_reduction"] = restricted * (1 - oe["retime_share"][lv])
        info["scope_gus"] = "all"
        info["restricted"] = restricted
    if scen in ("C", "E"):
        sel = CFG["selective"]
        tg = ctx["targets"][period]
        r = cs * sel["reduction"][lv]
        in_t = b.gu.isin(tg)
        m[peak & in_t] *= 1 - r
        adj = set().union(*(ctx["adj"][g] for g in tg)) - set(tg)
        in_adj = b.gu.isin(adj)
        lt, la = b.loc[peak & in_t].length_m.sum(), b.loc[peak & in_adj].length_m.sum()
        m[peak & in_adj] *= 1 + r * sel["detour_share"][lv] * lt / la
        info["sel_reduction"] = r * (1 - sel["detour_share"][lv])
        info["sel_gus"] = tg
        info["detour_gus"] = sorted(adj)
    if scen in ("D", "E"):
        bus = CFG["bus"]
        r = cs * bus["car_reduction"][lv]
        m[peak & b.gu.isin(ctx["bus_gus"])] *= 1 - r
        info["bus_reduction"] = bus["car_reduction"][lv]
    return m, info


# ---------------------------------------------------------------- 속도 반응
def new_speed(b, m, method, lv):
    if method == "탄력성":
        eps = b.road_class.map({k: v[lv] for k, v in CFG["speed_model"]["elasticity"].items()})
        v = b.v_base * m ** eps
    else:
        a, be = CFG["speed_model"]["bpr"]["alpha"], CFG["speed_model"]["bpr"]["beta"]
        R = b.v_free / b.v_base
        x0 = ((R - 1).clip(lower=0) / a) ** (1 / be)
        v = b.v_free / (1 + a * (x0 * m) ** be)
    return np.minimum(v, b.v_free)


def indicators(b, v, m, hours):
    d = b.hour_start.isin(hours)
    L, v0, v1, mm = b.length_m[d], b.v_base[d], v[d], m[d]
    sms0, sms1 = L.sum() / (L / v0).sum(), L.sum() / (L / v1).sum()
    return {"평균속도_기준": sms0, "평균속도": sms1, "속도변화_pct": (sms1 / sms0 - 1) * 100,
            "통행시간변화_pct": ((L / v1).sum() / (L / v0).sum() - 1) * 100,
            "정체비율_기준_pct": (L * (v0 < b.jam_thr[d])).sum() / L.sum() * 100,
            "정체비율_pct": (L * (v1 < b.jam_thr[d])).sum() / L.sum() * 100,
            "VKT변화_pct": ((L * mm).sum() / L.sum() - 1) * 100}


def transit_demand(info, lv, period, ctx):
    """감소한 승용차 통행(인) 중 대중교통 전환분을 자치구별로 배분."""
    tr = ctx["transit"][period]
    ratio = CFG["car_to_transit_ratio"][lv]
    occ_shift = CFG["to_transit_share"][lv]
    car_persons = tr.board * ratio  # 출발지(승차) 기준 승용차 통행 인원 근사
    add = pd.Series(0.0, index=tr.index)
    if "car_trip_reduction" in info:
        add += car_persons * info["car_trip_reduction"] * occ_shift
    if "sel_reduction" in info:
        dest = tr.alight * ratio  # 도착지(하차) 기준 — 대상 지역으로 오는 통행
        add.loc[add.index.isin(info["sel_gus"])] += dest[dest.index.isin(info["sel_gus"])] * info["sel_reduction"] * occ_shift
    if "bus_reduction" in info:
        g = add.index.isin(ctx["bus_gus"])
        add.loc[g] += car_persons[g] * info["bus_reduction"] * occ_shift
    cap_add = pd.Series(0.0, index=tr.index)
    if "bus_reduction" in info:
        cap_add.loc[cap_add.index.isin(ctx["bus_gus"])] = CFG["bus"]["added_capacity"]
    pct = add / tr.board * 100
    return add, pct, cap_add


def area_changes(b, v, pk, scen, info, ctx, sm):
    """정책 대상 지역·우회 유입 지역 속도 변화(%)."""
    out = {"대상지역_속도변화_pct": np.nan, "우회유입지역_속도변화_pct": np.nan}
    if scen in ("C", "E"):
        t = b.gu.isin(info["sel_gus"])
        out["대상지역_속도변화_pct"] = (sm(t, v) / sm(t, b.v_base) - 1) * 100
        a = b.gu.isin(info["detour_gus"])
        out["우회유입지역_속도변화_pct"] = (sm(a, v) / sm(a, b.v_base) - 1) * 100
    elif scen == "D":
        t = b.gu.isin(ctx["bus_gus"])
        out["대상지역_속도변화_pct"] = (sm(t, v) / sm(t, b.v_base) - 1) * 100
    elif scen == "B":
        out["대상지역_속도변화_pct"] = (sm(pd.Series(True, index=b.index), v) / sm(pd.Series(True, index=b.index), b.v_base) - 1) * 100
    return out


# ---------------------------------------------------------------- 실행
def run():
    b = baseline()
    adj = adjacency()
    cand = pd.read_csv(TABLES / "fig_C06_추가대중교통수요_후보자치구.csv")
    ctx = {"vol_ratio": hourly_volume_ratio(), "adj": adj,
           "bus_gus": list(cand.head(CFG["bus"]["n_target_gu"])["자치구"]),
           "targets": {p: target_gus(b, p, CFG["selective"]["n_target_gu"])[0] for p in PERIODS},
           "transit": {p: transit_by_gu(h) for p, h in PERIODS.items()}}
    vulnerable = set(ctx["bus_gus"])
    rows, gu_rows, link_out = [], [], []
    for scen, lv, period, method in product(SCEN, LEVELS, PERIODS, ["탄력성", "BPR"]):
        m, info = (pd.Series(1.0, index=b.index), {}) if scen == "A" else multipliers(b, scen, lv, period, ctx)
        v = new_speed(b, m, method, lv) if scen != "A" else b.v_base
        ind = indicators(b, v, m, PERIODS[period])
        sh = indicators(b, v, m, list(SHOULDER[period].values()))
        add, pct, cap = transit_demand(info, lv, period, ctx) if scen != "A" else (pd.Series(0.0), pd.Series(0.0), pd.Series(0.0))
        thr = CFG["excess_threshold_pct"]
        over = list(pct[(pct - cap * 100) >= thr].index) if scen != "A" else []
        # 취약지역(추가 대중교통 수요 후보 자치구) 속도 변화 vs 그 외
        pk = b.hour_start.isin(PERIODS[period])
        def sm(mask, vv):
            L = b.length_m[pk & mask]
            return L.sum() / (L / vv[pk & mask]).sum()
        vul = b.gu.isin(vulnerable)
        rows.append({"시나리오": SCEN[scen], "가정수준": LEVEL_KO[lv], "첨두": period, "속도모형": method, **ind,
                     "앞뒤시간_속도변화_pct": sh["속도변화_pct"],
                     "대중교통_추가수요_인시": add.sum(), "대중교통_증가율_pct": add.sum() / ctx["transit"][period].board.sum() * 100,
                     "잠재초과수요_자치구수": len(over), "잠재초과수요_자치구": ", ".join(over),
                     "취약지역_속도변화_pct": (sm(vul, v) / sm(vul, b.v_base) - 1) * 100,
                     "그외지역_속도변화_pct": (sm(~vul, v) / sm(~vul, b.v_base) - 1) * 100,
                     **area_changes(b, v, pk, scen, info, ctx, sm),
                     "홀짝제_제한비율": info.get("restricted", np.nan)})
        if lv == "mid" and scen != "A":
            gu_rows.append(pd.DataFrame({"시나리오": SCEN[scen], "첨두": period, "속도모형": method, "자치구": add.index,
                                         "추가수요_인시": add.values, "증가율_pct": pct.values}))
            if method == "탄력성":
                link_out.append(b[pk][["link_id", "hour_start", "gu", "v_base"]].assign(
                    v_new=v[pk], 시나리오=SCEN[scen], 첨두=period))
    res = pd.DataFrame(rows)
    res.round(3).to_csv(TABLES / "sim_results.csv", index=False, encoding="utf-8-sig")
    gus = pd.concat(gu_rows)
    gus.round(2).to_csv(TABLES / "sim_transit_by_gu.csv", index=False, encoding="utf-8-sig")
    links = pd.concat(link_out)
    meta = pd.DataFrame([{"항목": "선택관리 대상 자치구(출근)", "값": ", ".join(ctx["targets"]["출근"])},
                         {"항목": "선택관리 대상 자치구(퇴근)", "값": ", ".join(ctx["targets"]["퇴근"])},
                         {"항목": "버스 증편 대상 자치구", "값": ", ".join(ctx["bus_gus"])},
                         {"항목": "기준선 링크 수", "값": b.link_id.nunique()}])
    meta.to_csv(TABLES / "sim_targets.csv", index=False, encoding="utf-8-sig")
    print(meta.to_string(index=False))
    return res, gus, links, ctx


# ---------------------------------------------------------------- 그림
def fig_compare(res):
    scen = [SCEN[k] for k in "BCDE"]
    panels = [("속도변화_pct", "첨두 평균속도 변화 (%)", True), ("정체비율_pct", "정체 도로 연장비율 (%)", True),
              ("대중교통_추가수요_인시", "대중교통 추가수요 (만 명/시)", False), ("VKT변화_pct", "주행거리(VKT) 변화 (%) — CO2 대리 지표", False)]
    for period in PERIODS:
        fig, axes = plt.subplots(2, 2, figsize=(13, 7.6))
        tab = []
        for ax, (col, lbl, by_method) in zip(axes.ravel(), panels):
            methods = ["탄력성", "BPR"] if by_method else ["탄력성"]
            w = 0.36 if by_method else 0.55
            for j, meth in enumerate(methods):
                d = res[(res["첨두"] == period) & (res["속도모형"] == meth)]
                mid = d[d["가정수준"] == "중간"].set_index("시나리오").reindex(scen)[col]
                lo = d[d["가정수준"] == "낮음"].set_index("시나리오").reindex(scen)[col]
                hi = d[d["가정수준"] == "높음"].set_index("시나리오").reindex(scen)[col]
                scale = 1e4 if "추가수요" in col else 1
                x = np.arange(len(scen)) + (j - (len(methods) - 1) / 2) * (w + 0.04)
                ax.bar(x, mid / scale, width=w, color=SERIES[j], label=f"속도모형: {meth}" if by_method else None)
                err = np.vstack([(mid - np.minimum(lo, hi)) / scale, (np.maximum(lo, hi) - mid) / scale])
                ax.errorbar(x, mid / scale, yerr=np.abs(err), fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
                top, bot = np.maximum(lo, hi) / scale, np.minimum(lo, hi) / scale
                for xi, val, t_, b_ in zip(x, mid / scale, top, bot):
                    # 라벨은 오차선 끝 바깥에 둔다
                    ax.text(xi + w / 2 + 0.02, val, f"{val:+.1f}" if "변화" in col else f"{val:.1f}", ha="left",
                            va="center", fontsize=8.5, color=INK2)
                tab.append(pd.DataFrame({"지표": col, "속도모형": meth, "시나리오": scen, "중간": mid.values,
                                         "낮음": lo.values, "높음": hi.values}))
            if col == "정체비율_pct":
                base = res[(res["첨두"] == period)].정체비율_기준_pct.iloc[0]
                ax.axhline(base, color=INK2, lw=1)
                ax.text(len(scen) - 0.5, base, f"A 현행 {base:.1f}%", ha="right", va="bottom", fontsize=9, color=INK2)
                ax.set_ylim(0, base * 1.25)
            ax.axhline(0, color=AXIS, lw=1)
            ax.set_xticks(range(len(scen)), [s.split(" ", 1)[1] for s in scen], fontsize=9.5)
            ax.set_title(lbl, fontsize=11.5)
            ax.grid(axis="x", visible=False)
            if col == "속도변화_pct":  # 범례는 한 번만
                ax.legend(loc="upper right", fontsize=8.5)
        titles(fig, f"정책 시나리오 비교 — {period} 첨두 (가정 기반 시뮬레이션)",
               f"막대: 중간 가정 · 오차선: 낮음~높음 가정 범위 · 기준선 2026-10-14·15 관측 속도 · 대중교통 2025년 일평균 · 실측 결과 아님")
        fig.tight_layout(rect=(0, 0.02, 1, 0.92))
        save(fig, f"F01_시나리오비교_{period}", SRC, pd.concat(tab).round(3))


def fig_map(links, scen_key="C", period="출근"):
    gu = gpd.read_file(PROC / "seoul_gu.gpkg")
    sl = gpd.read_file(PROC / "seoul_service_links.gpkg")[["link_id", "geometry"]]
    d = links[(links["시나리오"] == SCEN[scen_key]) & (links["첨두"] == period)]
    d = d.groupby("link_id").agg(v_base=("v_base", "mean"), v_new=("v_new", "mean"))
    d["chg"] = (d.v_new / d.v_base - 1) * 100
    m = sl.merge(d, left_on="link_id", right_index=True)
    lim = max(5, np.ceil(np.percentile(np.abs(m.chg), 98)))
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("div", [DIVERGE[2], DIVERGE[1], DIVERGE[0]])
    fig, ax = plt.subplots(figsize=(11, 9))
    gu.plot(ax=ax, color="#f0efec", edgecolor="white", linewidth=1)
    m.plot(column="chg", ax=ax, cmap=cmap, vmin=-lim, vmax=lim, linewidth=1.1)
    ax.set_axis_off()
    sm_ = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-lim, lim))
    cb = fig.colorbar(sm_, ax=ax, orientation="horizontal", fraction=0.035, pad=0.01, aspect=45)
    cb.set_label("속도 변화 (%) · 파랑: 개선 · 빨강: 악화(우회 유입)", color=INK2)
    cb.outline.set_visible(False)
    titles(fig, f"{SCEN[scen_key]} — {period} 첨두 구간별 속도 변화 (가정 기반)",
           "중간 가정 · 실측 탄력성 모형 · 기준선 2026-10-14·15 관측 속도 · 실측 결과 아님")
    save(fig, f"F02_구간별_속도변화_{scen_key}_{period}", SRC,
         m.drop(columns="geometry").round(2), svg=False)


def fig_transit(gus, ctx):
    d = gus[(gus["속도모형"] == "탄력성") & (gus["첨두"] == "출근")]
    b_pct = d[d["시나리오"] == SCEN["B"]]["증가율_pct"].median()
    piv = d.pivot_table(index="자치구", columns="시나리오", values="증가율_pct").fillna(0)
    cols = [SCEN[k] for k in "CDE"]
    piv = piv.reindex(columns=cols)
    piv = piv[(piv > 0).any(axis=1)].sort_values(SCEN["E"])
    fig, ax = plt.subplots(figsize=(10, 6.8))
    y = np.arange(len(piv))
    h = 0.26
    for j, col in enumerate(cols):
        k = list(SCEN.values()).index(col) - 1  # 시나리오별 색 고정(B=0, C=1, D=2, E=3)
        ax.barh(y + (1 - j) * h, piv[col], height=h * 0.9, color=SERIES[k], label=col)
        for yi, v in zip(y + (1 - j) * h, piv[col]):
            if v > 0:
                ax.text(v, yi, f" {v:.1f}", va="center", fontsize=8, color=INK2)
    ax.axvline(b_pct, color=SERIES[0], lw=1.5)
    ax.text(b_pct, len(piv) - 0.4, f" B 전역 홀짝제: 모든 자치구 +{b_pct:.1f}%", fontsize=9, color=INK2, va="bottom")
    ax.set_yticks(y, piv.index, fontsize=9.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("출근 첨두 대중교통 승차 증가율 (%, 현재 대비)")
    ax.legend(loc="lower right", fontsize=9)
    thr = CFG["excess_threshold_pct"]
    titles(fig, "자치구별 대중교통 추가수요 — 출근 첨두 (가정 기반)",
           f"중간 가정 · 현재 승차: 2025년 일평균 08~10시 · 모든 자치구가 검토 기준 {thr}%(가정) 미만 · 실측 결과 아님")
    fig.tight_layout(rect=(0, 0.02, 1, 0.9))
    save(fig, "F03_자치구별_대중교통추가수요_출근", SRC, d.pivot_table(index="자치구", columns="시나리오",
                                                              values=["추가수요_인시", "증가율_pct"]).round(2).reset_index())


def main():
    setup()
    res, gus, links, ctx = run()
    show = res[(res["가정수준"] == "중간")][["시나리오", "첨두", "속도모형", "속도변화_pct", "대상지역_속도변화_pct",
                                         "우회유입지역_속도변화_pct", "정체비율_pct", "대중교통_추가수요_인시", "VKT변화_pct",
                                         "앞뒤시간_속도변화_pct", "잠재초과수요_자치구수"]]
    print(show.round(2).to_string(index=False))
    fig_compare(res)
    fig_map(links, "C", "출근")
    fig_map(links, "B", "출근")
    fig_transit(gus, ctx)


if __name__ == "__main__":
    main()
