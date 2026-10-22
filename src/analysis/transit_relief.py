"""지하철 혼잡과 버스 재배분 근거 — 서울교통공사 혼잡도(2026-06 기준본) + 버스 노선·정류소 승하차(최근 12개월).

1) 지하철: 평일 30분 단위·역·방향별 혼잡도(%, 100% = 열차 정원 탑승)로 첨두 정원 초과 구간 집계
2) 버스: 노선별 출근 첨두(07·08시) 승차와 업무지구 하차 비중으로 '증회 노선'과 '배차 조정 노선' 구분
3) 정류소: 업무지구 출근 첨두 하차 상위 정류소(승하차 안내 인력 배치 후보)
4) 시차출근 근거: 도심(종로·중구) 지하철 시간대별 하차(2025년 월평균 일 하차)
결과는 가정 없는 실측 집계. 배차 간격·차량 대수는 공개 자료로 확보하지 못해 노선 수·승차 인원 기준으로만 비교한다.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW, PROC, TABLES = ROOT / "data/raw", ROOT / "data/processed", ROOT / "outputs/tables"
CBD = ["중구", "종로구", "강남구", "서초구", "영등포구"]  # 출근 대중교통 순유입 상위 업무지구(fig_C04)
AM_SLOTS = ["7시00분", "7시30분", "8시00분", "8시30분"]
MONTHS = ("202510", "202609")


def subway():
    d = pd.read_csv(RAW / "subway_congestion_20260630.csv", encoding="utf-8-sig")
    d = d[d["구분"] == "평일"].copy()
    slots = [c for c in d.columns if "시" in c and c.endswith("분")]
    d[slots] = d[slots].apply(pd.to_numeric, errors="coerce")
    m = d.melt(id_vars=["호선", "역명", "상하구분"], value_vars=slots, var_name="시각", value_name="혼잡도")
    prof = m.groupby("시각", sort=False).agg(최고=("혼잡도", "max"), 정원초과_구간수=("혼잡도", lambda s: int((s >= 100).sum())),
                                           구간수=("혼잡도", "size")).reset_index()
    top = m[m["시각"].isin(AM_SLOTS)].sort_values("혼잡도", ascending=False).drop_duplicates(["호선", "역명", "상하구분"]).head(10)
    gu = pd.read_parquet(PROC / "subway_monthly_hourly.parquet", columns=["line", "station", "SIGUNGU_NM"]).drop_duplicates()
    am = m[m["시각"].isin(AM_SLOTS)].merge(gu, left_on=["호선", "역명"], right_on=["line", "station"], how="left")
    by_gu = am.groupby("SIGUNGU_NM").혼잡도.agg(["max", lambda s: int((s >= 100).sum())]).set_axis(["출근최고혼잡도", "정원초과_구간시각수"], axis=1)
    print(f"지하철 역 이름 → 자치구 매칭 {am.SIGUNGU_NM.notna().mean():.0%}")
    return prof, top, by_gu.sort_values("출근최고혼잡도", ascending=False)


def bus():
    b = pd.read_parquet(PROC / "bus_monthly_hourly.parquet",
                        columns=["route", "route_name", "stop_name", "ym", "hour_start", "daily_board", "daily_alight", "SIGUNGU_NM"])
    b = b[(b.ym >= MONTHS[0]) & (b.ym <= MONTHS[1])]
    n_month = b.groupby("route").ym.nunique()
    am = b[b.hour_start.isin([7, 8])]
    r = am.groupby("route").agg(이름=("route_name", "first"), 승차=("daily_board", "sum"))
    r["승차"] = r["승차"] / r.index.map(n_month) / 2  # 월평균 일 승차, 시간당
    cbd = am.assign(c=am.SIGUNGU_NM.isin(CBD)).groupby(["route", "c"]).daily_alight.sum().unstack(fill_value=0)
    r["업무지구_하차비중"] = cbd.get(True, 0) / cbd.sum(axis=1)
    r = r[r["승차"] > 0].dropna()
    hi, lo = r["승차"].quantile(0.5), r["승차"].quantile(0.25)
    r["구분"] = "유지"
    r.loc[(r["업무지구_하차비중"] >= 0.5) & (r["승차"] >= hi), "구분"] = "증회(업무지구행 고수요)"
    r.loc[(r["업무지구_하차비중"] < 0.2) & (r["승차"] < lo), "구분"] = "배차 조정(저수요)"
    grp = r.groupby("구분").agg(노선수=("승차", "size"), 첨두_시간당_승차=("승차", "sum"), 노선당_평균=("승차", "mean")).round(0)
    grp["노선_비중_pct"] = (grp["노선수"] / grp["노선수"].sum() * 100).round(1)
    grp["승객_비중_pct"] = (grp["첨두_시간당_승차"] / grp["첨두_시간당_승차"].sum() * 100).round(1)
    stops = (am[am.SIGUNGU_NM.isin(CBD)].groupby(["stop_name", "SIGUNGU_NM"]).daily_alight.sum() / n_month.max() / 2)
    stops = stops.sort_values(ascending=False).rename("출근첨두_시간당_하차").reset_index()
    return r.sort_values("승차", ascending=False), grp, stops


def cbd_alight():
    d = pd.read_parquet(PROC / "subway_monthly_hourly.parquet", columns=["ym", "hour_start", "daily_alight", "SIGUNGU_NM"])
    d = d[d.ym.astype(str).str.startswith("2025") & d.SIGUNGU_NM.isin(["종로구", "중구"])]
    h = (d.groupby("hour_start").daily_alight.sum() / d.ym.nunique()).rename("일평균_하차").to_frame()
    h["09시대_대비_배수"] = h["일평균_하차"] / h.loc[9, "일평균_하차"]
    return h.loc[6:10].round(3)


def main():
    prof, top, by_gu = subway()
    routes, grp, stops = bus()
    prof.to_csv(TABLES / "subway_congestion_profile.csv", index=False, encoding="utf-8-sig")
    top.to_csv(TABLES / "subway_congestion_top10_am.csv", index=False, encoding="utf-8-sig")
    by_gu.to_csv(TABLES / "subway_congestion_by_gu_am.csv", encoding="utf-8-sig")
    routes.to_csv(TABLES / "bus_route_realloc.csv", encoding="utf-8-sig")
    grp.to_csv(TABLES / "bus_route_realloc_summary.csv", encoding="utf-8-sig")
    stops.head(100).to_csv(TABLES / "bus_cbd_stops_top100.csv", index=False, encoding="utf-8-sig")
    cbd = cbd_alight()
    cbd.to_csv(TABLES / "subway_cbd_alight_by_hour.csv", encoding="utf-8-sig")
    am = prof[prof["시각"].isin(AM_SLOTS)]
    print(am.to_string(index=False))
    print(top.to_string(index=False))
    print(by_gu.head(10).to_string())
    print(grp.to_string())
    print(routes[routes["구분"].str.startswith("증회")].head(8).to_string())
    print(stops.head(10).to_string(index=False), "\n상위 100 정류소 하차 합", round(stops.head(100)["출근첨두_시간당_하차"].sum()))
    print(cbd.to_string())


if __name__ == "__main__":
    main()
