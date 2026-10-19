"""지하철·버스 시간대별 승하차(월 합계) 전처리: wide → long, 좌표·자치구 매핑.

- 원본 완전 중복 행은 long 변환 전에 제거(pivot 합산이 중복을 가리기 때문)
- HR_n = n시~n+1시 (hour_start = n). 값은 해당 월 전체 합계 → 일평균은 월 일수로 나눈 값(평일/주말 구분 불가)
- 버스: 순환노선은 같은 정류소를 두 번 지나므로 정류소명 뒤 순번(00002 등)으로 구분 → 키에 stop_name 포함
- 버스 컬럼명이 시간대에 따라 *_NOPE / *_TNOPE로 섞여 있어 정규식으로 통일
- 지하철역 좌표: subwayStationMaster (역명·호선 매칭), 버스정류소 좌표: busStopLocationXyInfo (STOPS_ID = STOPS_NO)

출력: data/processed/subway_monthly_hourly.parquet, bus_monthly_hourly.parquet, transit_points.gpkg
"""
import re

import geopandas as gpd
import pandas as pd

from common import CFG, PROC, RAW, add_peak, quality_report

HR = re.compile(r"HR_(\d+)_GET_(ON|OFF)_T?NOPE")


def drop_exact_dups(df, name):
    """원본 API가 같은 달 행을 통째로 두 번 반환한 경우(예: 지하철 2026-03, 2026-07) 제거.
    이후 pivot 합산이 중복을 가리므로 반드시 long 변환 전에 수행."""
    dup = df.duplicated()
    if dup.any():
        print(f"{name} 원본 완전 중복 {dup.sum():,}행 제거 — 월별: {df.loc[dup].iloc[:, 0].value_counts().sort_index().to_dict()}")
    return df.loc[~dup]


def to_long(df, id_cols, ym_col):
    hr_cols = [c for c in df.columns if HR.fullmatch(c)]
    m = df.melt(id_vars=id_cols + [ym_col], value_vars=hr_cols, var_name="col", value_name="n")
    parts = m["col"].str.extract(HR)
    m["hour_start"] = parts[0].astype(int)
    m["kind"] = parts[1].map({"ON": "board", "OFF": "alight"})
    m["n"] = pd.to_numeric(m["n"], errors="coerce")
    out = m.pivot_table(index=id_cols + [ym_col, "hour_start"], columns="kind", values="n", aggfunc="sum").reset_index()
    out = out.rename(columns={ym_col: "ym"})
    out["month"] = pd.to_datetime(out["ym"], format="%Y%m")
    out["daily_board"] = out["board"] / out["month"].dt.days_in_month
    out["daily_alight"] = out["alight"] / out["month"].dt.days_in_month
    return add_peak(out)


# 역명 변경으로 좌표 마스터와 이름이 다른 역 (승하차 자료 이름 → 마스터 이름)
STATION_ALIAS = {"당고개": "불암산", "자양": "뚝섬유원지"}


def norm_station(s):
    s = s.str.replace(r"\(.*?\)", "", regex=True).str.replace("역$", "", regex=True).str.strip()
    return s.replace(STATION_ALIAS)


def points():
    gu = gpd.read_file(PROC / "seoul_dong.gpkg")[["ADM_CD", "ADM_NM", "SIGUNGU_CD", "SIGUNGU_NM", "geometry"]]
    st = pd.read_csv(RAW / "subway_station_locations.csv", dtype=str)
    st = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st.LOT.astype(float), st.LAT.astype(float)), crs=4326)
    bs = pd.read_csv(RAW / "bus_stop_locations.csv", dtype=str)
    bs = gpd.GeoDataFrame(bs, geometry=gpd.points_from_xy(bs.XCRD.astype(float), bs.YCRD.astype(float)), crs=4326)
    st = gpd.sjoin(st.to_crs(CFG["crs"]["analysis"]), gu, how="left", predicate="within").drop(columns="index_right")
    bs = gpd.sjoin(bs.to_crs(CFG["crs"]["analysis"]), gu, how="left", predicate="within").drop(columns="index_right")
    return st, bs


def main():
    st, bs = points()

    sub = pd.concat([pd.read_csv(f, dtype=str) for f in sorted((RAW / "subway_hourly").glob("*.csv"))])
    sub = drop_exact_dups(sub, "지하철")
    sub = to_long(sub, ["SBWY_ROUT_LN_NM", "STTN"], "USE_MM").rename(
        columns={"SBWY_ROUT_LN_NM": "line", "STTN": "station"})
    # 역명 기준 좌표(같은 역명은 호선이 달라도 위치가 거의 같으므로 첫 좌표 사용)
    st["key"] = norm_station(st["BLDN_NM"])
    st_one = st.drop_duplicates("key")
    sub["key"] = norm_station(sub["station"])
    sub = sub.merge(pd.DataFrame(st_one.drop(columns="geometry"))[["key", "ADM_CD", "ADM_NM", "SIGUNGU_CD", "SIGUNGU_NM"]],
                    on="key", how="left")
    no_coord = sorted(set(sub.loc[~sub.key.isin(st.key), "station"]))
    outside = sub.loc[sub.key.isin(st.key) & sub.SIGUNGU_NM.isna(), "station"].nunique()
    print(f"지하철역 {sub.station.nunique()}개: 서울 밖 {outside}개, 좌표 이름 매칭 실패 {len(no_coord)}개 {no_coord[:20]}")
    quality_report(sub, "subway_monthly_hourly", key=["ym", "line", "station", "hour_start"], numeric=["board", "alight"])
    sub.to_parquet(PROC / "subway_monthly_hourly.parquet", index=False)

    bus = pd.concat([pd.read_csv(f, dtype=str) for f in sorted((RAW / "bus_hourly").glob("*.csv"))])
    bus = drop_exact_dups(bus, "버스")
    bus = to_long(bus, ["RTE_NO", "RTE_NM", "STOPS_ID", "STOPS_ARS_NO", "SBWY_STNS_NM", "TRFC_MNS_TYPE_NM"], "USE_YM")
    bus = bus.rename(columns={"RTE_NO": "route", "RTE_NM": "route_name", "STOPS_ID": "stop_id",
                              "STOPS_ARS_NO": "ars_no", "SBWY_STNS_NM": "stop_name", "TRFC_MNS_TYPE_NM": "bus_type"})
    bus = bus.merge(pd.DataFrame(bs.drop(columns="geometry"))[["STOPS_NO", "ADM_CD", "ADM_NM", "SIGUNGU_CD", "SIGUNGU_NM"]]
                    .drop_duplicates("STOPS_NO").rename(columns={"STOPS_NO": "stop_id"}), on="stop_id", how="left")
    # stop_id '~'는 가상정류소(위치 없음)
    print(f"stop_id '~' 행: {(bus.stop_id == '~').sum()}")
    n_stop = bus.stop_id.nunique()
    print(f"버스정류소 {n_stop:,}개 중 좌표/서울 자치구 미매핑 {bus.loc[bus.SIGUNGU_NM.isna(), 'stop_id'].nunique():,}개")
    quality_report(bus, "bus_monthly_hourly", key=["ym", "route", "stop_id", "stop_name", "hour_start"], numeric=["board", "alight"])
    bus.to_parquet(PROC / "bus_monthly_hourly.parquet", index=False)

    st.assign(kind="subway").to_file(PROC / "transit_points.gpkg", layer="subway")
    bs.assign(kind="bus").to_file(PROC / "transit_points.gpkg", layer="bus")


if __name__ == "__main__":
    main()
