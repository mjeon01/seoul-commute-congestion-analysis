"""교통량 이력(VolInfo) + 스마트교차로 1시간 교통량 전처리.

- VolInfo: hh는 00~23 → 해당 시각부터 1시간(hour_start)으로 가정. 차로별 값을 방향별 합계로 집계
- 교통량 지점 좌표(grs80tm, 중부원점 FN 500000 → EPSG:5181)로 자치구 매핑
- 스마트교차로: 다운로드 파일마다 직전 11일이 겹쳐 있음 → 동일 키 중복 제거, 값 충돌 여부 점검

출력: data/processed/volume_hourly.parquet, smart_intersection_hourly.parquet
"""
import geopandas as gpd
import pandas as pd

from common import CFG, EXT, PROC, RAW, add_calendar, add_peak, quality_report


def spot_volume():
    spots = pd.read_csv(RAW / "volume_spots.csv", dtype={"spot_num": str})
    spots = gpd.GeoDataFrame(spots, geometry=gpd.points_from_xy(spots.grs80tm_x, spots.grs80tm_y),
                             crs=CFG["crs"]["volume_spot"]).to_crs(CFG["crs"]["analysis"])
    gu = gpd.read_file(PROC / "seoul_gu.gpkg")
    # 시계(市界) 지점은 경계 바로 밖에 있어 within으로 안 잡힘 → 1km 이내 가장 가까운 자치구로 매핑하고 표시
    spots = gpd.sjoin_nearest(spots, gu[["SIGUNGU_CD", "SIGUNGU_NM", "geometry"]], how="left",
                              max_distance=1000, distance_col="dist_to_gu_m")
    spots["boundary_spot"] = spots["dist_to_gu_m"] > 0
    print(f"교통량 지점 {len(spots)}개, 경계 밖 최근접 매핑 {spots.boundary_spot.sum()}개, "
          f"미매핑 {spots.SIGUNGU_NM.isna().sum()}개")

    v = pd.concat([pd.read_csv(f, dtype=str) for f in sorted((RAW / "volume_hourly").glob("*.csv"))])
    v["vol"] = pd.to_numeric(v["vol"], errors="coerce")
    quality_report(v, "volume_raw_lane", key=["spot_num", "ymd", "hh", "io_type", "lane_num"], numeric=["vol"])
    g = (v.groupby(["spot_num", "ymd", "hh", "io_type"])
         .agg(vol=("vol", "sum"), n_lanes=("lane_num", "nunique"), lane_missing=("vol", lambda s: s.isna().sum()))
         .reset_index())
    g["date"] = pd.to_datetime(g["ymd"], format="%Y%m%d")
    g["hour_start"] = g["hh"].astype(int)
    g["datetime"] = g["date"] + pd.to_timedelta(g["hour_start"], unit="h")
    g = g.merge(spots[["spot_num", "spot_nm", "SIGUNGU_CD", "SIGUNGU_NM", "boundary_spot"]], on="spot_num", how="left")
    g = add_peak(add_calendar(g)).drop(columns=["ymd", "hh"])
    quality_report(g, "volume_hourly", key=["spot_num", "datetime", "io_type"], numeric=["vol", "n_lanes"])
    g.to_parquet(PROC / "volume_hourly.parquet", index=False)
    spots.drop(columns=["index_right"]).to_file(PROC / "volume_spots.gpkg")


def smart_intersection():
    files = sorted((EXT / "smart_intersection").glob("*.csv"))
    s = pd.concat([pd.read_csv(f, encoding="utf-8-sig", dtype={"STND_DT": str, "STND_TM": str}) for f in files])
    key = ["CROSS_ID", "DIR_CD", "STND_DT", "STND_TM"]
    n0 = len(s)
    s = s.drop_duplicates()
    conflict = s.duplicated(key, keep=False).sum()
    print(f"스마트교차로 원본 {n0:,}행 → 완전 중복 제거 후 {len(s):,}행, 같은 키에 다른 값 {conflict}행")
    s = s.drop_duplicates(key, keep="last")  # 충돌 시 가장 최근 다운로드 값 사용
    vol_cols = ["LTURN_VOL", "RTURN_VOL", "STURN_VOL", "UTURN_VOL"]
    s["vol"] = s[vol_cols].sum(axis=1)
    s["date"] = pd.to_datetime(s["STND_DT"], format="%Y%m%d")
    s["hour_start"] = s["STND_TM"].str[:2].astype(int)
    s["datetime"] = s["date"] + pd.to_timedelta(s["hour_start"], unit="h")
    s = add_peak(add_calendar(s))
    quality_report(s, "smart_intersection_hourly", key=["CROSS_ID", "DIR_CD", "datetime"], numeric=["vol"])
    print(f"교차로 {s.CROSS_ID.nunique()}개 {s.CROSS_NM.unique().tolist()}, 기간 {s.date.min():%Y-%m-%d}~{s.date.max():%Y-%m-%d}")
    s.to_parquet(PROC / "smart_intersection_hourly.parquet", index=False)


if __name__ == "__main__":
    spot_volume()
    smart_intersection()
