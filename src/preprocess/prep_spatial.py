"""공간정보 전처리: 서울 자치구·행정동 경계, 서울 표준링크(차로수·제한속도 등) + 자치구/행정동 매핑.

입력: data/raw/ext/boundary/bnd_*_2025_2Q.shp (SGIS), data/raw/ext/nodelink_20261014/MOCT_LINK.shp
      data/raw/service_link_vertices.csv (TOPIS 서비스링크 형상)
출력: data/processed/seoul_gu.gpkg, seoul_dong.gpkg, seoul_links.gpkg, seoul_service_links.gpkg
"""
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from common import CFG, EXT, PROC, RAW

EPSG = CFG["crs"]["analysis"]
SIDO = CFG["seoul_sido_code"]


def boundaries():
    gu = gpd.read_file(EXT / "boundary" / "bnd_sigungu_00_2025_2Q.shp")
    gu = gu[gu["SIGUNGU_CD"].str.startswith(SIDO)].set_crs(EPSG, allow_override=True)
    dong = gpd.read_file(EXT / "boundary" / "bnd_dong_00_2025_2Q.shp")
    dong = dong[dong["ADM_CD"].str.startswith(SIDO)].set_crs(EPSG, allow_override=True)
    # 행정동 코드 앞 5자리 = 시군구 코드
    dong["SIGUNGU_CD"] = dong["ADM_CD"].str[:5]
    dong = dong.merge(gu[["SIGUNGU_CD", "SIGUNGU_NM"]], on="SIGUNGU_CD", how="left")
    print(f"자치구 {len(gu)}개, 행정동 {len(dong)}개, 행정동 자치구 미매칭 {dong.SIGUNGU_NM.isna().sum()}")
    gu.to_file(PROC / "seoul_gu.gpkg")
    dong.to_file(PROC / "seoul_dong.gpkg")
    return gu, dong


def links(dong):
    seoul = dong.dissolve().to_crs(CFG["crs"]["nodelink"])
    lk = gpd.read_file(EXT / "nodelink_20261014" / "MOCT_LINK.shp", bbox=tuple(seoul.total_bounds.round()),
                       encoding="cp949")
    lk = lk.set_crs(CFG["crs"]["nodelink"], allow_override=True).to_crs(EPSG)
    keep = ["LINK_ID", "F_NODE", "T_NODE", "LANES", "ROAD_RANK", "ROAD_NAME", "MAX_SPD", "LENGTH", "geometry"]
    lk = lk[keep]
    # 링크 중간점이 속한 행정동/자치구로 매핑 (경계를 지나는 링크는 중간점 기준)
    mid = lk.copy()
    mid["geometry"] = lk.geometry.interpolate(0.5, normalized=True)
    mid = gpd.sjoin(mid, dong[["ADM_CD", "ADM_NM", "SIGUNGU_CD", "SIGUNGU_NM", "geometry"]],
                    how="inner", predicate="within")
    lk = lk.merge(mid.drop(columns=["geometry", "index_right"]).drop_duplicates("LINK_ID")[
        ["LINK_ID", "ADM_CD", "ADM_NM", "SIGUNGU_CD", "SIGUNGU_NM"]], on="LINK_ID", how="inner")
    print(f"서울 표준링크 {len(lk):,}개")
    lk.to_file(PROC / "seoul_links.gpkg")
    return lk


def _bearing(geom):
    (x0, y0), (x1, y1) = geom.coords[0], geom.coords[-1]
    v = np.array([x1 - x0, y1 - y0])
    n = np.linalg.norm(v)
    return v / n if n else v


def service_links(dong, std):
    """TOPIS 서비스링크 형상 생성 → 자치구/행정동 매핑 + 표준링크에서 차로수·제한속도 이전.

    표준링크 매칭: 서비스링크 15m 버퍼 안에 길이 50% 이상 들어가고 진행방향이 같은(코사인>0.7) 표준링크들의
    길이가중 중앙값. 매칭 실패 시 결측으로 둔다.
    """
    v = pd.read_csv(RAW / "service_link_vertices.csv", dtype={"link_id": str})
    v = v.sort_values(["link_id", "ver_seq"])
    geom = v.groupby("link_id").apply(lambda g: LineString(zip(g.grs80tm_x, g.grs80tm_y)), include_groups=False)
    sl = gpd.GeoDataFrame({"link_id": geom.index}, geometry=geom.values,
                          crs=CFG["crs"]["volume_spot"]).to_crs(EPSG)
    sl["length_m"] = sl.length

    mid = sl.copy()
    mid["geometry"] = sl.geometry.interpolate(0.5, normalized=True)
    mid = gpd.sjoin(mid, dong[["ADM_CD", "ADM_NM", "SIGUNGU_CD", "SIGUNGU_NM", "geometry"]], predicate="within")
    sl = sl.merge(mid.drop(columns=["geometry", "index_right", "length_m"]).drop_duplicates("link_id"),
                  on="link_id", how="left")
    print(f"서비스링크 {len(sl):,}개, 행정동 미매핑 {sl.ADM_CD.isna().sum()}개 (서울 경계 밖 중간점)")

    buf = sl[["link_id", "geometry"]].copy()
    buf["geometry"] = sl.buffer(15)
    cand = gpd.sjoin(std[["LINK_ID", "LANES", "MAX_SPD", "geometry"]], buf, predicate="intersects")
    cand["inside"] = [g.intersection(b).length / g.length if g.length else 0
                      for g, b in zip(cand.geometry, buf.set_index("link_id").loc[cand.link_id].geometry)]
    sb = dict(zip(sl.link_id, sl.geometry.map(_bearing)))
    cand["cos"] = [float(_bearing(g) @ sb[i]) for g, i in zip(cand.geometry, cand.link_id)]
    cand = cand[(cand.inside >= 0.5) & (cand.cos > 0.7)]
    cand["w"] = cand.length

    def wmedian(g, col):
        g = g.sort_values(col)
        return g[col].iloc[(g.w.cumsum() >= g.w.sum() / 2).argmax()]

    agg = cand.groupby("link_id").apply(lambda g: pd.Series({
        "lanes": wmedian(g, "LANES"), "max_spd": wmedian(g, "MAX_SPD"), "n_std_links": len(g)}),
        include_groups=False)
    sl = sl.merge(agg, on="link_id", how="left")
    print(f"표준링크 속성 매칭: {sl.lanes.notna().sum():,} / {len(sl):,}개")
    sl.to_file(PROC / "seoul_service_links.gpkg")


if __name__ == "__main__":
    _, dong = boundaries()
    std = links(dong)
    service_links(dong, std)
