"""t-data 1시간 소통정보(구간별) 전처리.

- 시간코드 hh는 '~hh시' 구간(직전 1시간)을 뜻함 → hour_start = hh - 1 (예: 08 = 07:00~08:00)
- 같은 기간을 담은 파일이 여러 개면 (기준일, 시간코드, 링크ID)로 중복 제거
- 서비스링크 형상(자치구·행정동, 표준링크에서 이전한 차로수·제한속도) 결합, 혼잡 등급(config 기준) 산출

입력: data/raw/manual/1시간 소통정보*.csv 및 .env의 TRAFFIC_DATA_PATH
출력: data/processed/speed_hourly.parquet
"""
import unicodedata

import geopandas as gpd
import numpy as np
import pandas as pd
from dotenv import dotenv_values

from common import CFG, PROC, RAW, ROOT, add_calendar, add_peak, quality_report

RENAME = {"기준일": "date", "시간코드": "time_code", "첨두시구분(T1/T2/T3:오전/낮/오후)": "topis_peak",
          "도로명": "road_name", "도로방향구분명": "direction", "구간순서": "seq", "링크ID": "link_id",
          "시작노드명": "from_node", "종료노드명": "to_node", "평균속도": "speed",
          "도로구분명": "road_class"}


def load():
    # macOS 파일명은 NFD로 저장되므로 정규화 후 비교
    files = {str(f) for f in (RAW / "manual").glob("*.csv")
             if unicodedata.normalize("NFC", f.name).startswith("1시간 소통정보")}
    env_path = dotenv_values(ROOT / ".env").get("TRAFFIC_DATA_PATH")
    if env_path:
        files.add(env_path)
    df = pd.concat([pd.read_csv(f, encoding="utf-8-sig", dtype=str) for f in sorted(files)])
    print(f"소통정보 파일 {len(files)}개, {len(df):,}행")
    return df.rename(columns=RENAME)[list(RENAME.values())]


def main():
    df = load()
    n0 = len(df)
    df = df.drop_duplicates(["date", "time_code", "link_id"])
    print(f"파일 간 중복 제거: {n0 - len(df):,}행")

    df["speed"] = pd.to_numeric(df["speed"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["hour_start"] = df["time_code"].astype(int) - 1
    df["datetime"] = df["date"] + pd.to_timedelta(df["hour_start"], unit="h")

    links = gpd.read_file(PROC / "seoul_service_links.gpkg", ignore_geometry=True)
    links = links.rename(columns=str.lower)[["link_id", "length_m", "lanes", "max_spd",
                                              "adm_cd", "adm_nm", "sigungu_cd", "sigungu_nm"]]
    df = df.merge(links, on="link_id", how="left")
    for col in ["sigungu_nm", "lanes"]:
        print(f"{col} 미매핑 링크: {df.loc[df[col].isna(), 'link_id'].nunique()}개 / {df.link_id.nunique()}개")

    # 이상치 표시(삭제하지 않음): 0 이하, 제한속도의 1.5배 초과
    df["speed_outlier"] = (df["speed"] <= 0) | (df["speed"] > df["max_spd"] * 1.5)

    c = CFG["congestion"]
    exp = df["road_class"].eq("도시고속도로")
    jam = np.where(exp, c["urban_expressway"]["jam"], c["general"]["jam"])
    slow = np.where(exp, c["urban_expressway"]["slow"], c["general"]["slow"])
    df["congestion"] = np.select([df.speed < jam, df.speed < slow], ["정체", "서행"], "원활")
    df["speed_ratio"] = df["speed"] / df["max_spd"]  # 제한속도 대비 속도

    df = add_peak(add_calendar(df))
    quality_report(df, "speed_hourly", key=["datetime", "link_id"], numeric=["speed", "lanes", "max_spd", "length_m"])
    print(f"기간: {df.datetime.min()} ~ {df.datetime.max()}, 이상치 표시 {df.speed_outlier.sum():,}행")
    df.to_parquet(PROC / "speed_hourly.parquet", index=False)


if __name__ == "__main__":
    main()
