"""전처리 공통: 경로, 설정, 공휴일/요일 구분, 첨두 구분, 품질 점검 기록."""
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
EXT = RAW / "ext"
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
CFG = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())

PROC.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

WEEKDAY_KO = ["월", "화", "수", "목", "금", "토", "일"]


def holidays():
    h = pd.read_csv(RAW / "holidays.csv", dtype=str)
    h = h[h["isHoliday"] == "Y"]
    return pd.Series(h["dateName"].values, index=pd.to_datetime(h["locdate"])).groupby(level=0).first()


def add_calendar(df, date_col="date"):
    """요일·평일/주말/공휴일 구분 추가. date_col은 datetime64(날짜)."""
    hol = holidays()
    d = df[date_col]
    df["weekday"] = d.dt.weekday.map(dict(enumerate(WEEKDAY_KO)))
    df["holiday_name"] = d.map(hol)
    df["day_type"] = "평일"
    df.loc[d.dt.weekday >= 5, "day_type"] = "주말"
    df.loc[df["holiday_name"].notna(), "day_type"] = "공휴일"
    return df


def add_peak(df, hour_col="hour_start"):
    """원문 기준 출근(08~10)·퇴근(17~19) 구분."""
    h = df[hour_col]
    df["peak"] = "비첨두"
    df.loc[h.isin(CFG["peak"]["am"]), "peak"] = "출근"
    df.loc[h.isin(CFG["peak"]["pm"]), "peak"] = "퇴근"
    return df


def quality_report(df, name, key=None, numeric=None):
    """결측·중복·수치 요약을 outputs/tables/quality_<name>.csv로 저장하고 요약 dict 반환."""
    rep = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(),
                        "missing_pct": (df.isna().mean() * 100).round(2)})
    if numeric:
        rep = rep.join(df[numeric].describe().T[["min", "50%", "max"]])
    rep.to_csv(TABLES / f"quality_{name}.csv", encoding="utf-8-sig")
    dup = int(df.duplicated(key).sum()) if key else int(df.duplicated().sum())
    print(f"[{name}] {len(df):,}행, 중복({key or '전체'}) {dup:,}건, 결측 컬럼: "
          f"{rep.loc[rep.missing > 0, 'missing_pct'].to_dict()}")
    return {"rows": len(df), "duplicates": dup}
