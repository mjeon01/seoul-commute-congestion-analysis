"""AI 반사실(counterfactual) 모델 — 시범사업 효과를 판정하는 장치.

정책이 없었다면 관리 구역(종로·중구)의 첨두 교통량이 얼마였을지를, 정책 영향을 받지 않는 정보
(멀리 떨어진 자치구의 같은 시각 교통량, 달력, 강수)만으로 예측한다. 시행 후 '실제 - 예측'이 정책 효과다.

검증: 정책이 없던 2025년 하반기(시험기간)에 가짜 효과(−X%)를 넣어, 4주 시범 창에서 몇 %부터 잡아내는지
(최소 검출 효과, MDE) 계산하고 단순 전후 비교·이중차분과 비교한다.
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from models.forecast_volume import load_volume  # noqa: E402

PROC, RAW, TABLES = ROOT / "data/processed", ROOT / "data/raw", ROOT / "outputs/tables"
TREAT = ["종로구", "중구"]
NEAR = ["서대문구", "은평구", "성북구", "동대문구", "용산구", "성동구", "마포구"]  # 우회 유입 가능 → 비교군에서 제외
HOURS = [8, 9, 17, 18]  # 정책 시간(원문 첨두)
SPLIT = {"train": ("2024-01-01", "2024-12-31"), "val": ("2025-01-01", "2025-06-30"), "test": ("2025-07-01", "2025-12-31")}
WINDOW = 20  # 시범 평가 창: 평일 20일(약 4주)


def daily_panel():
    g = load_volume(RAW / "volume_peak")
    g = g[g.vol > 0]
    g["date"], g["hour"] = g.datetime.dt.normalize(), g.datetime.dt.hour
    spots = gpd.read_file(PROC / "volume_spots.gpkg")[["spot_num", "SIGUNGU_NM"]]
    g = g.merge(spots, on="spot_num")
    # 결측 없이 비교하려고 날짜×시각 관측률 95% 이상 지점만
    full = g.groupby("spot_num").size()
    keep = full[full >= g.date.nunique() * g.hour.nunique() * 0.95].index
    g = g[g.spot_num.isin(keep)]
    # 지점이 하루 일부 빠진 경우 합계가 줄지 않도록, 날짜·시각마다 그룹별 '지점 평균 × 지점 수'로 집계
    g["grp"] = np.where(g.SIGUNGU_NM.isin(TREAT), "treat", np.where(g.SIGUNGU_NM.isin(NEAR), "near", g.SIGUNGU_NM))
    n_sp = g.groupby("grp").spot_num.nunique()
    agg = g.groupby(["date", "hour", "grp"]).vol.mean().unstack("grp")
    agg = agg.mul(n_sp, axis=1)
    return agg, n_sp


def features(agg):
    donors = [c for c in agg.columns if c not in ("treat", "near")]
    d = agg.reset_index()
    X = d[["date", "hour"]].copy()
    for c in donors:
        X[f"d_{c}"] = d[c]
    X["donor_total"] = d[donors].sum(axis=1, min_count=len(donors) - 2)
    X["treat_12"] = d.merge(agg["treat"].xs(12, level="hour").rename("t12"), left_on="date", right_index=True, how="left")["t12"].values
    hol = pd.read_csv(RAW / "holidays.csv", dtype=str)
    hol = set(pd.to_datetime(hol.locdate[hol.isHoliday == "Y"], format="%Y%m%d"))
    X["weekday"] = X.date.dt.weekday
    X["holiday"] = X.date.isin(hol).astype(int)
    X["month"] = X.date.dt.month
    w = pd.read_parquet(PROC / "weather_hourly.parquet", columns=["datetime", "rain_mm"]).set_index("datetime").rain_mm
    X["rain"] = (X.date + pd.to_timedelta(X.hour - 1, unit="h")).map(w).fillna(0).values
    X["y"] = d["treat"].values
    X = X[X.hour.isin(HOURS) & (X.weekday < 5) & (X.holiday == 0)].dropna(subset=["y", "donor_total", "treat_12"])
    return X


def split(X, name):
    a, b = SPLIT[name]
    return X[(X.date >= a) & (X.date <= b)]


def main():
    agg, n_sp = daily_panel()
    print("지점 수:", n_sp.to_dict())
    X = features(agg)
    feats = [c for c in X.columns if c not in ("date", "y")]
    tr, va, te = split(X, "train"), split(X, "val"), split(X, "test")
    models = {
        "AI(LightGBM)": lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=10,
                                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1),
        "선형(Ridge)": RidgeCV(alphas=np.logspace(-2, 3, 20)),
    }
    preds, rows = {}, []
    vt = pd.concat([va, te])
    for name, m in models.items():
        m.fit(tr[feats].fillna(0), tr.y)  # 2024로만 학습 → 2025 전체를 '처음 보는 기간'으로 예측
        p = m.predict(vt[feats].fillna(0))
        preds[name] = p
        pv, pt = p[: len(va)], p[len(va):]
        rows.append({"모델": name, "검증 MAPE(%)": round((np.abs(pv - va.y) / va.y).mean() * 100, 2),
                     "시험 MAPE(%)": round((np.abs(pt - te.y) / te.y).mean() * 100, 2)})
    perf = pd.DataFrame(rows)
    print(perf.to_string(index=False))
    best = perf.sort_values("검증 MAPE(%)").iloc[0]["모델"]

    # ---- 시범 창 플라시보: 정책이 없는 시험기간에서 '추정 효과'가 0 주변에 얼마나 흩어지는지
    vt = vt.assign(pred=preds[best])
    allday = vt.groupby("date").agg(y=("y", "sum"), pred=("pred", "sum"), donor=("donor_total", "sum")).sort_index()
    dates = allday.index[allday.index >= SPLIT["test"][0]]
    est = {"AI 반사실(사전 4주 보정)": [], "이중차분(비교 구 대비)": [], "단순 전후 비교": []}
    for i in range(0, len(dates) - WINDOW + 1):
        win = dates[i:i + WINDOW]
        w = allday.loc[win]
        before = allday[allday.index < win[0]].tail(WINDOW)  # 시행 직전 4주(정책 없음)
        calib = before.y.sum() / before.pred.sum()  # AI 예측의 수준 보정
        est["AI 반사실(사전 4주 보정)"].append((w.y.sum() / (w.pred.sum() * calib) - 1) * 100)
        est["단순 전후 비교"].append((w.y.mean() / before.y.mean() - 1) * 100)
        est["이중차분(비교 구 대비)"].append(((w.y.mean() / before.y.mean()) / (w.donor.mean() / before.donor.mean()) - 1) * 100)
    day = allday.loc[dates]
    out = []
    for k, v in est.items():
        v = np.array(v)
        sd = v.std(ddof=1)
        out.append({"방법": k, "효과 없을 때 추정값 평균(%)": round(v.mean(), 2), "추정 오차 SD(%p)": round(sd, 2),
                    "95% 오판 범위(±%p)": round(1.96 * sd, 2), "최소 검출 효과 MDE(%)": round(2.8 * sd, 1), "창 수": len(v)})
    mde = pd.DataFrame(out)
    print(mde.to_string(index=False))
    # 가짜 효과 주입 실험: 실제 교통량을 −X% 줄였을 때 AI 반사실 추정이 얼마로 나오나
    inj = []
    ai = np.array(est["AI 반사실(사전 4주 보정)"])
    thr = 1.96 * ai.std(ddof=1)
    for x in [2, 3, 5, 10, 15]:
        e = (1 - x / 100) * (1 + ai / 100) * 100 - 100
        res = {"넣은 효과(%)": -x, "AI 추정 평균(%)": round(e.mean(), 2), "AI 검출률(%)": round((e < -thr).mean() * 100, 1)}
        for k in ("이중차분(비교 구 대비)", "단순 전후 비교"):
            v = np.array(est[k])
            res[f"{k} 검출률(%)"] = round((((1 - x / 100) * (1 + v / 100) * 100 - 100) < -1.96 * v.std(ddof=1)).mean() * 100, 1)
        inj.append(res)
    inj = pd.DataFrame(inj)
    print(inj.to_string(index=False))
    perf.to_csv(TABLES / "cf_model_perf.csv", index=False, encoding="utf-8-sig")
    mde.to_csv(TABLES / "cf_placebo_mde.csv", index=False, encoding="utf-8-sig")
    inj.to_csv(TABLES / "cf_injection.csv", index=False, encoding="utf-8-sig")
    day.assign(pred=day.pred.round(0)).to_csv(TABLES / "cf_test_daily.csv", encoding="utf-8-sig")
    print("최선 모델:", best)


if __name__ == "__main__":
    main()
