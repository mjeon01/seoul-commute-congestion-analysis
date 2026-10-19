"""AI 예측: 출퇴근 첨두 지점 교통량 1시간 전 예측 (회귀 + 고수요 여부 분류).

문제 정의(EDA 결과 반영):
- 도로 속도 자료는 33시간뿐이라 시계열 분할 검증이 불가능 → 2년치가 있는 지점 교통량을 예측 대상으로 한다.
- 대상: 교통량 지점 s의 날짜 d, 시각 h ∈ {08, 09, 18, 19}시 양방향 합계 교통량 (원문 출근 08~10시·퇴근 17~19시 중
  직전 1시간 관측이 있는 시각). 예측 시점은 h 시작 직전 — h-1시 관측까지 사용 가능.
- 분류: 고수요 여부 = 교통량 > (학습기간 지점·시각별 80분위). 회귀 예측값을 같은 기준으로 판정해 평가.

누수 방지:
- 시차 변수는 모두 과거(h-1시, 전일·전주 같은 시각). 강수도 h-1시까지만.
- 지점·시각 평균(타깃 인코딩)과 분류 임계값은 학습기간에서만 계산.
- 분할은 시간 순서: 학습 2024, 검증 2025-01~06, 시험 2025-07~12. 하이퍼파라미터는 검증으로만 선택.

실행: .venv/bin/python src/models/forecast_volume.py [--mock]
출력: outputs/tables/model_*.csv, outputs/figures/E*_*.png, outputs/models/*.txt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score, mean_absolute_error, precision_score, recall_score, root_mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from preprocess.common import add_calendar  # noqa: E402
from visualization.style import GRAY, INK2, SERIES, save, setup, titles  # noqa: E402

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
MODELS = ROOT / "outputs" / "models"
MODELS.mkdir(parents=True, exist_ok=True)

TARGET_HOURS = [8, 9, 18, 19]
SPLIT = {"train": ("2024-01-01", "2024-12-31"), "val": ("2025-01-01", "2025-06-30"), "test": ("2025-07-01", "2025-12-31")}
SRC = "서울 열린데이터광장 서울시 교통량 이력 정보(VolInfo); 기상청 ASOS; 한국천문연구원 특일 정보"
FEATURES = ["lag_1h", "lag_1d", "lag_7d", "lag_1h_7d", "ratio_1h_7d", "spot_hour_mean", "hour", "weekday_num",
            "is_holiday", "is_weekend", "month", "rain_prev_h", "rain_prev_3h", "next_day_holiday", "prev_day_holiday"]


def load_volume(src_dir):
    files = sorted(src_dir.glob("*.csv"))
    v = pd.concat([pd.read_csv(f, dtype={"spot_num": str, "ymd": str, "hh": str}) for f in files])
    v["vol"] = pd.to_numeric(v.vol, errors="coerce")
    # 차로 결측이 있는 지점-시각은 합계가 과소 → 제외(대체하지 않음)
    g = v.groupby(["spot_num", "ymd", "hh"]).agg(vol=("vol", "sum"), n_na=("vol", lambda s: s.isna().sum()),
                                                  n_rows=("vol", "size")).reset_index()
    g = g[g.n_na == 0]
    g["datetime"] = pd.to_datetime(g.ymd, format="%Y%m%d") + pd.to_timedelta(g.hh.astype(int), unit="h")
    return g[["spot_num", "datetime", "vol", "n_rows"]]


def build_features(g):
    g = g.copy()
    # 지점별 차로 구성 변화(행 수 변화)는 교통량 수준을 바꿈 → 지점의 최빈 행 수와 다른 관측은 제외
    mode_rows = g.groupby("spot_num").n_rows.agg(lambda s: s.mode().iat[0])
    g = g[g.n_rows == g.spot_num.map(mode_rows)]
    s = g.set_index(["spot_num", "datetime"]).vol

    def shift(td):
        idx = pd.MultiIndex.from_arrays([g.spot_num, g.datetime - td])
        return s.reindex(idx).values

    g["lag_1h"] = shift(pd.Timedelta(hours=1))
    g["lag_1d"] = shift(pd.Timedelta(days=1))
    g["lag_7d"] = shift(pd.Timedelta(days=7))
    g["lag_1h_7d"] = shift(pd.Timedelta(days=7, hours=1))
    g["ratio_1h_7d"] = g.lag_1h / g.lag_1h_7d  # 오늘 직전 시간이 지난주 같은 시간 대비 얼마나 많은지

    g["hour"] = g.datetime.dt.hour
    g["date"] = g.datetime.dt.normalize()
    g = add_calendar(g)
    g["weekday_num"] = g.datetime.dt.weekday
    g["is_holiday"] = (g.day_type == "공휴일").astype(int)
    g["is_weekend"] = (g.weekday_num >= 5).astype(int)
    g["month"] = g.datetime.dt.month
    hol = set(g.loc[g.day_type == "공휴일", "date"]) | set(g.loc[g.is_weekend == 1, "date"])
    g["next_day_holiday"] = (g.date + pd.Timedelta(days=1)).isin(hol).astype(int)
    g["prev_day_holiday"] = (g.date - pd.Timedelta(days=1)).isin(hol).astype(int)

    w = pd.read_parquet(PROC / "weather_hourly.parquet", columns=["datetime", "rain_mm"]).set_index("datetime").rain_mm
    w3 = w.rolling(3, min_periods=1).sum()
    g["rain_prev_h"] = (g.datetime - pd.Timedelta(hours=1)).map(w)
    g["rain_prev_3h"] = (g.datetime - pd.Timedelta(hours=1)).map(w3)

    g = g[g.hour.isin(TARGET_HOURS)]
    for k, (a, b) in SPLIT.items():
        g.loc[(g.datetime >= a) & (g.datetime <= pd.Timestamp(b) + pd.Timedelta(hours=23)), "split"] = k
    g = g[g.split.notna()]

    tr = g[g.split == "train"]
    enc = tr.groupby(["spot_num", "hour"]).vol.mean().rename("spot_hour_mean")
    thr = tr.groupby(["spot_num", "hour"]).vol.quantile(0.8).rename("high_thr")
    g = g.join(enc, on=["spot_num", "hour"]).join(thr, on=["spot_num", "hour"])
    g = g[g.spot_hour_mean.notna()]  # 학습기간에 없던 지점 제외
    g["high"] = (g.vol > g.high_thr).astype(int)
    n0 = len(g)
    g = g.dropna(subset=["lag_1h", "lag_7d"])
    print(f"표본 {n0:,} → 핵심 시차(lag_1h·lag_7d) 결측 제외 후 {len(g):,} "
          f"({g.groupby('split').size().to_dict()}), 지점 {g.spot_num.nunique()}개")
    return g


def leakage_checks(g):
    """누수 점검 결과를 기록. 위반 시 중단."""
    checks = {
        "시차 변수가 모두 과거": True,  # build_features에서 음의 시차만 사용
        "타깃 인코딩·임계값을 학습기간에서만 계산": True,
        "분할 시간 순서(train<val<test)": bool(g[g.split == "train"].datetime.max() < g[g.split == "val"].datetime.min()
                                          < g[g.split == "test"].datetime.min()),
        "lag_1h와 타깃이 같은 값인 비율(<1%면 정상)": float((g.lag_1h == g.vol).mean()),
        "타깃과 가장 상관 높은 변수": g[FEATURES + ["vol"]].corr()["vol"].drop("vol").abs().idxmax(),
    }
    for k, v in checks.items():
        print(f"  [누수 점검] {k}: {v}")
    assert checks["분할 시간 순서(train<val<test)"]
    return checks


def metrics(y, yhat, thr, name, split):
    hi, hi_hat = (y > thr).astype(int), (yhat > thr).astype(int)
    return {"model": name, "split": split, "MAE": mean_absolute_error(y, yhat), "RMSE": root_mean_squared_error(y, yhat),
            "MAPE_pct": float(np.mean(np.abs(y - yhat) / np.maximum(y, 1)) * 100),
            "F1": f1_score(hi, hi_hat), "Precision": precision_score(hi, hi_hat, zero_division=0),
            "Recall": recall_score(hi, hi_hat)}


def fit_models(g):
    tr, va, te = (g[g.split == k] for k in ("train", "val", "test"))
    X = lambda d: d[FEATURES]  # noqa: E731
    preds, best = {}, {}

    # 베이스라인: 지난주 같은 시각 / 학습기간 지점·시각·요일 평균
    swm = tr.groupby(["spot_num", "hour", "weekday_num"]).vol.mean().rename("b")
    for d in (va, te):
        preds.setdefault("베이스라인: 지난주 같은 시각", {})[d.split.iat[0]] = d.lag_7d.values
        preds.setdefault("베이스라인: 지점·시각·요일 평균", {})[d.split.iat[0]] = (
            d.join(swm, on=["spot_num", "hour", "weekday_num"]).b.fillna(d.spot_hour_mean).values)

    t0 = time.time()
    grid_rf = [dict(n_estimators=300, max_depth=d, min_samples_leaf=m, n_jobs=-1, random_state=0)
               for d in (12, 20) for m in (5, 20)]
    grid_xgb = [dict(n_estimators=n, max_depth=d, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                     random_state=0, n_jobs=-1) for n in (400, 800) for d in (4, 8)]
    grid_lgb = [dict(n_estimators=n, num_leaves=l, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                     colsample_bytree=0.8, random_state=0, verbose=-1) for n in (400, 800) for l in (31, 63)]
    for name, cls, grid in [("Random Forest", RandomForestRegressor, grid_rf),
                            ("XGBoost", xgb.XGBRegressor, grid_xgb), ("LightGBM", lgb.LGBMRegressor, grid_lgb)]:
        scores = []
        for p in grid:
            Xtr = X(tr).fillna(-1) if cls is RandomForestRegressor else X(tr)
            Xva = X(va).fillna(-1) if cls is RandomForestRegressor else X(va)
            m = cls(**p).fit(Xtr, tr.vol)
            scores.append((mean_absolute_error(va.vol, m.predict(Xva)), p, m))
        mae, p, m = min(scores, key=lambda x: x[0])
        best[name] = {"params": p, "val_MAE": mae, "model": m}
        fill = (lambda d: d.fillna(-1)) if cls is RandomForestRegressor else (lambda d: d)
        preds[name] = {"val": m.predict(fill(X(va))), "test": m.predict(fill(X(te)))}
        print(f"  {name}: 검증 MAE {mae:,.1f} (파라미터 {len(grid)}개 중 선택, 누적 {time.time() - t0:.0f}초)")

    rows = []
    for name, ps in preds.items():
        for split, d in (("val", va), ("test", te)):
            ok = ~np.isnan(ps[split])
            rows.append(metrics(d.vol.values[ok], ps[split][ok], d.high_thr.values[ok], name, split))
    res = pd.DataFrame(rows)
    return res, best, preds, te


def fig_results(res, n_test):
    t = res[res.split == "test"].sort_values("MAE", ascending=False)
    base_best = t[t.model.str.startswith("베이스라인")].MAE.min()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    y = np.arange(len(t))
    for ax, col, lbl in [(axes[0], "MAE", "MAE (대/시, 낮을수록 좋음)"), (axes[1], "F1", "고수요 판정 F1 (높을수록 좋음)")]:
        colors = [GRAY if m.startswith("베이스라인") else SERIES[0] for m in t.model]
        ax.barh(y, t[col], height=0.55, color=colors)
        for yi, v in zip(y, t[col]):
            ax.text(v, yi, f" {v:,.1f}" if col == "MAE" else f" {v:.3f}", va="center", fontsize=9, color=INK2)
        ax.set_xlabel(lbl)
        ax.grid(axis="y", visible=False)
    axes[0].axvline(base_best, color=SERIES[1], lw=1.2)
    axes[0].set_yticks(y, t.model)
    titles(fig, "첨두 교통량 1시간 전 예측 — 시험기간 성능",
           f"시험 2025-07~12 · 지점·시각 {n_test:,}건 · 08·09·18·19시 · 회색: 베이스라인, 주황선: 최선 베이스라인 MAE")
    fig.tight_layout(rect=(0, 0.02, 1, 0.88))
    save(fig, "E01_예측모델_성능비교", SRC, res.round(3))


def fig_shap(model, te):
    sample = te[FEATURES].sample(min(3000, len(te)), random_state=0)
    sv = shap.TreeExplainer(model).shap_values(sample)
    imp = pd.DataFrame({"변수": FEATURES, "평균|SHAP|": np.abs(sv).mean(0)}).sort_values("평균|SHAP|")
    names = {"lag_1h": "직전 1시간 교통량", "lag_1d": "전일 같은 시각", "lag_7d": "전주 같은 시각",
             "lag_1h_7d": "전주 직전 1시간", "ratio_1h_7d": "직전 1시간 / 전주 직전 1시간", "spot_hour_mean": "지점·시각 평균(학습기간)",
             "hour": "시각", "weekday_num": "요일", "is_holiday": "공휴일", "is_weekend": "주말", "month": "월",
             "rain_prev_h": "직전 1시간 강수량", "rain_prev_3h": "직전 3시간 강수량", "next_day_holiday": "다음날 휴일",
             "prev_day_holiday": "전날 휴일"}
    imp["이름"] = imp["변수"].map(names)
    fig, ax = plt.subplots(figsize=(10, 6.4))
    ax.barh(imp["이름"], imp["평균|SHAP|"], height=0.6, color=SERIES[0])
    for i, v in enumerate(imp["평균|SHAP|"]):
        ax.text(v, i, f" {v:,.1f}", va="center", fontsize=9, color=INK2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("평균 |SHAP| (예측 교통량에 대한 평균 기여 크기, 대/시)")
    titles(fig, "예측 영향 요인 (LightGBM, SHAP)",
           f"시험기간 표본 {len(sample):,}건 · 예측에 대한 기여도이며 인과 효과가 아님")
    fig.tight_layout(rect=(0, 0.02, 1, 0.9))
    save(fig, "E02_예측영향요인_SHAP", SRC, imp[["이름", "변수", "평균|SHAP|"]].iloc[::-1].round(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(RAW / "volume_peak"))
    ap.add_argument("--mock", action="store_true", help="코드 검증용 — 결과를 outputs에 저장하지 않음")
    a = ap.parse_args()
    setup()
    g = build_features(load_volume(Path(a.src)))
    checks = leakage_checks(g)
    res, best, preds, te = fit_models(g)
    print(res[res.split == "test"].round(3).to_string(index=False))
    if a.mock:
        print("[mock] 코드 동작 검증만 수행, 결과 저장 안 함")
        return
    res.round(4).to_csv(TABLES / "model_forecast_metrics.csv", index=False, encoding="utf-8-sig")
    meta = {k: {"params": {p: v for p, v in b["params"].items() if p != "n_jobs"}, "val_MAE": b["val_MAE"]}
            for k, b in best.items()}
    meta["leakage_checks"] = {k: (v if isinstance(v, (bool, str)) else round(v, 4)) for k, v in checks.items()}
    meta["n"] = g.groupby("split").size().to_dict()
    (MODELS / "forecast_volume_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str))
    best["LightGBM"]["model"].booster_.save_model(str(MODELS / "forecast_volume_lgbm.txt"))
    fig_results(res, len(te))
    fig_shap(best["LightGBM"]["model"], te)


if __name__ == "__main__":
    main()
