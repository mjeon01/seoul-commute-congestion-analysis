"""코드 동작 검증용 모의 교통량 생성 — data/mock/volume_peak/ (실제 분석에 사용 금지).

실제 VolInfo 파일과 같은 컬럼으로, 지점·시각·요일 패턴 + 잡음을 가진 임의 값을 만든다.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "mock" / "volume_peak"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
spots = [f"M-{i:02d}" for i in range(20)]
base = {s: rng.uniform(500, 3000) for s in spots}
hours = {7: 0.9, 8: 1.0, 9: 0.85, 12: 0.7, 17: 0.9, 18: 0.95, 19: 0.8}
for d in pd.date_range("2024-01-01", "2025-12-31"):
    wk = 0.75 if d.weekday() >= 5 else 1.0
    rows = [{"spot_num": s, "ymd": d.strftime("%Y%m%d"), "hh": f"{h:02d}", "io_type": io, "lane_num": 1,
             "vol": int(base[s] * f * wk * rng.normal(1, 0.08) / 2)}
            for s in spots for h, f in hours.items() for io in (1, 2)]
    pd.DataFrame(rows).to_csv(OUT / f"{d:%Y%m%d}.csv", index=False)
print(f"모의 파일 {len(list(OUT.glob('*.csv')))}개 → {OUT}")
