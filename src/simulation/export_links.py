"""시나리오별 구간 속도(출근 첨두, 중간 가정, 두 속도모형)를 저장 — 기획서 지도용."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate as sim  # noqa: E402


def main():
    b = sim.baseline()
    cand = pd.read_csv(sim.TABLES / "fig_C06_추가대중교통수요_후보자치구.csv")
    ctx = {"vol_ratio": sim.hourly_volume_ratio(), "adj": sim.adjacency(),
           "bus_gus": list(cand.head(sim.CFG["bus"]["n_target_gu"])["자치구"]),
           "targets": {p: sim.target_gus(b, p, sim.CFG["selective"]["n_target_gu"])[0] for p in sim.PERIODS},
           "transit": {p: sim.transit_by_gu(h) for p, h in sim.PERIODS.items()}}
    pk = b.hour_start.isin(sim.PERIODS["출근"])
    out = []
    for scen in ["B", "E"]:
        m, _ = sim.multipliers(b, scen, "mid", "출근", ctx)
        for method in ["탄력성", "BPR"]:
            v = sim.new_speed(b, m, method, "mid")
            out.append(b[pk][["link_id", "hour_start", "v_base"]].assign(v_new=v[pk], 시나리오=scen, 속도모형=method))
    d = pd.concat(out).groupby(["시나리오", "속도모형", "link_id"]).agg(v_base=("v_base", "mean"), v_new=("v_new", "mean")).reset_index()
    d["chg"] = (d.v_new / d.v_base - 1) * 100
    d.round(3).to_csv(sim.TABLES / "sim_links_출근_중간.csv", index=False, encoding="utf-8-sig")
    print(d.groupby(["시나리오", "속도모형"]).chg.describe().round(2))


if __name__ == "__main__":
    main()
