"""1단계 시범(종로·중구 구역 관리) 시뮬레이션 — 가정 기반, 실측 결과 아님.

simulate.py의 시나리오 C(혼잡지역 선택 관리)와 같은 산식·가정값을 쓰되 대상 자치구만 종로구·중구로 고정한다.
정책 시간은 config/simulation.yaml peak_hours와 같다(출근 08~10시, 퇴근 17~19시, hour_start 8·9 / 17·18).
버스 재배분은 차량 대수를 바꾸지 않는 노선 간 재배치라 승용차 감소 효과를 넣지 않았다(효과 미반영).

출력 outputs/tables/sim_pilot_jongno_jung.csv — 가정수준 × 첨두 × 속도모형별
  서울 전체·관리 구역·우회 유입 지역 속도 변화, VKT 대리 지표 변화, 10km 통행시간(관리 구역 평균속도 기준),
  대중교통 추가수요, 승용차 통행 감소·CO2 '가정 산출 예시'(산식은 아래 CO2_* 상수, 출처 미확인 가정).
실행: .venv/bin/python src/simulation/simulate_pilot.py
"""
import sys
from itertools import product
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate as sim  # noqa: E402

PILOT = ["종로구", "중구"]
CO2_KM = 10      # 승용차 통행 1회 주행거리(km) — 출처 미확인 가정
CO2_KG_KM = 0.17  # 승용차 배출계수(kg/km) — 출처 미확인 가정(차종·속도별 계수 미반영)
CO2_DAYS = 250   # 연간 적용 평일 수 — 가정


def main():
    b = sim.baseline()
    ctx = {"adj": sim.adjacency(), "bus_gus": [], "targets": {p: PILOT for p in sim.PERIODS},
           "transit": {p: sim.transit_by_gu(h) for p, h in sim.PERIODS.items()}}
    rows = []
    for lv, period, method in product(sim.LEVELS, sim.PERIODS, ["탄력성", "BPR"]):
        m, info = sim.multipliers(b, "C", lv, period, ctx)
        v = sim.new_speed(b, m, method, lv)
        ind = sim.indicators(b, v, m, sim.PERIODS[period])
        pk = b.hour_start.isin(sim.PERIODS[period])

        def sm(mask, vv):
            L = b.length_m[pk & mask]
            return L.sum() / (L / vv[pk & mask]).sum()
        t, a = b.gu.isin(PILOT), b.gu.isin(info["detour_gus"])
        v0, v1 = sm(t, b.v_base), sm(t, v)
        add, _, _ = sim.transit_demand(info, lv, period, ctx)
        rows.append({"가정수준": sim.LEVEL_KO[lv], "첨두": period, "속도모형": method,
                     "서울전체_속도변화_pct": ind["속도변화_pct"], "VKT대리_변화_pct": ind["VKT변화_pct"],
                     "관리구역_평균속도_기준": v0, "관리구역_평균속도": v1, "관리구역_속도변화_pct": (v1 / v0 - 1) * 100,
                     "우회유입지역_속도변화_pct": (sm(a, v) / sm(a, b.v_base) - 1) * 100,
                     "관리구역_10km_분_기준": 10 / v0 * 60, "관리구역_10km_분": 10 / v1 * 60,
                     "대중교통_추가수요_인시": add.sum(),
                     "대중교통_증가율_pct": add.sum() / ctx["transit"][period].board.sum() * 100,
                     "우회유입_자치구": ", ".join(info["detour_gus"])})
    r = pd.DataFrame(rows)
    # 승용차 통행 감소(회/일) = 대중교통 추가수요(인/시) × 2시간 ÷ 대중교통 전환율 ÷ 재차인원 — 출근+퇴근 합
    occ = {sim.LEVEL_KO[k]: v for k, v in sim.CFG["occupancy"].items()}
    shift = {sim.LEVEL_KO[k]: v for k, v in sim.CFG["to_transit_share"].items()}
    d = r[r["속도모형"] == "탄력성"].groupby("가정수준")["대중교통_추가수요_인시"].sum()
    trips = d * 2 / d.index.map(shift) / d.index.map(occ)
    r["승용차통행감소_회일_가정"] = r["가정수준"].map(trips)
    r["CO2_톤년_가정예시"] = r["승용차통행감소_회일_가정"] * CO2_KM * CO2_KG_KM * CO2_DAYS / 1000
    r.round(3).to_csv(sim.TABLES / "sim_pilot_jongno_jung.csv", index=False, encoding="utf-8-sig")
    print(r.drop(columns="우회유입_자치구").round(2).to_string(index=False))
    print("우회 유입 자치구:", r["우회유입_자치구"].iloc[0])


if __name__ == "__main__":
    main()
