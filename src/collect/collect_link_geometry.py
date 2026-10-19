"""TOPIS 서비스링크 형상(꼭짓점) 수집 — 열린데이터광장 LinkVerInfo (링크ID별 호출, XML).

소통정보 링크ID는 국가 표준링크ID와 대부분 다르므로(약 70% 불일치) 서비스링크 형상을 직접 받는다.
예) python src/collect/collect_link_geometry.py "<소통정보 CSV 경로>"
출력: data/raw/service_link_vertices.csv (link_id, ver_seq, grs80tm_x, grs80tm_y)
"""
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from common import RAW, seoul_all


def main():
    ids = sorted(set().union(*(pd.read_csv(f, encoding="utf-8-sig", dtype=str)["링크ID"] for f in sys.argv[1:])))
    out = RAW / "service_link_vertices.csv"
    done = set(pd.read_csv(out, dtype=str)["link_id"]) if out.exists() else set()
    todo = [i for i in ids if i not in done]
    print(f"링크 {len(ids):,}개 중 신규 {len(todo):,}개")
    with ThreadPoolExecutor(8) as ex:
        rows = [r for p in ex.map(lambda i: seoul_all("LinkVerInfo", (i,), fmt="xml"), todo) for r in p]
    missing = set(todo) - {r["link_id"] for r in rows}
    if missing:
        print(f"형상 없음 {len(missing)}개: {sorted(missing)[:10]}")
    pd.DataFrame(rows).to_csv(out, mode="a", header=not done, index=False, encoding="utf-8-sig")
    print(f"꼭짓점 {len(rows):,}행 저장")


if __name__ == "__main__":
    main()
