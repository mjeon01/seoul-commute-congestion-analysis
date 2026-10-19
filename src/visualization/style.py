"""그림 공통 스타일: 한글 폰트, 팔레트(검증된 기본 팔레트), 저장 규칙(300DPI PNG + SVG + 수치 표).

모든 그림은 save()로 저장 — 제목 아래 부제에 기간·단위·대상을, 하단에 출처를 표기한다.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "outputs" / "figures"
HTML = ROOT / "outputs" / "html"
TABLES = ROOT / "outputs" / "tables"
for d in (FIG, HTML, TABLES):
    d.mkdir(parents=True, exist_ok=True)

# 범주형 슬롯 순서 고정(validate_palette.js 통과: 1~4 인접쌍 CVD ΔE ≥ 9.1). 3·4번은 대비 3:1 미만 → 직접 라벨 필수
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # 순차(파랑)
DIVERGE = ("#2a78d6", "#f0efec", "#e34948")  # 파랑-회색-빨강
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
GRAY = "#c3c2b7"  # 강조하지 않는 대상


def setup():
    for f in ["NanumGothic", "AppleSDGothicNeo", "Apple SD Gothic Neo"]:
        if any(f.replace(" ", "") in x.name.replace(" ", "") for x in font_manager.fontManager.ttflist):
            mpl.rcParams["font.family"] = f
            break
    mpl.rcParams.update({
        "axes.unicode_minus": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "font.size": 10.5, "lines.linewidth": 2, "lines.solid_capstyle": "round",
        "legend.frameon": False, "svg.fonttype": "none", "savefig.facecolor": SURFACE,
    })


def titles(fig, title, subtitle):
    """제목(좌측 정렬) + 부제(기간·단위·대상)."""
    fig.suptitle(title, x=0.01, ha="left", fontsize=15, fontweight="bold", y=0.995)
    fig.text(0.01, 0.945, subtitle, ha="left", va="top", fontsize=10.5, color=INK2)


def save(fig, name, source, table=None, svg=True):
    """outputs/figures/<name>.png(300dpi)·.svg, 수치 표는 outputs/tables/fig_<name>.csv."""
    fig.text(0.01, 0.005, f"출처: {source}", ha="left", va="bottom", fontsize=8.5, color=MUTED)
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    if svg:
        fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    if table is not None:
        table.to_csv(TABLES / f"fig_{name}.csv", index=False, encoding="utf-8-sig")
    print(f"저장: {name}")


PLOTLY_LAYOUT = dict(
    font=dict(family="NanumGothic, Apple SD Gothic Neo, sans-serif", color=INK, size=13),
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, colorway=SERIES, hovermode="x unified",
    xaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False),
    yaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False),
    margin=dict(l=60, r=30, t=90, b=60),
)


def save_html(fig, name, title, subtitle):
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(
        text=f"<b>{title}</b><br><span style='font-size:13px;color:{INK2}'>{subtitle}</span>", x=0.01))
    fig.write_html(HTML / f"{name}.html", include_plotlyjs="cdn")
    print(f"저장: {name}.html")
