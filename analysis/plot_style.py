"""제출용 노트북 공통 서식: 그림 스타일, 그림 제목·주석, 본문 수치 대조."""
import matplotlib.pyplot as plt
from matplotlib import font_manager

NAVY = "#1F3864"    # 합계·기준값
BLUE = "#2E5AA8"    # 주 계열
ORANGE = "#D9822B"  # 비교 계열
GRAY = "#7F8792"    # 논문·헤지 없음·음(-)의 항목
LIGHT = "#C9CED6"   # 보조 영역
INK = "#2B2F36"     # 글자·축
MUTED = "#6B7280"   # 주석

_KOREAN_FONTS = ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP")  # Noto CJK는 한글 포함


def apply_style():
    """그림 공통 서식을 적용한다."""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams.update({
        "font.family": [name for name in _KOREAN_FONTS if name in installed] + ["DejaVu Sans"],
        "font.size": 9,
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 9.5,
        "axes.titlelocation": "left",
        "axes.axisbelow": True,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#E5E7EB",
        "grid.linewidth": 0.6,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "none",
        "legend.fontsize": 8.5,
    })


def finish(fig, title, note=None):
    """그림 제목과 하단 주석을 넣고 여백을 정리한다."""
    fig.suptitle(title, x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK)
    bottom = 0.24 / fig.get_figheight() if note else 0.0
    fig.tight_layout(rect=(0, bottom, 1, 1))
    if note:
        fig.text(0.01, 0.008, note, fontsize=7.5, color=MUTED)


def verify_statements(checks):
    """본문에 적은 수치와 계산값을 대조하고, 불일치하면 실행을 중단한다."""
    for label, computed, stated in checks:
        assert computed == stated, f"{label}: 계산 {computed} ≠ 본문 {stated}"
    print(f"본문 수치 {len(checks)}건 일치")
