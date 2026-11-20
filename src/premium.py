"""
STEP 3. 환율 대비 괴리율(김치 프리미엄) 분석
- 입력: data/processed/merged_1h.csv
- 출력: data/processed/features_1h.csv   (괴리율 컬럼 추가, 이후 단계에서 계속 사용)
        data/processed/premium_stats.md  (리포트용 통계 표)
        images/01_price_premium.png      (시각화 1)
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------- 한글 폰트 ----------------
try:
    import koreanize_matplotlib  # noqa: F401  (설치돼 있으면 자동으로 한글 폰트 적용)
except ImportError:
    plt.rcParams["font.family"] = ["Malgun Gothic"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------- 경로·설정 ----------------
BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"
IMG = BASE / "images"
IMG.mkdir(exist_ok=True)
KST = "Asia/Seoul"

Z_WINDOW = 168        # 괴리율 z-score 기준 기간: 168시간(1주)
Z_TH = 2.0            # ±2σ 를 과열/역과열 기준으로 사용

# 색상 (검증된 기본 팔레트)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SHADE = "#0b0b0b", "#52514e", "#e5e4e0", "#ecebe7"


# ---------------- 1) 불러오기 ----------------
def load():
    df = pd.read_csv(PROC / "merged_1h.csv")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(KST)
    df = df.set_index("datetime")
    for c in ["is_filled", "is_ret_outlier", "fx_stale", "is_analysis"]:
        df[c] = df[c].astype(str).str.lower().eq("true")
    return df


# ---------------- 2) 괴리율 계산 ----------------
def add_premium(df):
    df["premium_pct"] = (df["close"] / df["usdkrw"] - 1) * 100          # 괴리율(%)
    roll = df["premium_pct"].rolling(Z_WINDOW, min_periods=Z_WINDOW)
    df["premium_mean"] = roll.mean()                                     # 1주 이동평균
    df["premium_std"] = roll.std()                                       # 1주 표준편차
    df["premium_z"] = (df["premium_pct"] - df["premium_mean"]) / df["premium_std"]
    df["premium_state"] = np.select(
        [df["premium_z"] >= Z_TH, df["premium_z"] <= -Z_TH],
        ["과열", "역과열"], default="정상")
    return df


# ---------------- 3) 평균회귀 반감기 (AR(1)) ----------------
def half_life(p):
    """Δp(t) = a + b·p(t-1)  →  반감기 = -ln2 / ln(1+b)  (b<0 이어야 평균회귀)"""
    p = p.dropna()
    dp = p.diff().dropna()
    lag = p.shift(1).loc[dp.index]
    b, _ = np.polyfit(lag, dp, 1)
    return (np.inf if b >= 0 else -np.log(2) / np.log(1 + b)), b


# ---------------- 4) 과열 이후 괴리율 변화 ----------------
def after_event(df, state, hours=(6, 12, 24)):
    """과열/역과열 '진입 시점' 이후 h시간 뒤 괴리율이 평균 얼마나 변했는지"""
    a = df["is_analysis"]
    enter = (df["premium_state"] == state) & (df["premium_state"].shift(1) != state) & a
    out = {"진입 횟수": int(enter.sum())}
    for h in hours:
        chg = (df["premium_pct"].shift(-h) - df["premium_pct"])[enter].dropna()
        out[f"{h}h 후 평균 변화(%p)"] = round(float(chg.mean()), 3) if len(chg) else "-"
    return out


def stale_spans(mask):
    """True 가 연속된 구간의 (시작, 끝) 목록 - 그래프 음영용"""
    spans, start, prev = [], None, None
    for t, v in mask.items():
        if v and start is None:
            start = t
        if not v and start is not None:
            spans.append((start, prev + pd.Timedelta(hours=1)))
            start = None
        prev = t
    if start is not None:
        spans.append((start, prev + pd.Timedelta(hours=1)))
    return spans


# ---------------- 5) 시각화 1 ----------------
def plot_premium(a, path):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.22})
    fig.patch.set_facecolor("white")

    # 환율 휴장 구간 음영
    for s, e in stale_spans(a["fx_stale"]):
        for ax in (ax1, ax2):
            ax.axvspan(s, e, color=SHADE, zorder=0, lw=0)

    # (상) 가격 vs 환율 - 같은 단위(원)이므로 축 하나
    ax1.plot(a.index, a["close"], color=BLUE, lw=2, label="빗썸 USDT/KRW 종가")
    ax1.plot(a.index, a["usdkrw"], color=ORANGE, lw=2, label="USD/KRW 환율")
    ax1.set_ylabel("원 (KRW)", color=INK2)
    ax1.set_title("USDT/KRW 가격과 환율, 괴리율 (2026-11-01 ~ 11-10, 1시간봉)",
                  loc="left", fontsize=13, color=INK, pad=30)
    h, l = ax1.get_legend_handles_labels()
    h.append(plt.Rectangle((0, 0), 1, 1, color=SHADE))
    l.append("환율 휴장(직전 환율 유지)")
    ax1.legend(h, l, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, ncol=3, fontsize=9, borderaxespad=0.2)

    # (하) 괴리율 + 1주 평균 ± 2σ 밴드
    up = a["premium_mean"] + Z_TH * a["premium_std"]
    dn = a["premium_mean"] - Z_TH * a["premium_std"]
    ax2.fill_between(a.index, dn, up, color="#cde2fb", lw=0, label="1주 평균 ±2σ")
    ax2.plot(a.index, a["premium_mean"], color=INK2, lw=1, ls="--", label="1주 이동평균")
    ax2.plot(a.index, a["premium_pct"], color=BLUE, lw=2, label="괴리율(%)")
    hot, cold = a[a["premium_state"] == "과열"], a[a["premium_state"] == "역과열"]
    ax2.scatter(hot.index, hot["premium_pct"], marker="^", s=40, color=INK,
                edgecolor="white", lw=1, zorder=3, label="과열(+2σ 이상)")
    ax2.scatter(cold.index, cold["premium_pct"], marker="v", s=40, color=INK2,
                edgecolor="white", lw=1, zorder=3, label="역과열(-2σ 이하)")
    ax2.axhline(0, color=INK2, lw=0.8)
    ax2.set_ylabel("괴리율 (%)", color=INK2)
    ax2.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, ncol=5, fontsize=9, borderaxespad=0.2)

    # 최대·최소 괴리율 직접 표기
    for t, fmt in [(a["premium_pct"].idxmax(), "최대 {:+.2f}%"),
                   (a["premium_pct"].idxmin(), "최소 {:+.2f}%")]:
        v = a.loc[t, "premium_pct"]
        ax2.annotate(fmt.format(v), (t, v), xytext=(6, 6), textcoords="offset points",
                     fontsize=9, color=INK)

    for ax in (ax1, ax2):
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
    ax2.xaxis.set_major_locator(mdates.DayLocator(tz=a.index.tz))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d", tz=a.index.tz))

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------- 실행 ----------------
if __name__ == "__main__":
    df = add_premium(load())
    a = df[df["is_analysis"]]

    # STEP 2 수정 확인
    n_filled = int(a["is_filled"].sum())
    print(f"[확인] 분석 기간 채운 봉: {n_filled}개"
          + ("  ✅" if n_filled == 0 else "  ⚠ collect.py 수정 후 다시 수집이 필요합니다"))

    hl, b = half_life(a["premium_pct"])
    p = a["premium_pct"]
    real_fx = ~a["fx_stale"]
    corr = (a["close"].pct_change()[real_fx]
            .corr(a["usdkrw"].pct_change()[real_fx]))

    stats = {
        "괴리율 평균": f"{p.mean():+.2f}%",
        "괴리율 중앙값": f"{p.median():+.2f}%",
        "괴리율 표준편차": f"{p.std():.2f}%p",
        "최대 괴리율": f"{p.max():+.2f}% ({p.idxmax():%m/%d %H시})",
        "최소 괴리율": f"{p.min():+.2f}% ({p.idxmin():%m/%d %H시})",
        "괴리율 +1% 이상 시간 비중": f"{(p >= 1).mean()*100:.1f}%",
        "괴리율 마이너스(역프) 시간 비중": f"{(p < 0).mean()*100:.1f}%",
        "과열(+2σ↑) 시간": f"{(a['premium_state'] == '과열').sum()}시간",
        "역과열(-2σ↓) 시간": f"{(a['premium_state'] == '역과열').sum()}시간",
        "평균회귀 계수 b": f"{b:.3f}",
        "평균회귀 반감기": f"{hl:.1f}시간" if np.isfinite(hl) else "회귀 없음",
        "평일 평균 괴리율(실시간 환율)": f"{p[real_fx].mean():+.2f}%",
        "주말 평균 괴리율(직전 환율)": f"{p[~real_fx].mean():+.2f}%",
        "USDT 변동률 vs 환율 변동률 상관계수": f"{corr:.2f}",
    }
    hot = after_event(df, "과열")
    cold = after_event(df, "역과열")

    print("\n========== 괴리율 분석 결과 (11/1~11/10) ==========")
    for k, v in stats.items():
        print(f"{k:<28}: {v}")
    print("\n[과열 진입 이후]  ", hot)
    print("[역과열 진입 이후]", cold)

    # 저장
    df.to_csv(PROC / "features_1h.csv", encoding="utf-8-sig")
    md = "| 항목 | 값 |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in stats.items())
    md += "\n\n| 이벤트 | " + " | ".join(hot.keys()) + " |\n|---|" + "---|" * len(hot) + "\n"
    md += "| 과열 진입 | " + " | ".join(str(v) for v in hot.values()) + " |\n"
    md += "| 역과열 진입 | " + " | ".join(str(v) for v in cold.values()) + " |\n"
    (PROC / "premium_stats.md").write_text(md, encoding="utf-8")

    plot_premium(a, IMG / "01_price_premium.png")
    print("\n저장: data/processed/features_1h.csv, data/processed/premium_stats.md, "
          "images/01_price_premium.png")