"""
STEP 4. 기술적 지표 4종 계산 + 매매 신호
- 지표: MACD(12,26,9), Stochastic RSI(14,14,3,3), Stochastic(14,3,3), BB %B(20,2)
- 입력: data/processed/features_1h.csv   (STEP 3 결과)
- 출력: data/processed/features_1h.csv   (지표·신호 컬럼 추가해서 덮어쓰기)
        data/processed/indicator_stats.md
        images/02_indicators.png           (시각화 2)
신호 값: +1 = Long(매수), -1 = Short(매도), 0 = 신호 없음
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------- 한글 폰트 ----------------
try:
    import koreanize_matplotlib  # noqa: F401
except ImportError:
    plt.rcParams["font.family"] = ["Malgun Gothic"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------- 경로·설정 ----------------
BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"
IMG = BASE / "images"
IMG.mkdir(exist_ok=True)
KST = "Asia/Seoul"

BLUE, ORANGE = "#2a78d6", "#eb6834"
LONG_C, SHORT_C = "#e34948", "#0d366b"      # Long ▲ 빨강(상승), Short ▼ 진한 남색(하락)
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e5e4e0", "#cde2fb"


# ---------------- 불러오기 ----------------
def load():
    df = pd.read_csv(PROC / "features_1h.csv")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(KST)
    df = df.set_index("datetime")
    for c in ["is_filled", "is_ret_outlier", "fx_stale", "is_analysis"]:
        df[c] = df[c].astype(str).str.lower().eq("true")
    return df


# ---------------- 교차 판정 도우미 ----------------
def cross_up(a, b):
    """a 가 b 를 아래→위로 돌파한 시점"""
    return (a > b) & (a.shift(1) <= b.shift(1))


def cross_down(a, b):
    return (a < b) & (a.shift(1) >= b.shift(1))


def safe_ratio(num, den, fill):
    """분모가 0(가격이 14시간 내내 같음)이면 중립값으로 채움.
    빗썸 USDT는 1원 단위라 가격이 멈춰 있는 구간이 자주 생김."""
    return (num / den.replace(0, np.nan)).fillna(fill)


# ---------------- 1) MACD ----------------
def add_macd(df, fast=12, slow=26, signal=9):
    ema_f = df["close"].ewm(span=fast, adjust=False).mean()
    ema_s = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_f - ema_s
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["sig_macd"] = np.select(
        [cross_up(df["macd"], df["macd_signal"]), cross_down(df["macd"], df["macd_signal"])],
        [1, -1], 0)
    return df


# ---------------- 2) Stochastic RSI ----------------
def rsi(close, n=14):
    diff = close.diff()
    gain = diff.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()   # Wilder 방식
    loss = (-diff.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + safe_ratio(gain, loss, np.inf))


def add_stoch_rsi(df, n_rsi=14, n_stoch=14, k=3, d=3, low=20, high=80):
    df["rsi"] = rsi(df["close"], n_rsi)
    lo = df["rsi"].rolling(n_stoch).min()
    hi = df["rsi"].rolling(n_stoch).max()
    raw = safe_ratio(df["rsi"] - lo, hi - lo, 0.5) * 100
    df["srsi_k"] = raw.rolling(k).mean()
    df["srsi_d"] = df["srsi_k"].rolling(d).mean()
    kk, dd = df["srsi_k"], df["srsi_d"]
    df["sig_srsi"] = np.select(
        [cross_up(kk, dd) & (np.minimum(kk, kk.shift(1)) < low),     # 과매도권 골든크로스
         cross_down(kk, dd) & (np.maximum(kk, kk.shift(1)) > high)],  # 과매수권 데드크로스
        [1, -1], 0)
    return df


# ---------------- 3) Stochastic ----------------
def add_stoch(df, n=14, k=3, d=3, low=20, high=80):
    ll = df["low"].rolling(n).min()
    hh = df["high"].rolling(n).max()
    raw = safe_ratio(df["close"] - ll, hh - ll, 0.5) * 100
    df["stoch_k"] = raw.rolling(k).mean()
    df["stoch_d"] = df["stoch_k"].rolling(d).mean()
    kk, dd = df["stoch_k"], df["stoch_d"]
    df["sig_stoch"] = np.select(
        [cross_up(kk, dd) & (np.minimum(kk, kk.shift(1)) < low),
         cross_down(kk, dd) & (np.maximum(kk, kk.shift(1)) > high)],
        [1, -1], 0)
    return df


# ---------------- 4) Bollinger Band %B ----------------
def add_bb(df, n=20, k=2.0):
    mid = df["close"].rolling(n).mean()
    std = df["close"].rolling(n).std(ddof=0)
    df["bb_mid"], df["bb_up"], df["bb_dn"] = mid, mid + k * std, mid - k * std
    df["bb_pctb"] = safe_ratio(df["close"] - df["bb_dn"], df["bb_up"] - df["bb_dn"], 0.5)
    b = df["bb_pctb"]
    df["sig_bb"] = np.select(
        [(b.shift(1) < 0) & (b >= 0),     # 하단 이탈 후 밴드 안으로 복귀
         (b.shift(1) > 1) & (b <= 1)],    # 상단 이탈 후 밴드 안으로 복귀
        [1, -1], 0)
    return df


# ---------------- 시각화 2 ----------------
def style(ax, ylabel):
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1), frameon=False, fontsize=8)


def marks(ax, a, sig_col, y):
    lg, sh = a[a[sig_col] == 1], a[a[sig_col] == -1]
    ax.scatter(lg.index, y.loc[lg.index], marker="^", s=55, color=LONG_C,
               edgecolor="white", lw=1, zorder=4, label=f"Long ({len(lg)})")
    ax.scatter(sh.index, y.loc[sh.index], marker="v", s=55, color=SHORT_C,
               edgecolor="white", lw=1, zorder=4, label=f"Short ({len(sh)})")


def plot_indicators(a, path):
    fig, axes = plt.subplots(5, 1, figsize=(13, 13), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1, 1.2, 1, 1], "hspace": 0.12})
    fig.patch.set_facecolor("white")
    ax0, ax1, ax2, ax3, ax4 = axes

    # (1) 가격 + 볼린저밴드
    ax0.fill_between(a.index, a["bb_dn"], a["bb_up"], color=BAND, lw=0, label="볼린저밴드(20, 2σ)")
    ax0.plot(a.index, a["bb_mid"], color=INK2, lw=1, ls="--", label="중심선(20)")
    ax0.plot(a.index, a["close"], color=BLUE, lw=2, label="USDT/KRW 종가")
    marks(ax0, a, "sig_bb", a["close"])
    ax0.set_title("기술적 지표 4종과 매매 신호 (2026-11-01 ~ 11-10, 1시간봉)",
                  loc="left", fontsize=13, color=INK, pad=10)
    style(ax0, "가격 (원)")

    # (2) BB %B
    ax1.plot(a.index, a["bb_pctb"], color=BLUE, lw=1.6, label="%B")
    for v in (0, 1):
        ax1.axhline(v, color=INK2, lw=0.8, ls="--")
    marks(ax1, a, "sig_bb", a["bb_pctb"])
    style(ax1, "BB %B")

    # (3) MACD
    pos = a["macd_hist"].clip(lower=0)
    neg = a["macd_hist"].clip(upper=0)
    ax2.bar(a.index, pos, width=0.035, color="#86b6ef", label="히스토그램(+)")
    ax2.bar(a.index, neg, width=0.035, color="#f2b49b", label="히스토그램(-)")
    ax2.plot(a.index, a["macd"], color=BLUE, lw=1.6, label="MACD")
    ax2.plot(a.index, a["macd_signal"], color=ORANGE, lw=1.6, label="시그널(9)")
    ax2.axhline(0, color=INK2, lw=0.8)
    marks(ax2, a, "sig_macd", a["macd"])
    style(ax2, "MACD")

    # (4) Stochastic RSI
    ax3.plot(a.index, a["srsi_k"], color=BLUE, lw=1.6, label="%K")
    ax3.plot(a.index, a["srsi_d"], color=ORANGE, lw=1.6, label="%D")
    for v in (20, 80):
        ax3.axhline(v, color=INK2, lw=0.8, ls="--")
    marks(ax3, a, "sig_srsi", a["srsi_k"])
    ax3.set_ylim(-5, 105)
    style(ax3, "Stoch RSI")

    # (5) Stochastic
    ax4.plot(a.index, a["stoch_k"], color=BLUE, lw=1.6, label="%K")
    ax4.plot(a.index, a["stoch_d"], color=ORANGE, lw=1.6, label="%D")
    for v in (20, 80):
        ax4.axhline(v, color=INK2, lw=0.8, ls="--")
    marks(ax4, a, "sig_stoch", a["stoch_k"])
    ax4.set_ylim(-5, 105)
    style(ax4, "Stochastic")

    ax4.xaxis.set_major_locator(mdates.DayLocator(tz=a.index.tz))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d", tz=a.index.tz))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------- 실행 ----------------
if __name__ == "__main__":
    df = load()
    df = add_macd(df)
    df = add_stoch_rsi(df)
    df = add_stoch(df)
    df = add_bb(df)
    a = df[df["is_analysis"]]

    # 준비 구간 덕분에 분석 기간에는 결측이 없어야 함
    ind_cols = ["macd", "macd_signal", "srsi_k", "srsi_d", "stoch_k", "stoch_d", "bb_pctb"]
    n_nan = int(a[ind_cols].isna().sum().sum())
    print(f"[확인] 분석 기간 지표 결측: {n_nan}개" + ("  ✅" if n_nan == 0 else "  ⚠"))

    names = {"sig_macd": "MACD", "sig_srsi": "Stoch RSI", "sig_stoch": "Stochastic", "sig_bb": "BB %B"}
    rows = []
    for col, nm in names.items():
        rows.append({"지표": nm,
                     "Long 신호": int((a[col] == 1).sum()),
                     "Short 신호": int((a[col] == -1).sum()),
                     "합계": int((a[col] != 0).sum())})
    sig_table = pd.DataFrame(rows)

    last = a.iloc[-1]
    state = {
        "MACD": f"{last['macd']:.2f} / 시그널 {last['macd_signal']:.2f} → "
                + ("상승 추세(MACD>시그널)" if last["macd_hist"] > 0 else "하락 추세(MACD<시그널)"),
        "Stoch RSI": f"%K {last['srsi_k']:.1f} / %D {last['srsi_d']:.1f} → "
                     + ("과매수(>80)" if last["srsi_k"] > 80 else "과매도(<20)" if last["srsi_k"] < 20 else "중립"),
        "Stochastic": f"%K {last['stoch_k']:.1f} / %D {last['stoch_d']:.1f} → "
                      + ("과매수(>80)" if last["stoch_k"] > 80 else "과매도(<20)" if last["stoch_k"] < 20 else "중립"),
        "BB %B": f"{last['bb_pctb']:.2f} → "
                 + ("상단 돌파(>1)" if last["bb_pctb"] > 1 else "하단 이탈(<0)" if last["bb_pctb"] < 0
                    else "밴드 상단부" if last["bb_pctb"] > 0.8 else "밴드 하단부" if last["bb_pctb"] < 0.2 else "밴드 중앙부"),
    }

    print("\n========== 지표별 신호 발생 횟수 (11/1~11/10) ==========")
    print(sig_table.to_string(index=False))
    print(f"\n========== 마지막 봉 상태 ({a.index[-1]:%m/%d %H시}, 종가 {last['close']:,.0f}원) ==========")
    for k, v in state.items():
        print(f"{k:<11}: {v}")

    # 신호가 여러 지표에서 동시에(3시간 이내) 나온 횟수 - STEP 6 신호 통합 준비
    sig_sum = sum(df[c].replace(0, np.nan).ffill(limit=3).fillna(0) for c in names)[df["is_analysis"]]
    print(f"\n2개 이상 지표가 같은 방향(3시간 이내): Long {int((sig_sum >= 2).sum())}시간, "
          f"Short {int((sig_sum <= -2).sum())}시간")

    df.to_csv(PROC / "features_1h.csv", encoding="utf-8-sig")
    md = "| " + " | ".join(sig_table.columns) + " |\n|" + "---|" * len(sig_table.columns) + "\n"
    md += "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in sig_table.values)
    md += f"\n\n마지막 봉 ({a.index[-1]:%m/%d %H시}) 상태\n\n| 지표 | 상태 |\n|---|---|\n"
    md += "\n".join(f"| {k} | {v} |" for k, v in state.items())
    (PROC / "indicator_stats.md").write_text(md, encoding="utf-8")

    plot_indicators(a, IMG / "02_indicators.png")
    print("\n저장: data/processed/features_1h.csv, data/processed/indicator_stats.md, "
          "images/02_indicators.png")