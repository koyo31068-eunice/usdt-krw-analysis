"""
STEP 6. 신호 통합(Long/Short 점수) + 신뢰성 확률(%) 백테스트
- 통합 점수: 4개 지표 신호(최근 3시간 유지)를 더한 값  (-4 ~ +4)
            +2 이상 → Long, -2 이하 → Short, 그 외 → Neutral
- 검증 방법: 신호가 나온 '다음 봉 시가'에 진입했다고 가정 (미래 정보 사용 방지)
            h시간 뒤 종가 기준으로 방향이 맞았는지 확인
- 신뢰성 확률 = 적중률 = 승 / (승 + 패)   (보합은 제외, 95% 신뢰구간 함께 표시)
- 입력: data/processed/features_1h.csv
- 출력: data/processed/features_1h.csv (점수 컬럼 추가)
        data/processed/backtest_stats.md
        images/04_hit_rate.png        (시각화 4)
        images/05_composite_score.png (시각화 5)
"""
from math import sqrt
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

HOLD = 3                    # 개별 지표 신호를 3시간 동안 유효한 것으로 봄
TH = 2                      # 통합 점수 ±2 이상이면 Long/Short
H_LIST = (1, 4, 12, 24)     # 검증 보유 시간
MAIN_H = 4                  # 대표 보유 시간: 4시간
FEE = 0.08                  # 왕복 수수료 가정(%) : 0.04% x 2

SIG_COLS = {"sig_macd": "MACD", "sig_srsi": "Stoch RSI", "sig_stoch": "Stochastic", "sig_bb": "BB %B"}

BLUE, ORANGE = "#2a78d6", "#eb6834"
LONG_C, SHORT_C = "#e34948", "#0d366b"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"


def load():
    df = pd.read_csv(PROC / "features_1h.csv")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(KST)
    df = df.set_index("datetime")
    for c in ["is_filled", "is_ret_outlier", "fx_stale", "is_analysis", "vol_anom", "iso_anom"]:
        df[c] = df[c].astype(str).str.lower().eq("true")
    return df


# ---------------- 1) 통합 점수 ----------------
def add_score(df):
    for c in SIG_COLS:
        df[c + "_hold"] = df[c].replace(0, np.nan).ffill(limit=HOLD).fillna(0)
    df["score"] = sum(df[c + "_hold"] for c in SIG_COLS).astype(int)
    df["decision"] = np.select([df["score"] >= TH, df["score"] <= -TH], ["Long", "Short"], "Neutral")
    return df


def entry_events(direction):
    """방향(+1/-1/0)이 새로 바뀌는 시점만 진입으로 셈 (같은 신호 반복 진입 방지)"""
    return direction.where((direction != 0) & (direction != direction.shift(1)), 0)


# ---------------- 2) 백테스트 ----------------
def wilson(k, n, z=1.96):
    """적중률의 95% 신뢰구간 (표본이 적을 때도 안정적인 Wilson 방식)"""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (center - half) * 100, (center + half) * 100


def evaluate(df, events, h):
    """events: +1(Long)/-1(Short) 이 찍힌 시리즈. 다음 봉 시가 진입 → h시간 뒤 종가 청산"""
    entry = df["open"].shift(-1)
    exit_ = df["close"].shift(-h)
    ret = (exit_ / entry - 1) * 100 * events
    ev = events[(events != 0) & df["is_analysis"]]
    r = ret.loc[ev.index].dropna()
    win, loss, tie = int((r > 0).sum()), int((r < 0).sum()), int((r == 0).sum())
    n = win + loss
    lo, hi = wilson(win, n)
    return {"신호수": len(r), "승": win, "패": loss, "보합": tie,
            "적중률(%)": round(win / n * 100, 1) if n else np.nan,
            "95%CI 하한": round(lo, 1) if n else np.nan,
            "95%CI 상한": round(hi, 1) if n else np.nan,
            "평균수익률(%)": round(float(r.mean()), 3) if len(r) else np.nan,
            "수수료차감(%)": round(float(r.mean()) - FEE, 3) if len(r) else np.nan}


def baseline(df, h):
    """기준선: 아무 시간에나 Long 했을 때의 적중률 (Short은 100 - 이 값)"""
    a = df["is_analysis"]
    r = (df["close"].shift(-h) / df["open"].shift(-1) - 1)[a].dropna()
    up, dn = (r > 0).sum(), (r < 0).sum()
    return round(up / (up + dn) * 100, 1)


# ---------------- 시각화 ----------------
def base_style(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def plot_hit_rate(tbl, base_long, path):
    t = tbl.dropna(subset=["적중률(%)"]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(t) + 1.8))
    fig.patch.set_facecolor("white")
    y = np.arange(len(t))
    err = [t["적중률(%)"] - t["95%CI 하한"], t["95%CI 상한"] - t["적중률(%)"]]
    ax.barh(y, t["적중률(%)"], height=0.55, color=BLUE)
    ax.errorbar(t["적중률(%)"], y, xerr=err, fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
    ax.axvline(50, color=INK2, lw=1, ls="--")
    ax.axvline(base_long, color=ORANGE, lw=1.2, ls=":")
    ax.text(50, len(t) - 0.35, " 50%(동전 던지기)", color=INK2, fontsize=8, va="bottom")
    ax.text(base_long, len(t) - 0.35, f"아무 때나 Long {base_long:.0f}% ", color=ORANGE, fontsize=8,
            va="bottom", ha="right")
    for yi, (_, r) in zip(y, t.iterrows()):
        ax.text(min(r["95%CI 상한"], 100) + 1.5, yi,
                f"{r['적중률(%)']:.0f}%  (승 {int(r['승'])} / 패 {int(r['패'])})", va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(t.index, fontsize=10, color=INK)
    ax.set_xlim(0, 125)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("적중률 (%) - 막대 끝 가는 선은 95% 신뢰구간", color=INK2, fontsize=9)
    ax.set_title(f"신호별 {MAIN_H}시간 후 방향 적중률 (2026-11-01 ~ 11-10, 다음 봉 시가 진입)",
                 loc="left", fontsize=13, color=INK, pad=24)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    base_style(ax)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_score(a, events, path):
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1], "hspace": 0.12})
    fig.patch.set_facecolor("white")
    ev = events[a.index]
    lg, sh = a[ev == 1], a[ev == -1]

    ax0.plot(a.index, a["close"], color=BLUE, lw=2, label="USDT/KRW 종가")
    ax0.scatter(lg.index, lg["close"], marker="^", s=90, color=LONG_C, edgecolor="white", lw=1.2,
                zorder=4, label=f"통합 Long 진입 ({len(lg)})")
    ax0.scatter(sh.index, sh["close"], marker="v", s=90, color=SHORT_C, edgecolor="white", lw=1.2,
                zorder=4, label=f"통합 Short 진입 ({len(sh)})")
    ax0.set_title("통합 신호 점수와 Long/Short 진입 시점 (2026-11-01 ~ 11-10)",
                  loc="left", fontsize=13, color=INK, pad=10)
    ax0.set_ylabel("가격 (원)", color=INK2, fontsize=9)

    s = a["score"]
    ax1.fill_between(a.index, 0, s.where(s > 0, 0), step="mid", color="#f4a3a3", lw=0, label="Long 쪽 점수")
    ax1.fill_between(a.index, 0, s.where(s < 0, 0), step="mid", color="#9ec5f4", lw=0, label="Short 쪽 점수")
    ax1.step(a.index, s, where="mid", color=INK2, lw=0.8)
    for v in (TH, -TH):
        ax1.axhline(v, color=INK2, lw=0.9, ls="--")
    ax1.axhline(0, color=INK2, lw=0.8)
    ax1.set_ylim(-4.5, 4.5)
    ax1.set_yticks(range(-4, 5, 2))
    ax1.set_ylabel(f"통합 점수 (±{TH} 기준)", color=INK2, fontsize=9)

    for ax in (ax0, ax1):
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        base_style(ax)
        ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1), frameon=False, fontsize=8)
    ax1.xaxis.set_major_locator(mdates.DayLocator(tz=a.index.tz))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d", tz=a.index.tz))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def tidy(t):
    """표 정리: 개수 컬럼은 정수로"""
    for c in ["신호수", "승", "패", "보합"]:
        t[c] = t[c].astype(int)
    return t


def to_md(t, index=True):
    t = t.reset_index() if index else t
    s = "| " + " | ".join(map(str, t.columns)) + " |\n|" + "---|" * len(t.columns) + "\n"
    return s + "\n".join("| " + " | ".join("-" if pd.isna(v) else str(v) for v in r) + " |" for r in t.values)


# ---------------- 실행 ----------------
if __name__ == "__main__":
    df = add_score(load())
    a = df[df["is_analysis"]]

    # 검증할 신호 묶음
    comp_dir = pd.Series(np.select([df["score"] >= TH, df["score"] <= -TH], [1, -1], 0), index=df.index)
    strong_dir = pd.Series(np.select([df["score"] >= 3, df["score"] <= -3], [1, -1], 0), index=df.index)
    prem_dir = pd.Series(np.select([df["premium_state"] == "과열", df["premium_state"] == "역과열"],
                                   [-1, 1], 0), index=df.index)
    candle = np.sign(df["close"] - df["open"])
    vol_dir = (candle * df["vol_anom"]).astype(int)

    sources = {nm: df[c] for c, nm in SIG_COLS.items()}
    sources["괴리 과열→Short(평균회귀)"] = entry_events(prem_dir)
    sources["거래량 급증 방향 추종"] = entry_events(vol_dir)
    sources[f"통합 점수 ±{TH}"] = entry_events(comp_dir)
    sources["통합 점수 ±3 (강신호)"] = entry_events(strong_dir)

    # ① 대표 보유시간(4h) 신호별 적중률
    main = tidy(pd.DataFrame({k: evaluate(df, v, MAIN_H) for k, v in sources.items()}).T)
    base_long = baseline(df, MAIN_H)
    print(f"========== 신호별 {MAIN_H}시간 후 적중률 (다음 봉 시가 진입, 왕복 수수료 {FEE}% 가정) ==========")
    print(main.to_string())
    print(f"\n[기준선] 아무 때나 Long 적중률: {base_long}%  |  아무 때나 Short 적중률: {100 - base_long:.1f}%")

    # ② 통합 신호: 보유시간별 + Long/Short 별
    comp_ev = sources[f"통합 점수 ±{TH}"]
    by_h = tidy(pd.DataFrame({f"{h}시간": evaluate(df, comp_ev, h) for h in H_LIST}).T)
    by_side = pd.DataFrame({
        "Long": evaluate(df, comp_ev.where(comp_ev == 1, 0), MAIN_H),
        "Short": evaluate(df, comp_ev.where(comp_ev == -1, 0), MAIN_H)}).T
    by_side = tidy(by_side)
    print(f"\n========== 통합 점수 ±{TH}: 보유 시간별 ==========")
    print(by_h.to_string())
    print(f"\n========== 통합 점수 ±{TH}: Long / Short 별 ({MAIN_H}시간) ==========")
    print(by_side.to_string())

    # ③ 마지막 봉 판단 + 신뢰성 확률
    last = a.iloc[-1]
    side = last["decision"]
    if side == "Neutral":
        conf = "-"
    else:
        r = by_side.loc[side]
        conf = f"{r['적중률(%)']}% (95%CI {r['95%CI 하한']}~{r['95%CI 상한']}%, 과거 {int(r['승'] + r['패'])}회 기준)"
    held = ", ".join(f"{nm} {'+' if last[c + '_hold'] > 0 else ''}{int(last[c + '_hold'])}"
                     for c, nm in SIG_COLS.items())
    print(f"\n========== 최종 판단 ({a.index[-1]:%m/%d %H시}, 종가 {last['close']:,.0f}원) ==========")
    print(f"통합 점수     : {int(last['score']):+d}  ({held})")
    print(f"트렌드 판단   : {side}")
    print(f"신뢰성 확률   : {conf}")
    print(f"괴리율        : {last['premium_pct']:+.2f}% (상태: {last['premium_state']})")
    print(f"분석 기간 시간 비중: Long {(a['decision'] == 'Long').mean()*100:.1f}% / "
          f"Short {(a['decision'] == 'Short').mean()*100:.1f}% / "
          f"Neutral {(a['decision'] == 'Neutral').mean()*100:.1f}%")

    # 저장
    df.to_csv(PROC / "features_1h.csv", encoding="utf-8-sig")
    md = f"### 신호별 {MAIN_H}시간 후 적중률\n\n" + to_md(main.rename_axis("신호"))
    md += f"\n\n기준선: 아무 때나 Long {base_long}% / Short {100 - base_long:.1f}%\n\n"
    md += f"### 통합 점수 ±{TH} 보유 시간별\n\n" + to_md(by_h.rename_axis("보유"))
    md += f"\n\n### 통합 점수 ±{TH} Long/Short 별\n\n" + to_md(by_side.rename_axis("방향"))
    md += (f"\n\n### 최종 판단 ({a.index[-1]:%m/%d %H시})\n\n| 항목 | 값 |\n|---|---|\n"
           f"| 통합 점수 | {int(last['score']):+d} ({held}) |\n| 트렌드 | {side} |\n| 신뢰성 확률 | {conf} |\n")
    (PROC / "backtest_stats.md").write_text(md, encoding="utf-8")

    plot_hit_rate(main, base_long, IMG / "04_hit_rate.png")
    plot_score(a, comp_ev, IMG / "05_composite_score.png")
    print("\n저장: data/processed/backtest_stats.md, images/04_hit_rate.png, images/05_composite_score.png")