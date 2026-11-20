"""
STEP 5. 거래량 이상 징후 분석
- 방법 1: 로그 거래량 z-score (직전 168시간 기준) ≥ 3  → 거래량 급증
- 방법 2: IsolationForest (거래량 z, 가격 변동폭, 괴리율 변화폭 3가지를 함께 봄)
- 입력: data/processed/features_1h.csv
- 출력: data/processed/features_1h.csv (이상징후 컬럼 추가)
        data/processed/anomaly_stats.md
        images/03_volume_anomaly.png    (시각화 3)
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

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

WINDOW = 168          # 거래량 기준 기간: 직전 1주
Z_TH = 2.0            # 거래량 z-score 2 이상 = 이상 징후 (z≥3은 0건이라 완화)
ISO_RATE = 0.03       # IsolationForest: 전체의 약 3%를 이상치로 지정
HORIZONS = (1, 4, 12)

BLUE, ORANGE = "#2a78d6", "#eb6834"
ANOM_C, ISO_C = "#d03b3b", "#eda100"
INK, INK2, GRID, BAR = "#0b0b0b", "#52514e", "#e5e4e0", "#b7d3f6"


def load():
    df = pd.read_csv(PROC / "features_1h.csv")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(KST)
    df = df.set_index("datetime")
    for c in ["is_filled", "is_ret_outlier", "fx_stale", "is_analysis"]:
        df[c] = df[c].astype(str).str.lower().eq("true")
    return df


# ---------------- 1) 거래량 z-score ----------------
def add_volume_z(df):
    lv = np.log1p(df["volume"])
    # shift(1): 현재 봉은 기준값 계산에서 빼야 급증한 봉이 자기 기준을 올리지 않음
    base_mean = lv.rolling(WINDOW, min_periods=WINDOW).mean().shift(1)
    base_std = lv.rolling(WINDOW, min_periods=WINDOW).std().shift(1)
    df["vol_z"] = (lv - base_mean) / base_std
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(WINDOW).median().shift(1)  # 평소 대비 몇 배
    df["vol_anom"] = df["vol_z"] >= Z_TH
    return df


# ---------------- 2) IsolationForest ----------------
def add_iso(df):
    df["abs_ret"] = (np.log(df["close"]).diff().abs() * 100)
    df["abs_dprem"] = df["premium_pct"].diff().abs()
    a = df["is_analysis"]
    X = df.loc[a, ["vol_z", "abs_ret", "abs_dprem"]].fillna(0)
    model = IsolationForest(contamination=ISO_RATE, random_state=42)
    df["iso_anom"] = False
    df.loc[a, "iso_anom"] = model.fit_predict(X) == -1
    return df


# ---------------- 3) 이상징후 이후 변화 ----------------
def add_forward(df):
    for h in HORIZONS:
        df[f"fwd_ret_{h}h"] = (df["close"].shift(-h) / df["close"] - 1) * 100
        df[f"fwd_absret_{h}h"] = df[f"fwd_ret_{h}h"].abs()
        df[f"fwd_absdprem_{h}h"] = (df["premium_pct"].shift(-h) - df["premium_pct"]).abs()
    return df


def count_events(mask):
    """연속된 이상 시간은 하나의 사건으로 묶어서 셈"""
    return int((mask & ~mask.shift(1, fill_value=False)).sum())


# ---------------- 시각화 3 ----------------
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


def plot_anomaly(a, path):
    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1, figsize=(13, 9), sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1.2, 1], "hspace": 0.12})
    fig.patch.set_facecolor("white")
    va, iso = a[a["vol_anom"]], a[a["iso_anom"] & ~a["vol_anom"]]

    # (1) 가격 + 이상징후 위치
    ax0.plot(a.index, a["close"], color=BLUE, lw=2, label="USDT/KRW 종가")
    ax0.scatter(va.index, va["close"], s=70, color=ANOM_C, edgecolor="white", lw=1.2,
                zorder=4, label=f"거래량 급증 z≥{Z_TH:.0f} ({len(va)}시간)")
    ax0.scatter(iso.index, iso["close"], s=55, marker="D", color=ISO_C, edgecolor="white",
                lw=1.2, zorder=4, label=f"IsolationForest만 탐지 ({len(iso)}시간)")
    ax0.set_title("거래량 이상 징후와 가격 (2026-11-01 ~ 11-10, 1시간봉)",
                  loc="left", fontsize=13, color=INK, pad=10)
    style(ax0, "가격 (원)")

    # (2) 거래량 막대 (만 USDT)
    vol = a["volume"] / 1e4
    colors = np.where(a["vol_anom"], ANOM_C, BAR)
    ax1.bar(a.index, vol, width=0.035, color=colors)
    ax1.plot(a.index, (a["volume"] / a["vol_ratio"]) / 1e4, color=INK2, lw=1, ls="--",
             label="직전 1주 중앙값")
    # 최대 거래량 직접 표기
    t = vol.idxmax()
    ax1.annotate(f"{vol.max():,.0f}만 ({t:%m/%d %H시})", (t, vol.max()),
                 xytext=(6, -4), textcoords="offset points", fontsize=8, color=INK)
    style(ax1, "거래량 (만 USDT)")
    h, l = ax1.get_legend_handles_labels()
    h = [plt.Rectangle((0, 0), 1, 1, color=BAR), plt.Rectangle((0, 0), 1, 1, color=ANOM_C)] + h
    l = ["거래량", "이상 거래량"] + l
    ax1.legend(h, l, loc="upper left", bbox_to_anchor=(1.005, 1), frameon=False, fontsize=8)

    # (3) 거래량 z-score
    ax2.plot(a.index, a["vol_z"], color=BLUE, lw=1.4, label="거래량 z-score")
    ax2.axhline(Z_TH, color=ANOM_C, lw=1, ls="--", label=f"기준선 z={Z_TH:.0f}")
    ax2.axhline(0, color=INK2, lw=0.8)
    style(ax2, "z-score")

    ax2.xaxis.set_major_locator(mdates.DayLocator(tz=a.index.tz))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d", tz=a.index.tz))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------- 실행 ----------------
if __name__ == "__main__":
    df = add_forward(add_iso(add_volume_z(load())))
    a = df[df["is_analysis"]]
    n_nan = int(a["vol_z"].isna().sum())
    print(f"[확인] 분석 기간 거래량 z-score 결측: {n_nan}개" + ("  ✅" if n_nan == 0 else "  ⚠"))

    va, normal = a[a["vol_anom"]], a[~a["vol_anom"]]

    # ① 요약
    summary = {
        "평소 시간당 거래량(중앙값)": f"{a['volume'].median()/1e4:,.0f}만 USDT",
        "최대 시간당 거래량": f"{a['volume'].max()/1e4:,.0f}만 USDT ({a['volume'].idxmax():%m/%d %H시})",
        f"거래량 급증(z≥{Z_TH:.0f}) 시간": f"{len(va)}시간 / 사건 {count_events(a['vol_anom'])}건",
        "IsolationForest 이상치": f"{int(a['iso_anom'].sum())}시간",
        "두 방법 모두 탐지": f"{int((a['vol_anom'] & a['iso_anom']).sum())}시간",
        "급증 시간 중 상승봉 비율": f"{(va['close'] > va['open']).mean()*100:.0f}%" if len(va) else "-",
        "가격급변(STEP2) 2건과 겹침": f"{int((a['vol_anom'] & a['is_ret_outlier']).sum())}건",
    }
    print("\n========== 거래량 이상 징후 요약 (11/1~11/10) ==========")
    for k, v in summary.items():
        print(f"{k:<24}: {v}")

    # ② 이상징후 이후 vs 평상시
    rows = []
    for h in HORIZONS:
        rows.append({"구간": f"{h}시간 후",
                     "가격변동폭_급증후(%)": round(float(va[f"fwd_absret_{h}h"].mean()), 3),
                     "가격변동폭_평소(%)": round(float(normal[f"fwd_absret_{h}h"].mean()), 3),
                     "괴리변화폭_급증후(%p)": round(float(va[f"fwd_absdprem_{h}h"].mean()), 3),
                     "괴리변화폭_평소(%p)": round(float(normal[f"fwd_absdprem_{h}h"].mean()), 3)})
    cmp = pd.DataFrame(rows)
    print("\n========== 거래량 급증 이후 vs 평상시 (평균 변동폭) ==========")
    print(cmp.to_string(index=False))

    # ③ 급증 사건 목록 (z 높은 순)
    ev = va.sort_values("vol_z", ascending=False).head(10)
    ev_tbl = pd.DataFrame({
        "시각": ev.index.strftime("%m/%d %H시"),
        "종가": ev["close"].astype(int),
        "봉": np.where(ev["close"] > ev["open"], "상승", np.where(ev["close"] < ev["open"], "하락", "보합")),
        "거래량(만)": (ev["volume"] / 1e4).round(0).astype(int),
        "평소대비(배)": ev["vol_ratio"].round(1),
        "z": ev["vol_z"].round(2),
        "괴리율(%)": ev["premium_pct"].round(2),
        "4h후 수익률(%)": ev["fwd_ret_4h"].round(2),
    })
    print("\n========== 거래량 급증 시각 TOP 10 ==========")
    print(ev_tbl.to_string(index=False))

    # 저장
    df.to_csv(PROC / "features_1h.csv", encoding="utf-8-sig")

    def to_md(t):
        s = "| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns) + "\n"
        return s + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in t.values)

    md = "| 항목 | 값 |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in summary.items())
    md += "\n\n" + to_md(cmp) + "\n\n" + to_md(ev_tbl)
    (PROC / "anomaly_stats.md").write_text(md, encoding="utf-8")

    plot_anomaly(a, IMG / "03_volume_anomaly.png")
    print("\n저장: data/processed/features_1h.csv, data/processed/anomaly_stats.md, "
          "images/03_volume_anomaly.png")