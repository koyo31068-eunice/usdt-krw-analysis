"""
STEP 2. 데이터 정제
- 기본 정보 확인 → 결측/이상치 처리 → 빗썸 + 환율 병합
- 결과: data/processed/merged_1h.csv, data/processed/data_summary.md
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
PROC = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

KST = "Asia/Seoul"
START = pd.Timestamp("2026-10-25 00:00", tz=KST)
END = pd.Timestamp("2026-11-10 23:00", tz=KST)
ANALYSIS_START = pd.Timestamp("2026-11-01 00:00", tz=KST)
FULL_INDEX = pd.date_range(START, END, freq="h")     # 1시간 간격 전체 시각(408개)


# ---------------- ① 불러오기 & 기본 정보 ----------------
def load_csv(name):
    df = pd.read_csv(RAW / name)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(KST)
    return df


def basic_info(df, title):
    print(f"\n----- [{title}] 기본 정보 -----")
    print(f"행 x 열      : {df.shape[0]} x {df.shape[1]}")
    print(f"기간         : {df['datetime'].min():%Y-%m-%d %H시} ~ {df['datetime'].max():%Y-%m-%d %H시}")
    print(f"컬럼         : {list(df.columns)}")
    print(f"중복 시각    : {df['datetime'].duplicated().sum()}개")
    print("컬럼별 결측치:", df.isna().sum().to_dict())


# ---------------- ②③④ 빗썸 정제 ----------------
def clean_bithumb(bt):
    bt = bt.drop_duplicates("datetime", keep="last").set_index("datetime").sort_index()

    # ③ 논리 오류 제거
    price_cols = ["open", "high", "low", "close"]
    bad = (
        (bt["high"] < bt["low"])
        | (bt["high"] < bt[["open", "close"]].max(axis=1))
        | (bt["low"] > bt[["open", "close"]].min(axis=1))
        | (bt[price_cols] <= 0).any(axis=1)
    )
    n_bad = int(bad.sum())
    bt = bt[~bad]

    # ② 빠진 시간대 채우기
    bt = bt.reindex(FULL_INDEX)
    bt["is_filled"] = bt["close"].isna()
    n_missing = int(bt["is_filled"].sum())
    bt["close"] = bt["close"].ffill().bfill()
    for c in ["open", "high", "low"]:
        bt[c] = bt[c].fillna(bt["close"])
    bt[["volume", "value"]] = bt[["volume", "value"]].fillna(0)

    # ④ 가격 급변 표시 (삭제 X)
    log_ret = np.log(bt["close"]).diff()
    z = (log_ret - log_ret.mean()) / log_ret.std()
    bt["is_ret_outlier"] = z.abs() > 5

    stats = {"논리오류_제거": n_bad, "빠진시간대_채움": n_missing,
             "가격급변_표시": int(bt["is_ret_outlier"].sum())}
    return bt, stats


# ---------------- ⑤⑥ 환율 정제 ----------------
def clean_fx(fx):
    fx = fx.drop_duplicates("datetime", keep="last").set_index("datetime").sort_index()
    fx = fx[fx["usdkrw"] > 0]

    # ⑤ 오류 틱: 앞뒤 5시간 중앙값과 1% 이상 차이 나면 제거
    med = fx["usdkrw"].rolling(5, center=True, min_periods=1).median()
    spike = (fx["usdkrw"] / med - 1).abs() > 0.01
    n_spike = int(spike.sum())
    fx = fx[~spike]

    # ⑥ 주말·휴장 공백: 직전 값 유지
    stale = ~FULL_INDEX.isin(fx.index)
    fx_full = fx.reindex(fx.index.union(FULL_INDEX)).sort_index()
    fx_full["usdkrw"] = fx_full["usdkrw"].ffill().bfill()
    fx_full = fx_full.loc[FULL_INDEX]
    fx_full["fx_stale"] = stale

    stats = {"환율_오류틱_제거": n_spike, "환율_공백_채움": int(stale.sum())}
    return fx_full, stats


# ---------------- 실행 ----------------
if __name__ == "__main__":
    bt_raw = load_csv("bithumb_usdtkrw_1h_raw.csv")
    fx_raw = load_csv("usdkrw_1h_raw.csv")
    basic_info(bt_raw, "빗썸 USDT/KRW 원본")
    basic_info(fx_raw, "USD/KRW 환율 원본")

    bt, s1 = clean_bithumb(bt_raw)
    fx, s2 = clean_fx(fx_raw)

    # ⑦ 병합
    df = bt.join(fx)
    df["is_analysis"] = df.index >= ANALYSIS_START
    df.index.name = "datetime"
    df.to_csv(PROC / "merged_1h.csv", encoding="utf-8-sig")

    # ---------- 정제 결과 요약 ----------
    a = df[df["is_analysis"]]
    summary = {
        "분석 기간": f"{a.index.min():%Y-%m-%d %H시} ~ {a.index.max():%Y-%m-%d %H시} (KST)",
        "분석 데이터 개수": f"{len(a)}개 (1시간봉)",
        "전체(준비구간 포함)": f"{len(df)}개",
        **{k: f"{v}건" for k, v in {**s1, **s2}.items()},
        "분석기간 빗썸 채운 봉": f"{int(a['is_filled'].sum())}개",
        "분석기간 환율 공백 비율": f"{a['fx_stale'].mean()*100:.1f}%",
        "USDT/KRW 종가 범위": f"{a['close'].min():,.0f} ~ {a['close'].max():,.0f}원 (평균 {a['close'].mean():,.1f})",
        "USD/KRW 환율 범위": f"{a['usdkrw'].min():,.1f} ~ {a['usdkrw'].max():,.1f}원",
        "최종 결측치": f"{int(df.isna().sum().sum())}개",
    }

    print("\n========== 정제 결과 요약 ==========")
    for k, v in summary.items():
        print(f"{k:<22}: {v}")

    # 리포트용 마크다운 표 저장
    md = "| 항목 | 값 |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in summary.items())
    (PROC / "data_summary.md").write_text(md, encoding="utf-8")
    print("\n저장: data/processed/merged_1h.csv, data/processed/data_summary.md")
    print("\n[병합 데이터 앞 3행 (분석 기간)]")
    print(a.head(3).round(2).to_string())