"""
STEP 1. 데이터 수집
- 빗썸 USDT/KRW 1시간봉 (출처: Bithumb Public API v1)
- USD/KRW 환율 1시간봉   (출처: Yahoo Finance, 티커 KRW=X)
- 기간: 2026-10-25 00시 ~ 2026-11-10 23시 (KST)
        └ 분석 기간은 11/1~11/10, 그 이전 7일은 지표 준비 구간
"""
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

# ---------------- 설정 ----------------
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

KST = "Asia/Seoul"
START_KST = pd.Timestamp("2026-10-25 00:00", tz=KST)   # 준비 구간 시작
END_KST   = pd.Timestamp("2026-11-11 00:00", tz=KST)   # 이 시각 '직전'까지 (11/10 23시봉 포함)
ANALYSIS_START = pd.Timestamp("2026-11-01 00:00", tz=KST)


# ---------------- 1) 빗썸 USDT/KRW ----------------
def fetch_bithumb(start_kst, end_kst):
    url = "https://api.bithumb.com/v1/candles/minutes/60"
    rows = []
    # 'to'에 KST 시각 전달.
    # (빗썸이 UTC로 해석하더라도 겹쳐 받을 뿐 빠지지 않으므로 어느 쪽이든 안전)
    to = end_kst.tz_localize(None)

    while True:
        params = {
            "market": "KRW-USDT",
            "count": 200,
            "to": to.strftime("%Y-%m-%d %H:%M:%S"),
        }
        r = requests.get(url, params=params,
                         headers={"accept": "application/json"}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data:
            break

        rows.extend(data)
        oldest = pd.Timestamp(data[-1]["candle_date_time_kst"])   # KST 기준
        print(f"  {len(data):>3}개 수신 | 가장 오래된 봉(KST): {oldest:%Y-%m-%d %H:%M}")

        if oldest <= start_kst.tz_localize(None):
            break
        to = oldest
        time.sleep(0.2)

    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["candle_date_time_kst"]).dt.tz_localize(KST)
    df = df.rename(columns={
        "opening_price": "open",
        "high_price": "high",
        "low_price": "low",
        "trade_price": "close",
        "candle_acc_trade_volume": "volume",
        "candle_acc_trade_price": "value",
    })
    df = df[["datetime", "open", "high", "low", "close", "volume", "value"]]
    df = (df.drop_duplicates("datetime")
            .sort_values("datetime")
            .query("@start_kst <= datetime < @end_kst")
            .reset_index(drop=True))
    return df

# ---------------- 2) USD/KRW 환율 ----------------
def fetch_fx(start_kst, end_kst):
    # 10/25, 11/1이 일요일이라 주말 공백을 메울 금요일 값까지 받도록 3일 앞당김
    fx = yf.download(
        "KRW=X",
        start=(start_kst - pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
        end=(end_kst + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1h", progress=False, auto_adjust=False,
    )
    if isinstance(fx.columns, pd.MultiIndex):    # ('Close','KRW=X') → 'Close'
        fx.columns = fx.columns.get_level_values(0)

    fx.index = fx.index.tz_convert(KST)
    fx = fx[["Close"]].rename(columns={"Close": "usdkrw"})
    fx.index.name = "datetime"
    return fx.reset_index()


# ---------------- 실행 ----------------
if __name__ == "__main__":
    print("[1] 빗썸 USDT/KRW 1시간봉 수집 중...")
    bt = fetch_bithumb(START_KST, END_KST)
    bt_path = RAW_DIR / "bithumb_usdtkrw_1h_raw.csv"
    bt.to_csv(bt_path, index=False, encoding="utf-8-sig")

    print("\n[2] USD/KRW 환율 1시간봉 수집 중...")
    fx = fetch_fx(START_KST, END_KST)
    fx_path = RAW_DIR / "usdkrw_1h_raw.csv"
    fx.to_csv(fx_path, index=False, encoding="utf-8-sig")

    # ---------- 요약 ----------
    n_analysis = (bt["datetime"] >= ANALYSIS_START).sum()
    print("\n========== 수집 요약 ==========")
    print(f"빗썸  : {len(bt)}행 | {bt['datetime'].min():%m/%d %H시} ~ {bt['datetime'].max():%m/%d %H시}")
    print(f"        └ 분석 기간(11/1~11/10) 봉 개수: {n_analysis} / 240")
    print(f"환율  : {len(fx)}행 | {fx['datetime'].min():%m/%d %H시} ~ {fx['datetime'].max():%m/%d %H시}")
    print(f"저장  : {bt_path.name}, {fx_path.name}")
    print("\n[빗썸 앞 3행]")
    print(bt.head(3).to_string(index=False))