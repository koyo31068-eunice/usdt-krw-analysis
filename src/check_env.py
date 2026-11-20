import sys
import pandas as pd, numpy as np, requests, yfinance as yf, matplotlib, sklearn

print("Python :", sys.version.split()[0])
print("pandas :", pd.__version__, "| numpy:", np.__version__)
print("matplotlib:", matplotlib.__version__, "| sklearn:", sklearn.__version__)

# 1) 빗썸 API 연결 테스트 (USDT/KRW 1시간봉 3개)
r = requests.get(
    "https://api.bithumb.com/v1/candles/minutes/60",
    params={"market": "KRW-USDT", "count": 3},
    timeout=10,
)
print("\n[빗썸] 상태코드:", r.status_code)
print(r.json()[0])

# 2) 환율(USD/KRW) 연결 테스트
fx = yf.download("KRW=X", period="5d", interval="1h", progress=False)
print("\n[환율] 받은 행 수:", len(fx))
print(fx.tail(2))