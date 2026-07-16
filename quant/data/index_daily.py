"""指数日线落仓：新浪接口 → warehouse index_daily（v2 Step3 regime 特征的原料）。

qlib bin 包只带基准 SH000906，regime 特征还需要沪深300/中证1000 算大小盘风向，
统一从新浪拉全历史落仓（与 daily_brief 同源，稳定）。每周跑一次保持最新。

用法: python -m quant.data.index_daily
"""
from __future__ import annotations

import sys
import time

import pandas as pd

from quant.data import warehouse

DATASET = "index_daily"
INDEXES = ["sh000300", "sh000852", "sh000906", "sh000688"]  # 沪深300 / 中证1000 / 中证800 / 科创50
THROTTLE_SEC = 0.4


def build() -> int:
    import socket

    import akshare as ak

    socket.setdefaulttimeout(60)
    # 盘中跑会拿到当日未收盘的半根 bar，以收盘价身份入仓直到下次更新；收盘前丢弃当日行
    cutoff = pd.Timestamp.today().normalize()
    if pd.Timestamp.now() >= cutoff + pd.Timedelta(hours=15, minutes=30):
        cutoff += pd.Timedelta(days=1)

    frames = []
    for sym in INDEXES:
        df = ak.stock_zh_index_daily(symbol=sym)
        out = pd.DataFrame({
            "datetime": pd.to_datetime(df["date"]),
            "close": df["close"].astype(float),
            "instrument": sym.upper(),
        })
        frames.append(out[out["datetime"] < cutoff])
        time.sleep(THROTTLE_SEC)
    warehouse.append(DATASET, pd.concat(frames, ignore_index=True))
    print(f"[index_daily] {warehouse.coverage(DATASET)}")
    return 0


def load_closes() -> pd.DataFrame:
    """宽表：index=datetime, columns=SH000300/SH000852/SH000906/SH000688 收盘价。"""
    df = warehouse.load(DATASET)
    if df.empty:
        return pd.DataFrame()
    return df["close"].unstack(level="instrument").sort_index()


if __name__ == "__main__":
    sys.exit(build())
