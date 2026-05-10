"""流动性维度数据：复用共享 K 线即可."""
from __future__ import annotations
import pandas as pd
from factors._kline import get_kline


def get_kline_for_liquidity(symbol: str, days: int = 90) -> pd.DataFrame:
    """近 days 日 K 线，用于计算量比 / 换手率拐点."""
    return get_kline(symbol, days=days)
