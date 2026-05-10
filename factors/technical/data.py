"""技术维度数据：复用共享 K 线."""
from __future__ import annotations
import pandas as pd
from factors._kline import get_kline


def get_kline_for_tech(symbol: str, days: int = 200) -> pd.DataFrame:
    return get_kline(symbol, days=days)
