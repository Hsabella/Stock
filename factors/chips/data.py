"""筹码维度数据：复用共享 K 线."""
from __future__ import annotations
from factors._kline import get_kline


def get_kline_for_chips(symbol: str, days: int = 250):
    return get_kline(symbol, days=days)
