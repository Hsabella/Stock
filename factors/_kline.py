"""共享 K 线获取 + 缓存（新浪 stock_zh_a_daily，列最全）.

供 liquidity / technical / market_action 等多个维度复用。
- 新浪源直接含 volume / amount / outstanding_share / turnover（小数，已转为 %）
- 缓存目录: cache/kline/{symbol}_daily_qfq.csv
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd

try:
    import akshare as ak
except ImportError:
    ak = None

CACHE_DIR = Path(__file__).resolve().parents[1] / "cache" / "kline"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_HOURS = 6


def _is_fresh(p: Path, ttl_hours: int = CACHE_TTL_HOURS) -> bool:
    if not p.exists():
        return False
    return (time.time() - p.stat().st_mtime) / 3600 < ttl_hours


ETF_PREFIXES = ("51", "56", "58", "159")  # 沪 51x/56x/58x + 深 159x


def is_etf(symbol: str) -> bool:
    """场内 ETF 代码段判定（quant/live/portfolio.py 有同款副本，venv 隔离不互相 import）。"""
    return str(symbol).startswith(ETF_PREFIXES)


def _sina_symbol(symbol: str) -> str:
    return f"sh{symbol}" if symbol.startswith("6") else f"sz{symbol}"


def _fetch_etf(symbol: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    """东财 ETF 日线（复权）→ 与股票缓存同列名。新浪股票接口对基金代码直接报错，
    新浪 fund_etf_hist_sina 不复权（份额拆分会污染距高/MA 计算），都不能用。"""
    df = ak.fund_etf_hist_em(symbol=symbol, period="daily",
                             start_date=start, end_date=end, adjust=adjust)
    return df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close",
                              "最高": "high", "最低": "low", "成交量": "volume",
                              "成交额": "amount", "换手率": "turnover"})


def get_kline(symbol: str, days: int = 250, adjust: str = "qfq",
              force_refresh: bool = False) -> pd.DataFrame:
    """日线 K 线（前复权）, 列: date / open / high / low / close / volume / amount /
    outstanding_share / turnover (单位: %).

    force_refresh=True 时绕过缓存重拉 (forward 校验当天用).
    """
    p = CACHE_DIR / f"{symbol}_daily_{adjust}.csv"
    if not force_refresh and _is_fresh(p):
        try:
            df = pd.read_csv(p, parse_dates=["date"])
            return df.tail(days).reset_index(drop=True)
        except Exception:
            pass

    if ak is None:
        return pd.DataFrame()

    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=int(days * 1.6))
    s = start.strftime("%Y%m%d")
    e = end.strftime("%Y%m%d")
    try:
        if is_etf(symbol):
            df = _fetch_etf(symbol, s, e, adjust)
        else:
            df = ak.stock_zh_a_daily(
                symbol=_sina_symbol(symbol), start_date=s, end_date=e, adjust=adjust
            )
    except Exception as ex:
        print(f"  [kline] {symbol} error: {ex}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    for c in ("open", "high", "low", "close", "volume", "amount", "outstanding_share", "turnover"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "turnover" in df.columns and not is_etf(symbol):
        df["turnover"] = df["turnover"] * 100.0  # 新浪是小数: 0.0038 → 0.38 (%)；东财已是 %
    df.to_csv(p, index=False)
    return df.tail(days).reset_index(drop=True)
