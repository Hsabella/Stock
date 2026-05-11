"""大盘 / 行业维度数据."""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd

try:
    import akshare as ak
except ImportError:
    ak = None

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "regime"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _is_fresh(p: Path, hours: int = 6) -> bool:
    if not p.exists():
        return False
    return (time.time() - p.stat().st_mtime) / 3600 < hours


def get_index_kline(symbol: str = "sh000001", days: int = 120,
                    force_refresh: bool = False) -> pd.DataFrame:
    """指数日 K（默认上证）. force_refresh=True 绕过缓存."""
    p = CACHE_DIR / f"{symbol}_daily.csv"
    if not force_refresh and _is_fresh(p):
        try:
            return pd.read_csv(p, parse_dates=["date"])
        except Exception:
            pass
    if ak is None:
        return pd.DataFrame()
    try:
        df = ak.stock_zh_index_daily(symbol=symbol)
        if df is None or df.empty:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").tail(days * 2).reset_index(drop=True)
        df.to_csv(p, index=False)
        return df
    except Exception as e:
        print(f"  [index_kline {symbol}] error: {e}")
        return pd.DataFrame()


def get_csi300_kline(days: int = 120, force_refresh: bool = False) -> pd.DataFrame:
    """沪深 300 指数日 K，作为相对强度基准."""
    return get_index_kline("sh000300", days=days, force_refresh=force_refresh)


def get_sector_fund_flow_rank() -> pd.DataFrame:
    """行业资金流当日排名."""
    p = CACHE_DIR / "sector_fund_flow_rank.csv"
    if _is_fresh(p, hours=6):
        try:
            return pd.read_csv(p)
        except Exception:
            pass
    if ak is None:
        return pd.DataFrame()
    try:
        df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流")
        if df is None or df.empty:
            return pd.DataFrame()
        df.to_csv(p, index=False)
        return df
    except Exception as e:
        print(f"  [sector_fund_flow_rank] error: {e}")
        return pd.DataFrame()


def get_north_total_inflow() -> float | None:
    """当日北向资金净流入（万元）.

    注: 2024-08 起监管停止披露当日总额，仅保留个股持股变化。
    取最近 5 个有效日的均值作为环境代理；全 NaN 时返回 None。
    """
    p = CACHE_DIR / "north_total_recent5.csv"
    if _is_fresh(p, hours=12):
        try:
            return float(pd.read_csv(p)["value"].iloc[-1])
        except Exception:
            pass
    if ak is None:
        return None
    try:
        df = ak.stock_hsgt_hist_em(symbol="北向资金")
        if df is None or df.empty:
            return None
        s = pd.to_numeric(df["当日成交净买额"], errors="coerce").dropna()
        if s.empty:
            return None
        v = float(s.tail(5).mean())
        pd.DataFrame({"value": [v]}).to_csv(p, index=False)
        return v
    except Exception as e:
        print(f"  [north_total] error: {e}")
        return None
