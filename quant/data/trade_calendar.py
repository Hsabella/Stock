"""A 股交易所交易日历（含未来日期）：晨报 today、K 线陈旧判定、回补冷静期按它数天。

为什么不用 pd.offsets.BDay：BDay 只跳周末不认节假日。2026-10-08 实测 09-30 收盘
+1 BDay 推出 10-01（国庆），晨报标签与三份流水都记成了假日；陈旧判定与冷静期
也会把长假里的 5 个工作日算成交易日。

为什么不用 qlib calendars/day.txt：它只有历史日期，推不出"下一个交易日"。

数据源新浪 tool_trade_date_hist_sina（覆盖到当年 12-31），缓存到 cache/trade_calendar.csv。
要问的日期超出缓存末日才重拉，每个进程最多重拉一次——跨年时新浪可能还没发布
下一年日历，不设上限的话晨报里 130+ 只票每只都会打一次网络。
拿不到日历时退回工作日近似（即旧口径），degraded() 供调用方提示。
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

import pandas as pd

CACHE_FILE = Path(__file__).resolve().parents[2] / "cache" / "trade_calendar.csv"

_refetched = False
_degraded = False


def _fetch() -> pd.DatetimeIndex:
    import akshare as ak

    df = ak.tool_trade_date_hist_sina()
    return pd.DatetimeIndex(pd.to_datetime(df["trade_date"])).sort_values()


def _refresh() -> pd.DatetimeIndex | None:
    global _degraded
    try:
        days = _fetch()
    except Exception as e:  # 日历挂了不能拖垮晨报，退回工作日近似
        _degraded = True
        print(f"[trade_calendar] 交易日历获取失败，退回工作日近似: {type(e).__name__}: {e}",
              file=sys.stderr)
        return None
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(days.strftime("%Y-%m-%d"), name="trade_date").to_csv(CACHE_FILE, index=False)
    return days


@functools.lru_cache(maxsize=1)
def trading_days() -> pd.DatetimeIndex | None:
    """全部交易日（升序）；磁盘缓存优先，没有就拉。拿不到返回 None。"""
    if CACHE_FILE.exists():
        try:
            return pd.DatetimeIndex(pd.read_csv(CACHE_FILE)["trade_date"]).sort_values()
        except Exception as e:  # 缓存损坏 → 重拉，别静默用半份日历
            print(f"[trade_calendar] 缓存不可读，重拉: {type(e).__name__}: {e}", file=sys.stderr)
    return _refresh()


def _covering(day: pd.Timestamp) -> pd.DatetimeIndex | None:
    """覆盖到 day 的日历；覆盖不到时（每进程一次）重拉，仍不行返回 None。"""
    global _refetched, _degraded
    cal = trading_days()
    if cal is not None and cal[-1] >= day:
        return cal
    if not _refetched:
        _refetched = True
        _refresh()
        trading_days.cache_clear()
        cal = trading_days()
        if cal is not None and cal[-1] >= day:
            return cal
    _degraded = True
    return None


def next_trading_day(after: pd.Timestamp) -> pd.Timestamp:
    """严格晚于 after 的第一个交易日。"""
    after = pd.Timestamp(after).normalize()
    cal = _covering(after + pd.Timedelta(days=1))
    if cal is None:
        return after + pd.offsets.BDay(1)
    return cal[cal > after][0]


def trading_days_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """(start, end] 区间内的交易日数；start 当天不算。"""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if end <= start:
        return 0
    cal = _covering(end)
    if cal is None:
        return len(pd.bdate_range(start, end)) - 1
    return int(((cal > start) & (cal <= end)).sum())


def degraded() -> bool:
    """本进程是否有过回退到工作日近似（晨报据此加一行提示）。"""
    return _degraded
