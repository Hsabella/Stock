"""交易日历：晨报 today / K 线陈旧判定 / 回补冷静期都要按交易所日历数天。

2026-10-08 实测：旧口径 "数据末日 + 1 BDay" 不认节假日，09-30 收盘后推出 10-01（国庆），
晨报标签与三份流水都记成了假日。
"""
import pandas as pd
import pytest

from quant.data import trade_calendar as tc

# 2026 国庆：09-30 之后下一个交易日是 10-08
FAKE_DAYS = ["2026-09-28", "2026-09-29", "2026-09-30",
             "2026-10-08", "2026-10-09", "2026-10-12", "2026-10-13"]


@pytest.fixture
def cal(monkeypatch, tmp_path):
    calls = []

    def fake_fetch():
        calls.append(1)
        return pd.DatetimeIndex(FAKE_DAYS)

    monkeypatch.setattr(tc, "CACHE_FILE", tmp_path / "trade_calendar.csv")
    monkeypatch.setattr(tc, "_fetch", fake_fetch)
    monkeypatch.setattr(tc, "_refetched", False)
    monkeypatch.setattr(tc, "_degraded", False)
    tc.trading_days.cache_clear()
    yield calls
    tc.trading_days.cache_clear()


def test_next_trading_day_skips_national_day(cal):
    assert tc.next_trading_day(pd.Timestamp("2026-09-30")) == pd.Timestamp("2026-10-08")


def test_next_trading_day_skips_weekend(cal):
    assert tc.next_trading_day(pd.Timestamp("2026-10-09")) == pd.Timestamp("2026-10-12")


def test_next_trading_day_from_holiday(cal):
    """周末/假期里手动跑：落到下一个交易日。"""
    assert tc.next_trading_day(pd.Timestamp("2026-10-03")) == pd.Timestamp("2026-10-08")


def test_count_across_holiday_is_one(cal):
    """(09-30, 10-08] 只有 1 个交易日——节后首日缓存停在节前收盘是"数据齐"。"""
    assert tc.trading_days_between(pd.Timestamp("2026-09-30"), pd.Timestamp("2026-10-08")) == 1


def test_count_same_day_is_zero(cal):
    assert tc.trading_days_between(pd.Timestamp("2026-10-08"), pd.Timestamp("2026-10-08")) == 0


def test_cache_reused_without_refetch(cal):
    tc.next_trading_day(pd.Timestamp("2026-09-30"))
    tc.trading_days.cache_clear()  # 模拟新进程：内存缓存没了，磁盘缓存还在
    tc.next_trading_day(pd.Timestamp("2026-09-30"))
    assert len(cal) == 1


def test_stale_cache_refetched(cal, monkeypatch):
    """磁盘缓存覆盖不到要问的日期 → 重拉（跨年时新浪才发布下一年日历）。"""
    tc.next_trading_day(pd.Timestamp("2026-09-30"))
    tc.trading_days.cache_clear()
    tc.next_trading_day(pd.Timestamp("2026-10-13"))  # 末日本身，往后无覆盖
    assert len(cal) == 2


def test_fetch_failure_falls_back_to_weekdays(monkeypatch, tmp_path):
    """日历拿不到不能拖垮晨报：退回工作日近似（即旧口径），并让调用方知道。"""
    def boom():
        raise ConnectionError("sina down")

    monkeypatch.setattr(tc, "CACHE_FILE", tmp_path / "missing.csv")
    monkeypatch.setattr(tc, "_fetch", boom)
    monkeypatch.setattr(tc, "_refetched", False)
    monkeypatch.setattr(tc, "_degraded", False)
    tc.trading_days.cache_clear()
    try:
        assert tc.trading_days() is None
        assert tc.next_trading_day(pd.Timestamp("2026-09-30")) == pd.Timestamp("2026-10-01")
        assert tc.trading_days_between(pd.Timestamp("2026-09-30"), pd.Timestamp("2026-10-08")) == 6
        assert tc.degraded()
    finally:
        tc.trading_days.cache_clear()


def test_beyond_coverage_falls_back(cal):
    """问的日期超出日历末日：用工作日近似续上，而不是报错。"""
    assert tc.next_trading_day(pd.Timestamp("2026-10-13")) == pd.Timestamp("2026-10-14")
