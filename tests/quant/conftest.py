import pandas as pd
import pytest

from quant.data import trade_calendar


@pytest.fixture(autouse=True)
def weekday_trade_calendar(monkeypatch, tmp_path):
    """测试默认用纯工作日日历：不联网、不读写真实 cache/，且与旧 BDay 口径一致。
    要测节假日的用例自己再 monkeypatch _fetch。"""
    monkeypatch.setattr(trade_calendar, "CACHE_FILE", tmp_path / "trade_calendar.csv")
    monkeypatch.setattr(trade_calendar, "_fetch",
                        lambda: pd.bdate_range("2000-01-03", "2030-12-31"))
    monkeypatch.setattr(trade_calendar, "_refetched", False)
    monkeypatch.setattr(trade_calendar, "_degraded", False)
    trade_calendar.trading_days.cache_clear()
    yield
    trade_calendar.trading_days.cache_clear()
