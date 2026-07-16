"""portfolio 数据层：止损线手算对拍 / atr_k 解析 / 陈旧判定 / 引擎信号解析。"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant.live import portfolio  # noqa: E402
from quant.live.portfolio import (  # noqa: E402
    _atr, load_kline, load_portfolio_config, stop_lines,
)


def _kline(n=40, close_start=100.0, step=-1.0, spread=2.0):
    """合成日K：每日 close 递变 step，high/low 对称 spread/2。"""
    dates = pd.bdate_range("2026-05-01", periods=n)
    close = pd.Series([close_start + step * i for i in range(n)])
    return pd.DataFrame({
        "date": dates, "close": close,
        "high": close + spread / 2, "low": close - spread / 2,
        "open": close,
    })


def test_stop_lines_hand_calculated():
    """恒定 TR=spread 时 ATR14=spread；追踪线=20日最高收盘-k×spread。"""
    k = _kline(n=40, close_start=100.0, step=-1.0, spread=2.0)
    # 手算: TR = max(high-low, |high-pc|, |low-pc|) = max(2, |−1+1|+... ) = 2? 逐日:
    # high-low=2; high-pc = (c+1)-(c_prev)= c_prev-1+1-c_prev = 0? 实际 close 递减 1:
    # high_t = c_t+1, pc = c_t+1 → high-pc = 0; low-pc = c_t-1-(c_t+1) = -2 → abs=2 → TR=2
    atr = _atr(k["high"], k["low"], k["close"], n=14)
    assert abs(atr.iloc[-1] - 2.0) < 1e-9
    out = stop_lines(k, entry_price=90.0, hard_stop_pct=0.08, atr_k=2.0)
    # 近20日最高收盘 = 40根中最后20根的最高 = close[20] = 100-20 = 80
    assert abs(out["atr_stop"] - (80.0 - 2.0 * 2.0)) < 1e-9
    assert abs(out["hard_stop"] - 90.0 * 0.92) < 1e-9
    # stop_line 取两者较严（较高）者 = 82.8; 现价 61 < 82.8 → 已破
    assert abs(out["stop_line"] - 82.8) < 1e-9
    assert out["broken"]
    assert out["dist_pct"] < 0


def test_stop_lines_not_broken_when_price_above():
    k = _kline(n=40, close_start=100.0, step=+0.5, spread=1.0)  # 上涨趋势
    out = stop_lines(k, entry_price=100.0, hard_stop_pct=0.08, atr_k=2.0)
    assert not out["broken"]
    assert out["dist_pct"] > 0


def test_atr_k_parsed_from_strategy(tmp_path):
    p = tmp_path / "w.yaml"
    p.write_text(
        "default_stop_strategy: atr_3\nhard_stop_pct: 0.05\n"
        "watchlist:\n"
        "  - { symbol: '600000', name: x, state: HELD, position: 100, entry_price: 10.0 }\n"
        "  - { symbol: '600001', name: y, state: WATCHING }\n")
    cfg = load_portfolio_config(p)
    assert cfg["atr_k"] == 3.0
    assert cfg["hard_stop_pct"] == 0.05
    assert [h["symbol"] for h in cfg["held"]] == ["600000"]


def test_load_kline_stale_detection(tmp_path, monkeypatch):
    """缓存末日落后 >3 个交易日 → stale=True（akshare 失败时仍返回旧数据）。"""
    kdir = tmp_path / "kline"
    kdir.mkdir()
    _kline(n=30).assign(date=pd.bdate_range("2026-06-01", periods=30)).to_csv(
        kdir / "600000_daily_qfq.csv", index=False)
    monkeypatch.setattr(portfolio, "KLINE_DIR", kdir)
    monkeypatch.setattr(portfolio, "STALE_TRADING_DAYS", 3)

    import akshare as ak
    monkeypatch.setattr(ak, "stock_zh_a_daily",
                        lambda **kw: (_ for _ in ()).throw(RuntimeError("网络挂了")),
                        raising=False)
    last = pd.Timestamp(pd.bdate_range("2026-06-01", periods=30)[-1])
    df, stale = load_kline("600000", today=last + pd.offsets.BDay(1))
    assert not stale and not df.empty
    df, stale = load_kline("600000", today=last + pd.offsets.BDay(10))
    assert stale and not df.empty  # 降级失败仍给旧数据, 由调用方标 ⚠️


def test_latest_engine_signals_parses_risks(tmp_path, monkeypatch):
    dec = tmp_path / "decisions"
    dec.mkdir()
    (dec / "partial_20260714.csv").write_text("symbol,decision,dec_risks\n600000,HOLD,[]\n")
    (dec / "partial_20260715.csv").write_text(
        'symbol,decision,dec_risks\n600000,DROP,"[\'PE_TTM=296（估值偏高）\', \'板块弱势\', \'第三条\']"\n')
    monkeypatch.setattr(portfolio, "DEC_DIR", dec)
    signals, date = portfolio.latest_engine_signals()
    assert date == "20260715"  # 取最新一期
    assert signals["600000"]["decision"] == "DROP"
    assert len(signals["600000"]["risks"]) == 2  # 只取前两条
