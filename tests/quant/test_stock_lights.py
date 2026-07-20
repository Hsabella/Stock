"""stock_lights 核心逻辑：事件化/信号/涨跌停/仿真/晨报数据层（合成序列，不碰 qlib 与网络）。"""
import numpy as np
import pandas as pd
import pytest

from quant.research import stock_lights as sl


def _mk(prices):
    """收盘序列 → OHLCV df（开盘=昨收，涨跌停不触发，无停牌）。"""
    idx = pd.bdate_range("2024-01-02", periods=len(prices))
    c = pd.Series(prices, index=idx, dtype=float)
    return pd.DataFrame({
        "open": c.shift(1).fillna(c.iloc[0]), "high": c * 1.01, "low": c * 0.99,
        "close": c, "volume": 1e6,
    }, index=idx)


def _crash_path(flat=130, down=30, step=0.98, rebound=0):
    prices = [100.0] * flat
    for _ in range(down):
        prices.append(prices[-1] * step)
    for _ in range(rebound):
        prices.append(prices[-1] * 1.03)
    return prices


def test_fresh_fires_only_on_first_crossing():
    cond = pd.Series([False, True, True, False, True, np.nan, True])
    out = sl._fresh(cond)
    assert out.tolist() == [False, True, False, False, True, False, True]


def test_fresh2_fires_on_second_consecutive_day_once():
    cond = pd.Series([False, True, True, True, False, True, True])
    out = sl._fresh2(cond)
    assert out.tolist() == [False, False, True, False, False, False, True]


def test_signal_frame_l1_and_s0_eventized_on_crash():
    ind = sl.compute_indicators(_mk(_crash_path()))
    sigs = sl.signal_frame(ind, thr=-0.35)
    assert sigs["S0基准"].sum() == 1        # dd 只向下穿越 -35% 一次
    assert sigs["L1接刀"].sum() == 1        # setup 内 RSI≤30 首穿一次
    crash_zone = ind["dd120"] <= -0.35
    assert sigs["L1接刀"][crash_zone].any()


def test_signal_frame_r1_fires_after_rebound_above_ma20():
    ind = sl.compute_indicators(_mk(_crash_path(rebound=15)))
    sigs = sl.signal_frame(ind, thr=-0.35)
    assert sigs["R1站回MA20"].sum() >= 1
    assert sigs["R1站回MA20"].to_numpy().nonzero()[0].min() > 130  # 只出现在反弹段


def test_limit_schedule_by_board_and_date():
    idx = pd.DatetimeIndex(["2020-08-21", "2020-08-24", "2024-01-02"])
    assert sl.limit_schedule("SH688001", idx).tolist() == [0.195] * 3
    assert sl.limit_schedule("SZ300724", idx).tolist() == [0.095, 0.195, 0.195]
    assert sl.limit_schedule("SH600026", idx).tolist() == [0.095] * 3
    assert sl.limit_schedule("BJ430017", idx).tolist() == [0.295] * 3


def test_simulate_left_stop_exits_after_plunge():
    prices = [100.0] * 140 + [60.0 * 0.97 ** i for i in range(3)]  # 已在低位再阴跌
    prices = _crash_path(flat=140, down=0)
    prices += [70.0, 70.0, 68.0, 50.0, 50.0, 50.0]  # 入场后暴跌破 3×ATR 止损
    ind = sl.compute_indicators(_mk(prices))
    n = len(ind)
    sig = np.zeros(n, dtype=bool)
    sig[n - 5] = True                                # 68 收盘日出信号
    trades = sl.simulate(ind, sig, np.ones(n, dtype=bool), "left",
                         np.full(n, 0.095))
    assert len(trades) == 1
    tr = trades[0]
    assert tr["ret"] < -0.2 and not tr["censored"]   # 50/68 - 成本
    assert tr["hold"] >= 1


def test_simulate_censored_at_data_end():
    prices = _crash_path(flat=140, down=0) + [70.0] * 4
    ind = sl.compute_indicators(_mk(prices))
    n = len(ind)
    sig = np.zeros(n, dtype=bool)
    sig[n - 3] = True
    trades = sl.simulate(ind, sig, np.ones(n, dtype=bool), "left",
                         np.full(n, 0.095))
    assert len(trades) == 1 and trades[0]["censored"]


def test_lamp_rows_uses_kline_and_flags_fire(monkeypatch, tmp_path):
    wl = tmp_path / "watchlist.yaml"
    wl.write_text(
        "watchlist:\n"
        "  - { symbol: \"002074\", name: \"国轩高科\", state: \"WATCHING\" }\n"
        "  - { symbol: \"600519\", name: \"贵州茅台\", state: \"WATCHING\" }\n"
        "  - { symbol: \"000001\", name: \"无K线票\", state: \"WATCHING\" }\n",
        encoding="utf-8")
    monkeypatch.setattr(sl, "REPO", tmp_path)

    crash = _mk(_crash_path()).reset_index(names="date")       # 深跌票（L1 会触发）
    flat = _mk([100.0] * 200).reset_index(names="date")        # 未到 setup

    from quant.live import portfolio

    def fake_load_kline(sym, today):
        if sym == "002074":
            return crash, False
        if sym == "600519":
            return flat, False
        return pd.DataFrame(), True

    monkeypatch.setattr(portfolio, "load_kline", fake_load_kline)
    out = sl.lamp_rows(today=pd.Timestamp("2024-08-01"))
    by_sym = {r["symbol"]: r for r in out["rows"]}
    assert by_sym["600519"]["lamp"] == "none"
    assert by_sym["000001"]["error"] == "K线不足"
    assert by_sym["002074"]["lamp"] in ("fire", "watch")       # 深跌必在灯上
    assert by_sym["002074"]["dd120"] < -0.35


def test_lamp_rows_skips_etf(monkeypatch, tmp_path):
    """ETF 不打灯（参数只在个股上回测过）——即使 K 线是深跌形态也不进灯列表。"""
    wl = tmp_path / "watchlist.yaml"
    wl.write_text(
        "watchlist:\n"
        "  - { symbol: \"515050\", name: \"5G通信ETF\", state: \"WATCHING\" }\n"
        "  - { symbol: \"002074\", name: \"国轩高科\", state: \"WATCHING\" }\n",
        encoding="utf-8")
    monkeypatch.setattr(sl, "REPO", tmp_path)
    crash = _mk(_crash_path()).reset_index(names="date")
    from quant.live import portfolio
    monkeypatch.setattr(portfolio, "load_kline", lambda sym, today: (crash, False))

    out = sl.lamp_rows(today=pd.Timestamp("2024-08-01"))
    syms = [r["symbol"] for r in out["rows"]]
    assert "515050" not in syms and syms == ["002074"]


def test_event_rows_survives_kline_shorter_than_horizon():
    """K线短于最长 horizon(60) 时主切片负索引会回绕——须有防护不崩。"""
    ind = sl.compute_indicators(_mk([100.0] * 50))
    sigs = sl.signal_frame(ind, thr=-0.35)
    gmap = {"hs300": np.zeros(50, dtype=bool), "csi1000": np.zeros(50, dtype=bool)}
    rows = sl.event_rows("SH600000", ind, sigs, -0.35, gmap,
                         np.ones(50, dtype=bool), np.zeros(50, dtype=bool))
    assert rows == []  # 平盘无 setup，无事件；关键是不抛 ValueError


def test_lamp_rows_fired_carries_signal_date(monkeypatch, tmp_path):
    wl = tmp_path / "watchlist.yaml"
    wl.write_text("watchlist:\n  - { symbol: \"002074\", name: \"国轩高科\", state: \"WATCHING\" }\n",
                  encoding="utf-8")
    monkeypatch.setattr(sl, "REPO", tmp_path)
    crash = _mk(_crash_path()).reset_index(names="date")
    from quant.live import portfolio
    monkeypatch.setattr(portfolio, "load_kline", lambda sym, today: (crash, False))

    out = sl.lamp_rows(today=pd.Timestamp("2024-08-01"))
    row = out["rows"][0]
    if row["lamp"] == "fire":  # 深跌路径 L1 在窗口内触发时必须带信号日
        for s in row["signals"]:
            assert set(s) == {"signal", "signal_date"}
            assert s["signal"] in ("L1接刀", "L6超卖修复")
            pd.Timestamp(s["signal_date"])  # 可解析日期
