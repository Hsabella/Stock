"""结构灯：raw_ma_off 时间对齐 + 配置 schema + 回测指标单测。"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant.backtest.timing import debounce, raw_ma_off  # noqa: E402
from quant.config import load_config  # noqa: E402
from quant.research.timing_lights import episode_false_rate, evaluate  # noqa: E402


def test_raw_ma_off_uses_t_minus_1_close():
    """T 日跌破 MA 时，raw 信号最早 T+1 生效（同主灯口径，无未卜先知）。"""
    cal = pd.bdate_range("2023-01-02", periods=300)
    close = pd.Series(100.0, index=cal)
    k = 250
    close.iloc[k:] = 50.0
    raw = raw_ma_off(close, ma_days=200, calendar=cal)
    assert not raw.iloc[k]
    assert raw.iloc[k + 1]


def test_raw_ma_off_without_spx_never_triggers_spx_leg():
    """纯 MA 腿：价格恒在 MA 上方时全程 False（空标普序列不产生触发）。"""
    cal = pd.bdate_range("2023-01-02", periods=260)
    close = pd.Series(range(100, 360), index=cal, dtype=float)  # 单调上涨
    raw = raw_ma_off(close, ma_days=200, calendar=cal)
    assert not raw.fillna(False).any()


def test_debounce_on_ma_light_confirm3():
    raw = pd.Series([False, True, True, False, True, True, True, False, False, False])
    out = debounce(raw, confirm_days=3)
    # idx1-2 只连续 2 日不切换；idx4-6 连续 3 日才切 off；idx7-9 连续 3 日切回
    assert out.tolist() == [False, False, False, False, False, False, True, True, True, False]


def test_timing_lights_config_schema():
    lights = load_config("backtest")["timing_lights"]
    assert len(lights) == 2
    for light in lights:
        assert {"symbol", "name", "ma_days", "confirm_days", "mode"} <= set(light)
        assert light["mode"] in ("advice", "display_only")
        if light["mode"] == "advice":
            assert light.get("advice_off")


def test_evaluate_protects_in_crash():
    """构造先涨后崩的序列：策略 MDD 必须小于买入持有 MDD。"""
    cal = pd.bdate_range("2020-01-01", periods=500)
    up = list(range(100, 400))          # 300 日上涨
    down = list(range(400, 200, -1))    # 200 日阴跌
    close = pd.Series((up + down)[:500], index=cal, dtype=float)
    row = evaluate(close, ma_days=120, confirm_days=2)
    assert row["mdd_strat"] > row["mdd_bh"]  # 回撤更浅（负数更大）
    assert row["mdd_compress"] > 0.25
    assert row["switches_py"] < 6


def test_episode_false_rate():
    idx = pd.bdate_range("2024-01-01", periods=8)
    close = pd.Series([100, 90, 95, 101, 100, 80, 85, 84], index=idx, dtype=float)
    off = pd.Series([False, True, True, True, False, True, True, True], index=idx)
    # 区间1: idx1→idx4 期间 90→100 涨(假信号); 区间2: idx5→idx7 80→84 涨(假信号)
    assert episode_false_rate(off, close) == 1.0
