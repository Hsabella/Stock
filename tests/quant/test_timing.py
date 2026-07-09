"""择时层时间对齐与防抖单测——锁死"严禁用 T 日及以后信息"。"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant.backtest.overlay import apply_overlay  # noqa: E402
from quant.backtest.timing import debounce, raw_risk_off  # noqa: E402


def _calendar(n=300, start="2023-01-02"):
    return pd.bdate_range(start, periods=n)


def test_ma_signal_uses_t_minus_1_close():
    """大盘 T 日跌破 MA 时，risk_off 最早只能在 T+1 生效（用的是 T-1 收盘）。"""
    cal = _calendar()
    hs300 = pd.Series(100.0, index=cal)
    k = 250
    hs300.iloc[k:] = 50.0  # 第 k 天暴跌跌破 MA200
    spx = pd.Series(100.0, index=cal)  # 美股恒定，不触发
    df = raw_risk_off(hs300, spx, ma_days=200, spx_threshold=-0.015, calendar=cal)
    assert not df["raw_risk_off"].iloc[k]        # T 日当天不能"未卜先知"
    assert df["raw_risk_off"].iloc[k + 1]        # T+1 才看到 T 收盘


def test_spx_overnight_uses_bar_dated_before_t():
    """美股日期为 T-1 的 bar（北京 T 日凌晨收盘）可用于 T 日；T 日 bar 不可用。"""
    cal = pd.DatetimeIndex(["2024-01-08", "2024-01-09", "2024-01-10"])
    hs300 = pd.Series(100.0, index=pd.bdate_range("2023-01-02", periods=400)).reindex(cal).fillna(100.0)
    # 构造完整点位序列：01-08 暴跌 5%（美东日期），01-09 收平
    spx = pd.Series([100.0, 100.0, 95.0, 95.0],
                    index=pd.DatetimeIndex(["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"]))
    df = raw_risk_off(hs300, spx, ma_days=5, spx_threshold=-0.015, calendar=cal)
    assert df.loc["2024-01-08", "spx_overnight"] == 0.0     # 只见 01-05 的平盘
    assert df.loc["2024-01-09", "spx_overnight"] < -0.04    # 01-08 暴跌北京 01-09 晨可见
    assert df.loc["2024-01-09", "raw_risk_off"]


def test_debounce_requires_consecutive_confirmation():
    raw = pd.Series([False, True, False, True, True, True, False, False])
    out = debounce(raw, confirm_days=2)
    # 单日毛刺（idx1）不切换；idx3-4 连续两日才切 risk_off；idx6-7 连续两日切回
    assert out.tolist() == [False, False, False, False, True, True, True, False]


def test_overlay_scales_and_charges_switch():
    idx = pd.bdate_range("2024-01-01", periods=4)
    net = pd.Series([0.01, 0.01, 0.01, 0.01], index=idx)
    exposure = pd.Series([1.0, 0.5, 0.5, 1.0], index=idx)
    out = apply_overlay(net, exposure, overlay_cost=0.001)
    assert abs(out.iloc[0] - 0.01) < 1e-12
    assert abs(out.iloc[1] - (0.005 - 0.0005)) < 1e-12  # 减半 + 切换成本
    assert abs(out.iloc[2] - 0.005) < 1e-12
    assert abs(out.iloc[3] - (0.01 - 0.0005)) < 1e-12
