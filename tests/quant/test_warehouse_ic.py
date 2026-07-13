"""warehouse append 去重 + ic_report 统计原语的纯逻辑单测。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant.research.ic_report import quantile_returns, rank_ic, weekly_anchors  # noqa: E402


def test_weekly_anchors_picks_first_trading_day():
    # 2024-01-01(周一) 缺席 → 该周锚点应是 01-02(周二)；下周正常取周一 01-08
    cal = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08", "2024-01-10"])
    anchors = weekly_anchors(cal)
    assert list(anchors) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-08")]


def test_rank_ic_perfect_and_inverse():
    n = 400  # 超过 MIN_STOCKS_PER_ANCHOR
    x = pd.Series(np.arange(n, dtype=float))
    assert abs(rank_ic(x, x) - 1.0) < 1e-9
    assert abs(rank_ic(x, -x) + 1.0) < 1e-9


def test_rank_ic_too_few_returns_nan():
    x = pd.Series(np.arange(10, dtype=float))
    assert np.isnan(rank_ic(x, x))


def test_quantile_returns_monotonic_when_factor_predicts():
    n = 500
    f = pd.Series(np.arange(n, dtype=float))
    r = f / n + 0.001
    q = quantile_returns(f, r)
    assert q is not None and len(q) == 5
    assert q == sorted(q)  # 完美预测时五分位收益单调递增


def test_warehouse_append_dedupes(tmp_path, monkeypatch):
    from quant.data import warehouse

    monkeypatch.setattr(warehouse, "warehouse_dir", lambda: tmp_path)
    df1 = pd.DataFrame({
        "instrument": ["SH600000", "SH600000"],
        "datetime": ["2024-01-02", "2024-01-03"],
        "val": [1.0, 2.0],
    })
    assert warehouse.append("t", df1) == 2
    # 同 key 覆盖 + 新增一行
    df2 = pd.DataFrame({
        "instrument": ["SH600000", "SH600000"],
        "datetime": ["2024-01-03", "2024-01-04"],
        "val": [99.0, 3.0],
    })
    assert warehouse.append("t", df2) == 3
    out = warehouse.load("t")
    assert out.loc[("SH600000", pd.Timestamp("2024-01-03")), "val"] == 99.0

    # 原子替换语义：第二次 append 后旧版本转存 .bak（累积数据写坏可回退上一版）
    bak = tmp_path / "t.parquet.bak"
    assert bak.exists()
    assert len(pd.read_parquet(bak)) == 2
    assert not (tmp_path / "t.parquet.tmp").exists()
