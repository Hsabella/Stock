# tests/test_metrics.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dashboard import data_loader as dl, metrics as mx


def test_group_counts():
    df = dl.load_partial_csv("20260529")
    c = mx.decision_group_counts(df)
    assert c.get("BUY", 0) == 7  # 与 20260529 报告摘要一致


def test_forward_group_perf_buy_underperforms_drop():
    df = dl.load_forward_csv("20260522")
    perf = mx.forward_group_perf(df, "ret_T+1").set_index("decision")
    assert perf.loc["BUY", "mean_ret"] < perf.loc["DROP", "mean_ret"]  # 已知反向


def test_buy_vs_drop_spread_negative_sample():
    df = dl.load_forward_csv("20260522")
    spread = mx.buy_vs_drop(df, "ret_T+1")
    assert spread < 0
