# tests/test_reweight.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import reweight_backtest as rb

CURRENT = {"fundamental_rank": 0.22, "fund_flow_rank": 0.18, "liquidity_rank": 0.12,
           "chips_rank": 0.08, "regime_rank": 0.07, "tech_rank": 0.05,
           "sector_rank": 0.18, "news_rank": 0.10}
TWO_DIM = {"fund_flow_rank": 0.5, "regime_rank": 0.5}

def test_current_config_reproduces_negative_ic():
    res = rb.run(CURRENT, use_penalty=True)
    assert res["days"] >= 10
    assert res["ic_mean"] is not None and res["ic_mean"] < 0  # 复现反向

def test_two_dim_beats_current():
    cur = rb.run(CURRENT, use_penalty=True)["ic_mean"]
    two = rb.run(TWO_DIM, use_penalty=False)["ic_mean"]
    assert two > cur  # 两维模型应优于当前 8 维
