"""端到端组合回测入口：预测分 → 周度 Topk 组合 → 成本后绩效报告。

两个 sanity check（M3 验收项）：
    --random-signal   随机信号跑一遍，结果应 ≈ 基准收益 - 成本（防回测器虚增收益）
    --cost-mult 0     成本清零对照，成本拖累量级应 ≈ 换手 × 费率

用法:
    python -m quant.backtest.run_backtest --pred results/quant/pred_lgb_fixed.parquet
    python -m quant.backtest.run_backtest --pred ... --random-signal
    python -m quant.backtest.run_backtest --pred ... --cost-mult 2   # 敏感性
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quant.backtest.report import render
from quant.backtest.strategy import WeeklyTopkDropoutStrategy
from quant.config import load_config

OUT_DIR = Path("results/quant")


def load_signal(path: Path, randomize: bool, seed: int = 42) -> pd.Series:
    score = pd.read_parquet(path)["score"]
    if randomize:
        rng = np.random.default_rng(seed)
        score = pd.Series(rng.standard_normal(len(score)), index=score.index, name="score")
        print("[backtest] 已用随机信号替换预测分（sanity check）")
    return score


def run(pred_path: Path, random_signal: bool, cost_mult: float, tag: str | None,
        start: str | None = None, end: str | None = None, topk: int | None = None,
        neutralize: bool = False) -> int:
    import qlib
    from qlib.contrib.evaluate import backtest_daily

    cfg = load_config("backtest")
    dcfg = load_config("data")
    qlib.init(provider_uri=dcfg["qlib"]["provider_uri"], region="cn")

    score = load_signal(pred_path, random_signal)
    if neutralize:
        from quant.features.neutralize import neutralize as _neutralize
        score = _neutralize(score)
    bcfg = cfg["backtest"]
    scfg = dict(cfg["strategy"])
    if topk is not None:
        scfg["topk"], scfg["n_drop"] = topk, max(2, round(topk * 0.27))

    exchange_kwargs = dict(cfg["backtest"]["exchange"])
    exchange_kwargs["open_cost"] = exchange_kwargs["open_cost"] * cost_mult
    exchange_kwargs["close_cost"] = exchange_kwargs["close_cost"] * cost_mult
    if cost_mult == 0:
        exchange_kwargs["min_cost"] = 0

    strategy = WeeklyTopkDropoutStrategy(
        signal=score,
        topk=scfg["topk"],
        n_drop=scfg["n_drop"],
        only_tradable=True,
        forbid_all_trade_at_limit=True,
        risk_degree=scfg["risk_degree"],
    )
    report, _positions = backtest_daily(
        start_time=start or bcfg["start"],
        end_time=end or bcfg["end"],
        strategy=strategy,
        account=bcfg["account"],
        benchmark=bcfg["benchmark"],
        exchange_kwargs=exchange_kwargs,
    )

    name = tag or pred_path.stem.replace("pred_", "")
    if neutralize:
        name += "_neutral"
    if random_signal:
        name += "_random"
    if cost_mult != 1:
        name += f"_cost{cost_mult:g}x"
    if topk is not None:
        name += f"_top{topk}"
    render(report, OUT_DIR / f"bt_{name}", title=f"WeeklyTopk{scfg['topk']} · {name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--random-signal", action="store_true")
    parser.add_argument("--cost-mult", type=float, default=1.0)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--neutralize", action="store_true",
                        help="加载 pred 后先做行业+市值中性化（v2 Step1）")
    args = parser.parse_args()
    return run(args.pred, args.random_signal, args.cost_mult, args.tag, args.start, args.end,
               args.topk, args.neutralize)


if __name__ == "__main__":
    sys.exit(main())
