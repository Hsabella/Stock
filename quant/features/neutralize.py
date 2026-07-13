"""横截面中性化：剥掉预测分中能被行业哑变量 + log 市值解释的部分，取残差。

v2 Step1（docs/quant/iteration_plan_v2.md）：每个截面日 OLS
    score ~ C(industry) + log_mktcap
残差再按行业组内标准化，即"纯选股观点"，让组合的行业/市值结构贴着基准走。
行业缺失的股票归入 UNK 桶、市值缺失用当日截面中位数填——都不把股票踢出候选。

组内标准化是必须的第二步：UNK 等异质组的残差散度显著高于普通行业组（实测约
1.2-1.3 倍），不归一会机械占领 topk 头部（实测 top30 UNK 占比 15%→30%）；
每组残差除以组内 std 后各组按人口比例进入头部。不足 MIN_GROUP 只的小组用
全截面 std 兜底，避免小样本 std 噪声放大。

用法:
    python -m quant.features.neutralize --pred results/quant/pred_lgb_rolling.parquet
    # → 写 results/quant/pred_lgb_rolling_neutral.parquet（喂给 run_backtest / evaluate）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quant.data import industry, warehouse


MIN_GROUP = 5


def _cross_residual(y: np.ndarray, log_mktcap: np.ndarray, ind: np.ndarray) -> np.ndarray:
    dummies = pd.get_dummies(ind, dtype=float).to_numpy()
    x = np.column_stack([np.ones(len(y)), log_mktcap, dummies[:, 1:]])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ beta


def _standardize_by_group(resid: pd.Series, ind: pd.Series) -> pd.Series:
    grp = resid.groupby(ind)
    std = grp.transform("std")
    size = grp.transform("size")
    fallback = resid.std()
    if not fallback > 1e-12:
        return resid
    scale = std.where((size >= MIN_GROUP) & (std > 1e-12), fallback)
    return resid / scale


def neutralize_frame(score: pd.Series, ind_map: pd.Series, log_mktcap: pd.Series) -> pd.Series:
    """逐截面日回归取残差。score/log_mktcap 索引 (datetime, instrument)，ind_map 索引 instrument。"""
    out = []
    n_unk = 0
    for dt, cross in score.groupby(level="datetime"):
        y = cross.droplevel("datetime")
        try:
            lm = log_mktcap.xs(dt, level="datetime").reindex(y.index)
        except KeyError:
            lm = pd.Series(np.nan, index=y.index)
        lm = lm.fillna(lm.median())
        if lm.isna().all():  # 当日整个截面缺市值 → 退化为仅行业中性化
            lm = pd.Series(0.0, index=y.index)
        ind = ind_map.reindex(y.index).fillna("UNK")
        n_unk += int((ind == "UNK").sum())
        resid = pd.Series(_cross_residual(y.to_numpy(float), lm.to_numpy(float), ind.to_numpy()),
                          index=y.index)
        resid = _standardize_by_group(resid, ind)
        idx = pd.MultiIndex.from_product([[dt], y.index], names=["datetime", "instrument"])
        out.append(pd.Series(resid.to_numpy(), index=idx, name="score"))
    res = pd.concat(out).sort_index()
    n_dates = score.index.get_level_values("datetime").nunique()
    print(f"[neutralize] {n_dates} 个截面完成, 行业缺失(UNK)占比 {n_unk / len(score):.1%}")
    return res


def neutralize(score: pd.Series) -> pd.Series:
    ind_map = industry.load_map()
    if ind_map.empty:
        raise RuntimeError("warehouse 缺 industry 快照，先跑 python -m quant.data.industry")
    ext_path = warehouse.warehouse_dir() / "ext_features.parquet"
    if not ext_path.exists():
        raise RuntimeError("warehouse 缺 ext_features.parquet，先跑 python -m quant.features.ext_features")
    log_mktcap = -pd.read_parquet(ext_path)["size"]  # ext 的 size 定义为 -log(流通市值)
    return neutralize_frame(score, ind_map, log_mktcap)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, type=Path)
    args = parser.parse_args()

    score = pd.read_parquet(args.pred)["score"]
    res = neutralize(score)
    out = args.pred.with_name(f"{args.pred.stem}_neutral.parquet")
    res.to_frame().to_parquet(out)
    print(f"[neutralize] 已写 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
