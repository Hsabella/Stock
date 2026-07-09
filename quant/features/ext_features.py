"""扩展特征构建：warehouse 原始数据 → 对齐 qlib 交易日历的特征 parquet。

产出 ~/.qlib/quant_warehouse/ext_features.parquet，索引 (datetime, instrument)
（与 qlib DataLoader 输出同序），列见 alpha_pv.EXT_FEATURES：
- turn_20            20 日换手率均值（新浪 turnover）
- turn_ratio_5_120   5 日/120 日换手比（量能异动，方向由 IC 定）
- size               -log(流通市值)，流通市值 = 不复权收盘价 × 流通股本
- ep / bp            1/PE_TTM、1/PB（百度估值为稀疏采样点，前向填充限 63 个交易日）

用法: python -m quant.features.ext_features
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from quant.config import load_config
from quant.data import warehouse

FFILL_LIMIT = 63  # 估值稀疏点最多前向填充一个季度


def _trade_calendar() -> pd.DatetimeIndex:
    cfg = load_config("data")
    cal_file = f"{cfg['qlib']['provider_uri']}/calendars/day.txt"
    with open(cal_file) as f:
        return pd.DatetimeIndex([line.strip()[:10] for line in f if line.strip()])


def build_turnover_size(sina: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    out = []
    for code, g in sina.groupby(level="instrument"):
        s = g.droplevel(0).sort_index().reindex(calendar[(calendar >= g.index.get_level_values(1).min()) &
                                                         (calendar <= g.index.get_level_values(1).max())])
        turn = s["turnover"]
        feat = pd.DataFrame({
            "turn_20": turn.rolling(20, min_periods=10).mean(),
            "turn_ratio_5_120": turn.rolling(5, min_periods=3).mean()
            / turn.rolling(120, min_periods=60).mean(),
            "size": -np.log((s["raw_close"] * s["outstanding_share"]).replace(0, np.nan)),
        })
        feat["instrument"] = code
        out.append(feat)
    df = pd.concat(out)
    df.index.name = "datetime"
    return df.reset_index().set_index(["datetime", "instrument"])


def build_valuation(baidu: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    out = []
    for code, g in baidu.groupby(level="instrument"):
        s = g.droplevel(0).sort_index()
        # 稀疏采样点对齐日历后限量 ffill；PE<=0（亏损）时 EP 记为负值而非缺失
        s = s.reindex(calendar[(calendar >= s.index.min()) & (calendar <= s.index.max())])
        s = s.ffill(limit=FFILL_LIMIT)
        feat = pd.DataFrame({
            "ep": 1.0 / s["pe_ttm"].where(s["pe_ttm"].abs() > 1e-6),
            "bp": 1.0 / s["pb"].where(s["pb"] > 1e-6),
        })
        feat["instrument"] = code
        out.append(feat)
    df = pd.concat(out)
    df.index.name = "datetime"
    return df.reset_index().set_index(["datetime", "instrument"])


def build(save: bool = True) -> pd.DataFrame:
    calendar = _trade_calendar()
    sina = warehouse.load("sina_daily")
    if sina.empty:
        raise RuntimeError("warehouse 缺 sina_daily，先跑 python -m quant.data.akshare_fetcher --dataset sina_daily")
    feats = build_turnover_size(sina, calendar)

    baidu = warehouse.load("baidu_valuation")
    if baidu.empty:
        print("[ext] 警告: baidu_valuation 为空，ep/bp 缺席（可先跑量价+换手，估值到位后重建）")
    else:
        feats = feats.join(build_valuation(baidu, calendar), how="left")

    feats = feats.sort_index()
    if save:
        path = warehouse.warehouse_dir() / "ext_features.parquet"
        feats.to_parquet(path)
        print(f"[ext] 已写 {path}: {feats.shape[0]:,} 行 × {feats.shape[1]} 列, "
              f"{feats.index.get_level_values('instrument').nunique()} 只")
    return feats


if __name__ == "__main__":
    build()
    sys.exit(0)
