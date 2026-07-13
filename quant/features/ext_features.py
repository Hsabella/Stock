"""扩展特征构建：warehouse 原始数据 → 对齐 qlib 交易日历的特征 parquet。

产出 ~/.qlib/quant_warehouse/ext_features.parquet，索引 (datetime, instrument)
（与 qlib DataLoader 输出同序），列见 alpha_pv.EXT_FEATURES：
- turn_20            20 日换手率均值（新浪 turnover）
- turn_ratio_5_120   5 日/120 日换手比（量能异动，方向由 IC 定）
- size               -log(流通市值)，流通市值 = 不复权收盘价 × 流通股本
- ep / bp            1/PE_TTM、1/PB（百度估值为稀疏采样点，前向填充限 63 个交易日）
- roe_ttm / profit_yoy / roe_delta   财报 PIT 因子（v2 Step2）：单季 ROE 滚动 4 季、
  归母净利累计同比（分母取绝对值防负基数翻符号）、roe_ttm 环比。财报数据的
  datetime 已是法定披露截止日（akshare_fetcher.pit_available_date），特征在该日
  之后才可见，前向填充限 130 交易日（下一份财报最迟 ~85 交易日后接棒）

用法: python -m quant.features.ext_features
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from quant.config import load_config
from quant.data import warehouse

FFILL_LIMIT = 63  # 估值稀疏点最多前向填充一个季度
FIN_FFILL_LIMIT = 130  # 财报特征：披露截止日最大间隔=三季报(10-31)→次年年报(4-30)约 122 交易日


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


def _fin_one(byp: pd.DataFrame) -> pd.DataFrame:
    """单只股票：报告期累计值 → roe_ttm / profit_yoy / roe_delta（索引=报告期）。"""
    pidx = pd.period_range(byp.index.min(), byp.index.max(), freq="Q").to_timestamp(how="end").normalize()
    byp = byp.reindex(pidx)  # 缺季补 NaN，保证 shift/rolling 的"4 期"就是"1 年"
    roe_q = byp["roe"].groupby(pidx.year).diff()
    roe_q = roe_q.fillna(byp["roe"].where(pidx.month == 3))  # 每年首期(一季报)即单季值
    roe_ttm = roe_q.rolling(4, min_periods=4).sum()
    prev = byp["net_profit"].shift(4)
    return pd.DataFrame({
        "roe_ttm": roe_ttm,
        "profit_yoy": (byp["net_profit"] - prev) / prev.abs(),
        "roe_delta": roe_ttm.diff(),
    })


def build_financials(fin: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    from quant.data.akshare_fetcher import pit_available_date

    out = []
    for code, g in fin.groupby(level="instrument"):
        byp = g.droplevel(0).sort_index()  # 索引即报告期
        byp = byp[~byp.index.duplicated(keep="last")]
        feat = _fin_one(byp)
        # 报告期 → 披露截止日才可见；对齐日历后限量 ffill
        feat.index = pd.DatetimeIndex([pit_available_date(p) for p in feat.index])
        feat = feat[~feat.index.duplicated(keep="last")].sort_index()
        span = calendar[(calendar >= feat.index.min()) & (calendar <= calendar[-1])]
        if len(span) == 0 or feat.dropna(how="all").empty:
            continue
        feat = feat.reindex(span.union(feat.index)).ffill(limit=FIN_FFILL_LIMIT).reindex(span)
        feat["instrument"] = code
        out.append(feat)
    if not out:
        return pd.DataFrame()
    df = pd.concat(out)
    df.index.name = "datetime"
    return df.reset_index().set_index(["datetime", "instrument"])


def build_regime(calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """全截面同值的风格切换特征（v2 Step3），索引单层 datetime，join 时按日广播。"""
    from quant.data.index_daily import load_closes

    px = load_closes()
    if px.empty or not {"SH000300", "SH000852", "SH000906"}.issubset(px.columns):
        print("[ext] 警告: index_daily 不全，regime 特征缺席（先跑 python -m quant.data.index_daily）")
        return pd.DataFrame()
    out = pd.DataFrame({
        "mkt_vol_20": px["SH000906"].pct_change().rolling(20).std(),
        "mkt_mom_20": px["SH000906"].pct_change(20),
        "size_spread_20": px["SH000852"].pct_change(20) - px["SH000300"].pct_change(20),
    })
    out = out.reindex(calendar).ffill(limit=5)  # 指数仓晚更新几天时容忍
    out.index.name = "datetime"
    return out


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

    fin = warehouse.load("financials")
    if fin.empty:
        print("[ext] 警告: financials 为空，财报因子缺席（先跑 akshare_fetcher --dataset financials）")
    else:
        feats = feats.join(build_financials(fin, calendar), how="left")

    regime = build_regime(calendar)
    if not regime.empty:
        feats = feats.join(regime, on="datetime")

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
