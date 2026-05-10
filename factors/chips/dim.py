"""筹码维度（自算筹码分布）→ chips_rank.

近似算法（spec §chips_dim）:
- 对每根 K 线 i: 权重 = volume_i × decay^(t-i) × 1/(high_i - low_i)
  把权重均匀撒在 [low_i, high_i] 价格区间
- 衰减系数 decay 用平均换手率代理: decay = (1 - turnover_avg/100)
  换手率越高，过去筹码沉淀越快被覆盖

原子因子:
- chips_concentration  = 90% 筹码区间宽度 / 当前价（越小越集中）
- cost_deviation       = (当前价 − 加权平均成本) / 加权平均成本
- chips_peak_distance  = 距最近筹码峰中心的价差比例
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from engine.ranker import cross_section_rank
from .data import get_kline_for_chips


def _chip_distribution(df: pd.DataFrame, bins: int = 200, lookback: int = 120) -> tuple[np.ndarray, np.ndarray]:
    """返回 (price_grid, chip_weight)."""
    df = df.tail(lookback).reset_index(drop=True)
    n = len(df)
    if n < 20:
        return np.array([]), np.array([])

    # decay 用平均换手率代理
    if "turnover" in df.columns and df["turnover"].notna().any():
        turn_avg = df["turnover"].mean() / 100.0
    else:
        turn_avg = 0.02
    decay = max(0.90, min(0.995, 1 - turn_avg))
    age = np.arange(n)[::-1]  # 最新=0
    decay_w = decay ** age

    p_min = float(df["low"].min())
    p_max = float(df["high"].max())
    if p_max <= p_min:
        return np.array([]), np.array([])
    grid = np.linspace(p_min, p_max, bins)
    chip = np.zeros(bins)
    lows = df["low"].to_numpy()
    highs = df["high"].to_numpy()
    vols = df["volume"].to_numpy()
    for i in range(n):
        lo, hi, vol = lows[i], highs[i], vols[i]
        if hi <= lo or vol <= 0 or pd.isna(vol):
            continue
        idx = (grid >= lo) & (grid <= hi)
        nb = int(idx.sum())
        if nb == 0:
            continue
        chip[idx] += vol * decay_w[i] / nb
    s = chip.sum()
    if s > 0:
        chip = chip / s  # 归一化为概率
    return grid, chip


def _quantile_price(grid: np.ndarray, chip: np.ndarray, q: float) -> float:
    """筹码累积分位对应价格."""
    cum = chip.cumsum()
    idx = int(np.searchsorted(cum, q))
    idx = max(0, min(idx, len(grid) - 1))
    return float(grid[idx])


def compute_one(symbol: str, name: str = "", kline: pd.DataFrame | None = None) -> dict:
    out = {"symbol": symbol, "name": name}
    if kline is None:
        kline = get_kline_for_chips(symbol)
    if kline is None or kline.empty or len(kline) < 30:
        out.update({"chips_concentration": None, "cost_deviation": None,
                    "chips_peak_distance": None, "chips_raw": 0.0})
        return out

    grid, chip = _chip_distribution(kline)
    if grid.size == 0:
        out.update({"chips_concentration": None, "cost_deviation": None,
                    "chips_peak_distance": None, "chips_raw": 0.0})
        return out

    cur = float(kline["close"].iloc[-1])
    p05 = _quantile_price(grid, chip, 0.05)
    p95 = _quantile_price(grid, chip, 0.95)
    concentration = (p95 - p05) / cur if cur > 0 else None
    avg_cost = float((grid * chip).sum())
    cost_dev = (cur - avg_cost) / avg_cost if avg_cost > 0 else None
    peak_idx = int(chip.argmax())
    peak_price = float(grid[peak_idx])
    peak_dist = (cur - peak_price) / cur if cur > 0 else None

    # ---- 合成（自定义，spec 未给具体公式）----
    # 1. 集中度: 90% 区间宽度占比 < 0.20 = 强集中（90% 筹码挤在 ±10% 内）
    if concentration is not None:
        con_score = max(0.0, 1.0 - concentration / 0.5)  # >0.5 完全分散为 0
    else:
        con_score = 0.5
    # 2. 成本偏离: 微浮盈区 [-5%, +20%] 最佳；深亏（套牢盘巨大）压低；过度浮盈过热
    if cost_dev is not None:
        if -0.05 <= cost_dev <= 0.20:
            dev_score = 1.0
        elif cost_dev < -0.05:
            dev_score = max(0.0, 1.0 + cost_dev / 0.30)  # -35% → 0
        else:
            dev_score = max(0.0, 1.0 - (cost_dev - 0.20) / 0.30)
    else:
        dev_score = 0.5
    chips_raw = 0.6 * con_score + 0.4 * dev_score

    out.update({
        "chips_concentration": concentration,
        "cost_deviation": cost_dev,
        "chips_peak_distance": peak_dist,
        "chips_raw": float(chips_raw),
    })
    return out


def compose_dim(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    raw = pd.to_numeric(df["chips_raw"], errors="coerce").fillna(0.0)
    df["chips_rank"] = cross_section_rank(raw)
    return df
