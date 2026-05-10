"""流动性活跃度维度（量比 + 换手率触发器）.

详见 .omc/specs/deep-interview-decision-engine.md → liquidity_dim 章节。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from engine.ranker import cross_section_rank
from .data import get_kline_for_liquidity


def _last(s: pd.Series, default=np.nan):
    return float(s.iloc[-1]) if len(s) and pd.notna(s.iloc[-1]) else default


def compute_one(symbol: str, name: str = "", kline: pd.DataFrame | None = None) -> dict:
    out = {"symbol": symbol, "name": name}
    if kline is None:
        kline = get_kline_for_liquidity(symbol)
    if kline is None or kline.empty or len(kline) < 25:
        out.update({
            "volume_ratio": None, "volume_ratio_5d_avg": None,
            "turnover_today": None, "turnover_ma5_ratio": None,
            "liq_signal": "no_data", "liq_signal_strength": 0.0,
            "liquidity_raw": 0.0, "risk_tags": [],
        })
        return out

    df = kline.copy().reset_index(drop=True)
    vol = df["volume"].astype(float)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)

    vol_ma5 = vol.rolling(5).mean()
    vol_ratio = vol / vol_ma5
    out["volume_ratio"] = _last(vol_ratio)
    out["volume_ratio_5d_avg"] = _last(vol_ratio.rolling(5).mean())

    if "turnover" in df.columns and df["turnover"].notna().any():
        turn = df["turnover"].astype(float)
        out["turnover_today"] = _last(turn)
        turn_ma20 = turn.rolling(20).mean()
        ratio_series = turn / turn_ma20
        out["turnover_ma5_ratio"] = _last(ratio_series)
        # 自身年内分位（约 250 个交易日）
        if len(turn.dropna()) >= 30:
            t_today = out["turnover_today"]
            hist = turn.dropna()
            out["turnover_self_percentile"] = float((hist <= t_today).mean()) if t_today is not None else None
        else:
            out["turnover_self_percentile"] = None
    else:
        out["turnover_today"] = None
        out["turnover_ma5_ratio"] = None
        out["turnover_self_percentile"] = None

    # ---- 形态特征 ----
    today_close = close.iloc[-1]
    today_open = open_.iloc[-1]
    today_high = high.iloc[-1]
    today_low = low.iloc[-1]
    is_up = today_close > today_open
    rng = max(today_high - today_low, 1e-9)
    upper_shadow_ratio = (today_high - max(today_open, today_close)) / rng

    low60 = low.tail(60).min()
    high20 = high.tail(21).iloc[:-1].max() if len(high) > 21 else high.max()
    dist_low60_pct = (today_close - low60) / low60 if low60 > 0 else 1.0
    breakout_high20 = today_close > high20

    vr = out["volume_ratio"] or 0.0
    vr5 = out["volume_ratio_5d_avg"] or 0.0
    t_today = out["turnover_today"]

    risks: list[str] = []
    candidates: list[tuple[str, float]] = []

    # liq_shrink_floor: 缩量止跌（左侧底部）
    if vr < 0.7 and is_up and dist_low60_pct <= 0.08:
        candidates.append(("liq_shrink_floor", 3.0))
    # liq_mild_breakout: 温和放量突破（右侧）
    if 1.5 <= vr <= 3.0 and is_up and breakout_high20:
        candidates.append(("liq_mild_breakout", 3.0))
    # liq_active_warmup: 持续活跃但不过热
    if 1.2 <= vr5 <= 2.0 and out["turnover_self_percentile"] is not None \
            and 0.60 <= out["turnover_self_percentile"] <= 0.80:
        candidates.append(("liq_active_warmup", 2.0))
    # liq_abnormal_vol_up: 巨量异动 + 收阳 + 上影 < 30%
    if vr > 5.0 and is_up and upper_shadow_ratio < 0.3:
        candidates.append(("liq_abnormal_vol_up", 1.5 * 0.6))  # 谱里 *0.6 折扣

    # ---- 风险标签 ----
    if vr > 5.0 and (not is_up or upper_shadow_ratio > 0.5):
        risks.append("RISK_distribution")
    if t_today is not None:
        if (df["turnover"].tail(5) < 0.5).all():
            risks.append("RISK_illiquid")
        high60 = high.tail(60).max()
        dist_high60_pct = (high60 - today_close) / high60 if high60 > 0 else 1.0
        if t_today > 15.0 and dist_high60_pct <= 0.05:
            risks.append("RISK_overheat")  # RSI 在 dim 合并时再校验

    if candidates:
        sig, strength = max(candidates, key=lambda x: x[1])
    else:
        sig, strength = "none", 0.0

    out["liq_signal"] = sig
    out["liq_signal_strength"] = strength
    out["liquidity_raw"] = strength
    out["risk_tags"] = risks
    return out


def compose_dim(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    raw = pd.to_numeric(df["liquidity_raw"], errors="coerce").fillna(0.0)
    df["liquidity_rank"] = cross_section_rank(raw)
    return df
