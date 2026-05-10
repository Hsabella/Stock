"""技术维度合成：RSI / MACD / KDJ / BOLL / OBV 五族信号 → technical_rank.

详见 .omc/specs/deep-interview-decision-engine.md → technical_dim 章节。
合成: tech_raw = 0.30*RSI_max + 0.25*MACD_max + 0.20*OBV_max + 0.15*KDJ_max + 0.10*BB_max
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from engine.ranker import cross_section_rank
from .data import get_kline_for_tech


# ---------- 基础指标计算（独立实现，避免依赖 BasicIndicator 的 fillna 副作用）----------

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_g = gain.rolling(n).mean()
    avg_l = loss.rolling(n).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist


def _kdj(high: pd.Series, low: pd.Series, close: pd.Series, n=9, m1=3, m2=3):
    low_n = low.rolling(n).min()
    high_n = high.rolling(n).max()
    rsv = (close - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    k = rsv.ewm(alpha=1 / m1, adjust=False).mean()
    d = k.ewm(alpha=1 / m2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def _bbands(close: pd.Series, n=20, k=2.0):
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std()
    upper = mid + k * sd
    lower = mid - k * sd
    width = (upper - lower) / mid
    return upper, mid, lower, width


def _atr(high, low, close, n=14):
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _obv(close: pd.Series, vol: pd.Series) -> pd.Series:
    sign = np.sign(close.diff().fillna(0))
    return (sign * vol).cumsum()


# ---------- 信号识别（取每族最强）----------

def _rsi_signals(rsi: pd.Series, close: pd.Series) -> float:
    if rsi.dropna().shape[0] < 20:
        return 0.0
    cur = rsi.iloc[-1]
    strengths = [0.0]
    # rsi_oversold_breakout: 过去 5 日内出现 RSI<30 且当前上穿 35
    last5 = rsi.iloc[-6:-1]
    if (last5 < 30).any() and rsi.iloc[-2] < 35 and cur >= 35:
        strengths.append(3.0)
    # rsi_low_turn: 当前 ∈ [30,45] 且 RSI[t]-RSI[t-2] > 3
    if 30 <= cur <= 45 and (cur - rsi.iloc[-3]) > 3:
        strengths.append(2.0)
    # rsi_bullish_divergence: 过去 20 日价格新低 + RSI 同期未创新低
    if close.shape[0] >= 20:
        win_close = close.iloc[-20:]
        win_rsi = rsi.iloc[-20:]
        if win_close.iloc[-1] == win_close.min() and win_rsi.iloc[-1] > win_rsi.min():
            strengths.append(3.0)
    return max(strengths)


def _macd_signals(macd: pd.Series, sig: pd.Series, hist: pd.Series, atr14: pd.Series) -> tuple[float, list[str]]:
    if macd.dropna().shape[0] < 30:
        return 0.0, []
    risks = []
    strengths = [0.0]
    cur_m, cur_s = macd.iloc[-1], sig.iloc[-1]
    prv_m, prv_s = macd.iloc[-2], sig.iloc[-2]
    golden = prv_m <= prv_s and cur_m > cur_s
    if golden:
        if cur_m < 0 and abs(cur_m) < (atr14.iloc[-1] or 0):
            strengths.append(3.0)  # 低位金叉
        elif cur_m > 0:
            risks.append("RISK_high_macd_cross")  # 高位金叉风险标签
    # macd_zero_cross_up
    if prv_m <= 0 and cur_m > 0:
        strengths.append(2.0)
    # hist_reversal: 连续 2 天负值绝对值缩小（即将转正）
    if hist.shape[0] >= 3:
        h1, h2, h3 = hist.iloc[-3], hist.iloc[-2], hist.iloc[-1]
        if h3 < 0 and h2 < 0 and abs(h2) < abs(h1) and abs(h3) < abs(h2):
            strengths.append(2.0)
    return max(strengths), risks


def _kdj_signals(k: pd.Series, d: pd.Series, j: pd.Series) -> tuple[float, list[str]]:
    if j.dropna().shape[0] < 5:
        return 0.0, []
    risks = []
    strengths = [0.0]
    if j.iloc[-2] < 0 and j.iloc[-1] > 0:
        strengths.append(3.0)  # j_zero_cross
    golden = k.iloc[-2] <= d.iloc[-2] and k.iloc[-1] > d.iloc[-1]
    if golden:
        # dull skip: K 连 ≥3 天 <20
        dull = (k.iloc[-3:] < 20).all()
        s = 2.0 * (0.3 if dull else 1.0)
        if k.iloc[-1] < 25:
            strengths.append(s)
        if j.iloc[-1] > 80:
            risks.append("RISK_high_kdj_cross")
    return max(strengths), risks


def _bb_signals(close: pd.Series, lower: pd.Series, width: pd.Series, open_: pd.Series) -> float:
    if close.shape[0] < 50:
        return 0.0
    strengths = [0.0]
    # bb_lower_breakout: 收盘从 < BB_LOWER 上穿 BB_LOWER 且收阳
    if close.iloc[-2] < lower.iloc[-2] and close.iloc[-1] >= lower.iloc[-1] and close.iloc[-1] > open_.iloc[-1]:
        strengths.append(3.0)
    # bb_squeeze_release: BB_WIDTH 在 50 日 30% 分位以下 → 当日扩张
    win = width.iloc[-50:]
    if width.iloc[-2] <= win.quantile(0.3) and width.iloc[-1] > width.iloc[-2]:
        strengths.append(2.0)
    return max(strengths)


def _obv_signals(close: pd.Series, obv: pd.Series) -> tuple[float, list[str]]:
    if close.shape[0] < 60:
        return 0.0, []
    risks = []
    strengths = [0.0]
    win = close.iloc[-20:]
    win_obv = obv.iloc[-20:]
    if close.shape[0] >= 20 and win.iloc[-1] == win.min() and win_obv.iloc[-1] > win_obv.min():
        strengths.append(3.0)  # bullish divergence
    if close.shape[0] >= 60:
        win60_obv = obv.iloc[-60:]
        win60_close = close.iloc[-60:]
        if obv.iloc[-1] >= win60_obv.max() and close.iloc[-1] < win60_close.max():
            strengths.append(2.5)  # obv 创新高 价格未创
        if close.iloc[-1] >= win60_close.max() and obv.iloc[-1] < win60_obv.max():
            risks.append("RISK_obv_bearish_divergence")
    if obv.shape[0] >= 20:
        ma20_obv = obv.rolling(20).mean()
        ma20_close = close.rolling(20).mean()
        if ma20_obv.diff().iloc[-1] > 0 and close.iloc[-1] > (ma20_close.iloc[-1] or 0):
            strengths.append(1.5)
    return max(strengths), risks


def compute_one(symbol: str, name: str = "", kline: pd.DataFrame | None = None) -> dict:
    out = {"symbol": symbol, "name": name}
    if kline is None:
        kline = get_kline_for_tech(symbol)
    if kline is None or kline.empty or len(kline) < 60:
        out.update({
            "rsi_strength": 0.0, "macd_strength": 0.0, "kdj_strength": 0.0,
            "bb_strength": 0.0, "obv_strength": 0.0,
            "tech_raw": 0.0, "rsi_value": None, "risk_tags": [],
        })
        return out

    df = kline.copy().reset_index(drop=True)
    close, high, low, open_, vol = df["close"], df["high"], df["low"], df["open"], df["volume"]

    rsi = _rsi(close)
    macd, sig, hist = _macd(close)
    k, d, j = _kdj(high, low, close)
    bb_u, bb_m, bb_l, bb_w = _bbands(close)
    atr14 = _atr(high, low, close)
    obv = _obv(close, vol)

    rsi_s = _rsi_signals(rsi, close)
    macd_s, macd_risks = _macd_signals(macd, sig, hist, atr14)
    kdj_s, kdj_risks = _kdj_signals(k, d, j)
    bb_s = _bb_signals(close, bb_l, bb_w, open_)
    obv_s, obv_risks = _obv_signals(close, obv)

    tech_raw = (
        0.30 * rsi_s + 0.25 * macd_s + 0.20 * obv_s + 0.15 * kdj_s + 0.10 * bb_s
    )

    out.update({
        "rsi_strength": rsi_s,
        "macd_strength": macd_s,
        "kdj_strength": kdj_s,
        "bb_strength": bb_s,
        "obv_strength": obv_s,
        "tech_raw": float(tech_raw),
        "rsi_value": float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None,
        "risk_tags": macd_risks + kdj_risks + obv_risks,
    })
    return out


def compose_dim(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    raw = pd.to_numeric(df["tech_raw"], errors="coerce").fillna(0.0)
    df["tech_rank"] = cross_section_rank(raw)
    return df
