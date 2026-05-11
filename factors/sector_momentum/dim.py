"""sector_momentum 维度: 板块"启动信号"打分（反追高版）.

设计原则（见 feedback_anti_chasing_high.md）: 不用纯 N 日涨幅排名 / 连涨天数,
否则会把"已经涨完"的板块推到前面.

因子:
- ret_5d   : 近 5 日涨幅 (短期动量)
- ret_30d  : 近 30 日涨幅 (用于判断"是否还在低位")
- pos_60d  : 当前价格在过去 60 日 (high, low) 中的相对位置 0~1, 越小越早期
- accel    : 加速度 = ret_5d - ret_20d * (5/20), >0 表示短期跑得比长期均速快
- sector_rsi : 板块自身 14 日 RSI, >70 视为过热
- rel_5d   : 5d 板块涨幅 - 5d 沪深300 涨幅 (相对强度)

打分:
breakout = (ret_5d > 0.03) AND (pos_60d < 0.70)   # 短期强 + 还没在 60d 高位
sector_raw = 0.40*z(breakout)
           + 0.25*z(accel)
           + 0.20*z(rel_5d)
           - 0.30*z(rsi_overheat)                  # 过热直接扣
- 0.0xx*z 列表里负权重表示"越大越扣"
sector_rank = cross_section_rank(sector_raw)       # 0=最强板块（启动早期）, 1=最弱

个股 sector_rank = 个股所在 SW2 板块的 sector_rank.
"""
from __future__ import annotations
import pandas as pd

from engine.ranker import cross_section_rank, zscore
from .data import get_sw2_list, get_sw2_kline, get_symbol_to_sw2


def _ret(close: pd.Series, n: int) -> float | None:
    if len(close) <= n:
        return None
    return float(close.iloc[-1] / close.iloc[-1 - n] - 1)


def _pos_in_range(close: pd.Series, n: int = 60) -> float | None:
    """当前价在过去 n 日 (low, high) 中的位置, 0~1, 越小越早期."""
    if len(close) < n + 1:
        return None
    window = close.iloc[-n:]
    lo, hi = float(window.min()), float(window.max())
    if hi <= lo:
        return 0.5
    return float((close.iloc[-1] - lo) / (hi - lo))


def _rsi(close: pd.Series, n: int = 14) -> float | None:
    """简单 RSI."""
    if len(close) < n + 1:
        return None
    diff = close.diff().iloc[-n:]
    up = diff.clip(lower=0).mean()
    dn = (-diff.clip(upper=0)).mean()
    if dn == 0:
        return 100.0
    rs = up / dn
    return float(100 - 100 / (1 + rs))


def compute_sector_scores(csi300_5d_ret: float | None = None) -> pd.DataFrame:
    """对所有 SW2 板块计算"启动信号"分.

    返回列: code, name, ret_5d, ret_30d, pos_60d, accel, sector_rsi,
            rel_5d, breakout_flag, sector_raw, sector_rank
    """
    sw2 = get_sw2_list()
    if sw2.empty:
        return pd.DataFrame()

    rows = []
    fresh_cutoff = pd.Timestamp.today() - pd.Timedelta(days=10)
    for _, r in sw2.iterrows():
        code, name = r["code"], r["name"]
        kl = get_sw2_kline(code)
        if kl.empty or kl["date"].max() < fresh_cutoff:
            continue
        close = kl["close"].astype(float).reset_index(drop=True)
        r5, r20, r30 = _ret(close, 5), _ret(close, 20), _ret(close, 30)
        accel = (r5 - r20 * (5 / 20)) if (r5 is not None and r20 is not None) else None
        rows.append({
            "code": code, "name": name,
            "ret_5d": r5, "ret_30d": r30,
            "pos_60d": _pos_in_range(close, 60),
            "accel": accel,
            "sector_rsi": _rsi(close, 14),
        })
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["rel_5d"] = pd.to_numeric(df["ret_5d"], errors="coerce") - (csi300_5d_ret or 0)

    r5 = pd.to_numeric(df["ret_5d"], errors="coerce")
    pos = pd.to_numeric(df["pos_60d"], errors="coerce")
    accel = pd.to_numeric(df["accel"], errors="coerce")
    rel = pd.to_numeric(df["rel_5d"], errors="coerce")
    rsi = pd.to_numeric(df["sector_rsi"], errors="coerce")

    # breakout: 短期涨 > 3% 且 60d 位置 < 70% (还没冲到顶)
    breakout = ((r5 > 0.03) & (pos < 0.70)).astype(float)
    df["breakout_flag"] = breakout

    # 过热项: RSI 越过 60 开始扣分, 70 满格扣
    overheat = ((rsi - 60) / 10).clip(lower=0, upper=1).fillna(0)

    raw = (0.40 * zscore(breakout)
           + 0.25 * zscore(accel.fillna(accel.median() if accel.notna().any() else 0))
           + 0.20 * zscore(rel.fillna(rel.median() if rel.notna().any() else 0))
           - 0.30 * zscore(overheat.fillna(0)))
    df["sector_raw"] = raw
    df["sector_rank"] = cross_section_rank(raw)
    return df.sort_values("sector_rank").reset_index(drop=True)


def map_to_symbols(symbols: list[str], sector_df: pd.DataFrame) -> pd.DataFrame:
    """把板块得分映射回个股.

    返回列: symbol, sw2_code, sw2_name, sector_ret_5d, sector_ret_30d,
           sector_pos_60d, sector_rsi, sector_breakout, sector_raw, sector_rank
    缺映射的个股 sector_rank=0.5 (中性), 不影响 composite.
    """
    s2b = get_symbol_to_sw2()
    by_code = sector_df.set_index("code") if not sector_df.empty else pd.DataFrame()

    def _opt(v):
        return float(v) if pd.notna(v) else None

    rows = []
    for sym in symbols:
        code, name = s2b.get(sym, (None, None))
        row: dict = {"symbol": sym, "sw2_code": code, "sw2_name": name}
        if code and code in by_code.index:
            sec = by_code.loc[code]
            row.update({
                "sector_ret_5d": _opt(sec.get("ret_5d")),
                "sector_ret_30d": _opt(sec.get("ret_30d")),
                "sector_pos_60d": _opt(sec.get("pos_60d")),
                "sector_rsi": _opt(sec.get("sector_rsi")),
                "sector_breakout": int(sec.get("breakout_flag", 0)),
                "sector_raw": float(sec["sector_raw"]),
                "sector_rank": float(sec["sector_rank"]),
            })
        else:
            row.update({
                "sector_ret_5d": None, "sector_ret_30d": None,
                "sector_pos_60d": None, "sector_rsi": None,
                "sector_breakout": 0, "sector_raw": 0.0, "sector_rank": 0.5,
            })
        rows.append(row)
    return pd.DataFrame(rows)
