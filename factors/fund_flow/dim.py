"""主力资金面维度合成 → fund_flow_rank."""
from __future__ import annotations
import pandas as pd

from engine.ranker import cross_section_rank, percentile, zscore
from .data import get_fund_flow_rank, get_north_holding, get_lhb_recent


def compute_one(
    symbol: str,
    name: str = "",
    lhb_df: pd.DataFrame | None = None,
    flow_3d: pd.DataFrame | None = None,
    flow_5d: pd.DataFrame | None = None,
) -> dict:
    """单只股票的主力资金原子因子（基于 bulk 快照）."""
    out = {"symbol": symbol, "name": name}

    # ---- 主力净流入（3 日 / 5 日，来自同花顺 bulk 排行）----
    # 同花顺只给"资金流入净额"绝对金额，没有净占比；
    # 我们用 净额 / 流通市值 估算占比（流通市值 = 最新价 × 流通股本，从 fundamental.individual_info 取，
    # 但 info_em 也限流；此处直接用净额做横截面 zscore，省去对流通市值的依赖）。
    if flow_3d is None:
        flow_3d = get_fund_flow_rank("3日排行")
    if flow_5d is None:
        flow_5d = get_fund_flow_rank("5日排行")

    def _amt_and_chg(df):
        if df is None or df.empty:
            return None, None
        sub = df[df["代码"] == symbol]
        if sub.empty:
            return None, None
        amt = pd.to_numeric(sub["资金流入净额_元"].iloc[0], errors="coerce") if "资金流入净额_元" in df.columns else None
        chg_col = next((c for c in ("阶段涨跌幅", "涨跌幅") if c in df.columns), None)
        chg = pd.to_numeric(sub[chg_col].iloc[0], errors="coerce") if chg_col else None
        return (None if amt is None or pd.isna(amt) else float(amt)), (None if chg is None or pd.isna(chg) else float(chg))

    amt3, chg3 = _amt_and_chg(flow_3d)
    amt5, chg5 = _amt_and_chg(flow_5d)
    out["big_order_inflow_3d_amount"] = amt3
    out["big_order_inflow_5d_amount"] = amt5
    out["price_change_3d"] = chg3
    out["price_change_5d"] = chg5
    # ratio 留空（缺流通市值），合成时用 zscore(amt) 即可

    # ---- 北向持股快照（持股比例 + 持股市值 5 日变化）----
    nh = get_north_holding(symbol)
    if not nh.empty:
        try:
            row = nh.iloc[-1]
            pct = pd.to_numeric(row.get("持股数量占A股百分比"), errors="coerce")
            close = pd.to_numeric(row.get("当日收盘价"), errors="coerce")
            value = pd.to_numeric(row.get("持股市值"), errors="coerce")
            chg5 = pd.to_numeric(row.get("持股市值变化-5日"), errors="coerce")
            chg10 = pd.to_numeric(row.get("持股市值变化-10日"), errors="coerce")
            out["north_holding_pct"] = float(pct) if pd.notna(pct) else None
            # 用 持股市值变化-5日 占当前持股市值 作为 5 日变化代理
            if pd.notna(chg5) and pd.notna(value) and value > 0:
                out["north_holding_change_5d"] = float(chg5) / float(value)
            else:
                out["north_holding_change_5d"] = None
            out["north_heavy_holding_flag"] = 1.0 if pd.notna(pct) and pct >= 5.0 else 0.0
            # 自身分位无历史时序，用 5d/10d 比值粗代（>1 说明 5 日内加仓快于过去 10 日）
            if pd.notna(chg5) and pd.notna(chg10) and chg10 != 0:
                out["north_holding_self_percentile"] = min(max(float(chg5) / float(chg10), 0.0), 1.0)
            else:
                out["north_holding_self_percentile"] = None
        except Exception as e:
            print(f"  [north parse] {symbol} {e}")
            out.update(_north_empty())
    else:
        out.update(_north_empty())

    # ---- 龙虎榜机构席位次数（近 30 天） ----
    if lhb_df is None:
        lhb_df = get_lhb_recent(days=30)
    if not lhb_df.empty:
        sym_col = next((c for c in lhb_df.columns if "代码" in str(c)), None)
        cnt = 0
        if sym_col:
            sub = lhb_df[lhb_df[sym_col].astype(str).str.zfill(6) == symbol]
            cnt = len(sub)
        out["lhb_count_30d"] = cnt
    else:
        out["lhb_count_30d"] = 0

    # 融资融券：免费接口质量较差，先置 0 占位
    out["margin_balance_change_5d"] = 0.0

    return out


def _north_empty() -> dict:
    return {
        "north_holding_pct": None,
        "north_holding_change_5d": None,
        "north_heavy_holding_flag": 0.0,
        "north_holding_self_percentile": None,
    }


def compose_dim(rows: list[dict]) -> pd.DataFrame:
    """合成 fund_flow_raw + fund_flow_rank.

    权重（基于同花顺 + 北向 + 龙虎榜数据可得性）:
    - 0.35 big_order_inflow_3d_amount (主力 3 日净流入金额，zscore)
    - 0.15 big_order_inflow_5d_amount (主力 5 日，zscore，趋势确认)
    - 0.20 north_holding_change_5d (北向 5 日持股市值变化，zscore)
    - 0.15 north_heavy_holding_flag (北向重仓标记，0/1)
    - 0.10 north_holding_self_percentile (越高越积极)
    - 0.05 lhb_count_30d (zscore)
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    def _z(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(0.0, index=df.index)
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() == 0:
            return pd.Series(0.0, index=df.index)
        return zscore(s.fillna(s.median()))

    big3 = _z("big_order_inflow_3d_amount")
    big5 = _z("big_order_inflow_5d_amount")
    nh_chg = _z("north_holding_change_5d")
    nh_flag = pd.to_numeric(df["north_heavy_holding_flag"], errors="coerce").fillna(0.0)
    nh_pct = pd.to_numeric(df["north_holding_self_percentile"], errors="coerce").fillna(0.5)
    lhb = _z("lhb_count_30d")

    raw = (
        0.35 * big3
        + 0.15 * big5
        + 0.20 * nh_chg
        + 0.15 * nh_flag
        + 0.10 * nh_pct
        + 0.05 * lhb
    )
    df["fund_flow_raw"] = raw
    df["fund_flow_rank"] = cross_section_rank(raw)
    return df
