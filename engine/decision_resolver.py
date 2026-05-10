"""按持仓状态 + composite + gates → 最终 decision.

参考: .omc/specs/deep-interview-decision-engine.md
"""
from __future__ import annotations
import math
import pandas as pd


# ---- 阈值（spec 2026-05-10 放宽版）----
GATE_FUND_RANK = 0.60
GATE_FLOW_RANK = 0.60
GATE_LIQ_RANK = 0.70
BUY_THRESHOLD = 0.55
WATCH_THRESHOLD = 0.45
HARD_RISK_TAGS = {"RISK_distribution", "RISK_overheat"}


def _f(x, default=None):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return default
    try:
        return float(x)
    except (ValueError, TypeError):
        return default


def _gates_passed(row: dict) -> tuple[bool, list[str]]:
    """检查 BUY 三道 gates + 大盘 + 风险标签."""
    fails = []
    if (_f(row.get("fundamental_rank"), 1.0) > GATE_FUND_RANK):
        fails.append(f"fundamental_rank={row.get('fundamental_rank'):.2f}>{GATE_FUND_RANK}")
    if (_f(row.get("fund_flow_rank"), 1.0) > GATE_FLOW_RANK):
        fails.append(f"fund_flow_rank={row.get('fund_flow_rank'):.2f}>{GATE_FLOW_RANK}")
    if (_f(row.get("liquidity_rank"), 1.0) > GATE_LIQ_RANK):
        fails.append(f"liquidity_rank={row.get('liquidity_rank'):.2f}>{GATE_LIQ_RANK}")
    if row.get("index_regime") == "bear":
        fails.append("index_regime=bear")
    risks = set(str(row.get("risk_tags", "")).split(";")) if row.get("risk_tags") else set()
    hard_hits = risks & HARD_RISK_TAGS
    if hard_hits:
        fails.append(f"hard_risk={','.join(hard_hits)}")
    return (len(fails) == 0), fails


def _build_drivers(row: dict) -> list[str]:
    """从原子因子构造可解释 drivers（最少 3 条）."""
    drivers = []
    pe_self = _f(row.get("pe_self_percentile"))
    if pe_self is not None and pe_self < 0.30:
        drivers.append(f"PE 自身 5 年分位 {pe_self*100:.0f}%（便宜）")
    pe_sec = _f(row.get("pe_sector_percentile"))
    if pe_sec is not None and pe_sec < 0.30:
        drivers.append(f"PE 行业分位 {pe_sec*100:.0f}%")
    roe = _f(row.get("roe_ttm"))
    if roe is not None and roe > 10:
        drivers.append(f"ROE_TTM {roe:.1f}%")
    if _f(row.get("north_heavy_holding_flag"), 0) > 0:
        nh = _f(row.get("north_holding_pct"))
        drivers.append(f"北向重仓股（{nh:.1f}%）" if nh else "北向重仓股（≥5%）")
    big3 = _f(row.get("big_order_inflow_3d_amount"))
    if big3 is not None and big3 > 0:
        drivers.append(f"主力 3 日净流入 {big3/1e8:+.2f} 亿")
    sig = row.get("liq_signal")
    if sig and sig not in ("none", "no_data"):
        drivers.append(f"liq 信号: {sig}")
    rsi_strength = _f(row.get("rsi_strength"), 0)
    if rsi_strength >= 3:
        drivers.append("RSI 超卖反弹")
    elif rsi_strength >= 2:
        drivers.append("RSI 低位拐头")
    obv_strength = _f(row.get("obv_strength"), 0)
    if obv_strength >= 2.5:
        drivers.append("OBV 创新高（资金先于价格）")
    macd_strength = _f(row.get("macd_strength"), 0)
    if macd_strength >= 3:
        drivers.append("MACD 低位金叉")
    rs20 = _f(row.get("stock_rs_20d"))
    if rs20 is not None and rs20 > 0.05:
        drivers.append(f"20d 跑赢沪深300 {rs20*100:+.1f}%")
    chips_con = _f(row.get("chips_concentration"))
    if chips_con is not None and chips_con < 0.15:
        drivers.append(f"筹码集中（90% 区间宽度 {chips_con*100:.1f}%）")
    return drivers


def _build_risks(row: dict) -> list[str]:
    risks = []
    raw_risk = row.get("risk_tags") or ""
    if raw_risk:
        risks.extend([t for t in raw_risk.split(";") if t])
    rsi = _f(row.get("rsi_value"))
    if rsi is not None:
        if rsi > 70:
            risks.append(f"RSI={rsi:.0f}（过热）")
        elif rsi < 25:
            risks.append(f"RSI={rsi:.0f}（极超卖，注意接飞刀）")
    pe = _f(row.get("pe_ttm"))
    if pe is not None and pe > 80:
        risks.append(f"PE_TTM={pe:.0f}（估值偏高）")
    return risks


def resolve(row: dict, state: str = "NONE") -> dict:
    """为单只股票决策.

    输入 row 需含: veto_hit / fundamental_rank / fund_flow_rank /
    liquidity_rank / partial_composite / risk_tags / index_regime / 各原子因子
    """
    composite = _f(row.get("partial_composite"), 0.0)
    veto = bool(row.get("veto_hit"))
    drivers = _build_drivers(row)
    risks = _build_risks(row)

    # ---- Hard Veto ----
    if veto:
        return {
            "decision": "AVOID" if state in ("NONE", "WATCHING") else "STOP",
            "confidence": 0.0,
            "composite": composite,
            "drivers": drivers,
            "risks": ["veto_hit（基本面硬否决）"] + risks,
            "gates_failed": ["veto_hit"],
        }

    # ---- 状态机 ----
    if state in ("NONE", "WATCHING"):
        passed, fails = _gates_passed(row)
        if not passed:
            decision = "DROP" if state == "WATCHING" and composite < WATCH_THRESHOLD else "CONTINUE_WATCH"
            if state == "NONE":
                decision = "NO_ACTION"
            return {"decision": decision, "confidence": composite, "composite": composite,
                    "drivers": drivers, "risks": risks, "gates_failed": fails}
        if composite >= BUY_THRESHOLD:
            decision = "BUY"
        elif composite >= WATCH_THRESHOLD:
            decision = "CONTINUE_WATCH"
        else:
            decision = "NO_ACTION" if state == "NONE" else "CONTINUE_WATCH"
        return {"decision": decision, "confidence": composite, "composite": composite,
                "drivers": drivers, "risks": risks, "gates_failed": []}

    if state == "HELD":
        # 简化版（缺 entry_price/target_prices/ATR 等持仓元数据时退化为 HOLD/REDUCE 标签）
        rsi = _f(row.get("rsi_value"))
        liq_rank = _f(row.get("liquidity_rank"), 0.5)
        # 强卖信号：过热 + RSI>75
        if rsi is not None and rsi > 75 and "RISK_distribution" in str(row.get("risk_tags", "")):
            return {"decision": "TAKE", "confidence": composite, "composite": composite,
                    "drivers": drivers, "risks": risks + ["过热+派发风险"], "gates_failed": []}
        if rsi is not None and rsi > 75:
            return {"decision": "REDUCE", "confidence": composite, "composite": composite,
                    "drivers": drivers, "risks": risks + ["RSI 过热"], "gates_failed": []}
        # 弱化信号：composite 跌穿 0.40
        if composite < 0.40:
            return {"decision": "REDUCE", "confidence": composite, "composite": composite,
                    "drivers": drivers, "risks": risks + ["composite 走弱"], "gates_failed": []}
        # 加仓信号：基本面强 + 主力流入 + 量价配合
        if (composite > 0.65 and liq_rank < 0.30
                and _f(row.get("fund_flow_rank"), 1.0) < 0.30):
            return {"decision": "ADD", "confidence": composite, "composite": composite,
                    "drivers": drivers, "risks": risks, "gates_failed": []}
        return {"decision": "HOLD", "confidence": composite, "composite": composite,
                "drivers": drivers, "risks": risks, "gates_failed": []}

    if state == "EXITED":
        return {"decision": "NO_ACTION", "confidence": 0.0, "composite": composite,
                "drivers": [], "risks": ["cooldown 期"], "gates_failed": ["EXITED"]}

    return {"decision": "NO_ACTION", "confidence": composite, "composite": composite,
            "drivers": drivers, "risks": risks, "gates_failed": []}


def resolve_batch(df: pd.DataFrame, state_map: dict[str, str] | None = None) -> pd.DataFrame:
    """对整张 6 维 composite 表批量出决策."""
    state_map = state_map or {}
    out = []
    for _, row in df.iterrows():
        d = row.to_dict()
        st = state_map.get(d["symbol"], "NONE")
        res = resolve(d, state=st)
        d["state"] = st
        d.update({f"dec_{k}" if k in ("drivers", "risks", "gates_failed") else k: v
                  for k, v in res.items()})
        out.append(d)
    return pd.DataFrame(out)
