"""基本面维度合成: Hard Veto + 6 软因子 → fundamental_rank."""
from __future__ import annotations
import pandas as pd

from engine.ranker import cross_section_rank, percentile, zscore
from .data import (
    get_indicator_lg,
    get_financial_abstract,
    get_individual_info,
    parse_abstract_to_metrics,
)


VETO_KEYS = [
    "veto_loss_2y",
    "veto_high_goodwill",
    "veto_high_leverage",
    "veto_st",
    "veto_audit_concern",
]


def compute_one(symbol: str, name: str = "") -> dict:
    """单只股票的基本面原子因子."""
    info = get_individual_info(symbol)
    abstract = get_financial_abstract(symbol)
    metrics = parse_abstract_to_metrics(abstract)
    indicator = get_indicator_lg(symbol)

    pe_ttm = pb = None
    pe_history = pb_history = pd.Series(dtype=float)
    if not indicator.empty:
        latest = indicator.iloc[-1]
        pe_ttm = float(latest.get("pe_ttm")) if pd.notna(latest.get("pe_ttm")) else None
        pb = float(latest.get("pb")) if pd.notna(latest.get("pb")) else None
        # 5年自身分位：取近 1250 个交易日
        recent5y = indicator.tail(1250)
        pe_history = recent5y["pe_ttm"] if "pe_ttm" in recent5y.columns else pd.Series(dtype=float)
        pb_history = recent5y["pb"] if "pb" in recent5y.columns else pd.Series(dtype=float)

    sector = str(info.get("行业", "")) if info else ""

    # ---- Hard Veto ----
    is_st = ("ST" in name.upper()) or ("ST" in str(info.get("股票简称", "")).upper())
    net_profit = metrics.get("net_profit")
    net_profit_prev = metrics.get("net_profit_yoy_prev")
    veto_loss_2y = (
        net_profit is not None and net_profit < 0
        and net_profit_prev is not None and net_profit_prev < 0
    )
    goodwill = metrics.get("goodwill")
    total_assets = metrics.get("total_assets")
    veto_high_goodwill = (
        goodwill is not None and total_assets is not None and total_assets > 0
        and (goodwill / total_assets) > 0.5
    )
    debt_ratio = metrics.get("debt_ratio")
    # 个股 info 接口不可用时 sector="", 退化用名称关键字判定金融
    finance_kw = ["银行", "证券", "保险", "金融", "信托", "财险", "人寿", "平安"]
    is_finance = any(kw in sector for kw in finance_kw) or any(kw in name for kw in finance_kw)
    veto_high_leverage = (
        not is_finance and debt_ratio is not None and debt_ratio > 80
    )
    vetos = {
        "veto_loss_2y": bool(veto_loss_2y),
        "veto_high_goodwill": bool(veto_high_goodwill),
        "veto_high_leverage": bool(veto_high_leverage),
        "veto_st": bool(is_st),
        "veto_audit_concern": False,  # 接口暂缺，留 False
    }
    veto_hit = any(vetos.values())

    # ---- 估值陷阱标签 ----
    risk_tags = []
    eps_growth = None
    if metrics.get("eps_basic") and metrics.get("eps_yoy_prev"):
        prev = metrics["eps_yoy_prev"]
        if abs(prev) > 1e-6:
            eps_growth = (metrics["eps_basic"] - prev) / abs(prev)
    roe_growth = None
    if metrics.get("roe_ttm") is not None and metrics.get("roe_yoy_prev") is not None:
        prev = metrics["roe_yoy_prev"]
        if abs(prev) > 1e-6:
            roe_growth = (metrics["roe_ttm"] - prev) / abs(prev)

    # ---- 软因子 ----
    pe_self_pct = percentile(pe_ttm, pe_history) if pe_ttm and pe_ttm > 0 else (1.0 if pe_ttm and pe_ttm <= 0 else None)
    pb_self_pct = percentile(pb, pb_history) if pb and pb > 0 else None

    return {
        "symbol": symbol,
        "name": name,
        "sector": sector,
        "pe_ttm": pe_ttm,
        "pb": pb,
        "pe_self_percentile": pe_self_pct,
        "pb_self_percentile": pb_self_pct,
        "roe_ttm": metrics.get("roe_ttm"),
        "roe_growth_yoy": roe_growth,
        "eps_growth_yoy": eps_growth,
        "net_profit": metrics.get("net_profit"),
        "debt_ratio": debt_ratio,
        "goodwill_ratio": (goodwill / total_assets) if (goodwill and total_assets) else None,
        **vetos,
        "veto_hit": veto_hit,
        "risk_tags": risk_tags,
    }


def compose_dim(rows: list[dict]) -> pd.DataFrame:
    """将多只股票的原子因子合成 fundamental_raw + fundamental_rank.

    规则:
    - veto_hit=True 的股票 fundamental_rank = NaN（让上层决策直接 AVOID）
    - 行业横向 PE 分位（pe_sector_percentile）：在同 sector 内做横截面分位
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # pe_sector_percentile：按 sector 分组，在该 sector 内做 0~1 分位
    # sector 全为空（个股 info 接口不可用）时，退化为全局 PE 分位
    pe_pos = df["pe_ttm"].where(df["pe_ttm"] > 0)
    if df["sector"].astype(str).str.strip().eq("").all():
        df["pe_sector_percentile"] = pe_pos.rank(pct=True)
    else:
        df["pe_sector_percentile"] = pe_pos.groupby(df["sector"]).rank(pct=True)

    # 估值陷阱标签
    def _risk(row):
        tags = list(row.get("risk_tags") or [])
        if (row.get("pe_sector_percentile") is not None and row.get("pe_sector_percentile") < 0.20
                and row.get("roe_growth_yoy") is not None and row.get("roe_growth_yoy") < -0.10):
            tags.append("RISK_value_trap")
        if (row.get("roe_ttm") is not None and row.get("roe_ttm") > 20
                and row.get("pe_self_percentile") is not None and row.get("pe_self_percentile") > 0.80):
            tags.append("RISK_priced_in")
        return tags

    df["risk_tags"] = df.apply(_risk, axis=1)

    # 软合成：清洗后加权
    valid = df[~df["veto_hit"]].copy()

    def _safe(s):
        return s.fillna(s.median()) if s.notna().any() else pd.Series(0.0, index=s.index)

    pe_self = _safe(valid["pe_self_percentile"])
    pe_sec = _safe(valid["pe_sector_percentile"])
    pb_self = _safe(valid["pb_self_percentile"])
    roe = _safe(valid["roe_ttm"])
    roe_g = _safe(valid["roe_growth_yoy"]).clip(-1, 1)
    eps_g = _safe(valid["eps_growth_yoy"]).clip(-1, 1)

    raw = (
        0.30 * (1 - pe_self)
        + 0.20 * (1 - pe_sec)
        + 0.10 * (1 - pb_self)
        + 0.20 * zscore(roe)
        + 0.10 * roe_g
        + 0.10 * eps_g
    )
    valid["fundamental_raw"] = raw
    valid["fundamental_rank"] = cross_section_rank(raw)

    df = df.merge(
        valid[["symbol", "fundamental_raw", "fundamental_rank"]], on="symbol", how="left"
    )
    return df
