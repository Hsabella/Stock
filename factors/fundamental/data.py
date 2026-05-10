"""基本面数据拉取 + 缓存（akshare 免费接口）.

数据源原则: 默认禁用东财 em 接口（项目级原则，见 .omc/wiki/data.md）.

接口:
- ak.stock_zh_valuation_baidu(symbol, indicator) : PE_TTM / PB 时序（百度估值）★ 主源
- ak.stock_financial_abstract(symbol)            : 财务摘要（多源混合，可用）
- ak.stock_individual_info_em(symbol)            : 行业归属（em，已确认限流；保留为可选探测，
                                                   失败即返回空 dict，由 dim.py 用名称关键字推断金融行业）
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd

try:
    import akshare as ak
except ImportError:
    ak = None

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "fundamental"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_HOURS = 12  # 基本面变化慢，半天缓存够用


def _is_fresh(p: Path, ttl_hours: int = CACHE_TTL_HOURS) -> bool:
    if not p.exists():
        return False
    age_h = (time.time() - p.stat().st_mtime) / 3600
    return age_h < ttl_hours


def get_indicator_lg(symbol: str) -> pd.DataFrame:
    """日级 PE_TTM / PB 时序（用百度估值接口，覆盖率比理杏仁稳定）.

    返回列: trade_date, pe_ttm, pb
    """
    p = CACHE_DIR / f"{symbol}_indicator_lg.csv"
    if _is_fresh(p):
        try:
            return pd.read_csv(p, parse_dates=["trade_date"])
        except Exception:
            pass
    if ak is None:
        return pd.DataFrame()
    pe_df = pb_df = None
    try:
        pe_df = ak.stock_zh_valuation_baidu(symbol=symbol, indicator="市盈率(TTM)", period="全部")
    except Exception as e:
        print(f"  [valuation_baidu PE] {symbol} error: {e}")
    try:
        pb_df = ak.stock_zh_valuation_baidu(symbol=symbol, indicator="市净率", period="全部")
    except Exception as e:
        print(f"  [valuation_baidu PB] {symbol} error: {e}")
    parts = []
    if pe_df is not None and not pe_df.empty:
        pe_df = pe_df.rename(columns={"date": "trade_date", "value": "pe_ttm"})
        pe_df["trade_date"] = pd.to_datetime(pe_df["trade_date"])
        parts.append(pe_df.set_index("trade_date")["pe_ttm"])
    if pb_df is not None and not pb_df.empty:
        pb_df = pb_df.rename(columns={"date": "trade_date", "value": "pb"})
        pb_df["trade_date"] = pd.to_datetime(pb_df["trade_date"])
        parts.append(pb_df.set_index("trade_date")["pb"])
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=1).reset_index().sort_values("trade_date")
    out.to_csv(p, index=False)
    return out


def get_financial_abstract(symbol: str) -> pd.DataFrame:
    """财务摘要（ROE/EPS/净利润/资产负债率/商誉等）."""
    p = CACHE_DIR / f"{symbol}_financial_abstract.csv"
    if _is_fresh(p, ttl_hours=24 * 7):  # 季报频率，缓存 7 天
        try:
            return pd.read_csv(p)
        except Exception:
            pass
    if ak is None:
        return pd.DataFrame()
    try:
        df = ak.stock_financial_abstract(symbol=symbol)
        if df is None or df.empty:
            return pd.DataFrame()
        df.to_csv(p, index=False)
        return df
    except Exception as e:
        print(f"  [financial_abstract] {symbol} error: {e}")
        return pd.DataFrame()


def get_individual_info(symbol: str) -> dict:
    """个股信息（行业、流通股本）."""
    p = CACHE_DIR / f"{symbol}_info.csv"
    if _is_fresh(p, ttl_hours=24 * 30):
        try:
            df = pd.read_csv(p)
            return dict(zip(df["item"], df["value"]))
        except Exception:
            pass
    if ak is None:
        return {}
    # em 接口已知限流；只尝试一次，失败即放弃，由 dim.py 名称关键字推断金融
    try:
        df = ak.stock_individual_info_em(symbol=symbol)
        if df is None or df.empty:
            return {}
        df.to_csv(p, index=False)
        return dict(zip(df["item"], df["value"]))
    except Exception:
        return {}


def parse_abstract_to_metrics(abstract_df: pd.DataFrame) -> dict:
    """将 stock_financial_abstract 的长格式解析为 latest 指标 dict.

    财务摘要 DataFrame: 行=指标名（如"净资产收益率"）, 列=报告期(20240331/20231231 等).
    我们取最近一个报告期，并为同比指标取去年同期。
    """
    out: dict = {}
    if abstract_df is None or abstract_df.empty:
        return out
    df = abstract_df.copy()
    # 兼容列结构：第一列为指标名，其余为报告期
    if "指标" in df.columns:
        idx_col = "指标"
    elif "选项" in df.columns:
        idx_col = "选项"
    else:
        idx_col = df.columns[0]
    df = df.set_index(idx_col)
    period_cols = [c for c in df.columns if str(c).strip().isdigit() and len(str(c).strip()) == 8]
    period_cols = sorted(period_cols, reverse=True)  # 最新在前
    if not period_cols:
        return out
    latest = period_cols[0]
    yoy = period_cols[4] if len(period_cols) > 4 else None  # 4 个季度前 = 去年同期

    def _g(name_keywords: list[str], period: str) -> float | None:
        for kw in name_keywords:
            for idx in df.index:
                if isinstance(idx, str) and kw in idx:
                    try:
                        v = df.loc[idx, period]
                        if isinstance(v, pd.Series):
                            v = v.iloc[0]
                        return float(v) if pd.notna(v) else None
                    except (ValueError, TypeError, KeyError):
                        continue
        return None

    # 用更精确的关键字防误匹配（"总资产周转率/总资产报酬率" 等）
    out["roe_ttm"] = _g(["净资产收益率(ROE)", "摊薄净资产收益率"], latest)
    out["eps_basic"] = _g(["基本每股收益"], latest)
    out["net_profit"] = _g(["归母净利润", "归属于母公司股东的净利润"], latest)
    out["debt_ratio"] = _g(["资产负债率"], latest)
    out["goodwill"] = _g(["商誉"], latest)
    # stock_financial_abstract 不直接给"总资产"绝对值，goodwill/总资产 比例此处置 None
    out["total_assets"] = None

    if yoy:
        out["roe_yoy_prev"] = _g(["净资产收益率(ROE)", "摊薄净资产收益率"], yoy)
        out["eps_yoy_prev"] = _g(["基本每股收益"], yoy)
        out["net_profit_yoy_prev"] = _g(["归母净利润", "归属于母公司股东的净利润"], yoy)

    out["latest_period"] = latest
    return out
