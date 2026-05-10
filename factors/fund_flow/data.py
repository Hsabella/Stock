"""主力资金面数据拉取 + 缓存（akshare 免费接口）.

数据源原则: 默认禁用东财 em 接口（项目级原则，见 .omc/wiki/data.md）.

接口:
- ak.stock_fund_flow_individual(symbol)        : 同花顺主力资金流（"即时"/"3日排行"/"5日排行"）★ 主力流向源
- ak.stock_hsgt_individual_em(stock)           : 北向持股快照（em 但实测可用，白名单）
- ak.stock_lhb_detail_em                       : 龙虎榜明细（em 但实测可用，白名单）
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd

try:
    import akshare as ak
except ImportError:
    ak = None

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "fund_flow"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_HOURS = 6


def _is_fresh(p: Path, ttl_hours: int = CACHE_TTL_HOURS) -> bool:
    if not p.exists():
        return False
    return (time.time() - p.stat().st_mtime) / 3600 < ttl_hours


def _market_of(symbol: str) -> str:
    return "sh" if symbol.startswith("6") else "sz"


def _parse_cn_money(s) -> float:
    """解析中文金额字符串 -> 元. 支持 '1.78亿' / '2491.80万' / '-8094.95万' / 数字."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return float("nan")
    if isinstance(s, (int, float)):
        return float(s)
    txt = str(s).strip().replace(",", "")
    if txt in ("", "--", "-"):
        return float("nan")
    sign = -1 if txt.startswith("-") else 1
    txt = txt.lstrip("+-")
    try:
        if txt.endswith("亿"):
            return sign * float(txt[:-1]) * 1e8
        if txt.endswith("万"):
            return sign * float(txt[:-1]) * 1e4
        if txt.endswith("%"):
            return sign * float(txt[:-1])
        return sign * float(txt)
    except ValueError:
        return float("nan")


def get_fund_flow_rank(indicator: str = "3日排行") -> pd.DataFrame:
    """全市场个股资金流快照（同花顺源，bulk）.

    indicator ∈ {"即时", "3日排行", "5日排行", "10日排行", "20日排行"}.
    缓存 6 小时，所有股票共享一次拉取。

    返回标准化列：代码 / 名称 / 资金流入净额(元) / 阶段涨跌幅 / 连续换手率 / 最新价
    """
    safe = indicator.replace("排行", "").replace("日", "d")
    p = CACHE_DIR / f"fund_flow_rank_ths_{safe}.csv"
    if _is_fresh(p, ttl_hours=6):
        try:
            df = pd.read_csv(p)
            df["代码"] = df["代码"].astype(str).str.zfill(6)
            return df
        except Exception:
            pass
    if ak is None:
        return pd.DataFrame()
    last_err = None
    for attempt in range(3):
        try:
            df = ak.stock_fund_flow_individual(symbol=indicator)
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.rename(columns={"股票代码": "代码", "股票简称": "名称"})
            df["代码"] = df["代码"].astype(str).str.zfill(6)
            # 标准化金额列：3日/5日/10日/20日 用 资金流入净额; 即时 用 净额
            amt_col = "资金流入净额" if "资金流入净额" in df.columns else "净额"
            if amt_col in df.columns:
                df["资金流入净额_元"] = df[amt_col].map(_parse_cn_money)
            for col in ("阶段涨跌幅", "涨跌幅", "连续换手率", "换手率"):
                if col in df.columns:
                    df[col] = df[col].map(_parse_cn_money)
            df.to_csv(p, index=False)
            return df
        except Exception as e:
            last_err = e
            time.sleep(2 + attempt * 2)
    print(f"  [fund_flow_rank {indicator}] error: {last_err}")
    return pd.DataFrame()


def get_north_holding(symbol: str) -> pd.DataFrame:
    """北向持股快照（最新一条 + 1日/5日/10日 持股市值变化列）.

    使用 ak.stock_hsgt_individual_em(stock=...) — 单只股票，含最新持股比例和滚动变化。
    返回单行 DataFrame, 至少含 持股日期 / 持股数量占A股百分比 / 持股市值变化-5日 等。
    """
    p = CACHE_DIR / f"{symbol}_north.csv"
    if _is_fresh(p, ttl_hours=12):
        try:
            return pd.read_csv(p)
        except Exception:
            pass
    if ak is None:
        return pd.DataFrame()
    last_err = None
    for attempt in range(3):
        try:
            df = ak.stock_hsgt_individual_em(stock=symbol)
            if df is None or df.empty:
                return pd.DataFrame()
            df.to_csv(p, index=False)
            return df
        except Exception as e:
            last_err = e
            time.sleep(1 + attempt)
    print(f"  [north_holding] {symbol} error: {last_err}")
    return pd.DataFrame()


def get_lhb_recent(days: int = 30) -> pd.DataFrame:
    """近 N 天龙虎榜明细（全市场）— 做一份缓存供所有股票查询."""
    p = CACHE_DIR / f"lhb_last_{days}d.csv"
    if _is_fresh(p, ttl_hours=12):
        try:
            return pd.read_csv(p)
        except Exception:
            pass
    if ak is None:
        return pd.DataFrame()
    try:
        end = pd.Timestamp.today()
        start = end - pd.Timedelta(days=days)
        for fn_name in ["stock_lhb_detail_em", "stock_lhb_stock_statistic_em"]:
            fn = getattr(ak, fn_name, None)
            if fn is None:
                continue
            try:
                df = fn(start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d")) \
                    if fn_name == "stock_lhb_detail_em" else fn(symbol="近一月")
                if df is None or df.empty:
                    continue
                df.to_csv(p, index=False)
                return df
            except Exception:
                continue
        return pd.DataFrame()
    except Exception as e:
        print(f"  [lhb] error: {e}")
        return pd.DataFrame()
