"""多源新闻流: 财联社 + 同花顺 + 新浪 → 累积到 cache.

每个源每次只能拉到最新 ~20 条, 必须按天累积, 才能形成 3-7 天滑窗.
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd

try:
    import akshare as ak
except ImportError:
    ak = None

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "news"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_CSV = CACHE_DIR / "history.csv"


def _normalize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """统一各源 schema → ts, title, content, source."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["ts", "title", "content", "source"])
    out = pd.DataFrame()
    if source == "cls":
        out["ts"] = pd.to_datetime(
            df["发布日期"].astype(str) + " " + df["发布时间"].astype(str),
            errors="coerce")
        out["title"] = df["标题"].fillna("").astype(str)
        out["content"] = df["内容"].fillna("").astype(str)
    elif source == "ths":
        out["ts"] = pd.to_datetime(df["发布时间"], errors="coerce")
        out["title"] = df["标题"].fillna("").astype(str)
        out["content"] = df["内容"].fillna("").astype(str)
    elif source == "sina":
        out["ts"] = pd.to_datetime(df["时间"], errors="coerce")
        out["title"] = df["内容"].astype(str).str.slice(0, 80)  # sina 没标题, 用内容前 80 字
        out["content"] = df["内容"].fillna("").astype(str)
    out["source"] = source
    return out.dropna(subset=["ts"])


def fetch_global_news() -> pd.DataFrame:
    """三源全局新闻汇总, 去重."""
    if ak is None:
        return pd.DataFrame()
    parts = []
    for src, fn_name in [("cls", "stock_info_global_cls"),
                         ("ths", "stock_info_global_ths"),
                         ("sina", "stock_info_global_sina")]:
        try:
            fn = getattr(ak, fn_name)
            d = _normalize(fn(), src)
            if not d.empty:
                parts.append(d)
        except Exception as e:
            print(f"  [news {src}] error: {e}")
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    # 去重: 同一标题或内容前 60 字
    df["dedup_key"] = df["title"].fillna("").str.slice(0, 60).where(
        df["title"].str.len() > 0, df["content"].str.slice(0, 60))
    df = df.drop_duplicates(subset=["dedup_key"], keep="first").drop(columns="dedup_key")
    return df.sort_values("ts").reset_index(drop=True)


def append_to_history(df: pd.DataFrame) -> int:
    """新增条目追加到 history.csv, 用 (ts, title 前 60) 去重."""
    if df.empty:
        return 0
    if HISTORY_CSV.exists():
        try:
            old = pd.read_csv(HISTORY_CSV, parse_dates=["ts"])
        except Exception:
            old = pd.DataFrame(columns=df.columns)
    else:
        old = pd.DataFrame(columns=df.columns)
    combined = pd.concat([old, df], ignore_index=True)
    # 统一 ts 为 ISO 字符串 (避免 CSV 重读后 str() 格式不一致导致重复)
    combined["ts"] = pd.to_datetime(combined["ts"], errors="coerce")
    combined["dedup_key"] = (combined["ts"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
                             + "|" + combined["title"].fillna("").astype(str).str.slice(0, 60))
    combined = combined.drop_duplicates(subset=["dedup_key"], keep="first").drop(columns="dedup_key")
    # 仅保留近 30 天, 控制文件大小
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=30)
    combined = combined[combined["ts"] >= cutoff].sort_values("ts").reset_index(drop=True)
    new_count = len(combined) - len(old)
    combined.to_csv(HISTORY_CSV, index=False)
    return new_count


def get_news_recent(days: int = 3, refresh: bool = True) -> pd.DataFrame:
    """读取近 N 天新闻流. refresh=True 时先拉一次最新追加进来."""
    if refresh:
        fresh = fetch_global_news()
        added = append_to_history(fresh)
        print(f"  [news] 抓取 {len(fresh)} 条 → 新增 {added} 条 (去重后)")
    if not HISTORY_CSV.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(HISTORY_CSV, parse_dates=["ts"])
    except Exception:
        return pd.DataFrame()
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=days)
    return df[df["ts"] >= cutoff].reset_index(drop=True)
