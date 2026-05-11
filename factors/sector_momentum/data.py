"""申万二级板块数据 (sina/akshare 源, 避开被限流的东财 em).

接口:
- ak.sw_index_second_info()         : 131 个 SW2 板块清单 (代码/名称/成份数)
- ak.index_hist_sw(symbol)          : 板块日 K (代码 6 位, 去掉 .SI)
- ak.index_component_sw(symbol)     : 板块成份 (symbol → board_code 反查表)
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd

try:
    import akshare as ak
except ImportError:
    ak = None

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "sector_momentum"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _is_fresh(p: Path, hours: float) -> bool:
    if not p.exists():
        return False
    return (time.time() - p.stat().st_mtime) / 3600 < hours


def get_sw2_list(ttl_hours: float = 24 * 7) -> pd.DataFrame:
    """申万二级板块清单. 列: code, name, cons_count."""
    p = CACHE_DIR / "sw2_list.csv"
    if _is_fresh(p, ttl_hours):
        try:
            return pd.read_csv(p, dtype={"code": str})
        except Exception:
            pass
    if ak is None:
        return pd.DataFrame()
    try:
        df = ak.sw_index_second_info()
        if df is None or df.empty:
            return pd.DataFrame()
        out = pd.DataFrame({
            "code": df["行业代码"].astype(str).str.replace(".SI", "", regex=False),
            "name": df["行业名称"].astype(str),
            "cons_count": pd.to_numeric(df["成份个数"], errors="coerce").fillna(0).astype(int),
        })
        out.to_csv(p, index=False)
        return out
    except Exception as e:
        print(f"  [sw2_list] error: {e}")
        return pd.DataFrame()


def get_sw2_kline(code: str, ttl_hours: float = 6) -> pd.DataFrame:
    """单个 SW2 板块日 K. 列: date, close, open, high, low, vol, amount."""
    p = CACHE_DIR / f"kline_{code}.csv"
    if _is_fresh(p, ttl_hours):
        try:
            return pd.read_csv(p, parse_dates=["date"])
        except Exception:
            pass
    if ak is None:
        return pd.DataFrame()
    try:
        df = ak.index_hist_sw(symbol=code, period="day")
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={
            "日期": "date", "收盘": "close", "开盘": "open",
            "最高": "high", "最低": "low", "成交量": "vol", "成交额": "amount",
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").tail(120).reset_index(drop=True)
        df.to_csv(p, index=False)
        return df
    except Exception as e:
        print(f"  [sw2_kline {code}] error: {e}")
        return pd.DataFrame()


def get_symbol_to_sw2(ttl_hours: float = 24 * 7) -> dict[str, tuple[str, str]]:
    """构建 {symbol: (sw2_code, sw2_name)}.

    遍历所有 SW2 板块的 cons 接口构建反查. 一次调用 ~131 次, 缓存 7 天.
    """
    p = CACHE_DIR / "symbol_to_sw2.csv"
    if _is_fresh(p, ttl_hours):
        try:
            df = pd.read_csv(p, dtype={"symbol": str, "sw2_code": str})
            return {r["symbol"]: (r["sw2_code"], r["sw2_name"]) for _, r in df.iterrows()}
        except Exception:
            pass
    if ak is None:
        return {}
    sw2_list = get_sw2_list()
    if sw2_list.empty:
        return {}
    rows = []
    for _, r in sw2_list.iterrows():
        code, name = r["code"], r["name"]
        try:
            cons = ak.index_component_sw(symbol=code)
            if cons is None or cons.empty:
                continue
            for sym in cons["证券代码"].astype(str).str.zfill(6):
                rows.append({"symbol": sym, "sw2_code": code, "sw2_name": name})
        except Exception as e:
            print(f"  [sw2_cons {code}] error: {e}")
        time.sleep(0.05)
    if not rows:
        return {}
    out = pd.DataFrame(rows).drop_duplicates(subset="symbol", keep="first")
    out.to_csv(p, index=False)
    print(f"  [symbol_to_sw2] built: {len(out)} symbols → {out['sw2_code'].nunique()} boards")
    return {r["symbol"]: (r["sw2_code"], r["sw2_name"]) for _, r in out.iterrows()}
