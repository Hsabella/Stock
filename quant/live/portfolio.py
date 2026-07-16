"""持仓健康度（晨报新段的数据层）：止损线 + 引擎排雷信号。

只读三个数据源，全部 REPO 锚定路径、与 cwd 无关：
- watchlist.yaml：HELD 持仓 + account / target_structure / hard_stop_pct 配置
- cache/kline/<symbol>_daily_qfq.csv：旧引擎 18:30 刷新的共享 K 线缓存；
  末日落后超 STALE_TRADING_DAYS 个交易日 → akshare 现拉降级并标 ⚠️
- results/decisions/partial_<date>.csv：旧引擎最新决策（排雷信号按 symbol 匹配，
  CSV 的 state 列可能滞后于 watchlist，不采信）

不 import 旧引擎模块（factors/data 链上有 .venv-quant 缺失的依赖），
_atr 复制自 factors/technical/dim.py:54。
"""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
WATCHLIST = REPO / "watchlist.yaml"
KLINE_DIR = REPO / "cache" / "kline"
DEC_DIR = REPO / "results" / "decisions"
STALE_TRADING_DAYS = 3
ATR_N = 14
TRAIL_WINDOW = 20  # ATR 追踪线的滚动最高收盘窗口


def load_portfolio_config(path: Path = WATCHLIST) -> dict:
    """watchlist.yaml → {held, account, target_structure, hard_stop_pct, atr_k}。"""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    held = [it for it in cfg.get("watchlist", []) if it.get("state") == "HELD"]
    strategy = str(cfg.get("default_stop_strategy", "atr_2"))
    atr_k = float(strategy.split("_", 1)[1]) if strategy.startswith("atr_") else 2.0
    return {
        "held": held,
        "account": cfg.get("account", {}),
        "target_structure": cfg.get("target_structure", {}),
        "hard_stop_pct": float(cfg.get("hard_stop_pct", 0.08)),
        "atr_k": atr_k,
    }


def _atr(high, low, close, n=ATR_N):
    # 复制自 factors/technical/dim.py:54（避免跨 venv import 旧引擎依赖链）
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def load_kline(symbol: str, today: pd.Timestamp) -> tuple[pd.DataFrame, bool]:
    """(日K df[date/open/high/low/close], 是否陈旧)。缓存陈旧时降级 akshare 现拉。"""
    p = KLINE_DIR / f"{symbol}_daily_qfq.csv"
    df = pd.DataFrame()
    if p.exists():
        try:
            df = pd.read_csv(p, parse_dates=["date"])
        except Exception:
            df = pd.DataFrame()
    stale = df.empty or len(pd.bdate_range(df["date"].iloc[-1], today)) - 1 > STALE_TRADING_DAYS
    if stale:
        try:
            import akshare as ak

            prefix = "sh" if symbol.startswith("6") else "sz"
            fresh = ak.stock_zh_a_daily(symbol=f"{prefix}{symbol}", adjust="qfq",
                                        start_date=(today - pd.Timedelta(days=120)).strftime("%Y%m%d"),
                                        end_date=today.strftime("%Y%m%d"))
            fresh["date"] = pd.to_datetime(fresh["date"])
            df = fresh.sort_values("date").reset_index(drop=True)
        except Exception:
            pass  # 现拉也失败 → 返回旧数据（stale=True 由调用方标 ⚠️）
    return df, stale


def stop_lines(kline: pd.DataFrame, entry_price: float, hard_stop_pct: float,
               atr_k: float) -> dict:
    """两条止损线：硬止损（成本-N%）+ ATR 追踪线（近20日最高收盘 - k×ATR14）。"""
    close = kline["close"].astype(float)
    hard = entry_price * (1 - hard_stop_pct)
    atr = _atr(kline["high"].astype(float), kline["low"].astype(float), close)
    trail = float(close.tail(TRAIL_WINDOW).max() - atr_k * atr.iloc[-1])
    line = max(hard, trail)  # 两线取严者判"已破"
    last = float(close.iloc[-1])
    return {
        "hard_stop": hard, "atr_stop": trail, "stop_line": line,
        "broken": last < line,
        "dist_pct": last / line - 1,  # 现价距止损线（负=已破）
    }


def latest_engine_signals() -> tuple[dict, str]:
    """最新 partial CSV → {symbol: {decision, risks}}，附 CSV 日期。按 symbol 匹配。"""
    dates = sorted(p.stem.split("_")[1] for p in DEC_DIR.glob("partial_*.csv"))
    if not dates:
        return {}, ""
    date = dates[-1]
    df = pd.read_csv(DEC_DIR / f"partial_{date}.csv", dtype={"symbol": str})
    out = {}
    for _, r in df.iterrows():
        try:
            risks = ast.literal_eval(r.get("dec_risks", "[]") or "[]")
        except (ValueError, SyntaxError):
            risks = []
        out[str(r["symbol"]).zfill(6)] = {
            "decision": str(r.get("decision", "")),
            "risks": [str(x) for x in risks][:2],
        }
    return out, date


def holdings_health(cfg: dict | None = None, today: pd.Timestamp | None = None) -> dict:
    """晨报【持仓健康度】数据：每只 HELD 的现价/浮亏/止损线/引擎信号。"""
    cfg = cfg or load_portfolio_config()
    today = today or pd.Timestamp.today().normalize()
    signals, sig_date = latest_engine_signals()
    rows, any_stale = [], False
    for it in cfg["held"]:
        sym = str(it["symbol"]).zfill(6)
        kline, stale = load_kline(sym, today)
        if kline.empty or len(kline) < ATR_N + 1:
            rows.append({"symbol": sym, "name": it.get("name", sym), "error": "无K线数据"})
            continue
        any_stale |= stale
        entry = float(it["entry_price"])
        last = float(kline["close"].iloc[-1])
        sig = signals.get(sym, {})
        rows.append({
            "symbol": sym, "name": it.get("name", sym),
            "position": float(it["position"]),
            "close": last, "close_date": str(kline["date"].iloc[-1].date()),
            "stale": stale,
            "entry": entry, "pnl_pct": last / entry - 1,
            "value": last * float(it["position"]),
            **stop_lines(kline, entry, cfg["hard_stop_pct"], cfg["atr_k"]),
            "decision": sig.get("decision", ""),
            "risks": sig.get("risks", []),
        })
    return {"rows": rows, "signal_date": sig_date, "any_stale": any_stale}
