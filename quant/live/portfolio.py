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
    """watchlist.yaml → {held, exited, account, target_structure, hard_stop_pct, atr_k, cooldown_days}。

    exited 只收带 exit_date 的 EXITED 条目（回补观察跟踪对象）；
    不带 exit_date 的视为历史清仓，不跟踪。
    """
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    items = cfg.get("watchlist", [])
    held = [it for it in items if it.get("state") == "HELD"]
    exited = [it for it in items if it.get("state") == "EXITED" and it.get("exit_date")]
    strategy = str(cfg.get("default_stop_strategy", "atr_2"))
    atr_k = float(strategy.split("_", 1)[1]) if strategy.startswith("atr_") else 2.0
    return {
        "held": held,
        "exited": exited,
        "account": cfg.get("account", {}),
        "target_structure": cfg.get("target_structure", {}),
        "hard_stop_pct": float(cfg.get("hard_stop_pct", 0.08)),
        "atr_k": atr_k,
        "cooldown_days": int(cfg.get("cooldown_days", 10)),
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

            # 400 日历日 ≈ 270 交易日：抄底灯的 dd120+信号窗要 140 根，无缓存也够
            prefix = "sh" if symbol.startswith("6") else "sz"
            fresh = ak.stock_zh_a_daily(symbol=f"{prefix}{symbol}", adjust="qfq",
                                        start_date=(today - pd.Timedelta(days=400)).strftime("%Y%m%d"),
                                        end_date=today.strftime("%Y%m%d"))
            fresh["date"] = pd.to_datetime(fresh["date"])
            fresh = fresh.sort_values("date")
            if not df.empty:  # 与旧缓存拼接保长历史（除权日复权口径或有微差，以新段为准）
                fresh = pd.concat([df[df["date"] < fresh["date"].iloc[0]], fresh])
            df = fresh.reset_index(drop=True)
        except Exception:
            pass  # 现拉也失败 → 返回旧数据（stale=True 由调用方标 ⚠️）
    return df, stale


def trail_line(kline: pd.DataFrame, atr_k: float) -> float:
    """ATR 追踪线 = 近20日最高收盘 - k×ATR14（止损与回补观察共用同一条趋势线）。"""
    close = kline["close"].astype(float)
    atr = _atr(kline["high"].astype(float), kline["low"].astype(float), close)
    return float(close.tail(TRAIL_WINDOW).max() - atr_k * atr.iloc[-1])


def stop_lines(kline: pd.DataFrame, entry_price: float, hard_stop_pct: float,
               atr_k: float) -> dict:
    """两条止损线：硬止损（成本-N%）+ ATR 追踪线（近20日最高收盘 - k×ATR14）。"""
    close = kline["close"].astype(float)
    hard = entry_price * (1 - hard_stop_pct)
    trail = trail_line(kline, atr_k)
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


def structure_drift(cfg: dict | None = None, health: dict | None = None) -> dict:
    """晨报【结构修正跟踪】数据：当前 vs 目标结构的偏离与待执行动作（规则生成）。

    ETF 按成本估值（宽基波动对 10 个点的结构带宽影响可忽略，避免再引入行情依赖）。
    """
    cfg = cfg or load_portfolio_config()
    health = health or holdings_health(cfg)
    target = cfg["target_structure"]
    lo, hi = target.get("core_etf_pct", [0.40, 0.50])
    max_count = int(target.get("satellite_max_count", 3))
    max_pct = float(target.get("satellite_max_pct", 0.10))

    rows = [r for r in health["rows"] if not r.get("error")]
    stock_value = sum(r["value"] for r in rows)
    etf_value = sum(float(e["position"]) * float(e["entry_price"])
                    for e in cfg["account"].get("etf", []))
    cash = float(cfg["account"].get("cash", 0))
    total = cash + stock_value + etf_value
    if total <= 0:
        return {"error": "总资产为 0（检查 account.cash 与持仓）"}

    etf_pct, stock_pct = etf_value / total, stock_value / total
    over_weight = [r for r in rows if r["value"] / total > max_pct]
    count_over = len(rows) - max_count

    # 减仓候选按严重度排序：引擎 DROP/STOP > REDUCE > 已破止损
    def severity(r):
        return (0 if r["decision"] in ("DROP", "STOP") else
                1 if r["decision"] == "REDUCE" else
                2 if r["broken"] else 3)

    candidates = sorted(rows, key=severity)
    actions = []
    if count_over > 0:
        def label(r):
            if r["decision"] in ("DROP", "STOP", "REDUCE", "TAKE"):
                return r["decision"]
            return "已破止损" if r["broken"] else "弱势"

        names = "·".join(f"{r['name']}({label(r)})" for r in candidates[:count_over + 1])
        actions.append(f"卫星个股 {len(rows)} 只超上限 {max_count} 只，优先处置: {names}")
    for r in over_weight:
        actions.append(f"{r['name']} 占总资产 {r['value'] / total:.0%} 超单票上限 {max_pct:.0%}，"
                       f"至少减 {r['value'] - total * max_pct:,.0f} 元")
    if etf_pct < lo:
        actions.append(f"核心宽基ETF 建仓至 ≥{lo:.0%}（缺口约 {total * lo - etf_value:,.0f} 元，分批）")
    return {
        "total": total, "cash": cash, "cash_pct": cash / total,
        "stock_value": stock_value, "stock_pct": stock_pct,
        "etf_value": etf_value, "etf_pct": etf_pct,
        "stock_count": len(rows),
        "target": {"core_etf_lo": lo, "core_etf_hi": hi,
                   "max_count": max_count, "max_pct": max_pct},
        "actions": actions,
        "compliant": not actions,
    }


NEG_DECISIONS = ("DROP", "STOP", "REDUCE")


def reentry_watch(cfg: dict | None = None, today: pd.Timestamp | None = None,
                  lights: list[dict] | None = None) -> dict:
    """晨报【回补观察】数据：止损卖出票（EXITED+exit_date）的接回三条件 + 冷静期。

    三条件：①对应结构灯转绿（688xxx→科创50，其余→中证1000；lights 缺失时未知）
    ②收盘价站回 ATR 追踪线（让股价自证跌完，而不是猜底）
    ③引擎最新信号非 DROP/STOP/REDUCE 且风险项无"板块…弱势"（匹配自家引擎的固定文案）。
    ready = 三条件全为 True 且冷静期已满；未知(None)按未满足处理——宁可晚接，不抢跑。
    冷静期按自然工作日近似交易日（与 load_kline 的陈旧判定同一口径）。
    """
    cfg = cfg or load_portfolio_config()
    today = today or pd.Timestamp.today().normalize()
    signals, sig_date = latest_engine_signals()
    light_by_symbol = {lt["symbol"]: lt for lt in (lights or [])}
    rows, any_stale = [], False
    for it in cfg["exited"]:
        sym = str(it["symbol"]).zfill(6)
        base = {"symbol": sym, "name": it.get("name", sym),
                "exit_date": str(it["exit_date"]), "exit_price": it.get("exit_price")}
        kline, stale = load_kline(sym, today)
        if kline.empty or len(kline) < ATR_N + 1:
            rows.append({**base, "error": "无K线数据"})
            continue
        any_stale |= stale
        elapsed = max(0, len(pd.bdate_range(pd.Timestamp(it["exit_date"]), today)) - 1)
        last = float(kline["close"].astype(float).iloc[-1])
        trail = trail_line(kline, cfg["atr_k"])
        light = light_by_symbol.get("sh000688" if sym.startswith("688") else "sh000852")
        sig = signals.get(sym, {})
        decision = sig.get("decision", "")
        sector_weak = any("板块" in r and "弱势" in r for r in sig.get("risks", []))
        conds = {
            "light_ok": None if light is None else light["state"] == "risk_on",
            "trend_ok": last >= trail,
            "engine_ok": (None if not decision
                          else decision not in NEG_DECISIONS and not sector_weak),
        }
        rows.append({
            **base, "stale": stale, "close": last,
            "trail": trail, "trail_dist": last / trail - 1,
            "cooldown_elapsed": elapsed, "cooldown_total": cfg["cooldown_days"],
            "cooldown_done": elapsed >= cfg["cooldown_days"],
            "light_name": light["name"] if light else "",
            "decision": decision, "sector_weak": sector_weak, **conds,
            "met": sum(1 for v in conds.values() if v is True),
            "ready": (all(v is True for v in conds.values())
                      and elapsed >= cfg["cooldown_days"]),
        })
    return {"rows": rows, "signal_date": sig_date, "any_stale": any_stale}


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
