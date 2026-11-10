"""持仓健康度（晨报新段的数据层）：止损线 + 引擎排雷信号。

只读三个数据源，全部 REPO 锚定路径、与 cwd 无关：
- watchlist.yaml：HELD 持仓 + account / target_structure / hard_stop_pct 配置
- cache/kline/<symbol>_daily_qfq.csv：旧引擎 18:30 刷新的共享 K 线缓存；
  末日落后超 STALE_TRADING_DAYS 个交易日 → akshare 现拉降级回写；仍追不上才标 ⚠️
- results/decisions/partial_<date>.csv：旧引擎最新决策（排雷信号按 symbol 匹配，
  CSV 的 state 列可能滞后于 watchlist，不采信）

不 import 旧引擎模块（factors/data 链上有 .venv-quant 缺失的依赖），
_atr 复制自 factors/technical/dim.py:54。
"""
from __future__ import annotations

import ast
import time
from pathlib import Path

import pandas as pd
import yaml

from quant.data import trade_calendar

REPO = Path(__file__).resolve().parents[2]
WATCHLIST = REPO / "watchlist.yaml"
KLINE_DIR = REPO / "cache" / "kline"
DEC_DIR = REPO / "results" / "decisions"
STALE_TRADING_DAYS = 1  # 缓存末日落后上一交易日即算陈旧 → 现拉降级。
#   个股原为 3：晨报 today = 指数末日的下一交易日，所以落后 1 = 正常、落后 2 = 少一根 K 线，
#   3 的容忍度会把"旧引擎漏跑一晚"整个吞掉——2026-09-02 实盘踩过：daily_run 09-01 没跑，
#   晨报拿 08-31 收盘当"现价"且不标 ⚠️，止损距离虚报 +8.6%（实际 +5.1%）。
MIN_CACHE_ROWS = 140  # 回写下限：抄底灯 dd120+信号窗要 140 根，低于此数的应答一律当残缺
FETCH_PAUSE_S = 0.15  # 现拉之间的间隔，与旧引擎 run_partial_engine 同一量级。
#   缓存命中不 sleep，只在真走网络时付这个钱：旧引擎漏跑一晚时全 130+ 只票会一起现拉，
#   连珠炮打免费接口容易被掐（多花约 20s，换一次批量刷新不半途失败）。
ATR_N = 14
TRAIL_WINDOW = 20  # ATR 追踪线的滚动最高收盘窗口（趋势读数用；止损锚点是 since-entry，见 stop_lines）
LEFT_INIT_ATR_K = 3.0  # 左侧臂初始止损 = 入场价 - 3×ATR14(信号日)；与 stock_lights.py LEFT_STOP_ATR 同源定案
ARM_SUSPECT_DD = -0.25  # 左臂标注体检：入场时距 120 日高应 ≤ -25%（抄底灯 setup 最浅档），不满足疑似标错臂
ETF_PREFIXES = ("51", "56", "58", "159")  # 沪 51x/56x/58x + 深 159x；LOF/封基不算
DD_WINDOW = 120  # ETF 观察的"距 N 日最高收盘"窗口（与抄底灯 setup 同口径，便于对照）
STATE_RANK = {"HELD": 0, "EXITED": 1, "WATCHING": 2, "NONE": 3}  # 同一 symbol 重复登记时的取舍序


def is_etf(symbol: str) -> bool:
    """场内 ETF 代码段判定（与 factors/_kline.py 的副本保持一致，venv 隔离不互相 import）。"""
    return str(symbol).startswith(ETF_PREFIXES)


def dedupe_watchlist(items: list[dict]) -> list[dict]:
    """watchlist 条目按 symbol 去重，保留状态最强的那条（HELD > EXITED > WATCHING > NONE）。

    watchlist.yaml 是手工维护 + 批量扫描追加的，同一只票很容易被登记两次。持仓票若
    以 WATCHING 的身份漏出来，就会绕过抄底灯的"持仓票不列入"过滤被当成新钱候选
    ——2026-09-02 实盘踩过：002487 大金重工 同时挂着 HELD 和 WATCHING，晨报把自己的
    持仓推荐成了抄底标的。yaml 那边已经改干净，这里兜底，防再犯。
    """
    best: dict[str, dict] = {}
    for it in items:
        sym = str(it["symbol"]).zfill(6)
        cur = best.get(sym)
        fallback = len(STATE_RANK)
        if cur is None or (STATE_RANK.get(str(it.get("state", "")), fallback)
                           < STATE_RANK.get(str(cur.get("state", "")), fallback)):
            best[sym] = it
    return list(best.values())


def load_portfolio_config(path: Path = WATCHLIST) -> dict:
    """watchlist.yaml → {held, exited, account, target_structure, hard_stop_pct, atr_k, cooldown_days}。

    exited 只收带 exit_date 的 EXITED 条目（回补观察跟踪对象）；
    不带 exit_date 的视为历史清仓，不跟踪。
    """
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    items = dedupe_watchlist(cfg.get("watchlist", []))
    held = [it for it in items if it.get("state") == "HELD"]
    exited = [it for it in items if it.get("state") == "EXITED" and it.get("exit_date")]
    etf_watching = [it for it in items
                    if it.get("state") == "WATCHING" and is_etf(str(it["symbol"]))]
    strategy = str(cfg.get("default_stop_strategy", "atr_2"))
    atr_k = float(strategy.split("_", 1)[1]) if strategy.startswith("atr_") else 2.0
    return {
        "held": held,
        "exited": exited,
        "etf_watching": etf_watching,
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


def _fetch_etf_kline(symbol: str, start: str, end: str) -> pd.DataFrame:
    """东财 ETF 日线（前复权）→ 与股票缓存同列名。新浪 fund_etf_hist_sina 不复权
    （份额拆分会把距高/MA 全算错），禁用。"""
    import akshare as ak

    df = ak.fund_etf_hist_em(symbol=symbol, period="daily",
                             start_date=start, end_date=end, adjust="qfq")
    df = df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
                            "最低": "low", "成交量": "volume", "成交额": "amount",
                            "换手率": "turnover"})
    df = df[[c for c in ("date", "open", "high", "low", "close",
                         "volume", "amount", "turnover") if c in df.columns]]
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def _lag(df: pd.DataFrame, today: pd.Timestamp) -> int:
    """缓存末日距 today 的交易日数（交易所日历）。晨报的 today = 指数末日的下一交易日，
    所以"数据齐"时恒为 1，长假后首日缓存停在节前收盘也是 1。"""
    return trade_calendar.trading_days_between(df["date"].iloc[-1], today)


def load_kline(symbol: str, today: pd.Timestamp) -> tuple[pd.DataFrame, bool]:
    """(日K df[date/open/high/low/close], 是否陈旧)。缓存陈旧时降级 akshare 现拉并回写。

    ETF 与个股只剩数据源一处不同：ETF 走东财 fund_etf_hist_em（qfq），个股走新浪
    stock_zh_a_daily。陈旧阈值与回写路径两者一致——现拉回来不落盘的话，旧引擎没覆盖
    到的票每天早上都要重新下载一遍且缓存永远停在老日期（2026-09-02 实盘踩过：
    30 只优质股池新票的缓存卡在 08-07 三周没动，每天白跑 30 次网络请求）。
    """
    p = KLINE_DIR / f"{symbol}_daily_qfq.csv"
    df = pd.DataFrame()
    if p.exists():
        try:
            df = pd.read_csv(p, parse_dates=["date"])
        except Exception:
            df = pd.DataFrame()
    etf = is_etf(symbol)
    stale = df.empty or _lag(df, today) > STALE_TRADING_DAYS
    if stale:
        try:
            # 400 日历日 ≈ 270 交易日：抄底灯的 dd120+信号窗要 140 根，无缓存也够
            start = (today - pd.Timedelta(days=400)).strftime("%Y%m%d")
            end = today.strftime("%Y%m%d")
            if etf:
                fresh = _fetch_etf_kline(symbol, start, end)
            else:
                import akshare as ak

                prefix = "sh" if symbol.startswith("6") else "sz"
                fresh = ak.stock_zh_a_daily(symbol=f"{prefix}{symbol}", adjust="qfq",
                                            start_date=start, end_date=end)
                fresh["date"] = pd.to_datetime(fresh["date"])
                fresh = fresh.sort_values("date")
            fresh = fresh[fresh["date"] < today]  # 盘中重跑时丢掉当日未收线的 bar
            if fresh.empty:
                raise ValueError(f"{symbol} 现拉返回空")  # 别拿空表覆盖缓存
            cached_len = len(df)
            cached_last = df["date"].iloc[-1] if cached_len else None
            # 与旧缓存拼接保长历史（除权日复权口径或有微差，以新段为准）。
            # 两段必须相接：若新段起点还在旧缓存末日之后，中间就是空洞，拼起来会让
            # ATR/dd120 算在断裂的序列上——这种情况宁可只留新段。
            if not df.empty and fresh["date"].iloc[0] <= df["date"].iloc[-1]:
                fresh = pd.concat([df[df["date"] < fresh["date"].iloc[0]], fresh])
            df = fresh.reset_index(drop=True)
            # 回写门槛：确实往前推进了，且长度够 min(原有, 140)——够抄底灯用即可，
            # 且从不要求比原来更多（无缓存时门槛为 0，新票历史短也照写）。
            # 别写成"不准比旧缓存短"：现拉窗口固定 400 日历日，比攒久了的缓存短是常态，
            # 那样卡会把正常刷新整片挡掉（本次改动初版就是这么写的，30 只票一只没刷上）。
            advanced = cached_last is None or df["date"].iloc[-1] > cached_last
            if advanced and len(df) >= min(cached_len, MIN_CACHE_ROWS):
                KLINE_DIR.mkdir(parents=True, exist_ok=True)
                df.to_csv(p, index=False)  # 次日命中缓存，不再重复下载
            # 现拉成功=已经拿到源上最新的，不是降级，别再标 ⚠️；仍追不上才是真陈旧（停牌/源滞后）
            stale = _lag(df, today) > STALE_TRADING_DAYS
        except Exception as ex:
            # 现拉失败 → 返回旧数据（stale=True 由调用方标 ⚠️）。必须留痕：静默吞掉的话，
            # 批量刷新被源端掐断只会表现为"某些票的缓存莫名不更新"，很难查。
            print(f"  [kline] {symbol} 现拉失败，沿用缓存: {type(ex).__name__}: {ex}")
        finally:
            time.sleep(FETCH_PAUSE_S)
    return df, stale


def trail_line(kline: pd.DataFrame, atr_k: float) -> float:
    """ATR 趋势线 = 近20日最高收盘 - k×ATR14（回补观察/ETF观察的趋势读数，非止损锚）。"""
    close = kline["close"].astype(float)
    atr = _atr(kline["high"].astype(float), kline["low"].astype(float), close)
    return float(close.tail(TRAIL_WINDOW).max() - atr_k * atr.iloc[-1])


def since_entry_ratchet(kline: pd.DataFrame, entry_date, atr_k: float) -> float | None:
    """入场以来棘轮止损 = max_u( 入场至 u 日最高收盘 - k×ATR14(u) )，u 遍历入场后每个交易日。

    与 stock_lights.py 回测持仓循环同构（stop = max(stop, hi - k×atr)），从当前 K 线
    整段确定性重算、不持久化——qfq 前复权在每个除权日整体重排历史价，存盘的 stop
    价位会跨复权基准失效，重算天然免疫。
    K 线起点晚于入场日（缺入场早期高点）、入场后尚无收盘 bar、或窗口内 ATR 有缺
    → 返回 None 由调用方降级，不静默用残缺窗口。
    """
    dates = pd.to_datetime(kline["date"])
    entry = pd.Timestamp(entry_date)
    if dates.iloc[0] > entry:
        return None
    mask = (dates >= entry).to_numpy()
    if not mask.any():
        return None
    close = kline["close"].astype(float)
    atr = _atr(kline["high"].astype(float), kline["low"].astype(float), close)
    lines = close[mask].cummax() - atr_k * atr[mask]
    if lines.isna().any():
        return None
    return float(lines.max())


def _signal_day_atr(kline: pd.DataFrame, entry_date) -> float | None:
    """入场前一交易日（信号日）的 ATR14——回测左侧臂用 atr[t] 而非入场当日，对齐。"""
    dates = pd.to_datetime(kline["date"])
    close = kline["close"].astype(float)
    atr = _atr(kline["high"].astype(float), kline["low"].astype(float), close)
    before = atr[(dates < pd.Timestamp(entry_date)).to_numpy()]
    if before.empty or pd.isna(before.iloc[-1]):
        return None
    return float(before.iloc[-1])


def stop_lines(kline: pd.DataFrame, entry_price: float, hard_stop_pct: float,
               atr_k: float, entry_date=None, entry_arm: str = "right") -> dict:
    """止损合成（与 stock_lights.py 回测同构）+ 展示用 ATR 趋势线。

    初始止损按入场臂选：right = entry×(1-8%)（回测 RIGHT_STOP 0.92）；
    left = entry - 3×ATR14(信号日)（回测 LEFT_STOP_ATR，宽止损防打脸）。
    最终 stop = max(初始止损, since-entry 棘轮)，只升不降。
    entry_date 缺失 / K 线覆盖不足 / ATR 不可算 → 降级为仅初始硬止损并标 degraded。
    """
    close = kline["close"].astype(float)
    last = float(close.iloc[-1])
    hard = entry_price * (1 - hard_stop_pct)
    arm = str(entry_arm or "right").lower()
    init, degraded, reason = hard, False, ""
    if entry_date is None:
        degraded, reason = True, "缺entry_date"
    if arm == "left" and not degraded:
        atr_sig = _signal_day_atr(kline, entry_date)
        if atr_sig is None:
            degraded, reason = True, "入场前ATR不可算"
        else:
            init = entry_price - LEFT_INIT_ATR_K * atr_sig
    ratchet = None
    if not degraded:
        ratchet = since_entry_ratchet(kline, entry_date, atr_k)
        if ratchet is None:
            degraded, reason = True, "K线不足入场窗"
            init = hard  # 左臂算出的 init 也不可信（窗口残缺），一并退回硬止损
    suspect, dd_at_entry = None, None
    if arm == "left" and entry_date is not None:
        upto = close[(pd.to_datetime(kline["date"]) < pd.Timestamp(entry_date)).to_numpy()]
        if len(upto) >= 60:  # 不足半个 setup 窗不下判断
            dd_at_entry = float(upto.iloc[-1] / upto.tail(DD_WINDOW).max() - 1)
            suspect = dd_at_entry > ARM_SUSPECT_DD
    line = init if ratchet is None else max(init, ratchet)
    trend = trail_line(kline, atr_k)
    return {
        "hard_stop": hard, "init_stop": init, "ratchet": ratchet,
        "stop_line": line, "broken": last < line,
        "dist_pct": last / line - 1,  # 现价距止损线（负=已破）
        "trend_line": trend, "above_trend": last >= trend,
        "degraded": degraded, "degrade_reason": reason,
        "arm_suspect": suspect, "dd_at_entry": dd_at_entry,
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


def _etf_last_close_sina(symbol: str, today: pd.Timestamp) -> float:
    """新浪 ETF 日线的最新已收盘价。不复权——只用于取最新一根（与 qfq 在最新日相等），
    历史段会被份额拆分搞错，别拿去算 MA/ATR（那条禁令见 _fetch_etf_kline）。"""
    import akshare as ak

    prefix = "sh" if symbol.startswith(("5", "6")) else "sz"
    df = ak.fund_etf_hist_sina(symbol=f"{prefix}{symbol}")
    df = df[pd.to_datetime(df["date"]) < today]  # 盘中跑时丢掉当日未收线的 bar
    if df.empty:
        raise ValueError(f"{symbol} 新浪返回空")
    return float(df["close"].iloc[-1])


def _core_etf_value(etfs: list[dict], today: pd.Timestamp) -> tuple[float, list[str]]:
    """核心宽基 ETF 市值 = Σ 持仓 × 最新收盘。取价顺序：K 线缓存/东财现拉 → 新浪最新收盘
    → 成本价；落到成本价的返回其名单（调用方标 ⚠️）。核心宽基不在任何扫描名单里，
    通常没有 K 线缓存，东财又常被封（2026-10-21 实况），新浪这一层是常走路径。

    曾经一律按成本估——注释称"宽基波动对结构带宽影响可忽略"，实盘证伪：2026-10-20
    159919 成本 4.874、现价 4.430（-9%），17800 股多算 7.9k，总资产 18.5万 报成 19.3万，
    科创50ETF 超限减仓额从 2,320 少报成 1,530，核心占比 42.7% 报成 45%。
    """
    value, fallback = 0.0, []
    for e in etfs:
        price = float(e["entry_price"])
        sym = str(e.get("symbol", "")).zfill(6) if e.get("symbol") else ""
        kline = load_kline(sym, today)[0] if sym else pd.DataFrame()
        if not kline.empty:
            price = float(kline["close"].iloc[-1])
        else:
            try:
                if not sym:
                    raise ValueError("缺 symbol")
                price = _etf_last_close_sina(sym, today)
            except Exception as ex:
                print(f"  [etf] {sym or e.get('name')} 取价失败，按成本估值: {type(ex).__name__}: {ex}")
                fallback.append(e.get("name") or sym or "?")
        value += float(e["position"]) * price
    return value, fallback


def structure_drift(cfg: dict | None = None, health: dict | None = None,
                    today: pd.Timestamp | None = None) -> dict:
    """晨报【结构修正跟踪】数据：当前 vs 目标结构的偏离与待执行动作（规则生成）。

    卫星按 holdings_health 的现价市值，核心 ETF 按 _core_etf_value 的现价市值。
    """
    cfg = cfg or load_portfolio_config()
    today = today or pd.Timestamp.today().normalize()
    health = health or holdings_health(cfg, today)
    target = cfg["target_structure"]
    lo, hi = target.get("core_etf_pct", [0.40, 0.50])
    max_count = int(target.get("satellite_max_count", 3))
    max_pct = float(target.get("satellite_max_pct", 0.10))

    rows = [r for r in health["rows"] if not r.get("error")]
    stock_value = sum(r["value"] for r in rows)
    etf_value, etf_cost_fallback = _core_etf_value(cfg["account"].get("etf", []), today)
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
        "etf_cost_fallback": etf_cost_fallback,
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

    三条件：①对应结构灯转绿（688xxx/588xxx→科创50，其余→中证1000；lights 缺失时未知）
    ②收盘价站回 ATR 追踪线（让股价自证跌完，而不是猜底）
    ③引擎最新信号非 DROP/STOP/REDUCE 且风险项无"板块…弱势"（匹配自家引擎的固定文案）；
      ETF 不进 8 维引擎，排雷不适用 → 条件③视为满足。
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
        elapsed = trade_calendar.trading_days_between(pd.Timestamp(it["exit_date"]), today)
        last = float(kline["close"].astype(float).iloc[-1])
        trail = trail_line(kline, cfg["atr_k"])
        light = light_by_symbol.get(
            "sh000688" if sym.startswith(("688", "588")) else "sh000852")
        sig = signals.get(sym, {})
        decision = sig.get("decision", "")
        sector_weak = any("板块" in r and "弱势" in r for r in sig.get("risks", []))
        conds = {
            "light_ok": None if light is None else light["state"] == "risk_on",
            "trend_ok": last >= trail,
            "engine_ok": (True if is_etf(sym)  # ETF 无引擎排雷，条件不适用=满足
                          else None if not decision
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
            "etf": is_etf(sym),
            "close": last, "close_date": str(kline["date"].iloc[-1].date()),
            "stale": stale,
            "entry": entry, "pnl_pct": last / entry - 1,
            "value": last * float(it["position"]),
            **stop_lines(kline, entry, cfg["hard_stop_pct"], cfg["atr_k"],
                         entry_date=it.get("entry_date"),
                         entry_arm=it.get("entry_arm", "right")),
            "decision": sig.get("decision", ""),
            "risks": sig.get("risks", []),
        })
    return {"rows": rows, "signal_date": sig_date, "any_stale": any_stale}


def etf_watch(cfg: dict | None = None, today: pd.Timestamp | None = None) -> dict:
    """晨报【ETF观察】数据：watchlist 里 WATCHING 状态 ETF 的趋势读数。

    纯展示、不给买卖信号——抄底灯的参数与期望值只在个股中性池上回测过，
    对 ETF 不适用（"无回测不改参"，扩展需先过 stock_lights 级别的回测）。
    读数与既有口径对齐：dd120=距 120 日最高收盘（抄底灯 setup 同源）、
    vs_ma200=主灯/结构灯的趋势腿、trail=持仓止损共用的 ATR 追踪线。
    """
    cfg = cfg or load_portfolio_config()
    today = today or pd.Timestamp.today().normalize()
    rows, any_stale = [], False
    for it in cfg.get("etf_watching", []):
        sym = str(it["symbol"]).zfill(6)
        base = {"symbol": sym, "name": it.get("name", sym)}
        kline, stale = load_kline(sym, today)
        if kline.empty or len(kline) < ATR_N + 1:
            rows.append({**base, "error": "无K线数据"})
            continue
        any_stale |= stale
        close = kline["close"].astype(float)
        last = float(close.iloc[-1])
        ma200 = float(close.tail(200).mean()) if len(close) >= 200 else None
        trail = trail_line(kline, cfg.get("atr_k", 2.0))
        rows.append({
            **base, "stale": stale,
            "close": last, "close_date": str(kline["date"].iloc[-1].date()),
            "chg_1d": last / float(close.iloc[-2]) - 1 if len(close) >= 2 else 0.0,
            "dd120": last / float(close.tail(DD_WINDOW).max()) - 1,
            "vs_ma200": last / ma200 - 1 if ma200 else None,
            "trail": trail, "above_trail": last >= trail,
        })
    return {"rows": rows, "any_stale": any_stale}
