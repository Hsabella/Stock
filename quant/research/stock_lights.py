"""个股抄底红绿灯敏感性回测：左侧 vs 右侧 vs 大盘 regime 切换的事件式评估。

背景：现有体系里指数层有红绿灯/结构灯，个股层只有止损线和排雷，缺"什么时候能买"。
上线前按项目纪律先回测。规则经三视角设计评审(2026-07-19)修订，评审实测数据见
对话记录；本 docstring 同时是"事先注册"的判胜规则（跑之前写死，防多重检验挑噪声）。

信号臂（全部事件化：条件由 false→true 的首日才算信号；交易仿真一次一笔+冷却10日）：
- S0 基准臂：setup 首次成立即买（"见到便宜就买"的裸收益，其他臂的价值=相对 S0 超额）
- 左侧 L1 接刀：setup 且 RSI14≤30 首穿
- 左侧 L6 超卖修复：setup 且 RSI14 连续≥3日≤30 后首次回升穿越 30（"刀落地后第一个抬头"）
- 左侧 L4 底背离：setup 且 收盘创近60日新低 且 RSI14 高于其近60日最低 ≥5 且 RSI14≤40
- 右侧 R1 站回MA20：底部区（近60日内曾达 setup 深度）且收盘>MA20 连续2日（首次）
- 右侧 R3 走平站回：R1 且 MA20 三日不再下行（过滤下行均线上的死猫反弹）
- 右侧 R4 低点翻转线：底部区 且 收盘 ≥ 近20日最低收盘+2×ATR14 连续2日（首次）
setup：收盘 / 近120日最高收盘 - 1 ≤ 阈值，扫 -25% / -35% / -45%（主参 -35%）。

出场（左右分开、各自与入场逻辑自洽；追踪线锚定"入场以来最高收盘"且只升不降）：
- 初始止损：左侧臂+S0 = 入场价 - 3×ATR14(信号日)；右侧臂 = 入场价×0.92
- 追踪：stop = max(stop, 入场以来最高收盘 - 2×ATR14)，收盘跌破次日卖
- 60 交易日强制评估；右侧臂另跑"无时间止损"敏感行
- 涨跌停按板块定阈值（688/创业板2020-08后 19.5%，主板 9.5%，北交 29.5%），
  另用"一字板"（开=高=低）识别兜住 ST ±5% 等未建模档位：
  开盘触限/一字涨停不买；卖出遇跌停开盘/一字跌停/停牌顺延到首个可成交日，
  顺延亏损计入；数据走完仍未了结（退市/长停）按末日收盘强平计入——不丢弃坏尾部。
  事件研究的 fwd 尾部同样按末日收盘截断强平（与 trades 同口径，不静默剔除坏尾部）。
- regime 灯取"入场日早晨可见"的灯（信号日收盘算出的灯 = 实盘晨报口径）。

判胜规则（事先注册）：
- 主假设 H1 左 vs 右、H2 大盘灯色×风格交互；主指标 = 交易级收益相对 S0 的差。
- 单格 n<30 不评；结论须在 2019-22 与 2023-26 两段方向一致、且在 ≥2/3 个 setup
  阈值上方向一致；选触发器只看 2019-22 段，2023-26 段只做确认。
- 自选池是 2026 年事后挑的（幸存者偏差，且对左侧臂偏袒）：主结论以 PIT 中性池
  （csi_union 抽样，含已退市票）为准，另按"是否沪深300成分(PIT)"拆大小盘画像。

用法: .venv-quant/bin/python -m quant.research.stock_lights [--pools watch,neutral]
      .venv-quant/bin/python -m quant.research.stock_lights --status   # 今日各票灯态
输出: results/quant/stock_lights_{events,trades}.csv + stdout markdown 表
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quant.backtest.timing import debounce, raw_ma_off
from quant.data.index_daily import load_closes

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / "quant"
KLINE_DIR = REPO / "cache" / "kline"
INSTRUMENTS_DIR = Path("~/.qlib/qlib_data/cn_data/instruments").expanduser()

EVAL_START = "2019-01-01"
SETUP_THRS = [-0.25, -0.35, -0.45]
PRIMARY_THR = -0.35
CONTEXT_LOOKBACK = 60   # 右侧触发要求近 60 日内曾达 setup 深跌（反弹后 dd 收窄仍算"底部区"）
EVENT_SPACING = 20      # 事件研究去重间隔（交易日）
HORIZONS = [5, 10, 20, 60]
MAX_HOLD = 60
COOLDOWN = 10
LEFT_STOP_ATR = 3.0     # 左侧初始止损 = 入场价 - 3×ATR14（约2个ATR的8%对小票是打脸机器）
RIGHT_STOP = 0.92
TRAIL_ATR = 2.0         # 持仓追踪 = 入场以来最高收盘 - 2×ATR14（与实盘 atr_2 同参、锚不同）
BUY_COST, SELL_COST = 0.0005, 0.0015
NEUTRAL_SAMPLE = 400
MIN_ROWS = 180

REGIMES = {  # 与晨报主灯/结构灯同参（只用 MA 腿）
    "hs300": ("SH000300", 200, 2),
    "csi1000": ("SH000852", 200, 3),
}
LEFT_STYLES = ["L1接刀", "L6超卖修复", "L4底背离"]
RIGHT_STYLES = ["R1站回MA20", "R3走平站回", "R4低点翻转"]
STYLES = ["S0基准", *LEFT_STYLES, *RIGHT_STYLES]
SWITCH_PAIRS = [("L6超卖修复", "R4低点翻转"), ("L1接刀", "R1站回MA20")]
GEM_20CM_FROM = pd.Timestamp("2020-08-24")  # 创业板注册制改 20% 涨跌幅


# ---------- 指标 ----------

def wilder_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-delta).clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr14(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """df: 单只股票 open/high/low/close/volume（已去停牌 NaN 行）。"""
    out = df.copy()
    c = out["close"]
    out["dd120"] = c / c.rolling(120, min_periods=60).max() - 1
    out["rsi"] = wilder_rsi(c)
    out["ma5"] = c.rolling(5).mean()
    out["ma20"] = c.rolling(20).mean()
    out["atr"] = atr14(out["high"], out["low"], c)
    out["min20c"] = c.rolling(20).min()
    out["trail20"] = c.rolling(20).max() - TRAIL_ATR * out["atr"]  # 实盘同款（仅展示用）
    return out


def _fresh(cond: pd.Series) -> pd.Series:
    """条件 false→true 的首日（事件化）。"""
    cond = cond.astype("boolean").fillna(False).astype(bool)
    return cond & ~cond.shift(1, fill_value=False)


def _fresh2(cond: pd.Series) -> pd.Series:
    """连续 2 日成立且此前不成立（右侧确认的事件化）。"""
    cond = cond.astype("boolean").fillna(False).astype(bool)
    two = cond & cond.shift(1, fill_value=False)
    return two & ~two.shift(1, fill_value=False)


def signal_frame(ind: pd.DataFrame, thr: float) -> pd.DataFrame:
    """各臂事件信号（信号日=收盘判定，次日开盘成交）。"""
    c, rsi = ind["close"], ind["rsi"]
    setup = ind["dd120"] <= thr
    context = ind["dd120"].rolling(CONTEXT_LOOKBACK, min_periods=1).min() <= thr
    below30 = rsi <= 30
    newlow = c <= c.shift(1).rolling(60).min()
    diverge = (rsi - rsi.shift(1).rolling(60).min() >= 5) & (rsi <= 40)
    r4line = ind["min20c"] + TRAIL_ATR * ind["atr"]
    ma20_flat = ind["ma20"] >= ind["ma20"].shift(3)
    r1_two = (c > ind["ma20"]) & (c > ind["ma20"]).shift(1, fill_value=False)
    return pd.DataFrame({
        "S0基准": _fresh(setup),
        "L1接刀": _fresh(setup & below30),
        "L6超卖修复": (setup & ~below30 & below30.shift(1, fill_value=False)
                       & (below30.rolling(3).sum().shift(1) >= 3)),
        "L4底背离": _fresh(setup & newlow & diverge),
        "R1站回MA20": context & _fresh2(c > ind["ma20"]),
        "R3走平站回": context & _fresh2(c > ind["ma20"]) & ma20_flat,
        "R4低点翻转": context & _fresh2(c >= r4line),
    }, index=ind.index).fillna(False)


# ---------- 板块涨跌停 ----------

def limit_schedule(code: str, index: pd.DatetimeIndex) -> np.ndarray:
    """逐日涨跌停幅度（留 0.5pct buffer 在使用处减）。code 如 SH688001/SZ300724。"""
    num = code[2:]
    if num.startswith("688") or code.startswith("BJ"):
        return np.full(len(index), 0.195 if num.startswith("688") else 0.295)
    if num.startswith(("300", "301")):
        return np.where(index >= GEM_20CM_FROM, 0.195, 0.095)
    return np.full(len(index), 0.095)


# ---------- regime ----------

def regime_greens() -> dict[str, pd.Series]:
    """{name: 绿灯布尔序列(按指数日历)}；缺数据直接抛错（回测不能带病跑）。"""
    closes = load_closes()
    out = {}
    for name, (inst, ma, confirm) in REGIMES.items():
        close = closes[inst].dropna()
        calendar = close.index[ma:]
        off = debounce(raw_ma_off(close, ma, calendar), confirm)
        out[name] = ~off
    return out


# ---------- 事件研究 ----------

def spaced_events(sig: np.ndarray, min_gap: int) -> list[int]:
    picked, last = [], -10**9
    for t in np.flatnonzero(sig):
        if t - last >= min_gap:
            picked.append(int(t))
            last = t
    return picked


def event_rows(inst: str, ind: pd.DataFrame, sigs: pd.DataFrame, thr: float,
               gmap: dict[str, np.ndarray], valid: np.ndarray,
               in300: np.ndarray) -> list[dict]:
    """单股全部臂的事件样本（fwd 为次日开盘→h 日后开盘，减同股无条件基线；
    数据尾部不足 h 日的按末日收盘截断强平，与 trades 同口径，不剔除退市坏尾部）。"""
    open_, close = ind["open"].to_numpy(), ind["close"].to_numpy()
    n = len(ind)
    fwd = {}
    for h in HORIZONS:
        f = np.full(n, np.nan)
        f[: n - 1 - h] = open_[1 + h:] / open_[1: n - h] - 1
        for t in range(max(0, n - 1 - h), n - 1):  # 尾部截断：末日收盘强平
            f[t] = close[n - 1] / open_[t + 1] - 1
        fwd[h] = f
    eval_ok = np.asarray(ind.index >= EVAL_START) & valid
    base = {h: np.nanmean(fwd[h][eval_ok]) if eval_ok.any() else np.nan for h in HORIZONS}

    rows = []
    for style in STYLES:
        sig = sigs[style].to_numpy() & eval_ok
        for t in spaced_events(sig, EVENT_SPACING):
            e = min(t + 1, n - 1)  # 灯用入场日早晨可见口径
            row = {"instrument": inst, "style": style, "thr": thr,
                   "date": str(ind.index[t].date()),
                   "hs300_green": bool(gmap["hs300"][e]),
                   "csi1000_green": bool(gmap["csi1000"][e]),
                   "in_csi300": bool(in300[t])}
            for h in HORIZONS:
                row[f"fwd{h}"] = fwd[h][t]
                row[f"ex{h}"] = fwd[h][t] - base[h]
            rows.append(row)
    return rows


# ---------- 交易仿真 ----------

def simulate(ind: pd.DataFrame, sig: np.ndarray, allowed: np.ndarray, stop_kind: str,
             limits: np.ndarray, max_hold: int = MAX_HOLD) -> list[dict]:
    """单股单臂：一次一笔+冷却；止损=初始线与"入场以来最高收盘-2ATR"棘轮取高；
    卖出遇跌停开盘/停牌顺延；数据走完未了结按末日收盘强平。"""
    open_, close = ind["open"].to_numpy(), ind["close"].to_numpy()
    high, low = ind["high"].to_numpy(), ind["low"].to_numpy()
    vol, atr = ind["volume"].to_numpy(), ind["atr"].to_numpy()
    min20c = ind["min20c"].to_numpy()
    n = len(ind)
    trades, cooldown_until, ptr = [], -1, 0
    sig_idx = np.flatnonzero(sig & allowed)
    while ptr < len(sig_idx):
        t = int(sig_idx[ptr])
        ptr += 1
        if t <= cooldown_until or t + 1 >= n or not np.isfinite(atr[t]):
            continue
        e = t + 1
        o = open_[e]
        if (not np.isfinite(o) or vol[e] <= 0 or abs(o / close[t] - 1) >= limits[e] - 0.005
                or high[e] == low[e]):
            continue  # 停牌/开盘触限/一字板（含 ST ±5% 档）不买
        stop = (o - LEFT_STOP_ATR * atr[t]) if stop_kind == "left" else o * RIGHT_STOP
        hi, exit_u = close[e], None
        for u in range(e, n):
            hi = max(hi, close[u])
            line = hi - TRAIL_ATR * atr[u]
            if np.isfinite(line):
                stop = max(stop, line)
            if close[u] < stop or u - e >= max_hold:
                exit_u = u
                break
        if exit_u is None:  # 数据尾部仍持仓：按末日收盘估值离场（不丢弃）
            exit_x, exit_px, censored = n - 1, close[n - 1], True
        else:
            x = exit_u + 1
            while x < n and (not np.isfinite(open_[x]) or vol[x] <= 0
                             or open_[x] / close[x - 1] - 1 <= -(limits[x] - 0.005)
                             or (high[x] == low[x] and open_[x] < close[x - 1])):
                x += 1  # 跌停开盘/一字跌停(含 ST ±5% 档)/停牌卖不出，顺延（亏损计入）
            if x < n:
                exit_x, exit_px, censored = x, open_[x], False
            else:
                exit_x, exit_px, censored = n - 1, close[n - 1], True
        ret = exit_px * (1 - SELL_COST) / (o * (1 + BUY_COST)) - 1
        premium = o / min20c[t] - 1 if np.isfinite(min20c[t]) and min20c[t] > 0 else np.nan
        trades.append({"signal_t": t, "entry_date": str(ind.index[e].date()),
                       "exit_date": str(ind.index[exit_x].date()), "ret": ret,
                       "hold": exit_x - e, "premium": premium, "censored": censored})
        cooldown_until = exit_x + COOLDOWN
        while ptr < len(sig_idx) and sig_idx[ptr] <= cooldown_until:
            ptr += 1
    return trades


def build_variants() -> dict[str, tuple]:
    """variant 名 → (styles, regime源, 用法, max_hold)。styles 为单臂或(左,右)对。"""
    v: dict[str, tuple] = {}
    for s in STYLES:
        v[f"{s}|all"] = (s, None, "all", MAX_HOLD)
        for src in REGIMES:
            v[f"{s}|{src}_green"] = (s, src, "green", MAX_HOLD)
            v[f"{s}|{src}_red"] = (s, src, "red", MAX_HOLD)
    for s in RIGHT_STYLES:  # 右侧趋势靠右尾，60日强评可能砍半山腰 → 长持敏感行
        v[f"{s}|长持"] = (s, None, "all", 10**6)
    for l_s, r_s in SWITCH_PAIRS:
        for src in REGIMES:
            v[f"切换[{l_s[:2]}绿/{r_s[:2]}红]|{src}"] = ((l_s, r_s), src, "switch", MAX_HOLD)
    return v


def simulate_universe(subset: dict[str, pd.DataFrame], sigs_by_inst: dict, thr: float,
                      gmaps: dict, valids: dict, in300s: dict) -> list[dict]:
    variants = build_variants()
    rows = []
    for inst, ind in subset.items():
        sigs = sigs_by_inst[inst]
        # 灯用"入场日早晨可见"口径：信号日 t 的门控/标签取 t+1 日灯（=t 收盘算出）
        gmap = {name: np.append(g[1:], g[-1]) for name, g in gmaps[inst].items()}
        eval_ok = np.asarray(ind.index >= EVAL_START) & valids[inst]
        limits = limit_schedule(inst, ind.index)
        for vname, (style, src, usage, mh) in variants.items():
            if usage == "switch":
                l_s, r_s = style
                green = gmap[src]
                sig = ((sigs[l_s].to_numpy() & green) | (sigs[r_s].to_numpy() & ~green))
                allowed, kind = eval_ok, "left"  # 切换臂止损按左侧参数（保守）
            else:
                sig = sigs[style].to_numpy()
                allowed = eval_ok.copy()
                if usage == "green":
                    allowed &= gmap[src]
                elif usage == "red":
                    allowed &= ~gmap[src]
                kind = "left" if style in (*LEFT_STYLES, "S0基准") else "right"
            for tr in simulate(ind, sig, allowed, kind, limits, mh):
                rows.append({"instrument": inst, "variant": vname, "thr": thr, **tr,
                             "hs300_green": bool(gmap["hs300"][tr["signal_t"]]),
                             "csi1000_green": bool(gmap["csi1000"][tr["signal_t"]]),
                             "in_csi300": bool(in300s[inst][tr["signal_t"]])})
    return rows


# ---------- 数据 ----------

def watch_symbols() -> list[str]:
    import yaml

    with open(REPO / "watchlist.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return sorted({str(it["symbol"]).zfill(6) for it in cfg.get("watchlist", [])})


def to_qlib_code(symbol: str) -> str:
    return ("SH" if symbol[0] == "6" else "SZ") + symbol


def load_ranges(name: str) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    ranges: dict[str, list] = {}
    for line in (INSTRUMENTS_DIR / f"{name}.txt").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            ranges.setdefault(parts[0], []).append(
                (pd.Timestamp(parts[1]), pd.Timestamp(parts[2])))
    return ranges


def load_panel(codes: list[str]) -> dict[str, pd.DataFrame]:
    import qlib
    from qlib.data import D

    from quant.config import load_config

    qlib.init(provider_uri=load_config("data")["qlib"]["provider_uri"], region="cn")
    df = D.features(codes, ["$open", "$high", "$low", "$close", "$volume"],
                    start_time="2018-01-01")
    df.columns = ["open", "high", "low", "close", "volume"]
    out = {}
    for inst, sub in df.groupby(level=0):
        sub = sub.droplevel(0).dropna(subset=["close"])
        if len(sub) >= MIN_ROWS:
            out[str(inst)] = sub
    return out


def range_mask(index: pd.DatetimeIndex, ranges: list | None) -> np.ndarray:
    if not ranges:
        return np.zeros(len(index), dtype=bool)
    mask = np.zeros(len(index), dtype=bool)
    for s, e in ranges:
        mask |= np.asarray((index >= s) & (index <= e))
    return mask


# ---------- 汇总 ----------

def summarize_events(events: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    def agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "n": len(g),
            "win20": float((g["fwd20"] > 0).mean()),
            "fwd20": g["fwd20"].mean(), "ex20": g["ex20"].mean(),
            "ex20_med": g["ex20"].median(), "ex60": g["ex60"].mean(),
            "ex20_t": g["ex20"].mean() / (g["ex20"].std() / np.sqrt(len(g)) + 1e-12),
        })
    return events.groupby(by).apply(agg, include_groups=False).reset_index()


def summarize_trades(trades: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    def agg(g: pd.DataFrame) -> pd.Series:
        wins, losses = g.loc[g["ret"] > 0, "ret"], g.loc[g["ret"] <= 0, "ret"]
        return pd.Series({
            "n": len(g), "win": float((g["ret"] > 0).mean()),
            "avg": g["ret"].mean(), "med": g["ret"].median(),
            "pf": wins.sum() / max(1e-12, -losses.sum()),
            "hold": g["hold"].mean(), "prem": g["premium"].median(),
        })
    return trades.groupby(by).apply(agg, include_groups=False).reset_index()


# ---------- 主流程 ----------

def run_backtest(pools: list[str]) -> None:
    greens = regime_greens()
    union_ranges = load_ranges("csi_union")
    csi300_ranges = load_ranges("csi300")
    watch = [to_qlib_code(s) for s in watch_symbols()]

    pool_codes: dict[str, list[str]] = {}
    if "watch" in pools:
        pool_codes["watch"] = watch
    if "neutral" in pools:
        symbols = sorted(union_ranges)
        step = max(1, len(symbols) // NEUTRAL_SAMPLE)
        pool_codes["neutral"] = symbols[::step]

    all_codes = sorted({c for cs in pool_codes.values() for c in cs})
    data = load_panel(all_codes)
    print(f"[stock_lights] 载入 {len(data)}/{len(all_codes)} 只（≥{MIN_ROWS} 交易日）")

    ind_cache = {inst: compute_indicators(df) for inst, df in data.items()}
    gmaps = {inst: {name: g.reindex(ind.index).ffill().fillna(False).to_numpy()
                    for name, g in greens.items()}
             for inst, ind in ind_cache.items()}
    in300s = {inst: range_mask(ind.index, csi300_ranges.get(inst))
              for inst, ind in ind_cache.items()}

    ev_frames, tr_frames = [], []
    for pool, codes in pool_codes.items():
        subset = {c: ind_cache[c] for c in codes if c in ind_cache}
        valids = {c: (range_mask(subset[c].index, union_ranges.get(c))
                      if pool == "neutral" else np.ones(len(subset[c]), dtype=bool))
                  for c in subset}
        for thr in SETUP_THRS:
            sigs_by_inst = {c: signal_frame(subset[c], thr) for c in subset}
            rows = []
            for inst, ind_df in subset.items():
                rows += event_rows(inst, ind_df, sigs_by_inst[inst], thr, gmaps[inst],
                                   valids[inst], in300s[inst])
            ev = pd.DataFrame(rows)
            ev["pool"] = pool
            ev_frames.append(ev)
            tr = pd.DataFrame(simulate_universe(subset, sigs_by_inst, thr, gmaps,
                                                valids, in300s))
            if not tr.empty:
                tr["pool"] = pool
                tr_frames.append(tr)
            print(f"[stock_lights] {pool} thr={thr:+.0%}: "
                  f"事件 {len(ev)}, 交易 {0 if tr.empty else len(tr)}")

    events = pd.concat(ev_frames, ignore_index=True)
    trades = pd.concat(tr_frames, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    events.to_csv(OUT_DIR / "stock_lights_events.csv", index=False)
    trades.to_csv(OUT_DIR / "stock_lights_trades.csv", index=False)

    events["period"] = np.where(events["date"] < "2023-01-01", "19-22", "23-26")
    trades["period"] = np.where(trades["entry_date"] < "2023-01-01", "19-22", "23-26")
    for pool in pool_codes:
        sub = events[events["pool"] == pool]
        tsub = trades[trades["pool"] == pool]
        print(f"\n### 事件研究 pool={pool}（ex=减同股无条件基线, t=ex20 的 t 值）")
        print(summarize_events(sub, ["thr", "style"]).round(4).to_markdown(index=False))
        prim = sub[sub["thr"] == PRIMARY_THR]
        print(f"\n### pool={pool} thr={PRIMARY_THR:.0%} × 沪深300灯")
        print(summarize_events(prim, ["style", "hs300_green"]).round(4)
              .to_markdown(index=False))
        print(f"\n### pool={pool} thr={PRIMARY_THR:.0%} × 中证1000灯")
        print(summarize_events(prim, ["style", "csi1000_green"]).round(4)
              .to_markdown(index=False))
        print(f"\n### pool={pool} thr={PRIMARY_THR:.0%} 分段稳定性")
        print(summarize_events(prim, ["style", "period"]).round(4)
              .to_markdown(index=False))
        if pool == "neutral":
            print(f"\n### pool=neutral thr={PRIMARY_THR:.0%} 大小盘画像（in_csi300）")
            print(summarize_events(prim, ["style", "in_csi300"]).round(4)
                  .to_markdown(index=False))
        print(f"\n### 交易仿真 pool={pool}（含成本/板块涨跌停/顺延, prem=入场价距20日低点溢价中位）")
        print(summarize_trades(tsub, ["thr", "variant"]).round(4)
              .to_markdown(index=False))
        tprim = tsub[tsub["thr"] == PRIMARY_THR]
        base_variants = tprim[tprim["variant"].str.endswith("|all")]
        print(f"\n### 交易仿真 pool={pool} thr={PRIMARY_THR:.0%} 分段稳定性（|all 臂）")
        print(summarize_trades(base_variants, ["variant", "period"]).round(4)
              .to_markdown(index=False))
    print(f"\n[stock_lights] 明细已写 {OUT_DIR}/stock_lights_{{events,trades}}.csv")


# ---------- 晨报接入 ----------

LAMP_SIGNALS = ["L6超卖修复", "L1接刀"]  # 仅回测过线的左侧臂进晨报；全部臂看 --status
FIRE_WINDOW = 3     # 最近 N 个交易日内触发过都算"在灯上"（晨报日更，防漏看）
LAMP_MIN_ROWS = 140  # dd120 需 120 根 + 裕量


def lamp_rows(today: pd.Timestamp | None = None) -> dict:
    """晨报【个股抄底灯】数据层：watchlist 全部票的三态灯。

    K 线走 quant.live.portfolio.load_kline（旧引擎缓存 + 陈旧时 akshare 降级），
    与持仓健康度同源同口径。lamp: fire=左侧触发 / watch=底部区观察 / none=未到 setup。
    """
    import yaml

    from quant.live import portfolio

    today = today or pd.Timestamp.today().normalize()
    with open(REPO / "watchlist.yaml", encoding="utf-8") as f:
        items = (yaml.safe_load(f) or {}).get("watchlist", [])
    rows, any_stale = [], False
    for it in items:
        sym = str(it["symbol"]).zfill(6)
        base = {"symbol": sym, "name": it.get("name", sym), "state": it.get("state", "")}
        kline, stale = portfolio.load_kline(sym, today)
        if kline.empty or len(kline) < LAMP_MIN_ROWS:
            rows.append({**base, "error": "K线不足"})
            continue
        any_stale |= stale
        ind = compute_indicators(kline.set_index("date"))
        sigs = signal_frame(ind, PRIMARY_THR)
        fired = [s for s in LAMP_SIGNALS if bool(sigs[s].tail(FIRE_WINDOW).any())]
        in_zone = bool(ind["dd120"].rolling(CONTEXT_LOOKBACK, min_periods=1)
                       .min().iloc[-1] <= PRIMARY_THR)
        last = ind.iloc[-1]
        rows.append({
            **base, "stale": stale,
            "lamp": "fire" if fired else "watch" if in_zone else "none",
            "signals": fired, "close": float(last["close"]),
            "dd120": float(last["dd120"]), "rsi": float(last["rsi"]),
        })
    return {"rows": rows, "any_stale": any_stale, "thr": PRIMARY_THR}


# ---------- 今日灯态 ----------

def status(thr: float = PRIMARY_THR) -> None:
    """用 cache/kline 新鲜数据给 watchlist 每只票打当前灯态（不依赖 qlib bin 更新）。"""
    import yaml

    with open(REPO / "watchlist.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    greens = regime_greens()
    glabel = {name: "🟢" if bool(g.iloc[-1]) else "🔴" for name, g in greens.items()}
    print(f"大盘 regime: 沪深300 {glabel['hs300']} / 中证1000 {glabel['csi1000']}"
          f"（灯截至 {max(g.index[-1] for g in greens.values()).date()}）\n")

    rows = []
    for it in cfg.get("watchlist", []):
        sym = str(it["symbol"]).zfill(6)
        p = KLINE_DIR / f"{sym}_daily_qfq.csv"
        if not p.exists():
            rows.append({"代码": sym, "名称": it.get("name", ""), "状态": it.get("state", ""),
                         "灯": "⚪无K线"})
            continue
        df = pd.read_csv(p, parse_dates=["date"]).set_index("date")
        ind = compute_indicators(df)
        sigs = signal_frame(ind, thr)
        last = ind.iloc[-1]
        recent = sigs.tail(3).any()
        lamp = "⚪未到setup"
        if ind["dd120"].rolling(CONTEXT_LOOKBACK, min_periods=1).min().iloc[-1] <= thr:
            fired = [s for s in STYLES if s != "S0基准" and recent[s]]
            lamp = "🟢" + ",".join(fired) if fired else "🟡底部区观察"
        rows.append({
            "代码": sym, "名称": it.get("name", ""), "状态": it.get("state", ""),
            "收盘": round(float(last["close"]), 2), "K线至": str(ind.index[-1].date()),
            "距120日高": f"{last['dd120']:+.0%}", "RSI14": round(float(last["rsi"]), 1),
            "vs MA20": f"{last['close'] / last['ma20'] - 1:+.1%}",
            "R4线": f"{last['close'] / (last['min20c'] + TRAIL_ATR * last['atr']) - 1:+.1%}"
            if np.isfinite(last["atr"]) else "-",
            "灯": lamp,
        })
    print(pd.DataFrame(rows).to_markdown(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools", default="watch,neutral")
    parser.add_argument("--status", action="store_true", help="只打当前灯态，不回测")
    args = parser.parse_args()
    if args.status:
        status()
        return 0
    run_backtest([p.strip() for p in args.pools.split(",") if p.strip()])
    return 0


if __name__ == "__main__":
    sys.exit(main())
