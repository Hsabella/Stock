"""晨间一句话仓位建议：择时层（M4 已回测验证）的实盘出口。

数据不依赖 qlib bin 包的更新节奏——大盘和美股都现拉 akshare：
- 沪深300 日线（新浪指数接口，取到 T-1 收盘）→ 是否跌破 MA200
- 标普500 日线（新浪美股接口，美东 T-1 的 bar 北京时间今晨已收完）→ 隔夜跌幅
与回测同一套规则/防抖逻辑（复用 timing.raw_risk_off/debounce），保证"实盘看到的
就是回测验证过的"。

用法:
    python -m quant.live.daily_brief                        # 打印到终端
    python -m quant.live.daily_brief --out-dir results/brief  # 同时写 brief_<日期>.md + latest.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from quant.backtest.timing import debounce, raw_ma_off, raw_risk_off
from quant.config import load_config


def fetch_index(symbol: str) -> pd.Series:
    """新浪指数日线收盘序列（取到 T-1 收盘）。"""
    import akshare as ak

    df = ak.stock_zh_index_daily(symbol=symbol)
    s = pd.Series(df["close"].to_numpy(), index=pd.to_datetime(df["date"]), dtype=float)
    return s.sort_index()


def fetch_live_series() -> tuple[pd.Series, pd.Series]:
    import akshare as ak

    hs300 = fetch_index("sh000300")
    us = ak.index_us_stock_sina(symbol=".INX")
    spx = pd.Series(us["close"].to_numpy(), index=pd.to_datetime(us["date"]), dtype=float)
    return hs300, spx.sort_index()


def _streak(raw: pd.Series) -> int:
    """原始信号当前值已连续几天（供"确认进度"展示）。"""
    streak = 1
    for v in raw.iloc[:-1][::-1]:
        if bool(v) != bool(raw.iloc[-1]):
            break
        streak += 1
    return streak


def lights_status(today: pd.Timestamp) -> list[dict]:
    """结构灯：配置驱动的单指数 MA 腿规则（参数经 timing_lights 回测定案）。"""
    out = []
    for cfg in load_config("backtest").get("timing_lights", []):
        close = fetch_index(cfg["symbol"])
        calendar = pd.DatetimeIndex(
            [*close.index[close.index >= "2019-01-01"], today]).unique().sort_values()
        raw = raw_ma_off(close, cfg["ma_days"], calendar)
        off = debounce(raw, cfg["confirm_days"])
        ma = close.rolling(cfg["ma_days"]).mean()
        out.append({
            "name": cfg["name"], "symbol": cfg["symbol"],
            "state": "risk_off" if off.iloc[-1] else "risk_on",
            "close": float(close.iloc[-1]),
            "vs_ma": float(close.iloc[-1] / ma.iloc[-1] - 1),
            "ma_days": int(cfg["ma_days"]), "confirm_days": int(cfg["confirm_days"]),
            "mode": cfg["mode"], "advice_off": cfg.get("advice_off", ""),
            "raw_today": bool(raw.iloc[-1]), "raw_streak": _streak(raw),
        })
    return out


def brief(today: pd.Timestamp | None = None) -> dict:
    cfg = load_config("backtest")["timing"]
    hs300, spx = fetch_live_series()
    today = today or (hs300.index[-1] + pd.offsets.BDay(1))  # 数据最新日的下一交易日=今天

    calendar = pd.DatetimeIndex([*hs300.index[hs300.index >= "2019-01-01"], today]).unique().sort_values()
    df = raw_risk_off(hs300, spx, cfg["ma_days"], cfg["spx_threshold"], calendar)
    df["risk_off"] = debounce(df["raw_risk_off"], cfg["confirm_days"])

    t = df.index[-1]
    ma_series = hs300.rolling(cfg["ma_days"]).mean()
    state = "risk_off" if df["risk_off"].iloc[-1] else "risk_on"
    raw = df["raw_risk_off"]
    out = {
        "date": str(t.date()),
        "state": state,
        "exposure": cfg["exposure_off"] if state == "risk_off" else 1.0,
        "hs300_close": float(hs300.iloc[-1]),
        "hs300_vs_ma200": float(hs300.iloc[-1] / ma_series.iloc[-1] - 1),
        "ma200": float(ma_series.iloc[-1]),
        "vs_ma200_prev5": float(hs300.iloc[-6] / ma_series.iloc[-6] - 1),
        "hs300_5d_chg": float(hs300.iloc[-1] / hs300.iloc[-6] - 1),
        "spx_overnight": float(df["spx_overnight"].iloc[-1]),
        "spx_threshold": float(cfg["spx_threshold"]),
        "confirm_days": int(cfg["confirm_days"]),
        "exposure_off": float(cfg["exposure_off"]),
        "raw_today": bool(raw.iloc[-1]),
        "raw_streak": _streak(raw),
    }
    # 结构灯/持仓段独立降级：任何异常只损失本段，绝不拖垮主灯（红灯日被顶掉=真金白银风险）
    try:
        out["lights"] = lights_status(today)
    except Exception as e:
        out["lights_error"] = f"{type(e).__name__}: {e}"
    try:
        from quant.live.portfolio import holdings_health

        out["holdings"] = holdings_health(today=today)
    except Exception as e:
        out["holdings_error"] = f"{type(e).__name__}: {e}"
    try:
        from quant.live.portfolio import structure_drift

        out["structure"] = structure_drift(health=out.get("holdings"))
    except Exception as e:
        out["structure_error"] = f"{type(e).__name__}: {e}"
    try:
        from quant.live.portfolio import reentry_watch

        out["reentry"] = reentry_watch(today=today, lights=out.get("lights"))
    except Exception as e:
        out["reentry_error"] = f"{type(e).__name__}: {e}"
    try:
        from quant.live.portfolio import etf_watch

        out["etf_watch"] = etf_watch(today=today)
    except Exception as e:
        out["etf_watch_error"] = f"{type(e).__name__}: {e}"
    try:
        from quant.research.stock_lights import lamp_rows

        out["stock_lamps"] = lamp_rows(today=today)
    except Exception as e:
        out["stock_lamps_error"] = f"{type(e).__name__}: {e}"
    return out


def _csi1000_light(b: dict) -> dict | None:
    return next((lt for lt in b.get("lights") or [] if lt["symbol"] == "sh000852"), None)


def _format_lights(b: dict) -> list[str]:
    if "lights_error" in b:
        return ["", f"⚠️ 结构灯生成失败: {b['lights_error']}"]
    lights = b.get("lights")
    if not lights:
        return []
    lines = ["", "【结构灯】持仓风格监控（参数回测定案: docs/quant/timing_lights_report.md）"]
    for lt in lights:
        icon = "🟢" if lt["state"] == "risk_on" else "🔴"
        seg = f"  {lt['name']} {icon} 昨收 {lt['close']:.0f}  MA{lt['ma_days']} {lt['vs_ma']:+.1%}"
        if lt["mode"] == "display_only":
            seg += "（仅展示，未过回测验收线）"
        elif lt["state"] == "risk_off":
            seg += f" → {lt['advice_off']}"
        if lt["raw_today"] != (lt["state"] == "risk_off"):
            left = max(1, lt["confirm_days"] - lt["raw_streak"])
            flip = "🔴" if lt["state"] == "risk_on" else "🟢"
            seg += f"（翻转信号第 {lt['raw_streak']} 天，再 {left} 天确认转 {flip}）"
        lines.append(seg)
    return lines


def _format_holdings(b: dict) -> list[str]:
    if "holdings_error" in b:
        return ["", f"⚠️ 持仓健康度生成失败: {b['holdings_error']}"]
    h = b.get("holdings")
    if not h or not h["rows"]:
        return []
    head = "【持仓健康度】现价=缓存收盘"
    if h["signal_date"]:
        head += f" · 引擎信号={h['signal_date']}"
    if h["any_stale"]:
        head += " ⚠️ 部分K线陈旧"
    lines = ["", head]
    for r in h["rows"]:
        if r.get("error"):
            lines.append(f"  {r['name']} {r['symbol']} ⚠️ {r['error']}")
            continue
        stop = (f"止损线 {r['stop_line']:.2f} 已破❗"
                if r["broken"] else f"止损线 {r['stop_line']:.2f}（距离 {r['dist_pct']:+.1%}）")
        if r.get("degraded"):
            stop += f" ⚠️({r['degrade_reason']}, 仅硬止损)"
        trend = (f"ATR趋势线 {r['trend_line']:.2f} "
                 f"{'上方🟢' if r['above_trend'] else '下方🔴'}")
        tag = "[ETF] " if r.get("etf") else ""
        seg = (f"  {r['name']} {r['symbol']} {tag}{r['close']:.2f}/{r['entry']:.2f} "
               f"{r['pnl_pct']:+.1%} | {stop} | {trend}")
        if r.get("arm_suspect"):
            seg += f" ⚠️左臂标注存疑(入场时距120日高 {r['dd_at_entry']:+.0%})"
        if r["decision"] in ("REDUCE", "DROP", "STOP", "TAKE"):
            seg += f" | 引擎:{r['decision']}"
        if r["risks"]:
            seg += " ⚠️" + "·".join(r["risks"])
        lines.append(seg)
    return lines


def _format_structure(b: dict) -> list[str]:
    if "structure_error" in b:
        return ["", f"⚠️ 结构修正跟踪生成失败: {b['structure_error']}"]
    s = b.get("structure")
    if not s:
        return []
    if s.get("error"):
        return ["", f"⚠️ 结构修正跟踪: {s['error']}"]
    t = s["target"]
    if s["compliant"]:
        return ["", f"【结构修正跟踪】✅ 已达标（核心ETF {s['etf_pct']:.0%} · "
                    f"卫星 {s['stock_count']} 只 · 现金 {s['cash_pct']:.0%}）"]
    lines = [
        "",
        f"【结构修正跟踪】目标: 核心ETF {t['core_etf_lo']:.0%}-{t['core_etf_hi']:.0%}"
        f" · 卫星≤{t['max_count']}只×≤{t['max_pct']:.0%} · 余为现金",
        f"  当前: 总资产 {s['total'] / 1e4:.1f}万 = 现金 {s['cash_pct']:.0%}"
        f" + 个股 {s['stock_count']}只 {s['stock_pct']:.0%}"
        f" + 核心ETF {s['etf_pct']:.0%}",
    ]
    lines += [f"  待执行{i}: {a}" for i, a in enumerate(s["actions"], 1)]
    return lines


def _mark(ok: bool | None) -> str:
    return "✅" if ok is True else "⚠️" if ok is None else "❌"


def _format_reentry(b: dict) -> list[str]:
    if "reentry_error" in b:
        return ["", f"⚠️ 回补观察生成失败: {b['reentry_error']}"]
    w = b.get("reentry")
    if not w or not w["rows"]:
        return []
    head = "【回补观察】止损卖出票的接回条件（三条全✅且过冷静期才考虑，接回按单票≤10%）"
    if w["any_stale"]:
        head += " ⚠️ 部分K线陈旧"
    lines = ["", head]
    for r in w["rows"]:
        if r.get("error"):
            lines.append(f"  {r['name']} {r['symbol']} ⚠️ {r['error']}")
            continue
        price = (f"卖{r['exit_price']:.2f} 现{r['close']:.2f}" if r.get("exit_price")
                 else f"现{r['close']:.2f}")
        cool = ("冷静期已过" if r["cooldown_done"]
                else f"冷静期 {r['cooldown_elapsed']}/{r['cooldown_total']}")
        seg = (f"  {r['name']} {r['symbol']} {price} | {cool}"
               f" | ①{r['light_name'] or '结构灯'}{_mark(r['light_ok'])}"
               f" ②站回趋势线{r['trail']:.2f}{_mark(r['trend_ok'])}({r['trail_dist']:+.1%})"
               f" ③引擎{r['decision'] or '无信号'}"
               f"{'·板块弱' if r.get('sector_weak') else ''}{_mark(r['engine_ok'])}"
               f" → {r['met']}/3")
        if r["ready"]:
            seg += " ✅ 条件齐备，可按新仓规则考虑接回"
        lines.append(seg)
    return lines


def _format_etf_watch(b: dict) -> list[str]:
    if "etf_watch_error" in b:
        return ["", f"⚠️ ETF观察生成失败: {b['etf_watch_error']}"]
    w = b.get("etf_watch")
    if not w or not w["rows"]:
        return []
    head = "【ETF观察】趋势读数，无买卖信号（抄底灯未在 ETF 上回测，不适用）"
    if w["any_stale"]:
        head += " ⚠️ 部分K线陈旧"
    lines = ["", head]
    for r in w["rows"]:
        if r.get("error"):
            lines.append(f"  {r['name']} {r['symbol']} ⚠️ {r['error']}")
            continue
        ma = (f"MA200 {r['vs_ma200']:+.1%}" if r["vs_ma200"] is not None
              else "MA200 -（历史不足）")
        trend = "上方🟢" if r["above_trail"] else "下方🔴"
        lines.append(f"  {r['name']} {r['symbol']} 收{r['close']:.3f}({r['chg_1d']:+.1%})"
                     f" 距120日高{r['dd120']:+.0%} {ma}"
                     f" ATR趋势线{r['trail']:.3f}{trend}")
    return lines


def _format_stock_lamps(b: dict) -> list[str]:
    if "stock_lamps_error" in b:
        return ["", f"⚠️ 个股抄底灯生成失败: {b['stock_lamps_error']}"]
    sl = b.get("stock_lamps")
    if not sl or not sl["rows"]:
        return []
    rows = [r for r in sl["rows"] if not r.get("error")]
    errors = len(sl["rows"]) - len(rows)
    # 持仓票不列入：灯只管新钱入场，防止被误读成补仓依据
    fired = [r for r in rows if r["lamp"] == "fire" and r["state"] != "HELD"]
    zone = sum(1 for r in rows if r["lamp"] == "watch")
    head = (f"【个股抄底灯】setup=距120日高≤{sl['thr']:.0%} · 左侧L6/L1"
            f"（回测定案: docs/quant/stock_lights_report.md）")
    if sl["any_stale"]:
        head += " ⚠️ 部分K线陈旧"
    lines = ["", head]
    c1k = _csi1000_light(b)
    if c1k is None:
        lines.append("  窗口: ⚠️ 中证1000灯缺失——触发仅记录")
    elif c1k["state"] == "risk_off":
        lines.append("  窗口: 中证1000 🔴 恐慌期=左侧超额最大 → 可按纪律执行"
                     "（单票半仓≤5% · 止损=入场价-3×ATR14 · 并发新仓≤3只）")
    else:
        lines.append("  窗口: 中证1000 🟢 —— 绿灯期独跌票无超额，以下触发仅记录、不建议入场")
    for r in fired:
        note = "（EXITED，接回归回补观察）" if r["state"] == "EXITED" else ""
        names = "+".join(s["signal"] for s in r["signals"])
        lines.append(f"  🟢 {r['name']} {r['symbol']} {names} "
                     f"收盘{r['close']:.2f} 距高{r['dd120']:+.0%} RSI{r['rsi']:.0f}{note}")
    if not fired:
        lines.append("  今日无触发")
    tail = f"  （🟡底部区观察 {zone} 只；持仓票不列入——灯只管新钱，不是补仓依据）"
    if errors:
        tail += f" ⚠️ {errors}只K线不足"
    lines.append(tail)
    return lines


def format_brief(b: dict) -> str:
    green = b["state"] == "risk_on"
    tone = "🟢 正常" if green else "🔴 防守"
    narrowing = b["hs300_vs_ma200"] < b["vs_ma200_prev5"]
    lines = [
        f"[{b['date']}] 今日风险状态: {tone} → 建议股票仓位上限 {b['exposure']:.0%}",
        "",
        f"【大盘趋势】沪深300 昨收 {b['hs300_close']:.0f}",
        f"  相对 MA200({b['ma200']:.0f}): {b['hs300_vs_ma200']:+.1%}"
        f"（5 个交易日前 {b['vs_ma200_prev5']:+.1%}，缓冲{'收窄中' if narrowing else '加厚中'}）",
        f"  近 5 个交易日大盘 {b['hs300_5d_chg']:+.1%}",
    ]
    if b["hs300_vs_ma200"] >= 0:
        fall = abs(b["ma200"] / b["hs300_close"] - 1)
        lines.append(f"  → 距离防守线: 再跌 {fall:.1%} 即跌破 MA200，"
                     f"之后还需连续 {b['confirm_days']} 日确认才转 🔴")
    else:
        rise = b["ma200"] / b["hs300_close"] - 1
        lines.append(f"  → 距离恢复线: 需涨 {rise:+.1%} 收复 MA200，"
                     f"之后还需连续 {b['confirm_days']} 日确认才转 🟢")
    lines += _format_lights(b)
    hit = b["spx_overnight"] <= b["spx_threshold"]
    lines += [
        "",
        f"【隔夜美股】标普500 {b['spx_overnight']:+.2%}"
        f"（防守触发线 {b['spx_threshold']:.1%}，{'⚠️ 已触发' if hit else '未触发'}）",
    ]
    if b["raw_today"] != (not green):
        left = max(1, b["confirm_days"] - b["raw_streak"])
        flip, verb = ("🔴", "防守信号已触发") if green else ("🟢", "恢复信号已出现")
        lines += ["", f"⚠️ {verb}（第 {b['raw_streak']} 天），再连续确认 {left} 天即转 {flip}"
                      f"{'——可提前想好减仓动作' if green else ''}"]
    lines += _format_holdings(b)
    lines += _format_structure(b)
    lines += _format_reentry(b)
    lines += _format_etf_watch(b)
    lines += _format_stock_lamps(b)
    lines += [
        "",
        f"【规则】趋势级过滤器，单日涨跌不触发：跌破 MA200 或 隔夜标普≤{b['spx_threshold']:.1%}，"
        f"连续 {b['confirm_days']} 日确认才转 🔴（仓位上限降至 {b['exposure_off']:.0%}），反向同理转回 🟢。",
        "  回测参考: 2024-01 崩盘段该规则把 -12.6% 回撤压到 -3.9%（docs/quant/phase1_report.md）",
    ]
    return "\n".join(lines)


def append_signal_log(path: Path, b: dict) -> None:
    """攒实盘信号流水（date,state,...），为红绿灯 track record 积累对账数据。"""
    line = f"{b['date']},{b['state']},{int(b['raw_today'])},{b['exposure']:.2f},{b['hs300_close']:.1f}\n"
    if not path.exists():
        path.write_text("date,state,raw_today,exposure,hs300_close\n" + line)
        return
    content = path.read_text()
    if f"{b['date']}," not in content:  # 同日重跑不重复记
        path.write_text(content + line)


def append_lights_log(path: Path, b: dict) -> None:
    """结构灯信号流水（长表 date,light,state,raw_today,close,ma_days），同日重跑去重。"""
    lights = b.get("lights") or []
    if not lights:
        return
    rows = "".join(
        f"{b['date']},{lt['symbol']},{lt['state']},{int(lt['raw_today'])},"
        f"{lt['close']:.1f},{lt['ma_days']}\n" for lt in lights)
    if not path.exists():
        path.write_text("date,light,state,raw_today,close,ma_days\n" + rows)
        return
    content = path.read_text()
    if f"{b['date']}," not in content:
        path.write_text(content + rows)


def append_stock_lamps_log(path: Path, b: dict) -> None:
    """个股抄底灯流水：每 (信号日,代码,信号) 只落一行——灯展示 FIRE_WINDOW 天，
    按晨报日去重会把同一事件记 3 遍、对账 3 倍计数。仅记非持仓的 🟢 触发。"""
    sl = b.get("stock_lamps") or {}
    fired = [r for r in (sl.get("rows") or [])
             if not r.get("error") and r.get("lamp") == "fire" and r.get("state") != "HELD"]
    if not fired:
        return
    c1k = _csi1000_light(b)
    header = "signal_date,symbol,signal,wl_state,close,dd120,rsi,csi1000,brief_date\n"
    content = path.read_text() if path.exists() else header
    added = ""
    for r in fired:
        for s in r["signals"]:
            key = f"{s['signal_date']},{r['symbol']},{s['signal']},"
            if key not in content and key not in added:
                added += (f"{key}{r['state']},{r['close']:.2f},{r['dd120']:.3f},"
                          f"{r['rsi']:.1f},{c1k['state'] if c1k else ''},{b['date']}\n")
    if added or not path.exists():
        path.write_text(content + added)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    try:
        b = brief()
        text = format_brief(b)
    except Exception as e:  # cron 场景下失败也要落盘可见
        b = None
        text = f"⚠️ 晨间建议生成失败: {type(e).__name__}: {e}\n（数据源可能暂不可用，可稍后手动重跑）"

    if args.out_dir:
        alert = args.out_dir / "data_health_alert.txt"  # weekly_data.sh 失败时写入，成功后清除
        if alert.exists():
            text += f"\n⚠️ {alert.read_text().strip()}"
    text += f"\n（生成于 {pd.Timestamp.now():%Y-%m-%d %H:%M}）"

    print(text)
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = b["date"].replace("-", "") if b else pd.Timestamp.now().strftime("%Y%m%d")
        if b:
            (args.out_dir / f"brief_{stamp}.md").write_text(text + "\n")
            (args.out_dir / "latest.md").write_text(text + "\n")
            append_signal_log(args.out_dir / "signal_log.csv", b)
            append_lights_log(args.out_dir / "lights_log.csv", b)
            append_stock_lamps_log(args.out_dir / "stock_lamps_log.csv", b)
        else:
            # 失败不覆盖 latest.md——保留最后一次有效建议（红灯日被"生成失败"顶掉=真金白银的风险）
            (args.out_dir / f"brief_{stamp}_failed.md").write_text(text + "\n")
    return 0 if b else 1


if __name__ == "__main__":
    sys.exit(main())
