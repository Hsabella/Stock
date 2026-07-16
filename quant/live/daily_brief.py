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
    # 结构灯独立降级：任何异常只损失本段，绝不拖垮主灯（红灯日被顶掉=真金白银风险）
    try:
        out["lights"] = lights_status(today)
    except Exception as e:
        out["lights_error"] = f"{type(e).__name__}: {e}"
    return out


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
        else:
            # 失败不覆盖 latest.md——保留最后一次有效建议（红灯日被"生成失败"顶掉=真金白银的风险）
            (args.out_dir / f"brief_{stamp}_failed.md").write_text(text + "\n")
    return 0 if b else 1


if __name__ == "__main__":
    sys.exit(main())
