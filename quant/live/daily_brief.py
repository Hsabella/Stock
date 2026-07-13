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

from quant.backtest.timing import debounce, raw_risk_off
from quant.config import load_config


def fetch_live_series() -> tuple[pd.Series, pd.Series]:
    import akshare as ak

    hs = ak.stock_zh_index_daily(symbol="sh000300")
    hs300 = pd.Series(hs["close"].to_numpy(), index=pd.to_datetime(hs["date"]), dtype=float)
    us = ak.index_us_stock_sina(symbol=".INX")
    spx = pd.Series(us["close"].to_numpy(), index=pd.to_datetime(us["date"]), dtype=float)
    return hs300.sort_index(), spx.sort_index()


def brief(today: pd.Timestamp | None = None) -> dict:
    cfg = load_config("backtest")["timing"]
    hs300, spx = fetch_live_series()
    today = today or (hs300.index[-1] + pd.offsets.BDay(1))  # 数据最新日的下一交易日=今天

    calendar = pd.DatetimeIndex([*hs300.index[hs300.index >= "2019-01-01"], today]).unique().sort_values()
    df = raw_risk_off(hs300, spx, cfg["ma_days"], cfg["spx_threshold"], calendar)
    df["risk_off"] = debounce(df["raw_risk_off"], cfg["confirm_days"])

    t = df.index[-1]
    ma = hs300.rolling(cfg["ma_days"]).mean().iloc[-1]
    state = "risk_off" if df["risk_off"].iloc[-1] else "risk_on"
    return {
        "date": str(t.date()),
        "state": state,
        "exposure": cfg["exposure_off"] if state == "risk_off" else 1.0,
        "hs300_close": float(hs300.iloc[-1]),
        "hs300_vs_ma200": float(hs300.iloc[-1] / ma - 1),
        "spx_overnight": float(df["spx_overnight"].iloc[-1]),
        "raw_today": bool(df["raw_risk_off"].iloc[-1]),
    }


def format_brief(b: dict) -> str:
    tone = "🟢 正常" if b["state"] == "risk_on" else "🔴 防守"
    lines = [
        f"[{b['date']}] 今日风险状态: {tone} → 建议股票仓位上限 {b['exposure']:.0%}",
        f"  沪深300 昨收 {b['hs300_close']:.0f}, 相对 MA200 {b['hs300_vs_ma200']:+.1%}"
        f"{'（跌破均线）' if b['hs300_vs_ma200'] < 0 else ''}",
        f"  隔夜标普500 {b['spx_overnight']:+.2%}"
        f"{'（触发防守阈值）' if b['spx_overnight'] <= -0.015 else ''}",
    ]
    if b["raw_today"] != (b["state"] == "risk_off"):
        lines.append("  ⚠️ 原始信号与当前状态不一致（防抖确认中），连续第二天出现将切换状态")
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
        else:
            # 失败不覆盖 latest.md——保留最后一次有效建议（红灯日被"生成失败"顶掉=真金白银的风险）
            (args.out_dir / f"brief_{stamp}_failed.md").write_text(text + "\n")
    return 0 if b else 1


if __name__ == "__main__":
    sys.exit(main())
