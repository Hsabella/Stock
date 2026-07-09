"""晨间一句话仓位建议：择时层（M4 已回测验证）的实盘出口。

数据不依赖 qlib bin 包的更新节奏——大盘和美股都现拉 akshare：
- 沪深300 日线（新浪指数接口，取到 T-1 收盘）→ 是否跌破 MA200
- 标普500 日线（新浪美股接口，美东 T-1 的 bar 北京时间今晨已收完）→ 隔夜跌幅
与回测同一套规则/防抖逻辑（复用 timing.raw_risk_off/debounce），保证"实盘看到的
就是回测验证过的"。

用法: python -m quant.live.daily_brief    # 每天开盘前跑，输出今日建议仓位
"""
from __future__ import annotations

import sys

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


def main() -> int:
    b = brief()
    tone = "🟢 正常" if b["state"] == "risk_on" else "🔴 防守"
    print(f"[{b['date']}] 今日风险状态: {tone} → 建议股票仓位上限 {b['exposure']:.0%}")
    print(f"  沪深300 昨收 {b['hs300_close']:.0f}, 相对 MA200 {b['hs300_vs_ma200']:+.1%}"
          f"{'（跌破均线）' if b['hs300_vs_ma200'] < 0 else ''}")
    print(f"  隔夜标普500 {b['spx_overnight']:+.2%}"
          f"{'（触发防守阈值）' if b['spx_overnight'] <= -0.015 else ''}")
    if b["raw_today"] != (b["state"] == "risk_off"):
        print("  ⚠️ 原始信号与当前状态不一致（防抖确认中），连续第二天出现将切换状态")
    return 0


if __name__ == "__main__":
    sys.exit(main())
