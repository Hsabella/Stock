"""择时层基础版：每日 risk-on/off 两档 → 仓位系数序列。

信号源（全部为 T 日开盘前可得，时间对齐是本模块的生命线）：
1. 沪深300 趋势闸门：T-1 收盘 < MA200(T-1) → risk_off 倾向
2. 隔夜美股：T 日开盘前最近一根标普500 日线（美东 T-1 收盘，北京时间 T 日凌晨
   收完）跌幅 ≤ -1.5% → risk_off 倾向
（A50 期货夜盘/日韩早盘留作二期实盘增强，历史分段数据不可得，不进回测。）

合成：任一触发即 raw risk_off；状态切换需连续 confirm_days 日确认（防抖）。
输出：exposure 序列（risk_on=1.0 / risk_off=exposure_off），落
results/quant/timing_exposure.csv，供 overlay.py 叠加到组合日收益。

用法: python -m quant.backtest.timing [--start 2019-01-01]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from quant.config import load_config
from quant.data import warehouse

OUT_DIR = Path("results/quant")


def fetch_spx(refresh: bool = False) -> pd.Series:
    """标普500 日线收盘（新浪美股指数），缓存进 warehouse。"""
    cached = warehouse.load("spx_daily")
    if not cached.empty and not refresh:
        return cached.droplevel("instrument")["spx_close"]

    import akshare as ak

    df = ak.index_us_stock_sina(symbol=".INX")
    out = pd.DataFrame({
        "instrument": "SPX",
        "datetime": pd.to_datetime(df["date"]),
        "spx_close": df["close"].astype(float),
    })
    warehouse.append("spx_daily", out)
    return out.set_index("datetime")["spx_close"]


def load_hs300() -> pd.Series:
    """沪深300 收盘（qlib bin 包）。"""
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=load_config("data")["qlib"]["provider_uri"], region="cn")
    df = D.features(["SH000300"], ["$close"], start_time="2015-01-01")
    return df["$close"].droplevel(0)


def raw_risk_off(hs300: pd.Series, spx: pd.Series, ma_days: int, spx_threshold: float,
                 calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """逐个 A 股交易日 T 计算原始 risk_off 标志（只用 T-1 及更早信息）。"""
    ma = hs300.rolling(ma_days, min_periods=ma_days).mean()
    below_ma = (hs300 < ma)  # 按 T-1 收盘判定，映射到 T 日时向后挪一位
    below_ma_at_t = below_ma.reindex(calendar).shift(1)

    spx_ret = spx.pct_change(fill_method=None)
    # T 日开盘前最近一根美股日线：日期 < T 的最后一根（美东 T-1 的 bar 北京 T 凌晨收完）
    spx_at_t = pd.Series(
        spx_ret.reindex(pd.DatetimeIndex(sorted(set(spx_ret.index) | set(calendar)))).ffill(),
    ).reindex(calendar - pd.Timedelta(days=0))
    # 上面 ffill 会把 T 日自身（若与美股同日期字符串）也算进来；美股日期必然 < A 股使用日，
    # 因新浪美股 bar 的 date 是美东日期，T 日凌晨收盘的 bar 日期为 T-1，因此按 < T 对齐：
    spx_prev = pd.Series(index=calendar, dtype=float)
    spx_dates = spx_ret.dropna()
    pos = spx_dates.index.searchsorted(calendar) - 1  # 最后一个 < T 的位置
    valid = pos >= 0
    spx_prev[valid] = spx_dates.iloc[pos[valid]].to_numpy()

    df = pd.DataFrame({
        "below_ma200": below_ma_at_t.astype("boolean"),
        "spx_overnight": spx_prev,
    }, index=calendar)
    df["raw_risk_off"] = df["below_ma200"].fillna(False).astype(bool) | (df["spx_overnight"] <= spx_threshold)
    return df


def raw_ma_off(close: pd.Series, ma_days: int, calendar: pd.DatetimeIndex) -> pd.Series:
    """单指数纯 MA 腿的原始 risk_off（无隔夜美股触发源），供结构灯复用。"""
    empty_spx = pd.Series(dtype=float)
    df = raw_risk_off(close, empty_spx, ma_days, spx_threshold=-1.0, calendar=calendar)
    return df["raw_risk_off"]


def debounce(raw: pd.Series, confirm_days: int) -> pd.Series:
    """状态切换需连续 confirm_days 日反向信号确认。"""
    state = False  # 初始 risk_on
    streak = 0
    out = []
    for v in raw.fillna(False):
        if bool(v) != state:
            streak += 1
            if streak >= confirm_days:
                state = bool(v)
                streak = 0
        else:
            streak = 0
        out.append(state)
    return pd.Series(out, index=raw.index, name="risk_off")


def build(start: str, exposure_off: float | None = None) -> pd.DataFrame:
    cfg = load_config("backtest")["timing"]
    if exposure_off is not None:
        cfg = {**cfg, "exposure_off": exposure_off}

    hs300 = load_hs300()
    calendar = hs300.index[hs300.index >= start]
    spx = fetch_spx()

    df = raw_risk_off(hs300, spx, cfg["ma_days"], cfg["spx_threshold"], calendar)
    df["risk_off"] = debounce(df["raw_risk_off"], cfg["confirm_days"])
    df["exposure"] = df["risk_off"].map({False: 1.0, True: float(cfg["exposure_off"])})
    df.index.name = "datetime"
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2019-01-01")
    args = parser.parse_args()

    df = build(args.start)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "timing_exposure.csv"
    df.to_csv(out)
    on_rate = (~df["risk_off"]).mean()
    print(f"[timing] {df.index[0].date()} → {df.index[-1].date()}, risk_on 占比 {on_rate:.1%}, 已写 {out}")
    print(df["risk_off"].astype(int).diff().abs().sum(), "次状态切换")
    return 0


if __name__ == "__main__":
    sys.exit(main())
