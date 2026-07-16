"""结构灯敏感性回测：中证1000 / 科创50 上 MA+确认 择时规则的纯保护力评估。

背景：晨间红绿灯只盯沪深300，与用户题材小票持仓错配。上线"结构灯"前按项目纪律
先回测——规则与主灯同构（T-1 收盘 vs MA，连续 confirm 日确认，T 日生效），只用
MA 腿不接隔夜美股。exposure 用 1/0（评估灯本身的保护力，不与 0.7 折扣混淆）。

验收门槛（三项同时满足才可给操作建议，否则该灯降级为"仅展示"）：
1. MDD 压缩 ≥ 25%（策略 MDD 相对买入持有）
2. 状态切换 ≤ 6 次/年（防 MA 附近来回打脸）
3. 踏空成本 ≤ 3pct 年化（买入持有年化 - 策略年化）

用法: .venv-quant/bin/python -m quant.research.timing_lights
输出: results/quant/timing_lights_sensitivity.csv + stdout markdown 表
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from quant.backtest.timing import debounce, raw_ma_off
from quant.data.index_daily import load_closes

OUT_CSV = Path("results/quant/timing_lights_sensitivity.csv")
ANN_DAYS = 244
SWITCH_COST = 0.001  # |Δexposure| 的近似调仓成本

GRID_MA = [120, 200]
GRID_CONFIRM = [2, 3]
LIGHTS = [("SH000852", "中证1000"), ("SH000688", "科创50")]
# 专项压力段：2022 熊市 / 2024-01 微盘崩盘 / 2025H2 起权重股行情
SEGMENTS = [("2022熊市", "2022-01-01", "2022-12-31"),
            ("2024-01崩盘", "2024-01-01", "2024-02-08"),
            ("2025H2-26", "2025-07-01", "2099-01-01")]


def max_drawdown(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1).min())


def episode_false_rate(off: pd.Series, close: pd.Series) -> float:
    """假信号率：off 区间内指数实际收涨的区间占比（躲了个寂寞）。"""
    episodes, start = [], None
    for dt, v in off.items():
        if v and start is None:
            start = dt
        elif not v and start is not None:
            episodes.append((start, dt))
            start = None
    if start is not None:
        episodes.append((start, off.index[-1]))
    if not episodes:
        return float("nan")
    rose = sum(1 for s, e in episodes if close.loc[e] > close.loc[s])
    return rose / len(episodes)


def evaluate(close: pd.Series, ma_days: int, confirm_days: int) -> dict:
    """close: 单指数收盘序列（无 NaN）。评估段 = MA warmup 之后的全部交易日。"""
    calendar = close.index[ma_days:]  # 前 ma_days 根做 warmup
    raw = raw_ma_off(close, ma_days, calendar)
    off = debounce(raw, confirm_days)
    exposure = (~off).astype(float)

    ret = close.pct_change().reindex(calendar).fillna(0.0)
    cost = exposure.diff().abs().fillna(0.0) * SWITCH_COST
    strat_ret = ret * exposure - cost

    nav_bh = (1 + ret).cumprod()
    nav_st = (1 + strat_ret).cumprod()
    years = len(calendar) / ANN_DAYS
    ann_bh = nav_bh.iloc[-1] ** (1 / years) - 1
    ann_st = nav_st.iloc[-1] ** (1 / years) - 1
    mdd_bh, mdd_st = max_drawdown(nav_bh), max_drawdown(nav_st)

    row = {
        "start": str(calendar[0].date()), "end": str(calendar[-1].date()),
        "ann_bh": ann_bh, "ann_strat": ann_st,
        "lag_cost": ann_bh - ann_st,                      # 踏空成本（负=反而增强）
        "mdd_bh": mdd_bh, "mdd_strat": mdd_st,
        "mdd_compress": 1 - mdd_st / mdd_bh if mdd_bh < 0 else float("nan"),
        "switches_py": float(off.astype(int).diff().abs().sum()) / years,
        "off_ratio": float(off.mean()),
        "false_rate": episode_false_rate(off, close),
    }
    for name, s, e in SEGMENTS:
        seg = calendar[(calendar >= s) & (calendar <= e)]
        if len(seg) < 10:
            row[f"seg_{name}"] = ""
            continue
        seg_bh = max_drawdown(nav_bh.loc[seg] / nav_bh.loc[seg].iloc[0])
        seg_st = max_drawdown(nav_st.loc[seg] / nav_st.loc[seg].iloc[0])
        row[f"seg_{name}"] = f"{seg_bh:.1%}→{seg_st:.1%}"
    row["pass_gates"] = (row["mdd_compress"] >= 0.25
                         and row["switches_py"] <= 6
                         and row["lag_cost"] <= 0.03)
    return row


def main() -> int:
    closes = load_closes()
    rows = []
    for inst, cname in LIGHTS:
        if inst not in closes.columns:
            print(f"[timing_lights] warehouse 缺 {inst}，先跑 python -m quant.data.index_daily")
            return 1
        close = closes[inst].dropna()
        for ma in GRID_MA:
            for confirm in GRID_CONFIRM:
                row = {"index": inst, "name": cname, "ma": ma, "confirm": confirm}
                row.update(evaluate(close, ma, confirm))
                rows.append(row)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    show = df[["name", "ma", "confirm", "start", "ann_bh", "ann_strat", "lag_cost",
               "mdd_bh", "mdd_strat", "mdd_compress", "switches_py", "false_rate",
               "seg_2022熊市", "seg_2024-01崩盘", "seg_2025H2-26", "pass_gates"]].copy()
    for col in ["ann_bh", "ann_strat", "lag_cost", "mdd_bh", "mdd_strat",
                "mdd_compress", "false_rate"]:
        show[col] = show[col].map(lambda v: f"{v:.1%}" if pd.notna(v) else "-")
    show["switches_py"] = show["switches_py"].round(1)
    print(show.to_markdown(index=False))
    print(f"\n[timing_lights] 已写 {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
