"""择时叠加：exposure × 组合日收益，近似扣调仓成本，出对比报告。

qlib 无每日仓位系数机制（risk_degree 是静态常数），故择时与 qlib 解耦：
overlaid_t = exposure_t × net_t − |Δexposure_t| × overlay_cost，现金部分记 0。

同时对基准做同样叠加（长窗口 sanity：择时规则本身在指数上是否改善回撤），
并输出 exposure_off ∈ {0.3, 0.5, 0.7} 敏感性。

用法:
    python -m quant.backtest.overlay --daily results/quant/bt_lgb_rolling_daily.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from quant.backtest.report import perf_stats
from quant.backtest.timing import build as build_timing
from quant.config import load_config

OUT_DIR = Path("results/quant")


def apply_overlay(net: pd.Series, exposure: pd.Series, overlay_cost: float) -> pd.Series:
    exp = exposure.reindex(net.index).ffill().fillna(1.0)
    switch_cost = exp.diff().abs().fillna(0) * overlay_cost
    return exp * net - switch_cost


def compare_table(variants: dict[str, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame({name: perf_stats(r.dropna()) for name, r in variants.items()})


def window_table(variants: dict[str, pd.Series], windows: dict[str, tuple[str, str]]) -> pd.DataFrame:
    rows = {}
    for wname, (lo, hi) in windows.items():
        row = {}
        for vname, r in variants.items():
            seg = r.loc[lo:hi]
            if len(seg) == 0:
                row[vname] = None
                continue
            nav = (1 + seg).cumprod()
            row[vname] = float((nav / nav.cummax() - 1).min())  # 窗口内最大回撤
        rows[wname] = row
    return pd.DataFrame(rows).T


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily", required=True, type=Path, help="run_backtest 产出的 *_daily.csv")
    args = parser.parse_args()

    cfg = load_config("backtest")["timing"]
    daily = pd.read_csv(args.daily, index_col=0, parse_dates=True)
    net = daily["return"] - daily["cost"]
    bench = daily["bench"]

    variants: dict[str, pd.Series] = {"策略(无择时)": net, "基准(中证800)": bench}
    for off in (0.3, 0.5, 0.7):
        timing = build_timing(start=str(net.index[0].date()), exposure_off=off)
        overlaid = apply_overlay(net, timing["exposure"], cfg["overlay_cost"])
        variants[f"策略+择时(off={off})"] = overlaid
        if off == cfg["exposure_off"]:
            variants["基准+择时(sanity)"] = apply_overlay(bench, timing["exposure"], cfg["overlay_cost"])

    total = compare_table(variants)
    # 压力窗口：测试段内已知的大盘回撤期（不足一段则自动跳过）
    windows = {
        "2024-01~02 流动性危机": ("2024-01-01", "2024-02-29"),
        "2024-10 脉冲后回落": ("2024-10-08", "2024-12-31"),
        "2025 全年": ("2025-01-01", "2025-12-31"),
    }
    wt = window_table(variants, windows)

    out_md = OUT_DIR / f"overlay_{args.daily.stem.replace('_daily', '')}.md"
    out_md.write_text("\n".join([
        "# 择时叠加对比", "",
        f"- 规则: 沪深300<MA{cfg['ma_days']} OR 隔夜SPX≤{cfg['spx_threshold']:.1%}，"
        f"连续 {cfg['confirm_days']} 日确认；risk_off 仓位={cfg['exposure_off']}",
        "", "## 全区间（成本后）", "", total.round(4).to_markdown(), "",
        "## 压力窗口最大回撤", "", wt.round(4).to_markdown(), "",
    ]))
    print(total.round(4))
    print(f"[overlay] 报告已写 {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
