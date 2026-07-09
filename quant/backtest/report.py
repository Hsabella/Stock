"""回测绩效报表：净值/夏普/回撤/分年度/换手，输出 md + csv + png。

输入是 qlib backtest_daily 的 report_normal DataFrame（列含 return/bench/cost/turnover），
所有"净"口径 = return - cost。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ANN = 52 * 5 / 5  # 交易日年化基数用 252
TRADING_DAYS = 252


def _ann_return(daily: pd.Series) -> float:
    cum = (1 + daily).prod()
    return cum ** (TRADING_DAYS / len(daily)) - 1


def _mdd(daily: pd.Series) -> float:
    nav = (1 + daily).cumprod()
    return (nav / nav.cummax() - 1).min()


def perf_stats(daily: pd.Series) -> dict:
    ann = _ann_return(daily)
    vol = daily.std() * np.sqrt(TRADING_DAYS)
    mdd = _mdd(daily)
    return {
        "年化收益": ann,
        "年化波动": vol,
        "夏普": ann / vol if vol > 0 else np.nan,
        "最大回撤": mdd,
        "Calmar": ann / abs(mdd) if mdd < 0 else np.nan,
    }


def summarize(report: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (总表, 分年度表)。"""
    net = report["return"] - report["cost"]
    bench = report["bench"]
    excess = net - bench

    total = pd.DataFrame({
        "策略(净)": perf_stats(net),
        "基准(中证800)": perf_stats(bench),
        "超额(净-基准)": {
            "年化收益": _ann_return(net) - _ann_return(bench),
            "年化波动": excess.std() * np.sqrt(TRADING_DAYS),
            "夏普": (excess.mean() * TRADING_DAYS) / (excess.std() * np.sqrt(TRADING_DAYS)),  # 信息比率
            "最大回撤": _mdd(excess),
            "Calmar": np.nan,
        },
    })

    rows = []
    for year, g in report.groupby(report.index.year):
        n = g["return"] - g["cost"]
        rows.append({
            "年度": year,
            "策略净收益": (1 + n).prod() - 1,
            "基准收益": (1 + g["bench"]).prod() - 1,
            "超额": (1 + n).prod() - (1 + g["bench"]).prod(),
            "年内单边换手(倍)": g["turnover"].sum() / 2,
            "成本拖累": g["cost"].sum(),
        })
    return total, pd.DataFrame(rows).set_index("年度")


def render(report: pd.DataFrame, out_prefix: Path, title: str) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    total, annual = summarize(report)

    md = [
        f"# 回测报告：{title}", "",
        f"- 区间: {report.index[0].date()} → {report.index[-1].date()}；口径: 成本后",
        f"- 日均单边换手: {report['turnover'].mean() / 2:.2%}（周度调仓）", "",
        "## 总体", "", total.round(4).to_markdown(), "",
        "## 分年度", "", annual.round(4).to_markdown(), "",
        f"![净值曲线]({out_prefix.name}_nav.png)", "",
    ]
    Path(f"{out_prefix}.md").write_text("\n".join(md))
    report.to_csv(f"{out_prefix}_daily.csv")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    net = report["return"] - report["cost"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot((1 + net).cumprod(), label="strategy(net)")
    axes[0].plot((1 + report["bench"]).cumprod(), label="CSI800")
    axes[0].plot((1 + net - report["bench"]).cumprod(), label="excess", linestyle="--")
    axes[0].legend()
    axes[0].set_title(title)
    nav = (1 + net).cumprod()
    axes[1].fill_between(nav.index, nav / nav.cummax() - 1, 0, alpha=0.4)
    axes[1].set_ylabel("drawdown")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_nav.png", dpi=120)
    plt.close(fig)
    print(f"[report] 已写 {out_prefix}.md / _daily.csv / _nav.png")
    print(total.round(4))
