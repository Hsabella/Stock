"""第一期验收评估：聚合全部回测产物 → 硬门槛 checklist → go/no-go 报告。

输入（results/quant/ 下由前序步骤产出）：
- bt_<tag>_daily.csv    各回测变体日度序列（run_backtest 产出）
- pred_<tag>.parquet    模型预测分（baseline 产出，用于算样本外周频 IC）
输出：docs/quant/phase1_report.md

用法: python -m quant.research.evaluate [--main bt_lgb_rolling]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from quant.backtest.report import TRADING_DAYS, _ann_return, _mdd, perf_stats
from quant.research.ic_report import rank_ic, weekly_anchors

RESULTS = Path("results/quant")

# 验收硬门槛（第一期 plan 定案）
GATES = [
    ("年化超额 vs 中证800 ≥ 8%", "excess_ann", 0.08, ">="),
    ("信息比率 ≥ 0.8", "ir", 0.8, ">="),
    ("夏普(净) ≥ 0.8", "sharpe", 0.8, ">="),
    ("最大回撤 ≤ 25% 且 ≤ 基准", "mdd_ok", 1, ">="),
    ("周度 RankIC ≥ 0.02", "ic", 0.02, ">="),
    ("ICIR ≥ 0.25", "icir", 0.25, ">="),
    ("周单边换手 ≤ 35%", "weekly_turnover", 0.35, "<="),
    ("成本翻倍后年化超额 ≥ 4%", "excess_2x", 0.04, ">="),
]

# v2 及格线（iteration_plan_v2 定案）：超额/IR 现实化 + 分年度稳定性；
# 夏普不再单列（v1 的 0.8 随 8% 超额目标一起失效）；成本翻倍保留"翻倍后剩一半"的比例逻辑
GATES_V2 = [
    ("年化超额 vs 中证800 ≥ 4%", "excess_ann", 0.04, ">="),
    ("信息比率 ≥ 0.5", "ir", 0.5, ">="),
    ("分年度超额为正的年份占比 ≥ 2/3", "annual_pos", 2 / 3, ">="),
    ("最大回撤 ≤ 25% 且 ≤ 基准", "mdd_ok", 1, ">="),
    ("周度 RankIC ≥ 0.02", "ic", 0.02, ">="),
    ("ICIR ≥ 0.25", "icir", 0.25, ">="),
    ("周单边换手 ≤ 35%", "weekly_turnover", 0.35, "<="),
    ("成本翻倍后年化超额 ≥ 2%", "excess_2x", 0.02, ">="),
]


def load_daily(tag: str) -> pd.DataFrame | None:
    path = RESULTS / f"{tag}_daily.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


def bt_metrics(daily: pd.DataFrame) -> dict:
    net = daily["return"] - daily["cost"]
    bench = daily["bench"]
    excess = net - bench
    stats = perf_stats(net)
    return {
        "ann": stats["年化收益"],
        "sharpe": stats["夏普"],
        "mdd": stats["最大回撤"],
        "bench_ann": _ann_return(bench),
        "bench_mdd": _mdd(bench),
        "excess_ann": _ann_return(net) - _ann_return(bench),
        "ir": (excess.mean() * TRADING_DAYS) / (excess.std() * TRADING_DAYS ** 0.5),
        "weekly_turnover": daily["turnover"].mean() / 2 * 5,  # 日均单边×5≈周单边
    }


def pred_weekly_ic(pred_tag: str) -> pd.Series:
    """样本外预测分的周频 RankIC 序列。"""
    import qlib
    from qlib.data import D

    from quant.config import load_config
    from quant.features.alpha_pv import LABEL_EXPRESSION

    qlib.init(provider_uri=load_config("data")["qlib"]["provider_uri"], region="cn")
    pred = pd.read_parquet(RESULTS / f"{pred_tag}.parquet")["score"]
    dts = pred.index.get_level_values("datetime")
    label = D.features(D.instruments("csi_union"), [LABEL_EXPRESSION],
                       start_time=str(dts.min().date()), end_time=str(dts.max().date()))
    label.columns = ["LABEL"]
    df = label.swaplevel().sort_index().join(pred, how="inner")
    ics = {}
    for dt in weekly_anchors(df.index.get_level_values(0).unique()):
        try:
            cross = df.xs(dt, level=0)
        except KeyError:
            continue
        v = rank_ic(cross["score"], cross["LABEL"])
        if v == v:
            ics[dt] = v
    return pd.Series(ics).sort_index()


def gate_table(m: dict, ic: pd.Series, m_2x: dict | None,
               annual: pd.DataFrame | None = None, gates: list = GATES) -> pd.DataFrame:
    vals = {
        "excess_ann": m["excess_ann"],
        "ir": m["ir"],
        "sharpe": m["sharpe"],
        "mdd_ok": 1 if (m["mdd"] >= -0.25 and m["mdd"] >= m["bench_mdd"]) else 0,
        "ic": ic.mean(),
        "icir": ic.mean() / ic.std(),
        "weekly_turnover": m["weekly_turnover"],
        "excess_2x": m_2x["excess_ann"] if m_2x else float("nan"),
        "annual_pos": (annual["超额"] > 0).mean() if annual is not None else float("nan"),
    }
    rows = []
    for name, key, thr, op in gates:
        v = vals[key]
        ok = (v >= thr) if op == ">=" else (v <= thr)
        rows.append({"门槛": name, "实测": round(float(v), 4), "结果": "✅ 过" if ok else "❌ 未过"})
    return pd.DataFrame(rows).set_index("门槛")


def annual_table(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, g in daily.groupby(daily.index.year):
        net = g["return"] - g["cost"]
        rows.append({
            "年度": year,
            "策略净收益": (1 + net).prod() - 1,
            "基准": (1 + g["bench"]).prod() - 1,
            "超额": (1 + net).prod() - (1 + g["bench"]).prod(),
        })
    return pd.DataFrame(rows).set_index("年度")


def variants_table() -> pd.DataFrame:
    rows = {}
    for path in sorted(RESULTS.glob("bt_*_daily.csv")):
        tag = path.stem.replace("bt_", "").replace("_daily", "")
        d = load_daily(f"bt_{tag}")
        if d is None or d.empty:
            continue
        m = bt_metrics(d)
        rows[tag] = {"净年化": m["ann"], "超额": m["excess_ann"], "夏普": m["sharpe"],
                     "MDD": m["mdd"], "周单边换手": m["weekly_turnover"]}
    return pd.DataFrame(rows).T


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", default="bt_lgb_rolling", help="主评估变体（bt_*_daily.csv 的前缀）")
    parser.add_argument("--pred", default="pred_lgb_rolling")
    parser.add_argument("--out", default="docs/quant/phase1_report.md")
    parser.add_argument("--gates", choices=["v1", "v2"], default="v1", help="v2=迭代及格线(超额4%/IR0.5/分年稳定)")
    args = parser.parse_args()

    daily = load_daily(args.main)
    if daily is None:
        print(f"[evaluate] 缺 {args.main}_daily.csv，先跑 run_backtest")
        return 1
    m = bt_metrics(daily)
    d2x = load_daily(f"{args.main}_cost2x")
    m2x = bt_metrics(d2x) if d2x is not None else None
    ic = pred_weekly_ic(args.pred)

    annual = annual_table(daily)
    gates = gate_table(m, ic, m2x, annual, GATES_V2 if args.gates == "v2" else GATES)
    n_pass = (gates["结果"] == "✅ 过").sum()
    verdict = ("GO：全部硬门槛通过，可启动第二期"
               if n_pass == len(gates)
               else f"NO-GO（as-is）：{len(gates) - n_pass}/{len(gates)} 项未过 → 按计划限时迭代一轮再判")

    ic_annual = pd.DataFrame({
        "IC": ic.groupby(ic.index.year).mean(),
        "正比率": ic.groupby(ic.index.year).apply(lambda s: (s > 0).mean()),
    })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([
        f"# 验收报告（{args.gates} 及格线，go/no-go）", "",
        f"- 主变体: {args.main}（滚动 walk-forward，2023 起全样本外，成本后）",
        f"- 区间: {daily.index[0].date()} → {daily.index[-1].date()}", "",
        f"## 判定：**{verdict}**", "",
        "## 硬门槛 checklist", "", gates.to_markdown(), "",
        "## 样本外周频 RankIC 分年", "", ic_annual.round(4).to_markdown(), "",
        "## 分年度收益", "", annual.round(4).to_markdown(), "",
        "## 全部变体对照", "", variants_table().round(4).to_markdown(), "",
    ]))
    print(gates)
    print(f"\n[evaluate] {verdict}")
    print(f"[evaluate] 报告已写 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
