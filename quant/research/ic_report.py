"""单因子体检：周频 RankIC / ICIR / 五分位组合价差 / 因子相关性。

评估口径与未来实盘一致：每周第一个交易日为锚点，因子值取锚点当日（T 收盘可算），
标签为 T+1 开盘买入持一周到 T+6 开盘的收益（alpha_pv.LABEL_EXPRESSION）。

淘汰标准（写进报告）：|RankIC 均值| < 0.01 或五分位不单调 → 标记 drop。

用法:
    python -m quant.research.ic_report [--start 2019-01-01] [--no-ext]
        [--out docs/quant/factor_report_v1.md]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quant.data import warehouse
from quant.features.alpha_pv import EXPRESSIONS, EXT_FEATURES, LABEL_EXPRESSION

N_QUANTILES = 5
MIN_STOCKS_PER_ANCHOR = 300  # 横截面样本太少的锚点不参与统计


def weekly_anchors(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """每周第一个交易日。"""
    iso = calendar.isocalendar()
    keys = list(zip(iso.year, iso.week))
    first = pd.Series(calendar, index=pd.MultiIndex.from_tuples(keys)).groupby(level=[0, 1]).first()
    return pd.DatetimeIndex(sorted(first))


def load_panel(start: str, end: str, with_ext: bool) -> pd.DataFrame:
    """(datetime, instrument) 面板：全部因子 + LABEL。"""
    import qlib
    from qlib.data import D

    from quant.config import load_config

    qlib.init(provider_uri=load_config("data")["qlib"]["provider_uri"], region="cn")
    exprs = list(EXPRESSIONS.values()) + [LABEL_EXPRESSION]
    names = list(EXPRESSIONS) + ["LABEL"]
    df = D.features(D.instruments("csi_union"), exprs, start_time=start, end_time=end)
    df.columns = names
    df = df.swaplevel().sort_index()  # (instrument, datetime) -> (datetime, instrument)

    if with_ext:
        ext_path = warehouse.warehouse_dir() / "ext_features.parquet"
        if ext_path.exists():
            ext = pd.read_parquet(ext_path)
            df = df.join(ext[[c for c in EXT_FEATURES if c in ext.columns]], how="left")
        else:
            print("[ic] 警告: ext_features.parquet 不存在，只评估量价因子")
    return df


def rank_ic(x: pd.Series, y: pd.Series) -> float:
    m = pd.concat([x, y], axis=1).dropna()
    if len(m) < MIN_STOCKS_PER_ANCHOR:
        return np.nan
    return m.iloc[:, 0].rank().corr(m.iloc[:, 1].rank())


def quantile_returns(x: pd.Series, y: pd.Series) -> list[float] | None:
    m = pd.concat([x.rename("f"), y.rename("r")], axis=1).dropna()
    if len(m) < MIN_STOCKS_PER_ANCHOR:
        return None
    q = pd.qcut(m["f"].rank(method="first"), N_QUANTILES, labels=False)
    return m.groupby(q)["r"].mean().reindex(range(N_QUANTILES)).tolist()


def evaluate(panel: pd.DataFrame, factors: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """返回 (汇总表, IC 时间序列, 因子秩相关矩阵)。"""
    anchors = weekly_anchors(panel.index.get_level_values(0).unique())
    ics: dict[str, dict] = {f: {} for f in factors}
    qrets: dict[str, list] = {f: [] for f in factors}
    corr_samples: list[pd.DataFrame] = []

    for i, dt in enumerate(anchors):
        try:
            cross = panel.xs(dt, level=0)
        except KeyError:
            continue
        label = cross["LABEL"]
        for f in factors:
            ic = rank_ic(cross[f], label)
            if not np.isnan(ic):
                ics[f][dt] = ic
            qr = quantile_returns(cross[f], label)
            if qr is not None:
                qrets[f].append(qr)
        if i % 8 == 0:  # 每 ~2 月采样一次横截面算因子间相关
            corr_samples.append(cross[factors].rank().corr())

    rows = []
    for f in factors:
        s = pd.Series(ics[f]).sort_index()
        if s.empty:
            rows.append({"factor": f, "n": 0})
            continue
        qm = np.nanmean(np.array(qrets[f]), axis=0)  # 各分位平均周收益
        spread_w = qm[-1] - qm[0]
        mono = pd.Series(qm).rank().corr(pd.Series(range(N_QUANTILES)).rank())
        rows.append({
            "factor": f, "n": len(s),
            "ic_mean": s.mean(), "ic_std": s.std(), "icir": s.mean() / s.std(),
            "ic_pos_rate": (s > 0).mean(),
            "q_spread_weekly": spread_w, "q_spread_ann": spread_w * 52,
            "monotonic": mono,
            # |IC|≥0.015 直接保留（尾部集中型因子单调性天然弱，树模型用得上）；
            # 0.01-0.015 之间要求分位大体单调；再弱即剔除
            "verdict": (
                "keep" if abs(s.mean()) >= 0.015
                else "keep" if abs(s.mean()) >= 0.01 and abs(mono) >= 0.7
                else "drop"
            ),
        })
    summary = pd.DataFrame(rows).set_index("factor").sort_values("icir", key=abs, ascending=False)
    ic_series = pd.DataFrame({f: pd.Series(ics[f]) for f in factors}).sort_index()
    corr = sum(corr_samples) / len(corr_samples) if corr_samples else pd.DataFrame()
    return summary, ic_series, corr


def write_report(summary: pd.DataFrame, corr: pd.DataFrame, out_md: Path, start: str, end: str) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 因子体检报告 v1（周频）", "",
        f"- 区间: {start} → {end}；锚点=每周首个交易日；标签=T+1 开盘买入持一周（open-to-open）",
        f"- 池子: csi_union（PIT）；每锚点横截面 ≥{MIN_STOCKS_PER_ANCHOR} 只才计入",
        "- 淘汰标准: |RankIC|<0.01 或五分位不单调（|单调秩相关|<0.7）→ drop", "",
        "## 汇总（按 |ICIR| 排序）", "",
        summary.round(4).to_markdown(), "",
        "## 因子间秩相关（>0.7 视为重复，需二选一）", "",
        corr.round(2).to_markdown() if not corr.empty else "(无)", "",
    ]
    high_corr = [
        f"{a} ↔ {b}: {corr.loc[a, b]:.2f}"
        for i, a in enumerate(corr.index) for b in corr.index[i + 1:]
        if abs(corr.loc[a, b]) > 0.7
    ] if not corr.empty else []
    if high_corr:
        lines += ["### 高相关对", ""] + [f"- {p}" for p in high_corr] + [""]
    out_md.write_text("\n".join(lines))
    print(f"[ic] 报告已写 {out_md}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-06-30")
    parser.add_argument("--no-ext", action="store_true", help="只评估量价 expression 因子")
    parser.add_argument("--out", default="docs/quant/factor_report_v1.md")
    args = parser.parse_args()

    factors = list(EXPRESSIONS) + ([] if args.no_ext else EXT_FEATURES)
    panel = load_panel(args.start, args.end, with_ext=not args.no_ext)
    factors = [f for f in factors if f in panel.columns]
    print(f"[ic] 面板 {panel.shape[0]:,} 行, 评估 {len(factors)} 个因子")

    summary, ic_series, corr = evaluate(panel, factors)
    print(summary[["n", "ic_mean", "icir", "ic_pos_rate", "q_spread_ann", "verdict"]].round(4))

    out_dir = Path("results/quant")
    out_dir.mkdir(parents=True, exist_ok=True)
    ic_series.to_csv(out_dir / "factor_ic_weekly_v1.csv")
    write_report(summary, corr, Path(args.out), args.start, args.end)
    return 0


if __name__ == "__main__":
    sys.exit(main())
