#!/usr/bin/env python3
"""逐因子信息系数(IC)计算。

约定: rank 是 [0,1] 分位, 0=最强/最好, 1=最弱。
signal = (1 - rank), 所以 IC = Spearman(signal, ret) = -Spearman(rank, ret)。
正 IC = 有预测力/好。
没有 scipy, 用 pandas rank + Pearson 手算 Spearman(并列取平均秩)。
"""
import glob
import os
import re

import pandas as pd

BASE = "/Users/wangbo/VSCodeProjects/Stock/results"
DEC_DIR = os.path.join(BASE, "decisions")
FWD_DIR = os.path.join(BASE, "forward")

RANK_COLS = [
    "fundamental_rank", "fund_flow_rank", "liquidity_rank", "tech_rank",
    "chips_rank", "regime_rank", "sector_rank", "news_rank",
]
RET_COL = os.environ.get("IC_RET_COL", "ret_T+1")


def spearman(a: pd.Series, b: pd.Series):
    """Spearman 相关 = 秩的 Pearson 相关。并列用平均秩(method='average')。"""
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(df) < 3:
        return None, len(df)
    ra = df["a"].rank(method="average")
    rb = df["b"].rank(method="average")
    if ra.std(ddof=0) == 0 or rb.std(ddof=0) == 0:
        return None, len(df)  # 因子全相同(常量), 相关无定义
    return ra.corr(rb), len(df)


def find_dates():
    out = []
    for p in sorted(glob.glob(os.path.join(DEC_DIR, "partial_*.csv"))):
        m = re.search(r"partial_(\d+)\.csv$", p)
        if not m:
            continue
        date = m.group(1)
        fwd = os.path.join(FWD_DIR, f"forward_{date}.csv")
        if os.path.exists(fwd):
            out.append((date, p, fwd))
    return out


def main():
    dates = find_dates()
    # 每个因子: list of (date, daily_ic, n)
    daily = {c: [] for c in RANK_COLS + ["composite"]}
    total_join_rows = 0
    per_date_n = []

    for date, pfile, ffile in dates:
        pdf = pd.read_csv(pfile)
        fdf = pd.read_csv(ffile)
        # base_date 一致性检查
        base_dates = fdf["base_date"].dropna().unique().tolist() if "base_date" in fdf else []
        # 按 symbol 内连接, 只取需要的列
        keep_p = ["symbol"] + [c for c in RANK_COLS + ["composite"] if c in pdf.columns]
        merged = pdf[keep_p].merge(fdf[["symbol", RET_COL]], on="symbol", how="inner")
        n = len(merged)
        total_join_rows += n
        per_date_n.append((date, n, base_dates))

        for col in RANK_COLS + ["composite"]:
            if col not in merged.columns:
                continue
            if col == "composite":
                # composite 是分数, 越大越好(BUY), 直接与 ret 正相关 => 基线
                ic, nn = spearman(merged[col], merged[RET_COL])
            else:
                # rank: 0=好, signal=1-rank, IC = -corr(rank, ret)
                ic, nn = spearman(merged[col], merged[RET_COL])
                if ic is not None:
                    ic = -ic
            if ic is not None:
                daily[col].append((date, ic, nn))

    print("=" * 70)
    print(f"重叠日期数(partial & forward 都有): {len(dates)}")
    print(f"日期列表: {', '.join(d for d, _, _ in dates)}")
    print(f"连接后总样本行数(sum over days): {total_join_rows}")
    print("-" * 70)
    print("每日连接样本量 & forward.base_date:")
    for date, n, bd in per_date_n:
        print(f"  {date}: n={n:3d}  base_date={bd}")
    print("=" * 70)

    # 汇总
    results = []
    for col in RANK_COLS + ["composite"]:
        recs = daily[col]
        if not recs:
            results.append((col, None, None, 0, 0))
            continue
        ics = [r[1] for r in recs]
        n_days = len(ics)
        ic_mean = sum(ics) / n_days
        pos_frac = sum(1 for x in ics if x > 0) / n_days
        n_total = sum(r[2] for r in recs)
        results.append((col, ic_mean, pos_frac, n_days, n_total))

    # 因子部分(不含 composite)按 IC 从好到坏排序
    factor_res = [r for r in results if r[0] != "composite"]
    factor_res.sort(key=lambda r: (r[1] is not None, r[1] if r[1] is not None else -9), reverse=True)
    comp_res = [r for r in results if r[0] == "composite"][0]

    print("逐因子平均日 IC (从好到坏, 正=有预测力):")
    print(f"{'factor':<18}{'ic_mean':>10}{'pos_days_frac':>15}{'n_days':>8}{'n_total':>9}")
    for col, ic_mean, pos_frac, n_days, n_total in factor_res:
        im = f"{ic_mean:+.4f}" if ic_mean is not None else "  N/A "
        pf = f"{pos_frac:.3f}" if pos_frac is not None else " N/A "
        print(f"{col:<18}{im:>10}{pf:>15}{n_days:>8}{n_total:>9}")
    print("-" * 70)
    col, ic_mean, pos_frac, n_days, n_total = comp_res
    im = f"{ic_mean:+.4f}" if ic_mean is not None else "N/A"
    pf = f"{pos_frac:.3f}" if pos_frac is not None else "N/A"
    print(f"BASELINE {col:<9}{im:>10}{pf:>15}{n_days:>8}{n_total:>9}  (应≈-0.10)")
    print("=" * 70)

    # 机器可读输出
    print("MACHINE_READABLE_START")
    for col, ic_mean, pos_frac, n_days, n_total in results:
        print(f"{col}|{ic_mean}|{pos_frac}|{n_days}|{n_total}")
    print("MACHINE_READABLE_END")

    # ---- 落盘 per-(date,factor) IC 时间序列, 供看板因子体检页 ----
    out_csv = os.path.join(os.path.dirname(BASE), "results", "factor_ic.csv")
    rows = []
    for col in RANK_COLS + ["composite"]:
        for date, ic, nn in daily[col]:
            rows.append({"date": date, "factor": col, "ic": round(ic, 6), "n": nn})
    pd.DataFrame(rows, columns=["date", "factor", "ic", "n"]).to_csv(out_csv, index=False)
    print(f"\n[written] {out_csv}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
