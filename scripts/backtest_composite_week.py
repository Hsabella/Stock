#!/usr/bin/env python3
"""1 周 composite 区分度回测.

思路:
1. 加载今日 partial_composite (results/decisions/partial_*.csv 最新)
2. 拉每只股票过去 30 个交易日 K 线（腾讯接口）
3. 每个回看锚点日 (anchor): T-1...T-5
   对该 anchor: 统计高 composite 组（top 1/3）vs 低 composite 组（bottom 1/3）的
   T+1, T+3, T+5 forward 收益均值/胜率
4. 同时报告 anchor 序列上的 rank-IC（composite 与 fwd_return 的 spearman 相关）

注: 由于 fundamental/fund_flow 数据本质周级慢变, 用 today composite 作为
    历史 5 天的 proxy 是合理近似（5 天内排名几乎不变）.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from factors._kline import get_kline


def load_latest_composite() -> pd.DataFrame:
    d = REPO / "results" / "decisions"
    files = sorted(d.glob("partial_*.csv"))
    if not files:
        raise SystemExit("无 partial_*.csv，先运行 scripts/run_partial_engine.py")
    df = pd.read_csv(files[-1])
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    return df


def fetch_kline(symbol: str, end: str, days: int = 60) -> pd.DataFrame:
    """复用共享 K 线缓存（新浪源）"""
    return get_kline(symbol, days=days)


def fwd_return(df: pd.DataFrame, anchor: pd.Timestamp, horizon: int) -> float | None:
    on_or_before = df[df["date"] <= anchor]
    after = df[df["date"] > anchor]
    if on_or_before.empty or len(after) < horizon:
        return None
    base = float(on_or_before.iloc[-1]["close"])
    if base == 0:
        return None
    target = float(after.iloc[horizon - 1]["close"])
    return (target / base - 1) * 100


def main():
    comp = load_latest_composite()
    print(f"composite 加载: {len(comp)} 只")

    # 拉 K 线
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    klines: dict[str, pd.DataFrame] = {}
    for i, row in enumerate(comp.itertuples(), 1):
        print(f"  [{i}/{len(comp)}] {row.symbol} {row.name} K线...", flush=True)
        df = fetch_kline(row.symbol, end, days=60)
        if not df.empty:
            klines[row.symbol] = df
        time.sleep(0.4)

    print(f"\nK 线拉取成功 {len(klines)}/{len(comp)}")

    # 选 N 个 anchor 日（默认 30）。注意 anchor 必须留 5 天 forward 空间。
    # 注意方法论局限: composite 是"今天的"快照（fundamental/fund_flow 用 today 数据），
    # 假设 30 个 anchor 范围内排名相对稳定（基本面季报频率慢，资金流轮动短期内有持续性）。
    # 这是横截面 signal stability 验证, 非严格 walk-forward.
    n_anchors = 30
    sample = next(iter(klines.values()))
    trading_days = sample["date"].sort_values().tolist()
    if len(trading_days) < n_anchors + 7:
        raise SystemExit(f"K 线长度不足: 需要 {n_anchors + 7} 天, 仅 {len(trading_days)}")
    # 倒数第 6 天往前取 N 个（anchor 之后留 5 个交易日做 forward）
    anchors = trading_days[-(n_anchors + 5):-5]

    print(f"\nAnchor 日: {[d.strftime('%Y-%m-%d') for d in anchors]}")

    # 对每个 anchor 计算 forward return
    horizons = [1, 3, 5]
    rows = []
    for anchor in anchors:
        for r in comp.itertuples():
            if r.symbol not in klines:
                continue
            rec = {"anchor": anchor.strftime("%Y-%m-%d"), "symbol": r.symbol, "name": r.name,
                   "composite": r.partial_composite, "veto": r.veto_hit}
            for h in horizons:
                rec[f"T+{h}"] = fwd_return(klines[r.symbol], anchor, h)
            rows.append(rec)
    bt = pd.DataFrame(rows)
    print(f"\nbacktest 行数: {len(bt)}")

    # ---- 区分度: 分位组 vs forward 收益 ----
    print("\n========= 分组对比 (composite top1/3 vs bot1/3) =========")
    for h in horizons:
        col = f"T+{h}"
        sub = bt.dropna(subset=[col, "composite"]).copy()
        if sub.empty:
            continue
        # 每个 anchor 内做分位
        def _bin(g):
            g = g.copy()
            try:
                g["bin"] = pd.qcut(g["composite"], 3, labels=["bot", "mid", "top"], duplicates="drop")
            except ValueError:
                g["bin"] = "mid"
            return g
        sub = sub.groupby("anchor", group_keys=False).apply(_bin)
        agg = sub.groupby("bin", observed=True)[col].agg(["count", "mean", "median",
                                                          lambda x: (x > 0).mean() * 100])
        agg.columns = ["n", "mean%", "med%", "win%"]
        print(f"\n[T+{h}]")
        print(agg.to_string(float_format=lambda x: f"{x:+.2f}"))
        if {"top", "bot"}.issubset(set(agg.index)):
            spread = agg.loc["top", "mean%"] - agg.loc["bot", "mean%"]
            print(f"  → top-bot spread: {spread:+.2f} pp")

    # ---- rank-IC ----
    print("\n========= rank-IC (composite vs forward return) =========")
    print(f"{'anchor':<12} " + " ".join(f"{f'IC(T+{h})':>10}" for h in horizons))
    ic_rows = []
    for anchor in anchors:
        sub = bt[bt["anchor"] == anchor.strftime("%Y-%m-%d")]
        ics = []
        for h in horizons:
            valid = sub.dropna(subset=[f"T+{h}", "composite"])
            if len(valid) >= 5:
                ics.append(valid["composite"].rank().corr(valid[f"T+{h}"].rank()))
            else:
                ics.append(None)
        ic_rows.append((anchor, ics))
        line = f"{anchor.strftime('%Y-%m-%d'):<12} "
        for v in ics:
            line += f"{v:>+10.3f} " if v is not None else f"{'NA':>10} "
        print(line)

    # 平均 IC
    ic_avg = pd.DataFrame([r[1] for r in ic_rows], columns=[f"IC(T+{h})" for h in horizons]).mean()
    print("\n平均 rank-IC:")
    print(ic_avg.to_string(float_format=lambda x: f"{x:+.3f}"))

    out = REPO / "results" / "backtest_composite_week.csv"
    bt.to_csv(out, index=False)
    print(f"\n保存: {out}")


if __name__ == "__main__":
    main()
