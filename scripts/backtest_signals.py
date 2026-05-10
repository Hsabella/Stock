#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Forward-look验证：拿最近一次扫描的信号，看 T+1/+5/+10/+20/+60 的实际收益."""
import csv
import sys
import time
from pathlib import Path

import akshare as ak
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SIGNAL_FILE = REPO / "results" / "stock_signals_20251125_180850.csv"
HORIZONS = [1, 5, 10, 20, 60]


def load_signals(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append({
                "symbol": r["symbol"].zfill(6),
                "name": r["name"],
                "score": int(r["score"]),
                "date": r["date"],
                "price": float(r["price"]),
            })
    return rows


def fetch_forward(symbol, start, end):
    for attempt in range(5):
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol, period="daily",
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
                adjust="qfq",
            )
            if df is not None and not df.empty:
                df = df.rename(columns={"日期": "date", "收盘": "close"})
                df["date"] = pd.to_datetime(df["date"])
                return df[["date", "close"]].sort_values("date").reset_index(drop=True)
        except Exception as e:
            print(f"  retry {symbol} attempt {attempt+1}: {e}", file=sys.stderr)
            time.sleep(3 + attempt * 2)
    return pd.DataFrame()


def fetch_index(start, end):
    df = ak.stock_zh_index_daily(symbol="sh000300")  # 沪深300
    df = df.reset_index() if "date" not in df.columns else df
    df["date"] = pd.to_datetime(df["date"])
    return df[(df["date"] >= start) & (df["date"] <= end)][["date", "close"]].reset_index(drop=True)


def fwd_return(df, signal_date, horizon):
    on_or_before = df[df["date"] <= signal_date]
    after = df[df["date"] > signal_date]
    if on_or_before.empty or len(after) < horizon:
        return None
    base = on_or_before.iloc[-1]["close"]
    target = after.iloc[horizon - 1]["close"]
    if base == 0:
        return None
    return (target / base - 1) * 100


def main():
    signals = load_signals(SIGNAL_FILE)
    print(f"加载信号 {len(signals)} 条")

    start = "2025-11-25"
    end = "2026-05-09"

    print("拉取沪深300基准...")
    bench = fetch_index(start, end)
    bench_returns = {}
    for h in HORIZONS:
        signal_d = pd.Timestamp(start)
        bench_returns[h] = fwd_return(bench, signal_d, h)
    print(f"  沪深300 forward returns: " +
          ", ".join(f"T+{h}={bench_returns[h]:+.2f}%" if bench_returns[h] is not None else f"T+{h}=NA"
                    for h in HORIZONS))

    print("\n逐只拉取并计算...")
    rows = []
    for i, s in enumerate(signals, 1):
        print(f"[{i}/{len(signals)}] {s['symbol']} {s['name']} (score={s['score']})", flush=True)
        df = fetch_forward(s["symbol"], s["date"], end)
        if df.empty:
            print("  ⚠️ 无数据"); continue
        signal_d = pd.Timestamp(s["date"])
        rets = {h: fwd_return(df, signal_d, h) for h in HORIZONS}
        rows.append({"symbol": s["symbol"], "name": s["name"], "score": s["score"],
                     "signal_date": s["date"], "signal_price": s["price"], **{f"T+{h}": rets[h] for h in HORIZONS}})
        time.sleep(1.5)

    if not rows:
        print("无可用数据"); return

    df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("逐只收益（%）")
    print("=" * 80)
    cols = ["symbol", "name", "score"] + [f"T+{h}" for h in HORIZONS]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:+.2f}" if pd.notna(x) else "  NA"))

    print("\n" + "=" * 80)
    print("汇总统计")
    print("=" * 80)
    print(f"{'horizon':<8} {'mean%':>8} {'median%':>9} {'win_rate':>9} {'≥+5%':>7} {'≤-5%':>7} {'bench%':>8} {'alpha%':>8}")
    for h in HORIZONS:
        col = f"T+{h}"
        s = df[col].dropna()
        if s.empty:
            print(f"T+{h:<5}     NA"); continue
        win = (s > 0).mean() * 100
        big_win = (s >= 5).sum()
        big_loss = (s <= -5).sum()
        bench = bench_returns.get(h)
        alpha = s.mean() - bench if bench is not None else None
        print(f"T+{h:<5} {s.mean():>+8.2f} {s.median():>+9.2f} {win:>8.1f}% {big_win:>7d} {big_loss:>7d} "
              f"{bench:>+8.2f}" if bench is not None else f" {'NA':>8}",
              end="")
        if alpha is not None:
            print(f" {alpha:>+8.2f}")
        else:
            print()

    print("\n" + "=" * 80)
    print("按 score 分组（T+20）")
    print("=" * 80)
    df["score_bin"] = pd.cut(df["score"], bins=[0, 70, 85, 101], labels=["低分(≤70)", "中分(71-85)", "高分(86+)"])
    grp = df.groupby("score_bin", observed=True)["T+20"].agg(["count", "mean", "median",
                                                              lambda x: (x > 0).mean() * 100])
    grp.columns = ["n", "mean%", "median%", "win%"]
    print(grp.to_string(float_format=lambda x: f"{x:+.2f}"))

    out = REPO / "results" / "backtest_forward_20251125.csv"
    df.to_csv(out, index=False)
    print(f"\n明细已保存: {out}")


if __name__ == "__main__":
    main()
