#!/usr/bin/env python3
"""Forward-look v2: 用腾讯接口规避东财限流."""
import csv, sys, time
from pathlib import Path
import akshare as ak
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SIGNAL_FILE = REPO / "results" / "stock_signals_20251125_180850.csv"
HORIZONS = [1, 5, 10, 20, 60, 110]


def load_signals(p):
    rows = []
    with open(p, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append({"symbol": r["symbol"].zfill(6), "name": r["name"],
                         "score": int(r["score"]), "date": r["date"], "price": float(r["price"])})
    return rows


def fetch_tx(symbol, start, end):
    if symbol.startswith("6"):
        sym = f"sh{symbol}"
    elif symbol.startswith(("0", "3")):
        sym = f"sz{symbol}"
    else:
        sym = f"sz{symbol}"
    for attempt in range(4):
        try:
            df = ak.stock_zh_a_hist_tx(symbol=sym, start_date=start.replace("-", ""),
                                        end_date=end.replace("-", ""), adjust="qfq")
            if df is not None and not df.empty:
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                else:
                    df = df.rename(columns={df.columns[0]: "date"})
                    df["date"] = pd.to_datetime(df["date"])
                return df.sort_values("date").reset_index(drop=True)
        except Exception as e:
            print(f"  retry {symbol}.{attempt}: {e}", file=sys.stderr)
            time.sleep(2 + attempt * 2)
    return pd.DataFrame()


def fetch_index(start, end):
    df = ak.stock_zh_index_daily(symbol="sh000300")
    df = df.reset_index() if "date" not in df.columns else df
    df["date"] = pd.to_datetime(df["date"])
    return df[(df["date"] >= start) & (df["date"] <= end)][["date", "close"]].reset_index(drop=True)


def fwd_return(df, signal_date, horizon):
    sd = pd.Timestamp(signal_date)
    on_or_before = df[df["date"] <= sd]
    after = df[df["date"] > sd]
    if on_or_before.empty or len(after) < horizon:
        return None
    base = float(on_or_before.iloc[-1]["close"])
    target = float(after.iloc[horizon - 1]["close"])
    return (target / base - 1) * 100 if base else None


def main():
    signals = load_signals(SIGNAL_FILE)
    print(f"信号数 {len(signals)}")
    start, end = "2025-11-25", "2026-05-09"
    bench = fetch_index(start, end)
    bench_ret = {h: fwd_return(bench, start, h) for h in HORIZONS}
    print("沪深300 基准:", {f"T+{h}": (f"{v:+.2f}%" if v else "NA") for h, v in bench_ret.items()})

    rows = []
    for i, s in enumerate(signals, 1):
        print(f"[{i}/{len(signals)}] {s['symbol']} {s['name']} score={s['score']}", flush=True)
        df = fetch_tx(s["symbol"], s["date"], end)
        if df.empty or "close" not in df.columns:
            print("  无数据"); time.sleep(1); continue
        rets = {h: fwd_return(df, s["date"], h) for h in HORIZONS}
        rows.append({"symbol": s["symbol"], "name": s["name"], "score": s["score"],
                     "signal_date": s["date"], **{f"T+{h}": rets[h] for h in HORIZONS}})
        time.sleep(0.8)

    if not rows:
        print("无数据"); return
    df = pd.DataFrame(rows)
    print("\n逐只:")
    cols = ["symbol", "name", "score"] + [f"T+{h}" for h in HORIZONS]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:+6.2f}" if pd.notna(x) else "    NA"))

    print("\n汇总:")
    print(f"{'h':<6} {'n':>3} {'mean':>8} {'med':>8} {'win%':>6} {'≥+5':>4} {'≤-5':>4} {'bench':>8} {'alpha':>8}")
    for h in HORIZONS:
        col = f"T+{h}"
        sr = df[col].dropna()
        if sr.empty:
            print(f"T+{h:<4}  no data"); continue
        b = bench_ret.get(h)
        alpha = sr.mean() - b if b is not None else float("nan")
        print(f"T+{h:<4} {len(sr):>3d} {sr.mean():>+7.2f}% {sr.median():>+7.2f}% "
              f"{(sr>0).mean()*100:>5.1f}% {(sr>=5).sum():>4d} {(sr<=-5).sum():>4d} "
              f"{b:>+7.2f}% {alpha:>+7.2f}%")

    print("\n按 score 分组（T+20）:")
    df["bin"] = pd.cut(df["score"], [0, 70, 85, 101], labels=["low(≤70)", "mid(71-85)", "high(86+)"])
    g = df.groupby("bin", observed=True)["T+20"].agg(["count", "mean", "median",
                                                       lambda x: (x > 0).mean() * 100]).rename(
        columns={"<lambda_0>": "win%"})
    print(g.to_string(float_format=lambda x: f"{x:+.2f}"))

    df.to_csv(REPO / "results" / "backtest_v2.csv", index=False)
    print("\n保存:", REPO / "results" / "backtest_v2.csv")


if __name__ == "__main__":
    main()
