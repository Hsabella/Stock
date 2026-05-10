#!/usr/bin/env python3
"""出场规则对比：裸持 vs 8日强平 vs 止损止盈."""
import csv, sys, time
from pathlib import Path
import akshare as ak
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SIGNAL_FILE = REPO / "results" / "stock_signals_20251125_180850.csv"
END = "2026-05-09"

STRATEGIES = {
    "A_hold20": {"max_days": 20, "stop_loss": None, "take_profit": None},
    "B_hold8":  {"max_days": 8,  "stop_loss": None, "take_profit": None},
    "C_sl5_tp10_d20": {"max_days": 20, "stop_loss": -0.05, "take_profit": 0.10},
    "D_sl5_tp10_d8":  {"max_days": 8,  "stop_loss": -0.05, "take_profit": 0.10},
}


def load_signals():
    rows = []
    with open(SIGNAL_FILE, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append({"symbol": r["symbol"].zfill(6), "name": r["name"],
                         "score": int(r["score"]), "date": r["date"]})
    return rows


def fetch_tx(symbol, start, end):
    if symbol.startswith("6"):
        sym = f"sh{symbol}"
    else:
        sym = f"sz{symbol}"
    for attempt in range(4):
        try:
            df = ak.stock_zh_a_hist_tx(symbol=sym, start_date=start.replace("-", ""),
                                       end_date=end.replace("-", ""), adjust="qfq")
            if df is not None and not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                return df.sort_values("date").reset_index(drop=True)
        except Exception as e:
            print(f"  {symbol} retry {attempt}: {e}", file=sys.stderr)
            time.sleep(2 + attempt)
    return pd.DataFrame()


def simulate(df, signal_date, max_days, stop_loss, take_profit):
    """返回 (return_pct, exit_day, exit_reason)."""
    sd = pd.Timestamp(signal_date)
    on_or_before = df[df["date"] <= sd]
    after = df[df["date"] > sd].head(max_days)
    if on_or_before.empty or after.empty:
        return None, None, "no_data"
    base = float(on_or_before.iloc[-1]["close"])
    if base == 0:
        return None, None, "bad_base"
    for i, row in after.iterrows():
        high, low, close = float(row["high"]), float(row["low"]), float(row["close"])
        day = list(after.index).index(i) + 1
        # 用 high/low 触发止损/止盈（更真实）
        if take_profit is not None:
            tp_price = base * (1 + take_profit)
            if high >= tp_price:
                return take_profit * 100, day, "take_profit"
        if stop_loss is not None:
            sl_price = base * (1 + stop_loss)
            if low <= sl_price:
                return stop_loss * 100, day, "stop_loss"
    # 持有到期
    final_close = float(after.iloc[-1]["close"])
    return (final_close / base - 1) * 100, len(after), "max_days"


def fetch_index():
    df = ak.stock_zh_index_daily(symbol="sh000300")
    df = df.reset_index() if "date" not in df.columns else df
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def main():
    signals = load_signals()
    print(f"信号 {len(signals)} 条 / 4 种出场规则\n")

    bench_df = fetch_index()
    bench_rets = {}
    for name, cfg in STRATEGIES.items():
        # 基准：用相同 max_days 在沪深300上做"持有 N 日"
        sd = pd.Timestamp("2025-11-25")
        after = bench_df[bench_df["date"] > sd].head(cfg["max_days"])
        on_before = bench_df[bench_df["date"] <= sd]
        if not after.empty and not on_before.empty:
            b = float(on_before.iloc[-1]["close"])
            t = float(after.iloc[-1]["close"])
            bench_rets[name] = (t / b - 1) * 100
        else:
            bench_rets[name] = None

    # 拉每只股票
    stock_data = {}
    for i, s in enumerate(signals, 1):
        print(f"[{i}/{len(signals)}] {s['symbol']} {s['name']}", flush=True)
        df = fetch_tx(s["symbol"], s["date"], END)
        if not df.empty and "high" in df.columns:
            stock_data[s["symbol"]] = (s, df)
        time.sleep(0.6)

    print(f"\n成功 {len(stock_data)}/{len(signals)} 只\n")

    # 对每个策略模拟
    results = {name: [] for name in STRATEGIES}
    for sym, (s, df) in stock_data.items():
        for name, cfg in STRATEGIES.items():
            ret, exit_day, reason = simulate(df, s["date"], **cfg)
            if ret is not None:
                results[name].append({
                    "symbol": sym, "name": s["name"], "score": s["score"],
                    "return": ret, "exit_day": exit_day, "reason": reason,
                })

    print("=" * 90)
    print(f"{'strategy':<22} {'n':>3} {'mean%':>8} {'med%':>8} {'win%':>6} "
          f"{'≥+5':>4} {'≤-5':>4} {'maxL%':>7} {'maxW%':>7} {'bench%':>8} {'alpha%':>8}")
    print("=" * 90)
    for name in STRATEGIES:
        rs = pd.Series([r["return"] for r in results[name]])
        b = bench_rets.get(name)
        alpha = rs.mean() - b if b is not None else None
        print(f"{name:<22} {len(rs):>3d} {rs.mean():>+7.2f} {rs.median():>+7.2f} "
              f"{(rs>0).mean()*100:>5.1f}% {(rs>=5).sum():>4d} {(rs<=-5).sum():>4d} "
              f"{rs.min():>+6.2f} {rs.max():>+6.2f} "
              + (f"{b:>+7.2f} " if b is not None else f"{'NA':>7} ")
              + (f"{alpha:>+7.2f}" if alpha is not None else f"{'NA':>7}"))

    print("\n各策略出场原因分布（含止损命中率）:")
    for name in STRATEGIES:
        if not results[name]:
            continue
        rsn = pd.Series([r["reason"] for r in results[name]]).value_counts()
        avg_day = pd.Series([r["exit_day"] for r in results[name]]).mean()
        print(f"  {name}: 平均持仓 {avg_day:.1f} 天, " +
              ", ".join(f"{k}={v}" for k, v in rsn.items()))

    print("\n按 score 分组（策略 D：-5%/+10%/8天，最有希望的方案）:")
    rs_d = pd.DataFrame(results["D_sl5_tp10_d8"])
    if not rs_d.empty:
        rs_d["bin"] = pd.cut(rs_d["score"], [0, 70, 85, 101], labels=["low(≤70)", "mid(71-85)", "high(86+)"])
        g = rs_d.groupby("bin", observed=True)["return"].agg(["count", "mean", "median",
                                                              lambda x: (x > 0).mean() * 100])
        g.columns = ["n", "mean%", "med%", "win%"]
        print(g.to_string(float_format=lambda x: f"{x:+.2f}"))

    # 保存明细
    rows = []
    for name in STRATEGIES:
        for r in results[name]:
            rows.append({"strategy": name, **r})
    pd.DataFrame(rows).to_csv(REPO / "results" / "backtest_v3_exit.csv", index=False)
    print(f"\n保存: {REPO / 'results' / 'backtest_v3_exit.csv'}")


if __name__ == "__main__":
    main()
