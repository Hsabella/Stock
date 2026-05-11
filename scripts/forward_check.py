#!/usr/bin/env python3
"""对比"昨天的决策"和"今天/最近 N 天的实际收益", 沉淀基线数字.

用法:
    python3 scripts/forward_check.py                          # 拿最近一份 partial_*.json
    python3 scripts/forward_check.py --decision 20260510      # 指定决策日期
    python3 scripts/forward_check.py --horizons 1 3 5         # 自定义 T+N 窗口

输出:
- results/forward/forward_{decision_date}.md  人类可读
- results/forward/forward_{decision_date}.csv 完整明细
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from factors._kline import get_kline
from factors.regime.data import get_csi300_kline


def load_decision(decision_date: str | None) -> tuple[str, dict]:
    dec_dir = REPO / "results" / "decisions"
    if decision_date:
        path = dec_dir / f"partial_{decision_date}.json"
    else:
        files = sorted(dec_dir.glob("partial_*.json"))
        if not files:
            raise SystemExit("no decision file found")
        path = files[-1]
    with open(path, encoding="utf-8") as f:
        return path.stem.replace("partial_", ""), json.load(f)


def _ret(close: pd.Series, base_idx: int, horizon: int) -> float | None:
    """从 base_idx 到 base_idx+horizon 的收益."""
    target = base_idx + horizon
    if target >= len(close) or base_idx < 0:
        return None
    return float(close.iloc[target] / close.iloc[base_idx] - 1)


def _max_drawdown(close: pd.Series, base_idx: int, horizon: int) -> float | None:
    """从 base_idx 到 base_idx+horizon 之间的最大回撤 (相对 base 价)."""
    end = min(base_idx + horizon + 1, len(close))
    if end - base_idx <= 1:
        return None
    seg = close.iloc[base_idx:end]
    base = seg.iloc[0]
    return float(seg.min() / base - 1)


def compute_forward(symbol: str, decision_date: pd.Timestamp,
                    horizons: list[int], force_refresh: bool = False) -> dict:
    kl = get_kline(symbol, days=400, force_refresh=force_refresh)
    if kl.empty:
        return {h: None for h in horizons} | {"max_dd": None, "_no_data": True}
    kl = kl.sort_values("date").reset_index(drop=True)
    # 找决策当天的 idx (≤ decision_date 的最后一根)
    mask = kl["date"] <= decision_date
    if not mask.any():
        return {h: None for h in horizons} | {"max_dd": None, "_no_data": True}
    base_idx = mask[::-1].idxmax()  # 最后 True 的位置
    close = kl["close"]
    out: dict = {f"ret_T+{h}": _ret(close, base_idx, h) for h in horizons}
    out["max_dd"] = _max_drawdown(close, base_idx, max(horizons))
    out["base_price"] = float(close.iloc[base_idx])
    out["base_date"] = str(kl["date"].iloc[base_idx].date())
    return out


def csi_benchmark(decision_date: pd.Timestamp, horizons: list[int],
                  force_refresh: bool = False) -> dict:
    csi = get_csi300_kline(force_refresh=force_refresh)
    if csi.empty:
        return {f"csi_T+{h}": None for h in horizons}
    csi = csi.sort_values("date").reset_index(drop=True)
    mask = csi["date"] <= decision_date
    if not mask.any():
        return {f"csi_T+{h}": None for h in horizons}
    base = mask[::-1].idxmax()
    close = csi["close"]
    return {f"csi_T+{h}": _ret(close, base, h) for h in horizons}


def aggregate(df: pd.DataFrame, decisions: list[str], horizons: list[int]) -> pd.DataFrame:
    rows = []
    for d in decisions:
        sub = df[df["decision"] == d]
        for h in horizons:
            col = f"ret_T+{h}"
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            if vals.empty:
                continue
            rows.append({
                "decision": d,
                "horizon": f"T+{h}",
                "count": len(vals),
                "hit_rate": float((vals > 0).mean()),
                "mean_ret": float(vals.mean()),
                "median_ret": float(vals.median()),
                "worst": float(vals.min()),
                "best": float(vals.max()),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decision", default=None, help="决策日期 YYYYMMDD")
    ap.add_argument("--horizons", nargs="*", type=int, default=[1, 3, 5])
    ap.add_argument("--refresh", action="store_true",
                    help="强制刷新 K 线缓存 (当天评估必须开)")
    args = ap.parse_args()

    dec_date, report = load_decision(args.decision)
    decision_ts = pd.Timestamp(dec_date)  # YYYYMMDD → Timestamp
    horizons = sorted(set(args.horizons))
    print(f"决策日: {dec_date} → 评估 T+{horizons}")
    print(f"决策数: {len(report['decisions'])}")

    rows = []
    for d in report["decisions"]:
        sym = d["symbol"]
        fwd = compute_forward(sym, decision_ts, horizons, force_refresh=args.refresh)
        rows.append({
            "symbol": sym, "name": d["name"], "decision": d["decision"],
            "state": d.get("state"), "confidence": d.get("confidence"),
            "composite": d.get("composite"),
            "sector": d.get("sector"),
            **fwd,
        })
    df = pd.DataFrame(rows)

    bench = csi_benchmark(decision_ts, horizons, force_refresh=args.refresh)
    print(f"\n沪深 300 基准: {bench}")

    # 数据可用性检查: 任何一档 horizon 全部 None → 提示数据未到
    available = df[[f"ret_T+{h}" for h in horizons]].notna().any()
    if not available.any():
        print(f"\n⚠ 决策日 {dec_date} 之后还没有可用 K 线数据 (可能数据源未推送).")
        print("  → A 股数据通常当晚 18:00 后才更新, 请稍后或明天再跑.")
        sys.exit(2)

    # 按 decision 聚合
    agg = aggregate(df, ["BUY", "CONTINUE_WATCH", "DROP", "AVOID"], horizons)
    print("\n=== 分组表现 ===")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:+.3f}" if isinstance(x, float) else x))

    out_dir = REPO / "results" / "forward"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"forward_{dec_date}.csv"
    df.to_csv(csv_path, index=False)

    md_path = out_dir / f"forward_{dec_date}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Forward 回测 · 决策日 {dec_date}\n\n")
        f.write(f"**沪深 300 基准**: " +
                " / ".join(f"T+{h}={bench.get(f'csi_T+{h}'):+.2%}"
                          if bench.get(f'csi_T+{h}') is not None else f"T+{h}=NA"
                          for h in horizons) + "\n\n")
        f.write("## 分组表现\n\n")
        f.write(agg.to_markdown(index=False, floatfmt="+.3f") + "\n\n")

        for grp in ["BUY", "CONTINUE_WATCH", "DROP"]:
            sub = df[df["decision"] == grp].copy()
            if sub.empty:
                continue
            f.write(f"## {grp} 明细 ({len(sub)} 只)\n\n")
            cols = ["symbol", "name", "sector", "confidence"] + [f"ret_T+{h}" for h in horizons] + ["max_dd"]
            tbl = sub[cols].sort_values(f"ret_T+{horizons[0]}", ascending=False)
            f.write(tbl.to_markdown(index=False, floatfmt="+.2%") + "\n\n")

    print(f"\n保存: {csv_path}")
    print(f"保存: {md_path}")


if __name__ == "__main__":
    main()
