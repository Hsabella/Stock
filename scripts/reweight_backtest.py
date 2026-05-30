#!/usr/bin/env python3
"""离线再加权回测: 用历史 partial_*.csv 里存的 8 维 rank 重算 composite,
复测 IC 与 BUY-DROP 价差。不重跑引擎、不抓数据。
仅用 20260511+ 全 schema 日(43 列的 20260510 缺 sector/news rank, 自动跳过)。
"""
import argparse
import glob
import os
import re

import pandas as pd

BASE = "/Users/wangbo/VSCodeProjects/Stock/results"
DEC_DIR = os.path.join(BASE, "decisions")
FWD_DIR = os.path.join(BASE, "forward")

CURRENT = {"fundamental_rank": 0.22, "fund_flow_rank": 0.18, "liquidity_rank": 0.12,
           "chips_rank": 0.08, "regime_rank": 0.07, "tech_rank": 0.05,
           "sector_rank": 0.18, "news_rank": 0.10}


def spearman(a, b):
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(df) < 3:
        return None
    ra, rb_ = df["a"].rank(method="average"), df["b"].rank(method="average")
    if ra.std(ddof=0) == 0 or rb_.std(ddof=0) == 0:
        return None
    return ra.corr(rb_)


def penalty_from_rsi(rsi):
    overheat = ((rsi - 70) / 15).clip(0, 1).fillna(0)
    extreme = ((25 - rsi) / 10).clip(0, 1).fillna(0) * 0.4
    return (0.5 * overheat + extreme).clip(0, 0.7)


def recompute_composite(pdf, weights, use_penalty):
    score = pd.Series(0.0, index=pdf.index)
    tw = sum(weights.values())
    for col, w in weights.items():
        r = pd.to_numeric(pdf[col], errors="coerce").fillna(1.0)
        score = score + w * (1 - r)
    comp = score / tw
    if use_penalty and "rsi_value" in pdf.columns:
        comp = comp * (1 - penalty_from_rsi(pd.to_numeric(pdf["rsi_value"], errors="coerce")))
    if "veto_hit" in pdf.columns:
        comp = comp.where(pdf["veto_hit"] != True, 0.0)  # noqa: E712
    return comp


def _days():
    for p in sorted(glob.glob(os.path.join(DEC_DIR, "partial_*.csv"))):
        m = re.search(r"partial_(\d+)\.csv$", p)
        if not m:
            continue
        date = m.group(1)
        fwd = os.path.join(FWD_DIR, f"forward_{date}.csv")
        if os.path.exists(fwd):
            yield date, p, fwd


def run(weights, use_penalty=True, ret_col="ret_T+1", buy_th=0.55, drop_th=0.45):
    ics, spreads, used = [], [], 0
    for date, pf, ff in _days():
        pdf = pd.read_csv(pf)
        if not all(c in pdf.columns for c in weights):
            continue  # 跳过缺列的旧 schema 日
        fdf = pd.read_csv(ff)
        if ret_col not in fdf.columns:
            continue
        pdf = pdf.copy()
        pdf["new_comp"] = recompute_composite(pdf, weights, use_penalty)
        m = pdf[["symbol", "new_comp"]].merge(fdf[["symbol", ret_col]], on="symbol", how="inner")
        m = m.dropna(subset=[ret_col])
        if len(m) < 3:
            continue
        ic = spearman(m["new_comp"], m[ret_col])
        if ic is not None:
            ics.append(ic)
        buy, drop = m[m["new_comp"] >= buy_th][ret_col], m[m["new_comp"] < drop_th][ret_col]
        if len(buy) and len(drop):
            spreads.append(buy.mean() - drop.mean())
        used += 1
    return {
        "days": used,
        "ic_mean": (sum(ics) / len(ics)) if ics else None,
        "ic_pos_frac": (sum(1 for x in ics if x > 0) / len(ics)) if ics else None,
        "buy_minus_drop": (sum(spreads) / len(spreads)) if spreads else None,
    }


CONFIGS = {
    "baseline(当前8维+penalty)": (CURRENT, True),
    "去penalty": (CURRENT, False),
    "砍sector/liq/tech": ({k: v for k, v in CURRENT.items()
                           if k not in ("sector_rank", "liquidity_rank", "tech_rank")}, False),
    "再降fund/chips": ({"fund_flow_rank": 0.40, "regime_rank": 0.25,
                        "news_rank": 0.15, "fundamental_rank": 0.10, "chips_rank": 0.10}, False),
    "两维(flow+regime)": ({"fund_flow_rank": 0.5, "regime_rank": 0.5}, False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ret-col", default="ret_T+1")
    args = ap.parse_args()
    print(f"再加权回测 (ret_col={args.ret_col}) — 目标: ic_mean 转正 且 buy_minus_drop > 0\n")
    print(f"{'config':<26}{'days':>5}{'ic_mean':>10}{'ic_pos%':>9}{'BUY-DROP':>10}")
    for name, (w, pen) in CONFIGS.items():
        r = run(w, use_penalty=pen, ret_col=args.ret_col)
        ic = f"{r['ic_mean']:+.4f}" if r["ic_mean"] is not None else "N/A"
        pf = f"{r['ic_pos_frac']:.2f}" if r["ic_pos_frac"] is not None else "N/A"
        bd = f"{r['buy_minus_drop']:+.4f}" if r["buy_minus_drop"] is not None else "N/A"
        print(f"{name:<26}{r['days']:>5}{ic:>10}{pf:>9}{bd:>10}")


if __name__ == "__main__":
    main()
