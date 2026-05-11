#!/usr/bin/env python3
"""部分决策引擎: fundamental + fund_flow + liquidity + technical 四维, 输出 partial_composite.

权重（spec 完整 8 维归一化后）:
- fundamental 0.30 / fund_flow 0.25 / liquidity 0.15 / technical 0.05
- 归一化分母 = 0.75
partial_composite = Σ wi * (1 - rank_i) / Σ wi

用法:
    python3 scripts/run_partial_engine.py [--watchlist watchlist.yaml]
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from factors.fundamental import dim as fund_dim
from factors.fund_flow import dim as flow_dim
from factors.fund_flow.data import get_lhb_recent, get_fund_flow_rank
from factors.liquidity import dim as liq_dim
from factors.technical import dim as tech_dim
from factors.chips import dim as chips_dim
from factors.regime import dim as regime_dim
from factors.regime.data import get_csi300_kline, get_north_total_inflow
from factors.sector_momentum import dim as sector_dim
from factors.news import dim as news_dim
from factors.news.data import get_news_recent
from factors._kline import get_kline
from engine.decision_resolver import resolve_batch
from engine.state_machine import load_positions, effective_state


def load_watchlist(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    items = cfg.get("watchlist", [])
    seen = set()
    out = []
    for it in items:
        sym = str(it["symbol"]).zfill(6)
        if sym in seen:
            continue
        seen.add(sym)
        out.append({"symbol": sym, "name": it.get("name", ""), "state": it.get("state", "WATCHING")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watchlist", default=str(REPO / "watchlist.yaml"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    items = load_watchlist(Path(args.watchlist))
    state_map_declared = {it["symbol"]: it["state"] for it in items}
    positions = load_positions(REPO / "state" / "positions.json")
    state_map = {sym: effective_state(sym, st, positions)
                 for sym, st in state_map_declared.items()}
    print(f"watchlist 共 {len(items)} 只 | states: " +
          ", ".join(f"{k}={v}" for k, v in pd.Series(list(state_map.values())).value_counts().items()))

    # ---- 共享 K 线（liquidity + technical 复用）----
    print("\n[fetch] 日 K (近 200 天, 新浪源) ...")
    klines: dict[str, pd.DataFrame] = {}
    for i, it in enumerate(items, 1):
        try:
            klines[it["symbol"]] = get_kline(it["symbol"], days=200)
        except Exception as e:
            print(f"  [{i}/{len(items)}] {it['symbol']} kline error: {e}")
            klines[it["symbol"]] = pd.DataFrame()
        time.sleep(0.15)
    print(f"  K 线已加载: {sum(1 for v in klines.values() if not v.empty)}/{len(items)}")

    # ---- 资金流共享缓存 ----
    print("\n[fetch] 龙虎榜近 30 天 + 主力资金 bulk 快照 ...")
    lhb_df = get_lhb_recent(days=30)
    flow_3d = get_fund_flow_rank("3日排行")
    flow_5d = get_fund_flow_rank("5日排行")
    print(f"  lhb={len(lhb_df)}, flow_3d={len(flow_3d)}, flow_5d={len(flow_5d)}")

    # ---- 基本面 ----
    print("\n[compute] fundamental_dim ...")
    fund_rows = []
    for i, it in enumerate(items, 1):
        try:
            fund_rows.append(fund_dim.compute_one(it["symbol"], it["name"]))
        except Exception as e:
            print(f"  {it['symbol']} fund error: {e}")
        time.sleep(0.2)
    fund_df = fund_dim.compose_dim(fund_rows)

    # ---- 主力资金 ----
    print("\n[compute] fund_flow_dim ...")
    flow_rows = []
    for it in items:
        try:
            flow_rows.append(flow_dim.compute_one(
                it["symbol"], it["name"], lhb_df=lhb_df, flow_3d=flow_3d, flow_5d=flow_5d,
            ))
        except Exception as e:
            print(f"  {it['symbol']} flow error: {e}")
        time.sleep(0.1)
    flow_df = flow_dim.compose_dim(flow_rows)

    # ---- 流动性 ----
    print("\n[compute] liquidity_dim ...")
    liq_rows = []
    for it in items:
        try:
            liq_rows.append(liq_dim.compute_one(it["symbol"], it["name"], kline=klines.get(it["symbol"])))
        except Exception as e:
            print(f"  {it['symbol']} liq error: {e}")
    liq_df = liq_dim.compose_dim(liq_rows)

    # ---- 技术 ----
    print("\n[compute] technical_dim ...")
    tech_rows = []
    for it in items:
        try:
            tech_rows.append(tech_dim.compute_one(it["symbol"], it["name"], kline=klines.get(it["symbol"])))
        except Exception as e:
            print(f"  {it['symbol']} tech error: {e}")
    tech_df = tech_dim.compose_dim(tech_rows)

    # ---- 筹码 ----
    print("\n[compute] chips_dim ...")
    chips_rows = []
    for it in items:
        try:
            chips_rows.append(chips_dim.compute_one(it["symbol"], it["name"], kline=klines.get(it["symbol"])))
        except Exception as e:
            print(f"  {it['symbol']} chips error: {e}")
    chips_df = chips_dim.compose_dim(chips_rows)

    # ---- 大盘 / 行业 / 个股 RS ----
    print("\n[compute] regime_dim ...")
    csi = get_csi300_kline()
    index_regime = regime_dim.get_index_regime()
    north_5d = get_north_total_inflow()
    north_pos = (north_5d is not None and north_5d > 0)
    print(f"  index_regime={index_regime}, north_5d_avg={north_5d}")
    regime_rows = []
    for it in items:
        try:
            regime_rows.append(regime_dim.compute_one(
                it["symbol"], it["name"], kline=klines.get(it["symbol"]), csi300=csi))
        except Exception as e:
            print(f"  {it['symbol']} regime error: {e}")
    regime_df = regime_dim.compose_dim(regime_rows, regime=index_regime, north_pos=north_pos)

    # ---- 板块动量 (SW2) ----
    print("\n[compute] sector_momentum_dim ...")
    csi_5d_ret = None
    if not csi.empty and len(csi) > 6:
        csi_5d_ret = float(csi["close"].iloc[-1] / csi["close"].iloc[-6] - 1)
    sector_scores = sector_dim.compute_sector_scores(csi_5d_ret)
    print(f"  打分板块数: {len(sector_scores)}")
    if not sector_scores.empty:
        top5 = sector_scores.head(8)[["name", "ret_5d", "ret_30d", "pos_60d",
                                       "sector_rsi", "breakout_flag", "sector_rank"]]
        print("  TOP 8 启动信号板块（高 breakout/低位/未过热）:")
        print(top5.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    sector_df = sector_dim.map_to_symbols([it["symbol"] for it in items], sector_scores)

    # ---- 新闻情绪 ----
    print("\n[compute] news_dim ...")
    news_df = get_news_recent(days=3, refresh=True)
    sw2_lookup = dict(zip(sector_df["symbol"], sector_df["sw2_name"]))
    news_rows = []
    for it in items:
        sw2 = sw2_lookup.get(it["symbol"])
        try:
            news_rows.append(news_dim.compute_one(it["symbol"], it["name"], sw2, news_df))
        except Exception as e:
            print(f"  {it['symbol']} news error: {e}")
    news_out = news_dim.compose_dim(news_rows)
    if not news_out.empty:
        hits = news_out[(news_out["news_direct_hits"] + news_out["news_sector_hits"]) > 0]
        print(f"  命中新闻的股票: {len(hits)}/{len(news_out)}")
        risk_hits = news_out[news_out["news_risk_tags"].apply(lambda t: bool(t))]
        if not risk_hits.empty:
            print(f"  ⚠ 触发风险事件: {len(risk_hits)} 只")
            for _, r in risk_hits.iterrows():
                print(f"    {r['symbol']}: {r['news_events']}")

    # ---- 合并 ----
    keep_f = ["symbol", "name", "sector", "pe_ttm", "pe_self_percentile",
              "pe_sector_percentile", "roe_ttm", "veto_hit", "fundamental_rank"]
    keep_g = ["symbol", "big_order_inflow_3d_amount", "north_holding_pct",
              "north_heavy_holding_flag", "lhb_count_30d", "fund_flow_rank"]
    keep_l = ["symbol", "volume_ratio", "volume_ratio_5d_avg", "turnover_today",
              "liq_signal", "liquidity_rank"]
    keep_t = ["symbol", "rsi_value", "rsi_strength", "macd_strength",
              "obv_strength", "tech_raw", "tech_rank"]
    keep_c = ["symbol", "chips_concentration", "cost_deviation", "chips_rank"]
    keep_r = ["symbol", "stock_rs_20d", "stock_rs_60d", "regime_rank", "index_regime"]
    keep_s = ["symbol", "sw2_code", "sw2_name", "sector_ret_5d", "sector_ret_30d",
              "sector_pos_60d", "sector_rsi", "sector_breakout", "sector_rank"]
    keep_n = ["symbol", "news_direct_hits", "news_sector_hits", "news_bull_score",
              "news_bear_score", "news_events", "news_sample", "news_risk_tags", "news_rank"]

    out_df = (fund_df[keep_f]
              .merge(flow_df[keep_g], on="symbol", how="left")
              .merge(liq_df[keep_l], on="symbol", how="left")
              .merge(tech_df[keep_t], on="symbol", how="left")
              .merge(chips_df[keep_c], on="symbol", how="left")
              .merge(regime_df[keep_r], on="symbol", how="left")
              .merge(sector_df[keep_s], on="symbol", how="left")
              .merge(news_out[keep_n] if not news_out.empty
                     else pd.DataFrame({c: [] for c in keep_n}),
                     on="symbol", how="left"))

    # 风险标签合并 (含新闻事件)
    risk_map = {}
    for r in (liq_rows + tech_rows):
        risk_map.setdefault(r["symbol"], []).extend(r.get("risk_tags") or [])
    for r in fund_rows:
        risk_map.setdefault(r["symbol"], []).extend(r.get("risk_tags") or [])
    for r in news_rows:
        risk_map.setdefault(r["symbol"], []).extend(r.get("news_risk_tags") or [])
    out_df["risk_tags"] = out_df["symbol"].map(lambda s: ";".join(sorted(set(risk_map.get(s, [])))))

    # ---- composite (8 维归一化, 完整版) ----
    weights = {"fundamental_rank": 0.22, "fund_flow_rank": 0.18,
               "liquidity_rank": 0.12, "chips_rank": 0.08,
               "regime_rank": 0.07, "tech_rank": 0.05,
               "sector_rank": 0.18, "news_rank": 0.10}
    total_w = sum(weights.values())
    score = pd.Series(0.0, index=out_df.index)
    for col, w in weights.items():
        r = pd.to_numeric(out_df[col], errors="coerce").fillna(1.0)
        score = score + w * (1 - r)
    out_df["partial_composite_raw"] = score / total_w

    # ---- 过热 / 极超卖下跌中 降权 ----
    rsi = pd.to_numeric(out_df["rsi_value"], errors="coerce")
    # RSI >70 线性衰减到 RSI=85 时 0.5×（避免追高）
    overheat = ((rsi - 70) / 15).clip(0, 1).fillna(0)
    # RSI <25 且 7 日内创新低 = 接飞刀，0.6× 降权（用 tech_raw=0 + RSI 极低代理）
    extreme_oversold = ((25 - rsi) / 10).clip(0, 1).fillna(0) * 0.4
    penalty = (0.5 * overheat + extreme_oversold).clip(0, 0.7)
    out_df["overheat_penalty"] = penalty
    out_df["partial_composite"] = out_df["partial_composite_raw"] * (1 - penalty)
    out_df.loc[out_df["veto_hit"] == True, "partial_composite"] = 0.0

    out_df = out_df.sort_values("partial_composite", ascending=False).reset_index(drop=True)

    # ---- 状态机 + decision resolver ----
    decided = resolve_batch(out_df, state_map=state_map)

    # 摘要表
    summary_cols = ["symbol", "name", "state", "decision", "confidence",
                    "fundamental_rank", "fund_flow_rank", "liquidity_rank",
                    "chips_rank", "regime_rank", "tech_rank", "sector_rank",
                    "news_rank", "sw2_name", "risk_tags"]
    print("\n========= 决策摘要（8 维完整版） =========")
    print(decided[summary_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # 决策计数
    print("\n决策分布:", decided["decision"].value_counts().to_dict())

    # 保存 CSV + JSON
    today = pd.Timestamp.today().strftime("%Y%m%d")
    csv_path = Path(args.out) if args.out else REPO / "results" / "decisions" / f"partial_{today}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    decided.to_csv(csv_path, index=False)

    # 结构化 JSON 报告
    json_path = csv_path.with_suffix(".json")
    report = {
        "scan_date": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "market_regime": str(decided["index_regime"].iloc[0]) if "index_regime" in decided else "unknown",
        "decisions": [
            {
                "symbol": r["symbol"],
                "name": r["name"],
                "state": r["state"],
                "decision": r["decision"],
                "confidence": round(float(r["confidence"]), 3),
                "composite": round(float(r["composite"]), 3),
                "ranks": {k.replace("_rank", ""): round(float(r[k]), 3)
                          for k in ("fundamental_rank", "fund_flow_rank", "liquidity_rank",
                                    "chips_rank", "regime_rank", "tech_rank",
                                    "sector_rank", "news_rank")
                          if pd.notna(r.get(k))},
                "sector": (r.get("sw2_name") if pd.notna(r.get("sw2_name")) else None),
                "drivers": r["dec_drivers"],
                "risks": r["dec_risks"],
                "gates_failed": r["dec_gates_failed"],
            }
            for _, r in decided.iterrows()
        ],
    }
    import json as _json
    with open(json_path, "w", encoding="utf-8") as f:
        _json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {csv_path}")
    print(f"保存: {json_path}")


if __name__ == "__main__":
    main()
