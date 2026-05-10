#!/usr/bin/env python3
"""读取最新决策 JSON → 生成易读的 Markdown 报告.

用法:
    python3 scripts/generate_report.py                      # 取最新 partial_*.json
    python3 scripts/generate_report.py --json <path>        # 指定 JSON
    python3 scripts/generate_report.py --out <path.md>      # 指定输出
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

DECISION_ICON = {
    "BUY": "🟢", "ADD": "🟢", "CONTINUE_WATCH": "🟡", "HOLD": "🟡",
    "DROP": "🔴", "REDUCE": "🟠", "STOP": "⛔", "TAKE": "💰",
    "AVOID": "⛔", "NO_ACTION": "⚪",
}
DECISION_ORDER = ["BUY", "ADD", "TAKE", "REDUCE", "STOP", "HOLD",
                  "CONTINUE_WATCH", "DROP", "AVOID", "NO_ACTION"]


def fmt_rank(v) -> str:
    if v is None:
        return "—"
    return f"{v*100:.0f}%"


def render(data: dict) -> str:
    L = []
    scan = data.get("scan_date", "")
    regime = data.get("market_regime", "?")
    decisions = data.get("decisions", [])

    L.append(f"# 决策报告 · {scan}")
    L.append("")
    L.append(f"**大盘**: `{regime}` | **股票池**: {len(decisions)} 只")
    L.append("")

    # ---- 摘要计数 ----
    counts = {}
    for d in decisions:
        counts[d["decision"]] = counts.get(d["decision"], 0) + 1
    summary = " · ".join(
        f"{DECISION_ICON.get(k,'')} **{k}**: {counts[k]}"
        for k in DECISION_ORDER if k in counts
    )
    L.append(f"## 摘要\n{summary}")
    L.append("")

    # ---- 分组 ----
    groups: dict[str, list[dict]] = {}
    for d in decisions:
        groups.setdefault(d["decision"], []).append(d)

    # ---- 行动组（BUY/ADD/REDUCE/STOP/TAKE）详情 ----
    actionable = [k for k in DECISION_ORDER if k in groups
                  and k in ("BUY", "ADD", "REDUCE", "STOP", "TAKE")]
    for kind in actionable:
        items = sorted(groups[kind], key=lambda x: -x["confidence"])
        L.append(f"## {DECISION_ICON.get(kind,'')} {kind} · {len(items)} 只")
        L.append("")
        for d in items:
            L.append(f"### {d['symbol']} {d['name']}  `confidence={d['confidence']:.2f}`")
            ranks = d.get("ranks", {})
            r_parts = [f"{k} {fmt_rank(v)}" for k, v in ranks.items()]
            L.append(f"- **ranks**: " + " · ".join(r_parts))
            L.append(f"- **state**: `{d.get('state','')}`")
            if d.get("drivers"):
                L.append("- **drivers**:")
                for x in d["drivers"]:
                    L.append(f"  - {x}")
            if d.get("risks"):
                L.append("- **risks**:")
                for x in d["risks"]:
                    L.append(f"  - ⚠ {x}")
            L.append("")

    # ---- 观察组（HOLD/CONTINUE_WATCH）紧凑表 ----
    watch_kinds = [k for k in ("HOLD", "CONTINUE_WATCH") if k in groups]
    for kind in watch_kinds:
        items = sorted(groups[kind], key=lambda x: -x["confidence"])
        L.append(f"## {DECISION_ICON.get(kind,'')} {kind} · {len(items)} 只")
        L.append("")
        L.append("| symbol | name | conf | fund | flow | liq | chips | regime | tech | risks |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for d in items:
            r = d.get("ranks", {})
            risks = ", ".join(d.get("risks", []))[:50]
            L.append(f"| {d['symbol']} | {d['name']} | {d['confidence']:.2f} "
                     f"| {fmt_rank(r.get('fundamental'))} | {fmt_rank(r.get('fund_flow'))} "
                     f"| {fmt_rank(r.get('liquidity'))} | {fmt_rank(r.get('chips'))} "
                     f"| {fmt_rank(r.get('regime'))} | {fmt_rank(r.get('tech'))} | {risks} |")
        L.append("")

    # ---- 排除组（DROP/AVOID/NO_ACTION）简洁列表 ----
    exclude_kinds = [k for k in ("DROP", "AVOID", "NO_ACTION") if k in groups]
    for kind in exclude_kinds:
        items = sorted(groups[kind], key=lambda x: -x["confidence"])
        L.append(f"## {DECISION_ICON.get(kind,'')} {kind} · {len(items)} 只")
        L.append("")
        for d in items:
            reason = ""
            if d.get("gates_failed"):
                reason = " ← " + "; ".join(d["gates_failed"])
            elif d.get("risks"):
                reason = " ← " + "; ".join(d["risks"][:2])
            L.append(f"- {d['symbol']} **{d['name']}** `conf={d['confidence']:.2f}`{reason}")
        L.append("")

    # ---- 备注 ----
    L.append("---")
    L.append("")
    L.append("> **使用提示**: BUY 信号需人工审核 drivers/risks 后再决策；"
             "改 `watchlist.yaml` 中的 state 为 HELD 即可纳入持仓跟踪。"
             "止损/止盈触发后用 `engine.state_machine.mark_exited()` 写入 cooldown。")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="输入 JSON（默认取 results/decisions/ 最新）")
    ap.add_argument("--out", default=None, help="输出 markdown 路径")
    args = ap.parse_args()

    if args.json:
        json_path = Path(args.json)
    else:
        files = sorted((REPO / "results" / "decisions").glob("partial_*.json"))
        if not files:
            sys.exit("无 partial_*.json，先运行 scripts/run_partial_engine.py")
        json_path = files[-1]
    print(f"读取: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    md = render(data)

    out_path = Path(args.out) if args.out else json_path.with_suffix(".md")
    out_path.write_text(md, encoding="utf-8")
    print(f"输出: {out_path}")
    print(f"\n{'─'*60}\n{md[:2000]}\n... (截断, 完整请看文件)")


if __name__ == "__main__":
    main()
