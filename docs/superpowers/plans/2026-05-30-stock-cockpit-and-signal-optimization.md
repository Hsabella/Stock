# A股决策驾驶舱 + 信号优化 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建一个 Streamlit 只读驾驶舱(3 页)+ 极简控制台,并用"离线再加权回测"把当前反向的 composite 信号(IC≈−0.097)证据驱动地扭正。

**Architecture:** 看板只消费 `results/` 现成产物(partial/forward CSV·JSON·MD),分 `app/data_loader/metrics/controls` 四个单一职责模块。信号优化**不重跑引擎**——历史 `partial_*.csv` 已存全部 8 维 rank + `rsi_value`,用 `reweight_backtest.py` 在存量数据上重算 composite 并复测 IC,选出最优权重后再回写 `run_partial_engine.py`。

**Tech Stack:** Python 3.10、pandas 2.2(已装)、streamlit + ruamel.yaml(需装)、pytest(开发期需装)。

参考设计: `docs/superpowers/specs/2026-05-30-stock-cockpit-and-signal-optimization-design.md`

---

## 文件结构

| 文件 | 职责 | 新建/改 |
|---|---|---|
| `scripts/reweight_backtest.py` | 离线再加权回测:存量 rank→重算 composite→IC/BUY−DROP | 新建 |
| `compute_factor_ic.py` | 逐因子 IC,新增 `results/factor_ic.csv` 落盘 + 参数化 RET_COL | 改 |
| `scripts/daily_run.sh` | forward 校验改 `--horizons 1 3 5` | 改(第28行) |
| `scripts/run_partial_engine.py` | 应用最优权重 + 关 penalty | 改(228-249) |
| `dashboard/app.py` | Streamlit 布局/路由,装配 4 个 tab | 新建 |
| `dashboard/data_loader.py` | 唯一读数据层(@st.cache_data) | 新建 |
| `dashboard/metrics.py` | 纯函数聚合(分组统计/IC/BUY-DROP) | 新建 |
| `dashboard/controls.py` | 写操作:save_watchlist / run_daily | 新建 |
| `dashboard/requirements.txt` `dashboard/README.md` | 依赖与启动说明 | 新建 |
| `tests/` | pytest:metrics / data_loader / reweight / controls | 新建 |

---

## Task 0: 分支 + 开发环境

**Files:** 无(环境)

- [ ] **Step 1: 建特性分支**(当前在 main,先隔离)

```bash
cd /Users/wangbo/VSCodeProjects/Stock
git checkout -b feat/cockpit-and-signal-fix
```

- [ ] **Step 2: 装依赖**

```bash
python3 -m pip install streamlit "ruamel.yaml" pytest
```

- [ ] **Step 3: 验证可导入**

Run: `python3 -c "import streamlit, ruamel.yaml, pytest, pandas; print('deps ok', pandas.__version__)"`
Expected: 打印 `deps ok 2.2.3`

- [ ] **Step 4: 建测试目录骨架**

```bash
mkdir -p tests dashboard
touch tests/__init__.py
```

- [ ] **Step 5: Commit**

```bash
git add tests/__init__.py
git commit -m "chore: 建 feat/cockpit-and-signal-fix 分支与测试骨架

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

# Part 1 — 信号优化(离线、快、先做)

> 全部基于 `results/` 存量文件,2 秒一轮,不抓数据。先把"反向能否扭正"用证据验证,再回写引擎。

## Task A1: forward 扩到 T+3/T+5(数据基座)

**Files:**
- Modify: `scripts/daily_run.sh:28`

- [ ] **Step 1: 改 daily_run.sh 第 28 行 horizons**

把:
```bash
        "$PY" scripts/forward_check.py --decision "$PREV" --horizons 1 --refresh \
```
改成:
```bash
        "$PY" scripts/forward_check.py --decision "$PREV" --horizons 1 3 5 --refresh \
```

- [ ] **Step 2: 批量回补历史 forward 的 T+3/T+5**(`forward_check.py` 原生支持 `--horizons`,无需改算法)

```bash
cd /Users/wangbo/VSCodeProjects/Stock
PY="/Library/Frameworks/Python.framework/Versions/3.10/bin/python3"
for f in results/decisions/partial_*.json; do
  d=$(echo "$f" | sed -E 's/.*partial_([0-9]{8})\.json/\1/')
  echo "=== backfill $d ==="
  "$PY" scripts/forward_check.py --decision "$d" --horizons 1 3 5 --refresh || echo "  (exit=$?, 多为 T+3/T+5 数据未到, 跳过)"
done
```

- [ ] **Step 3: 验证至少部分 forward 出现 ret_T+3 列**

Run: `head -1 results/forward/forward_20260518.csv | tr ',' '\n' | grep -n 'ret_T'`
Expected: 看到 `ret_T+1`、`ret_T+3`、`ret_T+5`(早期日期可能仅 T+1,属正常;较早决策日的 T+3/T+5 现在已有 K 线)

- [ ] **Step 4: Commit**

```bash
git add scripts/daily_run.sh results/forward/
git commit -m "feat(forward): daily_run 与历史回补扩展到 T+1/T+3/T+5

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task A2: compute_factor_ic.py 落盘 + 参数化 RET_COL

**Files:**
- Modify: `compute_factor_ic.py`
- Test: `tests/test_factor_ic.py`

- [ ] **Step 1: 写失败测试**(IC 落盘后文件存在且列正确)

```python
# tests/test_factor_ic.py
import subprocess, os, csv

REPO = "/Users/wangbo/VSCodeProjects/Stock"

def test_factor_ic_csv_written():
    out = os.path.join(REPO, "results", "factor_ic.csv")
    if os.path.exists(out):
        os.remove(out)
    subprocess.run(["python3", "compute_factor_ic.py"], cwd=REPO, check=True)
    assert os.path.exists(out), "compute_factor_ic.py 应写 results/factor_ic.csv"
    with open(out) as f:
        header = next(csv.reader(f))
    assert header == ["date", "factor", "ic", "n"]
    # 至少含 fund_flow_rank 与 composite 两行
    txt = open(out).read()
    assert "fund_flow_rank" in txt and "composite" in txt
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /Users/wangbo/VSCodeProjects/Stock && python3 -m pytest tests/test_factor_ic.py -v`
Expected: FAIL(`results/factor_ic.csv` 不存在)

- [ ] **Step 3: 在 `compute_factor_ic.py` 顶部把 RET_COL 改成可由环境变量覆盖**

把第 23 行:
```python
RET_COL = "ret_T+1"
```
改成:
```python
RET_COL = os.environ.get("IC_RET_COL", "ret_T+1")
```

- [ ] **Step 4: 在 `main()` 的 `print("MACHINE_READABLE_END")` 之后、`return`/函数结束前,追加落盘代码**

在 `compute_factor_ic.py` 的 `main()` 末尾(第 130 行 `print("MACHINE_READABLE_END")` 之后)加:
```python
    # ---- 落盘 per-(date,factor) IC 时间序列, 供看板因子体检页 ----
    out_csv = os.path.join(os.path.dirname(BASE), "results", "factor_ic.csv")
    rows = []
    for col in RANK_COLS + ["composite"]:
        for date, ic, nn in daily[col]:
            rows.append({"date": date, "factor": col, "ic": round(ic, 6), "n": nn})
    pd.DataFrame(rows, columns=["date", "factor", "ic", "n"]).to_csv(out_csv, index=False)
    print(f"\n[written] {out_csv}  ({len(rows)} rows)")
```

- [ ] **Step 5: 运行确认通过**

Run: `cd /Users/wangbo/VSCodeProjects/Stock && python3 -m pytest tests/test_factor_ic.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add compute_factor_ic.py tests/test_factor_ic.py results/factor_ic.csv
git commit -m "feat(ic): compute_factor_ic 落盘 factor_ic.csv 并支持多周期 RET_COL

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task A3: reweight_backtest.py — 离线再加权回测工具

**Files:**
- Create: `scripts/reweight_backtest.py`
- Test: `tests/test_reweight.py`

- [ ] **Step 1: 写失败测试**(用当前权重重算的 composite IC 应≈ 存量 composite 的 IC,且为负)

```python
# tests/test_reweight.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import reweight_backtest as rb

CURRENT = {"fundamental_rank": 0.22, "fund_flow_rank": 0.18, "liquidity_rank": 0.12,
           "chips_rank": 0.08, "regime_rank": 0.07, "tech_rank": 0.05,
           "sector_rank": 0.18, "news_rank": 0.10}
TWO_DIM = {"fund_flow_rank": 0.5, "regime_rank": 0.5}

def test_current_config_reproduces_negative_ic():
    res = rb.run(CURRENT, use_penalty=True)
    assert res["days"] >= 10
    assert res["ic_mean"] is not None and res["ic_mean"] < 0  # 复现反向

def test_two_dim_beats_current():
    cur = rb.run(CURRENT, use_penalty=True)["ic_mean"]
    two = rb.run(TWO_DIM, use_penalty=False)["ic_mean"]
    assert two > cur  # 两维模型应优于当前 8 维
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /Users/wangbo/VSCodeProjects/Stock && python3 -m pytest tests/test_reweight.py -v`
Expected: FAIL(`reweight_backtest` 不存在)

- [ ] **Step 3: 实现 `scripts/reweight_backtest.py`**

```python
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
```

- [ ] **Step 4: 运行确认测试通过**

Run: `cd /Users/wangbo/VSCodeProjects/Stock && python3 -m pytest tests/test_reweight.py -v`
Expected: PASS(两维模型 IC > 当前)

- [ ] **Step 5: Commit**

```bash
git add scripts/reweight_backtest.py tests/test_reweight.py
git commit -m "feat(opt): 离线再加权回测工具 reweight_backtest

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task A4: 跑实验,选最优权重(决策步)

**Files:** 无(实验记录)

- [ ] **Step 1: 跑 T+1 对照表**

Run: `cd /Users/wangbo/VSCodeProjects/Stock && python3 scripts/reweight_backtest.py`
Expected: 一张 5 行表;`baseline` ic_mean 为负、`两维` 与 `砍sector/liq/tech` 明显更高(目标转正)

- [ ] **Step 2: 跑 T+3 复核(抗单日噪声)**

Run: `python3 scripts/reweight_backtest.py --ret-col ret_T+3`
Expected: 方向与 T+1 一致(同一批 config 排序不应翻转)

- [ ] **Step 3: 记录结论到设计文档**

把两张表贴到 `docs/superpowers/specs/2026-05-30-stock-cockpit-and-signal-optimization-design.md` 末尾新增的 `## 附录: 再加权实验结果(YYYY-MM-DD)` 小节,选出 `ic_mean` 转正且 `BUY-DROP>0`、T+1/T+3 方向一致的**胜出 config** 作为下一步要回写引擎的权重。

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-05-30-stock-cockpit-and-signal-optimization-design.md
git commit -m "docs(opt): 记录再加权实验结果与胜出权重

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task A5: 把胜出 config 回写 run_partial_engine.py

**Files:**
- Modify: `scripts/run_partial_engine.py:228-249`

> 用 Task A4 选出的权重替换;关 penalty(已验证 penalty 惩罚赢家)。下方为示例(以"砍 sector/liq/tech + fund_flow/regime 加权"为例,具体数值以 A4 胜出 config 为准)。

- [ ] **Step 1: 替换权重字典(228-232 行)**

把:
```python
    weights = {"fundamental_rank": 0.22, "fund_flow_rank": 0.18,
               "liquidity_rank": 0.12, "chips_rank": 0.08,
               "regime_rank": 0.07, "tech_rank": 0.05,
               "sector_rank": 0.18, "news_rank": 0.10}
```
改成(数值替换为 A4 胜出 config):
```python
    # 2026-05-30 再加权: 砍掉反向维度 sector/liquidity/tech, 权重让给 fund_flow+regime.
    # 依据 reweight_backtest 实验 (见 specs 附录). fundamental 仍保留在 gate, 不进评分.
    weights = {"fund_flow_rank": 0.45, "regime_rank": 0.25,
               "news_rank": 0.10, "fundamental_rank": 0.10, "chips_rank": 0.10}
```

- [ ] **Step 2: 关 penalty(240-248 行)**——保留计算供 forward 透明,但不施加到 composite

把第 248 行:
```python
    out_df["partial_composite"] = out_df["partial_composite_raw"] * (1 - penalty)
```
改成:
```python
    # 2026-05-30: penalty 经验证系统性惩罚赢家(spearman(penalty,ret)=+0.107), 暂停施加.
    # 仍写出 overheat_penalty 列以便 forward/看板观察, 但 composite 不再打折.
    out_df["partial_composite"] = out_df["partial_composite_raw"]
```

- [ ] **Step 3: 删除误导的旧 4 维 docstring**

查看 `scripts/run_partial_engine.py` 文件头(第 4-7 行)若有 `0.30/0.25/0.15/0.05` 旧权重说明,改为指向真实权重一行注释:`# composite 权重见 build 段 (228 行); 历史 4 维注释已删`。

- [ ] **Step 4: 冒烟——引擎能跑且产出新 partial(注意东财限流,部分维度缺失可接受)**

Run: `cd /Users/wangbo/VSCodeProjects/Stock && /Library/Frameworks/Python.framework/Versions/3.10/bin/python3 scripts/run_partial_engine.py 2>&1 | tail -25`
Expected: 看到"决策分布: {...}"与"保存:"路径;个别维度报限流属已知、不阻断

- [ ] **Step 5: 次日 forward 验收(人工跟进)**

在驾驶舱"因子体检/Forward兑现"页观察:新决策日的 composite IC 是否转正、BUY 组是否跑赢 DROP。未达标回 Task A4 调权重。

- [ ] **Step 6: Commit**

```bash
git add scripts/run_partial_engine.py
git commit -m "feat(engine): 再加权 composite(砍反向维度)并暂停 overheat penalty

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

# Part 2 — 驾驶舱看板(Streamlit,3 读页 + 极简控制台)

## Task B1: data_loader.py — 唯一读数据层

**Files:**
- Create: `dashboard/__init__.py`、`dashboard/data_loader.py`
- Test: `tests/test_data_loader.py`

- [ ] **Step 1: 写失败测试**(覆盖:列日期、43 列旧文件不崩、丢垃圾列、抓沪深300基准)

```python
# tests/test_data_loader.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dashboard import data_loader as dl

def test_list_scan_dates_desc():
    dates = dl.list_scan_dates()
    assert "20260529" in dates
    assert dates == sorted(dates, reverse=True)  # 倒序

def test_load_old_43col_file_ok():
    df = dl.load_partial_csv("20260510")  # 仅 43 列, 缺 sector_rank
    assert "decision" in df.columns
    assert "sector_rank" not in df.columns  # 容错: 不强造列

def test_forward_drops_junk_cols():
    df = dl.load_forward_csv("20260521")  # 含 '1'/'_no_data' 垃圾列
    assert "1" not in df.columns and "_no_data" not in df.columns
    assert "ret_T+1" in df.columns

def test_parse_csi_benchmark():
    val = dl.parse_csi_benchmark("20260528")  # md 头: T+1=-0.45%
    assert val is not None and abs(val - (-0.0045)) < 1e-6
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /Users/wangbo/VSCodeProjects/Stock && python3 -m pytest tests/test_data_loader.py -v`
Expected: FAIL(模块不存在)

- [ ] **Step 3: 实现 `dashboard/__init__.py`(空)与 `dashboard/data_loader.py`**

```python
# dashboard/__init__.py
```
```python
# dashboard/data_loader.py
"""唯一读数据层。只读 results/ 与 watchlist.yaml, 绝不重算引擎。
非 streamlit 环境下 @st.cache_data 退化为普通函数, 便于单测。
"""
import glob
import json
import os
import re

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEC_DIR = os.path.join(REPO, "results", "decisions")
FWD_DIR = os.path.join(REPO, "results", "forward")

try:
    import streamlit as st
    cache = st.cache_data
except Exception:  # 单测/无 streamlit 时
    def cache(func=None, **_):
        return func if func else (lambda f: f)

JUNK_COLS = {"1", "_no_data", "Unnamed: 0"}


@cache
def list_scan_dates():
    dates = []
    for p in glob.glob(os.path.join(DEC_DIR, "partial_*.csv")):
        m = re.search(r"partial_(\d{8})\.csv$", p)
        if m:
            dates.append(m.group(1))
    return sorted(set(dates), reverse=True)


@cache
def list_forward_dates():
    dates = []
    for p in glob.glob(os.path.join(FWD_DIR, "forward_*.csv")):
        m = re.search(r"forward_(\d{8})\.csv$", p)
        if m:
            dates.append(m.group(1))
    return sorted(set(dates), reverse=True)


@cache
def load_partial_csv(date):
    return pd.read_csv(os.path.join(DEC_DIR, f"partial_{date}.csv"))


@cache
def load_partial_json(date):
    with open(os.path.join(DEC_DIR, f"partial_{date}.json")) as f:
        return json.load(f)


@cache
def load_partial_md(date):
    p = os.path.join(DEC_DIR, f"partial_{date}.md")
    return open(p, encoding="utf-8").read() if os.path.exists(p) else ""


@cache
def load_forward_csv(date):
    df = pd.read_csv(os.path.join(FWD_DIR, f"forward_{date}.csv"))
    return df.drop(columns=[c for c in df.columns if c in JUNK_COLS], errors="ignore")


@cache
def parse_csi_benchmark(date, horizon=1):
    """从 forward md 头部抓 'T+1=-0.45%' 返回小数(-0.0045)。"""
    p = os.path.join(FWD_DIR, f"forward_{date}.md")
    if not os.path.exists(p):
        return None
    m = re.search(rf"T\+{horizon}=([-+]?\d+\.?\d*)%", open(p, encoding="utf-8").read())
    return float(m.group(1)) / 100 if m else None


@cache
def load_factor_ic():
    p = os.path.join(REPO, "results", "factor_ic.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame(columns=["date", "factor", "ic", "n"])
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /Users/wangbo/VSCodeProjects/Stock && python3 -m pytest tests/test_data_loader.py -v`
Expected: PASS(4 项)

- [ ] **Step 5: Commit**

```bash
git add dashboard/__init__.py dashboard/data_loader.py tests/test_data_loader.py
git commit -m "feat(dashboard): data_loader 读数据层(含旧schema/垃圾列容错)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task B2: metrics.py — 纯函数聚合

**Files:**
- Create: `dashboard/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_metrics.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dashboard import data_loader as dl, metrics as mx

def test_group_counts():
    df = dl.load_partial_csv("20260529")
    c = mx.decision_group_counts(df)
    assert c.get("BUY", 0) == 7  # 与 20260529 报告摘要一致

def test_forward_group_perf_buy_underperforms_drop():
    df = dl.load_forward_csv("20260522")
    perf = mx.forward_group_perf(df, "ret_T+1").set_index("decision")
    assert perf.loc["BUY", "mean_ret"] < perf.loc["DROP", "mean_ret"]  # 已知反向

def test_buy_vs_drop_spread_negative_sample():
    df = dl.load_forward_csv("20260522")
    spread = mx.buy_vs_drop(df, "ret_T+1")
    assert spread < 0
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /Users/wangbo/VSCodeProjects/Stock && python3 -m pytest tests/test_metrics.py -v`
Expected: FAIL(模块不存在)

- [ ] **Step 3: 实现 `dashboard/metrics.py`**

```python
# dashboard/metrics.py
"""纯函数聚合: 输入 DataFrame, 输出展示用 DataFrame/标量, 无 IO。"""
import pandas as pd


def decision_group_counts(df):
    return df["decision"].value_counts().to_dict()


def forward_group_perf(df, ret_col="ret_T+1",
                       groups=("BUY", "CONTINUE_WATCH", "DROP", "AVOID", "HOLD", "REDUCE")):
    rows = []
    for g in groups:
        sub = df[df["decision"] == g][ret_col].dropna()
        if len(sub) == 0:
            continue
        rows.append({
            "decision": g, "count": len(sub),
            "hit_rate": (sub > 0).mean(),
            "mean_ret": sub.mean(), "median_ret": sub.median(),
            "worst": sub.min(), "best": sub.max(),
        })
    return pd.DataFrame(rows)


def buy_vs_drop(df, ret_col="ret_T+1"):
    buy = df[df["decision"] == "BUY"][ret_col].dropna()
    drop = df[df["decision"] == "DROP"][ret_col].dropna()
    if len(buy) == 0 or len(drop) == 0:
        return float("nan")
    return buy.mean() - drop.mean()


def factor_ic_summary(ic_df):
    """factor_ic.csv -> 每因子 平均IC/为正占比/天数, 按 IC 降序。"""
    if ic_df.empty:
        return ic_df
    g = ic_df.groupby("factor")["ic"]
    out = pd.DataFrame({
        "ic_mean": g.mean(),
        "pos_frac": g.apply(lambda s: (s > 0).mean()),
        "n_days": g.size(),
    }).reset_index().sort_values("ic_mean", ascending=False)
    return out
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /Users/wangbo/VSCodeProjects/Stock && python3 -m pytest tests/test_metrics.py -v`
Expected: PASS(3 项)

- [ ] **Step 5: Commit**

```bash
git add dashboard/metrics.py tests/test_metrics.py
git commit -m "feat(dashboard): metrics 纯函数聚合(分组/forward/IC)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task B3: app.py 骨架 + Tab① 今日决策

**Files:**
- Create: `dashboard/app.py`

- [ ] **Step 1: 写 app 骨架与 Tab①**

```python
# dashboard/app.py
"""A股决策驾驶舱。启动: streamlit run dashboard/app.py
只读 results/; 写操作仅改 watchlist.yaml/state, 非交易系统。"""
import os
import sys

# `streamlit run dashboard/app.py` 只把 dashboard/ 放进 sys.path, 这里补上仓库根,
# 让 `from dashboard import ...` 与 tests 一致可用。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st  # noqa: E402

from dashboard import data_loader as dl, metrics as mx  # noqa: E402

st.set_page_config(page_title="A股决策驾驶舱", layout="wide")
st.title("📊 A股决策驾驶舱")
st.caption("只读 results/ · 写操作仅改 watchlist/state · **非交易系统**")

tab1, tab2, tab3, tab4 = st.tabs(["今日决策", "Forward 兑现", "因子体检", "控制台"])

DECISION_EMOJI = {"BUY": "🟢", "REDUCE": "🟠", "HOLD": "🟡",
                  "CONTINUE_WATCH": "🟡", "DROP": "🔴", "AVOID": "⛔"}
RANK_COLS = ["fundamental_rank", "fund_flow_rank", "liquidity_rank", "tech_rank",
             "chips_rank", "regime_rank", "sector_rank", "news_rank"]

with tab1:
    dates = dl.list_scan_dates()
    if not dates:
        st.warning("results/decisions 下暂无 partial_*.csv")
    else:
        date = st.selectbox("决策日期", dates, key="t1_date")
        j = dl.load_partial_json(date)
        st.markdown(f"**大盘**: `{j.get('market_regime', '?')}` · **股票池** {len(j.get('decisions', []))} 只")
        df = dl.load_partial_csv(date)
        counts = mx.decision_group_counts(df)
        st.write(" · ".join(f"{DECISION_EMOJI.get(k,'')}{k} {v}" for k, v in counts.items()))

        decs = sorted(df["decision"].dropna().unique().tolist())
        pick = st.multiselect("筛选 decision", decs, default=decs, key="t1_pick")
        kw = st.text_input("搜索 symbol/name", key="t1_kw").strip()
        view = df[df["decision"].isin(pick)]
        if kw:
            view = view[view["symbol"].astype(str).str.contains(kw) |
                        view["name"].astype(str).str.contains(kw)]
        show_cols = [c for c in ["symbol", "name", "sector", "decision", "state",
                                 "confidence", "composite"] + RANK_COLS if c in view.columns]
        sort_col = "composite" if "composite" in view.columns else "confidence"
        cfg = {c: st.column_config.ProgressColumn(c, min_value=0.0, max_value=1.0, format="%.2f")
               for c in RANK_COLS if c in view.columns}
        st.dataframe(view[show_cols].sort_values(sort_col, ascending=False),
                     use_container_width=True, hide_index=True, column_config=cfg)
        st.caption("提示: rank 0=最强(绿/满), 1=最弱。")

        sel = st.text_input("查看某 symbol 的 drivers/risks", key="t1_sel").strip()
        if sel:
            hit = next((d for d in j["decisions"] if str(d["symbol"]) == sel), None)
            if hit:
                st.json({k: hit.get(k) for k in ("decision", "confidence", "ranks",
                                                 "drivers", "risks", "gates_failed")})
            else:
                st.info("该 symbol 不在当日决策集")
        with st.expander("原始 .md 报告"):
            st.markdown(dl.load_partial_md(date) or "(无 md)")
```

- [ ] **Step 2: 启动冒烟(后台起服务 + 抓首页 + 检查无 traceback)**

> curl 拿 200 不够——Streamlit 即使脚本抛异常也会返回 200 并把错误显示在页内。所以同时抓 stderr 检查 `Traceback`/`ModuleNotFoundError`。

Run:
```bash
cd /Users/wangbo/VSCodeProjects/Stock
streamlit run dashboard/app.py --server.headless true --server.port 8765 >/tmp/st.log 2>&1 &
sleep 7
echo "HTTP: $(curl -s -o /dev/null -w '%{http_code}' http://localhost:8765/)"
grep -E "Traceback|ModuleNotFoundError|NameError" /tmp/st.log && echo "❌ 有异常" || echo "✅ 无异常"
pkill -f "streamlit run" || true
```
Expected: `HTTP: 200` 且 `✅ 无异常`

- [ ] **Step 3: Commit**

```bash
git add dashboard/app.py
git commit -m "feat(dashboard): app 骨架 + Tab① 今日决策

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task B4: Tab② Forward 兑现

**Files:**
- Modify: `dashboard/app.py`(填充 `with tab2:`)

- [ ] **Step 1: 在 app.py 末尾追加 `with tab2:` 块**

```python
with tab2:
    fdates = dl.list_forward_dates()
    if not fdates:
        st.warning("results/forward 下暂无 forward_*.csv")
    else:
        fdate = st.selectbox("决策日(看 T+N 兑现)", fdates, key="t2_date")
        fdf = dl.load_forward_csv(fdate)
        ret_cols = [c for c in fdf.columns if c.startswith("ret_T+")]
        ret_col = st.radio("周期", ret_cols, horizontal=True, key="t2_h") if ret_cols else "ret_T+1"
        csi = dl.parse_csi_benchmark(fdate, horizon=ret_col.split("+")[-1])
        if csi is not None:
            st.metric(f"沪深300 {ret_col} 基准", f"{csi:+.2%}")
        perf = mx.forward_group_perf(fdf, ret_col)
        st.dataframe(perf, use_container_width=True, hide_index=True,
                     column_config={c: st.column_config.NumberColumn(c, format="%.2f%%")
                                    for c in ["hit_rate", "mean_ret", "median_ret", "worst", "best"]})
        spread = mx.buy_vs_drop(fdf, ret_col)
        st.metric("BUY − DROP 价差 (负=BUY跑输, 当前核心问题)", f"{spread:+.2%}")
        if not perf.empty:
            st.bar_chart(perf.set_index("decision")["mean_ret"])
        st.markdown("**个股明细**")
        cols = [c for c in ["symbol", "name", "sector", "decision", "confidence",
                            "composite", ret_col, "max_dd"] if c in fdf.columns]
        st.dataframe(fdf[cols].sort_values(ret_col, ascending=False),
                     use_container_width=True, hide_index=True)
```

- [ ] **Step 2: 冒烟**(同 B3 Step 2 起停;人工切到"Forward 兑现"页确认表与柱状图渲染)

Run: `cd /Users/wangbo/VSCodeProjects/Stock && python3 -c "import ast; ast.parse(open('dashboard/app.py').read()); print('app.py syntax ok')"`
Expected: `app.py syntax ok`

- [ ] **Step 3: Commit**

```bash
git add dashboard/app.py
git commit -m "feat(dashboard): Tab② Forward 兑现(多周期+BUY-DROP)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task B5: Tab③ 因子体检

**Files:**
- Modify: `dashboard/app.py`(填充 `with tab3:`)

- [ ] **Step 1: 追加 `with tab3:` 块**

```python
with tab3:
    st.warning("⚠ 14 天样本 · IC 统计上不显著 · 仅方向性参考")
    ic_df = dl.load_factor_ic()
    if ic_df.empty:
        st.info("尚无 results/factor_ic.csv, 先运行 `python3 compute_factor_ic.py`")
    else:
        summ = mx.factor_ic_summary(ic_df)
        st.markdown("**逐因子平均日 IC**(正=有预测力, 负=反向)")
        st.bar_chart(summ.set_index("factor")["ic_mean"])
        st.dataframe(summ, use_container_width=True, hide_index=True,
                     column_config={"ic_mean": st.column_config.NumberColumn(format="%.4f"),
                                    "pos_frac": st.column_config.NumberColumn(format="%.2f")})
        comp = summ[summ["factor"] == "composite"]
        if not comp.empty:
            st.metric("composite 基线 IC", f"{comp.iloc[0]['ic_mean']:+.4f}")
        st.markdown("**IC 时间序列**")
        facs = st.multiselect("选因子", summ["factor"].tolist(),
                              default=["fund_flow_rank", "sector_rank", "composite"], key="t3_f")
        if facs:
            piv = ic_df[ic_df["factor"].isin(facs)].pivot_table(
                index="date", columns="factor", values="ic")
            st.line_chart(piv)
        st.info("诊断: penalty 系统性惩罚赢家 · technical/liquidity 接飞刀 · sector 反追高在动量市做反")
```

- [ ] **Step 2: 语法检查**

Run: `cd /Users/wangbo/VSCodeProjects/Stock && python3 -c "import ast; ast.parse(open('dashboard/app.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add dashboard/app.py
git commit -m "feat(dashboard): Tab③ 因子体检(IC 柱状+时间序列)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task B6: controls.py — 写操作(watchlist 保存 + 触发 daily_run)

**Files:**
- Create: `dashboard/controls.py`
- Test: `tests/test_controls.py`

- [ ] **Step 1: 写失败测试**(保存前备份、非法 symbol 被拦、注释保留)

```python
# tests/test_controls.py
import sys, os, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dashboard import controls as ct

def _tmp_yaml(tmp_path):
    src = os.path.join(os.path.dirname(__file__), "..", "watchlist.yaml")
    dst = tmp_path / "watchlist.yaml"
    shutil.copy(src, dst)
    return str(dst)

def test_save_creates_backup_and_persists(tmp_path):
    path = _tmp_yaml(tmp_path)
    rows = [{"symbol": "002716", "name": "湖南白银", "state": "HELD",
             "position": 500, "entry_price": 12.63}]
    ct.save_watchlist(rows, path=path)
    baks = [f for f in os.listdir(tmp_path) if f.startswith("watchlist.yaml.bak.")]
    assert baks, "应生成备份"
    reloaded = ct.load_watchlist_rows(path=path)
    assert any(r["symbol"] == "002716" and r["state"] == "HELD" for r in reloaded)

def test_invalid_symbol_rejected(tmp_path):
    path = _tmp_yaml(tmp_path)
    try:
        ct.save_watchlist([{"symbol": "ABC", "state": "WATCHING"}], path=path)
        assert False, "非法 symbol 应抛错"
    except ValueError:
        pass

def test_held_requires_position(tmp_path):
    path = _tmp_yaml(tmp_path)
    try:
        ct.save_watchlist([{"symbol": "002716", "state": "HELD"}], path=path)
        assert False, "HELD 缺 position/entry_price 应抛错"
    except ValueError:
        pass
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /Users/wangbo/VSCodeProjects/Stock && python3 -m pytest tests/test_controls.py -v`
Expected: FAIL(模块不存在)

- [ ] **Step 3: 实现 `dashboard/controls.py`**

```python
# dashboard/controls.py
"""所有写操作与子进程, 集中审计。写前备份, 严格校验, 零字符串拼接。"""
import os
import re
import shutil
import subprocess
import time

from ruamel.yaml import YAML

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WATCHLIST = os.path.join(REPO, "watchlist.yaml")
VALID_STATES = {"NONE", "WATCHING", "HELD", "EXITED"}
_yaml = YAML()
_yaml.preserve_quotes = True


def load_watchlist_rows(path=WATCHLIST):
    with open(path, encoding="utf-8") as f:
        data = _yaml.load(f)
    return [dict(r) for r in (data.get("watchlist") or [])]


def _validate(rows):
    for r in rows:
        sym = str(r.get("symbol", ""))
        if not re.fullmatch(r"\d{6}", sym):
            raise ValueError(f"symbol 必须 6 位数字: {sym!r}")
        state = r.get("state", "WATCHING")
        if state not in VALID_STATES:
            raise ValueError(f"非法 state: {state!r} (允许 {VALID_STATES})")
        if state == "HELD" and (r.get("position") in (None, "") or r.get("entry_price") in (None, "")):
            raise ValueError(f"HELD 必须有 position+entry_price: {sym}")


def save_watchlist(rows, path=WATCHLIST):
    """校验→备份→用 ruamel 写回 watchlist 段(保留顶层键与注释)。"""
    _validate(rows)
    if os.path.exists(path):
        ts = time.strftime("%Y%m%d_%H%M%S")
        shutil.copy(path, f"{path}.bak.{ts}")
    with open(path, encoding="utf-8") as f:
        data = _yaml.load(f) or {}
    data["watchlist"] = rows
    with open(path, "w", encoding="utf-8") as f:
        _yaml.dump(data, f)
    return load_watchlist_rows(path)  # 回读供 diff


def run_daily():
    """非阻塞触发 scripts/daily_run.sh; 返回 Popen。固定参数, 无字符串拼接。"""
    return subprocess.Popen(["bash", os.path.join(REPO, "scripts", "daily_run.sh")],
                            cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def latest_log_tail(n=40):
    logs = sorted([f for f in os.listdir(os.path.join(REPO, "logs")) if f.startswith("daily_")])
    if not logs:
        return "(无日志)"
    p = os.path.join(REPO, "logs", logs[-1])
    return "".join(open(p, encoding="utf-8", errors="ignore").readlines()[-n:])
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /Users/wangbo/VSCodeProjects/Stock && python3 -m pytest tests/test_controls.py -v`
Expected: PASS(3 项)

- [ ] **Step 5: Commit**

```bash
git add dashboard/controls.py tests/test_controls.py
git commit -m "feat(dashboard): controls 写操作(watchlist 保存校验+备份, daily_run 触发)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task B7: Tab④ 控制台

**Files:**
- Modify: `dashboard/app.py`(填充 `with tab4:`)

- [ ] **Step 1: 追加 `with tab4:` 块**

```python
with tab4:
    st.warning("⚠ 非交易系统 · 写操作仅改 watchlist.yaml · 保存自动备份")
    rows = ct.load_watchlist_rows()
    import pandas as pd
    edited = st.data_editor(pd.DataFrame(rows), num_rows="dynamic", use_container_width=True,
                            key="t4_edit",
                            column_config={"state": st.column_config.SelectboxColumn(
                                options=["NONE", "WATCHING", "HELD", "EXITED"])})
    confirm = st.checkbox("我确认保存对 watchlist.yaml 的修改", key="t4_ok")
    if st.button("💾 保存 watchlist", disabled=not confirm):
        try:
            new_rows = [ {k: v for k, v in r.items() if pd.notna(v)}
                         for r in edited.to_dict("records") ]
            reloaded = ct.save_watchlist(new_rows)
            st.success(f"已保存并备份, 共 {len(reloaded)} 行")
        except ValueError as e:
            st.error(f"校验失败, 未写入: {e}")

    st.divider()
    if "daily_proc" not in st.session_state:
        st.session_state.daily_proc = None
    running = st.session_state.daily_proc is not None and st.session_state.daily_proc.poll() is None
    if st.button("▶ 一键 daily_run(约数分钟)", disabled=running):
        if st.session_state.get("t4_run_ok"):
            st.session_state.daily_proc = ct.run_daily()
            st.toast("已在后台启动 daily_run")
        else:
            st.warning("请先勾选下方确认框")
    st.checkbox("我确认触发全量 8 维 daily_run", key="t4_run_ok")
    st.caption("运行状态: " + ("🟢 进行中" if running else "⚪ 空闲"))
    with st.expander("最新运行日志尾部"):
        st.code(ct.latest_log_tail())
```

- [ ] **Step 2: 在 app.py 顶部 import 处补 `from dashboard import controls as ct`**

确认 app.py 头部 import 段含:
```python
from dashboard import data_loader as dl, metrics as mx, controls as ct
```

- [ ] **Step 3: 语法检查 + 全量测试**

Run: `cd /Users/wangbo/VSCodeProjects/Stock && python3 -c "import ast; ast.parse(open('dashboard/app.py').read())" && python3 -m pytest tests/ -v`
Expected: 语法 ok;`tests/` 全绿

- [ ] **Step 4: Commit**

```bash
git add dashboard/app.py
git commit -m "feat(dashboard): Tab④ 控制台(watchlist 编辑保存 + 一键 daily_run)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task B8: 依赖文件 + README + 端到端冒烟

**Files:**
- Create: `dashboard/requirements.txt`、`dashboard/README.md`

- [ ] **Step 1: 写 `dashboard/requirements.txt`**

```
streamlit>=1.30
ruamel.yaml>=0.18
pandas>=1.5
```

- [ ] **Step 2: 写 `dashboard/README.md`**

```markdown
# A股决策驾驶舱

只读 `results/` 的 Streamlit 看板 + 极简控制台。**非交易系统**:写操作仅改 `watchlist.yaml`。

## 启动
    pip install -r dashboard/requirements.txt
    streamlit run dashboard/app.py

## 标签页
- 今日决策:任一扫描日的 50 只票决策 + 8 维 rank + drivers/risks
- Forward 兑现:决策的 T+1/T+3/T+5 真实兑现、BUY−DROP 价差、沪深300 基准
- 因子体检:逐因子 IC(哪个维度有预测力/反向)、IC 时间序列
- 控制台:编辑 watchlist(自动备份)、一键 daily_run、看运行日志

数据由 `scripts/daily_run.sh` 产出;IC 由 `compute_factor_ic.py` 落盘 `results/factor_ic.csv`。
```

- [ ] **Step 3: 端到端冒烟(起服务→抓首页→查异常→关)**

```bash
cd /Users/wangbo/VSCodeProjects/Stock
streamlit run dashboard/app.py --server.headless true --server.port 8765 >/tmp/st.log 2>&1 &
sleep 7
echo "HTTP: $(curl -s -o /dev/null -w '%{http_code}' http://localhost:8765/)"
grep -E "Traceback|ModuleNotFoundError|NameError" /tmp/st.log && echo "❌ 有异常" || echo "✅ 无异常"
pkill -f "streamlit run" || true
```
Expected: `HTTP: 200` 且 `✅ 无异常`

- [ ] **Step 4: Commit**

```bash
git add dashboard/requirements.txt dashboard/README.md
git commit -m "docs(dashboard): requirements + README

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## 收尾:整合验收(人工)

- [ ] `python3 -m pytest tests/ -v` 全绿
- [ ] `streamlit run dashboard/app.py` 四个 tab 都能渲染;切日期/筛选/搜索正常
- [ ] 控制台改一行 watchlist 保存→确认生成 `.bak`、回读 diff 正确
- [ ] `python3 scripts/reweight_backtest.py` 选出的胜出 config 已回写 `run_partial_engine.py`
- [ ] 次日 forward 数据出来后,在"因子体检"页确认 composite IC 较 −0.097 改善、"Forward兑现"页 BUY 不再跑输 DROP;未达标回 Task A4
- [ ] 决定是否合并分支 / 是否注册 streamlit 为常驻(`finishing-a-development-branch`)
