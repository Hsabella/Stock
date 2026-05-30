# CLAUDE.md — Stock 多维决策引擎

A 股自选股决策引擎：收盘后从 8 个维度给 watchlist 打分 + 状态机出 BUY/HOLD/REDUCE/STOP 建议，**非自动交易系统**（最终下单由人拍板）。详细用法看 [USAGE.md](./USAGE.md)，术语看 [GLOSSARY.md](./GLOSSARY.md)。

## 常用命令

```bash
# 一键全套（引擎 + 报告 + forward 校验，cron 入口）
./scripts/daily_run.sh

# 分步
python3 scripts/run_partial_engine.py                 # 主引擎 → results/decisions/partial_<今日>.{csv,json,md}
python3 scripts/forward_check.py --decision YYYYMMDD --horizons 1 3 5 --refresh
python3 compute_factor_ic.py                           # 逐因子 IC → results/factor_ic.csv
python3 scripts/reweight_backtest.py                   # 离线再加权回测（选最优权重，可加 --ret-col ret_T+3）

# 测试
python3 -m pytest tests/ -q

# 可视化驾驶舱
streamlit run dashboard/app.py                         # http://localhost:8501
```

## 架构

- `factors/<dim>/dim.py` — 8 个维度各自算 raw→rank：fundamental / fund_flow / liquidity / technical / chips / regime / sector_momentum / news
- `engine/ranker.py` — 横截面 rank（**0=最强，1=最弱**）
- `engine/decision_resolver.py` — gates + BUY/DROP 规则 + drivers/risks（阈值在文件头）
- `scripts/run_partial_engine.py` — 主引擎，组装 composite（权重字典在 ~228 行）
- `dashboard/` — Streamlit 看板，只读 `results/`（app / data_loader / metrics / controls 各单一职责）
- `results/decisions/` · `results/forward/` · `results/factor_ic.csv` — 产出

## 关键约定 / 现状

- **rank 0 = 最强**（百分位，反直觉但是业内传统）。
- **composite 越大越优**；BUY 阈值 0.55；gates：fund/flow rank ≤ 0.60、liq rank ≤ 0.70。
- **2026-05 再加权（重要）**：composite 评分只用 5 维 `fund_flow .40 / regime .25 / news .15 / fundamental .10 / chips .10`；`sector_momentum / liquidity / technical` 权重置 0（14 天 forward 实测对 T+1 反向）；**overheat penalty 已停用**。改权重前先用 `reweight_backtest.py` 在存量数据上复测 IC，别拍脑袋。
- **验收信号好坏**：看驾驶舱"因子体检 / Forward兑现"——composite IC 是否为正、BUY 是否跑赢 DROP，且 T+3/T+5 方向一致（抗单日噪声）。当前基线 composite IC：+0.095(T+1) / +0.217(T+3)。
- **数据源限流**：东财 `stock_individual_info_em` 等对本机 IP 限流，已用 baidu/同花顺/新浪/申万规避；部分维度偶发缺数据属正常，不阻断主流程。

## 提交规范（重要）

- 单次 commit 计入绩效统计的**代码文件新增行数 ≤ 2000**（超了整笔不计），细则见全局 `~/.claude/CLAUDE.md`。`.md`/配置/lock 文件不计入。
- commit message 用 `feat/fix/docs/...` 前缀，结尾带 `Co-Authored-By: Claude ...`。
- 设计文档在 `docs/superpowers/specs/`，实施计划在 `docs/superpowers/plans/`。
