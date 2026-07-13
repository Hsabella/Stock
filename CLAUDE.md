# CLAUDE.md — Stock 多维决策引擎

A 股自选股决策引擎：收盘后从 8 个维度给 watchlist 打分 + 状态机出 BUY/HOLD/REDUCE/STOP 建议，**非自动交易系统**（最终下单由人拍板）。详细用法看 [USAGE.md](./USAGE.md)，术语看 [GLOSSARY.md](./GLOSSARY.md)。

> **2026-07 重构（quant/ 新系统）**：watchlist 打分器收益已证伪（38 天全样本 BUY 组合 -11.9% 跑输自选池），项目重构为 qlib 量化系统，见下方 quant 段落与 [docs/quant/phase1_report.md](./docs/quant/phase1_report.md)。旧引擎照跑到二期切换。

## quant/ 新系统（qlib 重构，第一期已完成）

```bash
# 全部用独立 venv: .venv-quant（pyqlib+lightgbm+akshare）
.venv-quant/bin/python -m quant.live.daily_brief      # 晨间仓位建议(用户每日唯一必看,
                                                      #  cron 工作日 09:00 → results/brief/latest.md)
.venv-quant/bin/python -m quant.data.bootstrap        # 更新 qlib 数据包(chenditc, 日更)
.venv-quant/bin/python -m quant.data.universe          # 重建 PIT 股票池 csi_union(~1800只)
.venv-quant/bin/python -m quant.data.checks            # 数据质检(每次更新必跑)
.venv-quant/bin/python -m quant.data.akshare_fetcher --dataset em_fundflow --refresh  # 资金流滚动累积
                                                      # (每周; --refresh 清断点, 周期性增量必加)
.venv-quant/bin/python -m quant.data.akshare_fetcher --dataset financials --refresh   # 财报摘要(PIT)
.venv-quant/bin/python -m quant.data.industry          # 行业标签快照(申万缓存+新浪补缺)
.venv-quant/bin/python -m quant.data.index_daily       # 指数日线(300/1000/800)落仓
.venv-quant/bin/python -m quant.features.ext_features  # 重建扩展特征 parquet
.venv-quant/bin/python -m quant.features.neutralize --pred results/quant/pred_lgb_rolling.parquet  # 行业+市值中性化(研究用)
.venv-quant/bin/python -m quant.research.ic_report     # 单因子周频 IC 体检
.venv-quant/bin/python -m quant.model.baseline --model lgb --mode rolling --tag <tag>  # 滚动训练
.venv-quant/bin/python -m quant.backtest.run_backtest --pred results/quant/pred_<tag>.parquet --start 2023-01-04  # [--neutralize] [--topk N]
.venv-quant/bin/python -m quant.backtest.overlay --daily results/quant/bt_<tag>_daily.csv  # 择时叠加
.venv-quant/bin/python -m quant.research.evaluate --gates v2  # 验收报告(v2 及格线: 超额≥4%/IR≥0.5/分年≥2/3正)
.venv-quant/bin/python -m pytest tests/quant/ -q
```

关键事实：股票池=300+500+1000 成分 PIT 并集；周度调仓 open-to-open；成本足额建模；
所有评估 2023 起全样本外（walk-forward）。**v2 迭代已执行完毕（2026-07-13），结论仍
NO-GO**：主线（+roe_delta 财报因子、限池中证800、embargo 修正）超额 -2.2%→**+3.89%**、
夏普 0.79、MDD 减半，但 IR/分年稳定性/成本翻倍未过线（alpha 绑定中小盘行情，25/26
反向）；中性化和 regime 特征两个假设被实测证伪（过程与数据见
[docs/quant/iteration_plan_v2.md](./docs/quant/iteration_plan_v2.md) 各 Step 实测小节 +
[docs/quant/v2_report.md](./docs/quant/v2_report.md)）。**选股封存观察，项目收敛为
"指数 ETF + 择时红绿灯 + 排雷器"**；红绿灯信号每日落 `results/brief/signal_log.csv` 攒实盘对账。
用户日常只看 `results/brief/latest.md`（晨间红绿灯）；旧引擎 BUY 名单已证伪勿采信，仅风险预警可用。
数据在 `~/.qlib/`（bin 包 + quant_warehouse parquet），不进 git；**东财资金流仅 120 天
历史，需每周跑 fetcher 累积**（已挂 cron 周五 19:30 `scripts/weekly_data.sh`：fetcher+bootstrap+checks，
连同工作日 09:00 `scripts/morning_brief.sh` 于 2026-07-10 装入 crontab）；
macOS 下 qlib 大面板调用必须有 `__main__` 保护。

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
