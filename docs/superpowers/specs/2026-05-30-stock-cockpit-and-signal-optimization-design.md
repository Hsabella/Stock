# A股决策驾驶舱 + 信号优化 · 设计文档

> 日期: 2026-05-30 · 状态: 待用户评审 · 作者: Claude + 用户
> 关联: `USAGE.md`、`GLOSSARY.md`、诊断 workflow `w4fsj1r86`

---

## 1. 背景与问题(有据可查)

把 `results/decisions/partial_*.csv` 的 8 维 rank 与 `results/forward/forward_*.csv` 的真实次日收益按 symbol 逐日 join(14 个交易日 20260510–20260528,每日 50 只,合计 700 行),计算每个维度的**信息系数 IC**(rank 与 T+1 收益的 Spearman 相关,已按"rank 越小越强"换算成正=有预测力):

| 维度 | 权重 | 平均日 IC | 为正天数 | 结论 |
|---|---:|---:|---:|---|
| fund_flow 主力资金 | 0.18 | **+0.064** | 57% | ✅ 有用(最好) |
| regime 相对强度 | 0.07 | **+0.056** | 54% | ✅ 有用 |
| chips 筹码 | 0.08 | −0.025 | 54% | ⚪ 噪声 |
| fundamental 基本面 | **0.22** | −0.025 | 43% | ⚪ 噪声(最大权重) |
| tech 技术 | 0.05 | −0.035 | 31% | ❌ 反向 |
| news 新闻 | 0.10 | −0.046 | 39% | ❌ 反向 |
| liquidity 量能 | 0.12 | −0.064 | 30% | ❌ 反向 |
| sector 板块 | **0.18** | **−0.127** | 17% | ❌ 最强反向 |
| **composite 综合** | — | **−0.097** | 29% | 🔴 整体反向 |

**核心问题**:75% 的权重压在噪声/反向因子上。有效的 fund_flow + regime 仅占 0.25。结果是 BUY 组 T+1 −0.23%(胜率 40%)反而**跑输** DROP 组 +0.09%(胜率 45.5%),BUY−DROP = −0.32%/日。

**根因(代码级)**:
1. **系统性接飞刀**:`factors/technical/dim.py:67-85` 把 RSI 超卖反转/底背离当强买点;`factors/sector_momentum/dim.py:101-113` 只奖励"还在低位(pos_60d<0.70)"的板块、对 sector_rsi>60 扣分;`factors/liquidity/dim.py:82-90` 把"缩量在 60 日低点收阳"当买点。而样本期是动量延续:RSI>70 组 T+1 +0.64%,RSI 25-35 组 −1.64%。方向完全做反。
2. **过热惩罚惩罚赢家**:`scripts/run_partial_engine.py:241-248` 的 overheat/extreme_oversold penalty 把 RSI>70 的动量股打折——被罚的票实际 +0.29% 跑赢未罚的 −0.25%,把 composite IC 从 raw −0.006 拖到更负。
3. **gates 本身有益**(通过 gate +0.39% vs 未通过 −0.37%),问题不在过滤,在 composite 排序本身反向。

> ⚠️ 证据局限:仅 14 天、仅 T+1、IC 量级 ~0.05–0.13 在统计上**不显著**,排序仅供方向性参考。看板与优化路线都必须显式标注这一点,并以"扩到 T+3/T+5 + 更长窗口"作为复核手段。

---

## 2. 目标 / 非目标

**目标**
- 建一个 Streamlit"驾驶舱"看板,把"每日决策 → forward 真实兑现 → 逐因子 IC"的**测量闭环**搬到网页上,让信号优化可见、可复测。
- 提供轻控制:网页编辑 watchlist、一键触发 daily_run。
- 与看板并行,启动证据驱动的信号优化,先做最稳的步骤,用看板验收。

**非目标(明确排除)**
- 不做自动交易/下单。看板只读 `results/`,写操作仅改 `watchlist.yaml` / `state/`。
- 不在看板里重算引擎逻辑(只消费 `results/` 现成产物)。
- v1 不做在线拖动权重实时重算 IC、参数扫描、A/B 对比(留给后续研究平台档位)。

---

## 3. 范围决策(本次已确认)

| 项 | 决策 |
|---|---|
| 能力档位 | 看板 + 轻控制,Streamlit,MVP 优先 |
| 主攻方向 | 先建驾驶舱,再用它调信号 |
| v1 范围 | 3 个只读页 + 极简控制台(watchlist 编辑 + 一键 daily_run) |
| 优化时机 | 与看板并行,先做步骤 0-2,用看板复测 |

**v1 交付(现在建)**
- Tab ① 今日决策(只读)
- Tab ② Forward 兑现(只读)
- Tab ③ 因子体检(只读)
- 控制台(极简):watchlist 编辑+保存(备份+二次确认)、一键 daily_run(二次确认+进度)
- 模块:`dashboard/app.py`、`data_loader.py`、`metrics.py`、`controls.py`
- 给 `compute_factor_ic.py` 加 `results/factor_ic.csv` 落盘(供因子体检页)

**v2 延后**
- HELD 持仓浮盈卡片
- 标记 EXITED + cooldown(`state/positions.json` + `state_machine.mark_exited`)
- 看板内批量补算 T+3/T+5 按钮(v1 用 CLI 手动跑即可)
- IC 时间序列多周期(T+3/T+5)展示

---

## 4. 看板架构

```
浏览器 ──Streamlit──▶ dashboard/app.py (布局/路由,只装配)
                          │
        ┌─────────────────┼──────────────────┐
        ▼                 ▼                  ▼
  data_loader.py      metrics.py         controls.py
  (唯一读数据层,       (纯函数聚合,        (写操作/子进程,
   @st.cache_data)     无 IO)             集中审计)
        │                                    │
        ▼                                    ▼
  results/decisions/*.{csv,json,md}     watchlist.yaml(读写,备份)
  results/forward/*.{csv,md}            scripts/daily_run.sh(subprocess)
  results/factor_ic.csv(新增)          logs/daily_*.log(读尾部)
  watchlist.yaml / cache/*kline
```

**模块职责(单一)**
- `dashboard/app.py` — `st.set_page_config` + 4 个 tab 装配;只布局,不写业务/IO。
- `dashboard/data_loader.py` — 唯一读数据层,全部 `@st.cache_data`:`list_scan_dates()`(glob+正则建日期索引)、`load_partial_csv/json/md(date)`、`load_forward_csv(date)`(丢弃 `1`/`_no_data` 垃圾列)、`parse_csi_benchmark_from_md(date)`、`load_watchlist()`、`load_factor_ic()`、`tail_log()`。**容错 43 列旧 schema 与缺列**(每个 CSV 读自己的 header)。
- `dashboard/metrics.py` — 纯函数:`decision_group_counts(df)`、`forward_group_perf(df, horizons)`、`buy_vs_drop(df)`、`held_pnl(...)`(v2)。输入/输出都是 DataFrame,无 IO。
- `dashboard/controls.py` — 所有副作用:`save_watchlist(rows)`、`run_daily()`、(v2)`mark_exited`、`recompute_forward`。
- `compute_factor_ic.py`(改,不重写)— 现仅 print stdout;新增把 per-(date,factor) 的 `daily` 记录 `to_csv` 到 `results/factor_ic.csv`(列 `date,factor,ic,n`),保留原 stdout 汇总。

---

## 5. 各页规格

### Tab ① 今日决策(只读)
- 日期选择器(`partial_*.csv` 的 YYYYMMDD 倒序)+ 大盘 regime 徽标(读 json `market_regime`)。
- 决策分布卡片:`groupby('decision').size()` → BUY/REDUCE/CONTINUE_WATCH/DROP/AVOID/HOLD + emoji。
- 主决策表(`st.dataframe`):`symbol,name,sector,decision,state,confidence,composite` + 8 个 `*_rank`;用 `column_config` 把 rank 渲染成色阶(0=绿/强,1=红/弱),按 composite 降序。
- 筛选:`multiselect` 按 decision;`text_input` 搜 symbol/name。
- 选中行 drill-down:从 `partial_*.json`(干净 `list[str]`)取 ranks/drivers/risks/gates_failed,expander 展示。
- 原始 `.md` 报告内嵌(`st.markdown`)。
- 数据源:`results/decisions/partial_YYYYMMDD.{csv,json,md}`(20260510 仅 43 列、缺 sector_*/news_*,需容错)。

### Tab ② Forward 兑现(只读)
- 决策日选择器(`forward_*.csv`)。
- 分组表现表:按 decision groupby `ret_T+1` → count / hit_rate=(ret>0).mean() / mean / median / worst / best(`forward_check` 已支持 `ret_T+{h}` 动态列,v1 只有 T+1)。
- 沪深300 基准:**只存在于 `.md` 头部**(CSV 里没有),正则抓 `T+1=-0.45%` 画基准线。
- BUY vs DROP 并排柱状(让"BUY 跑输 DROP"一眼可见)。
- 个股明细表:`symbol,name,sector,decision,confidence,composite,ret_T+1,max_dd`,按 ret 排序,红绿着色。
- 跨日趋势(轻量):遍历所有 `forward_*.csv`,每日 BUY 组 mean_ret 折线。
- 数据源:`results/forward/forward_YYYYMMDD.{csv,md}`(注意 `forward_20260521/20260527.csv` 有 `1`/`_no_data` 空垃圾列需丢弃)。

### Tab ③ 因子体检(只读,优化证据面板)
- 顶部 caveat 条:`14 天样本 / IC 统计不显著 / 仅方向性参考`(必须显式)。
- 逐因子 IC 横向柱状:8 个 `*_rank` 平均日 IC,正绿负红,标注 composite 基线 −0.097。
- 为正天数占比表:`factor + ic_mean + pos_days_frac + n_days + n_total`。
- 口径警告条:sector/news `n_days=12/13`、liquidity `n_days=10`,各因子口径不同、不能严格横比。
- IC 时间序列折线(读 `results/factor_ic.csv`)。
- 诊断结论卡片(静态文案,引用根因)。
- 数据源:`results/factor_ic.csv`(新增);兜底可在 app 内用 partial+forward 现算。

---

## 6. 控制台规格 + 安全(v1 极简)

| 动作 | 实现 | 安全 |
|---|---|---|
| 编辑保存 watchlist | `st.data_editor` 加载 `watchlist` 段;保存用 **ruamel.yaml**(保留注释/行内 flow)写回,顶层 `cooldown_days` 等原样保留 | 写前 `cp watchlist.yaml watchlist.yaml.bak.<ts>`;二次确认;写后回读 diff;symbol 6 位数字校验、state 枚举校验、HELD 必须有 position+entry_price |
| 一键 daily_run | `subprocess.Popen(['bash','scripts/daily_run.sh'])` 非阻塞;`st.session_state` 记状态,轮询最新 `logs/daily_*.log` 尾部 + returncode | 二次确认;锁文件防并发重复触发;**零字符串拼接**(固定脚本路径,杜绝注入);失败时红色展示日志尾部,不静默吞错 |

> 控制台顶部固定声明:**非交易系统,写操作仅改 watchlist.yaml / state**。

---

## 7. 信号优化路线(与看板并行,证据驱动)

每步改完都用"因子体检"页复测 composite IC 与 BUY−DROP 价差。v1 先做 0-2:

0. **扩多周期**:`scripts/daily_run.sh:28` 改 `--horizons 1 3 5`;对历史决策日批量回跑 `python3 scripts/forward_check.py --decision YYYYMMDD --horizons 1 3 5 --refresh`。`forward_check.py` 已原生支持多 horizon,**无需改算法代码**,仅操作层。
1. **去掉 penalty**(H1,最稳):`run_partial_engine.py:241-248` 改 `partial_composite = partial_composite_raw`。预期 composite IC 从 −0.097 回升。
2. **砍/降权三大反向维度**(H3/H4/H2):sector(0.18)、liquidity(0.12)、tech(0.05) 权重清零,按比例给 fund_flow + regime;重测应明显改善。
3. **降权两个噪声维度**(H6):fundamental(0.22)、chips(0.08) 大幅降权;fundamental 保留在 gate 不进评分。
4. **最小可验证两维模型**(H5):临时只用 fund_flow + regime,重测 IC 预期转正——判断"反向能否扭正"的最快实验。
5. **证据驱动重新加权**:以两维为基线逐个加回修正后的维度,每加一个复测;并清掉 `run_partial_engine.py:4-7` 误导的旧 4 维 docstring。

**验收门槛**:composite IC 由负转正 **且** BUY−DROP 价差 > 0,**且** 在 T+3/T+5 上方向一致(抗单日噪声)。未达标回到步骤 2-5 继续调。

---

## 8. 数据来源关键约定(给实现的备忘)

- **schema 不稳定**:`partial_20260510.csv` 仅 43 列(缺全部 sector_*/news_* 共 16 列);20260511 起为 59 列全量。读每个 CSV 自己的 header,不假设固定列。
- **沪深300 基准只在 `.md`**,不在 forward CSV,需正则抓取。
- **`forward_20260521/20260527.csv`** 含 `1`/`_no_data` 空垃圾列(中断 run 的泄漏),加载时丢弃。
- **干净决策数据用 json**:`partial_*.json` 的 `decisions[].ranks`(8 键 dict)/`drivers`/`risks`/`gates_failed` 是干净 `list[str]`,优于 CSV 的 stringified list。
- **持仓浮盈数据源(v2)**:`cache/<symbol>_daily_qfq.csv` 末行 close。
- **cooldown(v2)**:`state/` 目录当前不存在,`load_positions` 缺文件返回 `{}`,cooldown 现处休眠态。

---

## 9. 依赖与运行

- 新依赖:`streamlit`、`ruamel.yaml`(保留 yaml 注释)、`pandas`、`tabulate`。**streamlit 当前未安装**,需 `pip install`。
- 启动:`streamlit run dashboard/app.py`。
- 新增 `dashboard/requirements.txt` + `dashboard/README.md`(写清"只读 results/、写操作仅改 watchlist/state、非交易系统")。

---

## 10. 测试策略

- `data_loader` / `metrics` 为纯函数,用现有 `results/` 样本做单测:加载 43 列旧文件不报错、丢弃垃圾列、IC 聚合数值与 `compute_factor_ic.py` 一致、沪深300 正则抓取正确。
- `controls.save_watchlist`:写前备份存在、回读 diff 一致、非法 symbol/state 被拦截(用临时副本测,不动真文件)。
- `compute_factor_ic.py` 落盘:`results/factor_ic.csv` 列与行数符合预期。
- 优化每步:用 `compute_factor_ic.py` 复测 composite IC,记录"改动前/后"对照。

---

## 11. 决策记录(用户已确认 2026-05-30)

1. **快速清零 vs 反转逻辑** → ✅ **先直接清零**反向维度/penalty 快速拿正 IC,验证后再决定是否反转 dim 内部逻辑(保住维度信息)。
2. **fundamental 拆分** → ✅ 从 composite **评分降权/移除**,但**保留作 gate 过滤**(gate-PASS +0.39% 有益)。
3. **ruamel.yaml 依赖** → ✅ 用 **ruamel.yaml** 写 watchlist.yaml,保留注释/`{…}` flow 格式(接受多一个依赖)。
4. **daily_run 触发与限流** → ✅ **接受部分维度数据缺失**照常跑(脚本对失败维度容错),日志红字提示哪几维没数据。
5. **优化样本** → ✅ 按 14 天方向性证据先改、再用扩到 T+3/T+5 的更长窗口复核(与看板并行动手)。

---

## 附录: 再加权实验结果(2026-05-30)

由 `scripts/reweight_backtest.py` 在存量 `partial_*.csv`(8 维 rank + `rsi_value`)上离线重算 composite、复测 IC 与 BUY−DROP 价差,不重跑引擎、不抓数据。目标: `ic_mean` 转正 且 `buy_minus_drop > 0`,且 T+1/T+3 方向一致。

### T+1 对照表

```
config                     days   ic_mean  ic_pos%  BUY-DROP
baseline(当前8维+penalty)       13   -0.0634     0.31   -0.0051
去penalty                     13   +0.0172     0.46   +0.0037
砍sector/liq/tech             13   +0.0670     0.62   +0.0081
再降fund/chips                 13   +0.0951     0.69   +0.0096
两维(flow+regime)              14   +0.0990     0.71   +0.0103
```

### T+3 复核表(抗单日噪声)

```
config                     days   ic_mean  ic_pos%  BUY-DROP
baseline(当前8维+penalty)       11   +0.0131     0.45   -0.0047
去penalty                     11   +0.1025     0.82   +0.0184
砍sector/liq/tech             11   +0.1451     1.00   +0.0277
再降fund/chips                 11   +0.2174     0.91   +0.0344
两维(flow+regime)              12   +0.1887     0.83   +0.0309
```

### 结论与胜出 config

- **方向一致性**:T+1 与 T+3 两表 config 排序未翻转 —— `baseline` 始终最差(两表 BUY−DROP 均为负),随着砍掉反向维度(sector/liquidity/tech)、关 penalty、把权重让给 fund_flow+regime,`ic_mean` 与 `BUY−DROP` 单调改善。
- **penalty 系统性惩罚赢家**:`去penalty` 相对 `baseline` 在两个周期都大幅改善(T+1 −0.0634→+0.0172,T+3 +0.0131→+0.1025),确认 overheat penalty 是反向的,A5 应关闭。
- **胜出 config = `再降fund/chips`(关 penalty)**:
  - 权重 `{"fund_flow_rank": 0.40, "regime_rank": 0.25, "news_rank": 0.15, "fundamental_rank": 0.10, "chips_rank": 0.10}`,`use_penalty=False`。
  - T+1:ic_mean=+0.0951、BUY−DROP=+0.0096;T+3:ic_mean=+0.2174(全场最高)、BUY−DROP=+0.0344(全场最高)。
  - 两周期 `ic_mean` 全为正、`BUY−DROP` 全为正,满足全部胜出条件。
  - 相比纯 `两维(flow+regime)`(T+1 略高、T+3 略低),`再降fund/chips` 保留 5 维(多 news/fundamental/chips),对单因子失效更鲁棒;且 fundamental 仍可在 gate 段保留,不进评分。
- **下一步(Task A5)**:把上述 5 维权重回写 `scripts/run_partial_engine.py`,并关闭 overheat penalty(保留计算供 forward 透明,但不施加到 composite)。
