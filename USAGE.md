# 多维决策引擎使用说明

> A 股自选股 **8 维决策引擎** — 收盘后扫描 + 多维 rank 加权 + 状态机决策
> 看不懂的术语？查 [GLOSSARY.md](./GLOSSARY.md)

---

## 一句话理解

每天收盘后，把你 watchlist 里的股票从 8 个维度做横截面打分：

| 维度 | 权重 | 看什么 |
|---|---|---|
| fundamental | 0.22 | PE/PB 自身+行业分位、ROE、Hard Veto |
| fund_flow | 0.18 | 主力 3/5 日净流入、北向持仓、龙虎榜 |
| sector_momentum | 0.18 | 申万二级板块**启动信号**（反追高） |
| liquidity | 0.12 | 量比 + 换手率拐点 |
| news | 0.10 | 多源新闻流命中 + 利好/利空词典 + 风险事件 |
| chips | 0.08 | 90% 筹码区间、当前价相对成本均价 |
| regime | 0.07 | 大盘趋势 + 个股 vs 沪深 300 |
| technical | 0.05 | RSI/MACD/OBV 拐点 |

按你的持仓状态（NONE/WATCHING/HELD）输出 **BUY / HOLD / REDUCE / STOP** 等动作建议，附带 3-5 条人话理由和风险提示。

**这不是自动交易系统**——它是给你做参考的"决策助手"。最终下不下单、下多少、什么时机由你拍板。

---

## 日常工作流

### 方式 A: 自动跑（已装 crontab）

```
0 10,11,13,14,15 * * 1-5  fetch_news_only.py    # 盘中每小时累积新闻样本
30 15            * * 1-5  daily_run.sh          # 收盘后跑决策 + forward 对照
```

第二天起床看 `results/decisions/partial_<昨日>.md` + `results/forward/forward_<前日>.md`。

查看 / 调整：`crontab -l` / `crontab -e`

### 方式 B: 手动跑

```bash
cd /Users/wangbo/VSCodeProjects/Stock

# 一键跑全套（引擎 + 报告 + forward 校验）
./scripts/daily_run.sh

# 或者分步
python3 scripts/run_partial_engine.py     # 跑 8 维引擎，约 90 秒
python3 scripts/generate_report.py        # 生成 markdown 报告
python3 scripts/forward_check.py --refresh # 前一份决策 vs 今日实际
```

输出目录：
- `results/decisions/partial_YYYYMMDD.{csv,json,md}` — 当日决策
- `results/forward/forward_YYYYMMDD.{csv,md}` — 前一日 → 今日真实表现
- `logs/daily_YYYYMMDD_HHMMSS.log` — 完整运行日志（保留近 30 份）
- `cache/news/history.csv` — 滚动 30 天新闻流（cron 自动累积）

---

## 看报告怎么操作

### 🟢 BUY 区
**含义**: 通过基本面 Hard Veto + 三道 gates（fund/flow/liq rank）+ composite ≥ 0.55

**操作**:
1. 看 drivers — 至少 3 条买入理由（含**板块启动信号** / 新闻利好等）
2. 看 risks — 有没有 RSI 过热、PE 偏高、**板块过热**、**新闻负面事件**
3. **人工决策**：要买就把 `watchlist.yaml` 里这只票的 `state` 改为 `HELD`，加上 `position`（手数）和 `entry_price`（成交均价）

### 🟠 REDUCE / ⛔ STOP（HELD 票）
持仓中、引擎觉得该减/清仓 → 人工判断后减仓，仓位减完后改 yaml 把 `state` 设为 `EXITED`

### 🟡 CONTINUE_WATCH / 🔴 DROP / ⛔ AVOID
- **CONTINUE_WATCH**: 通过 gates 但 composite < 0.55，留在关注池
- **DROP**: gates fail（fund_rank>60% 或 liq_rank>70%），从 watchlist 移除
- **AVOID**: Hard Veto 命中（连亏 / 商誉过高 / 高负债 / ST），永远别碰

---

## 反追高设计 ⚠️

本引擎在两个层面对"追高"做了显式抑制（见 `feedback_anti_chasing_high.md`）：

1. **个股 overheat_penalty**: RSI > 70 → composite × 0.5，RSI > 85 时直接 ×0.5
2. **板块启动信号**:
   - 板块 5d 涨但 60d 位置已超 70% → **不算启动**，不加分
   - 板块 RSI > 70 → **过热惩罚**，扣分
   - "连涨天数"**已从因子中移除**（避免接最后一棒）

所以 BUY 列表不会把"已经涨完"的标的推到前面，但**也不能预测当天的事件冲击**（如运价骤跌）。

---

## watchlist.yaml 写法

```yaml
cooldown_days: 10                   # 卖出后多少天内不再 BUY 同一只
default_stop_strategy: "atr_2"      # 默认止损策略
default_max_position_pct: 0.10      # 单票最大仓位

watchlist:
  - { symbol: "002716", name: "湖南白银", state: "HELD",
      position: 500, entry_price: 12.630 }
  - { symbol: "688001", name: "华兴源创", state: "WATCHING" }
  - { symbol: "300750", name: "宁德时代", state: "WATCHING",
      target_entry: [240, 250] }
```

| state | 含义 | engine 输出 |
|---|---|---|
| `NONE` | 普通候选 | BUY / NO_ACTION |
| `WATCHING` | 关注未持仓 | BUY / CONTINUE_WATCH / DROP |
| `HELD` | 持仓中 | HOLD / ADD / REDUCE / STOP / TAKE |
| `EXITED` | cooldown 中 | NO_ACTION |

---

## Forward 校验

每天 cron 跑完决策后自动校验前一份决策的真实表现：

```bash
python3 scripts/forward_check.py --decision 20260510 --horizons 1 3 5 --refresh
```

输出每个决策组的：命中率 / 平均收益 / 中位 / 最差 / 最好 / 最大回撤，外加沪深300 同期基准。

> ⚠ A 股 K 线接口（新浪）通常**当晚 18:00 后**才有当日收盘。盘后立刻跑会提示"数据未到，明天再跑"。

---

## 项目结构

```
Stock/
├── factors/              # 8 个维度
│   ├── _kline.py             # 共享日 K 线（新浪，含换手率）
│   ├── fundamental/          # 基本面 + Hard Veto
│   ├── fund_flow/            # 主力资金 + 北向 + 龙虎榜
│   ├── liquidity/            # 量比/换手率拐点
│   ├── technical/            # RSI/MACD/KDJ/BOLL/OBV
│   ├── chips/                # 筹码集中度
│   ├── regime/               # 大盘 + RS
│   ├── sector_momentum/      # 申万 SW2 启动信号
│   └── news/                 # 多源新闻流 + 词典命中
├── engine/
│   ├── ranker.py             # 横截面 rank/zscore
│   ├── decision_resolver.py  # gates + 决策规则 + drivers/risks
│   └── state_machine.py      # 状态转移
├── scripts/
│   ├── daily_run.sh              # 一键全套（cron 入口）
│   ├── run_partial_engine.py     # 8 维主引擎
│   ├── fetch_news_only.py        # 仅累积新闻（盘中小时跑）
│   ├── forward_check.py          # T+N 真实收益对照
│   ├── generate_report.py        # markdown 报告
│   └── backtest_composite_week.py # 历史 anchor 回测
├── watchlist.yaml        # 自选股池（手动维护）
├── state/positions.json  # cooldown 跟踪（自动）
├── cache/                # 数据缓存（自动）
│   ├── kline/                # 个股日 K
│   ├── news/history.csv      # 滚动 30 天新闻
│   ├── sector_momentum/      # 板块数据
│   └── ...
├── results/
│   ├── decisions/        # 每日决策 csv/json/md
│   └── forward/          # 每日 forward 校验
└── logs/                 # cron + 手动跑的日志
```

---

## 常见问题

**Q: 为什么 rank 是 0% 表示最强（反直觉）？**
A: rank=0 表示在 N 只股票里排第 1（最强），rank=1 表示倒数第 1。业内"分位"传统就这样。

**Q: 为什么有时候报错 `stock_individual_info_em error: limited`？**
A: 东财个股信息接口对本机 IP 限流（见 `memory/project_eastmoney_blocked.md`）。已用 baidu / 同花顺 / 新浪 / 申万规避，不影响主流程。

**Q: forward_check 提示"数据未到"是 bug 吗？**
A: 不是。新浪 stock_zh_a_daily 接口通常当晚 18:00 后才推送当日收盘。盘后立即跑会拿不到 T+1 数据，明早 cron 自然就有了。

**Q: 新闻命中怎么算？没看到 NLP？**
A: MVP 阶段用**关键词词典**（utf-8 中文），不依赖 NLP。词典在 `factors/news/dim.py`：
- `BULLISH_KW` / `BEARISH_KW` — 利好/利空词
- `RISK_EVENT_KW` — 减持/解禁/立案/预亏/退市
- `SECTOR_ALIASES` — 板块别名（油运/航运、芯片/半导体...）

新闻流由 cron 每小时累积一次（cls + 同花顺 + 新浪），单日决策时拉近 3 天滑窗。

**Q: 信号每天都不一样吗？**
A: 大部分会变（K 线/资金流/RSI/板块动量/新闻都是日更）。基本面（PE/ROE）季报频率慢。

---

## 历史 / 下一步

完成进度参见 `memory/project_stock_engine.md`。

下一步候选（按优先级）：
1. **新闻逆向规则词典**：原油涨→油运空、锂价跌→锂矿空（等 forward 数据积累 1-2 周后基于真实坑迭代）
2. **HELD 真规则**：在 yaml 加 `target_prices` / `stop_loss`，让 STOP/TAKE/ADD 基于价格触发
3. **sector_cap**：BUY 列表同行业最多 N 只，避免油运三兄弟同时进 TOP
4. **推送**：BUY 信号自动 Telegram
5. **walk-forward 严格回测**：每天用当时数据重算 composite
