# 多维决策引擎使用说明

> A 股自选股 8 维决策引擎 — 收盘后扫描 + 多维 rank 加权 + 状态机决策
> 看不懂的术语？查 [GLOSSARY.md](./GLOSSARY.md)

---

## 一句话理解

每天收盘后，把你 watchlist 里的股票从 6 个维度（基本面 / 主力资金 / 流动性 / 筹码 / 大盘 / 技术）做横截面打分，按你的持仓状态（NONE/WATCHING/HELD）输出 **BUY / HOLD / REDUCE / STOP** 等动作建议，附带 3-5 条人话理由和风险提示。

**这不是自动交易系统**——它是给你做参考的"决策助手"。最终下不下单、下多少、什么时机由你拍板。

---

## 日常工作流（每天收盘后 5 分钟）

```bash
cd /Users/wangbo/VSCodeProjects/Stock

# 1. 跑 6 维 engine（约 60-90 秒，取决于网络）
python3 scripts/run_partial_engine.py

# 2. 生成易读 markdown 报告
python3 scripts/generate_report.py

# 3. 打开看
open results/decisions/partial_$(date +%Y%m%d).md
```

输出 3 份：
- `results/decisions/partial_YYYYMMDD.csv` — 全字段表格（给你自己分析用）
- `results/decisions/partial_YYYYMMDD.json` — 结构化数据（给程序消费）
- `results/decisions/partial_YYYYMMDD.md` — 人读报告（推荐每天看这个）

---

## 看报告怎么操作

### 🟢 BUY 区
**含义**：通过了基本面 Hard Veto + 三道 gates（fund/flow/liq rank）+ composite ≥ 0.55

**操作**：
1. 看 drivers — 至少 3 条买入理由是什么
2. 看 risks — 有没有 RSI 过热、PE 偏高等警告
3. **人工决策**：要买就把 `watchlist.yaml` 里这只票的 `state` 改为 `HELD`，加上 `position`（手数）和 `entry_price`（成交均价）

### 🟠 REDUCE / ⛔ STOP（HELD 票）
**含义**：你持仓的票引擎觉得该减/清仓
- 雪人集团示例：`fundamental_rank 94%` 意思是 33 只里基本面排第 31 → 实锤垃圾
- 处理：人工判断后减仓，仓位减完后改 yaml 把 `state` 设为 `EXITED`

### 🟡 CONTINUE_WATCH
**含义**：通过 gates 但 composite < 0.55，差一口气
**操作**：留在关注池，明天再看

### 🔴 DROP
**含义**：gates fail（比如 fund_rank > 60% 或 liq_rank > 70%）
**操作**：建议从 watchlist 移除，省得污染统计

### ⛔ AVOID
**含义**：基本面 Hard Veto 命中（连续 2 年亏损 / 商誉过高 / 资产负债率 > 80% / ST）
**操作**：永远别碰，从 watchlist 删掉

---

## watchlist.yaml 写法

```yaml
cooldown_days: 10                   # 卖出后多少天内不再 BUY 同一只
default_stop_strategy: "atr_2"      # 默认止损策略
default_max_position_pct: 0.10      # 单票最大仓位

watchlist:
  # 持仓格式
  - { symbol: "002716", name: "湖南白银", state: "HELD",
      position: 500, entry_price: 12.630 }

  # 关注格式
  - { symbol: "688001", name: "华兴源创", state: "WATCHING" }

  # 关注但有目标位（未来支持 STOP/TAKE 用）
  - { symbol: "300750", name: "宁德时代", state: "WATCHING",
      target_entry: [240, 250] }
```

**state 说明**：
| state | 含义 | engine 输出动作 |
|---|---|---|
| `NONE` | 普通候选股 | BUY / NO_ACTION |
| `WATCHING` | 关注但未持仓 | BUY / CONTINUE_WATCH / DROP |
| `HELD` | 持仓中 | HOLD / ADD / REDUCE / STOP / TAKE |
| `EXITED` | 刚卖出，cooldown 中 | NO_ACTION |

---

## 验证 / 回测 signal

跑 30 个交易日的样本外验证：

```bash
python3 scripts/backtest_composite_week.py
```

输出：
- 每只票每个 anchor 日的 forward T+1/T+3/T+5 收益
- 分位组对比（top 1/3 vs bot 1/3）
- 每日 rank-IC + 平均 IC

**当前基线**（截至 2026-05-10）：
- IC T+5 = +0.225（30 anchor 样本外）
- 28/30 天 T+5 IC > 0
- top-bot spread T+5 = +1.64 pp

---

## 项目结构

```
Stock/
├── factors/              # 6 个维度（每个 dim 一个子目录）
│   ├── _kline.py         # 共享日 K 线（新浪源, 含换手率/流通股本）
│   ├── fundamental/      # 基本面 + Hard Veto
│   ├── fund_flow/        # 主力资金（同花顺 + 北向 + 龙虎榜）
│   ├── liquidity/        # 量比/换手率拐点
│   ├── technical/        # RSI/MACD/KDJ/BOLL/OBV
│   ├── chips/            # 筹码集中度自算
│   └── regime/           # 大盘环境 + 个股相对沪深300
├── engine/
│   ├── ranker.py             # 横截面 rank/zscore
│   ├── decision_resolver.py  # gates + 决策规则
│   └── state_machine.py      # NONE/WATCHING/HELD/EXITED 转移
├── scripts/
│   ├── run_partial_engine.py      # 主入口（每天跑这个）
│   ├── backtest_composite_week.py # 回测验证
│   └── generate_report.py         # 生成 md 报告
├── watchlist.yaml        # 你的自选股池（手动维护）
├── state/positions.json  # cooldown 跟踪（自动写）
├── cache/                # 数据缓存（自动）
└── results/decisions/    # 每天的决策输出
```

---

## 常见问题

**Q: 为什么 rank 是 0% 表示最强（反直觉）？**
A: rank=0 表示在 N 只股票里排第 1（最强），rank=1 表示排倒数第 1。这是因为业内"分位"传统就这样，先忍着，未来可考虑反转显示。

**Q: 为什么有时候报错 "stock_individual_info_em error: limited"？**
A: 东方财富个股信息接口对单 IP 限流，已知问题。不影响主流程，sector 字段会退化为空，行业分位用全局分位代替。

**Q: 为什么有些股票 ETF 没决策？**
A: 当前引擎只支持 A 股个股（K 线 + 基本面接口）。ETF 用 fund_etf_hist_em 接口，未来再接。`512880 证券ETF` 在 yaml 里被注释掉了。

**Q: 为什么北向资金维度是个 5 日均值？**
A: 2024-08 起监管停止披露当日北向总额，只能用近 5 日平均代理。

**Q: 信号每天都不一样吗？**
A: 大部分会变，因为 K 线 / 主力流入 / RSI 都是日更。基本面（PE/ROE）季报频率慢，不会日变。

---

## 下一步可加的功能

按价值降序：
1. **HELD 真规则**：在 yaml 加 `target_prices` / `stop_loss`，让 STOP/TAKE/ADD 真正基于价格触发
2. **ETF 支持**：把 K 线接口换 fund_etf_hist_em，纳入证券 ETF
3. **市场动作维度** market_action_dim（影线/量价背离）— 凑齐 7/8 维
4. **推送**：BUY 信号自动发 Telegram/邮件
5. **walk-forward 回测**：每天用当时数据重算 composite，更严谨验证
