# 实施计划：止损线锚点修正（v2，已实施）

日期：2026-07-28 ｜ 状态：**已实施**（评审修订后落地；与初稿的差异见设计文档 §7 修订记录）
设计文档：[../specs/2026-07-28-stop-line-anchor-fix-design.md](../specs/2026-07-28-stop-line-anchor-fix-design.md)

## 评审结论摘要（为何偏离初稿）

初稿评审发现 4 个实质问题，v2 全部吸收：

1. **`stop_ratchet.csv` 持久化被否决**：qfq 前复权在除权日整体重排历史价，存盘 stop
   跨基准失效；棘轮可从 K 线确定性重算（回测本就这么做），无需状态文件。
2. **左臂 ATR 锚点 off-by-one**：回测用信号日（入场前一交易日）`atr[t]`，初稿伪代码取
   入场当日；并补 ATR NaN 降级保护。
3. **"复测一轮"验证不了新映射**：右臂+棘轮组合回测本就在跑，重跑只是回归检查；
   回补接回入场分布的证据只能靠实盘 log 攒。
4. **并行观察期无观察对象**（当前 0 HELD）：改为真实 K 线历史回放验收（见附录）。

## Step 1 — schema 扩展 ✅

**`watchlist.yaml`**：HELD 记法新增 `entry_date`（首次建仓日，加仓不改）、
`entry_arm`（`left`=抄底灯建仓 / `right`=回补接回等右侧入场，缺省 `right`），
顶部注释块已补记法说明与示例。当前无 HELD 存量，无需回填。
**下次建仓时人工填写这两个字段**（漏填 entry_date → 晨报降级标 ⚠️，不会静默错算）。

## Step 2 — 止损计算重写 ✅（`quant/live/portfolio.py`）

- `trail_line()` **原义不动**（近20日最高收盘 − k×ATR14），仅改名义：注释明确为
  "趋势读数"；`reentry_watch` / `etf_watch` 两个调用点不受影响。
- 新增 `since_entry_ratchet(kline, entry_date, atr_k)`：对入场以来每日 u 算
  `cummax(close) − k×ATR14(u)`，取历史最大——与回测持仓循环
  `stop = max(stop, hi − k×atr)` 逐日等价；**每日全量重算，零持久化**。
  窗口不完整（K 线起点晚于入场 / 入场后无收盘 bar / ATR 有缺）→ 返回 None。
- 新增 `_signal_day_atr()`：左臂初始止损用**信号日** ATR14，对齐回测 `atr[t]`。
- `stop_lines()` 重写：`stop = max(初始止损, 棘轮)`；
  初始止损按臂选（right = entry×0.92；left = entry − 3×ATR14(信号日)，
  常量 `LEFT_INIT_ATR_K = 3.0` 注释指向 `stock_lights.py` LEFT_STOP_ATR 定案）。
  四条降级路径（缺 entry_date / K线不足入场窗 / 入场前ATR不可算 / 左臂窗口残缺）
  全部退回仅硬止损 + `degraded` 标记。
  附带左臂标注体检：入场时距 120 日高 > −25%（`ARM_SUSPECT_DD`）→ `arm_suspect`。
- 返回值新增 `init_stop / ratchet / trend_line / above_trend / degraded /
  degrade_reason / arm_suspect / dd_at_entry`；`hard_stop / stop_line / broken /
  dist_pct` 语义不变（`structure_drift` 等下游零改动）。

## Step 3 — 展示层拆分 ✅（`quant/live/daily_brief.py`）

持仓健康度行拆分两个概念（永久形态，非过渡期双印）：

```
  东材科技 601208 43.32/41.47 +4.5% | 止损线 38.15（距离 +13.6%）| ATR趋势线 61.10 下方🔴
```

- **止损线** = 新 `stop_line`，唯一卖出驱动，破线才印 ❗
- **ATR趋势线** = `trail_line()` 原值（与【回补观察】【ETF观察】命名统一），仅趋势读数
- 降级：`止损线 38.15（距离 +x%）⚠️(缺entry_date, 仅硬止损)`
- 左臂存疑：追加 `⚠️左臂标注存疑(入场时距120日高 -12%)`

## Step 4 — 测试 ✅（`tests/quant/`，全部合成 K 线，不依赖真实缓存）

1. `test_stop_lines_bottom_entry_not_broken_on_day_one` — 东材场景形状：坑底新仓
   首日不破（旧 20 日高锚 76 vs 新 57），锁死本次 bug。
2. `test_stop_lines_ratchet_locks_profit` — 588000 场景形状：涨到 130 回撤，
   棘轮保持峰值日线 118 > 成本，破线锁利。
3. `test_stop_lines_ratchet_monotonic_as_days_pass` — 逐日截断重算，stop 单调不降；
   末段 ATR 放大使当日线下移，棘轮不得跟随。
4. `test_stop_lines_left_arm_wide_init` — 左臂初始 = entry − 3×ATR14(信号日)，宽于硬止损。
5. `test_stop_lines_degraded_paths` — 三条降级路径逐一断言 degraded + 退回硬止损。
6. `test_stop_lines_left_arm_suspect_flag` — 左臂体检三态（存疑/正常/右臂不适用）。
7. `test_format_holdings_shows_stops_and_engine_signal` — 展示层：双线拆分 + 降级 ⚠️ 文案。

`.venv-quant/bin/python -m pytest tests/quant/ -q` → **90 passed**（2026-07-28）。

## Step 5 — 验收：真实 K 线历史回放 ✅（替代并行观察）

回放脚本对两笔实盘逐日重算新旧口径（旧口径 07-24 读数 60.916 与晨报实盘 60.92
对上，回放可信）：

**东材科技 601208**（entry 41.465 @ 07-23，右臂）：

| 日期 | 收盘 | 旧止损 | 旧判 | 新止损 | 新判 | 新距离 |
|---|---|---|---|---|---|---|
| 07-23 | 41.48 | 63.23 | ❗ | 38.15 | 不破 | +8.7% |
| 07-24 | 41.72 | 60.92 | ❗ | 38.15 | 不破 | +9.4% |
| 07-27 | 43.32 | 61.10 | ❗ | 38.15 | 不破 | +13.6% |

旧口径全程误判已破（bug 复现）；新口径全程不破，该仓位不会被赶出。
棘轮实测 33.63 < 硬止损，未生效（符合设计 §4.3）。

**科创50ETF 588000**（entry 1.835 @ ~06-16，右臂；建仓日为估算，峰值在 6 月底，
更早的真实建仓日不改变棘轮路径）：

- 新棘轮随涨势抬升 1.69 → **2.153**（06-30 峰值日）后保持，**07-02 起稳定判破**
  ——比实际 07-22 卖出（1.968，+7.2%）早 14 个交易日，若按新线执行约在 2.10
  离场（+14.5%），锁利更多。
- 旧口径同期线值从 2.153 **回落**到 2.047，且 07-07~07-14 一度"解破"又"复破"
  （非单调线做卖出驱动的决策噪声）；新棘轮只升不降，无此问题。

## 回滚

改动集中在 `portfolio.py`（`stop_lines`/`since_entry_ratchet`/`_signal_day_atr`）与
`daily_brief.py` 展示串，revert 两处即可；**无状态文件需清理**（棘轮零持久化）。
`watchlist.yaml` 新字段留存无害（旧代码不读）。

## 不在本次范围

- 【回补观察】条件① 对 588000 的「科创50 在 MA200 之上」判定失真，建议改「站回
  MA50」——**另开一次改动**。
- 核心宽基 ETF（`account.etf`）维持不设个仓止损。
- entry_price 名义价 vs qfq 基准的除权错位（前置已存在，仅影响入场锚定线，
  棘轮免疫；见设计 §5.5）。
- 回测 `MAX_HOLD=60`（60 交易日强制评估）的实盘化——entry_date 已解锁数据条件，
  未实装提醒，另行立项。
