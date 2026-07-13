#!/bin/bash
# quant 每周数据维护 · cron 入口（每周五 19:30）
# ① 东财资金流累积（仅存 120 天，必须每周攒） ①b 财报增量 ①c 指数日线 ①d 行业快照
# ② qlib 数据包更新 ③ 数据质检
# 失败不静默：任一步骤非零 → 写 results/brief/data_health_alert.txt（晨报每日拼入，
# 直到下次成功才清除）+ macOS 通知 + 整体 exit 非零。不用 set -e：单步失败不阻断后续。
# 日志: logs/weekly_data.log（只留最近 1000 行）
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv-quant/bin/python

mkdir -p logs results/brief
# 并发锁：上一轮挂死或手动补跑撞上 cron 时直接退出（macOS 无 flock，用 mkdir 原子性）
if ! mkdir logs/.weekly_data.lock 2>/dev/null; then
  echo "$(date '+%F %T') 上一轮仍在运行，本轮退出" >> logs/weekly_data.log
  exit 1
fi
trap 'rmdir logs/.weekly_data.lock' EXIT

FAILED=""
step() {
  local name="$1"; shift
  echo "--- $name ---"
  "$@" || FAILED="$FAILED $name"
}

{
  echo "===== $(date '+%F %T') ====="
  step "em_fundflow" $PY -m quant.data.akshare_fetcher --dataset em_fundflow --refresh
  step "financials"  $PY -m quant.data.akshare_fetcher --dataset financials --refresh
  step "index_daily" $PY -m quant.data.index_daily
  step "industry"    $PY -m quant.data.industry
  step "bootstrap"   $PY -m quant.data.bootstrap --force-download
  step "checks"      $PY -m quant.data.checks
  if [ -n "$FAILED" ]; then
    echo "===== 完成(失败:$FAILED) $(date '+%F %T') ====="
  else
    echo "===== 完成(全部成功) $(date '+%F %T') ====="
  fi
} >> logs/weekly_data.log 2>&1

tail -n 1000 logs/weekly_data.log > logs/weekly_data.log.tmp && mv logs/weekly_data.log.tmp logs/weekly_data.log

if [ -n "$FAILED" ]; then
  echo "$(date '+%F') 周五数据维护失败步骤:$FAILED（东财资金流断档不可补，尽快手动重跑）" > results/brief/data_health_alert.txt
  osascript -e "display notification \"失败:$FAILED — 见 logs/weekly_data.log\" with title \"⚠️ 周五数据维护\"" 2>/dev/null
  exit 1
fi
rm -f results/brief/data_health_alert.txt
