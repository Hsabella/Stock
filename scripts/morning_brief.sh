#!/bin/bash
# 晨间仓位建议 · cron 入口（工作日 09:00）
# 产出: results/brief/brief_<日期>.md + results/brief/latest.md（每天看这个）
# 结果推 macOS 通知——红绿灯的价值集中在一年里少数几个红灯日，必须主动触达；
# 生成失败时 daily_brief 不会覆盖 latest.md（保留最后一次有效建议），并推 ⚠️ 通知。
# 日志: logs/brief.log（只留最近 500 行）
set -u
cd "$(dirname "$0")/.." || exit 1

mkdir -p logs results/brief
{
  echo "===== $(date '+%F %T') ====="
  .venv-quant/bin/python -m quant.live.daily_brief --out-dir results/brief
} >> logs/brief.log 2>&1
rc=$?

tail -n 500 logs/brief.log > logs/brief.log.tmp && mv logs/brief.log.tmp logs/brief.log

if [ $rc -eq 0 ]; then
  headline=$(head -n 1 results/brief/latest.md)
  osascript -e "display notification \"${headline//\"/}\" with title \"晨间仓位红绿灯\"" 2>/dev/null
else
  osascript -e 'display notification "生成失败，latest.md 保留昨日建议；可手动重跑 daily_brief" with title "⚠️ 晨间仓位红绿灯"' 2>/dev/null
fi
exit $rc
