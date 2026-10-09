#!/bin/bash
# 晨间仓位建议 · cron 入口（工作日 09:00）
# 产出: results/brief/brief_<日期>.md + results/brief/latest.md（每天看这个）
# 结果推 macOS 通知——红绿灯的价值集中在一年里少数几个红灯日，必须主动触达；
# 生成失败时 daily_brief 不会覆盖 latest.md（保留最后一次有效建议），并推 ⚠️ 通知。
# 日志: logs/brief.log（只留最近 500 行）
set -u
cd "$(dirname "$0")/.." || exit 1
. scripts/lib_no_proxy.sh   # 国内行情源绕过系统代理（K线/指数缓存刷新走这里）
. scripts/lib_timeout.sh

mkdir -p logs results/brief

# 失败重试：launchd 常在 DarkWake（合盖电池下几秒的后台唤醒）里触发本任务，
# 此时网络/代理 DNS 还没起来 → NameResolutionError。09-28~10-08 连续 7 天
# 都是 09:00 DarkWake 6 秒内跑挂、09:05 真唤醒后已无人重跑。
# sleep 在机器再次睡眠时一起暂停，醒来后继续，所以间隔按"清醒时间"算。
MAX_TRIES=4
RETRY_GAP_SEC=300
# 单次时限兜底（正常 ~30s）：挂死的进程会让 launchd 之后每天都跳过本任务
RUN_LIMIT_SEC=900

for try in $(seq 1 $MAX_TRIES); do
  {
    echo "===== $(date '+%F %T') (第 $try/$MAX_TRIES 次) ====="
    run_limited "$RUN_LIMIT_SEC" .venv-quant/bin/python -m quant.live.daily_brief --out-dir results/brief
  } >> logs/brief.log 2>&1
  rc=$?
  [ $rc -eq 124 ] && echo "!!! 超过 $((RUN_LIMIT_SEC / 60)) 分钟被终止" >> logs/brief.log
  [ $rc -eq 0 ] && break
  [ "$try" -lt $MAX_TRIES ] && sleep $RETRY_GAP_SEC
done

tail -n 500 logs/brief.log > logs/brief.log.tmp && mv logs/brief.log.tmp logs/brief.log

if [ $rc -eq 0 ]; then
  headline=$(head -n 1 results/brief/latest.md)
  osascript -e "display notification \"${headline//\"/}\" with title \"晨间仓位红绿灯\"" 2>/dev/null
else
  osascript -e 'display notification "生成失败，latest.md 保留昨日建议；可手动重跑 daily_brief" with title "⚠️ 晨间仓位红绿灯"' 2>/dev/null
fi
exit $rc
