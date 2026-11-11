#!/bin/bash
# quant 每周数据维护 · cron 入口（每周五 19:30）
# ① 财报增量 ①b 指数日线 ①c 行业快照（东财资金流累积 2026-11-11 废弃，见 docs/quant/fundflow_ic_preview.md）
# ② qlib 数据包更新 ③ 数据质检
# 失败不静默：任一步骤非零 → 写 results/brief/data_health_alert.txt（晨报每日拼入，
# 直到下次成功才清除）+ macOS 通知 + 整体 exit 非零。不用 set -e：单步失败不阻断后续。
# 日志: logs/weekly_data.log（只留最近 1000 行）；终端里手动跑时同时打到屏幕
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv-quant/bin/python
. scripts/lib_timeout.sh
. scripts/lib_no_proxy.sh   # 国内行情源绕过系统代理（不绕过 GitHub, bootstrap 仍走代理）

mkdir -p logs results/brief
# 并发锁：手动补跑撞上定时任务时直接退出（macOS 无 flock，用 mkdir 原子性）。
# 锁里记 PID：持有者已死（被 kill -9 / 断电，EXIT trap 没机会跑）就回收，
# 否则一把死锁会让之后每周都静默跳过。
LOCK=logs/.weekly_data.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  holder=$(cat "$LOCK/pid" 2>/dev/null)
  if [ -n "$holder" ] && kill -0 "$holder" 2>/dev/null; then
    echo "$(date '+%F %T') 上一轮(PID $holder)仍在运行，本轮退出" >> logs/weekly_data.log
    osascript -e "display notification \"上一轮 PID $holder 仍在运行，本轮跳过\" with title \"⚠️ 周五数据维护\"" 2>/dev/null
    exit 1
  fi
  echo "$(date '+%F %T') 回收失效锁(持有者 PID ${holder:-未知} 已不在)" >> logs/weekly_data.log
  rm -rf "$LOCK" && mkdir "$LOCK" || exit 1
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT

# 单步时限（秒）。requests 层已有 30s 读超时（quant/data/net.py），这里是兜底：
# 09-11 那轮 em_fundflow 卡在无超时的 SSL read 13 天，占着锁挡掉了 09-18。
# 正常耗时：抓取类 ~1h（全被封时重试退避可到 5h+），bootstrap 下载 ~1h。
FETCH_LIMIT=$((3 * 3600))
QUICK_LIMIT=$((30 * 60))

FAILED=""
step() {
  local name="$1" limit="$2"; shift 2
  echo "--- $name (时限 $((limit / 60)) 分钟) ---"
  local rc=0
  run_limited "$limit" "$@" || rc=$?
  [ "$rc" -eq 124 ] && echo "!!! $name 超过 $((limit / 60)) 分钟被终止"
  [ "$rc" -ne 0 ] && FAILED="${FAILED} $name"
  return 0
}

# 手动在终端跑（stdout 是 tty）时边写日志边上屏；launchd 后台跑只写日志。
# 走管道后 {} 块在子 shell 里执行，FAILED 带不出来，所以块尾落到状态文件再读回。
# PYTHONUNBUFFERED：python 输出进管道默认整块缓冲，不关的话屏幕上几分钟才蹦一段。
LOG=logs/weekly_data.log
STATUS=logs/.weekly_data.failed
export PYTHONUNBUFFERED=1
log_sink() {
  if [ -t 1 ]; then tee -a "$LOG"; else cat >> "$LOG"; fi
}

rm -f "$STATUS"
{
  echo "===== $(date '+%F %T') ====="
  step "financials"  $FETCH_LIMIT $PY -m quant.data.akshare_fetcher --dataset financials --refresh
  step "index_daily" $QUICK_LIMIT $PY -m quant.data.index_daily
  step "industry"    $QUICK_LIMIT $PY -m quant.data.industry
  step "bootstrap"   $FETCH_LIMIT $PY -m quant.data.bootstrap --force-download
  # 必须紧跟 bootstrap：extract 会把整个 cn_data 目录替换掉，
  # csi_union.txt 是我们自己生成的、不在 release 包里，不重建就没了
  # （ext_features / daily_brief 都读它）
  step "universe"    $QUICK_LIMIT $PY -m quant.data.universe
  step "checks"      $QUICK_LIMIT $PY -m quant.data.checks
  if [ -n "${FAILED}" ]; then
    echo "===== 完成(失败:${FAILED}) $(date '+%F %T') ====="
  else
    echo "===== 完成(全部成功) $(date '+%F %T') ====="
  fi
  printf '%s' "${FAILED}" > "$STATUS"
} 2>&1 | log_sink
FAILED=$(cat "$STATUS" 2>/dev/null || echo " 未知(状态文件缺失)")
rm -f "$STATUS"

tail -n 1000 logs/weekly_data.log > logs/weekly_data.log.tmp && mv logs/weekly_data.log.tmp logs/weekly_data.log

if [ -n "${FAILED}" ]; then
  echo "$(date '+%F') 周五数据维护失败步骤:${FAILED}（尽快手动重跑，见 logs/weekly_data.log）" > results/brief/data_health_alert.txt
  osascript -e "display notification \"失败:${FAILED} — 见 logs/weekly_data.log\" with title \"⚠️ 周五数据维护\"" 2>/dev/null
  exit 1
fi
rm -f results/brief/data_health_alert.txt
