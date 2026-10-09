# 单步时限兜底（source 进来用）：run_limited <秒> <命令...>
# 超时 SIGTERM 终止命令，返回 124（同 GNU timeout）；否则透传命令退出码。
#
# 为什么不用 perl alarm / Homebrew timeout：
# - launchd 的 PATH 里没有 Homebrew 的 timeout
# - /usr/bin/perl 是 universal 二进制，在本机以 x86_64 (Rosetta) 起，exec 出的
#   .venv-quant python 继承 x86_64 → numpy(arm64) 导入失败。2026-10-08 实测踩中。
# 纯 bash 后台 watchdog 不改变子进程架构。
run_limited() {
  local limit="$1"; shift
  "$@" &
  local pid=$!
  # 只有 sleep 正常睡满（wait 返回 0）才算超时；收尾时 sleep 被杀，bash 会打一行
  # "Terminated: 15" 噪音，watchdog 本身没有有用输出，stderr 直接丢掉
  ( sleep "$limit" & wait $! && kill -TERM "$pid" && touch "/tmp/.run_limited.$pid" ) 2>/dev/null &
  local watchdog=$!
  # 非交互脚本里 & 起的后台命令会忽略 Ctrl+C（bash 规则），手动跑时中断脚本得顺手杀子进程，
  # 否则 python 成孤儿继续抓数。trap 是全局的，用完即还原
  trap 'kill -TERM "$pid" 2>/dev/null; pkill -P "$watchdog" sleep 2>/dev/null; exit 130' INT TERM
  local rc=0
  wait "$pid" || rc=$?
  trap - INT TERM
  # 收尾命令返回非零是常态（watchdog 被杀 / 没 sleep 可杀），调用方开了 set -e
  # （daily_run.sh）时会把整个脚本带走——10-08 那晚引擎跑完就退出、forward 没跑
  pkill -P "$watchdog" sleep 2>/dev/null || true
  wait "$watchdog" 2>/dev/null || true
  if [ -e "/tmp/.run_limited.$pid" ]; then
    rm -f "/tmp/.run_limited.$pid"
    return 124
  fi
  return $rc
}
