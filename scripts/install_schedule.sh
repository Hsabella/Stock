#!/usr/bin/env bash
# 把三个定时任务从 crontab 迁到 launchd（幂等，可重复跑）。
#
# 为什么换：macOS 的 cron 在机器睡眠时不触发、醒来也不补跑，任务就这么静默丢了。
# 实测 2026 年 8-9 月 daily_run 漏了 08-20 / 08-27 / 09-01 三晚，logs/ 里连日志都没有，
# 直接后果是 cache/kline 停在上一交易日、晨报拿隔日收盘当"现价"。
# launchd 的 StartCalendarInterval 会在机器唤醒后补跑错过的那次，正好治这个病。
#
# 用法: ./scripts/install_schedule.sh          安装/更新
#       ./scripts/install_schedule.sh --uninstall  卸下（不自动恢复 crontab，见文末提示）
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
AGENTS="$HOME/Library/LaunchAgents"
PLIST_SRC="$REPO/scripts/launchd"
LABELS=(com.wangbo.stock.daily-run com.wangbo.stock.morning-brief com.wangbo.stock.weekly-data)

uninstall() {
    for label in "${LABELS[@]}"; do
        launchctl bootout "gui/$UID/$label" 2>/dev/null || true
        rm -f "$AGENTS/$label.plist"
        echo "  卸下 $label"
    done
}

if [ "${1:-}" = "--uninstall" ]; then
    uninstall
    echo "已卸下。如需回到 cron，手动 crontab -e 加回原来三行（见 git log 本文件）。"
    exit 0
fi

mkdir -p "$AGENTS" "$REPO/logs"
for label in "${LABELS[@]}"; do
    # 用 REPO 实际路径渲染模板，避免 plist 里写死别人机器上的路径
    sed -e "s|@REPO@|$REPO|g" -e "s|@LABEL@|$label|g" "$PLIST_SRC/$label.plist" \
        > "$AGENTS/$label.plist"
    plutil -lint "$AGENTS/$label.plist" >/dev/null   # 语法错的 plist 会被 launchd 静默忽略
    launchctl bootout "gui/$UID/$label" 2>/dev/null || true
    launchctl bootstrap "gui/$UID" "$AGENTS/$label.plist"
    echo "  装上 $label"
done

# 从 crontab 摘掉同名任务，避免 cron 和 launchd 同时跑（幂等：没有就什么也不做）
if crontab -l 2>/dev/null | grep -qE 'daily_run\.sh|morning_brief\.sh|weekly_data\.sh'; then
    crontab -l 2>/dev/null > "$REPO/logs/crontab.backup.$(date +%Y%m%d_%H%M%S)"
    crontab -l 2>/dev/null \
        | grep -vE 'daily_run\.sh|morning_brief\.sh|weekly_data\.sh' \
        | grep -v '^# quant 新系统: 每周数据维护' \
        | crontab -
    echo "  已从 crontab 摘掉这三个任务（原表备份在 logs/crontab.backup.*）"
fi

echo
echo "当前已注册:"
launchctl list | grep -E 'com\.wangbo\.stock' || echo "  (无)"
echo "查看下次触发: launchctl print gui/$UID/com.wangbo.stock.morning-brief | grep -A3 'next fire'"
