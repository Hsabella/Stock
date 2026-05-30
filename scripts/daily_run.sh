#!/usr/bin/env bash
# 每日决策引擎 (推荐 18:30 跑一次).
# 18:30 时同花顺新闻 / 东财资金流 / akshare 日线均已稳定, 避免 15:30 拉到不完整数据.
# 工作流: 拉新闻 → 8 维 composite → 输出报告 + 决策 JSON.
# 用 crontab 注册: 30 18 * * 1-5 /Users/wangbo/VSCodeProjects/Stock/scripts/daily_run.sh
set -euo pipefail

REPO="/Users/wangbo/VSCodeProjects/Stock"
PY="/Library/Frameworks/Python.framework/Versions/3.10/bin/python3"
LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/daily_${TS}.log"

cd "$REPO"
{
    echo "=== daily_run started: $(date) ==="
    "$PY" scripts/run_partial_engine.py
    echo ""
    "$PY" scripts/generate_report.py 2>&1 || echo "  (generate_report 失败 / 不存在, 跳过)"
    echo ""
    echo "--- forward 校验 (前一份决策 vs 今日) ---"
    # 找倒数第二份决策做对照. exit=2 = 数据未到, 不算失败
    PREV=$(ls -1 "$REPO"/results/decisions/partial_*.json 2>/dev/null | tail -2 | head -1 \
           | sed -E 's/.*partial_([0-9]{8})\.json/\1/')
    if [ -n "$PREV" ]; then
        "$PY" scripts/forward_check.py --decision "$PREV" --horizons 1 3 5 --refresh \
            || echo "  (forward exit=$? 通常为数据未到, 明天再跑)"
    else
        echo "  尚无历史决策, 跳过"
    fi
    echo "=== daily_run ended: $(date) ==="
} >"$LOG" 2>&1

# 保留最近 30 个日志, 其余删除
ls -1t "$LOG_DIR"/daily_*.log 2>/dev/null | tail -n +31 | xargs -I{} rm -f {} || true

echo "log: $LOG"
