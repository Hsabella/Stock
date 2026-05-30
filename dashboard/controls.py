# dashboard/controls.py
"""所有写操作与子进程, 集中审计。写前备份, 严格校验, 零字符串拼接。"""
import os
import re
import shutil
import subprocess
import time

from ruamel.yaml import YAML

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WATCHLIST = os.path.join(REPO, "watchlist.yaml")
VALID_STATES = {"NONE", "WATCHING", "HELD", "EXITED"}
_yaml = YAML()
_yaml.preserve_quotes = True


def load_watchlist_rows(path=WATCHLIST):
    with open(path, encoding="utf-8") as f:
        data = _yaml.load(f)
    return [dict(r) for r in (data.get("watchlist") or [])]


def _validate(rows):
    for r in rows:
        sym = str(r.get("symbol", ""))
        if not re.fullmatch(r"\d{6}", sym):
            raise ValueError(f"symbol 必须 6 位数字: {sym!r}")
        state = r.get("state", "WATCHING")
        if state not in VALID_STATES:
            raise ValueError(f"非法 state: {state!r} (允许 {VALID_STATES})")
        if state == "HELD" and (r.get("position") in (None, "") or r.get("entry_price") in (None, "")):
            raise ValueError(f"HELD 必须有 position+entry_price: {sym}")


def save_watchlist(rows, path=WATCHLIST):
    """校验→备份→用 ruamel 写回 watchlist 段(保留顶层键与注释)。"""
    _validate(rows)
    if os.path.exists(path):
        ts = time.strftime("%Y%m%d_%H%M%S")
        shutil.copy(path, f"{path}.bak.{ts}")
    with open(path, encoding="utf-8") as f:
        data = _yaml.load(f) or {}
    data["watchlist"] = rows
    with open(path, "w", encoding="utf-8") as f:
        _yaml.dump(data, f)
    return load_watchlist_rows(path)  # 回读供 diff


def run_daily():
    """非阻塞触发 scripts/daily_run.sh; 返回 Popen。固定参数, 无字符串拼接。"""
    return subprocess.Popen(["bash", os.path.join(REPO, "scripts", "daily_run.sh")],
                            cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def latest_log_tail(n=40):
    logs = sorted([f for f in os.listdir(os.path.join(REPO, "logs")) if f.startswith("daily_")])
    if not logs:
        return "(无日志)"
    p = os.path.join(REPO, "logs", logs[-1])
    return "".join(open(p, encoding="utf-8", errors="ignore").readlines()[-n:])
