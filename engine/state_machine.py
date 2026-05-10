"""持仓状态机 + cooldown 跟踪.

状态来源:
1. watchlist.yaml 显式 state: NONE / WATCHING / HELD
2. state/positions.json 跟踪 EXITED + cooldown_until

简化: 仅做 cooldown 检查，不自动转移状态（HELD→EXITED 由用户手动改 yaml）。
"""
from __future__ import annotations
import json
from datetime import date, timedelta
from pathlib import Path

VALID_STATES = ("NONE", "WATCHING", "HELD", "EXITED")
DEFAULT_COOLDOWN_DAYS = 10


def load_positions(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_positions(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def effective_state(symbol: str, declared_state: str, positions: dict, today: date | None = None) -> str:
    """返回有效状态: 若 positions 里有未到期 cooldown，强制 EXITED."""
    today = today or date.today()
    declared = (declared_state or "NONE").upper()
    if declared not in VALID_STATES:
        declared = "NONE"
    pos = positions.get(symbol, {})
    cd = pos.get("cooldown_until")
    if cd:
        try:
            cd_date = date.fromisoformat(cd)
            if cd_date >= today:
                return "EXITED"
        except ValueError:
            pass
    return declared


def mark_exited(positions: dict, symbol: str, exit_date: date | None = None,
                cooldown_days: int = DEFAULT_COOLDOWN_DAYS) -> dict:
    """记录止损/止盈后进入 cooldown."""
    exit_date = exit_date or date.today()
    until = exit_date + timedelta(days=cooldown_days)
    positions[symbol] = {**positions.get(symbol, {}),
                         "exited_on": exit_date.isoformat(),
                         "cooldown_until": until.isoformat()}
    return positions
