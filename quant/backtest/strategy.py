"""周度调仓策略：TopkDropoutStrategy 子类，仅在每周首个交易日交易。

qlib 的 TopkDropoutStrategy 每个交易日都会换 n_drop 只；周度调仓的标准做法
是子类里判断"当前步是否本周第一个交易日"，否则返回空决策（照常持仓）。
信号时点约定沿用 qlib：T 日交易用的是 T-1 日（上一交易日）收盘后的预测分。
"""
from __future__ import annotations

from qlib.backtest.decision import TradeDecisionWO
from qlib.contrib.strategy import TopkDropoutStrategy


def _iso_week(ts) -> tuple[int, int]:
    iso = ts.isocalendar()
    return int(iso[0]), int(iso[1])


class WeeklyTopkDropoutStrategy(TopkDropoutStrategy):
    """仅每周首个交易日执行 Topk-Dropout 换仓，其余交易日空决策。"""

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        start, _ = self.trade_calendar.get_step_time(trade_step)
        if trade_step > 0:
            prev_start, _ = self.trade_calendar.get_step_time(trade_step - 1)
            if _iso_week(prev_start) == _iso_week(start):
                return TradeDecisionWO([], self)  # 本周已调过仓
        return super().generate_trade_decision(execute_result)
