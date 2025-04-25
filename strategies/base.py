#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略基类 - 所有交易策略的抽象基类
"""

import abc
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
from loguru import logger


class StrategyBase(abc.ABC):
    """策略基类"""

    def __init__(self, name: str, description: str = "", params: Dict = None):
        """初始化策略

        Args:
            name: 策略名称
            description: 策略描述
            params: 策略参数
        """
        self.name = name
        self.description = description
        self.params = params or {}
        self.is_enabled = True
        self.weight = 1.0
        self.min_score = 0  # 最低分数阈值
        self._last_result = None

    def set_params(self, params: Dict):
        """设置策略参数

        Args:
            params: 策略参数字典
        """
        self.params.update(params)
        logger.debug(f"策略 '{self.name}' 参数已更新: {params}")

    def enable(self):
        """启用策略"""
        self.is_enabled = True
        logger.info(f"策略 '{self.name}' 已启用")

    def disable(self):
        """禁用策略"""
        self.is_enabled = False
        logger.info(f"策略 '{self.name}' 已禁用")

    def set_weight(self, weight: float):
        """设置策略权重

        Args:
            weight: 权重值，范围0-1
        """
        if 0 <= weight <= 1:
            self.weight = weight
            logger.debug(f"策略 '{self.name}' 权重已设置为: {weight}")
        else:
            logger.warning(f"无效的权重值: {weight}，权重范围应为0-1")

    def set_min_score(self, score: int):
        """设置最低分数阈值

        Args:
            score: 分数阈值，范围0-100
        """
        if 0 <= score <= 100:
            self.min_score = score
            logger.debug(f"策略 '{self.name}' 最低分数阈值已设置为: {score}")
        else:
            logger.warning(f"无效的分数值: {score}，分数范围应为0-100")

    @property
    def last_result(self):
        """获取最近一次策略执行结果"""
        return self._last_result

    @abc.abstractmethod
    def scan(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """扫描数据执行策略

        Args:
            data: 包含价格和指标数据的DataFrame
            **kwargs: 额外参数

        Returns:
            pd.DataFrame: 策略信号结果
        """
        pass

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成详细的策略信号

        Args:
            data: 包含价格和指标数据的DataFrame

        Returns:
            pd.DataFrame: 带有信号标记的DataFrame
        """
        # 默认实现，子类可重写
        return data

    def calculate_score(self, data: pd.DataFrame) -> pd.Series:
        """计算策略得分

        Args:
            data: 数据DataFrame

        Returns:
            pd.Series: 每个时间点的策略得分（0-100）
        """
        # 默认实现返回0分，子类应该重写此方法
        return pd.Series(0, index=data.index)

    def backtest(self, data: pd.DataFrame, **kwargs) -> Dict:
        """回测策略表现

        Args:
            data: 历史数据
            **kwargs: 回测参数

        Returns:
            Dict: 回测结果统计
        """
        # 默认简单回测实现，子类可重写
        signals = self.scan(data, **kwargs)

        # 计算基本统计信息
        if "signal" not in signals.columns or signals.empty:
            return {"error": "无有效信号"}

        # 提取买入信号
        buy_signals = signals[signals["signal"] > 0]

        # 计算信号后的收益率（假设持有1天）
        signals["next_return"] = signals["close"].pct_change().shift(-1)

        # 只关注买入信号的收益
        signal_returns = signals.loc[buy_signals.index, "next_return"]

        results = {
            "signal_count": len(buy_signals),
            "win_rate": (signal_returns > 0).mean() if not signal_returns.empty else 0,
            "avg_return": signal_returns.mean() if not signal_returns.empty else 0,
            "max_return": signal_returns.max() if not signal_returns.empty else 0,
            "min_return": signal_returns.min() if not signal_returns.empty else 0,
            "std_return": signal_returns.std() if not signal_returns.empty else 0,
        }

        # 记录结果
        self._last_result = results
        return results

    def __str__(self) -> str:
        """字符串表示"""
        return f"策略: {self.name} - {self.description} (启用: {self.is_enabled}, 权重: {self.weight})"

    def get_info(self) -> Dict:
        """获取策略详细信息

        Returns:
            Dict: 策略信息字典
        """
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.is_enabled,
            "weight": self.weight,
            "min_score": self.min_score,
            "params": self.params,
            "last_result": self._last_result
        }
