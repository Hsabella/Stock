#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
交易策略配置文件
"""

# 策略权重配置（总和应为1.0）
STRATEGY_WEIGHTS = {
    "macd_rsi_strategy": 0.35,
    "ma_crossover_strategy": 0.25,
    "volume_price_strategy": 0.15,
    "support_resistance_strategy": 0.15,
    "multi_factor_strategy": 0.10
}

# MACD + RSI 策略配置
MACD_RSI_STRATEGY = {
    "enabled": True,
    "name": "MACD与RSI超卖策略",
    "description": "MACD金叉与RSI低位共振策略",
    "params": {
        "macd_fast": 12,  # 调整为标准值以适应大多数市场
        "macd_slow": 26,
        "macd_signal": 9,
        "rsi_length": 14,  # 使用标准RSI周期
        "rsi_oversold": 35,  # 放宽超卖条件（增大阈值）
        "rsi_overbought": 70
    },
    "conditions": {
        "macd_crossover": True,  # MACD金叉
        "macd_zero_crossover": False,  # 不强制要求零线上穿
        "rsi_oversold": True,  # RSI处于超卖区
        "price_above_ma": False,  # 价格位于均线之上
        "rsi_turning_up": True  # 添加RSI转向向上的条件
    },
    "logic": "OR",  # 条件逻辑：修改为OR，只要满足任一条件
    "min_score": 50  # 降低最低分数要求（0-100）
}

# 均线交叉策略配置
MA_CROSSOVER_STRATEGY = {
    "enabled": True,
    "name": "均线交叉策略",
    "description": "短期均线上穿长期均线，结合成交量确认",
    "params": {
        "short_ma": 5,
        "mid_ma": 20,
        "long_ma": 60,
        "volume_ma": 10,
        "price_ma": 30
    },
    "conditions": {
        "short_cross_mid": True,  # 短期均线上穿中期均线
        "mid_cross_long": False,  # 中期均线上穿长期均线
        "price_above_mid": True,  # 价格位于中期均线之上
        "volume_expand": True,    # 成交量放大
        "trend_confirm": True     # 趋势确认（收盘价高于开盘价）
    },
    "logic": "CUSTOM",  # 自定义逻辑
    "custom_logic": "(short_cross_mid AND volume_expand) OR (mid_cross_long AND trend_confirm)",
    "min_score": 75
}

# 量价关系策略配置
VOLUME_PRICE_STRATEGY = {
    "enabled": True,
    "name": "量价背离策略",
    "description": "价格与指标背离识别策略",
    "params": {
        "lookback_days": 20,
        "volume_ratio": 1.5,
        "price_change_threshold": 0.02
    },
    "conditions": {
        "price_lower_low": True,  # 价格创新低
        "indicator_higher_low": True,  # 指标不创新低（背离）
        "volume_shrink": True,  # 成交量萎缩
        "price_reversal": True   # 价格反转迹象
    },
    "logic": "AND",
    "min_score": 85
}

# 支撑阻力策略配置
SUPPORT_RESISTANCE_STRATEGY = {
    "enabled": True,
    "name": "支撑位反弹策略",
    "description": "价格触及支撑位并反弹确认",
    "params": {
        "sr_lookback": 60,
        "bounce_threshold": 0.03,
        "volume_confirm": 1.2
    },
    "conditions": {
        "near_support": True,  # 接近支撑位
        "bounce_confirm": True,  # 反弹确认
        "higher_low": True,     # 形成更高低点
        "volume_increase": True  # 成交量增加
    },
    "logic": "AND",
    "min_score": 78
}

# 多因子综合策略配置
MULTI_FACTOR_STRATEGY = {
    "enabled": True,
    "name": "多指标综合策略",
    "description": "综合技术、量价、动量等多维度指标",
    "factors": {
        "technical": {
            "weight": 0.4,
            "indicators": ["macd", "rsi", "kdj", "boll"]
        },
        "momentum": {
            "weight": 0.3,
            "indicators": ["roc", "cci", "mfi"]
        },
        "volume": {
            "weight": 0.2,
            "indicators": ["obv", "vr", "vwap"]
        },
        "volatility": {
            "weight": 0.1,
            "indicators": ["atr", "std"]
        }
    },
    "threshold_score": 75,
    "require_factors": 3  # 至少需满足的因子数量
}

# 动态策略调整配置
DYNAMIC_ADJUSTMENT = {
    "enabled": True,
    "market_regime_based": True,  # 基于市场状态动态调整
    "auto_optimize": True,  # 自动优化参数
    "optimization_interval": 30,  # 优化间隔（天）
    "max_param_change": 0.2,  # 最大参数变化比例
    "market_regimes": {
        "bull": {  # 牛市配置
            "macd_rsi_strategy.rsi_oversold": 35,  # 牛市中提高RSI超卖阈值
            "ma_crossover_strategy.weight": 0.35,  # 牛市中增加均线策略权重
            "support_resistance_strategy.weight": 0.10  # 牛市中降低支撑阻力策略权重
        },
        "bear": {  # 熊市配置
            "macd_rsi_strategy.rsi_oversold": 25,  # 熊市中降低RSI超卖阈值
            "volume_price_strategy.weight": 0.25,  # 熊市中增加量价策略权重
            "multi_factor_strategy.threshold_score": 85  # 熊市中提高多因子策略分数要求
        },
        "sideways": {  # 震荡市配置
            "support_resistance_strategy.weight": 0.30,  # 震荡市中增加支撑阻力策略权重
            "ma_crossover_strategy.weight": 0.15,  # 震荡市中降低均线策略权重
        }
    }
}
