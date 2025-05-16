#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
交易策略配置文件
"""

# 策略权重配置（总和应为1.0）
STRATEGY_WEIGHTS = {
    "macd_rsi_strategy": 0.35,
    "ma_crossover_strategy": 0.20,
    "volume_price_strategy": 0.20,  # 提高量价策略权重
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
        "macd_zero_crossover": True,  # 零线上穿
        "rsi_oversold": True,  # RSI处于超卖区
        "price_above_ma": False,  # 价格位于均线之上
        "rsi_turning_up": True,  # 添加RSI转向向上的条件
        "kdj_golden_cross": True  # 添加KDJ金叉条件
    },
    "logic": "CUSTOM",  # 修改为自定义逻辑
    "custom_logic": "(macd_crossover AND rsi_oversold) OR (macd_zero_crossover AND rsi_turning_up) OR (kdj_golden_cross AND rsi_oversold)",
    "min_score": 60,  # 降低最低分数要求（0-100）
    "risk_filter": {
        "enable_trend_filter": True,  # 启用趋势过滤
        "downtrend_period": 10,       # 下跌趋势判断周期
        "downtrend_threshold": -5,    # 下跌趋势阈值（百分比）
        "volume_filter": True,        # 启用成交量异常过滤
        "volatility_filter": True,    # 启用波动率风险过滤
        "max_continuous_down_days": 4  # 最大连续下跌天数
    }
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
        "trend_confirm": True,    # 趋势确认（收盘价高于开盘价）
        "ma_alignment": True      # 均线多头排列
    },
    "logic": "CUSTOM",  # 自定义逻辑
    "custom_logic": "(short_cross_mid AND volume_expand) OR (mid_cross_long AND trend_confirm) OR (short_cross_mid AND ma_alignment)",
    "min_score": 70,  # 略微降低最低分数要求
    "risk_filter": {
        "enable_trend_filter": True,
        "max_continuous_down_days": 3
    }
}

# 量价关系策略配置
VOLUME_PRICE_STRATEGY = {
    "enabled": True,
    "name": "量价关系策略",
    "description": "基于量价关系和背离识别的策略",
    "params": {
        "lookback_days": 20,
        "volume_ratio": 1.5,
        "price_change_threshold": 0.02,
        "volume_shrink_threshold": 0.6,     # 成交量萎缩阈值
        "volume_expansion_threshold": 1.8,  # 成交量放大阈值
        "reversal_confirm_days": 2          # 反转确认天数
    },
    "conditions": {
        "price_lower_low": True,      # 价格创新低
        "indicator_higher_low": True,  # 指标不创新低（背离）
        "volume_shrink": True,        # 成交量萎缩
        "price_reversal": True,       # 价格反转迹象
        "volume_price_divergence": True,  # 量价背离
        "volume_shock_reversal": True,    # 成交量剧变后反转
        "low_volume_rebound": True        # 低量反弹
    },
    "pattern_weights": {
        "volume_climax": 25,           # 成交量高潮
        "shrinking_volume_drop": 20,    # 萎缩量下跌
        "rising_volume_rebound": 30,    # 放量反弹
        "breakout_volume_surge": 25     # 突破性放量
    },
    "logic": "CUSTOM",
    "custom_logic": "(price_lower_low AND indicator_higher_low) OR (volume_shrink AND price_reversal) OR (volume_price_divergence)",
    "min_score": 65  # 调低最低分数要求
}

# 支撑阻力策略配置
SUPPORT_RESISTANCE_STRATEGY = {
    "enabled": True,
    "name": "支撑位反弹策略",
    "description": "价格触及支撑位并反弹确认",
    "params": {
        "sr_lookback": 60,
        "bounce_threshold": 0.03,
        "volume_confirm": 1.2,
        "support_zone_width": 0.02,    # 支撑区域宽度
        "resistance_zone_width": 0.02  # 阻力区域宽度
    },
    "conditions": {
        "near_support": True,      # 接近支撑位
        "bounce_confirm": True,    # 反弹确认
        "higher_low": True,        # 形成更高低点
        "volume_increase": True,   # 成交量增加
        "candle_pattern": True,    # 蜡烛图形态确认
        "support_test_count": True  # 多次测试支撑位
    },
    "logic": "CUSTOM",
    "custom_logic": "(near_support AND bounce_confirm) OR (near_support AND higher_low AND volume_increase)",
    "min_score": 70
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
    "threshold_score": 70,  # 调低门槛
    "require_factors": 2     # 降低需要满足的因子数量
}

# 策略协同效应配置
STRATEGY_SYNERGY = {
    "enabled": True,
    "boost_threshold": 2,      # 至少需要多少个策略同时看好
    "boost_factor": 1.25,      # 信号强度提升因子
    # 主要策略
    "primary_strategies": ["macd_rsi_strategy", "volume_price_strategy"],
    "synergy_pairs": [
        {"strategies": ["macd_rsi_strategy",
                        "volume_price_strategy"], "factor": 1.3},
        {"strategies": ["support_resistance_strategy",
                        "ma_crossover_strategy"], "factor": 1.2}
    ],
    "conflict_penalty": 0.7    # 策略冲突时的惩罚因子
}

# 动态策略调整配置
DYNAMIC_ADJUSTMENT = {
    "enabled": True,
    "market_regime_based": True,      # 基于市场状态动态调整
    "auto_optimize": True,            # 自动优化参数
    "optimization_interval": 30,      # 优化间隔（天）
    "max_param_change": 0.2,          # 最大参数变化比例
    "performance_lookback": 90,       # 性能回溯期
    "stock_specific_adjustment": True,  # 根据股票特性调整
    "market_regimes": {
        "bull": {  # 牛市配置
            "strategy_weights": {
                "macd_rsi_strategy": 0.40,
                "ma_crossover_strategy": 0.30,
                "volume_price_strategy": 0.10,
                "support_resistance_strategy": 0.10,
                "multi_factor_strategy": 0.10
            },
            "params": {
                "macd_rsi_strategy.rsi_oversold": 40,    # 牛市中提高RSI超卖阈值
                "macd_rsi_strategy.min_score": 60,       # 牛市中降低MACD+RSI最低分数
                "macd_rsi_strategy.risk_filter.downtrend_threshold": -8,  # 放宽下跌过滤条件
                "ma_crossover_strategy.min_score": 65    # 牛市中降低均线策略门槛
            }
        },
        "bear": {  # 熊市配置
            "strategy_weights": {
                "macd_rsi_strategy": 0.25,
                "ma_crossover_strategy": 0.15,
                "volume_price_strategy": 0.30,
                "support_resistance_strategy": 0.20,
                "multi_factor_strategy": 0.10
            },
            "params": {
                "macd_rsi_strategy.rsi_oversold": 25,    # 熊市中降低RSI超卖阈值
                "macd_rsi_strategy.min_score": 75,       # 熊市中提高MACD+RSI最低分数
                "macd_rsi_strategy.risk_filter.downtrend_threshold": -3,  # 加强下跌过滤条件
                "volume_price_strategy.min_score": 60,   # 熊市中降低量价策略门槛
                "multi_factor_strategy.threshold_score": 80  # 熊市中提高多因子策略分数要求
            }
        },
        "sideways": {  # 震荡市配置
            "strategy_weights": {
                "macd_rsi_strategy": 0.25,
                "ma_crossover_strategy": 0.15,
                "volume_price_strategy": 0.20,
                "support_resistance_strategy": 0.30,
                "multi_factor_strategy": 0.10
            },
            "params": {
                "macd_rsi_strategy.rsi_oversold": 35,    # 震荡市中适中的RSI超卖阈值
                "macd_rsi_strategy.min_score": 65,       # 震荡市中适中的MACD+RSI最低分数
                "support_resistance_strategy.min_score": 60,  # 震荡市中降低支撑阻力策略门槛
                "macd_rsi_strategy.risk_filter.downtrend_threshold": -5  # 适中的下跌过滤条件
            }
        }
    },
    "stock_characteristics": {
        "high_volatility": {
            "volume_price_strategy.min_score": 70,
            "support_resistance_strategy.min_score": 75
        },
        "low_liquidity": {
            "volume_price_strategy.volume_ratio": 1.2,
            "ma_crossover_strategy.volume_expand": False
        },
        "large_cap": {
            "macd_rsi_strategy.rsi_oversold": 38,
            "ma_crossover_strategy.short_ma": 10
        },
        "small_cap": {
            "volume_price_strategy.volume_expansion_threshold": 2.5,
            "risk_filter.volatility_filter": True
        }
    }
}
