#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志工具模块
"""

import os
import sys
from loguru import logger

from config.settings import LOG_LEVEL, LOG_FILE, LOG_ROTATION


def setup_logger():
    """配置日志系统

    Returns:
        logger: 配置好的日志对象
    """
    # 创建日志目录
    log_dir = os.path.dirname(LOG_FILE)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 清除默认处理器
    logger.remove()

    # 添加控制台输出
    logger.add(sys.stderr, level=LOG_LEVEL)

    # 添加文件输出
    logger.add(
        LOG_FILE,
        rotation=LOG_ROTATION,
        retention="30 days",
        level=LOG_LEVEL,
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )

    logger.info("日志系统初始化完成")

    return logger


# 设置日志格式中文支持
logger = setup_logger()
