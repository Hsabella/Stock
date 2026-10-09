"""给 requests 补默认超时——akshare 底层调 requests 基本不传 timeout。

为什么不用 socket.setdefaulttimeout：requests 不传 timeout 时会显式给 urllib3
传 None，urllib3 随即 sock.settimeout(None)，把全局默认值覆盖成永久阻塞。
2026-09-24 实测 09-11 周维护的 em_fundflow 卡在 SSL read 整整 13 天，
进程占着锁，把 09-18 那次定时任务也挡掉了。

只在调用方没给 timeout 时注入；显式传了（含元组）一律尊重。
"""
from __future__ import annotations

import functools

import requests

CONNECT_TIMEOUT_SEC = 10
READ_TIMEOUT_SEC = 30


def install_requests_timeout(connect: float = CONNECT_TIMEOUT_SEC,
                             read: float = READ_TIMEOUT_SEC) -> None:
    """进程级补丁：requests.get/post/Session.* 未传 timeout 时用 (connect, read)。幂等。"""
    current = requests.Session.request
    original = getattr(current, "_without_default_timeout", current)

    @functools.wraps(original)
    def request(self, method, url, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = (connect, read)
        return original(self, method, url, **kwargs)

    request._without_default_timeout = original
    requests.Session.request = request
