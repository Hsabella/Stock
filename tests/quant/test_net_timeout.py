"""requests 默认超时补丁：akshare 调 requests 不传 timeout，半死连接会永久阻塞。

2026-09-24 实测：09-11 周维护的 em_fundflow 卡在 SSL read 13 天，
socket.setdefaulttimeout 无效（requests 显式传 timeout=None 覆盖了全局默认）。
"""
import socket
import threading
import time

import pytest
import requests

from quant.data import net


@pytest.fixture
def silent_server():
    """接受连接但永不回包的 TCP 服务，模拟半死连接。"""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    conns = []
    stop = threading.Event()

    def accept_loop():
        srv.settimeout(0.2)
        while not stop.is_set():
            try:
                conns.append(srv.accept()[0])
            except OSError:
                pass

    t = threading.Thread(target=accept_loop, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.getsockname()[1]}/"
    stop.set()
    t.join()
    for c in conns:
        c.close()
    srv.close()


@pytest.fixture(autouse=True)
def restore_session_request():
    original = requests.Session.request
    yield
    requests.Session.request = original


def test_setdefaulttimeout_alone_does_not_bound_requests(silent_server):
    """记录根因：全局 socket 超时对不传 timeout 的 requests 调用无效。"""
    old = socket.getdefaulttimeout()
    socket.setdefaulttimeout(1)
    try:
        result = {}

        def call():
            try:
                requests.get(silent_server)
                result["outcome"] = "returned"
            except Exception as e:  # noqa: BLE001
                result["outcome"] = type(e).__name__

        t = threading.Thread(target=call, daemon=True)
        t.start()
        t.join(3)
        assert t.is_alive(), f"预期仍阻塞，实际 {result}"
    finally:
        socket.setdefaulttimeout(old)


def test_install_bounds_calls_without_timeout(silent_server):
    net.install_requests_timeout(connect=1, read=1)
    start = time.monotonic()
    with pytest.raises(requests.exceptions.ReadTimeout):
        requests.get(silent_server)
    assert time.monotonic() - start < 5


def test_explicit_timeout_is_respected(silent_server):
    net.install_requests_timeout(connect=1, read=60)
    start = time.monotonic()
    with pytest.raises(requests.exceptions.ReadTimeout):
        requests.get(silent_server, timeout=1)
    assert time.monotonic() - start < 5


def test_reinstall_does_not_stack_wrappers():
    # 全量跑时别的用例可能已经调过 main() 装了补丁，基线要取未包装的原函数
    current = requests.Session.request
    original = getattr(current, "_without_default_timeout", current)
    net.install_requests_timeout()
    net.install_requests_timeout()
    assert requests.Session.request._without_default_timeout is original
