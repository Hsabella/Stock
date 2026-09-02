"""bootstrap.download_archive 断点续传/重试逻辑单测（不依赖网络）。

回归背景：2026-08-09 周五维护 bootstrap 死于一个 read timeout，原实现无重试无续传
→ 整包作废 → extract 没执行 → cn_data 悄悄停在 08-04 一周没人发现。
这里锁住四条不变量：中断能接着下、下满即停、零进展不空转、force 不接旧残包。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant.data import bootstrap  # noqa: E402

TOTAL = 1000
PARTIAL_ON_ERROR = 100  # 中断那轮在断开前已落盘的字节数


@pytest.fixture
def fake_download(monkeypatch):
    """把网络换掉：每轮按剧本写 N 字节或抛异常，返回每轮开工时的已有字节数列表。

    剧本用完还来一轮 = 被测代码多发了请求，用 AssertionError 立刻炸出来。
    """
    monkeypatch.setattr(bootstrap, "_remote_size", lambda url: TOTAL)
    offsets: list[int] = []

    def install(script: list):
        """script 每项是 int（成功写 n 字节）或 (n, 异常)（写 n 字节后中断）。"""
        rounds = iter(script)

        def fake_once(url, tmp, total):
            have = tmp.stat().st_size if tmp.exists() else 0
            offsets.append(have)
            step = next(rounds, "OVERRUN")
            assert step != "OVERRUN", f"多发了一轮下载请求（已有 {have} 字节）"
            written, exc = step if isinstance(step, tuple) else (step, None)
            if written:
                with open(tmp, "ab") as fh:
                    fh.write(b"x" * written)
            if exc:
                raise exc
            return have + written

        monkeypatch.setattr(bootstrap, "_download_once", fake_once)
        return offsets

    return install


def test_resumes_from_partial_after_timeout(fake_download, tmp_path):
    """中断后下一轮必须从已有字节续传，而不是从 0 重来。"""
    offsets = fake_download([
        (PARTIAL_ON_ERROR, TimeoutError("read timed out")),
        TOTAL - PARTIAL_ON_ERROR,
    ])
    dest = tmp_path / "qlib_bin.tar.gz"

    bootstrap.download_archive("http://x/qlib_bin.tar.gz", dest)

    assert offsets == [0, PARTIAL_ON_ERROR]        # 第二轮接着第一轮落下的字节下
    assert dest.stat().st_size == TOTAL
    assert not dest.with_suffix(".part").exists()  # .part 已改名，不留残包


def test_stops_when_complete(fake_download, tmp_path):
    """下满即停，不多发一轮（多发会拿到 416）。"""
    offsets = fake_download([TOTAL])
    bootstrap.download_archive("http://x/qlib_bin.tar.gz", tmp_path / "a.tar.gz")
    assert offsets == [0]


def test_raises_on_zero_progress(fake_download, tmp_path):
    """连续两轮一个字节都没进 = 对端不给了，报错退出而不是空转 30 轮。"""
    dead = [(0, ConnectionResetError("reset")) for _ in range(bootstrap._MAX_ATTEMPT)]
    offsets = fake_download(dead)
    dest = tmp_path / "a.tar.gz"

    with pytest.raises(IOError, match="停滞"):
        bootstrap.download_archive("http://x/qlib_bin.tar.gz", dest)

    assert len(offsets) == 2                       # 第 2 轮就认输
    assert not dest.exists()                       # 残包绝不能改名成正式包


def test_keeps_retrying_while_making_progress(fake_download, tmp_path):
    """每轮只挪一点也要接着下——真实代理经常这样，不能因"慢"就判死。"""
    slow = [(TOTAL // 5, TimeoutError("stall")) for _ in range(4)] + [TOTAL // 5]
    offsets = fake_download(slow)
    dest = tmp_path / "a.tar.gz"

    bootstrap.download_archive("http://x/qlib_bin.tar.gz", dest)

    assert offsets == [0, 200, 400, 600, 800]
    assert dest.stat().st_size == TOTAL


def test_reuses_existing_archive_without_force(fake_download, tmp_path):
    """已有整包且非 force 时直接复用，一轮网络都不发。"""
    dest = tmp_path / "a.tar.gz"
    dest.write_bytes(b"y" * TOTAL)
    offsets = fake_download([])

    bootstrap.download_archive("http://x/qlib_bin.tar.gz", dest)

    assert offsets == []


def test_force_discards_stale_part(fake_download, tmp_path):
    """--force-download 是"我要最新包"，残包可能属于上个 release，必须丢弃重下。"""
    dest = tmp_path / "a.tar.gz"
    dest.with_suffix(".part").write_bytes(b"z" * 300)   # 上个 release 的残包
    offsets = fake_download([TOTAL])

    bootstrap.download_archive("http://x/qlib_bin.tar.gz", dest, force=True)

    assert offsets == [0]                    # 从 0 开始，没接在旧残包后面
    assert dest.stat().st_size == TOTAL
