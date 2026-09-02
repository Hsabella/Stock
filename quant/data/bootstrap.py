"""qlib 数据包引导：下载/解压 chenditc qlib_bin 并做初始化冒烟测试。

数据源是社区维护的 chenditc/investment_data（qlib 官方 cn_data 已停更），
每日发布 qlib_bin.tar.gz（2005 至今日线，tushare+akshare 交叉校验）。

用法:
    python -m quant.data.bootstrap                # 缺包则下载，解压，冒烟
    python -m quant.data.bootstrap --force-download
    python -m quant.data.bootstrap --skip-extract  # 只跑冒烟测试
"""
from __future__ import annotations

import argparse
import shutil
import ssl
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from quant.config import load_config

_CHUNK = 1 << 20  # 1 MB
_MAX_ATTEMPT = 30  # 每轮至少推进一点就不算浪费；跨代理下载 560MB 常要十几轮
_READ_TIMEOUT = 90


def _ssl_context() -> ssl.SSLContext:
    """显式用 certifi 的根证书建 SSL 上下文。

    python.org 版 macOS Python 装完不跑 Install Certificates.command 的话，
    /Library/Frameworks/.../etc/openssl/ 是空的、ssl 默认信任库为空，
    urllib 走默认上下文必然 CERTIFICATE_VERIFY_FAILED（2026-08 踩坑：
    bootstrap 从装机起就 100% 挂，而 akshare 走 requests+certifi 所以没暴露）。
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _remote_size(url: str) -> int:
    """HEAD 拿 Content-Length（跟随重定向）；拿不到返回 0。"""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30, context=_ssl_context()) as resp:
            return int(resp.headers.get("Content-Length") or 0)
    except Exception:
        return 0


def _download_once(url: str, tmp: Path, total: int) -> int:
    """从 tmp 已有字节处断点续传一轮，返回本轮结束时的文件大小。

    异常不外抛：单轮失败（超时/连接断）属常态，由上层循环换新连接再续。
    """
    have = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")

    # 不用 urlretrieve：它不接受 context 参数，没法指定 certifi 根证书
    with urllib.request.urlopen(req, timeout=_READ_TIMEOUT, context=_ssl_context()) as resp:
        # 服务端不认 Range（200 而非 206）就只能从头写，否则会把响应接在残包后面写坏
        mode = "ab" if resp.status == 206 else "wb"
        if mode == "wb":
            have = 0
        next_report = have + _CHUNK * 100
        with open(tmp, mode) as fh:
            while chunk := resp.read(_CHUNK):
                fh.write(chunk)
                have += len(chunk)
                if have >= next_report:
                    print(f"  ... {have / 1e6:.0f}/{total / 1e6:.0f} MB", flush=True)
                    next_report += _CHUNK * 100
    return have


def download_archive(url: str, dest: Path, force: bool = False) -> Path:
    """下载 release 包（断点续传 + 多轮重试）；已存在且非 force 时直接复用。

    必须能续传：包 ~560MB，跨代理下载单条连接经常中途卡死。2026-08-09 周五维护
    就是死在这——原实现一条连接读到底、无重试无续传，一个 read timeout 就整包作废，
    bootstrap 非零退出后 extract 根本没执行，cn_data 悄悄停在 08-04 又过了一周。
    （注意 cron 里 bootstrap 走系统代理，不在 lib_no_proxy.sh 的绕过名单内。）
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"[bootstrap] 复用已有归档: {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest

    print(f"[bootstrap] 下载 {url}")
    tmp = dest.with_suffix(".part")
    total = _remote_size(url)
    if force:
        tmp.unlink(missing_ok=True)  # force 是"我要最新包"，残包可能是上个 release 的

    last_size = -1
    for attempt in range(1, _MAX_ATTEMPT + 1):
        try:
            size = _download_once(url, tmp, total)
            if not total:
                break  # 拿不到 Content-Length 时，一轮读到流结束就只能认为下完了
        except Exception as e:  # 超时/连接重置/代理抖动，换连接重来
            size = tmp.stat().st_size if tmp.exists() else 0
            print(f"[bootstrap] 第 {attempt} 轮中断({type(e).__name__})，已下 {size / 1e6:.0f} MB，续传", flush=True)
        if size >= total:
            break
        if size == last_size:  # 连续两轮零进展 = 对端不给了，别空转
            raise IOError(f"下载停滞在 {size}/{total} 字节（第 {attempt} 轮无进展）")
        last_size = size
    else:
        raise IOError(f"下载未完成: {last_size}/{total} 字节，已重试 {_MAX_ATTEMPT} 轮")

    if total and tmp.stat().st_size != total:
        raise IOError(f"下载不完整: {tmp.stat().st_size}/{total} 字节")
    tmp.replace(dest)
    print(f"[bootstrap] 下载完成: {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


def _find_data_root(extract_dir: Path) -> Path:
    """在解压目录里定位包含 calendars/day.txt 的数据根目录（容忍不同打包层级）。"""
    for cand in [extract_dir, *extract_dir.rglob("*")]:
        if cand.is_dir() and (cand / "calendars" / "day.txt").exists():
            return cand
    raise FileNotFoundError(f"解压结果中找不到 calendars/day.txt: {extract_dir}")


def extract_archive(archive: Path, provider_uri: Path) -> None:
    """解压归档并原子替换 provider_uri（旧数据先备份为 .bak 再删除）。"""
    print(f"[bootstrap] 解压 {archive} ...")
    with tempfile.TemporaryDirectory(dir=provider_uri.parent if provider_uri.parent.exists() else None) as td:
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(td)
        root = _find_data_root(Path(td))

        provider_uri.parent.mkdir(parents=True, exist_ok=True)
        bak = provider_uri.with_suffix(".bak")
        if provider_uri.exists():
            if bak.exists():
                shutil.rmtree(bak)
            provider_uri.rename(bak)
        shutil.move(str(root), str(provider_uri))
        if bak.exists():
            shutil.rmtree(bak)
    print(f"[bootstrap] 数据就位: {provider_uri}")


def smoke_test(provider_uri: Path) -> dict:
    """qlib.init + 基础可用性检查，返回摘要信息（供 checks/文档使用）。"""
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=str(provider_uri), region="cn")

    cal = D.calendar(freq="day")
    info = {
        "calendar_start": str(cal[0].date()),
        "calendar_end": str(cal[-1].date()),
        "n_trading_days": len(cal),
    }

    inst_dir = provider_uri / "instruments"
    info["instruments_files"] = sorted(p.stem for p in inst_dir.glob("*.txt"))

    # 任选一只老牌股票探测字段清单
    sample_dir = provider_uri / "features" / "sh600000"
    if sample_dir.exists():
        info["fields"] = sorted(p.name.split(".")[0] for p in sample_dir.glob("*.day.bin"))

    df = D.features(["SH600000"], ["$close", "$volume", "$factor"], start_time="2020-01-01", end_time="2020-01-31")
    if df.empty:
        raise RuntimeError("冒烟失败: SH600000 2020-01 无数据")
    info["sample_rows"] = len(df)

    print("[bootstrap] 冒烟通过:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-extract", action="store_true", help="数据已就位时只跑冒烟")
    args = parser.parse_args()

    cfg = load_config("data")
    provider_uri = Path(cfg["qlib"]["provider_uri"])

    if not args.skip_extract:
        archive = download_archive(
            cfg["download"]["release_url"], Path(cfg["download"]["archive_path"]), force=args.force_download
        )
        extract_archive(archive, provider_uri)

    smoke_test(provider_uri)
    return 0


if __name__ == "__main__":
    sys.exit(main())
