"""daily_brief 落盘行为：失败不覆盖 latest.md、信号流水去重、健康警报拼入。"""
import sys

import pandas as pd
import pytest

from quant.live import daily_brief

FAKE = {
    "date": "2026-07-13", "state": "risk_on", "exposure": 1.0,
    "hs300_close": 4781.0, "hs300_vs_ma200": 0.02, "ma200": 4687.0,
    "vs_ma200_prev5": 0.04, "hs300_5d_chg": -0.019,
    "spx_overnight": 0.0042, "spx_threshold": -0.015,
    "confirm_days": 2, "exposure_off": 0.7,
    "raw_today": False, "raw_streak": 3,
}


def _run(monkeypatch, tmp_path, brief_impl):
    monkeypatch.setattr(daily_brief, "brief", brief_impl)
    monkeypatch.setattr(sys, "argv", ["daily_brief", "--out-dir", str(tmp_path)])
    return daily_brief.main()


def test_failure_keeps_last_good_latest(monkeypatch, tmp_path):
    (tmp_path / "latest.md").write_text("昨日有效建议\n")

    def boom():
        raise RuntimeError("数据源挂了")

    assert _run(monkeypatch, tmp_path, boom) == 1
    assert (tmp_path / "latest.md").read_text() == "昨日有效建议\n"
    assert list(tmp_path.glob("brief_*_failed.md"))


def test_success_writes_files_and_signal_log_dedupes(monkeypatch, tmp_path):
    assert _run(monkeypatch, tmp_path, lambda: dict(FAKE)) == 0
    assert "🟢" in (tmp_path / "latest.md").read_text()
    assert (tmp_path / "brief_20260713.md").exists()

    assert _run(monkeypatch, tmp_path, lambda: dict(FAKE)) == 0  # 同日重跑
    log = (tmp_path / "signal_log.csv").read_text().strip().splitlines()
    assert log[0] == "date,state,raw_today,exposure,hs300_close"
    assert len(log) == 2  # 表头 + 唯一一行


def test_format_green_shows_distance_and_rules():
    text = daily_brief.format_brief(dict(FAKE))
    assert "距离防守线" in text and "再跌 2.0%" in text   # 4687/4781-1 ≈ -2.0%
    assert "缓冲收窄中" in text                            # +4.0% → +2.0%
    assert "单日涨跌不触发" in text


def test_format_pending_red_shows_confirm_progress():
    b = dict(FAKE, raw_today=True, raw_streak=1)
    text = daily_brief.format_brief(b)
    assert "防守信号已触发（第 1 天）" in text
    assert "再连续确认 1 天即转 🔴" in text


def test_format_red_state_shows_recovery_line():
    b = dict(FAKE, state="risk_off", exposure=0.7, hs300_vs_ma200=-0.03,
             hs300_close=4550.0, raw_today=True)
    text = daily_brief.format_brief(b)
    assert "🔴 防守" in text and "70%" in text
    assert "距离恢复线" in text and "收复 MA200" in text


def test_health_alert_appended(monkeypatch, tmp_path):
    (tmp_path / "data_health_alert.txt").write_text("2026-07-11 周五数据维护失败步骤: em_fundflow\n")
    assert _run(monkeypatch, tmp_path, lambda: dict(FAKE)) == 0
    assert "em_fundflow" in (tmp_path / "latest.md").read_text()


FAKE_LIGHTS = [
    {"name": "中证1000", "symbol": "sh000852", "state": "risk_on", "close": 7123.0,
     "vs_ma": 0.051, "ma_days": 200, "confirm_days": 3, "mode": "display_only",
     "advice_off": "", "raw_today": False, "raw_streak": 5},
    {"name": "科创50", "symbol": "sh000688", "state": "risk_off", "close": 1035.0,
     "vs_ma": -0.032, "ma_days": 200, "confirm_days": 3, "mode": "advice",
     "advice_off": "科创/题材持仓暂停加仓，新增仓位减半", "raw_today": True, "raw_streak": 4},
]


def test_format_lights_states_and_advice():
    text = daily_brief.format_brief(dict(FAKE, lights=[dict(x) for x in FAKE_LIGHTS]))
    assert "【结构灯】" in text
    assert "中证1000 🟢" in text and "仅展示" in text
    assert "科创50 🔴" in text and "暂停加仓" in text


def test_format_lights_flip_progress():
    lt = dict(FAKE_LIGHTS[0], raw_today=True, raw_streak=1)  # 绿灯但翻转信号第1天
    text = daily_brief.format_brief(dict(FAKE, lights=[lt]))
    assert "翻转信号第 1 天" in text and "再 2 天确认转 🔴" in text


def test_lights_error_degrades_without_killing_main_light():
    text = daily_brief.format_brief(dict(FAKE, lights_error="RuntimeError: 新浪接口挂了"))
    assert "⚠️ 结构灯生成失败" in text
    assert "🟢 正常" in text and "【隔夜美股】" in text  # 主灯照常


FAKE_HOLDINGS = {"signal_date": "20260715", "any_stale": False, "rows": [
    {"symbol": "601698", "name": "中国卫通", "position": 1200, "close": 27.42,
     "close_date": "2026-07-15", "stale": False, "entry": 30.306, "pnl_pct": -0.0952,
     "value": 32904.0, "hard_stop": 27.88, "atr_stop": 25.9, "stop_line": 27.88,
     "broken": True, "dist_pct": -0.0165, "decision": "DROP",
     "risks": ["PE_TTM=296（估值偏高）"]},
    {"symbol": "002716", "name": "湖南白银", "position": 1000, "close": 7.45,
     "close_date": "2026-07-15", "stale": False, "entry": 8.705, "pnl_pct": -0.1442,
     "value": 7450.0, "hard_stop": 8.01, "atr_stop": 7.0, "stop_line": 8.01,
     "broken": True, "dist_pct": -0.07, "decision": "HOLD", "risks": []},
]}


def test_format_holdings_shows_stops_and_engine_signal():
    text = daily_brief.format_brief(dict(FAKE, holdings=FAKE_HOLDINGS))
    assert "【持仓健康度】" in text and "引擎信号=20260715" in text
    assert "中国卫通 601698" in text and "已破❗" in text and "引擎:DROP" in text
    assert "湖南白银" in text and "引擎:HOLD" not in text  # HOLD 不重复展示


def test_holdings_error_degrades_without_killing_main_light():
    text = daily_brief.format_brief(dict(FAKE, holdings_error="FileNotFoundError: x"))
    assert "⚠️ 持仓健康度生成失败" in text and "🟢 正常" in text


def test_lights_log_dedupes(monkeypatch, tmp_path):
    fake = dict(FAKE, lights=[dict(x) for x in FAKE_LIGHTS])
    assert _run(monkeypatch, tmp_path, lambda: dict(fake)) == 0
    assert _run(monkeypatch, tmp_path, lambda: dict(fake)) == 0  # 同日重跑
    log = (tmp_path / "lights_log.csv").read_text().strip().splitlines()
    assert log[0] == "date,light,state,raw_today,close,ma_days"
    assert len(log) == 3  # 表头 + 两盏灯各一行
