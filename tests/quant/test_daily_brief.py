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
    (tmp_path / "data_health_alert.txt").write_text("2026-07-11 周五数据维护失败步骤: financials\n")
    monkeypatch.setattr(daily_brief, "data_health", lambda today=None: [])
    assert _run(monkeypatch, tmp_path, lambda: dict(FAKE)) == 0
    assert "financials" in (tmp_path / "latest.md").read_text()


def _health_cfg(tmp_path, output_name="csi_union"):
    return {"qlib": {"provider_uri": str(tmp_path / "qlib")},
            "warehouse": {"path": str(tmp_path / "wh")},
            "universe": {"output_name": output_name}}


def _write_qlib(tmp_path, last_day: str, codes: int = 100):
    cal = tmp_path / "qlib" / "calendars"
    cal.mkdir(parents=True, exist_ok=True)
    (cal / "day.txt").write_text(f"2026-01-02\n{last_day}\n")
    inst = tmp_path / "qlib" / "instruments"
    inst.mkdir(parents=True, exist_ok=True)
    (inst / "csi_union.txt").write_text(
        "".join(f"SH{600000 + i}\t2020-01-01\t2026-08-05\n" for i in range(codes)))


def test_data_health_clean_when_fresh(monkeypatch, tmp_path):
    monkeypatch.setattr(daily_brief, "load_config", lambda _: _health_cfg(tmp_path))
    _write_qlib(tmp_path, "2026-08-04")
    assert daily_brief.data_health(pd.Timestamp("2026-08-05")) == []


def test_data_health_flags_stale_qlib(monkeypatch, tmp_path):
    monkeypatch.setattr(daily_brief, "load_config", lambda _: _health_cfg(tmp_path))
    _write_qlib(tmp_path, "2026-07-03")
    issues = daily_brief.data_health(pd.Timestamp("2026-08-05"))
    assert len(issues) == 1
    assert "落后 33 天" in issues[0] and "bootstrap" in issues[0]


def test_data_health_ignores_retired_fundflow(monkeypatch, tmp_path):
    """em_fundflow 2026-11-11 废弃：仓库里没有它不能再报警。"""
    monkeypatch.setattr(daily_brief, "load_config", lambda _: _health_cfg(tmp_path))
    _write_qlib(tmp_path, "2026-08-04")
    assert not (tmp_path / "wh" / "em_fundflow.parquet").exists()
    assert daily_brief.data_health(pd.Timestamp("2026-08-05")) == []


def test_data_health_reports_missing_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(daily_brief, "load_config", lambda _: _health_cfg(tmp_path))
    issues = daily_brief.data_health(pd.Timestamp("2026-08-05"))
    assert any("qlib 数据包缺失" in i for i in issues)


def test_data_health_survives_probe_failure(monkeypatch, tmp_path):
    """体检自身抛错不能拖垮晨报——红绿灯比数据告警重要。"""
    def boom(_):
        raise RuntimeError("配置读不到")

    monkeypatch.setattr(daily_brief, "load_config", boom)
    assert _run(monkeypatch, tmp_path, lambda: dict(FAKE)) == 0
    text = (tmp_path / "latest.md").read_text()
    assert "数据体检自身失败" in text and "🟢" in text


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
     "value": 32904.0, "hard_stop": 27.88, "init_stop": 27.88, "ratchet": 25.9,
     "stop_line": 27.88, "broken": True, "dist_pct": -0.0165,
     "trend_line": 25.9, "above_trend": True,
     "degraded": False, "degrade_reason": "", "arm_suspect": None, "dd_at_entry": None,
     "decision": "DROP", "risks": ["PE_TTM=296（估值偏高）"]},
    {"symbol": "002716", "name": "湖南白银", "position": 1000, "close": 7.45,
     "close_date": "2026-07-15", "stale": False, "entry": 8.705, "pnl_pct": -0.1442,
     "value": 7450.0, "hard_stop": 8.01, "init_stop": 8.01, "ratchet": None,
     "stop_line": 8.01, "broken": True, "dist_pct": -0.07,
     "trend_line": 7.0, "above_trend": True,
     "degraded": True, "degrade_reason": "缺entry_date",
     "arm_suspect": None, "dd_at_entry": None,
     "decision": "HOLD", "risks": []},
]}


def test_format_holdings_shows_stops_and_engine_signal():
    text = daily_brief.format_brief(dict(FAKE, holdings=FAKE_HOLDINGS))
    assert "【持仓健康度】" in text and "引擎信号=20260715" in text
    assert "中国卫通 601698" in text and "已破❗" in text and "引擎:DROP" in text
    assert "ATR趋势线 25.90 上方🟢" in text          # 趋势读数与止损线拆分展示
    assert "⚠️(缺entry_date, 仅硬止损)" in text      # 降级路径必须显式标注
    assert "湖南白银" in text and "引擎:HOLD" not in text  # HOLD 不重复展示


def test_holdings_error_degrades_without_killing_main_light():
    text = daily_brief.format_brief(dict(FAKE, holdings_error="FileNotFoundError: x"))
    assert "⚠️ 持仓健康度生成失败" in text and "🟢 正常" in text


FAKE_REENTRY = {"signal_date": "20260715", "any_stale": False, "rows": [
    {"symbol": "002842", "name": "翔鹭钨业", "exit_date": "2026-07-17", "exit_price": 35.96,
     "stale": False, "close": 34.10, "trail": 38.20, "trail_dist": -0.1073,
     "cooldown_elapsed": 3, "cooldown_total": 10, "cooldown_done": False,
     "light_name": "中证1000", "decision": "DROP", "sector_weak": True,
     "light_ok": False, "trend_ok": False, "engine_ok": False, "met": 0, "ready": False},
    {"symbol": "002639", "name": "雪人集团", "exit_date": "2026-07-01", "exit_price": 12.50,
     "stale": False, "close": 13.80, "trail": 13.10, "trail_dist": 0.0534,
     "cooldown_elapsed": 12, "cooldown_total": 10, "cooldown_done": True,
     "light_name": "中证1000", "decision": "HOLD",
     "light_ok": True, "trend_ok": True, "engine_ok": True, "met": 3, "ready": True},
]}


def test_format_reentry_shows_conditions_and_ready():
    text = daily_brief.format_brief(dict(FAKE, reentry=FAKE_REENTRY))
    assert "【回补观察】" in text
    assert "翔鹭钨业 002842 卖35.96 现34.10" in text
    assert "冷静期 3/10" in text and "①中证1000❌" in text
    assert "②站回趋势线38.20❌(-10.7%)" in text and "③引擎DROP·板块弱❌" in text and "→ 0/3" in text
    assert "雪人集团" in text and "冷静期已过" in text
    assert "→ 3/3 ✅ 条件齐备" in text


def test_format_reentry_hidden_when_no_exited():
    text = daily_brief.format_brief(dict(FAKE, reentry={"rows": [], "signal_date": "", "any_stale": False}))
    assert "【回补观察】" not in text


def test_reentry_error_degrades_without_killing_main_light():
    text = daily_brief.format_brief(dict(FAKE, reentry_error="RuntimeError: x"))
    assert "⚠️ 回补观察生成失败" in text and "🟢 正常" in text


def test_lights_log_dedupes(monkeypatch, tmp_path):
    fake = dict(FAKE, lights=[dict(x) for x in FAKE_LIGHTS])
    assert _run(monkeypatch, tmp_path, lambda: dict(fake)) == 0
    assert _run(monkeypatch, tmp_path, lambda: dict(fake)) == 0  # 同日重跑
    log = (tmp_path / "lights_log.csv").read_text().strip().splitlines()
    assert log[0] == "date,light,state,raw_today,close,ma_days"
    assert len(log) == 3  # 表头 + 两盏灯各一行


FAKE_STOCK_LAMPS = {"any_stale": False, "thr": -0.35, "rows": [
    {"symbol": "002074", "name": "国轩高科", "state": "WATCHING", "stale": False,
     "lamp": "fire", "signals": [{"signal": "L6超卖修复", "signal_date": "2026-07-11"}],
     "close": 26.91, "dd120": -0.35, "rsi": 44.6},
    {"symbol": "002639", "name": "雪人集团", "state": "HELD", "stale": False,
     "lamp": "fire", "signals": [{"signal": "L1接刀", "signal_date": "2026-07-11"}],
     "close": 11.14, "dd120": -0.52, "rsi": 28.1},
    {"symbol": "601611", "name": "中国核建", "state": "EXITED", "stale": False,
     "lamp": "fire", "signals": [{"signal": "L1接刀", "signal_date": "2026-07-10"}],
     "close": 9.70, "dd120": -0.49, "rsi": 35.6},
    {"symbol": "002466", "name": "天齐锂业", "state": "WATCHING", "stale": False,
     "lamp": "watch", "signals": [], "close": 46.99, "dd120": -0.42, "rsi": 32.4},
]}


def _c1k(state):
    return dict(FAKE_LIGHTS[0], state=state)


def test_format_stock_lamps_red_window_lists_fires_excludes_held():
    b = dict(FAKE, lights=[_c1k("risk_off")],
             stock_lamps={**FAKE_STOCK_LAMPS, "rows": [dict(r) for r in FAKE_STOCK_LAMPS["rows"]]})
    text = daily_brief.format_brief(b)
    assert "【个股抄底灯】" in text and "≤-35%" in text
    assert "恐慌期=左侧超额最大" in text and "止损=入场价-3×ATR14" in text
    assert "🟢 国轩高科 002074 L6超卖修复" in text
    assert "雪人集团 002639" in text and "持仓中·非补仓依据" in text  # HELD 列出但标非补仓
    assert "中国核建" in text and "接回归回补观察" in text
    assert "🟡底部区观察 1 只" in text


def test_format_stock_lamps_green_window_downgrades_to_record_only():
    b = dict(FAKE, lights=[_c1k("risk_on")],
             stock_lamps={**FAKE_STOCK_LAMPS, "rows": [dict(r) for r in FAKE_STOCK_LAMPS["rows"]]})
    text = daily_brief.format_brief(b)
    assert "绿灯期独跌票无超额" in text and "仅记录、不建议入场" in text


def test_stock_lamps_error_degrades_without_killing_main_light():
    text = daily_brief.format_brief(dict(FAKE, stock_lamps_error="RuntimeError: x"))
    assert "⚠️ 个股抄底灯生成失败" in text and "🟢 正常" in text


def test_stock_lamps_log_dedupes_by_signal_date_and_excludes_held(monkeypatch, tmp_path):
    fake = dict(FAKE, lights=[_c1k("risk_off")],
                stock_lamps={**FAKE_STOCK_LAMPS, "rows": [dict(r) for r in FAKE_STOCK_LAMPS["rows"]]})
    assert _run(monkeypatch, tmp_path, lambda: dict(fake)) == 0
    assert _run(monkeypatch, tmp_path, lambda: dict(fake)) == 0          # 同日重跑
    # 次日晨报同一信号仍在 FIRE_WINDOW 展示窗口内 → 不得再落一行（防对账 3 倍计数）
    assert _run(monkeypatch, tmp_path, lambda: dict(fake, date="2026-07-14")) == 0
    log = (tmp_path / "stock_lamps_log.csv").read_text().strip().splitlines()
    assert log[0] == "signal_date,symbol,signal,wl_state,close,dd120,rsi,csi1000,brief_date"
    assert len(log) == 3                              # 表头 + 国轩高科 + 中国核建
    assert not any("002639" in ln for ln in log)      # HELD 不落流水
    assert log[1].startswith("2026-07-11,002074,L6超卖修复,")
    assert log[1].endswith(",risk_off,2026-07-13")


# ---- 2026-11-10 晨报抄底灯三处误导：信号日不标 / 卫星满额仍"可执行" / 科创板最小手超预算 ----

def _lamp_brief(rows, structure=None, kc50="risk_on", date="2026-11-10"):
    lights = [_c1k("risk_off"), dict(FAKE_LIGHTS[1], state=kc50)]
    b = dict(FAKE, date=date, lights=lights, stock_lamps={**FAKE_STOCK_LAMPS, "rows": rows})
    if structure is not None:
        b["structure"] = structure
    return daily_brief.format_brief(b)


def _fire(symbol, name, close, signal_date, signal="L1接刀"):
    return {"symbol": symbol, "name": name, "state": "WATCHING", "stale": False, "lamp": "fire",
            "signals": [{"signal": signal, "signal_date": signal_date}],
            "close": close, "dd120": -0.5, "rsi": 30.0}


def _struct(count, total=180_000.0):
    return {"total": total, "stock_count": count, "compliant": True, "cash_pct": 0.36,
            "etf_pct": 0.43, "etf_cost_fallback": [],
            "target": {"core_etf_lo": 0.4, "core_etf_hi": 0.5, "max_count": 3, "max_pct": 0.1}}


def test_lamp_shows_signal_date_and_marks_missed_entry():
    # 11-10(周二)晨报：11-09 信号 → 今日开盘入场；11-06 信号 → 入场日 11-09 已过
    text = _lamp_brief([_fire("002463", "沪电股份", 77.1, "2026-11-06"),
                        _fire("600176", "中国巨石", 27.4, "2026-11-09")])
    assert "信号11-06" in text and "入场日11-09已过" in text
    assert "信号11-09" in text and "今日开盘入场" in text


def test_lamp_window_blocks_new_entries_when_satellites_full():
    text = _lamp_brief([_fire("600176", "中国巨石", 27.4, "2026-11-09")], structure=_struct(3))
    assert "卫星已满 3/3" in text and "不可新开仓" in text
    assert "可按纪律执行" not in text


def test_lamp_window_shows_free_slots():
    text = _lamp_brief([_fire("600176", "中国巨石", 27.4, "2026-11-09")], structure=_struct(2))
    assert "可按纪律执行" in text and "卫星剩 1 个名额" in text


def test_star_board_min_lot_over_budget_flagged_and_kc50_red_halves():
    # 688 最小 200 股：62.81×200=12,562 > 18万×5%=9,000；科创50🔴再减半→4,500
    text = _lamp_brief([_fire("688183", "生益电子", 62.81, "2026-11-09")],
                       structure=_struct(2), kc50="risk_off")
    assert "最小200股≈12,562" in text and "超预算4,500" in text and "科创50🔴减半" in text


def test_main_board_lot_within_budget_not_flagged():
    text = _lamp_brief([_fire("600176", "中国巨石", 27.4, "2026-11-09")], structure=_struct(2))
    assert "超预算" not in text


def test_held_lamp_with_broken_stop_says_stop_wins():
    held = dict(_fire("601208", "东材科技", 34.4, "2026-11-09"), state="HELD")
    b = dict(FAKE, date="2026-11-10", lights=[_c1k("risk_off")],
             stock_lamps={**FAKE_STOCK_LAMPS, "rows": [held]},
             structure=_struct(3, total=180_000.0),
             holdings={"rows": [{"symbol": "601208", "broken": True}]})
    text = daily_brief._format_stock_lamps(b)
    joined = "\n".join(text)
    assert "东材科技 601208" in joined and "止损已破❗以止损为准" in joined
    assert "超预算" not in joined                      # 持仓票不算新仓预算
