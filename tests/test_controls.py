# tests/test_controls.py
import sys, os, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dashboard import controls as ct


def _tmp_yaml(tmp_path):
    src = os.path.join(os.path.dirname(__file__), "..", "watchlist.yaml")
    dst = tmp_path / "watchlist.yaml"
    shutil.copy(src, dst)
    return str(dst)


def test_save_creates_backup_and_persists(tmp_path):
    path = _tmp_yaml(tmp_path)
    rows = [{"symbol": "002716", "name": "湖南白银", "state": "HELD",
             "position": 500, "entry_price": 12.63}]
    ct.save_watchlist(rows, path=path)
    baks = [f for f in os.listdir(tmp_path) if f.startswith("watchlist.yaml.bak.")]
    assert baks, "应生成备份"
    reloaded = ct.load_watchlist_rows(path=path)
    assert any(r["symbol"] == "002716" and r["state"] == "HELD" for r in reloaded)


def test_invalid_symbol_rejected(tmp_path):
    path = _tmp_yaml(tmp_path)
    try:
        ct.save_watchlist([{"symbol": "ABC", "state": "WATCHING"}], path=path)
        assert False, "非法 symbol 应抛错"
    except ValueError:
        pass


def test_held_requires_position(tmp_path):
    path = _tmp_yaml(tmp_path)
    try:
        ct.save_watchlist([{"symbol": "002716", "state": "HELD"}], path=path)
        assert False, "HELD 缺 position/entry_price 应抛错"
    except ValueError:
        pass
