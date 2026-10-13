"""Audit-trail tests. ALL DATA IS SYNTHETIC."""
import shutil
import sqlite3

import pytest

from src.audit import AuditLog, verify_chain


def _add(log: AuditLog, n: int) -> None:
    for i in range(n):
        log.append(conversation_id=i, source_week=17, model_name="M", model_version="v1",
                   threshold_raw=0.0927, score_raw=0.01 * i, score_calibrated=0.02 * i,
                   decision=i % 2, text=f"synthetic text {i}", commit=False)
    log.commit()


@pytest.fixture
def log(tmp_path):
    lg = AuditLog(tmp_path / "audit.sqlite")
    _add(lg, 10)
    yield lg
    lg.close()


def _drop_triggers(conn):
    conn.execute("DROP TRIGGER audit_log_no_update")
    conn.execute("DROP TRIGGER audit_log_no_delete")


def test_update_is_blocked(log):
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        log.conn.execute("UPDATE audit_log SET decision = 1 WHERE id = 1")


def test_delete_is_blocked(log):
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        log.conn.execute("DELETE FROM audit_log WHERE id = 1")
    assert log.conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] == 10


def test_intact_chain_verifies(log):
    res = verify_chain(log.conn)
    assert res["ok"] and res["n_checked"] == 10


def test_chain_continues_after_reopen(log, tmp_path):
    path = log.path
    log.close()
    again = AuditLog(path)
    _add(again, 3)
    assert verify_chain(again.conn)["ok"] and again.conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] == 13
    again.close()


def test_edit_detected_even_without_triggers(log, tmp_path):
    copy = tmp_path / "copy.sqlite"
    shutil.copy(log.path, copy)
    c = sqlite3.connect(str(copy))
    _drop_triggers(c)
    c.execute("UPDATE audit_log SET score_raw = 0.99 WHERE id = 4")
    c.commit()
    res = verify_chain(c)
    assert not res["ok"] and res["first_bad_id"] == 4 and "modified" in res["reason"]


def test_middle_deletion_detected_without_triggers(log, tmp_path):
    copy = tmp_path / "copy.sqlite"
    shutil.copy(log.path, copy)
    c = sqlite3.connect(str(copy))
    _drop_triggers(c)
    c.execute("DELETE FROM audit_log WHERE id = 5")
    c.commit()
    res = verify_chain(c)
    assert not res["ok"] and res["first_bad_id"] == 6


def test_no_raw_text_column(log):
    cols = [r[1] for r in log.conn.execute("PRAGMA table_info(audit_log)")]
    assert "text" not in cols and "text_sha256" in cols
