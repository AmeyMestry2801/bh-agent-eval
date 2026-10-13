"""
Phase 10: append-only audit trail in SQLite.

- Triggers abort UPDATE and DELETE on audit_log.
- A hash chain (prev_hash -> row_hash) makes edits and mid-log deletions detectable even if
  the triggers are dropped. Deleting the LAST row is not detectable from the file alone;
  production would anchor the head hash externally.
- No raw text is stored, only its SHA-256.

ALL DATA IS SYNTHETIC. Demo (rebuilds results/audit.sqlite; a real log is never rebuilt):
    python -m src.audit
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

GENESIS = "GENESIS"
FIELDS = ["ts_utc", "conversation_id", "source_week", "model_name", "model_version",
          "threshold_raw", "score_raw", "score_calibrated", "decision", "text_sha256", "prev_hash"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_utc TEXT NOT NULL,
  conversation_id TEXT NOT NULL,
  source_week INTEGER,
  model_name TEXT NOT NULL,
  model_version TEXT NOT NULL,
  threshold_raw REAL NOT NULL,
  score_raw REAL NOT NULL,
  score_calibrated REAL NOT NULL,
  decision INTEGER NOT NULL CHECK (decision IN (0, 1)),
  text_sha256 TEXT NOT NULL,
  prev_hash TEXT NOT NULL,
  row_hash TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS audit_log_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only: UPDATE blocked'); END;
CREATE TRIGGER IF NOT EXISTS audit_log_no_delete BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only: DELETE blocked'); END;
"""


def compute_hash(rec: dict) -> str:
    payload = json.dumps([rec[k] for k in FIELDS], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode()).hexdigest()


class AuditLog:
    def __init__(self, path):
        self.path = Path(path)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.executescript(SCHEMA)
        self._last = self._read_last()

    def _read_last(self) -> str:
        row = self.conn.execute("SELECT row_hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
        return row[0] if row else GENESIS

    def append(self, *, conversation_id, model_name, model_version, threshold_raw, score_raw,
               score_calibrated, decision, text, source_week=None, ts_utc=None, commit=True) -> str:
        rec = {
            "ts_utc": ts_utc or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "conversation_id": str(conversation_id),
            "source_week": None if source_week is None else int(source_week),
            "model_name": str(model_name), "model_version": str(model_version),
            "threshold_raw": float(threshold_raw), "score_raw": float(score_raw),
            "score_calibrated": float(score_calibrated), "decision": int(decision),
            "text_sha256": sha256_text(text), "prev_hash": self._last,
        }
        h = compute_hash(rec)
        cols = FIELDS + ["row_hash"]
        self.conn.execute(
            f"INSERT INTO audit_log ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
            [rec[k] for k in FIELDS] + [h])
        self._last = h
        if commit:
            self.conn.commit()
        return h

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


def verify_chain(conn: sqlite3.Connection) -> dict:
    """Recompute the chain. Detects edited rows, mid-log deletions, id gaps and reordering."""
    cur = conn.execute(f"SELECT id, {', '.join(FIELDS)}, row_hash FROM audit_log ORDER BY id")
    expected_prev, expected_id, n = GENESIS, None, 0
    for row in cur:
        rid, rec, stored = row[0], dict(zip(FIELDS, row[1:-1])), row[-1]
        if expected_id is not None and rid != expected_id:
            return {"ok": False, "n_checked": n, "first_bad_id": rid, "reason": "id gap (row deleted)"}
        if rec["prev_hash"] != expected_prev:
            return {"ok": False, "n_checked": n, "first_bad_id": rid,
                    "reason": "prev_hash mismatch (row deleted or reordered)"}
        if compute_hash(rec) != stored:
            return {"ok": False, "n_checked": n, "first_bad_id": rid, "reason": "row content modified"}
        expected_prev, expected_id, n = stored, rid + 1, n + 1
    return {"ok": True, "n_checked": n, "first_bad_id": None, "reason": ""}


def _file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def _try(conn: sqlite3.Connection, sql: str) -> str:
    try:
        conn.execute(sql)
        conn.commit()
        return "NOT BLOCKED (unexpected)"
    except sqlite3.DatabaseError as e:
        return f"blocked -> {e}"


def main() -> None:
    import joblib
    import pandas as pd

    from src.calibrate import apply_platt
    from src.config import DATA_PROCESSED, RESULTS, ensure_dirs, set_seed
    from src.evaluate import CRISIS_IDX
    from src.models import M2TfidfLR  # noqa: F401  (needed to unpickle M2)

    set_seed()
    ensure_dirs()
    cfg_path = RESULTS / "decision_config.json"
    cfg = json.loads(cfg_path.read_text())
    thr = float(cfg["threshold_raw"])
    a, b = float(cfg["platt"]["a"]), float(cfg["platt"]["b"])
    model_path = RESULTS / "models" / "M2.joblib"
    m2 = joblib.load(model_path)
    version = f"M2:{_file_sha(model_path)}|cfg:{_file_sha(cfg_path)}"

    df = pd.read_csv(DATA_PROCESSED / "conversations_split.csv")
    mon = df[df.split == "monitor"].reset_index(drop=True)
    p_raw = m2.predict_proba(mon)[:, CRISIS_IDX]
    p_cal = apply_platt(p_raw, a, b)

    db = RESULTS / "audit.sqlite"
    db.unlink(missing_ok=True)  # demo only: a production audit log is never rebuilt
    log = AuditLog(db)
    for i, r in enumerate(mon.itertuples()):
        log.append(conversation_id=r.conversation_id, source_week=r.week, model_name="M2_tfidf_lr",
                   model_version=version, threshold_raw=thr, score_raw=p_raw[i],
                   score_calibrated=p_cal[i], decision=int(p_raw[i] >= thr), text=r.text, commit=False)
    log.commit()

    con = log.conn
    n, esc = con.execute("SELECT COUNT(*), SUM(decision) FROM audit_log").fetchone()
    print(f"logged {n} decisions from the monitor window ({esc} escalations) | version {version}")
    print("last 3 rows (no raw text stored):")
    for row in con.execute("SELECT id, conversation_id, source_week, round(score_raw,4), decision, "
                           "substr(text_sha256,1,10), substr(row_hash,1,10) FROM audit_log "
                           "ORDER BY id DESC LIMIT 3"):
        print("  ", row)

    upd = _try(con, "UPDATE audit_log SET decision = 0 WHERE id = 1")
    dele = _try(con, "DELETE FROM audit_log WHERE id = 1")
    print(f"\nUPDATE attempt: {upd}\nDELETE attempt: {dele}")
    ver = verify_chain(con)
    print(f"chain verification of the intact log: {ver}")
    log.close()

    tamper = {}
    for label, sql in [("edit row 100 (flip a decision)", "UPDATE audit_log SET decision = 1 - decision WHERE id = 100"),
                       ("delete row 200", "DELETE FROM audit_log WHERE id = 200")]:
        copy = RESULTS / "audit_tamper_copy.sqlite"
        shutil.copy(db, copy)
        c = sqlite3.connect(str(copy))
        c.execute("DROP TRIGGER audit_log_no_update")
        c.execute("DROP TRIGGER audit_log_no_delete")
        c.execute(sql)
        c.commit()
        tamper[label] = verify_chain(c)
        c.close()
        copy.unlink()
        print(f"tamper on a COPY with triggers dropped ({label}): {tamper[label]}")
    print("known limit: deleting the LAST row, or replacing the whole file, is not detectable "
          "from the file alone; production needs an externally anchored head hash.")

    (RESULTS / "audit_demo.json").write_text(json.dumps({
        "n_logged": n, "n_escalations": int(esc), "model_version": version,
        "update_attempt": upd, "delete_attempt": dele, "intact_chain": ver, "tamper_demo": tamper},
        indent=2))
    print("\nwrote results/audit.sqlite, results/audit_demo.json")


if __name__ == "__main__":
    main()
