"""
Run history and system log.

Always kept in memory (last 200 events / 30 runs) so the dashboard works even
before Data Store is set up. If DATASTORE_TABLE is set (e.g. ForecastRuns) and
the Catalyst SDK is available, each run summary is also inserted into that
Data Store table. Create the table in the Catalyst console with these columns:
run_at (Text), source (Text), rows (Int), peak_kw (Double),
storage_hours (Int), overload_hours (Int), first_mode (Text).
"""
import os
import threading
from collections import deque
from datetime import datetime, timezone

_lock = threading.Lock()
EVENTS = deque(maxlen=200)
RUNS = deque(maxlen=30)


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(level, message):
    with _lock:
        EVENTS.appendleft({"at": now_iso(), "level": level, "message": message})


def record_run(summary, zc_app=None):
    with _lock:
        RUNS.appendleft(summary)
    table = os.environ.get("DATASTORE_TABLE", "").strip()
    if not table or zc_app is None:
        return False
    try:
        zc_app.datastore().table(table).insert_row({
            "run_at": summary["run_at"], "source": summary["source"],
            "rows": summary["rows"], "peak_kw": summary["peak_kw"],
            "storage_hours": summary["storage_hours"],
            "overload_hours": summary["overload_hours"],
            "first_mode": summary["first_mode"],
        })
        return True
    except Exception as e:                                # noqa: BLE001
        log("warn", f"Data Store insert skipped: {e}")
        return False


def recent_events(limit=50):
    with _lock:
        return list(EVENTS)[:limit]


def recent_runs(limit=10):
    with _lock:
        return list(RUNS)[:limit]
