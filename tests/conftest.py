import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    """Every test gets its own throwaway SQLite file — never touches data/pathfinder.db."""
    test_db_path = str(tmp_path / "test_pathfinder.db")
    monkeypatch.setenv("PATHFINDER_DB_PATH", test_db_path)

    # database.py reads DB_PATH at import time, so re-point it directly too
    # in case it was already imported by an earlier test module.
    from src import database as db
    monkeypatch.setattr(db, "DB_PATH", test_db_path)
    db.init_db()
    yield db
