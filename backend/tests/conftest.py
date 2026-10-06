"""Run backend tests against a throwaway database, never the dev sentinelops.db."""
import os
import tempfile
from pathlib import Path

_test_db = Path(tempfile.mkdtemp(prefix="sentinelops-test-")) / "test.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_test_db.as_posix()}"
