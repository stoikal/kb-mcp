"""Deployment artefacts: backup script against the test database, unit files, env example."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / "deploy" / "backup.sh"


@pytest.mark.skipif(not shutil.which("pg_dump") or not shutil.which("pg_restore"), reason="needs postgres client tools")
def test_backup_creates_private_dump_and_prunes_old(tmp_path):
    old = tmp_path / "ragmcp-old.dump"
    old.write_text("stale")
    os.utime(old, (1, 1))
    env = {**os.environ, "RAGMCP_BACKUP_DIR": str(tmp_path), "RAGMCP_BACKUP_KEEP_DAYS": "7"}
    r = subprocess.run([str(BACKUP)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    dumps = list(tmp_path.glob("ragmcp-2*.dump"))
    assert len(dumps) == 1 and not old.exists()
    assert oct(dumps[0].stat().st_mode & 0o777) == "0o600"
    assert not list(tmp_path.glob("*.partial"))
    # The password must not leak into output.
    assert "ragmcp_local_dev" not in r.stdout + r.stderr


def test_backup_fails_without_url_or_bad_keep(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "RAGMCP_STORE_URL"}
    assert subprocess.run([str(BACKUP)], env=env, capture_output=True).returncode != 0
    env = {**os.environ, "RAGMCP_BACKUP_DIR": str(tmp_path), "RAGMCP_BACKUP_KEEP_DAYS": "x; rm -rf /"}
    assert subprocess.run([str(BACKUP)], env=env, capture_output=True).returncode == 2


def test_unit_files_and_env_example_are_consistent():
    unit = (ROOT / "deploy" / "ragmcp.service").read_text()
    assert "EnvironmentFile=" in unit and "--transport streamable-http" in unit and "Restart=on-failure" in unit
    assert "[Timer]" in (ROOT / "deploy" / "ragmcp-backup.timer").read_text()
    env = (ROOT / "deploy" / "ragmcp.env.example").read_text()
    for key in ("RAGMCP_STORE_URL", "RAGMCP_HOST", "RAGMCP_ADMIN_TOKENS", "RAGMCP_READER_TOKENS"):
        assert key in env
    # Copying the example without editing it must not start a server with known tokens.
    from ragmcp.auth import TokenRegistry

    values = dict(line.split("=", 1) for line in env.splitlines() if line.startswith("RAGMCP_") and "=" in line)
    with pytest.raises(ValueError, match="CHANGE_ME"):
        TokenRegistry([values["RAGMCP_ADMIN_TOKENS"]], [])
    with pytest.raises(ValueError, match="CHANGE_ME"):
        TokenRegistry([], [values["RAGMCP_READER_TOKENS"]])
