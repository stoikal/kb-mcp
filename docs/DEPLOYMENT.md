# Deploying ragmcp on the LAN (Linux, systemd user service)

Goal: one shared knowledge base, reachable only from the local network, admin writes, reader searches.

## 1. Postgres (once)

Postgres stays on `localhost` only (`listen_addresses = 'localhost'` in `postgresql.conf`; this is the default).

```sql
CREATE ROLE ragmcp LOGIN PASSWORD '...';
CREATE DATABASE ragmcp OWNER ragmcp;
\c ragmcp
CREATE EXTENSION vector;
```

## 2. Fixed IP

Clients keep the URL `http://<LAN-IP>:8000/mcp`, so the address must not change. Either:

- **Router DHCP reservation** (preferred): bind the machine's MAC address to e.g. `192.168.1.20`; or
- a **static address** on the machine (NetworkManager: `nmcli con mod <conn> ipv4.method manual ipv4.addresses 192.168.1.20/24 ipv4.gateway 192.168.1.1 ipv4.dns 192.168.1.1`).

Check with `ip -4 addr`.

## 3. Install the service

```bash
mkdir -p ~/.config/systemd/user ~/.config/ragmcp ~/.local/state/ragmcp
cp deploy/ragmcp.service deploy/ragmcp-backup.service deploy/ragmcp-backup.timer ~/.config/systemd/user/
cp deploy/ragmcp.env.example ~/.config/ragmcp/ragmcp.env
chmod 600 ~/.config/ragmcp/ragmcp.env      # holds the DB password and tokens
$EDITOR ~/.config/ragmcp/ragmcp.env        # set RAGMCP_HOST (LAN IP), DB password, tokens
systemctl --user daemon-reload
systemctl --user enable --now ragmcp.service ragmcp-backup.timer
loginctl enable-linger "$USER"             # start at boot, not only at login
```

Adjust `WorkingDirectory`/`ExecStart` in the unit if the project or `uv` live elsewhere (`command -v uv`).
The server refuses to start in HTTP mode if no tokens are configured.

Day to day: `systemctl --user status ragmcp`, `journalctl --user -u ragmcp -f`, `systemctl --user restart ragmcp` (after editing tokens).

## 4. Firewall: LAN only

Allow the port from the local subnet only (adjust the subnet and port):

```bash
sudo ufw allow from 192.168.1.0/24 to any port 8000 proto tcp
sudo ufw deny 8000/tcp        # anything else
sudo ufw status numbered
```

Do not forward the port on the router. Postgres (5432) needs no rule: it is not exposed.
The traffic is plain HTTP, so tokens are readable on the wire; fine on a trusted home network.
If a client requires HTTPS or the network is shared, put Caddy with a local certificate in front, or use Tailscale.

## 5. Tokens

Generate: `python3 -c 'import secrets; print(secrets.token_urlsafe(32))'`.
`RAGMCP_ADMIN_TOKENS` are yours (ingest, delete, reindex); give people only `RAGMCP_READER_TOKENS`
(search, get_chunk, get_document, list_*, stats). Several tokens per role are comma-separated, one per
person so you can revoke one: remove it from the env file and restart. Tokens are never logged
(log records are redacted for `Bearer ...`, DB passwords and token variables).

Client config, e.g. Claude Code:

```bash
claude mcp add --transport http kb http://192.168.1.20:8000/mcp --header "Authorization: Bearer <token>"
```

Smoke test: `curl -i -X POST http://192.168.1.20:8000/mcp` must return `401`.

## 6. Backups

`deploy/backup.sh` runs daily through `ragmcp-backup.timer` (03:30). It writes
`~/backups/ragmcp/ragmcp-<UTC time>.dump` (mode 600), verifies the dump can be listed, and prunes dumps older
than `RAGMCP_BACKUP_KEEP_DAYS` (14). Run now: `systemctl --user start ragmcp-backup.service`; check
`systemctl --user list-timers`. Copy the folder to another disk or machine too: a backup on the same disk
does not survive that disk.

Restore (test it once): `pg_restore --clean --if-exists --no-owner -d "$RAGMCP_STORE_URL" <dump>`.
Embeddings are in the dump, so no re-embedding is needed unless you change the embedding model (then call `reindex`).

## 7. Logging

Logs go to stderr (the journal). Set `RAGMCP_LOG_LEVEL` (default `INFO`) and optionally `RAGMCP_LOG_FILE`
for a rotating file (5 MB x 5). Access lines show method, path and status, never headers or bodies.

## 8. Retrieval eval

`uv run ragmcp-eval -v` ingests `eval/corpus/` into an in-memory store with the real embedder and scores
`eval/questions.jsonl` (labels are `file#heading path`) by recall@5 and MRR for vector, keyword and hybrid.
Add `--rerank` to compare the cross-encoder and `--min-recall 0.85` to fail on regressions. Re-run after any
change to chunk size, embedding model or reranker. Extend the set with your real documents' questions.

## Windows

See PLAN.md ("If it runs on Windows instead"): NSSM or Task Scheduler for the service, a `LocalSubnet`
firewall rule, `pgvector` availability as the main risk, `pg_dump` from a scheduled task.
