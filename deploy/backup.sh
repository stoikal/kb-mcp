#!/usr/bin/env bash
# Dump the ragmcp database (custom format, compressed) and prune old dumps.
# Reads RAGMCP_STORE_URL from the environment (systemd loads the env file; for a manual run:
#   set -a; . ~/.config/ragmcp/ragmcp.env; set +a; deploy/backup.sh).
# The password is passed to pg_dump through PGPASSWORD, never on the command line or in logs.
# Restore:  pg_restore --clean --if-exists -d "$RAGMCP_STORE_URL" <dump>
set -euo pipefail
umask 077

: "${RAGMCP_STORE_URL:?RAGMCP_STORE_URL is not set}"
dir="${RAGMCP_BACKUP_DIR:-$HOME/backups/ragmcp}"
keep_days="${RAGMCP_BACKUP_KEEP_DAYS:-14}"
[[ "$keep_days" =~ ^[0-9]+$ ]] || { echo "RAGMCP_BACKUP_KEEP_DAYS must be a number" >&2; exit 2; }

mkdir -p "$dir"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out="$dir/ragmcp-$stamp.dump"
tmp="$out.partial"
trap 'rm -f "$tmp"' EXIT

# Split the URL so the password never appears in argv (visible in `ps`).
url="${RAGMCP_STORE_URL#*://}"
creds="${url%%@*}"
if [[ "$url" == *@* && "$creds" == *:* ]]; then
  export PGPASSWORD="${creds#*:}"
  conn="postgresql://${creds%%:*}@${url#*@}"
else
  conn="$RAGMCP_STORE_URL"
fi

pg_dump --format=custom --compress=6 --no-owner --dbname="$conn" --file="$tmp"
# A dump we cannot list is not a backup.
pg_restore --list "$tmp" > /dev/null
mv "$tmp" "$out"
trap - EXIT

find "$dir" -maxdepth 1 -name 'ragmcp-*.dump' -type f -mtime "+$keep_days" -delete
echo "backup ok: $out ($(du -h "$out" | cut -f1))"
