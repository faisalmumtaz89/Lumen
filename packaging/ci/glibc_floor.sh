#!/usr/bin/env bash
# Fails when a Linux binary needs a newer glibc than the published binaries start on:
# GLIBC_2.34, which RHEL 9 and the Ubuntu 22.04 base of the published image both
# provide. A binary needs the newest GLIBC_ version among the symbols it references.
# Usage: glibc_floor.sh <binary>...
set -euo pipefail

FLOOR=2.34
status=0
for bin in "$@"; do
  need="$(objdump -T "$bin" | { grep -o 'GLIBC_[0-9.]*' || true; } | sed 's/^GLIBC_//' | sort -uV | tail -1)"
  if [ -n "$need" ] && [ "$(printf '%s\n%s\n' "$need" "$FLOOR" | sort -V | tail -1)" != "$FLOOR" ]; then
    echo "::error::$bin needs GLIBC_$need, newer than GLIBC_$FLOOR"
    status=1
  else
    echo "$bin needs GLIBC_${need:-none} (at most GLIBC_$FLOOR)"
  fi
done
exit "$status"
