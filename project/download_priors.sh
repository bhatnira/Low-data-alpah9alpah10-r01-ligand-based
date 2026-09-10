#!/usr/bin/env bash
#
# Download REINVENT4 prior models from Zenodo and verify them against the
# sha256 hashes pinned in configs/config.yaml (reinvent.priors).
#
# Source: REINVENT4 priors  (DOI 10.5281/zenodo.20701824, Apache-2.0)
#   https://zenodo.org/records/20701824
#
# Idempotent: skips already-downloaded files whose sha256 matches.
#
# Usage:
#   ./download_priors.sh            # download the two priors this project uses
#   ./download_priors.sh --force    # re-download even if checksums match
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

# name -> URL fragment (record files API)
declare -A PRIORS=(
  ["reinvent_pubchem.prior"]="https://zenodo.org/records/20701824/files/reinvent_pubchem.prior?download=1"
  ["libinvent.prior"]="https://zenodo.org/records/20701824/files/libinvent.prior?download=1"
)

# name -> sha256 (kept in sync with configs/config.yaml)
declare -A SHA256=(
  ["reinvent_pubchem.prior"]="fe8cd1678452ad292a8f93e97cb19a85959b729e17113e157180d5e69ae89ef3"
  ["libinvent.prior"]="03e6cbe8a53e59a4ac3aa6728d041f1957bdd07b5eefdf2cfc5c8591036075af"
)

mkdir -p "$ROOT/priors"
for name in "${!PRIORS[@]}"; do
  dst="$ROOT/priors/$name"
  want="${SHA256[$name]}"
  if [ "$FORCE" -eq 0 ] && [ -f "$dst" ]; then
    got="$(shasum -a 256 "$dst" | awk '{print $1}')"
    if [ "$got" = "$want" ]; then
      echo "OK  $name (sha256 already verified, skipping)"
      continue
    fi
    echo "WARN $name sha256 mismatch ($got); re-downloading"
  fi
  echo "=> downloading $name"
  curl -fL --retry 3 -o "$dst" "${PRIORS[$name]}"
  got="$(shasum -a 256 "$dst" | awk '{print $1}')"
  if [ "$got" != "$want" ]; then
    echo "ERROR $name sha256 mismatch after download: $got != $want" >&2
    rm -f "$dst"
    exit 1
  fi
  echo "OK  $name (sha256 verified)"
done
echo "All prior models present and sha256-verified in $ROOT/priors/"