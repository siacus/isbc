#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
EXTERNAL_ROOT="${EXTERNAL_METHOD_ROOT:-$REPO_DIR/.external_methods}"

HYPOTHESAES_URL="https://github.com/rmovva/HypotheSAEs.git"
HYPOTHESAES_COMMIT="706d8979c71befe8e3151f51a8d4dbf87d4d7e41"
BINARYSENTEMB_URL="https://github.com/Linear95/BinarySentEmb.git"
BINARYSENTEMB_COMMIT="2aaaae8062ea68fdce700c5765822a09d525eaf1"

prepare_checkout() {
  local name="$1"
  local url="$2"
  local commit="$3"
  local destination="$EXTERNAL_ROOT/$name"

  if [[ ! -d "$destination/.git" ]]; then
    git clone "$url" "$destination"
  fi
  if ! git -C "$destination" diff --quiet || \
     ! git -C "$destination" diff --cached --quiet; then
    echo "Refusing to alter dirty external checkout: $destination" >&2
    exit 1
  fi
  git -C "$destination" fetch origin "$commit"
  git -C "$destination" checkout --detach "$commit"
  printf '%s\t%s\n' "$name" "$(git -C "$destination" rev-parse HEAD)"
}

mkdir -p "$EXTERNAL_ROOT"
prepare_checkout HypotheSAEs "$HYPOTHESAES_URL" "$HYPOTHESAES_COMMIT"
prepare_checkout BinarySentEmb "$BINARYSENTEMB_URL" "$BINARYSENTEMB_COMMIT"

echo "Official external methods are ready under $EXTERNAL_ROOT"
