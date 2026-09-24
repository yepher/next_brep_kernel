#!/usr/bin/env bash
# Import new releases of the BREP_kernel crate from crates.io into this repo.
#
# Each release newer than the version in Cargo.toml is downloaded, verified
# against its crates.io checksum, unpacked over the working tree, and
# committed to main as "Version X https://crates.io/crates/BREP_kernel/X".
# Intermediate releases each get their own commit, so `git diff` between
# commits shows exactly what changed upstream.
#
# Usage:
#   tools/bin/sync-crate.sh              # import every newer release
#   tools/bin/sync-crate.sh --dry-run    # just list what would be imported
#   tools/bin/sync-crate.sh --push       # import, then push main to origin
#   tools/bin/sync-crate.sh 0.6.0        # import one specific version
#
# Files under tools/ are never touched, so local scripts survive imports.

set -euo pipefail

CRATE="BREP_kernel"
BRANCH="main"
API="https://crates.io/api/v1/crates/${CRATE}"
UA="${CRATE}-sync-script (https://github.com/yepher/next_brep_kernel)"

dry_run=0
push=0
requested=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry_run=1 ;;
    --push) push=1 ;;
    -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown option: $arg" >&2; exit 2 ;;
    *) requested+=("$arg") ;;
  esac
done

repo="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$repo"

current="$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)"
[[ -n "$current" ]] || { echo "could not read version from Cargo.toml" >&2; exit 1; }

meta="$(curl -fsSL -A "$UA" "$API")"

# Newer, non-yanked, non-prerelease versions in ascending order.
if [[ ${#requested[@]} -gt 0 ]]; then
  versions=("${requested[@]}")
else
  versions=()
  while IFS= read -r v; do
    [[ -n "$v" ]] && versions+=("$v")
  done < <(
    jq -r '.versions[] | select(.yanked | not) | .num | select(test("-") | not)' <<<"$meta" |
      { cat; echo "$current"; } | sort -V -u |
      awk -v cur="$current" 'found { print } $0 == cur { found = 1 }'
  )
fi

if [[ ${#versions[@]} -eq 0 ]]; then
  echo "Up to date: ${CRATE} ${current}"
  exit 0
fi

echo "Current: ${current}"
echo "To import: ${versions[*]}"
[[ $dry_run -eq 1 ]] && exit 0

if [[ "$(git symbolic-ref --short HEAD)" != "$BRANCH" ]]; then
  echo "not on ${BRANCH}; check out ${BRANCH} first" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "working tree has uncommitted changes; commit or stash them first" >&2
  exit 1
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

for v in "${versions[@]}"; do
  echo "==> ${CRATE} ${v}"
  checksum="$(jq -r --arg v "$v" '.versions[] | select(.num == $v) | .checksum' <<<"$meta")"
  [[ -n "$checksum" ]] || { echo "version $v not found on crates.io" >&2; exit 1; }

  archive="$tmp/${CRATE}-${v}.crate"
  curl -fsSL -A "$UA" -o "$archive" "https://static.crates.io/crates/${CRATE}/${CRATE}-${v}.crate"
  actual="$(shasum -a 256 "$archive" | cut -d' ' -f1)"
  if [[ "$actual" != "$checksum" ]]; then
    echo "checksum mismatch for $v: expected $checksum, got $actual" >&2
    exit 1
  fi

  rm -rf "$tmp/src"
  mkdir -p "$tmp/src"
  tar -xzf "$archive" -C "$tmp/src"
  src="$tmp/src/${CRATE}-${v}"

  # Remove tracked files (except tools/) so files deleted upstream go away,
  # then lay the new release down and stage exactly its contents.
  git ls-files -z -- . ':!tools' | xargs -0 rm -f
  (cd "$src" && tar -cf - .) | tar -xf -
  git add -u -- . ':!tools'
  (cd "$src" && find . -type f -print0) | git add --pathspec-from-file=- --pathspec-file-nul

  if git diff --cached --quiet; then
    echo "    no changes; skipping commit"
    continue
  fi
  git commit -q -m "Version ${v} https://crates.io/crates/${CRATE}/${v}"
  echo "    committed $(git rev-parse --short HEAD)"
done

if [[ $push -eq 1 ]]; then
  git push origin "$BRANCH"
fi
