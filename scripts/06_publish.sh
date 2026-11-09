#!/usr/bin/env bash
# Publish the contents of results/ and figures/ to a dedicated branch.
#
# The token is read from PAT_FILE at run time and handed to git through a
# temporary credential helper, so it never lands in .git/config or in the branch.
set -euo pipefail
cd "$(dirname "$0")/.."

BRANCH=${BRANCH:-experiments/$(date -u +%Y%m%d-%H%M)}
PAT_FILE=${PAT_FILE:-/data/lab/.divspace/pat}
REMOTE=${REMOTE:-origin}
STAGE=${STAGE:-/tmp/divspace-publish}
PY=${PYTHON:-python3}

if [ ! -s "$PAT_FILE" ]; then
  echo "no token at $PAT_FILE; skipping publish" >&2
  exit 1
fi
if [ ! -d results ]; then
  echo "no results directory; nothing to publish" >&2
  exit 1
fi

echo "== staging results =="
rm -rf "$STAGE"
mkdir -p "$STAGE"
cp -r results "$STAGE/results"
[ -d figures ] && cp -r figures "$STAGE/figures"

"$PY" analysis/manifest.py --out "$STAGE/manifest.json" --results results

# Raw generations are gzipped so the branch stays small; summaries stay readable.
find "$STAGE/results" -name '*.jsonl' -print0 2>/dev/null |
  while IFS= read -r -d '' file; do gzip -9 "$file"; done || true

echo "== building branch $BRANCH =="
WT=$(mktemp -d)
git worktree add --detach "$WT" HEAD >/dev/null
trap 'git worktree remove --force "$WT" >/dev/null 2>&1 || true; rm -rf "$STAGE"' EXIT

(
  cd "$WT"
  git checkout --orphan "$BRANCH" >/dev/null 2>&1
  git rm -rf --cached . >/dev/null 2>&1 || true
  find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
  cp -r "$STAGE"/. .
  git add -A -f
  git -c user.name="divspace-bot" -c user.email="divspace@users.noreply.github.com" \
    commit -q -m "Experiment run ${BRANCH#experiments/}"
)

HELPER=$(mktemp)
printf '#!/bin/sh\necho "username=x-access-token"\necho "password=$(cat %s)"\n' \
  "$PAT_FILE" > "$HELPER"
chmod 700 "$HELPER"

echo "== pushing =="
git -C "$WT" -c credential.helper="$HELPER" -c http.version=HTTP/1.1 \
  push --force origin "$BRANCH"
rm -f "$HELPER"

echo "published to branch $BRANCH"
