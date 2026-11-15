#!/usr/bin/env bash
# Commit exactly the named files on top of the CURRENT HEAD, through a private
# index, so that parallel lanes committing in the main checkout never revert one
# another (a stale `git read-tree HEAD` taken before a peer's commit records the
# peer's files at their old blobs).
#
# Usage: tools/safe-commit.sh <message-file> <path>...
#
# The index is built from HEAD at commit time, only the named paths are added
# from the working tree, and the commit is refused unless the staged diff
# against HEAD names exactly those paths (a path with no change is an error
# too). The shared index is left untouched. Deleted paths are staged as
# deletions.
set -euo pipefail
[ $# -ge 2 ] || { echo "usage: $0 <message-file> <path>..." >&2; exit 2; }
msg=$1; shift
[ -f "$msg" ] || { echo "safe-commit: no message file $msg" >&2; exit 2; }
cd "$(git rev-parse --show-toplevel)"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
export GIT_INDEX_FILE=$tmp/index
for attempt in 1 2 3; do
  head=$(git rev-parse HEAD)
  git read-tree "$head"
  git add -A -- "$@"
  want=$(printf '%s\n' "$@" | sort -u)
  got=$(git diff --cached --name-only "$head" | sort -u)
  if [ "$got" != "$want" ]; then
    echo "safe-commit: staged paths differ from the named paths; refusing." >&2
    echo "named:"; printf '  %s\n' $want >&2
    echo "staged:"; printf '  %s\n' $got >&2
    exit 1
  fi
  git diff --cached --stat "$head"
  tree=$(git write-tree)
  commit=$(git commit-tree "$tree" -p "$head" -F "$msg")
  # Advance the branch only if HEAD has not moved since read-tree; otherwise
  # rebuild on the new HEAD (the named files are re-read from the working tree).
  if git update-ref -m "safe-commit" "$(git symbolic-ref HEAD)" "$commit" "$head" 2>/dev/null; then
    unset GIT_INDEX_FILE
    git reset -q -- "$@" 2>/dev/null || true
    git log --oneline -1 "$commit"
    exit 0
  fi
  echo "safe-commit: HEAD moved during commit (attempt $attempt); rebuilding on the new HEAD" >&2
done
echo "safe-commit: HEAD kept moving; giving up" >&2
exit 1
