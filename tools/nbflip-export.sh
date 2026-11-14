#!/bin/zsh
# A read-only export of a commit for the NB-FLIP port's references and timing rounds.
#   tools/nbflip-export.sh COMMIT DIR [PATCH]
# git archive COMMIT (lib, tools, package.json, tsconfig.json) into DIR, the main checkout's
# node_modules linked, an optional PATCH applied (a scratch perturbation, recorded), lib and
# tools made read-only, and DIR/PROVENANCE written: the commit, the lib/methods/uniform and
# lib/core tree hashes (equal hashes: the same NB-FLIP), and the patch's hash.
set -eu
MAIN=/Users/petersuggate/code/me/fluid
C=${1:?COMMIT}; D=${2:?DIR}; P=${3:-}
[ -e $D ] && { echo "$D exists" >&2; exit 2; }
cd $MAIN
full=$(git rev-parse --verify "$C^{commit}")
mkdir -p $D
git archive $full lib tools package.json tsconfig.json | tar -x -C $D
ln -s $MAIN/node_modules $D/node_modules
if [ -n "$P" ]; then (cd $D && patch -p1 < $P); fi
{
  echo "commit $(git rev-parse --short $full) ($(git log -1 --format=%cd --date=iso $full))"
  echo "lib/methods/uniform tree $(git rev-parse --short=10 --verify ${full}:lib/methods/uniform), lib/core tree $(git rev-parse --short=10 --verify ${full}:lib/core)"
  [ -n "$P" ] && echo "patched with $P (sha1 $(shasum $P | cut -c1-12)): NOT the commit"
  echo "exported $(date '+%Y-%m-%d %H:%M:%S')"
} > $D/PROVENANCE
chmod -R a-w $D/lib $D/tools
cat $D/PROVENANCE
