#!/bin/bash
# Merge-gate bundle for the NB-FLIP solver: splash p20 70 frames x2, seep 60 x3,
# letters 60 x3, scored by tools/gate-repeat-spread.ts (windowed repeat noise,
# 1.66 x A/A spread). The solver is nondeterministic by design, so the gate is
# "within repeat noise", never bit identity.
#
#   tools/gpu/gate-bundle.sh NAME BASE_REV CAND_REV [extra gate-repeat-spread args...]
#
# BASE_REV and CAND_REV are git revisions of this repo; each is exported with
# git archive into $FLUID_GATE_DIR/snap-<sha> (node_modules symlinked from this
# checkout) so the gate never runs against a stale tree. Results land in
# $FLUID_GATE_DIR/gates/NAME-*.{log,json}. Extra args go to every scene, e.g.
# --set=volumeControlSeconds:0 or --differ=crowdedCells for a default flip.
# GPU runs go through tools/gpu/fluid-gpu-run.sh (splash as a gate, the
# registered scenes as timing so they never share the GPU).
set -u
name=${1:?NAME}; base=${2:?BASE_REV}; cand=${3:?CAND_REV}; shift 3
repo=$(cd "$(dirname "$0")/../.." && pwd)
dir=${FLUID_GATE_DIR:-/private/tmp/fluid-gates}
node=${FLUID_NODE:-$(command -v node)}
runner="$repo/tools/gpu/fluid-gpu-run.sh"
mkdir -p "$dir/gates"
snap() {
  local sha; sha=$(git -C "$repo" rev-parse --short=8 "$1") || exit 1
  local d="$dir/snap-$sha"
  if [ ! -d "$d" ]; then
    mkdir -p "$d.tmp" && git -C "$repo" archive "$sha" | tar -x -C "$d.tmp" && ln -s "$repo/node_modules" "$d.tmp/node_modules" && mv "$d.tmp" "$d"
  fi
  echo "$d"
}
B=$(snap "$base"); C=$(snap "$cand")
echo "baseline $B"; echo "candidate $C"
run() { local scene=$1 mode=$2; shift 2
  local log="$dir/gates/$name-$scene.log"
  (cd "$C" && env WEBGPU_NODE_MODULE="$repo/node_modules/webgpu/index.js" "$node" --import tsx tools/gate-repeat-spread.ts "$name-$scene" \
    --baseline="$B" --candidate="$C" --out="$dir/gates" --gate="$runner $mode" "$@") > "$log" 2>&1
  local rc=$? verdict
  verdict=$(grep -oE '"phase":"complete","pass":(true|false)' "$log" | tail -1 | grep -oE '"pass":(true|false)')
  echo "$scene rc=$rc $verdict log=$log"
  [ $rc -eq 0 ] && [ "$verdict" = '"pass":true' ]
}
fail=0
run splash gate --scene=splash --length=128 --pool-blocks=20 --steps=70 --repeats=2 --stats-every=10 "$@" || fail=1
run seep timing --scene=reservoir-seep --steps=60 --repeats=3 --stats-every=10 "$@" || fail=1
run letters timing --scene=nbflip-figure-8-letters --steps=60 --repeats=3 --stats-every=10 "$@" || fail=1
echo "bundle $name: $([ $fail = 0 ] && echo PASS || echo FAIL)"
exit $fail
