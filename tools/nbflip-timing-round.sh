#!/bin/zsh
# The NB-FLIP port's timing round (docs/plans/nbflip-port-2026-11-15.md §3 lane G, §4),
# generalised from the seep rewrite's lane H timing-round.sh to every gate scene.
#
#   tools/nbflip-timing-round.sh OUT BASE_TREE [CANDIDATE_TREE] [SCENES]
#
# BASE_TREE and CANDIDATE_TREE are read-only exports made by tools/nbflip-export.sh (each holds a
# PROVENANCE file), or the main checkout when it is clean under lib/ and tools/. SCENES defaults
# to "seep dam letters paddleboats". For each scene, back to back on the timing slot
# (tools/gpu/fluid-gpu-run.sh timing: exclusive GPU): baseline fenced, candidate fenced, baseline
# pipelined, candidate pipelined. Each run starts only after the GPU has been under 15% device
# utilisation for 5 consecutive seconds (600 s at most). A 1 s sampler logs utilisation and any
# other harness or node GPU job (OUT/gpu-util.log, OUT/contention.log). After the round the
# system log is searched for Metal command-buffer aborts and GPU restarts in the round's window
# (OUT/metal.log); the runner's own guard already fails a run whose processes saw one (rc 77).
# rc 76, 77 or 78 marks a run INVALID: reset and rerun it, never use it.
# Runs land in OUT/base and OUT/candidate; the §5-style table per scene is printed by
#   node --import tsx tools/nbflip-gates.ts --timing-only --ref-timing=OUT/base [--candidate-timing=OUT/candidate]
# and written to OUT/timing.md.
set -u
MAIN=/Users/petersuggate/code/me/fluid
R=$MAIN/tools/gpu/fluid-gpu-run.sh
O=${1:?OUT}; BASE=${2:?BASE_TREE}; CAND=${3:-}; SCENES=(${=${4:-seep dam letters paddleboats}})
# Every run starts inside its tree, so OUT and the trees must be absolute.
O=${O:A}; BASE=${BASE:A}; [ -n "$CAND" ] && CAND=${CAND:A}
mkdir -p $O/base; [ -n "$CAND" ] && mkdir -p $O/candidate
util() { ioreg -r -c IOAccelerator -d 1 2>/dev/null | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2; }
provenance() { # tree label
  local t=$1
  if [ -f $t/PROVENANCE ]; then echo "$2: $(tr '\n' ' ' < $t/PROVENANCE)"
  elif [ "$(cd $t && git rev-parse --show-toplevel 2>/dev/null)" = "$t" ]; then
    local d=$(cd $t && git status --porcelain -- lib tools)
    [ -n "$d" ] && { echo "$2: $t is a checkout with changes under lib/ or tools/: time a read-only export (tools/nbflip-export.sh)" >&2; exit 2; }
    echo "$2: checkout $t at $(cd $t && git rev-parse --short HEAD), clean under lib/ and tools/"
  else echo "$2: $t has no PROVENANCE file" >&2; exit 2; fi
}
{ echo "round start $(date '+%Y-%m-%d %H:%M:%S')"; provenance $BASE base; [ -n "$CAND" ] && provenance $CAND candidate; echo "main checkout HEAD $(cd $MAIN && git rev-parse --short HEAD), dirty lib/tools: $(cd $MAIN && git status --porcelain -- lib tools | wc -l | tr -d ' ') files"; } | tee $O/provenance.log
START=$(date '+%Y-%m-%d %H:%M:%S')
( while true; do echo "$(date +%H:%M:%S) util=$(util)" >> $O/gpu-util.log; ps -axo pid,etime,command | grep -E "webgpu|node .*(probe|bench|test|advance|slide|attribution|profile)" | grep -v -E "grep|ps -axo|nbflip-timing-round|$O" | sed "s/^/$(date +%H:%M:%S) /" >> $O/contention.log; sleep 1; done ) &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT
run() { # tree role scene mode
  local t=$1 role=$2 s=$3 m=$4 quiet=0 waited=0
  while [ $quiet -lt 5 ]; do u=$(util); if [ "${u:-100}" -lt 15 ]; then quiet=$((quiet+1)); else quiet=0; fi; sleep 1; waited=$((waited+1)); [ $waited -gt 600 ] && { echo "gave up waiting for a quiet GPU"; exit 2; }; done
  local name=$s-timing-$m out=$O/$role
  echo "$role $s $m start $(date +%H:%M:%S) (quiet after ${waited}s)"
  if [ $s = seep ]; then
    (cd $t && WEBGPU_NODE_MODULE=$t/node_modules/webgpu/index.js $R timing node --import tsx tools/seep-bench.ts --method=nbflip --steps=1200 --every=60 --mode=$m --slot=timing --out=$out --name=$name > $out/$name.log 2>&1)
  else
    # The dam is timed over 165 steps (nbflip-gates.ts TIMED_STEPS): a baseline from before lane D's fix fails a residency audit near step 178.
    local steps=(); [ $s = dam ] && steps=(--steps=165)
    (cd $t && WEBGPU_NODE_MODULE=$t/node_modules/webgpu/index.js $R timing node --import tsx tools/nbflip-gate-bench.ts --scene=$s $steps --mode=$m --slot=timing --out=$out --name=$name > $out/$name.log 2>&1)
  fi
  local rc=$?
  case $rc in 76|77|78) echo "$role $s $m rc=$rc INVALID (runner guard): reset and rerun" ;; *) echo "$role $s $m rc=$rc end $(date +%H:%M:%S)" ;; esac
  grep "^fluid-gpu-run" $out/$name.log | sed "s/^/  /"
}
for s in $SCENES; do
  run $BASE base $s fenced; [ -n "$CAND" ] && run $CAND candidate $s fenced
  run $BASE base $s pipelined; [ -n "$CAND" ] && run $CAND candidate $s pipelined
done 2>&1 | tee $O/round.log
kill $MON 2>/dev/null
/usr/bin/log show --start "$START" --style compact --predicate 'eventMessage CONTAINS[c] "command buffer was aborted" OR eventMessage CONTAINS[c] "GPU Restart" OR eventMessage CONTAINS[c] "kIOGPUCommandBufferCallbackError"' 2>/dev/null | grep -v -E "^Timestamp|log run noninteractively" > $O/metal.log
echo "Metal aborts / GPU restarts in the round's window: $(wc -l < $O/metal.log | tr -d ' ') (OUT/metal.log)" | tee -a $O/round.log
echo "round end $(date '+%Y-%m-%d %H:%M:%S')" | tee -a $O/round.log
(cd $MAIN && node --import tsx tools/nbflip-gates.ts --timing-only --ref-timing=$O/base ${CAND:+--candidate-timing=$O/candidate} --out=$O --name=timing) | tee -a $O/round.log
