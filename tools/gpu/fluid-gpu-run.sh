#!/bin/bash
# Shared GPU runner for the fluid repo's agents.
#   fluid-gpu-run.sh check  <cmd...>   correctness-only run (hashes, fatals, analytic scoring):
#                                       up to 3 at once, never alongside an exclusive (timing) run
#   fluid-gpu-run.sh gate   <cmd...>   merge-gate check: like check, but ordinary checks wait while a gate is queued
#   fluid-gpu-run.sh timing <cmd...>   timing run: exclusive. Takes fluid-gpu.lock and all 3 check slots.
# FLUID_GPU_LOCK_DIR overrides the lock directory; every agent on the machine must use the same one.
# Peer sessions that call lockf on fluid-gpu.lock directly still serialize against timing runs.
# Fairness: a check or gate that has waited AGED seconds stops yielding to queued timing runs,
# and a new timing run holds off (without queueing) while any such aged waiter exists.
# A timing command still only starts once it holds all 3 slots, so timing stays exclusive.
# Every command runs under fluid-gpu-guard.sh inside its slot: a GPU canary first (rc 76 when
# the GPU executes nothing), then a system-log check for Metal command-buffer aborts in the
# command's process tree (rc 77: Dawn reports none of them, so the results are invalid),
# and the GPU memory in use at start and at peak.
# Admission: a timing run is alone on the GPU; checks and gates run up to 3 at once, and each
# command may itself start several GPU processes (node --test runs test files in parallel).
set -u
D=${FLUID_GPU_LOCK_DIR:-/private/tmp/claude-501}
mkdir -p "$D"
LOCK=$D/fluid-gpu.lock
PENDING=$D/fluid-gpu-timing.pending
GATE=$D/fluid-gpu-gate.pending
CHECKWAIT=$D/fluid-gpu-check.waiting
AGED=600
GUARD=$(cd "$(dirname "$0")" && pwd)/fluid-gpu-guard.sh
mode=${1:?check|timing}; shift
# Metal shader cache, one directory per slot (lib/harness/metal-shader-cache.ts).
# A process reads and writes only the lowest-numbered shard of its directory that
# no other process holds, so harness processes compiling at once on the 3 slots
# filled shards 0-2 of one shared directory and missed what the slot before them
# stored. With one directory per slot, every run takes shard 0 of its slot's own.
# Metal caps a shard (about 1.1 GB of functions and 1.1 GB of libraries), so
# each directory stays near 2.3 GB. A timing run, which holds every slot, uses
# slot 1's. A FLUID_METAL_SHADER_CACHE the caller sets is kept as given.
slot_cache() {
  [ -n "${FLUID_METAL_SHADER_CACHE:-}" ] && { printf '%s' "$FLUID_METAL_SHADER_CACHE"; return; }
  local root
  root=$(getconf DARWIN_USER_CACHE_DIR 2>/dev/null) || return 0
  [ -n "$root" ] && printf '%s' "${root%/}/fluid-headless/metal-shader-cache-slot-$1"
}
# aged_waiter: true if some live queued check or gate (old or new runner) has waited AGED seconds.
aged_waiter() {
  local f p t
  for f in "$CHECKWAIT".* "$GATE".*; do
    [ -e "$f" ] || continue
    p=${f##*.}
    t=$(ps -o etime= -p "$p" 2>/dev/null | tr -d ' ') || continue
    [ -n "$t" ] || continue
    # etime is [[dd-]hh:]mm:ss
    t=$(echo "$t" | awk -F'[-:]' '{s=0; for(i=1;i<=NF;i++) s=s*60+$i; if(NF==4) s=$1*86400+$2*3600+$3*60+$4; print s}')
    [ "$t" -ge "$AGED" ] && return 0
  done
  return 1
}
# timing_waiting: true if a live timing run is queued. A timing run's pending file
# outlives it when it is killed while queued (its EXIT trap is gone once it execs
# lockf), so a file whose pid is gone is stale: remove it rather than yield to it.
timing_waiting() {
  local f live=1
  for f in "$PENDING".*; do
    [ -e "$f" ] || continue
    if kill -0 "${f##*.}" 2>/dev/null; then live=0; else rm -f "$f"; fi
  done
  return $live
}
case "$mode" in
  timing)
    while aged_waiter; do sleep 5; done
    touch "$PENDING.$$"
    trap 'rm -f "$PENDING.$$"' EXIT
    # Exclusive lock first, then drain the three check slots.
    cache=$(slot_cache 1); [ -n "$cache" ] && export FLUID_METAL_SHADER_CACHE="$cache"
    exec lockf -k "$LOCK" lockf -k "$D/fluid-gpu-check-1.lock" lockf -k "$D/fluid-gpu-check-2.lock" lockf -k "$D/fluid-gpu-check-3.lock" \
      /bin/bash -c 'rm -f "$0"; exec "$@"' "$PENDING.$$" "$GUARD" "$@"
    ;;
  check|gate)
    # gate: a merge-gate check, ahead of ordinary checks on slots 1-2.
    # Slot 3 goes to waiting ordinary checks first, so gates can't starve them.
    if [ "$mode" = gate ]; then
      touch "$GATE.$$"
      trap 'rm -f "$GATE.$$"' EXIT
    else
      touch "$CHECKWAIT.$$"
      trap 'rm -f "$CHECKWAIT.$$"' EXIT
    fi
    start=$SECONDS
    while :; do
      # Yield to a waiting timing run, unless this run has waited AGED seconds.
      if [ $((SECONDS-start)) -lt "$AGED" ] && timing_waiting; then sleep 2; continue; fi
      slots="1 2 3"
      if [ "$mode" = check ] && ls "$GATE".* >/dev/null 2>&1; then slots="3"; fi
      if [ "$mode" = gate ] && ls "$CHECKWAIT".* >/dev/null 2>&1; then slots="1 2"; fi
      for i in $slots; do
        [ "$mode" = check ] && rm -f "$CHECKWAIT.$$"
        # -t 0: take slot i only if free. A timing run holds all three slots, so a slot
        # is never free while one runs; plain fluid-gpu.lock holders are not waited for.
        # fd 3 keeps the command's stderr; lockf's own "already locked" goes to /dev/null.
        cache=$(slot_cache "$i")
        FLUID_METAL_SHADER_CACHE=${cache:-${FLUID_METAL_SHADER_CACHE:-}} lockf -k -t 0 "$D/fluid-gpu-check-$i.lock" /bin/bash -c '"$@" 2>&3; rc=$?; [ $rc -eq 75 ] && rc=1; exit $rc' -- "$GUARD" "$@" 3>&2 2>/dev/null
        rc=$?
        [ $rc -ne 75 ] && exit $rc
        [ "$mode" = check ] && touch "$CHECKWAIT.$$"
      done
      sleep 2
    done
    ;;
  *) echo "usage: $0 check|gate|timing <cmd...>" >&2; exit 2 ;;
esac
