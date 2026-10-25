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
set -u
D=${FLUID_GPU_LOCK_DIR:-/private/tmp/claude-501}
mkdir -p "$D"
LOCK=$D/fluid-gpu.lock
PENDING=$D/fluid-gpu-timing.pending
GATE=$D/fluid-gpu-gate.pending
CHECKWAIT=$D/fluid-gpu-check.waiting
AGED=600
mode=${1:?check|timing}; shift
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
case "$mode" in
  timing)
    while aged_waiter; do sleep 5; done
    touch "$PENDING.$$"
    trap 'rm -f "$PENDING.$$"' EXIT
    # Exclusive lock first, then drain the three check slots.
    exec lockf -k "$LOCK" lockf -k "$D/fluid-gpu-check-1.lock" lockf -k "$D/fluid-gpu-check-2.lock" lockf -k "$D/fluid-gpu-check-3.lock" \
      /bin/bash -c 'rm -f "$0"; exec "$@"' "$PENDING.$$" "$@"
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
      if [ $((SECONDS-start)) -lt "$AGED" ] && ls "$PENDING".* >/dev/null 2>&1; then sleep 2; continue; fi
      slots="1 2 3"
      if [ "$mode" = check ] && ls "$GATE".* >/dev/null 2>&1; then slots="3"; fi
      if [ "$mode" = gate ] && ls "$CHECKWAIT".* >/dev/null 2>&1; then slots="1 2"; fi
      for i in $slots; do
        [ "$mode" = check ] && rm -f "$CHECKWAIT.$$"
        # -t 0: take slot i only if free. A timing run holds all three slots, so a slot
        # is never free while one runs; plain fluid-gpu.lock holders are not waited for.
        # fd 3 keeps the command's stderr; lockf's own "already locked" goes to /dev/null.
        lockf -k -t 0 "$D/fluid-gpu-check-$i.lock" /bin/bash -c '"$@" 2>&3; rc=$?; [ $rc -eq 75 ] && rc=1; exit $rc' -- "$@" 3>&2 2>/dev/null
        rc=$?
        [ $rc -ne 75 ] && exit $rc
        [ "$mode" = check ] && touch "$CHECKWAIT.$$"
      done
      sleep 2
    done
    ;;
  *) echo "usage: $0 check|gate|timing <cmd...>" >&2; exit 2 ;;
esac
