#!/bin/bash
# fluid-gpu-guard.sh <cmd...>: runs one GPU command inside a slot held by
# fluid-gpu-run.sh. After a Metal out-of-memory fault, Metal can abort command
# buffers (every one, or one per step) while Dawn reports success, so a run
# can finish "healthy" having computed nothing or read back stale data
# (2026-11-07: frozen cloth at 11:11 and 11:58, then every process from 12:12).
#   1. Canary: one small kernel with a checked readback (gpu-canary.mjs) before
#      the command runs. A dead GPU fails in under a second with rc 76.
#   2. Abort check: when the command exits, the system log is searched for Metal
#      command-buffer aborts in the command's process tree (the child and every
#      descendant seen while it ran, polled every 0.5 s; a descendant that lives
#      shorter than that can be missed). Any match fails the run with rc 77,
#      whatever the command returned. Aborts in other processes over the same
#      window are reported as a warning but don't fail the run. If the log
#      can't be read (log show fails, or prints no header, as in a sandboxed
#      shell), the run fails with rc 78: an unread log is not "no aborts".
#   3. GPU memory: the system's "In use system memory" (IOAccelerator) at start
#      and its peak over the run, reported on stderr. It's machine-wide, so it
#      includes other slots' processes.
# FLUID_GPU_GUARD=0 runs the command bare; FLUID_GPU_CANARY=0 skips only the canary;
# FLUID_GPU_LOG replaces /usr/bin/log (tests of the rc 78 path).
set -u
[ "${FLUID_GPU_GUARD:-1}" = 0 ] && exec "$@"
here=$(cd "$(dirname "$0")" && pwd)
say() { echo "fluid-gpu-run: $*" >&2; }
gpu_bytes() { ioreg -r -d 1 -c IOAccelerator 2>/dev/null | sed -n 's/.*"In use system memory"=\([0-9]*\).*/\1/p' | head -1; }
gb() { awk -v b="${1:-0}" 'BEGIN{printf "%.2f GB", b/1073741824}'; }

if [ "${FLUID_GPU_CANARY:-1}" != 0 ] && ! node "$here/gpu-canary.mjs"; then
  say "GPU canary failed before the command ran: the GPU is not executing work (check /usr/bin/log for 'Execution of the command buffer was aborted'; a GPU reset may be needed)"
  exit 76
fi

start=$(date '+%Y-%m-%d %H:%M:%S')
pids=$(mktemp "${TMPDIR:-/tmp}/fluid-gpu-guard.XXXXXX")
trap 'rm -f "$pids"' EXIT
"$@" <&0 &
child=$!
# A signal goes to the child and every descendant seen so far, so a killed
# node --test leaves no GPU processes behind.
trap 'kill -TERM $(sort -u "$pids") 2>/dev/null' TERM INT HUP
echo "$child" >"$pids"
mem0=$(gpu_bytes); peak=${mem0:-0}
while kill -0 "$child" 2>/dev/null; do
  # Every live descendant of the child (BFS over one ps snapshot).
  ps -Ao pid=,ppid= | awk -v root="$child" '{p[NR]=$1; q[NR]=$2} END{keep[root]=1; do {n=0; for(i=1;i<=NR;i++) if(!(p[i] in keep) && (q[i] in keep)) {keep[p[i]]=1; n++}} while(n); for(k in keep) print k}' >>"$pids"
  m=$(gpu_bytes); [ -n "$m" ] && [ "$m" -gt "$peak" ] && peak=$m
  sleep 0.5
done
wait "$child"; rc=$?
trap - TERM INT HUP

sleep 1 # let the log catch up
logged=$("${FLUID_GPU_LOG:-/usr/bin/log}" show --start "$start" --style compact --predicate 'eventMessage CONTAINS "Execution of the command buffer was aborted"' 2>&1); logrc=$?
if [ $logrc -ne 0 ] || ! printf '%s\n' "$logged" | head -1 | grep -q '^Timestamp'; then
  say "cannot verify Metal aborts: /usr/bin/log show failed (rc $logrc): $(printf '%s' "$logged" | head -2 | tr '\n' ' ') (command rc $rc; FLUID_GPU_GUARD=0 skips the guard)"
  exit 78
fi
aborts=$(printf '%s\n' "$logged" | sed -n 's/^.*[[:space:]][^[:space:]]*\[\([0-9][0-9]*\):[0-9a-f]*\] (Metal) Execution of the command buffer was aborted.*$/\1/p' | sort | uniq -c)
mine=0; others=0
while read -r count pid; do
  [ -n "${pid:-}" ] || continue
  if sort -u "$pids" | grep -qx "$pid"; then mine=$((mine+count)); else others=$((others+count)); fi
done <<<"$aborts"
say "GPU memory in use $(gb "$mem0") at start, peak $(gb "$peak"); $(sort -u "$pids" | wc -l | tr -d ' ') process(es) in the tree"
[ "$others" -gt 0 ] && say "warning: $others Metal command-buffer abort(s) in other processes during this run"
if [ "$mine" -gt 0 ]; then
  say "$mine Metal command-buffer abort(s) in this run's processes: its results are invalid (command rc $rc)"
  exit 77
fi
exit "$rc"
