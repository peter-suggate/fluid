# Losasso topology growth — implementation hand-off

2026-08-05. Context: "fix the front dam wall advancement in symmetric expansion
scene" on the Dawn lane. The front is fixed; the lane is not yet green
end-to-end. This documents what landed, the forensics for the remaining red,
and the structural questions the losasso lane needs answered.

## What landed (uncommitted, this working tree)

1. **`losassoCoarseArenaAuthority()`** (lib/webgpu-octree.ts, WGSL near
   `coarseDirectoryAuthority`) OR'd into `liquidAuthorityAvailable()`.
   Root cause of the frozen front: the frontier finalize's availability
   predicate only recognized the Power corrected-coarse directory header
   (word0 `0x80000000`, generation == `pressureCapacity.w>>2`, which only
   `powerCoarseLevelSetSchedule` writes). On losasso, binding 15 is the
   coarse-phi **arena** (magic `0x4C504849`), so availability was false on
   every recurring advance. Zero-addition generations take the FRONTIER_REUSE
   path that *skips validation*, so the run looked healthy until the collapse
   front first needed row additions (~gen 44, t=0.17) — then every candidate
   was rejected, forever. Wet rows/faces stayed the t=0 set (1152/3872 for
   240 generations), the free surface had no faces ⇒ zero normal velocity ⇒
   sealed-box recirculation ratcheting to 7 m/s under a frozen interface.
2. **Extension-band membership probe clamp**
   (lib/webgpu-octree-losasso-extension-band.wgsl.ts `finePhiCells`): face
   centres in the outermost half cell of the domain have no in-lattice
   trilinear support; the probe returned invalid, corner faces were never
   published, and the transport hard-failed (control[7]=64, brick key 16383 =
   the lid/wall corner) as soon as the corner splash entered the band
   (t≈0.36). Clamped to the sample-centre lattice.
3. **`correctedCoarsePhi()` losasso branch** (lib/webgpu-octree.ts): reads the
   arena's restricted row phi (same lookup as `sampleCoarseOctreePhi`) so
   `liquidOwner`/`phi()` have a coarse backstop when a fine summary is
   missing. Correct in intent; currently **unreliable** — see arena ping-pong
   below.

Validated on Dawn/Metal: front reaches i=30/31 by t=0.2 (Martin–Moyce pace,
walls contacted ~t=0.22–0.25), rows grow 1152→2164, exact D4 symmetry holds at
every checkpoint that runs, transport clean through the old corner failure.

Diagnostics added (keep): `losassoFrontierDebug` getter + executor readbacks
`fluid-symmetry-losasso-frontier` (terminal) and `…-frontier-at-failure`
(printed when a checkpoint QA readback throws); transport `control[13]`
latches the first failing brick key; `FLUID_SPEED_MAP=1` now prints a
bottom-layer +x centerline (i, alpha, vx) — the front trajectory at a glance.

Repro lane (authored 250-step lane exceeds its own 240 s cap at ~1.3 s/step —
pre-existing; use 0.008 s checkpoints):

    FLUID_SCENE=symmetric-expansion FLUID_METHOD=octree FLUID_TARGET_S=0.4
    FLUID_ORACLE_STEPS=100 FLUID_CHECKPOINT_EVERY_S=0.008 FLUID_SPEED_MAP=1
    … (rest as package.json test:webgpu:symmetric-expansion)

## The remaining red: gen-91 all-dry retirement (t≈0.36, ±2 gens)

Sequence, established with per-step checkpoints and the at-failure dump:

- Residency shrink (16384 → ~15.8k fine pages) runs from ~gen 80 with healthy
  accepts and live summaries — **not** the trigger.
- Every readback through gen 90 is valid (rows 2152, arena magic, solver
  converged).
- Gen 91's candidate is built **cleanly empty**: 2144 previous rows, carried
  0, candidates 0, retired 2144, no failure flags (dirtyFailure `0x200`,
  reason 0). `required == previous + added − retired` holds, so the finalize
  accepts it. One `currentPressureOwnerWet` pass said "dry" for every owner.
- A zero-row topology is terminal by design (dirty marking only visits active
  tiles). The arena then rebuilds empty/non-magic, gen 92 rejects on
  availability, solver reads 0 rows nonfinite, fine volume logs ~890k owner
  lookup failures, QA trips "coarse publication is not valid".

Leads, in fix order:

1. **Arena ping-pong vs frozen bind groups.**
   `WebGPUOctreeLosassoCoarsePhiExchange.encode()` swaps `source.arena`
   between two buffers *every encode* (coarse-phi.ts:176-178), and the
   exchange encodes 2–3× per step (ready-flip reapplication + settle), while
   `refreshLosassoProjectionGroups()` snapshots the buffer only at ready-flip
   time. Binding 15 for the frontier classify/finalize can therefore be one
   arena stale or mid-clear. This is why fix 3 didn't change the outcome, and
   it makes fix 1's availability check flappable. Stop mutating
   `source.arena` (publish into a stable buffer) or rebind after every flip.
2. **Then re-observe the all-dry pass.** With (1) fixed, if gen 91 still
   retires everything, the summaries are returning found=true-but-dry (or
   found=false) for every owner in that one pass — probe
   `fluid-symmetry-losasso-fine-summary` at the exact generation; suspect the
   summary rebuild against the just-shrunk residency.

## Fundamental issues with the losasso lane to think about

These are design-level, not bugs; each contributed to how long the lane could
be silently wrong.

- **Wetness has a single authority with no backstop.** On the Power path,
  classification degrades to the corrected-coarse directory when summaries
  gap. On losasso, everything rests on the fine summaries; any one-generation
  hiccup classifies the world dry, and the system *validly* accepts the
  result. Retiring 100% of rows in one step should be structurally
  impossible or loudly fatal — an "empty-accept" tripwire (retired ==
  previous && previous > 0 ⇒ reject/fatal) is cheap and would have turned
  this from a silent death into a one-line diagnosis.
- **Fail-closed composes into deadlock.** Rejection-with-carry is sound per
  step but has two terminal absorbing states we have now hit in practice:
  the availability-reject retry loop (frozen graph, this fix) and the
  zero-row topology. The dry-identity memory documents the same family on the
  Power large lanes (rollback deadlock). The lane needs an explicit recovery
  policy: bounded retries then fatal, never silent carry forever. The
  attempted-vs-accepted generation gap (`frontier[8]` − `frontier[3]`) is the
  one-word health signal; it should be a standing tripwire, not a debug
  readback.
- **The sparse velocity coverage contract is implicit.** Semi-Lagrangian phi
  transport requires transport band ⊆ extension band *after* one step of
  surface motion, including at domain edges/corners. Today that holds by
  coincidence of constants (band widths, CFL, dilate count) and broke at the
  first corner contact. Worth asserting: any in-band sample whose own
  position lacks velocity coverage is a contract violation with a named
  counter (now latched via control[13]), and the band plan should derive the
  extension width from transport band + per-step CFL margin rather than a
  free constant.
- **Row-mean phi is a weak wetness/conditioning signal.** The arena stores
  volume-averaged restricted phi per row; half-wet front rows sit near zero
  and flip classification; the ghost conditioning's interior branch skips
  faces whose row-mean is non-negative (θ floors also differ: 1e-2 at walls
  vs 1e-4 interior). The first-order coupling is fine (see the convergence
  memory), but decisions keyed on row-mean phi near the front deserve an
  interval (min/max) rather than a mean — the arena entry layout has room.
- **The lane still under-observes.** The convergence memory's point 5 stands:
  no per-step tripwires on losasso (force-disabled for the cutover lane), the
  renderer retains rejected generations, and the authored 250-step gate
  cannot finish inside its own timeout at current readback cost, so the full
  D4 + wall-contact verdict has not actually been re-blessed post-cutover.
- **Nondeterminism** (±2 generations on the death, known lane property) means
  bisection needs signature-based comparison, not step-exact repro.

Memory: `losasso-symmetric-expansion-convergence` holds the full forensic
narrative (items 1–8). Scratchpad logs from the diagnosis session:
`symx-{fixed50,100b,100c,100d,88,91,92,250}.log`.
