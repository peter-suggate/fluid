# Why Uniform Geometric pressure cost barely follows scene size

Investigation of `cm12-figure-9` versus `coarse-first-pool-impact-half-slab`,
2026-09-24. Production numerical code was not changed. The working tree already
contained unrelated changes, which were preserved. Source hashes, summaries and
per-frame diagnostic measurements are in the adjacent JSON.

The main finding is **repeated, over-accurate coarse solves plus a large minimum
cycle schedule**. It is not simply the number of fine liquid cells, and not the
historical paging regression that encoded all seven cycles unconditionally.

## Geometry and actual execution

| | Figure 9 | Half slab |
|---|---:|---:|
| Fine grid | 128×128×64 | 64×48×8 |
| Fine cells | 1,048,576 | 24,576 |
| Pressure levels | 6 | 6 |
| Coarsest physical grid | 4×4×2 | 2×3×2 |
| Coarsest rows including halo | 144 | 80 |
| Coarsest spacing, metres | 1.6×1.6×1.6 | 3.2×1.6×0.4 |
| Coarse invocations per Full-Cycle | 6 | 6 |
| Encoded pressure dispatches with one Full-Cycle | 554 | 464 |
| Always-encoded finish dispatches | 163 | 163 |

Figure 9 has 42.67 times as many cells, but the slab's long axes still require
six levels. `pressure-plan.ts` rejects the slab's lockstep terminal grid as
larger than the preferred 256 haloed rows, then independently halves each even
axis until it cannot halve further. Its hierarchy is:

```
64×48×8 → 32×24×4 → 16×12×2 → 8×6×2 → 4×3×2 → 2×3×2
```

The slab ends with an 8:1 spacing ratio (64:1 inverse-square coefficient ratio
before geometry). That anisotropy is a plausible contributor to the observed
higher iteration count; this investigation did not isolate it with an alternate
hierarchy or smoother. Smaller row count does not imply an easier linear system.

`fullCycle()` starts at the coarsest grid and invokes a nested V-cycle from every
successively finer level. With six levels it calls the coarsest solver six times
and visits nonterminal smoothing levels 1+2+3+4+5 = 15 times. Each visit has six
pre- and six post-sweeps. The cycle budget cannot stop partway through this work.
Small-level smoothing fusion already exists; it is not an absent optimization.

The production Geometric path sets `simultaneousSmoothing=true`: updates are
Jacobi snapshots, including the compensated coarse solve. The UI still says
PRBGS. Reverting to red/black ordering would lose the reflection symmetry fix
documented in `uniform-airborne-expansion-2026-09-23.md`.

## Measurements

All GPU runs were sequential under the repository lease with the browser Fluid
page unloaded. Dawn/Metal on this machine; default balanced method parameters,
fine residual tolerance 10 s^-1, one 1/30 s step per advance. The user's browser
link had a different fine tolerance (21.9331) and was paused at 2.4333 s; these are
controlled authored-scene runs, not a replay of that exact browser state.

First, production stage instrumentation over 30 frames, discarding frames 1–4:

| Median stage, ms | Figure 9 | Half slab |
|---|---:|---:|
| Full-Cycles | 11.73 | 16.45 |
| Topology + RHS | 2.16 | 1.38 |
| Finish | 2.75 | 0.92 |

Thus the exact equal 11 ms was not reproduced, but the important anomaly was:
the much smaller scene's cycle stage was actually slower. These numbers are
pressure stages, not total rendered frame times; transport/rendering have other
costs. Individual stage medians need not sum to the median total.

Next, per-dispatch timestamps and controlled ablations over 60 frames,
discarding frames 1–4. This instrumentation disables production pass batching;
**do not compare these totals directly against the preceding stage timings**.
The timestamp quantization toggle was disabled for all arms in this table.

| Scene / diagnostic arm | Pressure total median ms | Coarse kernels median ms | Median max coarse iterations per frame | Max fine residual s^-1 |
|---|---:|---:|---:|---:|
| Slab, production numerics | 15.77 | 8.16 | 134 | 0.589 |
| Slab, inner tolerance 0.1 | 10.62 | 0.62 | 13 | 0.494 |
| Slab, V-cycles only, strict inner tolerance | 7.06 | 2.66 | 120 | 1.723 |
| Slab, V-cycles only + inner tolerance 0.1 | 5.90 | 0.07 | 1 | 2.128 |
| Figure 9, production numerics | 18.33 | 2.17 | 31.5 | 12.386 |
| Figure 9, inner tolerance 0.1 | 16.58 | 0.30 | 5 | 14.880 |

“Max coarse iterations” is the existing diagnostic's maximum across invocations,
not the sum of all coarse work. All slab arms executed one cycle per sampled
frame, converged, and used no recovery. V-only means `pressureFullCycles=0`,
retaining four available V-cycles and the existing demand controller, not a hard
one-cycle cap. It reduced the slab from 464 to 303 encoded dispatches and from
six coarse invocations to one. No warm start was added in this experiment.

There were no uncaptured WebGPU errors in the completed measurement arms. These
are single diagnostic runs with changed trajectories, not an ABBA performance
study or a long-term physics acceptance gate. For example, the slab's final
volume sum differs by about 0.0042% between the strict and relaxed Full-Cycle
arms, and maximum speed also differs. Passing a residual check does not establish
trajectory equivalence or visual quality.

## Specific sources of wasted work

1. **Fixed inner accuracy.** `uniform-coarse-solver.wgsl.ts` solves every coarse
   invocation to 1e-4 s^-1, independently of the incoming correction and requested
   fine tolerance. This is 100,000 times tighter than the default fine threshold.
   Coarse and fine residuals belong to different operators, so that ratio alone
   is not a proof of waste; the ablation supplies the evidence. The slab spends
   most of its coarse time improving a correction much further than necessary.
2. **Expensive small-system implementation.** One 256-lane workgroup owns the
   entire coarse solve. Each iteration snapshots storage, performs compensated
   arithmetic, recomputes six coefficients for updates and residuals, executes
   barriers, and computes several diagnostic atomics plus the worst-row census.
   For the slab there are only 18 physical cells and 80 haloed rows. This is not
   a GPU throughput problem solvable by allocating more fine-cell workgroups.
3. **Full-Cycle as the cheapest normal option.** Pressure is zeroed in
   `mgBuildFinestRhs()` every step. Lagged scheduling defaults to at least one
   cycle and truncates a prefix whose first operation is a Full-Cycle. It cannot
   choose a V-cycle first or preserve a useful previous pressure iterate.
4. **Idle recovery still launches.** Eight batches of eight recovery sweeps,
   checkpoints and final validation are always encoded: 163 dispatches. Their
   guards prevent arithmetic when recovery is unnecessary, but not dispatch
   costs. In the production measurements this section cost 0.92 ms on the slab
   and 2.75 ms on Figure 9. It is material, but does not explain the slab's main
   coarse-solve bottleneck.
5. **Spatial work reduction is only part of the problem.** Finest-cycle tile
   lists and liquid-only smoothing lists already reduce active work. Many
   coarse operators remain dense. Removing air threads does not remove the
   fixed visits, repeated inner convergence, or launch dependencies.

## Accuracy caveat in the current demand controller

A cycle is accepted when its residual is finite and non-increasing. Meeting the
requested tolerance is a separate condition. An accepted but unconverged solve
can exhaust its lagged prefix and be published; the CPU raises the next frame's
budget. Recovery is entered for rejection, not merely for missing tolerance.

Figure 9's strict baseline exceeded the default fine tolerance at frames 22,
25 and 35 (maximum 12.386). The relaxed arm did so at 23, 25 and 27 (maximum
14.880). Neither arm rejected a cycle or needed recovery. Therefore the current
acceptance/recovery gate alone is **not** a same-frame tolerance guarantee.
Loosening inner solves globally without addressing this would be premature.

## Recommended algorithm direction

1. **Make the current fine residual the work controller.** Preserve the finite,
   non-increasing acceptance gate, but continue the same solve if the requested
   tolerance is unmet. A practical staged design submits a cheap initial solve,
   observes its compact residual result, and encodes more work only when needed.
   Measure the extra submission/readback latency, especially on small scenes.
   A GPU-only alternative requires fusion/persistent execution where workgroup
   synchronization is valid; a grid-wide barrier cannot be emulated safely by
   simply putting a whole multi-workgroup solver into one kernel.
2. **Try a V-cycle first and escalate.** Use a V-cycle, then further V-cycles or
   a Full-Cycle if measured residual reduction is poor. Cache a pressure initial
   guess in dedicated storage, reinitialize newly liquid/solid rows, clamp to
   current constraints, and check the current system's residual before trusting
   it. The slab V-only experiment already saves most of the unnecessary work
   without a warm start. Do not replace all scenes' defaults from this one test.
3. **Use an inexact coarse solve.** Choose inner tolerance from the incoming
   coarse residual/correction and outer accuracy requirement, using a relative
   reduction target plus a scale-aware absolute bound. Start cheaper, tighten
   and retry when fine-grid progress is insufficient. Retain an exact/strict
   fallback, sweep cap and failure reporting. The fixed 0.1 experiment is a
   sensitivity test, not the proposed universal tolerance.
4. **Optimize the small coarse system.** Precompute its coefficients/diagonal
   once per invocation or topology generation, keep eligible rows in workgroup
   storage, and collect expensive worst-row diagnostics only on failure or a QA
   request. Preserve compensated arithmetic and projected constraints. For
   stretched terminal grids, compare an anisotropy-aware block/line smoother or
   a small active-set direct solve; retain reflection invariance. Test changing
   the hierarchy termination policy separately rather than assuming fewer rows
   or fewer levels always wins.
5. **Encode recovery only after a current rejection**, as part of the same
   staged continuation design. Do not merely omit it based on the previous
   frame: impacts and edits invalidate that prediction. Fuse small coarse-level
   restriction/prolongation/constraint operations where dependencies allow it.

Expose per-level liquid/constraint row counts, encoded versus executed
launches, coarse invocation count and *sum* of iterations, fine residual
reduction per cycle, and whether publication met tolerance. Correct the PRBGS
label for the simultaneous Geometric path. These reveal actual numerical work
and explain why scene cell count alone is misleading.

Before promoting changes: compare both scenes through impact and late motion,
hydrostatics, odd/semi-coarsened grids, reflection symmetry, moving solids,
constraint activation and injected failure/recovery. Retain tolerance and timing
ceilings. The existing CPU pressure budget/hierarchy tests passed (3 checks);
no production solver change was made, so the large Sparse CM12 refactor gate was
not applicable.

## Reproduction

```
node --import tsx tools/profile-uniform-geometric-dawn.ts \
  --scene=coarse-first-pool-impact-half-slab --frames=30 --out=/tmp/slab-stages.json

FLUID_UNIFORM_MG_LEVEL_LABELS=1 node --import tsx \
  tools/probe-uniform-pressure-scaling-dawn.ts \
  --scene=coarse-first-pool-impact-half-slab --steps=60 --out=/tmp/slab-pressure
```

For diagnostic variants add `--coarse-tolerance=0.1`,
`--values='{"pressureFullCycles":0}'`, or both. Replace the scene with
`cm12-figure-9` for the larger case. Run serially, with browser simulation and
rendering unloaded. The probe changes shader source only inside its own process;
it does not modify application defaults.
