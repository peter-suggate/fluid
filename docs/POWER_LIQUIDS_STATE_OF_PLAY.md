# Power Liquids perf — state of play

*2026-08-03. Entry point for the frame-rate program. There are sixteen other
`POWER_LIQUIDS_*` docs; they are history. Read this one first.*

## The one number that matters

**The guide lane has not measurably moved.**

`symmetric-expansion` (32x16x32, 2,124 rows) measured ~274 ms/advance when we
adopted it as the guide lane and 269.47 ms/advance in the most recent control.
The A/A floor is **3.5 ms**. That difference is noise.

Everything else in this document exists to explain why, and what to do instead.

## The reframe

**On droplet-256 the cost was concentrated. On the guide lane it is diffuse.**

Droplet rewarded surgery: one `atomicLoad` was worth 7.55 ms, one hoisted word
4.69 ms. We learned an instinct there — hunt the concentrated defect — and we
have been applying it to a lane that does not have one.

Four ceilings priced on `symmetric-expansion` so far:

| target | ceiling | floor |
|---|---:|---:|
| seam-fold scratch, whole fold ablated | 3.06 ms | 3.65 ms |
| the march's entire 120-candidate scan | 3.84 ms | 3.65 ms |
| 36 post-convergence frontier waves | 0.22 ms | 3.65 ms |
| E2c sorting network, disabled | 0.95 ms | 3.52 ms |

**Four independent targets, every one at or under the noise floor.** That is not
four unlucky guesses; it is the signature of cost spread thinly across many
sites. The guide lane's 270 ms is broad per-row work, not a few defects.

Which means the remaining wins are **structural** — fewer rows, less work per
row, better occupancy, a different solve shape — and surgical per-item removal,
the technique that produced every win so far, is now near exhausted here.

## What worked

**Removing per-item work from the inside of a loop body.** Every large win came
from this and nothing else:

| change | delta |
|---|---:|
| deleted one `atomicLoad` in the grading closure | **-7.55 ms** |
| resolved one page-invariant word once per page instead of per cell | **-4.69 ms** |
| grading page fill | **-17.2%** |
| `betterFace` tie-chain: 4 redundant gathers -> 1 | **-3.88 ms** (pass-level) |
| MGPCG stencil columns published for the 6.3 A2 apply | -1.17 ms |

Cumulatively `power-droplet-256` went **151.63 -> 73.08 ms/advance (-51.8%)**.
Those commits are banked and stay.

**Off the frame, but real:** GPU initialization is ~53% of every benchmark arm
and nothing reported it. Batching independent pipeline compiles is **-63%** on
that phase, **-35%** on construction. That is wall-clock per experiment, which
is the program's actual throughput constraint.

## What didn't work

**Every parallelism-shaped fix.** This is the headline lesson, with three
independent confirmations:

- fanning split materialization across lanes — the entire premise of a v2
  design — measured **-0.35 ms**
- flattening the same loop to stride cells instead of pages cost **+6.05 ms**
- 36 additional, provably post-convergence frontier waves cost **-0.22 ms**

**Depth, launch count and parallelism were never the constraint. Per-item cost
inside loop bodies was.** Stride pages, never cells.

**Also dead, do not reopen:**

- *Seam-fold per-thread scratch.* Ablating the entire fold — more than any
  correct rewrite could recover — bought 3.06 ms against a 3.65 ms floor.
- *The grading fixpoint short-circuit.* Grading is **0.2%** of the guide frame.
- *A workgroup-local queue drained behind a barrier.* Hangs: one lane runs the
  serial walk while 63 block. Producer and consumer must never share a barrier.
- *Memoizing the owner-page directory.* The load was already the hottest line in
  the arena, so the memo removes a cache hit and charges two registers.
- *The march's 120-candidate closest-face scan.* Priced by running the scan
  twice: **3.84 ms** for one complete scan against a 3.65 ms floor. Deleting the
  loop outright would barely clear noise, so the 4x quadrant redundancy, a
  conservative early-rejection bound, and analytic quadrant selection are all
  sub-floor before they are written.
- *The march's "always-zero" launch-gate `atomicLoad`.* It pattern-matched the
  -7.55 ms defect and is not one: it is a live fail-closed latch with **113
  setters**. Deleting it makes a failed generation continue silently.

## Why the guide lane didn't move

**The 51.8% was the container's ranking, not the fluid's.** `droplet-256` pins
the fluid at ~100 live rows and sweeps the domain, so it ranks whatever scales
with the container. `symmetric-expansion` runs 2,124 rows and spends its time
**per row**. The old ranking is void as a priority list.

Where the guide frame actually is (~270 ms):

| pass | ms | share | status |
|---|---:|---:|---|
| persistent MGPCG (ONE workgroup) | 61.33 | 22% | **open** |
| March Section 5 changed frontier x2 | 39.68 | 15% | -3.88 landed; scan+waves excluded |
| Advect structured families | 40.28 | 15% | **open — largest uncontended pass** |
| Advect fine phi rare + common | 29.01 | 11% | untouched |
| Transfer accepted velocity | 11.80 | 4% | untouched |
| Octree resident grading closure | 0.437 | 0.2% | mined out |

Top six = 70% of the frame. The three big slices are **64% of it and
essentially untouched.** That is the plot.

## Promising leads

**1. The MGPCG one-workgroup ceiling is partly self-imposed.**
`maxComputeInvocationsPerWorkgroup` sat at the conservative WebGPU default of
256 because `requiredFluidDeviceLimits` never requested it — in a function whose
own comment says devices expose conservative defaults unless asked. The M1 Max
hosts 1024. Lane width is now selectable and 256 emits a byte-identical module.
*Owed: the 512/1024 A/B.* Temper it — this moves ~3% of the machine to ~12% at
best, still on one core cluster, and wider barriers may cost more than they buy.
A non-monotone curve is the useful outcome: it would prove the solve is
barrier-bound, which is what a multi-workgroup design hinges on.

**2. The march's unattributed ~36 ms.** The pass is 39.68 ms, and its two most
plausible targets are now both excluded — the ungated waves (-0.22 ms) and the
candidate scan (3.84 ms ceiling). What remains is frontier bookkeeping across
12 waves x 3 kernels x 2 encodes: queue append/compaction, `settledSeedFace` /
`appendFrontierDestination`, and per-invocation setup on ~135 workgroups a wave.
*That residue is unattributed, not proven diffuse* — worth one per-kernel
attribution run before anyone concludes either way.

**3. Compactness, at 32x16x32 where constant waste dominates** (~205 MiB
resident, ~24% of it lane-independent): the immutable power catalog is resident
**four times** (35.56 MiB pure duplication); air support falls back to a dense
whole-domain arena of 16,384 records against **1,716 live**; the SPGrid slot
arena is **97.8% empty**; and owner pages are a full permutation on every lane,
so every `ownerAt()` pays a second dependent load. Footprint is charged three
times over: memory, cache locality, and construction wall-clock per arm.

**4. Two cancellations are unproven, not dead.** E2b and E2d were both cancelled
on droplet geometry, where `wPartials = 1`. It is 6 on `fill-800` and **17** on
the guide lane — the premise doesn't hold off droplet. One phase-repeat run
settles both.

## Flag defaults — audited 2026-08-03

**Every flag with evidence behind it is already at its winning default.** There
is nothing to flip.

| flag | default | why |
|---|---|---|
| `FLUID_OCTREE_GRADING_PAGE_FILL` | **ON** | the -17.2% win |
| `FLUID_OCTREE_GRADING_SPLIT_HELPERS` | **ON** | banked |
| `FLUID_OCTREE_MGPCG_STENCIL_COLUMNS` | **ON** | -1.17 ms |
| `FLUID_OCTREE_MGPCG_RESTRICTED_PREFIX_SORT` | **ON** | flipped `19cee7d`, D4 green |
| `FLUID_GPU_PARALLEL_PIPELINE_COMPILE` | **ON** | -63% on the compile phase |
| `FLUID_OCTREE_GRADING_MEMBERSHIP_LOAD` | **OFF** | OFF *is* the -7.55 ms win; the flag only restores the deleted load so an A/B can price it |
| `FLUID_OCTREE_GRADING_FIXPOINT` | **OFF** | does not boot, and grading is 0.2% of this frame — retire the flag, don't fix it |
| `FLUID_OCTREE_MGPCG_LANES` | **256** | no measurement yet; 256 emits a byte-identical module, so leave it until the A/B runs |

Diagnostics (`FLUID_GPU_INIT_CENSUS`, `FLUID_GPU_MEMORY_CENSUS`,
`FLUID_TRIPWIRES`) stay off; only `FLUID_TRIPWIRES=failfast` costs the ~27%, so
never compare a wall across tripwire modes.

## Measurement discipline that this program had to learn

- **Always quote the A/A floor next to the delta.** It is 3.5 ms on the guide
  lane and 0.27 ms on droplet-256. A 1 ms claim here is noise; the same claim
  there is real.
- **Ablate more than any correct fix could recover.** If that still doesn't
  clear the floor, stop — you have priced the ceiling without writing the fix.
  This killed the seam fold in one run.
- **Never ablate inside an iteration-to-fixed-point.** Short-circuiting the
  march's closest-face scan to 1/120th of the work made the pass **2x slower**
  and moved a neighbouring pass by +8.6 ms: a wrong winner stops the frontier
  converging, so the ablation prices its own divergence instead of the loop.
  Use a **result-preserving double execution** instead — run the loop twice and
  take the difference. It is valid wherever the loop is a deterministic function
  of unchanged inputs, and tripping zero tripwires is the evidence it was.
- **Check isolation on every ablation**, not just the headline pass. That check
  is exactly what separated the valid seam-fold measurement from the invalid
  march one.
- **A wrong cancellation is worse than a wrong finding**, because nobody
  revisits it. Say which lane a null was measured on.
- **Never compare cold construction against warm.** A cold Metal function cache
  reads 5.9-12.2 s against 1.0-1.6 s on the same arm.
- Gate on `symmetric-expansion`: D4 window is step **68** (volume/velocity/
  pressure/rhs) and **69** (diagonal/topology); the run exits rc=1 by
  construction.
- `benchmark-power-dam.ts` **swallows child stdout** it does not re-print. Any
  new child-emitted record needs an explicit forwarding branch.
