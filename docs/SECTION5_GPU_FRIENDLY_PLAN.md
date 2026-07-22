# GPU-Friendly Section 5: Restructuring Plan

Status: implemented through Phase 5 (Phase 6 deferred by its evidence gate)
Paper: Aanjaneya, Gao, Liu, Batty, Sifakis 2017, *Power Diagrams and Sparse
Paged Grids for High Resolution Adaptive Liquids*, Section 5.
Scope: the surface pipeline only — fine narrow-band level set (topology,
advection, redistance), coarse companion level set, face-band velocity
reconstruction/extrapolation, and regular-face-to-power-face publication.
The Section 4 variational power-diagram pressure discretization is out of
scope and unchanged.

## Implementation record (2026-07-22)

Phases 0–5 are now the product path: dedicated Section 5 timestamps use the
existing asynchronous query readback; topology pre-allocates the complete
support with a logarithmic Chebyshev flood; JFA-CPT is the default fixed-pass
redistance; face velocity propagation uses parallel CPT/BFS rather than the
single-lane heap; seed and adaptive-owner scans are parallel; support captures
write their next indirect arguments directly; coarse-phi parameters use an
encoder-local arena; and startup publishes the initial sparse authority in one
submission/fence by default. `?safeBringup=1`, `?gpu=safe`, or
`FLUID_SAFE_BRINGUP=1` retains the eight fenced diagnostic checkpoints.
`FLUID_FINE_REDISTANCE=fmm` selects the fixed-resident diagnostic oracle only
for at most 256 logical samples; larger requests fail before encoding so an
accidental production opt-in cannot monopolize a single backend submission.

The post-implementation audit also tightened four contracts that the initial
landing did not satisfy:

- band-phi relaxation is again bounded solely by band width (12 rounds for
  B4); CPT graph depth has a separate domain-derived bound used only to plan
  9–10 logarithmic pointer-jump rounds on the measured grids;
- the CPT jump kernel is now constructed and dispatched between graph linking
  and constant-time resolution, using ping-pong parent snapshots so no jump
  reads a value another workgroup is mutating;
- JFA materializes each seed's axial direction and 24-bit subcell fraction in
  otherwise-unused sample-flag bits, propagates resident sample indices, keeps
  the current winner distance locally, and evaluates each stride candidate
  once. Floods therefore do no seed hash lookup or six-neighbor phi walk and
  require no new buffer, binding, dispatch, or accounted byte;
- incomplete or out-of-order shared timestamp boundaries now suppress the
  complete aggregate and its children instead of publishing an impossible
  duration from an unresolved zero query.

Two attempted removals were rejected by evidence rather than hidden:

- the face-band phi graph extension remains for closure-only rows. Direct fine
  CPT coverage left 2,994 boundary/Delaunay closure rows without authority;
  the fail-closed extension is required until closest-point carriers cover the
  complete closure graph;
- the isolated fixed-resident FMM oracle still triggers a Dawn/Metal process
  fault when dispatched, despite compiling and despite removal of allocation
  and indirect-dispatch dependencies. Its bucket work is parallel and reads a
  frozen distance snapshot, but the backend fault remains; the 256-sample host
  guard prevents dam-break-scale dispatch. The default JFA path passes Dawn; the
  quantitative JFA-vs-FMM displacement gate therefore remains open on this
  backend and no oracle rebaseline was made.

Acceptance evidence from the integrated tree:

- 179 focused source/ABI tests are green (170 pass, 9 expected GPU skips), and
  the affected factor-4/factor-8 Dawn seed, JFA, portability, and P7 tests
  pass;
- both batched and safe-bring-up one-step UI dam smokes publish 19,935/19,935
  faces with zero unresolved rows and no validation errors; their harnesses
  stop only on the existing zero-CFL assertion for that one-step setup;
- the 50-step minimal power dam reaches all 50 steps with every fine generation
  committed, all final 11,749/11,749 band faces accepted, and no WebGPU
  validation errors. It ends on the downstream pressure threshold
  `0.01296999613907457 > 0.012`; the pre-change baseline stopped earlier at step 44 on a
  power-face generation audit;
- a timestamp-enabled 10-step rerun publishes no impossible values after the
  decoder guard. Dawn returned an incomplete boundary chain, so every affected
  Section 5 field was correctly reported unavailable (`0`) rather than treated
  as performance evidence; post-change wall-clock capture remains an open gate;
- the 24×18×16 UI grid uses 5,742 of 6,912 resident fine bricks (16.9% brick
  headroom; 5,879,808 of 7,077,888 payload bytes), while the deliberately
  capacity-tight 16³ smoke uses all 4,096 configured bricks without overflow;
- a dedicated Dawn capacity checkpoint now exercises the production-width
  60×45×40 factor-8 B4 lattice with the complete twelve-ring topology band.
  A full maximum-area planar interface publishes exactly 280,800 desired/active
  bricks into the production physical-band capacity of 337,500: 56,700 bricks
  (16.8% of capacity) remain, with 287,539,200 of 345,600,000 four-channel payload
  bytes active. The checkpoint uses the full Chebyshev support and performs no
  corner trimming; it proves this production-width planar case, not arbitrary
  fragmented-interface geometry, so overflow telemetry remains authoritative.

## 0. The paper's mandate, restated

Section 5 as written is a CPU procedure. The paper says so itself
(Section 8): *"we are presently reinitializing the signed distance with a
serial Fast Marching method on the narrow band; alternative reinitialization
schemes that admit parallelism certainly merit attention."* The paper is the
source of truth for **what** is computed:

- S5.a — a sparse narrow-band fine level set at 4–8× the finest octree
  spacing; velocities never live on the fine grid.
- S5.b — multi-step semi-Lagrangian advection: m backward segments of dt/m,
  velocity re-interpolated at each segment endpoint.
- S5.c — per-step dynamic topology: fresh narrow-band page set from the
  advected interface, new cells activated, stale cells dropped.
- S5.d — redistancing of the fine level set (paper: serial FMM; tetrahedral
  Sethian–Vladimirsky FMM for the coarse octree phi near T-junctions).
- S5.e — velocity interpolation: least-squares full vectors at cell centers,
  trilinear in cubic regions, barycentric on catalog Delaunay tetrahedra at
  transitions.
- S5.f — velocity extrapolation into air by closest-to-interface order over
  faces, with power-face ↔ regular-face transfer around the band.

None of these *outputs* requires the serial *orderings* the paper uses to
compute them. The one mathematical fact that unlocks everything: for
redistancing (speed F = 1, no obstacles) the Eikonal viscosity solution **is**
the Euclidean distance to the interface. Any method that finds the closest
interface point produces the same answer fast marching does; causal ordering
is an optimization for serial machines, not a correctness requirement.

## 1. What we have today (measured/audited baseline)

Steady state is already structurally sound: one `queue.submit` per step, zero
per-step readbacks, GPU-transactional A/B generation publish
(`lib/webgpu-uniform-eulerian.ts:1783`, `webgpu-octree.ts` `encodeSurface`
:3070–3232). The problems are *inside* that submission and at startup.

Audited hot spots (file:line refs from the 2026-07-22 audit):

| # | Hot spot | Structure | Cost shape |
|---|---|---|---|
| H1 | Fine redistance, bucketed causal FMM (`lib/webgpu-octree-fine-levelset-redistance.ts:113–198`) | 42 half-cell buckets × 14 dispatches = 595 ordered dispatches, 86 passes (one `clearBuffer` boundary per bucket) | 12 of the 14 per-bucket dispatches are page-allocation machinery, not marching |
| H2 | Face-band velocity extrapolation heap (`lib/webgpu-octree-face-fast-march.ts:1825`, loop :1333–1338) | `marchFaceHeapChunk` is `@workgroup_size(1)` — a binary heap popped by **one GPU lane**, `ceil(faceCapacity/1024)` chained dispatches | The single most GPU-hostile kernel in the repo |
| H3 | Support-tier closure S0→S6 (`webgpu-octree-face-fast-march.ts:1130–1233`) | ~7 serialized tiers; each tier's indirect args written by the previous tier's capture kernel; ~90 dispatches in one pass | Latency-bound chain of thin dispatches |
| H4 | Band-phi Jacobi ping-pong (face band, :1282–1289) 12 rounds; coarse redistance ping-pong 8 rounds (`webgpu-octree-power-coarse-levelset.ts` :252–255); summary pyramid per level | O(band)/O(levels) short passes | Acceptable order, fusable |
| H5 | Serial single-thread scans: `emitSeeds` (`webgpu-octree-fine-levelset-topology.ts:547`), `completeAdaptiveOwners` (`webgpu-octree-face-fast-march.ts:1214`) | `dispatchWorkgroups(1)` `@workgroup_size(1)` | Latency spikes proportional to leaf count |
| H6 | Pre-change startup: 8 fenced submissions, each `submit` + `await onSubmittedWorkDone` (`webgpu-uniform-eulerian.ts:866–888`, phases `webgpu-octree.ts:96–105`) | 8 full CPU–GPU round trips; now one by default, with all 8 retained in safe mode | Where the 120 s warmup timeout and >3 min serial-FMM incidents lived |
| H7 | Encoder/param coupling: 65 invocation-stable uniform slots requiring submit-and-retire before the next encode (`webgpu-octree-power-coarse-levelset.ts:154–191,263`) | Structural fence-like constraint | Blocks multi-step encoding |
| H8 | Observability: no dedicated timestamps — fine redistance is folded into `surface-update`, the face band into `projection`/`power-projection` (`lib/performance-stage-model.ts:71,101`) | — | Cannot prove or size wins |

Measured context (ocean-seiche 384×96×64, 33.5 ms/step): surface-update
2.29 ms (6.8 %), extrapolation 0.85 ms; the face band hides inside
projection (5.1 ms, already flagged suspicious). Precedent for the dispatch
argument: replacing 128 short pressure dispatches with 32 row-parallel
Chebyshev passes cut that stage **78 %**
(`docs/GPU_STAGE_DIAGNOSTICS_CAPTURE_PLAN.md:43–45`). Dam-break-class scenes
with large interface area, and every t=0 startup, are where H1/H2/H6 dominate
rather than the calm-ocean profile.

## 2. Tractability verdict

**Yes — every Section 5 output can be produced with GPU-native, order-free or
fixed-order algorithms, within WebGPU's portable constraints.** Verified
platform facts this plan relies on:

- Dispatches inside one compute pass are as-if-serial with respect to memory:
  each dispatch is its own usage scope (WebGPU spec §3.5.5, confirmed by
  spec maintainers, gpuweb discussion #4434). Ordered stages need **no** pass
  breaks, submissions, or fences — implicit visibility between dispatches is
  normative. The 8 fenced startup phases are not required for correctness.
- Indirect dispatch args may be written by earlier dispatches in the same
  pass — counts never need CPU readback; a fixed maximal chain with
  zero-workgroup slack is free.
- There is **no** device-scope barrier in WGSL and no inter-workgroup
  forward-progress guarantee (Metal rationale; Sorensen et al., OOPSLA 2021).
  Persistent-thread megakernels and decoupled-lookback single-pass scans are
  excluded by design in this plan. Reduce-then-scan only.
- 32-bit `atomicCompareExchangeWeak` key-claim hashing is portable; payload
  visibility to *later dispatches* is guaranteed by the point above. Insert
  wins double as dedup and can emit compact worklists inline.

## 3. General problems and their modern solutions

### P1 — Causally ordered Eikonal solve (H1)

The bucket structure exists only to respect FMM causality. Three
research-grade replacements, all producing the same distance field:

1. **JFA closest-point transform with subcell seeds** (Rong & Tan 2006;
   production-standard in GPU SDF pipelines). Seed each fine cell whose
   transported phi changes sign across an edge with the linear (optionally
   quadratic, à la Chopp 2001) root position; jump-flood closest-point
   records with strides band, band/2, …, 1; resolve
   `phi = sign(phi_transported) · |x − cp(x)|`.
   Cost: **O(log₂ bandRadius) passes** — for the default 21-cell band:
   1 seed + 5–6 flood + 1 resolve ≈ **8 dispatches, zero atomics, zero
   ordering, fixed at encode time**. Accuracy: identical error class to the
   current FMM (both are exact only up to the piecewise-linear interface
   sample used for seeding); JFA's rare propagation misses are bounded and
   reducible with the 1+JFA variant.
2. **Fast Iterative Method** (Jeong & Whitaker 2008) as fallback where a
   uniform flood domain is awkward: unordered Godunov relaxation with an
   active list, capped iteration chain via indirect dispatch (~10–30
   iterations for this band), converging to the same discrete solution as FMM.
3. **Russo–Smereka subcell-fix PDE reinit** as a cheap per-step *repair*
   mode: 4–8 fixed Jacobi iterations to heal ≤2–4 cells of per-step drift,
   with the full JFA rebuild amortized to every K steps (see P5).

Chosen: **JFA-CPT primary, FIM fallback, PDE-repair as amortization mode.**
The current bucketed FMM is retained behind a flag as the validation oracle.

### P2 — Serial-heap velocity extrapolation (H2)

The heap orders face acceptance by distance-to-interface. The CPT from P1
already *contains* the closest interface point for every band cell.
Replacement: **closest-point extrapolation** — one gather pass per staggered
component: `u(face) = u(cp(face))`, interpolated one-sidedly from wet faces
(this is exactly constant along true normals — the property the marching
order approximates). Where a face-graph-local scheme is preferable (power
faces at transitions), replace the heap with **integer BFS layers**: one
dispatch per layer over the ~4-cell air band actually needed by advection
CFL, i.e. 4–8 layer dispatches instead of `ceil(faceCapacity/1024)` serial
single-lane pops. The band-phi Jacobi (H4) collapses into the same CPT
resolve.

### P3 — Allocation interleaved mid-algorithm (H1's 12/14 dispatches)

Root cause: the march discovers pages it needs while marching. Invert the
order: **allocate the entire redistance support band before any distance
work**. The topology stage already dilates (`initialDilationBrickRings`,
currently 1 ring); raise dilation to `ceil(bandCells / B)` brick rings
(21-cell band, B=4 → 6 rings). Serial ring-by-ring dilation would cost 12
dispatches; better, dilate by **brick-level Chebyshev-distance flood** (a
3-4-pass mini-JFA on brick keys) so dilation is O(log rings). Result: the
redistance stage runs on a fixed, fully-resident domain — no requests, no
dedup, no publication between distance passes. Capacity is validated
GPU-side exactly as today (§18.5 transaction), with the §18.12 measured
headroom (21 % at factor-8) re-benchmarked for the wider resident band —
the plan accepts a memory-for-structure trade and must report the new
active-payload numbers honestly.

### P4 — Serial dependency closures (H3, H5)

- Support tiers S0–S6: keep the tier semantics (they encode octree grading,
  which is real), but the seven capture→prepare-args→resolve chains stay in
  **one pass** (already true) and get fused: capture and args-preparation
  merge into the tier's resolve kernel (one dispatch writes the next tier's
  indirect args directly), cutting ~3 dispatches per tier; tiers with zero
  rows dispatch zero workgroups. Target: ~90 → ~40 dispatches, no semantic
  change.
- `emitSeeds` / `completeAdaptiveOwners` single-thread scans: replace with
  parallel classify + reduce-then-scan compaction (3 dispatches each,
  deterministic order by leaf row index).

### P5 — Per-step topology rebuild cost (S5.c)

Adopt the GVDB/SPGrid amortization: **dilate the band by K·maxCFL cells and
rebuild topology every K frames (K = 2–4)**. Intermediate frames run only an
O(1) escape-detector kernel (any active cell touching the dilation margin
sets a flag that forces rebuild via indirect args — no readback). Per-frame
redistance in non-rebuild frames uses the P1 repair mode. Overflow/capacity
validation becomes a **deferred, non-blocking** async readback acted on 2–3
frames later; the GPU transaction still hard-stops publication on overflow
in-frame, so safety is unchanged — only the CPU's *knowledge* of it is
delayed.

### P6 — Startup round trips (H6)

With P1–P3 in place there is no algorithmic need for 8 fenced phases:
encode cold topology → power authority → surface/global-fine → face band →
render world as **one or two submissions** (spec guarantees ordering). Keep
the per-phase fencing as a `?safeBringup=1` diagnostic mode — its purpose
(localizing Dawn/driver failures to a bounded phase) is real and the
timeout-forensics history in this repo justifies keeping it available. The
implemented default path pays one submission/fence; safe mode retains the
eight diagnostic phase boundaries from `webgpu-octree.ts:96–105`.

### P7 — Encoder/param slot coupling (H7)

Replace the 65 invocation-stable uniform slots with a single params buffer +
dynamic uniform offsets (or per-dispatch u32 push-block in a storage
buffer), removing the submit-and-retire-before-next-encode constraint. This
is a prerequisite for multi-step batched encoding, not a speed win itself.

### P8 — Observability (H8)

Before any replacement lands: add timestamp boundaries (shared-boundary
style, zero extra readback, matching `tests/webgpu-octree-power-timing.test.ts`
conventions) splitting `surface-update` into `fineTopology_ms`,
`fineTransport_ms`, `fineRedistance_ms`, and `projection` into
`faceBand_ms`, `faceMarch_ms`, `powerPublication_ms`. Every phase below
gates on before/after numbers from these fields.

## 4. Implementation phases, gates, and objective benefits

Ordering favors risk isolation: instrument, then replace the two worst
algorithms behind flags with the old paths as oracles, then restructure
topology, then de-fence startup.

### Phase 0 — Instrumentation (small)
Add the P8 timestamp splits and surface them in the performance panel.
Gate: timing tests extended; zero extra readbacks; fields present in the
ocean-seiche and dam-break profiles.
Benefit: fine-redistance and face-march cost become measurable facts.
Everything below quotes these numbers in its acceptance evidence.

### Phase 1 — Pre-dilated band (P3)
Widen topology dilation to full redistance support via brick-level
Chebyshev flood; strip the 12 allocator dispatches from the redistance
bucket loop (buckets keep marching on a fixed domain, 2 dispatches each,
as an interim state).
Objective structural change: 595 → 4 + 42×2 + 3 = **91 ordered dispatches**
(−85 %), passes 86 → ~45 (clearBuffer boundary remains per bucket until
Phase 2 removes buckets entirely).
Gates: existing redistance parity tests; band-completeness and Eikonal
residual telemetry unchanged; §18.12-style capacity checkpoint re-measured
and reported (expected higher residency, must stay under configured
capacity with stated headroom).

Gate result (2026-07-22): the Dawn 60×45×40 factor-8 B4 checkpoint requests
the production twelve-ring support around the domain's maximum-area plane and
publishes 280,800/337,500 resident bricks with no topology flags or rollback
(56,700 bricks, 16.8%, free). The
source contract now assigns allocation and out-of-domain clipping to topology;
redistance consumes the immutable generation and never allocates pages. This
is a deterministic full-plane capacity case rather than a proof for every
possible interface topology; production overflow remains fail-closed.

### Phase 2 — JFA-CPT fine redistance (P1)
Implement subcell seed + jump flood + signed resolve on the pre-dilated
band, storing packed cp records in the existing `workA`/`workB` channels
(§18.4). Deterministic tie-break on (distance, seed key). Bucketed FMM
retained behind `fineRedistance=fmm` as a small diagnostic oracle. The host
rejects more than 256 logical samples before encoding because the current
Dawn/Metal backend faults even with parallel bucket dispatches; it is not a
dam-break-scale oracle on this backend.
Objective structural change: fine redistance becomes **1 compute pass,
~8–10 dispatches, fixed at encode time** (vs. 42-bucket chain). No atomics,
no indirect dependency on live page count.
Gates (oracle comparison on dam-break + tiny-hydrostatic + factor-8 B4):
zero-crossing displacement vs. FMM ≤ 0.1·h_fine max over the band;
|∇phi|−1 residual distribution no worse than FMM's recorded residuals;
sign preservation exact; 300-frame endurance gate (§18.11) passes.
Note: bit-exact fingerprints (e.g. the dry-scene 0xa37d0cdd contract) will
change wherever fine phi feeds them — re-baseline deliberately, with the
oracle diff attached as evidence, never silently.

### Phase 3 — CPT/BFS velocity extrapolation (P2)
Delete `marchFaceHeapChunk`. Wet-face-sourced closest-point gather for the
regular band (reusing Phase 2 CPT where the fine band overlaps; face-graph
BFS layers — 4–8 layer dispatches — for power/transition faces). Band-phi
Jacobi remains a separate, band-bounded 12-round closure conditioner. CPT
graph depth is used only to plan logarithmic pointer jumps, followed by one
constant-time resolve; an eight-layer BFS remains a bounded rare-case repair.
Objective structural change: extrapolation goes from
`ceil(faceCapacity/1024)` serial single-lane dispatches + 12 Jacobi rounds
to **link + ceil(log2 graph depth) jumps + resolve + eight repair layers**, all
fully parallel (20 dispatches at the 768-deep bound); the Jacobi loop remains
12 rounds and never inherits the domain bound. The last `@workgroup_size(1)`
kernel in the hot path is gone.
Gates: divergence-free-band audit unchanged; drag-sphere rigid-coupling
probe numbers within tolerance; velocity-extrapolation IoU vs. oracle on
the smoke harness meets the existing parity bar; extrapolation stage time
(new `faceMarch_ms`) reported before/after.

### Phase 4 — Support-tier and scan cleanup (P4, P7)
Fuse tier capture/args kernels; parallelize `emitSeeds` and
`completeAdaptiveOwners`; replace param-slot coupling with dynamic offsets.
Objective: face-band phase ~90 → ~40 dispatches; no `dispatchWorkgroups(1)`
scans remain; encoder reuse constraint removed.
Gates: face-band topology hashes identical to current output (this phase is
purely structural); dispatch counts asserted in a source-contract test.

### Phase 5 — Startup de-fencing (P6)
Default path: ≤ 2 submissions with one final fence + authority validation;
`?safeBringup=1` retains 8-phase fencing. Keep the documented startup counts
aligned with the implementation.
Objective: startup CPU–GPU round trips 8 → 1–2; t=0 wall time drops by the
sum of 6–7 fence latencies plus per-submission driver overhead; smaller
exposure window for the known Dawn unreaped-child hang.
Gates: staged Dawn bring-up gate still passes in safe mode; default-mode
t=0 wall measured and recorded on the 384-column safety grid.

### Phase 6 — Amortized rebuild (P5) [optional, measure first]
K-frame topology rebuild with escape flag + per-step PDE repair redistance.
Only land if Phase 0 numbers show topology+redistance still material after
Phases 1–3 (ocean-seiche says topology is 3.9 ms — the calm-scene
change-driven rebuild already cut it to 0.2 ms, so this phase may be
unnecessary; decide on evidence).
Gates: 300-frame endurance with K>1; no visible surface artifacts at brick
boundaries in the porcelain scenes; escape-flag path proven by a forced-CFL
test.

**Decision (2026-07-22): deferred; do not land K>1 yet.**  The optional entry
condition is not met by the evidence currently in the repository:

- the only post-change-driven topology measurement is the calm-scene
  `3.9 ms -> 0.2 ms` result above, which says that skipping whole rebuilds
  would target at most a small residual in the one measured steady-state
  case;
- Phase 0 now exposes `fineTopology_ms` and `fineRedistance_ms` through the
  existing asynchronous timestamp readback, but no post-Phase-2 dam-break or
  ocean-seiche capture of those fields has been checked in.  Dispatch-count
  reduction is not a substitute for the wall-clock measurement required by
  this phase;
- the product path still advances a transactionally published A/B fine
  generation every step.  There is no Russo-Smereka repair operator or GPU
  escape controller in the implementation, so introducing only a host-side
  frame counter would create stale generations rather than implement P5;
- the transported-payload 300-frame gate is deliberately disabled until the
  Section-5 velocity-coverage gate passes, and there is currently no
  forced-CFL escape test or porcelain brick-boundary visual baseline.  Thus
  none of the three Phase-6 acceptance gates can yet support a K>1 default.

The implementation therefore stays at K=1: every accepted surface step
transports, rebuilds the fine topology, runs the fixed-dispatch JFA-CPT
redistance, and publishes one complete generation.  The existing coarse
octree change-driven dirty-tile worklist remains the proven amortization
mechanism; it avoids stale pressure topology while reducing calm-scene work.
Re-open P5 only after a successful 300-frame transported-payload run and
timestamp captures show `fineTopology_ms + fineRedistance_ms` is material in
both an interface-heavy dam break and ocean-seiche.  Any subsequent K>1 A/B
must land the PDE repair, device-owned escape/indirect-force path, forced-CFL
test, porcelain comparison, and endurance gate together.

## 5. Summary of objective benefits

| Metric | Today | After plan | Basis |
|---|---|---|---|
| Fine redistance ordered dispatches | 595 (86 passes, 42 causal buckets) | ~8–10 (1 pass) | P1/P3; JFA is O(log band) |
| Extrapolation | serial 1-lane heap, `ceil(faceCap/1024)` chained dispatches + 12 Jacobi rounds | parallel CPT link + 9–10 jumps + resolve + 8 repair layers; separate 12 band-bounded Jacobi rounds | P2; CPT/BFS |
| Face-band dispatches (transitions phase) | ~90 serialized | ~40 | P4 fusion |
| Startup CPU–GPU round trips | 8 fenced submissions | 1–2 (8 behind debug flag) | Spec-guaranteed intra-submission ordering |
| Mid-march page allocation | 504 dispatches/step (12 × 42) | 0 (pre-dilated band) | P3 |
| `@workgroup_size(1)` kernels in hot path | 3 | 0 | P2/P4 |
| Per-step readbacks | 0 | 0 (unchanged; capacity validation stays deferred/non-blocking) | — |
| Wall-clock expectation | — | Startup: bounded by fence removal + FMM removal (both implicated in the 120 s/3 min incidents). Steady state: surface work is 2.3 ms + hidden face-band share of projection's 5.1 ms on calm ocean; interface-heavy scenes and t=0 gain the most. The 78 % pressure-stage precedent bounds what dispatch-count collapse of this magnitude has delivered in this codebase before. | Honest sizing requires Phase 0 numbers |

Accuracy is not traded away: JFA-CPT computes the same Euclidean distance
FMM approximates, from the same subcell interface seeds; extrapolation by
closest point is the limit the marching order approximates; and the paper
itself calls for exactly this class of parallel replacement. The variational
pressure solve, power-diagram geometry, and all Section 4 guarantees are
untouched.

## 6. Risks and mitigations

- **JFA propagation misses** (rare, bounded): 1+JFA variant; oracle diff
  gate in Phase 2; FIM fallback ready.
- **Fingerprint/bit-exactness contracts break**: planned re-baselining with
  oracle evidence; never silent (Phase 2 gate).
- **Wider resident band raises memory**: the re-run §18.12-style checkpoint
  records 280,800 active bricks and 287,539,200 four-channel payload bytes for
  a production-width maximum-area planar factor-8/twelve-ring case, leaving 16.8% of its
  337,500-brick capacity. Chebyshev corners are intentionally retained: no
  smaller support metric has been proved for backtrace plus interpolation.
  Fragmented-interface overflow remains fail-closed and must still be reported.
- **Sparse page boundaries in JFA strides**: floods run in fine-lattice
  coordinates through the page hash (neighbor-page IDs are already cached,
  §18.3); a stride that lands on a non-resident brick reads "no seed", which
  is correct because the pre-dilated band covers the entire support of any
  in-band closest point by construction — add a debug assert counter.
- **Determinism**: (distance, seed-key) tie-break; reduce-then-scan only;
  no decoupled lookback anywhere.

## 7. References

JFA: Rong & Tan, I3D 2006. FIM: Jeong & Whitaker, SISC 2008; Hong 2021
(GPU). Parallel FSM: Detrixhe, Gibou & Min, JCP 2013 (rejected: O(n) plane
sweeps, poor for narrow bands). Group marching/Dial buckets: Kim, SISC 2001
(current scheme's family). Subcell seeds: Chopp, SISC 2001; Russo &
Smereka, JCP 2000. Hopf–Lax redistancing: Lee et al., JCP 2017; Royston et
al., JCP 2018 (rejected: page-hostile gathers). PDE extrapolation: Aslam,
JCP 2004. GPU sparse-topology practice: Setaluri et al. (SPGrid) ToG 2014;
Hoetzlein (GVDB) HPG 2016; Wu et al. CGF 2018 (per-frame GPU rebuild,
no-marching redistance); Gao et al. ToG 2018, Wang et al. ToG 2020 (GPU MPM
page tables, zero readback). GPU hashing: Alcantara 2009; Ashkiani 2018.
WebGPU semantics: spec §3.5.5 usage scopes; gpuweb discussion #4434
(as-if-serial dispatches); WGSL §17.11 (no device-scope barrier); Sorensen
et al., OOPSLA 2021 (no forward-progress guarantee).
