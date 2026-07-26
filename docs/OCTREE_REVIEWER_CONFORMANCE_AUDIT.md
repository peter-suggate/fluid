# Octree M1 Max — reviewer conformance audit

Audit date: 2026-07-26 (later than, and superseding the status claims in,
`docs/OCTREE_M1_MAX_IMPLEMENTER_HANDOFF.md`).

Audits the working tree against the seven GPU-mapping recommendations for
`docs/papers/aanjaneya-2017-power-liquids.txt`, and against
`docs/WEBGPU_OCTREE_M1_MAX_IMPLEMENTATION_PLAN.md`.

**This is the first audit in this series that ran anything.** Every prior
document in the chain states explicitly that no lane was benchmarked. §1 records
what was executed. Derived figures — dispatch counts, serial iteration counts,
byte estimates — are labelled as derived every time they appear, and none of
them is a timing.

---

## Read this first

1. **The tree was being edited by a concurrent session throughout this audit**,
   including while the GPU runs in §1 executed (`webgpu-octree-air-velocity-support-gpu.ts`
   and `webgpu-octree-fine-levelset-transport.ts` changed minutes before and
   after). `npm run audit:octree-production-source` went 6 → 7 violations with
   line numbers shifting mid-audit. Findings are therefore anchored on **symbol
   names**; line numbers are indicative. Re-grep before acting.

2. **Consequence for the §1.2 failure specifically:** it is a real, specific,
   reproducible-looking diagnostic, but it was captured against a moving tree and
   **must be reproduced on a quiesced tree before being treated as a standing
   defect.** It may be in-flight breakage.

3. A stale exclusive lock may be left at `/tmp/fluid-webgpu-exclusive.lock` by an
   aborted run. That is deliberate crash evidence
   (`tools/webgpu-smoke-isolation.ts`, `acquireWebGPUExclusiveLock` docstring),
   not a bug. Confirm the recorded owner PID is dead, then call
   `releaseWebGPUExclusiveLock()`.

---

## 1. What was run

### 1.1 Suites

| Command | Result |
| --- | --- |
| `npm run test:webgpu:octree-m1-cutover` | **69 / 69 pass** |
| `npx tsc --noEmit` | clean |
| `npm run test:unit` | 1522 tests: **1446 pass, 11 fail, 65 skipped** |
| `npm run test:power-liquids-structure` | 32 / 32 pass |
| `npm run verify:octree-power-catalog` | 6 / 6 pass |
| `npm run test:octree-regression-attribution` | 10 / 10 pass |
| `npm run audit:octree-production-source` | **7 violations in 60 sources** (6 earlier in the same session) |
| `npm run lint` | 111 problems (74 errors) |

The cutover suite did real GPU work — 406.9 ms compiling structured dynamics
variants, 173.8 ms executing a one-row PCG solve. It did not silently skip.
**All five failures recorded as handoff item 4 are closed**, and the portability
question was resolved the right way: test 28 (`every structured boundary entry
point fits the hard ten-storage-buffer contract`) passes, so the shaders came
back under 10 bindings rather than the gate being loosened.

Lint is dominated by pre-existing style categories (37 `no-explicit-any`, 36
`no-unused-vars`, 29 `no-assign-module-variable` — a Next.js false positive on
WGSL modules). Not a blocker, though 36 unused bindings is consistent with the
dead-code findings in item 9.

**The 11 CPU failures.** Three are octree production contracts, all in files with
uncommitted edits — in-flight work, currently red:

| # | Test | Failed assertion |
| --- | --- | --- |
| 340 | rediscretized A2 owns coarse-fine contacts through spatial pages | destination-owned Eᵀ must gather through physical page adjacency |
| 1027 | fine-band support closes over the exact interpolation dependencies | transition demand must include catalog tetrahedron selector owners |
| 1175 | production regular structured transport matches affine interpolation | differential must stay attached to the production trilinear gather |

Two more are contract/lifecycle (554, 806). The remaining six (1322, 1394, 1395,
1411, 1420, 1446) are renderer/AO/shadow, outside this audit's scope.

### 1.2 The capture path — and what it reveals

`npm run capture:octree-regression-mini` and `-quiescent` both abort Dawn:

```
libc++abi: terminating due to uncaught exception of type
  std::out_of_range: absl::container_internal::raw_hash_map<>::at
WebGPU smoke PID ... exited from signal SIGABRT
```

**The discriminator is not the trace flags — it is `skip_validation`.**
`tools/benchmark-power-dam.ts` sets `FLUID_WEBGPU_DAWN_FEATURES = "skip_validation"`
unless `FLUID_BENCHMARK_VALIDATE=1`, and `tools/run-webgpu-smoke.ts` forwards it
as `enable-dawn-features=skip_validation`. The passing cutover suite runs with
**full Dawn validation on** and never constructs the octree solver end-to-end.
The code comment at the toggle site already documents this hazard: *"Dawn's
unchecked Metal path can abort inside native code instead of reporting a bad
command."*

Binary forensics on `node_modules/webgpu/dist/darwin-universal.dawn.node`
localises the throw: six `absl::ThrowStdOutOfRange` sites exist; three are
external-texture paths (the repo has no external textures) and two are
validation-gated (`ValidateBindGroupDescriptor`, `ValidateProgrammableStage`) and
therefore unreachable under `skip_validation`. The two survivors are
`ShaderModuleBase::GetEntryPoint` (a raw `mEntryPoints.at(name)`) and
`BuildSubstituteOverridesTransformConfig` — **both at pipeline creation**. With
validation skipped, an absent entry point, or a module whose WGSL failed to
compile and so has an empty entry-point map, reaches the raw `.at()` and throws
out of the Napi call.

**Then the decisive experiment.** Running the switch the harness provides for
exactly this purpose:

```
FLUID_BENCHMARK_VALIDATE=1 npm run capture:octree-regression-mini
```

does **not** abort. It produces a specific solver diagnostic and a clean throw
from `tools/webgpu-smoke-compact-field.ts` (`validateSnapshot`), reached via
`readCubicVolumeField`:

```
Compact octree QA coarse/fine generation mismatch:
  fineGeneration: 3          coarseGeneration: 2
  topologyPublished: 1       topologyRolledBack: 1
  topologyFlags: 24          downstreamFinalizeReason: 13
  topologyInterfaceBricks: 0 topologyDesiredBricks: 0  topologyActivatedBricks: 0
  topologyCapacityOrDilation: 6
  fineRestrictionValid: 0    fineRestrictionFlags: 8
  fineRestrictionCount: 4294967295 (INVALID)
  worklistActivePages: 4024  worklistGeneration: 3
  coarseRowCount: 1248       coarseControlCorrectedRows: 1001
  mgpcgControl: [1,0,0,0,1248,224,1,4294967295, 0…]
```

Three things follow, and they reorder this entire document:

- **`topologyRolledBack: 1` with coarse generation falling behind fine.** This is
  precisely the signature `docs/POWER_LIQUIDS_PERF_HANDOFF.md` attributes the
  30–100× nondeterministic slowdown to: *"Once the coarse power topology
  publication stalls (coarse gen ~17 vs fine 64, `topologyRolledBack=1`) … step
  cost explodes 30–100× nondeterministically with identical dispatch counts."*
  Here it is at coarse gen 2 vs fine gen 3 — **by step 3**.
- **The fine topology candidate is empty**: zero interface bricks, zero desired
  bricks, zero activated bricks, and fine restriction reports invalid
  (`fineRestrictionValid: 0`, count INVALID, flags 8). Something upstream of
  candidate construction is producing no interface, and the epoch is being
  rejected — correctly, fail-closed — every step.
- **`worklistActivePages: 4024` of 4096 logical bricks = 98.2%.** This is a
  **live confirmation** of the band-occupancy diagnosis in
  `docs/FINE_BAND_DENSITY_PLAN.md` (item 8 below), taken from a running solver
  rather than derived.

**So the capture path is not merely "unrun" as
`docs/OCTREE_M1_MAX_IMPLEMENTER_HANDOFF.md` item 1 states — and it is not a
harness bug either. It is the only thing in the repo currently reporting a real
solver failure, and `skip_validation` has been converting that report into a
native abort.**

Two premises in my own earlier reasoning were wrong and are recorded here so they
are not repeated: (i) "the crash happens very early because no step output
appears" is not evidence — `tools/benchmark-power-dam.ts` spawns the child with
`stdio: ["ignore","pipe","inherit"]` and feeds stdout into a readline that parses
without echoing, so *all* progress output is invisible by construction; (ii) the
timestamp-query path is **clean and irrelevant** — every `createQuerySet` site is
guarded by `GPUPerformanceTraceRecorder.supported`, counts are never zero, and no
`.at()` in the Dawn binary is reachable from `createQuerySet`,
`beginComputePass`, or `resolveQuerySet`. The deleted energy ledger is likewise
refuted: `energyRatio` is re-derived in TypeScript and a missing metric yields a
`null` → blocker string, not a crash.

**One latent defect to fix before any baseline is frozen:** for the quiescent
lane, `powerDamResultWindow` (`tools/power-dam-performance-report.ts`) differences
the command audit correctly but copies `stabilityEnvelope` **verbatim** from the
complete run. So the quiescent artifact's `residual`, `volumeDrift` and
`energyRatio` are maxima over all ~560 steps including the settling motion — not
over the quiescent window. That silently defeats the lane's purpose.

**Also note the artifact contract has never executed on real data.**
`docs/baselines/octree-regression/` does not exist and `artifacts/octree-regression/`
is empty. `tests/octree-regression-artifact.test.ts` builds every input by hand
and hardcodes `sourceSha256: "1".repeat(64)`; it never calls
`octreeRegressionRevision()`. So `dispatchAttribution`, `activeRatios` (an exact
`1e-12` consistency check), `authorityInventory` (exact byte-sum equality) and
`compactDissipationRatio` are unverified against real data. Expect the
"zero blockers" bar to fail on the first *successful* capture even after §1.2 is
fixed.

---

## 2. Scorecard against the seven recommendations

| # | Recommendation | State |
| --- | --- | --- |
| 1 | Keep SPGrid's layout, discard its VM machinery | **Largely met** |
| 2 | Power diagram as tables, not runtime geometry | **Met** |
| 3 | Split by topology class and specialize kernels | **Infrastructure only — specialization never written** |
| 4 | MGPCG rebuilt for a sync-poor GPU | **Mostly met; one-reduction property reverted in the worktree** |
| 5 | Particle transfers without atomics | **Met, and machine-enforced** |
| 6 | Topology rebuild as a parallel pipeline | **Not met** |
| 7 | Design for bandwidth | **Partly — band residency is 100%** |

---

## 3. Do not redo — these landed and are correct

### Recommendation 1 — page pool and directories

Every hot-path directory is O(1) with no search: `octreeOwnerPageLookup`
(2 dependent loads); `makeFineLevelSetSortedWorklistLookupWGSL` (direct index
despite the name); `pageFor`/`pageSlot` (brick-mask + `countOneBits` rank, with
`lookupProbeUpperBound: 1` accurate); `pageOf` in fine transport (`7u +
pageCapacity + key`, serving 8 taps per `sampleFine`). The fine-summary merge
ladder is genuinely gone, and the `publicationRow` nested binary search that
`FINE_BAND_DENSITY_PLAN` flagged is gone too — the transport shader now contains
zero `loop {`.

27-page physical adjacency for pressure/MG pages is built at publication and
**consumed** by both the smoother and the Section 6.3 apply.

### Recommendation 1 + 4 — the Chebyshev smoother is the recommendation done faithfully

`smoothPage` / `smoothPageChebyshevForward|Reverse`: `PAGE 8×8×4`,
`HALO 10×10×6`, `HALO_ELEMENTS 600`, five `var<workgroup>` arrays ≈ 12 KB (well
under 32 KiB), workgroup 128 for a 256-element page. The halo is staged **once**,
2–4 Chebyshev sweeps run entirely in local memory with `workgroupBarrier()`
between them, interior written back once, and the post-smooth consumes the same
weights in reverse so the V-cycle stays a symmetric linear map. This is both the
reviewer's block-sizing advice and the "shared-memory sub-iterations per global
pass" advice. (One real inefficiency remains inside it — item 10.)

### Recommendation 2 — tables, not geometry

- **No runtime power-diagram geometry in any recurring path.**
  `constructOctreePowerCell` is reachable only from the offline generator and
  tests. Nothing clips a polygon, computes a face area from vertices, builds
  Delaunay tetrahedra, or runs quadrature at runtime.
- Coefficients derive from baked `catalogSlots` through
  `reconstructStructuredCatalogSlot` — a signed-permutation transform plus an
  anchor-size affine map (area = `anchorSize² × slot.areaCentroid.x`, inverse
  distance = `slot.normalInverseDistance.w / anchorSize`).
- **Neighbour addresses are derived, not stored.** `canonicalDirection` supplies
  18 offsets; `worldDirection` applies the 6-bit transform code.
  `storedNeighbourIndices: 0` holds. The 36-slot gather form is **entirely absent
  from `lib/`** — `resolvedNeighbor` does not exist. Better than the handoff
  claims.
- **Catalog miss fails closed** through a complete chain: `LOOKUP_MISS` →
  candidate flags → `finalizePowerTopology` refuses promotion →
  `encodeReadyCommitGate` `poison()`. There is no fallback geometry path.
- **SPD is structural, not tolerance-based:** one canonical face owner
  (`row < neighbor`), the non-owner's handle rewritten to the owner's with
  `sign = -1` so the coefficient is bit-identical from both sides; cross-level
  coupling an exact adjoint pair via `finerAdjoint` reading the fine row's own
  coefficient.

### Recommendation 4 — the rest of the solver

- Pipelined MGPCG is the **only** production pressure solver — one call site, no
  mode switch. `webgpu-octree-mgpcg.ts`, `octree-power-galerkin.ts` and both
  `power-galerkin*` GPU modules are deleted and asserted absent. No RAP or triple
  product is reachable from production.
- Subgroups **required and fail closed** — the constructor throws before any
  `createBuffer` when `subgroups` or the 128-lane / 4096 B limits are absent. No
  portable reduction alternative is compiled.
- The reduction is genuinely three-level (`subgroupAdd` → workgroup partials →
  a separate one-workgroup finisher), reading `subgroup_size` from the builtin
  rather than assuming it.
- Spectral bounds are computed at publication from the rediscretized first-order
  stencil, validated per row, and the epoch is **rejected** on non-positive or
  non-finite bounds. Weight-0 sweeps are identity, so a rejected bound cannot
  corrupt the iterate.
- Fixed V-cycle schedule with **zero host readback in the hot path**; the whole
  solve is one command buffer (`PassBroker.fence()` only ends the pass);
  `prepareCorrectionDispatches` zeroes indirect X-counts on convergence so a
  converged solve does zero lane work.
- The coarse solve is `exactBottom` — a single exact division, structurally
  right-sized (`levelCount = ceil(log2 width) + 1` forces a 1×1×1 coarsest level,
  and the epoch is rejected unless the coarsest count is 1). No CPU, no
  `mapAsync`. **Better than the substitution the spec proposed** — see item 11.
- Restriction/prolongation are destination-owned gathers with
  `restrictionAtomicAddUpperBound: 0`. Because the transfer chain is immutable
  topology, summation order is fixed and bit-reproducible, and both directions
  share the same `correctionTransfer` records so E and Eᵀ cannot diverge.

### Recommendation 5 — the Eulerian deviation is real, complete, and machine-enforced

- No `P2G`/`G2P`/`FLIP`/`APIC` code exists; the only repo-wide hits are a
  source-guard regex and an unrelated quaternion flag. Two guards in
  `lib/webgpu-octree-work-accounting.ts` make it structural rather than
  conventional: `primary-particle-transport` and `floating-velocity-scatter`.
- **Zero float atomics in the octree path.** 96 `atomic<u32>` + 5 `atomic<i32>`,
  every one a control/error word, an integer compaction counter, an idempotent
  `atomicOr` mask, or an integer `atomicMin` election. The two hottest fine
  stages (transport, redistance) contain **no atomics at all**. The one
  `atomicAddFloat` CAS loop in the repo is in the *quadtree* method, off this
  path. The fixed-point `atomic<i32>` scatters in `webgpu-eulerian.ts` are also
  off-path (gated on `!this.octreeProjection`) and are correctly built anyway —
  integer addition is associative, so fixed-point scaling makes them
  deterministic.
- Secondary particles are one-way by *type*: solver fields are bound as sampled
  `unfilterable-float` textures, the only `read_write` bindings are
  particle-private, and the particle buffer is `STORAGE`-only so it cannot be a
  vertex or copy source. The system is not even constructed
  (`get secondaryParticles() { return undefined; }`).

  **Doc caveat worth recording:** at `HEAD` (424b8e3) this module *did* contain a
  solver write — `nearestParticlePhi: array<atomic<u32>>`, a
  `texture_storage_3d<r32float,write>`, and a `copyTextureToTexture` back over
  the solver's surface texture — inert only because `surfaceCorrectionStrength`
  defaulted to 0. The spec's unconditional "never" was overstated there; the
  uncommitted change makes it literally true.

### Recommendation 6 — what is correct

Refinement decisions are per-cell parallel (`@workgroup_size(4,4,4)` / `(256)`,
indirect). Grading is bounded parallel dilation
(`balanceRounds = ceil(log2 maxLeafSize)`). Block activation/deallocation is a
GPU prefix-sum/scatter chain. The directory **is** double-buffered and the flip
**is** fail-closed with byte-for-byte retention of the prior epoch.

**A genuinely quiescent topology does reach zero rebuild work.** The prior
handoff's "this gate cannot hold" is too strong: when the producer's dirty count
and add/retire counts are zero, `firstLevel` stays at `levels()`,
`markDirtyFrom` is never called, and all seven rebuild helpers early-return. See
item 7 for why it nevertheless fires every step in motion.

### Recommendation 7 — what landed

- **SoA is genuine.** The MG `state` buffer is channel-major
  (`at(c,l,s) = c*totalLevelSlots() + levelBase(l) + s`, 26 planes), so loading
  `A[s]` does not pull the other 22 channels. The smoother touches only
  A/RHS/DIAG/FLAGS + coefficients — no phi, no face velocity, no solid geometry,
  no topology metrics. The fine path is SoA at *sample* granularity
  (flags/phi/workA/workB separate buffers); the only AoS access is a 10-word
  brick record amortized 64:1.
- **f16 rule respected exactly** — zero f16 in any shader; the only mention is an
  optional device-feature request.
- **The pressure solve creates zero bind groups per advance in steady state** —
  the hottest, most-iterated code, correctly keyed per MG level.
- **Duplicate velocity caches were genuinely removed.** `publishFineVelocityCache`
  is gone; the far-air characteristic cache died with the deleted fused-transport
  module. Face velocity has one authority (plus one for the disjoint air-extension
  DOF set).
- **Fusion (a) landed** — divergence fused with RHS construction
  ("Fuse structured divergence RHS class", writing `divergenceRhs` in the same
  pass).
- **The fine transport band gate landed** (`FINE_BAND_DENSITY_PLAN` P1a, recorded
  there as dead). `inTransportBand` is defined *and gates real work* in all four
  class kernels: `if(!inTransportBand(old)){nextPhi[index]=old;continue;}`,
  skipping 4 velocity substeps and an 8-tap `sampleFine`.
- **The active-brick counter is fixed** (P0.1) — `webgpu-uniform-eulerian.ts` now
  reads `.activeCount` (word 1) with an explanatory comment, surfaced with
  capacity in `components/DiagnosticsPanel.tsx`. This is why §1.2 could report
  4024/4096 at all.
- The `createBindGroup` construction crash that `FINE_BAND_DENSITY_PLAN` recorded
  as blocking baseline capture is **fixed** by the `fineOnlyEntries` filter.

---

## 4. Work items

Ordered. Item 1 unblocks the ranking of everything after it.

### 1. Fix the capture path, and treat what it reports as the priority

Two distinct problems, and the second is more important than the first.

**(a) `skip_validation` converts solver faults into native aborts.** The
benchmark lanes run with Dawn validation off, so a bad command aborts inside
`ShaderModuleBase::GetEntryPoint` or `BuildSubstituteOverridesTransformConfig`
instead of being reported. Every capture and profile lane is affected, not just
`--artifact`. Options: default `FLUID_BENCHMARK_VALIDATE=1` for capture lanes
(validation costs CPU time, not GPU time, so it barely perturbs a GPU-bound
measurement), or keep `skip_validation` only for the final timing pass after a
validated warm-up run has proven the command stream.

To localise the abort if it recurs: change the spawn in
`tools/benchmark-power-dam.ts` to `stdio: ["ignore","inherit","inherit"]` (or tee
readline lines to stderr). The last `phase:"pipeline"` line from
`tools/run-webgpu-smoke.ts` names the exact entry point being compiled when it
died.

**(b) The validated run reports a topology rollback by step 3.** See §1.2 for the
full control-word dump. `topologyRolledBack: 1`, coarse generation 2 against
fine 3, an empty fine topology candidate (zero interface/desired/activated
bricks), and `fineRestrictionValid: 0`. This is the documented stall signature,
reached almost immediately.

**Reproduce this on a quiesced tree first.** If it holds, it is the highest
priority item in the repo — it is a correctness failure, it is the documented
cause of 30–100× nondeterministic step cost, and it makes every performance
number meaningless until fixed. Start from `fineRestrictionValid: 0` /
`fineRestrictionFlags: 8` and `topologyInterfaceBricks: 0`: the candidate found
no interface, so ask why interface classification produced nothing while 4024
pages were active.

**Also fix before freezing any baseline:** the quiescent lane's
`stabilityEnvelope` is not windowed (§1.2), so its residual/volume/energy
maxima describe the settling motion, not the quiescent window.

### 2. Non-convergence silently ships a stale pressure field

`finalizeAndPublish` (`lib/webgpu-octree-pipelined-mgpcg.ts`) on failure or
non-convergence writes:

```wgsl
let seed = select(0.0, pressureSeed[row], finite(pressureSeed[row]));
pressureOut[row] = select(seed, candidate, success && finite(candidate));
```

and `pressureSeed` is bound to `pressureIn` (`lib/webgpu-octree.ts`) — the
**previous step's pressure**. `control[20]` (`published`) records success, but
nothing in `lib/webgpu-octree.ts` reads it, and `encodeStructuredProjection`
receives only the pressure buffer and proceeds unconditionally. A solve that
exhausts its encoded iterations produces a divergent velocity field with no
runtime signal.

This matters more now than before: item 4 changed the smoothing depth without a
convergence measurement, and item 1(b) shows the solver is already in a degraded
regime.

**Done when:** projection is gated on the publication word, or a non-convergent
solve fails the substep closed the way a rejected topology epoch does.

### 3. Catalog completeness hole — a candidate mechanism for item 1(b)

`tools/generate-octree-power-catalog.ts` excludes every boundary mask with two
**opposite** faces set:

```ts
const validBoundaryMasks = Array.from({ length: 63 }, (_, index) => index + 1)
  .filter((mask) => !([[0, 5], [1, 4], [2, 3]] as const).some(([n, p]) =>
    (mask & (1 << n)) !== 0 && (mask & (1 << p)) !== 0));
```

But the runtime descriptor generator sets a mask bit for **every** out-of-domain
face probe (`lib/webgpu-octree-power-descriptor.ts`), and `octreeLeafSize`
(`lib/webgpu-octree.ts`) has **no clamp against the domain extent**. So any live
leaf whose size equals the domain extent along an axis sets both bits for that
axis, produces a mask absent from the catalog, and fail-closes the epoch.

Concrete trigger: the UI lane is 24×18×16 with `maximumLeafSize` defaulting to
16, so any size-16 leaf spans z fully.

This is *a* candidate mechanism for the rollback observed in item 1(b) — not
established as *the* mechanism; note the mini lane's `coarseMaximumLeafSize` was
2 in that run, which does not trigger this. Settle it with **no GPU at all**: a
pure-TS test enumerating every `(descriptor, mask)` pair the runtime generator
can emit, asserting each has a catalog key.

### 4. Two uncommitted worktree changes move against the plan's own gates, unmeasured

Both differ between the git index and the worktree:

| Symbol | Staged | Worktree |
| --- | --- | --- |
| `reductionsPerOuterIteration` | 1 | **2** |
| `OCTREE_SECTION43_PRODUCTION_SHELL_DEPTH` | 2 | **8** |

**The one-reduction property was reverted.** The staged version is a merged
Chronopoulos–Gear recurrence; the worktree removed the merged denominator
(`delta - beta * gamma / previousAlpha`) and its fused direction update, leaving
textbook PCG with an explicit `d·(Ad)` reduction. Phase 5's exit gate is *"one
global reduction synchronization per executed outer iteration"* — as written it
cannot hold. This looks like a deliberate robustness trade (the C–G denominator
relies on exact A-conjugacy, which f32 loses). If so, **amend the gate and record
the reason** rather than leaving a green-reading spec against reverted code.

**Encoded dispatch counts (derived from the production formulas, not timed).**
`encodedDispatchCount = 8 + 2·apply + setup + correction + E·(6 + apply + correction)`
with `apply = 5`, `setup = 18`, and
`correction = (l + 5 + 4(l−1)) + 2k + 4 + 5 + (2k−1)`:

| Lane | l | k | E | Encoded |
| --- | --- | --- | --- | --- |
| mini 16³ | 5 | 8 | 6 | **564** |
| UI 24×18×16 | 6 | 8 | 7 | **681** |
| envelope | 6 | 8 | 10 | **927** |
| (doc's stated figure) | 4 | 2 | 7 | 393–407 |

The plan sets the gate at *"under ~600"*. **Missed for every lane with E ≥ 7.**
The handoff's "≈ 393 at E = 7" is stale on four independent counts: `k` 2→8, the
per-iteration term 4→6, `firstOrderSetupDispatches` 9→10, and `l` = 4 in the doc
against 5–6 live.

At the repo's measured 3.6 µs dependent-dispatch tax, 681 dispatches ≈ 2.4 ms of
launch latency per advance. The larger cost is not the launches: `k = 8` means
**16 smoothing sweeps plus 15 residual applies of the expensive second-order
operator per preconditioner application**, over a band seeded as
`boundary || transition` then dilated 3 layers — which on a ≤ 24³ graded domain
is plausibly most of the domain.

Also: `countOctreePressureCommands` uses `firstOrderSetupDispatches + 7` where the
class computes `+5 + BOUNDARY_BAND_LAYERS` = `+8`, so the "exact mirror" in
`docs/WEBGPU_OCTREE_SOLVE_TAIL_POLICY.md` is one low throughout and applies
`l = 5` figures to a lane whose `l` is 6. Its test asserts against hand-written
literals rather than live class values — the shape that lets a mirror drift.

**Per `docs/EXECUTION_MULTIPLIERS_HANDOFF.md` §1 rule 2, neither change may ride
in a perf commit.** Both need their own differential gate, and both need item 1.

### 5. Recommendation 3 was reversed, and a test enforces the reversal

All five "class-specialized" pressure applies in
`lib/webgpu-octree-spgrid-vcycle.ts` are the **same body** with only a workset
index differing:

```wgsl
fn applyRegularInterior(...)    {let row=workRow(...,0u); ... applyRow(row);}
fn applyTransitionInterior(...) {let row=workRow(...,1u); ... applyRow(row);}
fn applyPhysicalBoundary(...)   {let row=workRow(...,2u); ... applyRow(row);}
fn applyTransitionBoundary(...) {let row=workRow(...,3u); ... applyRow(row);}
fn applyMergedBand(...)         {let row=workRow(...,4u); ... applyRow(row);}
```

`tests/webgpu-octree-spgrid-vcycle.test.ts` **asserts they stay that way**:

```js
assert.match(octreeSPGridAccurateOperatorShader,
  /applyTransitionInterior[\s\S]*applyRow\(row\)[\s\S]*applyPhysicalBoundary[\s\S]*applyRow\(row\)[\s\S]*applyTransitionBoundary[\s\S]*applyRow\(row\)/);
```

Writing a specialized regular body turns that test red. The reversal is
institutionalized, not accidental.

What a `regularInterior` row executes in `applyRow`, for what is at `caseId == 0`
a 7-point constant-coefficient stencil: a geometry + metrics struct load, three
validation branches, a topology-directory load, **19 coefficient loads**, an
18-channel loop to sum the diagonal, a second 18-channel loop applying them, a
~5-dependent-load `pageSlot` page-directory chase *per channel*, and
`finerAdjoint` — an 8-child × 18-direction loop whose inner
`coefficientForDirection` is itself an 18-entry linear search. The stencil is
7-point in value and 19-point in traffic; nothing knows that `caseId == 0` means
constant-coefficient.

This is the hottest kernel in the solver and the only major one with **no
threadgroup halo staging** — the accurate operator shader contains zero
`var<workgroup>`, as does `lib/webgpu-octree-section43-preconditioner.ts`.

The L1 multigrid smoother — the largest dispatch block — has **no class split at
all**, and the module intended to fix it declares in its own docstring that it
"contains no GPU class, WGSL, dispatch record, or alternate pressure path."

Four of the five promised fine-grid block sets (`interfaceBlocks`,
`redistanceBlocks`, `rareTransportBlocks`, `solidBoundaryBlocks`) return zero
code hits. The transport split that exists is real for regular vs transition, but
its `COMMON` and `RARE` twins differ **only by a diagnostic counter** — and since
the classifier marks a page RARE if it contains any single air-side sample, the
`COMMON` "fast path" is likely the near-empty one.

**The measurement that would justify any of this already exists in RAM and is
thrown away.** The four per-class row counts are read back every capture and then
summed into `classRows` in `lib/webgpu-octree-work-accounting.ts`; the four
per-class transport page counts sit inside the 68-word governor readback,
undecoded. Publishing them is a few lines. **Do that before writing any
specialized kernel** — if `regularInterior` is not ~95% of rows, the
specialization is not the win.

### 6. Recommendation 6 — the MG candidate rebuild is one GPU thread with a hash inside

`buildCandidateLevelSetsAndGhosts` and `buildCandidateLevelDeltas` are both
`@workgroup_size(1)` dispatched `[1,1,1]`, and together run the entire MG
candidate hierarchy rebuild on one lane. `cLookup` / `cInsert` remain **256-probe
linear open-addressed hash scans**. Reached per substep via
`encodeCandidateSetup` ← `encodeInactiveCoupledPowerCandidate` ←
`encodeInactiveTopologyCandidate`, called unconditionally at the substep tail.

Derived cost at mini scale (`levelCapacities [8192,1024,128,16,2]`,
`rowCapacity 4096`): a **fixed floor of ≈ 95,700 serial stores** paid whenever
any level is dirty, independent of row count, plus row- and slot-proportional
terms totalling **≈ 430K serial memory operations** (≈ 1.3M if the band fully
refines). Estimated **1.5–6 ms/advance**, central ~3 ms — against
`EXECUTION_MULTIPLIERS_HANDOFF` targets of ≤ 12 ms in motion, ≤ 3 ms quiescent.

Two corrections to the prior handoff's framing:

- The dominant term is **`rebuildCandidateTransferFor`'s 8-corner loop over
  Σ`cCount(l)`**, not the ghosts. The 18-channel ghost body executes once per row
  (guarded by `if (firstTrailingBit(h.y) != l) continue`), so it is `rows × 18`,
  not `levels × rows × 18`.
- "The exact shape of the 59.68 ms redistance" is the **wrong analogy** — that was
  a 49M-tap *parallel* kernel. The right in-repo calibration is the structurally
  identical lane-0 serial loop in `lib/webgpu-fluid-brick-residency.ts`,
  previously estimated at 1–3 ms.

The real problem is scaling: the fixed floor is O(domain cells) serial work
inside a method whose entire premise is sparsity.
`rebuildCandidatePageWorksetFor` is precisely the "block activation/deallocation
stream compaction" the reviewer specified, implemented as a serial
scan-and-append; `rebuildCandidateDirectoryFor` contains an explicit **serial
prefix sum** — the one primitive the reviewer named as the fix.

Also still single-lane or single-workgroup on recurring paths:
`summarizeStructuredFineTransport` and `publishStructuredFineDelta`
(`@workgroup_size(1)`, the latter looping `8 + 2 × pageCapacity`);
`publishStructuredFineTransportWorksets`; `publishRecurringSparseBand`
(dispatched `(1,1)` — 256 lanes each running a `(2r+1)³ = 15³ = 3375`-iteration
`atomicOr` loop per interface seed; `FINE_BAND_DENSITY_PLAN` P1b, unlanded);
`buildDirtyTileDelta` and `buildDirtyFrontierDelta` (`@workgroup_size(1)` over
tile capacity, the former with an **unbounded** full-table hash probe);
`buildOwnerPageCandidate` (an in-kernel bitonic sort plus per-row binary search
plus a lane-0 serial ranking loop, all in one workgroup). 57 of ~312 octree
compute entry points are `@workgroup_size(1)`.

### 7. The topology dirty predicate is all-or-nothing, so item 6 is paid every step in motion

In `planL1CaptureDelta`:

```wgsl
var topologyChanged=true;
var stencilChanged=true;
if(!topologyChanged){ ...compare against capturedGeometry... }   // unreachable
else{pageFirst=0u;}
```

The comparison branch is unreachable, and `deltaOldRow`, `sameL1Topology` and the
`newToOld` map it consumes are dead in this consumer. Worse than dead code:
`pageFirst = 0u` forces the first affected level to 0, so **every** level is
marked topology- and stencil-dirty, and the rebuild helpers clear and rebuild all
of `levelCapacity(l)` and all of `rowCapacity` regardless of how many rows
changed. One dirty row costs exactly what 4,096 dirty rows cost. The per-page
`changeStamp` and per-level row-range machinery is computed and discarded.

At mini scale "one dirty row" means "every row": `rowAuthorityDirtyGeneration`
returns true if the cell's tile **or any tile in its 3×3×3 ring** changed, and
`topologyTileSize() = max(8u, dimsMax.w)` gives a 16³ domain only 2³ tiles — so a
3×3×3 ring covers the whole tile domain.

**The handoff's 3-before-2 sequencing is no longer sound.** There is no
dirty-predicate win on the quiescent lane, because the guard already returns
false when nothing changed. The available win is *granularity* (level- and
row-range-scoped rebuild), which is the same work as the parallel rewrite. Do
items 6 and 7 together.

Note the rebuild is **deferred and double-buffered but not asynchronous**:
everything lands in one `GPUCommandEncoder` submitted once, so the two
single-lane singletons sit on the critical path occupying 1 of ~32 cores. The
recommendation's "asynchronous, CPU as helper" clause is not satisfiable as
written in WebGPU — the achievable forms are to make the work parallel, or to
submit it in a separate command buffer. Neither is done.

### 8. Recommendation 7 — the band is 100% of the lattice, at the shipped default

`docs/FINE_BAND_DENSITY_PLAN.md` derived this with
`interfaceRefinementBandCells = 3`. **The shipped default is 4**
(`lib/methods/octree.ts`), so it is worse than recorded:

```
transportBandFineCells  = min(256, max(4, 4 × 4))  = 16
redistanceBandFineCells = min(256, 16 + 4 + 3)     = 23   ← still stacks 3 radii (P2b unlanded)
requiredFineCells       = max(4 + 1, 23)           = 23
dilationBrickRings      = ceil(23/4) + 1           = 7
bandLayers              = 2 × 7 + 1                = 15
```

| lane | logical bricks | area×width | ×1.25 | capacity | occupancy |
| --- | --- | --- | --- | --- | --- |
| mini 16³ | 4096 | 3840 | 4800 | **4096** | **100%** |
| UI 24×18×16 | 6912 | 6480 | 8100 | **6912** | **100%** |

**Live confirmation:** §1.2's run reported `worklistActivePages: 4024` of 4096 =
98.2%. The paper's fine level set is **6.3%** occupied (543M of 8.6G logical at
2048³, Fig. 2) — a ~16× miss on the paper's central compression argument.

**The propagation mechanism is worse than the doc records.**
`lib/webgpu-octree-fine-levelset-topology.ts` sets
`dirtyHaloRings = ceil((redistanceBandFineCells + 1) / brickResolution) = 6` and
`supportHaloRings = 2 × dirtyHaloRings = 12`. The brick lattice is **16 bricks
across** on both lanes, so a 12-ring support halo spans 25 bricks > 16 and the
JFA support set **saturates the whole lattice from a single changed brick**. The
redistance stage's `dispatchWorkgroupsIndirect` is therefore only *cosmetically*
active-shaped — the indirect count it reads always equals capacity. Halving the
halo (P2c) is worth more than the doc suggests.

`globalFineSurfaceBandCells` (P2a) does not exist; P2b and P2c are unlanded.
P1c (`changedNeighborRadii`, still O(desired × changed), up to 8192 iterations
per desired page) and P1d are unlanded.

**Consequence worth stating plainly: 16³ and 24×18×16 cannot demonstrate
compact-band scaling at all.** Every sparsity conclusion drawn from those lanes
is measuring a dense grid. `FINE_BAND_DENSITY_PLAN` P4's 64³ lane and the plan's
own argument for a lane large enough to make the Section 6.3 streaming difference
measurable are the same request, and should land together.

### 9. Dead and duplicated work written every substep

- **`axisNeighbor`** (`lib/webgpu-octree-structured-dynamics.ts`) — defined,
  **zero callers**. The plan's "Verified closed this round" list credits "direct
  six-axis neighbour handles replacing the O(slots) scan"; the handles are
  written every publication and never read, and consumers still walk bounded slot
  loops.
- **The 27-page fine halo** (`haloBase` in
  `lib/webgpu-octree-fine-levelset-topology.ts`) — two writers, **zero readers**,
  and `includeHalo27` defaults to `true`, so 27 words × page capacity × 2 banks
  are allocated and 27 stores per brick per publication are paid for nothing.
- **Three dead capacity-shaped buffers**, all pure removals with no consumer to
  redirect: `transportedPhiSnapshot` ×2 (**2 MiB** + a per-publication indirect
  dispatch; `payloadSnapshot` has exactly one write and no read, and is
  `STORAGE`-only so no external consumer is possible — the real rollback path
  reads `rollbackPhi`); the air-support vector mirror in `recordArena`
  (**576 KiB** + one atomic store per support sample, no WGSL read); and
  `previousFineSeedLeaves` (**40 KiB** + a full-capacity write pass per
  publication, documented consumer does not exist).
- **The 19-channel catalog table is stamped and then overwritten.**
  `publishSection63Rows` writes
  `section63Coefficients[dest+c] = catalogCoefficients[src+c] * scale`; later in
  the same epoch `rebuildStructuredBoundaryRows` — bound to the *same buffer*
  (binding 14 = `section63.coefficients`) — zeroes all 19 channels and rebuilds
  them from `area × inverseDistance × aperture × scale`. Nothing reads the
  stamped values in between, so the per-case coefficient table is **write-only
  with respect to the pressure solve**; its live uses are validation and
  ghost-support decisions.
  *This does not violate recommendation 2* — the replacement values still come
  from baked `catalogSlots`, which is exactly the recommendation's "stamp
  per-face coefficient records at rebuild time." But the handoff's description of
  the catalog as the operator's source is wrong, the stamp loop is 19 wasted
  stores per row per substep, and an in-code comment claiming A2 "reads the
  immutable 19-channel catalog directly" is stale.
- **A per-fine-sample binary search where an O(1) directory exists.**
  `find` / `owner` in `lib/webgpu-octree-fine-levelset-volume.ts` binary-search
  the coarse directory with a Morton re-encode *per probe*, wrapped in a `loop{}`
  size ascent, dispatched over every active fine sample. `octreeOwnerPageLookup`
  answers this exact query in 2 loads.
- **Lesser duplicates:** pressure has 3 physical copies (the third defensible as
  the Krylov iterate; note `pressureB` is deliberately double-purposed as
  topology tile-state scratch, so collapsing A/B is not free); coarse phi is
  duplicated inline in both sample directories (24 KiB, removable by indirecting
  through `entry.row`); fine phi holds 10 channel-equivalents against the paper's
  four (`rollbackPhi` is legitimate transaction state with live consumers, since
  transport mutates `phi` in place); `rowVelocities` has 5 nameable consumers but
  no recorded measured win, so Phase 8 item 9 is formally unmet.

### 10. The MG smoother stages the payload but re-reads the operator every sweep

This inverts the intent of an otherwise excellent block design.
`pageAppliedA`/`pageAppliedB` run once per interior slot **per sweep**, and each
does:

```wgsl
for(var k=0u;k<18u;k+=1u){ let c=loadf(XP+k,l,slot); if(c==0.0){continue;}
  let relative=vec3i(decode(state[at(KEY,l,slot)],l))+stencilDirection(k)-…
```

- **18 stencil coefficients** are fetched from global `state` every sweep, though
  they are invariant across all sweeps.
- `state[at(KEY,l,slot)]` is loaded **inside the k loop** — up to 18× where 1
  suffices. Because `state` is `read_write` in this module, the compiler **cannot
  legally hoist it**. A one-line fix.
- `smoothable` reads the FLAGS channel from global per slot per sweep — also
  invariant.

Derived order of magnitude: staged payload ≈ 7 KB per page loaded once; operator
traffic ≈ 256 slots × ~76–148 B × 4 sweeps ≈ **78–151 KB per page**, i.e. roughly
**11–21× the staged payload**. Staging all 18 coefficients would cost
256×18×4 = 18.4 KB of threadgroup memory on top of the existing ~9.6 KB — tight
against the 32 KiB budget and possibly occupancy-negative. **The KEY hoist and
FLAGS staging need no budget and should be done first regardless.**

### 11. The source guards enforce nothing

`tests/webgpu-octree-work-accounting.test.ts` asserts only that each violation
*belongs to a discovered source* — never that the list is empty:

```js
assert.ok(audit.violations.every((violation) => sources.some((source) =>
  source.endsWith(violation.source))), "every violation must belong to ...");
```

And `audit:octree-production-source` is not part of `npm test`. So the 7 live
violations — four `domainVolume` full-domain dispatches in
`lib/webgpu-octree-air-velocity-support-gpu.ts` (3 pure full-domain sweeps per
advance, plus one capacity-shaped which at 100% occupancy is also full-domain)
and three capacity scans in `lib/webgpu-octree-fine-levelset-topology.ts` — are
unflagged Phase 7 / §2 violations.

Three structural gaps in the rules themselves:

- `unbounded-lookup` was widened to catch WGSL `loop {` (so handoff item 6 is
  half-closed), but still misses the forms the shaders use for search:
  `for (var probe = 0u; probe < 256u; ...)` and `while (lo < hi)`.
- `CAPACITY_AUTHORITY` is a **host-side vocabulary** (`rowCapacity`,
  `pageCapacity`, `dims.n[xyz]`, …). WGSL spells capacity as `levelCapacity(l)`,
  `p.capacity.x`, `rows()`, `brickCount(l)` — none match. `domainVolume` is
  absent too, which is why three `dims³` dispatches per publication are invisible
  to the gate.
- No rule has the form "small workgroup count × capacity-bounded body," so
  `[1,1,1]` dispatches are structurally unflaggable. That is why item 6 was never
  caught.

**A hot-path probe the docs mis-attribute as cold:** `classifyDesiredPageIdentities`
is dispatched **unconditionally every advance** (outside the bootstrap/recurring
branch), and calls `desiredContains` — a binary search over up to `pageCapacity`
entries — for every occupied page: ~4096 pages × ~12 dependent probes ≈ 49k probe
loads per advance. `docs/FINE_BAND_DENSITY_PLAN.md` records it as
*"cold-bootstrap expansion only"*. That attribution is wrong, and the site
violates the Phase 1 exit gate *"zero binary/hash probes in regular fine-brick
and pressure-page hot loops."*

### 12. Bind groups: ~25–30 uncached per advance, all in the fine A/B ping-pong

Uniform pattern: **single-slot caches keyed on buffers that alternate every
advance ⇒ 100% miss rate.**

- `lib/webgpu-octree-fine-levelset-summary-direct.ts` — **~22–24 per advance**
  (16 named groups + per-level parent groups), keyed on A/B-alternating
  `params/metadata/worklist/flags/phi` + `pageDelta`.
- `lib/webgpu-water-pipeline.ts` — **7 per rendered frame**. The early-out
  includes `previous.generation === source.generation`, but the octree bumps the
  generation every advance so it can never fire — and **no bind group actually
  depends on the generation**, only uniform *contents* do.
- `lib/webgpu-octree-fine-to-coarse-levelset.ts` — 3 per advance, same cause.
- `lib/webgpu-fluid-brick-residency.ts` — 2–4 per advance with **no cache at
  all**, and every resource is constructor-hoistable. Cheapest fix here.

Exemplary A/B patterns to copy already exist: the nested `WeakMap` caches in
`webgpu-octree-structured-boundary.ts` and `webgpu-octree-structured-dynamics.ts`.

Latent trap: `webgpu-octree-technique-overlay.ts` guards on `this.source ===
source`, but the supplier builds a fresh object literal on every read, so the
guard can never hit. Harmless only because the caller is transition-gated today.

### 13. Fusions (b) and (c), and the bytes-moved model

- **(b) pressure-gradient projection fused with closest-point extension seeding —
  ABSENT.** `projectFamily` ends at `setValue(handle, projected)`; extension lives
  in a separate module with its own `encode()` and 13+ passes. Only one fusion
  label exists in the entire octree path.
- **(c) mask generation fused with producers — not done, and correctly so.**
  `liquidMask` has ≥3 read-only consumers, and the spec conditions this fusion on
  *"where the mask has one consumer."* **Not a gap.**
- **Phase 8's bytes-moved gate is met by analytic constants, not measurements,
  and one is off by an order of magnitude.**
  `lib/webgpu-octree-work-accounting.ts` charges `transport[2] * 24` — 24 bytes
  per traced sample. Actual: `sampleFine` alone does 8 corners × (directory word
  + `flags` + `phi`) ≈ 96 B, plus 4 substeps of structured velocity
  interpolation with catalog lookups — realistically 200–500 B. **Understated
  ~10–20× for the dominant fine stage.** Two further blind spots: `fine-redistance`
  records `activeLanes: scheduled`, so its active ratio is definitionally 1.0 and
  it can never reveal wasted work; and only 3 of the plan's 5 fine stages are
  recorded (volume and restriction absent). The allocation inventory is
  module-scoped, not quantity-scoped, so it structurally cannot detect a duplicate
  representation of phi or velocity *within* a module — exactly where all four
  duplicates in item 9 live.

### 14. Minor

- **Compensated f32 is weaker than it reads.** No emulated f64 is used, but each
  reduction invocation owns one row, so the per-lane `lo` term is always zero and
  the subgroup stage does two *independent, uncompensated* `subgroupAdd`s.
  Effective error is O(32·ε) per 128-row block, not O(ε). `alpha` and `beta` are
  stored as `(x, 0.0)` — plain f32. Fix or rename.
- **`twoSumF32` is untested on GPU** — exactly the pattern Metal fast-math
  destroys by reassociation, and the Dawn test uses a one-row system where
  compensation is unexercised. A ≥256-row test seeded `+1e8, +1, +1, −1e8` would
  settle whether the compensated layer is real or decorative.
- **The even-`k` rule is invented w.r.t. the paper but load-bearing in the
  implementation.** §4.3 constrains nothing (`k` iterations, then `k` more) and
  the paper's examples use `k = 3` — which
  `normalizeOctreeSection43BoundarySmoothing` silently rounds to 4. But the
  pre-shell ping-pong leaves the iterate in `hybridA` only for even `k`, and
  `publishCorrection` reads `hybridB`. **Dropping the clamp without fixing the
  parity-dependent buffer selection would read a stale buffer** — contrary to the
  prior handoff's item 5.
- **The persistent coarse-solve interface is dead.** `persistentEnabled: false`;
  `OctreePersistentMGPCGExecutor` has zero production implementers, two tests
  assert its absence, and one work-accounting test exercises
  `persistentEnabled: true` — live coverage for a path production cannot produce.
  Since `exactBottom` is better than the spec's proposed substitution (§3),
  **amend the spec and delete the interface.**
- `compactBandIntersections` ranks by `atomicAdd` rather than the mandated prefix
  rank — numerically harmless, but it breaks the stated determinism guarantee.
- The band classifier re-derives row class by summing 18 coefficients
  (`section63Class`), and `dynamicRowClass` does it again — a third and fourth
  independent derivation of a class the published worksets already carry.
- `invalidRows` (workset slot 4) is never populated in the GPU lane, while
  `applyMergedBand` reads slot 4 as the band *union*. Only the calling convention
  keeps those apart.
- The JFA schedule may be running 8 passes rather than the intended 5:
  `planFineLevelSetJFAStrides(bandCells, bandCells)` passes the band as the
  *displacement* argument, taking the cold branch and rounding stride to 32. The
  in-code comment describes the 23-cell/stride-32 schedule as "former", but at
  the shipped default that is what production runs. Log `strides.length` once to
  confirm.
- The plan's *"one 128-lane subgroup implementation"* reads as a subgroup-width
  claim; the code correctly uses the runtime `subgroup_size` builtin with a
  128-*lane workgroup*. The sentence is misleading, the code is right.

---

## 5. The strategic point

`docs/EXECUTION_MULTIPLIERS_HANDOFF.md` §0 is the calibration that reframes this
list: the mini dam frame is **~95% shader execution, not launch overhead**, with
non-solve execution ~43 ms against an arithmetic floor of ~3–4 ms, and four
stacking multipliers — A thread-count, B per-thread, C schedule, D duplicated
quantities.

At 3.6 µs/dispatch the 681-dispatch encoded chain is ≈ 2.4 ms of a ~56 ms frame.
Recent effort has concentrated on that ~4%. Meanwhile:

- **A** is unaddressed where it matters most: the band is 100% of the lattice
  (item 8, live-confirmed at 98.2%), the JFA support halo saturates the lattice
  from one changed brick, and full-domain recurring dispatches remain (item 11).
- **B** is unaddressed: the regular apply pays 19 coefficient loads, two
  18-channel loops, a page-directory chase per channel, and a nested linear
  search for a 7-point stencil — and the specialization that would fix it is
  *forbidden by a test* (item 5). The smoother re-reads its operator every sweep
  (item 10).
- **C** moved in the wrong direction: `k` 2→8 quadrupled shell smoothing,
  unmeasured (item 4).
- **D** persists: dead adjacency records, ~2.6 MiB of dead buffers, a write-only
  coefficient table, and a redundant per-sample directory search (item 9).

The scaffolding for all seven recommendations is built to a high standard. The
two changes that would convert scaffolding into speed — specialize the regular
kernel, narrow the band — are both still ahead. And item 1(b) says something more
urgent than any of it: the solver may not currently be converging on its own
primary lane.

---

## 6. Undetermined, and what settles each

1. **Whether item 1(b) reproduces on a quiesced tree.** Everything else is
   subordinate to this. Stop the concurrent session, re-run
   `FLUID_BENCHMARK_VALIDATE=1 npm run capture:octree-regression-mini`.
2. **Actual per-stage cost ranking.** Every ms figure here is derived from
   dependent-load counts and published link taxes. A GPU timestamp bracketing the
   pass labelled `"SPGrid V-cycle · build inactive exact level deltas"` isolates
   item 6 almost exactly — one line of instrumentation.
3. **The real class proportions.** Is `regularInterior` 95% or 20%? The data is
   already read back and summed away (item 5). Publish four named lanes.
4. **`bandRows / liveRows` for the Section 4.3 shell.** Already computed, used
   only for a `<=` assertion. If it approaches 1.0, the merged-band collapse was
   free and the class split is pointless.
5. **What fraction of samples the transport band gate skips.** `transport[2]`
   counts only traced samples, so the instrumentation can answer it. One clean
   mini-lane read of `fine-transport` `activeLanes / scheduledLanes` sizes item
   8's remaining value — the cheapest measurement available.
6. **Whether item 3's mask hole can fire.** A pure-TS enumeration test; no GPU.
7. **Whether Dawn preserves `twoSumF32`.** A ≥256-row GPU test.
8. **Whether staging 18 coefficients fits the threadgroup budget** (item 10) —
   28 KB against 32 KiB, feasible on paper, occupancy unknown. Do the KEY hoist
   first; it needs no budget.
9. **Why the one-reduction revert happened.** The reason exists only in the
   author's head; the file is staged-but-modified.

---

## 7. Suggested order

**1 → 2 → 3 → 8 → (5-measurement) → 4 → 6+7 → 10**, with 11 alongside 6.

Item 1 first, in both halves: stop `skip_validation` from hiding faults, then
reproduce and fix the topology rollback. Nothing can be ranked or trusted before
that. Item 2 next because it is small and it is the correctness hole that makes a
degraded solve invisible. Item 3 because it is a no-GPU test that may explain
item 1(b). Item 8 next because it is the paper's central argument, it shrinks
five stages at once, and nothing else in recommendation 7 competes with it —
land P2a/P2b/P2c/P2d together, since P2d is the mitigation for a band that is
too narrow. Then publish the class counts before writing any specialized kernel.
Then choose `k` and the reduction count from measurement. Then items 6 and 7
together, since the sequencing that separated them no longer holds. Item 10's KEY
hoist is nearly free and can go any time.

The pure deletions in item 9 (~2.6 MiB, two recurring write passes, one dispatch,
two dead adjacency records) carry no risk and need no measurement.

## Standing constraints

- From `docs/EXECUTION_MULTIPLIERS_HANDOFF.md` §1 rule 2 — **no
  accuracy-affecting default flips in a perf commit.** Item 4 covers two such
  flips already in the worktree; they need their own commits and gates.
- From `docs/POWER_LIQUIDS_PERF_HANDOFF.md` — batching cycles behind extra pass
  boundaries **regressed** on M1 (4-cycle 11.60 ms, 8-cycle 11.21 ms against
  10.88 ms direct). More pass structure is not the answer to dispatch count.
- From `docs/FINE_BAND_DENSITY_PLAN.md` — **do not benchmark a tree that is being
  edited**, and treat any capture containing post-stall steps as measuring a
  correctness bug rather than a performance ceiling. Item 1(b) is that bug
  surfacing directly.
