# Power Liquids Performance Handoff

**Date:** 2026-07-25 · **Scene:** mini dam break 16³ (Dawn, M1 Max) and UI lane (`octree:sim-0.148000`)

## TLDR

The pipeline is not missing one clever optimization. It has a structural disease: **every phase pays a fixed, scene-size-independent cost** because the discretization is re-derived from scratch every step. The uniform solver being ~100x faster is the expected consequence — it does direct array indexing over a fixed dispatch list; this pipeline rebuilds topology, re-meshes the face band, and hash-probes directories per row per step. The plan that works is **deletions and direct lookups, not fusions** (dispatch-count cuts measured wall-neutral).

## 2026-07-25 characteristic-cache experiment

The M1 Max single-chunk lane now reuses the dormant velocity-prepass
result/status arena to publish complete velocity once per live fine node. A
fast trace interpolates that cache only in positive air farther than one
maximum characteristic displacement from the interface. Liquid, protected
interface samples, mixed direct/extrapolated cells, domain departures, and
cache misses run the original exact arbitrary-point Section-5 sampler.
Multi-chunk devices retain the exact path.

| Variant | Total | Complete-velocity cache | Cached trace | Exact fallback | Characteristic total |
|---|---:|---:|---:|---:|---:|
| Committed exact baseline | 69.44 ms | — | — | 15.48 ms | 15.48 ms |
| Mode-consistent cache, no interface guard | 63.85 ms | 4.72 ms | 0.39 ms | 5.18 ms | 10.29 ms |
| Protected positive-air cache | 66.92 ms | 4.65 ms | 0.20 ms | 6.82 ms | 11.67 ms |

The unguarded result is deliberately not the accepted number: its 500-step
run completed without non-finite values but crossed the strict connectivity
gate (98.48% dominant component versus the 99% minimum). The protected
variant retains a **2.52 ms/advance (3.6%)** total gain and a **24.6%**
characteristic-stage gain while keeping interface and liquid samples exact.
The exact two-step and 500-step gates pass; the latter reproduces the committed
baseline stability envelope exactly. Treat that full gate as mandatory
authority before changing the guard or accepting a faster variant.

Fine timestamp attribution owns all three new passes independently and the
compute-pass table is again closed. The cache adds two passes and two
dispatches per advance (50/796 versus 48/794) without allocating another
arena.

## 2026-07-25 Section-5 rows and power-face publication

The next mini-dam batch reduced the accepted protected-cache lane from roughly
**67.4 to 62.15 ms/advance** (about **7.8%**) with zero validation errors.
The final schedule has **792 dispatches in 50 compute passes**.

| Cut | Before | After | Result |
|---|---:|---:|---:|
| Affected power-face publication | ~8.45 ms | 3.08 ms | **−5.37 ms, −63.6%** |
| Section-5 support-row aggregate | ~13.17 ms | 12.85 ms | −0.32 ms |
| Recurring dispatches | 796 | 792 | −4 |

Power-face geometry is now counted, prefixed, and published row-parallel from
the exact affected-row delta. The former single-workgroup row walk was the
dominant cost. Compact/merge semantics and deterministic CSR order remain
unchanged.

Section-5 support identities now use a packed bitset instead of one `u32` per
possible `(cell, level)`. Marking uses `atomicOr`, scans popcount 256-word
blocks, and scatter enumerates set bits in canonical order. The duplicate
post-publication mark/scan/canonical-directory rebuild and validation launches
are deleted. Coarse signs are attached while core rows are copied and inherited
as support rows are scattered, deleting the separate full-row sign sweep.

The wall result also records a profiling lesson: a temporary fence before the
sign sweep made the next merged pass's ~10 ms occupancy appear under the sign
label. Removing the sign dispatch merely moved that time to
`Classify exact Section 5 catalog-adjacency delta`, proving it was attribution,
not execution, from the sign kernel. Fine timing must fence both sides of a
kernel before treating a merged pass's leading label as kernel authority.

Regression authority: exact two-step and full **500-step / 2 s** minimal-dam
gates pass. The long run ends with generation 501, 1,372 power rows, 5,263
faces, 8,320 incidences, converged MGPCG, and no validation errors.

## 2026-07-25 Galerkin defect cache and invariant cleanup

The pressure solve was the next arithmetic bottleneck. Restriction previously
evaluated the same fine-row sparse matrix product once for every incident
transfer record: 888,264 CSR-entry visits per V-cycle in the fixed mini
hierarchy, versus 156,136 for one rowwise product. The final even pre-sweep
leaves channel A authoritative and channel B dead, so a rowwise kernel now
caches `rhs - A*x` in B once and restriction gathers only `weight * B[fine]`.
Over the observed seven-cycle profile this removes roughly **5.1 million
repeated CSR-entry visits**.

The same batch deletes recurring validation of constructor-packed immutable
CSR offsets, columns, transfer records, RAP records, and diagonal indices.
Changing live headers, entries, coefficients, RAP totals, residuals, and
corrections retain their fail-closed checks. This distinction matters: the
invariant branches were nested in the hottest matrix/transfer loops even though
the packed topology cannot change after construction.

| Metric | Before | After | Result |
|---|---:|---:|---:|
| Power Galerkin solve | ~15.60 ms | 10.88 ms | **−4.72 ms, −30.3%** |
| Mini-dam advance | 62.15 ms | 58.39 ms | **−3.76 ms, −6.0%** |
| Encoded dispatches | 792 | 832 | +40 fixed-tail cache dispatches |
| Compute passes | 50 | 50 | unchanged |

The extra dispatches expose the next optimization directly: the 20-cycle
schedule usually converges in 6–8 cycles, but later direct dispatch commands
still launch and return through `stopped()`. Zero-work indirect dispatch
arguments can retain the hard 20-cycle safety ceiling while eliminating that
post-convergence launch tail.

All five focused Dawn Galerkin tests, TypeScript, the exact two-step dam, and
the full 500-step / 2 s gate pass. The two-step result remains six cycles with
the same residual and field hashes. The long run covers cycles 2–16, ends at
generation 501 with 1,372 rows / 5,263 faces / 8,320 incidences, and reports
zero validation errors.

## 2026-07-25 ownership-only power-face counting

Affected power faces were still running the complete reciprocal-slot,
shared-geometry, intersection, and polygon-clipping path twice: once to count
the output and again to publish it. Counting now validates the row authority,
slot reconstruction, world plane, neighbor lookup, and canonical ownership
only. Publication remains the sole full-geometry authority and retains all
reciprocal and polygon validation before transaction acceptance.

| Metric | Before | After | Result |
|---|---:|---:|---:|
| Affected power-face phase | ~3.08 ms | 1.835 ms | **−1.245 ms, −40.4%** |
| Mini-dam advance | 58.39 ms | 57.02 ms | **−1.37 ms, −2.3%** |
| Encoded dispatches | 832 | 832 | unchanged |
| Compute passes | 50 | 50 | unchanged |

TypeScript, the focused static shader tests, the exact two-step dam, and the
full 500-step / 2 s gate pass. The long gate again ends at generation 501 with
1,372 rows / 5,263 faces / 8,320 incidences, a converged pressure solve, valid
face and incidence topology, and zero validation errors.

An indirect-dispatch experiment also established an M1-specific scheduling
constraint: batching convergence-controlled cycles behind additional compute
pass boundaries regressed the Galerkin phase even when it removed stopped
launches. Four-cycle batches reached 11.60 ms and eight-cycle batches
11.21 ms, both slower than the 10.88 ms direct fixed-tail schedule. Empty
post-convergence launches are cheaper here than extra pass boundaries, so the
next solve cut must reduce active arithmetic without splitting the pass.

## 2026-07-24 implementation log — UI throughput authority

All values below are repeated free-running `benchmark:power-dam-ui` measurements
on the same M1 Max UI lane (`24×18×16`, 62 exact advances), with zero WebGPU
validation errors. The benchmark currently exits non-zero only because command
auditing reports 6.8 unattributed compute passes/advance; that attribution gap is
not a physics or shader-validation failure.

| Cut | Throughput | Incremental result | Shader-phase evidence |
|---|---:|---:|---|
| Starting tree | 396.66 ms/advance | authority baseline | fine topology 57–61 ms; restriction ~60 ms; transition adjacency 80–82 ms |
| Do not scan every fine sample for optional diagnostics | 363.44 / 363.52 ms | **−33.2 ms, −8.4%** | restriction 17–27 ms |
| Changed-seed halo scatter + immutable-domain publication | 301.39 / 300.65 ms | **−62.5 ms, −17.2%** | fine topology 3.29–3.57 ms, **~17× phase speedup** |
| Direct reciprocal endpoint scatter | 287.95 / 288.23 ms | **−12.7 ms, −4.2%** | transition adjacency 69.6–69.8 ms |
| Shared immutable live-face work package + mutable-φ frontier | 224.66 / 224.84 ms | **−63.3 ms, −22.0%** | transition adjacency 73.3→26.8 ms; closest-point 31.6→5.9 ms |

Net: **396.66 → ~224.75 ms/advance, −171.9 ms/advance (43.3%)**. This is a
realized gain, not a dispatch-count proxy. Final authority remained exact:
generation 18, 5,981 active fine pages, 3,392 face-band rows, 7,791 regular
faces, 16,800 incidences, 1,624 transient power rows, pressure convergence in
six iterations, and all Section-5 failure counters zero.

The two transformational shader changes establish the reusable algorithm:

1. Fine recurring topology used to launch one 256-lane workgroup across every
   logical brick key. Each key scanned every changed/external seed and performed
   a binary resident lookup: `O(domain × changed × log(resident))`, with one
   workgroup serializing the memory latency. It now scatters each changed seed's
   bounded Chebyshev halo into a direct atomic bit mask, then linearly publishes
   the immutable logical-key domain in canonical order:
   `O(changed × halo³ + domain)`. Publication order supplies sorting and the bit
   mask supplies deduplication.
2. Terminal face-band endpoints used to invert the graph by making every
   endpoint scan every source row and all 36 source edges:
   `O(endpoints × sources × 36)`, about 100 million comparisons in the measured
   shape. The source-edge producer now atomically appends the reciprocal record
   directly into each endpoint's fixed 36-slot range. A bounded per-endpoint
   insertion sort restores deterministic source order:
   `O(edges + endpoints × 36²)`.
3. The face arena has fixed row-owned addresses but is sparse: the measured
   transaction has 7,791 LIVE faces in an 82,944-slot capacity. The existing
   deterministic face-count walk now emits one canonical dense face-slot list
   and one indirect count, at no extra pass. Fine/coarse phi sampling,
   closest-point seeding and extension, all eight repair waves, and terminal
   diagnostics pass this same immutable package through the transaction instead
   of rescanning sparse capacity. A second GPU-built list contains only mutable
   band-phi rows (225 of 3,392 at final authority) for the 16 dependency waves.
   Its storage and indirect triplet are dedicated: an early implementation
   correctly solved phi but accidentally reused indirect slot 54, truncating
   later full-row consumers. Moving it to byte offset 252 fixed the complete
   downstream authority.

The immutable-state rule is precise: physical page identity, row identity,
fixed face/adjacency slot addresses, and committed generation metadata are
carried; φ, velocities, active membership, and symmetric-difference lifecycle
are recomputed. Immutable does **not** imply dense: the face arena is
row-strided with holes, whereas published row/key lists are dense prefixes.

One follow-up experiment is deliberately not retained: parallelizing the
capacity-wide scalar diagnostic/count reductions compiled successfully, but the
combined Dawn run aborted in bind-group/layout lookup before producing an
authority record. It was reverted. Do not count it as a gain; reintroduce only
one reducer at a time behind a standalone runtime test.

## Characteristic transport working-set cut (2026-07-25)

The stable mini-dam profile moved from **57.02 to 55.85 ms/advance** without
changing its 832 dispatches or 50 passes. The characteristic velocity-cache
publication fell from roughly **4.59 to 3.02 ms** (~34%). Exact two-step output
remained bit-identical, and the 500-step / 2-second smoke mini-dam completed
with `passedInvariants=true` and no validation errors.

The retained changes reduce bytes and arithmetic rather than launches:

1. Cache publication now excludes both the guarded interface and source samples
   farther into air than the transport band can reach in one characteristic.
2. Cached velocity and status share one `vec4f`; trajectory outcomes and
   diagnostic positions share one arena. This removes two storage bindings and
   one unused indirect-dispatch allocation while preserving fail-closed status.
3. Cached tracing leaves a per-lane exact-fallback marker in that arena. Exact
   workgroups scan their contiguous 4x4x4 brick first and return before preparing
   page data when no lane needs fallback.

Two measured designs were rejected. Putting cached and exact tracing in one
kernel regressed to **57.94 ms** through divergence/register pressure. Building
a compact indirect list of fallback bricks added more compaction cost than it
saved because almost every resident brick contained at least one exact lane.
That is the important temporal-coherence result: **brick occupancy is the wrong
delta granularity**. The next persistent representation should cache changed
owner/identity/authority or interpolation rows, not merely compact active
bricks.

## Measurements (keep these)

### Probe inflation — discount the UI panel numbers

- UI panel reading: **569.96 ms** GPU physics via the intrusive `segmented-queue-wall` probe (submit→queue-callback wall checkpoint per phase; serializes the GPU and adds a browser round-trip per row).
- Measured on mini-dam 16³: free-running (no probes/fences) = **246 ms/step**; serial phase probe reads ~**404 ms** → probe adds roughly **+40%** round-trip inflation. Use free-run as the throughput authority (`npm run benchmark:power-dam-ui`).

### Post-stall regime (check any capture for this first)

- Steps 1–15 of mini-dam run at **11.3 ms/step**. Once the coarse power topology publication stalls (coarse gen ~17 vs fine 64, `topologyRolledBack=1`, `powerVelocity INVALID_SOURCE=128`, MGPCG gate flags 3), step cost explodes **30–100x nondeterministically** with identical dispatch counts (62-step runs: 15.8 s / 16.8 s / 46 s total). Any 30-sample average that includes post-stall steps is measuring a correctness bug (escalating retry/rebuild), not a performance ceiling.
- `benchmark:power-dam-ui` fails final authority when stalled (power faces gen 10 vs fine 63); the `-performance` lane skips quality gates and exits 0 while broken — do not trust it alone.

### Dispatch overhead is a red herring

- Scratchpad dawn-node probe (`dispatch-probe.mjs`): hazard-chained tiny dispatches cost **~3 µs wall each**. The full unrolled 24-iteration Chebyshev MGPCG schedule (~1,569 dispatches/advance, 85% of all dispatches) ≈ **4.4–6.9 ms** including encode.
- Real preconditioner kernel time: **~230 µs/dispatch at ~1,085 rows** (Chebyshev A/B via `FLUID_CHEBYSHEV_ITERATIONS`: 24 iters → 550 ms pressure block, 2 iters → 276 ms ⇒ ~12.4 ms per outer iteration). Kernels are **execution-bound on hash-probe directory lookups** (`find()` ≤256 probes; 18 probes/row for descriptors), not launch-bound.
- Prior fusion evidence: 17x workgroup cut → 4.9% wall; 21% dispatch cut → 2.3%. Fusions are wall-neutral on this scene.

### Per-phase attribution (GPU timestamp trace, fence-every-step, healthy ~800 ms mini-dam steps)

| Phase | Cost |
|---|---|
| Pressure block (assembly + MGPCG + projection) | ~480–550 ms (solve ~350; assembly+projection ~200–250) |
| Fine SDF redistance | ~177 ms → 12 ms (varies) |
| Transition Delaunay adjacency | ~52 ms |
| Coarse topology | ~40 ms |
| Fine advection | ~19 ms |
| Face-band phases | ~17 ms each |

### Historical reference points

- 2026-07-23 state after smoother/live-face/topology-resolve fusion round: **66.81 ms/advance**, 1,456 dispatches, 244 passes; pressure 45.09 ms = 11.80 solve + 33.30 non-solve (assembly 13.79, transition adjacency 6.54); ~44 ms fine-surface not itemized.
- Paper Table 2 (Aanjaneya 2017): extrapolation is 2–4 s of a 58–81 s step — never their bottleneck, because everything topological is a table read.

## Diagnosis: one disease, three expressions

All stages, one command buffer, no CPU readbacks on the hot path — the cost is not sync, it's fixed per-step work:

1. **Fixed transaction scaffolding per step regardless of delta.** Owner-page generations (3 passes), fine-generation two-phase commit + rollback snapshot + deferred settle (~6 passes), full identity reassignment — every step, even when nothing changed.
2. **Full-domain/capacity dispatches inside nominally-delta stages.** Coarse refine/balance always `.full` (`webgpu-octree.ts:2316-2334`); frontier sort/merge sized at listCapacity (`:959`, `:2362-2374`, `encodeFrontierRows :2628-2666`); identity chain at R=4096 (`fine-levelset-topology.ts:796-808`); brick-residency commit full (`brick-residency.ts:1146-49`).
3. **Per-frame re-derivation of geometry that is fixed per descriptor.** The face band runs **three 10-digit radix sorts per frame** (`sortUniqueFaceBandCatalogSupport1/2/3`) solely to build a sorted directory so `rowOfIdentity` can binary-search (cell,size)→rowIndex, plus nested-binary-search dedup and ~29 serial transition-adjacency passes re-deriving catalog tetrahedra. Smoking gun: `transitionNeighbor` already computes `ownerAt(neighborOrigin)` — the answer — then discards it and binary-searches the directory.

The codebase already contains the correct pattern in two places, proving each fix shape works here:

- **Power-topology** `resolveDescriptor` is O(1) via `sameOrFinerDirect` (2^18) / `sameOrCoarserDirect` (2^9) direct tables, delta-driven indirect.
- **MGPCG persistent-small-domain executor** (`webgpu-octree-spgrid-vcycle.ts`) — why solve inner iterations are as cheap as they are.

The miss is that the rest of the pipeline was never converted to that shape.

## Fix list (ranked, with expected ladder)

Ladder: **246 → ~60–80 ms (T+S+P) → ~25 ms (pass collapse + direct lookups) → 2.5–5 ms (representation endgame).**

### Topology stage (~115 ms probe / 34 ms panel row + related)

- **T1 (S):** whole-stage skip on empty dirty-tile delta — the count already exists in `compaction` at `webgpu-octree.ts:2284`; gate `ownerPages.encode` (`:2245`), reset/refine (`:2295`), frontier (`:2346-74`).
- **T2 (M):** coarse refine/balance delta/indirect variants.
- **T3 (S):** sort/merge sized by dirty count (indirect from `topologyCandidateDispatch`), batch the ~12 indirect-arg copies (each copy fences: `pass-broker:45`).

### Section-5 / face band (~112 ms probe)

- **Face-band redesign (the big one):** identity→rowSlot direct table keyed by owner-arena slot; scatter-first-writer-wins replaces sort+unique (the scatter IS the dedup); topology-build 9→3 dispatches, deleting the three radix sorts (~30 storage round-trips/frame). Validate first: (1) do consumers of `rows`/`commitRows` require identity-sorted contiguous order? if so, ONE terminal sort (still 3→1); (2) ghost/tier flags (`ROW_BOUNDARY_GHOST`, `ROW_SUPPORT1/2/3_*`) must ride the row record, not directory position. Tests: bit-compare band rows/adjacency/incidence vs baseline up to row ordering; mixed-generation guards fail closed.
- **S1 (M):** replace the ~16 serially-dependent Jacobi band-extension rounds (`face-closest-point.ts:1332-40`, rounds = band 4 × fineFactor 4) by reusing the fine-SDF redistance closest-point output.
- **S2 (M):** catalog-indexed adjacency + weights; delete `describeCatalogRows`/`resolveCatalogAdjacency`/`validateCatalogAdjacency` and per-row tetra walks.
- **S3 (S):** fuse Phase C/D commit tails (`:1408-26`, `:1435-82`).
- Mechanical first fix: face band recreates bind groups **per dispatch per step** (`face-closest-point.ts:1127-1137`, ~50+/step uncached).

### Publication (~79 ms probe)

- **P1 (S):** reorder page-delta classification before identity chain; make identity chain indirect on changed-key count.
- **P2 (S):** kill full-domain clears; fold error clear into finalize.
- **P3 (S):** gate brick-residency commit + summary republish on VALUE delta, not band residency — 98% of a 16³ domain is band (4021/4096 bricks), so it currently republishes everything (`brick-residency.ts:1146`, `fine-levelset-summary.ts:284-300`). This is the panel's "Fine-to-coarse restriction" 68 ms + "Residency + sparse publication" 29 ms.

### Solve

- Encode diet: Galerkin encodes 217 dispatches/step; cut encoded cycles to the adaptive estimate.
- Representation endgame (V2 plan, `docs/POWER_LIQUIDS_REPRESENTATION_V2_PLAN.md`): P1 brick 64-bit occupancy masks + popcount row bases; P2 19-coefficient stencil catalog (~614 KB — coefficients are currently re-derived per generation via faces→CSR→19-channel scatter = the 15.8 ms assembly bucket); P3 catalog byte-selector adjacency. Kills the hash-probe execution-bound behavior in solve kernels.

### Known dead gate

- Change-driven topology gate was dead on the UI lane: `hasDenseField = !adaptiveSurfaceAuthority` meant `hasDensePhiSnapshot` false whenever surface pages were on. NOTE: the tree has moved (2026-07-24: `lib/octree-face-band.ts` added, `octree-face-fast-march.ts` deleted, `changeDrivenEligible` symbol no longer greps) — re-verify which items above have already landed before starting.

## Tools & environment

- Throughput authority: `npm run benchmark:power-dam-ui`; attribution only: `profile:power-dam-ui`. Env gates: `FLUID_MAX_ADVANCE_MS`, `FLUID_MAX_DISPATCHES_PER_ADVANCE`, `FLUID_MAX_PRESSURE_NON_SOLVE_MS`.
- Tracing: `FLUID_PHYSICS_TRACE_LOG=1` emits `"record":"physics-trace"` JSON lines; `FLUID_AWAIT_EVERY_STEPS=1` so the 250 ms-throttled timestamp trace resolves per advance; `FLUID_TRACE_DEBUG=1` warns on decode failures. `FLUID_CHEBYSHEV_ITERATIONS` for solve A/B.
- Traps: dynamic-trace boundary markers on the octree secondary encoder report ~0/non-monotonic timestamps — only main-chain boundaries are trustworthy. Dawn-node SIGSEGVs episodically at warmup closest-point extension; clean stale `/tmp/fluid-webgpu-exclusive.lock` if owner PID is dead. Verify no concurrent sessions are editing the tree before benchmarking (`vinext dev` has been live during past measurements).

## Related docs

- `docs/POWER_LIQUIDS_10X_PLAN.md` — phase-ordered plan
- `docs/POWER_LIQUIDS_REPRESENTATION_V2_PLAN.md` — representation endgame
- `docs/papers/aanjaneya-2017-power-liquids.txt` — paper; Table 2 grounds the "topology is table reads" thesis
