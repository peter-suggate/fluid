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
- **MGPCG persistent-small-domain executor** (`webgpu-octree-mgpcg.ts:97-117`) — why solve inner iterations are as cheap as they are.

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
