# Sparse CM12 pressure-and-tail 5× performance plan

Date: 2026-08-28
Scope: `pressure-topology` through `presentation-publication`
Target hardware: Apple M1 Max / Metal first, portable WebGPU execution model

## Executive decision

The next performance program should be built around two compiled images:

1. **Accepted Topology Execution Image (ATEI)** — compiled when candidate topology is accepted, normally at the previous frame's tail. It owns one topology epoch and publishes access-specific physical views: compact cell tiles, packed point-owner pages, split regular/seam face programs, and brick-neighbour descriptors.
2. **Pressure Epoch Execution Image (PEEI)** — compiled after scalar conditioning from ATEI plus current pressure membership and `theta`. It publishes compact canonical-rank vectors, regular neighbour/weight planes, ordered exceptional terms, diagonal, RHS inputs, and one submerged flag.

The pressure solve, projection, collocation, activity census, planning, and presentation then consume those images directly. They must not re-enter the generic authored/dynamic cell → incidence → row → term machinery.

This is the credible path to 5×. Dispatch-count work is explicitly not the program: the gains come from deleting unused computations, turning dependent pointer chains into coalesced SoA reads, keeping hot state in compact pressure-rank order, and reducing operator applications with a mathematically valid preconditioner.

## What HEAD actually costs

I captured clean `HEAD` `1daf91d4` with the repository's hardware timestamp lane:

```text
NODE_OPTIONS=--max-old-space-size=8192 \
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
FLUID_WEBGPU_BACKEND=metal \
node --import tsx tools/probe-sparse-cm12-stage-cost.ts \
  --scene=mini32 --brick-fine=8 --presentation-page=8 \
  --warmup=8 --frames=24 --final-qa=0 --quiet=1
```

The run had zero WebGPU validation errors and passed its diagnostic receipts.

| Stage | HEAD median | p95 |
|---|---:|---:|
| Pressure topology | 3.8011 ms | 9.2406 ms |
| Pressure RHS | 0.6554 ms | 2.7525 ms |
| Pressure solve | 28.1149 ms | 48.4966 ms |
| Velocity projection | 0.3932 ms | 2.1627 ms |
| Activity measurement | 2.6870 ms | 7.3400 ms |
| Resolution planning | 2.2282 ms | 5.6361 ms |
| Candidate transfer | 1.1796 ms | 2.6214 ms |
| Brick retirement | 0.0655 ms | 0.0655 ms |
| Presentation publication | 0.4588 ms | 3.3423 ms |

The sum of these stage medians is **39.5837 ms**. A 5× result therefore needs to be at or below **7.9167 ms** under the same measurement contract.

### The SparseWorld solve regression is real

| Receipt | Pressure cells | Iterations | Solve | Cost / iteration |
|---|---:|---:|---:|---:|
| Pre-change `10c691bb` | 6,150 | 40 | 9.5027 ms | 0.2376 ms |
| HEAD `1daf91d4` | 7,186 | 48 | 28.1149 ms | 0.5857 ms |

Pressure cells increased 16.8%; normalized iteration cost increased **146.5%**, or **2.46×**. Population growth is not the explanation.

The pre-change source is the checked-in [August 27 receipt](../artifacts/sparse-cm12-mini32-b8-p8-20260827-planning-wdr-bracket-control-stage-cost.json). The HEAD numbers above are from the clean capture made for this analysis.

## Root causes

### 1. Runtime B8 pressure cells miss the arithmetic operator

`applyOperator` permits only authored cells to enter its six-neighbour fast path:

```wgsl
if (cell < ta(2u)) {
  // arithmetic interior
}
// every runtime page falls through to incidence -> row -> term
```

See [webgpu-sparse-cm12-resident.wgsl.ts](../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts#L3871).

SparseWorld runtime pages are uniform B8 pages, but every recurring SpMV reconstructs their six incidences, row identity, term identity, membership, `theta`, dual weight, coefficient, and neighbour value. The dynamic accessors repeatedly divide IDs into page/local coordinates and reload the page base; see [sparse-cm12-row-access.wgsl.ts](../lib/methods/adaptive-mass/sparse-cm12-row-access.wgsl.ts#L87) and its dynamic incidence reconstruction at [line 196](../lib/methods/adaptive-mass/sparse-cm12-row-access.wgsl.ts#L196).

This is the direct mechanism behind the per-iteration regression: the new cells are structurally regular but execute as maximally general topology.

### 2. A full pressure-vector pass recomputes an unchanged value

`updatePipelinedState` already writes Jacobi `z = residual / diagonal`. The immediately following `applyJacobiPreconditioner` rereads residual and diagonal and writes the same `z`, with no intervening writer. The pass is visible in [the solve schedule](../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts#L5635) and its shader is [here](../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts#L3852).

Delete it. This is pure recurring vector traffic, not a numerical experiment.

### 3. Pressure topology builds coarse data that production never consumes

Production currently uses the scalar positive Jacobi preconditioner. The source defines `persistentBrickAggregate*`, `persistentHierarchy*`, wet-brick, and hierarchy-token accessors, but no numerical kernel calls them. Nevertheless, every pressure epoch seeds, repairs, validates, and freezes the aggregate and hierarchy caches in [webgpu-sparse-cm12-resident.ts](../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts#L5465).

The measured dead substages are:

| Unused product | Median |
|---|---:|
| PCA brick + aggregate-edge repair | 0.5898 ms |
| PCA hierarchy repair + freeze | 0.4588 ms |
| **Directly removable subtotal** | **1.0486 ms** |

This old cache must be deleted before a new preconditioner lands. A future hierarchy needs its own SPD contract; retaining unused non-SPD state “for later” only obscures ownership and consumes bandwidth.

### 4. Pressure publication re-derives topology instead of compiling execution data

The current pressure stage separately publishes PCM cells, PCM rows, PEI membership, fine coefficients, coarse caches, hierarchy caches, then returns to generic incidence for `preparePressure`. The largest current substages are PCM cells (0.9175 ms), fine coefficients (0.5243 ms), and PCM rows (0.3932 ms).

Runtime-page coefficient publication also walks generic incidence. Once any dynamic leaf exists, the old authored directed-edge image is deliberately not authoritative across the host/runtime seam. Yet its cache lifecycle survives around the replacement path.

The fix is one pressure-rank compiler, not another repair authority.

### 5. Post-solve stages throw away the compiled face program

Projection has already proved the split regular/seam face program. Immediately afterward, collocation and activity return to generic incidence and row-term traversal; see [collocateAndDiagnose](../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts#L4451) and [measureBrickActivity](../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts#L4796).

Activity now also performs detailed per-face phase tests to establish one geometric crossing owner. That ownership is structural for a fixed accepted epoch and belongs in ATEI. Recomputing it from density samples and topology addressing in every frame mixes topology compilation with numerical census.

## Target data model

The design rule is **one logical authority, multiple physical views**. TEI's failed face-sampling experiment already proved why a monolithic representation is insufficient: sharing authority is good; forcing every consumer through the same physical access shape is not.

### ATEI: immutable for an accepted topology generation

| View | Hot layout | Consumers |
|---|---|---|
| Cell tiles | compact rung-major tile ordinals → stable tile/cell base; descriptor SoA | transport, collocation, activity |
| Point owners | one packed finest 4³ page lookup → stable leaf/tile; u32 identity | sampling, presentation, diagnostics |
| Regular faces | arithmetic brick/tile programs; no exception-range read | pressure publication, projection, activity |
| Exceptional faces | compact ordered seam/sparse-air/2:1 packets | pressure, projection, activity |
| Brick descriptors | origin, rung, cell base, six/26 neighbour IDs, valid extent | planning, ownership, cross-page arithmetic |
| Static facts | wall/seam flags, geometric crossing owner, D4 family mapping | diagnostics, activity, symmetry |

ATEI should be compiled in candidate transfer before the selector flip, double-buffered with the topology generation. Changed pages plus their face neighbours are rewritten; quiescent topology does no structural work.

Do not use global `u16` interning for stable identities. The current ocean B8 build fails with `IRL1 u16 identity capacity exceeded`; pack local fields where profitable, but keep global identities as `u32`.

### PEEI: immutable for one pressure epoch

Preserve canonical PCM rank order so the established 64-lane reduction tree does not change. Store numerical vectors by **pressure rank**, not stable cell ID.

| Plane | Representation |
|---|---|
| Rank → stable cell | one `u32` per pressure rank, used only at frame boundaries/diagnostics |
| Stable cell → rank | page-local rank map or `INVALID`, compiled once per pressure epoch |
| Solver vectors | compact SoA `p, r, z, d, w`, diagonal and RHS indexed directly by rank |
| Regular operator | six direction-major neighbour-rank `u32` planes and exact `f32` weight planes |
| Exceptions | one rank-local range plus compact ordered neighbour-rank/weight records |
| Row projection | compact regular face ranks plus ordered exception ports; current `theta` attached |
| Per-cell facts | `submerged`, control volume and any dynamic-solid scale actually read downstream |

For 7,186 mini32 pressure cells, a literal six-neighbour/weight image is only about 345 KiB. At the historical 245k-cell ocean scale it is about 11.2 MiB—small enough to justify direct, coalesced access and still far below the current general topology working set. If regular weight classes can be coded without adding hot unpacking, do so after the direct baseline wins.

Regular and exceptional planes are physically separate. A regular lane never reads an exception range; exceptional terms preserve canonical arithmetic order.

## The 5× budget

| Block | HEAD | Target | Required leverage |
|---|---:|---:|---|
| Pressure topology + RHS | 4.4565 ms | 0.80 ms | delete dead PCA; direct accepted streams; publish PEEI/diagonal/RHS once |
| Pressure solve | 28.1149 ms | 3.80 ms | compact vectors; direct operator; remove duplicate Jacobi; 12–16 MGPCG iterations |
| Projection | 0.3932 ms | 0.25 ms | consume PEEI row/port image; compiled collocation |
| Activity | 2.6870 ms | 0.65 ms | ATEI cell/face tiles; precompiled crossing owner; tile reduction |
| Planning | 2.2282 ms | 0.70 ms | one compact brick-facts record; staged neighbour descriptors |
| Candidate + retirement + presentation | 1.7039 ms | 1.00 ms | delta-local ATEI build; direct point pages; no repeated topology decode |
| **Total** | **39.5837 ms** | **7.20 ms** | **5.50×** |

The target has 0.72 ms of margin below the strict 5× threshold. These are portfolio gates: an individual cut lands on measured merit, but the program is not complete until the total is met.

### Why solve needs both throughput and convergence work

Restoring the old 0.238 ms/iteration at 48 iterations still costs 11.4 ms; that alone misses the entire 7.9 ms tail budget. Conversely, reducing iterations while each dynamic-page application remains pointer-chased leaves too much cost and poor scaling.

The solve target therefore requires:

1. **≤0.22 ms per fixed-budget iteration** from compact vectors and the direct operator; then
2. **12–16 iterations** at the same true residual using a valid SPD multigrid preconditioner.

The new preconditioner must use a symmetric V-cycle: topology-time aggregation, volume-compatible `R = Pᵀ` (or its explicitly weighted equivalent), Galerkin `A_c`, symmetric pre/post smoothing, and correct Dirichlet/null-space propagation. Do not revive the old aggregate correction that the source itself records as non-SPD.

## Implementation sequence

### Phase 0 — Make the regression causal

- Add `adaptive-mass` to the Metal utilization profiler; record occupancy, SIMD-active lanes, L2 read hit, read/write bandwidth, ALU utilization, registers, and workgroup memory.
- Add a frozen dynamic-page pressure snapshot with fixed 48 iterations and a pressure census split by authored/runtime, regular/exceptional, and solid/free-surface class.
- Separate SpMV+partial, vector update, redundant Jacobi, and global reduction timings.
- Replace IRL1's global u16 identity assumption so ocean constructs and becomes a mandatory negative-control lane.

Gate: reproduce the current 0.586 ms/iteration and show whether the general operator is latency-bound, cache-bound, or both. Do not proceed from inferred counters.

### Phase 1 — Delete work with no numerical effect

- Remove `applyJacobiPreconditioner` from ordinary iterations after a same-run/raw-bit proof.
- Remove PCA aggregate/hierarchy repair, freeze, storage, PEI wet-brick/hierarchy streams, indirects, QA, and shader helpers that have no consumer.
- Retain only the current pressure-cell/row truth and fine numerical products needed until PEEI replaces them.

Expected result: about 1.05 ms from topology plus one complete pressure-vector read/write per iteration. This phase should reduce code and allocation as well as time.

### Phase 2 — Compile ATEI at topology acceptance

- Promote the accepted BFA/BTI/TEI work into one compiler with separate SoA views.
- Segment uniform runtime B8 tiles from authored/adaptive exception streams. Hot kernels must not branch on `id < host` for every access.
- Compile same-page and cross-page face identities, seam packets, brick neighbours, and geometric crossing ownership once.
- Make every view carry the same selector and generation; remove consumer-specific structural authorities as each view lands.

Gate: exact cell/row/point coverage on equal-rung, dynamic↔dynamic, dynamic↔authored, 2:1, clipped, sparse-air and solid fixtures. Compile cost is charged to changed topology, never hidden in a consumer stage.

### Phase 3 — Publish PEEI and cut over pressure assembly

- Ballot pressure membership per ATEI tile and produce the canonical rank stream.
- Build stable↔rank maps and compact numerical vector planes.
- Publish regular neighbour ranks/weights and compact exception terms directly from ATEI face programs.
- Compute `theta`, diagonal, RHS contribution, control volume and submerged once. Eliminate repeated row membership and `pressureCellSubmerged` walks.
- Make `preparePressure`, initial residual, recurring SpMV and projection consume PEEI only.

Gate: fixed 48-iteration cost ≤0.22 ms/iteration on mini32 and no worse than the pre-SparseWorld normalized cost on mini64/ocean. Canonical reduction order, true residual, projected velocity and divergence must match the declared exactness contract.

### Phase 4 — Install the SPD multigrid preconditioner

- Reuse ATEI structure, not the deleted PCA cache.
- Compile hierarchy structure only on topology change; refresh pressure-dependent coefficients per pressure epoch.
- Start with a topology-complete Galerkin hierarchy and symmetric smoother. Incremental hierarchy patching is later work.
- Keep outer PCG and its true-residual guard as the single convergence authority.

Gate: ≤16 iterations on mini32, mini64, long-dam and ocean at unchanged tolerance, no curvature recovery, and bounded iteration count across refinement.

### Phase 5 — Consume ATEI through the frame tail

- Collocation reads regular face slots and exception ports directly; its per-tile partial also publishes the velocity-change brick fact.
- Activity reads compiled endpoint pairs/crossing owners and performs coherent tile-local density/velocity comparisons. No incidence/row-term lookup.
- Planning reads one compact per-brick fact record and stages the neighbour descriptor block once per workgroup.
- Candidate transfer rewrites only delta pages and adjacent face programs.
- Presentation uses packed point-owner pages and staged tile descriptors; retain a dense derived cache wherever measured reuse beats on-demand ownership, as face preparation already demonstrated.

Gate: activity ≤0.65 ms, planning ≤0.70 ms, candidate+retirement+presentation ≤1.0 ms under the current mini32 timing contract.

## What not to do

- Do not optimize around dispatch count. The current regression is cost per useful pressure cell and per operator application.
- Do not send every consumer through TEI's physical layout. The direct TEI face sampler was approximately 3× slower; share authority, not necessarily storage order.
- Do not add workgroup-memory tile staging to SpMV without a fused canonical reduction proof. The prior split-tile arm regressed ocean by about 10.5%.
- Do not globally reorder pressure ranks. Prior 4³ and subgroup-major reorders either regressed or crossed the residual gate.
- Do not compact bytes while leaving dependent gathers unchanged. The previous compact-worklist experiment removed tens of MiB without a solve win.
- Do not keep old and new production authorities. Each accepted cut deletes its predecessor.
- Do not use `u16` for a world-scale identity or pack `f32` coefficients to `f16`; both violate current capacity or numerical contracts.
- Do not claim a 5× result by changing the pressure tolerance, accepted resolution, represented liquid, or iteration receipt.

## Measurement and acceptance contract

Every phase reports:

- fixed-state pressure cells, rows, regular faces, exceptional terms, authored/runtime split;
- topology changed vs quiescent frames;
- true and recursive residuals, maximum divergence, iterations and curvature recovery;
- GPU median/p95 for topology, RHS, solve, projection, activity, planning and publication;
- time per fixed-budget operator application;
- bytes read/written per operator application;
- occupancy, SIMD-active lanes, cache hit, bandwidth, ALU, registers and workgroup memory;
- allocation and lines/authorities deleted.

Required scenes:

1. dynamic-page mini32 fixed snapshot — regression reproducer;
2. mini64 — scaling and partial-pressure-page behavior;
3. symmetric expansion — exact/canonical-order gate;
4. dynamic↔authored seam fixture and dormant-receiver transition;
5. long dam — diameter and topology churn;
6. ocean — high wet occupancy and capacity negative control;
7. static-solid and moving-solid fixtures — dynamic coefficient path.

Final acceptance is **≤7.9 ms median from pressure topology through presentation**, **≤10 ms p95**, unchanged pressure tolerance and physical gates, no new authority/fallback, and no dense logical-world allocation. The 7.2 ms design budget is the working target.

## Evidence limitations

- The 39.5837 ms baseline is the sum of per-stage medians, not the median of a paired per-frame tail sum. The implementation gate should add a direct tail-span timestamp.
- The pre-change comparison is closely matched but not a frozen identical state. Its population-normalized result strongly localizes the regression; Phase 0 makes it causal.
- The current xctrace harness cannot profile `adaptive-mass`, so this report does not invent utilization percentages.
- Current ocean B8 construction fails its u16 IRL1 capacity guard; no HEAD ocean timing is claimed.

## Implementation checkpoint — 2026-08-28

The first throughput slice validates the plan; no architectural revision is required yet.

| Change | mini32 median | Result |
|---|---:|---|
| Original clean pressure solve | 28.1149 ms | 48 iterations, 0.5857 ms/iteration |
| Delete duplicate Jacobi pass | 21.8890 ms | exact true/recursive residual receipt |
| Retire unused PCA numerical work | 1.7695 ms pressure topology | PCA repair/freeze substages are zero |
| Arithmetic SparseWorld B8 page interiors | 9.0440 ms pressure solve | 48 iterations, **0.1884 ms/iteration** |

The dynamic-interior result also scales on mini64: solve median fell from 18.8088 ms to
8.7818 ms for the same 14,320 pressure cells, 48 iterations, true residual, recursive
residual, and zero curvature recovery.

The current mini32 sum from pressure topology through presentation is approximately
**14.0904 ms**, down from 39.5837 ms (**2.81×**). Fixed-iteration operator throughput
already beats the Phase 3 gate of 0.22 ms/iteration. The next critical path is therefore
Phase 4's SPD preconditioner: at current throughput, reducing 48 applications to 16 would
put the solve near 3.0 ms and the measured tail near the 5× boundary.

Two cleanup items remain intentionally separate from the accepted numerical cut:

- legacy PCA storage and QA hash surfaces still exist, although their numerical entry
  points and recurring work are retired;
- page-boundary, seam, solid, and authored exception cells still use the canonical generic
  incidence graph. They should move only when an access-specific compiled exception image
  wins against this measured arithmetic baseline.

## Phase 4 checkpoint — reject additive brick correction

A topology-complete SPD additive brick correction was tested before committing to a full
V-cycle:

```text
M^-1 = D^-1 + P diag(P^T A P)^-1 P^T
```

`P` contained one constant basis vector per accepted authored or runtime brick, and the
Galerkin diagonal used the canonical fine operator. The map reduced mini32 from 48 to 32
iterations with zero curvature recovery, but every physical implementation missed the
solve-time gate:

| Physical implementation | Iterations | Solve median | p95 |
|---|---:|---:|---:|
| Committed Jacobi baseline | 48 | 9.0440 ms | 11.2067 ms |
| Separate restrict/prolong vector passes | 32 | 10.4858 ms | 11.1411 ms |
| Fused brick update, full accepted-cell scan | 32 | 9.3061 ms | 17.5636 ms |
| Fused brick update, exact pressure-rank intervals | 32 | 9.9615 ms | 18.2190 ms |

The first arm retained the same 7,186-cell final census; the fused arms changed the evolving
mini32 trajectory and ended at 6,047 pressure cells, so their apparent medians are not a
fixed-state speedup. More importantly, even the same-census arm was slower: the local
reduction and second `z` traversal cost more than the sixteen avoided operator applications.

Plan revision: do not layer a coarse correction onto stable-cell PCG storage. Complete the
compact PEEI pressure-rank vectors and hierarchy planes first, then make restriction,
smoothing, coarse solve and prolongation replace fine-vector/operator traffic inside a
symmetric V-cycle. A future preconditioner arm must beat the Jacobi fixed-state solve in
absolute GPU time, not merely reduce iterations.
