# Sparse CM12 pressure-and-tail 5× performance plan

Date: 2026-08-28
Scope: `pressure-topology` through `presentation-publication`
Target hardware: Apple M1 Max / Metal first, portable WebGPU execution model

## Executive decision

The accepted pressure matrix—not SparseWorld pages—is the numerical authority. Keep vectors
in stable-cell space, compile a pressure-owned page schedule for locality, and preserve the
canonical row graph for coefficients, ordering and symmetry. Pages should improve GPU reuse;
they must not be treated as proof of a six-point stencil.

The path to 5× now has three coupled parts:

1. make the current 9–10 ms solve checkpoint causally understood with frozen-state hardware
   profiling;
2. execute the exact row operator page-locally so the 87.5% intra-brick neighbour traffic can
   reuse staged vector data without a duplicate coefficient image;
3. use that pressure-owned coverage and halo authority for a complete symmetric V-cycle that
   reduces 48 iterations to roughly 12–16 in absolute time.

ATEI remains the shared topology-generation authority for downstream consumers, but each
consumer gets an access-specific physical view. PEEI remains the pressure epoch transaction;
it does not compact Krylov vectors or publish a large per-cell neighbour/weight SoA. The
program targets GPU utilization, dependent data access, working-set size and deleted work—not
dispatch count.

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

### 1. The pressure matrix is an accepted masked row graph, not a hidden B8 stencil

The original diagnosis blamed runtime B8 cells missing the six-neighbour arithmetic path.
A post-frame structural census disproved that explanation. At mini32's terminal state all
7,186 pressure cells are authored-template cells and all execute the generic incidence → row
→ term path. Mini64 has only 846 dynamic pressure cells out of 20,442, and none qualifies
for the runtime-interior branch. Adding or removing that branch is timing-neutral.

The generic path is not merely an addressing accident. Mini32 contains 48,556 incidences,
but only 41,976 accepted row incidences; 4,334 pressure cells touch at least one inactive or
zero-`theta` row. Incidence counts range from 6 to 12, 1,557 cells touch multi-term/non-unit
geometry, and 1,391 touch sparse-air or dry-neighbour structure. A world-wide
`hasSolidBoundaries()` flag also disables every authored arithmetic path even though no
accepted mini32 pressure row has a non-unit local solid scale.

The root problem is therefore **global capability invalidation plus repeated traversal of an
irregular accepted matrix**, not runtime-page division by itself. A page is still valuable as
a scheduling and locality unit—87.5% of active off-diagonals remain within a brick—but it is
not a certificate that the operator is a six-point stencil.

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

The fix is one pressure-epoch transaction with pressure-owned page coverage and
access-specific consumers, not another full per-cell coefficient authority.

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

Preserve canonical PCM rank order for work selection and reductions, but keep numerical
vectors in stable-cell space. Rank-only Krylov compaction was slower because every neighbour
then required a reverse-map dependency; stable IDs are already monotone and page-local.

| Plane | Representation |
|---|---|
| Rank → stable cell | existing monotone `u32` pressure stream |
| Solver vectors | existing stable-cell `p, r, z, d, w`, directly addressed by neighbour ID |
| Pressure pages | pressure-owned brick IDs plus exact rank ranges/coverage receipt |
| Matrix authority | accepted active row graph; inactive rows removed only by a proved compiler |
| Optional regular certificate | one small ordered descriptor only for operand tuples proven bit-equivalent |
| Row projection | compact regular face ranks plus ordered exception ports; current `theta` attached |
| Per-cell facts | `submerged`, control volume and any dynamic-solid scale actually read downstream |

A literal per-cell six-neighbour/scalar image is no longer the baseline design. The exact-SoA
arm added up to 30 words per cell and regressed solve time by 8.7%. Any compiled descriptor
must be materially smaller than the topology it removes and preserve the generic row's
operation order. True exceptions continue to use the canonical matrix.

## The 5× budget

| Block | HEAD | Target | Required leverage |
|---|---:|---:|---|
| Pressure topology + RHS | 4.4565 ms | 0.80 ms | delete dead PCA; direct accepted streams; publish PEEI/diagonal/RHS once |
| Pressure solve | 28.1149 ms | 3.80 ms | page-local exact operator; remove duplicate Jacobi; 12–16 MGPCG iterations |
| Projection | 0.3932 ms | 0.25 ms | consume PEEI row/port image; compiled collocation |
| Activity | 2.6870 ms | 0.65 ms | ATEI cell/face tiles; precompiled crossing owner; tile reduction |
| Planning | 2.2282 ms | 0.70 ms | one compact brick-facts record; staged neighbour descriptors |
| Candidate + retirement + presentation | 1.7039 ms | 1.00 ms | delta-local ATEI build; direct point pages; no repeated topology decode |
| **Total** | **39.5837 ms** | **7.20 ms** | **5.50×** |

The target has 0.72 ms of margin below the strict 5× threshold. These are portfolio gates: an individual cut lands on measured merit, but the program is not complete until the total is met.

### Why solve needs both throughput and convergence work

Restoring the old 0.238 ms/iteration at 48 iterations still costs 11.4 ms; that alone misses the entire 7.9 ms tail budget. Conversely, reducing iterations while each dynamic-page application remains pointer-chased leaves too much cost and poor scaling.

The solve target therefore requires:

1. **≤0.22 ms per fixed-budget iteration** from page-local reuse and fewer dependent loads; then
2. **12–16 iterations** at the same true residual using a valid SPD multigrid preconditioner.

The new preconditioner must use a symmetric V-cycle: topology-time aggregation, volume-compatible `R = Pᵀ` (or its explicitly weighted equivalent), Galerkin `A_c`, symmetric pre/post smoothing, and correct Dirichlet/null-space propagation. Do not revive the old aggregate correction that the source itself records as non-SPD.

## Original implementation sequence (superseded)

This sequence records the initial hypotheses. The measured experiment sequence at the end of
the document is authoritative; rank compaction and a literal regular/exception SoA were
rejected by A/B.

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

The first throughput slice validates dead-work removal. It does **not** validate the original
runtime arithmetic diagnosis; the later structural census showed that branch had no live
mini32 consumer.

| Change | mini32 median | Result |
|---|---:|---|
| Original clean pressure solve | 28.1149 ms | 48 iterations, 0.5857 ms/iteration |
| Delete duplicate Jacobi pass | 21.8890 ms | exact true/recursive residual receipt |
| Retire unused PCA numerical work | 1.7695 ms pressure topology | PCA repair/freeze substages are zero |
| Combined retained checkpoint | 9.0440 ms pressure solve | 48 iterations, **0.1884 ms/iteration** |
| Remove unused runtime arithmetic branch | 9.70 vs 9.76 ms median A/B | neutral within one timestamp quantum |

The large solve change between 21.8890 and 9.0440 was originally attributed to the runtime
branch. That attribution is invalid: terminal mini32 executes it zero times, as does the
dynamic-interior class in mini64. The combined checkpoint is real and repeatable, but its
remaining causal split is unresolved. Do not use the branch as evidence for a page-stencil
architecture or claim its delta as an isolated win.

The current mini32 sum from pressure topology through presentation is approximately
**14.0904 ms**, down from 39.5837 ms (**2.81×**). Fixed-iteration operator throughput
already beats the Phase 3 gate of 0.22 ms/iteration. The next critical path is therefore
Phase 4's SPD preconditioner: at current throughput, reducing 48 applications to 16 would
put the solve near 3.0 ms and the measured tail near the 5× boundary.

Two cleanup items remain intentionally separate from the accepted numerical cut:

- legacy PCA storage and QA hash surfaces still exist, although their numerical entry
  points and recurring work are retired;
- every measured mini32 pressure cell uses the canonical generic incidence graph. It should
  move only when a structurally proved consumer wins against this measured baseline.

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

Plan revision: do not layer a coarse correction onto the existing PCG recurrence. A future
multilevel arm must make restriction, smoothing, coarse solve and prolongation replace fine
vector/operator traffic inside a symmetric V-cycle. It must beat the Jacobi fixed-state solve
in absolute GPU time, not merely reduce iterations. The rank-compaction experiment below
also rules out assuming that global pressure-rank vectors are a prerequisite.

## PEEI checkpoint — reject rank-only Krylov compaction

The next arm kept pressure, RHS and diagonal in stable-cell space while moving every
recurring Krylov field (`r`, `z`, `d`, `Ad`, `Az` and guarded residual) into canonical
pressure-rank order. The retired frozen cell-brick plane supplied a stable-cell-to-rank
map without increasing allocation. The operator then exchanged each membership probe plus
stable vector read for one reverse-rank read plus one compact vector read.

This was numerically contained and compiled across all resident shader entry points, but
mini32 pressure solve rose to **10.3547 ms median** (11.9276 ms p95) with the same fixed
iteration budget, versus the committed 9.0440 ms median. The arm was removed.

The result invalidates the assumption that vector address compaction alone is useful here.
Canonical stable IDs are already monotonically ordered and page-local; paying a reverse-map
dependency for every neighbour costs more than the denser vector load saves. The revised
PEEI rule is therefore: **compile the operator and its vector address together**. A future
rank-space arm must publish neighbour ranks (or a direction-major regular stencil) so a hot
SpMV never performs stable-to-rank lookup. Until then, keep the proven stable-cell canonical
operator and direct the next work at removing downstream generic topology traversal.

## Solve-envelope checkpoint — retain cadence-8 true residual guards

Reducing the guarded true-residual cadence from 8 to 16 removed three full `b-Ap`
applications before mini32's iteration-48 crossing and lowered its solve median to 7.8643 ms.
That local result did not survive the symmetric-expansion A/B. The baseline's cadence-8
drift recovery is numerically active there: cadence 16 changed the evolving fields and
topology and raised maximum pressure relative residual from **2.3756e-6** to **1.6210e-5**.
Cadence 32 also delayed mini32 convergence from iteration 48 to 64. Both arms were removed.

The convergence envelope is therefore part of the solver, not dispatch bookkeeping. Future
solve work must reduce the cost of the guarded operator application itself or improve the SPD
preconditioner while preserving the cadence-8 recovery trajectory exactly.

## Preconditioner checkpoint — reject accepted-leaf block polynomial

A two-step damped-Jacobi block map was implemented with one workgroup per accepted brick.
Its intermediate vector occupied 2 KiB of workgroup memory, so the second local operator
application read neighbour values on chip. It nevertheless failed on both tested damping
bounds: ω=2/3 and the conservative ω=1/4 each triggered all 16 curvature guards, missed the
pressure tolerance, changed the evolving census, and cost 18.2190–20.3817 ms.

This exposed two non-negotiable requirements for the eventual V-cycle:

- its brick work stream must be pressure-owned and prove complete coverage of the PEI cell
  list; the accepted-leaf manifest is not that authority;
- its local operator must be compiled alongside that stream. Re-entering canonical
  incidence/row/term storage for a second intra-brick traversal more than doubles solve cost,
  even when the vector itself is in workgroup memory.

Do not retry block smoothing on accepted topology leaves. The next solve architecture must
first publish pressure-brick ranges plus direct regular/exceptional local terms, and validate
that every pressure rank is covered exactly once before any numerical kernel consumes it.

A follow-up publication-only arm proved that PEI can derive the exact 7,186-cell wet-brick
stream from finalized membership, but the serial-in-one-workgroup brick scan added
0.065–0.13 ms to pressure topology. With no numerical consumer yet, that arm was also
removed. The stream and its direct operator must land atomically and show a net frame win;
dormant execution-image planes are not accepted as “infrastructure.”

## Exception-operator checkpoint — reject pairwise and exact-SoA packing

Two attempts isolated the remaining generic incidence path after B8 interior arithmetic:

| Arm | Pressure solve median | Structural cost |
|---|---:|---|
| Committed arithmetic baseline | 9.0440 ms | none |
| Pairwise-row shortcut inside generic SpMV | 9.4372 ms | branch and alternate code shape |
| Exact six-slot SoA exception image | 9.8304 ms | five scalar planes, neighbour plane and class plane |

The exact-SoA arm compiled each pressure rank's at-most-six incidences during coefficient
publication. To preserve canonical arithmetic it stored neighbour, dual weight, own
coefficient, other coefficient and `theta` separately; SpMV retained the exact
`dual*own*other*value/theta` order. Pressure topology remained 1.1141 ms, but solve regressed
8.7%. The arm added about 100 lines and a worst-case 30 words per cell, so it failed both the
absolute-time and structural-simplicity gates and was removed.

This is stronger evidence than “the first packing was imperfect.” The in-place pairwise arm
had no publication or allocation cost and still regressed. The remaining generic path is not
made faster by adding a second per-cell representation with a selection branch. Any next
operator experiment must increase the arithmetic fast-path domain using a much smaller
topology-time descriptor, leaving true exceptions on the existing canonical path.

## Architectural conclusions after falsification

### Measured matrix shape

The temporary classifier was run after timing and then removed; none of its buffers,
pipelines or entry points remains in production.

| Terminal mini32 fact | Count |
|---|---:|
| Pressure cells | 7,186 |
| Authored generic / runtime generic | 7,186 / 0 |
| Incidences / accepted row incidences | 48,556 / 41,976 |
| Active off-diagonals / cross-brick off-diagonals | 45,222 / 5,636 |
| Cells touching inactive or zero-`theta` rows | 4,334 |
| Cells touching multi-term/non-unit rows | 1,557 |
| Cells touching sparse-air or dry-neighbour structure | 1,391 |
| Six-incidence cells | 3,466 |

Mini64 confirms the same direction: 19,596 of 20,442 pressure cells are authored generic;
the remaining 846 are dynamic generic, with zero certified dynamic interiors.

### 1. Separate mathematical authority from locality scheduling

The mathematical object is the accepted masked row graph. Bricks/pages are a useful physical
schedule because 87.5% of active off-diagonals stay within a brick, but they do not imply a
six-point matrix. Future code must say which role a structure serves:

- the row graph owns coefficients, active membership, order and symmetry;
- stable IDs own vector addresses and are already monotone/page-local;
- pressure pages may own workgroup scheduling and vector reuse;
- a compact stencil descriptor is only an optimization certificate for a proved subset.

Conflating these roles caused both the runtime-B8 misdiagnosis and the accepted-leaf block
coverage failure.

### 2. Global capability flags destroy local regularity

`hasSolidBoundaries()` selects the generic operator for the whole world. In mini32 that makes
all 7,186 cells generic even though every accepted pressure row observed by the census has an
exact unit local solid scale. Similar global authored/runtime splits force hot kernels to
discard row-local facts already known at pressure publication.

The architectural fix is not to ignore solids. It is to compile a **local certificate from
the accepted numerical row image**. A world may contain solids while a particular pressure
row remains exactly regular. Local facts should select the physical path; global flags should
only select which compiler is needed.

### 3. Packing must preserve operands and operation order, not just the final coefficient

Rank-only vectors, pairwise shortcuts and the five-scalar exact SoA image all increased
dependencies or working-set size and lost. A later one-word mask appeared to classify 3,466
cells and made the broken solve very fast, but it mapped incidence ordinal directly to axis
direction. Canonical incidence order is not a direction-order contract; the arm produced zero
accepted iterations, curvature recovery and a relative residual of 6.75, so it was removed.

Even fixing that permutation is insufficient as a proof. The generic expression evaluates
`dual * own * other * value / theta`; equality of a precombined weight does not guarantee
bit-equivalent multiplication order. A valid compact class must certify the individual
operand tuple, encode directions in canonical incidence order, and pass a dual-operator
bitwise audit before it enters PCG.

### 4. Page-local reuse is still plausible, page-stencil replacement is not

Only 5,636 of 45,222 active off-diagonals cross a brick. A pressure-owned page kernel could
therefore stage most input-vector values once in workgroup memory while executing the exact
canonical row loops and arithmetic order. This attacks repeated vector traffic and dependent
global loads without duplicating all coefficients. It also creates the correct coverage and
halo authority for a later multilevel solver.

This is different from the rejected accepted-leaf block polynomial: pressure membership,
not accepted topology, must compile the page work stream, and the first consumer is an
ordinary baseline-equivalent SpMV rather than an additional numerical pass.

### 5. The remaining 5× gap is primarily convergence, but extra fine passes do not help

The current tail is about 14.09 ms versus the 7.92 ms 5× threshold; pressure solve alone is
about 9–10 ms at 48 iterations. The additive brick correction reduced the iteration count to
32 but was slower in absolute time, while the block polynomial lost positive curvature.

The required preconditioner must replace fine work inside a symmetric V-cycle. It cannot add
restriction, a second fine operator and prolongation beside every existing Jacobi iteration.
Its complete application must be cheaper than the operator applications it removes, use the
pressure-owned coverage stream, and keep cadence-8 true-residual guards unchanged.

### 6. Publication is charged to its consumer

The exact wet-brick stream cost 0.065–0.13 ms with no consumer; the exact-SoA image enlarged
code and data while slowing SpMV. New images land atomically with their first consumer and are
kept only on net frame time. There is no permanent “infrastructure” arm.

### 7. Retained-change audit

- Keep duplicate Jacobi removal: it deletes one recurring full-vector read/write and has an
  exact numerical A/B.
- Keep unused PCA scheduling removal: production Jacobi never consumed its aggregate or
  hierarchy results, and pressure topology fell from 3.80 ms toward 1.11 ms.
- Keep the unused PCG initialization-reduction deletion: it removes dead shader work and
  state without changing output.
- Remove the runtime B8 arithmetic branch: the census found no consumer and add/remove timing
  was neutral.
- Remove the frozen cell→brick publication: its accessor and stored plane had no consumer.
- Keep no failed rank, SoA, pairwise, block, cadence or mask representation.

The 9.044 ms combined solve checkpoint remains valid, but only 6.226 ms of its improvement is
currently isolated to duplicate-Jacobi removal. The rest must not be attributed to the now
proven-dead runtime branch. A frozen-state ablation is required before making another causal
claim.

## Revised experiment sequence

### Experiment A — make the current checkpoint causal

Capture one frozen accepted pressure epoch and time the existing exact operator, vector update
and reductions separately with hardware timestamps and Metal utilization counters. Ablate the
remaining source/scheduling differences between the 21.889 and 9.044 ms checkpoints without
changing the matrix or iteration trajectory. Report bytes, dependent loads, SIMD occupancy,
cache behaviour and bandwidth; dispatch count is not an objective.

Gate: account for the unexplained solve delta. If it cannot be reproduced on the frozen epoch,
replace the 9.044 checkpoint with a new paired baseline before doing more design work.

### Experiment B — pressure-owned page-local exact SpMV

Publish the exact pressure-page coverage stream together with its consumer. Run one workgroup
per pressure page, stage page-local input-vector values in workgroup memory, and execute the
existing incidence/row/term loop in identical order. Cross-page values remain direct stable-ID
loads. Do not publish neighbour/weight planes and do not change PCG mathematics.

Gate: exact mini32 pressure receipt, bit-identical symmetric-expansion fields relative to its
five-failure baseline, complete one-owner pressure coverage, pressure solve at most 8.2 ms and
topology increase at most 0.1 ms. If this misses, remove it and revisit the architecture before
any multilevel work—the 87.5% locality premise would not be translating into GPU utilization.

### Experiment C — ordered local regular certificate

Only after the exact page kernel establishes a reliable baseline, census individual operand
tuples and canonical direction permutations. If a large class has exact discrete operands,
encode one small ordered descriptor and verify baseline and compiled `A*x` bitwise on multiple
full pressure vectors before routing that class through the hot operator. Never infer safety
from a combined coefficient or six-incidence count.

Gate: at least 0.8 ms net solve improvement, no larger topology working set than the topology
reads removed, and the same numerical gates as Experiment B. Otherwise remove the descriptor.

### Experiment D — page-native symmetric V-cycle

Reuse the proven pressure-page coverage and exact halo path for one complete symmetric V-cycle.
Compile volume-compatible restriction/prolongation and a Galerkin coarse operator from the
accepted row graph. Benchmark the entire preconditioner before PCG integration.

Gate: absolute mini32 solve at or below 3.5–4.0 ms with cadence 8, zero curvature recovery and
the same true-residual and symmetry receipts. Iteration reduction alone is a rejection.

### Experiment E — downstream access-specific consumers

Projection already uses regular/seam face programs. Collocation and activity should receive
their own compact page-local views derived from the same accepted topology generation, not the
pressure matrix layout. Each view must remove generic traversal and win its own stage A/B. The
portfolio gate remains ≤7.92 ms median from pressure topology through presentation.

### Symmetric-expansion control

The cleaned working tree and committed `3a88f82a` baseline produce identical 20-step results:
the same fields and maxima, plus the same five existing gates (four step-1 publication/setup
failures and the step-18 mass-drift threshold). The cleanup therefore introduces no symmetric
regression; future arms compare against this five-failure control.
