# Sparse CM12 pressure performance — implementation handoff

Date: 2026-08-16. Scope checked against the current working tree at HEAD
`c418b265`; the tree has unrelated staged and unstaged user changes, so this
document does not claim a clean-tree receipt.

## 0. Scope and decision

This handoff covers the **non-spherical** Sparse CM12 pressure path:

- the global composite finite-volume operator over accepted sparse cells and
  face/seam rows;
- the resident GPU Jacobi-preconditioned CG solve;
- pressure warm starts and topology transitions;
- pressure-specific diagnostics, benchmarks, memory traffic, reductions, and
  command structure;
- a composite multigrid preconditioner and sparse-native specialization for
  regular regions.

It deliberately excludes the spherical separating-boundary projected-Jacobi
branch and all LCP/active-set work. It also excludes transport, sharpening,
gamma diffusion, rendering, and resolution-policy changes except where their
published topology is an input to pressure.

The implementation direction is:

1. **Make the GPU convergence receipt trustworthy before tuning.** The current
   resident path reports the recursively updated CG residual, not a fresh
   `b - A p`, and does not apply the CPU oracle's per-component null-space
   projection.
2. **Remove work that is structurally unnecessary.** Compact the pressure
   workset to live liquid cells/rows, bake effective edge weights once per
   pressure epoch, and make the warm start and early-stop behavior observable.
3. **Reduce synchronization cost only after the baseline is stable.** The
   current solver pays two global reductions and five dependent dispatches per
   iteration.
4. **Make convergence grid-independent.** The production endpoint is
   composite multigrid-preconditioned CG (MGPCG), retaining the same finest
   `G^T W G` operator.
5. **Keep one sparse pressure architecture.** Do not add a dense fallback,
   convert a frame to a dense logical lattice, or switch to the Uniform solver.
   Recover performance on regular regions by classifying their rows and using
   sparse-native regular kernels inside the same operator and hierarchy.

No performance change may land by weakening the pressure tolerance, changing
the finest operator, dropping wet bulk from the pressure graph, treating seams
as boundaries, or comparing solves at different true residuals.

---

## 1. Verified current implementation

### 1.1 Operator and pressure representation

`lib/methods/adaptive-mass/sparse-atlas-composite-projection.ts` defines the
pressure algebra. Every accepted face or seam port is one row of `G`, with dual
weight `W`. The unconstrained free-surface operator is

```text
A = G_l^T (W / theta) G_l
b = G_l^T W u* + M q
u = u* - (G_l p) / theta
```

where `q` is the CM12 target-divergence correction. Missing in-domain bricks
are sparse air and contribute one-sided Dirichlet rows when adjacent to liquid;
closed outer-domain faces have no row. A 2:1 port can contain one coarse term
and several fine terms, so the matrix is symmetric positive semidefinite but is
not the dense seven-point matrix.

The resident GPU stores a dimensionless pressure impulse. Diagnostic
publication maps it to physical pressure with

```text
pressureScale = fluidDensity * finestCellSize^2 / dt
```

Performance and parity comparisons should therefore prefer projected face
velocity and divergence. Raw Uniform and Sparse pressure arrays are not on the
same scale until this mapping is applied, and pressure remains gauge-dependent
on unanchored components.

### 1.2 Current pressure frame graph

The non-spherical pressure portion of
`lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts` is:

```text
pressure-topology
  classifyPressureCells       accepted cells
  classifyRows                accepted rows
  preparePressure             accepted cells

pressure-rhs
  initializePCG               accepted cells
  reduceInitialize            one workgroup

pressure-solve, repeated 128 times by default
  applyDirection              accepted cells + SpMV + d^T A d partials
  reduceCurvature             one workgroup
  updateResidual              accepted cells + p/r/z + r^Tz/r^Tr partials
  reduceResidual              one workgroup
  updateDirection             accepted cells

velocity-projection
  projectFaces                accepted rows
  collocateAndDiagnose        accepted cells

projection-diagnostics
  measureDivergenceDiagnostics accepted cells
  reduceDivergenceDiagnostics one workgroup
```

At the default 128 iterations, the solve body alone encodes **640 dependent
dispatches**, including **256 global scalar reductions**. Counting pressure
classification, initialization, projection, and diagnostics gives roughly 649
pressure-related dispatches when every stage is present.

The current defaults are:

```text
SPARSE_CM12_PRESSURE_ITERATIONS          = 128
SPARSE_CM12_PRESSURE_RELATIVE_TOLERANCE  = 0
minimum/maximum exposed iterations       = 8 / 256
```

Tolerance zero deliberately means fixed-budget execution. With a positive
tolerance, the same maximum command graph remains encoded. Later kernels gate
some arithmetic through `pressureIterationActive`, but the cell workgroups and
reduction dispatches are still launched; `applyDirection` and
`updateResidual` still enter their reduction machinery with zero values after
the gate closes. Early stop therefore saves edge traversal and vector updates,
not the whole remaining iteration cost.

### 1.3 Sparse work is accepted-topology work, not liquid-row work

The pressure kernels use accepted cell and row worklists. They do not currently
compact a pressure-epoch worklist containing only liquid cells and rows that
participate in the pressure system. `applyOperator` returns zero for a
non-liquid cell, but the invocation, accepted-cell lookup, workgroup reduction,
and partial publication have already been paid.

This distinction matters in:

- surface bricks containing a large air fraction;
- predictive/dormant receiver support retained for transport;
- recently vacated cells that remain accepted during hysteresis;
- large coarse bricks whose accepted storage is useful to other stages but
  whose pressure participation is small.

Pressure work must remain closed over every connected liquid cell and required
Dirichlet row. The optimization is a field-specific worklist, not more
aggressive residency retirement.

### 1.4 The SpMV is compact but still carries avoidable edge traffic

HEAD `c418b265` prepacked the pressure off-diagonal graph so every PCG
application no longer reconstructs cell-to-incidence-to-row-to-term chains.
The hot edge record remains three 32-bit words:

```text
row id, other cell id, static coefficient product
```

Every SpMV then:

1. reads the row id;
2. reads the per-frame row `theta`;
3. reads the neighbour id;
4. reads the static weight;
5. optionally reads rigid-body row scaling;
6. reads the neighbour vector value.

For a body-free pressure epoch, the effective off-diagonal coefficient is
constant for all PCG iterations once `classifyRows` has published `theta`.
Paying the row indirection and division on 128 applications is unnecessary.

### 1.5 Warm start exists but its value is not measured

`classifyPressureCells` clears pressure only for cells that are no longer
liquid. Continuing liquid cells retain the previous frame's pressure. Candidate
topology transfer also treats pressure as a warm start rather than conserved
state.

This is the correct policy, but current telemetry does not publish:

- zero-seed residual versus warm-seed residual;
- residual immediately before and after a topology transition;
- the number of iterations saved by the warm seed;
- physical-pressure consistency when `dt`, finest cell size, or pressure scale
  changes;
- a fallback count for newly liquid or poorly mapped cells.

With the default fixed 128 iterations, a better warm start improves the final
answer but saves no frame time.

### 1.6 The GPU convergence receipt is not a true residual

The non-spherical resident solver updates `r` recursively:

```text
r_(k+1) = r_k - alpha A d_k
```

and publishes `sqrt(r^T r / b^T b)` from that vector. It never reapplies the
operator to the final pressure to calculate a fresh `b - A p`. In f32, the
recursive residual can drift far below the true residual. This explains how a
reported relative pressure residual can reach approximately `1e-12` while the
recomputed post-projection maximum divergence remains around `1e-5 1/s`.

The CPU composite oracle already has the required reference behavior:

- assemble the same `A` and `b`;
- project unanchored connected components into an exact zero-mean quotient;
- periodically maintain that quotient during PCG;
- recompute the true residual after the solve;
- report both relative L2 and maximum residual;
- derive physical post-projection divergence from the equation residual.

The resident GPU path must reach diagnostic parity with that oracle before
iteration-count or tolerance changes are accepted.

### 1.7 Null-space handling differs from the CPU oracle

Every liquid component touching free-surface/sparse-air Dirichlet rows is
anchored and SPD. A completely enclosed all-liquid component is all-Neumann and
has a constant-pressure null space. The CPU oracle subtracts the mean from `b`,
pressure, residual, preconditioned residual, direction, and applied direction
for each unanchored component.

The resident GPU path currently does not publish component IDs/anchors and does
not perform that projection. CG can still appear to work from a zero seed when
the RHS is compatible, but small compatibility error can produce curvature
loss, iteration growth, a gauge drift, or misleading convergence. This is both
a correctness and a performance issue.

---

## 2. Performance model and expected crossover

Let:

- `N_d` be the dense Uniform cell count;
- `N_s` be the accepted Sparse pressure-cell count after pressure compaction;
- `E_s` be the effective off-diagonal edge count;
- `k` be executed PCG iterations;
- `L/h` be the connected pressure diameter measured in smallest cells.

The current Sparse cost is approximately

```text
T_sparse = topology/rhs setup
         + k * (SpMV(E_s) + vector passes(N_s) + two global reductions)
         + projection/diagnostics
```

Diagonal-preconditioned Poisson CG has iteration growth approximately
`O(L/h)` for fixed tolerance. Sparse is therefore not asymptotically linear
unless `k` remains bounded.

Uniform geometric multigrid is approximately

```text
T_uniform = sum over levels of smoothing + transfers + coarse solve
```

and is `O(N_d)` with grid-independent convergence when the hierarchy and
smoother are effective. Its disadvantages in the current implementation are a
cold start, a fixed 3 Full + 4 V schedule, and dense scans over air and
pass-through cells. Its advantages are regular texture access, no global dot
product in each ordinary smoothing sweep, and cheap removal of long-wavelength
error.

Expected regimes:

| Regime | Expected winner | Reason |
|---|---|---|
| all-fine Sparse, complete regular box | Uniform eventually | same unknowns; dense texture locality and multigrid beat CSR + reductions per degree of freedom |
| all-coarse Sparse versus matched coarse Uniform | Uniform eventually | same reduced problem; Sparse has abstraction and indirect-access overhead |
| localized liquid in a much larger empty domain | Sparse | `N_s << N_d`; avoided dense work dominates PCG overhead |
| broad shallow splash with many fine surface bricks | measurement-dependent | sparse occupancy helps, but accepted air support and seams increase `N_s`, `E_s`, and `k` |
| long thin connected jet/channel | multigrid required | low occupancy but large graph diameter gives slow low-frequency PCG convergence |
| deep ocean or filled tank | Uniform or Sparse MGPCG | pressure includes the wet bulk; surface sparsity alone is not pressure sparsity |
| tight residual at increasing resolution | multigrid | diagonal PCG iteration count grows with `L/h` |
| small grid | current result is implementation-dependent | Sparse reductions and Uniform's deep fixed command graph can both dominate useful cell work |

The existing all-fine end-to-end performance contract allows Sparse CM12 to be
up to 1.30x Uniform and targets 1.20x. That is a useful guard, not evidence that
the Sparse pressure solver is efficient. Pressure-specific work must be
reported separately at matched true residual.

---

## 3. Findings ledger

### F1 — Fixed 128-iteration execution hides warm-start value

Default tolerance is zero and every frame pays the complete 128-iteration
command graph. A stable hydrostatic or slowly changing frame receives no timing
benefit from a good seed.

### F2 — Current early stop is an arithmetic gate, not a work gate

Positive relative tolerance disables some post-convergence arithmetic, but all
iteration dispatches remain encoded and the reduction structure still runs.
The profiler must distinguish encoded, active, and post-gate iterations.

### F3 — Diagonal PCG is not grid-independent

The preconditioner removes local coefficient scale but not long-wavelength
Poisson error. Iterations grow with connected diameter/resolution and can erase
the sparse-cell advantage on long or deep liquid components.

### F4 — Two reductions serialize every iteration

`d^T A d` is required before `alpha`; `r^T z` is required before `beta`. The
current five-dispatch chain creates two device-wide synchronization points per
iteration. At small `N_s`, latency dominates; at large `N_s`, reduction and
SpMV bandwidth dominate.

### F5 — The hot SpMV performs redundant row/theta lookup

The topology is prepacked, but effective coefficients are not. Body-free runs
can bake `staticWeight/theta` once after row classification and reduce each hot
edge record from three words to `otherCell + effectiveWeight`.

### F6 — Accepted topology is wider than the pressure system

All accepted cells participate in PCG dispatch and reductions even when they
are air. Pressure-specific compaction is required; changing global residency
would incorrectly couple a pressure optimization to transport support.

### F7 — Recursive residual telemetry is too optimistic

A tiny recursive f32 residual cannot size iteration budgets or validate
accuracy without a true `b - A p` recomputation. Post-projection divergence is
currently the more trustworthy scalar.

### F8 — GPU null-space handling is incomplete

The CPU oracle's component quotient is absent on the GPU. Compatibility error
can look like solver slowness and must be separated from preconditioner quality.

### F9 — Mixed seams and small theta increase both traffic and conditioning

Mixed rows have more terms than regular faces and introduce wider matrix
couplings. Small ghost-fluid theta increases coefficient contrast. Publish
seam-degree and theta histograms alongside iteration counts; do not attribute
all iteration growth to domain size.

### F10 — Warm-start transfer has no quality receipt

Pressure transfer across topology epochs is allowed to accelerate convergence,
but its initial residual, physical scale, fallback population, and iteration
delta are not measured.

### F11 — Regular sparse regions pay the full irregular-row cost

All-fine and complete all-coarse regular regions still use the same compact CSR
application as mixed seam and cut-cell rows. The implementation repeatedly
loads offsets, column indices, and general row metadata even where topology
proves a fixed local stencil. Sparse should keep its compact ownership and
hierarchy, but it should not pay the exceptional-row cost on ordinary regular
interiors.

### F12 — Existing benchmarks do not isolate pressure at equal accuracy

End-to-end frame A/Bs include different transport and density states. Uniform
and Sparse residual scalars use different norms and, today, different residual
truthfulness. A solver optimization cannot be attributed cleanly until a
frozen-snapshot pressure benchmark exists.

---

## 4. Target production architecture

```text
accepted sparse topology (changes on topology epoch)
    -> connected-component / anchor metadata
    -> liquid cell + active row compaction
    -> finest composite operator structure
    -> solver hierarchy structure

per frame
    classify liquid and theta
    -> compact pressure worksets
    -> bake effective finest and coarse coefficients
    -> scale/remap warm pressure seed
    -> compute true initial residual
    -> MGPCG to shared physical tolerance
       [composite SpMV]
       [one multigrid V-cycle preconditioner]
       [global reductions]
       [periodic true-residual replacement]
    -> true final residual + KKT-independent pressure receipt
    -> compatible face projection
    -> physical divergence receipt
```

The finest operator remains the current `G^T W/theta G`. Multigrid is a
preconditioner, not a replacement discretization. The outer PCG retains a
single convergence authority and protects against an imperfect hierarchy.

The hierarchy is a **solver hierarchy**, not simply the simulation's accepted
LOD rungs. Simulation LOD decides physical approximation; solver aggregation
decides which error modes a coarse level removes. Parent relationships may be
seeded from brick ownership, but every coarse level must cover the complete
liquid graph and preserve Dirichlet anchors and unanchored-component gauges.

---

## 5. Work packages

Work packages are ordered. Do not start reduction-reordering or multigrid work
before WP0/WP1 establish a trustworthy baseline.

### WP0 — Frozen pressure benchmark and common receipts

**Goal:** make Uniform/Sparse pressure comparisons reproducible and
accuracy-matched.

Implement a pressure-only Dawn harness that freezes the exact pre-pressure
state and reports:

- `N_d`, accepted cells/rows, liquid cells, active pressure rows, `E_s`;
- regular, brick-face, mixed-seam, and sparse-air row counts;
- row-degree histogram and maximum degree;
- theta minimum plus logarithmic clamp/quantile histogram;
- warm-seed and zero-seed true residuals;
- recursive and true relative L2 residuals;
- maximum equation residual divided by control volume in `1/s`;
- volume-weighted post-projection divergence L2 and maximum;
- SpMV count, executed iterations, encoded iteration ceiling, and gated tail;
- pressure topology/RHS/solve/projection/diagnostic GPU times;
- bytes allocated by pressure state, effective edges, partials, and hierarchy;
- component count, anchored/unanchored counts, and compatibility correction.

Use the same frozen density, provisional face velocity, geometry, dt, liquid
threshold, theta rule, and target divergence for both solvers wherever the
operators are meant to match. Compare projected face velocities after mapping
pressure scale and gauge, not raw pressure first.

Add manufactured snapshots:

1. complete regular all-fine box;
2. matched complete regular all-coarse box;
3. localized liquid island in a large domain;
4. long thin connected component;
5. planar free surface with controlled theta;
6. one 2:1 seam and a seam corridor;
7. deep wet bulk with a fine surface band;
8. enclosed all-Neumann component with compatible RHS;
9. deliberately incompatible all-Neumann RHS, which must be detected;
10. one accepted topology transition with a transferred warm start.

**Deliverables:**

- `tools/benchmark-sparse-cm12-pressure.ts`;
- a JSON schema/receipt type under `lib/methods/adaptive-mass/`;
- CPU-oracle comparison for small snapshots;
- checked-in representative result under `benchmarks/results/` only after the
  harness and thresholds are accepted.

**Acceptance:** no performance claim is made from recursive residual alone;
both solvers are timed only after reaching the same true physical tolerance.

### WP1 — True GPU residual, compatibility, and actual iteration telemetry

**Goal:** make the resident convergence decision numerically defensible.

Add GPU kernels to:

1. identify pressure connected components and whether each has a Dirichlet
   anchor on topology publication;
2. accumulate each unanchored component's RHS sum and subtract its mean;
3. enforce the same zero-mean quotient on the initial pressure and the PCG
   vectors where required;
4. recompute `b - A p` after initialization, periodically during a long solve,
   and at the end;
5. publish actual executed iterations, first tolerance-crossing iteration,
   curvature breakdown, recursive/true residual ratio, and compatibility
   correction.

For the first implementation, periodic true-residual replacement every fixed
small batch (for example 8 or 16 iterations) may be evaluated, but the cadence
must be justified by the WP0 residual-drift trajectory, not chosen per scene.
Always recompute once at the end.

The stopping test should be based on a common physical criterion. Retain
relative L2 for conditioning analysis, but gate production on the greater of:

- an agreed absolute divergence-equivalent tolerance in `1/s`; and
- a relative tolerance appropriate to the RHS scale.

Do not hard-code a new universal epsilon before the benchmark matrix establishes
the f32 floor. Preserve the current `1e-5 1/s` post-divergence gate as a
regression reference, and report results relative to the matched Uniform
control.

**Acceptance:** GPU and CPU oracle agree on component anchoring,
compatibility correction, true residual, and projected divergence for the
small fixtures. No valid solve reports a recursive residual materially below a
failing true residual without setting a residual-drift diagnostic.

### WP2 — Pressure-specific worksets and effective-edge bake

**Goal:** make one existing PCG iteration proportional to the actual pressure
system and reduce hot edge bandwidth.

At each pressure epoch:

1. compact liquid cell IDs into a pressure-cell worklist;
2. compact rows with at least one liquid term and positive dual weight into a
   pressure-row worklist;
3. publish indirect counts and partial-reduction counts for those worklists;
4. bake the diagonal and effective off-diagonal values after theta
   classification;
5. store the body-free hot edge format as two words:
   `otherCell, effectiveWeight`;
6. retain a separate dynamic-solid path only when live rigid scaling prevents
   the bake.

The worklist builder must retain sparse-air Dirichlet rows and all pressure
edges required by a liquid cell. It must not use density sparsity to disconnect
wet components. Empty worklists must publish legal zero-work indirect records.

Measure two implementation options:

- stable template-cell IDs in vectors plus a compact invocation worklist;
- fully compact pressure vectors plus scatter/gather at projection boundaries.

Prefer stable vectors first: they avoid remapping warm pressure and topology
transfers while still eliminating air invocations. Fully compact vectors land
only if their bandwidth win repays remap cost.

**Acceptance:** bit-identical effective `A p` within f32 evaluation order on
regular and seam fixtures; true residual and projected velocity remain within
the agreed tolerance; PCG cell invocations equal liquid pressure cells rather
than accepted cells.

### WP3 — Residual-driven execution and warm-start quality

**Goal:** stop paying 128 useful iterations when the warm seed is already
close.

First make early stopping semantically correct:

- set a nonzero default only after WP0/WP1 establish the true-residual floor;
- stop on the true/guarded residual, not an unbounded recursive recurrence;
- publish executed versus encoded iterations;
- retain a hard maximum and fail closed if it is reached;
- never silently consume an unconverged field.

Then reduce the post-convergence command tail. Evaluate in this order:

1. **GPU arithmetic gate:** retain as the correctness backstop; move the gate
   before avoidable accepted-cell and edge reads where WGSL uniformity permits.
2. **Short encoded prefix from lagged telemetry:** size the next frame's prefix
   from recent executed iterations, with conservative growth after misses. This
   is a performance hint only; it may not remove the current frame's hard
   convergence backstop.
3. **GPU-written indirect iteration work:** investigate only with a legal
   WebGPU usage/synchronization design. Storage-written indirect arguments must
   not alias forbidden usage scopes. Do not add one compute-pass boundary per
   iteration merely to save launches.
4. **Small fixed batches:** acceptable if they reduce command construction and
   permit true-residual replacement without CPU mid-frame stalls.

Warm-start telemetry must include:

- initial residual from the retained seed;
- zero-seed residual measured in instrumentation builds;
- topology-transition residual ratio;
- newly liquid cell count;
- pressure-scale change and seed rescale count;
- fallback-to-zero count;
- iterations saved against the zero-seed control.

If `dt` or the finest physical pressure scale changes, rescale the stored
dimensionless pressure so the warm start represents the same physical field.

**Acceptance:** settling/hydrostatic frames execute materially fewer useful
iterations without increasing the true residual or divergence; abrupt topology
changes automatically recover to the hard ceiling and expose any miss.

### WP4 — Reduce per-iteration synchronization and vector traffic

**Goal:** lower the cost of PCG while keeping the same preconditioner.

Profile before changing the recurrence. Candidate changes:

- increase reduction workgroup width when device occupancy and register use
  allow it;
- use subgroup reductions where supported without changing determinism gates;
- pack scalar partials and reuse one compact partial arena;
- avoid writing diagnostic-only vectors in production;
- remove redundant pressure/diagonal/theta loads exposed by WP2;
- specialize regular-degree and mixed-seam SpMV paths so ordinary rows remain
  coherent;
- reorder compact pressure-cell work by brick/local index for neighbour cache
  locality;
- evaluate pipelined or communication-avoiding PCG to reduce two global
  reductions to one.

Pipelined PCG is a numerics-changing work package. It can amplify recurrence
drift in f32 and must include periodic true-residual replacement from WP1. Land
it independently from worklist and edge-layout changes.

Do not pursue kernel fusion that violates a global dependency. In particular,
`updateDirection` cannot simply be fused with the following SpMV: every
direction value must be globally visible before any neighbour reads it.

**Acceptance:** same true residual trajectory envelope and projected-velocity
accuracy; report time per executed iteration, SpMV GB/s, reduction time, and
vector time separately. Pass-count reduction alone is not an acceptance
criterion.

### WP5 — Seam-aware local preconditioning

**Goal:** reduce coefficient/seam-induced iterations before building the full
hierarchy and establish smoother candidates for multigrid.

Evaluate:

- damped Jacobi with a measured spectral bound;
- Chebyshev polynomial smoothing using a conservative estimate of
  `lambda_max(D^-1 A)`;
- block Jacobi or additive Schwarz patches around a mixed port, containing the
  coarse term and all fine terms;
- brick-interior blocks with seam corrections.

The current mixed-port expansion can introduce same-side fine-cell couplings,
so standard checkerboard red-black assumptions do not hold. Any block or
colouring must be derived from the actual composite graph.

This WP is promoted only if it materially reduces iterations on seam/theta
fixtures without excessive setup or application cost. Otherwise proceed
directly to WP6.

### WP6 — Composite multigrid-preconditioned CG

**Goal:** bound iteration growth as `L/h` and world size increase.

Build a global sparse solver hierarchy with:

1. one level-0 unknown per active liquid leaf;
2. aggregates seeded from stable brick parents but closed over the entire
   connected pressure graph;
3. prolongation `P` that preserves constants and respects physical cell
   volumes;
4. restriction `R` chosen as the compatible transpose/weighted transpose;
5. Galerkin coarse operators `A_c = R A_f P` initially;
6. explicit propagation of free-surface/Dirichlet anchors;
7. per-component gauge handling on levels without anchors;
8. weighted Jacobi, Chebyshev, or measured seam-block smoothing;
9. a compact high-precision-in-f32 coarsest solve;
10. one V-cycle as the outer PCG preconditioner.

Separate hierarchy **structure** from per-frame **coefficients**:

- rebuild aggregation, adjacency, `P/R`, and buffer sizes only on a topology
  generation change;
- refresh theta-dependent and solid-dependent coefficients every pressure
  epoch;
- retain stable vector slots and pressure warm starts across unchanged
  generations;
- patch only affected aggregates after a local topology transaction once the
  full rebuild is proven.

Start with a topology-complete rebuild. Incremental hierarchy patching is a
later optimization and must not precede correctness.

The hierarchy must remain sparse at every level, including the coarsest level.
The coarsest solve may use a serial or cooperative direct/iterative kernel over
the compact coarse graph, but it must not allocate or reconstruct the
equivalent dense logical domain.

**Target, not initial gate:** ordinary frames should converge in a low
double-digit or single-digit MGPCG iteration count, with no monotonic growth as
the same physical problem is uniformly refined. Final gates come from WP0
baselines, not this target.

**Acceptance:**

- constant pressure is a null mode only on legitimate unanchored components;
- a manufactured linear pressure field has the expected refinement behavior;
- `A_c` symmetry and positive curvature are verified;
- regular all-fine MGPCG matches the Uniform control after scale/gauge mapping;
- seam fixtures retain flux conservation and no divergence crease;
- iteration counts remain bounded across the refinement ladder;
- pressure time scales with live hierarchy cells/edges, not logical-domain
  volume.

### WP7 — Sparse-native regular-region specialization

**Goal:** avoid paying fully general CSR/seam costs on the regular majority of
the sparse pressure graph without adding another pressure backend.

Classify rows at topology-build time into a small number of structural classes:

- regular same-level interior rows with fixed degree and coefficient layout;
- free-surface or solid-adjacent same-level rows;
- mixed-resolution seam rows;
- other exceptional rows retained on the general compact path.

Preserve one unknown numbering, one sparse hierarchy, one finest operator, and
one convergence receipt. Store compact worklists/ranges for each class and
dispatch class-specific operator and smoother kernels. The regular kernel may
derive fixed neighbors from compact brick-local coordinates or consume a
tightly packed fixed-width neighbor record; it must not materialize a dense
world-sized grid. Seam and cut-cell kernels continue to consume explicit
general records.

Prototype in this order:

1. split regular and exceptional level-0 rows while retaining the existing
   numerical coefficients;
2. add a fixed-width regular-row representation and compare bytes per operator
   application against CSR;
3. propagate the same row classes into multigrid levels where structurally
   valid;
4. fuse regular-region smoothing/vector work only when true-residual receipts
   remain identical within the declared f32 tolerance.

All-fine Sparse versus Uniform remains a useful negative-control benchmark, not
a backend-selection trigger. All-coarse must compare against a physically
matched coarse Uniform lattice, not the fine Uniform reference.

**Acceptance:** exact row classification, zero dense logical-domain
allocation, identical finest-operator action to the general sparse path,
pressure-specific speedup at equal true residual, and no change to public
Sparse field/topology contracts.

### WP8 — Production policy and removal of legacy fixed-budget behavior

After WP0–WP7:

- use sparse MGPCG as the production solver, retaining plain sparse PCG only as
  a validation/debug path or a safe bring-up mode while MGPCG matures;
- preserve explicit debug overrides;
- remove the fixed 128-iteration default only after all production scenes have
  hard convergence receipts;
- fail closed or retain the prior accepted state on curvature loss, nonfinite
  values, hierarchy failure, or iteration-cap failure;
- publish hierarchy sizes, row-class counts, executed applications, and the
  convergence reason in `GPUEulerianInfo`.

No scene-specific magic iteration counts, occupancy thresholds, or solver
switches. Topology may select sparse row-class kernels, but it must not select a
dense pressure representation or the Uniform pressure solver.

---

## 6. Benchmark matrix and gates

### 6.1 Pressure-only fixtures

Every pressure implementation WP runs the WP0 frozen snapshots. Required
plots/tables per snapshot:

- true residual versus operator applications;
- post-projection divergence versus operator applications;
- GPU time split into setup, SpMV/smoothing, reductions, transfers, and
  projection;
- executed applications and encoded tail;
- `N_s`, `E_s`, hierarchy counts, seams, theta distribution;
- warm versus zero seed.

### 6.2 End-to-end scenes

Retain these roles:

| Scene/lane | Purpose |
|---|---|
| symmetric expansion 32x16x32 all-fine | equal-DOF overhead and D4 control |
| mini dam 32 and 64 all-fine | regular refinement ladder |
| mini dam 32/64 adaptive | moving mixed seams and warm-start transitions |
| long dam 192x96x32 adaptive | graph-diameter and front/residency adversary |
| hydrostatic/deep pool | warm-start and low-frequency pressure control |
| ocean seiche | high wet occupancy negative control; pressure may not fake sparsity by dropping deep liquid |
| localized droplet/brick seed in a large domain | sparse scaling positive control |

Construction and warmup remain excluded from recurring frame timing but are
reported separately. Alternate arm order and use fresh-process medians for
wall-time claims. Keep CPU encode, GPU stage time, and serialized queue-wall
time separate.

### 6.3 Correctness gates

- No NaN/Inf pressure, residual, coefficient, or projected velocity.
- Positive matrix diagonal for every active row; positive PCG curvature unless
  the solve is already converged.
- Component compatibility and gauge receipts agree with the CPU oracle.
- True final residual meets the agreed common tolerance.
- Maximum and volume-weighted divergence meet the greater of the solver target
  and the established matched-Uniform f32 floor; retain `1e-5 1/s` as the
  current regression reference.
- Projection does not increase kinetic energy beyond the established f32
  identity tolerance.
- Constant pressure gives zero interior gradient.
- Seam flux is single-owned and equal/opposite in adjacent control volumes.
- D4 pressure, velocity, topology, and divergence remain within existing
  scene gates.
- A topology transition may change pressure as a warm start, but the following
  global solve must independently converge; transfer is never accepted as the
  solve.

### 6.4 Performance gates

Do not freeze absolute millisecond gates before WP0 captures the current dirty
working tree in a clean reproducible revision. Required relative gates are:

- no regression in all-fine Sparse/Uniform end-to-end maximum ratio (1.30) or
  target ratio (1.20);
- pressure time and memory scale with pressure workset/hierarchy counts for the
  localized-domain ladder;
- no iteration growth across uniform refinement after MGPCG lands;
- post-convergence tail is separately measured and materially smaller than the
  useful solve;
- no simulation-sized host loop or synchronous convergence readback;
- no dense logical-domain allocation on the Sparse MGPCG path;
- high-occupancy negative controls do not claim a sparse win by changing
  represented resolution or tolerance.

---

## 7. Suggested file layout

Do not begin with a large extraction-only refactor. WP0/WP1 may add kernels to
the current resident files so performance changes remain attributable. Once
receipts are stable, isolate the subsystem along these boundaries:

```text
lib/methods/adaptive-mass/
  sparse-cm12-pressure-contract.ts
  sparse-cm12-pressure-receipt.ts
  webgpu-sparse-cm12-pressure.ts
  webgpu-sparse-cm12-pressure.wgsl.ts
  webgpu-sparse-cm12-pressure-multigrid.ts
  webgpu-sparse-cm12-pressure-multigrid.wgsl.ts

tools/
  benchmark-sparse-cm12-pressure.ts

tests/
  sparse-cm12-pressure-receipt.test.ts
  sparse-cm12-pressure-operator.test.ts
  sparse-cm12-pressure-nullspace-dawn.test.ts
  sparse-cm12-pressure-worklist-dawn.test.ts
  sparse-cm12-pressure-mgpcg-dawn.test.ts
  sparse-cm12-pressure-regular-rows-dawn.test.ts
```

The resident frame remains the owner of stage order and buffer lifetime. The
pressure module owns operator structure, vectors, hierarchy, convergence, and
projection receipts. The CPU composite oracle remains independent and is not
rewritten to mirror GPU implementation details.

---

## 8. Landing discipline

1. One numerical or layout hypothesis per commit/PR.
2. Do not combine recurrence changes, worklist compaction, edge packing, and
   multigrid in one benchmark delta.
3. Every optimization reports true residual, post-divergence, operator
   applications, workset sizes, and stage time before/after.
4. Preserve a forced classic-PCG backend until MGPCG is established.
5. Preserve a zero-seed control in instrumentation builds.
6. Do not re-bless density/transport differences as pressure performance.
7. Never use a lower-resolution Sparse arm as an equal-accuracy solver-speed
   comparison unless the Uniform arm is physically matched to that lattice.
8. A cap failure is a failed pressure solve, not telemetry-only success.
9. Keep topology transition and steady-topology samples separate.
10. Update this handoff when a WP lands: record commit, command, adapter,
    snapshot, work counts, residual, divergence, and median/p95 timing.

---

## 9. Recommended immediate sequence

The next implementation session should do exactly this:

1. Add WP0's pressure-only frozen snapshot harness without changing runtime
   numerics.
2. Add final true-residual recomputation and actual-iteration telemetry to the
   resident GPU path.
3. Port component anchor/null-space receipts from the CPU oracle.
4. Capture baseline trajectories for 8, 16, 32, 64, 128, and 256 iteration
   ceilings, warm and zero seed.
5. Land pressure-liquid cell/row compaction.
6. Land body-free effective-edge baking and measure bytes/application.
7. Turn on a true-residual-driven default only after the f32 floor is known.
8. Decide from profiles whether WP4 reduction work or WP6 MGPCG is the larger
   immediate win; do not spend a long cycle on local PCG tuning if iteration
   growth is already the dominant cost.

The likely production payoff is not a perfectly optimized 128-iteration PCG.
It is a compact live pressure graph, a trustworthy warm start, and a sparse
multigrid preconditioner that reduces the solve to a small bounded number of
global iterations.
