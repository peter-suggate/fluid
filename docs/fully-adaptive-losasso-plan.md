# Fully adaptive Losasso: implementation and handoff plan

Date: 2026-08-09  
Repository point examined: `61ff946` plus the local span-aware
`trackerPhi(position, span)` change in
`lib/webgpu-octree-losasso-coarse-phi.wgsl.ts`

Revision note: the graph design below retains the existing finest-lattice
node item as the stable logical key, while storing recurring physics fields in
deterministic compact adaptive slots. Delivery is a direct authority-cutover
sprint: build the adaptive graph and operators on the live lane, switch all
physics consumers as soon as the complete tuple is coherent, and validate with
symmetric expansion plus normal/coarse dam-break UI A/B runs. There is no
shadow implementation or unit-test tranche.

## Executive decision

The next step is not another coarse special case in the dense surface tracker.
It is to make one accepted octree graph own all recurring Losasso state:

- node-centred level set `phi`;
- node-centred extended velocity, including the MacCormack predictor field;
- adaptive face-normal liquid velocity;
- leaf-local interpolation, redistance, volume integration, pressure boundary
  data, topology sizing evidence, and render sampling.

The graph and its fields must participate in the same candidate/accepted
transaction as pressure rows and faces. Dense or factor-one data may remain as
an output adapter during migration, but no materialized view may feed a later
physics stage.

This is the smallest architecture that can make a size-4 region behave like a
size-4 method rather than a size-1 method sampled by size-4 pressure rows.

## What “fully adaptive” means here

For a leaf of span `s` and finest-grid spacing `h`:

- its scalar degrees of freedom are its eight octree corner nodes;
- its interpolation cell is `s h` wide;
- its face velocities live on the real adaptive faces and transition
  subfaces;
- its nodal velocity is reconstructed at the coarsest incident-face scale;
- its characteristic tracing samples the containing leaf, not a finest MAC
  staging lattice;
- its redistance update uses physical neighbor distances, including `s h`;
- its liquid volume is integrated from that leaf's nodal field;
- its pressure ghost distance and gradient are derived from that same field;
- its rendering representation is derived from that same field.
- its recurring graph and field storage, as well as its dispatch work, scale
  with adaptive leaves, nodes, edges, and faces rather than finest-domain
  capacity; an explicitly enabled render output is the only allowed dense
  allocation.

Adaptivity is therefore a property of the field representation and every
operator on it. Merely reducing the pressure-row count is not sufficient.

The first production cutover should target the current factor-one Losasso
lane. Factor-4/8 sparse fine level sets are a separate, deliberately
sub-cell-detail hybrid. They may remain as named experimental lanes, but they
must not be described as the fully adaptive Losasso lane until they either
become a one-way detail overlay or are retired.

## Why this has been difficult

### 1. There are currently two different simulations sharing a name

The pressure side is genuinely adaptive. `WebGPUOctreeLosassoCoarseBackend`
owns compact rows, adaptive faces, face-normal velocities, a pressure
operator, projection, and a staged candidate-to-accepted commit.

The factor-one surface side is a dense cell-centred simulation.
`WebGPUOctreeCoarseSummary` owns three full-domain phi banks and runs dense
prediction, dense Eikonal sweeps, dense volume work, and dense-complement
publication. `WebGPUOctreeLosassoCoarsePhiExchange` then restricts that result
onto compact pressure rows.

The dense field is not a cache. It is the recurring authority. This is why
changing only coarse row interpolation, gradients, or presentation cannot
change the underlying dynamics.

### 2. The field locations are incompatible

The paper stores `phi` and reconstructed velocity at octree nodes. The current
factor-one tracker stores `phi` at finest-cell centres. Moving between these
locations is not an identity operation, even when every pressure leaf has
size one. It changes:

- degrees of freedom and sample positions;
- boundary stencils;
- zero-crossing geometry;
- redistance arithmetic;
- volume quadrature;
- the result of every semi-Lagrangian interpolation.

The rejected adaptive-to-dense experiments repeatedly filtered the field
because each remap changed its representation. The problem was not a bad
interpolant; it was allowing both representations to be authorities.

### 3. The current adaptive node/edge publication is a census, not a solver graph

`WebGPUOctreeAdaptiveNodes` is valuable evidence, but it is not ready to carry
state:

- it scans all `(nx + 1)(ny + 1)(nz + 1)` possible nodes on every tracker
  advance;
- the stored node item is already a stable geometric identity, but compact
  list order is determined by atomics and cannot be a persistent field slot;
- node records contain only that dense linear coordinate key, with no compact
  physics-state mapping;
- edges contain coordinate keys, not compact node slots;
- there is no key-to-slot directory;
- there is no leaf-to-eight-corners table;
- there are no six directional neighbor records;
- there is no distinction between independent and hanging nodes;
- there are no edge/face T-junction constraint masters and weights;
- leaf edges of different spans can geometrically overlap, so the list is not
  a set of atomic Eikonal segments;
- it has no candidate bank and no joint validity/commit contract with the
  topology it describes.

The existing node item should be retained as the logical key for identity and
handoff. Using the current compact list and overlapping edges directly as the
solver graph would still make field storage, T-junction continuity, and edge
propagation ambiguous.

### 4. The topology and surface clocks are deliberately offset

One step currently does:

1. commit the previous step's topology candidate;
2. transport the surface with the previous projected/extended velocity;
3. solve and project on the accepted topology;
4. build the next topology candidate at the tail.

That pipeline is sound, but it means an adaptive surface cannot be an object
rebuilt casually inside `encodeSurface`. The candidate graph must be built at
the topology tail, phi must be remapped from the just-finalized accepted
surface, and the graph/phi/face-velocity tuple must commit atomically at the
next head. Otherwise a consumer sees node identities from one epoch and field
values from another.

### 5. Velocity is adaptively sourced but densely materialized

The liquid velocity authority already consists of adaptive pressure faces.
However `WebGPUOctreeLosassoExtensionBand` allocates for the complete finest
MAC lattice, stages accepted and predictor MAC values, and then stages two
complete finest nodal lattices. Face advection and surface transport consume
those dense nodal banks.

The staging kernels contain much of the intended Section 3 reconstruction,
but they execute it at every finest site. Coarse runs can therefore cost more
per staged site: each site searches the compact hierarchy and reconstructs a
temporary coarse value.

### 6. The current regression contract contains a real contradiction

A node-centred adaptive method cannot in general remain byte-for-byte equal to
a cell-centred dense method, even on a uniform size-1 topology. Requiring both
one new authority and the old full profile fingerprint as an exact final gate
is impossible unless the new code emulates the old cell-centred method, which
would no longer be the proposed method.

The cutover deliberately replaces the old fingerprint. The old lane is used
only as the A side of a short dam-break UI comparison while the new lane is
being brought up; it is not maintained as a shadow backend. Acceptance is
based on symmetric expansion and the normal/full-tank-size-4 dam-break UI A/B:
volume, surface profile, zero crossing, symmetry, stability, and structural
work reduction. Permanent byte identity to the cell-centred method is not a
goal.

## Current authority map

```text
accepted owner pages / pressure rows
        |                         \
        |                          -> adaptive pressure faces -> projected u
        v                                                   |
dense factor-one phi (3 banks)                             |
        |                                                   v
        +-> redistance / volume                    finest-MAC W7 graph
        |                                                   |
        +-> compact row phi                                 v
        |                                          dense MAC staging
        +-> topology sizing / ghost phi                     |
        |                                                   v
        +-> dense complement                       dense nodal staging
        |                                                   |
        +-> rendering                              face and phi advection

The two vertical chains meet through restriction and staging, not through one
shared adaptive state.
```

The target is:

```text
accepted adaptive graph + phi + face u + nodal extended u
        |
        +-> phi transport -> redistance -> volume correction
        +-> pressure row/ghost derivation -> solve -> projected face u
        +-> nodal reconstruction/extension
        +-> topology sizing and next candidate remap
        +-> renderer / temporary compatibility outputs
```

### Measured evidence at the current checkpoint

At `0.248 s`, the existing normal and full-tank-size-4 probes report:

| Metric | Normal | Full-tank minimum 4 |
| --- | ---: | ---: |
| Pressure-required rows | 1,296 | 24 |
| Accepted topology leaves | 4,665 | 178 |
| Published adaptive nodes | 5,717 | 357 |
| Published leaf edges | 16,493 | 935 |
| Dense predicted surface cells | 6,912 | 6,912 |
| Extension-band faces | 13,383 | 10,711 |
| Approximate probe wall time | 1.80 s | 1.55 s |

Pressure rows shrink by 54 times and topology leaves by 26 times, while dense
surface work does not shrink at all and the finest-face extension band shrinks
only modestly. This is direct evidence that the remaining problem is authority
and work-domain choice, not pressure coarsening quality.

The four full-lattice velocity staging passes are also semantically inverted
for a coarse run: every finest site reconstructs through the compact coarse
hierarchy. Accepted finest-MAC staging has measured about `0.67 ms` normally
and `1.40 ms` in the full-tank-size-4 arm.

## Target state and buffer contract

Introduce a versioned `LosassoAdaptiveState` with fixed-capacity accepted and
candidate banks. It should be owned by the Losasso backend, not by
`WebGPUOctreeCoarseSummary`.

| Record | Required contents | Identity/invariant |
| --- | --- | --- |
| Control | topology epoch, surface generation, velocity generation, counts, capacities, validity, error bits, dispatches | One published tuple; every consumer validates all generations it reads. |
| Leaves | origin, span, flags, eight compact node slots | Stable sorted leaf identity `(origin, span)`. |
| Nodes | finest-lattice node item, flags, local scale, constraint offset | The item is the permanent logical key; the deterministic compact slot is storage only. |
| Node directory | node item to compact slot | Bounded adaptive lookup with published load/probe receipts; no finest-lattice-sized inverse table is required. |
| Constraints | 2 edge masters or 4 face masters plus exact dyadic weights | Hanging nodes are derived values, never independent degrees of freedom. |
| Directional adjacency | nearest atomic neighbor slot and physical span for `-x,+x,-y,+y,-z,+z` | No edge jumps over an intermediate T-junction node. |
| Phi banks | accepted old/new scalar values and validity | Raw fluid phi; rigid carving is a derived query. |
| Nodal velocity banks | accepted extended vector, predictor extended vector, component-valid flags | Same graph and epoch as phi. |
| Band worklists | interface nodes/edges, redistance nodes, extension nodes | Published from adaptive distance and physical reach, not finest-domain scans. |
| Receipts | constraint residual, missing lookup, coverage, redistance residual, volume, invalid samples | Any nonzero structural error prevents publication. |

Use one allocation with bank offsets or stable accepted buffers plus an
in-stream commit. Do not rotate public `GPUBuffer` objects: existing bind-group
caches assume stable identities.

### Graph construction

The current node census should be replaced by graph v2, not incrementally
burdened with physics fields. Graph v2 reuses the existing node item as its
logical key but does not reuse the racing append slot or the overlapping edge
list as solver state.

1. Consume the candidate leaf list directly from the topology publisher. If
   that list is not yet available, enumerate resident candidate rows/owner
   payloads adaptively and deduplicate leaf identities; do not scan the full
   cell or node lattice. The topology builder should become the permanent leaf
   source before production cutover.
2. Emit the eight finest-lattice node items per leaf, radix-sort, and unique
   them. Reuse the existing GPU radix-sort infrastructure where its key bounds
   fit. Sorted items define deterministic compact storage slots; atomic append
   order never does.
3. Build a bounded item-to-slot directory and resolve every leaf's eight
   corner slots. Directory capacity scales with published nodes, not total
   finest-lattice nodes.
4. Classify each node as independent, edge-hanging, or face-hanging. Publish
   masters and exact weights. A node must have exactly one canonical
   classification.
5. Build nearest directional adjacency. Split a coarse geometric edge at every
   intermediate T-junction node; never use the current overlapping leaf-edge
   records as shortest-path edges.
6. Validate reciprocal adjacency, positive physical spans, leaf corner
   closure, constraint acyclicity, 2:1 assumptions, graph epoch, and all
   capacities before setting `published`.

The graph should be deterministic for a fixed topology. Sorted node items and
leaf identities make receipts, remaps, and coupled A/B comparisons
reproducible.
Generation-stamped per-item claims may remain as a diagnostic publication
technique, but finest-lattice-sized claim or field buffers are not part of the
final authority.

## Algorithms on the adaptive authority

### 1. Sampling phi

`samplePhi(position)` must:

1. find the accepted owner leaf containing the position;
2. fetch that leaf's eight node slots;
3. read constrained nodal phi values;
4. trilinearly interpolate in the leaf's local `[0,1]^3` coordinates.

Boundary policy is explicit. Closed walls use the same outward unit-slope
extension as transport/redistance; the open top may return exterior air. A
missing leaf, node, or valid generation is an error, not `phi = +h/2` and not
zero.

### 2. Topology handoff

At the tail of step `N`, build graph candidate `T+1` beside topology candidate
`T+1`, while accepted graph `T` and finalized phi generation `S` remain live.

For each candidate node:

- exact retained coordinate: copy phi by coordinate key;
- newly refined coordinate: sample accepted graph `T` in its containing leaf;
- coarsened corner: copy the surviving coordinate exactly;
- newly hanging node: compute from its candidate masters after master values
  are available;
- newly independent node: initialize from the accepted constrained field at
  that coordinate.

First publish a pure transfer receipt: retained values, prolongation values,
constraints, zero-crossing displacement, signed-distance residual, and volume
must all be measurable before any repair mutates the candidate. If the changed
topology requires signed-distance repair, redistance the candidate exactly
once, then use the same localized volume corrector as the recurring update.
Do not redistance both the finalized accepted field and an unchanged candidate,
and report whether topology transfer performed either mutation. Validate that
every candidate leaf needed by the interface/backtrace closure could sample
the accepted graph. Only after graph, phi, face velocity, nodal velocity,
pressure rows, and owner pages all pass does the ready-head commit publish
`T+1`.

On exact topology reuse, retain graph identities and phi; update generation
stamps without remapping.

This transaction is the cure for the prior repeated
adaptive-to-dense-to-adaptive filtering. It must not introduce a replacement
filter: stationary refine/coarsen churn has to show non-growing zero-crossing,
profile, and volume error, and an unchanged topology must be bit-preserving.

### 3. Semi-Lagrangian phi transport

Advect only independent adaptive nodes, using the previous projected and
extended nodal velocity on the same accepted graph:

1. evaluate velocity at the node;
2. midpoint-backtrace in world units;
3. sample old phi through the accepted leaf interpolant;
4. apply inflow and boundary policy;
5. project hanging-node values from their masters;
6. publish only if every required characteristic had graph coverage.

Start with the paper's first-order update. Add MacCormack/BFECC only after the
first-order adaptive oracle is stable; it requires a second adaptive predictor
velocity and a bounded donor clamp, not a return to dense scratch.

### 4. Redistance

Do not implement redistance as `min(neighbor + edgeLength)`. That computes
graph/Manhattan distance, not the Eikonal solution.

1. Detect sign changes on atomic adaptive edge segments.
2. Seed endpoint distances from the linearly interpolated zero crossing.
3. For a trial node, take the smallest accepted magnitude from each available
   axis and solve the upwind quadratic with the corresponding unequal physical
   spacings. Missing T-junction directions are omitted, matching the paper.
4. Use a deterministic bucketed march or a fail-closed fast-iterative ping-pong
   schedule. A parallel schedule must report its maximum update residual and
   reject an unconverged generation.
5. Reapply T-junction constraints after every write phase.
6. Stop at a physical narrow-band radius sufficient for topology sizing,
   characteristic reach, pressure ghosting, velocity extension, and rendering.

The GPU update must publish a monotone convergence receipt, maximum residual,
and reached physical band width. An unconverged generation is rejected rather
than accepted on visual plausibility.

### 5. Volume control

Measure liquid from adaptive leaves, not from finest cells or row-centre signs.
Use a fixed tetrahedral decomposition (or an equivalent deterministic
trilinear quadrature) of each leaf's eight nodal values and multiply by the
leaf's physical volume.

The production adaptive path is evidence-only. Immediately before redistance,
capture each leaf's represented volume. Immediately after redistance,
remeasure every leaf with the exact same integrator and publish signed total
drift, maximum absolute per-leaf drift, and total absolute drift. Coherent
graph/generation identity, complete live-leaf coverage, finite values, and
hanging-constraint closure are fail-closed requirements. The magnitude of
physical drift is telemetry: it neither mutates phi nor rejects publication.
Receipt word 43 is the ordinary Boolean transaction receipt (`1` valid, `0`
invalid).

The deleted experiment attempted to repair arbitrary post-step phi error with
bounded receiver patches, scalar offsets, and a migrated local-debt ledger. It
made symmetric topology exact but left scalar results dependent on reduction
order and produced visibly wild geometry after topology cutover. It is not a
dormant option or fallback; its buffers, pipelines, debt migration, tolerances,
and measure-only switch are removed.

Future mass conservation follows the concrete architecture in
`docs/papers/massConservingLiquids.txt`, not another post-step phi-offset patch:

- conservatively transport leaf-integrated mass/density alongside geometric
  nodal phi;
- partition mass exactly on refinement and sum it exactly on coarsening;
- trace mass lost during sharpening/remapping and scatter it locally along the
  level-set gradient within bounded `D*h` support;
- exclude solids and renormalize the remaining redistribution weights;
- persist sub-grid mass until the contour can represent it again;
- keep phi as the geometric/signed-distance authority, coupled to—but not
  overwritten by—the conserved mass field.

The current monotone higher-order phi-transport work is only an accuracy and
diffusion reduction. It is not the conservation mechanism above and must not
acquire a volume target, global scalar offset, post-redistance phi mutation, or
hidden repair fallback. The future correction is local because every mass
change produced by interface sharpening is returned by tracing along the
surface-density gradient toward the `0.5` contour, stopping at a bounded
`D*h` distance or a solid, then scattering with normalized non-solid weights.
Its conserved quantity survives topology publication independently of the
geometric phi field.

Store raw fluid phi. Rigid carving is derived as `max(rawPhi, -solidSdf)` for
pressure and rendering; it must not silently change the no-solid liquid
reference volume.

### 6. Pressure coupling and topology sizing

Replace row-centred restriction from the dense tracker with leaf-local
derivation from nodal phi:

- centre phi: trilinear value at `(0.5,0.5,0.5)`;
- min/max: the conservative corner extrema of the trilinear interpolant;
- gradient: analytic trilinear gradient at the query, expressed in world
  units;
- ghost distance: sample the same adaptive field on the wet-to-air dual and
  compute the zero-crossing fraction;
- interface/refinement evidence: query the finalized adaptive field directly.

The local uncommitted span-aware `trackerPhi` change is directionally correct:
a coarse row must not import a finest-cell gradient. It should remain as a
diagnostic improvement if separately accepted, but it does not remove the
dense authority and is not a substitute for this cutover.

### 7. Adaptive nodal velocity and extension

Keep projected velocity on the existing adaptive face authority. Replace the
finest-MAC and finest-node staging path with direct reconstruction on graph
nodes.

For each node and component:

1. inspect the four incident component-normal face locations;
2. choose the coarsest incident face scale;
3. area-average finer transition subfaces into temporary faces at that scale;
4. average the four same-scale values in a canonical order;
5. mark the component seeded only when its liquid/boundary inputs are valid.

This is the Section 3 rule. The current `stagedRawVelocity` neighborhood and
ray fallbacks should not be carried forward as hidden topology repair.

Extend nodal vectors out of liquid in increasing `abs(phi)` order over the
adaptive directional graph. A node uses already accepted closer-to-interface
neighbors; if none exist it remains trial until a later wave. Publish an exact
physical reach large enough for both phi and face characteristics. Build the
same adaptive field for the forward predictor before the reverse MacCormack
face pass.

Both phi transport and adaptive-face advection then use one function:
`sampleVelocity(position, fieldBank)`, implemented by containing-leaf
trilinear interpolation of the eight nodal vectors.

### 8. Rendering and compatibility

Rendering should eventually traverse the accepted leaf graph and sample its
nodal phi directly. During migration, a dense texture or compact row arena may
be materialized after finalization, with these restrictions:

- it carries the source graph epoch and surface generation;
- it is never read by transport, redistance, volume, pressure ghosting,
  topology sizing, or velocity extension;
- deleting it must leave every physics receipt unchanged.

That final property is the test that a compatibility view is truly a view.

## Step and commit order

The intended recurring sequence is:

```text
HEAD
  commit validated {topology, graph, phi, face velocity, nodal velocity}
  transport phi on accepted graph with carried post-projection nodal velocity
  redistance phi; correct volume; derive pressure-row surface data
  advect adaptive face velocity with accepted nodal velocity
  apply forces; solve pressure; project adaptive faces
  reconstruct and extend post-projection nodal velocity
TAIL
  build next topology and graph candidate
  remap finalized phi and projected face velocity into candidate
  redistance/correct candidate phi; reconstruct candidate nodal velocity
  jointly validate candidate transaction
```

Cold start has no special held-surface exception: publish analytic phi on the
cold graph, zero face velocity, and a valid zero nodal field before the first
positive-time transport.

## Direct cutover sprint

This is one implementation sprint, not a sequence of shadow tranches. Small
commits are encouraged for bisectability, but each commit advances the live
adaptive authority. Do not spend the sprint building CPU oracles, standalone
fixtures, manufactured-field tests, or a permanently selectable legacy lane.

### Sprint 1 — Build the live graph and state transaction

- Add `WebGPUOctreeLosassoSurfaceGraph` with fixed-capacity accepted/candidate
  banks, stable node-item keys, deterministic compact slots, leaf corner
  slots, explicit constraints, and nearest directional adjacency.
- Build candidate graph records directly from candidate leaves and wire graph
  validity into the existing topology ready/commit transaction.
- Add compact accepted/candidate phi and nodal-velocity banks plus generation,
  capacity, coverage, constraint, and adjacency receipts.
- Bootstrap cold phi from the analytic/imported scene source. Use the current
  finalized dense phi only as a one-time warm-start bridge when attaching to a
  running simulation; it must never be read again after the cutover commit.
- Keep `WebGPUOctreeAdaptiveNodes` only long enough to compare live node/edge
  counts during the dam-break A/B, then remove it from recurring execution.

The sprint does not pause for standalone graph validation. Invalid constraint,
adjacency, capacity, or epoch receipts fail the candidate transaction in the
live coupled run.

### Sprint 2 — Cut scalar authority over immediately

- Implement containing-leaf phi sampling, independent-node transport,
  constraint projection, physical-span Eikonal redistance, localized volume
  correction, and topology handoff on the live graph.
- Switch in the same cutover change: transport, redistance, volume, compact
  row phi/interval/gradient, pressure ghosts, refinement evidence, band
  membership, fine-seed publication, and renderer/diagnostic derivation.
- Stop encoding dense prediction, dense redistance, dense volume correction,
  dense-to-row restriction, and every dense-complement physics read.
- If rendering still needs the dense tail, materialize it one-way after
  adaptive finalization. No physics shader may bind it.

There is no half-authoritative fallback. A failed adaptive receipt holds the
last coherent accepted adaptive tuple; it does not resume dense physics.

### Sprint 3 — Cut velocity staging and extension over

- Reconstruct nodal velocity from the adaptive face directory at the coarsest
  incident-face scale for accepted and predictor fields.
- Extend nodal velocity over the atomic directional graph in phi order and
  derive leaf-scale air faces from constrained nodal values.
- Switch both phi and adaptive-face characteristics to the containing-leaf
  nodal velocity sampler.
- Remove the four finest-MAC/finest-node staging passes, span-one coarse-air
  records, Jacobi face extension, staged arenas, and their finest-domain
  capacity terms.
- Reconstruct and validate candidate nodal velocity before the joint topology
  commit. Missing coverage remains invalid; it never fabricates zero.

Sprint 2 and Sprint 3 should be developed back-to-back on the same cutover
branch. The lane is not accepted as fully adaptive until both are complete.

### Sprint 4 — Delete the former authority and tune only adaptive work

- Delete the factor-one Losasso dependency on the three dense phi banks,
  activity masks, dense redistance/volume kernels, dense-complement authority
  flag, coarse-only recurring transport, and staged velocity arenas.
- Keep generic Power or factor-4/8 code only where it belongs to a separately
  named backend.
- Specialize exact topology reuse, compact active bands, and fuse graph passes
  only after the two coupled validations are stable.
- Attribute performance using graph counts and pass timestamps from the normal
  and full-tank-size-4 dam-break runs.

## Cutover validation loop

Validation is deliberately limited to the two coupled scenarios that exercise
the target lane. Runtime structural receipts are implementation invariants,
not a separate unit-test program.

### 1. Symmetric expansion

Run the existing one-step check after every material physics change. Use the
three-step check at sprint boundaries and the existing long run only before
declaring the cutover complete. Require:

- no mixed generation or candidate publication error;
- exact expected symmetry receipts;
- no new volume or diagonal asymmetry;
- finite pressure, phi, and velocity state.

### 2. Dam-break UI A/B

At `0.248 s`, compare the pre-cutover baseline with both new configurations:

- normal refinement (`FLUID_REFINEMENT_REGION_FLOOR=0`);
- full-tank minimum size 4 (`FLUID_REFINEMENT_REGION_FLOOR=4`, scope `full`).

Judge the complete surface profile, volume, zero crossing, ridge, topology and
graph counts, invalid receipts, and visible wall/interface behavior together.
The normal lane may change because the discretization changed, but it must not
show a systematic reservoir rise, material liquid loss, instability, or a
new seam. The size-4 lane must be smooth at its authored scale and must encode
no dense surface or velocity staging passes.

Current commands:

```sh
FLUID_SAMPLE_TIMES_S=0.248 FLUID_REFINEMENT_REGION_FLOOR=0 \
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" \
node --import tsx tools/probe-dam-surface-shape.ts

FLUID_SAMPLE_TIMES_S=0.248 FLUID_REFINEMENT_REGION_FLOOR=4 \
FLUID_REFINEMENT_REGION_SCOPE=full \
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" \
node --import tsx tools/probe-dam-surface-shape.ts

npm run test:webgpu:symmetric-expansion:one-step
npm run test:webgpu:symmetric-expansion:three-step
npm run test:webgpu:symmetric-expansion:twenty-step
```

The twenty-step lane is the mandatory development tripwire after every
coherent adaptive change. It samples every `0.004 s` accepted state and gates
D4 volume, velocity, pressure, RHS, diagonal, and exact topology symmetry.
Stop on its first failure before returning to the dam A/B; the one-step lane
remains the finer pressure-stage diagnostic when that failure is in the solve.

Do not run Dawn concurrently with browser WebGPU. Use the repository's
isolated launchers for longer acceptance runs.

### A/B performance evidence

Report, per step and per topology epoch:

- accepted leaves, independent nodes, constrained nodes, atomic edges;
- transported/redistanced/extended nodes;
- scalar and velocity graph rebuild invocations;
- compatibility materialization invocations;
- GPU time per adaptive pass;
- allocated bytes by graph, phi, nodal velocity, and compatibility output.

The decisive assertion is not merely “faster.” In the full-tank-size-4 B arm,
transport/redistance/volume/velocity work and persistent physics allocation
must correlate with adaptive graph counts. A finest-domain-sized physics
dispatch or field allocation is a failed cutover even if its current
milliseconds are small.

## Non-negotiable invariants

1. Exactly one recurring phi authority exists in the fully adaptive lane.
2. Graph, phi, face velocity, and nodal velocity publish one coherent topology
   epoch.
3. A compatibility view never feeds physics.
4. Hanging-node constraints are enforced after every field write.
5. No invalid lookup silently becomes zero velocity, `+h/2` phi, or a held
   previous value.
6. Candidate failure cannot partially update accepted state.
7. Topology reuse does not remap or filter phi.
8. Redistance solves an Eikonal update with physical spans; edge shortest path
   is not accepted as a substitute.
9. Volume is measured and corrected on the same representation that advances.
10. Cold start publishes a complete valid state before positive time.

## Main risks and their containment

| Risk | Why it matters | Containment |
| --- | --- | --- |
| T-junction ambiguity | Independent hanging values create cracks and inconsistent marching. | Explicit constraint records, atomic segments, runtime residual gate, symmetric-expansion and dam-profile evidence. |
| Topology-remap mass loss | Coarsening discards interior shape even with perfect interpolation. | Pure-transfer receipts, candidate redistance only when required, same-integrator localized correction, dam A/B volume/profile checks. |
| GPU FMM ordering | A naive parallel minimum is not the Eikonal solution. | Unequal-spacing quadratic, monotone convergence receipt, fail closed, and dam profile/distance evidence. |
| Surface/velocity dependency cycle | Surface needs prior extended velocity; extension needs current phi and projection. | Carry post-projection nodal velocity in the accepted state; cold zero field; explicit head/tail transaction. |
| Nondeterministic atomics | Atomic append order changes identities and reductions. | Sort by coordinate/leaf key; canonical sums or fixed point; never persist append slots as identity. |
| Sparse coverage holes | Backtraces or transition stencils may leave the graph. | Physical support-closure validation in the candidate transaction; reject rather than hold. |
| WebGPU binding limits | Adding graph fields to existing large groups can exceed portable limits. | Small entry-point-derived layouts; stable arena bindings; keep graph samplers compact. |
| Misleading compatibility success | A dense output can quietly become input again. | Explicit dependency audit plus pass/binding census in both dam-break A/B arms. |
| False normal-regression gate | Bit equality to the old cell-centred method blocks the intended method. | Use the old normal profile only as the A side of the cutover comparison; accept the new physical profile without retaining a legacy backend. |

## Recommended first pull request

The first pull request is the cutover branch, not a schema-only prelude:

1. Add graph-v2 accepted/candidate storage and runtime receipts.
2. Add compact nodal phi and velocity banks plus topology handoff.
3. Implement adaptive scalar transport, redistance, volume, derivations, and
   adaptive nodal reconstruction/extension.
4. Switch every scalar and velocity consumer to the new state and stop
   encoding the former dense authority and staging passes.
5. Iterate on that live branch using symmetric expansion and the two dam-break
   UI A/B arms until the coupled gates pass.
6. Delete the former factor-one authority before merging; retain at most a
   one-way render materialization output.

Commits within the pull request should follow the sprint boundaries so a bad
change is bisectable, but the pull request is complete only when the authority
has flipped and the dense recurring path is gone.

## Implementation checkpoint — 2026-08-09

The live factor-one lane now owns an accepted/candidate adaptive surface graph,
nodal phi, adaptive face velocity, and reconstructed nodal velocity. The
architecture review for this checkpoint deliberately excluded the legacy
factor-4/8, global-fine, Power, W7, and render-adapter paths.

Completed correctness and structural changes:

- adaptive phi transport, redistance, pressure derivation, topology evidence,
  and topology handoff consume the compact accepted graph;
- accepted recurring phi passes use GPU-published live node, leaf, row, and
  face dispatch records instead of capacity-sized launches;
- warm invalid topology candidates publish zero candidate work without
  poisoning the accepted phi error state; cold invalid bootstrap remains
  fail-closed;
- velocity reconstruction consumes fixed 16-entry compiled face stencils;
  face-directory probing is confined to stencil compilation at topology
  publication;
- accepted reconstruction reads topology-migrated carried face values after a
  topology commit, rather than stale projected values in the prior face-ID
  order;
- arbitrary adaptive velocity sampling uses a graph-owned sparse leaf locator:
  the owner arena directly maps logical brick to resident physical page, and
  one page-local locator load returns the compact graph leaf slot. The recurring
  sampler has no dyadic span walk, binary search, hash probe, or
  topology-dependent indirect-memory loop;
- each locator bank contains a 16-word transactional header plus
  `ownerCapacity * 512` slots. Compilation and commit cover exactly
  `residentPageCount * 512` support cells at topology publication, not the
  finest domain;
- accepted and candidate volume evidence captures exact per-leaf represented
  volume before redistance and remeasures with the same integrator afterward.
  Coherent epoch/generation/counts, full leaf coverage, finite values, and
  hanging-constraint closure remain fail-closed; signed/max/absolute drift is
  telemetry and never mutates phi;
- scalar transport retains the `1/65536` phi lattice but snaps toward zero.
  This removes reflection-sensitive half-quantum ties without coarsening the
  field; the measured maximum raw reflected difference was only 0.0513 of one
  lattice quantum.

Before the patch-offset experiment became visible, the three-step
symmetric-expansion Dawn gate reported:

| Receipt | Result |
| --- | ---: |
| Volume, topology, diagonal D4 | exact |
| Maximum velocity D4 error | `2.9802322387695312e-8` |
| Maximum RHS D4 error | `3.814697265625e-5` |
| Maximum pressure D4 error | `3.662109375e-4` |
| Validation/non-finite errors | 0 |
| Accepted graph | 3,112 leaves, 4,157 nodes, errors 0 |

The post-locator dam-break UI A/B at `0.248 s` also passes its current
structural and profile gates:

| Metric | Normal | Full-tank minimum 4 | Reduction |
| --- | ---: | ---: | ---: |
| Accepted leaves | 2,509 | 150 | 16.7x |
| Adaptive nodes | 3,394 | 316 | 10.7x |
| Pressure-required rows | 960 | 13 | 73.8x |
| Node dispatch workgroups | 54 | 5 | 10.8x |
| Leaf dispatch workgroups | 40 | 3 | 13.3x |
| Seeded velocity receipt | 1,183 | 32 | 37.0x |

Both arms report zero validation, capacity, owner-coverage, graph-coverage,
adjacency, leaf-closure, and pressure-row mapping errors. Neither has an
interior ridge or enclosed dry column. The normal arm has leaf sizes
`1:2016, 2:476, 4:17`; the minimum-4 arm has `2:96, 4:48, 8:6`, where the
size-2 leaves are required transition/interface resolution rather than a dense
size-1 fallback. Wall-clock times from these cold/warm development probes are
not a controlled performance comparison and are intentionally not used as a
speedup claim.

### UI robustness checkpoint

The dam-break and symmetric-expansion presets now select the validated
factor-one profile directly: Losasso coarse pressure, causal-front velocity
extension, maximum leaf size 16, four interface-band cells, and global-fine
level-set factor 1. Factor-4 diagnostic lanes override this preset explicitly
instead of relying on the old UI defaults.

An exact two-step dam-break Dawn reproduction exposed a redistance receipt one
short of its required support (`3111 / 3112`). The unresolved independent node
was surrounded by constrained hanging nodes. Their projection had averaged an
unreached `FAR` sentinel with a reached master, producing a very large but
technically finite value that blocked the wave. Constraint projection now
propagates the minimum reached master until every master is reached, then uses
the canonical two- or four-master interpolation. Exact-zero interface samples
are also admitted as zero-distance seeds. Diagnostic failure readback can
report the graph slot, position, mask, both distance banks, and neighbours;
this is part of the explicit audit path and adds no recurring readback.

The exact Dawn reproduction now accepts and encodes both `0.004 s` steps,
publishes 960 pressure rows and an 18,882-vertex compact-coarse surface, and
reports zero validation and non-finite errors. The repaired shader also passes
the three-step symmetric-expansion Dawn gate with exact topology D4 at all
three checkpoints.

The later controlled symmetric A/B isolated the failure. With patch-offset
repair active, graph topology remained D4-exact but phi had 20,872 nodal and
5,932 renderer mismatches (maximum `1.220703125e-4`). With the same transport,
redistance, and topology but no phi mutation, graph, phi, and renderer were all
bit-exact D4 and both raster surfaces remained coherent without holes. That
evidence is why the patch-offset/debt implementation was deleted rather than
optimized. The evidence-only transaction must now be rerun through the exact
Dawn gates before an updated long-run physical claim is made.

Browser acceptance uses the same presets and therefore follows the Dawn-tested
path. Dam break reaches `0.008 s` after two manual steps with queue-confirmed
GPU completion, 960 pressure rows, and compact-coarse mesh generation 3.
Symmetric expansion reaches `0.012 s` after three manual steps with 1,120
pressure rows and compact-coarse mesh generation 4. Both show all instrumented
gates clear and no rejected-generation or missing-surface banner. The
diagnostics panel now identifies this source as `COMPACT COARSE`; deferred
surface-receipt recovery is scheduled independently of mesh extraction so the
250 ms extraction throttle cannot leave a healthy publication labelled empty.

The strict factor-one architecture audit found no unconditional recurring
finest-domain physics allocation or sweep. Its remaining ranked gaps are:

1. replace topology-sizing span/hash queries with topology-compiled direct
   evidence records;
2. skip candidate graph/remap construction when topology is exactly reused;
3. replace fixed full-graph redistance and velocity wave envelopes with
   compact live worklists while preserving convergence receipts;
4. remove serialization in whole-graph volume/evidence reductions where it is
   still measurable after the correctness gates above.

These are performance/architecture follow-ups and must not reintroduce a dense
compatibility authority or recurring topology search. Conservative transported
mass remains a separate correctness milestone under Section 5; the current
accepted and candidate paths publish evidence only.

### 2026-08-10 physics-parity audit: symmetry is necessary, not acceptance

The twenty-step symmetric-expansion gate remains mandatory after every
adaptive-physics change. It establishes D4 equivariance and catches ordering,
bank, and topology asymmetry early, but it does **not** establish dam-break
quality. The current adaptive path can pass all twenty symmetry checkpoints
while remaining materially more dissipative and slower than the retained HEAD
oracle.

The controlled HEAD arm was rerun with coarse/global volume correction
disabled. At `0.496 s`, correction-disabled HEAD still has 1,507 wet samples
and reaches front cell 24; enabling its correction raises the wet count to
1,626 but does not move the front. The adaptive arm has reached only about
1,111 wet samples and front cell 16 in the comparable strict-2:1 run. Global
volume correction therefore explains neither the slow front nor most of the
loss, and must remain disabled while the representation and transport defects
are isolated.

The common comparison metric is a deterministic 4x4x4 quadrature of the
published trilinear zero set in every finest logical cell. It reports volume,
centre of mass, and front position in finest-cell units for both HEAD and the
adaptive path. Wet-sample count remains useful supporting evidence but is not
a geometric volume metric.

Current ranked evidence is:

1. **Scalar transport is the dominant continuous loss.** Accepted steps lose
   roughly 15--20 quadrature cells while accepted redistance and candidate
   handoff receipts are nearly neutral. Disabling bounded MacCormack in favour
   of simple semi-Lagrangian phi transport makes the dam substantially worse,
   so the production default remains bounded MacCormack. This is an explicit
   stopgap for Losasso section 6's missing particle-level-set correction, not a
   claim that the paper specifies MacCormack; Ando--Batty section 3.2 likewise
   prefers first-order transport near liquid surfaces. The first-order arm
   remains executable evidence until the missing correction stage is present.
   A corrected true-time trace at `0.112--0.136 s` further separates the slow
   front from this loss: every crossing leaf and every front departure donor is
   span one, while adaptive front nodal x velocity is only
   `0.138--0.150 m/s` versus HEAD's `0.305--0.403 m/s`. RK2 sampling removes a
   further 8 percent. Forcing all-fine topology recovers only about 23 percent
   of the adaptive-to-HEAD velocity gap. The front-speed defect therefore
   begins upstream in face velocity/projection/reconstruction, not in a coarse
   scalar donor or marching.
2. **Redistance publication can freeze the visible simulation.** At the
   current `h=0.05`, the acceptance tolerance is `5e-6`, tighter than the
   `1/65536 m` published-phi quantum. Several steps barely miss that tolerance;
   the accepted bank then correctly remains unchanged, so the rendered zero
   set and front freeze despite nonzero transport work. Increasing the fixed
   iteration budget makes those steps publish but exposes still greater
   continuous transport loss. This is a real acceptance-policy defect and an
   important symptom, not the origin of the early loss.
3. **Topology support matters but is not the whole cure.** A one-cell
   factor-one protection rung improves late quadrature volume by about five
   percent without recovering HEAD front speed. Removing Power's stronger
   mixed-neighbour exclusion is still correct: a controlled A/B shows the
   strict-2:1 arm retains about 39 percent more late volume than the mixed-ring
   arm. A direct accepted-graph audit proves the initial zero-crossing shell is
   unit resolution and strongly sub-cell. Its first late-time report was
   invalid because the probe labelled requested checkpoint times while
   advancing only one step per checkpoint; true late-time crossing spans are
   being rerun. The earlier size-two "surface row" count is still a
   pressure-owner diagnostic, not a scalar crossing-leaf count. Do not add
   further interface refinement until the corrected accepted-graph audit
   supplies evidence for it.
4. **Pressure convergence is not the primary failure.** MGPCG reaches very
   small relative residuals while the surface depletes. Boundary-coefficient
   and rigid-aperture defects remain correctness bugs, but a more sophisticated
   T-junction pressure operator is a later manufactured-solution experiment,
   not the first dam fix.
5. **Velocity execution remains insufficiently adaptive.** Storage is compact,
   and accepted redistance now uses a compact frontier, but velocity
   reconstruction/extension still uses fixed full-graph wave envelopes.
   Replace the remaining velocity waves with compact live frontiers. Following
   *Tall Cells*, keep accurate
   extension within a two-cell interface band, restrict only known values to
   coarser hierarchy levels with renormalized weights, and prolongate back for
   far-field transport support. No production fix may recreate a
   finest-domain SDF or velocity sweep.

   A live receipt identified the immediate membership defect: at `0.240 s`,
   all 4,724 live adaptive nodes were scheduled for scalar transport. The band
   predicate used `abs(phi) <= reach`, while truncated far-field phi is exactly
   `reach`, so the clamp plateau became active work. The replacement must use a
   strict physical core plus an explicitly non-cascading transport-destination,
   donor, constraint, and topology-support halo; compacting the existing mask
   alone would merely compact the whole graph. A first bare `< reach` arm was
   reverted after the corrected true-time probe produced span-two crossing
   leaves by `0.112 s`, proving that readable retained plateau donors alone do
   not preserve the moving interface shell.

The first *Tall Cells* velocity slice is now integrated. The accurate causal
extension is limited to two finest-cell widths, reducing that schedule from 14
to 6 waves, while the existing sparse adaptive closure retains the full 7-cell
transport reach. Candidate transition reconstruction now omits unknown fine
faces and renormalizes the remaining exact area weights instead of discarding
the whole component. Existing repeated 2/4-master constraints supply the
coarse-to-fine T-junction interpolation. This reduces velocity wave-stage
dispatches by 19 percent and fixes a real migration defect, but each remaining
wave still dispatches the complete live graph; compact frontier scheduling is
still required. Dam loss remains severe after this slice, so it is not recorded
as a fidelity cure.

The adaptive-SDF slice is now integrated on the same accepted octree and nodal
phi authority. Scalar transport retains its full 7-cell topology-support set;
an A/B proved that adding an incident-leaf ring to that set legitimately
schedules every live graph node, so it is not a useful redistance frontier.
After transport, the accepted path rebuilds the same compact arena against the
transported bank using a 2-finest-cell core plus one immutable incident-leaf
ring. Candidate topology repair remains full-graph. Redistance initialize,
A/B sweeps, constraint projection, and finish consume the two indirect compact
lists. Mixed-leaf corners and the independent masters of constrained mixed
corners are encoded as frozen list entries and remain bit-identical.

At the corrected dam checkpoints `0.112--0.136 s`, transport has 4,208 live
nodes while accepted redistance uses 3,139--3,173 nodes, a 24.6--25.4 percent
reduction. All crossing leaves are unit-sized, and the explicit audits report
zero missing freeze bits, distance-marker mismatches, or master-closure
mismatches. Receipt words 52--53 publish the independent and constrained
redistance active counts. The final twenty-step symmetric gate passes with
exact volume, topology, and diagonal D4 symmetry, velocity error
`1.1920929e-7`, pressure error `7.32421875e-4`, RHS error
`2.145767e-4`, and zero validation errors. This removes recurring full-live
accepted redistance dispatch; it does not by itself repair scalar transport
loss.

The strongest current front-velocity defect is incomplete face-stencil
authority. Candidate reconstruction previously renormalized any known subset
of a component's face stencil and immediately set its validity bit. Both
causal and harmonic extension therefore skipped the partially supported
component. Partial restriction now retains its provisional numeric value but
does not publish validity; only complete geometric coverage is an extension
seed. The existing two-cell causal solve and sparse outer closure must repair
the component before final publication. The combined symmetric gate remains
green. In the corrected dam run, front nodal x velocity rises from the prior
`0.138--0.150 m/s` trace to `0.141--0.188 m/s`, while RK2 front velocity rises
from `0.127--0.138 m/s` to `0.129--0.153 m/s`; this is encouraging but still
well below HEAD's `0.305--0.403 m/s`. About 22 percent of front stencil weight
still lands on exact-zero faces, so a compact face-space owner/covering-face
support construction remains the next fidelity task.

The 32-cubed mini-dam robustness pass exposed a separate Section 6 extension
gap. Compiled independent-node packets stored each neighbour as a compact
packet id, but the propagation kernels consumed that word as a graph-node id
for phi lookup, T-junction constraint resolution, and frontier expansion. The
second lookup silently mapped an unrelated node. This violated Losasso's
nodal outward-extrapolation stage and Ando--Batty section 3.2's requirement
that velocity be extrapolated after redistancing. Compiled adjacency now keeps
the authoritative graph-node identity and maps to an independent packet only
at the execution boundary. At the eleven-step `0.044 s` Dawn checkpoint both
accepted and predictor fields changed from 42 unresolved nodes inside the 7h
characteristic reach to zero. The lane accepts with no rejected advances or
raster holes, reaches `0.813793 m/s`, keeps the largest one-step potential
energy increase at `0.00442221`, kinetic-energy drop at `0.0719548`, and
liquid-cell growth ratio at `1`. The 69-step `0.276 s` lane also accepts with
`0.264771 m` lateral spread, `3.35117 m/s` final speed, and `0.887423` retained
reconstructed liquid volume. The latter is now a required physics gate, so a
moving but rapidly disappearing level set cannot pass on speed alone.

The same A/B also sharpens the particle-correction gap. With the bounded
correction disabled, first-order semi-Lagrangian transport produced four tiny
enclosed raster cavities even after velocity reach became complete. Losasso
section 6 couples that simple transport to a particle level set; this lane does
not yet implement that correction. Production therefore retains bounded
MacCormack as an explicit stopgap, while `FLUID_ADAPTIVE_PHI_MACCORMACK=0`
remains the executable paper-comparison arm. This exception must disappear or
be re-evaluated when particle/local feature correction is implemented.

Immediate acceptance order is therefore: keep the twenty-step D4 gate green;
retain and continuously assert the existing unit-sized crossing shell; keep
the compact accepted-redistance freeze receipts exact; construct complete
compact face support before promoting a nodal velocity component to valid;
compact the remaining velocity frontiers; compare interface nodal velocities
and departures numerically against HEAD; then repair the transport
representation. Local conservative
feature repair remains a later independent milestone. Do not add global volume
correction to conceal any of these differences.

## Current worktree hygiene

This checkpoint sits in a broad dirty worktree containing unrelated renderer,
scene, foliage, and UI work. Preserve those changes. Validate the adaptive
lane with focused files and the coupled Dawn gates above; do not use a broad
reset or interpret unrelated repository-wide type/test failures as adaptive
regressions without tracing them to an edited adaptive file.

## Handoff definition of done

The fully adaptive Losasso effort is complete only when all of the following
are true:

- setting a regional minimum leaf size changes the resolution of surface and
  velocity operators, not only pressure;
- factor-one recurring physics has no finest-domain phi or velocity staging
  allocation/dispatch;
- one accepted graph/field generation supplies transport, redistance, volume,
  pressure boundaries, topology sizing, and rendering;
- topology changes remap and commit the complete state atomically;
- symmetric expansion passes its one-step, sprint-boundary three-step, and
  final long-run checks;
- the normal and full-tank-size-4 dam-break UI A/B arms pass the coupled
  profile, volume, stability, wall/interface, receipt, and structural-work
  contract;
- the coarse performance gain is attributable to adaptive surface and
  velocity work in pass timestamps and invocation counts;
- deleting every dense compatibility output leaves physics receipts
  unchanged.
