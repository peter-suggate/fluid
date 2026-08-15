# Sparse CM12 surface-activity resolution policy

Status: the GPU candidate selector is implemented; authoritative topology
publication remains a separate, GPU-only transaction. The default criterion is
deliberately simple: every free-surface or thin-fluid brick requests `8^3`, and
the rest grades outward through `4^3 -> 2^3 -> 1^3` with strict 2:1 face
balance. Velocity/deformation/temporal/detail signals are an explicit opt-in
criterion, not part of the default.

## Current failure and decision

`WebGPUAdaptiveMassSolver.createAsync` first asks scene initialization to keep
one half of the resident bricks fine. It then calls
`coarsenLargeQuiescentComponents`. For any connected wet component larger than
eight bricks, that function conservatively converts saturated fine bricks to
`4^3`, retaining interface bricks and at most one deterministic fine seam seed.
The mini dam break is one such component, so the deliberate negative fine half
collapses to a diagnostic seed. Component size is neither activity nor an
accuracy measure and must not remain the production policy.

The production selector is a hysteretic, brick-local surface/activity policy:

- absent bricks stay absent and receive zero classification work;
- fully interior, quiescent liquid may descend to `1^3`;
- every free-surface brick is `8^3`, independently of its
  activity score;
- resolution changes occur at a topology epoch, never in the middle of a CM12
  step;
- the accepted atlas remains authoritative if any planning, transfer, balance,
  capacity, or validation receipt fails.

This follows the useful parts of Ando and Batty's surface-adaptive sizing
method (`docs/papers/ando-batty-2020-practical-octree-liquid-simulator.txt`,
sections 6.1--6.6): evaluate activity near the interface, retain it over time,
and grade transitions. We deliberately start with cheaper dimensionless
signals already present in the Sparse CM12 state. CM12 density and persistent
gamma remain the surface/mass authority; a new level set is not introduced.

## Activity measured on accepted state

Classification runs after an accepted projected step and reads only the
compact cells and authoritative composite face rows. It must not materialize or
scan the finest presentation lattice.

A brick is in the surface band if either:

1. one of its cells has `0.05 < rho < 0.95`; or
2. one of its composite rows straddles the CM12 `rho = 0.5` isovalue, treating
   sparse air as `rho = 0`.

Surface-band membership is a hard fine-resolution floor. In the optional
`Surface + activity` criterion, bulk fluid also measures maximum accepted
velocity as finest-cell travel per step. That quantity selects a floor
independent of the brick's current level:

```text
travel = dt * max_rho>1e-5(length(u))       // finest cells per step

travel >= 1.00 -> 8^3
travel >= 0.50 -> 4^3
travel >= 0.25 -> 2^3
otherwise      -> 1^3
```

The surface floor wins over the velocity floor. In default `Surface distance`
mode, velocity, deformation, temporal change, and restriction detail do not
affect resolution.

Thin represented liquid is also a hard `8^3` floor, even when dilution makes
it too faint to cross the `rho = 0.5` surface. For each occupied composite leaf,
classification records the exposed positive and negative sides supplied by
its incidence rows. The leaf is thin when both sides of any axis are exposed
and its volume-fraction-weighted width is under two finest cells:

```text
representedThickness = clamp(rho, 0, 1) * leafWidthFineCells
thin = rho > max(residencyThreshold, thinFeatureThreshold)
    && representedThickness < 2
    && exists axis with exposedNegative(axis) && exposedPositive(axis)
```

This deliberately treats a one-cell sheet, filament, or dilute isolated leaf
as unresolved geometry and refines its whole brick. A two-cell-or-thicker body
with air on only its outer face is not classified as thin. Density below the
`0.005` residency threshold remains numerical residue and may be retired
instead of pinning fine topology. A brick must also contain at least one
finest-cell equivalent of integrated liquid mass; this rejects concentrated
subcell fragments that would otherwise pass a per-cell maximum test and pin a
whole region. These liveness cutoffs are deliberately distinct from CM12's
`1e-5` arithmetic dry epsilon.

For each surface brick compute these dimensionless values:

```text
shape = max(
  max_neighbor(1 - clamp(dot(n_i, n_j), -1, 1)) / 0.15,
  h4 * max_surface(abs(div(n))) / 0.25)

transport = max(
  dt * max_surface(abs(u)) / h4 / 0.25,
  dt * max_surface(frobenius(sym(grad(u)))) / 0.15)

temporal = max(
  max_surface(abs(rho - rho_previous_overlap)) / 0.05,
  max_surface(1 - clamp(dot(n, n_previous_overlap), -1, 1)) / 0.10)

activity = clamp(max(shape, transport, temporal), 0, 1)
scoreByte = roundToNearestEven(255 * activity)
```

`n` is the normalized density gradient derived from the same area/distance
weighted composite rows used by projection. A gradient below `1e-6` has no
normal and contributes zero to normal-change terms. Velocity is the accepted
post-projection velocity. Previous values are exact-overlap restrictions of the
last accepted snapshot; a missing history sample is `unknown`, not zero.

When an `8^3` brick is considered for demotion, add one signal that a `4^3`
brick cannot reconstruct after information has already been discarded:

```text
restrictionError = max_child(abs(rho8 - prolongConstant(restrictVolume(rho8))))
detail = restrictionError / 0.08
activity = max(activity, clamp(detail, 0, 1))
```

This is deliberately conservative. Constant prolongation is the stable first
rung, so a fine brick containing meaningful sub-coarse structure is not merged.
Later limited-linear prolongation can replace it and lower this measured error
without changing the policy API.

Reason bits are independent of the scalar maximum:

```ts
export const enum SparseSurfaceActivityReason {
  Surface       = 1 << 0,
  Deformation   = 1 << 1,
  Temporal      = 1 << 2,
  FineDetail    = 1 << 3,
  PredictedFace = 1 << 4,
  Unknown       = 1 << 5,
  Occupied      = 1 << 6,
  Energetic     = 1 << 7,
  ThinFluid     = 1 << 8,
}
```

The listed thresholds are reproducible defaults. Interactive overrides are
method parameters and therefore travel with captures/URLs, so an experiment
cannot silently change an A/B policy.

## Interactive experiment controls

The SIM panel's **Activity + resolution** stage exposes the selector rather
than requiring shader edits. Its controls fall into two deliberately explicit
classes:

- `Created-region floor`, `Initial fine band`, and `Receiver reach` are
  structural. They rebuild the accepted packed atlas. Until live topology
  publication is complete, setting the created-region floor to `8^3` is the
  conservative guarantee that a fast front cannot enter a coarse receiver.
- travel thresholds for the `8^3/4^3/2^3` rungs, front lookahead, thin-feature
  width/density, partial-surface bounds, restriction-error tolerance, topology
  cadence, promotion/demotion persistence, and score thresholds are runtime
  controls. They enter the next frame's small GPU uniform and change candidate
  requests without resetting simulation time.

The stage displays `CANDIDATE ONLY` while accepted composite rows remain
construction-time. This is intentional disclosure, not a selector mode: GPU
measurement, planning, 2:1 closure, and conservative transfer receipts run,
but their candidate fields do not become physics authority until the atomic
topology-publication transaction is implemented.

## Hysteretic state machine

Run a topology epoch every four accepted physics steps. Each resident brick
retains an eight-bit score and small counters:

```ts
export interface SparseBrickActivityHistory {
  readonly brickKey: number;
  readonly scoreByte: number;
  readonly reasons: number;
  readonly hotEpochs: number;
  readonly quietEpochs: number;
  readonly lastObservedGeneration: number;
  readonly previousSurfaceMassFineCells: number;
  readonly previousMeanNormal: readonly [number, number, number];
}

export interface SparseSurfaceResolutionPolicyOptions {
  readonly topologyCadenceSteps: 4;
  readonly promoteScore: 160;
  readonly emergencyPromoteScore: 224;
  readonly demoteScore: 96;
  readonly promoteEpochs: 2;
  readonly demoteEpochs: 1;
  readonly maximumPromotionLeafDelta: number;
  readonly maximumDemotions: number;
  readonly seamSentinel: "off" | "single" | "horizontal-d4-orbit";
  readonly sentinelAxis: 0 | 1 | 2;
  readonly sentinelSide: "negative" | "positive";
}
```

Transitions use a snapshot of all scores; coordinate iteration order cannot
affect another brick's result.

```text
coarse + score >= 224                       -> request fine now
coarse + score >= 160 for 2 epochs          -> request fine
fine   + no surface
       + no thin fluid
       + velocity floor below current level
       + score < 160
       + restrictionError <= 0.08 for 1 epoch -> request one rung coarser
otherwise                                   -> retain resolution
```

Refinement always wins over demotion. A brick changes by one rung in an epoch.
Newly resident receiver bricks start `8^3`, `Unknown`, with zero quiet epochs in
the first rung. A surface row whose outward characteristic can reach another
brick before the next topology epoch sets `PredictedFace` and immediately
requests that receiver fine. This uses
`topologyCadenceSteps * dt * max(abs(faceVelocity))`; it is a surface-motion
guard, not camera adaptation. A later calm-surface rung may create a `4^3`
receiver only when the predictor proves that the interface cannot reach it
before the following topology epoch.

Run a refine-only balance closure over all face neighbors after snapshot
classification. Three ordered passes close the four-level `1/2/4/8` ladder to
a strict 2:1 fixpoint. Omitted air is not a level and is not made resident
merely for balance.

Residency uses both a separate density threshold (`rho <= 0.005`) and a
one-finest-cell integrated-mass floor rather than exact floating-point zero or
CM12's much smaller arithmetic epsilon. A brick that fails either liveness test
may retire once it is outside directional surface support. Its discarded
integrated mass is published in the activity receipt and its fields are cleared,
so later receiver activation cannot resurrect stale residue. Actual retirement
receipts remain the authority for conservation diagnostics.

Every span-one surface/receiver brick retains candidate storage and topology
templates through `8^3`, independently of its authored resolution. Deep
quiescent liquid is represented by immutable dyadic macro-bricks; those leaves
allocate only their accepted cells and no candidate or fine-presentation page.
Splitting a macro is a sparse page-pool event, never a reason to preallocate its
covered fixed-brick volume. Runtime surface requests can therefore refine an
initially saturated span-one brick all the way to the fine rung without making
deep storage domain-shaped.

## Determinism and D4 symmetry

All reductions use fixed row/key order, dimensionless axis-symmetric formulas,
round-to-nearest-even byte quantization, and decisions from an immutable
snapshot. No random/hash tie breaking and no in-place neighbor propagation are
allowed.

Budget selection consumes whole equal-score buckets. If the next bucket does
not fit, defer the entire bucket rather than cutting it by brick key. This
preserves equal decisions for exactly symmetric inputs. Emergency requests may
exceed the budget because the alternative is knowingly losing an interface.
For a symmetric input with the seam sentinel off, the resolution map, score
bytes, reasons, and history counters must be exactly D4 equivariant.

The seam sentinel is an explicit method diagnostic, not a scene special-case:

- `off` is the production and physics/symmetry-gate setting;
- `single` retains one otherwise-demotable fine brick on the selected extreme
  side of each connected wet component, nearest its tangential centroid, and
  publishes `SeamSentinel`; this keeps a small visible live A/B seam;
- `horizontal-d4-orbit` retains the complete x/z D4 orbit of that seed for a
  symmetric seam diagnostic.

The sentinel is selected after activity scoring and before transfer. It must
not alter density, gamma, or velocity. UI diagnostics must distinguish pinned
fine bricks from activity-selected fine bricks. The present `fineHalf` option
can be removed once `single` is wired; retaining half a domain is too expensive
for a seam sentinel.

## GPU-only staggered publication design

Topology publication must not call `mapAsync`, `onSubmittedWorkDone`, rebuild a
CPU atlas, replace buffers from the host, or pause `advanceTo`. A prototype that
did this was immediately rejected: even compact readback introduces a queue
bubble and makes topology cadence visible as a hitch.

The accepted and preparing generations therefore live side by side on the
device. Storage is sparse-brick-capacity-shaped, never finest-domain-shaped:

```text
accepted brick levels + active bits
accepted compact cell state (resolution^3 slots per accepted leaf)
accepted compact cell/row worklists

shadow brick levels + preparation bits
shadow cell/face transfer slots (512 only for mutation-capable span-one leaves)
GPU topology-page free list (32..512 pages, scaled from the mutable frontier)
GPU-authored candidate cell descriptors (row/incidence publication-gated)
shadow local-row and seam-row descriptors
dirty queue: urgent segment | ordinary segment
indirect dispatch/count header
```

Fixed per-brick cell slots are an address space, not a compute commitment. The
accepted cell worklist contains only `resolution^3` live slots for each active
brick, and physics dispatches indirectly over that compact list. Row work is
likewise compact. This avoids the two bad alternatives: global prefix offsets,
which make one brick change relocate every later brick, and dispatching all 512
slots for every coarse brick, which would silently turn the adaptive method
into all-fine work.

An epoch is a device-side state machine:

```text
IDLE
  -> PLAN: snapshot evidence, request levels, close 2:1
  -> QUEUE: mark changed bricks plus their face-neighbour seam ring
  -> PREPARE: build shadow local topology in bounded round-robin batches
  -> READY: all dirty metadata prepared; accepted physics still advances
  -> COMMIT: transfer latest state, rebuild compact worklists, project, validate
  -> FLIP: one atomic accepted-generation change
  -> IDLE
```

Preparation may span frames because it writes only topology metadata. It must
not transfer evolving physical state early: that snapshot would be stale by the
time the generation flips. Cell/gamma/momentum and exterior-flux transfer runs
from the latest accepted state in the commit frame, followed immediately by a
zero-time composite projection and receipt validation. A failed receipt clears
the shadow transaction and leaves the accepted generation untouched.

The queue has two lanes:

- **urgent**: surface/thin-fluid promotion, swept-front receivers, and every
  balance brick needed to make those requests 2:1. Urgent work is prepared
  first and may exceed the ordinary budget. It is never delayed behind a bulk
  merge.
- **ordinary**: distance-driven coarsening and optional activity promotion.
  A device cursor consumes at most `prepareBricksPerFrame` entries, wrapping
  round-robin across epochs. Equal-score/D4 orbit buckets are indivisible.

Coarsening is lower priority than refinement. If new surface evidence touches a
brick while an older shadow transaction wants to merge it, the GPU cancels that
merge, raises the shadow requirement, and requeues its seam ring. If accepted
physics activates a dormant receiver during preparation, its `8^3` urgent
request is folded into the same transaction or starts the next one if shadow
capacity is already sealed.

Suggested initial budget is four ordinary bricks per frame, tunable from 1 to
64 after timings exist. The actual budget contract is work-based: preparation
stops when either the brick count or a row-descriptor count is exhausted. The
GPU header publishes `dirty`, `urgent`, `prepared`, `remaining`, `cancelled`,
and `generation` counters for diagnostics, but the host never consumes them to
schedule simulation.

The commit is atomic at generation granularity. Per-brick flips are forbidden:
pressure rows crossing a brick face must never observe one endpoint from the
accepted generation and the other from shadow. Presentation and scientific
overlays read the same accepted-generation slot as physics, so no separate
visual topology can get ahead of simulation.

## Conservative topology transfer

Build a complete candidate state beside the accepted state. The transfer is
over exact dyadic overlap and is independent of scene initialization.

### `8^3 -> 4^3`

For each coarse parent and its eight fine children:

```text
rho4   = sum(V8 * rho8) / sum(V8)
gamma4 = sum(V8 * gamma8) / sum(V8)
M4     = sum(V8 * rho8)
P4     = sum(V8 * rho8 * u8)
u4     = P4 / M4                         when M4 > epsilon
u4     = sum(V8 * u8) / sum(V8)          otherwise
p4     = sum(V8 * p8) / sum(V8)          warm start only
```

Normal face velocity is area averaged, so the total flux through every parent
face is unchanged. Tangential/collocated velocity is reconstructed from those
authoritative faces after transfer. Density, gamma integral, momentum, and
boundary volume flux therefore have explicit conservation receipts.

Pressure is not a conserved physical quantity. Volume averaging preserves a
useful warm-start mean; the next composite projection makes pressure and
divergence authoritative. Do not distort mass or velocity in an attempt to
claim pressure conservation.

### `4^3 -> 8^3`

The stable first rung uses constant injection:

```text
rho8 = rho4; gamma8 = gamma4; u8 = u4; p8 = p4
```

Each child's integrated mass, gamma, and momentum is its volume fraction of
the parent, so sums are exact. Parent normal face velocity is injected to its
four child faces, preserving flux. New internal faces are initialized from the
injected cell velocity. Projection follows immediately. This cannot recreate
detail that was already coarsened, which is why predictive promotion,
hysteresis, and the fine-detail demotion veto are correctness requirements.

History transfers by overlap: children inherit the parent's maximum score and
start with zero quiet epochs; a parent takes the maximum child score, bitwise
union of reasons, minimum quiet count, and zero hot epochs. The source state is
not released until all transfer and projection receipts pass.

## Proposed API and pipeline placement

Keep policy, planning, and state transfer separate:

```ts
export function measureSparseSurfaceActivity(
  state: SparseAtlasDynamicsState,
  history: ReadonlyMap<number, SparseBrickActivityHistory>,
  dt_s: number,
): SparseSurfaceActivityMeasurement;

export function planSparseAtlasResolutionEpoch(
  atlas: SparseAdaptiveMassAtlas,
  measurement: SparseSurfaceActivityMeasurement,
  history: ReadonlyMap<number, SparseBrickActivityHistory>,
  options: SparseSurfaceResolutionPolicyOptions,
): SparseAtlasResolutionPlan;

export function transferSparseAtlasResolution(
  accepted: SparseAtlasDynamicsState,
  plan: SparseAtlasResolutionPlan,
): SparseAtlasResolutionCandidate;

export function validateSparseAtlasResolutionCandidate(
  accepted: SparseAtlasDynamicsState,
  candidate: SparseAtlasResolutionCandidate,
): SparseAtlasResolutionTransferReceipt;
```

`SparseAtlasResolutionPlan` contains sorted `promoteKeys`, `demoteKeys`,
`sentinelKeys`, deferred score buckets, balance closure keys, predicted leaf
delta, and a policy version/threshold receipt. The transfer receipt contains
before/after integrals for density, gamma, XYZ momentum, exterior face flux,
pressure warm-start mean, plus first non-finite/failing keys. These conservation
values describe the candidate before projection. The zero-time projection
publishes its pressure impulse and energy identity separately.

Pipeline order is:

```text
accepted projected state
  -> compact activity measurement (every step)
  -> history update
  -> topology planning (every fourth accepted step)
  -> candidate transfer + composite-row rebuild
  -> zero-time global projection on candidate
  -> validate transfer + projection receipts
  -> atomic generation publication
  -> next normal CM12 step
```

Activity measurement should eventually be one GPU dispatch over resident
bricks with one compact summary record per brick. Planning is mark/scan/scatter
over those summaries. There is no dense clear, dense census, CPU leaf loop, or
GPU readback in the frame dependency chain.

## Work budget and performance contract

A `4^3 -> 8^3` promotion adds 448 compact cells. Default promotion budget per
topology epoch:

```text
maximumPromotionLeafDelta = clamp(
  max(448, floor(0.10 * acceptedLeafCount)),
  448,
  8 * 448)
maximumDemotions = max(1, 2 * floor(maximumPromotionLeafDelta / 448))
```

These cap topology churn, not correctness. Emergency/predicted requests may
overrun the cap and must publish that overrun. A steady topology performs only
the compact classifier and history update; row topology and transfer buffers
are reused.

The non-negotiable performance gate is measured after warm-up on mini dam break
32 and symmetric expansion:

- forced-all-fine Sparse CM12 at the same `32^3` resolution: median physics
  frame no more than 1.20x uniform CM12 and p95 no more than 1.30x;
- classifier + history: at most 5% of physics frame and at most 0.25 ms median
  on the reference GPU;
- no-change topology epoch: no allocation/readback and at most 2% of frame;
- amortized candidate build/transfer: at most 10% of frame over its four-step
  cadence;
- empty atlas: zero classifier, planner, transfer, and physics workgroups;
- quiet mini dam break must settle to zero promotions/demotions per epoch.

The current CPU dynamics authority cannot satisfy this GPU frame contract. The
policy must therefore be implemented directly against the compact GPU frame
pipeline rather than optimized around CPU leaf materialization.

## Physical topology cutover ABI

The GPU scheduler publishes brick resolution and double-buffered transferred
fields as one physical generation. Transport, pressure, projection,
diagnostics, and presentation all resolve the accepted worklists; the host
never substitutes construction-time cell or row counts after initialization.

Small construction lanes may pre-pack reusable operator templates. A large
accepted generation must instead stream generation zero directly into typed
arrays and leave candidate pages to the bounded GPU topology arena; Figure 6
proved that retaining the accepted graph plus four uniform and eighteen seam
object graphs exceeds the ordinary host heap.

The small-lane library contains:

- four intra-brick templates per receiver (`1^3`, `2^3`, `4^3`, `8^3`);
- one face template for every ordered neighbour-level pair (4 x 4 = 16), for
  each resident face adjacency;
- level-specific embedded-boundary geometry, exterior rows, term incidence,
  and transfer maps;
- two device-owned cell worklists and two row worklists, plus matching indirect
  dispatch arguments and active counts.

Stable template cell/row handles live in superset storage. A GPU commit writes
the inactive worklists by selecting each brick's intra template and each
adjacency's level-pair face template, compacts them with a scan, writes indirect
arguments, validates 2:1 and conservation receipts, then flips one generation
word. Every transport, pressure, projection, diagnostic and presentation
kernel must resolve its invocation through the selected worklist; a host
uniform `cellCount` or `rowCount` is no longer authoritative.

The face templates do not need 16 full copies of geometric row payload. A face
can retain a finest `8 x 8` micro-face lattice and pre-pack the 16 compact
owner/term mappings into it; the selected mapping supplies the active row
worklist. This avoids multiplying every adjacency's geometry by 16 while still
making the dispatch independent of CPU topology work.

Partial staggered commits must remain valid against the currently accepted
generation. Urgent promotion commits include their complete 2:1 closure as one
transaction. An ordinary one-rung demotion is committed only if its neighbours
still satisfy 2:1 at the commit gate; otherwise it remains queued for the next
round-robin visit. A topology generation becomes physically accepted only
after field and flux transfer receipts validate and an atomic slot flip
publishes its selected cell and row worklists. The next frame snapshots their
indirect arguments entirely on the GPU before transport and pressure consume
them.

## Acceptance gates for the first rung

1. Initialization: mini dam break retains the requested single seam sentinel;
   with sentinel off, flat saturated bulk coarsens without using component size.
2. Empty: an empty scene has zero resident activity work and zero topology work.
3. Conservation per accepted transition: relative density, gamma, and XYZ
   momentum error each `<= 1e-12` in the CPU oracle and `<= 5e-7` after GPU
   float32 transfer; exterior normal-flux error `<= 5e-7`.
4. Projection after transition: existing pressure residual and post-projection
   divergence gates pass without relaxed thresholds.
5. Stability: density/gamma remain finite and within the same overshoot bounds
   as the no-topology-change Sparse CM12 run; no promote/demote ping-pong in 250
   steps.
6. Prediction: a translating slab requests its receiving surface brick before
   the interface crosses the brick boundary.
7. Detail veto: a fine one-cell sheet or droplet with restriction error above
   `0.08` cannot demote, even at zero velocity.
8. D4: sentinel-off symmetric expansion has byte-exact D4 resolution, activity,
   reason, and history maps at every topology epoch and continues to pass all
   mass, velocity, pressure, divergence, connectivity, and time gates.
9. Seam A/B: `single` produces at least one actual `8^3 <-> 4^3` wet adjacency;
   disabling it changes topology only, not initial integrated state.
10. Performance: all frame gates above pass from captured GPU stage timings,
    not wall-clock estimates around asynchronous submission.

Camera distance, curvature-quality tuning, limited-linear prolongation, and
authored quality regions are later rungs. They
must compose as additional required-resolution evidence; they do not replace
surface activity or its conservation transaction.
