# Sparse CM12 surface-activity resolution policy

Status: first implementable rung. This is the specified replacement for
bootstrap component-size coarsening; it does not add camera-distance adaptation.

## Current failure and decision

`WebGPUAdaptiveMassSolver.createAsync` first asks scene initialization to keep
one half of the resident bricks fine. It then calls
`coarsenLargeQuiescentComponents`. For any connected wet component larger than
eight bricks, that function converts every fine brick in the component to
`4^3`. The mini dam break is one such component, so the deliberate negative
fine region disappears. Component size is neither activity nor an accuracy
measure and must not remain the production policy.

The first production rung is a hysteretic, brick-local surface policy:

- absent bricks stay absent and receive zero classification work;
- fully interior, quiescent liquid may be `4^3`;
- a free-surface brick is `8^3` only while measured shape or transport error
  requires it;
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

Only surface-band bricks evaluate the more expensive channels. Use the coarse
cell width `h4 = 2` in finest-cell units when making the score, irrespective of
the brick's current resolution; this makes a promote decision and the
corresponding keep-fine decision comparable.

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
  Shape         = 1 << 1,
  Transport     = 1 << 2,
  Temporal      = 1 << 3,
  FineDetail    = 1 << 4,
  PredictedFace = 1 << 5,
  Unknown       = 1 << 6,
  SeamSentinel  = 1 << 7,
}
```

The initial thresholds are calibration values, not scene knobs. Publish them
with captures so threshold changes cannot silently invalidate A/B results.

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
  readonly demoteEpochs: 8;
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
fine   + score <= 96 for 8 epochs
       + restrictionError <= 0.08           -> request coarse
otherwise                                   -> retain resolution
```

Refinement always wins over demotion. A brick changes by one rung in an epoch.
Newly resident receiver bricks start `4^3`, `Unknown`, with zero quiet epochs.
A surface row whose outward characteristic can reach another brick before the
next topology epoch sets `PredictedFace` and immediately requests that receiver
fine. This uses `topologyCadenceSteps * dt * max(abs(faceVelocity))`; it is a
surface-motion guard, not camera adaptation.

With only `4^3` and `8^3` bricks, every resident adjacency is already strictly
2:1 or equal. Still run a balance/validation pass and publish its zero-or-more
closure count, because a later `2^3` rung must refine the coarser neighbor to a
fixpoint. Omitted air is not a level and is not made resident merely for
balance.

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
pressure warm-start mean, plus first non-finite/failing keys.

Pipeline order is:

```text
accepted projected state
  -> compact activity measurement (every step)
  -> history update
  -> topology planning (every fourth accepted step)
  -> candidate transfer + composite-row rebuild
  -> next normal CM12 step and projection on candidate
  -> validate physics + transfer receipts
  -> atomic generation publication
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

Camera distance, curvature-quality tuning, limited-linear prolongation, more
than two resolution rungs, and authored quality regions are later rungs. They
must compose as additional required-resolution evidence; they do not replace
surface activity or its conservation transaction.
