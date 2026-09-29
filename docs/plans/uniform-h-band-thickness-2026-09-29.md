# Uniform Geometric: h band thickness (2026-09-29)

The band is thick because each dynamic layout must stay valid for 3 frames,
and the census over-estimates where the surface can reach in that time.
Reach, hysteresis, the speed and shape exceptions, and solid promotion add
nothing on minidam 64/128. Read from code; not yet measured.

## What decides the band

The census `decide` pass (`lib/methods/uniform/uniform-mixed-dynamic.ts:572-598`,
driven from `mixedCensusTail` in `webgpu-uniform-reference.ts:1515`) makes a
tile h when:

1. **The surface crosses it now.** Production passes `surfaceTolerance:0` and
   `fastTravel:0`, so no crossed tile stays 4h; `forwardFine` and the
   boundary rules never run.
2. **The surface could reach it within the horizon.** The tile is traced back
   3 frames (RK2 on signed velocity bounds); it is h if that departure box
   holds a current surface tile. This is what thickens the band.

The container shell lives in the solid mask's halo, so minidam has no
solid-coupled tiles.

## Why the look-ahead exists

Invariant: every tile holding the surface is h for the whole frame, from the
advection that moves the surface into it. A surface in a 4h tile breaks:

- **Shape.** Semi-Lagrangian advection writes at the destination owner's
  resolution; a 4h tile keeps phi only at its 4h corners. Refining later
  interpolates those corners and cannot recover the detail, so
  coarsen → refine does not return the same surface.
- **Pressure.** The band pressure re-solves only h tiles. A 4h owner with an
  air centre has no pressure row, so its rim liquid stops moving (comment at
  `webgpu-uniform-reference.ts:1522`; the far-wall run-up stall).
- **Momentum.** A sheet thinner than half a 4h owner has no velocity of its
  own at 4h.

Nothing can refine a tile between censuses. The layout lasts
`UNIFORM_MIXED_CENSUS_HORIZON = 3` frames (`webgpu-uniform-reference.ts:417`):
one frame on the old layout while the census is read back without blocking,
then two on the new one. So the census makes h now every tile the surface
could enter in those 3 frames.

## What inflates it on minidam

- **Long steps.** `dt` is fixed at 1/30 with one substep:

  | Lattice | h | cells/step at 2 m/s | 3-frame reach |
  | --- | --- | --- | --- |
  | 64³ | 12.5 mm | ≈ 5.3 | ≈ 16 cells ≈ 4 tiles |
  | 128³ | 6.25 mm | ≈ 10.7 | ≈ 32 cells ≈ 8 tiles |

  The reach runs ahead of a moving surface, and to both sides wherever the
  velocity bounds hold both signs.
- **Gravity on still liquid** (`:587`). The two older traced frames add 2g·dt
  and then g·dt of downward velocity everywhere, including where pressure
  cancels gravity. A still surface's band extends ≈ 2.6 cells (64³) or
  ≈ 5.2 cells (128³) down into the liquid.
- **Tile-wide bounds** (`:486-491`, `:469`). With `SAMPLE_REACH` = 2 cells, a
  tile's bound is the extreme velocity over its 3×3×3 tile neighbourhood, so an
  air tile beside a fast front takes the front's top speed. Boxes needing more
  than 64 cubes use larger power-of-two cubes that may overhang.
- **Fixed margins.** 0.5 cell of drift (`UNIFORM_MIXED_DYNAMIC_SURFACE_DRIFT`)
  and rounding up to whole 4-cell tiles.

## Levers toward surface-only

1. **Horizon 3 → 1.** Adopt the built layout on the GPU in the frame that
   censuses it. The speed-driven reach drops 3× and the gravity term goes
   away. One frame of travel is the floor for this design: the tile must be h
   before the advection that writes into it.
2. **Gravity only where liquid can fall.** Leave the shift out under liquid so
   a still surface stops extending downward. Independent of the horizon.
3. **Tighter bounds.** Push the actual surface points forward with their own
   velocity and mark the tiles they land in, instead of using
   neighbourhood-wide extremes.
4. **Below one frame.** Advect, recensus, relayout, then pressure within the
   frame; needs phi advected into a full-resolution scratch so no surface
   detail is lost first. A pipeline change, not a parameter.

Even surface-only costs one or two 4-cell tiles where the surface sits near a
tile face.

Open: one Dawn census run on minidam64 logging surface-crossed vs h tiles,
horizon 1 vs 3 and gravity shift on vs off, to split the thickness by cause.

## Design review: GPU-driven thin bands

29 September, inspected working tree at HEAD `6cfb9b77`, with existing local
solver edits. This section is a design proposal, not an implementation or a
benchmark result. The observations above describe the current coupling; the
requirement to keep one ownership layout throughout a frame is not a
fundamental requirement of a geometric liquid solver.

**Recommendation:** first remove host-dependent layout adoption and use a
one-step GPU classification. Then separate the broad region searched for an
arriving surface from the thin region receiving expensive h work. Use masks
for membership, compact lists for execution, and dirty sets for rebuilding
derived topology. These are separate responsibilities.

### What exists in this checkout

- `mixedCensusTail` already classifies and builds topology on the GPU. But
  `updateMixedDynamic` awaits the classifier and builder reads before calling
  `adoptBuiltLayout`. The builder reads tile words, constructs a host layout,
  and returns host counts. The horizon is still 3. The adjacent data-layout
  audit's statement that this checkout already has readback-free adoption
  and a one-frame cadence does not match these sources.
- `UniformMixedRemap.applyBuilt` already encodes target adoption, remap,
  live adoption and publication in one encoder. Its changed-tile worklist
  already limits field remapping to width changes and affected neighbours.
  Reuse this machinery; do not introduce a second remapper.
- `UniformMixedOwnership` still obtains dispatch sizes, fused/seam choices
  and hanging-cache allocation from host counts. The builder's output alone
  does not make its consumers independent of readback.
- `UniformMixedFrame.advance` still needs frame N−2's checked receipt for
  its pressure plan and lagged band size. Removing census readbacks alone
  leaves this control dependency.
- Comments in the frame call phi independent, but the active surface kernels
  still use mixed vertex authority; remap writes canonical phi and
  `UniformMixedPhiResolve` reconstructs hanging texels from coarse corners.
  A full-sized texture is not an independently authoritative h field.
- The h pressure solve already builds and consumes a GPU tile list. It
  includes liquid rows in h simulation tiles, not just sign-crossing tiles.
  Shrinking a surface mask alone will not shrink its work while ownership
  remains thick.

### Why horizon one is only the first step

For a surface of fixed physical area A, a genuinely thin band with O(1)
fine-cell thickness has O(A/h²) cells. A conservative travel slab of physical
thickness U·dt instead has O(A·U·dt/h³) cells, before domain saturation.
Reducing three frames to one improves the constant but does not remove this
high-resolution scaling at fixed dt. This describes the reach contribution,
not the cost of the whole solver: the global 4h solve still grows cubically.

There is another source of swept thickness beyond the three-step horizon:
`sampledFlow` always joins its bounds with zero. In a region translating at
one positive speed, the interval becomes [0,U], so the inverse box includes
the entire path from the destination back to its departure. This is safe for
clamping/contact but unnecessarily broad away from those cases. A certified
interior path could keep strictly signed bounds, with the current
zero-inclusive path retained wherever walls, solids or sampling require it.
That requires proving the whole RK2 footprint is interior, not testing only
the destination tile. It is a useful later classifier experiment, not a
reason to remove zero unconditionally.

For pure translation, the old surface and destination surface are two thin
sets even when they are many tiles apart. Their intervening swept volume
need not all acquire h ownership. Velocity sampling along characteristics
still needs valid data there, which may be the existing coarse field.

### First implementation: one-step relayout with no layout readback

Use the actual next step's dt and state, at frame entry after edits/body
updates that affect the trace. Encode the following dependencies on the GPU:

```text
current fields / ownership
  → support and velocity extension
  → one-step required-h mask
  → next topology, counts, lists and change mask
  → conservative remap from old to next ownership
  → publish next ownership and repair derived fields
  → ordinary advance
```

The existing frame-tail extension may supply the first stages when its state
and parameters still match. The layout must be adopted before the advection
it protects. No frame is admitted using an unadopted census merely because a
map callback has not arrived. The classifier and the subsequent trace must
agree on the sampled field: prove that remapping the extension preserves the
bound, or retain an immutable sampling view and its old topology. Recomputing
an arbitrary extension after classification invalidates that argument.

The concrete host dependencies to remove are:

1. Split GPU generation handles from optional host mirrors. A generation
   supplies buffers and a fixed-capacity ABI, not a `UniformMixedLayout`
   reconstructed by `read()`. Keep dimensions and capacities on the host;
   keep current membership, counts, changed count and generation on the GPU.
2. Make **all** topology-dependent launches GPU-counted: tiers, seams,
   regular/general work, fused work, phi resolve and hanging-cache producers.
   A lagged count is not a safe dispatch bound. Use indirect dispatch or a
   fixed grid that strides a GPU list; measure each choice.
3. Preallocate bounded hanging-cache capacity or use a bounded page pool.
   `reserveHanging` cannot allocate a larger GPUBuffer in response to a GPU
   count without host feedback. Overflow must be explicit and fail closed;
   clipping the list is not allowed.
4. Move `changedTiles` and `pressureMatchesSimulation` decisions to GPU
   control. Keep the all-4h pressure hierarchy fixed. Encode a layout-general
   path, or gate complete alternatives on the GPU, including the no-h case.
5. Replace live CPU mask joins with GPU unions for drops, sources, authored
   fine regions, solid edits and moving-body sweeps. A stale diagnostics
   mirror must never decide which new liquid receives h support. Explicit
   coarse-only regions need an explicit policy conflict; preserve current
   authored semantics rather than silently overriding them.

Initially retain the existing full topology builder. Making it incremental
at the same time would obscure whether failures came from adoption or stale
metadata. This step alone removes the multi-frame ownership lifetime. It
does not promise an asymptotically thin band.

### Target design: classify destinations before expensive h work

Keep these sets distinct; a reason bitmask can explain overlaps:

| Set | Meaning | Work it permits |
| --- | --- | --- |
| Current surface | h detail and thin liquid present in the source state | Preserve source phi, V and momentum |
| Candidate destinations | Conservative one-step arrival envelope | Classification and, where needed, h phi evaluation |
| Destination surface | Actual destination crossings/near-surface detail | Fine destination ownership |
| Stage support | Read/write footprints required by a particular operation | That operation's sampling, extension or halo work |
| Forced fine | Authored regions and liquid/solid coupling | Mandatory ownership and associated solves |
| Topology dirty | Ownership or adjacency changed | Remap and topology repair |

Do not make every candidate a fine transport owner. A broad candidate list
can still be useful if it only runs a relatively cheap classifier. It loses
its purpose if every candidate runs redistance, transport normalization,
eight sharpening rounds and band pressure.

A practical destination-classification path is:

1. Freeze the source phi, volume, velocity sampling view and ownership for
   the step. Build conservative candidate destinations using the one-step
   bounds. Include source/drop/body events that can create liquid locally.
2. Reject certified far tiles cheaply. For undecided candidates, evaluate
   the **actual backward-advection operator** at h vertices into scratch,
   preserving these values for reuse. Build crossing and near-surface masks
   from that result. Evaluate the support closure needed by later phi
   operations as well; a crossing-only list is too small.
3. Construct fine destination ownership from this mask, required stage
   support and forced regions. Retain source fine data until its final donor
   and momentum reader finishes. A transitional implementation can use the
   union of source and destination fine sets; it need not fill their hull.
4. Execute transport and momentum with an explicit source/destination
   contract, then correction, pressure and publication. Release unneeded h
   data only after their last readers, preserving coarse totals and fluxes.

This is a larger change than moving `adoptBuiltLayout`. Current transport
assumes one mixed ownership; its donor graph, normalization, target capacities
and self-fallback must either support separate source/destination ownership
or operate on a conservatively remapped union. If using a union, retain the
frozen sampling view or establish equivalence with the remapped trace. Do not
classify with one velocity field and silently advect with another.

Two additional contracts are essential:

- **Phi authority:** provisional h samples must survive ownership changes,
  vertex resolve, redistance and volume correction. A narrow h phi overlay
  needs explicit valid support and a defined far-field representation;
  coarse-corner interpolation cannot overwrite valid fine samples. Existing
  full-resolution textures can provide a correctness baseline before sparse
  pages. They need not imply full-domain fine execution.
- **Non-advective motion:** redistance, wall continuation, solid displacement
  and volume correction can change the final surface. Their read footprints
  and possible surface displacement must be covered before any dependent
  fine update. The current half-cell drift is an empirical margin, not a
  general proof. Either establish bounds for these operators or encode a
  bounded GPU repair/replay from preserved pre-stage data. Finding a miss
  after a coarse update cannot recover lost V/momentum detail. Overflow or
  exhausted repair must stop the frame, not publish a coarse surface.

Forward-advecting a few surface points is useful for proposing candidates,
but cannot certify their completeness. The solver gathers backward from
destination vertices; forward RK2 and backward RK2 are not exact inverses,
and point samples can leave holes under stretching, rotation or contact.
Likewise, eight destination corners can miss h corrugations. Same-sign
source taps certify a trilinear interpolation, but not an arbitrary cubic
one with negative weights; exclusion needs bounds for the actual operator.
Uncertain cases stay on the conservative path.

### Masks, lists and dirtying

Use stable spatial tile IDs. A membership bitset gives compact union,
difference and neighbour queries. Compact lists give dense GPU work. For a
new fine mask F and old mask O:

```text
promote = F & ~O
retire  = O & ~F
changed = F ^ O
```

Rebuild membership each step initially; do not accumulate OR bits forever.
Producers can atomically OR membership, then a subsequent dispatch compacts
it. Workgroup-local aggregation avoids one global append per cell. A fresh
list/epoch prevents duplicates and stale jobs. Device counts and list
capacities are checked before consumers run. Start with bitset clears;
epoch tagging is a later optimization requiring wrap and initialization
rules, not a prerequisite for residency.

Dirty sets follow the dependency being repaired:

- Width changes affect local face/vertex ownership and nearby seam records;
  the remapper already implements a neighbour closure, with narrower cases
  for unchanged h tiles. Reuse its proven rules.
- The builder's 3×3×3 stencil descriptors need changed tiles plus their
  neighbours. Solid geometry has its own edit/sweep dirty set.
- The exact distance-to-coarse certificate is **not local**: retiring one
  fine tile can lower distances far away. Initially rebuild it globally, or
  invalidate the fast-path certificate conservatively and use general
  sampling until repaired. Stale large distances can incorrectly admit a
  characteristic to the all-fine sampler.
- Velocity, phi and pressure values changing every step are field updates,
  not topology changes. Pressure has global influence; a local edit does not
  justify skipping the rest of the pressure solve.

The present packed owner ABI is the main obstacle to truly local topology
repair. A changed fine count shifts coarse owner bases (`fineCount*64`),
and insertion into sorted fine lists shifts later fine ranks. Cached
owner-indexed records can therefore become stale far from the changed tile.

For incremental topology, separate **execution order** from **storage
identity**: tile ID → stable fine page or fixed coarse slot, with arithmetic
indexing within each tile. Compact lists may reorder every frame without
moving field identity. Page allocation/release, seam slots and cached
references carry generations. Keep old pages alive through their last GPU
reader; initialize newly assigned pages before publishing their mapping.
No page-table lookup is needed for every regular in-tile neighbour.

Full GPU list compaction can remain even after field remaps become local.
At 256³ there are 262,144 four-cell tiles; one membership bitset is 32 KiB.
A metadata scan may be a good trade against a complex allocator. The current
builder's distance passes do whole-axis searches per tile, so “rebuild all
metadata” is not uniformly cheap: measure distance construction separately.
Stable paging is justified when rebuild/remap or capacity costs show up,
not required merely to eliminate the first readback.

Retain spare allocated pages without retaining active h work. An inactive
cached page is stale until refreshed. Allocation hysteresis can reduce churn
without spatially thickening the simulation band; ownership hysteresis
keeps paying for fine transport and pressure.

### Removing the remaining receipt dependency

The steady advance must remain correct when diagnostics are disabled and
no receipt resolves. Replace the frame N−2 pressure-plan dependency with GPU
state. Preserve the existing convergence rules and cycle caps. The host
encodes a bounded schedule; the GPU selects active V/full slots, projection
and the next plan. Band coarse-solve strategy must also cease depending on a
host-read band count. A layout-independent valid path is a baseline; GPU
selection between small/large alternatives is an optimization to measure.

This has real launch costs. The existing pressure-band code deliberately
uses fixed grids over compact lists, and prior measurements in the GPU
residency plan found expensive empty-dispatch schedules. GPU-driven does
not mean every dispatch must be indirect. Use direct bounded grids for
small list stages and compare indirect sizing for expensive variable work.
Do not trade a readback bubble for hundreds of costly empty launches without
measuring the complete frame.

Portable WebGPU provides GPU-buffer-supplied dispatch dimensions, not an
arbitrary GPU-generated command stream; the host still encodes the stage
graph. WGSL barriers synchronize a workgroup, so cross-workgroup list build,
finalization and consumption belong in ordered dispatches. A persistent
shader spinning on a global barrier is not a portable replacement.
See the [WebGPU indirect-dispatch specification](https://www.w3.org/TR/webgpu/#dom-gpucomputepassencoder-dispatchworkgroupsindirect)
and [WGSL synchronization rules](https://www.w3.org/TR/WGSL/#sync-builtin-functions).

Before removing receipt admission gates, add a sticky GPU failure record
with frame/generation and cause. Capacity failure, invalid support, pressure
rejection and solid-certificate failure must inhibit dependent work,
publication and subsequent queued advances. Preserve the last accepted
presentation independently if intermediate simulation fields are in-place.
The current projection gate alone does not establish that transaction.
An optional asynchronous status read reports failure to the UI; it cannot
be required for the GPU to stop. A full diagnostics ring drops telemetry
instead of blocking simulation. Ordinary queue backpressure may still bound
frames in flight, but must not require a numerical receipt to plan the next
step. Literal zero readbacks means the host cannot report GPU-computed
failure details until an explicit status query.

### Implementation and evidence order

1. **Measure the present band by reason.** Separate current crossings,
   one-step destinations, extra multi-frame reach, support and forced-solid
   work. Report unions and exclusive contributions so overlaps do not get
   counted twice. Compare horizon 1/3 on the same captured state first;
   this classifier comparison does not establish whole-run correctness.
2. **Implement GPU generation adoption at horizon one**, initially with the
   full metadata builder and conservative existing classifier. Audit every
   host topology consumer. Separately complete GPU pressure planning and
   sticky failure handling before claiming a readback-free advance.
3. **Prototype destination classification and independent phi authority.**
   Begin with a dense h reference path for correctness, then compare a
   conservative candidate list with certified rejection. Measure classifier
   and scratch costs as well as the expensive work they eliminate. Include
   correction/contact support and the source/destination transport contract.
4. **Make topology repair incremental only if needed.** Stable tile/page
   identity comes before local updates to packed-owner caches. Keep global
   cheap scans where they win; separate retained allocation from active work.

Required focused evidence: static pond (no gravity-only widening), fast
translation across several tiles, sub-4h sheets/corrugations, stretching and
rotation, wall run-up/contact, moving solids, drops/edits, coarse/fine churn,
empty/full layouts and forced capacity/pressure failures. Verify fine data
before the consuming update, conservative mass and face-flux transfer, and
no dependence on delayed/missing diagnostic callbacks. Test the zero-change
and zero-job paths as well as moving interfaces.

Then compare the same physical scene and timestep at 64/128/256: crossing
tiles, candidate tiles, actual h owners, seam tiles, band pressure rows,
changed tiles, allocations and peak memory, complete GPU time and sustained
frame throughput. A growing candidate region is acceptable only if total
cost falls; a thinner visual overlay is not evidence of less simulation work.
Use separate diagnostic captures so per-frame reads do not serialize the
throughput run. Run Dawn serially under the repository WebGPU lease, without
a browser simulation. After solver implementation, preserve the full gate:
`npm run check:types`, `npm run test:unit`, `npm run test:dawn`.

This review changes only this plan. No shaders, numerical settings, timing
ceilings or tests were changed, and no GPU measurements or clean-repo gate
were run for it.

## Step 1 result: band by reason (measured 29 September)

Probe: `tools/probe-uniform-band-reasons-dawn.ts` (production census plus
probe-owned classifiers with other policies on the same state; variant A
reproduces the production band exactly). Rerun:
`WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_GPU_COMPILATION_CONCURRENCY=1 node --import tsx tools/probe-uniform-band-reasons-dawn.ts --scene=minimal-power-dam-break-64 --capture=5,15,30,60,120`.

Sets are nested, so the terms are disjoint: X crossings ⊂ Z closure (dt 0)
⊂ C horizon 1 ⊂ D horizon 3 without gravity ⊂ A production.

| Share of production band | 64³ (steps 5–120) | 128³ (steps 5–30) |
| --- | --- | --- |
| Crossing tiles X | 23–32% | 14–24% |
| Closure Z−X (neighbour touches a crossing owner's face/edge/corner) | 19–28% | 10–16% |
| One-step travel C−Z | 9–19% | 20–22% |
| Multi-frame horizon D−C | 20–36% | 32–50% |
| Gravity shift A−D | 3–11% | 4–13% |

- Horizon 3 → 1 removes 31–39% of the band at 64³ and 40–54% at 128³. The
  band then sits at 2.1–2.6× X (64³) and 2.5–3.3× X (128³).
- **Correction to the analysis above:** the gravity shift applies only to
  older traced frames, so at horizon 1 it is exactly zero. Lever 2 is
  subsumed by lever 1.
- The closure floor (dt = 0) is 1.6–1.9× X and does not depend on dt. It
  protects shared vertices that would otherwise be rebuilt from 4h corners;
  removing it needs independent h vertex authority (step 3), not a
  classifier tweak. "Surface-only costs one or two tiles" understated it.
- The zero-join is up to 26% at horizon 3 (128³, step 5) but 2–13% at
  horizon 1: a minor lever after horizon 1. Drift margin is 1–3%.
- The band-to-crossing ratio grows with resolution (A/X 3.1–4.3× at 64³,
  4.2–7.2× at 128³), consistent with the travel-slab scaling argument.
- Open: one-frame travel peaks at 45 (64³) and 84 (128³) cells at step 15,
  about 16 m/s, on wet crossing tiles near a wall corner, against a
  crossing-tile P90 of about 2.5 m/s. It inflates neighbouring bounds.
- Open: the 128³ run died at frame 34, "Uniform mixed pressure did not
  converge: candidate 5.84, tolerance 5, 4 cycles" (the N−2 lagged-plan cap);
  no control run yet.

## Decisions (Peter, 29 September)

1. Rigid-body tiles are liquid-conditional: the GPU body sweep ORs into the
   coupled mask; no unconditional host join.
2. Inflow nozzles get a GPU h mark into the join bitset.
3. Pressure keeps the frame N−2 schedule for now; the full 4V+3F envelope
   is measured later with a one-line A/B.

## Horizon one landed (29 September, uncommitted)

The census, builder, remap and adoption now run at the frame head on the GPU,
with no readback (WP1, WP2a–e, WP3, P1–P3, WP5). Dynamic coarsening,
`uniform-dynamic-coarsening-dawn`, h tiles per step over 30 steps, same
settings in both arms:

| Scene | ec753b45 (horizon 3) mean | Horizon 1 mean | Change |
| --- | --- | --- | --- |
| minimal-power-dam-break-64 | 2353 | 1559 | −34% |
| high-resolution-dam-break (128³) | 12020 | 5721 | −52% |

This matches the Step 1 prediction (−31–39% at 64³, −40–54% at 128³). The
remaining floor is the closure (step 3). Frame time has not been measured.

- The 128³ lane's refine check now reads phi at the head's census phase,
  right after the remap. Read after the frame, a tile refined at the head has
  already had one h transport step (0.01 h off its 4h interpolant). 27794
  refined tiles match exactly; no relayout allocates or compiles.
- `uniform-pond-rest-dawn` fails at ec753b45 too. Its `pressureResidualTolerance: 0`
  control arm can never set the converged word, so the frame always fails.
  Pre-existing; not from this program.
- `tools/probe-uniform-band-reasons-dawn.ts` reads the host layout, which a GPU
  adoption no longer mirrors; it needs a GPU tile-word readback before reuse.
