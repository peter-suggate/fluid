# Coarse-first Activity census + frontier investigation

Scope: current working tree, adaptive-mass, `coarse-first-pool-impact`, paper
cadence (1/30 s). The investigation below describes the pre-optimization source.
Implementation and validation follow at the end. Existing curvature work is
preserved.

## Strongest suspect: curvature restriction

`webgpu-sparse-cm12-resident.wgsl.ts:6117–6180` computes normals at a common
physical support width. This is necessary for planar coarse/fine interfaces,
but the implementation performs expensive nested spatial queries:

- Each normal makes seven owner queries to choose its width, then up to six
  restricted density queries.
- Each restricted query first resolves an owner. A contained finer donor
  then loops through `(width / donorScale)^3` samples and resolves the sparse
  owner again for every sample, despite already knowing the containing leaf.
- Cross-leaf support retains `scale = 1`: work becomes `width^3`, even when
  many finest sites share the same coarse donor. At widths 8, 16 and 32 this
  is respectively 512, 4,096 and 32,768 owner queries per density sample.
  These are conditional operation counts, not observed pool frequencies.
- `compactOwnerCellAt` calls the world directory, whose owner search probes
  dyadic span levels and hash-table entries with atomic storage loads. It
  also reconstructs the leaf geometry and accepted cell range each time.

At resolution B1, `measureBrickActivity` has only one cell, so only lane zero
does this work. It requests seven normals: up to 49 width-probe queries and
42 restricted density queries. The other 63 lanes wait at the reduction.
For equal support widths the 42 density requests visit only 25 unique stencil
positions; varying selected widths require keys that include both origin and
width. Deduplication is therefore possible but must preserve support identity.

The current uncommitted curvature fix introduced the common-volume scans.
Reverting to the old tracer lookup is incorrect: activity never staged that
transport directory, and the old result could silently erase curvature.
The physical rationale and existing validation are in
`symmetric-expansion-coarse-first-physics-2026-09-06.md`.

## Additional slow patterns

1. **Lane-history predicate used for cell-local work.** `densityInterfaceCell`
   is declared before the cell loop (around line 6249), ORs in each crossing,
   and gates normal calculation at line 6420. Once true, every subsequent cell
   handled by that lane computes normals, even without its own crossing.
   B8 has eight cells per lane. Separate the cell-local crossing predicate
   from the accumulated census flag. This can change the normal extrema, so
   it needs numerical validation, not just a timing comparison.
2. **Clean bricks still reduce.** The host dispatches `brickCount` workgroups,
   not a compact dirty list (`webgpu-sparse-cm12-resident.ts:6809`). Clean
   invocations become INVALID, but only return after scratch writes and all
   six reduction rounds. A uniform clean-brick exit removes that work.
   Inactive *dirty* bricks must still publish their removal from the census.
3. **Large fused kernel.** Incidence/term traversal, density moments, velocity
   metrics, sibling residuals, curvature quadrature and history all share one
   entry point. Divergent quadrature delays the entire workgroup. Register
   pressure is a plausible additional cost; it requires compiler/counter
   evidence before being called a measured bottleneck.
4. **Capacity scans in frontier publication.** Directory finalization scans
   directory capacity and synthesis dispatches topology-page capacity. These
   are secondary candidates. Frontier allocation already uses accepted-leaf
   indirect work, support masks and a resolved-neighbor cache; it is not an
   unconditional 26-neighbor allocation on every frame.

## Existing approaches to borrow

| Existing shader | Useful pattern | Application |
| --- | --- | --- |
| `restrictedPresentationDensityAt`, resident WGSL:2149 | Resolve a containing leaf once; directly index its cell interval | Remove repeated directory queries inside contained curvature restriction |
| `cm12PresentationPreparePage`, resident WGSL:9406 | Uniform feature gate, cooperative density-cache fill, barrier, reuse | Stage common-volume samples across all 64 lanes; reuse them for normals |
| `cm12TeiStageDirectory`, transport-execution-image WGSL:93 | Cooperatively cache 27 leaf descriptors; exact fallback outside the cache | Activity-specific accepted-owner cache; size/cover it for macro support |
| `cm12StageTransportHomeFrameHalo`, transport-home-frame-halo WGSL:105 | Lane-strided halo fill, dyadic shifts/masks, exact cache-miss fallback | Avoid serial halo sampling and repeated variable integer division |
| Presentation and pressure execution-image kernels | `workgroupUniformLoad` before a barrier-containing branch | Uniform early return for clean activity workgroups |
| Existing activity sibling residual, resident WGSL:6474 | One elected owner loads shared data once | Deduplicate repeated density stencil requests |

Borrow addressing and scheduling, not presentation's field semantics:
curvature uses destination density, rejects cut-solid samples and treats
missing liquid as air. Presentation uses normalization/clamping and sometimes
previous activity receipts. Those predicates cannot be copied into the current
census without changing its meaning or reading stale classification.

## Recommended implementation order

1. Add exact-scene timing coverage at `timeStep: "paper"`; capture the census
   substage separately from frontier allocation and synthesis before impact,
   around impact (~0.6 s), and during settling. The pool interpolation probe
   uses scene cadence (1/60 s), so it is not a matching timing reproduction.
2. Implement direct cell addressing for fully contained restriction and a
   uniform clean-brick exit. Both can preserve the existing calculation.
3. Parallelize B1 halo queries and cache repeated common-volume samples.
4. Replace cross-leaf finest quadrature with exact overlap-volume restriction
   over accepted donor cells/leaf regions. Never choose one donor's stride
   for an entire mixed-resolution region: that can skip finer neighbors.
5. Evaluate the cell-local predicate separately; consider a separate compact
   curvature pass if kernel occupancy or the remaining serial tail warrants it.

The timestep does not directly set the restriction loop bounds. It changes
motion, dirty coverage and subsequent refinement; the resulting topology can
make this cost worse. Incoming prediction's radius-cubed scan is in resolution
planning, outside the reported Activity census + frontier interval.

Use hardware timestamps plus diagnostic counts for normal calls, restricted
samples, contained/cross-leaf cases and widths. Collect counters in a separate
diagnostic run so their atomics do not contaminate the performance comparison.
Any substantial implementation must pass the unchanged coarse-first planar,
solid, pool, impact, settling and expansion controls and the canonical
`npm run test:dawn:sparse-cm12` gate. Dawn must run without a competing browser
or Dawn process.

## Implemented changes

The common physical-volume curvature estimator is retained. Its implementation
now partitions cross-leaf support at logical-brick boundaries, resolves each
owner once, and integrates exact donor-cell overlap volumes. Contained support
uses direct native cell addressing. This replaces repeated finest-site sparse
lookups without replacing curvature with a presentation or tracer proxy.
B1 distributes its seven normals across seven lanes; clean activity workgroups
exit uniformly before reduction. The lane-history interface predicate remains
unchanged to avoid changing the estimator's support.

Activity timing now separates measurement/curvature, symmetry, and census
history. The existing frontier-finalization dispatch is also represented in the
stage contract; the dispatch itself was already present.

The live CPU planning path had another substantial cost: it mapped the full
activity capacity and accepted row IDs and constructed transfer geometry before
discovering that no generation replacement was needed. A conservative GPU
preflight now returns an eight-byte request receipt. Only positive requests
capture detailed planning state; only admitted replacements capture transfer
rows. Authored refinement regions and live policy edits retain detailed
planning. Source-topology lease validation remains in place. A no-op plan no
longer clears pressure-iteration feedback.

The preflight checks unbacked rerung/activation requests, macro motion/thin
fluid demands, and necessary quiet-merge conditions. It allows false positives;
fully backed in-place rerungs and dynamic B8 frontier pages need no host
replacement. Its persistent buffers are included in resident memory accounting.

## Physics and regression validation

- Independent finest-volume GPU oracle: non-planar dyadic donor fields across
  support widths 1, 2, 4, 8, 16 and all three axes; restriction error exactly
  zero. Planar-normal maximum error: 1.1920928955078125e-7. Cut-solid rejection
  remains covered.
- All 11 coarse-first Dawn tests pass, including pool, impact, settling,
  curvature and symmetric expansion against a max1 refinement region.
- Symmetric expansion at 1/30 s, checkpoints 3/5/8/13: unchanged A/B limits
  pass. At step 13 (0.4333 s), density L1 / reference mass is 1.0341%, height
  L1 / reference mass is 0.6999%, and height RMS is 0.03753 fine cells
  (1.8765 mm at the scene's 0.05 m lattice). This is a bounded comparison,
  not a claim of identical adaptive and max1 trajectories.
- New GPU preflight tests cover no-op, unbacked refinement and activation,
  backed rerungs, macro motion/thin fluid, merge count, span and surface vetoes.
  CPU regression tests verify no-op planning avoids transfer capture and
  preserves pressure feedback. Stage timing contract passes.
- Canonical `npm run test:dawn:sparse-cm12`: all 14 correctness lanes pass;
  mini32 passes at 30.54 ms (40 ms limit). Mini64 fails at 62.00 ms (50 ms
  limit). No limits were changed. An isolated matching mini64 stage probe
  measures 62.85 ms in the pre-change snapshot and 60.23 ms after these changes;
  this remaining failure predates the optimization. Pressure solve (~20.9 ms)
  and face preparation (~10.3 ms) dominate that case.
- Focused lint passes. Repository-wide TypeScript checking still reports 50
  existing errors outside the changed files; changed files have no reported
  errors.

## Reproduce the pool performance comparison

`tools/probe-coarse-first-activity-dawn.ts` selects the exact pool scene,
coarse-first, paper cadence, and hardware timestamps. It checks each encoded
step and trace context. Thirty steps cover one simulated second including
impact. The first three frames are excluded from reported medians. Run each
source tree sequentially with the Fluid browser closed:

```bash
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
ACTIVITY_READBACKS=1 ACTIVITY_OUT=/tmp/coarse-first-activity.json \
node --import tsx tools/probe-coarse-first-activity-dawn.ts
```

The probe waits 120 ms between samples to obtain per-step timestamps. Its wall
measurement includes advance, outstanding topology preparation and GPU queue
completion, but excludes that gap, solver startup, diagnostic field extraction
and rendering. Readback map durations include preceding GPU queue work and must
not be summed as transfer cost. These are simulation measurements, not browser
FPS or a claim that an existing deployed bundle has been updated.

## Final isolated pool measurements

Medians over steps 4–30, same scene, timestep and initial state:

| Measurement | Before | After |
| --- | ---: | ---: |
| Activity census + frontier | 98.70 ms | 3.21 ms |
| Total simulation GPU time | 141.62 ms | 44.43 ms |
| Serialized step wall time | 221.05 ms | 52.97 ms |

The final density SHA-256 is identical in both arms:
`fdc41931f38600945417b021c96e52944432038569068671d4d0e58ecf1ca412`.
Activity is ~30.7× faster and serialized step wall time is ~4.2× faster. An
intermediate tiled-shader-only capture measured 128.70 ms wall time; shader
savings alone left substantial CPU planning work.

At representative step 14, the old planner mapped 654,880 activity bytes,
384 header bytes, 163,760 row-ID bytes and 12 lease-check bytes: 819,036 bytes
before knowing that replacement was unnecessary. The new planner maps eight
bytes. Both arms also map a four-byte pressure receipt and 2,048 instrumentation
bytes. Detailed reads remain available when replacement is actually requested.

Raw receipts include per-frame phases, readbacks and source SHA-256 hashes:

- [Before](../artifacts/coarse-first-activity-before-2026-09-06.json)
- [After](../artifacts/coarse-first-activity-after-2026-09-06.json)
- [Mini64 remaining-cost comparison](../artifacts/coarse-first-mini64-comparison-2026-09-06.json)

The before source is an isolated copy of the working tree from the start of
this optimization, including the preexisting curvature correctness fix. It is
not a comparison against an unrelated old commit. Neither arm opens a browser.
