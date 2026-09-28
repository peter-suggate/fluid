# Uniform Geometric: dynamic coarsening

## Goal

Replace authored enforcement regions as the source of coarseness. Ownership
follows the simulation state: tiles near the free surface stay fine (h), and
submerged and far-air tiles become coarse (4h). (History: this originally kept
the 2h grading collar between them; the simulation layout is now ungraded h/4h,
and the 2h collar was retired in 8a80095f.) This uses the same single mixed frame, the same
ownership ABI and the same remap. There is no second solver and no fallback
path. Performance acceptance: on the 128³ high-resolution dam break, complete
frame time beats all-fine at equal simulated duration, with physically
plausible results.

## Definition

The unit is the 4h tile, which is the ownership word.

**Interface tile.** A tile is an interface tile if any of its owners is
neither clean interior nor clean air:

- *interior*: all eight owner-corner phi values are < 0 and `V ≥ 1 − fullTolerance`
- *air*: all eight corner phi values are > 0 and `V ≤ emptyTolerance`
- anything else counts as interface. That includes partial cells, phi sign
  changes, stray V in air (dust, orphan volume, film) and phi-only sheets with
  no volume.

This is deliberately conservative. In Uniform Geometric, V and phi can
disagree (see the phantom-sheet and non-compact-V notes), and any tile where
they disagree must keep fine resolution.

**Band.** `d(t)` is the Chebyshev tile distance from tile t to the nearest
interface tile. It is computed by three separable 1D passes and capped at
`DISTANCE_CAP`.

`R(t) = ceil(travel(t)·1.00001 / 4) + reach`, where `travel(t) = s(t)·dt / h_min`.
`s(t)` is the tile's box-maximum extended speed from the local speed
certificate (`ownership.speeds[UM_TILES + t]`). This is the same bound that
certifies fine characteristics. `reach` covers sampling, extension and
sharpening stencils. The default is 2 tiles, and it is a runtime parameter.

**Rule (hysteresis `k`):**

```
fine(t) = d(t) ≤ R(t)                          // refine
       ∨ (width(t) = 1 ∧ d(t) ≤ R(t) + k)      // keep fine until clearly away
       ∨ solidForced(t)                        // unchanged solid promotion
       ∨ authored region with maximum size 1   // regions become fine-only constraints
```

Every other tile prefers 4h. Coarse is only ever a preference, so strong 2:1
grading always resolves by demoting a coarse tile to 2h and never throws.

"Submerged" and "far air" are the same test: outside the band. The liquid
side and air side could take different radii later if measurements justify
it.

**Invariant (fail-fast).** No interface owner may sit in a non-h tile. The
classifier counts violations against the ownership that was live during the
frame. A nonzero count is a fatal diagnostic, not a silent coarse surface.

**Cadence.** Classify after every completed frame. Rebuild and remap only when
the fine mask differs from the previous one.

## Decisions (defaults taken, revisit on evidence)

- **Authored regions:** kept only as fine constraints. A region that caps
  cell size above h can no longer coarsen a surface tile, because the
  invariant wins. Region coarsening becomes redundant once dynamic mode is on.
- **Submerged width:** 4h. (History: 2h originally served as the collar; the
  layout is now ungraded h/4h and the collar was retired in 8a80095f.)
- **Mode switch:** runtime parameter `coarsening = regions | dynamic`, default
  `regions` until the lane passes. Then dynamic becomes the default and the
  region-coarsening path is retired, per the fail-fast / retire-legacy rule.

## Known problems, in the order they must be solved

1. **CPU layout build.** `createUniformMixedLayout`, `ownership.update`
   (breadth-first distance search, seam, regular and slot lists) and the
   pressure level layouts are all JavaScript, at about 27 × tiles work per
   build. Phase 1 accepts this and measures it. The end state is a GPU layout
   and work-list builder: Peter's standing rule is that work lists are rebuilt
   by the GPU frame plan on ownership change.
2. **Global owner renumbering.** `packUniformMixedLayout` numbers fine owners
   before coarse owners in key order, so one flip renumbers every owner after
   it. The fix is tile-stable addressing: base = key × (4 / w_level)³ within
   the already-reserved all-fine capacity.
3. **Pressure cold start.** `updateLayout` clears level-0 pressure, and the
   solve warm-starts from it every frame. Pressure must be restricted and
   prolonged with the other fields in the remap.
4. **Whole-domain remap and allocation on growth.** Remap runs two full
   `dispatchAll` passes and two submits. The hanging tap cache and the
   pressure seam records reallocate when a layout outgrows them. The fix is
   changed-tile worklists and worst-case preallocation.
5. **Seam area scales with the free surface.** The band edges are two seam
   sheets as large as the surface. Seams use the general sampler, 276-word
   hanging records and about 23 KB of pressure records per seam job, so
   record compaction is required before 256³. This is owned by the mixed
   performance session.
6. **Benefit needs scale.** Band thickness is (2R + 1) tiles plus two 2h
   collars, about 7–11 tiles. minidam64 (16 tiles tall) stays nearly all
   fine, so the target scene is 128³ or larger.
7. **Lossy remap.** Coarsening averages velocity (kinetic-energy loss) and
   refinement is piecewise linear. Hysteresis limits churn. Track KE drift per
   relayout.
8. **Lag.** The layout is decided from frame N and used for all of frame N+1.
   The band must cover a frame of travel. The invariant catches violations.
9. **V/phi disagreement.** The classifier keys on both fields (see the
   definition above).
10. **Dense-texture readers.** Fine vertices inside a tile that just
    coarsened keep stale values. Every dense phi/V reader (renderer, overlay,
    lenses, diagnostics) must resolve through ownership. Audit this before
    dynamic mode becomes the default.

## Coordination with the mixed performance program

The performance session (`docs/plans/uniform-small-region-performance.md`,
benchmark: 128³ high-resolution dam break) owns: ownership.ts, frame-plan.ts,
topology/face-dispatch/sampling WGSL, surface, extension, momentum,
pressure-stage/records/cycles, sharpening and transport. Dynamic coarsening
owns: `uniform-mixed-dynamic*.ts`, remap.ts, the host wiring in
`webgpu-uniform-reference.ts`, and this plan. Edits to frame.ts and layout.ts
are small and additive on both sides.

Incorporate its changes as they land:

- Record compaction and hanging-cache sizing: the prerequisite for item 5.
  Re-measure dynamic mode after each lands.
- Per-cell and per-row fast-path classification: under dynamic layouts, seam
  tiles are the common case, so these gains apply directly.
- Any GPU-side work-list rebuild in the frame plan: this is the natural home
  for item 1. Build the GPU layout builder on it rather than beside it.

## Phases

1. **Classifier and host loop (CPU layout build).** Add a GPU classifier
   (interface flags, separable distance, speed-adaptive band, hysteresis,
   invariant count, category counts), read back one bit per tile, rebuild via
   `createUniformMixedLayout(…, background 4, forced = solid ∪ band)`, and
   remap. Opt in through the `coarsening` parameter. Measure on the 128³ dam
   break: relayout count, CPU build ms, remap ms, pressure cycles after
   relayout, frame time against all-fine.
2. **Changed-tile remap including pressure** (items 3 and 4).
3. **Tile-stable owner addressing** (item 2), coordinated with the
   performance session because pressure storage and records index by owner.
4. **GPU layout and work-list builder** (item 1): removes the readback and
   CPU build from the frame loop.
5. **Dawn lane.** A moving surface at 128³ with checks for the invariant,
   mass and KE drift, no allocation after initialization, relayout count, and
   frame time against all-fine.
6. **Default on.** Retire region coarsening and audit dense readers (item 10).

## Status

- 2026-09-27: plan written. Phase 1 landed (uncommitted):
  - `lib/methods/uniform/uniform-mixed-dynamic.ts`: census with classify,
    separable distance, and a band/hysteresis decision; 1 bit per tile; cause
    counters.
  - Host loop in `webgpu-uniform-reference.ts` (`updateMixedDynamic`, runs
    inside the pending frame).
  - Parameters `coarsening`, `coarseningReach` and `coarseningHysteresis`.
  - `FLUID_MIXED_DYNAMIC_CENSUS=1` runs the census on authored-region layouts
    as a diagnostic and adopts nothing.

### Phase 1 measurements (128³ high-resolution dam break, tolerance 5, reach 2, hysteresis 1)

| step | all-fine pass sum | dynamic pass sum | fine / 2h / 4h tiles (dynamic) |
| ---: | ---: | ---: | --- |
| 1 | 75.6 ms | 88.2 ms | 10918 / 2682 / 19168 |
| 2 | 72.3 | 104.5 | 11336 / 2757 / 18675 |
| 4 | 74.2 | 121.3 | 17074 / 3004 / 12690 |
| 7 | 78.8 | 165.8 | 22504 / 2537 / 7727 |

- The GPU is slower in mixed mode even with 60% of tiles at 4h. Seam and
  general-path overhead outweighs the saved work (problem 5). The census
  itself costs 0.4–1.5 ms.
- The CPU cost is 20–30 ms of layout build plus about 80 ms of apply
  (remap and two `ownership.update` passes) on every relayout, which is every
  frame while the dam spreads (problems 1 and 4).
- The surface never escaped the band: 0 phi crossings in coarse owners over
  7 steps.
- **Retracted, 2026-09-27: coarse liquid V/phi drift is not a defect.** The
  earlier "blocker" compared deep coarse owners with near-surface fine owners.
  The authored "submerged" regions also missed the reservoir, which spans
  x, z 0–62.5% and y 0–92%, and reached dry floor, where wall vertices hold
  phi ≈ −1e-5.
  - Test: the same cells (x 6–40%, y 20–60%, z 6–40%, fully submerged) under
    both layouts, for six steps.
  - Largest V shortfall below full, all-fine: 0.015, 0.05, 0.084, 0.098, 0.104.
  - Largest V shortfall below full, 4h: 0.0002, 0.0005, 0.022, 0.049, 0.071.
  - Largest V excess: fine up to 0.044, 4h up to 0.026.
  - The 4h region had no phi crossings. At pressure tolerance 5, submerged V
    drifts about 10% at every resolution. Fine owners two or more widths
    inside phi reach shortfalls of 0.97 globally.
- **Invariant narrowed.** Only a phi sign change inside a 2h/4h owner is
  fatal: the geometric surface must stay in h. V/phi disagreement without a
  crossing is interface for classification, so the tile refines at the next
  census, but it is not a violation. It occurs deep in all-fine liquid too.
  `interiorDeficit` now measures owners with every corner at least two widths
  inside phi.
- **Dry floor stays fine.** Wall vertices never dry (phi ≈ −1e-5 under air),
  so dry floor tiles classify as interface and are never coarsened. This is
  conservative, and it costs a tile layer along every dry wall. The cure
  belongs to the wall-vertex drying issue (uniform voxel bath program), not to
  a classifier tolerance.
- `UNIFORM_MIXED_DYNAMIC_FULL_TOLERANCE` = 0.25 is above the method's own
  interior drift of about 0.1.
- **Next:** phases 2–4 (CPU relayout cost), measured against the GPU seam
  overhead that the performance program is removing.
- 2026-09-27, CPU relayout cuts (mock-device profile, 128³, a band of about
  12.8k h / 3.1k 2h / 16.9k 4h tiles):
  - `updateLayout` rebuilds a pressure level only when its tile widths change.
  - `UniformMixedOwnership.update` memoizes all of its derived arrays per
    layout object (`deriveOwnership`, a WeakMap). The remap target and the
    frame root share one derivation, and the breadth-first distance search
    uses a typed queue.
  - Per-update cost: root 23 → 0 ms, target 23 → 6 ms, L1 and L2 17–30 →
    6–7 ms (warm).
  - A relayout now costs about 63 ms on the CPU, down from 110–130 ms. What
    remains is `createUniformMixedLayout` (22–27 ms) and the two
    `uniformMixedPressureLevel` builds (7–16 ms each). Both belong to the
    phase 4 GPU builder.
- 2026-09-27, **phase 4 landed: GPU layout builder** (uncommitted).
  - `lib/methods/uniform/uniform-mixed-layout-builder.ts` builds, from the
    census band bits and a static fine mask (solid promotion plus fine-only
    regions, computed on the CPU only when solids or regions change), the h
    ownership and its 2h pressure level. The 4h level never changes.
  - The output is exactly the buffers `UniformMixedOwnership.update` uploads:
    - tile words (owner numbering by a two-level scan over 256-tile blocks);
    - h/2h/4h worklists and stencils;
    - counts and the frame-plan header;
    - chessboard distance (three separable passes, exact);
    - fine, seam and regular lists;
    - the hanging slot table.
  - The host reads back a 16-word receipt and the tile words.
    `UniformMixedOwnership.adopt` copies the rest on the GPU, and
    `UniformMixedRemap.applyBuilt` remaps in one encoder.
  - `uniformMixedLayoutFromTiles` gives the host an honest layout: worklists
    and stencils are derived from the tile words only on first host access.
  - Dawn test `tests/uniform-mixed-layout-builder-dawn.test.ts` (deleted in 8a80095f): every range
    `update()` writes, byte-compared against the builder, on 64×48×80 and
    256×128×160 lattices (the second needs a multi-chunk block scan). Also
    checks the lazy host arrays, changed-tile counts, and a rebuild after
    adopt reporting zero changes.
  - 30-step 128³ dam break, GPU builder (dynG) against the CPU path (dynF):
    - h/2h/4h tile counts are identical at every step, and so is volume drift
      to every printed digit.
    - Relayout host cost is 0.2 ms, down from about 50 ms. GPU build is
      0.13–0.26 ms, census 0.5–1.9 ms, remap 3.5–6.4 ms.
    - No phi crossings in coarse owners over 30 steps.
  - Authored region edits keep the CPU path; they are rare.
  - Same probe and same 30 steps against the unified all-fine arm (no
    regions). Medians:

    | steps | pass sum, all-fine | pass sum, dynamic | wall, all-fine | wall, dynamic |
    | ---: | ---: | ---: | ---: | ---: |
    | 1–7 | 74 ms | 98 ms | 94 ms | 130 ms |
    | 8–15 | 86 | 106 | 110 | 135 |
    | 16–30 | 108 | 110 | 153 | 139 |

    Dynamic already wins wall time once the front has formed. Early steps
    lose on GPU seam work (problem 5, the performance program). The earlier
    phase 1 table's 72–79 ms all-fine numbers were steps 1–7 only, and its
    dynamic pass sums (88–166 ms) predate the performance program's latest
    changes.
  - **Next:** phase 2 (changed-tile remap: 3.5–6.4 ms per relayout now) and
    phase 3 (tile-stable addressing), then the Dawn lane (phase 5).
- 2026-09-27: phase 2 and phase 5 landed. Phase 3 is not needed.
  - Phase 2, changed-tile remap (`uniform-mixed-remap.ts`). `markChanged`
    lists every tile whose 3×3×3 neighbourhood changed width, comparing the
    live and target tile words. The remap and publish passes then dispatch
    one group per listed tile, indirectly. Every other owner's remap is the
    identity, so its live value stays in place. Region edits (CPU `apply`)
    and GPU-built generations (`applyBuilt`) share the path.
  - A/B against the HEAD full remap on identical canonical fields, 64×48×80,
    4 random layout pairs with 1.7k–2.8k of 3840 tiles changing width:
    - phi is bitwise equal;
    - volume, velocity and the negative boundary differ by ≤ 2.2e-7
      relative (≤ 1 ulp). The full remap re-averaged unchanged coarse owners
      (64 equal values / 64), adding those ulps; the worklist no longer does.
  - 30-step 128³ probe: remap median 4.8 → 3.4 ms. The trajectory diverges
    after the step 9–12 splash (the ulps above), so pass sums are not
    comparable step by step.
  - Phase 3 (tile-stable addressing) is not needed. Fields are dense
    position-indexed textures, and pressure and records are rebuilt every
    frame, so renumbering owners costs nothing beyond the adopt copies.
    Pressure needs no remap either: the cold start costs no extra cycles
    (dynamic 3–4 against all-fine 2–6).
  - Phase 5 lane: `npm run test:dawn:uniform-dynamic-coarsening`
    (`tests/uniform-dynamic-coarsening-dawn.test.ts`). It runs the app's
    method with `coarsening:"dynamic"` on the 128³ dam break for 30 steps
    (`UNIFORM_DYNAMIC_LANE_STEPS`) and asserts:
    - per step: tile counts, zero phi crossings in coarse owners,
      |volume drift| < 1e-4;
    - relayouts ≥ half the steps, and fine tiles that start under half and
      then grow;
    - no shader or pipeline creation after frame 2, and no uncaptured errors.
    It passes with 29 relayouts and no allocations after frame 2.
  - Open lead, the band cap at steps 9–15. The splash drives max speed to
    about 25 m/s in both the dynamic and the unified all-fine arms. That is
    a solver property, not coarsening. The census radius is the tile's
    `speeds[UM_TILES+t]`, a box maximum over the *global* certificate
    radius, times the frame dt. It exceeds `UNIFORM_MIXED_DYNAMIC_DISTANCE_CAP`,
    so every tile goes fine (32768/0/0) until the spike decays. The sound
    local form:
    - each interface tile emits its own radius;
    - a tile is fine iff it lies in some interface tile's Chebyshev cube;
    - this is separable: x takes the max radius among row candidates with
      |dx| ≤ r, then y and z the same;
    - each radius comes from a local speed box grown to its fixed point, not
      the global box.
    This only pays off if the fast speeds are localized.
  - Pre-existing failures in the mixed suite, not caused by this work:
    - `uniform-mixed-remap-dawn` trips its no-allocation guard, because the
      tap cache grows on a live edit (a32e2ad9);
    - `uniform-mixed-solid-parity-dawn` coarse-region mass (the same
      5420.98 vs 6144 in logs from 2026-09-26 18:58);
    - `uniform-mixed-native-transport-dawn` has a stale byte formula.
  - Default stays Regions. Dynamic wins wall time after about step 16, but
    loses over the whole 30 steps (early GPU seam work, performance program).
- 2026-09-27: visual layers follow the live ownership. The grid-overlay slice
  draws the represented cell per owner: the grid lattice, sample dots, density
  bars, velocity arrows (one per owner, sampled at its centre), body occupancy,
  the volume readout (one number per owner) and the tiles layer's internal
  lattice (4×4 in h tiles, 2×2 in 2h, none in 4h). Phi, surface, density,
  volume, pressure, velocity and released faces already resolved through the
  shared presentation topology, which `updateLayout` rewrites in place, so a
  relayout reaches every layer with no rebind.
- 2026-09-27 overlay compile hang: the grid-overlay render pipeline never
  finished building on Metal (>200 s, also at HEAD), so no visual layer ever
  appeared (`createRenderPipelineAsync` just never resolved). Metal inlines
  every call, and the mixed presentation samplers were inlined at dozens of
  sites (per field mode, per layer iteration, per volume-march step). Fix:
  each fragment samples fluid/phi/velocity/zero-contour once, and
  `fragmentMain` has a single `gridSample` call site shared by the legacy
  slice, the layered composition and the volume march; presentation loop
  bounds are runtime (`umPresentationLoopBound`). Build is now ~11 s cold.
- 2026-09-27 surface-only census (predicted surface tiles). Reach and
  hysteresis now default to 0: tile t is fine iff it is a current interface
  tile or its RK2 departure box, cells `[4p - hi·s, 4p + 4 - lo·s]`, holds one
  (3D prefix sum over interface flags). Signed per-axis velocity bounds come
  from a separable min/max pyramid over radii 0,1,2,3,4,5,6,8,10,12,16 tiles.
  Each tile uses the smallest level that covers its own RK2 midpoints,
  `1 + ceil(travel/8)`. Three escapes shaped the rule:
  - Bounds must come from the field the next trace samples. Air faces at
    census time are stale, and the extension's hierarchy fills far faces
    from band tiles many tiles away (bounding by the nearest source, even
    widened by sqrt(3), missed a far-wall run-up and doubled the band). The
    host now calls `UniformMixedFrame.encodeExtension` (the last advance's
    plan and extension into velocityScratch) before the census, and every
    owner's extended faces feed the pyramid.
  - Zero joins every bound. Sampling near a wall or solid blends in the
    zero face, and the trace stops short at a solid or the domain clamp.
  - A current interface tile stays fine. Residual sheets behind a falling
    surface (the alternating -0.008/-0.036 wall-vertex pattern) do not move
    with the flow.
  Result, dam break with tolerance 5 over 30 steps: no violations, and 5-11k
  fine tiles against 24-27k for the old band. The lane passes (30
  relayouts, no allocations after frame 2). Medians over steps 16-30,
  against all-fine:
  - Wall 157 vs 153 ms; pass sum 132 vs 109 ms.
  - Census 4.3 ms, of which classify 3.3 and prefix/pyramid 1.0.
  - Remap 7.3 ms.
  - The census's second plan+extension run costs about 4 + 4 ms.
  - Seam-heavy surface stages lose about 40 ms: momentum +8.3, global
    surface volume +6.8, body forces +5.2, advect +4.3, redistance +4.1,
    fill +2.7, hanging taps +2.1.
  - Pressure and sharpening win about 28 ms: sweep -9.5, sharpening -6.4,
    mg smooth -3.0, rowsDivide -2.4.
  Next leads:
  - Reuse the census extension in the next frame instead of running it
    twice. The remap must then carry velocityScratch.
  - Cheaper remap under per-frame churn.
  - The seam-path cost in the performance program's stages. In a 1-2 tile
    band almost every fine tile is a seam tile.
- 2026-09-27 h/4h exploration (a temporary switch relaid out with
  `createUniformMixedLayout(..., stronglyBalanced=false)`; since removed).
  - Dam break, surface-only, same fine tiles: the pass sum was 74/89/92 ms
    against 91/103/117 ms graded at steps 2-4. Removing 2h seams cut
    momentum by 4.7, surface volume 4.7, advect 2.7, redistance 2.1 and
    forces 2.0 ms. At step 5 pressure failed ("did not converge", 18.8
    against tolerance 5 within the 4-cycle budget).
  - Still tank (tank-fill, 4h region x < 50%, y < 25%, 30 steps,
    tolerance 5): maximum spurious speed was 0.015-0.023 m/s for all-fine
    and for graded 2:1 alike. Ungraded 4:1 reached 0.61 m/s at step 1
    (residual stalled at 2.3), then pressure diverged at step 2 (non-improving
    cycle, candidate 60).
  - Verdict: the mixed pressure operator and multigrid are 2:1 only. The
    ungraded mode is not a usable 4:1 discretization. h/4h needs a
    different pressure coupling, such as a 4h global solve plus a
    flux-constrained fine band correction, before anything else can move.
    The hydrostatic split cannot be judged until then.
- 2026-09-27 pressure-only transitions (production; the only mode since the
  graded "all" mode, its UI select and `mixedPressureOnlyTransitions` were
  retired the same day).
  - The simulation layout L is ungraded h/4h. Every transport, surface,
    momentum and force stage runs on it, with no 2h seams.
  - Pressure level 0 runs on G = `uniformMixedPressureLayout(L)`. G is
    refine-only: a 4h tile in the 26-neighbourhood of an h tile becomes 2h.
    Levels 1 and 2 derive from G, so the existing 2:1 operator and
    multigrid are untouched. This avoids the T-junction problem without a
    new solver.
  - `UniformMixedOwnershipTransfer` maps L to G before the pressure
    stages. It does a whole-texture copy, then the tile-parallel remap over
    the `markTransfer` worklist: changed tiles plus unchanged coarse +axis
    neighbours. Volume and phi go to scratch; velocity and negative faces
    go into the rhs inputs.
  - Pressure geometry and authority write G's own target and centerPhi,
    so the renderer keeps L's. G to L remaps faces only. That is
    flux-exact (a 4h face is the mean of its 2h faces), so the field
    stays divergence-free on L.
  - `presentPressure` re-indexes level-0 pressure and phi by L owner for
    the overlay.
  - When G == L (no h tiles, or all fine), the direct bindings run and
    nothing is paid.
  - The builder emits [L ungraded, G graded, 2h] in one GPU pass.
  - The remap and transfer kernels are tile-parallel: one workgroup per
    tile, 64 cell lanes or 192 face lanes. Remap median 6 to 3 ms.
  - Dam break 128³, dynamic, steps 16-30 medians: pass sum 119.0 against
    130.6 ms graded, wall 148.8 against 154.6 ms.
    - Surface stages saved: surface volume 4.6, advect 2.8, momentum 2.5,
      plan 2.2, forces 2.1, census 1.2, geometry 1.0 and taps 0.9 ms.
    - Transfers cost 2.4 + 2.7 ms, the extension re-authority about 1 ms
      and the copies about 1 ms.
    - Pressure geometry on G costs 1.4 ms against 0.7 on L.
  - Still tank: 0.01 m/s from step 8, the same as graded. The old 4:1
    pressure reached 0.61 m/s.
  - Isolated timings only: Metal pass timestamps include queue waits. The
    presentation pass read 6.5 ms by timestamp and 0.16 ms in isolation.
  - Next leads:
    - Transfer velocity in place rather than copying whole textures.
    - A cheaper pressure geometry.
    - The 20-40 m/s splash spike at steps 8-15, which graded shows too.
- 2026-09-27 4h-first census (P1, P2, P3, P6, fast gate, forward dilation).
  h is kept only where 4h cannot represent the surface; everything else,
  including fast fronts, runs at 4h.
  - P1 face-aware dilation. classify packs, per required tile, six nibbles
    of cells from each face to the nearest crossing owner. Dilation reaches
    a neighbour only as far as the surface actually lies from that face.
  - P2 only phi crossings refine. V/phi disagreement in a coarse owner is
    counted (`uniformMixedDynamicCoarsePartialVolume`), not refined.
  - P3 resolvability, `coarseningSurfaceTolerance` tau (default 0.5
    cells).
    - An h tile may coarsen when trilinear phi from its 8 tile corners
      matches its 125 vertices within tau*h near the surface
      (min(|phi|,|I|) < 2h).
    - A 4h tile refines when its 4h-lattice second difference / 8 exceeds
      2*tau*h (the factor 2 is hysteresis).
  - Fast gate, `coarseningFastTravel` F (default 4 cells/step, one 4h cell;
    about 0.75 m/s on the dam scene).
    - A tile whose own face-velocity bounds give max|u|*dt/h >= F (F/2 for
      a 4h tile, hysteresis) is not required fine.
    - A fast front runs at 4h. It resolves no worse there: it moves more
      than one 4h cell per step either way, and 4h is the better-conditioned
      advection.
  - Forward dilation (F > 0). Tile p is fine when some required tile q
    within ceil(F/4)+margin reaches it:
    gap(q, facing face) + 4*(tile distance - 1) <= travel_q + 4*margin.
    The backward departure-box test remains only for F = 0.
  - required = crossing AND NOT fast AND NOT resolvable. The fatal
    "phi crossing in coarse owner" invariant is retired;
    `uniformMixedDynamicUnresolvedCoarse` counts 4h tiles the census
    refined, for information only.
  - P6. The census's plan and extension are reused by the next advance
    when the parameters are identical and nothing changed. Layout changes
    and every host mutator invalidate them (`invalidateExtension`).
  - Still tank: 0 fine tiles (was 1024), wall 24 ms (was 68.6 ms).
    Spurious speed 0.33 m/s is the all-4h free-surface problem (0.39 m/s
    all-4h), under investigation.
  - Dam, steps 16-30 medians: wall 101 ms against 148.8, h tiles 1970
    against 11320. Steps 3-15 run all-4h at 24-30 ms. Churn is 500-1200
    refined tiles per step. Late dust (26-37 cells) and drift -1.2e-4 at
    step 30 still fail the lane's 1e-4 gate.
- 2026-09-27 round-trip consistency of ownership changes.
  - Phi: coarsening samples canonical corners, and refinement fills fine
    vertices trilinearly from them, so coarse -> fine -> coarse is the
    identity. Fine -> coarse -> fine loses only detail the census judged
    below tau.
  - Faces: refinement interpolates each fine face between the coarse
    owner's two patches, and boundary fine faces take the patch itself.
    Coarsening averages the fine faces over the patch. Round trips are
    exact and each fine cell inherits its parent's divergence.
  - Volume: refinement used to broadcast the donor's V into every fine
    cell, a uniform V that contradicts the sharp phi it had just
    interpolated. The volume-correction/sharpening stages then moved mass
    to reconcile them, a spurious source at every refinement.
    - Now each cell takes its new owner's geometric fill (umSurfaceTarget
      on the remapped phi, the same target the solver compares against).
    - Fills are scaled so the donor's volume is conserved exactly. Filled
      fractions shrink when the donor holds less than phi implies, empty
      fractions shrink when it holds more, and every value stays in
      [0,1].
    - Averaging back on re-coarsening returns the donor's V exactly.
    - Measured with a forced all-fine -> all-4h -> all-fine -> all-4h
      probe. coarse -> fine -> coarse: dV 7e-7, dP 0, dU 2e-8.
    - Median |V - fill(phi)| after refinement fell from 0.38 to 0.05 in the
      tank and from 0.30 to 0.10 in the dam at step 20. The tank's residual
      is its 4h state's own V/phi disagreement, which the remap conserves
      rather than reshapes.
    - Overfull donors (V > 1), and donors whose phi fill is under half a
      cell or over n - 1/2, keep the uniform broadcast.
- 2026-09-27 free-surface theta. The all-4h still tank at 0.39 m/s was
  ghost-fluid theta, not width: an all-fine tank filled through a row of
  centres boiled the same way.
  - A liquid centre on the surface discharged its clamped pressure
    (rho g * 0.05 d) sideways through tangential faces.
  - The 0.05 clamp made the effective surface jump whenever a centre
    crossed phi = 0.
  - Fix: `umPressureSurfaceTheta` floors the liquid depth at
    theta_min * spacing before the ratio, with
    `UNIFORM_MIXED_THETA_MIN` = 1e-3. It is the single theta source for
    the rows, operator, ghost slopes, reconstruction, projection and the
    open lid.
  - Result: the all-4h tank runs at 0.03 m/s, and all-fine is
    bit-identical.
  - The remaining motion at tolerance 5 comes from cold-start pressure
    acceptance (the one-cycle residual has the same sign every step).
    Leads: warm-start pressure, or a width-aware tolerance.
  - Dam, dynamic, with both fixes: drift at most 7.8e-5 (was 1.24e-4),
    step-30 dust 8.9 (was 37), wall median steps 16-30 115 ms (h tiles
    2326 against 1970, a chaotic run with more cycles).
- 2026-09-27 surface break-up on refinement (4h -> h). The remap worklist
  skipped tiles that were h in both layouts. When a neighbour refines, an
  equal-width tie in `umVertexAuthority` hands such a tile the vertices on
  the neighbour's -x/-y/-z faces. The old layout derived those vertices from
  the 4h corners and never stored them, so the stale texel (from before the
  tile coarsened) became the surface.
  - Measured on the dam break: every refined tile's low faces disagreed with
    its 4h surface by up to 100 h, with 1-6k sign flips per step. Small
    liquid components rose from 2 to 260 against about 20 all-fine, and
    pressure diverged at step 33.
  - Fix: `markChanged` also lists an h -> h tile beside a refining tile. Its
    cells and faces remap as the identity, and its vertices are rewritten
    from the old layout's sampling.
  - Result: every stored vertex of every refined tile equals its 4h
    trilinear surface (40 steps). Small components match all-fine, and the
    run completes.
  - The lane asserts this on every refined tile.
- 2026-09-27 boundary rules (far-wall run-up stall). On the long dam
  (`sparse-cm12-long-dam-break`), the 4h-first census kept the whole
  impact at 4h: the fast gate exempted it, and the pile's phi is linear
  enough to pass P3. Arriving liquid sat in 4h owners whose centre phi is
  air ("hidden V": no pressure row, no momentum of its own). Hidden V in
  the last 16 columns grew from 1k to 3k cells (all-fine 0.1-0.3k), and the
  run-up stalled at 9-17 cells for five steps.
  - The fast 4h run-out also holds 17-20% hidden V and is harmless there,
    so hidden V alone cannot be the trigger. The trigger is the boundary.
  - `coarseningBoundaryTravel` (default 1 cell/step; 0 disables). A
    crossing tile is required h whatever its speed when either rule holds:
    - Impact: it is within one 4h cell of a closed wall or solid-coupled
      tile, moving toward it faster than along it.
    - Lift: it is on a closed vertical wall, moving up faster than along
      the wall.
    Floor run-outs and fronts passing a wall meet neither rule.
  - Boundary tiles dilate by their own directional travel (a second
    per-tile word, capped at 16 cells/step). Isotropic dilation by their
    speed made 4-7k h tiles at 128³. Required tiles, not partial-V tiles:
    refined spray fell under the dust threshold (9 cells in three steps).
  - Contact, not predicted arrival. Adding the step's travel to the reach
    refined the toe early and held it at x=180-185 for three steps.
  - Long dam: the front matches the rule-free arm, reaching the wall at
    step 23. Run-up is 12, 16, 29, 43, 57, 78 and 96 (the ceiling) at
    steps 24-34, where the old run stalled at 9-17. H = 0.5 m, so ideal
    u²/2g is ~80 cells. Whether 96 is physical is open (energy trace).
  - Cost: the lane passes with drift <1e-4, but its median wall is
    96.9 ms against 62.5 ms without the rules. h tiles peak at ~4.4k
    against ~1k at 128³. On the long dam, steps 20-40 cost 43-46 ms
    against 24 (rules off) and 27 (all-fine). Thresholds 1, 2 and 4
    cells/step give the same cost: the impact itself is what goes h.
