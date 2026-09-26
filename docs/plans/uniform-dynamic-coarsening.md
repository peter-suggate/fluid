# Uniform Geometric: dynamic coarsening

## Goal

Replace authored enforcement regions as the source of coarseness. Ownership
follows the simulation state: tiles near the free surface stay fine (h), and
submerged and far-air tiles become coarse (4h), with the existing 2h grading
collar between them. This uses the same single mixed frame, the same
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
- **Submerged width:** 4h, with 2h only as the collar.
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
  - Dawn test `tests/uniform-mixed-layout-builder-dawn.test.ts`: every range
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
- 2026-09-27: visual layers follow the live ownership. The grid-overlay slice
  draws the represented cell per owner: the grid lattice, sample dots, density
  bars, velocity arrows (one per owner, sampled at its centre), body occupancy,
  the volume readout (one number per owner) and the tiles layer's internal
  lattice (4×4 in h tiles, 2×2 in 2h, none in 4h). Phi, surface, density,
  volume, pressure, velocity and released faces already resolved through the
  shared presentation topology, which `updateLayout` rewrites in place, so a
  relayout reaches every layer with no rebind.
