# Fluid work inspection

Per-cell visibility for the power-liquids solver, in the spirit of the SVO pixel
trace. Status: the interactive cell trace and both measurement instruments are
in; none of the three has been confirmed visually in the running app.

## Why this is not a pixel trace

`svo-pixel-trace.ts` works because one pixel is one ray is one sequential shader
invocation: a probe re-runs the shipping traversal law and appends records in
execution order, exactly, with no perturbation of production.

A pressure cell has no such property. Its work is spread across every pass of
the frame and repeated once per smoother sweep, per V-cycle level, per outer
iteration. From `artifacts/xctrace-mini-dam-16-coarse-2frames-v2-2026-07-31`,
one advance of the **16³** mini dam runs **972 GPU passes**, 24.7 ms busy inside
58.7 ms wall, at **2.55% occupancy**, with the pressure solve alone accounting
for 769 calls and 14.0 ms. There is no replayable probe for that; anything
per-cell must either read what the frame left resident, or be derived from the
published schedule and topology.

Both instruments below take one of those two routes. Neither compiles a shader
variant and neither perturbs an accepted advance.

## Instrument 1 — flood provenance (resident-state route)

The redistance commit leaves the resolved closest-point seed in `workA` and the
packed sub-cell code above the sample flags. The flood's dependency graph is
therefore already in memory after the frame.

- `lib/fine-flood-provenance.ts` — the decode plus the ladder arithmetic.
- `lib/webgpu-fine-flood-provenance.ts` — accumulation pass over resident pages.
- `tools/report-fine-flood-provenance.ts` — the report.
- Overlay mode `flood-provenance`, on the fine-band lifecycle pipeline.

Each sample is binned by the **leading passes of the ladder that actually ran**
whose combined reach covers its hop, so bin *k* names pass *k* of that frame's
schedule rather than of an idealised one.

### Result: the ladder is well matched, not over-provisioned

| | mini dam (8 steps) | large dam (6 steps) |
| --- | --- | --- |
| resident / resolved samples | 171,963 / 171,963 | 199,696 / 199,696 |
| encoded ladder | `[8,4,2,1,1,1]` | `[8,4,2,1,1,1]` |
| total encoded reach | 17 fine cells | 17 |
| deepest hop observed | 16 fine cells | 16 |
| passes needed to cover it | 5 of 6 | 5 of 6 |
| closed by the first (stride-8) pass | 58.4% | 53.9% |
| unresolved | 0 | 0 |

The recurring flood is **not** where the waste is. Roughly half the field is
closed by the first pass and the schedule is one trailing pass longer than the
deepest straight-line hop requires — and that pass is a collar repair, which
exists for page-boundary turning routes a straight-line hop does not measure.

### A model correction worth keeping

The first version judged hops against the descending reach `2P - 1` and reported
an impossible **negative** surplus. Collar repairs extend reach by one cell each,
so the encoded reach is the sum of all strides (17), not `2P - 1` (15). The
summary now compares against the full encoded ladder and reports
`coveredByEncodedLadder` separately, because a warm publication carries and
remaps the previous closest-point field: a seed link can legitimately be older
and deeper than any single frame's flood could have built.

## Instrument 2 — pressure blast radius (derived route)

- `lib/fluid-blast-radius.ts` — schedule and cone growth.
- `tools/report-blast-radius.ts` — the report. No GPU work at all.

The operator is symmetric, so the inbound and outbound cones are the same set.
Each stage of the encoded schedule is a reach on the hierarchy: a smoother sweep
or operator application expands the cone by one cell at its level (`applied()`
reads an 18-point stencil); restriction carries it up through the parent map;
prolongation carries it back down into every child. Growth is tracked as one
axis-aligned box per level, which is exact for a dense hierarchy and therefore
an upper bound on any sparse resident set.

### Result: the blast radius is not a radius

`webgpu-octree-spgrid-vcycle.ts` reports `bottomOperation: "exact-single-cell"`
and refuses a `maximumLevels` that would truncate that bottom. So the pyramid
always descends to one cell, and prolonging a full one-cell level back down
doubles the box at every level.

| scene | domain | levels | stages to global | of total |
| --- | --- | --- | --- | --- |
| mini dam | 16³ | 5 | 24 | 260 |
| ocean seiche | 320×96×80 | 10 | 54 | 560 |

Every cell depends on every other cell within the **first V-cycle**, from
anywhere in the domain, at any domain size. The level-0 coverage curve shows a
long plateau while the cone climbs the coarse grids and then jumps to 100% on
the final `prolong L1->L0`.

The consequence for performance work: there is no locality left to exploit after
one V-cycle, so the question is not how far a cell reaches but how much level-0
work the schedule spends afterwards. The mini dam encodes **10 outer iterations
and 50 full sweeps of its 4,096-cell level-0 grid** — against a residual gate
that zeroes the tail, and a measured 769 pressure passes per advance.

### The sparse half: `lib/webgpu-fluid-blast-radius.ts`

The analytic result is topology-independent, which is exactly why it cannot show
what a given frame's adaptivity does to the *fine-grid* reach. Overlay mode
`blast-radius` floods the level-0 row graph as published — the 18-direction power
stencil resolved through the owner map the renderer already binds — and colours
every row by its hop distance from a seed.

The seed comes from the existing slice controls (`u.debug.x` axis,
`u.debug.y` fraction), so the slice slider sweeps it and no new UI was needed.
Flood is `clear` → `seed` → 24 × `relax`, each relax being one finest-grid
dispatch that offers `distance + 1` to the rows owning its eighteen neighbours;
cells interior to a row resolve to that same row and contribute nothing, so the
pass is exactly a relaxation over the row adjacency graph.

What it shows that the dense model cannot: **reach anisotropy**. One hop crosses
a whole 32³ coarse leaf but only one finest cell inside a refined band, so the
cone is lopsided in proportion to the adaptivity the scene chose.

Scope, stated in the module: this is the smoother/operator reach *only*. It does
not model restriction or prolongation, so it is the cone before the coarse grid
short-circuits it. Read it beside the analytic report, not instead of it.

## The interactive tool: cell trace

The two instruments above are measurement, not a tool. The tool is
`FluidCellTraceHud`, the solver-side counterpart of `PixelTraceHud`: enable it,
hover the fluid, click to pin, read what the solve does to that one cell.

- `lib/fluid-cell-trace.ts` — record ABI, decode, narrative builder.
- `lib/webgpu-fluid-cell-trace.ts` — the gather pass.
- `components/FluidCellTraceHud.tsx` — the HUD.

**Selection is a ray**, built from the pointer's pixel. The host sends the aim
in world space — `viewportRayForPixel`, the documented inverse of the WGSL
`cameraRay` — and the gather marches it through the owner map, recording every
leaf it crosses so the selection can be stepped into the interior.

The aim rather than the pixel is what a pin freezes, and that distinction is the
whole of it: a pixel names a different cell as soon as the camera moves, so a
pinned selection built on one slides onto whatever pans under it. Pinning simply
stops updating the ray.

### The honest structure of a cell trace

A ray can be replayed because it is one sequential invocation. A cell cannot, so
the HUD reports two things and never adds them:

- **Gathered** — what this frame published about the cell: leaf size, compact
  row, operator diagonal and right-hand side, entry count, power-cell volume,
  the eighteen stencil neighbours with their leaf sizes and which are coarser or
  finer, and a bounded probe of the fine-band samples inside the leaf.
- **Scheduled** — what the encoded command graph does to a row: level-0 sweeps,
  stencil re-reads, and stages until the dependency cone covers the domain.
  A residual gate may zero the tail, so these are badged `scheduled` in the UI
  and are never presented as observed.

The resolution-transition strip is the part that pays off fastest: it names each
neighbour sitting at a different leaf size, which is both the paper's hard case
and the reason fine-grid reach is uneven.

The fine-band figures come from a 512-probe lattice over the leaf, not a census
— a 32³ leaf at factor four holds over two million samples — and the HUD says so
rather than implying exactness.

### The fine-band half is now the larger half

What is described above as "a bounded probe of the fine-band samples" was four
scalars. It is now a per-probe record set and eleven views, including the one
that closes the loop from surface geometry back to the operator: the ghost-fluid
θ on each dual edge, which is what makes the row's diagonal explicable rather
than a bare float. See `FINE_BAND_CELL_INSPECTION.md` for the set, what did not
land, and the silent flag bug it uncovered.

## Not built / not confirmed

- The *field* blast-radius view still seeds from the slice plane, with the
  domain centre on the other two axes. The cell picker's cone decorator does
  follow the clicked cell; the volumetric hop field does not.
- No stage scrubber on the sparse cone; it shows the whole hop field at once,
  which is more legible than stepping but loses the per-stage story the analytic
  report carries.
- The sparse cone does not include the multigrid transfers. Adding them needs a
  flood over the persistent MGPCG `state`/`topology` level slots projected back
  through `rowMap` — the version that would show where ghost and air rows break
  the cone.
- ~~Both overlays and the cell trace are shader-validated and unit-tested, but
  none has been confirmed visually in the running app.~~ The **cell trace is now
  confirmed in the running app** on the dam break: pick mode armed from the
  scene's top-right toggle, a cell pinned by clicking the viewport, line work
  drawn around it, and every HUD panel rendering. The two measurement
  instruments and the field overlays are still shader-validated only.
- ~~The cell trace has no 3D line work.~~ It does now, through the
  visualization framework — see `VISUALIZATION_FRAMEWORK.md`. Hovering outlines
  the cell under the pointer; pinning draws the stencil, the resolution
  transitions, the dependency cone and the flood reach, all placed through the
  same transform the gather inverted to pick the cell.
- The trace's schedule inputs assume a dam-break tail policy with leaf 32 and an
  open top. For a scene with terrain, inflow or a closed lid the scheduled
  figures will be off; the gathered half is unaffected.
