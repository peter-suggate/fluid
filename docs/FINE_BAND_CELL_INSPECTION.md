# Fine-band cell inspection

What the Section 5 narrow band did to one pressure cell, and what it handed back.

Status: **eleven of sixteen views landed and are confirmed in the running app** —
pinned cell, line work in the scene, every panel rendering. The five that did not
land are listed at the bottom with the reason each is blocked. One real bug was
found and fixed on the way; it is recorded below because it was silent.

## The gap this closed

`FluidCellTrace` carried four fine-band numbers, all from a 512-probe lattice:

```
fineSamples  fineResolved  fineMaximumHop  fineInterface
```

and one decoration, `fine-band/flood-reach`, which dilates the leaf by the
deepest hop. Against a 32³ leaf at fine factor 4 — 2,097,152 samples — that is
one count, one ratio and one bound, and none of it says what the band handed the
pressure row.

## The organising question

The fine SPGrid stores φ only; velocity and pressure stay on the octree. So the
band's whole job is to collapse `(leafSize × m)³` signed distances into the few
numbers one row needs: a centre φ for the ghost-fluid free-surface condition
(§4.1, via Gibou et al. — "requires only liquid signed distance values at cell
centers"), a liquid fraction for `V_cell` in equations 3 and 4, and a crossing
per dual edge.

Everything below answers one of four questions about that collapse: what is in
there, how it got there, **what it handed the row**, and what it cost.

## The enabling change: records, not more scalars

The trace ABI went to **v2**. The load-bearing addition is not more counters but
`FLUID_CELL_TRACE_FINE_RECORD_CAPACITY` per-probe records — position, φ, flags,
seed cell, seed code and hop for each of a 4³ sub-lattice of the 8³ probe
lattice the statistics still come from. Line work needs positions, not totals,
and one addition unlocked four views at once.

Being a *regular sub-lattice of the same lattice* is what keeps the drawn sample
unbiased against the counted one. "The first 64 that resolved" would quietly
favour whichever corner the loop starts in.

Also added: per-neighbour φ on the stencil records (a crossing is a property of
the *pair*, so neither cell alone can name it), the row's coarse record, a
missing/stale/negative probe census, and interface flags on the ray-run hits.

---

## What landed

### The φ span bar — the orienting graphic

One axis, liquid blue to air orange, the leaf's own half-diagonal at each end.
Two spans are drawn rather than one because they can disagree, and their
disagreement is the finding:

- the **record** is what the pressure row will read;
- the **probes** are what the band actually holds.

`unresolvedInterface` is exactly their disagreement in the direction that costs
something — the probes straddle zero and the published record does not. That is
the paper's own §8 limitation made specific: "sub-grid droplets or air pockets
may at times be overlooked by the simulation." It renders as a failure-toned
warning, because when that artefact appears this names the cells responsible.

`interfaceOffCentre` is the milder neighbour: the surface is inside the cell but
not near its site, so second-order accuracy near the surface is resting entirely
on θ.

### The free-surface strip — why this row has this diagonal

One chip per dual edge the surface cuts, **sorted tightest first**, each carrying
θ and the coefficient scale `1/θ`. A crossing at 7% of the edge scales that
face's coefficient fourteenfold, and that is what explains a large diagonal.

The HUD had been printing `diagonal` and `rhs` as two bare floats with nothing to
interrogate them by. This is the view that turns them into a story.

Three deliberate refusals in the model:

- a neighbour with no published record contributes **nothing**, rather than being
  compared against a fabricated zero — that would put the surface exactly on a
  cell with no opinion about where it is;
- an **air** cell reports the crossing but no coefficient scale, because the
  condition is imposed from the liquid side and the scale belongs to the
  neighbour's row;
- an unpublished row has no crossings at all.

### Four new decorations

| Layer | Draws | Answers |
| --- | --- | --- |
| `surface` | a ring on the dual edge at the measured fraction, plus the liquid stub, graded to red as θ tightens | where the surface cuts, and how hard |
| `patch` | crosses at reconstructed closest points, each with a ring normal to the axis its crossing was found along | what the surface looks like inside this cell |
| `links` | each recorded sample joined to the closest point it inherited, coloured and swept by hop depth | where each distance came from |
| `gaps` | dashed brick boxes where no page was resident, or one was at another generation | where the band could not answer |

The fine three draw in the **fine lattice** via `DecorationBuilder.inSpace` —
that is what `DecorationSpace` exists for, and it is why a sample at fine
coordinate 34 at factor four lands at finest cell 8.625 rather than at 34.

Gaps are drawn at **brick** granularity because residency is a property of the
brick: claiming a hole the size of one sample where the real hole is sixty-four
would understate it by two orders of magnitude. Missing and stale get different
colours, because they are different faults with different fixes.

All four draw on hover, exactly as they draw pinned. They used to wait for a
pin, on the argument that the gather re-runs every frame; it does, but
`assembleDecorations` keys on the drawn facts, so sweeping the pointer across one
leaf rebuilds once. The saving was paid for by making the reader commit to a cell
before seeing anything about it, which is the opposite of what a picker is for.

### Band membership and ladder attribution

Two compact panels, side by side. The band nest puts all four widths on one axis
— the authored two arrive in finest cells and the derived two in fine cells, and
a reader should not have to do that conversion to see that transport reach sits
inside redistance support. The marker is the cell's own distance to the surface,
so "why is this cell refined at all" is answered by which bars it falls inside.

The ladder bar bins *this leaf's* probes against the ladder that actually ran.
The global report says roughly half the field closes on the first stride-8 pass;
that is a property of the domain, and this is the only way to know which side of
it a given cell falls on. Binning against a schedule that never ran would name
passes that do not exist, so no published ladder yields no histogram.

### Four narrative steps

`Read the corrected φ` · `Hold the liquid` · `Cover the leaf` · `Back one
unknown` · `Move and redistance`. The last is badged `scheduled` for the same
reason the row-update figure is, and the budget step is what makes the trade
concrete: one pressure unknown, revisited fifty-odd times by the solve, backed by
`(leafSize × m)³` samples the band moves and floods every step.

### Navigation

Selection by pixel names the nearest leaf, which on a liquid is a surface cell,
and `[`/`]` step inward blindly. The ray march now classifies each hit against
the coarse record, so:

- interface leaves are **picked out of the ray run** in the patch colour, giving
  the jump something visible to aim at;
- `≈` in the HUD and `i` on the keyboard jump to the next one, forward with wrap.

### Getting into the tool at all

Two fixes prompted by using it:

- **Pick mode is a toggle in the scene's top-right**, beside the frame rate,
  with a reticle glyph and the `C` shortcut. It was four scrolls into a collapsed
  section of the Render panel.
- **Clicking the viewport pins.** The HUD footnote had promised this since the
  picker landed and only the HUD button delivered it — the documented gesture did
  nothing. Pinning by click is the whole point: a cell you cannot hold still
  cannot be orbited, and orbiting is how the stencil and the patch become legible.

## The bug this found

`fine-flood-provenance.ts` restated the redistance commit's sample flags and put
`negative` on bit 2. The commit writes `NEGATIVE = 16u` — bit 2 is `known` in
`octree-fine-levelset-bricks.ts`'s layout. Nothing had read that flag until the
cell trace did, and the symptom was **every cell in the water reporting 0%
liquid**.

The flags now alias the brick layout rather than restating it, so the two cannot
disagree again, and a test asserts both the aliasing and that the gather's WGSL
interpolates the same values.

## The binding ceiling, spent exactly

The gather now holds **ten** storage buffers: `headers`, `metrics`, `pressure`,
`trace`, `fineWorklist`, `fineMetadata`, `fineFlags`, `fineSeeds`, `finePhi`,
`coarsePhi`. Ten is what browsers report on the Apple silicon this targets, and
`requiredFluidDeviceLimits` requests the adapter's value.

There is no slot left. `FLUID_CELL_TRACE_STORAGE_BINDINGS` names the list,
`initialize()` fails with that list rather than a driver message, and a test
asserts the count against `VISUALIZATION_STORAGE_BUFFERS_PER_STAGE`. An eleventh
publication needs a second gather pass, not another binding.

---

## What did not land, and why

**C2 — solid open fraction per face.** Needs `catalogFaces` in the gather, which
is the eleventh storage buffer. Blocked on the ceiling above; it needs a second
gather pass. The hidden `operator-open-fraction` field still renders it
volumetrically.

**B3/B4/B5 — the semi-Lagrangian departure trace, its interpolation branch, and
extrapolated-velocity provenance.** This was the flagship of the proposal and it
is the one thing deliberately *not* faked. The plan was to replay §5's m-substep
backtrace, and a replay is only honest if it samples the velocity the transport
actually sampled. That interpolant is not simple: `webgpu-octree-fine-levelset-transport.wgsl.ts`
does a weighted reconstruction over owner tags plus a barycentric path over the
Delaunay tetrahedra near T-junctions, needing `rowVelocity`, the air-support
arena, the owner directory, selectors and tetra geometry. Reproducing it with the
per-row velocity the gather could reach would draw a path the solver never took,
which is precisely what this codebase's conventions forbid.

Two routes remain, both real work:
1. a second gather pass carrying the transport's binding set; or
2. a trajectory tap inside the shipping transport pass, which would make the
   record genuinely `replayed` rather than re-derived — at the cost of editing a
   hot path.

**B6 — per-leaf page lifecycle.** Needs the transport topology delta keyed by
brick and intersected with the leaf; no publication exposes that today.

**E2 — single fine-sample pick.** A second selection level inside the picked
leaf. Cheap once B3 exists, and largely pointless before it.

## The evidence badge that is still unearned

The proposal argued for a third badge beside `gathered` and `scheduled`:

> **`replayed`** — the shipping law re-run against the published field, recorded
> in execution order.

The argument stands — a fine sample's advection *is* one sequential walk, unlike
a pressure row's work — but nothing in the tool earns that badge yet, so it has
not been added to the vocabulary. It arrives with B3 or not at all.

## Conventions carried

- Solid is measured, dashed is derived. `links` and the crossing rings are solid;
  `gaps` and the flood-reach envelope are dashed.
- Every figure declares `gathered` or `scheduled`, and the two are never summed.
- **Everything is a probe lattice, not a census.** 512 probes over up to 16.8M
  samples, with the line work drawn from a 64-probe subset of those. The HUD
  states both denominators rather than implying exactness.
