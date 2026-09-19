# Uniform Geometric: deep water solved at 4h, surface solved at h

Status: PLAN ONLY, 2026-09-19. Nothing here is implemented or measured. It is a design argued from the code
and from the tall-air A/B (`docs/research/uniform-geometric-tall-air-2026-09-19/ab-report.md`). Every number
marked *derived* is arithmetic from that report's fit, not a measurement of this idea.

Peter's question: can some regions (deep water) solve pressure at a 4h coarse level while others solve at fine
resolution, probably with a thin 2:1 seam so T-junctions do not cause artifacts, **without giving up the very
fast GPU-friendly dense approach**?

Short answer: yes, provided it is built as a **mask on the dense multigrid hierarchy we already run**, never as
an octree with T-junction stencils. It pays only on large liquid-heavy scenes. There is a no-risk first rung
that needs no seam at all.

## 1. The framing that keeps the GPU path

The CM11a V-cycle already owns a 4h level. With lockstep halving, level 2 is the 4h lattice, and **one 4h tile
(one 4³ workgroup at level 0, the unit every Uniform Geometric work map already classifies) is exactly one
level-2 cell**. So "this region solves at 4h" needs no new grid and no new stencil:

- A **deep tile** skips smoothing and residual work on levels 0 and 1 and keeps the prolonged level-2 answer.
- A **2h ring tile** skips level 0 only. The 2:1 graded seam falls out of the hierarchy for free.
- A **fine tile** runs every level, as now.

Same dense textures, same red-black passes, same launches, whole-workgroup early exit at level 0 (at level 1
a 4³ workgroup spans eight tiles, so the exit is per thread there). No pointer structure, no neighbour tables,
no variable stencils. Because no stencil changes anywhere, **a T-junction never exists as a discrete object**:
the seam is a place where the fine lattice stops being smoothed, and the 2h ring is an accuracy choice, not a
structural requirement. Start with a sharp seam placed deep; add the ring only if seam currents are measured.

This is the classical local-refinement multigrid family (Brandt's MLAT, McCormick's FAC, the Martin–Colella
adaptive projection). It is **not** the Losasso octree discretization, shares no kernels with it, and should
not borrow any ([[losasso-is-off-limits]]).

Tile classes after this plan. The first three exist today (`uvTwoLevelSeed` and the dilations):

| class | velocity | pressure | exists |
|---|---|---|---|
| far air | 4h face table | no rows | yes |
| fine (surface band, solids, sources) | fine | all levels | yes |
| shell | fine extension output | — | yes |
| **deep liquid** | fine storage, 4h content (§4.2) | levels ≥ 2 | new |
| **2h ring** (optional) | fine | levels ≥ 1 | new |

## 2. Why it is physically sound

Pressure in the liquid interior is harmonic apart from the divergence source. A surface feature of wavelength
λ decays with depth d as exp(−2πd/λ), so at a depth of two or three tiles nothing below ~4h wavelength is left
to resolve. The free surface, its second-order ghost-fluid boundary, every solid and every source stay in fine
tiles, so **the visible physics is untouched**. Deep tiles are all-liquid and solid-free by construction, so
the level-2 operator there is the plain seven-point Laplacian at 4h; none of the averaged ghost-fluid or
cut-cell coefficient baking is involved.

## 3. What it can buy, and when

From the tall-air fit, pressure costs `passes × 4.76 µs + workgroup-visits × 1.73 ns`.

- **At 64³ it buys almost nothing.** The launch floor is 85% of the small scene's pressure seam. Level-0
  arithmetic is about 1–2 ms of an 11.5 ms step (*derived*), and only the deep fraction of that is removable.
  A level-0 smoother tile map was already built and measured **slower** on mini64 and large-power
  (`docs/benchmarks/uniform-pressure-pass-floor-2026-09-19.md`): the classify barrier and atomic class reads
  cost more than the three-operation air path they skipped. Liquid cells are a better target than air cells
  (a real seven-tap update is skipped), but the ceiling is still small.
- **At 256³ with a deep pool it is the dominant term.** Level 0 has 64× the workgroups; its arithmetic alone
  is of order 75 ms/step (*derived*), and with 90% of liquid tiles deep nearly all of it goes. The setup
  pyramid (topology + RHS, ~10× a smoother sweep per cell) is also skippable in deep tiles.
- **It does not remove a single launch.** Pass count is untouched, so this is complementary to pass fusion
  and resident coarse levels, not a substitute. Rank it after them for every scene we ship today.
- **Memory** is unchanged unless deep tiles stop owning fine storage, which is the tile-resident program
  (`docs/research/uniform-geometric-empty-air.md`, rung 3), not this one.

Gate: do not build past rung 0 until a liquid-heavy lane at ≥128³ exists and shows level-0 + level-1 arithmetic
above ~25% of the step.

## 4. Risks and pitfalls

### 4.1 The stopping test breaks first
The residual tolerance gate (`mgCheckCycleConvergence`) and the lagged cycle budget read the fine residual.
In deep tiles the fine residual never converges: a trilinear-prolonged 4h field does not satisfy the fine
equations pointwise. Unfixed, the gate never fires and every step encodes and runs the full cycle budget —
a regression that would hide the saving. The residual measure must cover fine (and ring) tiles only, plus a
block-summed residual for deep tiles.

### 4.2 Deep fine velocity stops being divergence-free
CM11a uses the same trilinear operator for restriction and prolongation. The masked cycle's fixed point
satisfies the fine equations only in a restricted (weighted block-average) sense. After projection, deep fine
cells carry pointwise divergence, zero only on average over a tile. Consequences:

- Conservative volume transport sees a compressing/expanding field in the bulk, so V drifts from 1 in deep
  water: the same symptom family as the dam break's deep interior settling a third full. V is still conserved
  in total; it is mis-placed.
- Fine-scale divergence is never projected out while a tile stays deep. It is advected, and surfaces when the
  water does.

Fix: in deep tiles, **rebuild the fine face velocities from the tile's 4h face fluxes by linear interpolation
of the normal component across the tile**. That subdivision is exactly divergence-free whenever the 4h fluxes
balance, which the block-summed projection gives. Cost: sub-4h velocity detail in deep water is discarded
(higher numerical viscosity at depth, invisible unless a tracer or a field view looks there). This is the
liquid-side mirror of the two-level sampler that already serves far air from the 4h face table.

Exact block balance needs the 4h face flux to equal the sum of its sixteen fine face fluxes (refluxing, as in
Berger–Colella). With trilinear restriction the balance is smeared, not exact; check whether the residual
restriction can be made a pure 64-cell sum in deep tiles.

### 4.3 Popping when a tile changes class
When the surface drops and a deep tile becomes fine, any unprojected fine divergence is removed in one step:
a pressure spike that shows as a ripple at the surface above it. This is the adaptive-grid artifact this repo
has been bitten by before ([[blob-is-one-frame-topology-collapse]]). Mitigations, all needed: a thick fine
band (liquid depth ≥ 2–3 tiles below the lowest surface vertex plus the measured per-step reach), hysteresis
on the deep→fine transition, and §4.2's rebuild so there is nothing to pop.

### 4.4 Walls and the floor
`mgTrilinearPressure` clamps taps at the lattice edge. Hydrostatic pressure is linear through the floor, so a
clamped prolongation is wrong in the wall-adjacent fine cells of a deep tile by up to ~1.5h of head. The
symptom would be a slow persistent boil along the floor and up the walls of deep water — the coarse lane has
history here ([[coarse-lane-never-got-wall-fixes]], [[wall-sticking-mechanisms]]). Options: keep wall-adjacent
tiles fine (a 2D cost, like the surface band; the simple choice), extrapolate linearly at walls in the deep
prolongation, or solve deep tiles for dynamic pressure with the hydrostatic part removed.

### 4.5 The coarse operators become the answer
Levels ≥ 1 are rediscretized with baked, averaged coefficients, and today they only set the convergence rate.
In deep tiles level 2 is the solution. Restricting "deep" to tiles whose level-2 cell **and all 26 neighbours**
are full liquid with no solid keeps the operator the plain Laplacian. Known oddity to clear first: on
large-power, level 1 has zero liquid cells under the baked predicate
([[uniform-geometric-is-pass-count-bound]]). That would make every deep tile wrong; find out why before
trusting level 2 as an answer.

### 4.6 Separating boundary state
The CM11a minimum/shifted chain (`mgShiftMinimum`, `mgDownsampleMinimum`) assumes every level of a cell's
column is live. Deep tiles hold no solids, so they should be inert for it, but the downsample still reads
them. Audit, do not assume.

### 4.7 Hydrostatic rest and symmetry
Trilinear prolongation reproduces a linear field exactly away from walls, so a resting pool should stay still
with the seam deep (given §4.4). Tiles are 4h-aligned, so symmetric scenes stay symmetric when the domain is a
whole number of tiles about its mirror plane. Both are acceptance tests, not assumptions:
`hydrostatic-power-large-offset` must stay bit-still; the symmetry lanes must not move.

### 4.8 The class map is read a lot
The tall-air A/B found the two-level sampler's per-sample `atomicLoad` of the class map costs the vertex phi
seam ~38%. Pressure kernels must read the deep/ring class through a plain read-only binding, once per
workgroup where the kernel allows it, never per tap ([[atomic-load-is-an-fp-barrier]]).

## 5. The ladder

**Rung 0 — lazy deep smoothing (no seam, same answer).** Deep tiles run 2 pre + 2 post sweeps on levels 0
and 1 instead of 6 + 6; fine tiles keep 6 + 6. The six sweeps buy robustness at the free surface and at
solids, which are never deep. The discrete system and its fixed point are unchanged, so there is no seam, no
divergence issue, no popping and no wall issue; only the convergence rate can move. Removes about two thirds of
deep level-0/1 smoother arithmetic. Deliver as a live toggle on the pressure stage
(`Deep smoothing: Lazy | Full`, default Full), with a `Deep tiles` readout. Judge by executed cycles and the
residual series against the Full arm on a liquid-heavy lane: if cycle demand rises by more than the sweeps
saved, stop here.

**Rung 1 — deep classification and its instruments.** Add the deep bit to the tile classes (full-liquid
27-neighbourhood, no solid, no source, depth band + hysteresis), a field view for it, and the block residual.
No numerics change. Measure the deep fraction and its churn per step on real scenes. Rung 2 is not worth
building below roughly half of liquid tiles deep.

**Rung 2 — true 4h solve in deep tiles, sharp seam.** Skip levels 0–1 smoothing, residual and setup in deep
tiles; fine-only residual gate; wall-adjacent tiles forced fine; §4.2 velocity rebuild after projection.
Off arm unchanged. Acceptance: bit-still hydrostatic, symmetry lanes unmoved, volume conserved, no surface
ripple on a draining pool where tiles flip deep→fine every few steps, deep-interior mean V stays at 1.

**Rung 3 — the 2h ring,** only if rung 2 shows seam currents or reflections at the seam.

## 6. Open questions

1. Can restriction be a pure block sum in deep tiles so 4h flux balance is exact rather than smeared?
2. Is the level-1 "zero liquid cells" oddity on large-power a bake bug or a trivial problem?
3. Does semi-Lagrangian velocity advection re-create enough sub-4h divergence per step in deep tiles that the
   §4.2 rebuild must run every step rather than only on class change?
4. How thick must the fine band be under breaking waves, where surface depth changes by several tiles a step?
