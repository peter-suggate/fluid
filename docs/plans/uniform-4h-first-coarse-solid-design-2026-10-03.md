# Uniform Geometric: cut-cell 4h (H) owners — solid operator design

2026-10-03. Read-only study. No repository files were edited, and no Dawn, GPU or browser was run.

Inputs:
- docs/plans/uniform-4h-first-implementation-handoff-2026-10-03.md
- docs/plans/uniform-4h-first-storage-design-2026-10-03.md
- scratchpad/uniform-architecture-map.md
- docs/plans/uniform-coarse-first-2026-10-03.md (pond rejection: peak speed 0.020 → 3.25 m/s)

Paths below are relative to `lib/methods/uniform/` unless they are rooted.

## 0. Summary

1. **Cut H owners can work.** Today they are wrong at every stage, because each stage assumes "coarse ⇒ uncut".
   - The assumption is written down in several places: `uniform-mixed-solid.wgsl.ts:12-15`, `uniform-mixed-transport.wgsl.ts:96`, `uniform-mixed-surface-geometry.ts:128` and `uniform-mixed-sharpening.ts:114-118`.
   - Promotion makes it true: `uniform-mixed-layout.ts:97-176`, `uniform-mixed-dynamic.ts:382-397,800-822`.
   - The frame refuses violations: certificate bit 1 at `uniform-mixed-frame.ts:602`.
2. **The h discretization carries over to H almost unchanged,** provided two things hold.
   - (a) Every *volume* quantity uses the H open volume Ω_H. The record already has it: `UniformMixedSolid.coarse.x`, built at `uniform-mixed-solid.wgsl.ts:157-185`.
   - (b) Every *geometric* query (walks, buried tests, fill-target integration) keeps using the **h** solid mask.
   - The h mask stays a static asset at zero detail, per the storage design. `umCellOpen` at `uniform-mixed-solid.wgsl.ts:386-392` is valid for any world h cell, whatever the owner width.
   - So a cut H owner is "an H control volume whose open part is a union of h cells". Fluxes are H-face aggregates. Geometry is integrated at h.
3. **What changes is not cosmetic.** At h, solids are binary: SolidOccupancyMask closes any touched cell. That makes `target = fill × open` exact at h. At H, Ω is a staircase count/64, so `fraction × Ω` is wrong at every shoreline.
   - At rest, the fill target must be the exact staircase–plane intersection F_Ω (§2, W4). Otherwise the volume correction (half the excess per 1/30 s) pumps a resting pond.
   - This, together with the uncut flux weights in `umCoarseFlux` and the "empty" buried vertices that sit at the corners of cut H owners, is the most likely mechanism of the 0.020 → 3.25 m/s pond rejection.
4. **There is a known non-solid H defect.** The hose-x10 profile keeps its surface at h because "4h partial surface owners disturb hydrostatic balance in this shallow pond, even with the full pressure solve" (`/Users/petersuggate/code/me/fluid/lib/core/scenes.ts:2312-2320`). That defect must be isolated on a solid-free flat basin **before** any solid term is judged (Step 1).

## 1. Precise well-balanced condition for the existing discretization

**Setup.** At rest, uⁿ = 0. Forces add dt·g_y to every y face where `occupancy(owner) > 1e-5 ∨ occupancy(neighbour) > 1e-5 ∨ umCoarseMass` (`uniform-mixed-forces.ts:187`). Occupancy is clamp(0.5 − φ_c/(4·h_y·w)) (`:111-114`).

**Pressure solve.** It solves A p = b with:
- operator face weights V_f / (θ_f d_f);
- RHS b = −ρ(Σ_f V_f u*_f − c)/dt (`uniform-mixed-pressure-velocity.ts:182-207`);
- c = volume correction (`uniform-mixed-pressure-authority.ts:205-225,260-265`).

The projection is u = u* − (dt/ρ)·∇p, with ghost θ at open tops (`uniform-mixed-pressure-velocity.ts:234-250`). Faces with V ≤ 1e-6 return 0 (`:240`).

**Hydrostatic candidate.** Take p_o = ρ|g|(η − y_o) on liquid rows and p = 0 on air rows. With φ the planar SDF y − η:
- an interior liquid y face gives (p_up − p_o)/d = −ρ|g|;
- a liquid→air top face gives (0 − p_o)/(θ d), with θ d = η − y_o exactly. That also equals −ρ|g|.

So u_y = dt·g_y + (dt/ρ)·ρ|g| = 0 on every y face, for **any** V_f. Horizontal faces have equal p and no gravity, so u_x = u_z = 0.

The candidate is the solution, and the pond stays at rest exactly in f32, if and only if all four of these hold:

- **W1 — consistent weights.**
  - The RHS divergence uses the same V_f as the operator on every face of every row, including halo and seam faces. Then A·p_hydro = b.
  - Violated today for statically-cut H tiles: `umCoarseFlux` (`uniform-mixed-pressure-velocity.ts:164-167`) uses `fraction·u·16` with fraction 1/0.5, but the operator uses record V (`umProjectV`, `:219-224`).
  - `umCoarseCut` (`:143-147`) is `umSolidCut` = cut ∧ simulated-at-h (`uniform-mixed-solid.wgsl.ts:321-323`). So a cut tile that is simulated at H takes the uncut branch.
- **W2 — gravity on every operator face.**
  - Every face with V_f > 0 that borders a pressure-liquid row must receive dt·g_y.
  - This holds when occupancy and pressure-liquid classification derive from the **same** centre φ.
  - It breaks if the pressure φ at a cut owner comes from the h-vote (`uniform-mixed-pressure-authority.ts:161-183`) while occupancy uses centre φ.
  - Rule: at H, both use `umOwnerSurfacePhi`, the centre φ (`:186-189`).
- **W3 — the pressure φ is the planar SDF at the geometric H centre, including centres inside solid.**
  - Then classification depends on y only. That gives no horizontal liquid/air row pairs, and θ is exact.
  - It requires every corner vertex of a cut H owner to be *state*, holding the planar value.
  - Today `umBuried` (`uniform-mixed-surface.ts:157-161`) tests the 8 incident **h** cells. So a vertex inside solid at the corner of a cut H owner is buried, initialized "empty" by `uniformVolumeInitialPhi` (`uniform-volume-initial.ts:8-58`), and frozen by `umAdvected` (`:369-377`) and redistance (`:686,697`).
  - Centre φ is then biased positive. You get an air row mid-pond, a ghost face at the wrong height and a pressure kick. This alone can produce m/s speeds at a shoreline.
- **W4 — c = 0 at rest.**
  - The correction amount (`uvVolumeCorrectionAmountAt`), deficit, stranded and bulk terms all vanish when the fill target equals V exactly in every owner.
  - At H that requires target = F_Ω(φ), not fraction × Ω (§3.3). Initial V must use the same function.

Extension must then leave liquid-row faces untouched and write only air faces (from zero-velocity sources). That already holds: `umSource` at `uniform-mixed-extension.ts:155-158` uses phase.

**Detached mass is a W2 hazard.** An air H row with V·64 > 0.05 that does not touch liquid keeps dt·g (`uniform-mixed-detached-mass.wgsl.ts`, `uniform-mixed-pressure-velocity.ts:236-238`). At rest every such row touches the liquid row below it, so it is harmless, except for a thin film < H on a cut floor (T4).

## 2. Stage-by-stage: does the h helper hold at H?

Columns: helper (file:line) — at h — at H.

**Static solid queries**
- `umCellOpen` / `umCellInsideSolid` (`solid.wgsl:386-400`)
  - At h: binary per h cell.
  - At H: **valid unchanged.** Query h cells inside the H owner. The h mask is a static asset at zero detail.
- `umFaceOpen` (`solid.wgsl:454-469`), 4 transverse samples
  - At h: aperture.
  - At H: H aperture = mean over the 16 h faces of the slab. Needed by sharpening and cleanup only. Evaluate on the fly when the tile is cut.
- `umPressureFaceV` (`solid.wgsl:472-491`), dual-cell, 8 samples
  - At h: operator weight.
  - At H: **use record .yzw**, the slab mean of the 16 h V (`:157-185`). Do not form an H dual-cell V. The slab mean makes Σ_16 V_h·u = 16·V_H·u exact at seams (§2.6).
- `umSolidFaceVelocity` (`solid.wgsl:429-452`)
  - At h: body velocity at a face.
  - At H: V-weighted mean of the 16 h-face body velocities. Body-near tiles only.

### 2.1 Surface: advect, trace, redistance, buried

**`umSurfaceTrace` / `umWalk`** (`uniform-mixed-surface.ts`)
- The h half-cell walk stops at the first closed h cell.
- At H, keep the **h-step** walk against the h mask, sampling H velocities. An H-step walk would tunnel through h-thin walls.
- Cost: 4× more steps for H tracers, bounded by CFL·H/(h/2) ≈ 8 at CFL 1.
- Gate the walk on "tile cut", exactly as `umSolidClear` (`:155`) gates it now.

**`umBuried`** (`:157-161`)
- Change to: a vertex is buried ⇔ **every incident owner, at its own width, has capacity ≤ 1e-5.** For an h owner capacity = `umCellOpen`; for an H owner it is record .x.
- Corner vertices of cut H owners become state (W3).
- Data: `umOwnerAt` for the 8 incident cells, which is already used by `umSampleVertex`, plus record .x.

**`umEmbeddedAir` / `umEmbeddedContact`** (`:177-236`)
- These are the h contact-angle and air-pocket terms.
- At H they must use the same owner-width capacity test. Otherwise an H corner inside solid is pulled to "air" every frame.
- Recommendation: disable embedded contact for cut H owners. A resting pond needs no contact model. Document that the contact angle at H is the natural 90° of the extrapolated planar φ.

**Cubic taps** (`tapsClear` `:268`, buried-tap fallback `:278`)
- Use the new buried rule, unchanged otherwise.

**`umAdvectStore` solid gate** (`:419-433`) and the retirement evidence `solidClosed` / `solidClear` (`:556-571`)
- Replace the h-cell closure with the owner-width buried rule.

**Departure for cut owners** (`traceCells` traces `umTrace(centre)`)
- At H the geometric centre can be inside solid. The walk then stops at once and the departure sticks.
- Trace from the **open-volume centroid** c_Ω (new record data, §3), and apply the resulting displacement d = trace(c_Ω) − c_Ω to the **geometric** box.
- At u = 0, d = 0, so the mapping is the identity, which is required for rest.

**Redistance exclusions** (`:686,697`)
- Swap in the new buried rule.
- A planar field is a fixed point of the redistance PDE when no buried value enters a stencil.
- Buried vertices must be excluded from stencils, not merely left unwritten. Verify that the stencil reader honours `umBuried`. If it reads raw buried values, the "empty" ones corrupt the plane.

**Initial φ** (`uniform-volume-initial.ts:8-58`)
- Write the analytic SDF at every non-buried vertex under the new rule.
- Keep "empty" only for truly buried vertices.

### 2.2 Conservative transport

**`umRowCapacity`** (`transport.wgsl:96`)
- Today the coarse value is `w³` ("coarse rows are uncut (promotion certificate)").
- Change it to `w³·Ω_H` (record .x).

**Weights** (`:141`)
- Today grain 1 uses `min(select(1,open(origin),w==1), open(q))`.
- H receiver at grain 1: `min(Ω_r, open_h(q))`.
- Grain 4 (H donor q): add `× min(Ω_r, Ω_q)`. Today grain 4 has no solid factor at all.
- This is not exact at H, because the open overlap of two staircases is not their minimum. It is **conservative**, because rows are normalized to capacity and donors divide by their cell volume, and it is exact at rest (identity box).
- Exactness in motion near a cut is an accuracy question, not a stability one. Measure it with T3.
- **As landed (2026-10-04), two deviations from the weights above.**
  - Grain 4 onto a cut H tile weighs the **open part of the overlap, summed over that tile's h cells** (`tpOpenOverlap`: Σ overlap(box, cell)·`open_h(cell)`), not `overlap × min(Ω_r, Ω_q)`. The minimum is wrong for a staircase even at rest once the box is not the identity: a box over the dry part of a half-open tile still drew `Ω_q` of it. An uncut tile (Ω = 1) or a sealed one (Ω ≤ 1e-5) returns before the loop, so uncut scenes pay one record read per grain-4 edge.
  - An H receiver at grain 1 keeps the old form `open_h(q)` (no `min(Ω_r, ·)`): the receiver's Ω already enters through the row capacity the weights are normalized to.
  - Forcing grain 1 for every box that touches a cut tile was tried and dropped: 64 own edges per row lose the sole-sampler identity below, and V at rest drifted +1e-6 relative in 90 frames.

**`donorAt` capacity** (`:111-120`)
- Keep the cell volume `1<<(3·shift)`, which is 64 for H. Do **not** use 64·Ω. V is a fraction of the cell, so a donor's mass is 64·V.
- **As landed (2026-10-04), one deviation.** The capacity has two uses and they differ at a cut tile.
  - *Gather* keeps 64: a row takes `share × V_donor` with the share out of 64, as above.
  - *The division rounds' column target* is the donor's **open** volume, `64·Ω_q` (`tpDonorOpen`). The rounds scale a donor's samplers so their weights sum to the target; a cut donor's own row is normalized to `64·Ω_q`, so a target of 64 asked its samplers for more than the tile's rows can hold and the h rows of a seam beside a cut floor over-drew (T5 seam-floor arm fatal at frame 66 with 64, runs with the open target).
  - A donor's only sampler takes the capacity exactly (`tpShare`: weight == sum returns the capacity, no quotient). `64·Ω` is not a power of two, so `weight·capacity/sum` no longer returned the capacity and a row at rest lost V by rounding (1:3 slope: frame-1 correction 8.9e-7, where the lane asserts 0).

**`tpOverlap`** (`:190-193`) and the fallback (`:238-242`)
- Pass the receiver Ω through, as at h.

**Sealed preserve** (`:268-270`)
- Extend to Ω_H ≤ 1e-5 rows.

**Sources** (`:281`)
- For an H row, iterate the 64 h subcells with `open_h(q)`. Today it passes 1.0 when width ≠ 1.

**Departure clamp**
- d comes from the h-mask walk (§2.1), so it cannot exit through an h wall.
- The box may still overlap closed h cells. The min-open weight zeroes those donors.

### 2.3 Surface volume, geometry, sharpening, cleanup

**Fill target** (`uniform-volume.wgsl.ts:36-60`)
- `umSurfaceTarget` returns the plane–box fraction of the whole cell when the 8 quarter samples are planar to 1e-4. Otherwise it returns an 8-sample vote.
- **F_Ω for a cut H owner:** keep the same plane test. When planar, compute
  - F_Ω = (1/64)·Σ_{s ∈ open mask} geometricPlaneBoxFraction(∇, −(centre + ∇·o_s), vec3f(0.25)),
  - using `/Users/petersuggate/code/me/fluid/lib/core/geometric-plane-box.wgsl.ts:5`.
  - Skip subcells whose ±(|∇|·0.25·√3/2) bound puts them wholly inside or outside. About 16 of 64 cross the plane.
  - When the field is not planar: vote with the h-subcell trilinear signs over open subcells / 64.
- F_Ω is an **absolute** fill, at most Ω.

**`gcOwner`** (`uniform-mixed-surface-geometry.ts:96-140`)
- Line 128 stores `fraction × (w==1 ? open : 1)`. For width 4 this becomes F_Ω.
- The tile-level pressureTarget branch (`:129-137`) also becomes F_Ω over the tile mask. This gives one target convention: absolute.

**Pressure authority**
- `umDeficit` / `umFill` (`uniform-mixed-pressure-authority.ts:205-209,219`): drop `*umCapacity(o)` in the root (`this.coarse`), since targets are now absolute.
- Simulation `umCapacity` (`:119`) returns 1 for width ≠ 1. It becomes record .x for width 4.

**Surface volume** (`uniform-mixed-surface-volume.ts`)
- `umCapacity` (`:113`): use record .x.
- Seed (`:160,181`) and measure (`:370-391`): mass becomes 64·F_Ω, instead of fraction·w³·cap.
- Metric slopes (`:300-310`) need open incident cells and are width-1 only. Extend to "incident owners have capacity > 0.99999". Cut H owners then get no slope term, which is the same rule h applies to cut cells.

**Sharpening** (`uniform-mixed-sharpening.ts:114-118`)
- `umSharpenOpen` returns true for width ≠ 1. It becomes `Ω_H > 0.99999`.
- Face gates become record V > 0.99999. That skips cut owners and cut faces, exactly as at h.

**Cleanup** (`uniform-mixed-cleanup.ts:71`)
- Exempt cut H owners (Ω_H < 0.99999) from the orphan rule, as width-1 cut cells already are.

### 2.4 Momentum and forces

**`umUnitClosed`** (`uniform-mixed-forces.ts:107`)
- It is width-1 only. It becomes `capacity(o) ≤ 1e-5` at either width.

**Gravity** (`:187`)
- Unchanged, given W2.

**Capillarity** (gate `:188`) and curvature (skips closed neighbours, `:137`)
- Use the generalized closed test, so the curvature is one-sided at closed H neighbours.

**Viscosity**
- No solid term at h either. Unchanged.

**`umCoarseMass`** (`:117`)
- It uses V·w³. For cut owners keep V·64. It is a mass test, not a capacity test.

**Detached mass**
- A cut H air row with V > 0 that is supported (its lower face V_H,y < 1 or a closed neighbour below) must not keep free-fall velocity into the solid.
- Face V = 0 already returns 0 at `uniform-mixed-pressure-velocity.ts:240`.
- Partial faces need the rule: treat a supported cut owner as an air row with no detached exemption. Test it with T4.

### 2.5 Pressure and velocity

**Root** (`this.coarse`, `uniform-mixed-pressure-velocity.ts:143-167`)
- `umCoarseCut` becomes `umSolidStaticCut ∨ bodyNear`, so it is not limited to the h flag.
- For a cut H owner, `umCoarseFlux` becomes Σ_f [16·V_H,f·u_H,f + Σ_16 (open_inner,h − V_h,f)·u_s,h]. The body term is evaluated on the fly and only when `umBodyNear`. Static solids have u_s = 0, which leaves 16·V_H,f·u_H,f.
- This is `umCutFlux` (`:150-161`) with the u_h samples replaced by the single H face value. W1 then holds.

**Root topology / minimum** (`:201`)
- It uses record .x (`uniform-mixed-frame.ts:273-278`). Unchanged.
- A closed row (Ω ≤ 1e-5) gets p_min = 0.

**Simulation projection**
- `umProjectV` (`:219-224`) already reads the record for width 4. Confirm it does so for simulated-at-H owners and not only for root rows.

**Release** (`:262-267`)
- The root already tests record .x and V.
- Embedded-contact release (`:277-286`) is fine-only. Keep that: no embedded release for cut H. This is a documented limitation, affecting adhesion at H walls only.

**Moving bodies**
- Faces with V_H ≤ 1e-6 return 0 (`:240`) at both widths, so body velocity enters only through the RHS body term, as at h.

**Band** (`/Users/petersuggate/code/me/fluid/lib/methods/uniform/uniform-pressure-band.ts`)
- Rows (`:425-462`) already store `umPressureFaceV` per face.
- The Neumann RHS (`:541`) uses unweighted velocity. Multiply by the row's V (`rows[(15+f)·N]`) and skip faces with V ≤ 1e-6. A Neumann face with V < 1 then carries V_h·u_H.
- Drop certificate bit 1 (cut 4h liquid row) at `uniform-mixed-frame.ts:602` and `uniform-pressure-band.ts:411-416`. Keep bit 2 (cut Neumann face) only until the `:541` fix lands, then drop it.

### 2.6 Seams between h and H owners

**Flux consistency**
- The H side sees 16·V_H,f·u_H. The h side sees Σ_16 V_h,f·u_H, because the root face velocity is prolonged uniformly.
- These are equal because record V_H,f is defined as the slab mean of the h V (`solid.wgsl:157-185`). Keep that definition.

**Transport across seams**
- Grain-1 rows of h receivers over H donors use `min(open_h(r), Ω_q)`. Grain-4 rows of H receivers over h donors use `min(Ω_r, open_h(q))`. Both are conservative.

**Hydrostatics across a seam**
- The root is hydrostatic, so the seam face u = 0, so the band Neumann data is 0, so the band solution is hydrostatic. W1 must hold in both solves.

**As landed (2026-10-04): one deviation. A seam that crosses a free surface is still not at rest.**
- Deviation (surface, `uniform-mixed-surface.ts` search template): a rebuilt vertex takes `sign·|p−q| + phi(q)`, not `sign·|p−q|`. Newton stops at `|phi(q)| ≤ 5e-4 h`, and dropping that residual moved the vertex rows under a resting h surface against the surface row. With it an all-h surface slab rests (1.5e-5 m/s at frame 90, was 4.4e-4) and the pond lane's frame-90 speed falls from 0.025 to 2.8e-4 m/s. Without a Fine region the result is bit-identical (wide rebuilds are travel-gated).
- Cost: fig-9 within noise. dam64 +4.6 % over frames 13–36, all of it tile count: the surface after impact spans more tiles (interface tiles 508 → 630 at frame 31), the cost per h tile is unchanged.
- The argument above ("the root is hydrostatic, so the seam face u = 0") holds on frame 1 only. An open basin with one Fine box across the waterline (lane T5b, a `todo`) leaves rest in 120–180 frames. Measured chain:
  - **Pump.** With phi and V frozen the box gains 2.5e-6 m/s per frame, in through the tile-row seam faces below the surface and out through the box's free surface. It is the same at 4, 8 and 15 band cycles and at σ = 0, and absent at g = 0. The root sees 1–3e-5 1/s of leftover divergence in the band's tiles (the band's float32 floor, one sign) and answers it with seam inflow every frame; the band passes that inflow out through its own surface. A basin whose h tiles cover the whole surface has no second free surface to draw from and does not pump.
  - **Null mode.** A vertex's velocity is the mean of four surface faces and a cell's θ comes from the mean of eight vertices, so a face flux that alternates cell to cell moves no vertex and meets no restoring pressure. Every other part of the pumped flow is restored by the live surface; the alternating part accumulates as stripes normal to the nearest seam (±50 µm/s at frame 300 with the root correction off, ±100 µm/s at frame 90 with it on).
  - **Amplifier.** Transport carries V into the air row above the up columns (2e-3 of a cell by frame 150) while the phi surface stays within 1e-4 cells. The deficit correction answers a V that phi cannot see with a sink, and the root's correction on tiles the band also corrects is the strong path: off, the box holds the lane bounds for 300 frames (1.2e-4 m/s, still growing); on, it is fatal. Replacing it by the restriction of the band's own correction does not cure it (9.3e-4 m/s at frame 300, accelerating).
- Two candidate changes close phi feedback paths and are necessary, not sufficient (not landed):
  - The root's phi for an h tile is a least-squares plane through the tile's open h vertices inside the redistance band, and its target the mean of the 64 h fills. Today it is the trilinear cube of the eight 4-aligned corners: the deep corners lie outside the h band (advect-only: 0.024 cells stale by frame 150 with V frozen) and injection samples one phase of a vertex stripe. As written (64 `mat4x4f` of workgroup memory per job) it costs +0.4 ms on dam64 and +0.6 ms on fig-9; a separate pass with one lane per h tile is the cheap form.
  - The travel gate holds only the wide vertices that define the zero set (a face neighbour at their own stride has the other sign). Held as today, a still seam's wide vertices are never rebuilt, and the seam plane's hanging vertices interpolate values only advection has moved. No cost.
- Face-traced departure boxes do not address this: the V in the air row is real net inflow from the pump, and a conservative transport of any kind turns an alternating surface flux into alternating V. What is missing is a restoring force (or damping) for the alternating surface mode, or a root that does not answer the band's own residual.

**As landed (2026-10-05): two more deviations, and one rule tested and held.**
- Deviation (surface, travel gate, `umWideAdjacent`): the gate holds only a wide vertex that is a corner of a cut wide cell (one of its 26 neighbours at its own stride has the other sign), or one with a buried face neighbour. A still seam's other wide vertices are rebuilt. (Corrected 2026-10-06. As landed on 2026-10-05 it looked at the six face neighbours only, the same-sign corners of a cut cell were rebuilt and moved the zero set, and `uniform-coarse-redistance` was red: part 8, which also has the rule's cost.) A buried neighbour certifies nothing whatever sign its sentinel has: read by sign, the air sentinel held a shoreline's liquid vertices and let its air vertices search a field with that sentinel in it (5 cells of error one tile above a 1:1 shore). No cost (dam64 and fig-9 identical trajectories, the gate is not reached without a Fine region).
- Deviation (pressure band, `uniform-pressure-band.ts`): the band solves for the correction to its start. Each row stores its start pressure (one more row field), a `rebase` launch after `init` moves the start's differences into the right-hand side, and clamps, the wall halo, the projection and the presentation use start plus correction. Iterated as the total pressure, a float32 row holds its divergence to about 6 ulp(p)/h² only, one sign over a tile, and the root answered that with seam flux every frame: this was the pump's seed with the fields frozen (box speed 7.6e-5 → 1.9e-6 m/s at frame 30, band residual 2.3e-4 → 1.6e-8).
  - Cost: band solve pass +0.047 ± 0.005 ms (18 dam64 perturbation pairs; +0.5 % of the frame), fig-9 and the garden within ±0.4 %. Rows buffer +6.7 % (+4.8 % with solids). The launch cannot be folded into `init`: it reads the neighbours' starts.
  - Not enough alone: the open box still leaves rest (4.8e-4 m/s at frame 100, 0.04 at 150), and T5's floor box holds 200 frames (4e-4 m/s) and is fatal at 248 (before: fatal at 185). T5b stays a `todo`.
- Scatter check on the 2026-10-04 cost line: a 1e-6-cell change of the dam64 fill moves the frame-31 interface tiles 560 → 634 and the window's time by −2.6 % without the search-template change. Its +4.6 % was trajectory, not cost.
- Tested and held (not landed): **on a tile the band solves, the root row is a coarse model of the band problem, in its correction and in its wall.**
  - Correction: the root row takes the mean of the tile's h rows' corrections (carried in the transfer's x anchor) and not the pressure authority's own, which is the 4h mean V against a 4h target. With the correction-form band this holds the open box for 300 frames (3.3e-5 m/s at 150, 2.4e-4 at 300, still an exponential: e-fold about 87 frames, blown up by 600).
  - Wall: the root's halo on such a tile is never held (`umWallHeld`), as the band's is not. Without this the carried deficit sinks drive a held root wall to −1400..−4100 Pa up a wall column whose band answer is 0..+400 Pa, and four band cycles do not close a start that far off (128³ dam, Full: band residual above 50 for 17–33 frames in 4 of 8 perturbed runs, +10 % frame time inside; 0 of 8 with the wall rule, and 0 of 8 at HEAD).
  - Why held: it brings the floor pockets of 2026-10-01 back. `coarse-first-pool-impact` at 6 s (dt 1/60, declared defaults): floor h tiles 132 → 500, closed air vertices 1.9e3 → 1.3e4, liquid deficit 1.13e4 → 1.8e4 cells with the carried correction, with or without the wall rule and with or without the correction form; the correction form alone is unchanged (108, 1.8e3, 1.16e4). With all three parts the fig-9 dam front also frays (front sd 0.09 → 0.61 cells at frame 30; the ball is unchanged).
  - Reading, not yet measured directly: the pressure authority's own correction on a band tile is also what closes voids smaller than 4h. A void's h cells are air rows with no correction and the liquid rows around it are full, so the band asks for nothing; the 4h owner is liquid with a mean V under its target and asks for inflow.
  - If that reading holds, the pump at rest and the void repair in motion are the same flux, and the consistent rule has to keep the part of the root's deficit the band has no row for (the tile's non-liquid h cells under a liquid 4h owner) and replace only the part the band's rows already answer. Not built.
- What still grows at rest, measured (box x 56..88, z 32..64 in the open basin, dumps on z = 48, dt 1/30):
  - With the held candidate (correction form and carried correction), the alternating surface mode is fed at the seam. Surface-face u_y alternates ±10 µm/s at frame 150 and ±65 µm/s at frame 300 beside the seam, and falls to about a quarter of that at the box centre, sixteen cells in. V over the fill in the air row is 7e-4 then 9e-3 of a cell beside the seam against 2e-4 then 4e-4 at the box centre. The vertex height stripe is ±15e-6 then ±200e-6 cells and is the same over the whole box. Speed 3.3e-5 m/s at 150, 2.6e-4 at 300, blown up by 450 (e-fold about 75 frames).
  - An all-h slab whose seam does not cross the surface has no stripes in 600 frames. With the candidate it holds 1.7e-5 m/s, flat from frame 500, with V over the fill uniform and linear (1e-3 of a cell at 600). On the landed tree it shows only the smooth pump: 7.5e-5 m/s at 300, 2.9e-3 at 600, uniform upwelling of 70 µm/s and V over the fill 1.6e-2 of a cell at 600.
  - So the alternating mode is a defect of the seam where it crosses the surface, not of the h surface scheme.

**Round 5 (2026-10-05): nothing more landed at the seam; three feedback paths measured, none of the seam-local rules rests the box.**
Same box, max speed in m/s at frame 300 unless said, dt 1/30. Waterline w is the surface height in h cells (w20 on a tile boundary, w23 three liquid h rows in the surface tile).
- Split correction (held, not landed): on a banded tile the root keeps its deficit for the non-liquid h cells under a liquid 4h owner and takes the mean of the band's corrections for the rest.
  - Motion passes: `coarse-first-pool-impact` floor h tiles at 6 s 8–32 against 0–208 live, liquid deficit 6.5e3–7.1e3 against 1.1e4–1.2e4; 128³ Full, eight perturbations, 0 of 8 fail.
  - It moves the fig-9 front (spread at step 30 0.09 → 0.30) and does not rest the box: w20 3.2e-5, w21 2.4e-4, w22 2.3e-4, w23 4.2e-3.
  - Growth is per second, not per frame: the same amplitude at the same simulated time at dt 1/60 (e-fold about 2.2 s). The mode does not limit dt.
- Loop 1, the root's phi on a banded tile. The eight 4-aligned corners alias a 2h stripe into a level shift. A least-squares plane through the tile's h vertices within 3.5 h of the surface removes it: w20 1.2e-5, w21 1.1e-4, w22 3.9e-5, w23 1.2e-4 (35× at w23). Still an exponential at w21 and w23. Cost would sit in the region interior, per banded surface tile.
- Loop 2, the corrections on V the surface cannot see. With the plane fit:
  - h deficits not applied on banded tiles (root keeps the void part): w21 4.1e-5, w23 1.06e-3.
  - Root takes only the excess of the band's corrections: blown by frame 150 at every waterline (the root then pumps out and never in).
  - Surface-deficit balancing off everywhere: w21 2.0e-5, w23 5.7e-4. All corrections off: w21 3.9e-5, w23 2.7e-4.
  - So no rule that drops corrections where the band answers rests w23; the fit alone is the best arm there.
- Loop 3, the root's seam flux. Root given the authored rest plane and all corrections off, w23: 3.5e-4, e-fold about 75 frames.
  - No single stage carries it: redistance off 2.3e-4, sharpening off 1.75e-4, shift off 3.3e-4, cubic off 1.8e-4, band at 8 cycles 4.7e-4; the kept residual off 3.1e-3 (it restrains).
  - Not the hanging vertices: every vertex outside the box within 5 cells of the surface pinned to the rest plane, 2.2e-4, still an exponential. The band's Neumann data is the same frame's root flux, not the previous one.
  - Every Neumann face of the band carrying zero: 8.2e-6, bounded, fastest cells in the bulk. Only the surface tile row's seam faces zero 4.75e-5; only the deeper rows' 5.5e-5.
  - So with phi and corrections out of the root, what feeds the mode is the root's flux through the seam in answer to the box's own velocity: the root projects the restricted h velocity with a 4h surface row and the band takes the result as fixed data.
  - With corrections on the sealed box still grows (1.1e-4 to 1.3e-4): loop 2 needs no seam flux.
- The mode itself, from per-frame dumps: surface-face u_y alternates by column (−14, +20, −16, +21 µm/s beside the seam at frame 140), closed by u_x in the top liquid row, steady over ten frames. Phi moved 9e-5 cells in 150 frames where those faces would have moved it 4e-3. V is lifted in the up columns only, so excess grows linearly.
- Reading. The alternating mode has no restoring force, so it coasts, and each of the three paths alone turns it into an exponential. Removing one (the fit) buys time, not rest. A rule at the seam has to remove all three, or the mode needs a restoring force or damping; the restoring-force change stays parked (it touches every free surface).
- The first non-uniform quantity at rest is the root's seam flux at its float floor (−20 nm/s through the tile row under the surface, taken up by the first h column). The root in correction form would zero it for about one launch and one 4h field; not built, and an unstable mode with no seed is still unstable.
- Band launches (landed, `uniform-pressure-band.ts`): no band launch is encoded when the layout has no h tile (0.33 ms and 95 launches saved at zero detail), and a band of at most 8 buffered slots runs each level's consecutive half sweeps as one 256-lane launch (35 launches for 95: one band tile 0.88 → 0.54 ms, four 1.07 → 0.59). Larger bands and a census layout are unchanged, bitwise.

**Round 6 (2026-10-05): nothing landed; damping and a restoring term are refuted, the residue is traced to float quantisation of the phi trace, and HEAD does not rest the box either.**
Same box, dt 1/30 unless said, max speed in m/s at frames 50 / 150 / 300 / 600. "Split + fit" is the round-5 split correction with the plane-fitted root phi; an arm rests only if it is flat or decaying at 600.
- References. All-h slab over the same rows: 4.3e-6 / 9.8e-6 / 1.8e-5 / 1.8e-5 (flat). All-4h pool: 3.0e-6 (w21) and 6.7e-6 (w23) at 600. Split + fit box: w23 3.5e-6 / 2.0e-5 / 1.0e-4 / 1.9e-2, w21 5.6e-6 / 1.6e-5 / 1.4e-4 / 1.73.
- Damping the alternating mode (binomial filter of surface-face velocity before the pressure right-hand side) does not rest it. Within four tiles of the seam: w23 3.7e-2 at 600, w21 1.8. On every h surface face: w23 5.4e-3, w21 1.54. The projection puts the stripe back in the same frame (−1.26, 0.13, −1.44, 0.84 µm/s by column after a full-strength filter), so the mode is driven, not coasting; the round-5 reading is withdrawn.
- A restoring term (theta raised by the V an air cell holds above its target) is unstable: w23 1.3 at 150, w21 1.39 at 300, and it blows the all-h slab (0.97 at 300). With the filter the slab rests (2.2e-5) and the box does not.
- Sealed box (every Neumann face of the band zero), the upper bound of any change to the root's rows:
  - corrections on: w23 8.2e-6 / 1.8e-5 / 8.4e-5 / 1.7e-3, w21 0.33 at 600. Filtering the correction right-hand side or the velocity does not change it (1.4e-3 to 1.6e-3).
  - corrections off: w23 4.2e-6 / 6.2e-6 / 1.0e-5 / 2.5e-5, w21 6.8e-5 at 600. Slow, and not flat. No single stage shut off makes it flat (sharpening, mixed-width sharpening, trace, redistance, total volume, drain, cubic: 1.3e-5 to 5.7e-5 at 600).
  - At w21 and frame 600: neither feed 6.8e-5, seam flux alone 3.8e-4, corrections alone 0.33, both 1.73.
- A Galerkin root row on banded tiles was not built. Piecewise-constant aggregates are four times as stiff between tiles as the root's finite-difference rows, which is an O(ρ g depth) pressure jump at the seam at rest; scaled back it is the finite-difference row with the surface at the tile-centre distance, which is the plane fit. The sealed arm bounds it from above.
- Which term drives the stripe (band rows dumped at frame 144, top liquid row): the divergence of the forced velocity is 0.004–0.06 µm/s and the body force is uniform to two ulps; theta differs by column by 1e-6 to 6e-6 in every box arm (0.3–1.7 µm/s of unbalanced force per frame) and is uniform in the slab. Theta, from cell-centre phi, carries it.
- What moves phi first is the trace's arithmetic. The departure point is formed in absolute lattice coordinates (`end = p − dt·u/h`) and the interpolation weight is `p − floor(p)`, so a vertex at height 16–32 cells can only move in steps of 2⁻¹⁹ cells (1.907e-6). At dt 1/30 and h 12.5 mm that is 0.715 µm/s: below half of it a vertex does not move at all, above it a whole step, and one step of theta is 0.62 µm/s of force per frame, itself above the threshold.
  - Seen in the dumps: every surface vertex holds exactly the rest value until its first move, and every later value is a multiple of 2⁻¹⁹ cells.
  - The slab crosses the threshold between frames 143 and 300, the sealed box between 40 and 80, the unsealed box between 10 and 40; afterwards all three carry the same unstructured noise (rms 3e-6 to 1e-5 cells). The seam only supplies the first 0.36 µm/s sooner.
  - It applies to every free surface, the step doubles with each octave of height, and it is why no rest arm goes flat. Sampling relative to the vertex (the offset kept apart from the integer base) would keep sub-step motion at no cost; not tested.
- Column parity at the seam. With corrections off no parity mode stands out of the noise in vertex phi or surface velocity at any frame (amplitudes 0.2e-6 to 3e-6 cells against an rms of 4e-6 to 10e-6). With corrections on a 2h mode grows in both horizontal directions and is as large four cells inside the box as at the seam (sealed: 42e-6 and 43e-6 cells at frame 300 against 71e-6 rms). Hanging vertices stay the interpolant of their 4h neighbours. V in the air cells of the seam column is ten times the interior's by frame 300, in whole-tile blocks, not by parity. So the stripe is made by the corrections acting on V the surface cannot see (loop 2), not by anything that differs by column beside a 4h neighbour.
- HEAD (114ea4a7) on the same pond, the domain a Coarse region with the box Fine: w23 9.2e-4 / 1.8e-2 / 8.0e-2 / 0.61; w21 0.31 at frame 1. HEAD's all-4h pool with no box creeps at w23 (surface 0.30 mm at 600 against 0.002 mm here) and blows at w21 by frame 150. The defect predates the program; the old lanes never held a Fine box in still water.
- Live tree, first frame with max speed above 1 cm/s and above 10 cm/s (sampled every ten frames):

  | waterline | dt 1/30 | dt 1/60 |
  |---|---|---|
  | w20 | 160 / 190 (5.3 / 6.3 s) | 230 / 280 (3.8 / 4.7 s) |
  | w21 | 160 / 160 (5.3 s) | 210 / 210 (3.5 s) |
  | w22 | 90 / 90 (3.0 s) | 140 / 140 (2.3 s) |
  | w23 | 130 / 180 (4.3 / 6.0 s) | 180 / 250 (3.0 / 4.2 s) |
  | floor box, w21 | fatal at 219 (7.3 s), pressure not converged, at 8.7e-4 | 250 / 250 (4.2 s) |

  Against HEAD at w23 the live tree is ten times quieter to frame 50, reaches 1 cm/s at the same frame (130 against 140) and then grows faster (10 cm/s at 180 against 260).

**Round 7 (2026-10-05): the band start carries the 4h pressure by its own vertical difference where only closed owners are missing (landed). The displacement trace (step A) was landed, measured and pulled the same day: it returns with a loop 2 fix.**

- Why step A is out. It is the right trace, but its only present effect is to remove the stiction that hid loop 2: the box table is unchanged, the hero pond reaches 1 cm/s near frame 100 instead of 180, dam64's accepted residual is a little worse and the garden pays two more root cycles in 24 frames. It is kept as a diff against the tree (`ws3/stepA.diff` in the session scratchpad, also `trace_phi.py`). Before it returns it needs: a loop 2 fix; the `umStaleWide` exact tie decided on purpose; and, for the departure texture (step B), `tpShare`'s rounding twin. The names below (`umSurfaceDisplacement`, `umVertexCellAt`, `umTraceDisplacement`) are the diff's, not the tree's.

- Correction to round 6's naming: the quantised functions on this path were `umSurfaceTrace`, `umVertexCell` and `umCubicPhi` in `uniform-mixed-surface.ts` (the native `uvTrace` / `uvPhi` are compiled for the dense reference only).
- The trace (step A, not in the tree). `umSurfaceDisplacement(p)` returns `d = clamp(−dt·u(mid)/h, −p, D − p)` and never adds it to `p` (the midpoint only selects the velocity sample, whose error is the coordinate ulp times the velocity gradient). `umWalk(p, d)` walks the displacement. `umVertexCellAt(base, offset)` takes the cell from `base + floor(offset)` and both trilinear weights from the offset and the cell's vertices relative to the base (small exact integers); `umCubicPhi(cell)` takes its Catmull-Rom weights from the smaller fraction. `umRecordTravel`, the wall and embedded contact paths and the cut departure follow. Same loads, no branch.
  - First vertex move: frame 2 everywhere (smallest moves 0.7e-9 to 1.4e-8 cells) against frames 5 to 600 in steps of 1.907e-6 before.
  - The all-4h pool is flat under 1e-6 m/s for 600 frames at w20 to w23 and both time steps (before: creep to 4e-6). The sealed corrections-off box is two to three times quieter but still creeps (1e-5 at 600).
  - Timing: nil on dam64, fig-9, pool-impact and garden ×10 (advect, deferred advect and traceCells within ±0.008 ms).
- Not converted, with the reason. The cell-centre departure texture still stores `centre + d`: stored as a displacement (built, `trace_v.py`, not landed) transport resolves overlaps far below the coordinate ulp, and that exposes a value-space twin in `tpShare` (`weight·capacity/sum`): a donor's own share `1 − δ` rounds to the whole for `δ < 2⁻²⁵` while the receivers still take theirs, so a resting 4h pool gains 1e-8 of a surface tile's fill per frame (+0.78 cell volumes in 600 frames). The correct form is the donor's own share as capacity minus the others', from the exact fixed-point sum decode already holds; until then the absolute departure keeps transport's contract (a row at rest keeps V bit for bit). Momentum departure and velocity samplers: error 2e-12 m/s at rest. Redistance foot point and plane fits: tile-local regressors. Hanging, seam and restriction weights: integer differences.
- The box on the precise trace (outcome 2 of three): every corrections-on arm grows at the old rate. The quantum was the seed for the first tens of frames, not the gain. Split + fit box at 600 frames, w20 to w23: 4.7e-5 / 1.6 / 2.9e-4 / 1.9e-2 at dt 1/30; 1e-5 to 1.6e-4 at 1/60 (10 s), at or below the 1/30 values at frame 300: growth is per second, not per frame.
- Loop 2's gain, sealed box with corrections on (2h amplitude of vertex phi, rms over the seam window): 0.24e-6 cells at frame 40, 14e-6 at 100, 80e-6 at 280, 1172e-6 at 580; 1.05 to 1.13 per frame to frame 80, then 1.012 (e-fold 81 frames). A coherent 2h checkerboard in both horizontal directions.
  - The carrier is the excess expansion (`uvVolumeCorrectionAmountAt` on V above capacity): with surface-deficit balancing off it alone grows faster early (1.116 per frame, 98e-6 at frame 100) and at 1.009 late. Balancing restrains the first 100 frames forty-fold; total surface volume only fixes the phase. Corrections off: no parity mode.
  - At frame 300 the hidden air-row V (825e-6 mean) sits in the even columns and the top-row deficit (126e-6) in the odd ones: they are not the same columns.
  - Fix directions tried as patch arms: a dead band on excess and deficit at 2e-6 cell volumes (the old trace quantum): no effect, the amounts are 10 to 100e-6 by frame 40. At 1e-4 (fifty times any representable step): a 250-frame delay of the same exponential, nothing on the unsealed box. Corrections reading V with the air owner's hidden V credited below: worse (the credit is excess; blown by frame 60 at w21).
- What step A exposed: promotion over a sloped floor was fatal for any pond that had ever moved.
  - The lane's slope pond was bit-frozen under the old trace, so its 4h vertices kept the authored plane to all depths. `umStaleWide` re-searches a 4h vertex past the band beside one inside it, and a miss keeps the band value; the band-edge row holds exactly the band value, so `< band` is an exact tie that sub-quantum motion breaks, and the deep 4h vertices saturate at −16 cells one after another from frame 3. That is the normal state of a field that has moved.
  - With the deep field saturated, turning solid contact on kicked 14 to 30 times harder (0.8 to 1.7 mm/s against 0.055) and the next root solve failed its planned cycles. Reproduced on the old trace by patching the test to `<= band`: fatal at frame 7, 32 and 152 for switch frames 6, 30 and 150.
  - Cause: the band start carried the liquid-weighted 4h pressure to a row's depth by the ratio of the row's phi to the owners' wherever a closed or air owner was in the stencil. That is a depth carry only while phi is the depth; past the redistance band the ratio is one and the promoted rows over the slope start up to 2 ρ g h off, which four cycles do not remove.
- The band start now carries by cause (`bCoarseCarry` in `uniform-pressure-band.ts`; landed, and the tree's state is the "carry" rows below).
  - An air owner in the row's 4h stencil: the phi ratio, as before. Under a free surface phi is the depth, and the ratio takes the start to zero at the surface whatever the flow.
  - Only closed owners missing: the 4h pressure's own vertical difference. Per stencil column, the two owners' difference where both are liquid; where one is, that owner against the liquid owner beyond it; weighted as the liquid owners weigh in the mean. At rest that is ρ g dy, in free fall zero, otherwise whatever the root solve found. The result is kept to the mean's sign and to the range the ratio had (0 to 1/θ_min of the mean).
  - Neither an air owner nor any vertical liquid neighbour (a closed duct one 4h owner thick): the phi ratio, as before.
  - Rows with every owner liquid, and every row with an air owner, are bit for bit what they were. On the slope pond at the switch: 93 888 liquid band rows, 29 088 missing an owner, 18 527 changed (median 18.5 Pa, 0.15 cells of head; largest 152 Pa), none outside the marked set.
  - Rejected forms. The vertical difference on every row missing an owner (air included): at rest as good, but on the dam's splash front the extrapolation above a steep impact gradient started rows at −200 Pa and the band residual spiked to 14.8 (4.6 to 6.9 otherwise). Plain ρ g dy: exact at rest, wrong in free fall. Zero carry on the rows with no vertical difference: kick 0.29 mm/s, band residual 0.0255, fatal at frame 32. A tie margin in `umStaleWide`: restores the lane by keeping the pristine field and nothing else.
  - Cost term: boundary. Only band rows missing an owner from their 4h stencil pay, in the one init launch: eight owner re-reads to tell air from closed, and up to four new owner reads on the closed-only rows.
  - Promotion over the slope (lane arm; kick in m/s and band residual at the switch frame, then max speed 34 frames later), switch frames 6 / 30 / 150:

  | tree | kick | band residual | 34 frames later |
  |---|---|---|---|
  | before, pristine field | 5.5e-5 | 1.3e-3 | 1.6e-3 |
  | before, saturated field | 1.7e-3 / 7.8e-4 / 7.8e-4 | 2.4e-2 / 4.2e-3 / 4.2e-3 | fatal at 8 / fatal at 32 / 0.33 |
  | carry, pristine or saturated | 1.3e-6 | 2e-6 to 4e-6 | 7e-6 to 8e-6 |
  | carry + step A (pulled) | 3.4e-6 / 1.6e-5 / 3.5e-5 | 3.3e-6 | 2.5e-4 to 2.9e-4 |

    Step A's figures are the open shoreline (the h path's defect the lane names under T5), which the old trace's quantum held still for those frames.
  - Falling arms (frames 13–36 of dam64, fig-9, pool-impact): none has a closed owner in a band stencil, so the carry alone leaves their pressure receipts identical to every printed digit. With step A the trajectories differ: executed root cycles 1 in all; accepted residual mean 0.97 → 1.09 (dam64), 0.494 → 0.480 (fig-9), 0.200 → 0.200 (pool); band residual mean 1.63 → 1.20, 0.90 → 0.83, 0.339 → 0.341. Garden ×10: two more executed root cycles in 24 frames (2.42 → 2.50), which is its +1.8% (0.43 ms in the root V-cycle pass); the init launch itself reads 0.219 ms before and after.
- What neither change does.
  - The hero pond with the hose off does not rest in any tree: 1 to 3 cm/s by frame 300 before and after. With step A the pond reaches 1 cm/s near frame 100, the tree before near frame 180 (carry alone, the tree's state, 220; step A alone 150): the seed is earlier, the growth is the same loop. The lane's 90 frames hold in all (shoreline 0.006 mm against 0.5).
  - The Fine box in a still 4h pond, same tree before / after, first sampled frame above 1 cm/s: dt 1/30 w20–w23 180, 130, 110, 140 / 180, 130, 120, 150; dt 1/60 260, 120, 200, 220 / 350, 90, 290, 260. The floor box no longer ends in a pressure fatal in either (1 cm/s at frame 260); that came with the spare Full-Cycle slot and the start-up half kick, not from here.

### 2.7 Extension, remap and promotion

**Extension** (`uniform-mixed-extension.ts`)
- No solid term. Sources are liquid-phase owners (`:155-158`), and solid faces of liquid rows are already 0.
- Unchanged, but test that extension never overwrites a liquid-row face of a cut H owner (T1 speed check).

**Remap fine→coarse velocity**
- `tFineToPressure` / `tMean` (`uniform-mixed-remap.ts:331-368`) and `tCoarseToPressure` (`:296-329`) take the plain mean of the 16 h faces.
- Use the **V-weighted** mean instead: u_H = Σ V_h·u_h / Σ V_h, and 0 when Σ V_h = 0. Then the H flux equals the h flux, and a divergence-free h field coarsens to a divergence-free H field.
- Today every retire of a cut tile injects a divergence that the next solve converts into a kick.

**Remap coarse→fine V** (`remapTileCells`, `:694-752`)
- Split fills follow the target with no open cap. The overfull branch is at `:727-728`.
- Child h fill becomes F_h(φ)·open_h. The excess goes to children with room open_h − V_h, which is the capacity-aware redistribution. Σ children = 64·V_H stays exact.

**Remap coarse→fine velocity**
- Uniform prolongation of u_H to the 16 h faces keeps the face flux.

**Promotion becomes optional**
- `uniformMixedSolidTiles`, `uniformMixedLiquidSolidPromotion` and `assertUniformMixedSolidPromotion` (`uniform-mixed-layout.ts:97-176`), census `solidActive` / `solidPromote` (`uniform-mixed-dynamic.ts:800-822`) and the impact rule (`:382-397`) move into the detail policy (`uniform-detail-policy.ts`) as optional *requests*.
- Zero detail must be valid.
- Delete the assertion and certificate bit 1. This follows the fail-fast rule: no fallback that re-promotes on failure.

### 2.8 Live voxel edits (displacement)

**Scatter** (`uniform-mixed-solid-displacement.ts:43-76`) returns for width ≠ 1 (`:45`). Generalize it:
- Excess = 64·max(0, V − Ω_H′), in volume units, after the record rebuild.
- Scatter onto H-stride axial shells, weighted by capacity·w³ of the receivers. h receivers keep weight `umCellOpen`.
- Resolve already divides by w³.

**Order per edit:**
1. Rebuild the record (`invalidate`, which exists).
2. Displacement scatter and resolve.
3. Re-seed φ in newly opened space as air.
4. Redistance.

Newly opened space has V = 0. Newly closed vertices become buried by the new rule.

### 2.9 Rigid bodies

**`umBodyOwnerVelocity`** (`uniform-mixed-bodies.ts:67`)
- It reads `umOrigin(o)`. That is wrong for H owners, whose faces live at anchors.
- Fix: average the 6 face velocities of the owner via `umFace`. This is needed regardless of cut cells.

**`couple`** (`:81`, width-1 only)
- At H, loop the 64 h subcells.
- Per subcell: fraction from trilinear H φ, wet = open_body-free·fraction.
- Drag reaction ρ·h³·wet·(u_s − u)·blend, plus torque, accumulated in the same words. Buoyancy is the displaced wet volume Σ h³·wet_body.
- Gate on `umBodyNear(tile)`. Body-near H tiles evaluate Ω, V and the body term on the fly, because the record is static.

**Policy**
- Bodies may still *request* h detail via `markTiles`.
- The H coupling is required for zero-detail mode, and for the case where the request is refused by budget.

## 3. Data: what the record needs and where it is computed

The record today (`uniform-mixed-solid.wgsl.ts:94-101,157-185`) holds, per tile:
- vec4 (Ω, V+x, V+y, V+z), as slab means of h `umPressureFaceV`;
- halo wall slots;
- vec4 flags (x = static cut, y = simulated at h; written by `encodeSimulation`, `:259-272`).

Add per tile, built in the same kernel from the same 64 `umCellOpen` loads:
1. **open mask**, 2×u32, 64 bits. Used by F_Ω (§2.3), the transport subcell sources and the remap split. It avoids 64 texture loads per use.
2. **open centroid**, vec3 offset in cell units. Used by the cut-owner departure (§2.1). Pack it with mask popcount = 64·Ω as a check.
3. **No extra face data.** The −x/−y/−z faces are the neighbour's + faces. Domain-wall halos exist already.

About 24 B more per tile. That is 64× smaller than h-cell data. Rebuild it on `invalidate`, the live-edit path.

Body-near tiles: compute (Ω, V, mask, centroid) on the fly from `umCellOpen`, which includes bodies, wherever the record is read. Verify that `umCellOpen` includes body occupancy at the call sites that read the record; today the record is static-only.

**Data flow per frame (H cut owner):**
1. record (static)
2. `gcOwner`, giving F_Ω target and centre φ
3. authority, with absolute fill, capacity Ω and deficit
4. forces, with occupancy from centre φ and closed = Ω ≤ 1e-5
5. root RHS, with record V and the body term
6. band, with Neumann V·u
7. project, with record V and ghost θ
8. extension
9. transport, with capacity w³Ω, min-open weights and centroid departure
10. surface advect, with owner-width buried and the h-mask walk
11. redistance
12. surface volume (64·F_Ω)

Remap and displacement run at layout and edit events.

## 4. Ordered implementation plan

Every step ends with its Dawn lane only, targeted per the standing rules. Run `tools/capture-uniform-wgsl.mts` (the CPU shader preflight) before each Dawn run.

**Step 0 — test harness (Dawn only).**
- Extend `/Users/petersuggate/code/me/fluid/tools/probe-uniform-pond-rest-dawn.ts`:
  - `--detail=none` (all-4h);
  - `--basin=flat|slope|terrace|step`;
  - `--floor-offset-cells`;
  - existing `--waterline-shift-cells`;
  - output: peak speed, interior and shoreline surface error, excess, mass, residual and correction amount.
- Interior columns are redefined at H: the 3×3 H neighbourhood at and below the surface is uncut. Shoreline is everything else.
- New Dawn test file: `/Users/petersuggate/code/me/fluid/tests/uniform-coarse-solid-rest-dawn.test.ts`.

**Step 1 — isolate the non-solid H surface defect.**
- Run T1a: flat basin with no solid, all-4h, waterline offsets {0, .25, .5, .75}·H.
- Bisect by toggling one candidate at a time inside the probe (A/B env, not committed knobs):
  - (a) target vs initial V mismatch, i.e. c ≠ 0 at frame 0. Check excess, and compare `umSurfaceTarget` with the init fraction;
  - (b) sharpening on H owners (`umSharpenOpen` true at width 4);
  - (c) surface-volume shift;
  - (d) redistance wall metric at width 4;
  - (e) occupancy-gated gravity vs pressure classification (W2);
  - (f) θ floor 1e-3 at offset 0 when the waterline sits on the centre;
  - (g) detached mass;
  - (h) root acceptance tolerance and warm start.
- Fix the root cause. Expected first suspect: (a).
- Exit: T1a passes the pond-rest tolerances.
- Then remove the scenes.ts:2316 "keep the surface h" profile constraint in a separate change.

**Step 2 — target convention and F_Ω.**
- Record mask and centroid (`uniform-mixed-solid.wgsl.ts`).
- F_Ω in `uniform-volume.wgsl.ts` (mixed variant takes the record).
- `gcOwner` `:128-137`.
- Authority: `:119,205-209,219` (drop `*cap`).
- Surface-volume: `:113,160,181,370-391`.
- Initial V of cut H owners = F_Ω(φ₀) through the same WGSL, so that initial excess is 0.

**Step 3 — owner-width buried and initial φ.**
- `uniform-mixed-surface.ts`: `:157-161`, `:177-236` (disable embedded terms for cut H), `:268,278`, `:369-377`, `:419-433`, `:556-571`, `:686,697`.
- `uniform-volume-initial.ts:8-58`.
- Shared helper `umVertexBuried(p)` in `uniform-mixed-solid.wgsl.ts`.

**Step 4 — pressure W1.**
- `uniform-mixed-pressure-velocity.ts:143-167` (cut H flux with record V and the body term).
- Halo RHS `:205`.
- Authority voted φ `:161-183` restricted to h, with H using centre φ.

**Step 5 — remove mandatory promotion, then T1b and T2.**
- Gate `uniform-mixed-layout.ts:97-176` and `uniform-mixed-dynamic.ts:382-397,800-822` behind a detail request.
- Delete the assertion and frame bit 1 (`uniform-mixed-frame.ts:602`).
- Tests: T1b (floor at Ω = 0.25) and T2 (slopes).

**Step 6 — transport.**
- `uniform-mixed-transport.wgsl.ts`: `:96`, `:141`, `:193`, `:242`, `:268-270`, `:281`.
- Centroid departure at `uniform-mixed-surface.ts:945` (traceCells) and the transport departure.
- Tests: T3 (communicating vessels, motion), T4 (film).

**Step 7 — forces, sharpening, cleanup.**
- `uniform-mixed-forces.ts:107,137,188`; detached-mass rule.
- `uniform-mixed-sharpening.ts:114-118`.
- `uniform-mixed-cleanup.ts:71`.

**Step 8 — seams.**
- `uniform-pressure-band.ts:541` (V-weighted Neumann).
- Drop bit 2 (`:411-416`, `uniform-mixed-frame.ts:602`).
- Test T5.

**Step 9 — remap.**
- `uniform-mixed-remap.ts:296-368` (V-weighted means), `:694-752` (capacity-aware split).
- Test T8.

**Step 10 — displacement.**
- `uniform-mixed-solid-displacement.ts:43-76` (H scatter).
- Test T7.

**Step 11 — bodies.**
- `uniform-mixed-bodies.ts:62-67` (wet and velocity from faces), `:81` (H couple).
- Body-near on-the-fly record.
- Test T6.

**Step 12 — pond port.**
- T9, hose-x10 at zero detail.
- Then the existing lanes once: uniform-mixed-solid-parity, live-solid-edit, rigid-body, uniform-coarse-surface, uniform-pond-rest.

## 5. Acceptance tests (all Dawn) and tolerances

Scale: h = 6.25 mm (`/Users/petersuggate/code/me/fluid/lib/core/model.ts:411`), H = 25 mm. Read the actual h from probe JSON before asserting.

Tolerances come from physics, not fitting. A resting state is an exact discrete equilibrium (§1), so the error budget is solver tolerance plus f32 only:
- Surface drift: |δη| ≤ ε_div·D·T, with ε_div = 1e-4 s⁻¹ (the residual gate in `/Users/petersuggate/code/me/fluid/tests/uniform-pond-rest-dawn.test.ts:35`), depth D ≤ 40 mm and T = 1.5 s. That gives 6e-3 mm, under the existing 0.025 mm interior RMS.
- f32 quantum of η at y ≈ 0.5 m is about 6e-5 mm.
- Speed after one frame: ε_div·H ≈ 2.5e-6 m/s, far under 1e-3 m/s.

**The mm tolerances are resolution-independent and do not get rescaled for H.** The one that needs H justification is the **0.5 mm all-columns maximum**:
- At h it is about 0.08·h. At H it is 0.02·H.
- It holds at H only if W3 holds at shoreline cut owners.
- If T2 fails it while interior passes, the defect is a non-planar φ at buried-rule boundaries or embedded terms. Fix it there. Do not widen it.
- The same applies to the initial-excess bound 0.003. With F_Ω used for both V₀ and the target it should be about 1e-7. Assert < 1e-5 so that W4 regressions are caught.

| Test | Scene | Oracle and assertion |
|---|---|---|
| T1a | flat, no solid, all-4h; waterline offset {0, .25, .5, .75}·H | frame 1: surface < 0.001 mm and speed < 1e-3 m/s. 90 frames: interior RMS < 0.025 mm, max < 0.5 mm, residual < 1e-4, correction ≡ 0, mass 1e-6 |
| T1b | flat voxel floor at offsets {0, 1, 2, 3}·h, so floor Ω ∈ {1, .25, .5, .75}, plus T1a offsets | as T1a. Initial excess < 1e-5 |
| T2 | planar terrain 1:3, 1:1, diagonal (x+z) 1:2; waterline crosses cut owners | as T1a. Shoreline max < 0.5 mm. No air row below η (classification census) |
| T3 | two basins, terrace wall with one cut H window below η; levels differ by 2H at t = 0 | final common level = analytic (ΣV/ΣA, computed from the h mask) within 0.025 mm + one h-cell volume / area. Monotone level approach (no overshoot > 2 mm). Mass 1e-6 |
| T4 | film 0.3H on flat cut floor (Ω = .5) | film thickness constant within 0.025 mm. No downward speed > 1e-3 m/s on faces with V_H,y > 0 |
| T5 | T2 slope + h detail box straddling waterline and slope | as T2 on both sides. Seam face speed < 1e-3 m/s |
| T6 | (a) body ρ_b = ρ held at mid-depth, released; (b) ρ_b = 0.5ρ cube | (a) drift < 1 mm/s after 1 s. (b) equilibrium draft = 0.5·side within 0.5 mm. (c) prescribed body translation: liquid-region divergence residual < 1e-4 |
| T7 | live edit adds a k·H block fully under water | level rise = ΔV_solid/A_free within 0.025 mm + f32. Mass 1e-6. Remove the block and get the level back |
| T8 | rest pond; promote then retire a cut H band of tiles each frame for 10 frames | speed < 1e-3 m/s every frame. Surface error as T2. V exactly conserved (mass 1e-6) |
| T9 | hose-x10 pond-rest, zero detail | existing `tests/uniform-pond-rest-dawn.test.ts` assertions unchanged, with interior classification at H |

T1–T4 and T7 can share one process, one device and several scenes. The thin-liquid probe pattern rebuilds in about 0.3 s per scene. This is one Dawn run per arm.

## 6. Risks

1. **The non-solid H defect is unexplained** (Step 1). If it is in the root continuation from level 2, or in the h-band coupling rather than (a)–(h), every later step is blocked until it is found. Budget for it first.
2. **Transport accuracy near cuts.** min(Ω_r, Ω_q) over-weights receivers whose open part does not face the donor, for example a floor slab under a wall slab. This is conservative but diffusive, and it may smear thin films along walls. T3 and T4 detect it. The exact remedy is the subcell mask overlap at grain 1 (64-bit AND per overlap), at 4× gather cost on cut rows.
3. **Buried-rule churn.** More state vertices inside solid means advect and redistance write them. Redistance stencils must never read truly buried values. A leak shows as T2 shoreline error growing over time.
4. **Contact physics.** No embedded release at H means liquid can "stick" under overhangs. This is acceptable for ponds, and visible for drips. Document it. Detail requests at overhangs are the policy answer.
5. **Bodies at H** are 64-subcell loops on body-near tiles, which costs GPU time when a body spans many H tiles. Measure it.
6. **Live edits at H** move excess over 4h strides, so the jump is ≤ H. That is visible as a surface bump of ≤ ΔV/A, which is physical.
7. **Policy interplay.** Removing mandatory promotion changes frame cost and h-tile counts in every solid scene. That is intended (garden, hose), but it makes before/after frame-time comparisons cross-layout. Compare at matched detail.

## 7. What to measure

- **Pond rejection replay.** coarse-first on hose-x10 at peak speed (0.020 baseline, 3.25 rejected). Record which W-condition each step restores. This is the per-step "explain the residual" record.
- **Per step**, from the probe JSON: max |c| (correction), initial excess, number of air rows with centre below η, buried-vertex count at H corners, root and band residual, and iterations.
- **Cost:**
  - F_Ω evaluations per frame (cut surface owners × crossing subcells);
  - h-step walk length for cut H owners;
  - record build time on `invalidate`;
  - body-near on-the-fly overhead.
  - Compare against the frame time saved by retiring solid promotion. Reference: hose-x10 h tiles 1663 at 25.3 ms after 98394421. Garden h 2258 at 15.8 ms after 2c4784c2.
- **Frame time** on one target scene (hose-x10) at zero detail vs the HEAD layout. One Dawn run per arm.

## 8. All-4h Figure 9: the speed spike and the pressure fatal (2026-10-06)

Scene `mass-conserving-figure-9-dam-break`, Requested with no region, app settings (dt 1/60), 240 frames. A run is deterministic for a given code and perturbation (one extra dry h tile); the tallies below are over perturbations. Before: 7 of 81 runs fatal between frames 187 and 200 (4 "rejected a non-improving cycle", 3 "did not converge after 3 cycles"), and the peak speed after frame 150 was 35.8 m/s in the median run, 47.8 at the ninth decile and 63.4 at worst, against 21 m/s at the first impact (frames 40 to 60).

Three defects, each isolated by one switch. The hollow solid box is needed for none of them: with solid contact off (no h tile at all) the same two sites spike and the same fatal appears.

- **Lid limbo (speed, frames 100 to 185).** The pressure releases the lid's owners (centre phi above −4h, so not `umWallHeld`) and their V drains, but `umReleasedWalls` lifted a wide owner's wall vertex only within 2h of the surface, so phi stayed on the lid: at frame 133, 263 of the 283 phi-liquid lid owners held less than half their V. Such an owner is a pressure row with a half-weighted released wall face and nothing in it. The operator asks its wall face for twice the opposite face's speed, advection equalises the two, and the wall face doubles again the next step: −28 to −32 m/s at −13 to −70 kPa.
  - Switch (lift rule alone, unperturbed run): largest wall-face speed 28.1 → 10.6 m/s, frames with a wall face above 12 m/s 41 → 0, peak speed 36.7 → 29.2.
  - Landed (`umReleasedWalls`, `uniform-mixed-surface.ts`): deeper than the old rule reaches, a wall vertex separates where every owner around it has released, by the least of their gaps. The old rule is unchanged where it applied.
- **V over capacity in air owners (speed, frames 165 to 200).** The phantom lid sheet and the global surface-volume shift leave a layer of V under air phi on the reservoir floor. The returning bore piles it: 4 to 12 capacities in one owner, 40.5 at worst, in owners whose centre phi is air and which therefore have no pressure row. The liquid beside them sees a vacuum face inside that material, and the projection writes 20 to 36 m/s through it in the same frame.
  - Switch (with the lift rule): largest V 17.8 → 2.8 capacities, excess V 161 → 68 cells, peak speed 29.2 → 23.7. Without the lift rule the same switch runs into the fatal at frame 189.
  - Landed (`uniform-mixed-pressure-authority.ts`, one binding in `uniform-mixed-frame.ts`): over capacity, V claims the owner's row (`umOwnerSurfacePhi`, the native `pressureSurfacePhi` rule), only where the simulation holds the owner at 4h. A `rows` launch writes every owner's row phi first, so detached mass is decided on the same rows the velocity stage tests.
- **Enclosed air averaged away on the root's coarse levels (the fatal).** The root continuation starts at level 2 of the native hierarchy, and `mgDownsampleTopology` keeps the sign-aware phi rule for levels counted from level 0, so every coarsening of the root is a plain average. An air owner enclosed in 100 to 214 kPa liquid is then liquid on every level but the root, the coarse corrections carry no sink where it sits, and the root's smoother alone removes what they add: cycles contract by 0.6 to 0.8 where they usually contract by 0.1, and the three-slot budget ends above the tolerance, or its tightened slot raises the maximum norm.
  - Switch on a frozen state (frame 197 of a fatal run): 16.67, 12.04, then 16.35 and the fatal; with the enclosed owners given rows, 3.35 in one cycle.
  - Switch in the code (sign-aware root levels alone, 81 runs): fatal 7 → 0, frames with a stalled slot 46 → 0, and the speeds unchanged (35.4 m/s median). Without it (the two speed fixes, and rows for half-full enclosed air): speeds fixed, 6 of 241 runs still fatal.
  - Landed (`mgDownsampleTopology`, `webgpu-uniform-pressure-multigrid.ts` and `.wgsl.ts`): on the root's first coarsening, a mixed coarse cell whose average is liquid keeps its air (the positive mean) when none of the 26 coarse cells around it would come out air either (`mgAirVanishes`). At a free surface one always does, so the surface coarsens as before. The native dense solve is untouched (`control.z` is 0 there).

| Figure 9, zero detail | runs | fatal | frames with a stalled slot | late peak: median, p90, worst (m/s) | frames per run above 30 m/s |
|---|---|---|---|---|---|
| before | 81 | 7 | 46 | 35.8, 47.8, 63.4 | 2.86 |
| sign-aware root levels only (not landed) | 81 | 0 | 0 | 35.4, 44.0, 65.7 | 2.28 |
| lift, over-capacity rows and half-full enclosed rows; no solver change (not landed) | 241 | 6 | 60 | 25.2, 32.0, 52.6 | 0.17 |
| the three landed | 241 | 0 | 0 | 24.2, 30.3, 36.6 | 0.13 |
| the three and the 26-neighbour travel gate | 81 | 0 | 0 | 26.8, 33.7, 38.8 | 0.25 |

Second slot over first slot, ninth decile: 0.54 before, 0.15 after. First-slot residual, median: 0.141 before, 0.088 after.

Measured and not landed:
- The sign-aware rule for every mixed cell of the root's levels (one level or two). No stall either, but the whole surface is then coarse air and an ordinary cycle leaves 1.7 to 2.7 times the residual: dam64 Dynamic first slot 0.067 → 0.181, mean of the runs' largest accepted residual 3.41 → 4.41, Figure 9 Dynamic 1.60 → 4.13.
- Keeping enclosed air on two levels: 0 of 81 fatal as with one, and the cycles no better.
- An enclosed air owner at least half full counted as liquid (rows for the holes themselves). It removes the frozen frame's stall but not the class: bubbles whose centre owner hovers at the threshold stalled as before (2 of 99 and 6 of 241 fatal). Not needed once the coarse levels keep the air (0 of 81 with or without).
- The row rule applied to the root's owner of a tile the simulation holds at h: dam64 Dynamic's accepted residual rose (mean of the maxima 2.8 → 3.6).
- Rebuilding the neighbours' rows in the build lane and not in a launch of its own: +0.25 ms on Figure 9 and on the pool.

What remains: about one run in ten has a frame above 30 m/s and none of 241 has one above 40. The peak is a tip owner at the x = 0 wall at capacity, fed through its floor and side faces with one air face left: incompressible outflow through one face, at none of the three sites above.

Other scenes at app settings, accepted pressure residual (all-frame median, 99th percentile, mean of the runs' maxima; tolerance 5), before → after the three:

| scene | runs | median | p99 | mean of maxima | h tiles |
|---|---|---|---|---|---|
| dam64 Dynamic | 24 | 0.071 → 0.067 | 2.28 → 2.41 | 3.41 → 3.54 (sd 0.9) | 1356 → 1316 |
| Figure 9 Dynamic | 8 | 0.069 → 0.058 | 1.10 → 1.01 | 1.60 → 1.81 (sd 0.9) | 3388 → 3393 |
| pool impact Dynamic | 8 | 0.017 → 0.018 | 0.45 → 0.46 | 0.87 → 0.98 (sd 0.12) | 1384 → 1390 |
| Figure 9 Full | 1 | 0.065 → 0.069 | 2.86 → 2.48 | 3.91 → 4.17 | all |

No fatal in any of them. Figure 9 Full's peak speed falls from 114 to 43 m/s.

Frame time, the three together, interleaved pairs in both arm orders (sim GPU, extraction, host; ms): Figure 9 zero −0.27, −0.21, −0.02 (redistance −0.13, extension −0.07, pressure setup +0.06); pool zero +0.06, +0.03, −0.01; dam64 Dynamic +0.05, −0.01, −0.02; Figure 9 Dynamic +0.26 by medians and 0.00 paired, −0.01, +0.02; pool Dynamic +0.08 ± 0.22 sim over eight perturbed pairs (h tiles +7 ± 38).

**Travel gate (landed with them).** `umWideAdjacent` now holds every corner of a cut wide cell: one of the vertex's 26 neighbours at its own stride has the other sign, or a face neighbour is buried. `uniform-coarse-redistance` is green without a change to the lane; held by its face neighbours alone, a still sphere's same-sign corners were rebuilt against the zero set they shape and moved it (the lane's corner phi 0.21490 → 0.21927 at the first rebuild).
- The fatal: on the old code the rule raised it from 1 of 17 runs to 5 of 17. With the three in it does not: 0 of 81, as without it (and 0 of 99 against 2 of 99 on earlier forms of the fixes). What tied the two on the old code was not isolated; it goes when the root keeps enclosed air. Figure 9 zero's late peak is a little higher with the gate (26.8 against 24.2 m/s in the median run, none above 40).
- Residuals (mean of the runs' maxima, without → with the gate): dam64 Dynamic 3.54 → 3.34, Figure 9 Dynamic 1.81 → 1.32, pool Dynamic 0.98 → 1.08, Figure 9 Full identical.
- Cost in redistance: +0.07 ms on Figure 9 zero (sim +0.25, extraction +0.05), +0.04 on dam64 Dynamic, +0.15 on Figure 9 Dynamic. The rule is exactly "a corner of a cut wide cell", the least the lane's invariant allows, so there is no cheaper exact rule; the evidence test before the neighbours and the six faces before the twenty others made no measurable difference (±0.005 ms).
- Cost through the census, which is the larger one: under Dynamic the shape criterion (`umResolutionError`) requires more tiles. Pool impact Dynamic h tiles +19 % ± 2 (1390 → 1658) and sim +1.26 ± 0.19 ms (+8.6 %) over eight perturbed pairs; dam64 Dynamic +2.8 % ± 0.9 and +0.06 ± 0.04 ms; Figure 9 Dynamic +1.5 %. With the shape criterion off there is no inflation. Separated and fixed in the next paragraph: the criterion was reading held corners that are no longer distances.

**Held corners in the shape criterion (follow-up, landed).** The gate's census cost was the criterion reading held magnitudes, not surface the six-face rule had been eroding.

- Which test fires. Every shape trigger is an h tile that fails to coarsen: at width 1 `umResolutionError` compares the tile's h vertices within 2h of the surface with the trilinear interpolant of its eight wide corners. The 4h tiles' own test (second differences of wide phi over 16h) cannot pass 0.5 on a distance field and fired on none of 2675 cut 4h tiles at tolerance 0.5 or 0.25; 4h tiles turn h by dilation from required tiles.
- Fork, pool impact 128³ Dynamic, one state at each of four instants with the last redistance run under the 26-neighbour gate, the six-face gate and no hold. Shape triggers summed over the four: 1873, 1784, 1398. The six-face tree carried 386 of the 475.
- What the error is made of. Swapping only the eight corner values between the arms moves the count with them (at one instant: rebuilt h values under held corners 414 against 412 all held; held h values under rebuilt corners 239 against 234 all rebuilt). In the extra tiles the criterion reads 0.62 to 0.73 h (median) where the zero set of the h vertices and the zero set of the corners' interpolant are 0.19 to 0.31 h apart, and a measure of that displacement alone triggers the same on the three fields (704, 705, 667). The held corners' interpolant has slope 1.2 to 1.3 against 1.03 to 1.05 rebuilt: the zero set is where it should be, the magnitudes are not distances, and the tile's own h vertices are rebuilt every step.

| h tiles 16 frames after the fork | instant 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 26-neighbour gate | 1733 | 1016 | 1646 | 2462 |
| six-face gate | 1501 | 1007 | 1572 | 2310 |
| no hold | 1290 | 848 | 1011 | 2010 |

Landed, behind `uniformDetailHeldDistance` (Dynamic with the shape criterion at a tolerance above 0; the declared lane defaults do not evaluate the criterion and run exactly as before):
- `uniform-mixed-surface.ts`: `umBandState` returns kept, rebuilt or held. A held wide vertex that an h tile reads (`umHeldRead`) and that is not buried still goes through the search, but the search's result is not written: `found − initial` goes to one word per wide vertex after the travel words of the claims buffer (`umHeldIndex`). Phi, the travel words and the lane's bit-for-bit corners are untouched.
- `uniform-mixed-dynamic.ts`: the h-tile test adds that word to its corner values (`umHeldCorrection`) and skips the tile's vertices a 4h tile shares (`umWideShared`), which are wide vertices, held themselves, or hanging vertices, the corners' interpolant by construction (1.6e-6 h measured). The correction is additive, so the surface-volume shift that follows redistance does not invalidate it.
- The criterion on a field is now the old criterion on that field with its held corners rebuilt. The rule set and thresholds are unchanged.

Clean field. On the four fork fields rebuilt with no hold (every correction zero) the old and the new rule select the same tiles: 0 of 1398 triggers differ. On the held fields a CPU model of the channel gives the rebuilt fields' tiles to 3 of 1398, and the GPU scores equal that model on every cut h tile at four frames (710, 247, 606 and 1221 published vertices; correction magnitude 0.14 to 0.27 h in the median, 0.9 to 1.4 h at the ninth decile).

Against the three-fixes tree (no gate change), eight perturbed pairs in alternating arm order, app settings, 240 frames:

| scene (three-fixes sim, h tiles) | sim, ms | h tiles | mean of the runs' largest accepted residual |
|---|---|---|---|
| pool impact Dynamic (14.63, 1390), gate as first landed | +1.26 ± 0.19 | +268 | 0.98 → 1.08 |
| pool impact Dynamic, gate with held distance | +0.13 ± 0.33 | −12 ± 58 | 0.98 → 0.90 |
| Figure 9 Dynamic (21.91, 3393) | −0.44 ± 0.17 | −98 ± 33 | 2.17 → 2.25 (five trajectories) |
| dam64 Dynamic (9.79, 1333) | +0.04 ± 0.06 | −19 ± 29 | 3.19 → 3.29 |
| Figure 9 and pool, zero detail, against the landed gate | −0.01 ± 0.02 | none | identical |

- Four of the eight perturbations leave Figure 9 Dynamic's trajectory unchanged, so its eight runs are five trajectories (this also applies to the eight-run Figure 9 Dynamic rows above). Over the five: sim −0.64 ± 0.24 ms, h tiles −108 ± 54, largest accepted residual paired +0.07 ± 0.67, worst 3.42 → 3.63, frames above 2 of 1200: 3 → 4. Counted as eight runs the mean of maxima reads 1.81 → 2.77, which is the repeated trajectory (1.20 → 3.63) counted four times.
- Peak speed, paired: pool +3.4 ± 8.4 m/s (53.3 → 56.7), Figure 9 Dynamic +5.4 ± 4.3 over the five trajectories (46.0 → 51.4, largest 54.3 → 68.8), dam64 0.0 ± 2.4. No fatal in any run.
- At equal tile count the pool pairs put the gate, the published searches and the census read at about +0.2 ms together, inside the pair noise.
- Lanes green on the landed tree: `uniform-coarse-redistance`, `uniform-detail-policy`, `uniform-dynamic-coarsening`, `uniform-dynamic-band-mask`, `uniform-coarse-surface`, `uniform-pond-rest`.

Measured and not landed, or closed:
- Criterion-only normalisations (dividing by the interpolant's slope, a symmetric comparison, a 1h band, Lipschitz bounds from the tile, the corners or the h neighbours): each changed 300 to 780 tile-instances on rebuilt fields and none removed the excess.
- Keeping held magnitudes distances without moving the zero set needs one factor per connected set of cut cells, and rebuilding held corners is the instability the gate exists for. Both closed.
- A criterion of zero-set displacement alone would halve the shape triggers on every field, clean ones included. It changes what "needs h" means and is the owner's call; not done.

Not addressed: the 4h tiles' second-difference test still reads stored values (it does not fire at 0.5 or 0.25; at 0.125 held values give 13 and 9 tiles where rebuilt values give 1); an h tile's own corners beyond the 4h band are never rebuilt (as before, in every arm); the thin criterion reads stored magnitudes.

**Figure 9 zero, the late peak.** Full with the three fixes peaks later and higher than zero detail does: 43.2 m/s at frame 170 and 39.2 at frame 187, against 37.8 at the first impact; zero detail's late peak is 26.8 in the median run, 33.7 at the ninth decile and 38.8 at worst. The run-up jet at the wall is in the resolved flow, and a tip owner at capacity with one air face left is that jet at 4h. No fourth defect found and no candidate proposed. One Full run, and the location of Full's peak was not compared with the tip owner's.

**Shape by surface displacement, as an option (2026-10-06, later).** Peter: "yes, as an option". `detailShapeMetric` is `value` (default: `umResolutionError` with the held correction, unchanged, so the lane and app defaults give the census they gave) or `displacement`. Displacement (`umSurfaceDisplacement`, `uniform-mixed-dynamic.ts`) measures an h tile by how far its surface moves when its eight 4h corners carry it: at every zero crossing of an h edge, the corners' trilinear phi over its gradient, in h, against the same `detailShapeTolerance`. It reads the stored corners (their zero set is what 4h keeps), so `uniformDetailHeldDistance` is off under it and redistance publishes nothing. The 4h tiles' second-difference test is unchanged. The kernel equals a CPU model on every cut h tile at three frames of pool impact (713, 144 and 574 tiles, score bytes identical).

Off against on, Dynamic at app settings, 240 frames, the two arms interleaved in one process, unperturbed and gravity × (1 ± 1e-6); means of the three runs, whole frame = host + sim GPU + extraction:

| scene | whole frame, mean (ms) | per run | worst frame | h tiles | shape-triggered tiles | largest accepted residual |
|---|---|---|---|---|---|---|
| Figure 9, 128×128×64 | 23.79 → 23.38 | −0.30, −0.45, −0.50 | 38.4 → 36.2 | 3036 → 2975 (−2 %) | 1029 → 884 (−14 %) | 3.41 → 2.46 |
| pool impact 128³ | 24.23 → 23.69 | −0.39, +0.53, −1.74 | 44.7 → 40.4 | 1444 → 1229 (−15 %; −262, +5, −388) | 275 → 189 (−31 %) | 1.02 → 0.85 |

- Figure 9 is two halves. Frames 1 to 80 (the smooth collapse): h tiles 1133 → 711 and sim 11.8 → 9.9 ms. Frames 160 to 240: h tiles 3666 → 3999, tiles holding a sign change 1661 → 1836 and sim 18.5 → 19.7 ms, in all three runs. So the mean falls and the median frame rises (23.6 → 24.5 ms): with displacement on, the late surface is in more pieces.
- Quality, on against off (no Full reference in this run). Figure 9: the toe is 1 to 3 h ahead between frames 30 and 45 and reaches the wall on the same frame (47); kinetic energy within 5 % to frame 100 and equal over the run (ratio 0.99 to 1.01); volume drift −0.42 % either way, represented-volume drift at worst 5.1 → 4.4 %. Pool: the splash after impact carries 7 to 9 % less kinetic energy at frame 60 in all three runs, equal over the run (0.98 to 1.02); volume drift at worst 0.063 → 0.040 %. Peak speed moves both ways run to run (Figure 9 57 → 49 m/s, pool 47 → 53) and no run is fatal.
- Not measured: surface error against Full, which is what would say whether the tiles it drops were buying anything. The app default stays `value`.
