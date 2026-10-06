# Uniform Geometric 4h-first: storage design

3 October 2026. Implementation contract for the [4h-first handoff](uniform-4h-first-implementation-handoff-2026-10-03.md). It fixes the h-field storage ABI so the workstreams port against one rule.

**Scope.** This is compact storage for the existing mixed solver. It keeps every numerical kernel, so it also keeps that solver's transport defects, repair stages and time-step behaviour: completing it can leave Dynamic behaving exactly as it does at HEAD. A simpler 4h numerical method is a separate milestone and is not specified here. The physical layout and the order below were reopened on 4 October; see the last section, which supersedes "Order" and the first revision's cost and memory claims. Evidence: two read-only audits of the current mixed path (architecture map; canonical-texel audit), summarised below.

## The rule

The h lattice, its 4³ tiles (one H = 4h cell each), ownership words, owner lists and every numerical kernel stay as they are. Only **where an h-lattice texel lives** changes.

Every h-sized field is one physical texture holding:

- a **base block**, the compact H storage every tile has; and
- an **atlas** of patch slots, h storage for resident patches only. A patch is 32³ h = 8³ tiles; slot extents are 32³, or 33³ for vertex fields.

A patch directory (per patch: slot+1, or 0 when not resident) lives in the tail of the mixed topology buffer, so no new binding is needed. Every field access maps its logical h coordinate `p` through one function:

```
tile(p) resident  -> atlas texel  slotOrigin + (p - patchOrigin)
otherwise         -> base texel   (canonical coarse location; see classes)
```

Field classes are fixed at compile time per binding:

| Class | Fields | Base block | Non-resident canonical set |
|---|---|---|---|
| cell | volume (A/B/scratch), surface, gamma, phase, centre phi, pressure target, correction, curvature, targetFill | t³ at `p/4` | tile origin only. Loads anywhere in the tile return the coarse value; a store away from the origin is a violation. |
| face | velocity A/B/D and scratch, departure, extension output | (2t)³ at `2(p/4) + (p%4==3)` per axis | the origin and the three face anchors `o+3e_a`, each a full vec4 with its own `.w` bits |
| vertex | vertex phi, phi scratch | (t+1)³ at `p/4` | 4-aligned vertices. A hanging vertex is trilinear in its home tile's corners, the same rule PhiResolve applies. The home tile is `min(p/4, t-1)`. |
| atlas-only | unitVelocity, hanging caches | none | never touched by a non-resident tile |

**Residency** is all h-owned tiles plus their 26-connected one-tile neighbourhood, rounded up to patches. Seam, band and PhiResolve reads stay inside it. A resident coarse tile keeps its data at the h texels it uses today: a newly resident patch is broadcast-filled from the base block, and a retiring patch restricts its canonical texels back to the base. Width decides ownership; residency decides storage. Both are recorded per accepted generation.

With zero detail no patch is resident and every texture is its base block alone.

## Consequences

- Departures are unbounded, so safety rests on the owner-aware samplers (`umOwnerAt`, width, canonical anchors). Those already read only canonical texels of 4h owners. The `umRegularFine` raw fast paths run only under the frame-plan reach certificate. That certificate guarantees no 4h tile within `ceil(travel/4)+2`, so those paths run inside resident h tiles.
- Accesses are rewritten centrally when the shader is assembled. `rewritePressureTextureCalls` is generalised to every group, with class keyed by binding name. No module carries hand-edited per-site calls. The generic per-access lookup is the correctness baseline only; the handoff forbids it as the endpoint. The second step makes `umOrigin`, `umFace` and the tile prologue resolve the physical patch once per owner or workgroup. A fast patch-local offset then handles every tap that stays inside the owner's patch.
- Logical dimensions come from uniforms or constants and are asserted at bind time. Any `textureDimensions(field)` used as the lattice size is a bug under an atlas, and must be removed before landing.
- Whole-texture uploads, the t=0 dense authority/publication passes, and test readbacks go through a host field factory. It deposits canonical values, and decodes logical coordinates for readback.
- Remap sees both directories: new patches are allocated and filled before the remap, and freed slots are released only after every reader of the old generation completes.
- Running out of atlas capacity is a loud fatal error at admission, never a fallback.

## Known defects the audit found on the way

- `uniform-mixed-bodies.ts:67` reads the body velocity at `umOrigin`, but a 4h owner's faces live at its anchors.
- The raw centre-phi neighbour reads read stale texels from closed h cells: `uniform-mixed-surface.ts:157-236, 419-430`, and `umPressurePhiCell` / `umFinePressurePhiCell` in `uniform-mixed-pressure-authority.ts:109-137`. They should route through `umOrigin(umOwnerAt(q))`.
- The sharpening anchor-word cache (`uniform-mixed-sharpening.ts:101`) needs a key per (owner, axis).
- `mixedSource.openFraction` is gammaB, which holds the surface target after frame 1.
- The legacy name "resident" means far-air paging (`umResidentAllOwner`). In the new code, h storage residency is called **detail storage**.

## Order

1. Build the field factory, the classes and the central rewrite, with every patch resident. This is a full atlas, equivalent to today's storage. Gate: ulp-level agreement with HEAD on the mini dam and Figure 9 receipts.
2. Derive residency from ownership; zero detail allocates the base block only. Add allocation and retirement transactions, plus growth.
3. Size the owner-indexed buffers from fine capacity rather than `tiles*64`: transport rows, band rows, deposits, root pressure storage and the arena reservation. Remove the frame's all-fine capacity assertion.
4. Resolve addresses once per owner or workgroup. Run the 16³/32³ comparison on the frozen all-h frame.
5. Port the renderer and overlay to the accessor, and switch marching to H plus resident patches.

## Revision, 4 October 2026: patch-local addressing

Steps 1 and 2 are built (packed placement, a QA option; identity was production when this revision was written, and the domain placement below has been the default since 4 October 2026). The packed path is numerically sound but its addressing is not: on the 64³ dam it costs +19% to +26% over identity (9.7–10.0 ms against 11.6–12.3 ms), and the falling-water torus cold-compiles in 139 s against a 120 s lane ceiling. The cost is the form of the address, not the amount of arithmetic:

| Address form | Cost on dam64 |
|---|---|
| compile-time entry | +1.4% |
| arithmetic on a run-time value already in hand | +2.0% |
| directory read once per invocation, arithmetic taps | +2.5% |
| run-time entry with a residency branch per tap | +12% |
| directory read per tap, no branch (storage or uniform buffer) | +20% to +24% |
| `var<workgroup>` entry filled by invocation 0 | +88% |

So step 4 changes from "resolve once per owner" as an optimisation of the per-access lookup to a different contract: **a kernel addresses its own patch arithmetically and never tests residency per tap.** That needs each patch to carry what its fixed stencils read.

- **Halo.** Each slot carries a derived halo: r = 1 for cell and face fields, pitch P+3 for vertex fields. A halo texel is a copy of the canonical writer's value with the field's generation; it is never an authority and nothing stores to it except the refresh.
- **Data-dependent samplers** (momentum, surface advect, traceCells, advectDeferred, transport gather; 20.6% of the dam64 frame) resolve the directory once per sample point. With r = 1 a trilinear footprint lies inside the patch holding the sample point.
- **Out-of-owner stores** (band project and copy low slabs, vertex closure writers) stay directory-resolved and write the canonical texel.
- **4h taps** (owner-origin and 4h-lattice reads, redistance window) read the base block, or resolve their patches once per workgroup.

Cost on dam64. Measured: addressing +2.0% to +2.5% (fixed-region and Dynamic), 0% to +2.1% under Full. That probe uses one patch covering the lattice, so it shows arithmetic addressing can be cheap and nothing more: it has no patch boundaries, no halo refresh, no per-sample resolves and no base synchronisation. An earlier version of this section summed estimates for those into "+5% to +6.5%"; that figure is a guess, not a result, and is withdrawn until a multi-patch frame is measured in its real producer/consumer order.

## Review, 4 October 2026: layout and order reopened

A design review of the above found six problems. All six stand. The ownership and lattice invariants are kept; the physical layout and the delivery order are open again.

**1. Residency swamps sparsity.** Closure (one tile, 26-connected) followed by rounding to 32³ patches allocates far more than the h tiles. Allocated patch coverage, computed on the CPU for synthetic requests (`closure` = today's rule; `h-only` = patches holding an h tile, which needs halos to carry the neighbours; storage = cell field relative to identity, base plus haloed patches, r = 1):

| Request | n | h tiles | P=32 closure | P=32 h-only | P=16 h-only | storage P=32 / P=16 (h-only) |
|---|---|---|---|---|---|---|
| one tile at the centre | 64 | 0.02% | 100% | 12.5% | 1.6% | 0.17 / 0.04 |
| flat sheet, one tile thick | 64 | 6.3% | 100% | 50% | 25% | 0.62 / 0.37 |
| flat sheet, one tile thick | 128 | 3.1% | 50% | 25% | 12.5% | 0.32 / 0.19 |
| pool surface plus floor contact | 128 | 6.3% | 50% | 50% | 25% | 0.62 / 0.37 |
| 12.5% cube, centred | 128 | 12.5% | 100% | 12.5% | 12.5% | 0.17 / 0.19 |
| flat sheet, one tile thick | 256 | 1.6% | 25% | 12.5% | 6.3% | 0.17 / 0.10 |
| cube of edge n/4, moving | 128 | 1.6% | 12.5–18.8% | 6.3–12.5% | 1.6–2.3% | |
| scattered tiles, 1% | 128 | 0.9% | 100% | 100% | 44% | 1.22 / 0.64 |
| scattered tiles, 5% | 128 | 4.9% | 100% | 100% | 96% | 1.22 / 1.38 |

So at 64³ today's rule gives a full atlas for almost any request; a surface sheet, the common Dynamic shape, pays 8–16× its h tiles at P = 32; and fragmented detail defeats patches of either size. Halos must replace the closure, not sit on top of it, and P = 32 is not settled: these numbers favour 16³ for sheets and moving requests, and the choice waits on measured halo cost at both sizes. Real request masks (the importance census on the long dam and Figure 9) should replace the synthetic ones before the choice.

**2. Break-even is per field class.** Relative storage is about base fraction + allocated fraction × halo factor. At P = 32, r = 1 the pool is smaller than identity only below 82% allocated coverage for cell fields, 73% for face fields and 75% for vertex fields; the face figure is lower because the face base is (2t)³, one eighth of the h texture. That excludes pool slack, metadata, other buffers and the old-plus-new peak during growth. The face base keeps today's anchor convention at that price; a compact coarse face layout has to be compared before the layout is fixed.

**3. Base currency is unspecified.** The original rule writes the base only when a patch retires, yet the first revision lets 4h taps read the base while the patch is resident. Nothing keeps it current. Each class needs its own rule: volume is a restriction, a vertex value is a canonical selection, a face velocity is a reconstruction.

**4. Halo validity is unspecified.** "The field's generation" is not enough: volume changes across sharpening sweeps with no ownership change. Validity has to name the field version and the producing stage, including ping-pong pairs and writes that cross a patch boundary. Before any port, a table per consumer: required footprint, authoritative source, reconstruction rule, refresh dependency.

**5. Admission.** "Out of capacity is a loud fatal" conflated two cases. Exhaustion the host can see before authority changes is detected before adoption: the last accepted generation stays, and the request is reported deferred or rejected. A fatal is for an invariant violation, such as the GPU census exceeding a capacity the planner guaranteed.

**6. Gates.** Agreement of receipts is too weak as the main correctness gate. Compare canonical fields, boundary transfers and transition accounting.

### Order (supersedes the list above)

One vertical slice before any broad port:

1. **The all-4h endpoint.** Zero detail with scratch and owner-indexed buffers sized from capacity ([buffer plan](uniform-4h-first-buffer-capacity-plan-2026-10-04.md)), coarse contact ([coarse solid design](uniform-4h-first-coarse-solid-design-2026-10-03.md)), and the renderer consuming H rather than an expanded h volume.
2. **Two adjacent h patches**, with their shared face, edges and corners, real halos and every base synchronisation the consumers need.
3. **Measure that frame** against identical reference work: frame time in the real stage order, cold compile, refresh cost and peak memory.
4. **Lifecycle:** promotion, retirement, growth and two generations in flight.
5. **Occupancy curve**, then 16³ against 32³, then the rest of the port.

The accessor rewrite stays as a correctness bridge. Its compatibility needs do not fix the permanent layout.

## Domain placement, 4 October 2026: zero detail for the field textures (measured as QA; the default since the flip)

Step 1 of the order above, for the fields. Spec `domain[:checked][:parity][:unpinned]` (`UniformDetailDomain` in `uniform-detail-fields.ts`). This section was written and measured with the placement as a QA arm and identity as production; it became the default on 4 October 2026 ("The flip" below), and identity is now the comparison arm. It is the patch layout at its degenerate size, P = the lattice: no halos, no closure, no multi-patch, and it does not prejudge the patch size.

**Layout.** A field is one texture at a time. At C = 0 (the run-time h-tile capacity, `UniformMixedCapacity.fineTiles`) it is its base block; at C > 0 it is its logical h texture, the same texture identity allocates. The swap is the mode switch.

**Base currency, per class.** While the h generation is resident the base blocks do not exist, so nothing can read a stale one; the rule is a fact about allocation, not a refresh protocol.

| Class | Base block | Texel | Written | Read while C > 0 |
|---|---|---|---|---|
| cell | t³ | c = the tile's origin cell 4c | at retirement (restriction = selection of the origin cell, the canonical texel of a 4h tile) | never: it is freed after the admission fill |
| face | (t, t, 4t); `parity`: (2t)³ | (c.x, c.y, 4c.z + j.x + 2j.y + 3j.z), the four canonical texels only: the origin face texel and the three +face anchors 4c + 3e_a; `parity`: 2c + (l = 3) per axis, the other four texels zero | as cell | never |
| vertex | (t+1)³ | g = the tile corner 4g | as cell | never by the solver. The only 4h vertex data while C > 0 is `UniformCoarseVertexPhi`'s published texture, current at each published revision |
| per-tile cell (`createField(…, true)`: the split pressure's surface target and centre phi) | t³ | as cell | in place, at every occupancy: its kernels store and load tile origins only | it is the field: never admitted, never retired |
| atlas-only (`unitVelocity`) | none | a load at C = 0 is zero, a store a violation | never | never |

Checked in QA, not by comment: under `checked` a load of a non-canonical base texel raises a sticky bit (`UNIFORM_DETAIL_LOAD_VIOLATION`, cell 16, face 32, vertex 64) that the frame turns into a fatal, as a non-canonical store always does; and the storage verifies after every transition that each field is exactly one texture of the current generation and that base blocks beside a resident generation exist only as the targets of a retirement in flight. A group still bound to the other generation fails device validation, because that generation is destroyed.

**Vertex base = published coarse vertex phi: wired.** `UniformCoarseVertexPhi` creates its (t+1)³ texture with both copy usages and the frame hands it to the storage before the first layout (`adoptBase(phi, coarsePhi.texture)`): it is the vertex field's base block, the field itself at C = 0 and the target of its restriction at retirement. While C = 0 the three publish sites encode nothing (`adopted`: the kernels write the published texture in place); while C > 0 the publish runs as before. The surface-extraction lane's two contracts hold under the placement: base texel = phi[4g] bitwise, and the consumer source named `detailVertexPhi` exactly while capacity > 0.

**Lifecycle, on the capacity system.** `reserveFine` moves C; the storage follows it. 0 → C: the h textures are allocated and broadcast-filled from the bases before the remap (pass "Uniform detail admit"), the bases are destroyed. C → 0: after the phi resolve the canonical texels are restricted into fresh bases (pass "Uniform detail retire"), the swap commits after the layout submit, and the h generation is destroyed once the queue has drained. Field bytes are in the admission check (`fineReservation` adds `fieldBytesAt`), so a refusal happens before adoption: with a byte budget of 1 a request for 2048 h tiles is refused ("2048 h tiles need 53381192 bytes of capacity-sized buffers and fields beside the 4813892 in use"), capacity, storage and live bytes unchanged, and the refusal is recorded in `uniformDetail.rejected`.

**Addressing: one shader set, no recompile at C = 0.** Each access derives the mode from the bound texture, `b = u32(textureDimensions(name).x < lattice.x)`, and addresses without a branch: cell and vertex `q >> 2b`; compact face `(c.xy, c.z·(1 + 3b) + (j.x + 2j.y + 3j.z)·b)` on the clamped q; parity face `q >> b` (a canonical face's parity texel is its coordinate halved). A load a kernel marks `/*h*/` (`UNIFORM_DETAIL_H_LOAD`: it only ever runs in an h tile's stencil, so the texture is the logical one; the two regular-fine loops of the velocity samplers) is the raw load, and under `checked` a violation (256) on a base block. Measured on Dawn: zero modules and zero pipelines created after setup across zero → Fine region → Full → zero → Dynamic → zero.

**The face base.** Compact, the default, holds the four canonical texels per tile, n³/16: 16.8 MB per face field at 256³, 67.1 MB for the four. Parity, (2t)³ = n³/8, is twice that (134.2 MB) and stays selectable for A/B: its address is a pure shift, which compiles 2–3 s faster on the torus set (below) and is otherwise value-identical across the whole lifecycle (same fields, mass and tile words at every checkpoint, both checked runs clean). One texel per tile, with the three anchors in one vec4 beside the origin's bits, would be about 27 MB; not built.

### Measured

Bytes at zero detail, whole solver, mock-device census, Requested with no region (MB = 10⁶ bytes; fields in brackets):

| n | identity | packed | domain:parity | domain |
|---|---|---|---|---|
| 64³ | 34.4 (27.4) | 9.67 (2.64) | 9.30 (2.27) | 8.25 (1.22) |
| 128³ | 271.8 (218.5) | 71.7 (18.4) | 71.5 (18.1) | 63.1 (9.72) |
| 256³ | 2162.0 (1746.4) | 560.9 (145.2) | 560.4 (144.8) | 493.3 (77.7) |

At C > 0 the placement is identity plus 1,284 bytes, less the two per-tile pressure fields, which stay t³: 2·(n³ − t³)·4 bytes, 2.06 MB at 64³ and 132.1 MB at 256³. On Dawn at 64³ (measured before the per-tile fields) the live total goes 9.30 MB → 57.9 (lower-half region) → 79.6 (Full) → 9.31 and returns there after every visit; identity sits at 34.4 throughout. Peak, both generations held across a transition: 89.8 MB, the same as identity's peak.

Frame time, dam64 and Figure 9, two solvers alive on two devices in one process, frames alternated A B / B A, 12 warm + 24 timed, frame = sum of compute passes, statistic = median of the per-frame paired differences (unpaired medians are unusable when the GPU is shared). Re-timed on the code with compact faces, the h-only loads, the per-tile pressure fields and compile-by-need. Null pair identity against identity: +0.006 ms (+0.2%), quartiles −0.115 to +0.091.

| Scene, policy | identity ms | domain ms | paired difference (quartiles) | passes that moved (ms) |
|---|---|---|---|---|
| dam64 Requested, no region (C = 0) | 3.990 | 3.915 | −0.072 (−1.8%; −0.108 to −0.018) | none above 0.013 |
| dam64 Requested, lower-half Fine region (C = 2048) | 9.252 | 9.441 | +0.344 (+3.7%; +0.242 to +0.537) | pressure authority and volume correction +0.226, pressure rhs +0.050, advect +0.032, redistance +0.028, momentum +0.024 |
| dam64 Full (C = 4096) | 8.474 | 8.840 | +0.373 (+4.4%; +0.297 to +0.525) | pressure authority +0.231, pressure rhs +0.050, frame plan +0.021, advect deferred +0.019, advect +0.019 |
| dam64 Dynamic | 9.701 | 10.145 | +0.392 (+4.0%; +0.274 to +0.599) | pressure authority +0.189, extension +0.054, pressure rhs +0.048, authority phase +0.035, momentum +0.032 |
| Figure 9 Dynamic | 15.857 | 16.127 | +0.270 (+1.7%; +0.208 to +0.448) | momentum +0.073, extension +0.032, redistance +0.031, body forces +0.027, remap +0.019 |

The zero-detail frame, the app's default, is faster than identity: every access is a base access and the base textures are 12–22× smaller. With h tiles present the cost at 64³ is now 3.7–4.4%, up from 2.1–3.0% in the first pairing, and the growth is two passes that were not movers then: pressure authority and volume correction (+0.19 to +0.23 ms) and the pressure right-hand side (+0.05). Both read the two per-tile pressure fields (surface target, centre phi), the only domain-specific change to those passes since; that is the lead, not yet a measured cause. Figure 9 is unchanged within its scatter (+2.0% then, +1.7% now) and those passes are not among its movers.

Correctness on one running solver (dam64, canonical texels compared bitwise with identity at every checkpoint): zero, a Fine region entered from zero, Full and the return to zero are bitwise identical in every state field, with identical mass; at each retirement no base texel differs from the h texel it was restricted from. Mixed layouts are not bitwise: Dynamic parts at its second frame (48,495 canonical texels, largest 6.6e-7), a Fine region held from construction between frames 12 and 15 (75,845 texels). Tile words are equal at the onset; at frame 36 of the region run mass agrees to 4e-7 relative.

**The divergence is code generation, not data.** Three arms on the domain storage, same textures and lifecycle, only the accessor text changed: R, every accessor body the raw load or store (35 modules, 121 sites); N, the calls inlined with the coordinate cast (328 sites); U, the inlined text equal to identity's. All three are bitwise equal to identity in every texel, canonical or not, through Dynamic frame 3, and R and N are bitwise equal in the canonical texels of the region run through frame 18. Only the real accessor bodies part. At C > 0 they load the same texel of the same texture as the raw load (b = 0), so what differs is how Metal compiles the float arithmetic around an address that is now a function of a run-time value. `checked`, a third accessor shape, parts sooner again (Dynamic frame 1, region by frame 3) with no violation. Frame time matters more than bit identity, so this stands.

The other possible cause, a sampler reading a non-canonical texel of a 4h tile while C > 0, was measured on identity by overwriting those texels with 7777 between frames 1 and 2 of Dynamic, one field and one tile class at a time. Two combinations are read at all: the current velocity and the vertex phi, in 4h tiles within one tile of an h tile (velocity: 47,460 texels overwritten, 4,057 tile words of the next layout differ, pressure fails at frame 3; phi: 50,997 overwritten, 248,492 canonical texels differ). Tiles with no h tile in their 26-neighbourhood under either layout are never read off their canonical texels, in any of the 14 fields, and with a static region nothing is (0 canonical differences in 18 frames). So a relayout's frame-start consumers (census, remap) read last frame's resolved seam data, under either placement; the domain storage holds the same values there as identity while C > 0, which is why R, N and U are bitwise. Two consequences: real patches must keep the seam ring resident with the h tiles while deep 4h tiles need their canonical texels only (the overwrite reads as far air in a phi-like field, so the deep result is one-sided there); and at a 0 → C transition identity holds stale history in those texels where this placement holds the admission fill, by design. With a region held from construction 968 canonical texels of the two smoothed-surface scratch fields differ at t0 (far-field 27.48 against 27.66); they are overwritten by frame 3 and reach no state.

The checked runs, compact and parity, two rounds of zero ↔ Fine region ↔ Full ↔ Dynamic (54 frames) and Dynamic and a region from construction, raise no violation. The one read of non-canonical base texels found earlier, `umCutFlux` evaluated inside a `select` in `uniform-mixed-pressure-velocity.ts`, is now an `if`. The split pressure's surface target and centre phi, written by the geometry kernel at lane 0 of each tile and read through the all-4h pressure ownership, are per-tile fields: t³ at every occupancy, 132 MB less at 256³ while C > 0 and two fewer admission fills, with difference counts against identity identical before and after.

Transition cost, wall clock of the transition frame in ms, no readback inside the window, each transition taken twice in one run (first / second visit). 64³, dam64, steady frames ≈ 5 at zero, ≈ 10 with a region or Full, ≈ 12 Dynamic:

| 64³ | identity | domain | domain − identity, three runs |
|---|---|---|---|
| zero → Fine region (C = 2048) | 25.0 / 16.9 | 31.4 / 23.6 | +2 to +11, typically +7 |
| Fine region → zero | 11.1 / 11.7 | 12.1 / 11.9 | 0 to +5 |
| zero → Full (C = 4096) | 16.9 / 15.9 | 21.4 / 20.4 | +4 to +8 |
| Full → zero | 9.6 / 9.6 | 11.4 / 11.5 | +1 to +4 |
| zero → Dynamic (two frames) | 393 + 385 / 285 + 251 | 385 + 391 / 288 + 253 | none: stale launch widths, fixed below |
| Dynamic → zero | 9.0 / 10.3 | 10.8 / 13.1 | +2 to +6 |

128³ (the same scene rescaled, one run; steady frames in that run: identity ≈ 60 region, ≈ 15 zero; domain ≈ 40 and ≈ 8):

| 128³ | identity | domain | host `apply` (identity / domain) |
|---|---|---|---|
| zero → Fine region (C = 16384) | 104.1 / 89.1 | 107.6 / 97.2 | 39.6, 23.8 / 53.3, 42.6 |
| Fine region → zero | 44.6 / 159.2 | 34.5 / 34.3 | 20.6, 17.9 / 20.4, 19.5 |
| zero → Full (C = 32768) | 180.8 / 99.2 | 113.6 / 103.9 | 22.0, 23.7 / 45.3, 44.8 |
| Full → zero | 38.2 / 36.9 | 30.6 / 32.4 | 15.8, 14.3 / 16.2, 19.0 |
| zero → Dynamic (two frames) | 2203 + 2213 / 1756 + 1614 | 1925 + 1895 / 1518 + 1330 | 2.4, 1.1 / 24.0, 25.2 |
| Dynamic → zero | 28.3 / 30.9 | 39.6 / 30.1 | 15.9, 16.8 / 18.2, 15.7 |

Admission allocates and fills every h texel of every following field, so it scales with the lattice, not the request: about +7 ms at 64³, +14 to +23 ms of host `apply` at 128³ (the frame totals there sit inside identity's own scatter), and by the same scaling of order 0.1–0.2 s at 256³ beside allocating 1.75 GB; 256³ is unmeasured. Retirement is a restriction of t³ texels per field and adds 0–5 ms at 64³ and nothing visible at 128³. The cost that dominates every entry into Dynamic in these tables, two frames of 0.25–0.4 s at 64³ and 1.3–2.2 s at 128³, is the same under both placements; it was not the relayout switch but two frames of one-group launches, found and fixed afterwards (see *The two hitches* at the end): 38 + 15 / 16 + 14 ms at 64³ and 122 + 41 / 47 + 37 ms at 128³ on identity. This is the known limit of P = the lattice and the argument for real patches.

Cold compile before compile by need (the section below supersedes these figures), falling-water-torus (32³), one process, every compute entry point salted with a live statement so Metal compiles each pipeline (a dead-variable salt and an unsalted run both leave part of the cache warm, 10–20 s cheaper). Setup in s; "used" = pipelines dispatched in 48 frames:

| arm | setup | pipelines | used | never dispatched |
|---|---|---|---|---|
| identity (twice) | 111.7, 110.3 | 359 | 59.8, 58.9 | 51.4, 51.0 (159 pipelines) |
| domain (compact, twice) | 129.4, 128.3 | 364 | 62.9, 63.0 | 66.0, 64.9 (164) |
| domain:parity | 127.2 | 364 | 64.6 | 62.2 (164) |

Domain is identity + 18 s cold (parity + 16), over the 120 s ceiling by 8–9 s before the test body runs; identity has 8–10 s of margin. The target of identity + 2 s is not reachable by accessor shape. The residual by pipeline, compact against identity's mean: `advectDeferred` in its full-solids merged twin, which the torus never dispatches, 4.9 → 16.9 s (+12.0; parity +8.7); `momentumStep` +1.9; the dispatched `advectDeferred` twin +0.5; the five transfer pipelines +1.0 (new, used only at a transition); the other pipelines together +2.6. That one pipeline, compiled alone and cold: identity 4.8–5.1 s; compact 16.4–17.5; parity 13.6–14.6; every load raw and every store an accessor 4.9 (stores cost nothing); face loads raw 7.1, cell loads raw 16.0; the velocity loads of `umReleasedWalls`, `umLoadMixedFace` and `umContactReleased` raw together 7.5, each alone 17.0, 13.6 and 16.5 (superlinear in the number of addressed face loads); walls' two loads of one texel folded into one 16.6; the mode once per entry point in a private (parity) 14.8; branch, select, mask and no-clamp forms 14–19; an xy-packed compact layout 16.0. The h-only loads of the fine samplers do not touch it (129.0 s with them, 129.1 without). The reductions tried and dropped for no gain: the mode from the directory word, per access or per entry point; stores without the range and canonical tests.

Placement-independent, identity on the torus: 159 of its 359 pipelines are never dispatched and cost 51 of 111 s cold, mostly the full-solids twins of a solid-free scene. By module: forces 14.1 s unused of 19.5 (`forces` ×3 6.0, `forcesRegularCoarse` ×3 3.6, `forcesRegular` ×3 3.5); surface 13.2 of 22.3 (`advectDeferred` ×2 6.1, `advect` 1.2, `advectFine` 1.1, `redistanceFine` ×2 1.0, `traceCellsMerged` 0.9, `advectOwners` 0.9); conservative transport 5.0 (30 of 56 pipelines); pressure authority 3.3 (two modules); pressure band 2.2; sharpening 2.2; pressure hierarchy 1.8; surface volume 1.7; surface geometry 1.3; ownership transfer 1.1; pressure velocity 0.9; solid displacement 0.9; rigid bodies 0.8. The largest dispatched pipeline is `momentumStep`, 16 s. Compiling a solids twin when a scene first has solids would put the torus near 60 s under either placement and the domain residual at +3.6 s; that is a change to the solver's pipeline set; it is made in the next section.

Consumers outside the solver read a field through `lib/core/uniform-detail-abi.ts` (`udrInit`, `udrLoadCell`, `udrLoadFace`, `udrStoredVertex`, `udrLoadVertex`, `udrCellDims`, `udrVertexDims`), one shader at every occupancy: the grid overlay and its level-set volume, the secondary particles, the water pipeline; the harness readbacks go through `field.storage.read`, which returns the logical field. A compute probe of the ABI on dam64, every texel against the logical field, compact and parity: at C = 0 cell 0 of 262,144 wrong, face 0 of 262,144 (40,960 expected zero), stored vertex 0 of 274,625, interpolated vertex 0 of 269,712 (largest error 7e-8); at C > 0 (Fine region, Dynamic) 0 wrong; `udrCellDims` = 64 in both modes. Only the app can check: the overlay's render pipeline building and drawing, the renderer and overlay rebinding when a transition swaps a texture under them, and the secondary particles, which no headless path instantiates.

Lanes under `DETAIL=domain` before compile by need (compact faces, per-tile pressure fields, the adopted vertex base), one runner invocation, 8 of 8 files pass: falling-water-torus 118.8 s (ceiling 120; Metal's cache was warm for the modules whose text had not changed since earlier runs, so this is not the cold figure above), coarse-solid-rest 287.5 s (3 tests; identity 196 s), detail-policy 194.0, dynamic-coarsening 128.3, mixed-live-solid-edit 43.9, pond-rest 125.2, surface-extraction 152.3, and uniform-volume 115.8 (the overlay level-set compute probe through the ABI).

### Compile by need, 4 October 2026

Pipelines are created when a state can dispatch them, not because they exist. Each deferred pipeline (or solid twin) names its needs in `lib/methods/uniform/uniform-pipeline-needs.ts`; the predicates are the dispatch sites': `solids` (an in-domain voxel, terrain or body: the gated solid library), `solidFree` (none: the `umSolidsPresent = 0` twins), `bodies` (tile marking, coupling, the bodies-only record rebuild), `displace` (a live voxel edit or a body), `dynamic` (the census, the layout builder, the changed-tile launches), `forceCache` / `forceInline` (the force set of the surface width), `capillary` (the normal and curvature caches: cached forces with surface tension), `transfer` (domain placement: h capacity can leave or return to zero on a running state). Setup compiles, blocking as before, exactly the set the initial scene holds. No module text changed and no module is skipped.

A change whose pipelines are not built waits: the scene or values are held (latest wins), `createComputePipelineAsync` builds the set, the solver keeps advancing the accepted state, `uniformDetail.preparing` names the needs (the detail control shows "preparing"), and the change is replayed when they are ready. A first body waits the same way with the roster held empty. A voxel edit still needs no rebuild; the first one in a solid-free scene waits for a compile. Nothing falls back: a twin handle is a key that `select` resolves to a built variant or throws, the frame checks its state's needs before it encodes, and a headless caller that advances while a change waits gets a throw naming `pipelinesPrepared()` (the app's renderer opts in to presenting during a wait with `acceptPipelineWaits()`). `warmPipelines()` builds everything remaining in the background, one pipeline a round, and a waiting change is built ahead of it; the app's render worker calls it after the first presentation (`enableSolverPipelineWarmup` in `webgpu-render-worker.ts`), harness paths and lanes never do, so a lane compiles only what its states need.

Cold setup, falling-water-torus, live salt, one process, s:

| arm | before | after | pipelines | never dispatched after |
|---|---|---|---|---|
| identity (twice) | 111.7, 110.3 | 68.3, 65.7 | 359 → 224 | 24 pipelines, 4.1 and 3.9 s |
| domain (twice) | 129.4, 128.3 | 72.4, 70.6 | 364 → 229 | 29 pipelines, 5.1 and 4.9 s |

Margin under the 120 s ceiling: 52–54 s identity, 48–49 s domain, and the domain residual is +4 to +5 s. What is still compiled and never dispatched (identity, s): eight pressure-hierarchy tile-list smoother, cycle and residual variants 1.84, two `copyVolume` list variants 0.47, the rigid-body solver's `integrate` and `pickRigidBody` 0.45, two scratch copies 0.45, the pressure-authority phase build 0.30, sharpening `classify` and `classifyCoarse` 0.26, three `bandMeasure` cycle counts 0.16, the trace marker 0.10, the census `solidActive`, `solidPromote` and `budget` 0.07; under domain also the five transfer pipelines, 1.0, held from setup because Dynamic can cross zero. Their predicates are run-time sizes, not scene state, and were left. The largest dispatched pipeline is `momentumStep`, 15.8 s identity and 17.6 domain, a quarter of the cold setup.

Warm-up while frames run, dam64 Dynamic, live salt, 134 pipelines, headless Dawn on Metal; a frame is one advance, wall clock and sum of compute passes, medians:

| arm | duration | frames during | wall during / after | passes during / after | frames over 33 ms |
|---|---|---|---|---|---|
| one a round | 45.4 s | 3772 | 11.29 / 11.19 | 9.15 / 9.11 | 12 |
| four a round | 44.3 s | 3730 | 11.20 / 11.83 | 9.13 / 9.38 | 14 |
| all at once | 45.1 s | 3738 | 11.34 / 11.12 | 9.22 / 9.09 | 10 |
| control, nothing compiling, 3000 frames | 35.8 s | — | 11.22 | 9.13 | 13 |

Warm-up does not move the frame: at most +0.1 ms on the medians, and the 70–100 ms wall-only stalls every 240–260 frames are in the control at the same rate (4.3 per thousand frames against 3.8), so they are not the compile. (They are the node binding's garbage, not the solver: *The two hitches* at the end.) Metal compiles serially whatever the pace, so pacing changes nothing here; one a round is kept because a waiting change then queues behind a single pipeline. (The first four-a-round run shared the GPU with another process from 35 s in, passes 9 → 139 ms through the frames after the warm-up as well; it was discarded and re-run.) Chrome compiles in its GPU process, so the app can differ and has to be looked at.

Late changes on the running dam64, frames continuing at their normal 11.2–11.5 ms median throughout the wait:

| change | waits for | cold wait | frames meanwhile |
|---|---|---|---|
| first voxel edit, solid-free scene, identity | solids, displace (121 pipelines) | 36.8 s | 3063 |
| same, domain | solids, displace (121) | 49.8 s | 4063 |
| same, Metal cache warm | solids, displace (121) | 0.51 s | 41 |
| same, 3.5 s into the warm-up (13 built) | solids, displace (111 left) | 33.4 s, then the warm-up's last 10 in 7.1 s | 2799 |
| first rigid body, identity | solids, bodies, displace (124) | 36.0 s | 3018 |
| same, domain | solids, bodies, displace (124) | 47.8 s | 3897 |

The edit and the body land after the wait (the body falls 0.48 → 0.28 m in the next 30 frames, 0.31 under domain; no device error). Counts for the smaller waits, from the CPU census: Requested → Dynamic 30 pipelines, Dynamic → Requested 3, Requested → Full 3 and under domain 5 transfer pipelines more, an edit in a scene that already has solids 3. The cold waits are the price of a change made before the warm-up has finished on a cache that has never seen the shaders; with the cache warm it is half a second.

Lanes, one runner invocation per placement, 8 of 8 files under each (s, identity / domain): falling-water-torus 47.7 / 60.6 (cache partly warm; the cold figures are above), coarse-solid-rest 176.3 / 263.1, detail-policy 85.7 / 122.1, dynamic-coarsening 61.7 / 74.1 (its `compiled == []` assertion untouched), mixed-live-solid-edit 66.1 / 55.2, mixed-rigid-body 2.9 / 2.9, surface-extraction 57.0 / 68.8, uniform-volume 57.5 / 57.0. The live-solid-edit lane now starts solid-free in two arms, one waiting for its pipelines and one with everything compiled up front: the edit lands after the wait, advancing during it throws, and the volume field of the two arms is equal in every one of 32,768 cells just after the edit and at the end, under both placements. No lane covers a first body in a body-free scene; that is the probe row above.


### The flip, 4 October 2026, and what it left open

Flipped on 4 October 2026 at Peter's request, with his app checks (item 1) still outstanding: `DEFAULT_OPTIONS` in `uniform-detail-fields.ts` is the domain placement with compact faces (`{domain:{checked:false,compactFaces:true,unpinned:false}}`). Identity is the comparison arm (spec `identity`) and the patch atlas stays QA. The list is what was open at the flip and is still open.


1. In the app, since nothing headless covers them: the grid overlay drawing at C = 0 and across a transition, the renderer and overlay rebinding when a field's texture swaps, the secondary particles; and for compile-by-need, the warm-up after the first presentation and the three waits (first voxel edit, first body, first policy change) with their "preparing" status.
2. The two pressure passes that cost +0.23 ms (authority and volume correction) and +0.05 ms (rhs) under domain at 64³ with h tiles present. The per-tile pressure fields are not the cause, and neither is the run-time base/h switch: both were removed in process and the delta stayed (results below, "Part A item 3"). They stay pinned to tile resolution: no frame cost, 132 MB less at 256³. The delta is absent at 128³ on the same region and absent on Figure 9; what is left is that every non-identity placement runs a different trajectory from frame 1 on this layout, and the authority pass is data dependent. Unproven; it is 0.23 ms either way.
3. The hitch at 256³, unmeasured; any Fine region there allocates and fills 1.75 GB of fields. That is the degenerate patch size, and the reason this placement is the endpoint, not the layout.

4. Found after the flip, fixed in code, not yet re-run on Dawn: every scene with solids was wrong under the accessor placements. `uniform-pond-rest` (regression-no-forces arm: 34.4 mm and 0.0163 m/s in frame 1 with gravity off) and nine of the eleven `uniform-coarse-solid-rest` cases were red on the default; the one green case is the pond with no solid. Cause: the parameter block's first write was inside the t=0 authority and publication, which the accessor placements defer until the first layout is installed, while `UniformMixedFrame.updateLayout` builds the all-4h solid record and the tile cut map ahead of that layout (it does since the remap reads the record). Built from a zero block the record holds every tile uncut, the cut map answers clear, and `umCellOpen` returns 1 inside solids for the life of the frame. A bisection of the first advance (identity against domain, every field snapshotted before each pass) puts the first difference at the output of the head's geometric fill, the first solid-gated store: 1,900 texels of the fill target, 1 where identity has 0 at closed cells, with every input field equal. The pinned per-tile fields, a checked run, the t=0 seed order and the detail table were each ruled out on the way. Fix: `initializeMixedFrame` writes the parameter block before the first layout under the accessor placements, where identity's t=0 publication has already written it. Lanes owed on the default: `uniform-pond-rest`, `uniform-coarse-solid-rest`.

Closed since the first version of this section: the divergence explained (code generation); consumers through the ABI; `adoptBase` wired; compact faces the default; the `select` made an `if`; the two per-tile pressure fields; clean hitch numbers at 64³ and 128³; coarse-solid-rest re-run; the torus lane's cold margin (compile by need: 48–49 s under domain, 52–54 under identity); the paired frame-time table re-timed.


### Slice step 2: contract

Written before the code, 4 October 2026, and amended where the profile arm contradicted it (each amendment says so). The placement is a QA spec, `patch[:P][:edge][:r…][:cap<slots>|:all][:guard|:profile|:trust][:checked]`, beside `identity`, `packed`, `local` and `domain`; it holds one layout (the first), so promotion, retirement and growth are step 4. One stated answer per rule.

**Residency.** A patch is resident when it holds an h tile, and for no other reason: no closure, no dilation, no rounding beyond the patch grid. The seam ring (4h tiles within one tile of an h tile) gets no slots. Its canonical texels are base texels like any other 4h tile's. *Amended:* its non-canonical vertex phi and atlas velocity are read on a static layout too (the profile arm measured fixed reaches of 4 texels in both classes), so they are stored, in the halo: vertex and atlas fields carry a halo of 4 texels, one tile, which is the ring around a resident patch. Cell and face fields carry 1.

**Storage.** One physical texture per field: the base block unfolded at the texture origin (cell t³, face (2t)³ parity texels, vertex (t+1)³), then slots of pitch P + 2r (vertex P + 1 + 2r) from z = base depth. A slot holds its patch's interior at offset r and a halo of r texels on every side.

**Halos replace closure.** A work item reads its home entry once (the directory word of its tile's patch, zero for a tile outside every resident patch) into private origins and a box (interior plus halo). The arithmetic load (form G) has no branch: `select(base(p), origin + p, p in box)`, one `textureLoad`. It is correct when p is in the home box, or p is canonical (the base is current, next rule). What is left is an h texel outside the home box, which three kinds of site ask for: a data-dependent sampler, a 4h work item beside a patch reading its h neighbour, and (*amended*, found by the edge and corner arms) a work item with no home patch reading a cell or face texel that only a neighbour's low-side halo holds. Those sites keep the guarded load (form F): the same address, then, only when p is out of the box and not canonical, the home patch's directory word and after it the up to seven neighbours whose halo can hold p. Which sites they are is measured, not asserted: `patch:…:profile` records per load site which of those cases ran and how far past the box a tap went, and `patch:…:trust` compiles a site G only when its profile is clean.

The home entry comes from the owner functions of each kernel's source (they call `udHome` with the work item's tile). Tile-job kernels with no owner function (band, to-pressure, cache, unit velocity taps, resolve, live gather) are base-homed: every h read they make is a directory read.

Halo width is per class, from the widest fixed stencil the profile arm sees leave the box non-canonically. *Measured* (the first draft expected 1 / 1 / 1): cell 1, face 1, vertex 4, atlas 4. The momentum cache's `extended` field is its own class (atlas: no base, halo 4). The data-dependent samplers reach further (momentum `extended` 5 to 6 texels past a 4-texel halo, the advect departure taps 2 to 6) and are F at any halo that is not the whole patch. r is a spec parameter (`:r<n>` or `:r<cell>.<face>.<vertex>.<atlas>`).

**Halo validity.** By write-through, not by refresh. A halo texel is a replica of a texel some other patch (or the base) owns, and every store writes every replica: the home texel, the same texel in each resident patch whose box holds it (up to seven more, found from the directory), and the base texel when canonical. A store strictly inside its home patch's interior by more than r writes one texel and no directory is read; any other store walks the home patch and the neighbours its position can touch (a submask loop over at most seven, each one directory word). So a halo is valid after every store and before every load, in any stage order, including ping-pong pairs and the sharpening sweeps: **refresh launches per frame: 0.** The cost moves into the stores of the outer r texels of each patch and of every base-homed tile, and is measured in step 3. The alternative, a refresh launch after each writing stage, is rejected for this slice: it needs one launch per written field per writing pass (two for a face field, which cannot be read and written in one pass), and a missed one is a silent stale read.

**Base currency.** One rule for every based field (volume, velocity and its scratch, vertex phi and the other cell, face and vertex fields): the base holds every canonical texel of every tile at all times, resident or not, written by the same store as the patch texel. The readers that force it: a 4h work item reads its neighbour's canonical texel through the base whether or not that neighbour's patch is resident, and an h work item reads a canonical texel beyond its halo the same way. Nothing is restricted at retirement; a retiring patch is dropped. Atlas-only fields (unit velocity) have no base.

**Patch size.** P is a spec parameter (16, 32); nothing in the accessors or the host depends on its value.

**Exhaustion.** The patches a layout needs are counted on the host before `reserveFine`; a layout past the slot capacity returns a refusal through the same path (`updateLayout` returns it, the solver publishes `uniformDetail.rejected` and stays on its last generation). A store that lands nowhere (not canonical, in no resident box) sets a violation bit and the frame receipt throws: that is the invariant. *Amended:* a later layout whose patch set differs from the first is refused the same way ("haloed patches hold their first layout"), since step 2 has no admission fill.

**Checked arm.** *Amended naming.* `patch` (guard, the default) compiles every load F. `patch:…:profile` compiles every load in a recording form and writes the per-site profile to the support buffer. `patch:…:trust` compiles the profile's clean sites G and the rest F. `:checked` on any of them raises a violation (cell 16, face 32, vertex 64 for loads; 1, 2, 4, 8 for stores) when a G site leaves its box non-canonically or an F site finds no texel, and the frame receipt throws. `patch:…:all` makes every patch resident with the same shader text: the bitwise reference arm. There is no fallback to a logical texture: a load that is in no box, not canonical and in no resident patch has no texel.

**Not built in step 2.** Any change of layout after the first (admission fill, retirement, growth, two generations); Dynamic; consumers outside the solver (the ABI in `lib/core/uniform-detail-abi.ts` has no halo offset, so the overlay, renderer and particles are not ported; the harness reads through `storage.read`).


### Slice steps 2 and 3: results, 4 October 2026

**Two real patches do not hold the +2.0 to 2.5% address cost. The exact arms cost +26 to +40% of the frame.** Memory is proportional to detail: the two-patch region takes 32.5 MB of solver at 64³ against 45.0 under domain, and by census 700 MiB against 1,960 at 256³.

Scene: dam64 (`minimal-power-dam-break-64`), Fine regions in cells. FACE `0,0,0,64,32,32` (1,024 fine tiles: two face-adjacent P = 32 patches, sixteen P = 16), EDGE `0,0,0,32,32,32;32,0,32,64,32,64`, CORNER `0,0,0,32,32,32;32,32,32,64,64,64`, PART `8,0,0,56,24,40` (720 fine tiles, 4h tiles inside resident patches). 128³: the same FACE box. Four Dawn processes; the second was lost to a compile blow-up, and the accessor text changed once between the third and the fourth (the neighbour search for cell and face, a submask loop in place of eight tests, and the guarded load taking the arithmetic address first). "Old" and "new" below name those two texts.

#### Frame time (paired, per-pass timestamp queries, median paired difference against identity on the identical layout)

| arm | 64³ FACE, new text | old text | load forms (G / F) | exact |
|---|---|---|---|---|
| null (identity twice) | +0.037 ms (0.5%) | +0.079 | | |
| `domain` | +0.366 ms (4.7%) | +0.355 | | yes |
| `patch:32` (guard) | **+2.864 ms (36.4%)** | +2.966 (37.0%) | 0 / 143 | yes |
| `patch:32:trust` | +2.596 ms (33.0%) | +2.934 (36.6%) | 54 / 89 new, 58 / 85 old | old yes; new carries one wrongly arithmetic site (below) |
| `patch:32:all:trust` | +2.250 ms (28.6%) | +2.535 (31.6%) | the same | the same |
| `patch:16:trust` | **+3.098 ms (39.5%)** | noisy pair, discarded | 17 / 126 | yes |
| `patch:16:all:trust` | +2.629 ms (33.6%) | | 17 / 126 | yes |

Identity is 7.86 ms. 128³, the same 1,024 fine tiles (identity 10.96 ms): `domain` +0.067 ms (0.6%; +0.146 in the earlier process), `patch:32:trust` **+2.935 ms (26.8%)** new text with 20 G sites, +2.849 (26.1%) old text with 67. The cost is the same number of milliseconds at 64³ and 128³: it follows the h work, not the grid, and not the number of arithmetic sites either (0, 20, 58 or 67 G sites all land within 0.3 ms of each other).

Movers, `patch:32` guard at 64³ (ms, identity's pass median in brackets): momentum +0.846 (0.408), surface redistance +0.532 (0.572), advect deferred +0.291 (0.179), surface advect +0.276 (0.272), pressure authority and volume correction +0.251 (0.180; +0.227 of it is in `domain` too), traceCells +0.124 (0.082), body forces +0.117 (0.139), transport +0.060, rhs +0.058, extension +0.050. The pressure cycle and band solve are not on the list. At 128³: momentum +1.177 (0.758), redistance +0.574, advect +0.288, advect deferred +0.224, body forces +0.134, traceCells +0.115.

**Refresh launches per frame: 0.** Launches per frame are 50 at 64³ and 54 at 128³ in every arm; halos are kept by write-through.

#### The address-form check

Real halos brought the branch back, at the sites that carry the frame.

- With every load arithmetic (`trust@*`, inexact, timing only, since the samplers then read wrong texels), momentum is +0.109 ms against +0.82 with its samplers guarded, redistance leaves the top ten against +0.54, advect deferred is +0.125 against +0.25. So an arithmetic site with real halos is at the level the spike measured, and the write-through stores are not the cost. The whole frame of that arm is not a bound: its state diverges (5% more volume) and the transport live set and sharpening do more work, so it still reads +2.397 ms.
- The guarded form costs the same whether its fall-through is a branch chain with one directory read (old text) or a `select` plus one rare branch holding the neighbour loop (new text): +2.97 against +2.86 ms.
- It costs most of that even when the fall-through is almost never taken: with every patch resident (`:all`) the same text is +2.25 ms. The price is the guard's presence in the inlined kernel, not directory reads executed.
- Patch size and the share of stores on the slow path do not matter per pass (P = 16 against P = 32: momentum +0.90 / +0.85, redistance +0.60 / +0.53).

The sites that cannot be arithmetic under this contract are exactly the hot ones: the data-dependent samplers (momentum `extended` reaches 5 to 6 texels past the halo, the advect departure taps 2 to 6), every site a 4h work item shares with an h work item (the 4h item reads its fine neighbour through the directory), and every site in a base-homed kernel. The profile-then-trust discipline therefore buys nothing as built: 89 of 143 load sites stay guarded and they include the samplers.

#### Correctness

`patch` against `patch:all` (identical shader text, every patch resident), canonical fields and every live texel, `checked` on both sides:

| layout | text | frames | result |
|---|---|---|---|
| FACE P = 32 | old | 36 | bitwise, no violation |
| PART P = 32 | old | 36 | bitwise, no violation |
| FACE P = 16 | old and new | 36 | bitwise, no violation |
| FACE 128³ P = 32 | old and new | 12 | bitwise, no violation |
| EDGE P = 32 | new | 36 | bitwise, no violation |
| EDGE, CORNER P = 32, unchecked | new | 36 | bitwise (both sides share the wrongly arithmetic site below, so this shows the stores and guarded loads are layout independent, not that the arm is exact) |
| FACE, PART P = 32 | new | 3 | **violation 32** (one wrongly arithmetic site; not an addressing fault) |
| CORNER P = 32 | new | 19 | the same site |
| EDGE P = 16 | new | 1 | **violation 32** (six wrongly arithmetic sites; not an addressing fault) |

The edge and corner arms earned their place: under the old text both threw at frame 1, because a cell or face texel of a non-resident tile can live only in a neighbour's low-side halo, and a base-homed work item reading it searched no neighbour. The new text searches neighbours for every class.

The three red rows are the trust discipline failing, not the storage, and each was reconstructed on the CPU from the saved profiles:

- Profile keys are a hash of the module's source. A concurrent edit to a shared prelude between two processes changed the hash of 20 modules, their saved profile sites silently fell back to guarded, and the merged profile then covered those modules only from two fresh 12-frame runs. One site that is dirty over 36 frames (the advect kernel's `velocity` load, face class) was clean over 12, compiled arithmetic, and left its box at frame 3. Every P = 32 `trust` arm of the fourth process carries that site, which is why its end state differs from the guard arm's; the guard arm, the P = 16 arms and the 128³ arm are exact (identical end state across guard, trust, `:all`, P = 16 and P = 32 in the third process).
- The P = 16 EDGE profile was recorded under the old text, where the low-side texels above were "no texel anywhere", and the clean rule let a no-texel tap through ("garbage in either form"). Under the new text those taps find a texel through the neighbour search, so the six face sites must be guarded. The rule now treats a no-texel tap as dirty (`uniformDetailSiteClean`); that one-line change is not re-run on the GPU.

So: profile keys must survive unrelated source edits (module label and ordinal) or a stale profile must be refused outright, and a profile needs the full frame window of the run it licenses. Not covered on the new text with a complete profile: FACE and PART P = 32 checked, EDGE P = 16.

Against identity the comparison is not a tolerance statement. Every non-identity placement leaves identity at frame 1 on this layout (pressure residual 0.8995 against 0.9020; 65 canonical texels of the transport gamma field 0 against 15/64; velocity 2.5e-3), with or without the host's sync, and the trajectories separate from there. `domain` and `patch` agree with each other at frame 1 to 2.7e-7 and 2e-6 in two fields, with volume and residual identical. The cause of the frame-1 difference from identity is accessor code generation by the earlier analysis; it was not re-derived here.

#### Exhaustion

On Dawn: `patch:32:cap1` with a two-patch layout publishes `uniformDetail.rejected` = "the layout's h tiles need 2 detail patches of 32³ h; the pool holds 1"; a second layout under `patch:32` publishes "…need 2 detail patches other than the 1 resident (haloed patches hold their first layout)". Generation stays 1, no device error, the solver keeps advancing on its last layout.

#### Cold compile (salted, compile-by-need set: 197 pipelines, 56 modules)

| arm | old text | new text |
|---|---|---|
| identity | 64.4 s | 63.4 s |
| domain | 65.5 s | |
| `patch:32` guard | 117.4 s | |
| `patch:32:trust` | 122.7 s | **192.5 s** |

Metal flattens every call, so compile time follows the inlined text, and loops in it cost more than their characters: identity expands to 16.5 M characters, all-arithmetic to 18.7 M, the old guarded text to 24.0 M, the new to 25.8 M. The neighbour loop at every guarded load is what took 123 s to 193 s.

#### Memory

Measured, whole solver (field textures in brackets), FACE region:

| | identity | domain | `patch:32` | `patch:16` |
|---|---|---|---|---|
| 64³ | 47.1 MB (27.4) | 45.0 (25.3) | **32.5 (12.9)** | 36.9 (17.2) |
| 128³ | 299.4 MB (218.5) | 282.9 (202.0) | **127.1 (46.2)** | |

By census (MiB, whole solver; field bytes in brackets where they differ in kind):

| | identity | domain | `patch:32` | `patch:16` |
|---|---|---|---|---|
| 128³, two-patch region | 286.6 | 270.8 | 122.2 | 122.4 |
| 128³, `0,0,0,128,64,64` | 357.2 | 341.4 | 246.8 | 279.8 |
| 256³, two-patch region | 2,085.7 (1,746) | 1,959.7 (1,614) | **700.1 (294)** | 723.7 (318) |
| 256³, `0,0,0,256,128,128` (65,536 fine tiles) | 2,845.1 | 2,719.1 | 1,963.6 (822) | 2,227.2 (1,099) |

P = 16 pays for its halos: a quarter of the lattice Fine at 256³ costs 264 MiB more than P = 32, and its frame is no cheaper. Step 5 has no reason yet to prefer 16.

#### Part A item 3: the +0.23 ms under domain

Refuted twice. (a) Per-tile against h-resolution for the two pressure geometry fields: +0.399 ms pinned, +0.419 unpinned, the authority pass the same in both. They stay per-tile (132 MB less at 256³ for no frame cost). (b) The run-time base/h switch replaced by a constant in the authority kernels (+0.400, authority +0.227) and in every module (+0.274, authority +0.229, rhs +0.046): the authority delta does not move. Allocation, bind groups and launches are identity's. At 128³ on the same region the authority delta is −0.008 ms.

#### What step 2 leaves open

For the address cost, a step 2b before any port, in this order:

1. Give the base-homed kernels a home (they are tile jobs: the tile is known), so their h reads stop being directory reads.
2. Compile h work and 4h work as separate pipeline variants where they share a kernel, so the h variant is arithmetic and only the 4h variant keeps the guard. It removes the guard from the fixed taps of the h kernels and doubles the pipelines of every shared kernel, so it is weighed against cold compile.
3. For the data-dependent samplers, one box test per sample (the sample's whole footprint in the home box, else the guarded path for that sample) instead of one guard per tap; or a reach halo on the two fields they read.
4. Take the neighbour loop out of the load path (a single lookup for halo-held texels, or homes for the work items that need it), which also gives the compile time back.

If those do not bring the exact frame within a few per cent of domain, the layout question in the review is answered against per-patch halos for the sampled fields.

Step 4 needs, none of it built: admission fill for a newly resident patch (its interior from the relayout's remap, its halo from its neighbours and the base, and its neighbours' halos of it); retirement (drop, with the base already current); growth of the slot texture; two in-flight generations of the directory and the slots a retiring generation still reads; directory updates on the GPU rather than at setup; the profile, if it survives, keyed so it holds across layouts; the consumers outside the solver, whose ABI has no halo offset; and Dynamic, which changes the patch set every relayout and reads the seam ring while it does.

### The two hitches, 4 October 2026

Both were measured on the identity placement, production at the time, while timing the slice. One is a solver defect and is fixed; the other does not exist outside node.

**Entering Dynamic.** Direct launches are sized from completed-frame evidence (`uniformBufferedWork`: the count of two frames ago with 25% headroom, never less than one group; the kernels stride the live GPU list, so an undersized launch is complete but serial). A host layout primes those budgets from its own counts in `updateLayout`. A relayout attach primed nothing: from all-4h the pressure band's budget was one slot and the ownership's h-tile budgets one group, and the receipts that would correct them describe the first Dynamic frame and arrive two frames later. For those two frames the census admitted its h tiles (1,632 at 64³, 8,901 at 128³) and the band its rows (933, 4,869) while every band launch ran one workgroup wide. Per-pass timestamps, second visit so nothing else is new: the band solve 217 and 176 ms at 64³ and 1,224 and 1,120 ms at 128³ against 1.3 and 3.6 steady, band list and rows 13 and 73, band projection 5 and 27. On the first visit the h-tile launches of extension, transport, redistance, momentum and advection add another 300 ms a frame at 64³ in the same way; an arm with every pipeline built ahead shows the same 573 + 523 ms, so it is not compile. Host cost of the switch is 1–3 ms: 8 buffers (47 MB at 64³, 274 MB at 128³) and 23 bind groups, no blocking call. Entering from Full or from a Fine region never showed it, because those leave budgets at or above the need.

The fix is in `setRelayout`: on attach the band, ownership and sharpening budgets go to their ceilings, and receipts of frames up to the attach are not observed (they count the host layout that was replaced), so the first receipt applied is the first Dynamic frame's. Launch widths only: fine and band tile counts after each transition are identical before and after.

| zero → Dynamic, frames 1 + 2, wall ms | before | after | steady |
|---|---|---|---|
| 64³ first visit (30 pipelines built by need, outside the frames) | 591 + 523 | 38 + 15 | 14 |
| 64³ second visit | 250 + 207 | 16 + 14 | 14 |
| 64³ Dynamic from construction | 147 + 133 | 34 + 12 | 13 |
| 128³ first visit | 1,876 + 2,022 | 122 + 41 | 38 |
| 128³ second visit | 1,365 + 1,248 | 47 + 37 | 37 |
| 128³ Full → Dynamic (never affected) | 63 + 54 | 53 + 34 | 29 |

What remains on the first frame of a first visit is outside the passes (64³: 38 ms wall, 25 in passes; 128³: 122 wall, 43 in passes) and is absent on the second visit; first use of the pipelines just built is the likely cause and is not verified. The same mechanism inside steady Dynamic is the next paragraph.

**Decayed budgets in steady Dynamic.** Staged before it was fixed, one process, identity placement (the process started before the flip), GPU pass time. Under the app's importance settings a still pool holds no h tile and the budgets decay to one group by about frame 16. A Fine region drawn then admits its tiles at once and the next two frames run serial: 511 and 503 ms against 7.7 steady at the 64 class, 3,662 and 3,655 ms against 22.5 at the 128 class (band solve 2,038, transport 600, extension 374, advection 138, band list and rows 122, redistance 111, momentum 81). It does not happen under the declared defaults (shape tolerance 0: the budgets never decay), and a physical impact did not show it. With a floor of 1,024 groups the same two frames are 13.8 and 16.7 ms (64 class, the second noisy) and 42.3 and 24.0 ms (128 class); at the ceilings 11.3 and 8.3, and 39.9 and 22.5.

The fix is a reserve, not a tuned floor on every launch: `uniformBufferedWork` takes a reserve below which a budget does not decay, and `advance` passes `UNIFORM_WORK_RELAYOUT_RESERVE` (1,024 groups) only while a GPU relayout owns the layout, since only the census admits tiles with no host evidence. Zero detail and Requested pass no reserve and launch as before, so they pay nothing by construction; `tests/uniform-buffered-work.test.ts` still holds, the decay to one group being the no-reserve case. Launches stay fixed and direct. Idle cost, paired against the old budgets in the same process (median paired difference, with the null pair):

| at rest, ms a frame | reserve 256 | reserve 1,024 | ceilings | null pair |
|---|---|---|---|---|
| zero detail, 64 class (a blanket floor's cost; the reserve is not applied here) | | +0.17 ± 0.10 | +0.42 ± 0.14 | |
| zero detail, 128 class (a blanket floor's cost; the reserve is not applied here) | | +0.23 ± 0.30 | +0.46 ± 0.26 | |
| Dynamic, 64 class (4.21 ms) | | +0.47 ± 0.30 | +0.65 ± 0.34 | +0.02 ± 0.16 |
| Dynamic, 128 class (5.72 ms) | +0.21 ± 0.09 | +0.25 ± 0.04 | +0.61 ± 0.05 | +0.03 ± 0.05 |

The arms were timed by the probe setting the budgets; the landed form (the reserve passed from `advance`) has not itself run on Dawn, and nothing here was timed under the domain placement. A lane assertion for the attach fix is written and not yet run: `uniform-detail-policy` fails when frame 1 or 2 after zero → Dynamic exceeds 5 times the steady frame in GPU pass time (after the fix 2.5–2.8 and 1.4 times at 32³; before it 9–10, and 13–16 after twelve zero frames; calibrated on identity).

**Periodic stalls.** The 70–150 ms wall-only stalls are V8 major collections in the node process, driven by the Dawn node binding and not by the solver or the probe. With a device alive the binding pumps its event loop through the global `setImmediate` 45,000–65,000 times a second, passing a newly created native function each time; a bare device with no solver and no pending operation does it at 65,000/s. Each costs about 620 bytes that scavenges do not reclaim (a sampled heap profile of 320 Dynamic frames attributes 103.8 of 105.2 MB to `processImmediate`), so old space grows about 30 MB/s until the collector's limit, a 30 ms mark-compact and a further 40–60 ms of finalizers land in one frame. The period is in seconds, not frames: 3.5–3.6 s unpaced (315 frames) and 3.5 s with 8 ms of sleep between frames (175 frames). It is the same under Requested with a Fine region (89.9, 83.5, 83.3 ms at 3.5 s) and at zero detail (76 and 146 ms, 5.5 s apart at 33,000/s), and with no timestamp queries or probe readbacks. Rerouting the binding's `setImmediate` through a 1 ms timer removes it (737 calls/s, no major collection in 10 s, heap flat, maximum frame 32 ms) at +3.5 ms a frame of completion latency; an explicit collection every 60 frames spreads it (maximum 23 ms). The solver's own host path, run on a mock device, leaves about 1 KB a frame for old space. Chrome has no such pump, so the app does not have these stalls; headless Dawn timings should read medians, and a probe that needs clean maxima can collect explicitly.

## Requested on the GPU-built layout, 4 October 2026

Requested and Full built the layout on the host (`updateMixedRegions`, `placeMixedBodies`); Dynamic builds it on the GPU (census, then `UniformMixedLayoutBuilder`). A free rigid body over a host layout with 4h tiles threw, because a host layout cannot follow a body the GPU integrates. The aim is one layout path: Requested is Dynamic with every importance criterion off, so the census builds from requests only (Fine regions, the solid-contact request, bodies).

### What differs between the two paths (read from the code, before any change)

| | Host layout (Requested, Full) | GPU layout (Dynamic) |
|---|---|---|
| Capacity | Constructs at C = 0; `updateLayout` reserves the layout's exact h count (`reserveFine`, growth 2×, returns to a need under a quarter of what is held). The band holds every h tile (`liquidBand` off). | Constructs at C = T; `setRelayout` reserves T and sizes the band at its liquid bound (`capacityOf`: at most max(4096, T/2) slots). |
| Admission | Before adoption, on the host: `fineReservation` refuses a capacity the device cannot hold, the accepted generation keeps running, `uniformDetail.rejected` names the refusal. | None. The builder knows the h count `f` in `scan`, before `scatter` writes a tile word, but a count over capacity only raises the sticky `fineCapacity` bit: frame failure 6. |
| Remap | `remap.apply(layout)`: a transient target ownership, `markChanged` scans the lattice, two submissions between frames. | `remap.applyGpu` in the frame's first encoder: `markListed` over the build's dilated changed tiles, the census extension remapped too. Same cell and face kernels. |
| Detail storage (domain) | `detail.reserve(layout)` follows the capacity; a capacity that returns to zero retires in the same call (`encodeRetire`, `commitRetire`). | `detail.reserveAll()` once at attach; C never returns to zero while attached. |
| What is h | Fine regions (or every tile), every solid-contact tile whether wet or dry (`solid.forced`), the tiles the roster's poses can touch. | Fine regions and focus as builder statics, the census band (criteria, departure boxes, joins, source plugs), liquid-conditional solid and body promotion (`solidActive`, `solidPromote`). |
| Bodies | Host tiles from roster poses; a free body is refused. | `setBodies`: tiles marked from the GPU poses every census; h only where the body meets liquid. |
| 4h surface (`uniformDetailCoarseSurface`) | A frame parameter of the settings (Requested: always travel-gated). Unchanged by who builds the layout. | Same parameter (Dynamic: unless shape at tolerance 0). |
| Coarse solids (`uniformDetailCoarseSolids`) | Requested with solid contact off: static solids force nothing, cut 4h owners. | Not reachable: Dynamic always promotes from `solid.coupled`. |
| Frame head | Plan, geometry, authority, extension. No census. | Census (13 kernels and the cube levels), builder (6), admit, remap, resolve, retire, widths, plan, geometry over the changed tiles, authority; the tail extends for the next census. |
| Pipelines | None beyond the frame. | The `dynamic` need: 30 pipelines (census, builder, changed-tile launches) a Requested → Dynamic switch waits for. |
| Host mirror | `ownership.layout` is current. | None: launches are GPU counted, the tier counts reach the host in a receipt two frames late. |

Three things in the census are not requests and must be off under Requested: the six criteria (already gated by `criterionOn`), the hold and the budget, and the source rule in `decide` (a tile the drop or the inflow plug can fill is band), which has no switch today.

### Plan

1. **Requests only.** `uniformDetailImportance` returns no criterion, no hold and no budget unless the policy is Dynamic, with a `sources` flag the census reads in `decide`. Builder statics are the plan's h tiles (Full: every tile); the census solid mask is `solid.coupled` when solid contact is requested and empty otherwise; Coarse regions mask nothing (there is no automatic source to mask). Bodies arrive through `setBodies` as under Dynamic.
2. **Capacity under C < T: count, then admit.** The census can know its need before it commits. `widths` counts the h tiles the build asks for; a new `admit` kernel, between `widths` and `classify`, compares that count with the admission capacity and, over it, rewrites the staged widths to the generation the build is against and empties the changed list. The build then seals the unchanged generation (no remap, no generation bump) and the receipt carries the need. The host reads the receipt as it already does (never awaited), grows the capacity between frames, and the next build admits; a growth the device refuses is published as `uniformDetail.rejected` while the accepted layout keeps running. The `fineCapacity` fatal bit stays: after `admit` it can only mean the kept generation itself is over capacity, a true invariant violation. Dynamic passes no admission capacity (C = T; nothing to defer).
3. **Host bound first.** Requests the host can count are reserved before the build that needs them: the plan's tiles, `solid.forced` when contact is requested (the liquid-conditional promotion is a subset), and the tiles the roster's poses can reach. Deferral is then the path for what the host cannot see (a body moving faster than its roster pose says).
4. **Zero detail stays compact.** With no request (no Fine region, no body, no solid contact over solids, not Full) no relayout is attached: no census, no `dynamic` pipelines, C = 0, base blocks only. The frame's instruction stream at zero detail is today's.
5. **Idle builds.** A layout that only the host can change (Fine regions, Full, no body, no contact over solids) is built once per change; the frames between run the head without the census, as the host layout did.
6. **Capacity return.** Shrinking under a GPU-built generation is not safe from a receipt two frames old. A host request that lowers the bound enough to return storage (to zero, or below a quarter of what is held) is applied by the host layout builder between frames, as before, and the GPU builder resumes on it. The t = 0 layout is the host's too: the uploads are h data and are deposited once at h on it.

### Results, 4 October 2026

Landed as planned. `updateMixedRegions`, `placeMixedBodies`, `syncMixedRelayout` and the free-body throw are deleted; `syncMixedLayout` is the one entry. Evidence is one Dawn device per probe, pre-change solver (a scratch copy of the reference and the frame) against the tree's, frames interleaved in rotating order, GPU pass time from timestamp queries.

**Zero detail is unchanged.** dam64, Requested, no request, 90 frames:

| | before | after |
|---|---|---|
| h capacity, storage | C = 0, base blocks only | C = 0, base blocks only |
| allocated | 7.63 MB | 7.63 MB |
| passes per frame | 49 | 49 |
| pass time, median | 4.210 ms | 4.213 ms |
| paired difference | | +0.018 ± 0.027 ms (null pair, two instances of the tree: −0.003 ± 0.034 ms) |

The mock-device instruction stream at zero detail is digest-identical to the pre-change one.

**Fine regions and Full run the same through the GPU builder.** dam64 dam break, a 0.4 × 0.3 × 0.4 m Fine region added live, then Full, then back to zero:

| | h tiles (before, after) | tile words that differ | allocated | passes | pass time before | after | paired |
|---|---|---|---|---|---|---|---|
| Fine region | 384, 384 | 0 | 36.91 MB both | 50, 50 | 6.875 ms | 6.815 ms | −0.026 ± 0.042 ms |
| Full | 4096, 4096 | 0 | 73.21 MB both | 50, 50 | 8.656 ms | 8.682 ms | +0.041 ± 0.066 ms |
| zero again | 0, 0 | 0 | 7.63 MB both | 49, 49 | 3.981 ms | 4.033 ms | +0.009 ± 0.077 ms |

The frame after a change carries one census and build (60 passes against 52). The first Fine region or first body under Requested now waits for the `dynamic` pipelines (and `transfer` under domain placement), 28 more than before; Full does too.

**A free body works.** Still pool, 0.8 × 0.275 × 0.8 m at 64³, Requested, zero detail; a 0.2 m crate at 600 kg/m³ dropped from y = 0.6 m:

- Before the body: C = 0, 7.63 MB, 49 passes, 4.12 ms.
- The body waits for `solids, bodies, displace, dynamic, transfer`. In the air it has no h tile (the promotion is liquid-conditional); the census runs (70 passes, 10.8 ms) with capacity reserved at the host's count (900).
- The h set appears when the body's reach (bounding radius, the frame's travel, one cell) meets wet tiles, peaks at 1,072 on entry and settles at 750 to 800 around the floating crate (19% of the domain); no h tile was ever beyond the body's reach. 17.3 ms per frame.
- It couples: the crate plunges to y = 0.172 m, rises and floats at 0.27 to 0.28 m; the level away from it rises 5.5 to 8.4 mm (7.5 mm displaced); owner mass drifts −3.5e-4 over 60 frames (Dynamic, same drop: −4.5e-4), no device error, no fatal.
- Body removed: one cleanup frame (59 passes), then C = 0, base blocks only, 7.65 MB, 49 passes, 3.9 to 4.1 ms, mass unchanged.
- The same drop under Dynamic: 1,200 to 1,250 h tiles, C = 4096, 74.2 MB, 19.2 ms.

**Count, then admit, on a real device.** The roster the host counts from is not the GPU's pose: on entry the census needed 1,177 tiles against 900 held, deferred one build (the accepted layout ran one more frame), the receipt grew the capacity to 1,800 and the next build admitted. With the host's count cut to one tile the capacity followed the receipts 1 → 48 → 96 → 259 → 557 → 1,114 over five frames with no h tile, then admitted 1,094; no fatal, mass drift −1.8e-4. A deferred frame runs the body over 4h tiles: bounded by the growth rule (the larger of the need and twice what is held), not eliminated.

**Solid contact under Requested is now liquid-conditional** (the app default on a scene with solids). Garden hose ×10, 20,736 tiles:

| | before | after |
|---|---|---|
| h tiles | 6,992 (every contact tile) | 1,284 to 1,308 |
| capacity, allocated | 6,992, 232.74 MB | 6,992, 232.74 MB |
| passes per frame | 58 | 67 (a census every frame) |
| pass time, median | 20.96 ms | 17.29 ms (−3.90 ± 0.25 ms paired) |

The capacity is still reserved at the host's count of contact tiles, so the bytes do not follow the h count. Reserving from the receipts instead is the next step for memory and needs a rule for the first frames, when a cut tile with liquid would run at 4h.

**What keeps the host layout builder** (`hostMixedLayout`, one function):

1. The t = 0 layout: the uploads are h data, deposited once at h.
2. Zero detail: nothing is attached, so there is no census to build with.
3. Capacity return: to zero, below a quarter of what is held, and on leaving Dynamic. A GPU-adopted generation cannot be shrunk from a receipt two frames old.
4. The `patch` QA arm: its detail storage cannot grow to every patch, so requests never attach under it.

Removing 3 needs a staged admission cap (lower the cap, wait for a build that confirms the count under it, then retire storage). Leaving Dynamic for Requested keeps C = T as before (the reservation only returns below a quarter): 368 MB on the garden against 232 MB entered directly.

Lanes: `uniform-mixed-rigid-body` (new Requested arm: zero detail, a live free body, h tiles within reach, back to zero), `uniform-detail-policy`, `uniform-surface-extraction`, `uniform-mixed-live-solid-edit`, `uniform-coarse-solid-rest`, `uniform-pond-rest` pass.

### Live edits, contact count and early body tiles under Requested, 5 October 2026

**A live voxel edit under Requested joins the next census.** Requested is the app default, and before this the edit had no join outside Dynamic. With solid contact on, the next census already promoted the edit's wet tiles (coupled ∧ wet, plus the ring) and the displacement ran after the adopt in the same frame head. With contact off at zero detail nothing was attached: the displacement moves h owners only (`uniform-mixed-solid-displacement.ts` returns for width ≠ 1), so liquid in the new solid's 4h owners stayed there. Now the edit's touched tiles and their liquid-conditional promotion are joined to one census under Requested and Full as under Dynamic (`mixedEditJoin`): the edit waits for `solids, displace, dynamic, transfer` the first time, the relayout attaches at C = |join| (100 tiles for a fill, 27 for a clear on dam64), the join frame runs census → build → remap → phi resolve → displacement, and the sync after it lets the tiles go.

`uniform-mixed-live-solid-edit` has two new Requested arms at zero detail (64³ dam, a block filled into the liquid, then cleared):

| | contact off | contact on (a scene with no other solid) |
|---|---|---|
| before the edit | 0 h tiles, C = 0 | 0 h tiles, C = 0 |
| the frame after the fill | 80 h tiles; liquid inside the block 0; mass 6144.0011 against 6144.0009 | 80 h tiles; liquid inside the block 1.8e-7 |
| five frames later | 0 h tiles, C = 0; sealed owners hold 0; mass 6144.0016 | 80 h tiles (contact holds them); mass 6143.798 |
| after the clear | 0 h tiles, C = 0 | 0 h tiles, C = 0 |

There is no frame with liquid on a new solid at 4h and the displacement skipping it, and volume is kept. One residue: an edit on the same frame as a deferred build (only a body outrunning the host's count defers) would run its displacement before the join's tiles exist.

**`contactTiles` is published from the running layout.** Under Requested and Full it is the h tiles beyond the plan's admitted tiles (Dynamic reports none), taken from a host layout when one is built and from each build's receipt otherwise (non-blocking; a receipt that arrives after a host layout replaced its generation is dropped). On a frame with no census the control shows the last build's counts, which are the running layout's; an attached relayout that built nothing records the current generation as the `previous` stage view, and the reasons and importance views stay the last census's.

**The early h tiles before a dropped crate reaches the pool are liquid above the phi surface, not a dry admit.** Still pool to 0.275 m, crate from 0.6 m, Requested at zero detail, 20 settle frames. The first h tiles appear on the fourth frame of the fall, one frame before the reach sphere meets the phi surface:

| frame | sphere bottom (reach = radius + travel + one cell) | wet tiles | active tiles | h tiles after |
|---|---|---|---|---|
| 3 | 0.360 m | 1,560 | 0 | 0 |
| 4 | 0.316 m | 1,580 | 4 | 48 |
| 5 | 0.262 m (pool level 0.275 m) | 1,560 | 36 | 196 |

The four active tiles on frame 4 are the row above the surface (y = 0.30 to 0.35 m), each within the reach (0.185 to 0.198 m against 0.218 m), each flagged wet: their 4h owners hold a fill of 0.044 with every corner phi in air (least phi +2.7 cells). The census's wet flag is "not air", and air needs both no negative corner and V under the dust threshold, so volume a still 4h pool carries one owner above its phi surface counts as liquid. The pool has 1,536 tiles under the surface and between 16 and 56 such tiles above it, changing every frame, which is why the first h tile moved between frames 3 and 5 across runs. No active tile was dry or outside the reach. The cost is 48 h tiles one frame early.

## Benchmark gate, 5 October 2026

The handoff's closing validation: one path from zero to full h detail, Full within 5% of the solver before this program at full detail. Baseline is HEAD (114ea4a7, `git archive` into a scratch directory, its own module graph), the tree is the working tree at one fingerprint for every arm (after the contact agent's shader labels, before its root rhs change). Two Dawn processes, one raw device each, every arm of a scene created first and then advanced one frame at a time in rotating order; GPU pass time is the sum of timestamp queries over every compute pass; the median is over frames 13 to 60 (13 to 70 on fig-9) at dt = 1/30 s; bytes are every buffer and texture the arm created on the device. Setup is in creation order: the first arm of each root in a process compiles, the rest reuse its pipelines.

HEAD arms: all-fine is `coarsening: "regions"` with no region; Dynamic is the declared defaults. Tree arms: zero detail is Requested with solid contact off; the Fine region is a fifth to a twelfth of the domain; "app" is Dynamic with the app's importance defaults (shape tolerance 0.5, thin, approach, hold 8) under the scene's own overrides, still at dt = 1/30 s.

**dam64 through impact** (4,096 tiles)

| arm | pass ms, median | passes | h tiles, mean | bytes | setup s |
|---|---|---|---|---|---|
| HEAD all-fine | pressure rejected on frame 3 | 68, 65 | 4,096 | 75.1 MB | 86.3 (first HEAD arm) |
| HEAD Dynamic | 10.23 | 58 | 1,167 | 75.1 MB | 0.3 |
| zero detail | 4.16 | 49 | 0 | 7.8 MB | 2.1 (first tree arm) |
| Fine region | 7.87 | 50 | 768 | 40.7 MB | 0.7 |
| Dynamic, declared | 10.61 | 59 | 1,184 | 73.4 MB | 0.3 |
| Dynamic, app | 11.59 | 59 | 1,846 | 73.4 MB | 0.3 |
| Full | 8.95 | 50 | 4,096 | 73.4 MB | 0.3 |

**fig-9** (16,384 tiles)

| arm | pass ms, median | passes | h tiles, mean | bytes | setup s |
|---|---|---|---|---|---|
| HEAD all-fine | band over capacity on frame 2 | 68 | 16,384 | 248.2 MB | 103.8 (first HEAD arm) |
| HEAD Dynamic | 18.23 (mean 21.29) | 58 | 4,035 | 248.2 MB | 1.4 |
| zero detail | 5.08 | 49 | 0 | 30.2 MB | 54.5 (first tree arm) |
| Fine region | 11.89 | 50 | 3,072 | 161.6 MB | 3.6 |
| Dynamic, declared | 18.81 (mean 21.52) | 59 | 3,826 | 243.2 MB | 7.3 |
| Dynamic, app | 20.67 | 59 | 5,022 | 243.2 MB | 1.4 |
| Full | 21.19 | 50 | 16,384 | 292.1 MB | 1.4 |

**Garden hose ×10** (20,736 tiles)

| arm | pass ms, median | passes | h tiles, mean | bytes | setup s |
|---|---|---|---|---|---|
| HEAD all-fine | 19.28 over frames 13 to 33; pressure did not converge on frame 34 | 53 | 20,736 | 313.4 MB | 82.2 (first HEAD arm) |
| HEAD Dynamic | 24.86 | 62 | 1,578 | 313.4 MB | 1.1 |
| zero detail | 8.57 | 57 | 0 | 38.0 MB | 1.2 |
| Fine region | 16.42 | 58 | 1,728 | 184.1 MB | 1.1 |
| Requested, contact on (app default) | 17.72 | 67 | 1,285 | 233.4 MB | 5.6 |
| Dynamic, declared | 28.45 | 63 | 1,576 | 307.5 MB | 1.1 |
| Dynamic, app | 28.11 | 63 | 1,576 | 307.5 MB | 1.2 |
| Full | 22.98 | 54 | 20,736 | 369.5 MB | 84.4 (first tree arm) |

**Dam at 128³** (`high-resolution-dam-break`, 32,768 tiles)

| arm | pass ms, median | passes | h tiles, mean | bytes | setup s |
|---|---|---|---|---|---|
| HEAD all-fine | pressure rejected on frame 2 | | 32,768 | 493.4 MB | 104.5 (first HEAD arm) |
| HEAD Dynamic | 21.86 | 58 | 5,299 | 493.4 MB | 2.1 |
| zero detail | 6.59 | 53 | 0 | 59.5 MB | 60.9 (first tree arm) |
| Fine region | 19.93 | 54 | 6,144 | 322.2 MB | 4.0 |
| Dynamic, declared | 22.89 | 59 | 5,418 | 485.1 MB | 7.6 |
| Dynamic, app | 28.76 | 63 | 8,666 | 485.1 MB | 2.1 |
| Full | 33.88 over frames 13 to 32; pressure did not converge on frame 33 | 50 | 32,768 | 583.0 MB | 2.0 |

The null pair (a second Full instance against the first) is within ±0.17 ± 0.29 ms on every scene.

### Full against HEAD all-fine

**HEAD has no all-fine run to compare a window against, except on the garden.** It rejects pressure on frame 3 of dam64 (the limitation the full-fidelity addressing study already recorded for the pre-program solver), fills its pressure band past capacity on frame 2 of fig-9, rejects on frame 2 of the 128³ dam, and stops converging on frame 34 of the garden. The tree's Full runs dam64, fig-9 and the garden through their windows; on the 128³ dam it stops converging on frame 33 (accepted residual 5.59 against the tolerance 5 after 3 cycles, both instances identically).

- **Garden, frames 13 to 33 (the one sustained window): Full is +17.8% (22.95 against 19.49 ms mean, 22.99 against 19.28 median), paired +3.47 ± 0.13 ms, null +0.12 ± 0.10 ms. The 5% target is missed.** The passes that moved (ms per frame, HEAD then tree): pressure authority and volume correction 1.36 → 3.25 (+1.89), transfer to pressure 0.20 → 0.69 (+0.49), geometric fill 0.19 → 0.53 (+0.34), surface advect 1.64 → 1.87 (+0.24), extension 1.89 → 2.05 (+0.16), pressure authority phase 0.19 → 0.31 (+0.12), surface advect deferred 0.93 → 1.03 (+0.11), trace cells +0.07; the rest are under 0.05 each. The first three are 2.7 of the 3.5 ms. The source of all three changed in this program's solid work (the authority's cut-tile vote became a least-squares plane fit over the tile's 125 vertices and its capacity reads a tile's open fraction, the transfer to pressure reads the solid patch weights, the fill is scaled by open fractions), and the garden is the one scene of the four with terrain; that is a reading of the source, not a measurement.
- **Where HEAD all-fine only survives its first frames, Full is within noise of it.** Fresh solver per sample, order rotated: dam64 frame 1, Full +0.79 ± 0.93 ms on 19.3 (+4% ± 5%, ten samples; null −1.30 ± 1.00); frame 2, −2.42 ± 0.61 ms on 12.7 (HEAD is on its way to the rejection); fig-9 frame 1, −2.97 ± 1.14 ms on 32.6 (eight samples; null +1.07 ± 0.70). A fresh solver's first frame is twice a settled one and its spread is as large as the target, so these bound nothing tighter than about 5%.
- Bytes: Full holds 292 MB on fig-9 against HEAD's 248 MB, 369 against 313 on the garden and 583 against 493 at 128³, because Full reserves the pressure band for every tile where HEAD caps it at half the lattice (the cap its fig-9 run hits).

### Dynamic at declared defaults against HEAD Dynamic

| scene | HEAD | tree | paired | h tiles, HEAD and tree | ms per 1,000 h tiles | at HEAD's cost for the tree's tiles |
|---|---|---|---|---|---|---|
| dam64 | 10.18 ms mean | 10.54 | +0.36 ± 0.09 ms (+3.6%) | 1,167, 1,184 | 8.72, 8.90 | +0.33 ± 0.06 ms (+3.3%) |
| fig-9 | 21.29 | 21.52 | +0.23 ± 0.07 ms (+1.1%) | 4,035, 3,826 | 5.28, 5.62 | +0.91 ± 0.21 ms (+4.3%) |
| garden | 25.06 | 29.30 | +4.25 ± 0.92 ms (+16.9%) | 1,578, 1,576 | 15.88, 18.60 | +4.26 ± 0.88 ms (+17.0%) |
| dam 128³ | 21.91 | 22.91 | +1.00 ± 0.11 ms (+4.6%) | 5,299, 5,418 | 4.13, 4.23 | +0.94 ± 0.15 ms (+4.3%) |

The last column removes the trajectory: a line fitted to HEAD's own frames (pass time against h tiles) evaluated at the tree's tile counts, so the redistance fix's change of tile count is not charged to storage. **The lane-equivalent path is 3 to 5% slower on the dam breaks and fig-9 and 17% slower on the garden.** One pass is new (publish 4h vertex phi base, under 0.01 ms). What moved:

- dam64: extension +0.11, momentum +0.09, redistance +0.04, V-cycle +0.04.
- fig-9: extension +0.33, geometric sharpening +0.18, advect deferred +0.04; transport −0.16, sharpening geometry −0.10.
- dam 128³: extension +0.30, momentum +0.20, band solve +0.10, redistance +0.08, remap +0.07.
- garden: body forces 11.08 → 12.23 (+1.14), remap 0.13 → 1.00 (+0.88), pressure authority and volume correction 0.46 → 1.33 (+0.87), transport +0.28, transfer to pressure 0.06 → 0.32 (+0.26), geometric fill +0.15, trace cells +0.11.

Extension is the common mover off the garden; the garden's are the same solid passes as under Full, plus remap and body forces.

### Setup and bytes

- Cold setup, first use of a lattice size by a root in a process: HEAD compiles everything up front, 82 to 105 s. The tree compiles by need: 54.5 s (fig-9) and 60.9 s (128³) to zero detail, 68 and 75 s by the time every policy has been created on the same device, and 84.4 s for Full directly on the garden. Metal keeps compiled shaders on disk across processes (dam64's tree arm took 2.1 s here and HEAD's took 2.3 s in the second process against 86.3 s in the first), so these are upper bounds for a machine that has run the same sources before.
- Zero detail is a tenth to an eighth of HEAD's bytes (7.8 against 75.1 MB, 30.2 against 248.2, 38.0 against 313.4, 59.5 against 493.4). A Fine region over a fifth of the domain holds half to two thirds of Full's, because the field textures are the domain's as soon as one tile is h. Dynamic holds what HEAD does (C = T).

Decision needed before the capacity work: the garden's Full and Dynamic costs are outside the target, and the passes that carry it are pressure authority, the transfer to pressure, geometric fill, remap and body forces. Nothing was optimised inside this gate.

### The garden regression: attribution and fix, 5 October 2026

One Dawn process, HEAD from `git archive`, every arm a patched copy of the tree's modules against the unedited tree and HEAD, frames 13 to 33 (Full) and 13 to 60 (Dynamic), split timing per launch. The regression is four pieces of solid work, none of it storage:

| launch (ms a frame) | HEAD | tree before | tree after | what it was |
|---|---|---|---|---|
| pressure authority `cut`, Full | 0.44 | 2.09 | 0.35 | plane fit of a cut h tile's live vertices for its 4h pressure owner: 64 lanes, barriers and a serial reduction per tile, each lane resampling the solid to decide which vertices are live |
| the same, Dynamic | 0.22 | 1.07 | 0.33 | |
| transfer to pressure `tFineToPressure`, Full | 0.19 | 0.65 | 0.41 | +face patch weights resampled from the solid (`umPressureFaceV`) for every patch of a cut tile |
| geometric fill `geometry`, Full | 0.17 | 0.51 | 0.30 | lane 0 of a cut tile summed 64 open-fill terms serially (`umSurfaceTargetOpen`) |
| remap `remapCells`, Dynamic | 0.12 | 0.98 mean, **6.3 over 150 frames** | 0.19 | `planeCorner`: one lane walking 4,096 cells twice per buried corner handed to a cut 4h owner. 37 of 188 frames above 40 ms (worst 72; HEAD's worst 36), and a 60 ms first frame |

The remap fit was the largest cost and it is a hitch, not a steady tax: the 60-frame gate window caught one spike of it.

**Is the work needed with solid contact on?** Yes. Pressure's root solve runs at 4h under every policy, so a cut h tile has a 4h pressure owner whether or not the simulation ever runs a cut 4h owner: the fit, the patch weights and the open fill serve that root owner, and the remap fit serves the dry cut tiles that contact leaves at 4h. Nothing here could be compiled out on `uniformDetailCoarseSolids`. So the fix is the second and third preference:

- **Into the record, once per solid state.** The coarse solid record grows from two to five `vec4f` a tile: the tile's 64 open-cell bits and, for each of its 48 +face patches, the patch's open volume when it is a whole number of eighths (a 4-bit code; anything else falls back to the sample). `cut` reads the bits instead of sampling the solid; the transfers read the codes through `umPatchFaceV`. Cost: 48 bytes a tile (garden 1.0 MB, 128³ 1.6 MB, 256³ 12.6 MB).
- **Cheaper per-frame kernels, arithmetic unchanged.** `cut` is one lane a tile with a function-local vertex array and no barrier (same two fit loops, same closed form). The fill's 64 open terms are evaluated one a lane and summed in cell order. `umPressurePhiCell`'s open branch returns the centre phi it had already loaded.
- **`planeCorner` across the workgroup.** Lane i takes tile i of the corner's 4³ tiles in both passes; lane 0 reduces in tile order and solves with the original statements. The sums are reordered, so this one is not bit-equal: 8,968 vertex phi texels move by at most 7.0e-6 on the garden's first frame (volume and velocity equal through frame 8), and the Dynamic trajectory separates from there.

Everything but the plane rewrite is bit-equal to the unedited tree in volume, velocity and vertex phi: garden Full at frames 12 and 33, garden Dynamic without the plane rewrite at 12, 40 and 60, the garden with coarse solids on (Requested, contact off, zero detail) at 1, 12 and 40 with the plane rewrite in, and dam64 and fig-9 under Dynamic and Full at their first and last frames.

| garden arm | HEAD | before | after | after against HEAD |
|---|---|---|---|---|
| Full, frames 13 to 33 | 18.72 ms | 21.92 (+17.1%) | 19.59, 19.51 | **+0.88 ± 0.11 and +0.81 ± 0.10 ms (+4.7%, +4.3%)**; null −0.02 ± 0.07 |
| Dynamic, declared defaults, 13 to 60 | 24.08 | 28.17 (+17.0%, worst frame 67) | 25.93, 26.04 (worst 29) | **+1.85 ± 0.22 and +2.03 ± 0.22 ms (+7.7%, +8.5%)**; null −0.07 ± 0.22 |
| Requested, contact on (1,285 h tiles) | none | 17.22 | 16.14 | −1.08 ± 0.29 ms against before |

dam64 and fig-9 did not move: Dynamic −0.02 ± 0.03 and −0.05 ± 0.03 ms, Full +0.05 ± 0.03 and +0.04 ± 0.03 ms, all bit-equal. Lanes on the edited tree: `uniform-coarse-solid-rest` 10 of 10, `uniform-mixed-live-solid-edit`, `uniform-pond-rest`.

**Full meets the 5% target; Dynamic does not.** What is left of Dynamic's +1.9 ms: body forces +0.8 to +0.9 (`forcesRegularCoarse` 0.44 → 0.94, the seam `forces` launch +0.4 on 9.9), transport `build` 0.18 → 0.40, pressure authority +0.12, `traceCellsMerged` 0.10 → 0.21, extension +0.10, redistance +0.08, fill +0.08. The solid work that remains is about 0.3 ms.

**Body forces.** The +1.15 ms is not the closed-owner gate: with HEAD's unit-only test in `umOwnerClosed` the regular 4h launch reads 0.99 against 0.94, with HEAD's reconstruction sample 0.94, under identity storage (no accessor emitted) 0.93, with the plane fit disabled 0.93. The regular 4h list is built in the same order by the same builder. The cause of that doubling is not found. It stops mattering under the next finding.

**The garden's 10 ms force pass is the inline curvature set, and the cached set is 9 ms cheaper (measured, not landed).** The forces compile two sets: one that evaluates each owner's normal and curvature once (`cacheNormals`, `cacheCurvature`), and one that evaluates both inline at every face. The choice is tied to `uniformDetailCoarseSurface`, so Dynamic at shape tolerance 0 and Full run inline, Requested and Dynamic at the app's tolerance run cached. The garden is the only gate scene with surface tension (0.072 N/m; the dam breaks and fig-9 are 0, where neither set evaluates curvature). On it:

| garden | inline | cached |
|---|---|---|
| Dynamic: seam `forces` | 10.15 ms | 0.68 |
| Dynamic: `forcesRegularCoarse`, `forcesRegular` | 0.95, 0.39 | 0.13, 0.09 |
| Dynamic: `cacheNormals` + `cacheCurvature` | 0 | 0.23 + 0.12 |
| **Dynamic frame** | 26.42 ms (HEAD 24.10) | **15.83 (−8.27 ± 0.16 ms against HEAD, −34%)** |
| Full frame | 19.54 | 20.12 (+0.59 ± 0.08: `cacheCurvature` 0.73 + `cacheNormals` 0.34 against 0.48 saved) |

A seam owner supplies many faces through the general lookup, and inline each face evaluates two curvatures, each a centre normal and its face neighbours' normals; Full has no seam and its regular h faces are cheaper inline. The rule that follows is "cached unless Full". It is two lines (the need in `pipelineNeedsOf` and the `forces.encode` argument, decoupled from `coarseSurfaceTravel`), no kernel text. The cached set's first frame differs from the inline one in 15 velocity texels by at most 9.1e-13 (contraction of the same expression in a different kernel), then the trajectories separate. Not applied: it changes which pipelines Dynamic runs, in a file the contact work is editing, and is a decision for the program rather than a regression fix.

**The lagged cycle plan is one cycle short on some garden trajectories.** The plan for frame N comes from the receipt of frame N−2, and a frame that ends above tolerance after its planned cycles is fatal. At declared defaults (tolerance 0.001, no pressure reserve) in one process: the tree with the plane rewrite stops at frame 116 (initial 50.5; 0.109 → 0.0103 → 0.00125 after three cycles, frame 115 had asked for four), the cached-force arm at frame 142 (0.00155 after three), HEAD all-fine at frame 34 (0.0017 after three); the unedited tree, identity storage and HEAD Dynamic run 200 frames. Each of these converges at a factor of ten a cycle and is one cycle late, so this is the planner's lag meeting a perturbed trajectory, not a solver defect of any arm; it is the same failure the 128³ dam shows below.

### Full on the 128³ dam, frame 33: discriminator

Three arms to frame 40 in the same process:

| arm | result | frame 33 cycle history |
|---|---|---|
| tree | fails at frame 33, accepted 5.587 against 5 after 3 cycles | plan 3, initial 10,106.8; 22.99 → 11.25 → 5.587; no stall flag; frames 27 to 32 converged in 2 |
| `DETAIL=identity` | fails at frame 33, bit-identical history | the same |
| tree without the redistance residual (candidate A, `uniform-mixed-surface.ts` about 757) | runs to frame 40 | plan 4; 24.20 → 12.17 → 6.50 → 3.73 |

Storage is not involved. Candidate A is not the cause either: without it the residual after three cycles on frame 33 is 6.50, worse than the tree's 5.59; that arm survives only because its frame 31 happened to need three cycles and raised the plan to four in time. The frame is a halving-per-cycle impact frame (initial residual 10⁴) arriving one frame ahead of the plan.

### 256³ on the GPU, 5 October 2026

The first execution at this size since the storage, capacity and layout changes; every earlier 256³ figure in this program is a mock census. Scene `cm12-figure-7-256` (262,144 tiles: a 1 m ball dropped into an empty 6.4 m tank, no solids, impact about frame 25), dt = 1/30 s, compute passes only, no render. One Dawn process, one raw device, one arm at a time with the solver destroyed before the next; the device's live bytes returned to under 10 B after every arm. Pass time is the sum of timestamp queries over frames 10 to 50 with every frame awaited (the window timed on 30 September); bytes are every live buffer and texture on the device, in decimal MB. The tree is the working tree at about 09:50 with the contact agent's delta-form band solve; HEAD is 114ea4a7 from `git archive`.

| arm | pass ms, median (mean) | passes | h tiles, mean | bytes | setup s |
|---|---|---|---|---|---|
| zero detail (Requested) | 11.63 (12.99) | 49 | 0 | 500.6 MB | 78.5 (first tree arm) |
| Fine region over an eighth of the lattice, admitted on frame 5 | 36.22 (37.68) | 50 | 32,768 | 2,550.1 MB (peak 2,624.6) | 17.3 |
| Dynamic, declared | 37.40 (39.09); repeat 39.20 (39.70) | 59 | 7,882 | 4,099.5 MB at attach; 2,345.0 after the return on frame 32; 2,452.2 at frame 50 (peak 4,300.0) | 22.7 |
| HEAD Dynamic | 38.91 (38.56) | 58 | 7,805 | 4,100.7 MB throughout | 115.7 (first HEAD arm) |
| Full, 10 frames | 118.9 (119.8) over frames 1 to 10 | 50 | 262,144 | 4,954.6 MB (peak 5,042.8) | 17.4 |

No arm failed pressure (one V-cycle on 56 of 60 Dynamic frames, two on four, in the tree and in HEAD alike; no stall), no GPU error was raised, and Dynamic deferred nothing.

**Bytes against the census.** A mock census of the same tree gives 500,614,752 (zero detail), 2,550,137,320 (region), 4,099,474,408 (Dynamic at attach) and 4,954,588,136 B (Full). The device holds 672 to 940 B more in each case: the pressure receipt and mass accounting buffer is 1,200 B against the mock's 528, and a 264 B diagnostics readback appears once frames run. No other label differs. Against the figures recorded earlier in this program:

- Zero detail is inside the 493 to 561 MB range.
- Dynamic at attach is 33.6 MB above the 4,065.9 MB census, all of it the pressure band rows (704.6 → 738.2 MB: the delta-form solve's extra row field).
- Full's 5.30 GB was an earlier tree's (the one whose Dynamic was 4.48 GB). Today it is 4.95 GB: band rows 1,476.4 MB and stage scratch 1,030.9 MB are the two largest buffers, both under the device's 4,294,967,295 B limit.
- After the return Dynamic holds 2.35 to 2.54 GB, under the 2.6 to 2.8 GB estimate, because the need is lower than the estimate assumed (C of 12,426 to 31,580 against 41,216 to 57,768).

**The first h tile.** Admitting the region takes the device from 500.6 to 2,550.1 MB: 1,612.2 MB of domain-sized textures (the census's 1.6 GB: four rgba32float fields at 268.4 MB each, the two vertex phi fields at 67.9 MB, the r32float fields at 67.1 MB) and 511.8 MB of buffers sized by C = 32,768 (band rows 184.5, velocity tap cache 181.4, stage scratch +100.1). In time: `applySceneUniforms` 47.8 ms; pipeline preparation 2,478.6 ms (the h kernels' first use in the arm, awaited by the probe); 301.6 ms for the queue to drain the allocation and the whole-lattice fill, against the 0.1 to 0.2 s estimate; then the admission frame itself, 49.9 ms of passes and 97.4 ms wall against 25.7 and 28.3 ms on the frame after. Old and new buffers overlap by 74.5 MB during the change.

**Capacity through impact.** Dynamic attaches at C = T and returns on frame 32 to C = 12,426 (4,099.5 → 2,345.0 MB), then grows on frames 34 (16,912; 2,390.2 MB), 39 (23,056; 2,452.2 MB) and 51 (31,580; 2,538.2 MB): four changes, no deferral, for a need that reaches 16,974 h tiles on frame 55. The peak of 4,300.0 MB is the return, with the old and new buffers both live.

**Tree against HEAD, Dynamic at declared defaults.** Sequential arms, not interleaved; the tree ran before HEAD and again after it, with an identical tile trajectory both times.

| window | HEAD | tree | paired | h tiles, HEAD and tree | ms per 1,000 h tiles | at HEAD's cost for the tree's tiles |
|---|---|---|---|---|---|---|
| frames 10 to 50, first run | 38.56 ms mean | 39.09 | +0.53 ± 0.64 ms (+1.4%) | 7,805, 7,882 | 4.94, 4.96 | +0.42 ± 0.54 ms (+1.1%) |
| frames 10 to 50, repeat | 38.56 | 39.70 | +1.14 ± 0.39 ms (+3.0%) | 7,805, 7,882 | 4.94, 5.04 | +1.03 ± 0.63 ms (+2.7%) |
| frames 10 to 60, first run | 42.50 | 44.09 | +1.59 ± 0.63 ms (+3.7%) | 9,372, 9,609 | 4.54, 4.59 | +1.15 ± 0.75 ms (+2.7%) |
| frames 10 to 60, repeat | 42.50 | 44.53 | +2.03 ± 0.47 ms (+4.8%) | 9,372, 9,609 | 4.54, 4.63 | +1.59 ± 0.75 ms (+3.7%) |

**In pass time the tree is 1 to 3% slower than HEAD on the 30 September window at matched tile count, the same size as the other dam scenes; the first run is not resolved from zero and the two tree runs differ by 0.61 ± 0.45 ms.** HEAD's 38.6 ms reproduces the 30 September figure (about 39 ms). During free fall (frames 10 to 24) the tree is level or ahead (−0.63 ± 0.92, −0.22 ± 0.65 ms); the difference is in the impact frames (25 to 50: +1.19 ± 0.84, +1.92 ± 0.42 ms at 10,128 against 10,010 h tiles). Past frame 50 the trajectories part (16,383 against 14,245 h tiles on frame 60). By label over frames 10 to 60, HEAD then tree: extension 5.09 → 6.05 (+0.96), momentum 3.86 → 4.21 (+0.34), band solve 3.81 → 4.06 (+0.24), global surface volume +0.14, body forces +0.12, census +0.11; geometric sharpening 2.12 → 1.71 (−0.41), sharpening geometry −0.16, geometric fill −0.10. Extension is the common mover again.

**The host advance is 3 ms slower under the tree's Dynamic, and that is larger than the pass difference.** `advanceTo` takes 6.05 ms median against HEAD's 3.12 (paired +3.05 ± 0.07 ms, every frame, at C = T and after the return alike; the Requested arms take 2.8), and the awaited frame is +3.5 to +4.0 ms on frames 10 to 50. A CPU profile under the mock device names it: `UniformDetailPlanner.settled` (`uniform-detail-planner.ts`, `low.every(...)` over every tile), read by `detailStaticKey()` on each Dynamic frame from the frame head and from the relayout encode in `webgpu-uniform-reference.ts`; 3.2 ms per frame in the profile. It scales with the tile count (the garden's 20,736 tiles show +0.4 ms). Not changed here; a count of non-zero entries kept where `plan` writes them would make the read constant time.

**Full constructs and runs.** The device accepts it: 4,954.6 MB, setup 17.4 s on a warm device, a first frame of 141.1 ms and frames 2 to 10 at 110.3 to 125.8 ms, pressure converged on every frame. The largest passes are surface advect 15.0, extension 14.4, the shared frame plan 12.4, momentum 8.0 and global surface volume 7.1 ms.

**Setup and memory.** Cold setup to zero detail is 78.5 s and HEAD's is 115.7 s; a later tree arm on the same device takes 17 to 23 s (2 s at 128³). The first Dynamic frame is 193 ms of passes (454 ms wall) on the tree and 208 ms (611 ms wall) on HEAD. The process footprint peaked at 5.2 GB under Dynamic and 6.06 GB under Full on a 32 GB machine, free memory stayed at 43% or more, and swap in use did not change during any arm.

## Progressive-detail cost curve, 5 October 2026

The target model is **frame cost ≈ cheap coarse base + work inside refined regions + work at their boundaries**: a small moving region must add a small cost, and must not switch on a large fixed allocation, make the coarse work dearer, or need general addressing throughout the solver. This section measures how far the tree is from that, attributes the distance, and ranks the structural changes that would close it. One change landed (the solid world memo, under "Moving regions"); everything else is a proposal.

**The curve is not proportional today, and the gap is one step, not a slope.** The first h tile costs 0.6 to 1.9 ms at 128³ (1.6 ms at 64³, 4.6 ms at 256³) and 192 MB (1.54 GB at 256³), whatever its size. After that step detail is cheap: 64 h tiles add 0.1 to 0.4 ms over one, 512 add 0.8 to 1.1 ms, and the slope is about 1 to 2 µs per h tile plus 3 to 4 µs per seam face. A moved region costs the host a pass over the whole layout (14.5 ms at 128³ before the memo, 7.5 after) against 0.8 to 1.1 ms on the GPU. Dynamic adds a tile-count-sized chain every frame (1.07 ms at 128³, 3.94 ms at 256³) and makes every tile a boundary tile.

### How it was measured

One lab script (scratch `u21c/lab.mts`), arms in one process on one device (Apple, Dawn on Metal), compute only, every frame awaited, timestamp queries per pass, CPU shader preflight before each process. **dt is 1/30 s in every arm** (the app default is 1/60; the step is not the same price at another dt, so no number here transfers across dt). Policy is Requested with `detailSolidContact` on, and Fine regions are the only source of h, so tile counts are exact; regions are applied through the live edit path (`applySceneUniforms`) on frame 3. Medians are over frames 21 to 60 of 60 ("active") unless a quiet window (frames 5 to 10) is named.

- **Pool, 128³**: `cm12-figure-7` (6.4 m cube, T = 32,768 tiles) filled to 3.1 m, a 1 m ball dropped at the centre (impact near frame 11). Regions are s×s×s tiles at the quarter point: in the bulk (under the surface), across the surface, and in the air.
- **256³**: `cm12-figure-7-256` (T = 262,144; ball onto a dry floor; window frames 32 to 60), fewer arms.
- **Garden hose ×10** (T = 20,736) and the 64³ dam for the cross-checks.

A second set of "split" arms gives each dispatch its own timed pass (label and entry point). Splitting adds about 1.5 ms of pass overhead to a total, so totals come from the unsplit arms and only differences between split arms are used.

### The curve

Pool, 128³. Zero detail is 5.86 ms (quiet 5.71; 7.5 ms awaited frame; 476 dispatches; 140,581 workgroups; 64.0 MB). `advanceTo` on the host is 2.35 to 2.5 ms in every arm, Dynamic 2.68.

| arm | h tiles | seam faces | bulk ms | surface ms | air ms | device MB |
|---|---|---|---|---|---|---|
| zero | 0 | 0 | 5.86 | 5.86 | 5.86 | 64.0 |
| s = 1 | 1 | 6 | 7.38 | 7.76 | 6.47 | 256.4 |
| s = 2 | 8 | 24 | 7.39 | 7.93 | 6.45 | 256.6 |
| s = 4 | 64 | 96 | 7.47 | 8.13 | 6.63 | 258.2 |
| s = 8 | 512 | 384 | 8.19 | 8.88 | 7.27 | 271.1 |
| s = 16 | 4,096 | 768 to 1,024 (+512 to 768 wall faces) | 14.32 | 12.95 | 11.04 | 320.4 |

| arm | h tiles | seam faces | ms | C | device MB |
|---|---|---|---|---|---|
| two disjoint s = 4 at the surface | 128 | 192 | 8.38 | 128 | 260.1 |
| one box of 128 tiles at the surface | 128 | 160 | 8.15 | 128 | 260.1 |
| one s = 4 at the surface | 64 | 96 | 8.13 | 64 | 258.2 |
| surface slab, whole plan | 3,072 | 2,048 | 16.66 | 3,072 | 310.0 |
| Dynamic, declared defaults | 2,196 | 2,939 | 19.78 | 5,656 | 336.1 |
| Full | 32,768 | 0 | 41.84 | 32,768 | 621.3 |

**Cost follows tiles and seam faces, not the number of regions**: two regions against one of the same tile count is +0.23 ms for 32 more seam faces. The dispatch count is 476 at zero and 477 in every Requested arm (514 under Dynamic): detail does not add launches, it changes what the same launches cost.

**After stage 1 and the zero-detail launch change (sections "Stage 1" and "Launches not issued at zero detail" below), same-session A/B, median frame over the active window:**

| arm | before | after stage 1 | after both |
|---|---|---|---|
| pool 128³, zero | 5.51 | 5.55 | 5.36 against 5.53 in its own session |
| pool 128³, s = 1 at the surface | 7.30 | 7.21 | 7.32 against 7.30 in its own session |
| pool 128³, s = 4 | 8.13 | 8.10 | |
| pool 128³, s = 8 | 8.88 | 8.87 | |
| pool 128³, s = 16 | 13.03 | 12.99 | |
| pool 128³, Dynamic | 19.82 | 19.89 | |
| pool 128³, Full | 41.84 | 42.34 | |
| 256³, zero | 11.83 | 11.77 | 12.01 against 12.10 in its own session |
| 256³, first tile | 16.45 | 14.61 | 14.72 against 15.04 in its own session |
| 256³, s = 16 | 19.56 | 18.26 | |
| 256³, Dynamic | 47.14 | 48.59 (paired frames +0.18) | |

The "before" column is this tree without stage 1; it is below the tables above because the band schedule changed in between. Sessions differ by up to 0.2 ms, so the last column is given with its own baseline (the tree with stage 1 alone, measured beside it).

Other scenes:

| scene | zero | first tile | larger | Dynamic | Full |
|---|---|---|---|---|---|
| dam 64³ (T 4,096) | 4.00 ms, 8.4 MB | 5.59, 32.5 MB | s = 8: 6.82, 40.4 MB | 10.66 (1,068 h; C 4,096; 78.2 MB) | 8.76 (4,096 h; 78.2 MB) |
| 256³ | 12.26 ms (quiet 10.74), 500.6 MB | 16.89 (quiet 13.22), 2,038.3 MB | s = 4: 16.92; s = 16: 19.87 (2,156.1 MB); s = 32: 33.66 (32,768 h; 2,550.1 MB) | 46.26 (16,383 h; C 31,580; 2,538.2 MB) | not rerun (118.9 ms, previous section) |
| garden hose ×10, contact off | 8.41 ms, 40.9 MB | 8 tiles in the air: 8.91, 162.9 MB | 512 tiles: 9.84, 177.4 MB | | |
| garden hose ×10, contact on | 16.12 ms with 1,296 contact h tiles (C 2,824; 205.5 MB) | region inside the contact band: 16.17 | region of 263 new tiles: 16.64 | 26.12 (1,637 h) | |

The garden's "zero" arm with contact on is not zero detail: the terrain contact band is 1,296 h tiles and costs 7.7 ms over the true coarse base. Under Dynamic the garden's body forces pass is 11.3 ms against 1.14 ms under Requested with contact; that pass belongs to the forces decision and was not touched. **The Dynamic `advanceTo` median at 256³ is 3.29 ms** after the `settled` fix (6.05 ms before it, 3.12 at HEAD).

The timestep probe's datapoint (64 h tiles at the impact point of `coarse-first-pool-impact`: +2.0 ms at dt 1/60) is reproduced here at dt 1/30: +2.28 ms for 64 tiles at the surface, of which +1.90 ms is already there with one tile. The probe's +0.5 ms at dt 1/30 is not what this pool shows; the scenes and the region placement differ and that difference was not chased.

### The fit

Pool, active window, ms = jump + a·(h tiles) + b·(seam faces), fitted over s = 1 to 16 with zero as the origin:

| placement | jump at the first tile | a, per h tile | b, per seam face | rms |
|---|---|---|---|---|
| bulk | 1.24 ms | 1.25 µs | 2.74 µs | 0.29 ms |
| surface | 1.76 ms | 0.34 µs | 3.87 µs | 0.18 ms |
| air | 0.59 ms | 0.95 µs | 0.89 µs | 0.02 ms |
| all three | 1.20 ms | 0.66 µs | 3.53 µs | 0.82 ms |

With h tiles alone the fit is 1.61 ms + 2.14 µs per tile (bulk), 2.24 + 1.76 (surface), 0.68 + 1.10 (air). A line through zero leaves a residual at s = 1 of +1.50 ms (bulk), +1.87 (surface) and +0.59 (air): **the fixed jump is 25 to 32% of the zero-detail frame for one tile of liquid.** The jump is +1.59 ms at 64³ and +4.6 ms at 256³ in the wet window (+2.5 ms dry), after which 256³ rises 0.73 µs per tile to s = 16 and 0.5 µs to s = 32. The garden with contact off jumps +0.5 ms for tiles in the air, then 1.8 µs per tile.

Bytes: the first tile takes +192.4 MB at 128³ (twelve domain-sized textures: four rgba32float at 33.55 MB, two vertex phi at 8.59, six r32float at 8.39), +24.1 MB at 64³, +1,537.7 MB at 256³ (4 × 268.44 + 2 × 67.90 + 6 × 67.11) and +122 MB in the garden. After it, bytes follow C: 28.8 kB per tile of capacity up to 512, 15.6 kB at 4,096, 11.1 kB at Full, the same at 128³ and 256³. In time the first admission is 7 to 8 ms of `applySceneUniforms`, 34 to 112 ms of pipeline preparation and a 24 to 68 ms frame at 128³; at 256³ it is 35 ms, 312 ms warm (2.5 s on a first compile) and a 104 to 390 ms frame that allocates and fills the 1.54 GB.

### Four classes

Every timed entry point is put in one class: **coarse base** (what a zero-detail frame runs), **fixed with detail** (the first-tile jump at the same placement, from the unsplit arms), **fine interior** (band solve, `redistanceFine`, `rowsDivideFine`, `buildFine`, `advectFine`, `advectOwners`, the momentum unit, regular and deferred steps, `forcesRegular`, the extension's regular h sweep, seed and publish, `traceCells`, `liveSeed`, `retirementEvidence`), **boundary** (`sweepSeam`, `seedSeam`, `sweepCoarse`, `seedCoarse`, `publishCoarse`, `unitVelocityTaps`, phi resolve, `summarize`, `resolveScale`, the general forces kernel, the transfers to and from pressure, the 4h sampling cache, the 4h vertex phi publish) and **relayout** (census, layout build, remap, remap publish, widths, changed geometry). "Shared" is the growth of kernels that serve both regimes over owner lists and cannot be split by launch. Interior, boundary and shared are differences between split arms above the one-tile arm.

| arm | base | fixed with detail | fine interior | boundary | shared | relayout | total |
|---|---|---|---|---|---|---|---|
| pool, s = 4 at the surface (64 h) | 5.86 | 1.90 | 0.32 | 0.06 | −0.02 | | 8.13 |
| pool, s = 8 in the bulk (512 h) | 5.86 | 1.52 | 0.50 | 0.22 | 0.04 | | 8.19 |
| pool, s = 8 at the surface | 5.86 | 1.90 | 0.62 | 0.22 | 0.24 | | 8.88 |
| pool, s = 16 at the surface (4,096 h) | 5.86 | 1.90 | 3.76 | 0.53 | 0.84 | | 12.95 |
| pool, surface slab (3,072 h, 2,048 seam) | 5.86 | 1.90 | 4.01 | 1.70 | 2.97 | | 16.66 |
| pool, Dynamic (2,196 h, 2,939 seam) | 5.86 | 1.90 | 3.45 | 2.47 | 4.93 | 1.27 | 19.78 |
| pool, Full | 5.86 | 1.90 | 29.54 | 0.37 | 3.82 | | 41.84 |
| dam 64³, s = 8 | 4.00 | 1.59 | 0.70 | 0.34 | 0.20 | | 6.82 |
| dam 64³, Dynamic | 4.00 | 1.59 | 2.01 | 0.76 | 1.34 | 0.97 | 10.66 |
| dam 64³, Full | 4.00 | 1.59 | 3.51 | −0.21 | −0.45 | | 8.76 |
| 256³, s = 16 (4,096 h) | 12.26 | 4.6 | 1.74 | 0.66 | 0.90 | | 19.87 |

(Rows sum to the unsplit total within 0.4 ms; the split arms carry their own noise.) **For any region up to 512 tiles the fixed class is larger than the interior and boundary classes together, and at 256³ the jump exceeds the cost of 4,096 h tiles.** That is the first thing to remove.

### The first-tile jump, by label

One tile against zero, medians per pass label, ms:

| pass | 64³ | 128³ air | 128³ bulk | 128³ surface | 256³ |
|---|---|---|---|---|---|
| pressure band solve | +0.63 | 0.00 | +0.61 | +0.64 | +0.67 |
| band list and rows, band projection | +0.07 | 0.00 | +0.07 | +0.07 | +0.08 |
| extension | +0.11 | +0.10 | +0.12 | +0.52 | +0.76 |
| transport | +0.25 | +0.07 | +0.22 | +0.22 | +0.23 |
| surface redistance | +0.05 | +0.09 | +0.07 | +0.13 | +0.49 |
| transfer to pressure and to simulation | +0.02 | +0.07 | +0.06 | +0.07 | +0.60 |
| pressure authority and volume correction, authority phase | 0.00 | +0.05 | +0.05 | +0.05 | +0.40 |
| momentum | +0.07 | +0.02 | +0.06 | +0.06 | +0.21 |
| sharpening geometry, geometric sharpening, fill | +0.03 | +0.03 | +0.03 | +0.06 | +0.44 |
| frame plan, transport live set | +0.01 | +0.04 | +0.04 | +0.04 | +0.32 |
| body forces | +0.01 | +0.04 | +0.04 | +0.04 | +0.15 |
| surface advect | +0.05 | +0.04 | +0.04 | +0.04 | +0.13 |
| global surface volume | +0.11 | +0.02 | +0.02 | +0.10 | +0.11 |
| local speed, 4h cache, cleanup, phi resolve, 4h vertex phi publish | +0.05 | +0.08 | +0.07 | +0.08 | +0.22 |
| **sum of label medians** | **+1.47** | **+0.58** | **+1.47** | **+2.08** | **+4.81** |

Two mechanisms account for it, and they scale differently.

**1. Launch latency, the same at every domain size (about 1.0 to 1.3 ms when the tile is wet).** The split arms give it by entry point: `bandSweep` +0.36 ms (48 launches of one group: 7 µs each while empty, 13.6 µs once one wet h tile exists, and unchanged at 512 tiles), `bandMiddleSweep` +0.12, `bandCoarseSolve` +0.06, `bandMiddleBake` +0.04, `bandRestrict` +0.03, `bandPrep` +0.02: **the band solve's fixed schedule (`{cycles: 4, fineSweeps: 3, middleSweeps: 2, coarseSweeps: 16}`, 95 launches a frame, `uniform-pressure-band.ts:29` and `:1026`) is +0.6 to 0.7 ms as soon as one h tile holds liquid**, and nothing for a tile in the air (band membership is h tiles with a wet cell). The extension's seam launches are next: `sweepSeam` +0.23 ms at the surface (two launches going from 2 to 4 groups: each surface job runs serially inside its group, about 115 µs per job, and 740 groups at s = 8 cost the same) and `sweepCoarse` +0.13 (2 → 66 groups). Then the fine launches that stop being empty: transport `gather` +0.08, `rowsDivideFine` +0.07, `build` +0.04, `buildFine` +0.03, `advectFine` +0.03, the momentum deferred and unit steps +0.05.

**2. Coarse kernels that cost more per job once detail exists (about 0 at 64³, 0.45 ms at 128³, 2.9 ms at 256³).** These are launches over all T tiles or over the 4h owner list, with the same width and the same number of loads and stores as at zero detail: `toPressureCoarse` +0.005 / +0.034 / +0.34 ms (64³ / 128³ / 256³), `toSimulationCoarse` +0.007 / +0.03 / +0.26, extension `publishList` +0.002 / +0.02 / +0.25, redistance +0.04 / +0.08 / +0.23, authority `build` +0.18 and `resolve` +0.14 at 256³, frame plan `seed` +0.17, `liveSeedCoarse` +0.14, `forcesRegularCoarse` +0.14, `retirementEvidenceCoarse` +0.115, `momentumStep` +0.11, authority phase +0.10, sharpening `commit` +0.10 and `classify` +0.09, geometry fill +0.09, surface volume `seed` +0.08, cleanup +0.07, surface advect +0.07, `cacheGeometryPrepare` +0.07, 4h sampling cache +0.06, local speed +0.05. **The cause is the texture switch of the default domain placement** (`uniform-detail-fields.ts:438`, `:738`, `:842`): each non-pinned field is its tile-resolution base block at C = 0 and one domain-sized texture once capacity is above zero, so a far 4h job reads and writes its tile-origin texel in a texture 64 times sparser (16 MB of velocity becomes 1 GB at 256³). There is no early-out it loses and no branch on `umCounts.x`; the accessor (`udLoad_`/`udStore_`, `:1522`) has the same shape at C = 0 and C > 0 and tests `textureDimensions` on every access in both. The passes that read tile-resolution storage at any C do not move (pressure V-cycle 1.55 → 1.52 ms at 128³, 1.86 → 1.84 at 256³; pressure setup, measure and checkpoints). Three entries rise without reading a switched texture and are not explained from the code: surface volume `dilate` (+0.08 at 256³), `rowsFallbackCoarse`/`rowsDivideCoarse` (+0.02) and sharpening `limit` (+0.04, its width goes 252 → 374 with the seam owners).

**A base-behaviour change rides on the first tile.** `umShiftLimit()` (`uniform-mixed-surface-volume.ts:146`) is 4h while the layout has no h tile and h once it has one, for every surface in the domain: it bounds the Newton bracket and secant step in `clearBand` (`:169`) and the convergence test in `solve` (`:452`), and can change which owners `measure` counts as cut. That is a change to the coarse numerics at s = 1, not a cost: the loads are the same. It should be local to h owners and their seam, or the same at both. The halo indexing switch at `uniform-mixed-pressure-boundary.wgsl.ts:60` exists but is inert: its users bind the all-4h pressure ownership, whose `umCounts.x` is always zero.

**The zero-detail base already carries the h tier's launches.** Every counted launch is `Math.max(1, …)` (`uniform-mixed-ownership.ts:479` to `:510`), so a zero-detail frame issues **137 one-group launches that do nothing** (band solve 95, extension 12, transport 9, redistance 3, phi resolve 3, band prepare 3, band projection 3, surface advect 2, momentum 2, forces 2, hanging taps 1, `traceCells` 1, cleanup 1) and three wide ones whose every group is empty: the band `list` at 4,096 groups and `toPressureFine`/`toSimulationFine` at min(4,096, T) each. The band solve pass alone is 0.33 ms at zero detail; with the rest the floor is about 0.5 ms of the 5.86. Going to one tile adds one dispatch (the 4h vertex phi publish) and 1,700 workgroups: the jump is per-launch latency and per-job cost, not more or wider launches.

### Fixed items, ranked

| # | item | size | sized by |
|---|---|---|---|
| 1 | twelve domain-sized detail textures at the first h tile | +192 MB at 128³, +1.54 GB at 256³; a 0.1 to 0.4 s frame to allocate and fill, 0.3 s of pipeline preparation (2.5 s cold) | domain cells (n³), triggered by C > 0 |
| 2 | coarse kernels reading 4h values through those textures | +0.45 ms at 128³, +2.9 ms at 256³, every frame | T, triggered by C > 0 |
| 3 | band solve schedule | +0.6 to 0.7 ms on the first wet h tile; 0.33 ms at zero detail | nothing: 95 launches at any size |
| 4 | extension seam jobs | +0.25 to 0.35 ms on the first surface tile | nothing up to about 700 seam groups |
| 5 | fine launches that stop being empty (transport, momentum, advect, rows) | +0.3 ms | nothing |
| 6 | empty h tier launches at zero detail | about 0.5 ms of the base | nothing: 137 launches plus three of 4,096 groups |
| 7 | buffers sized by C | 28.8 kB per tile of capacity (band rows, velocity tap cache, stage scratch) | C: already proportional |
| 8 | 4h vertex phi publish | 0.008 ms at 128³, 0.03 at 256³ | (t + 1)³ |

### Dynamic against Requested

At equal tiles Dynamic adds three things. **A relayout chain every frame, sized by T**: census, layout build, remap and remap publish are 0.81 ms at 64³, 1.07 ms at 128³, 0.62 ms in the garden and 3.94 ms at 256³ (census 1.46, remap 1.96, remap publish 0.36, layout build 0.16), with 37 more dispatches, the frame plan and authority phase run twice, and lattice-sized phi resolve and changed-geometry launches. **Capacity above need** (C = 5,656 for 2,196 tiles on the pool, C = T on the 64³ dam). **Tiles that are all boundary**: its h set is a ragged surface band, so the pool's Dynamic (2,196 h, 2,939 seam faces) pays 2.47 ms of boundary and 4.93 ms of shared work where the slab (3,072 h, 2,048 seam faces) pays 1.70 and 2.97.

That is why Dynamic is slower than Full on the 64³ dam (10.66 against 8.76 ms): relayout +0.97, shared +1.79, boundary +0.97, interior −1.5. Full runs the regular kernels (`forcesRegular`, `momentumRegularStep`) over a layout with no seam and empty coarse launches; Dynamic pays the seam and general kernels and the T-sized chain for a quarter of the tiles. On a domain this small the chain and the seam cost more than the tiles they save.

### Moving regions

A region moved one tile along x at the surface, through the live edit path, every fourth frame and every frame:

| arm | steady median | static |
|---|---|---|
| s = 4, every 4 frames | 8.24 ms | 8.13 |
| s = 4, every frame | 9.02 | 8.13 |
| s = 8, every 4 frames | 9.05 | 8.88 |
| s = 8, every frame | 10.06 | 8.88 |

**On the GPU a move costs about 0.65 ms plus about 3 µs per changed tile**: +0.81 ms for s = 4 (32 changed tiles) and +1.06 ms for s = 8 (128 changed), made of census 0.27, remap 0.15 to 0.23, frame plan +0.08 to 0.10, layout build 0.07, redistance +0.07 to 0.09, geometric fill +0.05, authority phase +0.05, remap publish 0.04 to 0.05; 38 more dispatches and about 250,000 more workgroups on the move frame and 200,000 on the next (`forgetWork()` puts every counted launch at its ceiling for two frames), +0.13 to 0.2 ms on the frame after. No capacity change, no deferral and no pipeline preparation after the first admission. `advanceTo` is +0.45 ms on a move frame.

**On the host a move cost 14.5 ms at 128³ whatever the region's size**, outside `advanceTo` (3 to 5 ms at 64³; 80 to 130 ms at 256³ and about 150 ms in the garden on the CPU mock). The profile: `syncMixedLayout` 52% (layout pack 14%, `planDetail` 11%, remap apply 7%, ownership write 7%: all T-sized), `pipelineSceneNeeds` 40% (`solidWorldForScene` 28%, a probe occupancy mask of the whole domain 16%). So a move is proportional to the whole layout and to the domain, not to its 2·s² changed tiles.

**Landed: the solid world is no longer rebuilt for an edit that leaves the solids alone.** `solidWorldForScene` (`lib/core/solid-world.ts`) cached by scene object identity, and every edit arrives as a new scene object (always across the worker hand-off), so a region move rebaked the terrain, replayed the voxel patches and rewrote the mask pages. It now also keeps the last world by its content stamp (terrain, container, lattice dimensions and the patch fold: every input of the world), held weakly. Per move: pool 128³ 14.5 → 7.3 to 7.5 ms on Dawn (7.5 to 9.4 on the CPU mock); on the CPU mock 256³ 80 to 127 → 47 to 55 ms and garden hose ×10 about 150 → 17 to 36 ms (20 ms on Dawn for the one garden edit that ran). What remains is `syncMixedLayout` (81%, T-sized) and the probe mask while pipelines are deferred (18%).

### What the renderer consumes

**Simulation side.** Once C > 0 one launch of (t + 1)³ publishes the 4h vertex phi base each advance (0.008 ms at 128³, 0.03 at 256³); the presented pressure, phi and stage copies are T-sized and unchanged by detail.

**Extraction (renderer compute: window list, classify, polygonise), frame 30, median ms:**

| arm | classify | polygonise | total | ns per vertex |
|---|---|---|---|---|
| pool zero | 0.52 | 0.79 | 1.31 | 2.3 |
| one tile at the surface | 0.52 | 0.79 | 1.31 | |
| s = 4 at the surface | 0.52 | 1.11 | 1.64 | |
| s = 8 at the surface (bulk, air: 1.38) | 0.59 | 1.11 | 1.70 | |
| s = 16 at the surface | 0.72 | 1.38 | 2.10 | |
| surface slab | 0.98 | 2.62 | 3.60 | 7.7 |
| Dynamic | 0.98 | 2.62 | 3.60 | 7.4 |
| Full | 1.05 | 0.98 | 2.03 | 2.5 |
| 256³ zero (frame 45) | 2.49 | 1.97 | 4.46 | |
| 256³ s = 16 | | | 5.24 | |
| 256³ Dynamic | 3.74 | 7.14 | 10.88 | 8.0 |

Dam 64³: zero 0.52, s = 8 1.11, Dynamic 1.51, Full 0.66. Timestamps are quantised at about 0.066 ms. **Extraction has no jump at the first tile; it costs three times as much per vertex where the surface runs through h tiles in a mixed layout, and is cheap again at Full.** It is the one consumer whose detail cost is proportional already, with a steep constant.

**Sizing, from the code.** No render structure is sized by the domain or by C because of detail: the detail phi ((n + 1)³) is bound directly with no copy, mesh buffers are 64 MiB at 128³ at any detail, the classify launch is (t + 1)³ workgroups at every level and already paid at zero. What changes with detail: every window touching an h tile is listed and pays 216 vertex loads (there is no per-tile phi range to reject it); the presented phi and stage grid buffer is C-sized (256 B per band tile); at the first h tile the bind groups are rebuilt and one re-extraction is forced. Polygonise uses `dispatchWorkgroupsIndirect`.

**Plan for a rationed render measurement (not run).** One process, 128³, three arms: zero, an 8×8×8-tile region at the surface, Dynamic. Extend `tools/benchmark-water-extraction-dawn.ts` with one mode of about 60 lines: the tool's `classifyMatrix` solver setup on `high-resolution-dam-break`, its `renderSamples` raster setup at 640×360, the arm change applied live (`applySceneUniforms` for the region, `applyRuntimeValues({detailPolicy: "dynamic"})`), `setVolume` and `setDenseLevelSetVolumeSource` re-called after each change, then ten instrumented `water.encode` calls with `invalidateSurface()`, recording extract, polygonise, caustics, the four interface passes and composite, with medians, h tiles, C, vertex count and listed-window count per arm. Compile stays out of the samples and the tool takes the GPU lease itself: `WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/benchmark-water-extraction-dawn.ts --scenes=high-resolution-dam-break --render --wait`, in the foreground, nothing else on the GPU.

#### Extraction after rounds 1 and 2, 5 October 2026

Three changes since the table above, all in `lib/core/webgpu-water-pipeline.ts`, all exact (the worklist as a set, vertex and triangle counts, and the mesh against the reference polygonise, on all twelve arms below; the extraction lane passes). Round 1: the cube whose every tile around its vertices is h takes the all-h path, and the two tile-mask tests of the mixed path are closed-form. Round 2: the listed windows append their mixed cubes from the worklist's end (`mixedCubeCount`, a ninth counter word) and one workgroup moves them behind the others, so `[0, activeCubeCount)` is what every reader had, in class order; and the classify launch is 4096 workgroups sharing the window list out instead of one per slot of the (t + 1)³ lattice. Same process, same frames, median ms of 24 (classify / polygonise / total, the window list included in the total):

| arm | round-1 text | now |
|---|---|---|
| pool zero | 0.52 / 0.79 / 1.31 | 0.33 / 0.85 / 1.25 |
| surface slab | 0.85 / 2.10 / 2.95 | 0.79 / 1.51 / 2.36 |
| Dynamic | 0.85 / 1.84 / 2.69 | 0.85 / 1.44 / 2.29 |
| Full | 1.05 / 0.98 / 2.03 | 1.05 / 1.11 / 2.16 |
| dam 64³ zero | 0.13 / 0.39 / 0.52 | 0.13 / 0.39 / 0.52 |
| dam 64³ s = 8 | 0.20 / 0.72 / 0.92 | 0.20 / 0.59 / 0.85 |
| dam 64³ Dynamic | 0.26 / 0.79 / 1.05 | 0.33 / 0.66 / 0.98 |
| dam 64³ Full | 0.20 / 0.52 / 0.72 | 0.20 / 0.52 / 0.72 |
| 256³ zero (frame 45) | 2.36 / 1.97 / 4.52 | 0.85 / 2.03 / 3.02 |
| 256³ Dynamic | 3.28 / 4.00 / 7.41 | 1.90 / 3.80 / 5.77 |
| 256³, 24,576 h tiles holding no surface | 3.87 / 2.03 / 5.96 | 2.62 / 2.10 / 4.85 |
| 256³ Full | 7.93 / 3.15 / 11.08 | 7.54 / 3.28 / 10.88 |

**The base fell and the increment fell.** Zero detail: 1.31 to 1.25 at 128³, 4.52 to 3.02 at 256³ (the fixed launch was 0.26 and 1.6 ms of 7 ns slots). The increment over zero for the surface slab: 2.29 in the table above, 1.64 after round 1, 1.11 now; Dynamic 2.29, 1.38, 1.04; 256³ Dynamic 6.42, 2.89, 2.75. The mover is under the timer quantum on every arm (11,148 to 16,911 cubes moved at 128³). `water.encode` itself was run with its render passes replaced by no-op objects (on the launch striding the list, which measured the same as sharing it but for 0.25 ms more at 256³ Full): two compute passes, the same mesh, the same times as the QA entry.

**What is left.** (1) Polygonise workgroups holding two classes: 760 to 790 of about 900 at 128³ partial detail before, 400 to 490 now, all of them all-h beside all-4h cubes in the front range; three regions would remove them at three times the worklist memory. (2) h windows holding no surface: 24,576 h tiles in the bulk cost classify 1.77 ms at 256³ (72 ns a tile), the per-tile phi range the solver does not publish. (3) Full at 128³ is 0.13 ms dearer in polygonise (two timer quanta) on the worklist the shared launch builds, not explained: a polygonise workgroup's cubes come from as many windows as before (4.9), and one workgroup per window gives the 0.98 back for 0.20 more in classify. Triangle order in the vertex buffer follows the worklist, as it always did (blocks are allocated by a global atomic); the three vertex shaders group by `index - index % 3` and nothing reads across triangles.

### General addressing is not the way out

The patch placement removes the domain-sized textures, and at zero detail it is already dearer than the base block: 6.78 against 5.85 ms at 128³ (+0.93 ms: surface redistance 0.32 → 0.76, surface advect 0.17 → 0.27) and 15.54 against 12.09 ms at 256³ (+3.45 ms, +29%: redistance 0.77 → 1.79, advect 0.76 → 1.34, transfers 0.17 → 0.58, forces 0.45 → 0.70). That is the +26 to 40% of the earlier patch arm, now measured with no detail present: every kernel pays the page lookup whether or not a page exists. The patch arms with regions measured nothing: the QA haloed patches hold their first layout by design (`syncMixedLayout`: no relayout of either kind), so a region drawn live is never adopted there (h = 0 in every arm).

### Defects found, not fixed

- **Garden hose ×10, Requested with contact on: a live edit that adds region tiles far from the liquid fails the next frame** with "invalid support (layout build 4: fatal bits 8)", the residency audit. Seen for tiles [2,14,2] to [10,22,10] and for a moving two-tile box in the air; the same region with contact off raised nothing, and regions [19,1,11] to [27,9,19] (beside the contact band) and [4,18,4] to [6,20,6] with contact on work. **Fixed the same day, below.**
- **The surface volume shift limit** changes for the whole domain at the first h tile (above).

### The garden fatal: cause and fix, 5 October 2026

**Cause: the residency certificate did not cover the builder's static h tiles.** The census certifies which pages of 4³ tiles a frame may skip as far air (`pageSeed`, `pageMark`: pages within the closure radius of a near tile or of a tile its band makes h), and audits itself: an h tile in a page the last certificate left absent is the residency fatal (`classify`, `UNIFORM_MIXED_RESIDENCY_AUDIT.absentNear`). The layout builder makes a tile h when the census band or its static mask asks (`fineAt`: band, less static 4h, plus static h), and a Fine region reaches it only through the static mask (`refreshRequestStatics`, `refreshMixedBuilderStatics`). A region drawn far from the liquid therefore became h in a page the certificate of that same census had left absent; the next census found h tiles there and raised the fatal one frame after the gesture. It needs a census on the following frame, which solid contact over a solid gives (the census runs every frame while the liquid can move the need); with contact off the same edit left h tiles in absent pages silently, until the next edit ran a census. Not the admit pass, the edit join or the capacity: the tiles were admitted and built correctly, in pages the frame was told to skip.

**Fix: the census seeds the static h tiles.** `UniformMixedDynamicClassifier.setStatic` keeps the builder's static h mask in the third block of the join buffer (no new binding; the census now clears only the two join blocks), and `pageSeed` seeds a page from its near tiles, its band tiles and its static h tiles. Both callers of the builder's `setStatic` set it. The audit is unchanged.

**Evidence.** A new arm of `tests/uniform-detail-policy-dawn.test.ts` ("a Fine region drawn and moved far from the liquid under solid contact keeps its pages resident": 64³, a pool one tile deep with a block standing in it, a two-tile region drawn eight tiles above the near set, moved twice, removed) fails on the unfixed tree with "invalid support (layout build 5: fatal bits 8)" and passes with the fix (the region's 8 tiles h, no other h tile above the pool, mass within 2e-3). The detail-policy and live-solid-edit lanes pass (2/2 files). On the garden itself, one process: the 512-tile region that failed runs 30 frames (1,796 h tiles), the moving two-tile box runs seven moves, and Dynamic with the same region runs.

### Ranked proposal

Ranked by measured size against the target: a small moving region adds a small cost. Each holds for a moving region and under Dynamic, and none is specific to a scene. Sizes are the measured label differences above; the "after" figures are sums of those differences, not measurements.

**Status, later on 5 October 2026.** Item 1: stage 1 is in the tree and takes 1.8 ms of the 256³ first-tile jump; the mirrored stores (0.75 ms) and the h store's size remain (sections "Stage 1" and "Stage 2 design" below). Item 6: landed outside the band files, 0.17 to 0.22 ms off the zero-detail base. Item 4: not started.

**1. Keep 4h values in tile-resolution storage at every C, and separate kernels by regime. Removes fixed items 1 and 2: 2.9 ms a frame and 1.54 GB at 256³, 0.45 ms and 192 MB at 128³, and the 0.1 to 0.4 s first-tile frame.** The base block persists for the life of the solver and is the only thing a pure 4h kernel binds, at zero detail and beside detail alike, with a plain `textureLoad` (no `textureDimensions` test, no page lookup). h values live in a store sized by C. Three kernels per stage instead of one accessor that addresses everything:
- *pure 4h*, over the 4h owner list, reading and writing the base block only: this is today's zero-detail kernel, and its cost must not move when detail exists;
- *regular h interior*, over the h tiles whose neighbourhood is all h, reading the h store with a regular in-tile stencil and a neighbour-tile table: its own launch, its width the interior tile count;
- *boundary*, over seam tiles on both sides (h tiles with a 4h neighbour, 4h tiles with an h neighbour): the only kernel with the general accessor, its width the seam tile count.

The regime of a tile is a layout fact, so the three lists come out of the layout build that already makes the owner lists, and a moved region changes list membership, not kernels. This is the base block and patch store of the storage design above with one constraint added by the measurement: the general accessor is confined to the boundary kernel, because putting it in every kernel costs 0.9 ms at 128³ and 3.5 ms at 256³ before any detail exists.

**2. Make the host edit path proportional to changed tiles. 7.5 ms per move at 128³ and about 50 ms at 256³ on the host after the memo; +250,000 workgroups for two frames on the GPU.** Diff the request against the accepted one and hand the planner, the layout pack, the ownership write and the remap the changed tile list instead of T; raise the launch budgets by the host-known change instead of `forgetWork()`; run the relayout head (census, build, remap, phi resolve, changed geometry) over the changed list instead of the lattice or a fixed 4,096 groups; do not build the probe occupancy mask when the world's stamp is unchanged. The GPU's 0.65 ms fixed part per move is those lattice-sized launches.

**3. Band solve: stop paying a fixed 95-launch schedule. 0.6 to 0.7 ms on the first wet tile at every size, 0.33 ms at zero detail, and 4.8 ms of serial `bandCoarseSolve` at Full.** When C = 0 no h tile can exist, under any policy, so the band prepare, solve and projection launches (101 of the 137) are not issued. For a small band the sweeps of a cycle run inside one launch per level, in workgroup memory, as `bandCoarseSolve` already runs its 16; `bandCoarseSolve` itself is one workgroup, serial in band size, and needs a parallel form before Full or a large band is affordable. The convergence gate removes work but never a launch, so the schedule is the cost. This is in the pressure files and changes the sweep order, so it is WS3's to decide.

**4. Extension at the seam: a regular launch over seam faces. 0.25 to 0.35 ms on the first surface tile, and most of the shared growth on ragged layouts (2.97 ms on the slab, 4.93 under Dynamic).** Every extension entry runs its jobs serially inside a group (`for (umJob = group; umJob < jobs; umJob += groups)`), and a surface face in a seam tile makes six general neighbour evaluations. Seam faces get their own list and one invocation each; the regular h and regular 4h sweeps keep their launches and never take the general path.

**5. Dynamic: census and remap over candidates, not T. 1.07 ms a frame at 128³, 3.94 ms at 256³.** The census criteria are local to the surface, bodies and regions, so a candidate list (tiles within the criteria's reach of last frame's band; not checked that the frame builds one today) replaces the T-sized launch; remap and its publish run when the build changed a tile, over the changed list (the same list as item 2). With item 1 the Dynamic band's cost becomes its seam, which is what it should be compared on.

**6. Do not issue h tier launches when C = 0. About 0.5 ms of the 5.86 ms base at 128³.** The 36 non-band launches and the three 4,096-group launches join the band's 101 in item 3; the condition is capacity, known on the host with no readback and exact under Dynamic. Small and structural; not landed in this update because 101 of the 137 are in the pressure band files WS3 has open, and the rest are worth about 0.25 ms.

**7. Extraction: reject h windows by a per-tile phi range, and give seam windows their own polygonise path. +2.3 ms on the slab and under Dynamic at 128³, +6.4 ms under Dynamic at 256³.** A minimum and maximum of phi per h tile, written where the tile's phi is written, lets classify drop a window before its 216 loads; the factor of three per vertex is the seam sampling in mixed cubes and belongs to a boundary kernel, as in item 1. Renderer side.

**With 1, 3, 4 and 6 the first wet surface tile at 128³ would cost about 0.5 ms instead of 1.9 (the fine launches that stop being empty and what is left of the band and seam launches), and under 1 ms at 256³ instead of 4.6, and the zero-detail base would fall about 0.5 ms.** The slope after the first tile (1 to 2 µs per h tile, 3 to 4 µs per seam face) is already the proportional part and stays.

### Extension seam tier, 5 October 2026

Item 4 above, measured and landed in `uniform-mixed-extension.ts` alone. Volume, velocity, phi and occupancy are bitwise equal to the previous kernels in every arm compared (64³: zero, one tile, a region moved every frame, Dynamic, Full, frames 3 to 14; the 128³ rows below, frames 4 to 60; the 256³ rows, frames 4 to 60).

**Where the time went.** Deep pool 128³, one surface tile (1 seam h tile, 26 seam 4h tiles), per launch: `sweepSeam` 158 µs and `sweepCoarse` 70 µs, each twice a frame, beside `sweepList` 108 µs for all 32,741 regular 4h owners. The seam h tile has 160 to 192 live faces a sweep; 42 (first sweep) and 67 (second) of them on average are not sources, and those need 56 and 88 neighbour requests that search (224 at most). The old kernel queued 143 to 168 faces for six serial `umNeighbor` each. The count is not what costs: a seam launch takes the same time for 1 tile and for about 300. It is the latency of one job's chain: 26 µs for a live tile whose faces are all sources, about +50 µs with the first face that is not, 13 to 15 µs per further serial round. A tier with a launch of its own therefore pays that chain however few faces it holds. Under Dynamic (2,405 live seam h jobs, 57,000 and 109,000 non-source faces, 3,500 and 11,000 searching requests) the same launches are throughput-bound (508 and 373 µs).

**What changed.** (1) A seam tile job plans its patches in parallel (sources, closed walls and dead faces drop out; each of a patch's six requests is classed from the staged 3×3×3 widths as the one-load direct case or a search), queues only the searching requests, serves the queue one request per lane with the unchanged `umNeighbor`, then combines. (2) Seed, each sweep and publish are Jacobi passes over disjoint outputs, so the seam h and seam 4h jobs of a pass share one launch (`seedSeams`, `sweepSeams`, `publishSeams`), and a sweep's launch also holds the regular 4h owners, 64 a job: the seam chains run beside the coarse base's instead of after it. 16 counted launches a pass become 10; with C = 0 the four list launches are unchanged.

| pool 128³, ms | frame before | frame after | extension before | extension after |
|---|---|---|---|---|
| zero detail | 5.374 | 5.331 | 0.515 | 0.516 |
| 1 tile | 7.179 | 7.040 | 1.105 | 0.698 |
| 4³ tiles | 8.091 | 7.805 | 1.158 | 0.867 |
| 8³ tiles | 8.834 | 8.686 | 1.170 | 0.951 |
| 16³ tiles | 12.989 | 12.778 | 1.594 | 1.351 |
| Dynamic (2,506 h tiles) | 19.782 | 19.785 | 2.629 | 2.538 |
| Full | 42.354 | 42.150 | 3.911 | 3.729 |
| 256³ zero detail | 11.594 | 12.178 | 1.493 | 1.552 |
| 256³ first tile | 14.936 | 14.738 | 2.255 | 1.934 |

The first tile's extension cost at 128³ is 0.18 ms, from 0.59. The 256³ zero row is arm scatter, not the change (its launches are the list kernels of before): the same pair in the other order gives 1.490 before and 1.451 after, two baseline arms give 1.498 and 1.468, and the split arms 1.583 and 1.510.

**What did not work.** The scratch record per face and neighbour with a combine launch was not built: 13.8 kB an h tile at capacity (453 MB at 128³) and one more launch a sweep, when launch latency is the cost. One launch holding every job kind (all h tiles, seam 4h tiles, packs) was built and is bitwise equal, first tile 0.72 ms, but the regular h sweep runs 1.8 times slower inside the larger kernel (Full extension 3.96 to 5.65 ms) and the list sweep 1.3 times (+0.06 ms at zero detail); cutting its workgroup memory changed neither, so regular h and the C = 0 list keep their own kernels. Leaving the packs out of the seam launch is the measured alternative: extension 2.52 against 2.61 ms under Dynamic and 0.05 less at Full, 0.88 against 0.74 on the first tile.

**Left.** At 256³ the first tile's extension cost is now mostly `publishList` (130 to 320 µs once an h tile exists; the store side). The sweep's seam launch is sized by the `merged` work kind, which counts every h tile: 4,096 empty groups at Full, 37 µs a launch against 10. A work kind counting seam h tiles, seam 4h tiles and packs would size it (landed in round 2, below).

**Round 2: throughput in a thin band (same day, later).** In a band one to three tiles thick nearly every h tile is a seam h tile, so the seam kernel's rate is the band's rate. Landed in `uniform-mixed-extension.ts`, plus the `seams` work kind in `uniform-mixed-ownership.ts`; volume, velocity, phi and occupancy are bitwise equal to round 1 in every arm (64³: zero, one tile, a region moved every frame, Dynamic at lane and app settings, Full, frames 3 to 14; every 128³ row below, frames 4 to 60 or 120; 256³ first tile and Dynamic at both steps, frames 4 to 60 or 120).

*Rates on the same tiles.* Pool 128³ under Dynamic (2,405 seam h jobs, 2,901 seam 4h jobs, 27,734 regular 4h owners), µs per sweep launch, two sweeps a frame. The probe launches run each kernel over the seam lists with every store sent to a slot nothing reads.

| per sweep launch, µs | round 1 | round 2 |
|---|---|---|
| seam h list: job and 3×3×3 staging only | 15 | 15 |
| the regular h kernel on the seam h tiles | 218 | 150 |
| the seam kernel on them, searches not served | 364 | 221 |
| the seam kernel with its searches (the real launch) | 507 | 296 |
| seam 4h list, searches not served | 128 | 93 |
| seam 4h list, real | 281 | 214 |
| regular 4h owners (packs) | 72 | 68 |

Before, a seam h tile cost 2.3 times the regular kernel on the same tile (211 against 91 ns a sweep): 146 µs of a launch was the seam kernel's structure (per-face liveness and six width lookups a face), 124 µs the searches. Seed and publish have no such excess: `seedSeams` is 77 µs for both lists, the h part at Full's 10.6 ns a tile; `publishSeams` is 196 µs against 33 ns an h tile at Full (79 µs) plus the 4h side. Full's 0.11 µs per h tile is not the floor for a band: 30,000 of its tiles are all sources and end at one word, while every band tile holds faces still to reach. The floor is the regular kernel on the band's own tiles, 62 ns a sweep now.

*What changed.* (1) Open bit. The seed writes, in the tile's mask word, whether it left a patch that is not a source; a tile without one stores nothing in any sweep, so its job ends at that word (h tile jobs, seam 4h jobs and regular 4h owners alike). (2) Cell liveness. A unit face's six requests are anchored at the six cells around its anchor whatever its component, so an h tile job tests one bit per lane of the tile's mask spread one cell (in-tile shifts, plus the facing layer of each unit face neighbour) instead of 192 per-face tests; beside a 4h face neighbour the whole facing layer is live when that tile's mask, or the mask of the tile below it on another axis, is set (a superset there, exact everywhere else; an extra face returns its old slot and stores nothing). The same test makes an h tile job live only when a cell is, and runs in the regular h kernel too. (3) Request classes from six bits. An h tile's request can search only where it leaves the tile into a 4h face neighbour, so a face looks up widths only for those requests (at most 1.25 a face on average, instead of 6). (4) A searching request none of whose readable slots is finite answers INF without a search. (5) The sweep's seam launch is sized by `seams` (seam h + seam 4h + packs), not by every h tile: 37 → 10 µs a launch at Full.

| ms, round 1 → round 2 (same pair in the other order) | frame | extension |
|---|---|---|
| pool 128³ zero | 5.396 → 5.309 (5.372 → 5.396) | 0.522 → 0.513 (0.522 → 0.518) |
| pool, 1 tile | 6.897 → 6.915 (6.976 → 6.892) | 0.716 → 0.672 (0.704 → 0.672) |
| pool, 8³ tiles | 8.644 → 8.445 | 0.958 → 0.855 (0.971 → 0.864) |
| pool, 16³ tiles | 12.651 → 12.388 | 1.348 → 1.108 (1.389 → 1.133) |
| pool Dynamic, lane 1/30, 2,506 h | 18.706 → 18.165 (19.011 → 18.277) | 2.545 → 1.963 (2.554 → 1.985) |
| pool Dynamic, app 1/60, 1,450 h | 13.701 → 13.496 (13.853 → 13.410) | 1.634 → 1.405 (1.638 → 1.419) |
| pool Full | 42.208 → 40.898 (42.340 → 41.038) | 3.752 → 2.572 (3.738 → 2.576) |
| 256³ zero, single arms | 11.75 → 11.72 without the two empty passes | 1.519, 1.517 → 1.523, 1.504 |
| 256³ first tile, single arms | 14.561 → 14.616 (paired −0.10 ± 0.09) | 1.927 → 1.934 |
| 256³ Dynamic, lane 1/30, 13,621 h, single arms | paired −1.31 ± 0.16 | 7.186 → 6.065 |
| 256³ Dynamic, app 1/60, 20,819 h, single arms | 86.12 → 83.48 (paired −2.18 ± 0.12) | 15.22 → 12.58 |

Zero-detail split sums at 128³: extension 0.565 → 0.558 and 0.560 → 0.552 in the other order, the rest 5.903 → 5.887 and 5.841 → 5.833. At 256³ the one split pair reads 1.503 → 1.559 with the untouched `publishList` moving most (128 → 138 µs): arm scatter, as the four single arms say. The 256³ Dynamic frame medians (46.92 → 47.33) hide the change behind frame scatter; every pass median is equal or lower (sum 48.36 → 46.91) and the paired per-frame difference is the figure above. Extension per h tile over the zero arm: pool Dynamic 0.81 → 0.58 µs, 256³ at 1/60 0.66 → 0.53, Full 0.099 → 0.063.

*What did not pay.* A job-level test on facing layers alone (without the per-cell test) gave 0.08 ms at Full and nothing under Dynamic. Serving a queued request with only the search half of `umNeighbor` (no prefix loads) is bitwise equal and worth 0.015 ms; not kept, since it needs a width-4 corner case (the all-unit stencil above the plane) argued by hand. A request queue compacted across tiles was not built: a job's queue is bounded only by 240, so a dense list is 240 words a seam tile at capacity or an overflow path.

*Left.* Under Dynamic at 128³ the two sweep seam launches are 1.22 ms of the 1.96: seam h 0.59 (0.30 at the regular rate, 0.14 the queue's structure, 0.13 searches), seam 4h 0.43 (0.25 of it searches), packs 0.14. The searches are now the largest part, 0.38 ms a frame, two thirds on the 4h side, and they cost by job, not by request: a job with three requests occupies a workgroup for one `umNeighbor` chain (its site lookup, and for a width-4 patch over h tiles four tied site lookups). The lever is that chain: the candidates of a seam request are fixed by the staged widths (a unit request into a 4h tile reads that owner's two faces on the patch axis; a width-4 request over h tiles reads four unit patches), so they can be built from the staging without `umVelocitySite`, keeping `umNeighbor`'s selection arithmetic so the result stays bitwise. Publish (0.26 ms) and seed (0.12 ms) are at the regular rate.

### Boundary cost per seam tile, 5 October 2026 (late)

**Landed: sharpening's seam tier and surface volume's `dilate` cost what their seam faces cost. State is bitwise equal on every arm measured.** Each change is in the stage's own file; no launch, receipt or frame wiring changed.

Cost per seam was isolated on the pool at 128³ with 512 h tiles in four Requested layouts (a 16×2×16 slab, 64 boxes of 2×2×2, 256 columns of 1×2×1, 16 rods of 16×2×1: 640, 1,536, 2,496 and 1,568 seam faces), as increments over the zero arm of the same job.

| per frame | before | after |
|---|---|---|
| sharpening sweeps | 0.71 µs per 4h tile with an h tile anywhere in its 3×3×3, whatever its faces (boxes +1.99 ms, columns +1.97) | 0.33 µs per 4h tile with an h face neighbour + 0.10 µs per seam face (boxes +0.56, columns +0.66) |
| sharpening geometry | 0.08 µs per such tile (boxes +0.21 ms) | the same rate, face tiles only (+0.08 ms) |
| surface volume | 0.09 µs per tile + 0.045 µs per face (`dilate` 0.20 / 0.37 / 0.43 / 0.29 ms) | flat (`dilate` 0.151 to 0.156 ms on all four) |
| coarse rows, divide + fallback | 0.085 µs per seam face (+0.05 / +0.15 / +0.23 / +0.11 ms) | not changed |
| extension (the seam agent's file; measured only) | 0.18 µs per tile + 0.62 µs per face + 0.42 ms for the 512 h tiles (+0.96 / +2.03 / +2.63 / +1.60 ms) | |

Sharpening (`uniform-mixed-sharpening.ts`): only a 4h tile with an h tile across a face is a seam tile, and edge and corner tiles join the regular 4h list; `compact` writes the six side bits into the flag word, so a seam lane past its side's patches does no lookup and the reductions run over the real terms in the old order; a quiet seam owner leaves the sweeps as a quiet regular owner does (geometry walks every seam tile from a list at the end of the tier region, header word 7; the active ones form the list the sweeps and the receipt already used). What remains, 0.33 µs, is one 192-lane job per two seam owners. One lane per owner would by estimate halve it (0.3 ms on the pool under Dynamic); it is not done because `tests/helpers/uniform-sharpening-reference.ts` rewrites the sweep module by text and pins two owners of 96 lanes per job. `dilate` (`uniform-mixed-surface-volume.ts`): a 4h lane beside an h tile reads the facing layer's sixteen band values by address after one neighbour lookup. Rows are left: a grain-1 row is 125 real overlaps, any merge moves rounding, and the riser is 0.1 to 0.25 ms.

Interleaved pairs, median ms, old → new (paired mean difference ± standard error). The second arm of a pair reads slower on either tree (same-tree control +0.30 ± 0.16 ms at pool Full), so Full and the 256³ zero were also run reversed.

| | lane, 1/30 | app, 1/60 |
|---|---|---|
| pool 128³ zero | 6.92 → 6.83 (−0.14 ± 0.14) | 6.22 → 6.23 (+0.08 ± 0.08) |
| pool s = 1 | 7.16 → 7.09 (+0.07 ± 0.17) | |
| pool s = 8 | 8.82 → 8.75 (+0.07 ± 0.15) | |
| pool Dynamic | 19.96 → 18.91 (−0.80 ± 0.06) | 14.67 → 14.20 (−0.30 ± 0.17) |
| pool Full | 42.26 → 42.31 (+0.22 ± 0.20; reversed −0.13 ± 0.09) | 42.37 → 42.53 (+0.27 ± 0.16; reversed −0.22 ± 0.19) |
| fig-7 256³ zero | 15.58 → 15.50 (+0.40 ± 0.38; reversed −0.34 ± 0.32) | 16.16 → 16.20 (+0.16 ± 0.17) |
| fig-7 256³ Dynamic | 46.35 → 46.19 (+0.19 ± 0.28) | 85.73 → 84.61 (−1.10 ± 0.24) |
| fig-7 256³ Full | 126.8 → 130.2 (+2.0 ± 0.4; reversed −2.4 ± 0.5) | |

The 256³ lane run has 45 active seam tiles in its window, so only surface volume moves there (4.08 → 3.83 ms).

**The Dynamic frame in four classes (split arms, mean ms, after).** Base is what the zero arm pays for the same entry; fixed is census, layout build, remap and publish; boundary is extension's seam and coarse tiers, hanging taps, phi resolve, the general (uncertified) momentum, advect and force kernels, the excess of 4h kernels over zero, and the seam share of sharpening by the fit above; h interior is the rest.

| | zero | Dynamic | = base | + h interior | + boundary | + fixed | Full |
|---|---|---|---|---|---|---|---|
| pool 128³, lane, 1/30, 2,506 h (before) | 6.48 | 21.94 | 6.34 | 7.45 | 6.87 | 1.27 | 44.05 |
| same, after | 6.55 | 20.77 | 6.37 | 7.19 | 5.99 | 1.22 | 43.76 |
| pool 128³, app, 1/60, 1,450 h | 6.59 | 16.16 | 6.50 | 4.80 | 3.83 | 1.03 | 44.20 |
| fig-7 256³, lane, 1/30, 14,551 h | 13.35 | 52.18 | 11.91 | 23.46 | 12.86 | 3.94 | 131.32 |
| fig-7 256³, app, 1/60, 21,941 h | 13.37 | 82.02 | 12.77 | 40.80 | 23.77 | 4.67 | 143.28 |

Boundary by stage, pool / 256³ at 1/30 / 256³ at 1/60: extension seam and coarse tiers 1.99 / 4.13 / 7.93; general momentum 0.91 / 3.76 / 6.04; general advect 0.36 / 1.91 / 2.89; general forces 0.49 / 1.00 / 1.41; sharpening seam 0.90 (was 1.54) / 0.02 / 1.22; merged trace 0.09 / 0.62 / 0.83; phi resolve 0.17 / 0.35 / 0.52; hanging taps 0.16 / 0.36 / 0.54; surface volume 0.05 (was 0.18) / 0.11 / 0.16.

*Why Dynamic costs several times Full per h tile.* Half of its increment is boundary and fixed, not h work. Its h tiles are all surface-band tiles, where Full's average runs over every tile of the lattice. And the regular semi-Lagrangian kernels require no 4h tile inside the tile's speed-reach box (`certify`), which a band one to three tiles thick almost never satisfies: at 256³ and 1/30, `momentumUnitStep` is 2.9 ms and `advectFine` 1.9 ms against 0.04 ms each for the regular entries, although 47% of the h tiles have all 26 neighbours h.

*Why 256³ costs 85 ms at 1/60 and 46 ms at 1/30 over the same simulated second.* Single-arm medians at 1/30 and 1/60: zero 16.2 and 16.6, Dynamic with lane settings 46.2 and 88.1, with app settings 47.6 and 84.5, Full 125.8 and 144.2. The step is the cause, not the settings. The 1/60 window holds 1.3 to 1.5 times the h tiles and 1.5 to 1.6 times the seam faces (about 18 of the 30 ms between the split arms); the rest is surface state. Sharpening has nine times the active owners (45 k against 407 k; under Full 44 k against 172 k) and goes from 1.3 to 7.5 ms; redistance goes from 2.7 to 6.4. At t = 2 s the Full state has 1.07 M partially filled h cells at 1/60 against 0.58 M at 1/30: the liquid is more dispersed at the smaller step under every policy.

*Two measurement traps.* At zero detail the `phi resolve` and `hanging fine taps` passes are opened with no dispatch and their timestamps bracket idle time (0.2 to 0.9 ms at 128³, 1.4 to 4.7 ms at 256³, varying with device load), so unsplit zero totals overstate the work (256³: 16.2 ms unsplit, 13.4 split). Arms stepped in one process slow each other at 256³ (Dynamic 62.6 ms beside two other arms, 46.2 alone); the 256³ class figures are single-arm.

### Regular against general fine kernels, 5 October 2026 (evening)

**Nothing landed. The certificate is not the lever: on the same tiles the general kernels cost 1.25 to 1.5 times the regular ones, and no tile-grain certificate reaches a thin band.**

Same-tile rates: Full with every h tile forced onto the general list (scratch root) against Full as is, and a 32×8×32 Requested slab at the pool surface whose inner 2,000 tiles certify.

| general ÷ regular | slab, live surface tiles, µs per tile | Full 128³ | Full 256³ |
|---|---|---|---|
| momentum | 0.104 ÷ 0.083 = 1.26 | 1.28 | 1.25 |
| phi and V advect | 0.110 ÷ 0.074 = 1.49 | 1.40 | 1.42 |
| trace cells | 0.031 ÷ 0.025 = 1.26 | 1.36 | 1.32 |
| redistance | +0.025 | 1.08 | 1.03 |

The general entries already take the unit path per sample (`umUnitSampleKind`, `umFineStencilSample`), so that ratio is all a certificate can return: 0.19 of the 1.85 ms in the four general rows at the pool, 1.3 of 7.3 at 256³ and 1/30, 1.75 of 11.2 at 1/60, if every band tile certified. Few can: h tiles with all 26 neighbours h are 4% (pool, lane), 20% (pool, app), 47% and 48% (256³). A margin cut to the taps read (minus ⌈(T+1)/4⌉, plus max(2, 1+⌊(T+1)/4⌋) tiles for T cells of travel, against ⌈T/4⌉+2) moved the regular share 0 → 4% (pool, app), 0.1 → 4% (256³, 1/30) and 4 → 23% (256³, 1/60): about 0.5 ms at 256³ 1/60, inside ±2 ms of run noise. Forces is not on the certificate: its regular entry takes tiles with a uniform 3×3×3.

The "general momentum" row is the unit step near the regular rate (0.34 / 2.9 / 3.3 ms), the excess of seam 4h tiles (0.29 / 0 / 0.97) and `momentumDeferred` (0.28 / 0.83 / 1.77), which re-traces with the general sampler each cell whose characteristic sampled a 4h tile: 14% / 3% / 10% of h cells, 14 to 20 of 64 per listed tile, one 192-lane job per tile. Packing 64 listed cells to a round (scratch root; handed back as a diff, not applied): 0.277 → 0.200 ms (pool, lane), 0.135 → 0.113 (pool, app), 0.835 → 0.593 and 1.765 → 1.271 (256³), state bitwise equal at every frame hashed. Packed, a general trace still costs four unit traces per face; that, not certification, is the remaining lever.

### Figure 9: Full against Dynamic, 5 October 2026 (night)

**Full beats Dynamic on Figure 9 only between about 1.5 and 3.0 s, when Dynamic's band is 26 to 37% of the tiles. A band tile costs four to six times what a tile costs under Full, because the band holds nearly all the tiles that are expensive under Full too (the surface), and on top of them Dynamic pays for its seams and for moving the band.** Over the whole run Dynamic is 1.0 ms ahead (26.9 against 27.9 ms); at the worst window it is 5.7 ms behind. Nothing was landed: the two changes that lower the frame (no hold, a filled band) did not pass, for the reasons below.

**How it was measured.** `mass-conserving-figure-9-dam-break`, 128×128×64 cells (16,384 tiles), dt 1/60, 240 frames from reset, one Dawn process and one device for every arm, compute passes only. Dynamic is the app's (`UNIFORM_DETAIL_APP_IMPORTANCE`: tolerance 0.5, thin, approach, hold 8); zero is Requested with no region. CPU is the synchronous wall time of `advanceTo` (plan, encode, submit) plus the synchronous extraction encode; sim GPU is the sum of the timestamp-query pass times; extraction is the renderer's two compute passes ("Extract water isosurface", "Polygonise water isosurface") through the production `RasterWaterPipeline.encode` with a no-op render pass, forced every frame. Every figure is a median over its window. Dynamic is repeatable when every frame is awaited (repeat runs give equal volume, velocity, phi and tile hashes at frames 120 and 240, and equal tile counts every frame), so arms forked from it at frame K start from the same state. Root pressure ran 1 cycle (3 to 3.5 encoded) and the band 4 cycles in every arm and window.

**The crossover.** ms per frame as CPU / sim GPU / extraction = total.

| t (s) | zero (272 h) | Dynamic | Full | Dynamic h tiles | seam faces | flips a frame | of them reversed within 3 frames |
|---|---|---|---|---|---|---|---|
| 0.0–0.5 | 3.24 / 9.63 / 0.62 = 13.49 | 3.31 / 11.21 / 0.93 = 15.45 | 3.03 / 19.43 / 0.98 = 23.44 | 566 | 516 | 64 | 15 |
| 0.5–1.0 | 3.19 / 9.81 / 0.67 = 13.66 | 3.30 / 15.22 / 1.29 = 19.81 | 3.05 / 20.17 / 1.01 = 24.23 | 1,587 | 1,175 | 189 | 23 |
| 1.0–1.5 | 3.31 / 9.99 / 0.86 = 14.16 | 3.34 / 18.70 / 1.76 = 23.80 | 3.00 / 21.86 / 1.27 = 26.13 | 2,535 | 1,693 | 237 | 71 |
| 1.5–2.0 | 3.17 / 10.49 / 1.29 = 14.95 | 3.39 / 25.06 / 2.50 = 30.95 | 3.04 / 23.82 / 1.51 = 28.37 | 4,234 | 2,796 | 385 | 92 |
| 2.0–2.5 | 3.19 / 10.96 / 1.90 = 16.05 | 3.37 / 32.35 / 3.08 = 38.80 | 3.13 / 28.00 / 1.97 = 33.10 | 6,141 | 4,296 | 768 | 177 |
| 2.5–3.0 | 3.25 / 11.92 / 2.46 = 17.63 | 3.37 / 25.88 / 2.51 = 31.76 | 3.03 / 25.24 / 1.82 = 30.10 | 4,655 | 2,553 | 527 | 86 |
| 3.0–3.5 | 3.21 / 11.91 / 1.89 = 17.00 | 3.36 / 21.50 / 1.81 = 26.66 | 3.06 / 23.27 / 1.38 = 27.71 | 3,705 | 2,056 | 291 | 73 |
| 3.5–4.0 | 3.18 / 11.27 / 1.16 = 15.61 | 3.41 / 22.55 / 2.17 = 28.14 | 3.15 / 25.36 / 1.63 = 30.14 | 3,861 | 2,426 | 269 | 69 |
| mean | 15.32 | 26.92 | 27.90 | 3,411 | 2,189 | 341 | 76 |

Full's pressure band is 4,030 to 5,045 tiles through the run and Dynamic's 395 to 2,711; Full has no seam and no flip.

**Rates.** Over a true zero (10.7 ms on this scene, below), Full costs 1.05 µs a tile here, 1.07 on pool 128³ and 0.83 on dam64. Dynamic costs 4.7 µs a band tile here, 6.2 on the pool and 4.1 on dam64. So Dynamic loses once its band passes about a fifth of the lattice: the pool's band is 3.7% of the tiles (18.9 against 46.6 ms), Figure 9's peaks at 37%, and dam64's is 34% on average, where Full wins the whole run (12.09 against 14.39 ms; zero 8.71). Full's own cost is not per tile either: with the same 16,384 tiles it runs from 23.4 to 33.1 ms as the surface grows. The 10,000 tiles Dynamic keeps at 4h at the peak are air and deep bulk, which Full passes through cheaply.

**Where the 5.7 ms is (frames 121 to 150, GPU ms by pass, zero / Dynamic / Full).** Extension 0.83 / 3.97 / 2.00; redistance 1.30 / 3.02 / 1.74; sharpening 0.67 / 3.32 / 2.60; momentum 0.29 / 2.17 / 1.18; remap and its publish 0.06 / 1.25 / 0; census 0.18 / 0.49 / 0; forces 0.17 / 0.76 / 0.45; phi resolve and hanging taps 0.07 / 0.55 / 0.07; extraction classify 0.49 / 1.17 / 0.59 and polygonise 1.41 / 1.92 / 1.39. Dynamic is ahead only in transport (2.86 against 3.95) and the band solve (2.28 against 3.88). By entry point (a split arm, one pass a dispatch): `sweepSeams` 2.72 ms against Full's `sweep` 0.75; the 4h `redistance` launch 1.25 ms against 0.01 (1.14 at zero: it does not shrink as the 4h tiles do, because the 4h vertices beside the band are still searched); `momentumUnitStep` + `momentumStep` + `momentumDeferred` 2.14 against `momentumRegularStep` 1.12.

**Adaptation against layout.** Arms forked from the same Dynamic state at frame K; total ms over frames K+2 to K+6. "Frozen" is Requested over regions that reproduce the captured tile set exactly (solid contact off, no census); "frozen + census" is the same tile set with the census running and no tile changing.

| K (t) | Dynamic | frozen + census | frozen | Full | adaptation (Dynamic − frozen) | layout (frozen − Full) |
|---|---|---|---|---|---|---|
| 60 (1.0 s) | 22.48 | 21.51 | 20.84 | 25.84 | 1.64 | −5.00 |
| 105 (1.75 s) | 32.08 | 29.68 | 28.73 | 31.48 | 3.35 | −2.75 |
| 135 (2.25 s) | 39.82 | 37.66 | 36.80 | 33.73 | 3.02 | +3.07 |
| 165 (2.75 s) | 28.65 | 29.29 | 28.08 | 29.13 | 0.57 | −1.05 |
| 210 (3.5 s) | 27.12 | 25.97 | 25.38 | 27.93 | 1.74 | −2.55 |

At the worst instant the 6.1 ms deficit is half adaptation and half layout. Elsewhere the mixed layout is cheaper than Full and adaptation takes back part of that (at K = 165 the band is shrinking and the frozen arms keep tiles Dynamic releases, so its 0.57 is low). Adaptation has a fixed part of 0.6 to 0.9 ms with no tile changing (census 0.33, the second frame plan 0.23 and authority phase 0.17, both tile-count sized, layout build 0.07, CPU 0.3) and a part of about 3 µs a flipped tile (at 776 flips: remap 1.03, remap publish 0.15, redistance +0.50 from the vertices a flip leaves stale, census +0.16). The remap already runs a worklist of the changed tiles and their neighbours; what it has too much of is flips, 22% of which undo a flip made within the last three frames. The layout's +3.07 at K = 135 is extension +2.17, redistance +0.95, momentum +0.71, phi resolve and hanging taps +0.44, forces +0.27, extraction +1.13, less transport 1.39 and the band solve 1.79.

**Shape, same instant.** The frozen tile set transformed on the host and applied as regions; ms against the frozen thin band, K = 60 / 105 / 135 / 165 / 210.

| shape | h tiles, seam faces at K = 135 | ms against the thin band |
|---|---|---|
| thin band (as Dynamic left it) | 6,466, 4,553 | 0 |
| closed (26-neighbour closing) | 7,294, 3,636 | −0.36 / −0.74 / −1.05 / −1.51 / −0.78 |
| filled (4h tiles with four h or wall faces, to a fixed point) | 6,705, 4,129 | −0.42 / −0.46 / −0.82 / −0.41 / −0.55 |
| one tile thicker, face neighbours | 9,420, 3,994 | +0.65 / +0.40 / +1.02 / −0.37 / +0.01 |
| one tile thicker, 26 neighbours | 10,756, 3,232 | +0.69 / +1.44 / +0.33 / −1.40 / −0.24 |
| 2×2×2 tile pages | 8,824, 3,744 | +0.14 / 0.00 / −0.17 / −0.71 / −0.35 |
| 4×4×4 tile pages | 12,416, 2,048 | +0.85 / +0.95 / −1.49 / −1.07 / −0.12 |
| whole columns of the band | 10,708, 3,447 | −0.33 / +1.63 / +0.07 / −0.94 / −0.62 |
| everything below the band's top | 13,512, 1,786 | +1.47 / +3.24 / +0.90 / +1.59 / +1.36 |

Closing and filling holes are the only shapes cheaper at every instant: closing adds 4 to 13% of the tiles and removes 10 to 40% of the seam faces. A ring is not: at K = 135 it adds 2,954 tiles to remove 559 faces, where closing adds 828 to remove 917. A tile pays where it removes about a seam face or more. Cost does not fall steadily to Full's either: "everything below the surface" (13,512 h tiles, 1,786 seam faces) is 37.70 ms against Full's 33.73, since the tile-count sized 4h work and the seam kernels last until the final 4h tile is gone.

**Shape, maintained by Dynamic from reset.** The thicker bands with today's controls, whole-run mean against 26.88: `detailMarginTiles` 1 → 27.04 (5,526 h tiles), 2 → 28.04; coarsening hysteresis 1 tile → 27.11 (flips 341 → 187, reversals 76 → 1, 4,389 h tiles), 2 → 27.38. The churn it removes is worth less than the ring it adds. A fill pass in the census was built as a probe in the scratch root (after `decide` and `solidPromote`: a 4h-decided tile joins the band when its six faces score a threshold, 2 a band neighbour and 1 a domain wall, in one or more rounds; with the pass off the state is bitwise the tree's at frames 120 and 240). Eight variants on Figure 9: mean 26.2 to 27.3 against 26.9, and 34.6 to 36.3 against 37.0 in the 2 to 3 s window (every variant better there, by 0.6 to 2.4 ms). Pool 128³: 18.69, 19.89 and 20.09 against 19.05. dam64: 14.35, 14.48 and 14.59 against 14.39. The consolidated band is cheaper where the band is large and fragmented and no better elsewhere, and from reset the difference is inside the scatter between trajectories. Not proposed as a default; the diff is in the session scratchpad (`fig9/fill-probe.diff`).

**The hold.** The app's 8-step hold is where most of Dynamic's extra tiles come from, and it cannot simply be dropped.

| whole-run mean ms (h tiles) | hold 8 (app) | 4 | 2 | 1 | 0 | zero | Full |
|---|---|---|---|---|---|---|---|
| Figure 9 | 26.88 (3,411) | 27.09 (3,263) | 26.67 (2,885) | 26.08 (2,607) | 24.39 (2,098) | 15.32 | 27.90 |
| pool 128³ | 18.94 (1,207) | 18.73 (1,054) | 18.41 (963) | | 17.74 (701) | 11.47 | 46.63 |
| dam64 | 14.39 (1,396) | 14.40 (1,316) | 14.17 (1,167) | | 13.82 (906) | 8.71 | 12.09 |

Hold 0 is 2.5 ms cheaper on Figure 9 and meets Full in the 2 to 3 s window (32.7 against about 31.6). But the band's pressure residual after its four cycles is above 4 on 44 of the 240 frames (1 of 240 with hold 8; largest 17.5 against 12.0), on dam64 on 6 of 180 against 1, and one of the three hold-0 arms that also ran the fill probe (two rounds) stopped at frame 236 on the fatal "pressure rejected a non-improving cycle". Holds of 1 to 4 flip more tiles than either end (391 to 531 a frame against 341 and 300) and are not reliably cheaper. So the app default stays at 8.

**What this says to do.** (1) The gross boundary cost is the largest item: about 6.7 ms at the peak for 4,553 seam faces (the +3.07 above plus the 3.6 ms of transport, band solve and list work the 4h tiles save), in `sweepSeams`, the 4h `redistance` launch, the general momentum entries and deferred pass, and the mixed-cube extraction, in that order. (2) Adaptation is 0.6 to 3.4 ms: the flips themselves, then the tile-count sized second frame plan and authority phase (0.4 ms, which could run over the builder's dilated changed list), then the census. (3) Shape is worth at most about 1 ms at the peak and nothing reliable elsewhere. (4) Refining fewer tiles is worth the most (2.5 ms) and is the one that costs pressure convergence, so it needs the band solve to hold its residual on a thinner band first.

**Found on the way, not fixed.** `mass-conserving-figure-9-dam-break` clones the default scene and keeps its solid voxel entries 7 to 12, which form a hollow box one cell thick over cells [0, 24]×[0, 16]×[0, 16], inside the reservoir corner. With solid contact on (the declared and app default) Requested with no region therefore holds 272 h tiles and runs the census every frame: 13.5 to 17.6 ms, against 9.7 to 11.5 ms with contact off at the same instants (sim GPU 11.36 against 7.89 and CPU 3.3 against 2.35 at K = 135). The "zero" column above is the former. **Fixed 6 October 2026:** `createMassConservingFigure9DamBreak` now clears the cloned `solidVoxels`, so the scene holds only its own container shell; Figure 9 numbers from before that change include the box. `hose-tank`, `dam-break-boxes` and `sphere-jet` (`createPaperScenario`) keep the same stale 24×16×16 shell.

**Boundary follow-up, 5 October 2026 (late night).** The 4h `redistance` launch is now two launches, and its cost follows the wide vertices it searches. Landed in `uniform-mixed-surface.ts`; every arm is bitwise equal to the tree before it (volume, velocity, phi and tile mask hashes: Figure 9 zero, zero with solid contact off, Dynamic and Full at frames 15, 45, 90, 135, 150, 210 and 240; the frozen forks at K = 60, 105, 135, 165 and 210 at K and K + 16; pool 128³ zero, one tile, Dynamic and Full at 15, 60 and 120; dam64 Dynamic at the seven frames).

- **What the launch did.** It ran one job per eight seam 4h tiles and per 64 regular 4h owners (1,024 workgroups at zero, 2,048 under Dynamic), and each job searched its own band vertices on its own lanes. A job holds few of them, unevenly (a seam job's 64 corners mostly belong to another tile; a pack's owners are mostly off the band), so nearly every job ran one Newton chain with most of its lanes idle. The time followed the jobs that held any band vertex, which is nearly all of them: the wide band is four 4h cells either side of the surface, and on this lattice that is 5,200 to 10,000 of Dynamic's 13,700 to 18,200 wide vertices and up to 15,900 of zero's 18,270.
- **What it does now.** The job launch classifies and appends its band vertices to one list (advect's deferred list, dead by then; one atomic per workgroup, the count in a spare claim word). A second launch, `redistanceBand`, strides a fixed grid over the list with one search per lane, on the same lagged grid as before. Both are Jacobi (they read `phi` and write `outputPhi`), so the order is free. `umRebuildBand` also tests the band before the travel gate and the preserve rule, so an off-band vertex loads no neighbour (the same truth table).
- **Measured**, Figure 9, the same pass split per launch (ms, 0.5 s windows): Dynamic 0.90 / 0.92 / 0.96 / 0.91 / 1.21 / 0.86 / 0.77 / 1.24 before, classify + search 0.61 / 0.67 / 0.71 / 0.74 / 0.86 / 0.63 / 0.54 / 0.69 after; zero 0.93 to 1.82 before, 0.60 to 1.17 after. The search launch is 70 to 90 ns a search (6,100 searches 0.53 ms, 15,300 searches 1.10 ms). A search runs 3.3 to 4.5 Newton steps on average (of at most eight), and 27 to 62% of them end as misses that keep the advected value. That is the floor for this band width; a narrower wide band or a cheaper miss changes results and was not tried.

| whole frame, ms (interleaved pairs) | before | after | redistance pass |
|---|---|---|---|
| Figure 9 Dynamic, whole-run mean | 26.86 | 26.50 | 1.98 → 1.67 |
| Figure 9 Dynamic, 2.0–2.5 s | 38.79 | 38.32 | 3.01 → 2.67 |
| Figure 9 zero (272 h), whole-run mean | 17.67 | 16.86 | 1.68 → 1.13 |
| Figure 9 zero, solid contact off | 13.91 | 13.40 | 1.95 → 1.19 |
| Figure 9 Full, whole-run mean | 27.86 | 27.85 | 1.23 → 1.23 |
| pool 128³ zero / one tile / Dynamic / Full | 11.76 / 12.60 / 17.76 / 46.10 | 11.55 / 12.60 / 17.55 / 46.08 | −0.06 / −0.07 / −0.04 / 0.00 |
| dam64 Dynamic | 14.13 | 14.05 | 0.83 → 0.77 |

The two zero rows ran while another job loaded the CPU (their CPU column is 3.6 to 5.8 ms); the paired simulation GPU difference is −0.45 ± 0.14 and −0.57 ± 0.12 ms, and the landed tree alone reads 14.77 (zero), 26.50 (Dynamic, worst window 38.36) and 27.75 (Full, 32.88). The band-first test adds −0.01 to −0.04 ms to the pass (Full included: −0.029 ± 0.003 on Figure 9, −0.034 ± 0.003 on the pool).

- **Dynamic's `redistanceFine` against Full's is parity, not excess.** At frame 135 Dynamic's 6,466 h tiles hold 4,464 tiles with a band vertex and 199,619 band vertices; Full's 16,384 hold 4,567 and 190,541. The launches read 1.76 and 1.75 ms.
- **`sweepSeams`, measured and not changed.** Repeating parts of a job is idempotent in a Jacobi sweep (hashes equal), so the added time of a repeat is that part's cost. At 2.0–2.5 s (2,753 seam h jobs, 2,793 seam 4h jobs, two sweeps, 2.75 ms): seam h jobs 1.42 ms, seam 4h jobs 0.94, regular 4h packs 0.07; the queued neighbour searches are 0.28 of the first and 0.33 of the second, 0.61 in all. So the searches are a fifth of the launch, not its bulk, and a global request list like the one above would save at most about 0.5 ms here. The larger remainder is the 4h side's 0.6 ms of plan and combine for 3 to 48 patches in a 64-lane job; packing several seam 4h tiles into a workgroup is the lever, and it means staging more than one tile's neighbourhood per job (`ueWords`, `ueWidths`, `ueMasks`, `ueJob`).
- **Travel gate, tried and not landed.** `uniform-coarse-redistance-dawn` is red because the gate (`umWideAdjacent`, landed 5 October) holds only a wide vertex with a face neighbour of the other sign, and the lane wants every corner of a cut 4h cell bit-unchanged at rest. Holding every corner of a cut wide cell (26 neighbours at the vertex's stride; a buried face neighbour still holds, a buried diagonal cuts nothing) turns that lane green and keeps `uniform-pond-rest`, `uniform-coarse-solid-rest`, `uniform-dynamic-coarsening` and `uniform-coarse-surface` green. It was not landed for two reasons. It costs +0.04 to +0.05 ms in the pass with the band tested first (pool zero +0.048, Figure 9 zero +0.037), since on a flat surface it holds the same two layers and every other band vertex walks all 26 neighbours. And Requested with no region on Figure 9 stopped on "pressure rejected a non-improving cycle" between frames 185 and 200 in 5 of 17 runs with it (the unperturbed run and 4 of 16 with one dry h tile added somewhere) against 1 of 17 with the gate as it is and 0 of 7 with the older hold of every wide vertex. Dynamic ran through with both. The one stop with the gate as it is says the all-4h Figure 9 run is close to that failure near 3.2 s whatever the gate.

**Adaptation follow-up, 6 October 2026.** Adaptation on Figure 9 is a fixed 1.2 ms of launches plus about 1.4 µs a flipped tile, and it changes almost nothing downstream. Little of it is work that can be skipped: two bitwise-exact cuts landed, worth 0.18 ms on Dynamic over the run and 0.31 ms in the worst window, and the larger of the two is a launch every policy was paying, not adaptation. The 5 ms by which Dynamic trails Full at 2.0–2.5 s is not reachable from this side.

- **By pass.** The same forks as above (live Dynamic, frozen + census, frozen), GPU ms by label and CPU, mean over frames K+2 to K+5 while the three layouts still agree; live minus frozen, with frozen + census minus frozen in brackets. Root pressure encoded 3 cycles and the band ran 4 in every arm at every fork.

| K | flips | remap worklist | h tiles | seam faces | remap | remap publish | census | second frame plan | second phase | layout build | phi resolve + fill | CPU | all labels |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 60 | 244 | 955 | 2,216 | 1,563 | 0.46 (0.05) | 0.07 | 0.42 (0.26) | 0.11 (0.10) | 0.09 (0.09) | 0.08 (0.07) | 0.06 | 0.66 (0.19) | 2.19 |
| 105 | 367 | 1,720 | 4,550 | 2,812 | 0.65 (0.05) | 0.10 | 0.46 (0.29) | 0.17 (0.14) | 0.14 (0.12) | 0.08 (0.07) | 0.08 | 0.62 (0.07) | 3.31 |
| 135 | 772 | 3,142 | 6,487 | 4,476 | 1.10 (0.06) | 0.17 | 0.50 (0.34) | 0.21 (0.21) | 0.15 (0.16) | 0.08 (0.07) | 0.10 | 0.36 (0.40) | 3.07 |
| 165 | 477 | 1,807 | 4,511 | 1,998 | 0.74 (0.06) | 0.11 | 0.48 (0.32) | 0.11 (0.16) | 0.11 (0.14) | 0.08 (0.07) | 0.04 | 0.42 (0.32) | 1.16 |
| 210 | 249 | 1,112 | 3,589 | 2,234 | 0.49 (0.05) | 0.07 | 0.44 (0.29) | 0.14 (0.14) | 0.10 (0.13) | 0.08 (0.07) | 0.06 | 0.35 (0.39) | 1.63 |

  The chain's own launches (census, layout build, second plan, second phase, and the CPU that encodes them) are the fixed part: 39 launches a frame, 0.7 to 0.9 ms of GPU and 0.3 to 0.5 ms of CPU (8 to 10 µs a launch; medians over K+1 to K+15 are 3.11–3.23 ms live, 3.05–3.17 frozen + census, 2.78–2.90 frozen). The remap and its publish are the per-flip part. Knock-on in later stages is at most 0.15 ms while the layouts agree (redistance +0.04 to +0.09, momentum ±0.05, sharpening 0); the larger differences a longer window shows in extension, redistance, sharpening and transport are the live band growing or shrinking against the frozen one (about 0.7% of its tiles a frame), which is layout, not adaptation. "All labels" includes that drift, so it is not the sum of the columns.
- **By entry point**, whole run, one pass a dispatch (means: 348 flips, 1,424 worklist entries, 3,475 h tiles; fit against h tiles and flips). Census 0.46 ms in 17 launches of 13 entry points, none over 0.11 (`classify` 0.019 + 0.024 a thousand h tiles, `classifyCoarse` 0.10 flat, `decide` 0.07). Layout build 0.10 in 8. Remap: `remapFaces` 0.37 for its two launches (0.10 + 0.70 a thousand flips), `remapCells` 0.23 (0.05 + 0.36 a thousand flips), `copyFaces` 0.10 for two (0.02 + 0.17), `markListed` 0.02. Frame plan `seed` 0.13 a run (0.04 + 0.024 a thousand h tiles) and its three dilates 0.02. Authority phase 0.11 a run. So a flip is 1.4 µs of remap and publish: 0.5 ms at the mean, 1.1 to 1.3 ms at the peak. The whole frame fits 12.9 ms + 2.2 µs an h tile + 2.8 µs a seam face (flips are collinear with seam faces; 13.2 + 3.5 µs an h tile + 4.0 µs a flip is the same fit).
- **Landed: the remap lists fewer tiles** (`uniform-mixed-remap.ts`, bitwise). A tile that stays h was listed whenever any of its 26 neighbours refined, to write the shared vertices it takes over. It takes one over only from a +face or +edge neighbour: equal-width authority is the lowest owner, h owners are numbered in tile order, and a +corner neighbour shares one tile corner, which every layout stores. Such a tile is now listed only for a face it remaps or one of those six neighbours, and the entry carries a bit (26; the tile index is 26 bits) so a tile listed for its faces alone leaves the cell kernel at once. Worklist 1,394 → 889 entries a frame on Figure 9 (3,051 → 1,931 at 2.0–2.5 s), 637 → 421 on the pool, 534 → 345 on dam64; remap pass −0.06 ms over the run, −0.16 at 2.0–2.5 s.
- **Landed: one phase launch a frame is gone in every policy** (`uniform-mixed-pressure-authority.ts`, `uniform-mixed-frame.ts`, bitwise). Phase is simulation state (the donor flag momentum and the extension read). The split's all-4h pressure authority stored its own phase into the same texture, which nothing reads, and that is why the tail extension, and the head on a static layout, rebuilt it with a launch over every owner. The all-4h authority now stores no phase, so the frame's own simulation authority (ahead of the band rows) is the last writer: the tail extension skips its phase launch unless rigid bodies moved after it, and a head skips it while `geometryCurrent` holds under the same parameters (every host edit of volume, phi, solids or ownership clears that through `invalidateExtension`). Dynamic keeps the launch after its relayout. Figure 9: Dynamic 0.216 → 0.108 ms, Full 0.118 → 0.004, zero (272 h) 0.082 → 0.041, zero with contact off 0.032 → 0; pool Full 0.229 → 0.025; dam64 Dynamic 0.150 → 0.075.
- **Whole frame**, interleaved pairs of the tree before and after, CPU + sim GPU + extraction, mean of the 0.5 s window medians and the worst window; the paired differences are per-frame medians.

| ms | before | after | worst window before → after | paired sim GPU | paired wall |
|---|---|---|---|---|---|
| Figure 9 Dynamic (three pairs) | 26.92 | 26.65 | 38.93 → 38.41 | −0.18 ± 0.06 | −0.20 |
| Figure 9 Full | 28.17 | 28.08 | 33.15 → 32.99 | −0.10 ± 0.06 | −0.13 |
| Figure 9 zero (272 h) | 15.22 | 15.12 | 17.47 → 17.51 | −0.04 ± 0.03 | −0.09 |
| pool 128³ Dynamic | 19.06 | 18.85 | 22.48 → 22.25 | −0.13 ± 0.09 | −0.15 |
| pool 128³ Full | 45.90 | 45.56 | 46.81 → 46.70 | −0.25 ± 0.19 | −0.33 |
| dam64 Dynamic | 14.83 | 14.62 | 17.28 → 17.16 | −0.09 ± 0.06 | −0.12 |
| dam64 Full | 12.49 | 12.37 | 13.25 → 13.18 | −0.06 ± 0.06 | −0.08 |

  Volume, velocity, phi and tile-mask hashes are equal before and after in every pair: Figure 9 Dynamic, Full, zero and zero with contact off at frames 15, 45, 90, 135, 150, 210 and 240; the frozen fork at K = 135 at K and K + 16; pool Dynamic at 15, 60, 120 and 180 and pool Full at 15 and 60; dam64 Dynamic and Full at the seven frames; and with surface tension on (0.0728 N/m, where the curvature cache uses the phase texture as scratch ahead of the authority) Figure 9 Full and zero with contact off at 15, 30 and 60 and Dynamic at 15, 45, 90 and 120. Zero with contact off is the timing trap named above: its frame wall time is 8.55 → 8.48 ms and its phase label is gone, but the sum of its pass times reads 1.2 ms higher, because the empty phi resolve pass now spans the GPU's wait for the next submitted segment (1.77 ms against 0.28, with no work in it).
- **In the owner's terms**, Figure 9 Dynamic after the change, mean / 2.0–2.5 s: base (zero, contact off) 11.3 / 11.4 ms; h-tile interior at 2.2 µs a tile 7.5 / 13.5; boundary at 2.8 µs a seam face less the flips 5.6 / 11.0; adaptation fixed 1.2 / 1.2; adaptation per flip at 1.2 µs 0.4 / 0.9. That sums to 26.0 and 38.0 against 26.65 and 38.41 measured; the fit's coefficients are from the tree before the change.
- **Flips were examined and left alone.** By the census's own reason, a frame holds 1,489 h tiles for closure, 1,082 as crossing seeds, 378 as held bulk, 187 for travel, 172 at solids and 153 as held crossings. Of 181 tiles refined a frame, 96 are travel and 85 closure; of 165 coarsened, 150 are closure and 12 travel; seeds under the 8-step hold almost never flip. 22% of flips undo one made within three frames and 37 to 40% within eight, all of them closure or travel tiles. Avoiding a reversal saves two flips, about 3 µs; holding the tile instead costs 2.2 to 3.5 µs for every frame it is held. A hold pays only for a one-frame gap, which is what the neutral 3-step band hold above already found, so no flip rule is proposed.
- **Measured or costed and not built.** The post-relayout phase over changed tiles only (0.11 ms): a phase depends on the owner's face neighbours (detached mass, the closed-cell continuation), so the exact set is the dilated list plus its face neighbours, which needs a mark and a compact launch to replace one launch; net 0.05 ms or less. The second frame plan over changed tiles only (0.15 ms): its maximum-speed word is a global reduction and residency changes with the layout; an exact version needs a per-tile speed word and another reduce, net 0.07 ms or less. One launch for the extension and velocity face remaps (0.13 ms mean, 0.29 at the peak): it needs both face fields bound through the detail accessor in one kernel, past the storage-buffer budget. The census is at its launch floor, and running it less often would break "declared defaults reproduce the old census".

**Boundary follow-up, 6 October 2026 (second round).** Two launches lost boundary work: `sweepSeams` now runs two seam 4h tiles a job, and a mixed extraction window forms its interpolated vertices from the base it has already loaded. Both are bitwise. At 2.0–2.5 s on Figure 9 Dynamic the simulation and extraction GPU time falls by 0.34 to 0.40 ms a frame (0.15 over the run); zero and Full do not move. Sharpening has no seam lever that keeps its reference helper, and that is stated below with its measured size.

- **`sweepSeams`, 4h side** (`uniform-mixed-extension.ts`, `ueSweepCoarsePack`, `UE_PACK_TILES = 2`). A job stages both tiles' 3×3×3 words, widths and masks in one round, gives each live patch of the pair its own lane (`uePackItem`: a prefix over the six low sides, so a pair with 3 patches uses 3 lanes and one with 96 runs a second round), shares one request queue of the old size, and flushes each tile's mask. The values, the plan and the combine are the single-tile job's, per patch.
- **What the launch pays for**, at the frozen 2.25 s layout (two sweeps, 3.1 ms in the pass, every arm's hashes equal). Barriers are free (an extra back-to-back barrier a job: 0.00). The grid is not the limit: capping it at 1,024, 256 and 64 workgroups adds 0.24, 0.79 and 2.94 ms, so 4,096 is already past the knee. Repeating a phase (idempotent in a Jacobi sweep) costs, for the 4h jobs, stage 0.02, plan 0.27, serve 0.15 and combine 0.18; for the seam h jobs, plan 0.15, serve 0.27 and combine 0.46 (1.05 for the whole job). Repeats overstate what removing work returns: packing measured −0.13 where they implied −0.33. So the remainder is per-patch plan, serve and combine on occupied lanes, not job overhead.
- **Pack size.** Fixed two-round packs were slower wherever seams are few (+0.03 to +0.04 ms on dam64 and in Figure 9's first second): a second serial round lengthens every job's lane chain. With patches compacted, the extension pass against the old structure by 0.5 s window reads +0.01 / +0.02 / 0.00 / −0.10 / −0.17 at two tiles a job and +0.03 / +0.04 / +0.02 / −0.14 / −0.27 at four, where the same kernel at one tile a job (the control) reads +0.02 in every window; on dam64 two tiles are level with the control and four are +0.03 above it (staging 108 words takes two rounds of 64 lanes). Two landed: it is never slower than the old structure.
- **Queued neighbour searches, not rebuilt.** Serve is 0.42 ms by repeat (0.15 + 0.27), perhaps half of that real. A closed form from the staged widths would remove the queue; it changes which lane forms a value and was not attempted in a pass that must stay bitwise. Evaluating only what sweep 1 changed in sweep 2 needs a copy path (the sweeps ping-pong, so sweep 2 must still store every value) and a changed mask. `publishSeams` (0.40) and `seedSeams` (0.13) stage differently and were left.
- **Sharpening, measured, nothing landed.** By window Dynamic reads 3.08 and 3.91 ms at 1.5–2.0 and 2.0–2.5 s against Full's 2.23 and 3.05 and zero's 0.62 and 0.76. Repeating the seam jobs in `propose` and `limit` (16 of the 24 sweep launches; `commit` prepares the next budgets in place and cannot be repeated) adds 0.43 ms at 2.0–2.5 s, and repeating the regular jobs adds 1.15; geometry is 0.52. So the seam 4h jobs are at most 0.64 ms of the 3.41 in the sweeps, for at most 1,460 jobs of two tiles (2,919 4h tiles have an h face neighbour) in which a tile with one or two h sides fills 21 to 37 of its 96 lanes. That occupancy is the lever, and it is the expression `tests/helpers/uniform-sharpening-reference.ts` pins (`shSeamTile(2u*(job-jobs.x)+lane/96u)`, with `shSweepJobs` and `shSeamLane`). A patch-compacted pack like the extension's would return an estimated 0.15 to 0.2 ms and needs that helper replaced; nothing else in the seam kernels is seam-face-proportional (idle lanes already return after one flags word).
- **Extraction: where a mixed layout pays.** By ablation (extraction feeds nothing back, so the simulation is identical in every arm), at 2.0–2.5 s with classify at 1.17 ms: skipping mixed (class 3) windows leaves 0.30, and leaves 18,325 of the 62,380 surface cubes. Under Dynamic the h band is about three tiles thick, so 70% of the surface lies in windows with a 4h tile in a stencil. Keeping only their vertex phase reads 0.76: the vertex phase is 0.45 and cells plus cubes 0.41. Replacing the interpolant with a stored load reads 0.81: `umVertexFrom4` at 4h-incident vertices is 0.36 of the 0.45, and it is the eight-term loop, not the texture: reading the same loop's corners from workgroup memory returns 0.05.
- **What it does now** (`webgpu-water-pipeline.ts`, `uniformWindowBaseValues`, `uniformWindowFrom4`). The first 27 lanes load the base; a mixed window then takes one barrier and its lanes interpolate from those 27 values, the eight terms as two vectors: the same texels, the same tile, `umVertexFrom4`'s weights ((x·y)·z, all exact quarters), its zero-weight skip (a `select`, as `uniformNormalBase` has it) and `umVertexSum8`'s pairing. Each product is rounded alone in both forms (a conditional store there, a `select` here), so the values are the same bits. A lane also merges its sign bits before one atomic. Classify by window: Dynamic 0.30 / 0.44 / 0.58 / 0.85 / 1.17 / 0.84 / 0.68 / 0.74 → 0.26 / 0.37 / 0.48 / 0.69 / 0.92 / 0.67 / 0.53 / 0.59; pool 128³ Dynamic 0.48 / 0.64 / 0.58 / 0.55 → 0.43 / 0.54 / 0.50 / 0.48; dam64 Dynamic 0.29 to 0.40 → 0.24 to 0.34; zero and Full −0.02.

| GPU = simulation + extraction, ms (interleaved pairs) | whole frame, mean | whole frame, worst window | GPU, paired median | GPU, worst window paired |
|---|---|---|---|---|
| Figure 9 Dynamic | 26.53 → 26.59 | 38.21 → 38.04 | −0.15 | −0.34 (34.60 → 34.34) |
| Figure 9 Dynamic, arms reversed | 26.26 → 26.23 | 37.95 → 37.89 | −0.16 | −0.40 (34.60 → 34.23) |
| Figure 9 zero | 14.88 → 14.96 | 17.27 → 17.35 | 0.00 | +0.01 |
| Figure 9 Full | 27.88 → 27.91 | 33.02 → 33.04 | −0.02 | −0.02 |
| pool 128³ Dynamic | 17.68 → 17.70 | 19.59 → 19.67 | −0.11 | −0.13 |
| dam64 Dynamic | 14.24 → 14.24 | 17.08 → 16.82 | −0.05 | −0.04 |

  The whole-frame columns are within the lab's CPU noise and do not resolve either change: a root loaded into the lab process reads from 0.07 ms less to 0.17 ms more CPU a frame than the base root whatever it contains (probe roots that differ from the base only in extraction shader text span that range), the final root read 0.13 to 0.17 more in both orders, and neither change touches frame-path JavaScript or the launch count. The GPU columns are the result. This was not A/A-tested in a fresh process. The pack alone, before the extraction change: Figure 9 Dynamic 26.59 → 26.57 and 38.17 → 38.11, extension pass 3.98 → 3.80 at 2.0–2.5 s in both orders, simulation GPU −0.19 and −0.22 there; zero 14.86 → 14.88, Full 27.82 → 27.84, pool 17.66 → 17.62, dam64 14.18 → 14.18.

  Volume, velocity, phi and tile-mask hashes are equal before and after in every pair, for the pack alone and for both changes: Figure 9 Dynamic (both orders), zero and Full at frames 30, 120, 180 and 240; pool Dynamic at 15, 60 and 120; dam64 Dynamic at 30, 120, 180 and 240; the frozen fork at K = 135 at K + 40 for every probe arm. The extraction header (cube and vertex counts) is equal every fourth frame of every pair, and `uniform-surface-extraction-dawn` holds the window scan's worklist and mesh equal to the full scan's at zero, partial and full detail.
- **In the owner's terms**, Figure 9 Dynamic at 2.0–2.5 s: base 11.4 ms and h-tile interior 13.5 are unchanged (zero 0.00, Full −0.02); boundary 11.0 → 10.6 (the extension pass −0.18 and classify −0.25 by their own labels, −0.34 to −0.40 paired over the frame); adaptation 2.1. A seam face is 2.7 µs where it was 2.8.
- **Costed and not built.** The mixed window's cell phase (0.41 ms with its cubes): `uniformWindowCells` is two eight-term loops a cell; an h cell's weights are all 1/8, so kinds 1 and 2 are the same bits from the same inputs and differ only in the one tile corner a kind-2 cell may touch, which allows the all-h window's vector form with a corner fix-up. A 4h cell's three rounded products need the `select` form to stay unfused. Promoting windows to the all-h class by their own stencil is exact for the same reason but needs a 4×4×4 block of h tiles, which a three-tile band rarely holds. Rejecting listed windows still needs a per-tile phi range from the solver. Polygonise on mixed cubes (1.92 against Full's 1.39) was not opened.

**Surface and boundary attribution, 6 October 2026 (third round).** A quarter off Figure 9 Dynamic is not available from a change to the surface and boundary kernels one stage at a time. At the 2.25 s layout the seams cost 10.5 ms of the 22.0 ms of simulation above true zero, but as a dozen stage-sized pieces, the extension's 3.3 ms and eleven of 0.4 to 1.4 ms, each its own seam kernel; the largest single repeated reconstruction (the extension's cross-width neighbour search) is worth 0.34 ms of the mean frame and 0.82 ms of the worst window when removed outright. That removal was built and measured and is **not landed**: it is not bitwise, and it does not sit inside the reference's own perturbed spread on the pool or on dam64. Nothing in the tree changed in this round. All numbers are on the tree as it stood at 12:38 on 6 October, after the lid release, over-capacity rows, enclosed-air and travel-gate changes, in interleaved same-moment pairs.

- **How it was measured.** One Dawn process, frozen forks of Figure 9 Dynamic (app settings) at frame K = 135 (2.25 s), split passes, medians of frames K+3 to K+14, six layouts on the same state: D the Dynamic mask frozen, X only the tiles that held a crossing, X1 those plus their 26-neighbours, Dp the Dynamic mask plus its 26-neighbours, F all h, Z true zero. Crossing counts are each arm's own at K+8 (the mask is frozen and the surface moves on, so D's 2,641 crossing tiles at the freeze are 1,704 eight frames later).

| arm | h tiles | of them holding a crossing at K+8 | seam faces | seam h tiles (by face / by 26-neighbourhood) | simulation ms | extraction ms |
|---|---|---|---|---|---|---|
| D, Dynamic mask | 6,654 | 1,704 | 4,700 | 3,053 / 4,504 | 30.09 | 2.86 |
| X, crossing tiles only | 2,641 | 1,254 | 4,473 | 2,359 / 2,602 | 25.22 | 2.59 |
| X1, crossing plus neighbours | 7,287 | 2,166 | 4,095 | 2,815 / 4,024 | 31.13 | 2.64 |
| Dp, Dynamic plus neighbours | 10,990 | 2,449 | 3,305 | 2,356 / 3,487 | 32.34 | 2.63 |
| F, all h | 16,384 | 2,734 | 0 | 0 | 30.46 | 1.92 |
| Z, true zero | 0 | 0 | 0 | 0 | 8.07 | 1.28 |

- **Three-way fit** over the five non-zero arms (time above zero = a × crossing h tiles + b × other h tiles + c × seam faces; residual 0.28 ms on the total): 4.81 µs a crossing h tile, 0.66 µs any other h tile, 2.23 µs a seam face. In D that is 8.2 ms in h tiles holding a crossing, 3.3 ms in h tiles that hold none and 10.5 ms at seams. A negative entry is the fit trading a term against seams where the arms are nearly collinear, not a saving.

| stage, ms in D at 2.25 s | zero | h tiles with a crossing | h tiles without | seams | D | F |
|---|---|---|---|---|---|---|
| extension | 0.51 | −0.51 | 0.79 | 3.34 | 4.09 | 2.11 |
| hanging taps + phi resolve | 0.00 | 0.06 | −0.01 | 0.47 | 0.52 | 0.07 |
| phi advect | 0.57 | 0.87 | 0.31 | 0.82 | 2.55 | 2.63 |
| traceCells | 0.11 | 0.32 | −0.02 | 0.05 | 0.49 | 0.54 |
| redistance | 0.94 | 0.93 | −0.19 | 0.50 | 2.24 | 1.79 |
| transport + cleanup | 0.61 | 1.71 | 0.61 | 0.38 | 3.35 | 5.00 |
| surface volume | 0.70 | 0.96 | −0.06 | 0.41 | 2.08 | 2.03 |
| sharpening + geometry | 0.71 | 0.89 | 0.40 | 1.43 | 3.32 | 3.46 |
| momentum | 0.12 | 0.74 | 0.02 | 1.07 | 2.01 | 1.30 |
| forces | 0.08 | 0.09 | 0.10 | 0.50 | 0.79 | 0.46 |
| pressure band + authority | 0.20 | 1.50 | 1.24 | 0.84 | 3.71 | 6.28 |
| root pressure | 3.29 | 0.10 | 0.17 | 0.41 | 3.94 | 3.97 |
| other | 0.26 | 0.10 | 0.13 | 0.15 | 0.64 | 0.77 |
| **simulation** | 8.07 | 8.20 | 3.26 | 10.50 | 30.09 | 30.46 |
| extraction | 1.28 | −0.17 | 0.33 | 1.42 | 2.86 | 1.92 |

- **What the table says.** At the worst window Dynamic costs what Full costs (30.1 against 30.5 ms) with 41% of its tiles: Full pays 11 ms for 13,650 h tiles that hold no crossing, Dynamic pays 10.5 ms for 4,700 seam faces instead. 68% of Dynamic's h tiles are seam tiles by 26-neighbourhood, because its band is the crossing tiles plus about one ring (6,654 = 2,641 crossing + 2,849 one tile away + 929 two + 175 three + 60 beyond), so nearly every h tile runs each stage's seam kernel and an h tile costs 3.3 µs where Full's costs 1.4. Read stage by stage against Full's rate for the same tiles, the seam excess is extension 2.8, momentum 1.4 (unit step 0.26, deferred 0.58, the seam 4h part of the merged step 0.53), extraction 0.9, phi advect 0.85, sharpening 0.8, surface volume 0.8, redistance 0.6, forces 0.5, taps and resolve 0.5, transport 0.5: about 9.9 ms in ten stages, none above 3 ms. Skipping surface stages in h tiles with no crossing in reach returns little under Dynamic (its non-crossing tiles are the one-tile ring the surface can reach in a step, 3.3 ms for all of them over every stage) and something under Full, where 9,087 of 16,384 tiles are two or more tiles from a crossing: 0.5 to 1 ms a stage at 0.66 µs a tile. Only X moves the frame (−4.9 ms): that is a choice of layout, which belongs to the census, not to a kernel.
- **Other layouts.** The K = 135 coefficients predict D at K = 105 as 23.1 ms (measured 24.29: 3,902 h tiles, 1,866 crossing at the freeze) and at K = 165 as 22.5 (measured 22.94: 5,150 and 2,053). There D / X / Dp read 24.29 / 21.28 / 25.83 and 22.94 / 19.49 / 22.74 ms. Pool 128³ at frame 120 (1,180 h tiles, 688 crossing, 1,473 seam faces): Z 9.20, X 13.82, D 14.93, Dp 16.94, F 43.51 ms; D is 4.9 µs an h tile above zero where Full is 1.05, and the arms are too nearly collinear there to split three ways.
- **Repeated reconstruction, by quantity.** *Departure point:* five RK2 traces an h cell a frame against the same extended field (phi vertex, cell centre for V, three momentum faces), and again for each deferred item (`advectDeferred`; `momentumDeferred` re-traces all three faces of an escaped cell). They are different points, half a cell apart, so sharing means interpolating a departure map, which is the cell-centre compression already identified as the source of regenerated excess; not pursued. *Hanging-vertex phi:* materialised by phi resolve three times a frame and formed again per mixed window by extraction (now from the loaded base). *Extended face value across a width change:* searched (`umVelocitySite`, up to four candidate faces) once per request in each of the two seam sweeps; this is the candidate below. *Face geometry in sharpening:* an active owner's faces are derived 15 times a sweep for 8 sweeps, 120 `umFace` evaluations an owner a frame; the dose-scaled budgets never reach exactly zero, so there is no exact quiet-owner skip beyond the ones already there, and the seam lane expression is pinned by the reference helper.
- **Candidate, measured, not landed: extension within one width.** A neighbour request is answered only by the patch of its own width at its plane (`umNeighbor`'s direct cases); a request whose plane lies between tiles of another width has no answer. The sweeps then run inside each width's patch lattice and stop at a width change, and a patch left without a distance publishes the hierarchy's far value, as every patch beyond the two-patch reach already does. The search, the request queue and the plan and serve phases go; a seam job is stage, six loads a patch, combine, flush. Expected from the split attribution: 1.5 to 2.0 ms of `sweepSeams` at the worst window. Measured: `sweepSeams` 1.91 → 0.78 ms (frames 100–125) and 2.49 → 1.18 (125–150); what is left is per-job staging at about the regular kernel's rate. Bitwise on all-h and all-4h layouts.

| interleaved pairs, ms | whole frame, mean | whole frame, worst window | extension pass, mean | simulation, paired median | h tiles / seam faces at the worst window, before; after |
|---|---|---|---|---|---|
| Figure 9 Dynamic | 26.50 → 26.16 | 38.16 → 37.34 | 1.98 → 1.38 | −0.50 | 6,345 / 4,441; 6,328 / 4,462 |
| pool 128³ Dynamic (3 s) | 19.71 → 19.55 | 23.77 → 23.64 | 1.45 → 1.10 | −0.12 | 2,105 / 2,630; 2,316 / 2,913 |
| dam64 Dynamic | 14.32 → 14.23 | 17.28 → 16.84 | 0.88 → 0.72 | −0.13 | 2,361 / 1,006; 2,254 / 1,058 |
| Figure 9 Full | 27.87 → 27.83 | 33.12 → 33.11 | | +0.03 | hashes equal at frames 60 and 240 |
| Figure 9 true zero | 11.02 → 11.01 | 11.93 → 11.54 | | −0.16 | hashes equal at 60 and 240 |
| pool 128³ Full (2 s) | 46.24 → 46.47 | 47.43 → 47.73 | | +0.20 | hashes equal at 60 and 120 |

  Zero and Full run identical work before and after (equal volume, velocity, phi and mask hashes), so their −0.16 and +0.20 are the size of what this lab does not resolve. The trajectories differ after the first seam, so tile counts differ by window; the paired column is over the whole run.
- **Quality of the candidate.** Per scene, 17 runs before and 17 after (the unperturbed run and 16 with gravity scaled by 1 + n × 10⁻⁶), 4 s each, 102 runs: no fatal diagnostic, no GPU error, pressure converged on every frame, cycle counts identical (root 1, band 4). In-range is the fraction of samples of the candidate inside the reference family's min to max; a seventeenth reference run would read 0.89.

| in-range: unperturbed run; all 17 | volume drift | represented drift | max speed | front | h tiles | liquid vertices |
|---|---|---|---|---|---|---|
| Figure 9 Dynamic | 0.95; 0.85 | 0.80; 0.90 | 1.00; 0.92 | 1.00; 1.00 | 0.90; 0.92 | 0.90; 0.83 |
| pool 128³ Dynamic | 0.40; 0.77 | 0.85; 0.85 | 0.85; 0.82 | (at the wall throughout) | 0.75; 0.66 | 0.85; 0.76 |
| dam64 Dynamic | 0.25; 0.73 | 0.90; 0.76 | 0.65; 0.75 | (at the wall throughout) | 0.80; 0.81 | 0.75; 0.68 |

| accepted residual over 4,080 frames, before → after | root p50 | root p99 | root max | root mean | band p50 | band p99 | band max | band mean |
|---|---|---|---|---|---|---|---|---|
| Figure 9 Dynamic | 0.062 → 0.066 | 1.37 → 1.19 | 4.94 → 2.81 | 0.167 → 0.163 | 0.245 → 0.266 | 3.07 → 3.45 | 16.7 → 26.0 | 0.473 → 0.529 |
| pool 128³ Dynamic | 0.0171 → 0.0173 | 0.486 → 0.518 | 1.29 → 1.35 | 0.0490 → 0.0487 | 0.0388 → 0.0329 | 1.97 → 1.80 | 5.13 → 5.47 | 0.140 → 0.125 |
| dam64 Dynamic | 0.0698 → 0.0683 | 2.07 → 2.69 | 4.08 → 4.91 | 0.192 → 0.201 | 0.079 → 0.084 | 3.44 → 2.94 | 61.6 → 40.3 | 0.305 → 0.280 |

| liquid-indicator Hamming distance to the unperturbed reference at 4h vertices, family median / max, before then after | 1 s | 2 s | 3 s | 4 s |
|---|---|---|---|---|
| Figure 9 Dynamic (≈4,500 liquid vertices) | 146 / 175; 148 / 182 | 807 / 987; 825 / 946 | 812 / 1,030; 788 / 921 | 751 / 828; 752 / 931 |
| pool 128³ Dynamic (≈17,950) | 29 / 39; 64 / 76 | 63 / 118; 101 / 122 | 188 / 219; 220 / 273 | 156 / 206; 164 / 206 |
| dam64 Dynamic (≈1,820) | 183 / 221; 229 / 277 | 150 / 213; 144 / 223 | 128 / 160; 146 / 188 | 111 / 189; 143 / 174 |

  Figure 9 Dynamic is matched: every series sits inside the reference family at the expected rate, the fronts agree (3.2 in every run from 1.0 s), and the band residual's higher pooled mean is inside the run-to-run spread (per-run means 0.29 to 0.74 before, 0.34 to 0.74 after; rank test p ≈ 0.28). The pool is not: in its first second the reference family has not spread (160 h tiles at 0.2 s in all 17 reference runs, 124 in all 17 candidate runs) and the candidate's surface stays about twice as far from the reference as the reference's own perturbed runs through 1 s, and further at its extremes through 3 s. dam64 is not either: the root's accepted residual has a heavier tail (p99 2.07 → 2.69; the per-run maximum is above the reference family's maximum in 8 of 17 runs) and the unperturbed run's drift leaves the family three samples in four. Against Full as the truth (one unperturbed run each, mean Hamming distance over the first 2 to 2.5 s) the candidate is no further away than the reference: pool 540 → 523, dam64 214 → 193, Figure 9 523 → 538, and the two Dynamic runs differ from each other by a tenth of that on the pool. So the change is not shown to be worse, and it is not matched; at 1.3% of the mean frame it does not earn an exception. The lab's `ke`, `umax` and `vol` series read the h-resolution textures and so count only h tiles under Dynamic (0 on an all-4h frame, 5% of Full's on Figure 9 at 0.3 s); they follow the layout and were not used. Max speed is the solver's own.
- **In the owner's terms**, Figure 9 Dynamic, mean / worst window, unchanged by this round: base 11.0 / 11.9 (true zero, measured again), Dynamic 26.50 / 38.16, Full 27.87 / 33.12; interior 7.5 / 13.5, boundary 5.6 / 10.6 and adaptation 1.6 / 2.1 as fitted before. The candidate would have read boundary 5.3 / 9.8.
- **What would move the frame by the size asked.** The boundary is expensive because each stage has a second, general kernel for seam tiles, and most of Dynamic's tiles are seam tiles. Removing that is one design, not ten: give every h region a ghost ring (h storage over the 4h tiles beside it, filled from the 4h values once for each stage that reads them, restricted back where the 4h side owns the result), so every h tile runs the regular kernel and the 4h kernels read 4h values only. Costed from this attribution at 2.25 s: up to −10.5 ms of seam work, against the 2,919 4h tiles that share a face with an h tile becoming ghost tiles (+1.9 ms at the non-crossing rate if they run the stages, less if they are only filled, more if the stencils also need edge and corner tiles) and the fills: at best about −8 ms of 38 at the worst window and about −4 of 26.5 on the mean. This is arithmetic on the fit, not a measurement. It changes the numerics at every seam in every stage (the matched-quality comparison above, for all of them), needs at least 44% more h tile slots, and crosses the frame orchestration, the storage classes and the 4h entry points. It was not started. The cheaper alternative with a measured size is the layout: X (crossing tiles only) reads −4.9 ms at this state with the same seams.
- **Costed and not built.** An exact form of the candidate (answering a cross-width request from the staged widths in closed form instead of dropping it) keeps the values and removes the queue; it is bounded by the 0.34 / 0.82 ms above less its loads, and needs `umVelocitySite`'s three cases re-derived on staged data. Narrow-band skips under Full (above). The packed deferred-momentum change stays with its owner's decision. A patch-compacted sharpening seam pack stays at 0.15 to 0.2 ms behind the reference helper.

**Coarse side, 6 October 2026 (fourth round).** A quarter off the zero-detail frame is not available from the coarse pipeline. The true-zero frame is launches, not work, and every repair stage that was reduced or removed either breaks the flow or leaves the reference's own perturbed spread. One exact change landed: pressure slots encoded past the lagged need take a launch-minimal form, 84 fewer launches a frame at 128³ (36 on dam64), −0.3 to −0.6 ms of whole frame at every policy with equal hashes. Trees: the four-way attribution is on the tree before 11:57 on 6 October; everything else is on scratch copies of the live tree taken at 12:25 (after the lid release, over-capacity rows, enclosed-air and travel-gate changes) and at 13:45 (which also holds this landing and the held-distance census change), in interleaved same-moment arms.

- **How it was measured.** One Dawn process, compute passes only, app settings (dt 1/60, app importance), whole frame = host encode + simulation GPU + extraction, arms interleaved frame by frame with the reference first and last. Attribution: frozen forks of Figure 9 at frames K = 105, 135, 165 with each stage's launches split by tile class, medians of the frames after the fork; a seam tile is one whose 3×3×3 tile neighbourhood holds both widths; a split launch's zero-work floor is subtracted. Split passes add launches, so the totals read about 2 ms above the unsplit simulation; the shares are the result. Quality metrics integrate over owners (a 4h owner's V at its origin texel, its velocity at its three +face anchors, weight 64): kinetic energy at frames 6, 30 and 60 of Figure 9 reads 2.266 / 2.240 / 2.224 ×10⁵, 4.018 / 3.859 / 3.967 ×10⁶ and 6.305 / 6.290 / 6.241 ×10⁶ for zero / Dynamic / Full, with equal volume.

| frozen layout, ms | h tiles | seam 4h | regular 4h | seam h | regular h | shared | total |
|---|---|---|---|---|---|---|---|
| K = 105 Dynamic | 4,098 | 2.40 | 2.12 | 4.17 | 4.98 | 10.66 | 24.32 |
| K = 135 Dynamic | 6,466 | 3.67 | 2.09 | 5.80 | 5.76 | 12.08 | 29.40 |
| K = 165 Dynamic | 4,942 | 1.81 | 1.57 | 3.40 | 4.96 | 10.86 | 22.61 |
| K = 105 wet (every tile below the surface at h) | 11,295 | 1.46 | 1.66 | 2.26 | 10.99 | 13.25 | 29.62 |
| K = 135 wet | 13,512 | 1.74 | 1.46 | 2.83 | 12.94 | 14.33 | 33.30 |
| K = 165 wet | 7,330 | 1.16 | 1.14 | 1.82 | 8.02 | 12.31 | 24.45 |

| K = 135 Dynamic, ms | total | seam 4h | regular 4h | seam h | regular h | shared |
|---|---|---|---|---|---|---|
| extension | 4.05 | 1.14 | 0.31 | 1.90 | 0.39 | 0.32 |
| transport | 2.83 | 0 | 0.70 | 0 | 2.13 | 0 |
| sharpening | 2.79 | 0.78 | 0.18 | 0.98 | 0.85 | 0 |
| band solve | 2.31 | | | | | 2.31 |
| redistance | 2.09 | 0.35 | 0.41 | 0.78 | 0.49 | 0.06 |
| surface volume | 2.06 | 0.04 | 0.02 | 0.40 | 0.03 | 1.56 |
| momentum | 2.03 | 0.56 | 0.09 | 0.49 | 0.33 | 0.57 |
| phi advect | 1.81 | 0.46 | 0.14 | 0.66 | 0.49 | 0.07 |
| root V-cycle, Full-cycle (closed), rhs, authority | 4.19 | | | | | 4.19 |
| forces | 0.77 | 0.17 | 0.12 | 0.30 | 0.17 | 0 |

- **What the four-way split says.** Under Dynamic the 4h side is 3.4 to 5.8 ms, and it is two different things. Regular 4h (1.6 to 2.1 ms) is the zero-detail frame's own stage work, launch floors included; it does not grow with detail and cannot be removed without removing the base. Seam 4h (1.8 to 3.7 ms) is the general kernels of the 4h tiles beside h tiles: extension sweeps, sharpening, momentum, phi advect, redistance. It follows the seam count, not the 4h surface, and goes only with the seam kernels themselves (the ghost ring of the round above). The wet layout against Full at the same state (36.78 against 33.01 ms whole at K = 135, 32.24 / 32.12 at 105, 26.29 / 26.99 at 165) is the seam itself and the 4h wide-band redistance: the wet mask still carries 2.8 ms of seam h and 1.7 of seam 4h work at K = 135 that Full does not have, against Full's extra regular h tiles in air.
- **Who consumes the 4h side.** Nothing was found dead. The root pressure is consumed every frame (boundary data of the band solve, and the 4h projection). Seam 4h phi advect, redistance and extension are read by the seam h kernels beside them in the same frame. Regular 4h transport rows are the conservation of V. Regular 4h sharpening and redistance are consumed wherever a 4h tile owns the surface. The only work that merely keeps far state tidy is the wide redistance band past any reader: 1.1 to 2.0 thousand of about 10 thousand wide vertices, at most 0.10 to 0.15 ms.
- **The zero-detail frame, by pass** (medians, GPU / host ms, before the landing; 13:45 tree for Figure 9 and the pool, 12:25 for dam64).

| pass | Figure 9 true zero | pool 128³ zero | dam64 zero |
|---|---|---|---|
| root V-cycle (the one that runs) | 1.08 / 0.30 | 1.34 / 0.33 | 1.04 / 0.20 |
| root Full-cycle (encoded, closed) | 0.56 / 0.44 | 0.64 / 0.53 | 0.45 / 0.29 |
| pressure setup, measure, checkpoints, gates | 0.32 / 0.19 | 0.37 / 0.19 | 0.37 / 0.18 |
| authority, transfers, rhs, project (the host figure holds the last submit) | 0.38 / 0.60 | 0.49 / 0.64 | 0.41 / 0.50 |
| redistance | 0.88 / 0.03 | 0.29 / 0.03 | 0.36 / 0.03 |
| surface volume | 0.52 / 0.05 | 0.29 / 0.05 | 0.31 / 0.05 |
| sharpening + geometry | 0.54 / 0.10 | 0.46 / 0.09 | 0.57 / 0.09 |
| extension | 0.47 / 0.06 | 0.51 / 0.06 | 0.43 / 0.06 |
| transport + live set + cleanup | 0.51 / 0.08 | 0.55 / 0.07 | 0.29 / 0.07 |
| phi advect, deferred, traceCells, resolve | 0.70 / 0.21 | 0.39 / 0.19 | 0.38 / 0.19 |
| momentum, forces, cache, taps, plan, certificate, fill, and the host before the first pass | 0.43 / 0.36 | 0.52 / 0.35 | 0.34 / 0.35 |
| **simulation** | 6.40 / 2.41 | 5.85 / 2.54 | 4.94 / 2.02 |
| extraction | 0.98 | 1.26 | 0.79 |
| launches a frame | 428 | 461 | 313 |

  About 204 of the launches run; the other 224 are the slots the pressure schedule encodes beyond the one V-cycle that runs (floor 3, reserve, spare Full-cycle), closed by a GPU gate. A closed launch costs 1.3 to 2.3 µs of GPU and about 1.5 µs of host inside a pass; a launch that runs costs about 9 µs before its work. The three submits and the final transfer carry 0.5 to 0.8 ms of host. The work itself (cells touched) is about 3.7 ms. So a quarter (2.4 ms) would need most of the remaining launch floor gone, and the floor is the stage structure: each stage is two to nine launches over the same few thousand 4h owners.
- **Landed: spare pressure slots in launch-minimal form** (`webgpu-uniform-pressure-multigrid.ts` and `.wgsl.ts`, `uniform-mixed-pressure-cycles.ts`, one loop in `uniform-mixed-frame.ts`, the entry list in `webgpu-uniform-reference.ts`). A slot at or past the lagged need (`max(1, V + Full − 1)` of the receipt two frames old; every slot wide on the first frames and under the full envelope) encodes iterations 2 to 11 of each tiled Jacobi visit as four launches of `mgSmoothTilesJacobiCone` instead of ten of `mgSmoothTilesJacobi`: each lane relaxes its own dependency cone (an octahedron of radius at most 3, two private arrays of 343) for 3, 3, 2 and 2 iterations and stores its centre. The values are the wide form's bit for bit. A V slot is 59 → 35 launches, a Full slot 165 → 105. The schedule's policy (floor, reserve, spare Full) is untouched: its file documents the fatals without it. A spare slot runs only on a frame whose cycle count exceeds the largest of the last four accepted frames (1 frame in 240 on Figure 9 zero and on dam64 Dynamic, none on the other four runs below), and then costs more than the wide form: +1.8 ms on that frame at Figure 9 zero, +2.8 pool, +1.1 dam64 (measured by forcing every slot to the cone form, which also showed equal hashes with the kernel running every frame).

| 13:45 tree, 240 frames, before / before again → after / after again | launches | whole frame, mean | median | worst 0.5 s window |
|---|---|---|---|---|
| Figure 9 true zero | 428 → 344 | 9.85 / 9.96 → 9.55 / 9.47 | 9.79 / 9.81 → 9.36 / 9.35 | 11.00 / 10.94 → 10.66 / 10.75 |
| Figure 9 Dynamic | 608 → 524 | 27.77 / 27.61 → 27.10 / 27.09 | 26.85 / 26.89 → 26.51 / 26.44 | 39.97 / 38.56 → 38.05 / 38.01 |
| Figure 9 Full | 567 → 483 | 28.20 / 28.24 → 27.80 / 27.78 | 27.58 / 27.65 → 27.28 / 27.13 | 33.21 / 33.48 → 32.81 / 32.97 |
| pool 128³ zero | 461 → 377 | 9.69 / 9.79 → 9.38 / 9.35 | 9.64 / 9.72 → 9.29 / 9.24 | 9.85 / 9.95 → 9.69 / 9.61 |
| pool 128³ Dynamic | 637 → 553 | 19.77 / 19.83 → 19.46 / 19.43 | 19.38 / 19.49 → 19.04 / 18.92 | 23.31 / 23.39 → 22.97 / 22.86 |
| dam64 Dynamic | 489 → 453 | 14.23 / 14.35 → 14.02 / 14.01 | 13.90 / 13.90 → 13.71 / 13.73 | 16.89 / 17.22 → 16.74 / 16.79 |

  Volume, velocity, phi and mask hashes are equal at frames 120 and 240 in all 24 arms. By label on Figure 9 zero: closed Full-cycle 0.56 / 0.44 → 0.42 / 0.33 ms GPU / host, V-cycle 1.08 / 0.30 → 1.04 / 0.26.
- **Repair and maintenance stages.** Cost is the pass's mean over a 4 s run on the 13:45 tree (Figure 9 zero; Figure 9 Dynamic; Figure 9 Full). Removal and reduction are single unperturbed runs unless a protocol is named; kinetic energy and the 4h-vertex liquid indicator are against the same run unchanged.

| stage | cost, ms | removed outright | reduced | condition that already skips it | verdict |
|---|---|---|---|---|---|
| redistance | 0.88; 1.86; 1.24 | Figure 9 zero: max speed 37 m/s (20.6), drift −6.6×10⁻³ (−1.8×10⁻⁴), energy up to 3.5×; −0.97 ms | | band and travel gate, per wide vertex | stays |
| sharpening, 8 sweeps | 0.54; 2.44; 2.10 | energy 1.5 to 2.6× at zero detail, 2.7× at 4 s under Dynamic, drift −1.3×10⁻²; −1.07 ms at zero | 4 sweeps: −0.14 zero, −1.1 Dynamic, −0.8 Full, −0.5 pool Dynamic, −0.4 dam64 | an owner with zero budgets leaves the sweeps; the dose-scaled budgets do not reach zero inside the band | 4 sweeps not matched (below) |
| extension front sweeps, 2 | 0.50; 2.14; 1.69 | Figure 9 Dynamic: energy 0.54× at 2 s and 28% fewer h tiles; pool: energy 2.2× at 1 s | 1 sweep: −0.1 zero, −0.9 Dynamic, −0.25 Full, −0.5 pool Dynamic, −0.25 dam64 | none; the hierarchy fills what the front does not reach | 1 sweep not matched (below) |
| surface volume, 2 secant rounds | 0.52; 1.40; 1.46 | Figure 9 zero: root residual p50 0.117 → 0.229, drift 5×, indicator 5 to 34% away; −0.52 ms | 1 round: −0.12 to −0.2; Figure 9 zero root residual max 1.93 → 4.13 and drift 2×, dam64 zero max 3.70 → 4.58 | a converged solve skips the second round's work, not its launches | stays |
| deficit balancing | inside the authority, unresolved | Figure 9 zero: residual p50 0.117 → 0.209, drift −2.5×10⁻³; dam64 zero: two-cycle frames, drift −3.9×10⁻³ | | | stays; nothing to gain |
| cubic phi advection | | drift −1.3×10⁻² (Figure 9 zero), energy 0.67 to 0.99×; −1.05 ms | | | stays |
| dust and orphan clearing | 0.04 at zero | no change in time | | | not a cost |
| root smoothing, 6 + 6 sweeps a visit | V-cycle 1.04 to 1.34 | 1 + 1: fatal, pressure rejected a non-improving cycle | 3 + 3: −0.86 Figure 9 zero, −0.82 pool, −0.49 dam64; accepted residual p50 0.117 → 0.542, 5 and 17 two-cycle frames where there were 1 and 0 | | convergence degrades: no |
| root slots beyond the lagged need | closed: 0.56 GPU + 0.44 host | lagged plan alone −1.1 ms; documented fatals (128³ dam far-wall impact, 3 of 8 perturbed all-fine runs) | spare form, above | runs when the cycle count exceeds the last four frames' largest: 2 frames in 1,440 | landed, exact |
| band solve, 4 fixed V-cycles | band solve 1.98 Dynamic; 3.63 Full; 1.35 pool Dynamic | | residual gate, below | per-cycle residual is already measured on the GPU; the target is the constant 0, so the gate never closes | convergence degrades: no |

- **Sharpening and extension sweep counts: 17-run protocol, not matched.** Reference and three candidates (4 sharpening sweeps; 1 front sweep; both), each the unperturbed run and 16 with gravity scaled by 1 + n × 10⁻⁶, 4 s, 13:45 tree, 136 runs: no fatal diagnostic in any. Cells are the family median; the reference column adds its min and max; bold is a candidate median outside the reference family.

| Figure 9 Dynamic | reference | 4 sweeps | 1 front sweep | both |
|---|---|---|---|---|
| kinetic energy at 1 s / 2 s, against the unperturbed reference | 0.984 [0.975, 1.012] / 0.956 [0.903, 1.003] | 1.001 / **1.006** | 0.975 / 0.907 | 0.997 / 0.931 |
| front at 0.5 s | 0.85 [0.75, 0.85] | 0.85 | 0.75 in all 17 | 0.85 |
| root accepted residual, per-run p50 | 0.0613 [0.0526, 0.0684] | 0.0674 | **0.0696** | 0.0676 |
| band residual, per-run p90 | 0.861 [0.596, 1.169] | 1.083 | 0.992 | 1.129 |
| max speed, m/s | 44.6 [38.0, 58.6] | 46.6 | 49.5 | 52.6 |
| indicator distance to the unperturbed reference at 1 s, % of liquid vertices | 3.32 [2.99, 4.04] | **4.06** | 4.04 | 4.04 |
| volume drift, ×10⁻³ | −5.41 [−6.70, −4.59] | −5.69 | −5.02 | −4.88 |
| mean h tiles | 3,282 [3,219, 3,538] | 3,385 | 3,239 | 3,259 |

| pool 128³ Dynamic | reference | 4 sweeps | 1 front sweep | both |
|---|---|---|---|---|
| kinetic energy at 1 s / 2 s / 4 s | 0.981 [0.914, 1.020] / 0.941 [0.900, 1.016] / 1.026 [0.971, 1.072] | **1.107** / **1.093** / **1.174** | **1.053** / 0.979 / **0.957** | **1.167** / **1.105** / **1.116** |
| root accepted residual, per-run p50 | 0.0177 [0.0156, 0.0196] | **0.0201** | 0.0181 | 0.0190 |
| volume drift, ×10⁻³ | −0.543 [−0.871, −0.280] | −0.761 | −0.674 | **−0.892** |
| max speed, m/s | 58.5 [38.9, 83.7] | 55.2 | 52.4 | 49.3 |
| mean h tiles | 1,345 [1,218, 1,572] | **1,634** | 1,339 | **1,587** |

  Halving the sharpening sweeps on the pool raises kinetic energy 11 to 17% above every reference run and the Dynamic census's h tiles by 21%, so part of the saving returns as tiles. One front sweep holds the Figure 9 front a sample behind the reference at 0.5 s in every run and moves the pool's energy out of the family at 1 and 4 s. dam64, Full and the zero arms were not run through the protocol after these two failed. Both counts were set at the lanes' 1/30 s step and the app takes twice as many steps a second, which is why they were tried; the measurement says the per-step dose is what the flow sees.
- **Band solve: a residual gate, measured, not landed.** The band runs four V-cycles every frame from the trilinear root pressure; its restriction already writes each cycle's residual to a history word and `bMet` closes the remaining launches at a target, but the frame writes the target as 0. With the target set (a hook in a scratch root; single runs):

| target (the band's own residual units) | Figure 9 Dynamic: frames stopping at 1 / 2 / 3 / 4 cycles of 220; band solve ms; band residual p50 / p90 | Figure 9 Full | pool 128³ Dynamic | dam64 Dynamic |
|---|---|---|---|---|
| 0 (the tree) | 0 / 0 / 0 / 220; 1.98; 0.205 / 0.85 | 0 / 0 / 0 / 220; 3.63; 0.449 / 1.51 | 0 / 0 / 0 / 220; 1.35; 0.042 / 0.25 | 0 / 0 / 0 / 220; 1.34; 0.074 / 0.43 |
| 0.25 | 0 / 5 / 81 / 134; 1.86; 0.245 / 0.89 | 0 / 0 / 44 / 176; 3.55; 0.388 / 1.52 | 0 / 119 / 61 / 40; 1.13; 0.142 / 0.36 | 0 / 33 / 105 / 82; 1.24; 0.180 / 0.59 |
| 1 | 6 / 115 / 58 / 41; 1.60; 0.562 / 0.95 | 0 / 82 / 69 / 69; 3.02; 0.739 / 1.84 | 102 / 97 / 13 / 8; 0.89; 0.565 / 0.90 | 11 / 147 / 49 / 13; 1.05; 0.490 / 0.88 |
| 5 (the root's tolerance) | 105 / 99 / 14 / 2; 1.28; 2.48 / 4.49 | 65 / 134 / 21 / 0; 2.23; 2.20 / 4.51 | 179 / 25 / 3 / 0 (13 at none); 0.76; 1.28 / 3.71 | 157 / 50 / 12 / 1; 0.81; 1.57 / 3.41 |

  A cycle is about 0.29 ms of the Dynamic band and 0.64 of Full's; 0.8 to 1.1 ms of the pass is its start, rows and closed launches whatever the cycle count. The smallest target (0.25) returns 0.1 to 0.2 ms and already moves the pool's median residual from 0.04 to 0.14; a target that returns 0.4 to 1.4 ms raises the median three to thirty times. Pressure convergence may not degrade, so it is not landed. The band's residual is not part of the frame verdict today either: its largest value in a run is 2 to 17 in the unit of the root's tolerance of 5.
- **The first h tile, pool 128³, Requested with one region, contact off** (medians, 12:25 tree, before → after the landing).

| whole frame, ms | 0 tiles | 1 tile | 8 tiles |
|---|---|---|---|
| static region | 9.56 → 9.19 | 11.49 → 11.24 | 11.68 → 11.37 |
| region moved one tile every frame | 10.17 → 9.89 | 12.85 → 12.41 | 12.94 → 12.60 |
| launches (static; moving) | 461 → 377 | 538 → 454; 571 → 487 | |

  The jump is unchanged by the landing: 1.9 to 2.0 ms for the first static tile, 0.1 to 0.2 for the next seven, 0.9 to 1.3 more when the region moves. By label for the first tile (GPU + host): band solve +0.57 +0.07, transport +0.26, extension +0.14, band list and rows +0.08 +0.03, redistance +0.08, momentum +0.08, forces +0.04 +0.04, phi advect +0.06, transfers +0.06, surface volume +0.03; simulation GPU 5.57 → 7.09 → 7.18 for 0 / 1 / 8 tiles, host 2.54 → 2.88 → 2.81, extraction 1.22 → 1.34 → 1.44. All of it is execution switched on by any mixed layout, and its largest single item is the band's 95-launch schedule over one tile.
- **What the 4h side would need for a ghost ring.** Under an h tile the base block holds injections, not restrictions: the origin h cell, the single h face at each +face anchor, the tile-corner vertex phi, current at every stage because every store of a canonical texel writes both copies. Vertex phi at aligned vertices is therefore already the restricted value, everywhere and always; the hanging vertices of a seam are materialised by phi resolve three times a frame (after advect, after redistance, after surface volume). Restricted velocity exists in three places: the 4h sampling cache (fine patches averaged once a frame after the head extension, current for every trace of that frame), the extension's coarse slots (written by `restrictBand` once after the final sweep of each run), and the all-4h pressure ownership between the two transfers. There is no restricted V under an h tile in simulation ownership at any stage. So 4h phi advect, redistance and the traces could read under h tiles as ordinary neighbours today; 4h extension could after one restriction a run that it already performs; 4h sharpening, surface volume and transport cannot until V is restricted for them once a stage, and their conservation needs each seam face's flux computed once for both sides. This is read from the code and the canonical-texel audit, not measured.
- **In the owner's terms**, Figure 9 at app settings on the 13:45 tree, mean / worst window: base (true zero) 9.9 / 11.0 → 9.5 / 10.7; Dynamic 27.7 / 39.3 → 27.1 / 38.0; Full 28.2 / 33.3 → 27.8 / 32.9. The landing takes the same launches out of every policy, so the cost above base (interior, boundary, adaptation) is as the rounds above left it. At the 2.25 s layout the split reads: boundary 9.5 ms (seam h 5.8, seam 4h 3.7), interior 5.8 (regular h), base work that detail does not move 2.1 (regular 4h), and 12.1 shared, of which the band solve is 2.3 and the root 4.2 (tree before 11:57, split passes).
- **Measured and not landed, besides the above.** Jacobi weights of 1 for seven iterations and 0.5 for two in place of twelve at 2/3 (the same sum of weights, high-frequency factor 0.041 against 0.049): root residual distribution equal to the reference's on the three zero scenes, −0.2 to −0.45 ms median, not exact, no perturbed family run; weights above 1 are fatal under the pressure floor's clamp. A lagged plan without the floor, reserve and spare Full (−1.1 ms) and one V plus one Full (−0.45): the documented fatals. Costed, not built: the band's 95 launches as one for a band of a few tiles (about −0.4 ms of the first-tile jump; each band kernel is its own module with its own workgroup size, so it is a rewrite), and a restricted V under h tiles (above).

**Ghost ring assessment and variant B, 6 October 2026 (fifth round).** The third round's "up to −10.5 ms of seam work for about +1.9 ms of ghost tiles plus fills" is not real, and that bullet is withdrawn. The ring it asks for already exists for the two fields the trace stages read: `unitVelocityTaps` fills all 192 h face taps of every 4h tile in the 26-neighbourhood of an h tile once a frame, and phi resolve writes every hanging vertex of the same tiles three times a frame. Their cost (0.5 ms at the worst window) is the fit's "hanging taps + phi resolve" row; the rest of the 10.5 ms is not missing ghost data. Three scratch prototypes were measured against the tree as it stood at 13:40 (after the census follow-up; `diff` against the live `lib` empty at 13:52): the extension's exchange removed (the third round's candidate, −0.34 mean / −0.82 worst), the momentum sampler reading the ring as h storage (−0.1 ms on its stage, nothing on the frame), and variant B live (−0.53 / −1.09 on Figure 9 with 15% fewer h tiles, not matched; a loss on the pool). With every convertible stage converted the envelope is at most about −4 ms of 38 at the worst window, of which 1.4 is measured at label level (0.8 on the whole frame), all of it inexact. Nothing landed and no file under `lib`, `tests` or `tools` was touched; this part is the only change.

- **What the code already does, stage by stage.** Read from the kernels and their samplers; "ring" is the 4h tiles with an h tile among their 26 neighbours (the tiles with a mixed 3×3×3 stencil, which are exactly the tiles that hold a hanging slot).

| stage | what an h kernel reads beyond its tile | where it comes from at a 4h neighbour today | what a fill launch could replace | 4h side under and beside h tiles | conservation |
|---|---|---|---|---|---|
| phi advect, `traceCells` | velocity along an RK2 path (8 taps a sample, travel u·dt), phi at the departure point (8 taps, 64 with the cubic) | velocity: the unit texture, whose ring texels `unitVelocityTaps` filled from the 4h cache; phi: the resolved texels of the ring. A sample that lands in a 4h tile blends the h and 4h interpolants over two cells (`umVelocitySamplingWeights`) | nothing: the fill exists. What differs from the regular kernel is a per-sample test of where the sample landed and the two-cell blend | reads the 4h cache (16 h faces averaged once a frame after the head extension) and aligned phi vertices, both current | none to keep (semi-Lagrangian) |
| momentum | the same path, and the advected component at the departure point | the same unit texture inside h tiles; a face whose path leaves the h tiles is listed and re-traced by `momentumDeferred` with the blended sampler | the escape, for paths that end inside the ring (prototype M1 below) | the merged 4h step reads the cache; a 4h tile with an h tile across a positive face owns that face's 16 h patches and traces each | none to keep |
| forces | viscosity at MAC sites one cell around; capillarity through the pressure seam reconstruction | viscosity reads the unit texture (ring filled); the regular entry needs a uniform 3×3×3 | the regular entry's condition could become "h or ring", for viscosity only | 4h owners read the cache | none |
| extension | a neighbour patch's (value, distance) per sweep, two patches deep over the sweeps | searched across the width change per request in every seam sweep (`umVelocitySite`, plan and serve) | the search: one fill of the ring's unit patches before the sweeps and one restriction after (the third round's candidate is the limit with no exchange at all) | `restrictBand` restricts once after the final sweep of each run | none (an extrapolation) |
| redistance | phi within the Newton window | resolved ring texels | nothing | aligned vertices | none |
| transport | donor cells under each row's departure box | a row takes the open part of its overlap with a 4h donor; every donor's V is divided among its samplers by exact fixed-point sums | nothing: ghost cells would be 64 aliases of one donor, and the stage conserves by normalising over a single partition of the mass | 4h rows sample h donors cell by cell | the donor sums; a ring would need a flux across the seam computed once for both sides instead |
| surface volume | owner corners, the band one owner around | corners completed in place by phi resolve | nothing | owners of every width in one reduction by physical volume | the global shift's volume sum |
| sharpening | the flux across each face to the neighbouring owner, eight sweeps | a 4h tile beside an h tile takes a lane per unit patch (16 a side); one flux a patch, applied to both owners | not by a fill: V in ghost cells changes every sweep, and the single flux per patch is the conservation | seam 4h tiles do h-resolution work on their seam faces | one flux per physical patch |
| pressure | root pressure at the band's edge | the root is all-4h under every tile (`toPressure` restricts, `toSimulation` returns) and the h band solves inside it | nothing: this stage already has the design's shape | the root reads restricted velocity under h tiles | the 4h divergence is the sum of the h ones |
| extraction | 6³ vertices a window | a mixed window forms its hanging vertices from the 3³ base it loaded, in workgroup memory | the per-window interpolation, if the published phi kept the resolved ring texels | 4h windows march the base | none |

- **Where the seam term is, by launch** (frozen fork at 2.25 s, 12:38 tree, ms; Dynamic mask, then all h, then true zero where it applies). The regular h lists are nearly empty under Dynamic: `advectOwners` 0.04 against `advectFine` 1.14 (all h: 1.72 / 0.02); `momentumRegularStep` 0.02 against `momentumUnitStep` 0.76, `momentumDeferred` 0.58 and the merged 4h step 0.65 (zero: 0.12; all h regular: 1.23); `forcesRegular` 0.09 against `forces` 0.64 (all h: 0.43); extension `sweep` 0.18 against `sweepSeams` 2.82 (all h: 0.78 / 0.04), `publishSeams` 0.41, `seedSeams` 0.14. "Regular" for the trace stages is the reach certificate, a box of ceil(travel / 4) + 2 tiles of h on every side, not the 26-neighbourhood: one real ring of h tiles (arm Dp) certifies 18% of the trace work, and a ring of ghosts cannot do better than a ring of real tiles. On the same tiles the general kernels cost 1.25 to 1.5 times the regular ones (5 October, evening), so the trace stages' "seam" is mostly h tiles running the general kernel at about the regular rate, plus the deferred lists.
- **Per stage: the fit's seam term against what a one-tile ring can take** (Figure 9 Dynamic, 2.25 s, ms).

| stage | seam, by the fit | convertible | basis |
|---|---|---|---|
| extension | 3.34 | 1.3 measured, up to about 2.0 | `sweepSeams` 2.49 → 1.18 with the exchange removed (third round); the remainder is per-job staging |
| hanging taps + phi resolve | 0.47 | 0 | these are the ring's fills, already paid |
| phi advect, `traceCells` | 0.87 | up to 0.4 | general against regular on the same tiles (1.49), for tiles whose travel stays in the ring; not built |
| momentum | 1.07 | 0.1 measured | M1 below: `momentumDeferred` −0.34, `momentumUnitStep` +0.25. The merged 4h step's 0.53 is traces of real seam faces |
| forces | 0.50 | up to 0.4 | regular against general per tile; viscosity only, capillarity keeps the seam reconstruction; not built |
| sharpening + geometry | 1.43 | up to 0.3 | seam 4h jobs are at most 0.64 by repeat count, and their expression is pinned |
| surface volume | 0.41 | up to 0.2 | not built |
| redistance, transport, pressure, other | 2.28 | 0 | no fill replaces them (table above) |
| extraction | 1.42 | up to 0.9 | mixed against all-h windows; needs the published phi to keep ring texels; not built |
| **total** | 10.5 + 1.4 | **at most 3.6 to 4.3 of 38; 1.4 measured** | new fills: one for the extension's ring patches, none elsewhere |

- **Prototype M1, the ring read as h storage in momentum** (scratch root, `surf/p_m1.py`; not landed, quality not assessed). `umUnitSampleKind` takes a sample in a ring tile as the unit interpolant when every tap lies in an h or ring tile, instead of escaping to the deferred list. Not exact: it replaces the two-cell blend by the h interpolant of the ring's filled taps. Interleaved split pair, Figure 9 Dynamic, frames 125 to 150: `momentumDeferred` 0.59 → 0.24, `momentumUnitStep` 0.93 → 1.18 with 5% more h tiles (6,017 → 6,302), the stage −0.10 ms; frames 100 to 125 the same (0.38 → 0.15, 0.71 → 0.83). The deferred traces are not repeated work: they are the same traces, moved. Whole frame, mean / worst window: Figure 9 26.24 → 26.38 / 38.07 → 37.77 (paired per-frame simulation median −0.12), pool 128³ 18.38 → 18.83 / 21.54 → 23.94 (the run with the change holds 2,332 h tiles against 1,772 in its last window; paired median +0.10), dam64 14.07 → 14.12 / 16.87 → 16.52 (+0.08). Null on the frame.
- **Variant B, live from reset** (scratch root, `surf/p_b.py`). The census's closure is the 0.5 cell drift margin and the closed box in `decide`; B removes both and keeps crossing, travel, solids and holds. It runs on today's kernels with no new fill, which is B with ghosts for the two fields above. Interleaved pairs, whole frame ms, h tiles / seam faces at the worst window:

| scene, Dynamic at app settings | mean | worst window | h tiles / seam faces, before; after | h tiles, time mean of the 17-run family |
|---|---|---|---|---|
| Figure 9 | 26.18 → 25.64 | 38.28 → 37.18 | 5,981 / 4,469; 5,009 / 4,373 | 3,371 → 2,875 |
| pool 128³ (3 s) | 18.08 → 18.41 | 21.37 → 22.80 | 1,772 / 2,110; 1,758 / 2,487 | 1,392 → 1,272 |
| dam64 | 14.11 → 13.94 | 17.14 → 16.94 | 2,377 / 938; 2,011 / 1,096 | 1,326 → 1,132 |

  The seam does not shrink (it moves one tile towards the surface), so the saving is the removed tiles at about 1.1 µs each, and on the pool the changed flow raises more tiles later than the closure cost. Quality, 17 runs before and 17 after per scene (the unperturbed run and 16 with gravity scaled by 1 + n × 10⁻⁶), 4 s, 102 runs: no fatal, no GPU error, pressure converged on every frame, cycle counts unchanged.

| in-range: unperturbed run; all 17 (a seventeenth reference run reads 0.89) | volume drift | represented drift | max speed | kinetic energy | liquid vertices |
|---|---|---|---|---|---|
| Figure 9 | 0.85; 0.72 | 0.90; 0.75 | 0.80; 0.79 | 0.50; 0.71 | 0.85; 0.78 |
| pool 128³ | 0.15; 0.37 | 0.55; 0.61 | 0.55; 0.58 | 0.30; 0.27 | 0.55; 0.46 |
| dam64 | 0.80; 0.69 | 0.75; 0.59 | 0.90; 0.83 | 0.90; 0.81 | 0.45; 0.69 |

| accepted residual over 4,080 frames, before → after | root p50 | root p99 | root mean | band p50 | band p99 | band max |
|---|---|---|---|---|---|---|
| Figure 9 | 0.060 → 0.072 | 1.35 → 1.58 | 0.161 → 0.181 | 0.224 → 0.256 | 3.00 → 3.71 | 17.2 → 31.9 |
| pool 128³ | 0.0178 → 0.0204 | 0.572 → 0.418 | 0.0492 → 0.0474 | 0.0284 → 0.0345 | 1.70 → 1.20 | 4.44 → 4.15 |
| dam64 | 0.0698 → 0.0677 | 1.94 → 2.29 | 0.190 → 0.204 | 0.0791 → 0.0735 | 2.56 → 3.36 | 37.6 → 40.5 |

| liquid indicator at 4h vertices | Hamming distance to the unperturbed reference, family median / max, before; after, at 1 s | at 3 s | per-run mean distance to Full over 4 s, min / median / max, before; after | the same over the first second |
|---|---|---|---|---|
| Figure 9 | 151 / 184; 193 / 217 | 860 / 958; 900 / 989 | 735 / 767 / 789; 757 / 793 / 814 | 115 / 123 / 133; 139 / 150 / 162 |
| pool 128³ | 67 / 119; 176 / 204 | 226 / 287; 366 / 487 | 568 / 581 / 594; 589 / 610 / 615 | 384 / 399 / 406; 408 / 414 / 420 |
| dam64 | 219 / 256; 252 / 282 | 107 / 130; 112 / 137 | 153 / 170 / 186; 141 / 158 / 185 | 185 / 199 / 205; 170 / 182 / 192 |

  Figure 9 is not matched: in the first second every one of the 17 runs is further from Full than every reference run (150 against 123 wrong vertices of about 4,500), the root's median residual is above the reference family's largest in 13 of 17 runs, and the distance to the reference at 1 s is outside the family. The pool is not matched and is worse: kinetic energy at 1 s is 4.0 to 4.7 × 10⁵ in all 17 runs against 2.9 to 3.3 × 10⁵ in the reference (Full: Dynamic already carries 1.30 times Full's energy there, B 1.81 times), volume drift leaves the family, and 16 of 17 runs are further from Full over the whole run. dam64 is inside the reference's spread on most series and no further from Full (158 against 170), with a heavier root tail. What is lost one tile below the surface, measured directly: the tile +face velocity (the 4h face value, or the mean of an h tile's 16 faces on that side) against Full's, relative RMS over the wet tiles that are h in the reference and 4h under B, reference then B: Figure 9 at 0.25 s 0.03 → 0.65 (61 tiles) and at 0.5 s 0.41 → 0.58 (103); pool 0.22 → 0.42 (104) and 0.23 → 0.70 (154); dam64 0.11 → 0.20 (349) and 0.36 → 0.30 (195). From 0.75 s the runs have separated and the measure no longer discriminates (Figure 9 at 1 s 0.30 against 0.30). So the closure tiles carry resolved velocity that 4h does not reproduce, the pool's impact shows it in the energy, and B buys 2% of the mean frame on Figure 9 and nothing on the pool. It is not proposed.
- **The lab's layout metrics, fixed.** `ke`, `umax` and `vol` now integrate over owners (an h tile's 64 cells; a 4h tile's cell value with its three +face anchors at local 3 on each axis). On the pool's first frames Dynamic (all 4h) and true zero agree to seven digits and Full is within 4% on energy and equal on volume; the series above use them.
- **Reach.** Travel per step in cells, from the tile-mean face velocity of each wet h tile (cell maxima are higher, and the dry tiles of the band are not counted, so these fractions are optimistic). A one-tile ring holds 3 cells of travel (4 less the tap cell), a two-tile ring 7. Lowest value over nine samples of each run, with the time it occurs:

| h tiles whose travel stays inside | one tile, dt 1/60 | one tile, dt 1/30 | two tiles, dt 1/60 | two tiles, dt 1/30 |
|---|---|---|---|---|
| Figure 9 Dynamic | 0.92 (1.0 s; p99 4.4 cells, max 5.9) | 0.59 (1.0 s), 0.74 (2.5 s) | 1.00 | 0.95 |
| pool 128³ Dynamic | 0.97 (1.5 s) | 0.71 (0.5 s) | 1.00 | 0.98 |
| dam64 Dynamic | 0.94 (1.0 s) | 0.76 (1.0 s) | 1.00 | 0.97 |

  One tile is enough for every stencil stage (extension's sweeps, forces, redistance, sharpening, surface volume, extraction: one cell, or one patch a sweep). It is not enough for the traces: at 1/60 the fastest 3 to 8% of h tiles leave it at the impact, at 1/30 up to 41%, so any trace kernel that reads the ring unconditionally needs a per-tile travel certificate in cells and the present path for the rest. The existing certificate is not that one: it asks for ceil(travel / 4) + 2 tiles.
- **What it needs from storage.** Ring tiles as a fraction of h tiles, run-weighted on today's tree, by 26-neighbourhood then by face: Figure 9 0.64 and 0.45 (at the largest frame 6,250 h + 4,426 ring = 65% of the lattice's tiles); pool 1.22 and 0.75 (2,096 + 3,047 = 16%); dam64 0.62 and 0.44 (3,007 + 730 = 91%). The domain placement already has the texels, and the detail ring (an h tile within three tiles) is where non-canonical stores are legal, so the phi and velocity rings need nothing new. A stage converted later needs its own state in the ring: the extension's unit patches are slots by fine-tile rank (240 an h tile), so its ring needs slot capacity for h + ring, 1.6 to 2.2 times today's. What the ring needs that the storage direction could remove: the h texels of 4h tiles within one tile of an h tile must stay allocated and writable by a fill (hanging vertices and unit taps are non-canonical stores in 4h tiles), and the fills read the base, not the h texture's canonical texel, so stopping the publication of 4h values into the h textures removes nothing as long as the base stays loadable with a one-tile halo. A windowed or h-tile-only h store would have to allocate the ring: +62% to +122% patches.
- **What it needs from frame orchestration.** Nothing for the trace stages: the 4h cache and `unitVelocityTaps` run once after the head extension and stay valid until forces writes velocity; phi resolve follows each phi writer. A converted extension adds one fill before its sweeps and uses the `restrictBand` it already has, in every run of the stage (head, tail, census). An extraction that reads resolved ring texels needs the publish to follow the frame's last phi resolve, which it does. The 4h side is as the coarse-side part above states it; nothing here contradicts it.
- **The shape criterion at a real h tile whose neighbours are ghosts.** Nothing new: a ghost tile is a 4h tile, so the vertices on the shared face are hanging, hold the 4h interpolant that phi resolve wrote, and are the ones the criterion already skips. The tile is judged on its remaining vertices against the interpolant of its eight corners plus their held corrections. Under B the tiles that seed the band lose their h neighbours, so a seed can have all six faces skipped and is judged on its 27 interior vertices alone, and the surface on each skipped face is the 4h interpolant by construction: the criterion cannot see detail where it leaves the tile, which is where B's loss above sits.
- **Lanes and helpers that pin the current seam numerics** (from their assertions; none was run, none was edited). `uniform-sharpening-seam-dawn` with `tests/helpers/uniform-sharpening-reference.ts`: sweep outputs bitwise equal to the reference expressions for `shSeamTile`, `shSweepJobs` and `shSeamLane`, including a single active seam tile. `uniform-transport-pruning-dawn` with `uniform-transport-workgroup-reference.ts`: transport bitwise equal to the original operators on mixed layouts in each departure mode (the donor weights across widths). `uniform-cubic-surface-dawn`: the surface stages bitwise against the reference tap checks on a layout with both widths, after phi resolve (the resolved sampler and the cubic at a seam). `uniform-force-cache-dawn`: cached against inline forces to 2 × 10⁻⁶ on steps that exercise h/4h face stencils. `uniform-extension-symmetry-dawn`: reflection error below 10⁻⁶ and the pass count 13 + 3 × sweeps (a fill launch changes it). `uniform-nearest-extension-dawn`: nearest-source values to 10⁻⁶. `uniform-dynamic-band-mask-dawn`: the band equals the closed union of crossing owners grown by the drift margin, "touching only a corner still needs support" (variant B contradicts it directly). `uniform-surface-extraction-dawn`: windows against the reference polygoniser, and the published base as one texel a tile corner. `uniform-pressure-phase-dawn`: the band keeps the h phase. Trajectory bounds a seam change must stay inside: `uniform-dynamic-coarsening-dawn` (volume drift below 10⁻⁴ a step on the 128³ run), `uniform-coarse-surface-dawn` (three encoded cycles, mass loss below 0.5%, owner count and front position), `uniform-detail-policy-dawn` (mass, the over-capacity fatal's text).
- **A landing order in which every tree is correct**, if the owner accepts non-bitwise seam changes at all (each step below is inexact). (1) Extension: unit patches for the ring, one fill before the sweeps, sweeps inside each width, `restrictBand` after; replaces `sweepSeams`' search and queue; bounded by −0.34 mean / −0.82 worst measured without the fill; not matched on the pool or dam64 in its no-exchange form, so it needs the family again. (2) Forces: the regular entry over "h or ring" for viscosity. (3) Extraction: mixed windows load resolved ring texels. (4) Traces: a travel certificate in cells and unconditional ring reads for certified tiles, with the present kernels for the rest. Each stands alone on the domain placement; none needs another. Sharpening, surface volume, transport, redistance and pressure keep their seam paths. Measured so far, steps 1 and 4's momentum half together are worth about 0.4 ms of the mean frame.
- **In the owner's terms**, Figure 9 Dynamic on the 13:40 tree, mean / worst window: 26.2 / 38.2 (two pairs: 26.18 and 26.24, 38.28 and 38.07), with 5,981 h tiles and 4,469 seam faces at the worst window against 6,345 and 4,441 at 12:38. Boundary by the third round's fit stays 5.6 / 10.6; what a ring converts of it is at most about 2 / 4, of which 0.4 / 1.4 is measured at label level. The boundary is not expensive because ghosts are missing. It is expensive because most of Dynamic's h tiles sit within a tile of 4h, so their traces run the general kernels at 1.25 to 1.5 times the regular rate, the seam faces are real h unknowns owned by 4h tiles, and three stages conserve across the seam face by face.
- **Not done.** No ghost prototype of the extension beyond the third round's candidate, of forces, of extraction or of the surface traces; their rows above are estimates from label splits and from the 5 October kernel ratios. M1's quality. The reach fractions use tile means. The frozen-fork label splits are from the 12:38 tree, whose h tile counts are about 6% above today's. Scratch artefacts: `surf/out/022` to `041`, `surf/p_b.py`, `surf/p_m1.py`, `surf/e1.diff`; both scratch roots deleted.

**Coarse side: the root runs its own traversal, 6 October 2026 (sixth round, landed).** The pressure root no longer walks the native continuation's per-cycle plans. `uniform-mixed-pressure-cycles.ts` holds one traversal form for every slot, open or spare: a V slot is 18 launches (59 wide, 35 spare before) and a Full slot 50 (165 and 105). The frame is 94 to 146 launches shorter in every policy and 0.6 to 0.85 ms cheaper on the host; simulation GPU is within 0.2 ms at pool and 0.3 to 0.4 ms lower on Figure 9 zero and Full and on dam64. Not bitwise: the smoother changed.

- **The form.** A visit is nine damped Jacobi iterations at weights 1 ×7 then 0.5 ×2 (twelve at 2/3 before). A level of more than 1,000 cells is tiled: three iterations to a launch, each tile's workgroup loading the tile and a three-cell halo into workgroup memory and relaxing a region that shrinks a cell an iteration, so the stored values are those of one launch per iteration; descent and ascent are a launch each. The levels of 1,000 cells and fewer share one launch of one workgroup that carries their visits, descents, the coarsest solve (eight (1, 0.5) pairs in place of the convergence-tested solve) and ascents; a visit there is a lane to a row, which reads the row's coefficients, RHS and bound once and passes its iterates through workgroup memory. The last launch of a traversal adds its iterate to the mixed pressure. Bind groups are set once per pass.
- **Whole frame, app settings, dt 1/60, extraction included** (ms, mean of two interleaved pairs, before → after on one tree and one device; worst = the worst 30-frame window).

| | launches | mean | worst | host | sim GPU |
|---|---|---|---|---|---|
| Figure 9 zero | 342 → 228 | 9.32 → 8.28 | 9.9 → 8.9 | 2.28 → 1.64 | 5.78 → 5.38 |
| Figure 9 Dynamic | 521 → 407 | 23.96 → 23.33 | 34.3 → 33.7 | 3.27 → 2.63 | 18.54 → 18.53 |
| Figure 9 Full | 479 → 365 | 25.02 → 23.95 | 29.5 → 29.0 | 2.92 → 2.21 | 20.49 → 20.16 |
| pool zero | 377 → 231 | 10.04 → 9.03 | 10.4 → 9.3 | 2.57 → 1.72 | 6.17 → 6.00 |
| pool Dynamic | 556 → 410 | 20.66 → 19.75 | 24.8 → 23.4 | 3.50 → 2.68 | 14.98 → 14.89 |
| pool Full | 514 → 368 | 47.82 → 47.08 | 48.6 → 48.0 | 3.19 → 2.34 | 42.49 → 42.60 |
| dam64 Dynamic | 456 → 362 | 14.81 → 13.92 | 17.1 → 16.3 | 2.99 → 2.42 | 10.85 → 10.53 |

  Figure 9 Dynamic's simulation GPU is flat because the new trajectory carries 5% more h tiles (3,453 against 3,288 run-mean); its pressure labels fall 1.47 → 1.14 ms.
- **What the GPU charges, measured on the way.** A one-workgroup launch has one core's throughput: about 2 µs a barrier pass plus the rows it relaxes, so everything below the root lattice in one workgroup cost +0.85 ms and was withdrawn for the tiled form. A row relaxed against storage costs four to five times a row relaxed from lane registers and workgroup memory (pool, the 1,000-cell level: 21 µs a pass against about 4.5), which is what took pool's one-workgroup launch from 449 to 216 µs and removed the 0.1 to 0.25 ms pool had gained. A closed launch costs its thread starts: 11 to 15 µs for a root-lattice launch of 26 to 47 thousand threads, so the spare V and spare Full slots still cost about 0.5 ms of GPU a frame at these sizes. Split timings (a pass per launch) overstate a many-launch form by about 6 µs a launch and understate halo work; the per-label frame sums are the arbiter.
- **Quality, Dynamic, unperturbed and four gravity perturbations, before family → after family.** No fatal in 20 runs. Figure 9: drift −4.3 [−5.4, −3.5] → −4.7 [−5.0, −4.1] ×10⁻³; accepted root residual median 0.048 → 0.052, p90 0.45 → 0.39, max 2.4 → 2.0; max speed 56 [44, 65] → 45 [39, 48]; h tiles 3,047 [3,029, 3,172] → 3,201 [2,995, 3,240]; owner KE inside the family at 1, 2, 3 and 4 s; liquid owner counts 0.7 to 1.2% above the before median with overlapping ranges. Pool: drift −0.72 → −0.69 ×10⁻³; accepted p90 0.119 → 0.110; band residual p90 0.20 [0.18, 0.30] → 0.25 [0.21, 0.36] and max 2.5 → 3.7 against a tolerance of 5; max speed 47 [35, 66] → 50 [44, 54]; h tiles 1,438 → 1,438; KE at 4 s 0.965 [0.949, 1.00] → 0.933 [0.906, 1.03]. Every frame of the seven whole-frame runs closed in one cycle. At a tolerance the V slot cannot meet (0.1 on Figure 9, 0.01 at pool, zero detail, first form) the second-slot and Full-slot counts equal the reference's (one/two/three-plus cycles over 150 frames: 125/25/0 against 122/28/0; 54/94/2 against 58/89/3).
- **Deleted with it.** The continuation's v/full plans and their spare forms, `mgSmoothTilesJacobiCone`, the liquid list of levels below the continuation's entry, and fifteen CM11a pipelines the mixed frame compiled and no longer dispatches (20 → 5). `prepareMixedContinuation` returns the setup (`encodeSetup`) and the level fields.
- **Tried and not kept.** Two Jacobi iterations to a root-lattice launch and a reach-pruned (octahedral) halo: predicted −0.2 ms at pool from split timings, nothing on the frame, +16 launches. 512 smoothing lanes and a 512-lane parallel descent: slower. Tiling the 1,000-cell level: equal GPU, +8 launches a slot. 16 and 64 coarsest pairs: no fewer cycles, 0.14 to 0.3 ms dearer.
- **Open.** The band solve's schedule and the frame-wide encode (closed stage groups, bind group reuse, the three submits) were not touched; the submits are segmented on purpose so the GPU starts while the host encodes. The schedule gate still writes the native slot-gate word and the coarse-accuracy words, which nothing dispatched reads. Lanes after the landing: `uniform-pressure-phase` and `uniform-volume` pass; `uniform-pressure-local-visit` fails at "fine: must execute coarse visits (reference)" because it counts dispatches of `mgSmoothVisitLocalInPlace`, which the mixed frame no longer compiles, and needs a decision (untouched).

**Coarse side: spare slots, band launch forms, encode, 6 October 2026 (seventh round).** Three exact landings (hashes identical in every arm) and four forms measured and not kept. The frame at Dynamic sizes did not move; pool Full fell 3.4 ms.

- **What a launch costs, corrected.** The sixth round's "a closed launch costs its thread starts" was wrong. Thread starts that read a gate and return are nearly free (a 262-thousand-thread launch costs about 4 µs more than a two-workgroup one). A closed or idle launch inside a pass costs about 5 µs on the GPU whatever its width, and about 2 µs to encode plus 2 µs of `queue.submit` on the host. One workgroup has one core's throughput (about 21 ns a row relaxed against storage, 4.5 ns from lane registers, 2 µs a barrier pass at 1,024 lanes), so work moved into a single workgroup is serial. A narrow launch that has to work is slow: an open V slot at pool costs +0.16 ms at 16,384 threads, +1.5 at 4,096, +6.5 at 1,024.
- **Landed: spare root slots.** The root's tiled launches stride their jobs; listed launches are sized by `uniformBufferedWork` of the cycle-list count (a new receipt word), and a slot beyond the cycles the lagged plan expects launches at most 16,384 threads. Closed Full slot 0.38 → 0.31 ms at pool, 0.37 → 0.34 on Figure 9; whole frame inside noise. The 68 closed root launches of a three-slot frame are about 0.36 ms of GPU and 0.28 ms of host, and only fewer launches would remove them.
- **Landed: band launch forms.** A band of more than 8,192 slots runs its 4h level as a launch per half sweep across workgroups (`coarseSweep`) instead of one streaming workgroup; smaller bands keep the one-workgroup solve, with 64- and 256-lane forms for a band that fits them; the band list launch is sized by the ownership's h-tile evidence.

| ms, before → after | launches | mean | wall median | host | sim GPU | band solve |
|---|---|---|---|---|---|---|
| pool Full | 368 → 492 | 44.07 → 40.68 | 44.00 → 40.16 | 2.39 → 2.64 | 41.68 → 38.05 | 13.53 → 9.69 |
| Figure 9 Full | 365 → 365 | 22.00 → 22.11 | 21.37 → 21.62 | 2.31 → 2.30 | 19.68 → 19.82 | 3.60 → 3.60 |
| pool Dynamic | 410 → 410 | 17.28 → 17.58 | 16.85 → 16.98 | 3.25 → 3.26 | 14.03 → 14.32 | 1.39 → 1.38 |
| pool Requested, no tile | 231 | 7.31 → 7.48 | 6.92 → 6.90 | 1.79 → 1.78 | 5.51 → 5.70 | none |
| pool Requested, 1 tile | 308 | 9.27 → 9.11 | 8.75 → 8.69 | 2.14 → 2.08 | 7.13 → 7.03 | 0.58 → 0.54 |
| pool Requested, 8 tiles | 308 | 9.28 → 9.30 | 8.85 → 8.83 | 2.09 → 2.14 | 7.19 → 7.15 | 0.59 → 0.57 |

  The 8,192 threshold is measured: forcing the launch form costs +0.19 ms of band solve on Figure 9 Full (4,400 to 5,600 tiles) and +0.83 ms at pool Dynamic. Figure 9 Dynamic and the zero-detail frames were not rerun for this landing (their band path is the one Figure 9 Full takes, or absent).
- **Landed: dead stores.** The schedule gate no longer writes the native slot-gate word or the coarse-accuracy words; its coarse-accuracy state (control words 2, 9, 25), `setCoarseAccuracy`, `MG_SLOT_GATED` and the continuation's `tolerance` and `diagnostics` members are gone, and the pressure trace drops its native block. Timing flat (pool zero median 7.66 → 7.62, Figure 9 zero 7.03 → 6.98, dam64 Dynamic mean 12.82 → 12.67, 362 launches).
- **Not kept: h tiles relaxed in workgroup memory, several sweeps a launch.** At a matched band residual it costs what the live schedule costs (four launches a visit: pool Dynamic 1.33 against 1.39 ms, Figure 9 1.94 against 1.89; residual median/p90 0.023/0.218 against 0.041/0.245). Every cheaper form loses residual (two launches a visit: 1.03 ms at 0.138/0.70 at pool, and 0.41/2.82 against 0.204/1.16 on Figure 9), and one- and eight-tile bands got slower. Relaxing a tile against frozen neighbours converges worse per unit of work than the global red-black sweep. Sixteen 4h sweeps in place of eight also lost residual (0.041 → 0.139).
- **Not kept: the first level below the root lattice in the one-workgroup launch.** V 18 → 10 launches, Full 50 → 23, frame 231 → 188 at zero detail, but the open V pays the level serially: V label 1.34 → 2.02 ms at pool and 1.05 → 1.51 on Figure 9, against 0.07 and 0.17 off the closed Full; frame mean 7.52 → 8.10 and 7.33 → 7.42. This is the sixth round's withdrawn form again. In the spare slots only it would save about 0.07 to 0.17 ms of GPU and 0.11 of host a frame and cost about 2 ms on the frame a spare Full opens.
- **Encode.** The three submits stay: merged, host falls 0.16 ms and the awaited frame's wall median rises 1.9 ms at pool Dynamic (0.07 and 1.1 at Figure 9 zero), because the GPU starts later. Per frame at pool Dynamic the host is 2.85 ms: `queue.submit` 0.87, the head before the first pass 0.31, the rest encode at about 2 µs a launch. Bind group and view creation total 0.04 ms (1.3 and 10 calls), `writeBuffer` 0.13 ms (25 calls, 13 of them the frame parameters, which a skip-if-unchanged cache could save at the price of a stale write behind the kick's direct authority write). Nothing beyond what `empty` and `idle` already skip is provably empty from lagged receipts. The host is launch-bound, and it overlaps the previous frame's GPU work in the app, so it sets the frame only where it exceeds the GPU.
- **Open.** The first h tile still costs +1.7 ms of GPU at pool, 0.70 of it the band: a few-tile static band takes the 35-launch fused path at 0.54 ms, and one small workgroup running the whole solve (about −0.4 ms) remains a rewrite of the band's eight modules. Under Dynamic the 1,024-slot reserve keeps every band on the 95-launch schedule whatever its size. The root over-solves at these scenes (one cycle in every frame, accepted residual p90 0.04 to 0.56 against a target of 5); a six-iteration visit would take two launches where nine take three, and was not tried because the dam 128³ impact frames are the documented fatals.

## Stage 1: 4h values stay in tile-resolution storage, 5 October 2026

Proposal 1, stage 1, is in the tree. It does not shrink the h store and it does not meet the whole target: the coarse-kernel rise at the first tile falls from 2.9 ms to about 1.1 ms at 256³, not to zero. What is left is named below, with the change that would remove it.

### Kernels by regime, before the change

The entry-point inventory (187 entries) sorts by what a launch's jobs can touch:

| regime | entries | read a switched texture before stage 1 |
|---|---|---|
| pure 4h (4h owner list or T tiles, canonical texels only) | 13 | all 13, through `udLoad_` on the n³ texture once C > 0 |
| h interior (h tiles, in-tile stencil) | 16 | all 16, the h store, as intended |
| boundary (seam tiles on either side, transfers, remap, extension seam) | 33 | all 33 |
| mixed lattice (one launch over T or over all owners, 4h and h jobs in one kernel) | 26 | all 26; the 4h branch is separable in 22 |
| none (buffers and per-tile pinned textures only) | 99 | none |

Before stage 1 every one of the 88 texture-reading entries took its 4h values from the n³ texture when C > 0: a 4h tile's one value per field sat at the tile-origin texel, 64 texels apart, so a T-wide launch touched one texel in 64 of a 268 MB texture (256³) where at C = 0 it read a dense 4 MB block.

Six launches that are issued only over 4h jobs are still not pure 4h, because their samplers leave the owner's tile with no reach certificate: the merged surface `advect` and `redistance` (departure point anywhere; Newton search up to four tiles), `advectDeferred`, `traceCellsMerged`, `momentumStep` and the 4h sampling cache. They are "sampler at a boundary" kernels and keep a two-source accessor.

The mixed-lattice entries whose 4h branch is not separable are `traceCellsMerged` (raw loads, uncertified reach) and the simulation-side pressure authority `build` and `phase` at seams (a 4h owner reads its neighbour's origin `centerPhi`, which is an h texel beside an h tile).

### What changed

- **The base block is the home of 4h values at every C.** It is no longer adopted into the n³ texture at admission; each switching field keeps its tile-resolution block for the life of the solver. Device bytes at C > 0 rise by the twelve blocks: +8.9 MB at 128³ (252.4 against 243.5 MB), +71 MB at 256³ (2,006.9 against 1,935.9 MB), 3.7% in both.
- **Stores write through.** A store to a canonical texel writes the bound texture and its twin (`udStore_`), so the base is current for every tile (invariant I1) and the h texture is still the complete logical field (invariant I2). I2 is what lets the boundary kernels stay single-source; it is also what stage 1 still pays for.
- **Pure 4h loads read the base.** A load marked `/*4h*/` (`UNIFORM_DETAIL_4H_LOAD`) compiles to `udLoad4_<field>`, a plain load of the base twin, the same code at C = 0 and C > 0. About fifty load sites in twelve modules carry the mark. Four launches whose every load is canonical bind the base group instead (extension list tier, forces 4h tier, authority, the surface merged tier).
- **Samplers branch on the owner cell's width, once, outside their tap loops.** `umVertexCell` and `umCubicPhi` read a 4h cell's corners from the base (`umLoadCorner`) and an h cell's from the h store.

Two shapes were measured and rejected:

- *A universal two-source accessor* (every load tests its address and picks a texture): the general-addressing result of the previous section, 0.9 ms at 128³.
- *The width branch per tap*, inside the eight-tap and 64-tap loops: surface redistance +0.22 ms in every arm on Metal, zero detail included. Hoisting the branch outside the loops (two copies of each loop) removes it: redistance +0.03 and advect +0.00 at 64³. A further variant with runtime loop bounds on the rare branch moved advect and redistance by 0.03 ms at most at 256³, inside the order noise, and was not kept.

### The curve after stage 1

Same-session A/B, the tree with stage 1 against the same tree without it, one device, arms interleaved. Median frame over the active window, ms. The "before" column is lower than the table above (5.86 ms at zero) because WS3's band schedule change landed in between.

| pool 128³ | h tiles | before | after | difference |
|---|---|---|---|---|
| zero | 0 | 5.51 | 5.55 | +0.04 |
| zero, order reversed | 0 | 5.51 | 5.55 | +0.04 |
| s = 1, surface | 1 | 7.30 | 7.21 | −0.09 |
| s = 1, order reversed | 1 | 7.27 | 7.19 | −0.09 |
| s = 4 | 64 | 8.13 | 8.10 | −0.04 |
| s = 8 | 512 | 8.88 | 8.87 | −0.02 |
| s = 16 | 4,096 | 13.03 | 12.99 | −0.04 |
| Dynamic | 2,196 / 2,133 | 19.82 | 19.89 | +0.06 (paired frames: median +0.09, mean −0.06) |
| Full | 32,768 | 41.84 | 42.34 | +0.50 (order reversed: 42.13 and 42.53, +0.41) |

| 256³ | h tiles | before | after | difference |
|---|---|---|---|---|
| zero | 0 | 11.83 | 11.77 | −0.07 |
| first tile | 1 | 16.45 | 14.61 | −1.84 |
| s = 16 | 4,096 | 19.56 | 18.26 | −1.30 |
| Dynamic | 16,383 / 16,125 | 47.14 | 48.59 | paired frames: median +0.18, mean +0.15 (0.4%) |

The Dynamic medians are taken over a window in which h grows from 10,000 to 17,000 tiles and the two trajectories are not the same run, so the paired frame difference is the figure to read; the label sums agree with it (51.14 against 50.98 ms).

**The first-tile jump is 2.84 ms at 256³ instead of 4.62, and 1.65 ms at 128³ instead of 1.78.** At 128³ almost none of the jump was ever the coarse kernels: it is the band solve (+0.61 ms), the extension seam tier (+0.43), the fine transport launches (+0.25) and the surface-volume seam work (+0.11), all launch latency that does not depend on the domain.

256³, split arms, rise of each entry point at the first tile, ms:

| entry | before | after |
|---|---|---|
| surface `redistance` (merged) | 0.247 | −0.002 |
| frame plan `seed` | 0.176 | 0.018 |
| authority `build` | 0.176 | 0.025 |
| `liveSeedCoarse` | 0.141 | 0.041 |
| `forcesRegularCoarse` | 0.139 | 0.026 |
| `retirementEvidenceCoarse` | 0.117 | 0.001 |
| `momentumStep` | 0.112 | 0.030 |
| authority `resolve` | 0.109 | 0.067 |
| authority phase | 0.103 | 0.032 |
| sharpening `cacheGeometryPrepare` | 0.089 | 0.001 |
| sharpening `commit` | 0.089 | 0.042 |
| sharpening `classify` | 0.083 | 0.001 |
| surface volume `seed` | 0.080 | 0.000 |
| surface `advect` (merged) | 0.078 | 0.038 |
| cleanup `clean` | 0.061 | 0.016 |
| 4h sampling cache | 0.056 | 0.007 |
| `localSpeed` | 0.051 | 0.007 |
| **still paying: mirrored stores** | | |
| `toPressureCoarse` | 0.334 | 0.241 |
| extension `publishList` | 0.256 | 0.186 |
| `toSimulationCoarse` | 0.265 | 0.171 |
| surface geometry `geometry`, sharpening `gather` | | 0.077, 0.078 |
| **not stage 1** | | |
| band solve | 0.64 | 0.64 |
| extension `sweepSeam`, `sweepCoarse` | 0.236, 0.117 | 0.236, 0.117 |
| fine launches that stop being empty (`redistanceFine`, `rowsDivideFine`, `momentumDeferred`, `advectFine`) | 0.22 | 0.22 |
| **sum of all label rises** | **4.80** | **2.94** |

**The zero-detail arm pays for the two-copy samplers: +0.04 ms at 128³ in both run orders (0.7%).** At 256³ the frame is not slower (−0.07 ms), but in the split arms the merged surface `advect` is 0.84 against 0.70 ms and `redistance` 0.69 against 0.60 at zero detail, offset by other kernels; the likely cause is the doubled tap loops in those two kernels (the bulk of `advect` never executes the cubic and still slowed), and it is not resolved.

**Full pays 0.4 to 0.5 ms at 128³, 1.0 to 1.2% of its 42 ms, in both run orders.** By label, over the window's frames: `pressure authority phase` +0.09 ms, `global surface volume` +0.09, transport +0.05 to +0.09, surface redistance +0.04 to +0.05, sharpening geometry +0.04, authority and volume correction −0.07. These are h kernels in which every store now tests whether its texel is canonical and, one time in 64, writes the base as well. That is the probable cause; it was not isolated, and Full at 256³ was not rerun. The base at an h tile's canonical texel is read by 4h jobs beside it, so the write cannot simply be dropped under Full without a rule for which base texels are live.

Checked storage (`domain:checked`, every load and store audited against its home) ran clean on the 64³ dam (zero, one tile, a moving region, Dynamic, Full), on the pool (zero, one tile, s = 4, Dynamic; 30 frames) and at 256³ (s = 16 and Dynamic; 40 frames), the last two sets with the launch change below in the tree.

Lanes on the tree with stage 1 and the launch change, no bound changed: `uniform-pond-rest` (1 pass), `uniform-coarse-solid-rest` (10 pass), `uniform-detail-policy` (3 pass, including the garden fatal arm), `uniform-mixed-live-solid-edit` (1 pass), `uniform-mixed-rigid-body` (3 pass), `uniform-surface-extraction` (2 pass). The first run of `uniform-coarse-solid-rest` failed four exact-rest assertions; three solver files were rewritten by another session thirteen seconds after that run started, and the rerun on an unchanged tree passed all ten.

### The three unexplained risers

None of the three reads a 3D texture, so stage 1 does not move them; each is a seam effect on a kernel that works from buffers.

- **Surface volume `dilate`** gains an h job, and a 4h owner beside the h tile walks sixteen face parts on that side instead of one.
- **`rowsFallbackCoarse` and `rowsDivideCoarse`**: a 4h row that touches an h tile takes grain 1, so it has 126 edges instead of 9; lane 0 sums them serially and four rows share a workgroup.
- **Sharpening `limit`**: seam jobs appear (`shLimitSeam`, two owners of 96 lanes each), and each ends in a serial 96-term reduction.

They are launch latency of the same kind as the extension seam tier, about 0.1 ms each at any size.

### What remains of stage 1

**The mirrored stores, about 0.75 ms of the 256³ jump.** `toPressureCoarse`, `toSimulationCoarse`, `publishList`, `geometry` and `gather` are T-wide or 4h-list launches whose loads now come from the base and whose every store still also lands on one texel in 64 of the n³ texture, to keep I2. Removing them is the *mirror drop*: a regular 4h tile is never written to the h store, and only 4h tiles within the boundary kernels' reach of an h tile are mirrored. Every reader of a 4h canonical texel must then read the base or be a boundary kernel inside that reach, which includes the unmarked loads the stage 2 audit found reading the h store domain-wide (the census `umClassifyOwner`, `wallReach`, `umReleasedWalls`, diagnostics) and the overlay's `udrLoadCell` and `udrLoadFace`, which are handed no cell or face base today. The safe way in is a checked mode that poisons unmirrored texels of the h store and reports the first load of one. It is also the first step of stage 2, so it is costed there.

**The first-tile frame.** The admission seed is a T-sized launch (base to the canonical texels of the n³ textures), not a whole-lattice fill; it stays while the mirror stays. The whole-lattice cost of the first tile is allocating and zero-clearing the twelve n³ textures (192 MB at 128³, 1.54 GB at 256³). Stage 1 does not touch that; only an h store that is not domain-sized does.

**The zero-arm cost** above, in the two merged surface kernels.

### Launches not issued at zero detail

Proposal 6, outside the band files. `UniformMixedOwnership.coarseOnly` is true while the held layout has no h tile, which the host knows when it encodes; the counted, tiered, fused and buffered dispatch helpers return without a launch for the h tier, the seam tiers and the hanging taps in that state, the regular-fine twin of the deferred advect is skipped, and the two fine transfers are sized by the h tile count instead of 4,096 groups. Each of those entries runs its body once per listed job, so with no job it wrote nothing.

| arm | launches before | after | workgroups before | after | frame before | after |
|---|---|---|---|---|---|---|
| pool 128³, zero | 374 | 333 | 136,384 | 127,130 | 5.53 ms | 5.36 |
| pool 128³, zero, order reversed | | | | | 5.55 | 5.33 |
| pool 128³, s = 1 | 416 | 416 | 141,497 | 133,309 | 7.30 | 7.32 (reversed 7.24 and 7.38) |
| dam 64³, zero | | | | | 3.68 | 3.51 |
| dam 64³, s = 1 | | | | | 5.63 | 5.59 |
| 256³, zero | | | | | 12.10 (quiet 10.80) | 12.01 (quiet 10.62) |
| 256³, s = 1 | | | | | 15.04 | 14.72 |

**The zero-detail base falls 0.17 to 0.22 ms at 64³ and 128³ and about 0.1 to 0.2 ms at 256³; one tile is unchanged within noise.** At zero detail, volume, velocity, phi and occupancy are bitwise equal with and without the change at frames 20, 40 and 60 of the pool, in both run orders. The h-tier `classify` launch of WS3's `uniform-pressure-surface-band.ts` goes through the same helper and is skipped with the rest.

Proposal 4 (extension seam faces as their own launch) still stands after stage 1 and is not started: `sweepSeam` and `sweepCoarse` cost 0.118 and 0.058 ms per launch for one seam tile at 256³, twice a frame. `seedSeam` is cheap, so the cost is not staging; it is `umExtended`, which makes six general `umNeighbor` evaluations in series for each face, and a seam tile already fills its 64 lanes three deep. The fix is one invocation per (face, neighbour) writing to a scratch record, and a combine launch; it needs a scratch buffer and a second launch per sweep, and is a piece of work on its own.

## Stage 2 design: an h store that is not domain-sized, 5 October 2026

A design, not code. The question was whether one wrapped window holds: the h store as a single texture set of power-of-two extent W per axis, at least the bounding box of the admitted h tiles, addressed by `cell & (W − 1)`.

**It holds for the case it was proposed for, a small region that moves, and its accessor costs nothing measurable. It does not hold as stated (three of its claims fail against the code), and it saves at most a factor of two under Dynamic, because Dynamic's h tiles are a thin sheet whose bounding box is the whole domain in two axes on every scene measured.** The corrections first, then the measurements, then what would hold where the window does not.

### What breaks, from the code

**1. The h store can no longer be the complete logical field, so the mirror drop comes first.** Today a store from any launch writes the bound h texture at its address for any owner, 4h included, and unmarked loads read 4h canonical texels from it domain-wide (the census `umClassifyOwner`, `wallReach`, `umReleasedWalls`, diagnostics, the overlay's `udrLoadCell` and `udrLoadFace`). Under `& (W − 1)` every one of those aliases onto a live h texel. The window therefore needs the same change that removes stage 1's remaining cost: 4h values outside the window live in the base only, their stores do not reach the h store, and their readers read the base.

**2. "No halo" fails: the window is the bounding box plus a ring, and one more vertex.** The h store is also the home of texels that belong to 4h tiles beside h tiles:
- faces: the sixteen unit texels on the +a plane of a 4h tile whose +a neighbour is h (written by `toSimulationCoarse`, extension `publishCoarse`, the remap's face copies and the band projection; read by `toPressureCoarse`, the momentum `cache` and `publishCoarse`);
- the velocity tap atlas: all 64 `unitVelocity` texels of each slotted seam 4h tile;
- vertices: the hanging vertices of seam 4h tiles in `phi` and `phiScratch`.

That is a ring of one tile. Boundary kernels also read *canonical* 4h values up to two tiles from an h tile (the resolve lattice, `umResolvedVertex`'s window from tile − 1 to tile + 2); with a two-tile mirrored ring they stay single-source, as today. A vertex field has one more texel per axis than cells. So W ≥ bounding box + 4 tiles + 1 texel per axis, not the bounding box.

**3. "Equals the domain when h is everywhere" fails for vertices, and the extent test goes.** n + 1 vertices do not fit W = n under a wrap (vertex n aliases vertex 0, and the extraction reads the closure plane p = n). The full-domain case has to be today's unwrapped texture with the mask set to all ones, as a special case of the same accessor. And the accessor can no longer tell a base from an h store by `textureDimensions(name).x < D.x` (a window with W < n reads as a base); `extent()`, `verify()` and `adoptBase` assume the same. The window's box and mask have to be published, and the place exists: the detail table in the topology tail that every mixed shader already binds, and `udrEntry` for the external consumers.

**4. "One AND per axis, no table" holds for h-interior kernels only.** A kernel whose every load is certified inside an h tile's neighbourhood takes the one AND. A kernel that can address a 4h texel outside the ring needs a box test and a second source, which is the general accessor again. The regime split of stage 1 is what makes this affordable: pure 4h kernels bind the base and need nothing; the uncertified samplers already choose their source by the owner cell's width, and an h cell is always inside the window; the boundary kernels are inside the ring by construction. What is left to convert is the list of unmarked domain-wide loads in item 1.

**5. Out-of-lattice loads.** A load outside the lattice is left to the device's clamp today (`uniform-detail-fields.ts`, the accessor comment). Under a wrap it aliases instead, so those sites need an explicit clamp before the mask.

**6. Stale texels: not every h texel is written before it is read after admission.** `uniformDetailSiteClean` does not certify it: it says a profiled load site stayed within its home patch and halo, and nothing about writes.
- Rewritten in full every frame for every h tile, so safe for a tile that enters the window: `velocity` (`toSimulationFine`), `velocityScratch` (`publishFine`, forces `output`), `departure` (`traceCells`, momentum `output`), `volumeScratch` (`liveSeed`), `volume` (cleanup), `centerPhi` and `targetFill` (surface geometry), `phase` and `curvature`, `correction`, `unitVelocity`, and the canonical vertices of `phi` and `phiScratch`.
- Not safe: the remap's refine (`remapTileCells`) writes all 64 `volume` and face texels of a new h tile but skips every `phi` vertex the old layout stored (the eight tile corners, and vertices shared with an old h neighbour), and it reads its inputs (corners, donor volume, old face anchors) from the h store through unmarked loads. In a zero-cleared domain texture with seeded canonical texels that is correct; at a wrapped address that last held another tile it is not. The refine has to read its 4h inputs from the base and write the corners.
- Relied on to persist between frames, and harmless under a wrap as long as a tile's address does not change: the hanging vertices after `resolveListed`, `geometryChanged`, the reused tail extension. A wrap keeps addresses for as long as W does.

**7. "No copy on a move" holds; growth does not come free.** A box that slides keeps every surviving texel's address. W → 2W changes addresses: a re-wrapping copy, at most eight box regions per field with `copyTextureToTexture`, between frames, plus a rebind of every group (as `follow` does when C crosses zero; no pipeline recompiles). The box has to be known where the copy is issued. Under the GPU relayout the host learns the layout by a lagged receipt, so growth works like capacity: the build admits h tiles only inside the published window, reports the box it wanted, and the host grows the window for a later frame.

**8. Whole-store operations.** The admission seed and `encodeBaseRefresh` address the h store at `4g` over T tiles: box-gated under a window. `copy()` and the frame's whole-texture copies stay valid (both sides share W). `install`, `image`, `write`, `read`, `capture` and `logicalDomain` treat the texture as the logical image and need a wrapped upload and an unwrap with a base broadcast; that reaches `readDetailField3D`, `readMixedTexture` and several probes. The t = 0 dense authority and publication launch over the lattice into the h textures.

### Measured: the accessor

One patch arm: every accessor load and store of a switching field masked per axis (`& (textureDimensions(t) | 0xffff)`, an identity the compiler cannot fold, and dearer than a mask read once from a uniform), against the same tree without it, Full on the 128³ pool, split arms, both run orders. Median ms per launch over the window:

| h kernel | plain | masked | difference |
|---|---|---|---|
| `momentumRegularStep` | 2.079 | 2.084 | +0.005 |
| `advectOwners` | 1.991 | 1.974 | −0.018 |
| `redistanceFine` | 1.322 | 1.296 | −0.026 |
| `forcesRegular` | 0.913 | 0.914 | +0.001 |
| `traceCells` | 0.844 | 0.839 | −0.005 |
| frame, sum of entry medians | 43.67 | 43.79 | +0.13 (the two orders: +0.36 and −0.14) |

**The mask is not measurable in the three heaviest h kernels** (`forcesRegular` makes up to 372 loads an invocation, `advectOwners` up to 177, `momentumRegularStep` about 60 per lane). The cost of a window is not the AND; it is the kernels that would need a second source.

### Measured: Dynamic's bounding box against the domain

Bounding box of the h tiles under Dynamic at declared defaults, in tiles, read from the tile words every five frames:

| scene | tiles | h tiles | bounding box | box / domain |
|---|---|---|---|---|
| dam 64³ | 16³ | 914 to 1,380 (22 to 34%) | x and z full; y 8 to 14 of 16 | 0.50 to 0.88 |
| pool 128³ | 32³ | 1,796 to 2,701 (5 to 8%) | x and z full; y 6 to 12 of 32 (frames 15 to 30) | 0.19 to 0.38 |
| fig-9 | 32 × 32 × 16 | 1,780 to 4,870 (11 to 30%) | z full; x 23 then 32 of 32; y 18 to 30 of 32 | 0.56 to 0.84 |
| 256³ | 64³ | 10,249 to 17,089 (4 to 7%) | x and z full; y 5 tiles at frame 35, 26 at frame 60, still rising | 0.08 to 0.41 |

With the ring and the power-of-two extent the window is the domain on the dam and on fig-9, half the domain on the pool while the box is 11 tiles of y or fewer (it reached 12), and half the domain at 256³ until the splash passes 27 tiles. **Under Dynamic a window saves a factor of two at best, on scenes where h is 4 to 8% of the tiles.**

A Requested region is the other case. One tile needs W = 32 cells (one tile, a two-tile ring each side, one vertex): 3 MB of h store in place of 192 MB at 128³ and 1.54 GB at 256³. s = 4 and s = 8 fit W = 64 (25 MB); s = 16 is the domain at 128³ and W = 128 (192 MB in place of 1.54 GB) at 256³. The first-tile frame allocates and clears those bytes instead of the domain's.

### Verdict, and what holds where the window does not

**The window is worth building for Requested regions and is not the answer for Dynamic.** It turns the first tile's 192 MB or 1.54 GB into 3 MB and removes the whole-lattice allocation from the first-tile frame, with an accessor that costs nothing in h kernels; a region that moves keeps its addresses. Its preconditions are the list above, and the order matters:

1. *Mirror drop, with a checked mode that poisons unmirrored h texels.* Needed by any h store that is not the domain, and it recovers the 0.75 ms of mirrored stores stage 1 left at 256³. The unmarked domain-wide loads and the overlay's cell and face reads are converted here.
2. *Publish the window (box and mask) in the detail table; replace the extent test; clamp before the mask.* With the box equal to the domain and the mask all ones this is today's storage, so it lands with no change in behaviour and every lane still applies.
3. *Refine reads the base and writes the corners; the seed and refresh launches are box-gated; uploads and readbacks wrap and unwrap.*
4. *Shrink W for Requested layouts* (the host knows the box when it packs the layout), then growth by re-wrapping copy, then the GPU-built layout with the box in the receipt.

**For Dynamic the store has to follow the tiles, not their box: a paged h store, bytes proportional to C.** The h store becomes pages of one tile (or a small block of tiles) in a texture sized by capacity, and a tile's page comes from the directory the detail table already publishes for the external consumers. The general accessor through that directory in every kernel is what the previous section measured and rejected (0.9 ms at 128³, 3.5 ms at 256³). What stage 1 changes is that it no longer has to be in every kernel: pure 4h kernels never touch it; the h kernels run one workgroup per tile and can stage their 27 neighbour pages once per group, as `redistanceFine` and the extension already stage their neighbourhoods, so a tap costs an index into workgroup memory and no dependent load; only the uncertified samplers pay a directory load per cell, as they pay a width lookup today. Its preconditions are steps 1 to 3 of the window, with a page index where the window has a mask, so the window is on the way to it and not a detour. At 256³ Dynamic holds 2.4 GB for 6% of the tiles in h; that is the case for going on to pages, and it was not costed further here.

### Step 1 as built: the mirror drop, 5 October 2026 (late)

**The ring mirror is built and certified, and it is not the default. The 0.75 ms was never the twin test: it is the write of one texel per tile into the n³ textures, and the consumers outside the solver still need those texels. Until they read the base, dropping the mirror moves the cost and does not remove it.**

What is in the tree:

- `UniformDetailDomain.mirror`: `"all"` (the default, every tile's canonical texels in the h store, as before) or `"ring"` (only the tiles with an h tile within three tiles; QA spec `domain:ring`). The ring is one bit per tile in the second stencil word, written by the host packer and by a `mirror` entry in the GPU layout builder.
- Under the ring a 4h store outside it lands in the base alone; a tile that enters the ring is seeded from the base; the admission seed and the base refresh cover the ring only.
- Every domain-wide reader of a canonical texel names its home with a mark: `/*4h*/` (base, as before), `/*c*/` (base at a canonical address, else h), `/*r4h*/` (base under the ring, the plain load under the every-tile mirror), `/*g*/` (a compare-read that only guards a store of the same texel). Converted: the census, diagnostics, momentum's original face value, the forces' cell phi, coarse mass and forced velocity, the pressure-velocity face and correction loads, the wall-reach and released-wall loads, the embedded-contact volume, the surface-volume gradient, the remap's face copy.
- `domain:checked:ring:poison` fills every canonical h texel outside the ring with −2^100 and raises a violation when a load returns it. `domain:checked:mirror:survey` records every load site and compares the two homes bit for bit at each canonical load.

Certified by poison with no flagged site: dam 64³ (zero, one tile, a moving region, Dynamic, Full; 45 frames), pool 128³ (zero, s = 1, s = 4, Dynamic; 30 frames), 256³ (s = 16, Dynamic; 40 frames). It found one reader the code review had missed: the phi resolve's store-if-different read of a hanging vertex that was never written at the admission frame. Scenes with embedded solids or bodies were not in the matrix.

What it measured (256³, s = 16, GPU pass ms, same process, interleaved):

| arm | ms | against the tree before |
|---|---:|---:|
| every-tile mirror (before, and the default now) | 18.15 | |
| ring, consumers not served | 17.54 | −0.62 |
| ring, consumers' canonical texels seeded each frame | 18.21 | +0.06 |

By label the ring returns 0.69 ms (transfer to pressure 0.22, transfer to simulation 0.15, extension 0.14, authority and volume correction 0.08, the rest small) and the seeding pass costs 0.72. The grid overlay, the level-set overlay, the secondary particles, the extraction and the harness readbacks all bind one texture per field and read a 4h value at its canonical texel there. The renderer asks for those textures every frame, so seeding on demand would be seeding always.

The ring has costs of its own that the every-tile mirror does not: Dynamic +0.35 ms at 128³ and about +0.5 at 256³, Full +0.33 at 128³. Part is the canonical-or-h load in kernels that walk every owner (census +0.09, forces +0.05 at 128³ Dynamic); the rest (ring-entry seeds every relayout, the ring bit read by every canonical store) is not split. A base binding also moves the momentum kernel at the last bit under Metal: a control that loads both homes and returns the h bits diverges exactly as the converted kernel does, so the values are equal and the code generation is not. For those reasons the every-tile arm compiles `/*c*/` and `/*r4h*/` to the plain load, and the default is bitwise the tree before the change (dam 64³ five arms, pool 128³ four arms) at the same frame time (Dynamic 18.81 against 18.81, Full 42.38 against 42.49, 256³ s = 16 18.23 against 18.18).

Step 1 is therefore finished as a certificate and not as a saving. What turns it into one, in order: the runtime ABI (`udr*`) and the extraction take a 4h tile's value from the cell and face bases, and the harness readbacks compose them from the base; the seeding pass is deleted; the census branches on the owner's width instead of testing the address; then the default moves to the ring. Steps 2 to 4 above are unchanged and need the same consumer conversion, since a window that is not the domain has no canonical texel for a far tile at all.

## Timestep: the start-up half kick, 5 October 2026 (landed)

A frame drifts by dt·U, adds g·dt and projects, so the stored velocity is the drift's, half a step ahead of the positions. A state seeded at t = 0 therefore lagged free fall by g·t·dt/2. The host now encodes `UniformMixedFrame.kick` once, ahead of the first clocked frame of a solver: U ← P(U + ½dt·g). It is a mode of `advance`, not a kernel family: a frame's head, forces with gravity alone (no viscosity, capillarity or volume correction), the whole pressure solve on the envelope plan to `min(frame target, 2·lead·UNIFORM_PRESSURE_KICK_RESIDUAL_PER_STEP)`, and no transport, surface or sharpening stage. Its receipt sits in the ordinary ring under the ordinary fail-fast verdict; it plans no frame; there is no toggle. Cost: one extra head and envelope solve at step 1.

The head is the frame's own. The first version solved on the layout the seed was deposited on, and under Dynamic that layout is all-h: a tank more than half wet then holds more wet h tiles than the band's liquid bound (`UniformPressureBand.capacityOf`, T/2), and the kick died with "pressure band over capacity" (pool 128³, fill 0.484 plus a ball: 17084 tiles against 16384). No frame ever solves on that layout, because its census runs first; the kick now does the same (the relayout head where one is due, built for the frame's step) and is solved on the generation it adopts. After the fix the kick's band is 1484 tiles at lane settings and 0 at app settings on that scene, and fig-7 256³ starts under both.

Fall before impact, lag against y0 − gt²/2 in h (analytic lag without the kick in brackets): pool 128³ Dynamic, lane 1/30 at 0.3 s 0.03 (1.0), app 1/60 at 0.2 s 0.04 (0.33); pool 128³ Full 1/30 0.001 (1.0); fig-7 256³ Dynamic at 0.2 s, lane 0.01 (1.33), app −0.002 (0.67). A resting pool is kicked to at most 5e-4 m/s and settles (3e-5 m/s by 0.3 s).

Two lane bounds moved with it. `uniform-coarse-solid-rest` W4: the first frame's volume correction was exactly 0 and is now bounded by the lane's residual (1e-4 1/s; at most 1.3e-5 in the scratch run of the first version), the converged kick's residue. `uniform-geometric-boundary`, ceiling arms: the accepted residual is quantized near 2.6e-7 and domain-ceiling's frame 2 stalls at four quanta (1.0435e-6) behind the kick whatever the cycle count (5, 6 and 8 run), so the two ceiling arms accept 1.5e-6; the side-wall arms keep 1e-6.

Not done: a step change at run time needs its own (signed) kick; injected liquid, the hose plug and the rigid integrator still carry their own half-step offsets.

## Packed deferred momentum, 6 October 2026 (landed)

`momentumDeferred` (`uniform-mixed-momentum.ts`) re-traces with the general sampler the cells whose characteristic left the unit taps. At the app settings it lists 958 of fig-9's 3149 h tiles (pool 128³ 480 of 1275, fig-7 256³ 2804 of 8475) with 18 to 21 cells each, and 60 to 86% of those cells have one escaped face, the positive face on a 4h boundary. One tile a round ran about 53 of 192 lanes. A round now takes as many whole entries as fit its 64 cells and each lane finds its own cell among them (the n-th listed bit); the launch follows the seam h count instead of the h count. Every hashed frame is bitwise equal (fig-9 and pool 128³, Dynamic and Full, unperturbed and two gravity perturbations), so the quality read is the old one. The pass, mean ms: fig-9 Dynamic 0.316 → 0.232, pool Dynamic 0.126 → 0.092, fig-7 256³ Dynamic 0.916 → 0.607, Full (empty list) 0.025 → 0.011. Whole frame (host + passes + extraction, 240 frames at 1/60, two runs each): fig-9 Dynamic 23.83 → 23.72, pool Dynamic 20.51 → 20.45, fig-9 Full 24.96 → 25.02, pool Full 47.61 → 47.58, all inside the ±0.1 ms spread between identical runs. The saving is real on the pass and below what a whole-frame read resolves at 128³.

Measured and not kept (fig-9 / pool, same pass): the 5 October serial unpack on lane 0, 0.260 / 0.099; the unit sampler first and the general one only where a face escaped, 0.312 / 0.116 (a round pays both paths); escaped faces alone, 192 a round, the cell's other components recomputed with the unit sampler by one lane, 0.485 / 0.207. A general trace costs about 6× a unit trace per face here and filling the lanes recovers under a third of it, so the pass is bound by the traces, not by idle lanes. What is left needs fewer general traces: the unit step's values kept for escaped cells (storage per listed tile), or fewer escapes. The whole pass is 1.3% of the fig-9 frame.

## Seam sharpening: reference replaced, seam jobs packed by class, 6 October 2026 (landed)

`tests/helpers/uniform-sharpening-reference.ts` used to regex-rewrite the production module back to the one-seam-tile-per-workgroup sweep, and `uniform-sharpening-seam-dawn` compared the two bitwise. Both sides ran the same proposal, limit and commit text, so the lane pinned the job packing and could not see a wrong flux in the shared text. The helper is now a module of its own: every owner every sweep, one invocation each, a lattice-keyed store, five launches a sweep and one committed flux value per patch, with no production sharpening text in it (only the ownership topology, the phi sampler and the budget rule). The comparison is no longer bitwise: V agrees within 16 ulp of a full cell (1.91e-6; 7.2e-7 measured). That is a deliberate weakening of one clause, paid for by what the lane asserts beside it: per-owner balance against the reference's accumulated patch fluxes and dust (2 tolerances), V in [0, open], finite outputs, untouched non-owner texels, skip parity, field conservation against production's own dust words (2.5e-4 cells), each of the 1 to 6 h-side seam shapes moving volume, the single seam tile, closed owners, and moved volume above ten times dust; three policies by six layouts, with and without solids. In a scratch copy a 0.1% error on one seam part and a misplaced one-patch side each fail it. `--sharpening-baseline` in `tools/probe-uniform-stage-scaling-dawn.ts` is retired with the sweep it restored.

The lane found one production difference from dense semantics on its first run: `umProposal` tested the zero-budget return after the orphan branch, so under the orphan policy a negative sliver's negative give was proposed against a quiet neighbour (8e-6 in V). The return now comes first. The mixed frame always writes policy (0, 0), so the app is unchanged.

Seam work is now sized by the seam list. A seam owner needs 16 lanes per h side and one per other side, and the old sweep gave every seam tile half a job (96 lanes) whatever its shape. Geometry now packs the active seam owners into four classes by h-side count (1, 2, 3, 4 or more: blocks of 24, 48, 64 and 96 lanes, so 8, 4, 3 and 2 owners a job), as `tile | sides << 24 | class << 30`, and a lane finds its owner by division, with no per-lane list walk or flag load. The head lane reduces in the old side and part order, so the result is bitwise the old kernel's (state hashes equal at steps 60 to 240 on fig-9 and pool 128³ Dynamic at app settings, unperturbed and under two gravity perturbations). The work list grows by `tiles` + 4 words and the receipt's seam word is a weight in 24ths of a job. Sharpening passes, paired per frame: fig-9 −0.05 to −0.09 ms of 1.95, pool 128³ −0.11 to −0.17 ms of 1.6; the whole-frame change is inside the run-to-run spread. Per-side neighbour records were considered and not built: they save about one load at the same dependency depth.

## Windowed h store, ring by default, publication deleted, counted admission, 6 October 2026 (landed)

What is in the tree (`uniform-detail-fields.ts`, `uniform-detail-abi.ts`, and two host edits in `uniform-mixed-frame.ts` and `webgpu-uniform-reference.ts`):

- The every-tile mirror is retired; the ring is the only form. Consumers (the runtime ABI `udr*`, the extraction, uploads and readbacks) take a 4h value from the base through the detail table, and the per-frame publication pass is deleted. A ring tile outside the window box raises `WINDOW_FATAL`.
- The h store is a wrapped power-of-two window: per axis the smallest power of two (at least 8 texels) that holds the h-tile box plus the three-tile ring, or the lattice where that is no smaller. A vertex plane above the box belongs to a tile outside it, so the store holds 4w texels an axis and one more only at the lattice top (`boxEnd`). It grows at once by reallocation and a re-wrapping copy at admission, and shrinks after 16 calm syncs when a quarter would do. Dynamic and rigid bodies keep a lattice-sized store.
- A request that arrives on an all-4h host layout is a counted edit: the frame keeps its launch budgets and adds the edit bound (`setRelayout(..., counted)`), where it used to forget every budget and launch at the ceilings for two frames. A capacity change that fits the admission stales with a zero edit.

Whole frame (sim wall + extraction wall, medians, ms) and h-store MB (bases included), after / before, one process per pair on one device, 1/60 step:

| pool 128³ (T = 24,576) | frame after | before | h MB after | before |
|---|---:|---:|---:|---:|
| zero | 8.04 | 7.99 | 6.9 | 6.9 |
| 1 tile | 9.74 | 9.85 | 9.9 | 151.2 |
| 8 | 10.01 | 10.02 | 9.9 | 151.2 |
| 64 | 10.98 | 11.06 | 30.9 | 151.2 |
| 512 | 12.13 | 12.15 | 30.9 | 151.2 |
| moving 1 | 10.03 | 9.91 | 9.9 | 151.2 |
| moving 8 | 10.25 | 10.29 | 12.9 | 151.2 |
| moving 64 | 11.03 | 11.12 | 30.9 | 151.2 |
| Dynamic (about 2,600 h) | 23.69 | 23.09 | 151.2 | 151.2 |
| Full | 32.53 | 32.14 | 151.2 | 151.2 |

| fig-7 256³ (T = 262,144) | frame after | before | h MB after | before |
|---|---:|---:|---:|---:|
| zero | 14.03 | 13.98 | 73.0 | 73.0 |
| 1 tile | 15.07 | 15.80 | 74.5 | 1610.6 |
| 8 | 14.89 | 15.62 | 76.0 | 1610.6 |
| 64 | 14.91 | 15.94 | 85.0 | 1610.6 |
| 512 | 15.70 | 16.44 | 121.0 | 1610.6 |
| moving 8 | 15.29 | 15.94 | 79.0 | 1610.6 |

All solver allocations: pool 48.4 MB at one tile (189.7 before), 83.4 at 512 (203.7); 256³ 472 to 533 MB (about 2,010 before). The first tile at 256³ costs +1.04 ms (was +1.82); at 128³ it is still +1.7, launch-bound. Fig-9 (the scene as changed on 6 October): zero 7.34 / 7.27, Dynamic 20.67 / 20.23, Full 23.61 / 23.13, memory unchanged.

**Dynamic and Full are 0.4 to 0.6 ms (about 2%) slower.** By label, pool Dynamic: census +0.09, redistance +0.09, body forces +0.07, momentum +0.04, ring entry +0.03, remap +0.04; Full: momentum, redistance, extension, advect, forces and transfer +0.06 to +0.09 each. The causes are the clamp and mask per load and the wrapped store per write in every h kernel (the store extent is a run-time value), the canonical-or-h load in kernels that walk every owner, and the ring entry at each relayout. Not fixed. The choices: accept it; a compile-time unwrapped variant chosen by policy (a second pipeline set, and a recompile when the policy changes); or branch the census and forces on owner width.

Admission. Launch width at the arrival frame, pool, in thousands of workgroups (arrival / next; a zero-detail frame is 97): 126 / 104 for one tile, 144 / 121 for 64, 241 / 209 for 512; before 365 / 303, 373 / 310, 407 / 344. Wall at a warm arrival of 512 tiles 15 to 17 ms (24 to 29 before; steady 12.1); 256³ with 64 tiles 26 to 39 ms (133 to 287 before; steady 14.9). Pass counts at the arrival frame equal the steady frame's. What is left: the reallocation and rebind on the host (1 to 4 ms), first use of 29 pipelines at the very first admission of a session, and the T-sized relayout head (128³: census 0.2, remap 0.15, layout build 0.1; 256³: the extension run for the census +2.2, census 0.72, layout build 0.37). A host-counted request needs neither the census nor its extension; that is the next domain-wide item and it sits in the scheduling and dynamic files.

Quality (4 s, unperturbed and two gravity perturbations of 1e-6 an arm, before and after): no fatal and no GPU error. Full is bitwise the tree before on pool and fig-9. Requested and Dynamic on the pool agree to every printed digit through frame 30 and first differ at frame 40 by 1.6e-9 and 3.6e-10 relative in V, then diverge through the impact as the perturbed arms do. Volume drift, owner-integrated kinetic energy, maximum speed and h-tile counts sit inside the before spread, with two edge reads: pool Requested kinetic energy at frame 60 is 5% above a three-sample spread, and fig-9 Dynamic h tiles are 3% under at frame 120 and 4% over at 180. Accepted residuals were not printed by the lab.

Not done: Dynamic and bodies in a window (the census admits tiles the host learns two frames late; a box saves only the dry axis on surface-spanning scenes, so the paged store above, about 5.87 kB a tile, is the form that pays); the rest of the first-tile jump (band solve, the extension seam tier and the regular-fine kernels are launch-bound and owned elsewhere; `umShiftLimit` is one global scalar that turns h at the first h tile). The ghost-ring set needs no new list: the ring is one bit a tile written by the builder's `mirror` entry at each relayout, and its 26-neighbour subset is the compact seam-4h list (`support[7N+20…]`); `encodeRing` is the reusable launch site. The claims buffer grows by one word per 4h vertex for `umHeldIndex` (0.11 MB at pool 128³, 1.1 MB at 256³, by arithmetic), proportional to T.

Follow-up, 6 October 2026 (landed): the Dynamic and Full loss is recovered by one WGSL source in two store forms. Every domain module declares `override udWindow:bool=true` and carries each accessor twice behind it; a pipeline of the direct form is compiled with `udWindow` false, where the clamp, mask, wrap and extent test are gone and an h texture's canonical store writes both homes untested, and the storage hands a stage the form that matches its extent at dispatch (`uniformDetailPipeline` / `uniformDetailPick`; Dynamic, Full and bodies start direct, a Requested store stays general, a change of class compiles in the background and is adopted at the layout sync). The census no longer tests an address per load: tile corners read the base block and h-tile vertices the h texture, by owner width. Frame, three arms from one tree, before / landed / fixed in ms: pool Dynamic 21.29 / 21.63 / 21.32, Full 30.43 / 31.12 / 30.45; fig-9 Dynamic 18.77 / 19.18 / 18.83, Full 22.83 / 23.41 / 22.90 (two samples each); what remains under Dynamic is the ring entry pass at a relayout, 0.03 ms. Not run after the follow-up: `uniform-atlas-address`, `uniform-volume`, and a Requested arm against roots of the same tree (GPU use was stopped first).
