# The hero scene — porcelain pond, hose filling

Reference: `output/imagegen/garden-pond-hose-fill-simplified.png`

Goal: one scene that is the product's face. A cream-and-white porcelain pond,
teal water, a hose plunging in with a splash crown and concentric ripples,
ringed by mushroom-cap boulders, stepping discs, pebble beds, air plants and a
dense-canopy bonsai. Faithful — not "evocative of".

The bet is that this is the right stretch: an SVO whose leaves resolve exact
analytic surfaces is the one representation that can hold cauliflower-floret
detail and a 1.2 m pond in the same frame, and the fluid/terrain coupling is
the one that can make the water meet the coping where the coping actually is.

---

## 1. The reference, read as geometry

Nine object families. For each, the *generator* — the small parameter set that
makes one, and makes the next one different:

| Family | Count in frame | Essential form | Generator parameters |
|---|---:|---|---|
| Pond vessel | 1 | Closed organic plan curve; basin floor; rounded coping ring rising ~6 cm; inner face falling to water | plan-curve control points, rim height/width profile, basin depth, beach-shelf sector |
| Pebble beds | ~400 | Dense bands of graded ellipsoids hugging the coping's outer and inner feet | band curve, width, size grade, packing density, seed |
| Mushroom-cap boulders | 4 (a graded set) | Flattened ellipsoid cap on a short tapered stem, cap overhanging stem | cap radii, flatten, stem taper/height, tilt, seed |
| Stepping discs | 5 | Low cylinders, slightly domed, wading into the water on a curve | path curve, spacing, radius grade, submersion |
| Bonsai | 1 | Multi-trunk buttressed root flare, smooth pale limbs, canopy of dense fine florets in fused lobes | height, root count, flare, limb spiral, lobe count, floret scale, seed |
| Air plants | 4 | Rosette of ~9 tapered recurved blades | blade count, length, taper, recurve, splay, seed |
| Hose | 1 | Curved dark-teal tube over the back coping, brass ferrule, nozzle aimed down-in | polyline control points, bore, ferrule |
| Water | 1 | Teal body, visible floor, caustics, concentric ripples, plunge crater, splash crown, surface glints | solver |
| Ground plane | 1 | Pale textured plaster the pond is set into | terrain base |

Everything above except the water and the ground is *organic-by-variation*: the
same generator, a different seed. That is the requirement the user named, and it
is the reason none of this should be authored as literal coordinates.

---

## 2. Where each object has to live — the one rule

The engine has three authorities and they are not interchangeable:

| Authority | Who consumes it | Shapes available |
|---|---|---|
| **Terrain heightfield** (`lib/terrain.ts`) | fluid solver, rigid contacts, *and* the renderer (`terrainHeightfield` primitive kind) | any non-overhanging height function; analytic (≤8 features) or a sculpted grid up to 262 144 samples |
| **Scenery graph** (`lib/scenery-graph.ts`) | renderer only | box, cylinder, ellipsoid, capsule, torus, cone, procedural tree |
| **Rigid bodies** (`lib/model.ts:21`) | solver + contacts + renderer | sphere, box, capsule, cylinder — that is all |

**Scenery is invisible to the water.** `VoxelSceneSource` is
`VoxelContainerBoundarySource | VoxelTerrainSource | VoxelRigidSource`
(`lib/voxel-scene.ts:183`) — no scenery term. So the rule that decides
everything:

> Anything the water must touch is terrain or a rigid body. Everything else is
> scenery.

Applied:

- **Pond vessel → terrain, entirely.** Basin, coping ring, beach shelf, bank
  contours: one sculpted `TerrainGrid`. This buys one authority for both the
  solid the water rests in and the surface that gets rendered, so the waterline
  cannot disagree with the geometry. It costs the coping's slight *overhang* —
  a heightfield cannot overhang — which from the reference camera is not
  visible; the rounded crest and the fillet where rim meets ground are.
  This is the single most important decision in the plan and the one I would
  most want a look at a first render before committing to.
- **Stepping discs → static rigid cylinders.** They stand in the water; water
  must part around them. The existing garden already does exactly this
  (`lib/scenes.ts:717`).
- **Everything else → scenery.** Boulders, bonsai, air plants, hose, pebbles.
  All of it is dry, or dry enough that render-only is faithful.

Container and lattice: **1.8 × 0.6 × 1.2 m at `finestCellSize_m: 0.0075`,
`brickSize_cells: 8`** → 240 × 80 × 160, and every dimension is a whole number
of bricks. Chosen so the reference's ripples resolve: their wavelength reads as
~4 % of pond width ≈ 5 cm ≈ 6.7 cells, and the hose stream (~4 cm bore) lands
at ~5 cells across. The current garden runs 0.025 m, at which both features are
sub-cell and the reference is simply unreachable. This is 7.3× the current
garden's cell count and is deliberately the scene the Power Liquids leap program
exists to serve — see §6.

---

## 3. What the engine already gives us, and the five real gaps

Already there, and more than I expected:

- Marched-SDF primitives are a walked path. `torus` and `cone` were both added
  as `exactDistance | marchedIntersection` (`lib/svo-primitive-abi.ts:290-292`),
  with the ABI comment stating the thesis outright: a new shape "costs one
  signed-distance function on each of the CPU and the GPU, not a bespoke quartic".
  A new kind touches ~12 lib files plus tests, by the cone's own footprint.
- Procedural generation with a determinism contract (`voxel-scenery/procedural-tree.ts`) —
  seed in, geometry out, static revision keyed on the result.
- Terrain sculpting to 262 k samples with a bilinear form the WGSL mirrors exactly.
- Per-voxel exact analytic hits: the SVO stores a material owner per voxel and
  the ray solves the real primitive inside it. Silhouettes are exact, not voxel-stepped.
- Water refracts the composed scene (`sceneTexture` in the raster water pipeline)
  and secondary particles exist for spray, 16 k default and up to 65 k.
- Sway, so the canopy and the air plants can move without re-voxelizing.

The five gaps:

**G1 — Primitive budget: 4 096.**
`SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES = 4_096` (`lib/svo-primitive-candidates.ts:18`).
Counting the scene: ~400 pebbles + ~40 tree limbs + 12 boulder parts + 36 blades
+ 12 hose segments ≈ **500**. The budget is not the problem for anything except
the canopy. Good news, and it means pebbles can stay honest individual ellipsoids.

**G2 — One owner per voxel.** This, not the primitive count, is the real limit on
fine detail. A floret smaller than a voxel loses to whichever primitive owns that
voxel. At 0.0075 m a 3 cm floret is 4 voxels wide and renders cleanly; the
reference's finest florets are ~8 mm, i.e. one voxel. A faithful canopy is
~10⁴ florets. Both an explicit-primitive canopy and a finer lattice lose here.

**G3 — No aggregate primitive.** The answer to G2, and the actual stretch: a
marched SDF kind that owns a coarse region and resolves sub-voxel structure
analytically inside it. One record, unbounded detail, exact silhouette at any
zoom. `marchedIntersection` was built for precisely this and has two shipped
users already.

**G4 — Caustics are implemented and switched off.** `causticsEnabled = false`
(`lib/webgpu-water-pipeline.ts:1053`), with the projection shader, pipeline and
384² target all present. The reference's single most recognisable feature is the
caustic net on the pond floor. This is a re-enable-and-tune, not a build.

**G5 — No swept-tube or scatter-band generator.** The hose is a polyline, the
pebble beds are a band. Both are currently spelled as hand-placed beads; both
want to be generators.

---

## 4. The procedural vocabulary

Six generators. Each is pure, seeded, deterministic, and plans geometry before
publishing it — the contract `planProceduralTree` already establishes. Each
lives in `lib/voxel-scenery/`, each gets a declarative node kind in
`SceneryNode`, and each gets a CPU-oracle test that pins its extent.

1. **`pond-vessel`** *(terrain, not scenery)* — closed Catmull-Rom plan curve →
   signed distance in plan → height profile (outer fillet, rim crest, inner
   face, basin floor, beach ramp over an angular sector) → baked `TerrainGrid`.
   Variation: control points, rim profile, beach sector.
2. **`scatter-band`** — pebble beds. Poisson-disc scatter in the annulus between
   two offsets of a plan curve, each sample an ellipsoid with graded radii,
   random tilt, bedded to the local ground. Variation: density, grade, seed.
3. **`capped-boulder`** — the mushroom stones. Tapered stem cone + flattened
   ellipsoid cap with a small overhang and a slight tilt. Variation: 6 numbers
   and a seed; the reference's four are one generator at four scales.
4. **`rosette`** — air plants. N tapered cones on a splay/recurve profile.
5. **`swept-tube`** — the hose. Polyline → capsule chain with a shared bore,
   plus a ferrule cone. Also reusable for cables and rails.
6. **`floret-canopy`** *(the new primitive kind)* — see below.

Variation is one seed per object plus its shape parameters, which means "give me
another garden" is a re-seed and not a re-author. That is the property the user
asked for, and it is what makes the hero scene a *family* rather than a
one-off — which matters for the scene library and for marketing stills.

### The floret canopy — the stretch

A new SVO primitive kind, `floretCluster`, flagged
`exactDistance | marchedIntersection`, alongside torus and cone.

Its SDF: an ellipsoidal lobe envelope, with the surface displaced by two octaves
of domain-repeated sphere packing — a hashed jitter per repetition cell so the
florets are irregular, smoothly unioned so they fuse into cauliflower rather
than reading as beads. Sub-voxel detail then costs one SDF evaluation per march
step and *no* primitive records: a canopy of seven lobes is seven records that
resolve arbitrarily fine detail at any zoom.

This is the piece I would build second, not first — it is the highest-value and
highest-risk item, and it should be built against a canopy that already works.

---

## 5. Phases

Each phase ends somewhere shippable, and each names the oracle that says it
worked. Ordered so the riskiest item is de-risked by the cheapest possible
version of itself first.

**Phase 0 — Frame the target.**
Fix camera, container, lattice, palette and waterline. Stand up
`hero-garden-hose` in `SCENE_CATALOG` on the existing garden's geometry at the
new lattice, with the new camera. Ugly, but it renders and it establishes the
frame every later comparison is judged in.
*Oracle:* scene loads, `check:scenery` clean, one render at the hero camera.

**Phase 1 — The vessel.**
`pond-vessel` generator → `TerrainGrid`. Water settles in it. Coping reads as
rounded. Waterline sits where the reference's does.
*Oracle:* a settling test — fill to the waterline, run to rest, assert no
residual speed and that the wetted plan matches the generator's own curve within
a cell. This is the phase where the "coping as heightfield" bet is confirmed or
paid for; do not proceed past it on faith.

**Phase 2 — The dry set, at fidelity.**
`scatter-band`, `capped-boulder`, `rosette`, `swept-tube`. Pebble beds, four
boulders, four air plants, the hose. Bonsai on the *existing* tree generator
re-tuned toward the reference silhouette (multi-trunk, tighter lobes) —
explicitly not the new primitive yet.
*Oracle:* `report:svo-scenes` coverage, primitive count under budget with
headroom stated, and a side-by-side still against the reference.

**Phase 3 — Water that looks like the reference.**
Re-enable caustics (G4) and tune. Plunge crater, ripple train, splash crown via
secondary particles. Tune inflow bore/velocity so the falling column stays
coherent for its ~15 cm drop.
*Oracle:* a ripple-wavelength measurement against the analytic dispersion
relation for the depth — the repo's house style is an oracle with a known
answer, and this one has one. Plus a frame profile: expect the same
primary-visibility-bound structure the 1500² garden showed, and watch the node-mip
pyramid does not silently withdraw as the scene grows.

**Phase 4 — The floret canopy.**
`floretCluster` primitive kind: CPU SDF, WGSL mirror, ABI record, candidate
bounds, node-mip and coverage oracles, editor entity. Then `floret-canopy` as a
scenery node the bonsai's canopy switches to.
*Oracle:* the existing `webgpu-svo-primitive-exact` and `svo-primitive-ray`
suites extended to the new kind — CPU and GPU must agree on hits to the same
tolerance every other kind is held to.

**Phase 5 — Variation and the family.**
Expose seeds. Generate a handful of re-seeded gardens; confirm they are
recognisably siblings and none degenerate. Pick the hero seed and pin it.

---

## 6. Risks, honestly

**The solver is mid-cutover, and it is slow right now.** Today's status doc
records a real ~3× unexplained regression on the mini lane (245 ms/advance at
leaf 32 vs 69.6 in the 07-29 capture) and passes-per-advance flat at 80 against
a target of ≤25. This scene asks for 7.3× the cell count of the current garden.
**Mitigation:** the scene's *look* must not block on solver throughput. Phases
0–2 and 4 are dry-path work and are unaffected. Phase 3 should be built and
judged first as a still — a settled pond with caustics and a frozen ripple field
is most of the reference image — and only then pushed to interactive rates. If
it lands at 10 fps, that is a legitimate hero *still* and a known perf item, not
a failed scene.

**The coping-as-heightfield bet** (§2). Cheap to test in Phase 1, expensive to
discover in Phase 4. Fallback: heightfield keeps the wet inner face, a swept
torus chain adds the outer overhang, and the two are authored from the same plan
curve so they cannot drift.

**The floret SDF's march cost.** Two octaves of domain repetition inside a
48-iteration march, across a canopy that fills a good fraction of the frame, on
a path already measured as primary-visibility bound. Bound it by making octave
count a distance-driven parameter — the canopy is the one object in the scene
whose detail genuinely can fall off with range.

**Ripples at 7.5 mm.** 6.7 cells per wavelength resolves a ripple but will damp
it faster than the reference shows. If the ripple train dies before it reaches
the coping, the honest answers are a finer band local to the pond surface or
accepting a shorter train — not a shader fake, which would not survive the
camera moving.

---

## 6b. What phase 0/1 actually measured (2026-08-03)

Built: `lib/voxel-scenery/pond-vessel.ts` (generator), `lib/hero-garden-scene.ts`
(scene), `tests/pond-vessel.test.ts` (11 oracles, green). Registered as
`hero-garden-hose` on the Garden shelf. It loads and simulates at 25 mm.

**Four things the plan got wrong, found by building it.**

1. **The vessel was an extrusion.** The first generator held the coping to a
   constant section and had a test *pinning the crest to a constant height*. That
   is the wrong invariant: the eye reads an unvarying crest line long before it
   reads a wandering outline, so the rim looked machined however much the plan
   wobbled. The section now swells and narrows along its run — at a different
   lobe count from the plan, so the two read as two accidents of one hand rather
   than one repeating motif — over 2.5 mm of two-octave relief. §4 said "organic
   by variation" and then varied only the outline; varying the *section* is what
   the word was supposed to mean.

2. **Nothing was planned for surface grain.** The plaster speckle on every white
   surface is a large part of why the reference reads as hand-made, and it is
   sub-millimetre — a material, not geometry, at any lattice we will ever run.
   `lib/svo-procedural-material.ts` already carries `stone`/`ceramic`/`organic`
   noise policies over colour and roughness; the terrain material needs wiring to
   them. Cheap, and the highest return per line in the whole plan.

3. **The tree is not made of cones.** §5 phase 2 said to re-tune the existing
   procedural tree. But the reference's trunk is a *fused buttressed root mass* —
   smooth-min-unioned tubes, the melted-wax look — and its canopy is ~10⁴
   florets. Those are the same operation. One marched **smooth-union displaced
   cluster** primitive serves the trunk, the canopy, the worn boulders and
   possibly the pebble beds, which is a far stronger case for the one hard engine
   change than "florets" was on its own. It should be the plan's backbone, not a
   phase-4 stretch.

4. **Pebbles were undercounted and mis-modelled.** ~600–900, not ~400 — still
   inside the 4 096 budget, so they stay individual ellipsoids — but they are
   *contact-packed* in graded bands, not Poisson-scattered. `scatter-band` needs
   to be a packing.

**Three solver walls, in the order they were hit.**

| Wall | Evidence | Status |
|---|---|---|
| Waterline above the outer ground floods the domain | `tank-fill` wets every column the level clears; the outer plaster sat below it, so a 2-cell film covered the whole 1.8 × 1.2 m floor and the sparse authority published no liquid-row frontier at all | **Fixed.** The waterline is now measured *down* from the outer ground, in cells, and a test sweeps the ground outside the coping for clearance |
| 7.5 mm overruns a device limit | "Seed global fine bricks from every interface leaf" dispatches one workgroup per interface leaf on one axis: 65 536 against WebGPU's 65 535 | **Open, engine-shaped.** Needs a 2-D dispatch. No art direction moves it |
| 25 mm, 15 mm and 12.5 mm refuse the t=0 pressure gate | Identical failure across row counts, so not scale. Reproduced with an analytic basin in place of the generated grid, so **not** the sculpted-heightfield path — the coping-as-terrain bet is cleared | **Fixed, in the engine.** See §6c. The inner face was not implicated; all three lattices publish t=0 |

At 25 mm the scene initialises, solves and simulates; it fails only the
equilibrium *quality* gate — 5 disconnected components, 27 % volume drift — which
is the same shallow-wedge story seen from the other end, plus a coping only two
cells wide. No GPU lane is registered for it yet, and therefore no variant: an
authored variant no lane claims is a fork by another name.

**The scene opens dry, and the water is opted into.** `systems.fluid` already
existed as the gate `planSceneRuntime` reads — it is how a new document reaches
its first frame without a solver — so the hero scene sets it false by default and
`createHeroGardenHoseScene({ water: true })` turns it on. The two documents are
identical apart from that one leaf, which is asserted: the fill, the initial
condition and the jet are authored either way, so the switch fills the pond that
was designed rather than an empty tank, and a dry scene cannot quietly become a
second scene wearing the first one's name. The switch is also on the Fluid tab of
scene configuration, for any scene, since "render the set without bringing up a
solver" is not a hero-specific need.

The vessel is now baked at a quarter of the fluid cell (6.25 mm). The coping is a
`terrainHeightfield` primitive — the ray marches the grid rather than voxelised
cells — so the sample spacing, not the lattice, is what limits its rendered form;
at the solver's 25 mm the rim was two samples wide and there was nothing to
judge. A quarter cell also puts every solver column centre exactly on a grid
node, so the finer bake costs the water nothing.

**The next move is the inner face**, not the lattice: steepening it toward the
reference's near-vertical profile is the one change that plausibly addresses the
component count and the drift together, and it is a change toward the source
image rather than away from it. (It was also nominated for the t=0 gate. It was
not that — see §6c.)

## 6c. The t=0 gate was the page directory, not the vessel (2026-08-03)

The pressure solve refused this scene at t=0 on every lattice tried, and the
browser's own words named the row: `mgpcg[0]=2` is `ERR_ROW`, `mgpcg[6]=22` the
failing stage, `mgpcg[7]` the row. Reproduced in Dawn against the authored scene
through `WebGPUUniformEulerianSolver.createAsync`, byte-for-byte identical to
the browser, then re-walked on the CPU from a dump of the SPGrid topology
buffer.

Stage 22 is `opPageSlot` rejecting a page whose record decodes somewhere other
than the cell that was asked for, and the caller was not the row's own stencil —
it was `finerAdjoint`, which asks every row above the finest level whether its
eight children exist one level down. That question is answered by the dense
logical-page directory, guarded by `pageFor(...) != INVALID`.

**The directory could not answer it.** Its resting value was the buffer's
zero-fill, and zero is the perfectly legal index of physical page 0. The commit
path writes a directory slot only for a page that is *arriving* or *retiring*,
so a logical page that has never held a cell is never written at all: it
answered "page 0" to every query. Row 519 sat at level 1 over page (4, 1, 5),
which holds no finest cells; the directory handed back page 0, whose origin is
(24, 8, 12); the operator compared the two and rejected the publication.

Why this scene and not the tanks: a wide shallow basin in a large open container
has coarse rows standing over regions with no finest-level cells at all. The
mini dam has no empty logical page in its 16³ domain and so never asks. The
defect was present in every scene — `minimal-power-dam-break` carries four stale
slots at level 0 — and only this one queried them.

Fixed by authoring the page directory as INVALID at allocation, in both topology
banks (`lib/webgpu-octree-spgrid-vcycle.ts`), pinned by
`tests/octree-spgrid-page-directory.test.ts`. 25 mm, 15 mm and 12.5 mm all
publish t=0 after it; `minimal-power-dam-break` is bit-identical.

7.5 mm still fails, but earlier and differently — cold topology publishes no
liquid-row frontier — which is its own wall and not this one.

## 7. What I would do first

Phase 0 and Phase 1, in that order, and then stop and look at a render before
building any generator beyond the vessel. The vessel decides the frame, the
waterline and the coping bet; every other phase is comparatively independent and
can be parallelised once it is settled.
