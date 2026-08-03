# Hero garden scene — implementation plan

Evidence in `HERO_GARDEN_AGGREGATE_SDF_ASSESSMENT.md`. Intent in
`HERO_GARDEN_HOSE_SCENE_PLAN.md`.

The scene renders today at 19.9 ms / 720p with the full lighting stack on. It is
not blocked on perf. It is blocked on the primitive budget, on shading, and on
three registries that force every new thing to be authored as a one-off.

---

## Decisions

1. **The aggregate primitive is `smoothUnionCluster`: a smooth-min over a
   jittered, domain-repeated sphere lattice, intersected with a lobe envelope.**
   Not an envelope with noise displacement applied — that construction loses
   58.9 % of rays to tunnelling, because the march steps by `|d|` and a displaced
   field is not Lipschitz-1. Smooth-min understeps and is safe. Ship 8 neighbours,
   jitter ≤ 0.3 of the lattice period; measured 0 % missed at ~8 of the 48
   available march iterations.
2. **Parameters live in the scene arena, not the record.** The 64-byte record has
   three floats of per-kind space. Use the word-13 arena pointer, exactly as
   `terrainHeightfield` does.
3. **The cluster reports two shapes.** Full SDF to the render ABI; an inflated
   ellipsoid envelope to the voxeliser, node-mip oracle and every bounds
   formula. The pyramid samples one point per voxel centre and reads
   under-population as empty air, so it must over-occlude, not under-occlude.
4. **Registries before the kind.** Adding a primitive kind touches fourteen
   dispatch sites, thirteen of which fail silently. Adding a generator has no
   registration surface at all. Fix both before adding to either.
5. **Generators become document nodes.** They currently bake to flat geometry at
   build time, discarding seed and parameters. A saved scene should hold
   `{generator, seed, params}`, not hundreds of ellipsoids.
6. **Camera aperture becomes a scene property.** Hard-coded 104° lens against a
   reference that reads ~50 mm. Largest single change; gates the framing.

---

## Phase 0 — Unblock

- Single-face cull on the primitive-visibility raster (~2× on that pass).
- Wire the porcelain terrain to a procedural surface. It currently resolves to a
  flat passthrough, so the vessel — most of the frame — has no grain.
- Resolve the doubly-authored coping: the crest moved from the heightfield into
  the scenery sweep and nothing asserts they agree. Four tests are red on it.
- Register a GPU smoke lane. Nothing renders this scene in CI, and a withdrawn
  node-mip pyramid is a ~15× cliff reported only as a console warning.

*Gate:* tests green, scene renders in CI, pyramid withdrawal fails something.

## Phase 1 — Registries

- `SCENERY_GENERATORS` catalog + a `generator` scenery node kind; convert the six
  existing generators, then delete the hero-specific composition step.
- `surface` field on `SceneryMaterial`. Material selection is currently a regex
  over object names, which is why generators carry comments about what they may
  not be called.
- One frozen primitive-kind table driving the TS union, the u32 tag, the CPU SDF
  and the WGSL, with a completeness test over all fourteen per-kind arms.

All three follow the existing house pattern: frozen catalog, no mutable
`register()`, catalog test that proves the freeze.

*Gate:* hero document round-trips as generator nodes and re-expands identically.

## Phase 2 — The primitive

Build `smoothUnionCluster` through the Phase 1 table. CPU and WGSL SDFs agreeing
to the same 3e-4 the other marched kinds are held to.

*Gate:* a new CPU oracle asserting zero tunnelled rays and a bounded step count —
this is the test that stops displacement being reintroduced — plus CPU/GPU parity
and bounds conservativeness.

## Phase 3 — Move the set onto it

Budget is 2 227 of 4 096, with the bonsai (1 339) and coping (672) holding 90 %.
The planned set overruns. Below that sits a silent per-brick candidate ceiling
already exceeded at 25 mm.

- Canopy, trunk and root flare onto the cluster. This also removes the visible
  capsule seams at every limb junction.
- Boulders onto the cluster at low jitter.
- Raise pebbles to the planned population; re-measure per-brick candidates and
  halve the render lattice if still over.
- Wire the rosette generator — built and tested, never called by the scene.
- Fix the silhouette sparkle before this lands; the cluster multiplies the
  affected edge length.

*Gate:* stated primitive headroom, busiest brick under the candidate ceiling.

## Phase 4 — Water

Independent of 1–3.

- Stepping discs become rigid bodies. They are scenery today, so water passes
  through them.
- Caustics: the map is produced but never consumed — it is in no bind group, so
  enabling the flag changes nothing. Needs the consumer, the scene's own light
  direction, a heightfield receiver instead of a flat plane, and a real
  ray-bundle energy term.
- Teal as a scene property. The optics table is frozen and inlined at build time;
  at pond depth it yields almost no chroma.
- Spray: the render path is complete, the simulation is dead code that is never
  instantiated. Needs a solver owner and a plunge/impact spawn source.

## Phase 5 — Camera and tone

- Aperture as a `CameraState` field, threaded as a uniform. Re-frame the hero
  camera afterwards.
- Exposure and a filmic curve. Tonemapping is Reinhard with no exposure control,
  which is what the scene is currently compensating for with raised light
  intensity.

---

## Order

Phase 0 in any order. Phase 1's primitive table before Phase 2. Phase 3 after
Phase 2. Phases 4 and 5 are independent of everything else and of each other.

---

## Corrections found while building it

**Decision 1's "ship 8 neighbours" is unsound, and the measurement that
recommended it was reading the wrong failure.** The eight cells surrounding the
query point are the right *set*, but the set changes as the point crosses a cell
boundary, and at the instant a cell leaves, its sphere is only
`(1 − jitter)·period − radius` away — 1.2 mm at hero canopy scale, well inside
the smooth minimum's blend. The field therefore *jumps*. Measured over 200 000
sample pairs at the recommended configuration, its Lipschitz constant is **60**,
and 5 of 2 000 rays step clean through the surface. Twenty-seven cells do not fix
it (that block is centred on the point's own cell, so its membership changes at
the same boundaries), and neither does an order-independent fold.

A **four-cell block centred on the point** is the smallest that contains the
eight with a ring to spare; a leaving sphere is then `(2 − jitter)·period −
radius` away — 17.2 mm — and the measured constant is **1.000**, with zero
tunnelled rays. Sixty-four cells is not sixty-four sphere evaluations: the outer
fifty-six are rejected by the distance to their *unjittered* centre, which needs
no hash. The authoring contract is now `(2 − jitter)·period ≥ floret + smooth`,
enforced where a scene can reach it.

The residual is not tunnelling: about one ray in six hundred, all at grazing
incidence, reports a hit past the dense oracle's crossing because the field dips
to zero between two march samples without entering the 37 µm acceptance band.
That is the band buying the step count, and it is a silhouette pixel rather than
a hole.

**Phase 3's gate is not met at the authored lattice, and the aggregate is not
what stands between us and it.** The canopy move took the set from 2 227 records
to 1 209 and the bonsai from 1 339 to 191, so the 4 096 ceiling is comfortable.
The per-brick ceiling is not: at 25 mm cells (200 mm bricks) twelve bricks hold
more than the 64 candidates the hierarchy binds, worst 128. Attributed, the
overfull bricks are **the swept coping rail** (83 cones in one brick) and **the
bonsai's fork** (103) — the two things still spelled as chains of explicit
primitives, and the two an *ellipsoidal-lobe* aggregate cannot express. Sweeping
the lattice: 12.5 mm leaves six bricks over at worst 72; **7.5 mm clears it
entirely**, worst 51.

So the remedy the plan already names — halve the render lattice — works, but the
lattice is currently one number shared with the solver, where 7.5 mm overruns the
65 536-vs-65 535 one-axis dispatch. The scene opens dry, so the shipped default
would be fine and `water: true` would not. Separating the render lattice from the
solver's, or a **swept-arc aggregate** for the rail and the trunk, are the two
real routes. Neither was taken here.

**Phase 3's "trunk and root flare onto the cluster" is only half reachable.** A
cluster's envelope is an ellipsoid, so it serves the canopy lobes, the boulder
caps and any squat mass; it cannot express a tapered swept trunk, and chaining
clusters along one would reintroduce the seams at every hard lobe clip. The
capsule seams at limb junctions want the swept-arc kind, not this one.
