# Hero garden at 1000× — fidelity handoff

**Goal:** `hero-garden-hose` converges on its reference plate. Not "looks nicer" —
a registered, scored comparison against a fixed camera, driven to the point where
the difference is grading rather than geometry.

**Thesis in one line:** the 1000× is bought by *making leaf size a function of the
pixel* and *making surface detail a field rather than a record*. Neither is a
smaller `cellSize`, and record count barely moves.

**Supersedes nothing.** `docs/svo-raster-visibility-handoff.md` (W0–W4, landed)
is the substrate this stands on: it made visibility stop scaling with authored
record count. This document spends that headroom on image.

All `file:line` verified at HEAD `f7850f9`. Frame numbers trace to
`docs/svo-render-frame-anatomy.data.js` and carry that lane's ±5 % error bar
(`svo-dry-lane-noise-floor`).

---

## Status — H0 and H1 have landed

**H0 (fidelity gate) — done. H1 (grade half) — done. H2 — substrate only.**
Everything below this block is the original plan and is unchanged; this block is
what the plan turned into when it was run.

### The lane

```
npm run gate:hero-fidelity            # 4 reps, both scores, contact sheet
npm run gate:hero-fidelity:overlay    # the scored regions drawn on the plate
npm run solve:hero-camera             # re-solve the camera against the plate
npm run solve:hero-grade              # re-solve exposure + white balance
```

- `lib/hero-fidelity-score.ts` — ΔE₀₀ (CIEDE2000, checked against Sharma's
  34-pair table) and the gradient-magnitude histogram distance, the latter in
  **octaves** with a **signed bias** (positive = frame smoother than plate).
- `lib/png-codec.ts` — the tree had an encoder and no decoder; the plate needed
  one. Refuses 16-bit/paletted/interlaced rather than decoding approximately.
- `tools/hero-fidelity-gate.ts`, `tools/solve-hero-camera.ts`,
  `tools/solve-hero-grade.ts`.
- The plate is the one already in the tree:
  `output/imagegen/garden-pond-hose-fill-simplified.png`.

### Numbers, at 836 × 470 on the M1 Max

| Region | ΔE₀₀ before | ΔE₀₀ after | grad (oct) before | after |
|---|---|---|---|---|
| pond | 23.18 | 21.08 | 3.669 | 3.470 |
| coping | 20.15 | **3.31** | 2.158 | 1.144 |
| stones | 14.34 | 13.88 | 1.911 | 1.564 |
| canopy | 18.56 | *23.36* | 1.340 | 1.244 |
| ground | 21.88 | **2.43** | 1.419 | 1.538 |
| **mean** | **19.62** | **12.81** | **2.099** | **1.792** |

H1's gate was "ΔE₀₀ on the coping and ground regions halves". Coping fell 6.1×
and ground 9.0×, with no geometry changed.

### Three things the measurement corrected

1. **The camera was not registered, and it was contaminating every score.**
   Solving it (1.40 m → 1.667 m, elevation 0.40 → 0.50 rad; azimuth deliberately
   held at 5.4 so the derived key light did not move) improved the mean gradient
   score 2.504 → 2.099 octaves *with ΔE₀₀ flat* — the signature of a
   registration fix rather than an image change. Do not trust a pre-H0 score.

2. **The frame was over-*exposed*, not over-lit.** Neutral regions sat 27 L\*
   above the plate and at b\* ≈ 0.5 against the plate's ≈ 5. A photograph of a
   white set in sun is exposed so the white does not blow; dropping the key
   instead would have changed the lit-to-shadow ratio, which is a physical fact
   about the set. The grade now carries a **white balance** (new: authored as
   channel gains, luminance-normalised so it cannot double as a second
   exposure) — see "four places" below.

3. **The canopy's lighting ratio is inverted, and the cause is §2.6, not H1.**
   In the frame the canopy is *darker* than the coping (L\* 74.6 vs 86.8); in
   the plate it is *brighter* (66.2 vs 60.5). That is why the canopy region
   regresses under the grade solve. Cropping both at the canopy region and
   putting them side by side says why, and it is worth doing before anything
   else is tried there:

   > The plate's florets are **convex and bright**, with soft shallow shadow
   > between them. The frame's are **inverted** — a crust of bright ridges
   > around near-black hard-edged pits.

   `BONSAI_POND_CANOPY` is ~950 florets at a 29.3 mm radius, so a floret is
   ~2.3 cells across at the 25 mm lattice and *the gaps between them are about
   one cell*. The voxelizer resolves those gaps as occupancy, cone AO at a
   0.62 aperture reads them as deep crevices, and they go black.

   This is the three-band detail law of §2.6 being violated: a feature at or
   below the leaf size is being resolved as **geometry** when the band it
   belongs to is **voxel normal** (H4) or **filtered BRDF** (H5). Resolving it
   as occupancy is a lie, and here the lie is not merely aliasing — it inverts
   the relief and costs a whole region's tone.

   So the canopy is **not** a fill-and-bounce job, and adding a fill light to
   chase it would be treating a symptom. It is direct evidence for H4/H5 and
   for the band assignment being per-evaluation. Note also that the two scores
   disagree about it in a way that is informative rather than contradictory:
   the canopy's mean gradient is *higher* than the plate's (0.431 vs 0.327 —
   those black edges) while its EMD bias is *positive*, i.e. smoother (large
   flat lit areas between the cracks). A frame can be simultaneously too noisy
   and too smooth, at different scales, and that is exactly what a
   distribution metric is for.

### The grade now travels through four places, not three

`resolveDisplayGrade` → `packWaterSceneOptics` lane 6 → `unifiedDisplayGradeBalanced`
(app) **and** `tools/write-frame-png.ts` (offline, hand-transcribed). The green
gain is *not* transported: lane 6 had two spare floats, so it is reconstructed
in the shader from the luminance-normalisation constraint. Adding a grade knob
means editing `lib/webgpu-lighting.ts`, `lib/webgpu-water-pipeline.ts`,
`tools/write-frame-png.ts` and `lib/model.ts`'s validator, or it drifts silently.

### Measurement facts worth keeping

- **The lane is reproducible.** Four interleaved reps at this camera and size
  agreed to 0.000 on every region — better than §5 feared. The
  `binDirtyBrickCandidates` atomic race does not bind here. The gate still
  compares through a floor (0.25 ΔE₀₀, 0.02 octaves) rather than against zero.
- **Score the right lane.** `tools/run-svo-dry-render-smoke.ts` applies the
  scene's ACES grade; `tools/benchmark-svo-dry-frame-gpu.ts` applies a bare
  gamma 2.2 with no curve and no exposure. Scoring the second would measure the
  missing tonemap and nothing else.
- The two vignettes compound in the app but not in the offline PNG, so the gate
  scores a frame ~8 % brighter at the corners than the canvas shows.
- **Pinning the camera invalidates previously recorded frame times.**
  `heroGardenCamera` is shared by `hero-garden-hose`, `hero-garden-hose-x10`
  (`lib/scenes.ts:1350`) and `tools/hero-floor-far-scene.ts`, which overrides
  distance but *inherits* the elevation that moved. Those lanes now cover a
  different number of pixels, so any absolute figure captured before this
  change is not comparable. `tools/hero-floor-sky-scene.ts` overrides every
  affected axis and is unaffected. Re-baseline before reading a regression.

### H2 — what exists

`lib/svo-field-program.ts` + `tests/svo-field-program.test.ts`: the tape
encoding, the two-register-file machine, arena packing, **Lipschitz through the
graph carried from day one**, `domain-warp` (op 1) with its three source solids,
and the generated WGSL mirror sharing the tree's existing `svoProceduralNoise`
so both sides agree by construction. The Lipschitz claim is enforced by a
finite-difference assertion over a sample grid, which is §2.4's debug mode as a
test.

Not yet done: ops 2–8, interval pruning (§2.5 — genuinely not needed until the
tape branches, see the module note), and the four-call-site bit-for-bit oracle.

---

## 0. The gap, decomposed

Reference plate vs. current frame, ordered by share of pixels that disagree.
This ordering is the whole argument for the sequencing in §6, and it is *not*
the ordering the request implies.

| # | Gap | Pixel share | Mechanism | Work area |
|---|---|---|---|---|
| 1 | **No water.** Plate is a refractive pool: stones seen through the surface, caustic net on the bowl, ripple rings, splash column and spray at the jet | ~55 % | `createHeroGardenHoseScene` ships `systems.fluid = options.water === true` — off (`lib/hero-garden-scene.ts:600`). The SVO path has **no caustics at all** (they exist only in the legacy `lib/webgpu-water-pipeline.ts`), and `svo-fluid-coverage.ts` is explicit that its volume is "a fraction in [0, 1], never a shape" — it feeds shadow cones, it cannot *be* the water surface | **H6** |
| 2 | **No key light.** Plate has a warm low sun, soft penumbra, contact shadows, warm bounce into shadow, mild bloom, filmic roll-off. Current frame is flat ambient white with neutral grading | ~all | Cone shadow/AO machinery exists and is tuned image-forward (`giBounceStrength: 1.8`, `svo-render-tuning.ts:152`), but the authored light rig and the tonemap are not doing the plate's work | **H1** |
| 3 | **The geometry is spherical, at every scale.** Plate boulders are asymmetric fractured masses with erosion pitting; the canopy is a self-similar Romanesco. Current: smooth ellipsoids, one-radius sphere lattices, a blobby crown. **This is a form problem, not a surface problem** — no amount of noise on an ellipsoid fixes it | ~30 % | 11 primitive kinds, all smooth convex analytic solids (`svo-primitive-kinds.ts:39`). The only aggregate is `smooth-union-cluster` — a sphere lattice smooth-minned at **one fixed radius** (`svo-cluster-arena.ts:33`) — which also costs one record per feature, the reason `SVO_DRY_SCENE_CLUSTER_CAPACITY` (4 096, `webgpu-svo-dry-scene.ts:373`) is a live constraint | **H2** |
| 3b | **The walls and ground are placeholder.** Plate is a photographic seamless — bounded plaster floor sweeping into a backdrop, granular, no corner line. Current: six axis-aligned 25 mm boxes with the front face opened, on an **infinite** terrain plane that recedes to a fog horizon | ~20 % | `EnvironmentProxyShell` (`voxel-environments.ts:46–58`), front-face special case at `svo-scene-primitives.ts:232`; `traceTerrain` bounds `t` only in `y`. Sculpted terrain never reaches the renderer at all (`cloneTerrain` drops `grid`) | **H2b** |
| 4 | **Voxels are 25 mm.** Visible as blocky quantisation in the pond bowl's occlusion | ~15 % | The octree has **one scene-wide cell size** — `renderCellSize = sceneDomain.cellSize_m / refineScale` (`webgpu-octree-sparse-bricks.ts:949`), and `sparseSceneNeedsConservativeSampling` compares a feature radius against that one number (`webgpu-sparse-scene-proxies.ts:1112`). There is **no screen-space LOD anywhere in the proxy path** | **H3/H4** |
| 5 | **Aliasing budget already spent.** Even today's 4.5 mm speckle shimmers | — | Stated in-tree: "the material has no filtering… a grain finer than a pixel aliases rather than blurs. The real fix is a distance-driven octave count, which is a shading-path change" (`svo-procedural-material.ts:29–39`) | **H5** |
| 6 | **Set is not the plate's set.** Missing stepping-stone discs, mushroom caps, the jet itself, the coral canopy | ~10 % | Authoring | **H7** |

**Read this table honestly:** the request is for finer voxels, and finer voxels
are #4. Water and light are #1 and #2 and together they are most of the frame.
H2/H3/H4 are still the right structural bet — they are what makes the *next*
decade of scenes possible — but a plan that does them first ships a
higher-resolution version of the wrong image.

---

## 1. Why 1000× is not a smaller `cellSize`

The domain is 1.8 × 0.6 × 1.2 m.

| Cell | Lattice | Note |
|---|---|---|
| 25 mm (authored) | 72 × 24 × 48 = 82 944 | 445.6 MiB resident at 10× records |
| 10 mm | 180 × 60 × 120 = 1.3 M | 761.4 MiB — the measured ceiling today |
| 5 mm | 360 × 120 × 240 = 10.4 M | **Refused**: `planSvoNodeMipPyramid` returns incomplete at 25 479 pages against `maxTextureDimension2D` = 16 384 |
| 1 mm | 1 800 × 600 × 1 200 = **1.30 G** | 15 625× the authored lattice, uniformly, over a domain that is 96 % air |

A uniform 1 mm lattice is the naive reading of "1000×" and it is unreachable:
it spends its entire budget on the far wall and on empty space, and it still
does not produce a single floret, because the *shape* it subdivides is still a
smooth ellipsoid. Refining a sphere gives you a rounder sphere.

**The two things that actually buy 1000×:**

### 1.1 Leaf size proportional to the pixel, not to the domain

At the hero framing a pixel is **≈ 2.9 mm on the near coping**
(`svo-procedural-material.ts:34`, `:128`) and tens of millimetres at the far wall. A
refinement rule of "split while the projected voxel footprint exceeds τ pixels"
at τ = 1 gives ~3 mm leaves where the camera is looking and ≥ 25 mm where it is
not — for a resident set that is **O(visible surface in pixels), not O(volume)**.

That is the structural claim, and it is what makes this affordable: resident
voxel count becomes bounded by the framebuffer times a depth-complexity
constant, so **memory scales with resolution, not with detail**. Going from
25 mm to 3 mm near-field is 570× the volumetric density in the region that
matters and roughly *flat* in bytes.

The tree already has depth (`finestLevel`, `maximumDepth`,
`SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM = 3`). What it does not have is a
*camera-driven, per-node* target. `environmentRefinementDepth` ships at 0 with
the reason stated in-tree: "the excursion budget a swaying prop is held to is
still derived from one scene-wide cell size" (`svo-render-tuning.ts:124–126`).
That coupling is H3's real work.

### 1.2 Geometry is a program, not a record

501 records must stay ~501 records. A floret canopy authored as records is
~50 000 of them and hits the cluster arena, the material table
(`SVO_DRY_SCENE_MATERIAL_CAPACITY = 8 192`, `webgpu-svo-dry-scene.ts:328`) and
the BVH in that order.

Instead a record names a **field program** — a short tape of composed procedural
ops (domain warp, fracture, variable-radius smooth-min, Worley erosion, recursive
scatter) evaluated as one SDF. §2 is the whole design. Cost is paid **per voxel
near the surface at voxelization time**, amortized over frames, and it is already
the axis the voxelizer is budgeted along (W2b's per-frame chunk budget). Record
count: unchanged. Effective surface features: unbounded.

**This is the 1000×.** 501 records, each a tape whose recursion and octave depth
is driven by the leaf size the camera warrants, resolved at ~3 mm over ~3 m² of
visible surface, is on the order of 10⁶ resolved features against today's ~10³
authored ones — and every arena constant stays where it is.

---

## 2. The shape language — geometry stops being primitives

This is the centre of the program, and it is bigger than "add noise to an
ellipsoid". **The base shapes are the problem.** Every solid in the set is
currently an ellipsoid, a capsule, or a `smooth-union-cluster` — a lattice of
spheres smooth-minned at *one fixed radius* (`SVO_CLUSTER_ARENA_BLOCK.smoothRadiusWord`,
`svo-cluster-arena.ts:33`). That construction has exactly one look, and the
current frame is it: a lava lamp. You cannot noise your way out of it, because
the failure is in the *form*, not the surface.

A boulder in the plate has structure at four scales and none of it is spherical:
an asymmetric mass flattened on a bedding plane (10–30 cm), fracture facets with
tight arrises where it broke (2–8 cm), erosion where the soft matrix receded and
left hard inclusions proud (3–15 mm), and crystalline pitting (0.3–2 mm). A
sphere lattice can produce none of those four.

### 2.1 A record's geometry becomes a program

New primitive kind **`field-program`** (`arenaBacked | marchedIntersection`),
whose arena block is a short **tape** — a DAG of field ops, 20–80 nodes. This is
the pattern the ABI already uses twice: `smooth-union-cluster` and
`terrain-heightfield` are both arena-backed records whose block carries what the
64 B record cannot (`svo-primitive-kinds.ts:63–72`). `SVO_CLUSTER_ARENA_BLOCK` is
the template to copy, including its rule that a slot past capacity decodes as a
**miss, never as the envelope** — "an aggregate that quietly became an ellipsoid
is a shape nobody authored" (`svo-cluster-arena.ts:110`). Same discipline here.

### 2.2 The op set, ordered by how much it un-spheres the frame

| # | Op | What it buys | Why it matters here |
|---|---|---|---|
| 1 | **Domain warp** — `d(p) = base(p + A·w(p)) / L` | organic mass | **The single highest-value op.** One ellipsoid with a two-level warp (≈15 cm for the mass, ≈2 cm for the lobing) reads as a rock. A hundred smooth-unioned spheres does not, and never will. If only one op ships, ship this |
| 2 | **Anisotropic frame** — scale + shear in a hash-derived rotation | bedding, flattening, "lying the way it fell" | Rocks are not isotropic. The current set is, uniformly, which is half of why it reads as CGI |
| 3 | **Fracture** — `smax(d, halfspace_i, k_i)`, small `k`, 3–6 planes | **flat faces with tight arrises** | The strongest single "this is stone" cue and it is *completely absent* today. Everything currently has one curvature regime: convex and smooth |
| 4 | **Variable-k smooth-min** — `k` scaled by the smaller operand's radius | unions that read at both ends | Today one `smoothRadius_m` serves the whole aggregate, so a 2 cm lobe and a 20 cm mass blend identically. That is the blobbiness, precisely |
| 5 | **Worley subtract** / **ridged Worley add** | vesicles, pitting, proud inclusions | The 3–15 mm erosion band |
| 6 | **Erosion mask** — modulate 5 by `dot(n, up)` and a cheap curvature proxy | weathering that collects where water and grit collect | Unmasked noise reads as *fabric*. Masked noise reads as stone. This is the difference between "bumpy" and "eroded" |
| 7 | **Recursive scatter** — `scatter(child, s)` where child may itself be `scatter(child, s/2.5)` | **the florets** | See §2.3 |
| 8 | **Chip / crack defects** — sparse hash-placed negative rounded boxes and thin planes | the authored "defects" | Asymmetry that noise cannot give you, because a chip is a *rare event*, not a field |

Ops 1–4 are the reimagination. Ops 5–8 are the granularity. Both are needed;
5–8 alone on today's forms just gives you bumpy lava lamps.

### 2.3 The florets, reimagined

The plate's canopy is a **Romanesco**: self-similar florets, each a cluster of
smaller florets, three levels visible. That is op 7 and nothing else — a scatter
whose occupant is a scatter. It is one record.

The payoff is that recursion depth falls straight out of the LOD:

> **recursion depth = octave count = detail band count = a single number derived
> from the brick's leaf size.**

At the far bank the canopy evaluates one level and is a mass. Two metres closer
it evaluates three and is a thousand florets. Nothing is authored twice, no
record is added, and the transition is the same `log2` the material's octave fade
uses. This is the unification that makes "detail scaled by camera" a property of
the *shape language* rather than a special case bolted onto it.

### 2.4 Lipschitz through the graph — the constraint that kills naive versions

The primary and every visibility cone sphere-trace `d`. Domain warp **destroys
the distance bound**: a warp of gradient magnitude `g` inflates the true distance
by up to `1+g`, and a tracer that steps the returned value overshoots and punches
holes in every rock. Retrofitting this later is a rewrite.

So the ABI carries it from day one: every op returns `(distance, lipschitz)`.
Warp multiplies `L` by `1+|∇w|`; smooth-min takes `max(L)`; the tracer steps
`d / L`. Each op declares its bound analytically. A debug mode asserts
`|∇d| ≤ L` by finite difference over a sample grid, and the CPU oracle
(`svo-implicit-reference.ts`) checks the tape's TypeScript and WGSL evaluators
agree bit-for-bit.

### 2.5 Interval pruning — what makes an 80-op tape cost like a 5-op one

A deep tape evaluated at every voxelizer sample is unaffordable, and this is the
objection that would otherwise sink the whole design. The answer is that the
voxelizer works on **bricks** (8³, `webgpu-octree-sparse-bricks.ts:918`), not on
points.

Evaluate the tape over the brick's *box* in interval arithmetic, prune every
subtree whose interval cannot reach zero, and what remains is a short specialized
tape — typically 3–8 ops for a given brick, because a boulder's fracture planes
and its floret scatters are each relevant to a small fraction of its volume. Cache
the pruned tape per brick; re-derive only when the brick refines.

This is Keeter's MPR (SIGGRAPH 2020) / libfive, it is worth 10–30×, and it is
what makes tape depth roughly free. It also composes exactly with H3: pruning is
per-brick, refinement is per-brick, and the pruned tape is the natural place to
hang the brick's octave count.

### 2.6 The three-band detail law

Once the tape exists, one law governs where each scale of it is *consumed* —
switched on the ratio of feature period to leaf size. Getting this right is what
separates 1000× detail from 1000× shimmer.

| Band | Regime | Consumer | Why |
|---|---|---|---|
| period > 2 × leaf | **Geometry.** The op is resolved as occupancy; it breaks the silhouette | voxelizer + exact upgrade | A pit you can see the edge of must be geometry |
| 0.5–2 × leaf | **Voxel normal.** Not resolved as occupancy; its gradient perturbs the stored normal | brick attribute (H4) | Below the leaf, occupancy is a lie; the normal is not |
| period < pixel | **Filtered BRDF.** The op's *variance* widens roughness (Toksvig / LEAN); its mean does nothing | material shader (H5) | The only correct answer. Sub-pixel geometry that is not converted to roughness aliases, and no amount of resolution fixes it |

Band assignment is **per-evaluation, not per-record**: the same boulder is band-1
at the near coping and band-3 at the far bank, and the voxelizer and the shader
must compute the same band from the same leaf size or detail pops when a brick
refines.

---

## 3. Work areas

Effort: S ≈ days · M ≈ 1–2 weeks · L ≈ several weeks.

### H0 — Fidelity gate (S · risk low · deps none) — **do this first**

"Pixel perfect" is unfalsifiable without a score. Build the lane before the work,
or every change after this is an opinion.

- Register the plate: solve the reference camera (the plate reads as a ~50 mm at
  a shallow downward angle; `CameraState.tanHalfFov` is authored, see
  `renderer-aperture-is-fixed`), and pin `heroGardenCamera` to it.
- Score per region — pond surface, coping, stone set, canopy, ground — with
  **ΔE₀₀** for tone/colour and a **gradient-magnitude histogram distance** for
  detail density. The second is the one that measures "1000×": it is high when
  the frame has the plate's spatial frequency content and blind to a colour shift.
- Emit a contact sheet per run. Regression on either score fails the lane.
- **Gate:** the lane runs, produces both scores on the current frame, and its
  numbers are reproducible within the noise floor across four interleaved reps.

### H1 — Light rig and grading (S–M · risk low · deps H0) — **largest delta per day**

- Author the key: one warm directional at the plate's elevation, sky fill, and a
  cool bounce from the ground plane. The cone shadow/AO path already exists and
  is tuned (`shadowConeAperture: 0.065`, `aoConeAperture: 0.62`).
- Filmic tonemap + exposure + white balance + small-radius bloom on the
  specular highlights. The plate's read is substantially its roll-off.
- **Gate:** ΔE₀₀ on the coping and ground regions halves. No geometry changed.

### H2 — The shape language (L · risk medium-high · deps H0) — **the request's core**

Implements §2. This is the largest authoring-facing change in the program and it
is what the rest of the detail work stands on.

- **`lib/svo-field-program.ts`** — the op set of §2.2, TypeScript + WGSL mirror,
  in the shape `svo-procedural-material.ts` already uses (typed policy table →
  generated WGSL constants). Each op declares its Lipschitz bound and its
  coarsest period. **Ship op 1 (domain warp) first and re-render the stone set
  before writing op 2** — it is the highest-value op by a wide margin and it will
  recalibrate everything after it.
- **`field-program` primitive kind** (code 12), arena-backed. Copy
  `SVO_CLUSTER_ARENA_BLOCK`'s shape *and* its refusal discipline. **Do not widen
  the 64 B record stride** — BVH nodes share it (`svo-primitive-candidates.ts`
  `packNodes:154`); the tape lives in the arena and the record carries a
  reference, exactly as clusters do.
- **Interval pruning (§2.5)** lands with the tape, not after it. Without it the
  voxelizer budget is blown by the first authored boulder, and a naive-first
  implementation will produce a "field programs are too slow" measurement that is
  an artifact of skipping this.
- **Four call sites must agree bit-for-bit** on the tape and its band assignment:
  the voxelizer's occupancy test, its normal tap, `svoIntersectPrimitiveExact`'s
  march, and the material shader. Extend `svo-implicit-reference.ts` into the
  oracle over all four — this is the single highest-risk correctness surface in
  the program.
- Recursive scatter (op 7) retires the canopy's `smooth-union-cluster` records.
  Measure the cluster-arena occupancy drop; it is the headroom that pays for H7.
  `smooth-union-cluster` stays as a legacy kind — do not migrate the whole set at
  once.
- **Gate:** the gradient-histogram score on the stone-set and canopy regions moves
  toward the plate by ≥ 50 %, with authored record count within ±5 % and
  cluster-arena occupancy *down*. Zero sphere-trace holes at the Lipschitz
  assertion over a full camera orbit. Pruned tape length p50 ≤ 8 ops per brick.

### H2b — The shell: walls and ground, completely redone (M · risk low · deps H2)

The largest surface in frame and the least authored one. Today it is **six
axis-aligned boxes** — `EnvironmentProxyShell`, thickness defaulting to
`finestCellSize_m` = 25 mm (`voxel-environments.ts:46–58`) — with the front face
specially opened (`svo-scene-primitives.ts:232–236`), standing on an **infinite**
terrain plane: `traceTerrain` bounds `t` only in `y`, so the ground recedes to a
fog horizon. That is the flat white expanse in the current frame, and it is
nothing like the plate.

- **Cove, not a room.** The plate is a photographic seamless: a bounded floor
  sweeping into a backdrop through a large fillet (~0.25 m) so there is no corner
  line anywhere. Author it as a `field-program` — one swept fillet, not six boxes
  — and bound the ground so the infinite plane stops being visible.
- **Give it a material with form.** Trowel relief at ~8 cm, plaster granularity
  at 1–2 mm, slight wear at the pond's foot. The shell is where H3's screen-space
  refinement is most visible and easiest to validate: it is big, near-planar, and
  its detail requirement is *purely* a function of distance.
- **Check the terrain material before blaming the renderer.** `materialModel:
  "garden-terrain"` is a *lawn* — it paints near-white daisies over
  `floor(p.xz*24)` and clover over `floor(p.xz*14)`, which read as scattered
  hard-edged axis-aligned squares. If the current frame shows axis-aligned
  banding, check this first; it is not an SVO artifact. See
  `terrain-grid-invisible-to-renderer`.
- **Sculpted terrain still does not reach the renderer.** `cloneTerrain` in
  `voxel-scene.ts` drops `TerrainDescription.grid`; the `terrainHeightfield` SVO
  kind has an `externalTerrain` flag, a `terrainReference` word and **no
  producer**. If the cove is authored as terrain rather than as a field program,
  that gap must be closed first. Authoring it as a field program sidesteps it,
  which is the recommendation.
- The shell carries most of the frame's bounce light, so **H1's warm bounce is
  only as good as this surface** — which is the argument for doing it early
  rather than filing it under set dressing.
- **Gate:** no visible corner line or horizon at the hero camera; ΔE₀₀ on the
  ground and backdrop regions halves; near-ground leaf size tracks the pixel
  under a camera dolly (this is also H3's cleanest acceptance surface).

### H3 — Screen-space leaf sizing (L · risk high · deps H2)

The hardest workstream and the one that touches the most stateful machinery.

- Replace the scene-wide `renderCellSize` with a **per-node refinement target**:
  split while projected footprint > τ px, merge below τ/hysteresis. Camera-anchored,
  frame-budgeted (reuse W2b's chunk budget), hysteretic so a slow pan does not
  thrash.
- Decouple the excursion budget from the scene-wide cell size — the coupling
  named at `svo-render-tuning.ts:124`. A swaying prop's budget becomes a function
  of *its own* node's size.
- **Node-mip pyramid page capacity is the blocker below 10 mm** and it is an
  *atlas layout* limit, not a memory one: 25 479 pages against
  `maxTextureDimension2D` = 16 384. Fix by moving the atlas to a 2D array or by
  slab-packing pages; the address plan already supports the reserved/growable
  shape (`svo-node-mip-address-plan.ts`, `maximumReserveBytes: 48 MiB`). Do
  **not** raise memory to solve this.
- Watch `node-mip-pyramid-withdrawal-cliff` throughout: a scene that renders 15×
  slow after this change is a silently withdrawn pyramid, not the primary path.
- **Gate:** near-coping leaf size ≤ 1.2 × the projected pixel; far-field
  unchanged; resident voxel bytes within 1.5× of today's 25 mm figure; no
  pyramid withdrawal on a full camera orbit; refinement converges within N frames
  of a camera stop with an authored, measured N.

### H4 — Voxel-attribute shading (M–L · risk medium · deps H3)

**This is what makes H3 affordable, and it must land with it.**

Today every solid voxel upgrades its hit to `svoIntersectPrimitiveExact` over
the owner's field — named in the predecessor as the remaining wall, and with a
detail field on top of it the cost goes *up*, not down.

- Store the resolved surface **in the brick**: oct-encoded normal + material id +
  a signed sub-cell offset (Teardown's move, and the G-buffer already has the
  encoders — `svo-gbuffer.ts:207`, `encodeSvoGBufferNormalOct8`).
- Shading reads voxel attributes. `svoIntersectPrimitiveExact` survives **only**
  inside the near-field analytic band — which is what the band is for, and it
  ships disabled today (`nearFieldBandPixels: 0`, `svo-render-tuning.ts:170`).
  H3 shrinks the band's population by construction, which is the mitigation for
  the predecessor's "band blowup on macro shots" risk.
- **Gate:** exact-upgrade invocations per frame drop by ≥ 90 %; silhouette depth
  p99 on a crown crop stays inside the threshold that failed at W3 (97.7 mm at
  ×10) — this is the check that says the band is now doing its job.

### H5 — Filtered materials (M · risk low · deps H2)

- Thread the pixel cone width into the material evaluation (the shading path
  change the tree already names) and fade octaves out at Nyquist.
- Convert the faded octaves' *variance* into roughness (Toksvig/LEAN) rather than
  dropping them. Without this, detail does not just alias — it gets *brighter*
  as it recedes, because a rough surface averaged as a smooth one over-reflects.
- Revisit `plaster`'s `colorAmplitude: 0`. It is zero for a stated and good reason
  ("a cloud you can see is the thing that reads as dirt" on a white set), but that
  reasoning was against an *unfiltered* field. With filtering, a small amplitude
  at a period above the pixel is available again — and the plate's coping does
  have tonal variation.
- **Gate:** a slow camera orbit shows no temporal shimmer above an authored
  threshold on the stone set — measured, as per-pixel luminance variance across
  consecutive frames, not eyeballed.

### H6 — Water (L · risk medium-high · deps H1)

The other half of the frame, and a genuinely separate program. Roughly:

- Fluid on for the hero (`water: true`), with the pond at the plate's fill.
- A water **surface** in the SVO, not a coverage fraction — `svo-fluid-coverage.ts`
  is explicit that its volume is a shadow input and cannot serve as the interface.
- Dielectric refraction from the surface into the voxel scene: the machinery
  exists (`svo-media.ts` `traceSvoMediaRay`, `evaluateSvoDielectricTransition`;
  `svo-thick-glass.ts`). What is missing is the water surface as a boundary
  source and the depth-driven absorption tint that makes the plate's pool read
  teal at depth and clear at the rim.
- **Caustics**: absent from the SVO path entirely; present in the legacy
  `lib/webgpu-water-pipeline.ts`. Port or rebuild — the plate's bowl is a caustic
  net and it is one of the most recognisable things in the image.
- Jet, ripple rings from the impact point, and spray/foam
  (`webgpu-secondary-particles.ts`).
- **Gate:** ΔE₀₀ and gradient score on the pond region, which is ~55 % of the plate.

### H7 — Set dressing (M · risk low · deps H2)

Stepping-stone discs, mushroom-capped stones, the coral canopy rebuilt on
`cell-scatter`, the hose jet. Cheap once H2 exists, and it is what closes the
last ~10 %. Deliberately last: authoring against a renderer that cannot yet show
detail wastes the authoring.

---

## 4. Frame budget

| Subsystem | Today (1×) | After this program | How |
|---|---|---|---|
| Octree primary + voxelized SDF | 3.0 ms | ≤ 6 ms | H3 raises resident pages; H4 removes the per-voxel exact upgrade |
| Near-field analytic band | 346 ms (all records) | ≤ 3 ms | H4 confines exact to the band; H3 shrinks its population |
| Cone visibility + GI | 45 ms | ≤ 6 ms | unchanged plan (quarter rate + pyramid); H3 improves what cones *see* for free |
| Water (new) | 0 | ≤ 6 ms | H6 |
| Deferred shading + grading | 11 ms | ≤ 4 ms | resolution-bound; H5 adds filtering, H1 adds bloom |
| **Frame** | **405 ms** | **≤ 25 ms** | |

25 ms, not 16.7. The plate is a substantially harder image than the frame the
predecessor budgeted, and a 40 fps hero that looks like the plate is worth more
than a 60 fps hero that does not. Revisit once H4's measurement lands.

**Never quote a render-pass GPU timestamp on this machine** — they bracket
[vertex start, fragment end] on a tiler that hoists vertex stages. Compute passes
only. This produced one confident and entirely wrong conclusion already.

---

## 5. Do not revisit

- **Uniform `cellSize` reduction** as the route to fine detail (§1). 5 mm is
  already refused by the pyramid; 1 mm is 1.3 G cells over a domain that is 96 %
  air, and it still cannot produce a floret.
- **Noise on an ellipsoid.** The form is the problem (§2). A displaced sphere is
  a bumpy sphere; the plate has fracture facets, bedding anisotropy and
  self-similar scatter, none of which are perturbations of a convex solid. Ops
  1–4 before ops 5–8, always.
- **More sphere lattices.** `smooth-union-cluster` at one blend radius has one
  look, and the current frame is it. Adding octaves to it does not escape it.
- **A field-program tape without interval pruning** (§2.5). It will measure slow,
  the measurement will be an artifact of the omission, and the conclusion drawn
  from it will be wrong.
- **Authoring detail as records.** The cluster arena, the material table and the
  BVH all bind first, and `smooth-union-cluster` already costs one record per
  feature. This is the mistake recursive scatter exists to prevent.
- **Widening the 64 B record stride** for detail parameters — sidecar buffers.
- **Tightening proxy boxes**, **hardening the direct-fragment raster**, and
  **sparse/virtual-texture page indirection** — all refuted, see the
  predecessor's §7.
- **Any single-run A/B below ~10 ms at 10×.** The lane's within-arm sd is
  6.7–8.4 ms on a ~145 ms frame with ~9 ms of batch-to-batch drift. Two runs
  differing by 3 % have measured nothing.

## 6. Sequencing

```
H0 ──┬── H1 ─────────────────────────────┐
     │                                   ├── H7
     ├── H2 ──┬── H2b ───────────────────┤
     │        ├── H3 ── H4 ──────────────┤
     │        └── H5 ────────────────────┤
     └────────────  H6 ──────────────────┘
                (parallel, own program)
```

**H0 → H1 → H2** is the critical path to a frame that reads like the plate.

**Inside H2, ship the domain warp first and re-render the stone set before
writing anything else.** It is the highest-value op by a wide margin, it is one
op, and seeing it will recalibrate the priority of everything after it.

**H3 + H4 must land together** — H3 without H4 makes the exact-upgrade wall worse
in exact proportion to the refinement it buys.

**H2b is early, not late.** The shell is ~20 % of the frame's pixels, carries most
of its bounce light, and is the cleanest possible acceptance surface for H3's
screen-space refinement.

**H6 runs in parallel** by a different pair of hands if there are any; it shares
only the light rig with H1.
