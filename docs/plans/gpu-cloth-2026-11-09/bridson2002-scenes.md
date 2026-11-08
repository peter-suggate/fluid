# Bridson, Fedkiw and Anderson 2002: scene provenance

This document covers the scenes of R. Bridson, R. Fedkiw and J. Anderson, *Robust Treatment of Collisions, Contact and Friction for Cloth Animation*, SIGGRAPH 2002, rebuilt for the GPU cloth.

- **Code:** `lib/cloth/scenes/bridson-2002/`. `parameters.ts` is the single table of numbers, and each number carries its source.
- **Tests:** `tests/cloth-bridson-2002.test.ts`.
- **Lane:** S. Progress is logged in `progress-lane-s.md`.

**Status: authored and tested as data. Nothing is "reproduced" yet.** No GPU run exists, because the solver cannot yet do any of these:
- contact against voxel solids;
- moving voxel solids;
- run a cloth document that a scene holds.

The comparison table at the end is empty until captures exist.

## Sources

All downloads are kept in `~/code/me/fluid-scratch/bridson2002/`, outside git.

| Tag | Source | Where |
|---|---|---|
| paper | SIGGRAPH 2002 PDF: UBC copy, plus Stanford's 2.2 MB lossless copy for the figures | `pdf/cloth2002.pdf`, `web/cloth.pdf`, `figs-lossless/fig-000..010.png` |
| paper | Project page with three named stills: `test0_frame114`, `test1_frame053`, `test2_frame056` | https://graphics.stanford.edu/papers/cloth-sig02/, `web/` |
| video | The 2002 result videos `cloth0.avi` (curtain, 160 frames), `cloth1.avi` (sphere, 213 frames) and `cloth2.avi` (table, 93 frames). 1280×960, MS-MPEG4v2, 15 fps playback | Wayback: `web.archive.org/web/2011id_/http://physbam.stanford.edu/~fedkiw/animations/clothN.avi`; frames in `video/frames_clothN/fNNN.png` |
| video | Bridson's copies of the same runs, 640×480 | https://www.cs.ubc.ca/~rbridson/movies/ (`rb_curtain.mov`, `rb_spinningcloth.mov`, `rb_tablecloth.mov`) |
| thesis | R. Bridson, PhD thesis, Stanford 2003. Ch. 1 covers the cloth model; ch. 2 covers collisions and includes Figs 2.1–2.3, the same three scenes. §2.1: the objects in Figs 2.1 and 2.3 are level sets, i.e. sampled signed distance, as our voxel solids are | https://www.cs.ubc.ca/~rbridson/docs/rbridson_phd.pdf |
| physbam | PhysBAM `Projects/solids_3d/Standard_Tests/STANDARD_TESTS.h`, tests 5 ("Cloth Curtain and Ball"), 12 (table) and 13 ("Cloth and spinning sphere"), plus `DEFORMABLES_STANDARD_TESTS.cpp` (`Create_Cloth_Panel`) and `RIGIDS_STANDARD_TESTS.cpp` (`Add_Ground`). This is 2006–2008 descendant code by the same group, so it is later tuning and ranks below the paper and the video | GitHub `byrantwithyou/PhysBAM` |

The models PhysBAM loads (`skinnyhexlink` and `cut_pyramid`) are not public.

## Timing: 24 fps simulation frames

The videos play at 15 fps, but each frame is a 24 fps simulation frame:
- In PhysBAM test 5 the ball is at rest until t = 2 s. The video's ball starts moving at f52–56. At 24 fps, 2 s is f48; the gap is the lag in detecting the first motion.
- The ball comes back to the pile at 4.8 s, which is f115. The video shows f114, and Stanford names its still `test0_frame114`.
- PhysBAM's default `frame_rate` is 24.

| Scene | Video length |
|---|---|
| Curtain | 160 frames = 6.667 s |
| Sphere | 213 frames = 8.875 s |
| Table | 93 frames = 3.875 s |

Our scenes take one 1/60 s cloth step per frame and never substep. Video frame f falls at scene time `videoStart + f/24`.

## Shared cloth model and the mapping to VBD/AVBD

**The paper's model** is a mass-spring sheet (§3):
- structural springs between neighbours;
- diagonal shear springs;
- bending springs between every other node, each with a stabilisation spring to the middle node.

The paper adds two limits:
- a strain limit, "a triangle edge should not change length by more than 10%", with biphasic springs that are "much stiffer beyond 10%" (§4);
- a strain-rate limit (§4).

**Our mapping.** We don't implement springs; each spring term maps onto the new solver:
- **Structural and diagonal springs** become an orthotropic StVK membrane, made isotropic here.
- **The soft-then-stiff biphasic behaviour** becomes a soft membrane plus the AVBD hard per-edge limit `strainLimit` = 0.1, |e| ≤ 1.1 L. It applies to every mesh edge, diagonals included, as the paper limits every spring.
- **Strain-rate limiting** has no separate term. The paper's form, "no more than 10% change per step", is an explicit-step stabiliser. The implicit 1/60 s step does not need it, and the hard strain limit bounds the same quantity.
- **Every-other-node bending springs** have no linear bending modulus: their length change is quadratic in the hinge angle. They become a discrete-shell bending modulus, chosen as an assumption (below).

**Mesh.** We use PhysBAM's `Initialize_Herring_Bone_Mesh`: each cell's diagonal flips with its column only, so every interior vertex has six triangles. The repo's `clothGridData` flips the diagonal per cell instead, so interior vertices get four or eight triangles. Under area-lumped mass that is a 2:1 mass checkerboard.

**Mass.** The paper says (§3): "We make the edges and corners of the cloth slightly heavier by giving all particles the same mass instead of a mass proportional to the area of the surrounding cloth." The thesis, ch. 1, footnote 1, adds the hem argument. PhysBAM `Create_Cloth_Panel` sets `mass_node = aspect·side²/(m·n)`, i.e. 1 kg/m² shared equally.

- **Today's solver** lumps mass by area: ρ·A/3 per incident face. With the herring-bone mesh the interior masses are equal, but hem nodes get half the interior mass and corner nodes 1/6 or 1/3 of it, the opposite of the paper.
- **The scene carries `nodeMass_kg`** until C1 accepts a per-vertex mass (COORDINATION in `progress-lane-s.md`).

| Parameter | Value | Source | Quote, method or reason |
|---|---|---|---|
| Nodes across | 150 | paper | §9: "a piece of cloth with 150 × 150 nodes". §9 also says "tens of thousands [of nodes] in the dynamics" |
| Long side | int(aspect·149)+1 | physbam | `Create_Cloth_Panel`: `m=(int)(aspect_ratio*number_side_panels)+1`, which gives square cells. PhysBAM's own test uses 40 panels; we use the paper's resolution |
| Thickness | 1 mm | paper | §6: "Our cloth is given a realistic thickness, e.g. 1mm" |
| Strain limit | 0.1 | paper | §4: "a triangle edge should not change length by more than 10%" |
| Areal density | 1 kg/m² | physbam | `mass_node = aspect*sqr(side_length)/(m*n)` |
| Equal node mass | ρ·W·D/(m·n) | paper | §3, quoted above |
| Stretch (warp = weft) | 200 N/m | assumption | The paper gives no constant. PhysBAM's edge springs, 10/(1+√2) N per unit strain at 40 panels/m, come to ~170–220 N/m. This is soft below the 10% limit, as §4 asks: under gravity the hanging curtain strains about 2.5% |
| Shear | 100 N/m | assumption | For an isotropic lattice with equal diagonal springs, G = E/2 at Poisson 0 |
| Poisson | 0 | assumption | Mass-spring sheets have no designed Poisson coupling |
| Compression ratio | 1 | paper | The linear springs of §3 resist compression the same as extension. The thesis's later "compression to 0%" belongs to the 2003 model, not to these 2002 runs |
| Bending | 1.5e-4 N·m | assumption | Skip springs have no linear modulus, so the value is chosen for a bending length c = (B/ρg)^⅓ = 2.5 cm. **Check it against the Fig 1 fold spacing** |
| Damping (Rayleigh) | 0.002 s | assumption | The paper gives none. PhysBAM uses "overdamping fraction" 15. This value critically damps edge modes near 1000 rad/s |
| Friction model | one μ, static and kinetic | paper, thesis | §7.3; thesis §2.1: "This formula includes both the static and kinetic friction cases". The paper lists different μ_s/μ_k as future work (§10) |
| Self friction μ | 0.3 | assumption | The paper gives no value. PhysBAM later uses 3.2 with its repulsion clamping; we reject that as solver-specific tuning |
| Iterations K | 64 | assumption | This is the solver budget per step, set by the M1/M2 gates, not by the paper |
| Step | 1/60 s, one per frame | lane rule | The paper steps adaptively (§5); we never substep |

## Fig 1: curtain (PhysBAM test 5)

**Caption versus §9.**
- The caption: "The ball moves, flipping the curtain over on top of itself … Another ball enters the scene and pushes through the complicated structure of folds before slipping underneath unraveling the curtain."
- §9: "When the sphere moves up and away, the curtain flips back over on itself."

**Resolution: one sphere plays both roles.** PhysBAM test 5 drives a single kinematic sphere:
- at rest until 2 s;
- up and away along (1, 0.5, 0) m/s until 3.5 s;
- straight down behind the flipped curtain until 4 s;
- back along the ground in −x at 1.5 m/s.

That return is the caption's "another ball". The video agrees: only one ball is ever visible, and it leaves the frame before re-entering from +x at f104–116.

**Video sequence** (`video/frames_cloth0`):

| Frames | What happens |
|---|---|
| f0 | Flat sheet at the pins' height |
| f8–12 | Falls over the dome and the ground |
| f52–80 | The ball lifts the cloth up and away |
| f84–96 | The curtain flips back onto itself, orange (back) side up |
| f104–116 | The ball re-enters from +x |
| f120–150 | The ball pushes under the pile and carries it over |
| f159 | Unravelled on the −x side, back side up |

| Parameter | Value | Source | Quote, method or reason |
|---|---|---|---|
| Sheet | 1.7 × 1 m, 254 × 150 nodes | physbam, video | Test 5 `side_length=1`, `aspect_ratio=1.7`. In Fig 1d the run on the ground is ≈ 2.5× the hanging height, which fits 1.7 and not a square |
| Start | flat at y = 0.5 m, x ∈ [0, 1.7], z ∈ [0, 1] | physbam | `Create_Cloth_Panel` places the sheet at y = 0.5 |
| Pins | the two x = 0 corners, fixed | physbam, video | Test 5 constrains the two corners. In the video the top edge sags 0.19–0.23 of its width, a two-point catenary at about 10% stretch |
| Ground | voxel slab, top at y = 0, μ = 0.3 | physbam | `Add_Ground()` default friction 0.3 |
| Ball | voxel sphere r = 0.25 m, μ = 0 | physbam | `Add_Rigid_Body("sphere",.25,0)`, kinematic |
| Ball start | centre (0, 0, 0.5), a dome under the pinned edge | physbam | `frame.t.z=.5` |
| Ball script | t < 2: (0, 0, 0.5) | physbam | `Set_Kinematic_Positions` test 5 |
| | 2 ≤ t < 3.5: (t−2, (t−2)/2, 0.5) | physbam | same |
| | 3.5 ≤ t < 4: (1.5, 0.75 − 1.5(t−3.5), 0.5) | physbam | same |
| | t ≥ 4: (1.5 − 1.5(t−4), 0, 0.5) | physbam | Crosses x = 0 at 5 s and reaches −2.5 m at 6.67 s |
| Duration | 6.667 s | video | 160 frames at 24 fps |

## Fig 2: rotating sphere (PhysBAM test 13)

| Parameter | Value | Source | Quote, method or reason |
|---|---|---|---|
| Sheet | 1.9 m square, 150 × 150 nodes, centred at (0, 0.56, 0) | physbam, video | `Create_Cloth_Panel(n,1.9*side_length,…,FRAME(TV(0,.56,0)))`, starting 1 cm above the sphere top. Video f0 shows a flat sheet at the sphere top |
| Ground | μ = 0.1 | physbam | `Add_Ground((T).1)` |
| Sphere | r = 0.25 m, centre (0, 0.30, 0), μ = 1 | physbam | `Add_Rigid_Body("sphere",.25,1)`, `frame.t=TV(0,.30,0)`. It floats 5 cm clear of the ground |
| Spin | −π/4 rad/s about +y, from t = 0.75 s | physbam, video | `twist.angular=TV(0,-pi/4,0)` for t > 0.75. In the video the skirt flaps travel round the vertical |
| Spin pose | θ = −π/4·(t − 0.75) | assumption | PhysBAM sets `From_Rotation_Vector(time*ω)`, which jumps the pose at 0.75 s. That jump is invisible on a sphere, so we keep the pose continuous with the same velocity |
| Frustum | square, base 0.40 m, top 0.20 m, height 0.24 m (±25%) | video | The black frustum in `cloth1.avi` and Fig 2; the text never mentions it. Measured from perspective against the 0.5 m sphere |
| Frustum place | base on the ground at (−0.65, 0, 0.65), axis-aligned | physbam, assumption | `cut_pyramid` at (−0.65, 0.05, 0.65). The model is unpublished, so its base on the ground and its axis alignment are assumptions |
| Frustum base | on the ground, y = 0 | video | It sits on the ground as in the paper; its base face coincides with the ground's top face. The 67° base corner is above the 35° edge C2's 1 cm contact field holds at any lattice phase, so the field build accepts it (`tests/cloth-bridson-2002.test.ts`) |
| Frustum μ | 0.1 | physbam | `Add_Rigid_Body("cut_pyramid",.1,.1)` |
| Duration | 8.875 s | video | 213 frames at 24 fps. PhysBAM's own `last_frame` is 20 s |

## Fig 3: tablecloth

PhysBAM test 12 is a later and different table: a 1 m cloth on hex-link legs, with a 1 m/s sphere. So the geometry here is the 2002 video's.

**Camera reconstruction** (`cloth2.avi`):
- horizon at y ≈ 420 px of 960, and focal length f ≈ 798 px from the equal leg heights;
- the table is turned 21.8° to the camera.

The reconstruction gives ratios only. We scale it by one anchor, PhysBAM's sphere radius of 0.15 m, which gives a realistic 0.75 m table.

The 2002 run started from an already draped cloth:
- Video f0 shows the settled drape.
- Stanford's `test2_frame056` matches video f56. It is Fig 3b, cloth dragged down between the legs.

We must drape first, so the scene runs a 2 s pre-roll and video f0 falls at t = 2 s.

| Parameter | Value | Source | Quote, method or reason |
|---|---|---|---|
| Scale anchor | sphere r = 0.15 m | physbam | Test 12 `Add_Rigid_Body("sphere",.15,.5)` |
| Legs | 4 square posts, 0.11 × 0.11 × 0.75 m | video | Sphere diameter = 0.40 leg heights; height/width ≈ 6.7 |
| Leg layout | square, centres ±0.24 m | video | Centre spacing is 0.65 leg heights (4.3 leg widths) |
| Leg tops | 45° bevel, 22 mm | video, thesis | Tops are bevelled ≈ 0.2 leg widths. Thesis §2.1: it "may not be possible to actually round off the tip (as was actually done in figure 2.2)". The paper itself stresses "the sharp corners of the legs" |
| Leg μ | 0 | physbam | Test 12 `skinnyhexlink` friction 0 |
| Cloth | 1.59 m square, 150 × 150 nodes, flat at y = 0.76 | video, assumption | Side ≈ 2.1 leg heights. Diagonal corner overhang 0.71 m, which reaches the 0.75 m ground within the 10% limit, as at f0. The 1 cm gap copies test 13 |
| Ground μ | 0.3 | physbam | Test 12 `Add_Ground()` |
| Sphere μ | 0.5 | physbam | Test 12 |
| Pre-roll | 2 s | assumption | Drapes and settles the 0.55 m overhang before video f0 |
| Descent | y = 2.10 − 0.82 (t − 2) m | video | Highlight tracked at a constant 9.6 px/frame from f27 to f42 (`scripts/track_sphere.py`). PhysBAM test 12 uses 1.0 m/s |
| Landing | t − 2 = 2.378 s (f57) | video | Contact with the ground at f57 |
| Exit | +x at 0.62 m/s, between two legs, to the end | video | Pure translation is an assumption; rolling is not measurable at 640 px |
| Duration | 2 + 3.875 s | video | 93 frames at 24 fps |

## Voxel solids

Every solid is a voxel solid; there are no analytic colliders. Each shape (box, sphere, square frustum, bevelled post) is an implicit function used only to bake the shared 8³-brick, 9³-node field:
- **Static solids** go through `bakeProceduralSolid`, so they sit on the shared hit-test and edit path.
- **Moving solids** are body-local fields with a `RigidPoseTrack` (`lib/core/prescribed-rigid-motion.ts`), sampled in absolute scene seconds.
- **Faceted shapes** use the largest plane distance. It is exact inside, 1-Lipschitz, and has the exact zero set.

| Feature | Size | Cells at 1 cm |
|---|---|---|
| Leg bevel | 22 mm | 2.2, the smallest |
| Leg width | 110 mm | 11 |
| Frustum top | 200 mm | 20 |
| Sphere radii | 150–250 mm | 15–25 |
| Ground slab | 80 mm | 8, one brick |

**Cell: 1 cm.**
- Authoring refuses a feature narrower than two cells (`assertBridsonSolidResolved`). At a 3 cm cell the leg is refused, and the test checks this.
- The test also measures the worst sign disagreement between the stored trilinear field and the shape:
  - bevelled post: 2.5 mm, where its sharp vertical edges round inside one cell;
  - frustum: 1.35 mm;
  - spheres: 0.1 mm.
  - All are below half a cell.

**Cloth contact field** (C2's `buildClothSolidField`, M5 part 1). `bridsonClothSolidSource(solids, t, box, cell, band)` gives the union of a scene's solids at scene time t. Built at 1 cm with a 6 cm band, CPU:

| Scene | Bricks | Size | Time |
|---|---|---|---|
| Curtain at t = 0 | 1852 | 5.5 MB | 6 s |
| Sphere at t = 0 | 3062 | 9.1 MB | 25 s |
| Table at t = 0 | 1944 | 5.8 MB | 40 s |

All three pass its thin-feature check. Almost all of the time goes on sampling the check at h/8 near the surface.

**Bake cost.**
- The four legs bake in 1.4 s.
- The whole table, including a 3.8 × 2.8 m ground slab, takes 45 s on the CPU, almost all of it on the ground's partial bricks. At registration the ground should therefore be the scene's own voxel floor (y = 0). The slab stays in the data for the cloth contact field, which builds bricks from node values alone.

## Fig 4: robe — reference only

"Frames from a production animation of a robe draped over a digital character." The 2002 PDF prints "Not available in this version". The character, the garment and the "highly sophisticated model from an industrial production system" were never published. **Nothing stands in for this figure.**

## Faithfulness gaps

1. **Equal masses** are not yet expressible. The solver lumps mass by area, which makes the hems lighter where the paper makes them heavier. This is a C1 request.
2. **Friction:**
   - The contact config has one μ (`OgcContactConfig.mu`). The scenes need a μ per solid (0, 0.1, 0.3, 0.5 and 1) and a separate self-contact μ. This is a C2 request.
   - The paper's single μ for static and kinetic friction matches OGC's Coulomb model.
3. **Bending, damping and stretch** are assumptions, because the paper gives no constants. They are to be judged against the fold scale in Figs 1–3, not tuned per scene.
4. **Strain-rate limiting** is not modelled separately (see the mapping).
5. **Fig 3 geometry** comes from a single-camera reconstruction, with ±10% on ratios. The 2 s pre-roll and the pure-translation exit are assumptions.
6. **Fig 2 frustum:** the size is ±25% and the axis alignment is assumed.
7. **Collision handling differs by design.** The paper uses repulsions, impact zones and rigid impact zones; ours uses OGC offset geometry with Planar-DAT truncation. Matching is judged on visible behaviour: stable folds, no snagging on the leg corners, unravelling.

## Comparison record

| Scene | Moment | Reference | Capture | Verdict |
|---|---|---|---|---|
| — | — | — | none yet | Blocked on three things:<br>• M5's static field wired into C1's solver (C2's next gate);<br>• moving voxel solids (M6);<br>• cloth documents running in scenes (C1). |

**Ready as the first captures once M5 runs through C1's solver.** Each scene starts with a static stretch, so these moments need only static solids:
- **Curtain, t ≤ 2 s:** the ball holds still. Compare the drape at f12 and at f48 (Fig 1a).
- **Sphere, t ≤ 0.75 s:** before the spin. Compare at f12.
- **Table, until about 3.5 s:** the sphere is still far above. Compare the settled drape at video f0 (t = 2 s) with `video/full_t00.png`.
