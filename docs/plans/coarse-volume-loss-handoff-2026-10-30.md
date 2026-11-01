# Handoff: 4h free-surface volume loss (O6), 2026-10-30

Follow-up: [O6 investigation and abandon decision](coarse-volume-loss-investigation-2026-10-30.md).
The measured local alternatives did not satisfy the goal; no production fix
was adopted. That report corrects the slab-width, default-advection and
particle-presence assumptions below and records the reproducible regressions.

Goal: find out why a free surface held at 4h loses liquid, at rest and in
motion, and fix it to the papers. This is part of finishing BULK16/M4b. The
coarse start, refused slot claims and Coarse regions all leave free surfaces
at 4h, and none of them is usable while 4h liquid evaporates.

- **In scope:** the 4h tier's phi transport, its smoothing, redistance,
  reconstruction and classification at a free surface; the tests that pin
  it; the cross-scene gates.
- **Out of scope:** new or changed scenes and presets (the user's direction
  is to finish the bulk work, with no scene changes); flipping the coarse
  start to default; the slot accessor's +1.2 ms; M5.
- **Owner:** Codex. The BULK16/M4b agent is committing fix (a), free flight
  for detached 4h mass (it touches `uniform-mixed-pressure-authority.ts` and
  `uniform-mixed-detached-mass.wgsl.ts`); start from that commit. Work in the
  main checkout, with no stash, reset, checkout or rebase. Run GPU jobs only
  through `tools/gpu/fluid-gpu-run.sh`, one task-owned simulation per slot,
  and don't touch processes you didn't start.

## Evidence (measured by the agent, 2026-10-30)

All runs use the default h start with a Coarse region (min = max = 4 cells)
over the whole box, so every tile is 4h, with no solids and no h tiles.
V is the represented volume in h cells.

| case | size in 4h cells | result | rows? |
|---|---|---|---|
| drop r 0.2 m at a vertex | radius 1 | V 132 → 3 over 10 steps; 140 against the sphere's 268 at construction | none (free flight after (a)) |
| drop r 0.3 m | radius 1.5 | 710 → 248 cells over 12 steps (−65%); before (a): 710, 701, 686, 565 over steps 1–4 | yes |
| drop r 0.5 m | radius 2.5 | −20% in 12 steps; centre of mass falls 0.195 m (free fall gives 0.196 m) | yes |
| slab 1.2 m wide × 0.3 m deep, collapsing | 6 wide × 1.5 deep | about −50% in 30 steps | yes |
| drop r 0.5 m falling onto a pool (todo merge test) | radius 2.5 | 3847 → 2621 cells by contact at step 19; +316 cells at contact; free-fall increment 0.143 against g·dt 0.163 at step 2 | yes |
| film 0.08 m at rest on the floor | 0.4 thick | −3.9% in 60 steps, at speed 0, identical with and without (a) | none |

- **Motion is right and volume is wrong.** The r 0.5 drop's centre of mass
  tracks free fall to 0.5%.
- **Two regimes, which may have different causes:**
  - drops of 1–2.5 coarse cells radius are below what any grid level set
    holds (see Fix directions);
  - correction (2026-10-30): the slab is only 1.5 owners deep, so it is
    under-resolved too and does not on its own prove a defect. The
    convergence step (4) decides it. The r 0.5 drop (2.5 owners) is the
    best-resolved moving case; its free-fall increment is also low
    (0.143 against 0.163).
- **Loss at rest with zero velocity** rules out advection for the film.
  What remains is the non-advective passes: smoothing, redistance and the
  reconstruction or volume metric.

## Where it reaches, and why it matters

- **On the default path:** a 4h free surface exists only where a slot claim
  was refused at capacity, or under an authored Coarse region. Measured
  reach, counting 4h owners with 0.001 < V < 0.999
  (`artifacts/bulk16/m4b/o6/`):
  - **Seep 1200:** 960 owners at every sample. This is the left reservoir's
    resting surface, the same set throughout, and seep drift stays at
    +0.11%.
  - **Letters 240:** 2048 owners at frame 20, then 1772, 741 and 121 at
    frames 40, 60 and 80 as the impacts refine it. 0 from frame 100 on,
    except 52 at frame 180.
  So a large, flat, resting 4h surface conserves on the default path. The
  loss belongs to moving and under-resolved 4h surfaces, and any fix must
  keep the seep's resting 4h surface unchanged.
- **For the bulk work:**
  - The coarse start is M4b's memory win (1.74 GB on the seep). It needs a
    4h surface it can trust until the census refines it.
  - Refused claims are a silent degradation. BULK16's own design rule is
    "a promote with no free block is an explicit fatal, never a silent
    refusal". A refused claim that leaves a moving surface at 4h, where it
    evaporates, breaks that rule.

## Code map

- **Phi transport:**
  - `lib/methods/uniform/uniform-mixed-surface.ts`, class `UniformMixedSurface`
    (line 62).
  - umVertexFoot (line 919): the foot on the zero set, the gradient taps,
    and the horizontal fourth-difference smoothing.
  - umMovedSmoothing (line 997).
  - The cubic or trilinear choice is umCubicAt (around line 1366).
- **Smoothing weight:** `UNIFORM_MIXED_VERTEX_SMOOTHING_PER_S = 0.5`
  (line 32), so eps = 0.5·dt per step, capped at 1/32.
  - The fourth difference is in `uniform-surface-smoothing.wgsl.ts`.
  - The comment at 900–913 says an alternating pattern loses 16·eps a step
    (about 13% at dt 1/60). A drop one coarse cell across *is* that
    pattern.
- **4h advection mode:** `uniform-mixed-frame.ts:894–900` (surfaceFlags[1]).
  - The coarse default is cubic; `phiCoarseAdvection: "trilinear"` changes
    it.
  - umCubicAt's comment assumes "the census keeps a moving surface at h
    where 4h would round its corners or curves (umCurveError) … a 4h
    surface is flat or gently curved, which the trilinear carries". That
    assumption is false under a Coarse region or a refused claim.
  - Check which mode the failing runs used.
- **Redistance:** `uniform-narrow-band-redistance.wgsl.ts`.
  - "Fixed-band coarse values remain advected surface samples."
  - Adaptive coarse vertices beside retiring h owners recover their metric
    magnitudes, which can move a hanging zero.
- **Classification:**
  - `uniform-mixed-pressure-authority.ts` umSurfaceOwner, where the
    centre phi is the mean of the 8 corners.
  - The detached test is `uniform-mixed-detached-mass.wgsl.ts`, with (a)'s
    changes.
- **Volume metric and V:** `uniform-mixed-surface-volume.ts`. Check whether
  V at 4h is the same quantity the census and the gates count
  (volumeCellSum).
- **Curve refinement:** `uniform-mixed-dynamic.ts` umCurveError (line 730)
  and the shape-error promote (around line 1087). This is what normally
  keeps curved moving surfaces off 4h.

## Investigation plan (short, focused runs; test first)

1. **Pin it.** Add `tests/uniform-coarse-surface-volume.test.ts`, using the
   all-4h Coarse region set-up from `tests/uniform-coarse-free-fall.test.ts`.
   - Cases: the 6-cell slab, collapsing, 30 steps; a resolved r 0.5 drop,
     12 steps; the film at rest, 60 steps.
   - Assert volume drift per case. The slab is the case that must pass.
     Small drops get a bound from the convergence step (step 4), not 0.
2. **Split rest from motion.** Rerun the film, and the slab with gravity off
   and at rest, toggling one pass at a time:
   - smoothing eps = 0;
   - redistance off (`redistance: false` in the frame parameters);
   - cubic against trilinear at 4h (`phiCoarseAdvection`).
   Report the volume drift with each pass off. A pass that loses volume at
   zero velocity is a defect, whatever the resolution.
3. **Localise in motion.** Measure V before and after each surface pass in
   one step of the collapsing slab: advect, deferred advect, smoothing,
   redistance, reconstruct. Name the pass and the cells where volume
   leaves.
4. **Convergence.** Run the same slab and drop at h (no Coarse region) and
   at 4h. Grid level-set loss should fall with resolution.
   - If 4h loses more than about 4× what h loses for the same physical
     feature, the defect is 4h-specific (seams, hanging vertices, corner
     loads, the smoothing width), not plain level-set diffusion.
   - Report the ratio.
5. **Construction.** The r 0.2 drop starts at 140 of 268 cells. Confirm
   that this is the restriction of a one-cell feature (expected) and not a
   corner-load error like 4d1c5704's.

## Fix directions (grounded; decide after steps 2–4)

- **A defect in a pass:** smoothing on curved 4h features, redistance moving
  the zero, or a 4h-specific seam or corner error. Fix that pass to its own
  stated model. Example: if the fourth-difference damper eats curved 4h
  features, its comment says it should leave "a cubic alone", so test that
  claim at 4h widths. Each fix needs a measured case and a citation, the
  same as the tangential fix.
- **The resolution limit is the papers' answer, not a bug.**
  - Enright et al. 2002 (particle level set): a grid level set loses volume
    in under-resolved regions, and that is why particles exist.
  - Ferstl et al. 2016 (NB-FLIP): the free surface lives in a particle band
    at the finest resolution, and the grid level set is interior only.
  - Losasso, Gibou and Fedkiw 2004 (octree water): the interface is refined
    to the finest level.
  - Under these papers, a moving free surface at 4h with no particles is
    outside the method. The consistent fix is the invariant: **a moving
    free surface lives at h.** It has two parts, each needing a ruling:
    - **A refused claim on a moving free-surface tile.** Make it an explicit
      fatal, the same policy as pool exhaustion, with capacity by rule.
      Alternatively, give such tiles priority above region heat in the
      claim. Memory stays proportional to surface area, which is
      fluid-sized.
    - **A Coarse region over a moving surface.** Either Coarse regions
      govern only below the band, or they keep 4h and record the loss as
      that setting's documented cost. The Coarse region is the user's
      authoring choice, so bring it to Claude with the numbers; don't
      decide it alone.
- **Ruled out:**
  - volume correction or injection, clamps, damping, or reseeding to
    restore V;
  - per-scene exemptions, and scene changes;
  - a mechanism that helps 4h and moves seep, letters or rest.
  The fixed-dt, no-substep and allocate-upfront rules stand.

## Gates and reporting

- **Tests:**
  - the new volume test passes;
  - the free-fall and film tests stay green;
  - the `todo` merge and film-volume tests point at O6 and turn green
    if the fix covers them;
  - types 0 errors; unit tests 0 failures.
- **Cross-scene, in every report:**
  - seep 1200: worst drift ≤ 3%, and the right tank grows every 150-step
    window;
  - letters at 240: ≤ 3%;
  - the rest ladder at 1, 1.5, 2 and 2.5 m: drift 0.
- **Timing:** seep pipelined 3+3, no worse than the pre-fix tree beyond
  1.66× the spread.
- **GPU sweep:** against the known-failure list.
- **Memory:** construction bytes unchanged, or explained by rule.
- **Each report states:** the losing pass and its file:line, the before and
  after table above, and the convergence ratio.

## State of the tree when this was written

- **Committed this session:**
  - 9668db55: the slotless h readback;
  - 78e61625: the wall-list stub;
  - dd9217ce: the bulk-promote-image helper.
- **336ed429:** fix (a), free flight for detached 4h mass, with the
  floor-support exclusion along gravity's dominant axis.
  - `tests/uniform-coarse-free-fall.test.ts` has 2 pass and 3 todo, all
    pointing at O6: the film's volume, the r 0.2 merge and the r 0.5 merge.
  - Gates: seep +0.11% with the tank growing every window, letters
    −1.53% at 240, rest 0, timing within noise.
- **A draft O6 pinning test from the BULK16 agent** (not in the tree):
  `/private/tmp/claude-501/-Users-petersuggate-code-me-fluid/b858ab5d-4376-4cd9-9167-ad36a08a9e8e/scratchpad/o6/uniform-coarse-surface-volume.test.ts`.
- **The default-flip patch** (coarse start as default) is in the agent's
  scratchpad. It stays out of the tree until O6, seed rule (A) and the
  slot-accessor cost are resolved.
- **Seed rule (A) is ruled and not started.** The t=0 layout is frame 1's
  own census, run on the initial h fields, at rule capacities. It follows
  O6.
- **Other open items:**
  - O4: the dam and channel volume gain;
  - O5: the pool-rest flow growth;
  - the slot-accessor +1.2 ms (dense h at 36.4 ms against compacted h at
    37.6 ms on the seep);
  - the renderer's 2M vertex cap.
- **Plan of record:** `docs/plans/bulk16-plan-2026-10-28.md`.
