# Paddleboats to 60 fps: fewer fine tiles and particles, and a frame that fits the M1 Max (2026-11-10)

**Status: plan for review, revised after an adversarial review (§5). Nothing
below is implemented.**

Scene: `lib/core/paddleboats-scene.ts`. A 12.8 × 0.8 × 4.8 m pool, h = 2.5 cm,
water 24 cells deep, two hulls with eight paddle blades each. The app shows
about 10 fps. The target is 60.

## 1. What was measured today

HEAD c7c26dd1 plus the working tree. `tools/probe-reservoir-bulk.ts
--scene=paddleboats --bodies --pipelined` (hulls moored, wheels driven) and
`tools/probe-census-reasons.ts`. One run each.

Caveat: the 300-step timing run and the 300-step reasons run each logged one
Metal command-buffer abort (`ImpactingInteractivity`, a watchdog, not an
out-of-memory), so the runner marked them invalid (rc 77). Their numbers agree
with the 2026-11-03 status table, and I use them as indicative. The 24-step
and 12-step runs were clean.

| | step 12 | step 300 |
|---|---|---|
| step time, kernel timer off | ~32 ms | 52–53 ms |
| h tiles (4³ h cells each) | 1,624 | 3,526 |
| census reason "solid" | 1,536 (996 active) | 754 (299 active) |
| particles | ~0 | 312k (172k hot, 140k collar; 226k deeper than one cell) |
| pressure band tiles | 844 | 1,788 |
| regions COLD / WARM / HOT | 1208 / 240 / 88 | 591 / 810 / 135 |
| dispatches a step | — | 491 (417 direct, 74 indirect), 296 distinct kernels |

Census reasons at step 300 (first rule wins, and join outranks the others):

| reason | tiles | split |
|---|---|---|
| join | 2,491 | 1,740 hold particles; 751 are one tile from a particle tile, and 603 of those are dry |
| solid | 754 | |
| travel + closure | 172 | |
| crossing | 24 | |

Other counts at step 300:
- The shape criterion requires 759 tiles. Particle tiles number 2,196: 704
  triggered, 409 cooling, and 1,083 with zero heat (collar and carried).
- 6,647 tiles cross the free surface, so about 85% of the surface is already
  at 4h.

Kernel time by stage family (kernel timer on, so distorted; the differences
are the useful part):

| stage family | step ~12 | step ~290 | change |
|---|---|---|---|
| pressure (arena solve, rows, both projects, authority, rhs, transfers) | ~9 | ~11.5 | +2.5 |
| particles (move, transfer, reconstruction, seed, sort) | 2.3 | 10.6 | **+8.3** |
| level set (surface advect, arena metric, resolve, fill) | ~5 | ~9 | +4 |
| extension | 1.8 | 3.1 | +1.3 |
| census, layout, remap, frame plan, solid record | ~5 | ~7 | +2 |
| kernels under 0.1 ms each | 6.8 (215 kernels) | 5.4 (193 kernels) | |

No stage dominates. The largest single kernel is the particle move at 5.5 ms.

### What the items cost

My first draft fitted `17 ms + 8.7 µs × h tiles` through the two columns
above. The review showed that fit is confounded: WARM regions, band tiles and
particles all rose between the two points. The better evidence is from
interventions on this scene:

- **h tiles: about 3 µs each.** 56b738d7 removed 1,046 h tiles and about
  1,000 band tiles for 49.1 → 46.3 ms
  (`small-scene-step-2026-11-03/progress-census.md`).
- **Particles: about 27 µs per thousand.** That is the +8.3 ms of the particle
  family above over 312k particles, and it matches the move's 17–29 ns a
  particle in lane T's notes.

So of today's 52 ms, tiles carry roughly 10 ms, particles roughly 8 ms, and
**about 34 ms does not depend on either**. That base is the frame's length:
about 490 serial launches, each latency-bound at 9–17% occupancy and costing
tens of microseconds however little it has to do
(`brick-resident-step-2026-11-08.md` §1).

### The app is slower than the solver, and nobody has measured why

The headless step is 52 ms, which would be 19 fps. The app shows 10. No tool
attributes the other ~48 ms: `tools/benchmark-nbflip-ui.ts` leaves out the
renderer and the bodies. Known contributors, none of them sized:

- **CPU contacts every step.** Paddleboats has motor attachments and no
  interaction entities, so `enableNativeMotorContacts` is on
  (`webgpu-uniform-reference.ts:935`). Each step reads the rigid state back,
  runs Box3D on the CPU, writes each body back and submits again
  (`webgpu-rigid-body.ts:1088–1118`). The host holds one frame in flight
  (`mixedHostBusy`, `:3020`) and the draw waits on a queue fence
  (`:3036–3042`). This breaks the all-GPU rule whatever it costs.
- **Free hulls and a longer run.** The probe moors the hulls and stops at
  step 300. An older capture averaged 7.5–9.5k h tiles and 422–610k particles
  over steps 300–600 (`one-pass-m1max-2026-11-02.md` §8).
- **Host encode.** About 17 µs a dispatch in Node
  (`activity-step-2026-11-05/progress-stages.md`), so about 8 ms for 491
  dispatches; unmeasured in Chrome.
- **The renderer**, sharing the same GPU queue.

## 2. What 60 fps needs

A 16.7 ms frame leaves the solver about 10 ms once the renderer has its share.

| state | step |
|---|---|
| today, headless | 52 ms |
| h tiles 3,526 → ~2,000 and particles 312k → ~150k | ~43 ms |
| h tiles → ~1,000 and particles → ~100k (the shape criterion's own demand) | ~39 ms |

**Cutting tiles and particles is worth about 10 ms. It does not get this
scene near 60 fps.** The frame's 490 launches do, and the pressure solve alone
is about 160 of them. A 10 ms step needs roughly 120 launches in total at
today's cost per launch. That is the same conclusion the brick-resident plan
reached, and its fused passes are the only design on file that addresses it.

## 3. Plan

### Phase 0: attribute the app frame (measurement only)

1. Capture the app on paddleboats with free hulls, at the moment it shows
   10 fps: h tiles, particles, dispatches, solver GPU time, contact readback
   and CPU time, fence wait, host encode time in Chrome, render time, rAF
   period.
2. Headless A/B with native contacts on and off, free hulls, 600 steps.
3. Find the long command buffer behind today's two watchdog aborts.

Gate: a table that accounts for the app's 100 ms. Phases 1–3 are sized from
it, and their order may change.

### Phase 1: contacts on the GPU

4. Replace the per-step Box3D step with GPU contact for motor assemblies.
   The GPU rigid kernel cannot do this today: its pair test is sphere against
   sphere, it returns early for motor fixtures (`webgpu-rigid-body.ts:513–520`)
   and it skips contact iterations in native mode (`:629`). Box3D was added
   (defdf461) to give hull and wheel envelopes with travel-bounded stepping.
   The smallest all-GPU route:
   - `solidVoxelContact` already takes a support function (`:305–360`); make
     the support the maximum over the hull and axle envelopes;
   - add one box-against-box test for hull against hull.
5. Delete the readback, `deferredFramePublication` and the presentation fence
   for this case.

Regressions: `tests/paddleboat-contacts.test.ts` unchanged (wheel-first
impacts at 40 m/s, basin edits); a hull against an edited basin wall, as
AGENTS.md requires; save, load and undo of bodies.

### Phase 2: particles (312k → about 150k, worth about 4 ms)

Each item is its own interleaved A/B, gated on letters and channel-dam
volume as well as paddleboats.

6. **The cold collar.** 1,083 of 2,196 particle tiles have zero heat and 140k
   of 312k particles are collar. The collar is transfer support for the hot
   core ("its samples stop at an unseeded tile",
   `uniform-narrow-band-activity.wgsl.ts:107`). It spreads to all 26
   neighbours, including straight up into dry tiles, which is where the 603
   dry join tiles come from. Test dropping the spread into tiles with no
   liquid, and seeding collar tiles at the deep density (4 a cell) throughout.
7. **Band depth 3 → 2.5 cells.** Depth 2 was tested once, on the channel dam:
   +3.7% volume against a +4.35% base (`bulk16-plan-2026-10-28.md`), which is
   not a failure. Re-test on letters.
8. **Shape tolerance.** The 35–50% particle saving at 1 h predates the
   smoothed shape measure, and the remaining triggers have a median of about
   2 h, so expect little. One run to confirm, then close it.

### Phase 3: h tiles (3,526 → about 2,500, worth about 3 ms)

9. **Split the body tiles by cause.** Step 12 has 1,536 solid tiles of which
   996 are active; step 300 has 754 and 299. Whether the difference left or
   was relabelled "join" is not known. Extend `probe-census-reasons.ts` to
   report per body: tiles cut by the shape, tiles marked only by the motion
   bound, ring tiles, and wet, dry or inside the solid.
10. **Act only on what item 9 shows.** The obvious tightenings are small or
    already done: `markTiles`' slack is at most 1.46 h, the ring was trimmed
    in 56b738d7, and wet-only rings are recorded as unsafe
    (`uniform-mixed-dynamic.ts:2046`).
11. Tiles follow particles: every collar tile removed in phase 2 is an h tile
    removed here.

### Phase 4: the frame's length (the 34 ms)

This is the main line for frame rate, and the only one with a factor of three
in it.

12. **Static and body solid record split** (about 2.8 ms, open since
    2026-11-03): the basin's record is built once and only body-cut tiles
    refresh.
13. **Pressure launches.** About 160 a step, 38 a cycle, with three coarse
    tail cycles inside each. The pool is 6 coarse cells deep. Count launches
    by level on this scene and remove the levels and tail cycles that do no
    work here; the fused tail and the Galerkin operator are already on file
    as rejected and parked.
14. **Fused brick passes for the non-pressure frame**: continue
    `brick-resident-step-2026-11-08.md`. S2, S3 and B4 landed; the fused
    transport pass was accepted as a design and not landed, blocked on a
    640 s Metal compile. Paddleboats becomes its acceptance scene beside
    drop=2: launches a step, and the step at 12 and at 300.
15. **Re-measure after each of 12–14** with the step-12 and step-300 pair.

Gates for every phase: types, unit tests, seep 300 and letters 120 drift and
step time, paddleboats 300 with tiles, particles, band tiles, launches and
drift.

## 4. Decision for the user

Phases 0–3 are worth landing and take the solver from 52 to roughly 40 ms.
60 fps on this scene at h = 2.5 cm depends on phase 4 item 14, which has
already had one hard week. The alternatives that reach 60 fps sooner change
the scene rather than the solver:

- h = 3.5 cm halves the surface tiles, but the blades become 1.4 cells thick;
- simulating at 30 Hz and presenting at 60 doubles dt and the travel margins.

I recommend phases 0–1 now, phases 2–3 in parallel as cheap A/Bs, and a
decision on item 14 once phase 0 says how much of the app's 100 ms is the
solver at all.

## 5. Adversarial review (independent, read-only) and what changed

| # | Objection | Verdict | Change made |
|---|---|---|---|
| 1 | The 8.7 µs/tile fit is confounded; interventions on this scene give about 2.7 µs/tile, so tile and particle cuts reach ~43 ms, not ~32 | Draft was wrong | §1 and §2 rewritten on intervention slopes; launch count is now the main line |
| 2 | "Move contacts to the GPU" is a new contact solver: the GPU pair test is sphere–sphere and skips motor fixtures | Draft was wrong on scope | Phase 1 rescoped to a compound support function plus one box–box test, with the edit regression |
| 3 | The app's missing 48 ms was pinned on the contact path with no measurement; the step is GPU-bound, so two frames in flight will not shorten it | Unproven | Phase 0 widened; the claim that phase 1 restores frame rate is withdrawn |
| 4 | "1,536 → 600 body tiles" was mostly already taken; ring-by-wet-face contradicts a recorded failure | Draft was wrong | Target and the ring item dropped; the split-by-cause probe kept |
| 5 | Moving the rest surface off the tile face fails its own arithmetic (a 3.9 h band in a 4 h tile), and the dry tiles are the collar's vertical spread | Draft was wrong | Item dropped; the dry tiles are now attacked through the collar |
| 6 | Band depth 2 had been tested; tolerance 1 h is unlikely to release today's triggers; a face-only collar leaves edge tiles unseeded | Unproven | Items restated with those results; the face-only collar replaced by a no-liquid test |
| 7 | "Stop repeated extension and resolve, classify replay, layout" re-proposes measured-and-rejected work | Draft was wrong in part | Removed; the static/body record split kept; phase 4 now connects to the brick-resident work |

Still unverified:
- whether the headless probe with host-posed bodies takes the Box3D path;
- why step 12 holds 1,536 solid tiles;
- the pressure level structure on a pool 6 coarse cells deep;
- anything about Chrome's encode cost or the renderer on this scene.
