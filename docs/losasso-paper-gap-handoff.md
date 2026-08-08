# Losasso 2004 paper-gap audit — mini dam realism + rigid coupling handoff

**Date:** 2026-08-08
**State:** measured + code-forensic diagnosis at HEAD `bf3c360`; no fixes applied.
Every file:line below was verified against this tree by direct read; dynamic
claims come from the probe/smoke runs recorded in §2.
**Symptoms (user-reported, all reproduced):**

- S1 — the dam front runs up the far wall and sticks to the ceiling;
- S2 — the liquid loses its momentum far too early and settles into a
  persistently bumpy surface that never levels;
- S3 — rigid-body coupling in the settled tank
  (`interactive-water-box-settled`) is visibly wrong.

**Lane:** coarse-band Losasso (`coarseBackend=losasso`,
`globalFineLevelSetFactor=1`), scene `minimal-power-dam-break-32`
(0.8 m³ closed box, 25 mm cells, fill 23/64, dt 4 ms, free-slip,
closed top; profile `lib/scenes.ts:92` = leaf 32 / band 3 / grading 3).

**Two constraints that shape this whole document:**

1. **These defects are original to the lane.** All the load-bearing
   mechanisms below exist verbatim at `e839807` (the commit that shipped the
   coarse-band path): the transport wall clamp, the sign-frozen redistance,
   the zero-width lid hysteresis, the volume corrector, the missing particle
   correction. The four post-handoff commits (`7972eb4`..`bf3c360`) changed
   amplitudes, not mechanisms — treat them as background, not causes.
2. **The target configuration is surface band 1.** The shipped AUTO default
   on this scene resolves to a protection width that keeps the lattice
   near-uniformly fine around the water (width = `max(1,band) +
   (grading−1)·max(2,size)` → 7/11/19/35/67 cells for sizes 2/4/8/16/32,
   `lib/octree-runtime-dials.ts:383-392`), so the coarse-mixture defects of
   `docs/losasso-band-invariance-handoff.md` are largely dormant at defaults —
   and fully live at band 1. Every fix and every gate below must be evaluated
   at `FLUID_SURFACE_BAND=1`, not at AUTO.

---

## 1. TL;DR — the five load-bearing gaps

| # | Gap vs the paper | Symptom |
|---|---|---|
| G1 | Level-set transport and redistance have **no closed-wall extension**: the backtrace clamps to the boundary cell and re-samples the film's own phi; redistance skips out-of-domain neighbours and can never flip sign. A film on the lid (or any wall) is self-sustaining. | S1, S2-bumps at walls |
| G2 | **No particle level set** and first-order MAC-trilinear phi transport; the global volume corrector is the *only* volume authority and publishes a step offset **after** redistance. | S2 (volume, features), bumps |
| G3 | The unilateral separation active set has a **zero-width hysteresis band exactly at the lid** and no approach-velocity guard; the ghost probe can never see air above a wet lid face, so the lagged `FACE_SEPARATED` bit is the only opener. | S1 |
| G4 | Rigid scenes: the per-substep boundary refresh **re-seals every separated wall face** (the ghost re-run repair exists only on the fine lane; factor 1 has no fine bricks); buoyancy is analytic against a **frozen authored waterline**; drag is computed against **still water**; phi is never carved inside bodies at factor 1. | S3 |
| G5 | At band 1 the coarse-mixture machinery is the algorithm: the residual RC1 zero-velocity paths, the W=7 support shell freeze, and the size-quantized wetness/theta from `docs/losasso-band-invariance-handoff.md` are on the critical path. | S2 at band 1 |

Measured exonerations (§2, §5): the volume corrector is **not** the cause of
S1 or of the rest-state corrugation (A/B with `FLUID_COARSE_VOLUME_CONTROL=0`
changed neither), and the pressure discretization remains verified-correct.

---

## 2. Measured evidence (all runs this session, reproducible)

Instrument: `tools/probe-dam-surface-shape.ts` (Dawn, no renderer), plus one
`tools/run-webgpu-smoke.ts` run with `FLUID_SPEED_MAP=1`.

```
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
FLUID_SCENE=minimal-power-dam-break-32 FLUID_MAX_DT=0.004 \
FLUID_MAXIMUM_LEAF_SIZE=32 FLUID_OCTREE_INTERFACE_BAND=3 \
FLUID_SAMPLE_TIMES_S=0.5,1.0,...,6.0 FLUID_SURFACE_ASCII=1 \
node --import tsx tools/probe-dam-surface-shape.ts
```

**Timeline (band 3, grading 1, corrector on):**

- 0.1–0.3 s: collapse proceeds normally (median height 27.5 → 12.5 cells).
- 0.4–0.5 s: front reflects at x=31 and climbs to the **lid** (max height
  32.5 by 0.5 s); the middle of the tank is bone dry.
- 0.5–1.2 s: the far corner (31,31) stays welded to the lid.
- 1.6 s: lid clears briefly; 2.0–2.5 s: lid wet again, ~40% of the floor dry,
  a corner pillar standing to the ceiling.
- 2.5 s: max speed **rebounds 0.75 → 3.01 m/s** — the welded pillar finally
  releases and free-falls (~0.5 m drop ≈ 3.1 m/s; the release *is* the spike).
- 3–6 s: weak residual slosh (0.2–0.65 m/s), surface a static ±1–2-cell
  corrugation plus isolated multi-cell towers/pits at walls that appear and
  vanish between samples.
- Pressure: `mgpcg-nonconvergence` tripwire ×9 starting step 408 (t≈1.6 s).
- Receipt: `frozenCells` 10–14k of 32768 every advance (the D3 counter,
  `lib/webgpu-octree-coarse-summary.ts:755`, `state[31]`).

**Corrector A/B (`FLUID_COARSE_VOLUME_CONTROL=0`):** ceiling sticking
unchanged; rest corrugation unchanged (neighbor step ~1.0 both arms); the
off-arm reveals the underlying transport loss — volume drifts 10743 → 9587
of 11600 target by 6 s (~10%; ~1.7%/s). The corrector is *necessary* today
(G2) and *innocent* of S1/S2-corrugation. Its cost is where it puts the
compensation, not whether it runs.

**Band 1 (the target configuration):** see §2.1 below.

Physical read: energy drains 3.62 → 0.75 m/s in 1.5 s (far faster than 32³
first-order SL alone should), except the mass nonphysically parked at the
lid, which re-injects on release; sloshing that should persist for tens of
seconds in an inviscid box is effectively dead by ~4 s; the rest state keeps
frozen bumps because nothing in the lane can level them (§4.2).

### 2.1 Band-1 arm (the target configuration)

One-command repro: add `FLUID_SURFACE_BAND=1` to the probe line above.
Measured this session (corrector on):

- The violent phase is indistinguishable from band 3: same lid weld
  0.5–2.0 s, same overshoot, surface rows still size 1 while the flow is
  active (the moving front drags the fine shell with it).
- The band-1-specific failure appears **at rest**: by t = 6 s, 327 of 1024
  surface columns publish from **size-2 rows**, the neighbor step jumps
  from ~1.0 back to **4.6 cells**, minimum height drops to 6.5 (4-cell
  pits), and volume dips to 11491/11600. The ladder coarsens into the
  resting surface and the surface *roughens* — the S2-e/f + G5 machinery
  measured directly. A settled tank should be the easy case; at band 1 it
  is where the coarse-mixture defects live.
- `frozenCells` is flat (~14k) across the dial — at 32³ the W=7 shell
  covers most of the air anyway; expect this counter to move on larger
  domains, not here.

---

## 3. Paper audit — §-by-§ against `docs/papers/losasso-2004-octree-water-smoke.txt`

Verdicts: OK = faithful or a defensible supersession; GAP = missing/divergent
and implicated in a symptom; N/A = not exercised by these scenes.

| Paper mechanism | Ours | Verdict |
|---|---|---|
| §3 nodal averaging "coarsest neighboring face as the scale"; T-junction constraints | Staged nodal lattice with per-quadrant validity; since `7972eb4` a dyadic-bracket fallback (`stagedCoarsenedVelocity`, `lib/webgpu-octree-losasso-extension-band.wgsl.ts:128-163`) makes it near-total inside coarse leaves, but it takes the *first* resolving span (≤ maximumLeafSize=32) rather than the containing leaf, and air/high-wall corners still fail to `previous=0` (`losasso-dynamics.wgsl.ts:302-303`) | **PARTIAL** — the RC1 residue; bites at band 1 |
| §3 refinement/coarsening transfer is averaging, never zero | Coarsening exact; refinement mid-planes fixed in `7972eb4` (`velocity-migration.wgsl.ts:32-50,92-93`); remaining hazard: silent no-write inherits an unrelated face's slot across epoch compaction (`:85`, `:94`) — unobservable, D5 census never added | **PARTIAL** |
| §4 semi-Lagrangian velocity advection, trilinear nodal | Faces advected via staged nodal field; bounded MacCormack on coarse faces (velocity only) | OK |
| §4.1/4.2 symmetric Poisson discretization, PCG | Verified correct previously and unchanged; flux-conservative T-junction faces | **OK — do not touch** |
| §4 "20 iterations to machine precision" | `mgpcg-nonconvergence` ×9 during the violent phase of this 32³ scene | **GAP** (diagnose under load; may be BC-induced by G1/G3 pathologies rather than solver-intrinsic) |
| §6 particle level set (Enright 2002), particles RK2 | **Absent.** `secondaryParticles` is render-side spray and returns `undefined` on this lane (`lib/webgpu-uniform-eulerian.ts:1614`). The volume corrector self-describes as the "particle-level-set correction analogue" (`lib/webgpu-octree-coarse-summary.ts:769-774`) | **GAP (G2)** |
| §6 phi advected semi-Lagrangian with **nodal** velocities | RK2 midpoint trace (better than paper) but sampling **MAC staggered trilinear**, not nodal (`coarse-summary.ts:756-765` → `losassoVelocityAtGrid`); the staged nodal lattice exists and is unconsumed; no CFL clamp on the backtrace anywhere (dt is authored); first-order sample, no limiter | **PARTIAL** |
| §6 fast marching maintaining signed distance | Honest FMM-from-infinity seed + first-order Eikonal upwind sweeps (`coarse-summary.ts:786-811`) — but **sign is frozen** (`:793`, `:811`: magnitude-only, sign carried verbatim from the advected field), and out-of-domain neighbours are **skipped, not virtual air** (`:787`, `:798`) | **GAP (G1b)** |
| §6 velocity extrapolation into air (Strain 2000) | Fixed-Jacobi over hard-coded W=7 shell, 8 sweeps pinned; outside the band the 8-corner sample fails whole and phi **freezes** (observable: `frozenCells`) | **PARTIAL** — fine inside the band, frozen outside; the shell radius, not the CFL, sets the front's max speed |
| §6 refinement band about interface "focusing more heavily on the water side" | Symmetric `|phi|` distance test (`webgpu-octree.ts:8983`); no water-side bias; the Ando-Batty sizing bit is produced only by the fine transport kernel and **never fires at factor 1** | **PARTIAL** — half the refined shell is spent on air |
| §6 surface tension Dirichlet `pI = pair + σκ` | σ uploaded to `params.physical.y` (`webgpu-octree.ts:4188-4193`) and **never read by any octree shader**. The settled tank authors σ=0.072 and simulates σ=0 | **GAP** (silent divergence; mini dam authors σ=0 so N/A for S1/S2) |
| §5/§6 objects: Neumann on faces intersecting the object | Superseded by a *better* cut-cell aperture blend (`losasso-dynamics.wgsl.ts:387-390`, `losasso-projection.ts:38-43`) — legitimate | OK |
| §5 "clip the velocity component normal to the object so that it is guaranteed to be separating" | **Absent for solids** — only the closed-wall clip exists (`losasso-projection.ts:44-47`) | **GAP (G4)** |

---

## 4. Root causes per symptom, ranked

### 4.1 S1 — ceiling (and wall) sticking

The fine/structured lane fixed exactly this in July (see memory
`wall-sticking-mechanisms`): three compounding mechanisms. **The coarse lane
has analogues of none of the two transport-side fixes and a defective
analogue of the third.** All original to `e839807`.

- **S1-a (primary): transport treats every closed wall as an infinite liquid
  reservoir.** `predictSummaryCells` → `densePhiAt` → `centeredAxisSample`
  (`lib/webgpu-octree-coarse-summary.ts:756-762`, `:581`) clamps the
  departure point to the boundary cell centre and copies phi verbatim. A
  receding lid film's backtrace exits the domain, gets clamped, and
  re-samples the film's own negative phi forever. The fine lane's fix —
  clamp to the sample lattice *and add the exit distance* (unit outward
  slope, `finishSample`, `lib/webgpu-octree-fine-levelset-transport.wgsl.ts:633`)
  — has no coarse analogue.
- **S1-b: redistance cannot represent a surface at a wall.** Seed and sweep
  both `continue` past out-of-domain neighbours
  (`coarse-summary.ts:786-793`, `:794-811`); there is no virtual air sample
  at `center + h` (fine-lane fix: `seedClosestPointCode`,
  `lib/webgpu-octree-fine-levelset-redistance.ts:1107`). Combined with the
  sign-freeze (`:811`), redistance has **no path at all** to dry a lid film:
  a film flush against the lid has no in-lattice sign change and its sign
  can never flip. S1-a and S1-b are self-reinforcing.
- **S1-c: the separation active set has zero hysteresis width at the lid.**
  `losasso-projection.ts:50-68`: for overhead faces
  `releasePressure == contactPressure` and `renewalPressure =
  max(releasePressure, 1e-4·contactPressure)` — identical, so there is no
  band; and the `opening` predicate has **no approach-velocity guard** (the
  only velocity term clamps returning flux on already-separated faces).
  Additionally the ghost probe steps *inward*
  (`losasso-coarse-phi.wgsl.ts:121`), so `sampledAir` is dead at a wet lid
  face and the lagged `FACE_SEPARATED` bit is the only opener — a face can
  only open using pressure information from a solve in which it was fully
  closed.
- Velocity extension near the lid is clean (band includes wall faces,
  non-band faces re-zeroed each sweep — no frozen-impact-velocity pump on
  this lane). Dead code note: `closedDomainFace`
  (`losasso-dynamics.wgsl.ts:142-147`) is never called.

Measured signature: corner (31,31) welded 0.5–1.2 s and 2.0–2.5 s; release
at ~2.5 s re-injects 3 m/s into a nearly-settled tank. The lateral-wall
towers/pits in the rest state are the same S1-a/S1-b mechanism on the other
five walls (free-slip walls hold films laterally; the corrector-off arm
shows them more clearly).

### 4.2 S2 — early momentum death + surface that never levels

Momentum, in causal order:

- **S2-a: nonphysical potential-energy parking.** S1 holds mass at the lid
  and walls for seconds; what should be one clean reflection becomes
  weld-hold-release cycles that scramble coherent slosh into grid-scale
  noise the projection then dissipates. (The 2.5 s speed rebound is this.)
- **S2-b: first-order transport losses.** ~1.7%/s volume loss measured with
  the corrector off; the same trilinear resample diffuses momentum. No
  particle correction exists to restore interface sharpness (G2), and phi
  transport has no limiter. At 25–50 mm cells this is the paper's own
  scheme at 1/32 of the paper's resolution — but the paper's answer to
  exactly this is the particle level set it names as essential.
- **S2-c: MGPCG non-convergence ×9** during the violent phase: each
  non-converged projection leaves divergence the next advection converts to
  noise. Diagnose *after* G1/G3 (the welded-BC states it is being asked to
  solve are themselves pathological).
- **S2-d (band 1): the coarse-mixture machinery.** At band 1 the residual
  RC1 zero-paths (§3 row 1), the W=7 support shell (front speed capped at
  shell-growth rate; phi frozen outside — `frozenCells` is the counter),
  and size-quantized wetness put the degenerate scheme inside the active
  flow. This is `docs/losasso-band-invariance-handoff.md` §RC1/RC5/RC3;
  its WP-B1 (total nodal reconstruction via the containing *cell*) remains
  the structural fix — the `7972eb4` bracket fallback is not it.

Why the surface never levels (the "incredibly bumpy" rest state):

- **S2-e: nothing can flatten a frozen bump.** At rest, velocities are
  ~zero, so transport moves phi by ~nothing; redistance rebuilds magnitude
  but **cannot flip sign** (`:811`) — a one-cell tower of "liquid" stays
  liquid; there is no surface tension (σ never read); and there are no
  particles to delete escaped blobs. The only leveling force is hydrostatic
  pressure through the solve, which acts on velocity — and the velocities
  it creates are below the level where transport visibly moves the
  interface before the next projection re-zeroes them. Every mechanism that
  should erode a bump is either absent (σ, particles), disabled by
  construction (sign-freeze), or quantized (S2-f).
- **S2-f: solver wetness/theta quantization.** Wetness for size-1/2 rows
  uses the containing 4³ B4-node's **min phi**
  (`webgpu-octree.ts:8774`, `:9709-9716`) — over-wet by up to 2 cells — and
  theta still comes from the **row-centre** phi
  (`losasso-coarse-phi.wgsl.ts:125,136-137`); a newly-wet row whose centre
  phi ≥ 0 falls through to the staged open p=0 face
  (`:142`, the RC6 fallthrough, still open and now documented as intended).
  Sub-cell surface differences are thus invisible or wrongly-Dirichlet to
  the solve: adjacent columns differing by one cell of height can produce
  no leveling gradient. This is measurable (D-bump below).
- **Exonerated by A/B:** the volume corrector. Rest corrugation and ceiling
  sticking are identical with it off. (Its count/apply set mismatch —
  count gate `|phi|<h` at `coarse-summary.ts:820-822` vs apply gate `≤2h`
  + 1-ring at `:777`,`:841` — is real and worth fixing for hygiene, but it
  is not the driver of S2.)

### 4.3 S3 — rigid coupling in the settled tank

Scene facts: `interactive-water-box-settled` has **no methodProfile** → runs
the shipped defaults → factor 1, 50 mm cells, σ authored 0.072 (simulated 0),
closed top, two bodies (cork sphere 3.6 cells across; dense box tilted 26°).

Ranked (post-R1/R4/X5 — the zero-diagonal poisoning and face-area fractions
ARE fixed at HEAD):

- **S3-a: the per-substep rigid boundary refresh re-seals separated wall
  faces — in every rigid scene.** `conditionLosassoFaces` under refresh
  forces `openFraction=0`/`normalVelocity=0` on every `FACE_CLOSED_BOUNDARY`
  face (`losasso-backend.wgsl.ts:497-499`); the ghost pass that legitimately
  opened separated wall faces (`flags=3`) is not re-run because
  `encodeGhostRefresh` requires `currentFine` — undefined at factor 1
  (`webgpu-octree.ts:5172-5177`). Net: the moment you add a body, all six
  walls become permanent pressure anchors, re-welded every substep. This
  couples S3 back into S1/S2: a rigid scene gets a strictly worse version
  of the wall pathology.
- **S3-b: buoyancy/drag are decoupled from the actual water.** Buoyancy is
  analytic Archimedes against a **frozen authored waterline**
  (`hydrostaticReferenceY_m = fillFraction·height_m`, set once,
  `webgpu-octree.ts:2865-2866`; `immersedVolumeAtReference`,
  `rigid-pressure-reaction.ts:99-115`) — waves, displacement, and craters
  are invisible to it; the tilted box's immersion is a vertical-prism
  estimate that ignores orientation. Occupancy words 7–9 are **never
  written**, so `meanVelocity ≡ 0` (`webgpu-rigid-body.ts:232`) and drag
  damps the body toward *world-stationary* — a cork on a wave is
  continuously braked. A floating body therefore bobs on a spring to an
  authored plane while being dragged to a halt: precisely "not right" in a
  settled tank.
- **S3-c: phi is never carved inside bodies at factor 1** (carve constructed
  only under `fineA && fineB`, `webgpu-octree.ts:2986-2996`), and the coarse
  volume target is latched at the first advance and never body-adjusted
  (`coarse-summary.ts:826-827`). The level set keeps a liquid ghost inside
  the body; the corrector balances the books by eroding the retreating
  front — craters around a sinking body, and the render draws the ghost.
- **S3-d: no separating clip for solid faces** (§3 last row) — bites on
  body-wall overlap and pose jumps; secondary here.

---

## 5. Exonerated — do not spend time

- **Pressure discretization / T-junction fluxes** — verified correct
  previously; nothing here touches it.
- **The volume corrector as the cause of S1/S2** — A/B measured null. Fix
  its set mismatch for hygiene (WP-5), don't expect symptom relief.
- **Velocity extension staleness at the lid** — band membership and
  re-zeroing verified clean; there is no frozen-velocity pump on this lane
  (unlike the historic fine-lane bug).
- **Zero-diagonal publication poisoning (A1)** — fixed at HEAD, belt and
  braces; the settled-tank symptom is S3-a/b/c, not this.
- **The recent four commits as root cause** — the mechanisms predate them
  (verified at `e839807`). Note their hazards for later
  (`docs/losasso-band-invariance-handoff.md` §9 of the audit: bracket
  smoother, migration silent-carry, corrector set mismatch) but they are
  not why the dam sticks or dies.

## 6. Discriminating experiments (cheap, ordered)

- **D-lid (validates the S1 fix as it lands):** port the free-fall oracle
  pattern to this lane — `ceiling-slab-drop`/`corner-brick-drop` scenes
  already exist; run them with the coarse profile
  (`FLUID_METHOD=octree` + `coarseBackend=losasso`, factor 1) and gate on
  "ceiling wet cells = 0 by t=0.2". Today they run the structured lane; the
  port is an overrides change, not new scenes.
- **D-bump (nails S2-e/f):** settled 32³ tank, author a single one-cell
  surface bump; assert it decays. Instrument: probe already reports
  `maximumNeighborStepCells`. Read the solve: if adjacent columns with
  Δh = 1 cell produce identical pressure rows (quantized wet/theta), S2-f
  is confirmed independently of transport.
- **D-front (band 1):** probe `profileX` front position vs the Ritter
  solution across `FLUID_SURFACE_BAND=1..4`; divergence that *tracks the
  dial* is S2-d; divergence common to all dials is S2-a/b.
- **D-freeze:** `readReceipt().frozenCells` across the same sweep — the
  W-shell freeze signature is a step increase as the band shrinks.
- **D-rigid-still:** settled tank + one static submerged sphere (O1 scene
  exists: `rigid-hydrostatic`): assert fluid max speed stays ~0 over 2 s.
  Today S3-a should fail this within one substep — it is the cleanest
  possible repro of the re-seal.
- **D-mgpcg:** re-run the 6 s smoke after WP-1/WP-2 land; if the 9
  non-convergences vanish with the welded BCs, S2-c was induced, not
  intrinsic.

## 7. Work packages (ordered; each has a gate)

**WP-1 — Closed-wall contract for coarse phi (fixes S1-a/S1-b; the port of
the fine lane's two wall fixes).**
(a) In `predictSummaryCells`' sampler path, clamp the departure point to the
sample lattice and **add the clamped distance** to the sampled phi (unit
outward slope), exactly as `finishSample` does on the fine lane; openTop
exempt. (b) In `seedDenseRedistance`/`sweepDenseRedistance`, treat
out-of-domain neighbours across closed walls as virtual samples at
`center + h`. Gate: D-lid green (ceiling wet cells 0 by t=0.2 on the coarse
profile); the rest-state wall towers/pits in §2 disappear.

**WP-2 — Lid separation hysteresis + approach guard (S1-c).**
Give overhead faces a real hysteresis band (renewal while
`p < 0.25·contactPressure` after opening at `p < contactPressure`, mirroring
the fine lane's complementarity fix), and block opening while the liquid-side
normal velocity approaches the wall. Keep the existing cross-epoch
`FACE_SEPARATED` persistence. Gate: corner-brick-drop (coarse profile) clears
the lid on the free-fall parabola; no weld-hold-release speed spikes in the
6 s mini-dam smoke.

**WP-3 — Sign-mutable interface maintenance (S2-e).**
Let the redistance flip sign where the Eikonal solution contradicts the
carried sign within the interface band (or equivalently: seed sign from the
crossing topology, not verbatim), so isolated one-cell towers/blobs can dry.
This is the coarse-lane stand-in for the paper's escaped-particle deletion —
scope it to |phi| ≤ 1 cell to avoid eating real features. Gate: D-bump decays;
volume drift unchanged (corrector still owns global volume).

**WP-4 — Sub-cell hydrostatics at the surface (S2-f).**
(a) Take theta for the ghost Dirichlet from the face's own interpolated phi
crossing rather than the row-centre value; (b) close the RC6 fallthrough for
rows that are wet by interval but centre-dry (they must get a conditioned
interior face, not a staged p=0 face); (c) drop the containing-B4 min to the
row's own extent for size-1/2 wetness. Gate: D-bump shows a nonzero leveling
gradient for Δh = 1 cell; `flags=4` census ~0 below the bulk surface at rest.

**WP-5 — Volume authority hygiene (S2-b support; NOT a symptom fix).**
(a) Match the corrector's count/apply sets (`abs<h` guard at the apply site
or widen the count gate); (b) apply the offset *before* the redistance
sweeps so the published field is a distance function of the corrected
interface (kills the per-frame step at the region boundary); (c) longer
term, per-column or per-region controllers (Kim 2007 / Ando-Batty §8) —
and if S2-b's drift is still visually offensive after WP-1..4, that is the
particle-level-set discussion (G2), which is a program, not a patch.

**WP-6 — Rigid coupling on factor 1 (S3).**
(a) Make the boundary refresh preserve ghost-opened wall faces: skip
`FACE_CLOSED_BOUNDARY` faces whose `flags=3`/`FACE_SEPARATED` state came
from the ghost pass, or re-run `publishLosassoCoarseOnlyGhosts` after
refresh (the factor-1 analogue of X1.2). (b) Write occupancy words 7–9
(wet-cell mean velocity) so drag is relative to local fluid; (c) replace the
frozen `hydrostaticReferenceY_m` with a measured local waterline (cheap:
column height from the dense tracker above the body footprint); (d) carve
phi inside solids on the coarse tracker (a dense-lattice pass; the fine
carve kernel is the template) and subtract submerged body volume from the
corrector target; (e) add the solid separating clip alongside the wall one.
Gate: D-rigid-still green; cork sphere finds Archimedes depth within 10%
without the authored-plane spring (O2 scene exists); no craters around a
resting body over 10 s.

**WP-7 — Band-1 structural work (S2-d; the standing program).**
Unchanged from `docs/losasso-band-invariance-handoff.md`: WP-B1 total nodal
reconstruction **via the containing cell** (the `7972eb4` bracket is not
it — bound it to the resolved leaf as an interim), close RC6 (subsumed by
WP-4b), D5 migration census, and the WP-B7 band-sweep oracle — which after
this audit must run **at band 1**, not AUTO, to discriminate anything.

Suggested order: WP-1 → WP-2 (both small, both S1; re-run the 6 s smoke —
expect S2 to improve materially from these alone) → WP-4 → WP-3 → WP-6
(independent track, can parallel WP-3/4) → WP-5 → WP-7.

## 8. What NOT to do

- **Don't add viscosity/damping to "fix" S2.** The lane is already
  over-dissipative; S2 is parked energy + transport loss + quantization,
  not under-damping.
- **Don't re-enable the global uniform corrector variant** or tune corrector
  gains chasing S2 — measured null.
- **Don't fix S1 by opening walls unconditionally** (e.g., skipping the
  Neumann default at wet lid faces): the approach-velocity guard matters;
  an unconditionally open lid face is a pressure vent during impact.
- **Don't widen the band or W to hide band-1 defects** — the target
  configuration is band 1; widening is the anti-goal (droplet-in-vast-domain
  needs the shell thin).
- **Don't chase the MGPCG reds first** — they are downstream of pathological
  welded-BC states until proven otherwise (D-mgpcg).
- **Don't trust `projD`-style metrics** on this lane; use the probe's
  height/ridge/step metrics and `readReceipt()` fields — both already exist
  and cost nothing.
- Standing hygiene: ±5% within-arm noise, ±2-generation nondeterminism —
  bisect by signature, not step; never compare walls across tripwire modes.
