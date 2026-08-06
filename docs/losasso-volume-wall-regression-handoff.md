# Losasso realism work — volume-loss and wall-sticking regressions

**Date:** 2026-08-05
**State:** analytic attribution only — no fixes applied. All `file:line` refs
verified against this working tree (uncommitted diff on `main` @ da8f4d6).
**Symptoms (rendered tank scene):** (1) total liquid volume shrinks steadily
over time; (2) thin films/streaks remain stuck to the tank walls at old
waterlines and along splash paths. Everything else (energy, slosh liveliness)
looks correct — which is the realism work doing its job.

## What the realism work actually changed (5 structural changes)

1. **Bounded MacCormack on coarse faces.** S1 became S1a predict / S1b reverse
   / S1c correct (`lib/webgpu-octree-losasso-dynamics.ts:180-199`, WGSL
   `reverseLosassoFaces` at `lib/webgpu-octree-losasso-dynamics.wgsl.ts:328`,
   `correctLosassoFaces` at `:344`, clamp at `:359-365`). The forward
   predictor is gathered and Jacobi-extended over the W7 band in place
   (`lib/webgpu-octree-losasso-extension-band.ts:340-355`,
   `lib/webgpu-octree-losasso-velocity-extension.ts:181-193`;
   `predictorVelocity` **aliases** `projectedSeeds`, extension-band.ts:259).
2. **All-wall unilateral separation with hysteresis.** Previously only
   gravity-opposed (ceiling) faces could open; now every closed wall opens
   under tension, ceiling with hydrostatic bias, and a separated face re-closes
   only at `p ≥ max(release, 1e-4·contactPressure)`
   (`lib/webgpu-octree-losasso-projection.ts:37-56`).
3. **Fine-transport activity gating.** Pages classify awake/asleep; sleeping
   pages skip advect+commit entirely (`pageAwake` at
   `lib/webgpu-octree-losasso-fine-transport.wgsl.ts:69-71`, classify `:104`,
   wake halo bit set in topology repair,
   `lib/webgpu-octree-fine-levelset-topology.ts:2648-2650`).
4. **Global volume correction disabled for Losasso.** The ledger now only
   measures; the uniform phi offset is not applied
   (`lib/webgpu-octree.ts:4892-4903`). Rationale in the comment: the uniform
   offset regrew wall films and detached droplets to pay for loss elsewhere.
5. **Coarse-phi publication loosened.**
   - `sampleCoarseOctreePhi` total-miss fallback returns `+0.5·width` (air)
     instead of invalid (`lib/webgpu-octree-losasso-coarse-phi.ts:92`).
   - `restrictLosassoCoarsePhiRow` now publishes `VALID|FINITE`
     **unconditionally** — previously flags were inherited from the seed row
     (`flags=seed.w&(VALID|FINITE|FINE_RESTRICTED)`) so a row with an invalid
     seed and no fine coverage published as invalid
     (`lib/webgpu-octree-losasso-coarse-phi.wgsl.ts:119`, INTERFACE flag no
     longer gated on prior validity at `:133`).
   - New dense wet-face owner remap refreshes band→wet seed ids at topology
     boundaries (`encodeTopologyRemap`,
     `lib/webgpu-octree-losasso-extension-band.ts:358-389`; WGSL
     `remapLosassoWetSeedFaces` at
     `lib/webgpu-octree-losasso-extension-band.wgsl.ts:223-234`).

## Attribution — volume loss

### VL-1 (primary; near-certain for the *visible* loss): the corrector was the thing holding volume, and it is now off

`lib/webgpu-octree.ts:4899` calls `encodeMeasurement` (measure-only) on the
Losasso path. Before this change, `finalizeFineVolume` +
`applyFineVolumeCorrection` moved fine phi by
`clamp((V−V_ref)/A, ±h/2)` every advance and pinned `currentVolume` back to
`referenceVolume`. Semi-Lagrangian fine transport + redistancing + coarsen
churn have always dissipated volume; the corrector masked it. Disabling it
did not *create* loss — it unmasked the raw drift. The comment's reason for
disabling (uniform offset inflates wall films and droplets) is sound, so the
fix is **not** re-enabling it; it is a localized replacement (WP-V1).

Note the telemetry semantics also changed: `control.currentVolume` on Losasso
now reports measured volume (previously it was reset to `referenceVolume` when
unsaturated — `lib/webgpu-octree-fine-levelset-volume.ts:83-86`). Any test or
dashboard comparing `currentVolume` across this change compares different
quantities.

### VL-2 (contributor; leak, impact-correlated): all-wall separation has no approach-velocity guard

`projectLosassoFaces` opens any closed wall face at `solved < 0` and keeps it
open below `1e-4·contactPressure` (`projection.ts:49-52`). The active-set bit
is lagged ("makes the *next* coarse operator use the p=0 air ghost"). While a
face is open, nothing constrains normal velocity *into* the wall: when sloshing
fluid returns to a wall whose faces are still separated, the projection does
not remove the approaching u·n, fine transport advects phi into the wall, and
the closed-wall phi conventions (unit outward slope + `exitCells`
reconstruction in fine transport) subsequently reconstruct less liquid. Each
open→impact→close cycle bleeds a face-area × Δt × u·n sliver of volume.
Classical unilateral contact requires complementarity on (u·n, p), not on p
alone — the missing half is the velocity guard.

### VL-3 (contributor; front-lag): a sleeping page in the path of an advancing front loses the leading edge

A page is asleep when `PAGE_ACTIVITY_VALID` is set and none of
interface/moving/halo/dirty are (`fine-transport.wgsl.ts:69-71`). Its phi is
frozen that advance. The wake signal for "front approaching" is
`PAGE_WAKE_HALO`, set in topology `repairNeighbor` from JFA `support`
(`fine-levelset-topology.ts:2648-2650`). If halo refresh cadence is coarser
than the advance cadence, or a fast front crosses a page boundary within one
step, the leading edge is not transported into the sleeping page that step and
redistancing then erases it. Loss rate scales with front speed × sleep-boundary
area — small per step, monotone, dam-break-shaped.

### VL-4 (minor): total coarse-phi miss now means "air"

`coarse-phi.ts:92` returns `+0.5·width` on total lookup miss where it returned
invalid. Consumers that treated invalid as "no information" — new-page seeding
(`fine-levelset-topology.ts:2735`), external seed classification (`:1897`),
renderer global-fine (`webgpu-water-global-fine-tetra.ts:69`,
`-classify.ts:73`) — now see definite air. Any spurious miss (row directory
churn during refresh, race with epoch flip) silently converts liquid to air
instead of erroring. Probably intentional for growth into uncovered regions,
but it removed a tripwire; keep it on the suspect list until D5's counter says
misses are zero in steady state.

## Attribution — wall sticking

The free-fall 2×2 oracles previously proved vertical walls were *exactly
inert* under free-slip (see `docs/` history / memory: contact attribution
2026-07-29). So the streaks are new machinery, not the old adhesion bugs.
Three candidates, in order of suspicion:

### WS-1 (prime suspect for the *rendered* streaks): restrict publishes stale/garbage rows as valid interfaces

`coarse-phi.wgsl.ts:119`: `flags = VALID|FINITE` unconditionally. When
`centreFine.valid == 0` (fine band retired after a film dried, or a fresh
arena slot whose `rowPhi` holds a previous occupant's value or zeros), the row
republishes its stale seed `centrePhi/min/max` as a **valid** row — and a
zeroed seed publishes `phi = 0, min = max = 0`, which sets `INTERFACE`
(`:133`). Every such row is a phantom liquid surface exactly where fluid used
to be: old waterlines on the side walls, splash paths high on the back wall.
The renderer consumes precisely this field through `sampleCoarseOctreePhi`
(global-fine tetra/classify), and the coarse operator sees it too. The old
code's `flags = seed.w & mask` meant a never-validly-published row stayed
invisible. This is a 2-line-shaped defect with a distinctive signature:
**the streaks never dry, never move, and sit at historical fluid positions** —
which matches the image.

### WS-2: bounded MacCormack systematically re-damps the near-wall layer

`velocityAtNode` counts closed-domain faces as zero-velocity contributions
(dynamics.wgsl.ts:150-190 — pre-existing, no-slip-flavored). The MacCormack
stages triple this stencil's influence: the reverse trace samples the clamped
lattice (`velocityAtGrid` clamps to the domain box), so near a wall the
back-and-forth traces don't cancel and `0.5·(original − reversed)` is
systematically biased; and the limiter clamps to `sourceStencil` extrema
(`:359-365`) whose lower bound near a wall includes those zeros. Net effect:
bulk fluid gets its energy back, the half-cell wall layer does not — the bulk
drains and the wall layer lags behind as a film. This is the textbook reason
MacCormack/BFECC implementations revert to plain semi-Lagrangian for stencils
touching boundaries (Selle et al. 2008).

### WS-3: the topology seed remap can zero extension velocity near walls

`remapLosassoWetSeedFaces` requires a **uniform owner** across a retired
coarse record's span (`extension-band.wgsl.ts:229-233`); multi-owner or
missing spans get `seedWetFace = INVALID` and SEED cleared, and
`gatherLosassoProjectedSeeds` then writes `seedVelocity = 0` (`:238`). Zero
seeds are then dilated outward: a near-wall air band seeded with zeros gives
the interface there ~zero extension velocity — a film that *cannot* recede.
Refinement churn concentrates exactly at interfaces near geometry, so the
multi-owner case is not rare. Also: the remap runs inside
`encodeReadyCommit` (`backend.ts:745-749`); if `wetControl[3] != 1` at that
point in the frame, `scatterLosassoDenseWetFaces` publishes nothing and the
*entire* seed set drops for that epoch.

### Exoneration note

The all-wall hysteresis itself (change 2) is a *release* mechanism — a film at
p≈0 keeps its separated mark under the `1e-4·contactPressure` renewal
threshold, so it is not the sticking cause; its failure mode is the VL-2 leak,
not adhesion.

## What the successor literature actually did about these two problems

Checked directly against the papers (2026-08-05): Flynn et al. 2018 and
Ando & Batty 2020, plus the lines they cite.

**Flynn, Egbert, Holladay, Oborn 2018, "Adaptive Fluid Simulation Using a
Linear Octree Structure" (CGI 2018)** — does not address either problem. It
is a data-layout paper: a pointer-free linear octree (flat C array, width
words, Losasso-style node payload) giving 1.5–5× over a pointer octree with
*identical* simulation output. Its physics is FLIP + a hierarchical
projection (per-size uniform solves, T-junction pressures interpolated from
the coarser solve). The only relevance here: being FLIP-based, velocity and
effectively the surface ride on particles, so it never faces grid-SL volume
dissipation head-on — evidence for the particle route, not a fix for ours.
Its hierarchical cascade of uniform solves is *less* faithful than our
paper-aligned graded-face discretization; nothing to import.

**Ando & Batty 2020, "A Practical Octree Liquid Simulator with Adaptive
Surface Resolution" (TOG 39(4))** — the direct successor to Losasso 2004,
and it is unusually explicit on exactly our two questions:

- **Volume loss: they did NOT solve it structurally — they correct it.**
  §8 (p. 32:13): *"We also used a variant of global/regional volume
  correction schemes (e.g., [Kim et al. 2007; Thürey et al. 2010]) to
  compensate long-term accumulated volume loss/gain of liquid."* And
  §9.2.4 (Conservation): *"Since semi-Lagrangian advection is not
  conservative [Lentine et al. 2011], our method does not exactly preserve
  momentum/mass during advection… A momentum/mass conserving advection
  scheme [Chentanez and Müller 2012; Lentine et al. 2011] might help."*
  So the state of the art for octree level-set liquids is: accept SL drift,
  run a **regional** Kim-2007-style volume controller permanently. Our
  regression is precisely that we turned ours off (for the valid reason that
  a *uniform global* offset is the wrong variant — it inflates films). The
  fix the field converged on is the one WP-V1 already specifies: regional /
  divergence-source control, not no control. Conservation-by-construction
  alternatives if we ever want them: Lentine et al. 2011 (conservative SL),
  Chentanez & Müller 2012 (mass-conserving Eulerian), BiMocq (Qu et al.
  2019) — all bigger surgeries than WP-V1.

- **Energy loss: they explicitly evaluated MacCormack for liquids and
  rejected it.** §3.2 (p. 32:4): *"We use basic semi-Lagrangian advection
  for both the velocity and the level set… MacCormack advection is also
  possible, but as Selle et al. [2008] discuss, first order semi-Lagrangian
  advection is recommended for velocity near liquid surfaces anyway. We did
  not observe significant improvement in animation quality using MacCormack…
  The added cost of MacCormack may be worth paying in smoke simulation
  contexts."* Their chosen instrument against dissipation for liquids is
  **Extended Narrow-Band FLIP** (Sato et al. 2018) near the surface
  (§7.2), plus noting bi-directional / advection-reflection schemes
  (Qu 2019; Narain 2019) as the Eulerian alternatives (§9.2.2). Two
  consequences for us: (a) if we keep the MacCormack bet, the Selle-style
  first-order fallback near the **interface and solids** is not optional
  polish — it is the published usage contract of the scheme (WP-W1 should
  gate on interface proximity too, not just closed walls); (b) the
  literature's fidelity axis for liquids is particles/hybrid transport near
  the surface — in our architecture the analog is the fine band, which is
  exactly where the activity-handoff already puts the fidelity budget.

- **Wall separation / sticking: not addressed at all.** Their solid
  boundaries are standard cut-cell Neumann via face area fractions (Ng et
  al. 2009); no unilateral contact, no separation condition anywhere in the
  paper. Neither paper touches our second regression. The line that *does*
  solve it is the inequality-constrained projection: pressure solved as an
  LCP/QP with `p ≥ 0` and complementarity `p·(u·n) = 0` on separable solid
  faces (separating boundary conditions in Bridson's book; Batty, Bertails,
  Bridson 2007 for the variational weights; Narain et al. 2010's unilateral
  incompressibility for the LCP machinery; solvable GPU-side with a
  projected/bound-constrained CG such as MPRGP inside the existing MGPCG).
  Our lagged `FACE_SEPARATED` active set is a cheap explicit approximation
  of exactly that LCP — and its defect is that it enforces only the
  pressure half of the complementarity. WP-V2's approach-velocity guard
  restores the velocity half (no inflow through an open contact) at one
  line of cost; the full projected-CG LCP is the principled endpoint if the
  guard's residual artifacts ever matter.

**Bottom line:** the successors validate the handoff's direction rather than
replace it. Ando–Batty = "keep a regional volume controller on, prefer
first-order SL near liquid surfaces (or particles), cut-cell walls" — i.e.,
WP-V1 and WP-W1 are the published practice; WP-V2/full-LCP is from the
contact literature neither paper engages; Flynn18 contributes nothing
physical.

## Diagnostics plan (ordered; each is cheap and discriminating)

**D1 — decompose the drift curve (first, hours).** The ledger still runs
(measure-only). Extend `tools/webgpu-smoke-executor.ts` (it already unpacks
`FineLevelSetGPUVolumeControl`, line 11-12, and reports
`exactVolumeDrift`, `:814`) to log per checkpoint: `currentVolume`,
`fineVolume`, `coarseVolume`, `replacedCoarseVolume`, `interfaceArea`,
`correction`. The *shape* attributes the loss: constant slope → steady SL
dissipation (VL-1 unmasking); steps synchronized with wall impacts → VL-2;
steps at topology epochs → VL-3/WS-1.

**D2 — A/B the corrector.** `FLUID_VOLUME_CONTROL` exists
(`webgpu-smoke-executor.ts:647`) but the Losasso branch at
`webgpu-octree.ts:4898` ignores it — route it: when set, take the `encode()`
path even on Losasso. Expected: volume holds, films worsen. Confirms VL-1 is
the visible driver and gives the raw-drift number the WP-V1 gate needs.

**D3 — wall-leak ledger.** After projection, accumulate signed and absolute
`u·n · area` over closed faces with `FACE_SEPARATED` set (spare word in the
existing metrics; readback at checkpoints), plus the count of separated faces
below the initial waterline. Nonzero net into-wall flux = VL-2 confirmed;
correlate its integral with D1's step losses.

**D4 — activity-gate A/B.** Add an env-gated params flag that forces
`pageAwake() = true` (the plumbing pattern exists: `inflowVelocityStrength.w`
already forces wake, `fine-transport.wgsl.ts:71`). Compare D1 slopes gated vs
forced-awake — the difference is VL-3. Add a census counter: pages waking with
`|phi|_min < h` already inside them (woken late = front already arrived).

**D5 — phantom-row census + render bisect (decides WS-1 fast).**
(a) Counter in `restrictLosassoCoarsePhiRow`: rows published with
`centreFine.valid == 0 && (seed.w & (VALID|FINITE)) != (VALID|FINITE)`;
readback in `tests/webgpu-octree-losasso-regressions.test.ts`. Nonzero on the
tank scene ⇒ WS-1 real.
(b) On a frame with visible streaks, render the coarse-phi surface and the
fine-band surface separately (both paths exist in the global-fine
tetra/classify machinery). Streaks present in coarse-only ⇒ WS-1; present in
fine-only ⇒ WS-2/WS-3.

**D6 — seed-remap census.** After `encodeTopologyRemap`, count ACTIVE band
faces with `seedWetFace == INVALID` and the per-epoch SEED-flag delta. A/B:
skip the remap entirely (pre-change behavior — stale ids, but nonzero seeds).
Films receding again under the A/B ⇒ WS-3.

**D7 — MacCormack boundary A/B.** Params flag: skip the S1c correction for
faces whose forward-source stencil touched any `closedDomainFace` (track one
near-boundary bit through `VelocitySample`; the wall test already exists at
dynamics.wgsl.ts:141). Compare film persistence. Films receding ⇒ WS-2.

**D8 — a receding-wall oracle (new scene, analytic gate).** The free-fall 2×2
oracles test *approach/contact*; nothing tests *recession*. Add
`wetted-wall-drain`: a block seeded flush against one vertical wall, wetted to
height h₀, then allowed to slump. Gate: wet wall cells above the evolving bulk
surface → 0 within T; centroid matches the shallow-water slump to tolerance.
This is the regression harness for whichever of WS-1/2/3 is fixed, and it
isolates sticking from volume (short T, loss negligible).

## Implementation handoff (gated on diagnostics)

**WP-W2 — restrict validity (do first; smallest, likely biggest visual win).**
Restore seed-flag inheritance as the fallback: when `centreFine.valid == 0`,
publish `flags = seed.w & (VALID|FINITE|FINE_RESTRICTED)` (keep the
recompute-when-fine-valid path from this diff); never set `INTERFACE` unless
the row is valid by either route. Two lines in `coarse-phi.wgsl.ts:119,:133`.
Gate: D5 counter = 0; streak pixels gone in the coarse-only render.

**WP-V2 — approach-velocity guard on separated faces.** In
`projectLosassoFaces`, for a closed face currently `FACE_SEPARATED`, clamp the
projected normal velocity to outward-only:
`projected = outward · max(0, outward · projected)` — separation still
releases, but the wall is never porous to approaching flow during the
active-set lag. One expression at `projection.ts:33-36`. Gate: D3 into-wall
flux = 0; free-fall 2×2 oracles stay green (release must not regress).

**WP-W1 — MacCormack semi-Lagrangian fallback at boundaries AND near the
interface.** Promote the D7 flag to default-on behavior: skip S1c (keep
S1a's value) when either trace's stencil touched a closed wall, was clamped
by the domain box, **or lies within one coarse cell of the interface**
(coarse row INTERFACE flag is already in the stencil's reach). This is the
Selle et al. 2008 usage contract that Ando & Batty 2020 §3.2 restates:
first-order SL near liquid surfaces, MacCormack only in the interior. Gate: D8 oracle passes; interior energy metrics
unchanged (the realism win must survive — only the half-cell wall layer
changes).

**WP-W3 — seed remap multi-owner fallback.** When the span maps to multiple
refined wet owners, seed from the area-weighted mean of the refined faces'
projected values instead of dropping to INVALID/0; when `wetControl` is
invalid at remap time, retain the prior `seedWetFace` instead of clearing.
`extension-band.wgsl.ts:223-234`. Gate: D6 census small and stable; no
zero-velocity seeds adjacent to wet faces.

**WP-V1 — localized volume control (replaces the disabled global offset).**
This is the variant the successor literature runs permanently (Ando & Batty
2020 §8: "global/regional volume correction… [Kim et al. 2007; Thürey et
al. 2010]"). Preferred: divergence-source volume control in the pressure
RHS (Kim et al. 2007 style) — distribute `k·(V_ref − V)` as a source term over rows of the
**main connected reservoir only** (component labeling exists since da8f4d6
"bootstrap disconnected fluid reservoirs"), excluding interface rows whose
band pages are film-like (thin `|phi|` support, low `PAGE_ACTIVITY_MOVING`).
This conserves without the failure mode that got the global offset disabled:
films and stray droplets receive no correction by construction. Fallback
option: keep the phi-offset mechanism but mask application to samples on
`PAGE_ACTIVITY_MOVING` pages of the main component. Gate: D1
`exactVolumeDrift` bounded over a 50 s tank run; D8 and free-fall oracles
green; film pixels do not grow over 50 s.

**WP-V3 — halo wake in the classify pass.** If D4 shows front-lag loss:
dilate the awake set by one page-neighborhood inside
`classifyLosassoFineActivity` each advance (awake page marks face-adjacent
neighbors' `PAGE_WAKE_HALO`), rather than relying on topology-repair cadence.
Gate: D4 late-wake census = 0; activity-gating perf win preserved (this adds
one atomicOr per awake-page face, not a full-domain pass).

### Suggested order

1. D1 + D2 (attribution numbers, same run) → 2. D5 → WP-W2 (cheap, likely the
rendered streaks) → 3. D3 → WP-V2 → 4. D6/D7 → WP-W3/WP-W1 as indicated →
5. WP-V1 (the substantive design item) with D8 landed first as its regression
gate → 6. D4 → WP-V3 if the gated drift residual is still visible.

### Cross-cutting cautions

- `predictorVelocity` aliases `projectedSeeds` (extension-band.ts:259). The
  current encode order (S1 predictor gather → S3e re-gather from
  `wetProjected`) makes this safe; any reordering of S1/S3e breaks it
  silently. If WP-V1 touches the advance graph, assert on this.
- The volume ledger's `currentVolume` semantics changed on Losasso (measured,
  not pinned). Fix any test comparing it against `referenceVolume` before
  trusting D1 numbers.
- Never compare drift numbers across tripwire modes (`failfast` costs ~27%),
  and single-run A/B deltas under ~5% on the dry lanes are noise — run arms
  interleaved.
