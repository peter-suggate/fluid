# Refinement regions freeze the surface — coarse-cell transport handoff

**Date:** 2026-08-08
**State:** code-forensic diagnosis at working tree (HEAD `bf3c360` + uncommitted
refinement-region feature); no fixes applied. Every file:line verified by
direct read this session.
**Symptom (user-reported):** drawing a minimum-cell-size refinement region over
part of the dam-break scene freezes the fluid in and out of the box — the
surface inside never moves, and the wave outside piles against the region
boundary as if it were a wall.
**Lane:** factor-1 coarse Losasso (`globalFineLevelSetFactor: 1`, the product
default, `lib/methods/octree.ts:42`) — the only lane the region feature ships
on.

---

## 1. TL;DR

The paper does **not** say coarse cells cannot transport the free surface —
our lane added that invariant, and then built velocity machinery that is only
correct when the invariant holds. The region feature is the first thing that
violates it, so the machinery's failure mode fires:

> Every finest MAC plane near the interface gets a span-1 "air band" record —
> including planes strictly **interior** to a held coarse leaf. Those records
> can never reach a velocity seed, so the Jacobi extension writes them an
> exact `0.0`. The staged-lattice sampler then finds that span-1 record
> *first* and returns the zero as a **valid** velocity — short-circuiting
> `stagedCoarsenedVelocity`, the function that implements the paper's own
> coarse-face reconstruction and would have returned the right answer.
> Valid-zero velocity at every interface cell ⇒ the semi-Lagrangian backtrace
> lands on itself ⇒ phi never moves ⇒ the region is a static obstacle.

The inversion is diagnostic: cells in the **bulk** of a held leaf have no
span-1 record, fall through to the bracket reconstruction, and get correct
velocity. Only the **surface** freezes. "Coarse bulk flows, coarse surface
does not" is exactly this mechanism and nothing else.

The region gate itself is innocent: it is a pure split-refusal at three sites
(`lib/webgpu-octree.ts:9002` inside `pressureRefinementEvidence`, `:9179` in
`leafNeedsRefinement`, `:9567` in `refineCoarseBlock`), grading is
deliberately ungated, and a scene with no regions is bit-identical. The
freeze is entirely downstream, in code that assumed "surface ⇒ unit cells".

---

## 2. The paper's actual position (the misinterpretation)

`docs/papers/losasso-2004-octree-water-smoke.txt`:

- **§6:** *"We apply adaptive refinement to a band about the interface
  (focusing more heavily on the water side)"* — refinement near the interface
  is a **criterion for where to spend resolution**, exactly like the smoke
  criteria in §5 (objects, vorticity, optical depth). It is an accuracy/
  quality choice, never a precondition for transport.
- **§6:** phi is *"update[d] … with the semi-Lagrangian method using
  velocities defined at the nodes (see section 3)"* — on the nodes of **every
  cell**, whatever its size. Nothing conditions level-set transport on leaf
  size.
- **§3:** velocities live on the faces of all leaves; nodal averaging handles
  arbitrary size transitions (*"using the coarsest neighboring face as the
  scale, we compute temporary coarsened velocities on the other faces"*).
  That sentence is the license for a surface crossing a coarse cell: the
  velocity there is a *reconstruction from the coarse faces*, not zero and
  not absent.
- **§4.1/4.2:** the symmetric Poisson discretization is defined for any
  unrestricted octree configuration; the free-surface Dirichlet (§6) applies
  to whichever face separates a wet cell from an air cell, at any size.

So a coarse cell containing the zero isocontour is a *first-order-accurate*
configuration in the paper, not an illegal one. Our lane hardened the
refinement criterion into a correctness invariant:

- `lib/webgpu-octree-losasso-backend.ts:682` — `gradingPolicy =
  "uniform-fine-free-surface-shell"`.
- `lib/octree-interface-band-audit.ts:16-20` — audits *"no leaf coarser than
  one cell may contain the interface"* as an invariant.
- `lib/webgpu-octree.ts:9045-9050` — any interval zero crossing forces the
  split ladder to size 1.
- `lib/octree-runtime-dials.ts:170-192` — the `finestSurfaceCellSize` dial
  (clamped to 1..2) already documents that setting it to 2 *"currently
  exposes a stationary coarse-cut failure"* — **the same mechanism this
  handoff diagnoses**, previously observed at S=2, now exposed at S=8/16/32
  by the region. The region did not create a new bug; it made a known,
  dial-gated one reachable from the editor.

Strategic note: fixing this is not just region-enablement. The activity
program (`docs/losasso-activity-handoff.md`) records that *spatial*
fine-factor variation is blocked; a working coarse-surface transport path is
precisely what unblocks the spatial axis, and the region is the instrument
for measuring it.

---

## 3. The freeze mechanism, step by step (all verified)

Phi itself is fine. At factor 1 the level set is advected on the **dense
finest lattice** independent of octree topology (`predictSummaryCells`,
`lib/webgpu-octree-coarse-summary.ts:812-830`, runs over `p.domainVolume`),
so representation inside a held leaf is not the problem. The velocity that
transport samples is:

1. **Span-1 air records blanket the interface — even inside coarse leaves.**
   `publishLosassoCoarseAirBandFaces`
   (`lib/webgpu-octree-losasso-extension-band.wgsl.ts:296-309`) sweeps every
   finest MAC plane and appends a span-1, layer-1, ACTIVE, **non-SEED**
   record wherever the dense phi crosses or is within one finest cell.
   Leaf size is never consulted. Layers 2..7 dilate from these (`:315-333`).

2. **Only wet pressure faces are seeds, and a held leaf's wet faces are
   span-S records on its boundary planes.** `publishLosassoWetSeedFaces`
   (`:242-244`) appends wet faces at their own dyadic level;
   `gatherLosassoProjectedSeeds` fills velocities only where
   `seedWetFace != INVALID` (`:380-388`); `beginLosassoExtensionBand`
   resets `seedWetFace`/`seedVelocity` every publication (`:239`). The
   remap that would map a span-1 record onto its covering dense wet owner
   (`remapLosassoWetSeedFaces`, `:364-377`) runs only at topology commit
   (`lib/webgpu-octree-losasso-extension-band.ts:588` via
   `encodeTopologyRemap`), never in the per-advance path.

3. **The extension writes starved faces an exact 0.0.** The Jacobi kernel
   (`lib/webgpu-octree-losasso-velocity-extension.wgsl.ts:62-87`) computes a
   layer-1 face only from layer-0 neighbours (`neighborMetric.w >= layer`
   skips its layer-1 peers), and `else { outputVelocity[face] = 0.0; }`
   (`:86`). Adjacency resolves through span-1-first `containingCompact`, so
   an interface plane two or more finest cells from the held leaf's boundary
   has no reachable seed: its extended velocity is exactly zero. (Planes
   directly adjacent to the leaf boundary do pick up the span-S wet face's
   value — the freeze has a one-cell-thick leaky rim, which is why the
   boundary in the screenshot shows slight communication but no flow.)

4. **Staging prefers the starved span-1 record over the correct
   reconstruction.** `stageLosassoVelocityLattice` binds the extension
   output as the seed buffer (`lib/webgpu-octree-losasso-extension-band.ts:486`,
   binding 14 = `this.extended`), and `stagedRawVelocity`
   (`...extension-band.wgsl.ts:156-163`) does:
   ```wgsl
   let face=containingCompact(axis,q);          // span 1 first — finds the air record
   if(face!=INVALID){let value=seedVelocity[face];
     if(finite(value)){return bitcast<u32>(value);}   // returns the 0.0 as VALID
     return STAGED_INVALID;}
   ...
   return stagedCoarsenedVelocity(axis,q);      // never reached on these planes
   ```
   `stagedCoarsenedVelocity` (`:135-155`) is the paper's §3 reconstruction —
   find the smallest dyadic bracket of real faces and interpolate — and it
   would return the correct value for every plane the air records shadow.
   Its own doc comment (`:128-134`) states the intent the air records defeat.

5. **Downstream, everything reads the zero.** `stagedNodalSample` (`:171-187`)
   averages the zeros into valid nodes; `predictSummaryCells` gets
   `velocity.w > 0, |v| = 0` — the backtrace lands on itself and phi is
   carried unchanged (`coarse-summary.ts:820-829`); `advectLosassoFaces`
   (`lib/webgpu-octree-losasso-dynamics.wgsl.ts:297-316`) samples the same
   nodes for the leaf's face centres, so the coarse faces are re-zeroed each
   advance. The projection then sees a wet region with no interior motion
   and near-zero boundary flux: fluid neither enters nor leaves the box.

**Why the receipts look clean:** `frozenCells` (`state[31]`) counts only
*invalid* velocity (`:822`); a valid zero is invisible to it. The
discriminating triple is `interfaceVelocityQueries`/`interfaceVelocityValid`
high while `interfacePhiMoved` collapses
(`state[27]/[28]/[29]`, `readReceipt` at `coarse-summary.ts:508-513`).

---

## 4. Secondary gaps in the same region (real, but not the freeze)

These matter once transport moves — they set the *quality* of a coarse
surface, and several are the S³-scaled versions of gaps already catalogued in
`docs/losasso-paper-gap-handoff.md` §4.2:

- **G-a: ghost theta is row-centre-gated at row-width distance.** A held
  leaf's row phi is a centre point sample with the gradient hard-zeroed and
  the interval collapsed (`lib/webgpu-octree-losasso-coarse-phi.wgsl.ts:89-100`;
  `correctCoarseDirectory`, `coarse-summary.ts:995-1002`). The free-surface
  ghost takes theta from that centre value at `dual = rowSize · h`
  (`coarse-phi.wgsl.ts:127-148`); a leaf wet-by-minimum but centre-dry falls
  through to `flags=4u` — a staged p=0 Dirichlet a full row width away. The
  surface inside a region therefore has no sub-leaf hydrostatics: adjacent
  columns differing by less than S cells produce no leveling gradient. This
  is paper-gap WP-4 (S2-f) at S=8..32 instead of 2.
- **G-b: wet classification dilates to the whole leaf.** For size ≥ 2, one
  negative finest cell makes the entire leaf a wet pressure row
  (`webgpu-octree.ts:9818-9825`; `complete` is forced by the
  `COARSE_AUTHORITY` stamp, `coarse-summary.ts:954`). Per the paper this is
  legitimate — a partially wet cell *should* get an equation, with theta
  handling the partial — but it is only benign once G-a is fixed; today it
  inflates the liquid region by up to S cells with a wrongly-placed ghost.
- **G-c: sign-frozen redistance + directional corrector.** The redistance
  cannot flip sign (`coarse-summary.ts:884`) and the volume corrector applies
  only on cells transport actually moved (`:929-949`), so a frozen region
  contributes nothing and the global compensation is spent deforming the
  still-moving parts of the domain — the region exports its volume error.
- **G-d: the residency scheduler is not region-aware.**
  `fluidGatedBoundarySupport` must mirror the refinement policy exactly —
  the constructor comment records what happened last time they disagreed
  (`webgpu-octree.ts:2198-2202`: "the scheduler won and published topology
  for leaves that never existed"). A region covering a wall band recreates
  that disagreement by construction. Audit before shipping regions near
  walls.
- **G-e: the interface-band audit now red-flags by design.**
  `lib/octree-interface-band-audit.ts:16-20` treats a coarse interface leaf
  as a violation; `straddlingCoarseRows/Cells` will be non-zero inside every
  region. The audit needs region-awareness or its consumers will chase
  phantom regressions.
- **G-f: cost hygiene.** `redistanceReachCells` sizing
  (`coarse-summary.ts:114-149`) ignores regions, so the sweeps a region was
  drawn to avoid are still paid for. Fold the region floor into the reach
  computation once transport works — this is the feature's actual payoff.

---

## 5. Exonerated — do not spend time

- **The region gate and packing** (`lib/octree-refinement-regions.ts`) —
  pure split-refusal, floor clamped to `topologyMaximumLeafSize` (`:173`) so
  the tile-origin reachability hazard (`webgpu-octree.ts:2176-2183`) cannot
  fire; grading ungated by design; uniform-tier (no reset) verified by
  `tests/octree-refinement-regions.test.ts:264-278`.
- **The dense phi representation** — `predictSummaryCells` runs the full
  lattice; a coarse leaf's interior surface is representable and rendered
  (the screenshot's smooth in-region surface is the dense field drawing
  correctly while frozen).
- **Velocity migration across epochs** — averaging + dyadic-bracket
  reconstruction (`webgpu-octree-losasso-velocity-migration.wgsl.ts:33-98`),
  not a reset; the re-zeroing is the per-advance staging path above.
- **The pressure discretization** — size-agnostic and verified previously;
  nothing here touches it.

---

## 6. Discriminating experiments (cheap, run before fixing)

- **D-freeze-kind:** dam-break + region, read `readReceipt()`: expect
  `frozenCells` flat, `interfaceVelocityQueries`/`Valid` high,
  `interfacePhiMoved` ≈ 0 over the region. Confirms valid-zero (§3), not
  invalid-velocity, freeze.
- **D-floor-2:** region floor sweep 1 (inert) vs 2. If floor 2 already
  freezes, the mechanism is identical to the documented
  `finestSurfaceCellSize=2` stationary coarse-cut failure — one fix, two
  features unblocked.
- **D-rim:** ASCII surface probe (`tools/probe-dam-surface-shape.ts`) with a
  region across half the tank: expect motion in a ~1-cell rim just inside
  the region boundary (the leaky adjacency in §3.3) and none deeper. The rim
  is the mechanism's fingerprint; a uniformly dead region would point
  somewhere else.
- **D-bracket:** temporarily skip `publishLosassoCoarseAirBandFaces` records
  whose plane lies strictly inside a wet leaf of size > 1 (a 5-line guard)
  and re-run D-freeze-kind. If `interfacePhiMoved` recovers, §3 is confirmed
  end-to-end before committing to the real fix.

---

## 7. Work packages (ordered; each has a gate)

**WP-A — Starvation sentinel in the extension; bracket fallback in staging
(the fix).** Two coordinated edits:
(a) `losasso-velocity-extension.wgsl.ts:83-87` — when `count == 0`, write a
NaN/sentinel instead of `0.0` (the `causalFront` pre-zeroing at `:64` is
overwritten at the face's own sweep, so only genuinely starved faces end the
K=8 schedule holding the sentinel; the predictor path shares the kernel).
(b) `stagedRawVelocity` (`extension-band.wgsl.ts:156-163`) — on a non-finite
band value, fall through to `stagedCoarsenedVelocity` instead of returning
`STAGED_INVALID`. Net: a reachable air face keeps its extended velocity
(refined default bit-identical in behaviour), a starved one gets the paper's
§3 coarse-face reconstruction.
Gate: D-freeze-kind recovers `interfacePhiMoved` inside the region; the dam
wave visibly crosses the region boundary; default scene (no regions)
D4-stable.

**WP-B — Stop publishing shadow records inside coarse wet leaves
(structural companion).** In `publishLosassoCoarseAirBandFaces`, skip planes
already covered by a coarser face record (wet faces are published first —
run order `extension-band.ts:430-433` — so `containingCompact` at span > 1
answers this). Shrinks the band graph inside regions to the leaf's real
faces instead of carrying a blanket of zero-value span-1 records. Do after
WP-A, as a measured simplification: WP-A alone is correct; WP-B removes the
dead weight and the rim asymmetry.
Gate: band face count drops inside regions; no behaviour change vs WP-A
outside them.

**WP-C — Sub-leaf ghost placement (G-a; quality).** Take the free-surface
theta per pressure face from the dense tracker's phi crossing rather than
the row-centre value, and give wet-by-interval/centre-dry rows a conditioned
interior face instead of the `flags=4` staged p=0 fallthrough
(`coarse-phi.wgsl.ts:127-148`). This is paper-gap WP-4 generalized from
S=2 to arbitrary S — coordinate with that work package rather than forking
it.
Gate: settled tank with a region over half the surface levels flat across
the boundary; a one-cell authored bump inside the region produces a nonzero
leveling gradient.

**WP-D — Region-aware hygiene (G-d/G-e/G-f).**
(a) audit `fluidGatedBoundarySupport` vs the region-gated refinement policy
for wall-covering regions; (b) teach `octree-interface-band-audit` that
straddling coarse rows inside a region are expected; (c) fold region floors
into `redistanceReachCells` so the region actually cheapens the advance.
Gate: audit green with a wall-adjacent region; measured redistance cost drop
with a large region.

**WP-E — Dynamics coverage (the missing test class).**
`tests/octree-refinement-regions.test.ts` is topology/packing/editor only —
nothing moves. Add a Dawn dynamics test: 32³ dam break, floor-8 region over
the far half; assert (1) `interfacePhiMoved > 0` within the region window,
(2) column heights inside the region change by > 1 cell as the wave arrives,
(3) global volume tracks the no-region arm within noise. Plus a receipt
regression pinning the D-freeze-kind triple.
Gate: test red at today's tree, green after WP-A.

Suggested order: D-freeze-kind + D-bracket (half a day, de-risks everything)
→ WP-A → WP-E (lock it) → WP-C → WP-B → WP-D.

---

## 8. What NOT to do

- **Don't exempt interface leaves from the region gate.** "Refine wherever
  the surface is" deletes the feature — the entire point is a coarse surface
  in calm water. The paper supports it; make the machinery honor it.
- **Don't fix by seeding the air records with zero-distance wet velocities
  via a per-advance remap sweep** unless WP-A proves insufficient — it adds
  a dispatch to every advance to repair records WP-B says shouldn't exist.
- **Don't touch the pressure discretization or grading** — both are correct
  and size-agnostic already.
- **Don't read `frozenCells` as the freeze metric** — it cannot see this
  failure (valid zeros). Use the `state[27]/[28]/[29]` triple.
- **Don't evaluate region quality before WP-C lands** — with theta at row
  granularity, a working transport will still look terraced at S cells;
  that is G-a, not a transport regression.
- Standing hygiene: ±5% within-arm noise on this lane; never compare walls
  across tripwire modes.
