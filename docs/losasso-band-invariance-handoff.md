# Losasso band invariance — why a narrow band breaks the physics, and the handoff

**Date:** 2026-08-08
**State:** code-forensic diagnosis, no fixes applied. Every load-bearing claim
below was verified by direct read of this working tree; secondary refs come
from the same sweep and are marked where not independently re-verified.
**Symptom:** shrinking the fine surface band distorts the simulation instead of
merely reducing accuracy — e.g. the mini dam break stops collapsing.
**Lane:** coarse-band Losasso (`coarseBackend=losasso`, factor 1 default).

## The invariant, and why the paper has it and we don't

The claim being violated: *a very coarse sim behaves well, a very fine sim
behaves well, so a coarse/fine mixture should only modulate accuracy.* In
Losasso 2004 that invariant holds because of three properties of §3:

1. **Coarse cells are full fluid.** Advected, forced, projected — velocity
   history lives on every face at every size.
2. **Nodal velocity reconstruction never fails.** At a node whose incident
   faces differ in size, "using the coarsest neighboring face as the scale, we
   compute temporary coarsened velocities on the other faces to be used in the
   averaging." There is no invalid case — coarse neighborhoods produce coarse
   values, not missing values.
3. **Refinement/coarsening transfer is averaging.** New faces get nodal
   averages; coarsened faces get face averages. Velocity is never created at
   zero.

Our lane violates (2) and (3) outright, and adds two fail-closed layers the
paper doesn't have (leaf-centre wetness, a global volume corrector). The
result: **band width is not an accuracy dial here — it is the radius of the
region where the simulation is actually running the paper's algorithm.**
Everything outside that radius is a different, degenerate scheme. Pure-coarse
and pure-fine lanes never see this because a uniform lattice has span-1 faces
everywhere; only a *mixture* creates span≥2 leaves, and only a *narrow band*
pushes them into the dynamically active flow.

## Root causes, ranked

### RC1 — Coarse faces carry no momentum: velocity resets to 0 + dt·g every advance (primary; verified)

The face advection entry point does not read the face's own previous
velocity. It re-derives "previous" from the staged nodal field and falls back
to zero:

- `advectLosassoFaces` (`lib/webgpu-octree-losasso-dynamics.wgsl.ts:302-303`):
  `prior = velocityAtGrid(centre, BAND_FIELD)`; `previous = select(0.0,
  prior.value[axis], prior.valid)`. Binding 5 (`extendedVelocity`) is declared
  in the shader (`:36`) but deliberately absent from the bind group
  (`lib/webgpu-octree-losasso-dynamics.ts:69-75`) — the nodal field is the
  *sole* carrier of face history.
- `velocityAtGrid` (`dynamics.wgsl.ts:197`): one invalid corner node
  invalidates the whole trilinear sample.
- `stagedNodalSample` (`lib/webgpu-octree-losasso-extension-band.wgsl.ts:142-158`):
  a node is valid only if, **per axis**, every in-domain tangential-quadrant
  MAC face resolves (`:153` — any `STAGED_INVALID` quadrant kills the node).
- `stagedRawVelocity` → `containingCompact` (`extension-band.wgsl.ts:122-134`):
  the face-directory ladder snaps **only tangential** coordinates to the span
  (`:123`); the normal coordinate is never snapped. A MAC plane strictly
  interior to a coarse leaf therefore has no record *at any span* →
  `STAGED_INVALID`.

Geometry does the rest: a coarse face's centre node always lies on planes
interior to its leaf along both tangential axes, so its nodal sample needs
axis-b/axis-c MAC faces on interior planes — which don't exist. Every face
bordered only by coarse leaves gets `previous = 0`, then `forceLosassoFaces`
adds `dt·g` (`dynamics.wgsl.ts:380`), then projection removes the divergence.
**The coarse interior is memoryless hydrostatics recomputed from scratch each
advance.** It cannot accelerate, so a column of it cannot collapse.

Band dependence: the region where nodal reconstruction *does* work is
`{span-1 leaves} ∪ {W7 extension shell}`. With the authored band-4 profile,
span≥2 leaves sit in the near-hydrostatic deep interior and the defect is
invisible. Shrink the band and the memoryless region grows into the flow that
is supposed to be collapsing — the distortion tracks the band exactly.

### RC2 — Refinement creates a third of its new velocity DOFs at exactly zero (verified)

`migrateLosassoLaggedVelocity`
(`lib/webgpu-octree-losasso-velocity-migration.wgsl.ts:62-73`): each new face
gathers old faces over its span² finest sub-positions via `containingOld`
(`:28-31`) — again tangential-only snapping. Coarsening is the paper's exact
area-average (all sub-planes coincide with old fine planes). But when a coarse
cell of span S **refines**, the new mid-plane faces inside it sit on planes
that were interior to the old cell: `count == 0` → `newVelocity = 0` (`:72`).
For a 2:1 split that is 4 of every 12 new faces per axis. Also `:64`: an
invalid old authority zeroes the **entire** field, not just the epoch's delta.

A narrower band refines/coarsens closer to the action and the band edge sweeps
through faster-moving fluid, so this momentum sink scales up as the band
shrinks. The projection then "repairs" the injected zero-divergence error by
braking the surrounding fluid.

### RC3 — Solver wetness snaps to leaf centres: the pressure surface is quantized to leaf size

`currentPressureOwnerWet` (`lib/webgpu-octree.ts:9675-9712`): size ≥ 4 rows are
wet iff the **leaf-centre** phi sample is negative; size 2 falls through to the
row-centre arena value (`liquidOwner`, `:7939-7961`) because the factor-1
publisher stamps `COARSE_AUTHORITY` on every entry. A leaf straddling the
surface is all-or-nothing: either its whole size³ block is a pressure row or
none of it is. With a narrow band the straddling leaves are size 4/8, so the
solver's free surface moves in leaf-sized jumps and half-wet water gets no
pressure DOF at all — the hydrostatic head driving the collapse is
mis-measured at exactly the surface that matters.

### RC4 — The uniform volume corrector fights the collapse (verified encode)

The coarse-only tracker runs an **unconditional** uniform phi offset every
advance: measure (`summarizeDenseVolume`), compute
(`prepareVolumeCorrection`, `lib/webgpu-octree-coarse-summary.ts:741-747`,
target latched from the first advance at `:744`), apply to **every dense
cell** (`correctAndAggregateSummaryCells`, `:748-753`), encoded at `:381-382`.
Clamped at ±h/2 per advance. This is the same "uniform global offset" variant
that `docs/losasso-volume-wall-regression-handoff.md` §VL-1 documents was
disabled on the fine path for regrowing films. On this lane it is live: any
band-induced transport loss (RC1/RC5 freeze the surface locally) is measured
as volume loss, and the corrector answers by pushing phi negative
*everywhere* — re-inflating the dam at up to half a cell per step while
transport tries to drain it. Amplifier, not root cause: it converts local
band defects into global shape distortion.

### RC5 — Phi can only change phase inside the staged shell; elsewhere it silently freezes

Dense coarse-phi advection (`predictSummaryCells`,
`coarse-summary.ts:692-698`): if the staged velocity sample is invalid the
cell rewrites last frame's phi — no counter, no flag. The Losasso sampler is
all-or-nothing over its 8 corners
(`webgpu-octree-losasso-velocity-sampler.wgsl.ts:136-138`), and redistancing
rebuilds *magnitude only*, never sign (`coarse-summary.ts:713,:731`). So the
advectable set — the only place liquid can become air — is the staged-velocity
support region, whose radius is the **hard-coded W=7 extension shell**
(`OCTREE_LOSASSO_EXTENSION_WIDTH`,
`lib/webgpu-octree-losasso-extension-band.ts:6`), not the band dial and not
the CFL. The front advances at the speed the support region grows, not the
speed of the fluid. On the factor-4/8 fine path the same class of miss is
louder but worse-shaped: one unsupported sample rejects the **entire**
level-set advance (`fine-transport.wgsl.ts:202-216` → `acceptedStep()`).

### RC6 — Half-wet coarse rows can get a fictitious atmospheric surface

`publishLosassoCoarseOnlyGhosts`
(`lib/webgpu-octree-losasso-coarse-phi.wgsl.ts:125-137`): when a face has no
compact neighbor row, the default is correct Dirichlet air — but the
interior branch's fallthrough (`:137`, `flags=4`) keeps the staged
`openFraction=1`, p=0 face whenever the row-centre phi reads ≥ 0 or the
half-span air probe lands badly. A narrow band creates exactly the half-wet
coarse rows (row-mean/centre phi near zero — see the topology handoff's
"row-mean phi is a weak signal") that take this branch, so a submerged
interface between two bodies of water is solved as a free surface: the
hydrostatic column short-circuits at the band edge.

### RC7 — The dial system ships thinner than authored, and nothing couples the reaches

- **The default dial is not a no-op.** Live `interfaceBandCells` defaults to 3
  with no AUTO (`lib/octree-runtime-dials.ts:132-168`); resolution
  (`octreeDialledSurfaceBand`, `:382-409`) spends grading first, so the
  shipped authored pair (band 4 / grading 1) runs at an **effective band term
  of 1** (protection width 3) on every frame out of the box. The mini-dam
  profile (authored band 3 / grading 3 / leaf 32, `lib/scenes.ts:92-101`)
  drops from a 9-cell shell to 3. "The authored band" is never what runs
  unless the dial is wound to max.
- **The dry-boundary look-ahead shrinks with the dial with no floor**
  (`boundaryLiquidWouldRefine` call sites, `lib/webgpu-octree.ts:9076-9078`,
  `:9417-9421`, raw `solve.w` — 1 cell of warning at the default dial where
  the comment calibrates for 3).
- **The extension width is a constant justified by the default band.**
  `OCTREE_LOSASSO_EXTENSION_WIDTH = 7` is documented as "four-cell band +
  two-cell backtrace + one MAC stencil cell". Nothing ties it to the band in
  either direction, and `velocityExtensionSweeps` can be dialled to 2 while
  the graph stays W7 (knowingly stale outer layers).
- **No CFL–band validation exists on the factor-1 path.** There is no
  velocity-dependent dt clamp on the octree path (substeps = 1,
  `lib/webgpu-uniform-eulerian.ts:1882`), the semi-Lagrangian trace distance
  is unbounded a priori, and the only band-vs-reach assert in the codebase is
  the factor-4/8 residency floor
  (`lib/webgpu-octree-fine-levelset-topology.ts:294-306`).
- **Two independent copies of the protection-width formula** (host/authored at
  `webgpu-octree.ts:2919-2920` for redistance sweeps; shader/effective at
  `:8888-8890`) currently agree only because the dial can only thin.

## Exonerated (don't spend time here)

- **The pressure discretization is correct.** T-junction faces are emitted
  once at fine area, the symmetric `(p2−p1)/Δ` form with consistent
  operator/RHS/projection triple, flux-conservative across the interface
  (`webgpu-octree-losasso-backend.wgsl.ts:193-249,373-425`,
  `-operator.wgsl.ts:57-71`, `-projection.ts:39-43`).
- Gravity is applied on coarse faces, aperture-weighted
  (`dynamics.wgsl.ts:380`).
- Coarsening velocity transfer is the paper's exact area-average.
- The band-edge default BC is Dirichlet air, not a phantom wall (the RC6
  fallthrough is the exception, not the rule).
- Redistance sweep count over-covers the dialled width by construction
  (authored ≥ effective).

## Discriminating experiments (ordered; each is cheap and kills or crowns a cause)

- **D1 — momentum-retention A/B (decides RC1, hours).** One-line arm: in
  `advectLosassoFaces`, when `prior.valid` is false, fall back to the face's
  own lagged velocity (bind `extendedVelocity` — the binding is already
  declared) instead of 0. If the mini dam collapses again at a narrow band,
  RC1 is confirmed as the driver. This arm is *not* the fix (it freezes
  instead of advecting) — it is the discriminator.
- **D2 — corrector A/B (decides RC4's share).** Force `state[15] = 0` in
  `prepareVolumeCorrection` (env-gate it). Collapse recovering ⇒ RC4 was
  masking/fighting; volume drift curve becomes the RC1/RC5 loss signal.
- **D3 — freeze census (quantifies RC5).** Count cells where
  `predictSummaryCells` skipped advection (`velocity.w == 0` with
  `initialized`) into a spare state word; read back per advance across a band
  sweep. Today this is silent.
- **D4 — wet-row census by size (quantifies RC3).** Per advance, count wet
  rows by `owner.size` at dial 3 vs 8. A jump in size-4/8 wet rows at the
  surface at low dial = the solver surface is quantizing.
- **D5 — migration-zero census (quantifies RC2).** Count faces with
  `count == 0` in `migrateLosassoLaggedVelocity` per epoch, split
  refine/coarsen.
- **D6 — receipt ratio (RC5/authority).** `readReceipt()` advances vs
  completions and `predictedCells` vs `domainVolume` across the sweep; any
  ratio below 1 means whole advances silently lost authority.

## Work packages

**WP-B1 — Total nodal reconstruction (the structural fix; the paper's rule).**
Make `stagedNodalSample`/`velocityAtNode` never fail inside fluid: when a
quadrant's MAC plane is interior to a coarse leaf, resolve the **cell** (the
pressure-row directory lookup exists) and interpolate between that leaf's own
two bounding faces along the axis — the discrete analogue of "temporary
coarsened velocities, coarsest neighboring face as the scale." The nodal
field becomes total over the fluid; `advectLosassoFaces` then advects real
momentum on every face and the `select(0.0, …)` paths become genuinely
unreachable rather than load-bearing. This single change restores property
(2) of the paper and is what makes band width an accuracy dial again.
Gate: D1's arm becomes redundant; mini dam collapses at every dial setting.

**WP-B2 — Never create velocity at zero on refinement.** In
`migrateLosassoLaggedVelocity`, when a sub-position finds no old face
(mid-plane case), fall back to the average of the old parent cell's two
bounding faces along the axis (or the WP-B1 nodal field, once it exists).
Also: on invalid old authority, carry the previous field rather than zeroing
everything. Gate: D5 census = 0 for refine epochs on a moving-front scene.

**WP-B3 — Wetness from an interval, not a point.** Classify size ≥ 2 rows
wet from the dense tracker's per-leaf min phi (the aggregation hierarchy the
summary already maintains) or an arena min/max pair (the topology handoff
already recommends interval-over-mean); a leaf is wet if any covered cell is
wet, with the face's theta doing the sub-leaf work. Gate: D4 census stable
across dials; no all-or-nothing leaf flips at the surface.

**WP-B4 — Close the RC6 fallthrough.** In `publishLosassoCoarseOnlyGhosts`,
a missing-neighbor face whose own row is wet by WP-B3's interval must not
keep an open p=0 face; either find the coarse fluid neighbor (it exists —
the row set just didn't include it: that's an RC3 fix) or treat it as the
conditioned interior case. Gate: no `flags=4` faces below the bulk surface
on a resting tank.

**WP-B5 — Volume control: adopt the regional corrector.** This lane should
follow `docs/losasso-volume-wall-regression-handoff.md` WP-V1 (regional /
main-reservoir-only control), not the live uniform offset. Interim: env-gate
the offset (D2's switch) so band experiments stop being confounded. Gate:
D2's drift curve bounded without the offset once WP-B1/B2 land.

**WP-B6 — Reach contracts instead of coincidences.** (a) Derive the
extension width from `effectiveBand + backtraceCells + 1` with W7 as the
floor, or assert `W ≥ band + 3`; (b) couple `velocityExtensionSweeps ≥ W+1`
or shrink the graph with the sweeps; (c) floor the dry-boundary look-ahead
at 3 cells (`:9076`, `:9421`); (d) make the live dial default AUTO =
authored, so "shrink the band" is an experiment someone runs, not the
shipped state; (e) single-source the protection-width formula; (f) promote
D3's freeze counter to a standing tripwire (the fine path's control[7]
analogue — today the coarse path fails silent).

**WP-B7 — The band-invariance oracle (the regression gate this whole area
lacks).** Mini dam break, dial swept 3→8 (and authored band 1→4): the
front-position trajectory (`FLUID_SPEED_MAP=1` centerline already exists)
must (a) collapse at every setting, (b) converge monotonically toward the
fine result as the band widens. This is the executable form of the user's
invariant and the gate for WP-B1..B5. Run it as a PERF-class lane first
(single arm per dial) before blessing tolerances — the coarse-only D4 gate
is red at HEAD (velocity step 54 / volume step 245, pre-existing) and this
oracle must not inherit that noise.

Suggested order: D1+D2 same session (they're both one-line arms) → WP-B1 →
D5 → WP-B2 → D4 → WP-B3/B4 → WP-B5 → WP-B6/B7 alongside.

## What NOT to do

- **Don't fix it by widening the band or W.** That re-hides RC1-RC5 behind
  the default geometry; the invariant stays broken and the droplet-in-a-vast-
  domain program (band as a shell, cost ∝ n^0.6) needs narrow bands to work.
- **Don't touch the pressure discretization.** It verified clean; the
  distortion is upstream (transport/momentum) and orthogonal (classification).
- **Don't treat the corrector as the root cause.** Disabling it (D2) may
  visibly "fix" the collapse — that's the amplifier coming off, not the
  defect. RC1 is still eating momentum underneath.
- **Don't let D1's fallback ship as the fix.** Frozen velocity is better than
  zero but still not advection; WP-B1 is the fix.
- **Don't conflate the two band knobs.** Authored `interfaceRefinementBandCells`
  sizes structures; the live dial thins the running band. Any experiment must
  state both (plus grading), or it isn't reproducible.
- Standing lane hygiene: never compare walls across tripwire modes; ±5%
  within-arm noise; the lane's ±2-generation nondeterminism means bisecting
  by signature, not step-exact repro.
