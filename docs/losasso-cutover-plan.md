# Losasso cutover plan — phased, gated, no experiments

Supersedes the experiment list in `docs/simplify-power-liquids-handoff.md`.
That doc's pressure-solve update changed the picture: the exact fixed-point
reduction works, warm-started MGPCG converges in 3–4 iterations, and the
one-workgroup executor — not convergence — was the pressure wall. Consequences:

- **Every approximation bet on the projection is dead.** Fixed-cycle MG, CG
  deletion, iteration clamps (old E2), and the volume controller (old E3) are
  removed from the program. The projection stays converged and warm-started;
  it just becomes wide.
- **The signed-integer superaccumulator is a general D4 primitive.** Integer
  accumulation is exactly associative/commutative, hence invariant to workgroup
  partition, dispatch schedule, and D4 row permutation — strictly stronger than
  the canonical-fold fixed-order barrier. Anywhere a canonical fold exists only
  to protect D4 exactness, the superaccumulator can replace it and unlock wide
  parallelism. This is what makes the phases below safe to parallelize.
- **Ordering flips.** The hard part of the wide-solver port was transcribing
  power-diagram arithmetic (topology-LUT stencils, 18/8-term canonical folds,
  edge-channel plumbing) into the row-parallel path. The destination coupling
  (Losasso 2004) has none of that arithmetic. Therefore: **swap the coupling
  first on the existing persistent executor, then build the wide executor
  against the simple operator it will actually ship with.** The power-stencil
  transcription is never done at all.

**Destination.** First-order Losasso-2004 graded coupling with closed-form
coefficients; free-surface shell uniformly fine (transitions only interior and
terrain-cut — terrain needs no fine band, cut-cell fractions are resolution-
independent); plain first-order V-cycle-preconditioned warm-started MGPCG on a
wide executor with exact reductions; fixed-K Jacobi velocity extension on
axis-aligned faces, built once per advance; k-advance topology cadence.
Absent from the new backend by construction (and frozen, not deleted, in the
old one): power descriptors + catalog (14.2 MB), topology LUTs, edge-channel
velocities, local Delaunay tetrahedralizations, the §4.3 two-operator hybrid
boundary smoother, the march-to-fixed-point frontier, the one-workgroup solver
arena and its 65,536-row gate.

## The seam: two coarse backends, one switch

The full 2017 implementation stays permanently revertible. That is realized as
a **coarse-dynamics backend seam**, selected per scene/lane at construction:

- **`power2017`** — the pipeline exactly as it ships today, *frozen*: no new
  features are ported into it, it takes no interest in the phases below, and CI
  keeps it green on its own gate lanes. It is the revert path and the permanent
  physics reference for A/Bs.
- **`losasso`** — the new backend, with its **own pipelines, bind group
  layouts, and channel set**. Reduced bindings are the point: no edge-velocity
  channels, no descriptor/catalog binding, closed-form stencils instead of LUT
  tables, one operator at every level. None of the power machinery is linked
  into this path.

Seam rules that keep the cutover clean:

1. The seam sits at the *coarse level* — everything octree-side (coupling
   operator, preconditioner, pressure executor, velocity storage layout,
   air-support/extension, coarse grading policy) is behind it. Everything
   fine-band-side (fine advection, JFA redistance, narrow-band topology, volume
   correction) and all scene authoring is shared and backend-agnostic.
2. The seam is at **pipeline/layout creation, not inside WGSL**. No
   `if (backend)` in shaders, no superset bind group layouts carrying dead
   bindings through the losasso path — otherwise "reduced bindings" is fiction.
3. The shared fine-band code speaks to the coarse level through one interface
   (velocity sampling for advection/extension targets, φ exchange,
   epoch/topology commit). If a phase needs to widen that interface, widen it
   for both backends in the same change — that is the only ongoing maintenance
   `power2017` is allowed to cost.
4. Grading policy is a backend property: `losasso` *requires* the uniform-fine
   free-surface shell; `power2017` keeps today's grading. A scene runs under
   either backend without re-authoring.

## Gate protocol (applies to every phase)

- **Exact gate:** factor-4 symmetric-expansion, exact D4 (volume, velocity,
  pressure, topology, four-wall contact) through accepted step 67, and the
  240-advance lane at end of phase. Factor 4 is the shipping surface path, so
  this gate exercises fine-band transport and coarse/fine exchange as well as
  the coarse dynamics. Factor 1 remains a compact diagnostic lane, not the
  cutover gate. This gate checks *internal symmetry of the new physics*, so it
  survives physics-changing phases where bit-identity to the old trajectory
  is impossible by design.
- **Trajectory A/B:** factor-4 67-step lane. Not an exact gate (baseline loses
  symmetry at step 1–2). Rule: no field fails earlier than the frozen baseline;
  four-wall contact stays simultaneous. **Re-freeze this baseline at the end of
  each physics-changing phase (2, 3, 5)** — comparing across a physics change
  is meaningless.
- **Standing oracles:** dry-identity zero-RHS (class 4); free-fall drop oracles
  (wall/ceiling/seam sticking); hydrostatic-over-terrain (landed in Phase 0);
  dam-lane air-support counters for Phase 5; solver flags 0, convergence inside
  the iteration budget.
- **Measurement:** wall time, same tripwire mode as baseline, never proxies
  (launch counts, pass counts, gate greenness alone).
- **Backend hygiene:** `power2017` must stay green on its own gate lanes at
  every phase boundary — the revert path is only real if it is continuously
  verified. *Within* the `losasso` backend, no dual paths are carried forward:
  each phase's superseded machinery is absent from that backend, not flagged
  off. The persistent-style executor inside `losasso` is the single sanctioned
  temporary (Phase 4 stage-audit oracle, removed when Phase 4 lands).

## Phases

### Phase 0 — Gates become permanent infrastructure *(days)*
Land as shipping counters/scenes, not probes: (a) the three-bucket transition
audit — free-surface-band / terrain-cut / interior — validated on a scene
constructed to have surface transitions (the Bet-1 lesson: prove the counter
can read nonzero); (b) the hydrostatic-over-terrain oracle scene, authored
D4-symmetric, in the dry-identity family; (c) frozen factor-4 exact contract
and factor-4 trajectory baseline. **Gate:** audit reads correctly on the
constructed scene; oracle scene is class-4 green under today's coupling.

### Phase 1 — Reducer as a standalone primitive *(days)*
Extract the signed radix-256 limb superaccumulator with the fixed decoder
(convert each scaled limb, sum physical values — never form the unscaled
integer in f32) as a library primitive with its adversarial test set: finite-f32
extremes, cancellation, row permutations, multiple partitions, >65,536 rows;
identical integer total before the single rounding; compiles under Naga and
Dawn. **Gate:** primitive tests green; no production behavior change.

### Phase 2 — Grading policy: uniform fine free-surface shell *(days–week)*
Land |φ_liquid| < ~2Δx_fine ⇒ uniformly fine as a *selectable grading policy*
(the one `losasso` will require), and validate it **under `power2017`**, where
additive refinement is legal physics. This prices the policy's capacity cost on
real lanes before the new backend exists, and old vs new can be compared
directly on cells the rule didn't touch. **Gate:** bucket (a) of the audit
reads **0** on symmetric-expansion, droplet-256, and the ocean lane under the
policy; factor-4 exact; cell/page growth bounded (mini: 9,500 of 11,520 pages
today — verify headroom; large lanes: check *authored* capacity, capacity is
not inert). Re-freeze factor-4 under the policy.

### Phase 3 — Stand up the `losasso` backend *(1–2 weeks; the pivot)*
Build the new backend behind the seam: closed-form first-order graded coupling;
§4.3 hybrid preconditioner collapsed to one first-order operator at every level
(plain V-cycle, no boundary-band second-order smoothing); solid faces keep
cut-cell fraction weights (symmetric ⇒ SPD preserved) at any resolution,
including terrain-cut T-junctions; velocities on axis-aligned octree faces
only; the Phase-2 grading policy bound on. Its bind group layouts are authored
fresh from what this operator needs — the reduced-bindings win is realized
here, not in a later cleanup. The executor may initially borrow the
persistent-kernel *shape* (it is the Phase-4 audit oracle), but none of the
power machinery (LUTs, edge channels, descriptors, Delaunay tables) is linked
into this path. `power2017` is untouched. **Gate:** factor-4 exact D4 under
`losasso`; hydrostatic-over-terrain parasitic |u| bounded and decaying (write
the bound down before running); dry-identity green; free-fall drops green;
visual A/B `losasso` vs `power2017` on droplet-256 and ocean — the seam makes
this a config flip. Freeze a factor-4 `losasso` baseline. **Fallback:** any
scene that misses its bound ships on `power2017` via the switch while the
miss is diagnosed; the per-scene escape within `losasso` is *authored*
refinement near the offending terrain (allowed, never required).

### Phase 4 — Wide exact-reduction MGPCG on the simple operator *(1 week)*
Inside `losasso` only: build the row-parallel executor against the Phase-3
operator — closed-form stencils, no LUT, no staged power worksets; the port the
prototype wanted. All global scalars (dots, norms, curvature) through the
Phase-1 primitive. Keep the backend's persistent-style executor solely as a
stage-audit oracle (the audit must show first-mismatch location, as in the
prototype work — expect coefficient arithmetic, not partitioning, if anything
drifts), then remove it from the backend once green; `losasso` carries no
65,536-row gate. `power2017` keeps its persistent executor untouched.
**Gate:** factor-4 exact; factor-4 not-worse vs the Phase-3 `losasso`
baseline; complete pressure phase (setup + solve + exact finishes +
publication) measured interleaved, target ~10 ms vs 54.9 ms.

### Phase 5 — Velocity extension: fixed-K sweeps, built once *(1 week)*
The frontier march's 83,352 face unknowns and twice-per-advance schedule were
power-face artifacts. On axis-aligned faces: K Jacobi sweeps of ∇φ·∇u = 0 over
faces within W cells of the interface (seed set ~26k), built once at the S3e
position; S1 consumes the previous advance's field (rebuild only on epoch
change). Fixed K and W are constants chosen once from instrumentation of
today's fixed-point round count, then hard-coded — no adaptive termination, no
reductions. D4-safe by construction (no marching order). **Gate:** factor-4
exact; free-fall drop oracles (extrapolation quality is exactly what they
isolate); dam-248 air-support counters on the dam lane, not just mini; wall
target 67.4 ms → ≲15 ms. Re-freeze factor-4.

### Phase 6 — Topology cadence, default flip, freeze *(days)*
Inside `losasso`: rebuild the candidate epoch every k advances, padding
expressed as extra dilation rings (band 4 stays canonical); choose k from page
headroom per lane. Replace remaining D4 canonical folds that survive only as
rounding barriers with the Phase-1 primitive where wall time says it pays (the
folds cost 25.5 ms on the mini lane). Then flip the default backend to
`losasso` and formally freeze `power2017`: its ~104 MB power authority and
catalog are simply never allocated under the default, its gate lanes stay in
CI, and its code is off-limits to refactors that aren't seam-interface changes.
**Gate:** factor-4 exact at 240 advances under `losasso`; resident-page
ceiling respected on every lane; `power2017` gate lanes still green
post-flip; final re-profile captured and `symmetric-expansion-frame-anatomy`
re-issued for both backends.

## Expected end state

Pressure 54.9 → ~10 ms (Phase 4), air support 67.4 → ~15 ms (Phase 5), S4/S2
topology partially amortized (Phase 6), minus the catalog/LUT traffic the new
backend never carries. Roughly 238 → ~120–140 ms on the mini lane before any
further work. The shipping path is *simpler* — one coupling, one operator, one
executor, reduced bindings, no LUTs, no edge channels, no fixed-point marches —
while the full 2017 implementation remains one config flip away, frozen and
continuously verified. The remaining frame is then dominated by fine level-set
advection (50 ms) — a correctly-expensive, paper-parity subsystem — which is
the right place to be.

Risks worth naming: Phase 2 capacity on large lanes (authored capacity, not
default); the seam itself is the main Phase-3 engineering risk — if the
fine-band↔coarse interface turns out wider than expected (velocity sampling,
φ exchange, epoch commit), that widening lands as its own reviewed change
touching both backends *before* the backend work continues; Phase 5's fixed K
must be chosen from measurement, not taste, or the free-fall oracles will find
it; and the standing temptation to "just fix one thing" inside `power2017` —
resist it, a drifting reference is worse than no reference.
