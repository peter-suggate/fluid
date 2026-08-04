# Handoff — simplifying the Aanjaneya-2017 scheme for GPU frame steps

**Thesis.** Nearly all of the paper's expensive machinery (power diagram, topology
LUTs, edge-channel velocities, second-order boundary smoothing, marching-order
causality) exists to make one thing safe: *a resolution transition touching the
free surface*. Our scenes never want that configuration — the fine band is a
surface shell and transitions live deep interior/air. Exactness there is
invisible physics. The program: trade exactness at transitions only where it is
physically invisible, while first removing implementation bottlenecks that do
not require a quality trade. Keep the SPGrid pyramid, hierarchy, and fine narrow
band; make pressure data-parallel with exact reductions, and use fixed-count
approximations only when their additional wall win survives the scene gates.

**Ground truth.** `docs/symmetric-expansion-frame-anatomy.md` (captured
2026-08-04, 240 advances, 237.75 ms/advance shipping wall). Honest per-advance
means, not the representative frame:

| Subsystem | ms | % | Note |
|---|--:|--:|---|
| §5 air-support march (velocity extrapolation) | 67.4 | 29.8 | Run **twice** per advance (S1 topology-commit + S3e settled-fine, ~30 ms each). Paper spends 2.5–5 % here. |
| Fine level-set advection | 50.1 | 22.1 | In line with paper. |
| §4.3 MGPCG pressure solve | 54.9 | 24 | 4–6 CG iterations, swings 43.5–61.5 ms. Single persistent workgroup. |
| JFA redistance | 13.6 | 6.0 | **Already simplified** — FMM→JFA is done and is decisively ahead of the paper (23–26 %). Do not re-litigate. |

The two bottlenecks initially looked like the two places the paper's exactness
lives: marching-order extrapolation and converged-to-tolerance projection. The
pressure result below narrows that assessment: convergence is not the expensive
part by necessity; the one-workgroup executor is an implementation choice.

### Pressure-solve update (2026-08-04)

**Status: promising local prototype, not ready for production cutover.**

The original assessment bundled two different concerns: CG's global scalar
dependencies and the single-workgroup implementation chosen to make those
dependencies deterministic. Dawn testing shows they should be separated.

- The outer MGPCG recurrence can be widened across the GPU. Every finite f32
  reduction term is deposited by bit decomposition into signed radix-256
  integer limbs. Integer addition makes the reduction exactly associative and
  commutative, so it is invariant to workgroup partitioning and to a D4 row
  permutation. This is stronger than a fixed-order floating-point fold.
- The row-capacity planner now admits 65,537 rows; the old 65,536 executor gate
  is no longer structural. The remaining limit is the ordinary WebGPU
  one-dimensional row dispatch, not a one-workgroup solver arena.
- The first fixed-point decoder was wrong: it formed the enormous unscaled
  integer in f32 before applying the binary exponent, producing infinity and a
  stage-5 nonfinite failure. Converting each already-scaled limb and summing the
  physical values fixes it. After that fix, the 67-step Dawn run completed every
  solve in 3–4 iterations with solver flags 0.
- The current solver is already unconditionally warm-started from the previous
  accepted pressure remapped by row identity. Therefore former E1 is not an
  unclaimed optimization; only a cold-start measurement arm remains useful.
- Restoring the deleted historical row-parallel executor is not, by itself, a
  behavior-preserving cutover. The production persistent kernel has since gained
  D4-canonical stencil folds, staged worksets, and newer authority ABI details.
  Those changes must be transcribed into the wide path rather than assuming the
  old implementation is still an oracle.

The stage audit localized that last point. On the current factor-4 scene, the
initial residual and Section 6.3 case IDs were exact, but the published diagonal
was already asymmetric by 2.98e-7 (560 comparisons), and the first hybrid
preconditioner output differed by as much as 0.001953125. The final pressure was
still exact at step 1, so the exact outer reduction is doing its job; the drift
enters through coefficient/preconditioner arithmetic, not row partitioning.

The 67-step factor-4 A/B also proves that this lane cannot presently be treated
as a green exact-D4 cutover oracle. The baseline persistent solver first loses
volume symmetry at step 2, velocity/pressure at step 1, RHS at step 2, and
diagonal/topology at step 35; all four walls contact together at step 61. The
wide prototype first loses volume at step 2, velocity/pressure at step 12, RHS
at step 13, and diagonal/topology at step 31; all walls contact together at step
60. The prototype improves several field windows, but it is not baseline-
identical and moves topology and wall contact. That is a red A/B until explained.

This does **not** weaken the fixed-point-reduction result. It changes the
cutover sequence: prove the reducer independently, port the current stencil and
preconditioner arithmetic, then run the scene gate before switching production.
The prototype currently exists only in the local workspace; a visually good
browser result is useful evidence but is not proof that this code is deployed.

---

## Structural moves (the destination)

1. **Ban transitions from the *free-surface* band only; first-order
   Losasso-2004 coupling everywhere else.** Grading rule: uniform fine wherever
   |φ_liquid| < ~2Δx_fine. Then T-junction flux error is O(h) *where nobody can
   see it*, coupling coefficients become closed-form (no topology LUT, no edge
   channels), and the operator stays SPD.
   **Terrain/scenery explicitly does NOT require the fine band.** Solid
   boundaries are handled by the cut-cell/ghost-fluid machinery (face area
   fractions + solid SDF at vertices), which is resolution-independent and
   orthogonal to the power diagram: a coarse cell cut by terrain stays legal,
   and a T-junction face partially covered by solid keeps a symmetric
   coefficient (weight the flux by the unblocked fraction). Where terrain meets
   the free surface (shorelines, wetting fronts) the cells are already inside
   the fine shell via the |φ| criterion, so both boundary types are uniformly
   resolved there without any extra rule. What we accept: O(h) parasitic
   currents along terrain at coarse-region T-junctions — submerged or in air,
   away from the visible surface. Refinement near terrain remains *available*
   (authored, where flow detail matters, as in the paper's river scene) but is
   never *required* for correctness. The paper itself concedes the
   interior-first-order direction in §8.
2. **Keep warm-started MGPCG, but make only its global reductions exact and
   row-parallel.** The stencil applies and Jacobi/Chebyshev sweeps do not need a
   single workgroup. Dot products, residual norms, and curvature use signed
   integer superaccumulators, so the scalar result is independent of row
   partition and dispatch scheduling. This removes the 65,536-row executor gate
   without buying speed by accepting a worse projection. Fixed-cycle multigrid
   remains a later quality/performance experiment, not the default destination.
3. **Fixed-sweep extrapolation over a width-limited band; drop
   march-to-fixed-point.** K Jacobi sweeps of ∇φ·∇u = 0 over faces within W
   cells of the interface. Fixed pass count, symmetric by construction.
4. **Amortize topology.** Rebuild band/epoch every k advances with band padding
   sized to k·CFL. The band is conservative; the physics can't tell.

## Revised pressure-track sequence

This track is now a prerequisite chain, not a collection of independent bets:

1. **Freeze two different scene contracts.** Run the coarse/factor-1 symmetric
   expansion lane through accepted step 67 as the exact D4 gate. Separately,
   retain the current factor-4 67-step baseline as an observational A/B with its
   already-known failure steps and step-61 simultaneous wall contact. Do not
   silently call the red factor-4 lane an exact gate.
2. **Prove the reducer in isolation.** Test adversarial finite f32 terms,
   positive/negative cancellation, row permutations, several workgroup
   partitions, and a row capacity above 65,536. The integer total must be
   identical before the single f32 rounding. Compile with Naga and Dawn.
3. **Port forward, do not merely restore.** Bring the current persistent
   kernel's D4-canonical 18- and 8-term folds, staged stencil columns, workset
   rules, smoother order, and accepted-authority ABI into the row-parallel
   operator and Section 4.3 preconditioner. A stage audit must show where any
   first mismatch appears.
4. **Run correctness before wall time.** The factor-1 lane must remain exact
   through step 67. The factor-4 A/B must not move a baseline failure earlier,
   alter accepted topology unexpectedly, or move simultaneous wall contact
   without a written explanation. Solver flags must remain zero and every
   accepted solve must converge inside the existing iteration budget.
5. **Then measure the wall A/B.** Interleave persistent and wide arms in the
   same tripwire/timing mode. Attribute the complete pressure phase, including
   setup, exact reduction finishes, and publication. The target is roughly
   54.9–57.5 ms → ~10 ms, not merely a faster reduction microbenchmark.
6. **Cut over only after 1–5.** Keep the old persistent executor as an A/B oracle
   until the wide path passes both contracts. Remove the 65,536 production gate
   only with the successful cutover, not from a prototype that has changed the
   scene trajectory.
7. **Reassess approximation bets afterward.** If MGPCG is already near 10 ms,
   fixed-iteration PCG or CG deletion has much less upside and should compete
   against the air-support and topology work on measured frame impact.

---

## Experiments — in-scheme, D4-gated

All run on the `symmetric-expansion` lane
(`tools/profile-mini-dam-xctrace.ts --lane=symmetric-expansion --steps=240`),
which remains the end-state oracle: volume, velocity, pressure, topology and
four-wall contact must stay exactly D4-symmetric. The pressure-track prerequisites
above are sequential. E3–E8 below remain independently revertable after their
stated prerequisites.

**Gates for every experiment** (fail any → the experiment is red, not "close"):
- **D4, immediate cutover gate**: the factor-1 symmetric-expansion contract is
  exact through step 67. The 240-advance exact lane remains the end-state gate.
- **Factor-4 trajectory A/B**: until its pre-existing post-step asymmetry is
  repaired separately, no field may fail earlier than the frozen baseline and
  all four walls must still contact on one accepted step. Any changed failure
  or contact step requires an explanation before proceeding.
- **Dry-identity zero-RHS**: still scenes produce zero pressure increment
  (class-4 contract) — the natural oracle for warm-started δp.
- **Free-fall drop oracles**: no wall/ceiling/seam sticking regressions.
- **Volume drift** over 240 advances vs baseline (matters for E2/E3).
- **Wall time in the same tripwire mode as baseline** — `failfast` costs ~27 %;
  never compare across modes.
- Measure **wall time**, not launch counts or pass counts (the Bet-1 lesson:
  a green gate coexisted with a 54 % serialization regression).

### E1 — Cold-start A/B; close the supposed warm-start opportunity
The accepted row-identity transaction already remaps the previous pressure into
both seed banks, and MGPCG already starts from that seed. Preserve this as a
documented invariant. Use the existing cold-start measurement arm only to price
what would be lost by removing it. **Measure:** warm vs cold iteration-count
distribution and complete solve wall. **Expected:** the warm arm remains the
shipping arm; this experiment does not itself produce a cutover or a new win.

### E2 — Iteration clamp: exactly N iterations, no convergence check
Force the existing solver to run exactly N ∈ {2, 3} PCG iterations (keep the
preconditioner). This *simulates* fixed-cycle MG without touching structure.
**Measure:** wall, volume drift, visual diff on droplet-256 impact frames.
**Predict:** solve cost becomes flat ~2/5 of today's mean; the failure mode, if
any, is a one-frame "squish" at impacts. **D4 note:** clamping doesn't remove
the dot products yet — reduction order must stay canonical. The full move
(delete CG, keep V-cycles) removes that rounding-hazard class entirely; the
clamp tells us whether N is enough before we commit. **Revised priority:** run
only after the exact wide MGPCG wall is known. If the complete solve is already
~10 ms, this approximation probably loses to E4/E5 on value and risk.

### E3 — Volume controller, neutrality first
Add the global divergence-offset controller and verify it is a **no-op when
projection is converged** (offset ≈ 0 with today's solver, and exactly 0 on the
dry-identity scenes). Land it neutral; it derisks E2's drift before E2 ships.
Independent of E1/E2.

### E4 — Air-support march once per advance, not twice
S1 rebuilds air support against the committed epoch (~30 ms); S3e rebuilds it
against the settled fine generation (~30 ms). Instrument which consumers read
each rebuild, then skip S1's when the topology epoch is unchanged (reuse the
previous advance's S3e result). **Measure:** wall, and the dam-248 air-support
counters (corridor-island vs bug discrimination). **History:** the packet-BVH
cutover of this march was reverted (8ea9e27) — this experiment changes
*scheduling*, not the march itself, which is why it's safer. **Predict:** up to
−30 ms (−13 %) on quiescent advances. **Risk:** air support is the first domino
on the dam lane; run that lane's counters, not just the mini.

### E5 — Instrument the march's fixed point; then clamp sweeps + band width
First instrumentation-only: how many relaxation rounds does the frontier march
take to reach its fixed point today, and what fraction of the 83,352 face
unknowns (26,120 seeded) ever change after round K? Then: clamp to K sweeps
and restrict to faces within W cells of the interface. **Measure:** wall,
free-fall drop oracles (extrapolated velocity quality is exactly what they
isolate). **Predict:** if the fixed point is already reached in few rounds, the
win is the band restriction (83k → ~26k unknowns), not the clamp. Fixed-K
Jacobi is D4-safe by construction — no marching order to break symmetry.

### E6 — Surface-clean grading audit *(instrumentation only, no behavior change)*
Count power faces and "same-or-coarser" descriptors per advance, in **three
buckets**: (a) within |φ_liquid| < 2Δx of the free surface, (b) solid-cut
(terrain-intersected) elsewhere, (c) fully interior/air. Mini already publishes
only 26 same-or-coarser of 2,100 descriptors — likely near-zero in bucket (a).
Run the same audit on droplet-256 and the ocean lane (320×96×80, sculpted
terrain — the lane where bucket (b) is large by design). **If (a) ≈ 0
everywhere:** the first-order coupling swap can be scoped; bucket (b) sizes the
terrain-artifact exposure that E8 then quantifies, and bucket (c) is free.
**If (a) is not ≈ 0:** we learn exactly which scenes pay for the power diagram
at the one place it matters. Trust the counter, not the audit's existence — the
Bet-1 audit reported 0 while 68 existed; validate the probe on a scene
constructed to have surface transitions.

### E7 — Topology cadence: rebuild candidate epoch every k advances
Pad the fine band by k·CFL cells and run S4 (candidate build, 13 ms) and the
S2 topology delta every k-th advance only. Band 4 is canonical for this lane
(c8c84ee) — the pad must be expressed as extra dilation rings, not a band-width
change. **Measure:** wall amortized over k, resident-page growth (9,500 of
11,520 capacity today — check headroom before choosing k), D4 topology
symmetry. **Capacity is not inert**: check the large lane's authored capacity
before porting any k there.

### E8 — Hydrostatic-over-terrain oracle *(new scene, no solver change)*
The classic failure first-order T-junction coupling is accused of: parasitic
currents in a *still* pool. Build a small still-water-over-sculpted-terrain
scene where transitions deliberately land on terrain-cut coarse cells (bucket
(b) of E6), and add it to the dry-identity family: the class-4 zero-RHS
contract must hold under today's power-diagram coupling (baseline), and the
oracle then measures max |u| growth and decay under any future coupling change.
**Why now:** it costs nothing, it makes the terrain-artifact cost of structural
move 1 a measured number instead of an argument, and it doubles as a large-lane
still-scene probe (both large-lane reds converge on fine transport failing to
publish on still scenes — this oracle will trip on that too, so land it with
the tripwire provenance visible). **D4 note:** author the terrain D4-symmetric
so the same scene serves both oracles.

---

## Expected shape of the win

The exact wide MGPCG track attacks the 24 % pressure subsystem without changing
the projection tolerance. Hitting the ~10 ms target removes roughly 45 ms from
the ~238 ms frame, landing near ~193 ms before any approximation experiment.
E4 + E5 then attack the 29.8 % air-support subsystem and can plausibly move the
frame toward ~160 ms. E1 is now measurement-only, and E2 should be attempted
only if its remaining upside is compelling after the wide result. E6 scopes the
larger prize — deleting the power-diagram frontier machinery — but that is a
cutover, not an experiment, and should only be scoped after the audit lands.

A disappointing A/B here is a lead, not a verdict: explain it, then take an
alternative route to the same structural goal. Do not stash/checkout/reset in
this worktree to get baselines — use the lane's baseline capture instead.
