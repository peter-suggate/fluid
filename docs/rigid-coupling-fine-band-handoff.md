# Rigid coupling + fine-band fidelity handoff

Two observed symptoms, one research pass (Losasso 2004, Ando & Batty 2020) and a code
audit of the production Losasso lane behind each. Findings are code-verified with
file:line citations; the three most load-bearing claims were re-checked by hand.

- Symptom A: dropping a heavy rigid body into the fluid spreads the fluid far and wide.
- Symptom B: the fine band does not deliver the hoped-for visual fidelity.

---

## Part A — rigid coupling: why a heavy drop "explodes" the fluid

### Diagnosis (ranked by causal weight)

**A1. Publication poisoning freezes the solid BC at impact velocity (top cause).**
A wet pressure row whose faces are all fully blocked gets `diagonal = 0`, and
`sortLosassoIncidences` treats that as a fatal header error:
`webgpu-octree-losasso-backend.wgsl.ts:594-595` —
`if(!valid||…||!(diagonalValue>0.)){ fail(ERROR_HEADER); … }`.
`fail` marks the entire candidate publication invalid (`finishLosassoPublication`,
`:638-655`), and the ready-commit validator refuses to flip
(`webgpu-octree-losasso-authority-commit.wgsl.ts:42-54`). A heavy body guarantees
fully-enclosed wet cells within a frame or two of entry (interior cells rasterize to
solid fraction exactly 1.0 → `openFraction` 0 on all six faces), and the enclosed plug
never dries because phi is never carved (A2). Result: **every subsequent candidate also
fails**, the face graph accepted at impact — with `normalVelocity ≈ −v_impact` across
the body footprint — stays live indefinitely, and `divergenceLosassoRows`
(`webgpu-octree-losasso-dynamics.wgsl.ts:423-456`) demands ~`A_footprint · v_impact` of
outward flux **every substep**: a continuous volume/momentum pump of order 0.1–0.5 m³/s
until the body leaves. No GPU test drops a rigid body into Losasso fluid, so this is
unexercised.

**A2. Phi is never carved → sealed liquid plug → ~2× volume injection.**
Wet classification is phi-only (`currentPressureOwnerWet`, `webgpu-octree.ts:8906-8928`);
no kernel subtracts the body SDF from phi. Water inside the body stays wet, advects
*with* the body (all its faces carry `u_solid`), while the projection pushes a full
submerged-body-volume of water outward — the displaced volume exists twice. The
safeguard written for exactly this (`relaxSolidPhi`, comment at
`webgpu-uniform-eulerian.ts:2044-2051`) only dispatches when `!this.octreeProjection`
and its shader source is the empty string (`webgpu-uniform-eulerian.ts:103`) — **dead
code in the production lane**.

**A3. The body feels almost no fluid resistance.**
`integrate` (`webgpu-rigid-body.ts:194-201`) reads wet-occupancy words
`exchange[base+6..9]` for buoyancy/drag/added-mass, but nothing in the octree lane ever
writes them → all three are zero. The only resistance is the pressure-reaction impulse
(`webgpu-octree-losasso-rigid-pressure-reaction.ts:104-125`), which is computed against
the *accepted* (frozen, per A1) face graph while occupancy is fresh — so the reaction
shrinks as the body sinks past the frozen graph. The body plunges unphysically fast,
which scales the displacement rate and splash violence directly.

**A4. Face blockage is max-of-cell-fractions, not a face-area fraction.**
`conditionLosassoFaces` (`webgpu-octree-losasso-backend.wgsl.ts:424-455`) sets
`solid = max(adjacent cell fractions)` per fine column — every face of a partially
covered cell counts as blocked and moving at body velocity, including tangential and
trailing faces the surface does not cut. The imposed solid flux is over-wide by roughly
one cell shell around the body.

**A5. BC staleness.** Face `openFraction`/`normalVelocity` refresh only at topology
candidate acceptance (default cadence 1 advance, `octree-coarse-backend.ts:158-163`),
and the body integrates once per frame with the full frame delta
(`webgpu-uniform-eulerian.ts:2053`) while the fluid substeps — occupancy teleports
1–3 finest cells per frame for a fast body; the swept-bounds machinery serves rendering
residency only (`svo-primitive-motion.ts:85-86,147-153`).

### What the papers prescribe

- **Losasso 2004** treats solids in one paragraph: clip the normal velocity component so
  it is *separating*, Neumann on faces that intersect the object, drop the face's flux
  term symmetrically from both rows. No moving-solid formulation, no cut cells, no
  two-way coupling. Not much help here beyond confirming the Neumann drop must not
  destroy the row.
- **Ando & Batty 2020** use the standard variational cut-cell form: pressure system
  `−[∇]ᵀ[V][A][F][∇]{p}` where `[A]` is per-face **non-solid area fractions**
  (Ng 2009) — genuinely per-face, not cell-fraction max. Crucially, for degenerate rows
  where all adjacent faces are solid/degenerate they do **not** fail: they floor the
  face weight at a tiny positive value (`W_k = 10⁻²`), noting a zero weight "is
  equivalent to assuming a solid face in that position", and show no visible artifact
  results. They extrapolate phi and velocity *into* solids for interpolation, and their
  sizing function refines near liquid–solid contact via `∇·∇(ϕ + ϕ_solid)`.

### Work items (R-series, in causal order)

**R1 — enclosed wet rows must not poison the publication.**
Wet classification consults the already-resident `solidCells` fraction: a cell with
solid fraction ≥ threshold (start at 1.0, consider ≥ 0.9) is excluded from the pressure
system (classified non-wet). Belt-and-braces per Ando-Batty: in
`sortLosassoIncidences`, a row whose diagonal is 0 *because all faces are
solid-blocked* gets an identity row (or ε = 10⁻² face-weight floor) instead of
`fail(ERROR_HEADER)` — the fail path remains for genuinely malformed headers.
*Acceptance:* new GPU test drops a dense rigid sphere into a Losasso tank; assert
candidate publications keep committing (generation counter advances) throughout
submersion, and total liquid volume stays within corrector tolerance.

**R2 — carve the plug.**
A fine-band pass (piggyback on the JFA redistance encode,
`webgpu-octree-fine-levelset-redistance.ts`) clamps fine phi against the rigid SDF:
`phi = max(phi, −phi_solid)` inside bodies, restricted to the resident band. The global
volume corrector's target must subtract the submerged solid volume, or carving will be
re-inflated elsewhere (`webgpu-octree-fine-levelset-volume.ts:38-51` is cadence-1).
Per Ando-Batty §5.1, keep *extrapolated* phi values available for interpolation near
the boundary rather than hard air values (extend the existing velocity-extension band
pattern to phi if interpolation artifacts appear).
*Acceptance:* the R1 drop test additionally asserts no wet cells with solid fraction
1.0 after two frames of submersion, and volume error bounded during entry.

**R3 — give the body its fluid forces back.**
The pressure impulse already contains hydrostatic buoyancy (gravity is applied to face
velocities pre-solve, `webgpu-octree-losasso-dynamics.wgsl.ts:396-397`), so do **not**
wire the analytic-buoyancy words as-is — that double-counts (noted in audit; see
`webgpu-rigid-body.ts:194-201`). Instead: (a) compute wet-occupancy in the reaction
kernel and use it for drag + added-mass only; (b) make the reaction sample the same
epoch of face data as occupancy (fresh-vs-stale mixing goes away largely via R1).
*Acceptance:* dropped-sphere terminal behavior — sphere decelerates on entry and
reaches a bounded depth; compare against the CPU-reference lane qualitatively.

**R4 — true per-face area fractions.**
Replace max-of-cell-fractions in `conditionLosassoFaces` with a face-area fraction
sampled from the rigid SDF on the face plane (the structured lane's vertex-SDF aperture
`clamp(0.5 + sdf/h, 0, 1)` at `webgpu-octree-structured-boundary.ts:786` is the model;
4 sub-samples per fine face column is enough). This aligns the Losasso lane with the
Ando-Batty `[A]` matrix and stops tangential/trailing faces from carrying body flux.
*Acceptance:* existing rigid tests stay green; drop-test splash radius shrinks
measurably (record before/after plate).

**R5 — BC freshness for fast bodies.**
Re-evaluate face `normalVelocity` per substep from the analytic body state (cheap — the
shape is analytic; only `openFraction` needs the rasterized topology). Substep the body
integrator with the fluid substeps instead of one full-frame Euler step. Swept
occupancy is optional follow-on; with R1–R4 landed, re-measure before paying for it.

---

## Part B — fine band: why it under-delivers visually

### Diagnosis (ranked)

**B1. The band refines the surface's *representation*, not its *content*.**
The fine SPGrid stores phi only (`octree-fine-levelset-bricks.ts:1-8`); velocity and
pressure remain at the 25 mm finest octree cells. Fine phi is passively advected by
trilinearly interpolated coarse velocity
(`webgpu-octree-losasso-fine-transport.wgsl.ts:122-171`). A feature needs ~4 velocity
cells to be dynamically alive → smallest self-generated surface feature ≈ 10 cm. The
hero reference's ripple train (~5 cm) and plunge stream (~4 cm) sit at or below the
coarse dynamic floor — `hero-garden-scene.ts:27-33` states this exactly. The band can
sharpen a smooth 25 mm interface (worth having: fine cell ≈ 10 px vs coarse ≈ 40 px at
the hero camera) but cannot invent 6 mm structure. **Both papers agree this is
structural**: in Losasso 2004 and Ando-Batty 2020 the fine cells near the surface carry
*velocity and pressure too* — neither paper has a phi-only band; Ando-Batty explicitly
contrast against exactly this two-mesh design.

**B2. Shading throws away most of the 4× that was won.**
At factor 4 the drawn mesh gets one flat normal per 6.25 mm cube
(`webgpu-water-global-fine-tetra.ts:118`) — the per-vertex Gaussian normal compensator
explicitly bails for factor ≠ 1 to preserve the blessed baseline
(`webgpu-water-global-fine-tetra.ts:71-74`). Refraction, specular, and the caustic
Jacobian all consume these normals (`webgpu-water-pipeline.ts:959-977,1103-1212`), so
perceived curvature is faceted at fine-cube scale. This is also the cheapest fix in the
whole document, and it compounds with the hero-1000x finding that water shading is 55%
of the pixel gap.

**B3. First-order fine transport erodes what detail does arise.**
Fine phi transport is first-order semi-Lagrangian with trilinear resampling every
step; BFECC/MacCormack exists but is gated `fineFactor === 1 && env`
(`webgpu-octree-fine-levelset-transport.ts:60-70,333`). 1–2-fine-cell features decay
within a few advances; JFA redistance and the cadence-1 volume offset prune further.

**B4. Whole-frame fallback cliffs can silently draw coarse or stale surfaces.**
`validCurrentPublication()` (`webgpu-water-global-fine-classify.ts:33-62`) rejects the
entire fine extraction unless four generations agree; rejected frames draw
`global-fine-coarse` or retained meshes. Given the documented Losasso lifecycle bugs
(topology-growth handoff), stretches of a run may genuinely not be showing fine phi at
all — and Symptom A's publication poisoning (A1) *causes* exactly this: while the
candidate is poisoned, topology freezes, so a rigid-body drop also degrades the drawn
surface.

**B5. Band placement is distance-only.**
Seeding/retirement is purely geometric distance-to-interface
(`webgpu-octree-fine-levelset-topology.ts:156-184`). No curvature, velocity, or
solid-proximity term. Losasso's smoke criteria include "near objects"; Ando-Batty's
sizing is `γ_ϕ|∇·∇(ϕ+ϕ_solid)| + γ_u·|velocity-derivative|`, advected and decayed over
time so emerging detail isn't retired next step. Sub-band phenomena (droplets < 2 fine
cells, sheets < 1) cannot exist in trilinear phi regardless.

### Work items (F-series)

**F0 — measure before judging: source-mode HUD counter.**
A per-run counter of frames drawn per surface source (`global-fine`, retained,
`global-fine-coarse`) surfaced in the HUD/fidelity report. If a meaningful fraction of
frames aren't drawing fine phi, fix that (and A1) before evaluating band fidelity.

**F1 — extend the normal compensator to factor 4/8.**
Lift the factor-1 gate in `filteredNormalAt` behind an env flag (baseline preservation
is byte-for-byte by intent — re-bless the hero fidelity lane after). Expected: the
single largest per-pixel win available from the band as-built; curvature-driven
specular and caustics stop quantizing at 6.25 mm facets.

**F2 — enable bounded MacCormack fine transport at factor 4.**
Lift the `fineFactor === 1` gate on the corrector (same env-flag + re-bless pattern).
The coarse velocity advection already runs bounded MacCormack
(`webgpu-octree-losasso-dynamics.ts:196`); the surface field is where second order is
visually load-bearing.

**F3 — sizing-driven placement (Ando-Batty §6).**
Add a sizing term to band/octree refinement: curvature of `(phi + phi_solid)` plus the
diagonal velocity-derivative magnitude, evaluated near the surface, propagated by the
existing dilation machinery, and *temporally advected with decay*
(`max(R(Δt)·S_advected, S_evaluated)`, R = 0.9^(Δt/0.01 s)) so splash-born detail
persists. This concentrates the finest octree cells (hence real dynamics) at the plunge
point, contact lines, and high-curvature ripples instead of uniformly along the band.

**F4 — the content axis (structural; decide direction before building).**
Two routes to sub-25 mm *dynamics*, mutually exclusive in practice:
  (a) **Paper-faithful:** one deeper octree level (12.5 mm velocity/pressure) in a
      sizing-selected sub-band — what both papers do. Cost: touches the nine contracts
      pinning `fineFactor`, the solve, and the activity program; large.
  (b) **Program-aligned:** keep 25 mm dynamics and synthesize sub-coarse detail as a
      displacement/detail field on the fine surface (curvature- and activity-driven
      ripple synthesis), per the existing hero-1000x direction ("detail is a field,
      never a smaller cellSize"), plus wiring secondary particles to the global-fine
      source for spray.
Recommendation: (b) first — it is the standing program direction, it is orthogonal to
R-series and F1–F3, and F1+F2 must land first anyway or synthesized detail will be
faceted and eroded by transport. Revisit (a) only if (b) plateaus below the reference.

---

## Sequencing

1. **F0 + R1** (diagnosis + the frozen-publication fix): R1 alone likely removes the
   worst of Symptom A *and* a chunk of Symptom B's intermittent coarseness.
2. **R2 + R3** (plug carve + body forces): completes the physical entry behavior.
3. **F1 + F2** (normals + transport order): the cheap fidelity wins on the band as-built.
4. **R4 + R5** (face fractions + freshness): correctness polish, measured against the
   R1 drop-test plate.
5. **F3**, then the **F4 decision**.

Every gate-lifting change (F1, F2) follows the env-flag + re-bless pattern the
baselines require; every R item lands with the drop-test lane so this stays exercised.
