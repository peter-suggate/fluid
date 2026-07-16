# Handoff: quadtree tall-cell dam-break quality — diagnosis and work plan (2026-07-17)

Supersedes the prioritization in `docs/quadtree-handoff-plan-2026-07-16.md` and updates
`docs/quadtree-numerical-audit-2026-07-16.md` with the current tree state. Paper:
Narita et al., *Quadtree Tall Cells for Eulerian Liquid Simulation*, SIGGRAPH 2025
(`tmp/pdfs/sg2025narita.txt`), building on Chentanez & Müller 2011
(`tmp/pdfs/tall-cells/paper.txt`) and Ando & Batty 2020. We deliberately target a
GPU-resident (WebGPU) implementation — "conformance" below never means "port the paper's
CPU architecture".

## Implementation report (2026-07-17)

P0–P6 and D1–D5 are implemented in this tree. D6 remains optional and P7 remains the
post-stability strategic workstream; neither is included in this change.

| Item | Implemented result |
|---|---|
| P0 | Captured the pre-change and post-change one-second records in `tmp/tall-cell-audit/quadtree-dam-break-handoff-2026-07-17.jsonl`; unit, type, shader, build, settled-tank, and dam-break checks were rerun. |
| P1 | The Eq. 25 matrix scale remains capped at 100, while the velocity correction uses the shared `maximumVelocityUpdateFluidScale = 20` cap. |
| P2 | Debris culling is on by default and remains explicitly disableable; its intervention count is published to diagnostics and the viewport legend. |
| P3 | CPU profile rebuilds and both GPU construction/packing paths carry centre/min/max φ profiles. Centre φ remains the represented sample value; the footprint extrema conservatively force cubic interface bands. |
| P4 | Proactive substeps can rise to 64. A final GPU velocity pass clamps component and vector CFL to 0.9 and atomically counts interventions. |
| P5 | Pressure MLS no longer rejects air queries. Near-interface air candidates use bounded linear ghost pressures and retain the solved adaptive-face mean. |
| P6 | CPU/GPU sizing weights are shared and now include speed-gradient and Froude-like moving-front demand. The measured active sample count fell from 15,248 to 12,710 rather than increasing toward the 2× limit. |
| D1–D5 | The slice overlay now distinguishes dry tall cells and has Structure, CFL, Speed, Coverage, φ, Divergence, and Pressure modes. Divergence is computed after projection into a persistent GPU texture; pressure uses the persistent mapped-pressure texture. The legend exposes debris, clamp, pressure, topology, blocked-frame, and VOF-recovery state. |

The last valid one-second native-GPU sample after P1–P6 measured a 4.390 m/s peak,
0.900 peak component CFL, 1.731% peak exact-volume drift, two wet components, and a
0.99996 minimum dominant-component fraction. Final exact-volume drift was −0.904% and
the largest component contained 25,315 of 25,316 wet cells. This clears the numerical
speed, CFL, volume, and connectedness bars. That sample did record 32 blocked rebuild
attempts after the GPU sparse pack fell back to the CPU worker; the packer now sizes
retries from observed CSR and matrix requirements as well as face count.

The current host's native WebGPU module subsequently stopped executing even a minimal
one-workgroup compute shader (buffer copies still execute and validation reports no
shader error). Therefore that all-zero run is excluded from numerical comparison and a
fresh three-method IoU measurement remains a machine-runtime verification item. The
most recent valid three-method value before the final controller/front-sizing changes
was 0.581 wet-IoU at one second, below the aspirational 0.6 visual-parity bar. The code
and shader validators remain deterministic and green; do not treat a zero-velocity,
zero-rebuild native sample as a physical result.

## 0. The evidence this doc is based on

Two screenshots of the dam-break-ui scene (corner column, 92% height,
`lib/initial-fluid.ts:14`) at t ≈ 0.624 s:

1. **Water surface**: a near-symmetric fragmented mound with a spray fountain punching
   vertically to the domain ceiling. A corner dam break at this time should be a
   *directed surge wave* crossing the tank diagonally — the bulk shape is wrong
   (horizontal momentum lost), the surface is shredded into debris, and there is a
   localized energy injection feeding the fountain.
2. **Grid overlay (Z slice, structure mode)**: bright-teal liquid tall cells at the
   bottom (expected), a blue/pale cubic band around the surface (expected), and
   dark-teal *tall* columns spanning from the cubic band to the domain ceiling —
   including columns in the splash region that are a **single full-height tall cell with
   no cubic band at all**.

Code findings below were verified by direct read in the current working tree
(uncommitted state included). Quantitative history:

- Pre-fix @1 s numbers (audit doc §1, 2026-07-16): wet-IoU vs uniform 0.29, 367 φ
  components, peak speed 7.8 m/s, CFL 1.42, volume drift 0.04%.
- Most recent JSONL for the quadtree method is `tmp/tall-cell-audit/pf-quadtree.jsonl` /
  `pf-all.jsonl` (Jul 15, **0.2 s only**): CFL 0.74, peak 3.6 m/s, 1 component, converged
  residual 7.9e-9, all invariants pass. i.e. the method is clean early and degrades
  during the splash phase.
- **There is no post-fix ≥1 s measurement.** The screenshots are the freshest evidence.
  Re-baselining is task P0.

## 1. What has already landed (do not redo)

Most of the 2026-07-16 remediation plan (W-items) is now in the tree. Verified state:

| Prior finding | Status now | Evidence |
|---|---|---|
| φ advected once per rebuild, accumulated dt (A2) | **FIXED** — resident GPU φ ping-pong, bounded MacCormack (RK2 backtrace, 8-corner clamp) every substep with per-substep dt | `lib/webgpu-uniform-eulerian.ts:468,515`; `lib/webgpu-quadtree-builder.ts:135-169,469-476` |
| MAC staggering bug in φ backtrace (B2) | **FIXED** — adjacent-face averaging | `lib/webgpu-quadtree-builder.ts:78-87` |
| Free-surface BC frozen between rebuilds (A1) | **FIXED** — `refreshFaces` (Eq. 25 fluid scale) + `refreshRows` (DOF activity) run every solve; only the DOF/face *layout* is ≤2 steps stale | `lib/webgpu-quadtree-tall-cell.ts:436-449,483-506,1793`; `lib/webgpu-uniform-eulerian.ts:430-432` |
| Rebuild cadence 8–32 steps | **FIXED** — cadence 1, pipelined with `quadtreeTopologyStaleSteps = 2`, blocks past that | `lib/webgpu-uniform-eulerian.ts:112,116,366-368,406-440` |
| All air–air faces zeroed; no air-velocity extrapolation | **FIXED** — only far-field air (both φ > 2h) zeroed; 3-sweep φ-driven ring extrapolation after every projection | `lib/webgpu-quadtree-tall-cell.ts:879` (zeroing), `:922-960,1842-1851` (extrapolation) |
| VOF co-owns fluid identity (B1) | **MOSTLY FIXED** — construction/sizing/seeding are φ-only; VOF reconciliation is an armed circuit breaker (arms at −10% represented volume, releases at −2%), inert in healthy runs | `lib/webgpu-quadtree-builder.ts:23-26,206-217,281-308,492-494,576-578`; `lib/methods/quadtree-tall-cell.ts:9,49` |
| IC(0) preconditioner too slow | **FIXED** — default is `poly` (28× pressure-solve win measured); `ic0`/`blockic`/`line` selectable | `lib/methods/quadtree-tall-cell.ts:8,12,25` |
| Rigid impulses once per rebuild (A4) | **FIXED** — `coupleImpulse` every solve + per-step readback | `lib/webgpu-quadtree-tall-cell.ts:1813,1866-1890` |
| Per-step redistance | **FIXED** — JFA seed→flood→finalize every substep, narrow band preserved verbatim | `lib/webgpu-quadtree-builder.ts:251-310,472-476` |
| Volume controller | Active — narrow-band normal-speed correction, ±30 cells/s clamp; measured drift ≈0 | `lib/webgpu-quadtree-builder.ts:99-102,499-501` |

⚠️ **Much of this is uncommitted** (~5.2k insertions across 42 files, incl.
`webgpu-quadtree-tall-cell.ts` +837, `webgpu-quadtree-builder.ts` +160,
`quadtree-tall-cell-grid.ts` +113 with the MLS row-cap removal). First action of any
follow-up session: commit the verified source + test files (see W0 instructions in
`docs/quadtree-handoff-plan-2026-07-16.md` — the UI/overlay diffs are also wanted, as
they add the CFL/speed overlay modes).

## 2. Root-cause analysis

### Symptom A — fragmented mound + vertical fountain (screenshot 1)

**A1 (PRIMARY, unchanged from the audit): the ghost-fluid velocity kick is uncapped in
practice (100×) and shared with the matrix scale.**

- `refreshFaces` computes one Eq. 25 scale per face, clamped at **100** (θ floor 0.01):
  `lib/webgpu-quadtree-tall-cell.ts:443-445`; CPU constant `maximumFluidScale = 100` at
  `lib/quadtree-tall-cell-grid.ts:456`; direct-pack mirror at
  `lib/webgpu-quadtree-tall-cell.ts:1333`.
- `project` applies that *same* scale to the velocity update:
  `fluidScale = face.weights.y / face.weights.x` → `value[axis] -= fluidScale * gradient`
  (`lib/webgpu-quadtree-tall-cell.ts:883,891`). A nearly-dry face (θ ≈ 0.01) gets a 100×
  pressure-gradient kick. Nearly-dry vertical faces concentrate exactly at the crest of
  the collapsing splash — this is the fountain. The restricted tall-cell path clamps the
  equivalent scale at ~20 (θ ≥ 0.05) and does not exhibit the jet.
- Measured pre-fix: peak 7.8 m/s (vs 3.8 tall-cell, 2.6 uniform) and CFL 1.42. Nothing
  since has touched this mechanism.
- **Fix (XS)**: in `project`, clamp the *velocity-update* scale only:
  `let fluidScale = min(20.0, select(0.0, face.weights.y / face.weights.x, face.weights.x > 0.0));`
  Keep the matrix scale at 100 (the matrix side is SPD-safe and benefits from the sharper
  boundary). Mirror in any CPU velocity-update reference used by tests. This was audit
  item #3 / plan item W2-adjacent and was never applied.

**A2: CFL can exceed 1 — substepping is capped at 8 and there is no velocity clamp.**

- `proactiveQuadtreeSubsteps` (`lib/webgpu-uniform-eulerian.ts:22-35`) bounds the next
  frame's substep count from the *previous* frame's measured peak + gravity, `maximumSubsteps = 8`.
  A spike from A1 both exceeds the previous bound (one-frame lag) and can exceed 8
  substeps' worth of speed. Measured CFL 1.42 with the spikes present.
- **Fix (XS–S)**: after A1, spikes should vanish; still, add a safety: either raise the
  cap when `velocityBound*dt/h > 8` (it is readback-free, cost is linear in substeps) or
  add a last-resort per-component clamp at ~0.9·h/dt in `project`/extrapolation, with a
  telemetry counter so clamping is visible, not silent. Gate: dam-break peak CFL ≤ 1.

**A3: horizontal momentum is flattened inside coarse/tall regions → the surge becomes a
mound.**

- For any face with >1 sub-face (coarse leaf boundaries, tall faces), `project`
  *overwrites* the fine velocity with the solved area average and re-adds only the MLS
  variation: `value[axis] = (face.flux - face.solidFlux)/face.weights.x` then
  `gradient - face.mlsMean + solved` (`lib/webgpu-quadtree-tall-cell.ts:885-890`).
- The MLS variation degrades to Shepard (inverse-distance smoothing) wherever the 4×4
  normal equations are singular — which is systematically the case near the free surface
  because air DOFs are skipped (`mapPressure`, `:824,:837`); no ghost-air pressures are
  used. Net: near the surface and across the tall/cubic band, sub-leaf shear is destroyed
  every substep. Vertical shear of horizontal velocity is exactly what a dam-break surge
  is made of. The paper accepts *some* of this damping (§6 "Numerical Damping") but gets
  its surface detail back via EXNBFLIP particles and finer effective resolution; we have
  neither.
- **Fix (M)**: two independent steps, in order of value:
  1. Ghost-air pressures in `mapPressure` (Ando–Batty ghost values: extrapolate
     p through the interface with the same θ used in Eq. 25) so the MLS stays first-order
     at the surface instead of falling back to Shepard. Audit item #7.
  2. Sizing function: make sure the *moving front* refines. Today sizing = curvature +
     strain near the surface (`lib/webgpu-quadtree-builder.ts:648-665`); a fast flat
     front has low curvature and may stay coarse. Add a velocity-magnitude-gradient or
     Froude-like term so the surge tip runs at h_min. Cheap to A/B via
     `adaptivityStrength`.

**A4: liquid that the topology never sees gets no pressure at all → fragmentation and
pass-through splash.** See Symptom B, B2 — same root cause. A droplet or sheet living in
a column whose leaf-centre φ profile shows no crossing sits inside a Dirichlet air tall
run: `project` leaves its velocity untouched (fine φ < 0 → not zeroed at `:879`) but
there is no face DOF (`packedFace == 0`) and `mappedGradient` over Shepard-smoothed
distant pressures ≈ 0, so it free-falls and never pushes back. Fine for true droplets;
destructive for connected sheets, which is what shreds the crest.

**A5: nothing removes debris.** `cullDebris` exists
(`lib/webgpu-quadtree-builder.ts:311-323`: isolated wet voxel, all-dry 6-neighbourhood,
dry VOF → φ = +0.5h, counted in `culledDebrisCells`) but ships **disabled**
(`debrisCulling: values.debrisCulling === true`, `lib/methods/quadtree-tall-cell.ts:50,68`).
The per-step JFA faithfully preserves every isolated wet voxel forever. Pre-fix
measurement: 152 components at t = 0.25 s *before* heavy splashing.
**Fix (XS)**: default it on after a 1 s A/B confirming volume controller absorbs the loss
(it clamps at ±30 cells/s; culled debris is tiny). Gate: dominant-component fraction
≥ 0.995 (already an invariant in the smoke harness).

### Symptom B — "strange tall cells at the top" (screenshot 2)

**B1 (mostly benign): full-height air tall cells above the surface are by design.**
After the ≤2-cell cubic air band above each detected interface
(`airBandCells = 2`, `lib/quadtree-tall-cell-grid.ts:389,400`), the remaining air rows
coalesce into a single tall segment reaching `ny−1` (`:402-421`); nothing caps extent at
the ceiling. These carry **no pressure DOF** (Dirichlet p = 0: `dofBySample = −1` for
non-liquid samples, `:709-711`; GPU `dof = INVALID`,
`lib/webgpu-quadtree-pack-builder.ts:152-155`). This is paper-conformant (§4.2 forms tall
cells upward too) and numerically harmless. It *looks* alarming because the overlay draws
air tall cells (dark teal, `lib/webgpu-grid-overlay.ts:304-305`) in the same box style as
liquid tall cells. See debug-tooling task D1.

**B2 (the real defect): interface detection samples φ only along each leaf's horizontal
centre column.**

- Segmentation input is one φ value per (leaf, y) at the leaf centre
  (`lib/quadtree-tall-cell-grid.ts:373,336-351`; GPU profile
  `lib/webgpu-quadtree-builder.ts:710-723`; pack-builder mirror
  `lib/webgpu-quadtree-pack-builder.ts:76-85`). A droplet/sheet that does not cross the
  centre column of its leaf produces no `interfaceY`, hence no cubic band, hence the
  whole column above the bulk surface is one air tall run — the observed
  "full-height tall column with no cubic band" in the splash region, and mechanism A4.
- This bites whenever splash lands in a leaf with size > 1 (off-centre sub-columns
  unsampled) — sizing *should* refine splash to size 1 (it scans every fine column,
  `lib/webgpu-quadtree-builder.ts:648-665`, and takes the footprint max), but any
  droplet the sizing band misses, or that appears between rebuild kick and apply
  (≤2-step staleness), is invisible. Even at size 1 the interface test needs a
  centre-row sign change or |φ| ≤ h_min (`lib/quadtree-tall-cell-grid.ts:377`).
- **Fix (S–M)**: segment from a *footprint-conservative* profile instead of the centre
  sample: per (leaf, y) compute `minPhi` over the leaf's fine footprint (a second channel
  alongside the existing leaf-centre profile; the GPU already walks these columns in
  `sampleLeafProfiles`). Use centre φ for the tall-sample *values* (paper's choice) but
  flag `interfaceY` from the min/max profile so any sub-leaf crossing forces a cubic
  band. Both CPU (`populateTallPressureGrid`) and GPU (`classifySegments` in
  `lib/webgpu-quadtree-pack-builder.ts:94-116`) must change identically — there is a unit
  test asserting GPU/CPU topology equality ("GPU count-scan-emit rebuild reproduces CPU
  topology") that will catch drift.
- Note the CPU and GPU pack paths are otherwise byte-equivalent; switching backends will
  not change this behavior.

**B3 (transient): topology staleness.** With cadence-1 pipelined rebuilds the interface
can outrun the 2-cell air band for ≤2 steps (comment at
`lib/quadtree-tall-cell-grid.ts:386-388` acknowledges this; the air band exists for this
reason). Acceptable once B2 is fixed; do not spend effort here first.

### What the two symptoms share

A1/A2 inject energy at nearly-dry faces → spray; B2/A4 remove pressure coupling from
sub-leaf liquid → spray never rejoins the bulk; A5 keeps every resulting voxel alive;
A3 saps the coherent surge that would otherwise sweep debris along. Fixing A1 + B2 + A5
should collapse the component count and kill the fountain; A3 is the "shape" fix.

## 3. Honest gaps vs the paper (acknowledged, not bugs)

| Gap | Paper | Us | Position |
|---|---|---|---|
| Advection/transport on the adaptive grid | All fields adaptive (Table 1 advection 4.5 s → 0.77 s) | Dense fine-grid velocity/φ; quadtree accelerates projection only | Deliberate GPU choice (coalesced dense kernels). Caps memory/advection wins at uniform scaling. Revisit only if perf targets demand (old A3/R7). |
| EXNBFLIP splash enrichment | Yes (§5) | None | Largest visual-detail gap once stability lands. Strategic item. |
| Dual-contouring surface + high-res surface φ | Yes (§4.3.2, §6) | Marching cubes on sim-res φ | Renders every fragment faithfully — amplifies B2/A5 visually. Consider 2× render-φ later (Goldade-style). |
| MICCG/ICCG CPU solver | ICCG, rel 1e-4 | GPU PCG, poly preconditioner, rel 1e-4, indirect-dispatch early-exit | Ours converges (7.9e-9 measured); non-M-matrix note in code matches paper §5. No action. |
| Sizing oracle | curvature + velocity variation near surface | same, weights duplicated CPU (`quadtree-tall-cell-grid.ts:1044`) / GPU (`webgpu-quadtree-builder.ts:298` staticData) | Dedupe constants when touched (old B3). |
| Ando–Batty MLS with ghost-air | Implied by [F] second-order BC | Runtime MLS skips air DOFs, Shepard fallback | Fix under A3.1. |

## 4. Prioritized work plan

Ordered so each item is independently landable and measurable. P0–P3 are the "make the
screenshot stop happening" set.

| # | Task | Size | Files | Acceptance |
|---|---|---|---|---|
| **P0** | Commit the in-tree work (W0), then re-baseline: 1 s three-method smoke with field stats + envelope (command §6). Record IoU/components/CFL/peak-speed/drift per checkpoint as the new reference JSONL in `tmp/tall-cell-audit/`. | XS | — | Fresh JSONL; doc updated with real post-W0 numbers. |
| **P1** | Velocity-update ghost-fluid clamp ≤20× (θ ≥ 0.05) in `project` only; matrix keeps 100. | XS | `lib/webgpu-quadtree-tall-cell.ts:883` (+ CPU mirror/test) | Dam-break peak speed ≤ 5 m/s, peak CFL ≤ 1 (existing smoke gates flip to pass); fountain gone visually. |
| **P2** | Enable `cullDebris` by default after A/B; keep counter telemetry. | XS | `lib/methods/quadtree-tall-cell.ts:50` | Dominant component ≥ 0.995 @1 s; volume drift still ≤ 0.5%. |
| **P3** | Footprint-conservative interface detection (B2): min-φ profile channel; force cubic bands on any sub-leaf crossing; CPU + GPU pack in lockstep. | S–M | `lib/quadtree-tall-cell-grid.ts:373-401`, `lib/webgpu-quadtree-pack-builder.ts:94-116`, `lib/webgpu-quadtree-builder.ts:710-723` | GPU=CPU topology test stays green; overlay shows cubic bands around every splash region; no full-height no-band columns while liquid is present in the column. |
| **P4** | CFL safety: adaptive substep cap raise or clamped-with-telemetry fallback. | XS–S | `lib/webgpu-uniform-eulerian.ts:22-35` | CFL gate ≤1 holds even with induced spikes (test exists: "CFL subdivisions use current conservative velocity bound"). |
| **P5** | Ghost-air pressures in `mapPressure` MLS (drop the air-DOF skip; substitute ghost value −φ_air/φ_liq·p_liq per face θ). | M | `lib/webgpu-quadtree-tall-cell.ts:814-840` | Settled-tank stays quiet (no stirring regression — this failure mode is documented at `lib/quadtree-tall-cell-grid.ts:1019-1024`); dam-break IoU vs uniform ↑ at 0.5–1 s. |
| **P6** | Sizing: refine the moving front (velocity-magnitude/gradient term), constants deduped CPU/GPU. | S | `lib/webgpu-quadtree-builder.ts:298,648-665`, `lib/quadtree-tall-cell-grid.ts:1044` | Overlay shows size-1 leaves at the surge tip; IoU ↑; DOF count increase bounded (<2×). |
| **P7** | Strategic (post-stability): EXNBFLIP-style particle enrichment; 2× render φ; adaptive advection. Decide after P0–P6 measurements. | L | — | — |

Explicitly **not** on the list: preconditioner work (done — poly), per-step surface
refresh (done), VOF retirement (circuit breaker is inert; W7's 10 s φ-only soak gate
`test:webgpu:dam-quadtree-phi-only-soak` remains the retirement criterion), the restricted
tall-cell settling investigation (separate handoff:
`docs/tall-cell-dam-settling-handoff-2026-07-16.md`).

## 5. Visual debug tooling (requested; ranked by diagnostic value)

Existing: scientific-view "Solver grid" slice overlay (Z/X slice + draggable position)
with field modes **Structure / CFL load / Speed** and a live legend
(`components/VisualPanel.tsx:86-88`, `lib/webgpu-grid-overlay.ts` `fieldMode` in
`u.debug.w`, `lib/stores/ui-store.ts` `gridOverlayMode`). The CFL/speed modes are part of
the uncommitted diff — keep them. The overlay samples live GPU textures (velocity,
`adaptiveCells` topology, fluid field) with no readback, so new modes are cheap.

Add, in this order:

| # | Tool | Why | How |
|---|---|---|---|
| D1 | **Distinguish air tall cells + "unrepresented liquid" alarm mode.** Render air tall cells as outline-only (or a toggle to hide dry cells), and add a field mode that paints any fine cell with φ < 0 whose adaptive cell has no liquid DOF in **red**. | Directly visualizes B2/A4 — the exact defect in screenshot 2 becomes a red highlight instead of a guess; also de-alarms the by-design air columns. | `lib/webgpu-grid-overlay.ts:273-305` (adaptive branch already reads topology + φ via `fluidSample`); needs the DOF-validity bit — the topology word or `cellPressureSamples` texture already distinguishes sample liquidity. |
| D2 | **φ (level set) slice mode**: signed-distance colormap with zero-contour line, optional VOF-disagreement tint. | Debris, reseeding, band health, and redistance defects become visible per-step; today φ is only visible via the final mesh. | New `fieldMode = 3`; bind the resident φ texture (`WebGPUQuadtreeSurfaceState` canonical texture) to the overlay pipeline. |
| D3 | **Post-projection divergence mode**: per-cell ∇·u after `project` (compute into a small R16F texture behind a debug flag). | Separates "solver didn't converge / operator wrong" from "transport destroyed it" — the two are indistinguishable in the current overlays. Would have located the fountain source in minutes. | Small compute kernel in `lib/webgpu-quadtree-tall-cell.ts` or the orchestrator; overlay mode 4. |
| D4 | **Pressure slice mode**: mapped fine pressure (`mapPressure` output texture already exists per-solve). | Seams at tall/cubic band and leaf boundaries (A3, mechanism #2 for the jet) show up as pressure discontinuities. | Persist the transient mapped-pressure texture behind the debug flag; overlay mode 5. |
| D5 | **Per-frame HUD counters** in the legend: φ component count (approximate GPU union-find is overkill — reuse the smoke harness's counter on demand), `culledDebrisCells`, `pressureIterationsUsed`, rebuild staleness/blocked frames, reconciliation-armed flag. | The legend already shows `maxComponentCfl`/`maxSpeed`; these five are the ones every diagnosis session grep'd out of JSONL. | `components/VisualPanel.tsx` legend; diagnostics already surfaced by the engine (`lib/webgpu-uniform-eulerian.ts:597`). |
| D6 | **Velocity glyph slice** (small arrows or line integral streaks on the slice) gated to the extrapolation band (|φ| < 3h). | Verifies the new air-extrapolation actually feeds the backtrace (freezing crests vs moving crests). | Extend overlay shader; optional, do last. |

## 6. Reproduction and verification

```bash
# environment (recreate after reboot: tmp/tall-cell-audit/post-reboot.sh)
export WEBGPU_NODE_MODULE=/private/tmp/fluid-webgpu/node_modules/webgpu/index.js

# shader validation first
npm run test:quadtree-shaders

# the P0 baseline / regression run (three methods, 1 s, IoU + envelope):
FLUID_SCENE=dam-break-ui FLUID_METHOD=quadtree-tall-cell,tall-cell,uniform \
FLUID_TARGET_S=1.0 FLUID_CHECKPOINT_EVERY_S=0.25 FLUID_CPU_ORACLE=0 \
FLUID_FIELD_STATS=1 FLUID_STABILITY_ENVELOPE=1 \
node --import tsx tools/run-webgpu-smoke.ts

# quick gates: npm run test:webgpu:dam-break-regression   (0.2 s)
#              npm run test:webgpu:dam-quadtree-quality   (0.2 s + envelope)
#              npm run test:webgpu:quadtree-matrix        (all scenes)
# knobs: FLUID_QUADTREE_PRECONDITIONER=poly|line|blockic|ic0,
#        FLUID_QUADTREE_DEBRIS_CULLING=1, FLUID_QUADTREE_VOF_RECONCILIATION=0
```

Smoke gates already encode the targets (`tools/run-webgpu-smoke.ts:1022-1064`): dam-break
peak speed ≤ 5 m/s, peak CFL ≤ 1, pressure rel-residual ≤ 1e-4, dominant component
≥ 0.995, level-set drift ≤ 0.02, 2:1 neighbor ratio. **Today's failing gates are speed,
CFL, and component fraction** — P1–P3 target exactly those. Success bar for the visual
outcome: wet-IoU vs uniform ≥ 0.6 at 1 s (the "normal divergence" parity bar from the
audit), single dominant component, no fountain, visible directed surge at 0.6 s.

Unit tests to keep green: `tests/quadtree-webgpu.test.ts` (esp. "GPU count-scan-emit
rebuild reproduces CPU topology", "projection preserves and extrapolates near-surface air
velocity"), `tests/quadtree-tall-cell-grid.test.ts` (MLS exactness/coverage, segmentation
tests — extend "leaf-centre profile reproduces dense φ segmentation" when doing P3).

## 7. Caveats for the next agent

- The pressure operator itself checks out: variational assembly uses identical
  coefficients for gradient and divergence (symmetric by construction), [V] tall-face
  volumes are correct, and the solver converges to 1e-4+ every solve. Don't chase the
  matrix; the damage is in the velocity *update*, the *topology's blindness* to sub-leaf
  liquid, and *debris lifecycle*.
- The mechanism ranking for the fountain (A1 > MLS band seams > extrapolation smear) is
  from code inspection, not a bisected reproduction. P0's fresh baseline plus the D3
  divergence overlay will confirm or reorder it cheaply — do P0 before arguing with
  the ranking.
- The restricted `tall-cell` method's failing settling gate is a *different* workstream
  with its own handoff doc; several files are shared, so coordinate commits.
- `lib/tall-cell-multigrid.ts` is not on the quadtree path (unreferenced) — ignore it.
