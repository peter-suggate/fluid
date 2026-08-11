# Uniform (Chentanez–Müller 2012) performance handoff

**Scope.** The dense `uniform` method (`lib/webgpu-uniform-reference.ts` + its three
WGSL modules) implements `docs/papers/massConservingLiquids.txt` faithfully but runs
**62–88 ms per step on a 32×16×32 grid** (16K cells), where the paper reports
26.7 ms at 128×128×64 and 113 ms at 256×128×128 on a 2012 GTX 680. This document
is the implementation plan to close that gap, written after an adversarial
verification pass over every diagnostic claim. Each claim below carries a verdict:
**CONFIRMED** (verified in code or artifacts), **ESTIMATED** (arithmetic or cost
model, not yet measured), or **CORRECTED/RETRACTED** (the adversarial pass changed
or killed it).

**Direction constraints from Peter (do not re-litigate):**

- Dispatch/encode overhead is **not** the lens. Do not justify work by pass counts
  alone.
- The timestep policy (`maxDt_s`, advances-per-presentation) is **out of scope**.
  The frame must get fast at the current dt.
- The lenses that matter: **algorithmic parallelism** (critical-path depth),
  **compact data structures**, **cache locality**, and **no loops in shaders over
  indirect memory**.
- Structural do-less-work changes before cheaper launches (standing direction).

---

## 1. Baseline and how to measure

- Measured: `wallPerStep_ms` 62–88 (typical) / 155 (extra-level variant) for the
  uniform arm in `artifacts/integration-symmetric-expansion-coverage-*.json`.
  `simulationWall_ms` is solver-loop wall **excluding** the every-10-step QA
  reconstructions (`tools/webgpu-smoke-executor.ts:1741`), so the number is honest.
  **CONFIRMED.**
- Reproduce: `npm run benchmark:symmetric-expansion:uniform-vs-losasso`, or the
  short form in `docs/symmetric-expansion-uniform-losasso-benchmark.md`. Fresh
  isolated process per arm, WebGPU exclusive lock applies.
- Per-pass GPU attribution exists and should anchor every WP:
  `FLUID_GPU_FINE_TIMESTAMPS=1` on the isolated runner produces
  `gpuPassTimestamps` / `gpuFineTimestamps` reports
  (`tools/webgpu-smoke-executor.ts:1400`). The advance also has hardware seams via
  `UNIFORM_ADVANCE_PHASE` (`webgpu-uniform-reference.ts:127`) consumed by the
  in-app pipeline panel.
- **WP0 (do first): capture the per-phase split** of one uniform advance on the
  benchmark scene with fine timestamps, and archive it in the artifact. Every
  claim below marked ESTIMATED gets settled by this run. Do not start WP2+ until
  WP0 numbers exist; priorities below assume the cost model is roughly right and
  WP0 may reorder them.

---

## 2. Verified findings ledger

### F1 — The coarsest LCP solve spins its full 4,096-iteration cap. CONFIRMED (cost ESTIMATED)

`mgSolveCoarsest` (`webgpu-uniform-pressure-multigrid.wgsl.ts:313–370`) is one
256-lane workgroup. The loop is `for (iteration < mg.control.w)` with
`mg.control.w = UNIFORM_CM11A_COARSE_SWEEP_CAP = 4096` and **no break**;
`converged` only gates the work inside. Eight `workgroupBarrier()`s per iteration
execute unconditionally (verified count: 2 color + 1 clamp + 1 reset + 1 residual +
2 worst-lane + 1 converged). Every invocation therefore runs ~33K barrier waves on
one workgroup while the rest of the queue (a strict dependency chain) waits.

- Invocations per advance: **L per full cycle + 1 per V-cycle** = 16 at the
  benchmark's L=4 hierarchy, 25 at L=7 (128³). CONFIRMED from `buildPlan`
  (`webgpu-uniform-pressure-multigrid.ts:363–410`).
- The active work uses double-single arithmetic (`mgDSAdd`/`mgDSScale`/
  `mgDSDivide`, ~20 flops per neighbor accumulate) in both the smooth and the
  residual phase. CONFIRMED.
- Telemetry hides the spin: `iterations` stops incrementing at convergence, so
  `uniformCM11aCoarseIterations` reports the converged count while the hardware
  runs 4,096. CONFIRMED.
- Wall-time attribution (~tens of ms/advance from this alone) is **ESTIMATED** —
  WP0 measures it. This is the one place where the *algorithm* narrows to 256
  threads for thousands of dependent steps, so it is first in line regardless of
  the exact number.
- Dead machinery note: `mgSmoothColour`'s `coarseDone` gate reads
  `mg.control.w & 2`, but smooth emits pass `control[3] = 0` and the coarse solve
  passes 4096 (bit 2 clear) — the gate can never fire. CONFIRMED.

### F2 — Sequential depth of the pressure schedule. CONFIRMED (arithmetic)

Fixed 3 Full-Cycles + 4 V-Cycles, 4+4 PRBGS sweeps, every sweep = 3 dependent
full-level passes (2 `mgSmoothColour` + `mgProjectMinimum`). Exact plan sizes from
walking `buildPlan`: **939 dependent passes at the benchmark (L=4), 2,634 at 128³
(L=7), 3,373 at 256 (L=8)**. The finest level alone receives 56 sweeps (7
V-cycle-equivalents × 8) = 168 full-grid passes per advance. The fine residual is
measured on-GPU and read back but never steers the schedule (the panel tooltip
says so explicitly, `webgpu-uniform-reference.ts:1190`).

### F3 — Each smooth pass rewrites the whole level; the projection pass is redundant. CONFIRMED (redundancy needs one A/B)

`mgSmoothColour` dispatches the full level; wrong-color and non-liquid threads
copy their old value through (write-only storage textures force full rewrite per
ping-pong pass). The updated color is **already projected at write time**
(`max(p, p_min)`, line 249). After both colors, every liquid cell has been
projected; the trailing `mgProjectMinimum` re-projects everything, and the code
comment admits it is retained only to match the paper's prose summary. Removing it
should be bit-identical for liquid cells; solid/halo cells pass through the smooth
unprojected, so verify with the stage-audit A/B before deleting (see §4).

### F4 — Smoother coefficients are recomputed from textures every sweep. CONFIRMED

`mgCoefficient` costs ~3–5 texture loads per neighbor (`mgFaceV` topology loads,
`mgTheta` phi loads, `mgLiquid`), ×6 neighbors, evaluated in the smooth **and
again** in `mgResidual` — for coefficients that are constant across the entire
solve (topology and phi are fixed after setup). No LDS tiling anywhere in the
module.

### F5 — Solid geometry has no grid representation outside the multigrid. CONFIRMED

Solids exist only as ≤12 analytic primitives in a storage buffer. Verified
chains (all per texel, per pass):

- `sharpenDeltaRho` → 6 × `faceOpenFraction` → 4 quadrature samples ×
  `rigidBodyIndexAt` (≤12 `insideRigid` quaternion tests) ≈ **288 body tests per
  cell** in the sharpen pass (`webgpu-uniform-reference.wgsl.ts:777–806, 265–286`).
- `pressureDensity`'s V=0 fallback loops 6 neighbors × `cellOpenFraction`
  (12 bodies × 8 corners) ≈ up to **~670 tests** for solid-adjacent cells
  (`:120–140, 243–246`).
- `project` calls `pressureLiquid` ~7–8× per cell (center + neighbors + ghost
  fractions), each one a full `pressureDensity` chain; and — adversarial catch —
  `pressureFaceData` at `:705` is evaluated **unconditionally** because WGSL
  `select()` evaluates both arms: the `nearAnyBody` reject chooses the *value*,
  not the *cost*. With bodies present the 8-sample dual-cell quadrature plus
  `extrapolatedRigidVelocityAtFace` (12 × `rigidSignedDistance`) runs on every
  interior face of every cell.
- The transport passes pay it too: `transportStencilWeight` →
  `densityTransportDestination` → `cellInsideSolid` (12-body loop) **per corner**
  of every 8-corner gather, in trace, scatter, and gather (`:397–402`).
  `TraceAlongField` multiplies it along its 5-step walk (`:809–822`).

**Scale caveat (adversarial):** every loop breaks at `bodyCount == 0` and terrain
is one 2D load, so on the body-free benchmark scene these chains cost ~nothing —
they do **not** explain the 62–88 ms. They dominate per-cell work on any scene
with bodies or terrain, at every resolution. Fix for correctness of the cost
model at production scenes, not for the benchmark number.

The template for the fix already exists in-tree: `mgBuildFinestTopology` bakes
exactly this data (cell open fraction + 3 face dual-volumes + phi) into textures
once per advance and the whole multigrid runs on pure loads. Transport,
sharpening, projection, solid-excess, and coupling never got that bake.

### F6 — All interpolation is manual 8-corner loads; the hardware path is half-built and unusable today. CONFIRMED (amended)

`textureSampleLevel`/`textureSample` appear **zero times** across all three WGSL
modules (grep-verified). The transport texture carries a one-texel zero shell
built expressly "so hardware trilinear sampling reproduces the zero wall-face
boundary condition" (binding 14 comment) and a filtering sampler is bound at
binding 15 — unused. **Amendment from the adversarial pass:** the device never
requests `float32-filterable` (grep over `lib/*.ts`), so sampling rgba32float
with that sampler would fail validation today. The fix requires requesting the
feature (M1 exposes it via the Mac2 family; gate at adapter time) or sampling an
rgba16float copy (numerics change — see risks).

Verified per-cell load counts: `advect` = 3 MAC components × (RK2 `departurePoint`
= 2 × 24 loads + 8 final) = **168 texture loads/cell**; `reverseAdvection`
repeats all 168; `correctAdvection` adds a 24-load min/max clamp. The MacCormack
stage is ~360 manual 16-byte loads per cell versus ~21 hardware samples.

### F7 — Fragmented scalar fields, dead lanes, dense containers for face data. CONFIRMED

- ρ, γ, ρ′, and pressure scratch are four separate r32float textures; every
  Sec. 3.4/3.5 kernel that touches (ρ, γ) pays two fetch streams. The multigrid
  smoother's per-neighbor reads span pressure, phi, topology, minimum — four
  textures where (p, rhs, φ, p_min) fit one rgba texel per level.
- Velocity/transport rgba32float always store `w = 0.0` (verified in
  advect/reverse/correct/project stores); no uniform kernel reads velocity `.w`.
  25% of the most-fetched bytes are zeros. FIM state is 6 padded rgba32float
  textures with bitmasks stored as `f32(mask)` requiring `round()` decode.
- The four `boundaryVelocity` buffers are dense `n³×16 B` each, holding data
  meaningful only on the three negative domain faces (O(n²) payload).
  `carryBoundaryVelocity` **reads and rewrites the entire buffer** in every
  advect, reverse, and correct pass, and `project` reads + rewrites it again
  (`webgpu-uniform-reference.wgsl.ts:560, 575, 593, 678, 716`).
- `sharpenDeposits` scatter atomics go to a linear-layout buffer
  (`x + nx(y + ny·z)`) with three plane-strided regions (offsets 0 / count /
  2·count) from 3D-tiled thread groups — a second, cache-hostile address stream
  inside the transport kernels.

### F8 — The FIM front: fixed 24-iteration encode, full-grid granularity while active, run twice per advance. CONFIRMED (one earlier claim corrected)

- `activeFrontPasses = 4·2·3 = 24` for a band the module's own comment limits to
  **two cells** (`webgpu-uniform-velocity-extrapolation.ts:131–134`).
- **Correction from the adversarial pass:** converged iterations are *not*
  full-grid sweeps. `prepareActiveDispatch` zeroes the indirect args once the
  active counter empties, so post-convergence updates dispatch zero workgroups.
  The genuine waste while the front is live: the indirect args are **all-or-
  nothing** — `ceil(dims/4)` over the whole domain whenever any face is active
  (`webgpu-uniform-velocity-extrapolation.wgsl.ts:325–327`) — there is no
  compacted worklist, so each live iteration sweeps every cell to advance a
  2-cell band; and the 24 `prepare` links (one thread active, 63 masked) remain
  on the critical path every invocation.
- The entire extension (seed, front, resolve, hierarchy down/up, pack ≈ 61
  passes at benchmark size) runs **twice** per advance because MacCormack needs
  the predicted field extended. The two invocations bind the *same* density
  texture but at different times: the first sees pre-advection density, the
  second post-advection (the authority pass rebuilds ρ′ in between). Sharing one
  extension operator across both is therefore an **approximation decision**, not
  free reuse — extending both fields with one (post-advection) operator is
  arguably more consistent for MacCormack's error cancellation, but it changes
  results and needs the gates re-blessed. The paper-faithful alternative is plain
  semi-Lagrangian velocity advection (the paper has no MacCormack), which
  deletes the second extension and the reverse/correct passes outright.

### F9 — Sequential-depth census. CONFIRMED (arithmetic)

Whole advance at benchmark size ≈ **1,090 dependent passes**: 939 pressure + 122
extension (61 × 2) + ~30 transport/sharpening/copies. The three stages transcribed
from the paper's own GPU formulation (density trace/scatter/gather, sharpen
compute/scatter/resolve) are the only ones already at paper-shaped depth. The
depth explosion lives where iterative/ordered formulations replaced direct ones:
fixed-cap coarse LCP, 7 fixed multigrid cycles × 8 × 3-pass sweeps, 24-slot FIM
encode ×2, 12-pass mirrored gamma diffusion.

### F10 — Gamma diffusion: 13 passes + 4 dense copies, mirrored orders exist only for D4 symmetry. CONFIRMED

`webgpu-uniform-reference.ts:754–775`: 6 forward (xyz even/odd pairs) + 6 reverse
(zyx) + `averageGammaDiffusion`, plus 4 full-volume `copyTextureToTexture` to
stage and commit. The shader comment states the mirror-average exists because the
paper leaves axis order unspecified and an ordered traversal breaks D4. A
snapshot-based antisymmetric flux gather (each cell computes the identical
pairwise exchange with each neighbor from the same input state; one adds, one
subtracts) is conservative by construction and **order-free**, so the entire
mirrored machinery and its copies collapse to one pass per repetition. This
changes numerics (it is a Jacobi-style relaxation of the paper's sequential
pairwise rule) → gates must be re-blessed; the D4 gate should get *stronger*.

### F11 — Five full-volume copies per advance stand in for bind-group parity. CONFIRMED

`webgpu-uniform-reference.ts:754–755, 774–775, 796` (+3 more on the rigid path,
`:820–822`). The multigrid module already demonstrates the fix (texture pairs +
parity flip in the plan).

### Retracted / demoted claims

- **`buildHeight` serial column loop**: the kernel exists in the shader but is
  not in `PIPELINES` and never dispatched in the advance. RETRACTED.
- **"48 full-grid FIM updates per advance"**: overstated; converged iterations
  dispatch zero workgroups (see F8). CORRECTED.
- **Dispatch/encode overhead as a primary cost**: out of scope by direction;
  WP0's fine timestamps will attribute GPU time without relying on it.
- **Timestep multiplier vs the paper**: real, but out of scope by direction.
- **Per-advance `writeBuffer`/instrumentation/readback costs**: negligible;
  readback is throttled and pending-guarded. Dropped.

---

## 3. Work packages

Ordered by (expected effect ÷ risk), assuming WP0 confirms the cost model.
Each WP states its exactness contract: **BIT-IDENTICAL** (must produce identical
fields; verifiable with the stage audit), or **RE-BLESS** (numerics change;
benchmark gates must be consciously re-accepted).

### WP0 — Measure the per-phase split (no code change) — **DONE 2026-08-11**

Measured on the benchmark scene (100 steps, comparison-uniform lane, stage
audit on, fresh process). Control wall: 59.2 ms/step. Per-pass attribution
(`FLUID_GPU_PASS_TIMESTAMPS=1`, one sampled advance, 1,024 passes, 63.8 ms GPU
sum — archived in the WP0 scratch captures and reproduced below):

| Pass | total ms/advance | invocations | mean |
| --- | --- | --- | --- |
| `mgSolveCoarsest` | **51.05 (80%)** | 15 | 3,403 µs |
| `mgSmoothColour` | 6.78 | 455 | 14.9 µs |
| `mgProjectMinimum` | 1.36 | 228 | 6.0 µs |
| FIM front (all passes, both invocations) | ~1.6 | ~50 | 70–590 µs |
| `mgResidual` + restrict/prolongate/clears | ~1.4 | ~180 | 5–15 µs |
| transport + sharpening + diffusion + projection | ~1.0 | ~30 | — |

Verdicts settled: F1's ESTIMATED cost is **80% of the frame** — the 4,096-spin
coarse solve dominated everything. F5/F6 (analytic solids, manual
interpolation) are confirmed ~irrelevant on this scene: the whole MacCormack +
transport block is ~1 ms. The remaining post-WP1 frame is launch-depth bound
(455 × 15 µs smooth passes), which is F2/WP3 territory.

**Caveat:** the seam-level `FLUID_GPU_FINE_TIMESTAMPS` recorder misattributes
on this lane (it charged ~35 ms to extension phases that per-pass data shows
are sub-ms, and its per-advance sum exceeded the measured wall). Use
`FLUID_GPU_PASS_TIMESTAMPS=1` for attribution here.

### WP1 — Coarsest solve: break on convergence. BIT-IDENTICAL

`webgpu-uniform-pressure-multigrid.wgsl.ts:322–360`. `converged` is computed
identically by all lanes from a workgroup atomic after a barrier, but WGSL's
uniformity analysis cannot see that — which is why the current code gates work
instead of breaking (the repo's "select-by-count, never branch around a barrier"
lesson). The sanctioned escape: store the convergence flag to a
`var<workgroup>`, read it with **`workgroupUniformLoad`** (itself a barrier and
formally uniform), then `if (converged) { break; }` is legal. Post-convergence
iterations currently compute nothing, so breaking is bit-identical by
construction. Also fix the telemetry so physical iterations are reported
(publish both converged-at and executed counts; the current single counter is
the misleading one).

Expected: removes ~16 × (4096 − actual) barrier-only iterations per advance.
This is the cheapest change with the largest depth reduction; measure before/after
wall per step in the same session (±5%-style noise rules apply — fresh process
per arm, compare medians of ≥3 runs).

### WP2 — Bake solid topology once per advance; consumers read textures. BIT-IDENTICAL (body-free scenes trivially; body scenes by construction)

Add one bake pass per advance (bodies move once per step) writing, per cell:
cell open fraction, 3 face apertures (`faceOpenFraction` semantics), and — in a
second rgba texel or a separate texture — face solid velocity + the CM11a dual
face volume (`pressureFaceData` semantics). Rewire `sharpenDeltaRho`,
`scatterSolidExcess`/`resolve`, `project`, `coupleRigid`, and
`transportStencilWeight`/`cellInsideSolid` consumers to single loads. Evaluate
the *same helper functions* in the bake so stored values are bit-identical to
today's inline evaluations; storage is f32, so consumers read exact values.
`mgBuildFinestTopology` can then also read the bake instead of re-deriving.

Notes: `select()` evaluates both arms — when rewiring `project:705`, hoist the
load, don't keep the analytic call behind a select. The `pressureDensity`
6-neighbor continuation should be baked too (store ρ′ per cell; it is already
conceptually a field — the authority pass builds one for extension but the
pressure path re-derives it inline).

### WP3 — Pressure stage restructure. Split into three ratchets

- **WP3a (BIT-IDENTICAL, pending one A/B):** delete `mgProjectMinimum` from
  `sweep()` after verifying via stage audit that solid/halo pass-through cells
  never feed a consumer unprojected (F3). Sweep cost 3 passes → 2.
- **WP3b (BIT-IDENTICAL):** per-level coefficient bake — one setup pass per level
  storing (a_x⁺, a_y⁺, a_z⁺, flags) per cell (negative-face coefficients are the
  neighbor's positive ones); smoother and residual read 1 texel per neighbor
  instead of re-deriving via `mgFaceV`/`mgTheta`. Keep the arithmetic identical
  (same formula, same order) so results don't move.
- **WP3c (RE-BLESS, separate decision):** replace RBGS with damped/Chebyshev
  Jacobi (projection folded in) and steer cycle count with the already-measured
  residual. This is the depth win (~3× on the pressure stage) but changes
  convergence behavior; the coloring is D4-symmetric either way. Do not bundle
  with 3a/3b.

### WP4 — Transport sampling through the texture unit. RE-BLESS

Request `float32-filterable` at device creation when the adapter offers it;
convert `sampleVelocityComponent`, `sampleVolume`, `sampleGammaStencil`, and
`mgTrilinearPressure` to `textureSampleLevel` on the already-shelled transport
texture (the shell exists for exactly this). Keep the manual path compiled as
fallback for adapters without the feature. Risks: hardware filtering precision is
implementation-defined — the D4 symmetry gates are the acceptance test (mirrored
coordinates must filter symmetrically; if Apple's fixed-function weights break
the 1e-3/1e-4 gates, fall back to manual for the gated lanes and keep hardware
for production scenes, or use an rgba16float sampling copy with an explicit
re-bless). Expected: MacCormack stage ~360 → ~21 fetches/cell; also shrinks
sharpening's trace walk.

### WP5 — Gamma diffusion as one snapshot flux-gather pass. RE-BLESS

Implement F10's antisymmetric flux gather: per cell, for each of 6 neighbors,
compute the paper's pairwise transfer from the *pre-pass* (ρ, γ) snapshot with a
symmetric formula (so both sides compute the same flux bit-identically), apply
±. One pass per repetition (paper allows 1–7; start with 2 and A/B dissipation
via the benchmark's energy-retention indicators). Deletes 11 passes, the average
pass, the 4 staging copies, and the mirrored-order machinery. D4 symmetry holds
by construction — tighten the gate if anything, and re-bless drift/energy.

### WP6 — Boundary-velocity compaction + bind-group parity. BIT-IDENTICAL

Replace the four dense `n³×16 B` boundary buffers with three face-sized arrays
(or fold the negative faces into the transport texture's existing halo shell).
`carryBoundaryVelocity` then touches O(n²) not O(n³) per momentum pass. While in
the bind-group code, build both parities of the main-loop groups (as the
multigrid does) and delete the 5 per-advance `copyTextureToTexture` staging
copies (F11). Pure data-movement change; fields must be bit-identical.

### WP7 — Extension restructure. Two options, decide before starting

- **7a (paper-faithful, RE-BLESS):** drop MacCormack; velocity advection becomes
  one semi-Lagrangian pass (the paper's own scheme). Deletes the second extension
  (61 passes), reverse, and correct. Expect more velocity dissipation — the
  benchmark's `lateToMiddleKineticEnvelopeRatio` /
  `normalizedLateMechanicalEnergySlopePerSecond` indicators are the A/B; the
  uniform lane is already labeled "very dissipative", so this needs Peter's call.
- **7b (keep MacCormack, RE-BLESS):** build the extension *operator* once from
  post-advection density (distances + upwind sources), apply it to both current
  and predicted fields (one cheap apply pass each). Consistent extension of the
  MacCormack pair; changes results slightly vs today's two-geometry extension.

Either way: replace the 24-slot fixed front with a direct band computation — the
accurate band is ≤2 cells, so distances and upwind values are computable in one
bounded-stencil gather (radius-2 neighborhood), no wavefront iteration; keep the
log-depth hierarchy fill as-is. If keeping FIM instead, lower the ceiling to what
a 2-cell band needs and make the indirect dispatch a compacted worklist rather
than all-or-nothing.

### WP8 — Field packing. BIT-IDENTICAL (layout only)

Pack (ρ, γ, ρ′, flags) into one rgba texture for the transport band; pack
(p, rhs, φ, p_min) per multigrid level; store FIM state compactly (masks as
integer texels or packed with distances); stop writing the dead velocity `w`
lane (or use it for something — e.g., the ρ′ authority — and delete a texture).
Do after WP2–WP5, since those change who reads what.

---

## 4. Verification protocol

- **Bit-identical WPs:** run with `FLUID_UNIFORM_SYMMETRY_STAGE_AUDIT=1` and
  compare the stage-audit textures (`symmetryStageAuditTextures`,
  `webgpu-uniform-reference.ts:177`) byte-for-byte against a control run at
  fixed step counts (the audit captures every stage boundary: advection,
  diffusion, sharpening, prediction, reverse, correction, projection). Caveat
  from the SVO lanes: byte-identity can break legitimately when dataflow shape
  changes make the compiler reassociate floats — if WP3b or WP4 trip this,
  fall back to the benchmark gates plus a small absolute-difference bound and
  say so in the PR.
- **RE-BLESS WPs:** full `benchmark:symmetric-expansion:uniform-vs-losasso` run;
  gates that must hold: D4 volume ≤1e-3, D4 velocity ≤1e-4 m/s, ≤1% conservative
  drift, dominant-component and boundary-residue diagnostics. Energy indicators
  are reported A/B, not gated — bring regressions to Peter rather than silently
  accepting.
- **Wall-time claims:** fresh process per arm, ≥3 runs, compare medians; never
  compare across tripwire modes; a single-run 2–3% delta has measured nothing.
- Never stash/checkout/reset in this worktree to get a baseline — run the control
  from a separate clone or use the archived artifacts.

## 5. Appendix: plan arithmetic (recompute, don't trust)

For hierarchy depth `L = log2(min axis)`:
`sweep = 3` passes; `v(l) = 29 + v(l+1)` with `v(L−1) = 1`;
`full = 3 + 2(L−1) + 2 + Σ_{l=0}^{L−2}(1 + v(l)) + 1`;
total = setup `(2 + (L−1) + L)` + `3·full` + `4·v(0)` + 2.
Coarse-solve invocations per advance = `3L + 4`.
Benchmark (L=4): 939 passes, 16 coarse solves. 128³ (L=7): 2,634 / 25.
Extension per invocation = `4 + 2·24 + downs + ups` (+1 authority) ≈ 61 at
benchmark dims; ×2 per advance. Transport/sharpen/solid/copies ≈ 30.
