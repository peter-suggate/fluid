# All-fine pool: sharp seams from switching surface definitions

## Finding

A controlled current-tree Dawn comparison reproduces cross-shaped and rectangular ridges in `coarse-first-pool-impact-half`. Disabling only `presentationHeightPolicyEnabled` in diagnostic shader modules removes a family of raised publication seams, but it does not eliminate all surface artifacts. Accepted density is byte-identical at 2, 3 and 3.333 seconds. The broad four-lobed wave remains in the density field; its physical/numerical angular accuracy is not established by this experiment.

This is not a D4 averaging effect. The resident initializes `d4Capable`, scalar authority and face authority to false (`webgpu-sparse-cm12-resident.ts`, around line 4725), and the field-mutating D4 kernels were removed. The surviving frame-control names preserve ABI layout.

## Mechanism

In `lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts`:

- `presentationIntegratedColumnReceipt` (around 2474) integrates a bracket anchored to a density crossing. It accepts/rejects columns using hard monotonicity and full/empty endpoint tolerances of 0.01.
- `presentationContinuousColumnHeight` (around 2683) returns that receipt directly at finest resolution: all-fine enforcement does not bypass this path.
- `preparePresentationColumnHeights` populates per-column validity; `presentationColumnHeightValid` (around 2860) checks only whether the cached height is nonnegative.
- `cm12PresentationExactSample` (around 9700) first chooses `phi = (y + 0.5 - height) * h` for valid columns. Otherwise the fine-cell fallback is `phi = 4h * (0.5 - rho/open)` from `presentationPhiAt` (around 2227). The legacy publication kernel contains the same choice.

Those surfaces disagree in moving, diffuse or depleted columns. A validity boundary therefore creates a finite jump in displayed surface position, even with unchanged simulation state. The selector is binary, without continuity between the two definitions. The publication delta maps show the resulting axial strips and quadrant-shaped patches.

The older September 6 report describes a continuous vertical reconstruction that fixed similar symptoms then. That is historical, not current behavior: commit `30d572e8` removed the vertical-volume reconstruction and restored density presentation, while the conditional column-height path remains. Reapplying the older fix indiscriminately would require checking side walls, detached liquid, overturning interfaces and coarse/floor surfaces.

## Experiment

Matched the visible browser settings relevant to this case: whole-domain minimum/maximum cell size 1, selector `surface`, balanced quality, brick resolution 8, dt = 1/30 s. Both arms advance 100 steps from reset. Every sampled active brick is asserted to have resolution 8 (finest cells). The browser was closed before Dawn; processes ran serially under the repository GPU lease.

The diagnostic wraps `GPUDevice.createShaderModule` only for the ablation and replaces the body of `presentationHeightPolicyEnabled` with `return false`. The original A/B left production files unchanged. This is a causal test, not a proposed global production fix. A subsequently requested runtime control is described below.

| Time | Maximum publication displacement | Current height-Laplacian RMS | Height branch disabled |
| --- | ---: | ---: | ---: |
| 2.00 s | 39.46 mm | 44.55 mm | 41.95 mm |
| 3.00 s | 59.76 mm | 21.76 mm | 10.79 mm |
| 3.33 s | 147.75 mm | 24.55 mm | 16.72 mm |

Density checkpoint files are byte-identical across arms. With the height branch disabled, the published crossing agrees with the raw density-0.5 crossing within 0.011 mm. The 2-second maximum neighbour jump includes the central jet, so it is not a useful isolated seam metric. The Laplacian is the unscaled discrete height Laplacian, not physical curvature. These results measure the scalar field before rendering; they do not depend on lighting or triangle normals.

Captures and plots: `artifacts/axis-artifacts-2026-09-10/`, including `comparison.json` and `comparison.png`. The initial full-output attempt ran out of disk before 2 seconds; its files were deleted and it is excluded. The replacement probe keeps only three density and publication checkpoints.

## Reproduction

With the Fluid browser closed, run these commands serially:

```sh
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" POOL_MAX_CELL=1 POOL_STEPS=100 POOL_DT=0.03333333333333333 POOL_OVERRIDES='{"selectorMode":"surface"}' POOL_OUTPUT=artifacts/axis-artifacts-2026-09-10/base node --import tsx tools/probe-axis-artifacts-dawn.ts
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" AXIS_DISABLE_HEIGHT=1 POOL_MAX_CELL=1 POOL_STEPS=100 POOL_DT=0.03333333333333333 POOL_OVERRIDES='{"selectorMode":"surface"}' POOL_OUTPUT=artifacts/axis-artifacts-2026-09-10/no-height node --import tsx tools/probe-axis-artifacts-dawn.ts
uv run --offline --with numpy --with matplotlib python tools/analyze-axis-artifacts.py
```

The two successful captures and density equality support the diagnosis. ESLint passes for the diagnostic probe. The initial investigation did not change production behavior. A production fix should make the surface definition continuous across this validity boundary and then run the canonical gate, including thin-floor and coarse-waterline controls rather than globally discarding their height representation.


## Requested runtime control

Added **Simulation pipeline → Presentation pages → Column height**, default On. Off bypasses the column-height branch and uses density presentation. This is a runtime parameter: no solver reset or shader recompilation is required, and it applies on the next simulation step (Single step while paused). The setting is normalized, included in presets and URL/runtime parameter handling, forwarded at generation-zero publication, retained across resident replacement, and packed in a formerly reserved uniform lane. Common physics operators do not read this flag. On adaptive grids, presentation also feeds surface representability proofs, so a long adaptive run need not retain identical topology; the demonstrated bit-exact physics comparison is all-fine.

A production-path test switches the setting Off immediately before step 61, with no shader interception. At step 60 the published phi is byte-identical to the On baseline. At steps 90 and 100 it is byte-identical to the diagnostic Off arm. Density remains byte-identical to the baseline at all three checkpoints. See `runtime-verification.json`; reproduce with `AXIS_TOGGLE_HEIGHT_STEP=61` and output directory `runtime-toggle` using the same probe command.

The cleanup command initially failed while following a recursive `node_modules` symlink in an old capture. Its manifest traversal now uses `lstatSync` and skips symbolic links; deletion still removes the authorized artifact entries themselves. `npm run cleanup:artifacts` then succeeded: 658 entries, approximately 61.34 GiB reclaimed. Current captures were moved outside `artifacts` during cleanup and restored afterward.


## Residual valleys reported with the toggle Off

The user confirmed indented crescents/valleys remain after disabling the column-height branch. The visible panel was checked: Off is selected, all-fine enforcement and the surface selector are active. The running Dawn gate was interrupted before browser inspection to avoid concurrent WebGPU use.

The saved 3-second density field contains strong submerged depletion at the central X seam: at `(z=16,y=8)`, finest columns `x=31,32` have density approximately 0.699/0.696, while `x=30,33` have approximately 0.939/0.938. Integrated masses in those columns are 13.467/13.381 versus 16.990/16.995 finest-cell equivalents. Thus the underlying density has a narrow axial defect that changing the surface representation cannot remove. The identical Off publication and raw density crossing establishes this independently of renderer lighting. This does not yet attribute the defect to a specific physics operator, nor prove every visible valley has the same cause.

`residual-surfaces.png` compares normal shading of the density crossing, integrated column mass and the historical continuous local-volume reconstruction, all from the same saved fields. Column mass is an independent diagnostic, not a valid geometry oracle for depleted/overturning liquid. Its conspicuous central cross makes the submerged depletion clear.

A read-only stage-capture helper (`tools/axis-artifact-stage-audit.ts`, enabled by `AXIS_AUDIT_STEPS`) is prepared to isolate where depletion first appears. Additional Dawn execution requires the browser to be closed again.

The interrupted canonical gate is not a pass. It reported symmetric-expansion density D4 error 0.304505 versus 0.006 and topology/hydrostatic lane timeouts. Performance lanes also require an ignored baseline capture (`artifacts/sparse-cm12-ocean-b16-p16-stage-cost-baseline.json`) that the authorized full artifact cleanup removed. No performance baseline was regenerated or ceiling changed to mask that failure. Whole-tree TypeScript reports 52 existing errors outside the files changed for this control; targeted new/helper lint passes, while broader lint also reports existing hook-name violations in the resident.

### Does the depletion repeat at every brick seam?

A follow-up on the existing all-fine captures averages submerged density over y=4..11 and z=8..23, then compares the two cells touching each x brick boundary with their immediately adjacent cells. At 3 seconds, the centre boundary x=32 has means 0.6862/0.6821 against 0.9329/0.9310 beside it, a 0.2478 dip. Other interior brick boundaries x=8,16,24,40,48,56 have dip magnitudes below 0.0002 in this patch. At 2 seconds the central dip is 0.1738; at 3.33 seconds it is 0.2752.

This weakens a generic every-page transfer explanation. The central boundary is also the impact/reflection symmetry plane, so boundary-specific handling and stagnation-flow transport are still confounded. All-fine excludes coarse/fine restriction as a necessary cause, but does not exclude same-resolution shared-face sampling, sparse-page activation, or signed-coordinate boundary handling. A translated impact and a frozen-topology comparison would separate those explanations; the read-only stage audit should identify the first affected operator before any fix.

## Translated-impact test

Fresh `shift-center` and `shift-x1` arms use production Column height Off and otherwise identical method values and shader hashes. Only the sphere's X position moves by +0.05 m (one finest cell); tank, brick origin, enforcement and Z position stay fixed. The initial density arrays are exactly equal after that one-cell translation, including unchanged uniform tank fill. Both run to 3.333 seconds. The centered arm reproduces the earlier accepted density checkpoint files byte-for-byte.

At 1 second the strongest local two-column depression moves from edge x=32 to edge x=33: columns 31/32 become columns 32/33, inside one B8 brick. Its dip relative to its immediate flanks is 0.08857 centered and 0.08636 shifted. Thus this depletion can form away from a brick boundary. At 2 and 3 seconds the shifted dip drifts toward x=32 again; the tank was not translated, so later wall response and movement of the flow's stagnation region are confounded with fixed-grid position. The one-cell test does not rule out every later boundary contribution.

Stage audit at step 20 (0.667 seconds), measuring each arm at its own impact plane:

| Stage | Centered dip | Shifted dip |
| --- | ---: | ---: |
| Before transport | 0.048391 | 0.048386 |
| After conservative transport | 0.065684 | 0.065619 |
| After gamma diffusion | 0.058755 | 0.058688 |
| After sharpening | 0.058723 | 0.058654 |
| After capacity repair / scalar publication | 0.058470 | 0.058383 |

This identifies transport as deepening an existing local deficit at this step, independently of the impact landing on a brick seam. It does not yet identify the defective transport formula or exclude bad velocity from an earlier stage. At audited steps 20, 40 and 60, scalar-publication density is byte-identical to final accepted density in both arms: the subsequent topology transfer causes no scalar change at those checkpoints.

Reproduction: use `AXIS_IMPACT_SHIFT_X_CELLS=1`, `AXIS_AUDIT_STEPS=20,40,60`, `POOL_OVERRIDES='{"presentationColumnHeight":"off"}'`, and a distinct `POOL_OUTPUT` with `tools/probe-axis-artifacts-dawn.ts`. Omit the shift or use 0 for the centered arm. `tools/analyze-axis-impact-shift.py` asserts the initial integer translation and configuration equality, writes `impact-shift.json`, `impact-shift-stages.json`, and plots `impact-shift.png`.

### Four-cell offset confirms formation inside a brick

The third arm, `shift-x4`, moves the sphere +0.20 m along X. Its initial density is also exactly the integer translation of the centered state, and the recorded shader hash and method values match both prior arms. The run reaches 100 steps without a GPU validation failure.

At 1 second the strongest two-column dip is at edge 36 (columns 35/36), halfway through the brick spanning x=32..39. Its magnitude is 0.06909 against 0.08857 in the centered arm. At 2 seconds the depression is broader/weaker and its largest local two-column contrast is at edge 34; at 3 seconds that contrast is at edge 31. These later indices identify the largest local pair contrast in the diagnostic patch, not a geometrically exact stagnation plane. The density profiles visibly migrate and broaden rather than remaining pinned to the impact's initial brick edge.

The additional arm's scalar-publication and final accepted density arrays are again byte-identical at steps 20, 40 and 60. All three runs therefore show no density change during the subsequent topology transfer at those audited checkpoints. All active sampled bricks remain at the finest resolution.

**Interpretation:** fixed brick-boundary transfer is not needed to produce the initial depletion. The controlled translation and stage measurements implicate the transport/velocity interaction around the impact flow much more strongly than a page-copy seam. They do not yet establish the faulty transport expression, nor prove every artifact in every scene has this cause. Tank boundaries remain fixed, so late-time translated trajectories are not expected to be exact translations of one another.

Validation: initial-density translation error is exactly zero for offsets 0, 1 and 4 cells; shader and method equality assertions pass; the centered run reproduces the previous density checkpoints exactly; stage-to-final density maximum difference is zero at all nine audited checkpoints. ESLint passes for the probe and stage-audit helper, and `git diff --check` passes. This follow-up changes diagnostic tools and the report only; it does not introduce another production change or rerun the full regression gate. The four-cell test waited for an unrelated canonical regression run to release the GPU lease; two lock-rejected launch attempts executed no GPU work.

## Source audit: transport can generate an axial deficit without page seams

At the user's request, a subagent independently audited transport and coordinate handling while the parent audited pressure. Further GPU experiments were paused. The strongest mechanism found is the combination of backward donor-column normalization and independently forward-traced deficit redistribution:

- `traceGammaAndBeta` (resident WGSL around line 3749) sums receiver stencil contributions into each donor's beta.
- `cm12ConditionedRowCoefficient` (`lib/core/cm12-numerics.ts`, line 55) divides the backward weight times gamma by `max(1, donorBeta)`.
- `scatterDensityDeficit` (resident WGSL around line 3802) redistributes only donors' positive `1-beta` deficits using forward characteristics.
- `gatherConservativeDensity` (around line 3841) assembles density and records its coefficient row sum as gamma.

These operations conserve donor mass but do not enforce constant-density preservation at receivers. Where a velocity component changes sign, neighboring interpolation stencils reverse direction. The resulting beta pattern can have an axial ridge; normalization reduces the central receivers' incoming mass, while forward deficits from outward-moving neighboring donors cannot reach those receivers.

### Algebraic counterexample from the current formulas

Consider an unclipped, uniform all-fine interior with cell centers at half-integers relative to a flow symmetry plane, initial density and gamma both one, and affine divergence-free velocity `u=a*x, v=-a*y, w=0`. Choose `a*dt=0.1` and a local neighborhood in which the characteristic code uses one substep. Its midpoint RK2 backward trace gives factors `0.905` in x and `1.105` in y.

For donor x=0.5, backward weights from receivers x=-0.5, 0.5 and 1.5 sum to `0.0475 + 0.9525 + 0.1425 = 1.1425`. For each y donor contributing to receiver y=3.5, the corresponding sum is `0.895`. Therefore every donor used by receiver `(0.5,3.5)` has beta `1.1425*0.895 = 1.0225375`. Its gathered density is `1/1.0225375`, approximately **0.977959**, despite initially uniform density. The same applies at x=-0.5. The neighboring outward donor x=1.5 forward-traces to 1.6575, whose stencil cannot reach x=0.5; central donors have beta greater than one and emit no deficit. Thus forward redistribution does not restore this central band. This ignores fixed-point rounding, which is much smaller than the demonstrated deficit.

The issue is not solely RK2's map-volume error: an exactly volume-preserving backward affine map with factors `0.9` and `1/0.9` gives central beta `1.15*(8/9)=46/45` and gathered density `45/46`, again with no forward deficit reaching the central receivers. This is a consistency counterexample for the implemented transport construction, not a measured attribution of the captured scene's entire deficit.

### Amplification and persistence

`collocateAndDiagnose` (around line 6022) averages projected face velocities into cell vectors; `sampleEffectiveTransportVelocityAtSpansMode` (around line 3008) trilinearly samples those vectors for characteristic tracing. The native face-flux divergence diagnostic does not establish volume preservation of this interpolated characteristic map. That representation mismatch can contribute additional error, although the counterexample above does not require it.

Gamma diffusion (around line 4165) applies bounded neighbor transfers rather than enforcing unit row sums immediately. It can soften a band while transport repeatedly regenerates it, consistent with the previously captured stage deltas. Pressure's volume correction only responds to excess density. `preparePressure` (around line 5541) explicitly explains why underdensity is not given the opposite source: a previous attempt contracted transport-smoothed bulk liquid. Adding negative pressure recovery is therefore not an established repair.

The audited interpolation path contains ordinary floor/fract coordinates and world/solid clipping, with no tank-midpoint branch or active D4 reflection. This source mechanism predicts a defect associated with flow sign-change planes, consistent with initial impact-centered depletion and a possible later tank-centered pattern. It does not exclude a separate late-time page or boundary defect. The next repair investigation should address receiver consistency of the conservative transport/gamma scheme; no production numerical change was made during this audit.

## Implemented correction: retain cumulative gamma during backward gather

Following the request to implement a fix, checking the transport against the local CM12 paper exposed a specific implementation error. Section 3.4 step 4 evaluates gamma-prime **from gamma** using the conditioned backward interpolation. LAF11 likewise describes cumulative gamma as transported with the scalar to retain earlier incompressibility errors.

The existing code instead combined `A_backward * 1` with `A_forward * gamma_old`. That is neither the current operator's row sum nor cumulative gamma. The backward gather now uses `coefficient * sourceGamma[donor]`, matching density and the already-correct forward deficit. Both regular and packed-coarse sparse GPU paths, the sparse CPU reference, and the two-tile algebra oracle are corrected. No extra dispatches, iterations, density clamping, or pressure-volume sources are introduced.

An especially simple invariant exposes the omission: at zero velocity and uniform gamma `g`, the conservative density operator is identity. Old code changes gamma to `2g-g*g` for g below one, or to one for g above one, even though nothing moves. Correct code retains g. For nonzero motion on an all-wet closed grid, the corrected operator also conserves volume-weighted gamma and preserves constant `rho/gamma`, because it transports both fields identically.

This refines the earlier audit's description of gamma as a recorded row sum: that was what the backward code did, but it was the wrong history update. The first step from gamma=1 remains unchanged; the correction enables diffusion to respond to accumulated dilution/concentration. It does not make every individual transport step exactly constant-preserving, and its effect on the captured tank must still be measured.

Validation so far:

- A new repeated axial-flow CPU regression failed before the fix on cumulative gamma conservation; it passes after the fix on both uniform and mixed B8/B4 grids, conserving mass and gamma and retaining constant concentration for six steps.
- Stationary gamma cases 0.5, 0.75, 1.25 and 2 pass.
- The two-tile transport oracle passes all variants and six 128-step seam soaks; maximum mass error is below 6e-17.
- The broader sparse CPU test file has one unrelated large-CFL work-grid resolution failure, reproduced with temporary copies restoring the old transport implementation and its dynamics caller. No assertion was relaxed.
- A focused Dawn regression is prepared to execute both actual production gather kernels with stationary non-unit gamma. GPU verification and the canonical gate await closure of the reopened Fluid browser tab.

The separate uniform-reference solver has an analogous historical row-sum assumption, but its binding currently supplies advected gamma rather than old donor gamma. It is outside this Sparse CM12 correction and has not been changed by substituting the wrong gamma texture.

## Follow-up: broad recessed surface patches after the gamma correction

The reported recessed patches were investigated once the browser was closed. Both production gather kernels pass `tests/sparse-cm12-cumulative-gamma-dawn.test.ts`. A corrected all-fine half-pool run with Column height Off reaches 100 steps. At 3 seconds the submerged central-pair density dip is 0.03159, compared with 0.24781 before the gamma fix. At 3.333 seconds it is 0.01732 versus 0.27523. Final mass is 69715.47 versus initial 69724, improving on the pre-fix final 69706.35.

A second corrected run changes only Column height to On. Accepted density is byte-identical between On and Off at steps 60, 90 and 100. Nevertheless, height reconstruction lowers broad surface patches: maximum recession relative to Off is 47.4 mm at 2 seconds, 71.5 mm at 3 seconds and 70.0 mm at 3.333 seconds. At 3 seconds, 1520 of the 58×58 interior sample columns are lowered by more than 10 mm. The sharp curved borders in a shaded comparison resemble the reported screenshot. See `gamma-height-on-off.json` and `gamma-height-on-off-90.png` (On left, Off right); the image is a diagnostic normal shading of the published vertical zero crossing, not a renderer screenshot.

This identifies a concrete presentation mechanism for matching recesses: `presentationIntegratedColumnReceipt` accepts/rejects columns using hard monotonicity and anchoring thresholds; `cm12PresentationExactSample` selects integrated height for accepted columns and a density-derived scalar elsewhere. The corrected density changes that validity pattern. The two reconstructions disagree in height, producing abrupt borders even though the underlying accepted density is identical. Turning Column height Off removes this particular reconstruction mismatch; it does not imply all physical surface depressions should disappear. No additional production change was made in this follow-up.

The required canonical gate was run unchanged and **did not pass**: symmetric expansion reported D4 density error 0.42416 above 0.006; mini64 performance hit its 30-second timeout; the long-dam lane exhausted the remaining suite budget, leaving the final four lanes unrun. The overall budget was 180 seconds. Mini32 performance passed at 30.212 ms against its 40 ms ceiling. Full output is `/tmp/fluid-axis-gamma-gate.log`. This does not establish that every gate failure was introduced by gamma; the preceding run also had symmetry/timing failures, and unrelated workspace changes remain present.
