# Mass-Conserving Liquids: Exhaustive Uniform Implementation Map

**Reference:** [`mass-conserving-liquids-algorithm-reference.md`](mass-conserving-liquids-algorithm-reference.md)  
**Implementation audited:** the dense `uniform` WebGPU method in the current worktree  
**Audit date:** 13 August 2026
**Purpose:** map every algorithm, formula, invariant, edge case, and acceptance requirement in the reference to code; identify every observed difference and potential gotcha.

---

## 1. Scope and reading rules

This is a bidirectional audit:

1. **Reference to code:** all 37 entries in the reference's algorithm inventory are accounted for below.
2. **Code to reference:** implementation-only behavior on the live uniform path is catalogued in Section 13.
3. **Validation:** the relevant tests are mapped, including tests that currently fail.

The implementation boundary is not one file. The live path spans:

- method selection and defaults: [`lib/methods/uniform.ts`](../../lib/methods/uniform.ts)
- fixed paper step: [`lib/uniform-paper.ts`](../../lib/uniform-paper.ts)
- frame orchestration and resource wiring: [`lib/webgpu-uniform-reference.ts`](../../lib/webgpu-uniform-reference.ts)
- main density, velocity, solid, pressure-projection, and post-process kernels: [`lib/webgpu-uniform-reference.wgsl.ts`](../../lib/webgpu-uniform-reference.wgsl.ts)
- velocity extension: [`lib/webgpu-uniform-velocity-extrapolation.ts`](../../lib/webgpu-uniform-velocity-extrapolation.ts) and [`lib/webgpu-uniform-velocity-extrapolation.wgsl.ts`](../../lib/webgpu-uniform-velocity-extrapolation.wgsl.ts)
- CM11a pressure hierarchy: [`lib/webgpu-uniform-pressure-multigrid.ts`](../../lib/webgpu-uniform-pressure-multigrid.ts) and [`lib/webgpu-uniform-pressure-multigrid.wgsl.ts`](../../lib/webgpu-uniform-pressure-multigrid.wgsl.ts)
- marching-cubes rendering: [`lib/webgpu-water-pipeline.ts`](../../lib/webgpu-water-pipeline.ts) and [`lib/marching-cubes-lookup.wgsl.ts`](../../lib/marching-cubes-lookup.wgsl.ts)
- inflow extension: [`lib/inflow-boundary.ts`](../../lib/inflow-boundary.ts)

Line links identify the implementation audited on this date. Status labels mean:

- **Mapped:** the referenced operation is present with no material formula change found.
- **Variant:** the operation is present, but an underspecified engineering choice or a materially different discretization is used.
- **Extended:** the referenced operation is present and additional non-CM12 behavior executes.
- **Absent:** the reference includes the operation, but the uniform implementation does not implement it.
- **Dormant:** code exists in the shader source but is not compiled or dispatched by the uniform frame path.

“Potential gotcha” is deliberately broader than “confirmed defect.” It includes precision limits, unsupported boundary cases, stale diagnostics, missing guards, and implementation choices whose correctness depends on scene assumptions.

---

## 2. Executive differences and highest-risk gotchas

These are the material findings. They are expanded and code-linked later.

| ID | Finding | Classification | Consequence |
|---|---|---|---|
| D1 | The published excess-density divergence `min(0.5(rho'-1),1)/dx` is replaced by `min(rho'-1,1)/dt`. | Confirmed formula difference | At the published `dt=1/30`, `dx=0.05`, the small-excess slope is `30` instead of `10` per second; the cap is `30` instead of `20` and activates at excess `1` instead of `2`. |
| D2 | Interior cumulative transport gamma is clamped to `[0.5,2.5]` before beta construction and reset to `1` in cells whose new density is below `1e-5`; a partially or fully exterior released-wall gamma sample instead retains its deficient zero-extension value. | Confirmed algorithm difference | The method is still not the published unbounded cumulative-gamma recurrence, but conditioning no longer fills the gamma sample's missing exterior weight or invents a coefficient for a fully empty released trace. |
| D3 | A short-lived change multiplied the conditioned backward row sum by pre-advection `gamma_n[donor]` a second time. | Resolved regression | The extra snapshot and donor factor were removed. On exact Figure 9 at 4 ms, the bad recurrence erased the 97-layer reservoir by 0.344 s; the restored row-sum recurrence retains mass. |
| D4 | Gamma diffusion formerly used x-even/x-odd parity passes and two mirrored axis orders, repeated seven times by default. | Resolved defect | It now gathers both face fluxes from one immutable snapshot per axis, giving the source-prescribed Jacobi-within-axis update. One x/y/z repetition is the reference default; 2–7 are explicit extra diffusion. |
| D5 | Gamma diffusion formerly ignored solid connectivity and face openness. | Resolved defect | Each face flux is now weighted by the shared cut-cell open fraction, and a closed face transfers neither gamma nor density. |
| D6 | When solid excess has no valid scatter target, it is removed from live `rho` and only added to a telemetry counter. | Confirmed conservation difference | The simulation mass ledger loses that amount. Telemetry does not constitute a recovery buffer and does not restore mass later. |
| D7 | The no-valid-stencil fallback formerly credited beta to the destination's own index without a matching gather contribution. | Resolved defect | A fully empty released-wall stencil now contributes zero beta, so the donor's full missing column weight is recovered by the normalized forward remainder. Partially visible beta/gather stencils remain normalized. |
| D8 | Density transport at an open top does not implement an outflow ledger. Forward traces outside the grid fall back to the source cell. | Confirmed boundary difference | “Open” applies to pressure/velocity, but surface-density mass is effectively retained. Published open-boundary conservation accounting is absent. |
| D9 | Density characteristics use substepped midpoint/RK2 with per-substep domain clamping and a released-wall escape rule; velocity characteristics use one midpoint/RK2 trace with no explicit solid collision test. | Confirmed tracing variant | Density and velocity do not follow the same characteristic policy. Velocity can sample across a thin or moving solid even where density transport stops. |
| D10 | The sharpening trace uses 0.5-cell steps, nearest-cell gradients, no interpolated interface crossing, and at most five steps. | Confirmed reconstruction choice | It can overshoot the 0.5 contour, stall on quantized gradients, or miss a thin solid between tested endpoints. |
| D11 | Initial density is binary at cell centers rather than subcell occupancy/supersampling. | Confirmed initialization difference | Initial mass and interface position are resolution- and alignment-dependent; cut boundary cells do not start in `[0,1]` unless exactly dry/wet. |
| D12 | CM12's velocity-advection gap is exposed as selectable midpoint semi-Lagrangian or bounded MacCormack, and the force pass additionally applies explicit viscosity, CSF surface tension, and authored inflow velocity. | Declared extension | Comparisons are not CM12-only unless viscosity and surface tension are zero and the selected velocity profile is recorded. |
| D13 | The uniform path performs two-way rigid-body reaction/integration after projection. CM12 assumes one-way solid coupling. | Confirmed extension | Fluid and body states can diverge from the paper even if pressure and density kernels match. |
| D14 | The implementation has marching-cubes rendering but no oriented triangle-mesh volume integration. | Confirmed absence | `representedVolumeCellSum` is not enclosed mesh volume; it is a fixed-point sum of clamped occupancy. Paper mass-versus-visible-volume plots cannot be reproduced by current telemetry. |
| D15 | Diagnostic mass is accumulated in `u32` fixed point at 1/2048-cell resolution and clamps per-cell conservative density to `8`. | Confirmed diagnostic limitation | It is not a double-precision global mass reduction, can under-report cells above `8`, and can overflow on sufficiently large totals. |
| D16 | The Sec. 3.8 “scene” option is described and tested as scene-aware, but the gate currently returns true only for the literal value `"on"`. | Confirmed code/test inconsistency | Default `"scene"` mode is off for every scene. One non-GPU test and the GPU ceiling test's precondition disagree with live behavior. |
| D17 | The sharpening-step regression formerly assumed a literal `3*dt` after sharpening became runtime-tunable. | Resolved validation gap | The assertion now checks `3*dt*strength`; unit strength remains the paper value while the ceiling/gamma/boundary invariants stay independently guarded. |
| D18 | FIM and pressure convergence failures are telemetry only; the frame still consumes the resulting fields. | Confirmed runtime gotcha | A ceiling/cap failure is not rejected, retried, or rolled back. |
| D19 | Thin density below the pressure-liquid isovalue formerly received no gravity, semi-Lagrangian transport erased an existing away-wall boundary velocity, and projection reset its positive-wall face to the solid velocity. | Resolved separating-boundary defect | Sub-isovalue liquid now receives body force, transport preserves the away-wall sign, and projection clamps only motion into the wall. The wall inequality remains active even when the cell has no pressure row. |
| D20 | Density transport formerly classified cells by centre-point solid containment and ran before current-geometry excess ejection. | Resolved moving-solid conservation defect | A cut cell with `V>0` lost valid `rho<=V` donor mass when its centre was inside a body, while a newly covered `V=0` donor was zeroed before Sec. 3.6 could move its density. Current `V` is now reconciled before transport and `V>1e-5` defines the transport cell set. |

---

## 3. End-to-end frame map

The paper's four stages are present in the correct top-level order. The live encode order is:

| Order | Reference step | Live code |
|---:|---|---|
| 1 | Update body state/solid geometry inputs | body sync and parameter upload in [`advanceTo`](../../lib/webgpu-uniform-reference.ts#L734); geometry is evaluated procedurally in shader functions beginning at [`insideRigid`](../../lib/webgpu-uniform-reference.wgsl.ts#L203) |
| 2 | Velocity extrapolation | authority field in [`buildExtrapolationAuthority`](../../lib/webgpu-uniform-reference.wgsl.ts#L342), then FIM/hierarchy dispatch from [`encodeVelocityExtrapolation`](../../lib/webgpu-uniform-reference.ts#L695) |
| 3 | Current-geometry solid reconciliation | Sec. 3.6 scatter/resolve establishes the paper's `rho<=V` and `rho=0` for `V=0` state before transport can omit fully solid donors. |
| 4 | Conservative density advection | host dispatches [`traceGammaAndBeta`](../../lib/webgpu-uniform-reference.wgsl.ts), [`scatterDensityDeficit`](../../lib/webgpu-uniform-reference.wgsl.ts), and [`gatherConservativeDensity`](../../lib/webgpu-uniform-reference.wgsl.ts) |
| 5 | Gamma diffusion | seven iterations of paired axis diffusion |
| 6 | Density sharpening | compute/scatter/resolve |
| 7 | Post-density partial-solid excess ejection | the same Sec. 3.6 invariant is checked after advection, diffusion, and sharpening |
| 8 | Velocity advection and forces | selectable path |
| 9 | Pressure hierarchy and projection | fixed multigrid schedule and projection |
| 10 | Implementation-only two-way rigid coupling | post-projection reaction and integration |
| 11 | Optional render-only density reconstruction | presentation-only field |
| 12 | Fixed-point diagnostics | [`reduceDiagnostics`](../../lib/webgpu-uniform-reference.wgsl.ts) |

The order of CM12 Algorithm 1 is therefore preserved. The extra stages do not reorder density ahead of extrapolation or pressure ahead of velocity advection. The material differences are the added solid cleanup, two-way coupling, inflow, force models, and telemetry behavior.

### 3.1 Initialization and first-frame edge cases

| Reference requirement | Live mapping | Difference/gotcha |
|---|---|---|
| Initialize `rho` from liquid volume, preferably with subcell occupancy | [`initializeVolumeAndTerrain`](../../lib/webgpu-uniform-reference.ts#L623) evaluates the liquid and terrain predicates at cell centers and writes `0` or `1`. | Binary center sampling is D11; partial initial occupancy is not represented. |
| Initialize transport gamma to one | [`webgpu-uniform-reference.ts:645`](../../lib/webgpu-uniform-reference.ts#L645) fills both gamma textures with `1`. | Mapped exactly, including cells that are geometrically solid. The first advection/solid passes establish the live solid state. |
| Initialize pressure and velocity to zero | Newly allocated WebGPU textures and the explicit negative-face buffers at [`webgpu-uniform-reference.ts:648`](../../lib/webgpu-uniform-reference.ts#L648) start at zero. | Only negative-face boundary buffers are explicitly uploaded; the texture fields rely on WebGPU's zero-initialization guarantee. |
| Rasterize `V`, `Vf`, and solid velocity | Geometry is evaluated procedurally through `cellSolidFraction`, `faceSolidData`, and `solidVelocity` rather than initialized into persistent fraction fields. | Fractions are quadrature estimates and are recomputed in each consumer; see Section 8.2. |
| If initial `rho>V`, eject excess before pressure | The regular first-frame solid-excess dispatch occurs before pressure at [`webgpu-uniform-reference.ts:886`](../../lib/webgpu-uniform-reference.ts#L886). | It runs only when terrain or bodies are configured, and inherits the unplaceable-mass loss in D6. |
| Empty/full-domain validity | No special initialization branch exists; the same FIM, density, and pressure dispatch chain handles these cases. | No focused empty-domain or fully filled closed-domain regression was located. An authored open top deliberately creates exterior-air pressure behavior even for an otherwise full tank. |

---

## 4. Complete 37-item algorithm inventory map

### 4.1 Grid, frame, and transport primitives

| # | Reference algorithm | Status | Corresponding code | Differences and gotchas |
|---:|---|---|---|---|
| 1 | MAC staggered-grid storage | Variant | velocity allocation at [`webgpu-uniform-reference.ts:318`](../../lib/webgpu-uniform-reference.ts#L318); component sampling at [`sampleVelocityComponent`](../../lib/webgpu-uniform-reference.wgsl.ts#L183); separate negative-face buffers at [`webgpu-uniform-reference.ts:324`](../../lib/webgpu-uniform-reference.ts#L324) | Positive x/y/z faces are packed into one `rgba32float` texture with cell-grid dimensions, while three negative domain planes live in buffers. This is MAC geometry, but not three canonical `(nx+1,ny,nz)`-style component arrays. The unused `.w` channel and split boundary ownership make indexing mistakes easy. |
| 2 | Four-stage frame split | Mapped + extended | [`advanceTo`](../../lib/webgpu-uniform-reference.ts#L734) | Correct CM12 order. Adds partial-solid cleanup, two-way rigid coupling, optional rendering, inflow, and diagnostics. |
| 3 | Backward characteristic tracing | Variant | density trace [`integrateTraceOffset`](../../lib/webgpu-uniform-reference.wgsl.ts#L439); velocity trace [`departurePoint`](../../lib/webgpu-uniform-reference.wgsl.ts#L197) | Both use midpoint/RK2 rather than forward Euler. Density is substepped up to 16 times and collision-stopped by endpoint cell; velocity is one RK2 trace and has no solid-segment collision. |
| 4 | Forward characteristic remainder scatter | Variant | [`scatterDensityDeficit`](../../lib/webgpu-uniform-reference.wgsl.ts#L509) | Implemented with signed 20-bit-fraction fixed-point atomics. Rho and gamma logical scatters are fused. Outside/fully masked targets fall back to the donor cell, so no open-boundary outflow occurs. |
| 5 | Trilinear gather/scatter | Variant | density weights [`transportStencilWeight`](../../lib/webgpu-uniform-reference.wgsl.ts#L464); gamma gather [`sampleGammaStencil`](../../lib/webgpu-uniform-reference.wgsl.ts#L469); velocity interpolation [`sampleVelocityComponent`](../../lib/webgpu-uniform-reference.wgsl.ts#L183); sharpening scatter [`webgpu-uniform-reference.wgsl.ts:947`](../../lib/webgpu-uniform-reference.wgsl.ts#L947) | Backward beta/density transport and the forward conservative remainder normalize valid weights. The cumulative-gamma sample alone retains missing invalid/solid weight as the zero extension. The sharpening trace's scalar sample masks solids but does not renormalize. Velocity interpolation relies on the extrapolated zero shell and does not visibility-mask the sample stencil. |
| 6 | Donor column normalization | Mapped with quantization | beta construction and `max(1,beta)` scaling at [`webgpu-uniform-reference.wgsl.ts:478`](../../lib/webgpu-uniform-reference.wgsl.ts#L478) and [`webgpu-uniform-reference.wgsl.ts:541`](../../lib/webgpu-uniform-reference.wgsl.ts#L541) | Formula structure matches. Beta is quantized at `2^-20` per atomic contribution, so final donor sums are not exact even relative to the float weights used by gather. No production beta-sum assertion exists. |
| 7 | Cumulative gamma balancing | Variant | gamma textures initialized at [`webgpu-uniform-reference.ts:645`](../../lib/webgpu-uniform-reference.ts#L645); recurrence at [`webgpu-uniform-reference.wgsl.ts:478`](../../lib/webgpu-uniform-reference.wgsl.ts#L478) and [`webgpu-uniform-reference.wgsl.ts:533`](../../lib/webgpu-uniform-reference.wgsl.ts#L533) | Clamp `[0.5,2.5]` and dry-cell reset to `1` remain conditioning differences. The gather publishes the row sum of the already gamma-scaled, donor-normalized operator; applying donor gamma again is unstable in this conditioned matrix-free form. Solid cells are assigned gamma zero during advection, then may be averaged back above zero by diffusion. |
| 8 | Dimension-split gamma diffusion | Mapped with declared axis order | [`diffuseGammaAxis`](../../lib/webgpu-uniform-reference.wgsl.ts); one-repetition schedule in [`webgpu-uniform-reference.ts`](../../lib/webgpu-uniform-reference.ts) | Each cell gathers equal-and-opposite density and gamma face fluxes from the same axis snapshot, so the update is Jacobi within an axis and Gauss-Seidel between x/y/z sweeps. Cut-cell apertures block or scale face exchange. CM12 does not prescribe the axis traversal order; this implementation declares fixed x/y/z rather than averaging a custom mirrored operator. |
| 9 | CM12 three-scatter matrix-free advection | Variant | three live kernels at [`webgpu-uniform-reference.ts:834`](../../lib/webgpu-uniform-reference.ts#L834) | Three GPU passes are used, though steps 6 and 7 are fused and the third pass is a gather/resolve. Empty released-wall stencils leave beta untouched so the forward remainder remains conservative. Fixed-point atomics add deterministic quantization but still have integer overflow limits. |

### 4.2 Velocity extension

| # | Reference algorithm | Status | Corresponding code | Differences and gotchas |
|---:|---|---|---|---|
| 10 | Normal-extension PDE for air velocity | Reconstructed variant | [`upwindExtensionValue`](../../lib/webgpu-uniform-velocity-extrapolation.wgsl.ts#L194) | Implements the steady `n·grad(u)=0` upwind update on each native MAC component. Source classification uses effective density `rho/V`, not raw `rho` or a redistanced surface phi. This is a defensible choice but is not fixed by CM12. |
| 11 | JRW07 Fast Iterative Method | Mapped variant | source/front setup at [`seedActiveFront`](../../lib/webgpu-uniform-velocity-extrapolation.wgsl.ts#L118), anisotropic Godunov solve at [`godunovDistance`](../../lib/webgpu-uniform-velocity-extrapolation.wgsl.ts#L157), synchronous updates at [`updateActiveFront`](../../lib/webgpu-uniform-velocity-extrapolation.wgsl.ts#L255) | Uses `f=1`, a float sentinel `65504`, relative roundoff-scale convergence, and a fixed host ceiling. No local residual is checked in addition to change in distance. |
| 12 | JRW07 GPU tile FIM/reduction | Variant, not literal | indirect whole-domain schedule at [`webgpu-uniform-velocity-extrapolation.ts:304`](../../lib/webgpu-uniform-velocity-extrapolation.ts#L304) | Does not implement `8^3` active tiles or tile compaction. It repeatedly dispatches the whole domain indirectly until the aggregate active count reaches zero, with a pre-encoded ceiling of `max(nx,ny,nz)`. This preserves synchronous semantics but has different cost and activation behavior. |
| 13 | CM11b hierarchical far-field extrapolation | Mapped variant | hierarchy construction at [`webgpu-uniform-velocity-extrapolation.ts:195`](../../lib/webgpu-uniform-velocity-extrapolation.ts#L195), restriction/prolongation at [`webgpu-uniform-velocity-extrapolation.wgsl.ts:352`](../../lib/webgpu-uniform-velocity-extrapolation.wgsl.ts#L352) | Uses renormalized staggered trilinear transfers and a corresponding-cell fallback. Coarsening continues until the shortest axis collapses to one. Unknown open faces that remain unsupported are packed as zero; convergence telemetry does not reject the frame. The band distance is `2*max(h)`, which is wider than two cells along smaller-spacing axes on anisotropic grids. |

### 4.3 Sharpening and solid-density handling

| # | Reference algorithm | Status | Corresponding code | Differences and gotchas |
|---:|---|---|---|---|
| 14 | Godunov-style normal-flow sharpening increment | Mapped, anisotropic form | [`sharpenDeltaRho`](../../lib/webgpu-uniform-reference.wgsl.ts#L900) | The code algebraically folds the paper's `Delta x` and `1/Delta x^2` factors into `DeltaT/h_axis`. This is equivalent on cubic grids and a reasonable anisotropic generalization. |
| 15 | Sharpen jump limiter | Mapped | weight and neighbor maximum at [`webgpu-uniform-reference.wgsl.ts:916`](../../lib/webgpu-uniform-reference.wgsl.ts#L916) | Uses six face neighbors and `tau=0.4`. Out-of-domain and solid neighbors read as zero, so walls can contribute a full density jump and suppress sharpening. |
| 16 | CM12 local increment limiter | Mapped | [`webgpu-uniform-reference.wgsl.ts:919`](../../lib/webgpu-uniform-reference.wgsl.ts#L919) | Correct branch priority and `epsilon=1e-5`. |
| 17 | Gradient-field tracing with forward Euler | Variant | [`sharpenScatter`](../../lib/webgpu-uniform-reference.wgsl.ts#L932) | Five maximum forward-Euler steps of 0.5 cell, using the nearest cell-centered gradient. No interpolated crossing point, no step refinement, and only endpoint cell solid tests. The field is frozen after applying delta, which is consistent and deterministic. |
| 18 | Solid-aware trilinear local mass return | Mapped with quantization | [`webgpu-uniform-reference.wgsl.ts:947`](../../lib/webgpu-uniform-reference.wgsl.ts#L947) | Invalid/solid target weights are zeroed and renormalized; no-target fallback returns to the source. Each atomic contribution is rounded independently to `2^-20`, so per-source conservation has a quantization residual. Deposits can exceed one. |
| 19 | Face-fraction sharpening flux | Mapped variant | open-face terms at [`webgpu-uniform-reference.wgsl.ts:906`](../../lib/webgpu-uniform-reference.wgsl.ts#L906); face quadrature at [`faceSolidData`](../../lib/webgpu-uniform-reference.wgsl.ts#L282) | `Vf` is four-sample face quadrature, not exact solid rasterization. Shared values are procedurally recomputed, not stored once; identical calls should match, but moving geometry and floating arithmetic remain implicit dependencies. |
| 20 | Excess-density ejection from partial solids | Variant | [`scatterSolidExcess`](../../lib/webgpu-uniform-reference.wgsl.ts#L999) and [`resolveSolidExcess`](../../lib/webgpu-uniform-reference.wgsl.ts#L1027) | Uses `S=1` and a finite-difference signed-distance gradient. Zero gradient leaves the target at the source position. Targets are excluded only when their center is inside a solid, not when partially covered. Simultaneous deposits can overfill receivers. Unplaceable mass is removed and only metered, not retained for recovery. Geometry is current-position only; no swept solid interval is used. |

### 4.4 Pressure classification and projection

| # | Reference algorithm | Status | Corresponding code | Differences and gotchas |
|---:|---|---|---|---|
| 21 | Effective occupancy `rho/V` | Mapped variant | [`pressureDensityOpen`](../../lib/webgpu-uniform-reference.wgsl.ts#L137) and [`pressureDensity`](../../lib/webgpu-uniform-reference.wgsl.ts#L142) | Uses `V<=1e-5` as solid. One-cell continuation takes the **maximum** adjacent open-cell effective density, not an average. This preserves any adjacent liquid classification but can expand the pressure-liquid set around mixed neighborhoods. |
| 22 | Approximate density-derived liquid phi | Mapped | [`pressurePhi`](../../lib/webgpu-uniform-reference.wgsl.ts#L158) | Uses `dx=min(hx,hy,hz)` for rectangular cells. No redistancing. Coarse levels apply CM11a sign-aware averaging and one-cell continuation. |
| 23 | Ghost-fluid free-surface pressure | Mapped variant | main fraction [`ghostFluidFraction`](../../lib/webgpu-uniform-reference.wgsl.ts#L165); hierarchy fraction [`mgTheta`](../../lib/webgpu-uniform-pressure-multigrid.wgsl.ts#L65); projection [`webgpu-uniform-reference.wgsl.ts:803`](../../lib/webgpu-uniform-reference.wgsl.ts#L803) | `theta` is clamped to `0.05`, but clamp frequency is not reported. The authored open top uses a synthetic exterior phi and a special coefficient path. |
| 24 | Variational cut-cell pressure projection | Mapped variant | geometry and divergence at [`pressureFaceData`](../../lib/webgpu-uniform-reference.wgsl.ts#L318) and [`divergenceAt`](../../lib/webgpu-uniform-reference.wgsl.ts#L731); matrix coefficients at [`mgCoefficientRaw`](../../lib/webgpu-uniform-pressure-multigrid.wgsl.ts#L69) | Distinguishes face aperture from CM11a face-centered overlapping dual volume, which is correct. Geometry uses eight point samples. Body-union volume uses the maximum single-body fraction rather than union coverage, so overlapping solids are approximate. |
| 25 | Separating-wall complementarity | Mapped node-level approximation | fine lower bounds at [`mgBuildFinestRhs`](../../lib/webgpu-uniform-pressure-multigrid.wgsl.ts#L129); projected updates at [`mgSmoothColour`](../../lib/webgpu-uniform-pressure-multigrid.wgsl.ts#L288) | Domain halo and center-classified solid nodes use `p_min=0`. Partially covered cells whose centers are outside the solid remain unconstrained. Complementarity velocity/product residual is not reported; diagnostics report a projected pressure-system residual and pressure gap. |
| 26 | Projected red-black Gauss-Seidel | Mapped | host sweeps at [`webgpu-uniform-pressure-multigrid.ts:382`](../../lib/webgpu-uniform-pressure-multigrid.ts#L382); shader update at [`mgSmoothColour`](../../lib/webgpu-uniform-pressure-multigrid.wgsl.ts#L288) | Projection is folded into both color passes, removing a separate projection pass. This should preserve sweep-exit values, but is an optimized schedule rather than literal pseudocode. |
| 27 | Bubble-aware multigrid coarsening | Mapped | constants at [`webgpu-uniform-pressure-multigrid.ts:9`](../../lib/webgpu-uniform-pressure-multigrid.ts#L9); coarsening at [`mgDownsampleTopology`](../../lib/webgpu-uniform-pressure-multigrid.wgsl.ts#L145) | `C=2`; border/dual-volume handling has explicit domain-wall overrides. No regression directly verifies small-bubble survival end to end. |
| 28 | Constraint-aware V-cycle | Mapped | [`vCycle`](../../lib/webgpu-uniform-pressure-multigrid.ts#L398) | Four pre- and four post-sweeps; constraints transfer through three fine-to-coarse boundaries. Every invocation executes the fixed schedule regardless of fine residual. |
| 29 | Constraint-aware Full-Cycle | Mapped | [`fullCycle`](../../lib/webgpu-uniform-pressure-multigrid.ts#L413) | Three Full-Cycles precede four V-cycles, matching the CM11a performance schedule. Coarsest solve is capped at 4096 sweeps with `1e-4 s^-1` projected residual tolerance; a cap failure does not abort projection. |
| 30 | Capped excess-density divergence feedback | **Formula difference** | [`volumeCorrectionDivergence`](../../lib/webgpu-uniform-reference.wgsl.ts#L749); RHS use at [`webgpu-uniform-pressure-multigrid.wgsl.ts:134`](../../lib/webgpu-uniform-pressure-multigrid.wgsl.ts#L134) | Code is `min(max(rho'-1,0),1)/dt`, not `min(0.5(rho'-1),1)/dx`. The nearby comment claims the published small-excess slope is retained, but the executable expression does not retain it. |

### 4.5 Velocity, forces, rendering, and evaluation

| # | Reference algorithm | Status | Corresponding code | Differences and gotchas |
|---:|---|---|---|---|
| 31 | Velocity advection (CM12 specification gap) | Declared variant | option declaration at [`lib/methods/uniform.ts:13`](../../lib/methods/uniform.ts#L13); midpoint semi-Lagrangian at [`semiLagrangianAdvection`](../../lib/webgpu-uniform-reference.wgsl.ts#L637); MacCormack at [`boundedMacCormack`](../../lib/webgpu-uniform-reference.wgsl.ts#L670) | The implementation honestly exposes the choice. Default is one-pass semi-Lagrangian. MacCormack uses forward prediction, predicted-field extension, reverse trace, and donor-extrema fallback. Neither path collision-tests the complete velocity characteristic against solids. |
| 32 | Forward-Euler external force integration | Extended | [`applyVelocityForces`](../../lib/webgpu-uniform-reference.wgsl.ts#L606) | Gravity is explicit Euler on faces. The same function also applies explicit molecular viscosity, balanced-force CSF surface tension, and inflow velocity. Explicit diffusion has its own timestep stability limit despite semi-Lagrangian advection. |
| 33 | Detail discriminator and density amplification | Mapped | [`webgpu-uniform-reference.wgsl.ts:1034`](../../lib/webgpu-uniform-reference.wgsl.ts#L1034) | Implements `g=2 min(rho,.5)` and `rho_render=rho/min(max(g_blur,.01),1)`. Render isolation is correct. The UI/pipeline text says it reads simulation gamma, but the kernel derives a separate discriminator from rho. |
| 34 | Separable Gaussian blur | Mapped variant | [`blurPostprocessAxis`](../../lib/webgpu-uniform-reference.wgsl.ts#L1038) | Sigma is two cells and radius is six cells. Domain edges ignore invalid samples and renormalize. Samples are not masked by solids, so support can blur through rigid bodies or terrain. |
| 35 | Marching cubes | Mapped in shared renderer | lookup table at [`lib/marching-cubes-lookup.wgsl.ts:9`](../../lib/marching-cubes-lookup.wgsl.ts#L9); classification/interpolation at [`webgpu-water-pipeline.ts:458`](../../lib/webgpu-water-pipeline.ts#L458); polygonization at [`webgpu-water-pipeline.ts:495`](../../lib/webgpu-water-pipeline.ts#L495) | Uses the classic 256-case table and linear 0.5 crossings. Vertices are emitted per cube, not shared/welded. Ambiguous cases are not asymptotically resolved. A virtual zero side/top boundary closes optical meshes; the floor extends the bottom value, so extracted boundary geometry is not simply the paper's in-domain scalar contour. Capacity clipping can omit complete triangles. |
| 36 | Oriented mesh volume integration | **Absent** | no implementation on the uniform telemetry path | Triangles are oriented against the reconstructed gradient, but no `dot(a,cross(b,c))/6` reduction exists. `representedVolumeCellSum` must not be described as marching-cubes enclosed volume. |
| 37 | Global mass integration | Variant diagnostic | [`reduceDiagnostics`](../../lib/webgpu-uniform-reference.wgsl.ts#L1057) and [`readStats`](../../lib/webgpu-uniform-reference.ts#L1012) | Accumulates `round(clamp(rho,0,8)*2048)` into `u32`, not double precision and not multiplied by physical cell volume. It has per-cell clipping, quantization, possible integer overflow, and no outflow/source/sink ledger beyond the host's expected inflow volume. |

**Inventory closure:** 37 of 37 reference entries are accounted for. Item #12 replaces the literal tiled GPU FIM schedule with a whole-domain indirect schedule; item #36 is absent; the remaining 35 are present as mapped, variant, extended, or shared-infrastructure operations. No reference inventory row is unclassified.

---

## 5. Conservative density advection: formula-by-formula map

### 5.1 State and storage

| Mathematical object | Implementation |
|---|---|
| `rho_n` | `volumeA` before the step; despite the name, it stores surface density, not geometric volume or physical density. See bindings at [`webgpu-uniform-reference.ts:493`](../../lib/webgpu-uniform-reference.ts#L493). |
| `rho_{n+1}` | `volumeB` after gather and throughout diffusion/sharpening, copied back to `volumeA` at [`webgpu-uniform-reference.ts:892`](../../lib/webgpu-uniform-reference.ts#L892). |
| `gamma_n` | `gammaA` before advection. |
| backward-advected gamma | `gammaB` written by `traceGammaAndBeta`. |
| final gamma before diffusion | `gammaA` written by `gatherConservativeDensity`. |
| beta | first `cellCount` signed atomic integers in `conditioningScratch`, viewed through `sharpenDeposits`; scale `2^20`. |
| rho forward deposits | second `cellCount` atomic region. |
| gamma forward deposits | third `cellCount` atomic region. |

### 5.2 Backward trace and gamma advection

Reference:

`gamma_adv[i] = sum_l w_minus[l,i] gamma_n[l]`.

Code:

1. Integrates a substepped midpoint characteristic.
2. Masks invalid/solid corners.
3. Samples cumulative gamma with the unnormalized masked weights, but normalizes the surviving stencil for beta construction and density gather.
4. Clamps a fully interior `gamma_adv` to `[0.5,2.5]`; a partially or fully exterior released-wall gamma sample has only an upper bound and retains its deficient value, including zero.

The mixed normalization is important: cumulative gamma sees the scalar field's zero exterior, while beta and density gathering retain the normalized interpolation used by the conservative operator. A controlled exact-Figure-9 comparison found that making beta/gather deficient caused gamma diffusion to replenish more lid mass than advection removed; retaining normalized visible weights allowed the separating downward face velocity to drain the layer.

### 5.3 Beta construction

Reference:

`beta[l] += w_minus[l,i] gamma_adv[i]`.

Code at [`webgpu-uniform-reference.wgsl.ts:494`](../../lib/webgpu-uniform-reference.wgsl.ts#L494):

- visible backward beta weights are normalized by their total;
- each contribution is rounded to a signed integer at scale `2^20`;
- if no weight remains, beta receives no contribution.

If `total<=1e-9`, beta receives no synthetic self contribution and gather likewise has no backward term. The donor therefore retains a full deficit for the forward remainder in steps 6-7.

### 5.4 Backward density and gamma contributions

Reference:

`scale = gamma_adv[i] / max(1,beta[l])`

`rho_next[i] += scale * w_minus[l,i] * rho_n[l]`

`gamma_prime[i] += scale * w_minus[l,i] * gamma_n[l]`.

Code at [`webgpu-uniform-reference.wgsl.ts:541`](../../lib/webgpu-uniform-reference.wgsl.ts#L541):

`scaled = gamma_adv * weight / max(1,beta(donor))`

`rhoNext += scaled * volume(donor)`

`gammaNext += scaled`.

The density expression maps directly. The live implementation treats `gammaNext` as the row sum of the already gamma-scaled, donor-normalized transport operator: step 1 supplied `gamma_adv`, and that value is already present in `scaled`. A short-lived implementation of the literal extra `gamma_n[donor]` factor compounded the cumulative field in this conditioned recurrence. In the exact 128x128x64 Figure 9 scene at 4 ms it moved the reservoir toward the ceiling and then erased all wet cells by 0.344 s. Restoring the operator-row sum completed 0.4 s with raw volume drift `-3.69e-5`, maximum density `1.000958`, and no disconnected component.

### 5.5 Forward remainder

Reference:

`rho_next[k] += rho_n[j] * (1 - beta[j]) * w_plus[j,k]`

`gamma_next[k] += gamma_n[j] * (1 - beta[j]) * w_plus[j,k]`.

Code at [`webgpu-uniform-reference.wgsl.ts:510`](../../lib/webgpu-uniform-reference.wgsl.ts#L510) maps both formulas and fuses their atomics. It renormalizes visible forward weights. If none remain, it deposits both corrections back to donor `j`.

Gotchas:

- there is no explicit outflow; an out-of-domain forward trace returns to its source;
- the deficit is suppressed below `1/2^20`;
- each atomic term is rounded independently;
- `i32` atomic capacity bounds the total deposit per destination;
- inflow is added after the conservative operator and capped by `1-rhoNext`, so the host's analytic expected inflow can exceed actual inserted density when receiver cells lack capacity.

### 5.6 Dry-cell gamma reset

After both backward and forward terms, code sets `gammaNext=1` whenever `rhoNext<1e-5`. This is an implementation-only stabilization. It avoids gamma compounding in dry convergence regions, but it also destroys cumulative row-sum history and makes gamma depend on density support, which the published algebra does not.

### 5.7 Conservation precision

The algebraic donor-sum proof applies only to the coefficients actually used. Here beta is rounded before scaling, while gather recomputes floating trilinear weights. Therefore conservation is bounded by fixed-point beta/deposit quantization and atomic integer capacity, not merely floating-point reduction roundoff. There is no runtime reduction of `sum_j |beta_j-1|` in production; beta is only copyable under the opt-in symmetry audit.

---

## 6. Gamma diffusion map

The pair formula is faithful:

`m = rho_high * (gamma_high-gamma_low)/(2 gamma_high)`

and both gamma values become their average. The implementation correctly gives each invocation one output texel and explicitly copies unpaired boundary cells.

The scheduling is a variant:

```text
repeat 7 times:
    x-even pairs  -> ping
    x-odd pairs   -> pong, reading x-even output
    y-even pairs  -> ping
    y-odd pairs   -> pong
    z-even pairs  -> ping
    z-odd pairs   -> pong
```

Consequences:

- pair parities within an axis are ordered, not a single Jacobi snapshot;
- x, y, z order is fixed and can introduce measurable directional bias;
- every valid neighbor pair is eligible, including liquid-solid and air-solid pairs;
- the shader clamps negative output density to zero, which would hide rather than ledger any negative overshoot;
- the pipeline UI says “6 passes · 1 paper iteration,” while the encoder actually runs 42 passes/seven iterations.

---

## 7. Sharpening and partial-solid handling map

### 7.1 Equations 4–15

The six directional changes, Godunov positive/negative norms, cubic weight, `tau=0.4`, `DeltaT=3dt`, and six-neighbor maximum difference map to [`sharpenDeltaRho`](../../lib/webgpu-uniform-reference.wgsl.ts#L900). For cubic cells the rearranged factors yield the same final `dt/dx` scaling.

For anisotropic cells, the code generalizes each axis independently. The reference and paper assume one `Delta x`, so anisotropic equivalence is not source-specified.

### 7.2 Equation 17

The exact branch order maps:

1. zero a would-be negative or tiny cell;
2. do not modify `rho>0.5`;
3. retain the proposed delta otherwise.

### 7.3 Algorithm 2 trace and scatter

The code freezes `rho+delta` in `volumeA`, traces every negative delta against that field, and accumulates fixed-point deposits before resolve. This avoids read/write races.

Differences/gotchas:

- step length is 0.5 cell rather than the audit reference's robust recommendation of at most roughly 0.25 cell;
- gradient is sampled from `floor(p)`, not trilinearly interpolated;
- crossing is detected only after a complete step and the crossing position is not interpolated;
- solid crossing is tested only at the candidate endpoint's center-classification cell;
- a partial solid with open fraction below one is not necessarily considered “inside” for trace blocking;
- density sampling masks solid corners without renormalization, which can delay or prevent detection of the 0.5 contour near walls;
- fixed-point scatter rounding leaves a small per-donor mass residual;
- no diagnostic records stalled traces, maximum-distance exits, solid hits, fallback count, or actual maximum transport distance.

### 7.4 Partial-solid excess

The source is reduced to `min(rho,V)` and excess is scattered one minimum-cell-size away along the solid SDF gradient. This maps the intended main path.

For moving geometry, the same Sec. 3.6 operation is also applied before conservative density transport. This follows the paper's own state definitions: Sec. 3.6 removes `rho-V` and guarantees `rho=0` inside a completely solid cell; Sec. 3.7 defines that completely solid case by `V=0`. Applying the invariant only after transport allowed Sec. 3.4 to discard a newly covered donor first. The transport domain therefore uses `V>1e-5`; a centre-inside cut cell with `V>0` remains a valid donor holding at most `V` density. The post-density Sec. 3.6 pass remains because advection, gamma diffusion, and sharpening can create new `rho>V` states. Neither pass rescales global mass or applies a post-step volume correction.

The no-target path first searches the immediate 26-neighbourhood for an open receiver. If none exists, `rho-excess` has already been stored and only `reductions[4]` receives the amount. There is no persistent recovery field, retry, or source restoration. The public field `uniformUnplaceableSolidExcess_cells` is therefore a **mass-loss receipt**, not conserved pending mass. The Dawn moving-solid regression requires this receipt to remain zero.

---

## 8. Pressure and boundary map

### 8.1 Effective fill and phi

`rho'=rho/V` maps with a `1e-5` open-fraction threshold. Adjacent fully solid cells use the maximum effective density among open face neighbors. Pressure phi is `-(rho'-0.5)min(h)`.

Potential gotchas:

- maximum continuation can classify a solid cell liquid because of a single wet neighbor even when five neighbors are dry;
- the same continued solid cell becomes a pressure unknown before CM11a's `p_min` classification;
- a cell can have geometric `V=0` under quadrature while its center is not `cellInsideSolid`, leaving its pressure lower bound unconstrained;
- no count is published for tiny-V suppression or continued-solid unknowns.

### 8.2 Cut-cell divergence and matrix

The code intentionally distinguishes:

- face aperture `Vf` for sharpening and open/closed MAC faces;
- face-centered overlapping dual-cell volume for the CM11a pressure energy.

The RHS uses CM11a equations 8–10, including solid-velocity terms, and the pressure matrix uses compatible dual-volume/h² coefficients. Coefficients are baked once per level after phi continuation and reused by smoothing and residual evaluation.

Geometry gotchas:

- cell solid fraction uses eight offset samples at `±0.4` cell;
- face aperture uses four samples at `±0.35` transverse offsets;
- pressure dual volume uses eight `±0.4` samples;
- overlapping rigid bodies use the maximum body fraction rather than union occupancy;
- moving-solid geometry is evaluated at one time sample and characteristic collision is not swept;
- body velocity is extrapolated from the nearest body within one cell when dual samples contain no solid.

### 8.3 Ghost fluid and open top

Both liquid-to-air orientations receive the same theta-scaled coefficient, fixing a common asymmetric bake error. `theta_min=0.05` is a declared conditioning choice.

The open top is a code extension with synthetic exterior phi, an open pressure face, zero exterior pressure, and no corresponding surface-density outflow. Therefore the velocity/pressure and mass boundary models are not mutually open.

### 8.4 Separating complementarity and multigrid

The fixed schedule maps CM11a:

- hierarchy depth from the dyadic minimum grid axis;
- `C=2` bubble-preserving levels;
- `S=3` constraint transfer depth;
- four pre- and four post-PRBGS sweeps;
- three Full-Cycles then four V-cycles;
- high-precision-style coarsest solve using double-single arithmetic;
- projected residual diagnostics.

Potential gotchas:

- construction rejects non-dyadic minimum dimensions or a coarsest padded grid above 256 cells;
- the fixed schedule does not stop early on a fine residual;
- a coarsest cap failure is reported but the frame still projects with the result;
- no all-Neumann component/gauge analysis is explicit; the solid halo and complementarity formulation are relied on to regularize closed regions;
- the reported fine residual is a projected pressure-system residual, not the full volume-weighted post-projection divergence and complementarity product requested by the reference;
- no post-projection `L2`, `Linf` divergence, penetration velocity, or separation velocity is measured.

### 8.5 Excess-density divergence formula

Published:

`s_paper = min(0.5 * max(rho'-1,0), 1) / dx`.

Implemented:

`s_code = min(max(rho'-1,0), 1) / dt`.

At `dt=1/30` and `dx=0.05`:

| Excess `e=rho'-1` | Paper | Code |
|---:|---:|---:|
| small `e` | `10e s^-1` | `30e s^-1` |
| `e=1` | `10 s^-1` | `30 s^-1` |
| `e>=2` | capped at `20 s^-1` | capped at `30 s^-1` from `e>=1` |

This is the clearest formula-level deviation in the pressure path. The shader comment describes a resolution-independent reinterpretation, so it appears intentional, but it must not be presented as the published lambda/eta formula.

---

## 9. Velocity advection, forces, and rigid coupling

CM12 leaves velocity advection unspecified. The implementation records two profiles:

| Profile | Code | Notes |
|---|---|---|
| midpoint semi-Lagrangian | [`semiLagrangianAdvection`](../../lib/webgpu-uniform-reference.wgsl.ts#L637) | default; one backward trace per MAC component |
| bounded MacCormack | prediction/reverse/correction at [`webgpu-uniform-reference.ts:895`](../../lib/webgpu-uniform-reference.ts#L895) | CM11b-derived option; prediction is extrapolated before reverse tracing; correction falls back outside donor extrema |

Both apply forces exactly once. Additional behavior beyond the minimal reference is:

- explicit molecular viscosity `nu=mu/rho` via a seven-point velocity Laplacian;
- CSF surface tension using density-derived normals and curvature;
- authored inflow velocity enforcement;
- negative and positive domain MAC face special handling;
- two-way rigid reaction, drag-like impulse telemetry, and pose integration after projection.

Gotchas:

- explicit viscosity is not unconditionally stable; the paper's large `dt` can exceed a diffusion stability bound for high viscosity or fine grids;
- surface tension is not part of the audited CM12 core and adds its own capillary timestep concern;
- the velocity trace samples the hierarchy-filled field but does not stop a segment at a solid;
- positive high-domain faces are preserved during the one-pass advection path while negative faces are carried in separate buffers, so boundary symmetry depends on later projection;
- post-projection two-way coupling is intended not to rewrite velocity, but it still adds a non-CM12 body-state evolution path.

---

## 10. Rendering and evaluation map

### 10.1 Sec. 3.8 reconstruction

The shader implementation matches the published discriminator, Gaussian sigma, truncation radius, denominator floor, and amplification formula. The result is kept in `surfaceB`, while simulation continues from `volumeA`; render isolation is real.

Differences/gotchas:

- blur ignores invalid domain samples and renormalizes but does not mask/renormalize around solids;
- a tiny positive speck can be amplified up to 100x;
- the pipeline description says “reads surface density, gamma,” but transport gamma is not read;
- the `"scene"` gate is currently always false; only explicit `"on"` enables reconstruction;
- initial reconstruction runs only when the gate is already true.

### 10.2 Marching cubes

The shared renderer extracts a 0.5 contour with the standard Lorensen–Cline table, linear edge interpolation, and gradient-consistent triangle orientation.

Differences/gotchas:

- vertices are duplicated per cube, so the output is visually contiguous but not a welded topological mesh;
- classic ambiguous cases are accepted without an asymptotic decider;
- tank side/top boundaries receive a virtual zero layer; the floor receives a constant extension;
- capacity clipping can discard triangles and therefore break watertightness;
- no mesh validation or enclosed-volume integration follows extraction.

### 10.3 Telemetry is not paper visible volume

The live diagnostics publish:

- `volumeCellSum`: fixed-point sum of `clamp(rho,0,8)`;
- `representedVolumeCellSum`: fixed-point sum of `clamp(rho,0,1)` from the raw simulation field;
- `front_m`: maximum x index with raw `rho>=0.5`;
- `maxSpeed_m_s`: maximum packed face-vector magnitude.

Neither volume statistic is marching-cubes enclosed volume. Even with Sec. 3.8 enabled, the reduction bind group uses raw `volumeA` as `surfaceIn`, so “represented” does not measure the reconstructed render field.

---

## 11. Core invariant and acceptance-gate coverage

| Reference invariant/gate | Implementation coverage | Gap/gotcha |
|---|---|---|
| Finite state after every kernel | Not implemented as a per-kernel check | No NaN/Inf counters; bitcast max diagnostics are not a finite-state audit. |
| Nonnegative rho | Local `max(...,0)` in gather/diffusion and Eq. 17 in sharpening | No global assertion or repair ledger; clamping can hide negative numerical mass. |
| Closed-domain global mass | Conservative operator plus fixed-point diagnostics | No beta invariant check, double reduction, or per-stage mass ledger. Unplaceable solid excess is removed. |
| Open-boundary mass ledger | Absent | No cumulative outflow; forward outside fallback retains mass. Host reference volume accounts only expected inflow. |
| Solid exclusion | Solid gather writes zero; later excess ejection | Gamma diffusion can repopulate solids. Unplaceable excess is lost. No final `rho<=V` assertion. |
| Donor beta equals one | Algebraically intended | Fixed-point mismatch; no production diagnostic. Optional audit copies raw beta only. |
| Separating pressure `p>=0` | Enforced at constrained nodes during every color update | Constraint classification is center/halo based; no direct post-solve minimum-pressure statistic is published. |
| Projection residual | Fine projected pressure residual published | No post-projection divergence norms or component-wise flux ledger. |
| Rendering isolation | Implemented | Correct: `surfaceA/B` never replace `volumeA/B` in simulation bindings. |
| Failed masked scatter is conservative | Sharpen fallback and density forward fallback are conservative; an empty density backward stencil leaves the full forward deficit | Solid-excess no-target path is explicitly non-conservative. |
| No characteristic crosses a solid | Partial | Density checks substep endpoints; velocity has no solid crossing; no swept moving-solid test in kernels. |
| CPU/deterministic reference comparison | Not part of routine uniform tests | Most tests inspect source strings; no explicit sparse-matrix CM12 density oracle is present in the focused suite. |
| Serialize algorithm variants/thresholds | Partial | UI captures velocity, timestep, and render gate. Gamma clamp/reset, fixed-point scale, trace step, theta clamp, excess law, and geometry quadrature are hard-coded and not serialized as an algorithm profile. |

The implementation does not currently satisfy all 12 acceptance-gate items in Section 22 of the algorithm reference. In particular, gates 2–4, 6–9, and 11–12 are incomplete or only partially evidenced.

---

## 12. Test coverage and current test state

### 12.1 What the focused tests cover

| Area | Tests | Nature of evidence |
|---|---|---|
| Method defaults and shader entry points | [`uniform-reference-method.test.ts`](../../tests/uniform-reference-method.test.ts) | Mostly static option and source-presence assertions; optional one-step WebGPU smoke |
| Gamma pair formula and dispatch order | [`uniform-gamma-snapshot-diffusion.test.ts`](../../tests/uniform-gamma-snapshot-diffusion.test.ts) | Static regex assertions, not numerical pair-field execution |
| Sharpen `DeltaT=3dt` | [`uniform-paper-sharpening-step.test.ts`](../../tests/uniform-paper-sharpening-step.test.ts) | One source regex |
| Velocity profile wiring and released wall | [`uniform-paper-velocity-advection.test.ts`](../../tests/uniform-paper-velocity-advection.test.ts) | Source/paper regex assertions |
| Pressure boundary transfer and coefficient bake | [`uniform-pressure-boundary-transfer.test.ts`](../../tests/uniform-pressure-boundary-transfer.test.ts), [`uniform-pressure-coefficient-bake.test.ts`](../../tests/uniform-pressure-coefficient-bake.test.ts) | Static source assertions |
| FIM terminal-active telemetry | [`uniform-fim-convergence-diagnostics.test.ts`](../../tests/uniform-fim-convergence-diagnostics.test.ts) | Static source assertions |
| Published Figure 9 harness constants | [`mass-conserving-paper-cases.test.ts`](../../tests/mass-conserving-paper-cases.test.ts) | Numerical scene configuration only |
| Ceiling separation and mass | [`uniform-ceiling-separation-gpu.test.ts`](../../tests/uniform-ceiling-separation-gpu.test.ts) | 300-step, 1.2 s GPU readback after the lid-impact window when WebGPU is configured |
| Moving-solid displacement | [`run-uniform-displacement-smoke.ts`](../../tools/run-uniform-displacement-smoke.ts), `npm run test:webgpu:uniform-displacement` | Native Dawn/Metal runs with one and two descending rigid boxes. Gates raw conservative mass loss below `0.1%`, Sec. 3.6 unplaceable mass at zero, and transient clamped represented-volume loss below `2%`. On 13 August 2026 the one-body result was `0.00065%` final mass loss / `1.453%` maximum represented loss; two bodies measured `0.00072%` / `1.790%`. The pre-fix one-body reproduction lost `25.8%` raw mass. |

### 12.2 Focused test run performed for this audit

Command:

```text
node --import tsx --test \
  tests/uniform-reference-method.test.ts \
  tests/uniform-gamma-snapshot-diffusion.test.ts \
  tests/uniform-paper-sharpening-step.test.ts \
  tests/uniform-paper-velocity-advection.test.ts \
  tests/uniform-pressure-boundary-transfer.test.ts \
  tests/uniform-pressure-coefficient-bake.test.ts \
  tests/uniform-fim-convergence-diagnostics.test.ts \
  tests/mass-conserving-paper-cases.test.ts
```

Current source-focused result after D20: **39 tests; 38 passed, 0 failed, 1 WebGPU test skipped** across the uniform and pipeline-graph subset. The separate Dawn displacement command passes both one- and two-body cases.

### 12.3 Missing validation relative to the reference

No focused test located in this audit directly proves all of the following:

- random donor coefficients sum to one after fixed-point beta and deposit rounding;
- gamma diffusion cannot cross a solid separator;
- closed/open/periodic boundary mass ledgers;
- sharpening mass error and locality over random fields;
- solid excess recovery when all targets are blocked;
- `rho/V` behavior across tiny V and mixed continuation neighborhoods;
- post-projection divergence `L2/Linf` and complementarity product residual;
- hydrostatic rotated-wall convergence for the uniform method;
- mesh watertightness, ambiguity handling, and enclosed volume;
- long-run double-precision mass comparison;
- a deterministic CPU sparse-matrix oracle for the exact implemented density operator.

---

## 13. Reverse map: live implementation behavior not in the CM12 core

This section prevents the audit from missing code because it was absent from the reference inventory.

| Extension | Live code | Interaction/gotcha |
|---|---|---|
| Authored inflow mass and velocity | [`lib/inflow-boundary.ts`](../../lib/inflow-boundary.ts), inserted in [`gatherConservativeDensity`](../../lib/webgpu-uniform-reference.wgsl.ts#L550) and force/projection paths | Host expected volume integrates analytic flow, while shader insertion is capacity-limited. Difference becomes reported “drift” even if conservative transport is exact. |
| Open-top pressure/velocity boundary | face geometry and projection branches at [`webgpu-uniform-reference.wgsl.ts:294`](../../lib/webgpu-uniform-reference.wgsl.ts#L294) and [`webgpu-uniform-reference.wgsl.ts:786`](../../lib/webgpu-uniform-reference.wgsl.ts#L786) | Density remains closed, so boundary semantics differ by field. |
| Terrain height field | [`webgpu-uniform-reference.wgsl.ts:78`](../../lib/webgpu-uniform-reference.wgsl.ts#L78) | Terrain SDF is vertical height difference; steep/non-heightfield solids require rigid primitives instead. |
| Analytic rigid primitives and quadrature | [`webgpu-uniform-reference.wgsl.ts:201`](../../lib/webgpu-uniform-reference.wgsl.ts#L201) | Shapes are sphere, box, capsule, and cylinder; union fractions are approximate. |
| Explicit viscosity | [`webgpu-uniform-reference.wgsl.ts:596`](../../lib/webgpu-uniform-reference.wgsl.ts#L596) | Adds a timestep-sensitive diffusion scheme not documented in CM12. |
| CSF surface tension | [`webgpu-uniform-reference.wgsl.ts:614`](../../lib/webgpu-uniform-reference.wgsl.ts#L614) | CM12 instead notes that sharpening distance D can look like surface tension. This is a distinct physical force. |
| Two-way rigid reaction/integration | [`coupleRigid`](../../lib/webgpu-uniform-reference.wgsl.ts#L820) and host dispatch at [`webgpu-uniform-reference.ts:947`](../../lib/webgpu-uniform-reference.ts#L947) | CM12's one-way coupling assumption no longer holds in body scenes. |
| Released-domain-face density trace escape | [`backwardTraceExitsReleasedFace`](../../lib/webgpu-uniform-reference.wgsl.ts#L429) | Custom coupling between CM11a separation and density transport; the zero exterior sample now leaves the donor's full forward deficit available for conservative return. |
| Gamma operating-envelope clamp/reset | [`webgpu-uniform-reference.wgsl.ts:483`](../../lib/webgpu-uniform-reference.wgsl.ts#L483) and [`webgpu-uniform-reference.wgsl.ts:553`](../../lib/webgpu-uniform-reference.wgsl.ts#L553) | Stabilization is hard-coded rather than a declared method profile. |
| Fixed-point scatter arithmetic | [`TRANSPORT_FIXED`](../../lib/webgpu-uniform-reference.wgsl.ts#L395) | Makes atomic accumulation available without float atomics, but changes the conservation error model. |
| Fixed paper-step accumulation | [`uniformPaperAdvanceReady`](../../lib/uniform-paper.ts#L5) and [`advanceTo`](../../lib/webgpu-uniform-reference.ts#L734) | Paper mode refuses fractional advances; callers must repeatedly invoke to catch up to a far-ahead target. Scene mode is a different calibration. |
| Fixed pressure schedule and cap telemetry | [`webgpu-uniform-pressure-multigrid.ts:5`](../../lib/webgpu-uniform-pressure-multigrid.ts#L5) | No adaptive retry or rejection. |
| Virtual-boundary optical mesh closure | [`latticeValue`](../../lib/webgpu-water-pipeline.ts#L420) | Rendering geometry at tank contacts is a product choice, not CM12 simulation physics. |
| Dead/dormant uniform shader entry points | `buildHeight` and `relaxSolidPhi` in the main shader | They are not entries in `PIPELINES` and are not dispatched by `advanceTo`; they must not be cited as live uniform algorithms. |

---

## 14. Explicit dependency-exclusion audit

The reference deliberately lists algorithms from dependency papers that should not silently enter the CM12 profile. The uniform code was checked for each group:

| Reference Section 21 exclusion | Uniform-path finding |
|---|---|
| CM11b tall-cell compression/remeshing, level-set reinitialization, tall-cell Laplacian/multigrid, particle thickening, and spray/foam | Not present. The uniform path is dense-grid throughout. |
| CM11b rigid-body two-way approximation | Not imported literally, but the uniform path **does** add its own two-way rigid reaction/integration. This violates the reference's one-way CM12 profile boundary and is catalogued as D13. |
| JRW07 travel-time/geoscience applications | Not present; only the Eikonal/FIM machinery is used. |
| LAF11 conservative momentum, vorticity-confinement correction, and energy correction | Not present. Cumulative gamma and gamma diffusion are the only LAF11-derived live mechanisms identified. |
| LGF11 compressible-flow extensions and quadratic interpolation | Not present; density and velocity sampling are trilinear. |
| MMTD07 mean-curvature flow, general foliation processing, WENO-5 density fluxes, and narrow-band reinjection | Not present as those algorithms. The live CSF force computes curvature, but it is a separate surface-tension extension rather than MMTD07 mean-curvature-flow sharpening. |
| BBB07 PATH/QP solve | Not present; the code uses projected multigrid. BBB07-style two-way coupling is not used literally, but custom two-way coupling is present as noted above. |
| ENGF03 particle-level-set correction and surface-tension tests | Particle-level-set correction is absent. Surface tension exists as a force, but the reference's cited test suite was not found. |
| CM12 PLS comparison, anti-diffusion future work, and connected-component global sharpening correction | Not present. Sharpening performs bounded local return only. |
| Full LGF11 and LAF11 baseline transport algorithms used for reference comparison | Not present as selectable CPU/GPU baselines. Their component ideas feed the streamlined CM12 path, but the five-scatter reference methods requested by the validation section are absent. |

---

## 15. Parameter map

| Reference parameter | Published value | Live value/source | Difference |
|---|---:|---|---|
| simulation `dt` | `1/30 s` examples | [`UNIFORM_PAPER_DT_S`](../../lib/uniform-paper.ts#L2) | Exact in paper mode; scene mode can differ. |
| `dx` | `0.05 m` examples | scene lattice spacing | Figure 9 harness exact; general scenes vary and may be anisotropic. |
| gravity | `10 m/s²` | scene-authored | Figure 9 harness exact. |
| sharpening `DeltaT` | `3dt` | hard-coded in shader | Exact. |
| `tau` | `0.4` | hard-coded | Exact. |
| density epsilon | `1e-5` | hard-coded | Exact. |
| return distance `D` | `1.1–3.1`, usually `2.1` | hard-coded `2.1` | Typical value only; not configurable or serialized. |
| solid distance `S` | `1` cell | one `dx=min(h)` displacement | Exact on cubic grids; minimum-cell physical distance on anisotropic grids. |
| excess `lambda` | `0.5` | omitted from executable formula | Different. |
| excess cap `eta` | `1` before division by `dx` | cap `1` before division by `dt` | Different units/rate. |
| gamma diffusion iterations | `1–7` | `1` default, `1–7` selectable | The least-diffusive published schedule is the reference; larger counts deliberately add diffusion. |
| accurate extension band | `2` cells | `2*max(h)` physical distance | Exact only on cubic grids. |
| bubble levels `C` | `2` | `2` | Exact. |
| constraint levels `S` | `3` | `3` transfer boundaries | Exact host constant; see hierarchy indexing note. |
| ghost theta floor | reconstructed choice | `0.05` | Declared implementation choice, no clamp counter. |
| blur sigma | `2dx` example | two cells per axis | Exact in cell units; anisotropic physical sigma differs by axis. |
| blur radius | unspecified | six cells (`3 sigma`) | Declared implementation choice. |
| detail theta | `0.01` example | `0.01` | Exact. |
| beta/deposit precision | unspecified | `2^20` fixed-point | Implementation-only. |
| sharpening step | unspecified | `0.5` cell, max five steps | Implementation-only. |
| pressure schedule | CM11a performance case | 3 Full + 4 V, 4 pre/post | Matches cited CM11a case, not specified by CM12. |

---

## 16. Final completeness conclusion

The uniform implementation has a real, end-to-end CM12-derived execution path: MAC storage, two-zone velocity extension, matrix-free conservative density transport, persistent gamma and diffusion, local sharpening, cut-solid density handling, effective-fill pressure classification, ghost-fluid/variational/separating-wall projection, projected multigrid, optional Sec. 3.8 reconstruction, and marching-cubes rendering are all present.

It is **not a formula-identical implementation of the algorithm reference**. The most consequential differences are:

1. the `dx`/lambda excess-divergence formula is replaced by a stronger `dt`-based law;
2. cumulative gamma is clamped and reset, and the conditioned matrix-free gather publishes its normalized operator row sum rather than multiplying by donor gamma again;
3. gamma diffusion uses an ordered parity variant and ignores solid separation;
4. unplaceable solid excess is removed rather than recovered;
5. characteristic and boundary fallbacks do not cover every conservation/collision case, though positive-wall separating motion is now retained consistently for sub-isovalue liquid;
6. velocity forces and rigid coupling extend beyond CM12;
7. visible mesh volume is not implemented, and mass telemetry is fixed-point/clamped rather than a high-precision ledger;
8. the focused validation suite is currently not green and is far less exhaustive than the reference's acceptance gate.

Every reference algorithm has been mapped or explicitly marked absent, and every live uniform-path extension found in the audited files has been reverse-mapped. The document should be updated whenever any of the linked shader formulas, dispatch order, hard-coded thresholds, or renderer contour rules change.
