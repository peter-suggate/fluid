# Uniform Geometric → Rust 2D advance-lab

Status: first UI integration complete, 2026-09-20. Bare `/advance-lab` now
selects Uniform Geometric running in Rust/Wasm. This does not change the 3D
studio default or replace Rust `World3d`.

**Scope update:** the user deferred solve-window scheduling and full stage-option
support in favor of proper UI integration with defaults. The first release uses
the shared 3D defaults with `activeRegion: "off"`, shared stage descriptions and
read-only defaults, completed-frame field views, playback/reset and supported
static scenes. Live sources, rigid coupling and stage captures remain follow-up
work. The broader acceptance requirements below describe the original full
migration target; they are not blockers for this explicitly reduced UI scope.

See [implementation status and evidence](../research/uniform-geometric-2d-2026-09-20/implementation-status.md).
No unit tests were added; verification uses whole-scene runners, browser checks
and existing regression checks.

## Decision and behavioral contract

Build a dense `UniformGeometricWorld2d` execution backend for the **same Uniform Geometric method**. The purpose of 2D is to develop, inspect, and validate improvements that can be applied to 3D. A separate Rust execution backend must not become a separate product definition or a permanently diverging algorithm fork. Share options, defaults, validation, UI controls, stage definitions, scenario descriptions, and comparison machinery wherever possible. Keep adaptive topology out of the uniform numerical path.

“Behaviorally exact” means the same algorithm after explicit dimensional reduction: the same field authorities, sampling conventions, branch predicates, time integration, normalization, constraints, iteration schedules, and enabled options. It does not mean that an arbitrary 3D scene and its 2D slice have identical motion. Nor should GPU/Wasm floating-point bit identity be claimed without evidence. Discrete classifications should match exactly away from recorded threshold ambiguities; numeric comparisons need fixed, stage-specific absolute/relative/ULP bounds established before acceptance runs. A looser physical test cannot excuse a stage-parity failure.

The initial inspection reference was HEAD `8750e8b5aa1dfd758c2517d0dd77f574f9cb638e` **plus the working-tree changes**, especially the uniform shader files carrying contact-release fixes. The source is under active development; re-read it at implementation time. First remove the unused liquid-capacity-balancing feature as authorized below, then freeze a reference manifest with commit, patch/content hashes, resolved parameters, scenes, timestep, and backend. Do not discard concurrent changes or silently choose pre-fix HEAD as the oracle. Refresh the manifest explicitly if concurrent work changes the source.

Known reference limitation: the boundary review records conserved V but roughly 6–7% mini32 represented-phi-volume drift after the fixes. Record both quantities and reproduce the agreed reference behavior; improving phi/V agreement is a separate algorithm change. The same review records unrelated failing sparse gates. Its historical results are context, not a substitute for fresh validation.

## What exists and what must change

| Concern | Current implementation | Migration |
| --- | --- | --- |
| 3D source | `lib/methods/uniform/uniform-volume-method.ts`, `webgpu-uniform-reference.ts`, `uniform-volume.wgsl.ts` | Treat executable host scheduling and shaders as authoritative; prose alone is insufficient. |
| Velocity and pressure | Uniform reference shaders, `webgpu-uniform-velocity-extrapolation*`, `webgpu-uniform-pressure-multigrid*` | Port their 2D reductions, including coarse extension and constrained pressure; do not substitute the adaptive PCG path. |
| Rust 2D engine | `rust/crates/fluid-core/src/world.rs`, `levelset_volume.rs` and adaptive graph/topology machinery | Add a separate uniform engine with contiguous fields and fixed topology. Existing transport is a comparator, not the reference algorithm. |
| Wasm dispatch | `rust/crates/fluid-wasm/src/lib.rs`: `OwnedWorld::Slice` / `Volume` | Add explicit 2D method dispatch and uniform load options, advance, commands, receipts, and snapshots. |
| Client/view | `lib/physics-wasm/advance-controller.ts`, `advance-view.ts`, publication/protocol modules | Carry method identity and uniform diagnostics without presenting adaptive-only data as real work. |
| Lab | `advance-lab/*`, `app/advance-lab/page.tsx`, adaptive `features/advance-slice/*` | Make stages, controls, lenses, copy, and persistence method-aware; select uniform by default. |

Before this migration the lab defaulted to the `level-set-volume` transport experiment within the adaptive world. Its shared stage declarations say “Solve the motion → Transport → Adapt and publish.” Uniform requires a different loop; replacing the transport option alone would retain the wrong solver.

## 0. Establish shared ownership before porting

### One method definition, two execution backends

Extract a backend-neutral Uniform Geometric contract under `lib/methods/uniform/`, used directly by both the 3D method and advance-lab. It must not import the WebGPU solver just to render controls or resolve defaults. Use the existing method-parameter and feature-control infrastructure; do not create a parallel lab option framework.

- **One parameter schema:** keys, types, enum values, defaults, ranges, units, help text, runtime-versus-rebuild semantics, and applicability. Extract the definitions now embedded in `uniform-volume-method.ts` and the inherited uniform parameters into composable shared modules. Preserve the density-based uniform method's separate overrides.
- **One resolver:** both UIs, solver factories, harnesses, reset paths, and URL decoding consume the same fully resolved values. Remove duplicated fallback literals from constructor/runtime-update and lab paths. Quality and scene overrides must be shared and have the same precedence.
- **One cross-language contract:** generate Rust option types/defaults/validation data and numerical constants from the shared serializable schema; check generated output for staleness. Native Rust tests and Wasm must not maintain handwritten copies of defaults. TypeScript sends the resolved configuration, and Rust validates it without silently replacing values. Keep presentation-only metadata on the TypeScript side.
- **One UI control implementation:** reuse the same parameter controls and stage-control declarations in studio and lab through their host adapters. Dimension-specific wording such as area/volume and 4×4/4×4×4 is derived from dimension metadata. Hide controls only for a declared capability difference, never to mask an unfinished port at final cutover.
- **One stage contract:** share stage IDs, order, option gates, field authority, dependency descriptions, and diagnostic vocabulary. Generate a simple stage schedule description for Rust and consume it from the GPU host; compare actual execution traces with it. GPU dispatch counts and CPU loop timings remain backend-specific measurements.
- **Shared pure logic:** extract timestep and budget policies, parameter normalization, hierarchy/work-region planning, and scene configuration out of WebGPU ownership where practical. Execute host planning in shared TypeScript when appropriate; use generated tables or differential checks for policy that must also execute natively in Rust. Do not add per-cell JS/Wasm calls.

Rust CPU kernels and WGSL GPU kernels cannot directly share ordinary executable source. Keep backend-specific loops and storage adapters small, share constants and stencils through generation where straightforward, and enforce the remaining algorithm correspondence with stage tests. Avoid introducing a general kernel language/compiler as a prerequisite. Any unshared numerical rule must have a named source counterpart and a parity fixture; “same defaults” alone is insufficient.

### Delete liquid capacity balancing rather than porting it

User-authorized cleanup: delete Uniform Geometric's unused optional **Liquid capacity balancing** feature from 3D before extracting the common contract. Remove its three UI parameters, option fields, runtime updates, shader entry points, dispatches, dedicated buffers/scratch metadata, pipeline controls, diagnostics specific to that feature, and obsolete benchmark commands/tools. Audit references before deleting storage: the donor sums and scratch used by conservative transport must survive. Retain historical benchmark reports as historical evidence, not runnable feature documentation.

This does **not** remove the three transport row/donor normalization rounds, zero-donor fallback, conservative gather, or volume sharpening. Those remain part of the method. Replace the pipeline's optional balancing stage with an accurately named conservative-gather stage and update timings accordingly.

Some existing tests/tools explicitly enable balancing despite it being unused in normal operation. Retire tests solely about the deleted feature, and migrate their general conservation, finite-field, transport and sharpening coverage to the retained method. Compare the cleaned implementation against pre-cleanup `liquidCapacityBalancing=off` fixtures to prove cleanup preserves supported behavior. Legacy saved balancing keys are stripped through an explicit config migration; they never reactivate the feature. Do not remove analogous adaptive code unless it is proven unused by that separate method.

Exit: 3D runs from the extracted common schema with unchanged retained behavior, obsolete balancing references are gone from live code, and a shared options/defaults contract is ready for the Rust backend.

## 1. Freeze the reference and build the parity harness

Deliver a source-to-port specification and checked-in fixture manifest before implementing the production Rust solver.

- Extract **resolved** defaults through the same parameter-resolution path used by the 3D method. Some constructor fallbacks differ from parameter declarations, so calling the solver with `{}` is not a valid default oracle.
- Include initialization: scene lattice, origin, units, liquid/solid sampling, vertex phi, volume/open capacity, boundary faces, sources, initial velocity, initial pressure, and first publication.
- Document every 3D→2D change: eight corners to four, trilinear to bilinear sampling, face/neighbor stencils, plane-box to line-square geometry, 4×4×4 work tiles to 4×4, volume to area per unit depth, and three-axis hierarchy planning to two-axis planning. Preserve dimensionless thresholds; explicitly derive physical-unit conversions and rigid/source dimensional conventions.
- Capture stage inputs and outputs, not only final screenshots: phi, V, capacities, MAC velocities, contact-release bits, transport weights and donor sums, pressure coefficients/RHS/iterates/residuals, work masks, and final surface. Include schedule state such as lagged pressure-cycle demand.
- Use z-invariant extruded 3D fixtures with zero z velocity where dimensional equivalence is valid. Verify that z boundaries, stencils, and hierarchy schedules do not contaminate the comparison; a one-cell-thick box is not automatically a 2D oracle.
- For stages whose dimension-dependent hierarchy or geometry prevents direct extrusion parity, make a small explicitly reduced reference harness based on the frozen WGSL/host code, independent of the Rust implementation. Check its reductions against analytic fixtures and the applicable 3D invariants. Record exactly which tests use which oracle.
- Preserve f32 field arithmetic and the shader's accumulation/quantization rules, fallback order, and ping-pong semantics. Audit integer atomic donor accumulation and reductions before replacing them with ordinary Rust sums. Use f64 only for diagnostics unless the source requires it.

Exit: reviewed dimensional specification, reproducible fixtures, and fixed comparison rules. No undefined “looks equivalent” acceptance.

## 2. Implement the uniform Rust numerical core

Suggested location: `rust/crates/fluid-core/src/uniform_geometric/`, with modules for state/options, geometry, sampling/extension, transport, pressure, boundaries, world, and publication. Keep the public engine independent of adaptive `Graph` construction, remapping, and lifecycle transactions. A fixed presentation graph may be derived once if required by existing views; it must not drive the numerical hot path.

Represent cell V/open capacity/pressure, `(nx+1)×(ny+1)` authoritative vertex phi, and staggered u/v fields in reusable contiguous buffers. Store boundary-face values and contact-release state explicitly. Preallocate scratch and cache pressure/extension hierarchies. Give each stage a named observer boundary for tests and lab diagnostics without copying every field in normal playback.

Port in the production order:

1. Prepare the step, solid/source state, optional solve window, and fine/shell/transport work maps.
2. Extend the start-of-step velocity, including the front sweeps and coarse hierarchy sampler.
3. Apply enabled phi-feedback preparation; RK2-advect vertex phi with source and wall treatment; redistance or perform the exact disabled-path copy.
4. Build transport edges; accumulate donors; apply zero-donor fallback; execute **three** row/donor normalization rounds.
5. Gather conservative V and publish the matching capacity scratch. There is no optional liquid-capacity-balancing pass.
6. Classify sharpening work and execute the source's **eight** conservative sharpening rounds when enabled, preserving buffer ownership. Phi remains the surface authority.
7. Advect velocity and apply forces once. Default is semi-Lagrangian; implement the selected MacCormack alternative before advertising it.
8. Build and solve the CM11a pressure hierarchy, including bound constraints, residual gates, and cycle-budget policy; project velocity and publish contact-release state.
9. Execute supported rigid coupling, then publish phi surface and diagnostics. Preserve which release state is consumed by the next transport step.

Do not pull adaptive pre-transport projection, topology repair, closed-air logic, transport microsteps, or PLIC surface reconstruction into this path merely because they already exist. Helpers may be reused only after their semantics pass parity tests.

Defaults currently worth locking explicitly: semi-Lagrangian velocity; two front sweeps; redistance on; sharpening and work maps as resolved by the method; dust threshold `1e-6`; two-level velocity on; fine reach 2 and shell reach 1; tiled extension/advection/transport; transport margin 1; solve window on with a window-local pressure lattice; volume pressure rows, compaction, phi seeding, and phi agreement off. Inherited pressure controls currently specify residual tolerance 10 in the source's units, 3 full cycles, 4 V cycles, 6 pre/post sweeps, lagged budgeting and headroom 1. These are not interchangeable with the lab's PCG iteration/tolerance controls. This paragraph records inspection findings, not a second defaults authority: production defaults come only from the shared schema.

First establish an explicit all-fine/dense control profile, then port the default two-level sampler and work scheduling. The dense control is an intermediate verification target, not the final default: coarse velocity sampling changes behavior. During implementation, unsupported options must fail clearly. Final acceptance requires the retained dimension-applicable 3D numerical options in 2D, with the same defaults and semantics, so experiments can transfer between them. Backend-only instrumentation may differ; numerical options must not silently disappear.

Time stepping must follow the resolved 3D policy, including full `1/30 s` steps when paper stepping is enabled. Keep render cadence separate from simulation steps. A lab timestep override must be explicit, persisted, and applied identically to the oracle; no hidden fractional steps or extra substeps.

Exit: initialization, individual stages, one-step, and multi-step parity for both the dense control and the resolved production profile.

## 3. Treat boundaries and live changes as part of parity

Port the current contact fix, not the earlier gravity-direction heuristic. Keep mass-transport aperture separate from pressure dual-cell support. Preserve one-layer solid phi continuation, nonnegative solid pressure, outer-halo predicted-velocity RHS, atmospheric open boundaries, solved release conditions, and the `1e-4`-cell separating-travel threshold. Reduce the six-face release representation to the four relevant 2D faces without losing low-domain-face state.

Phi needs both incoming air on released walls and contact continuation that cannot re-wet a departing surface. Extension must retain separating velocity even where transport aperture is closed. Test the complete interaction, not pressure alone.

Use the existing authored scene documents. Audit the current Rust scene reduction against uniform initialization instead of assuming they agree. Cover static terrain, floor/ceiling/side/open/symmetry boundaries, sources and zero-rate sources, live liquid insertion, reset, and supported moving bodies. Derive circles/rigid mass and source rates per unit depth explicitly. Any existing unsupported 3D scene feature must remain visible as a capability limitation; expanding rigid/scene support is not implicit in this port.

Injection must use the reference's phase and one-shot consumption semantics, volume accounting, and work-map invalidation. Do not reuse an adaptive command if it also remeshes or immediately changes fields that the reference changes on the next advance.

Exit: boundary/contact and command-sequence fixtures pass with finite fields, no solid penetration, and separately accounted source, outflow, and dust contributions to V.

## 4. Wire Wasm and the lab without hiding method differences

- Add a method selector independent of `transportExperiment`, using the existing registered identity `uniform-volume` for Uniform Geometric. Keep adaptive experiments under the adaptive method.
- Route uniform load/advance/reset/commands/snapshot through `OwnedWorld`. Preserve worker ordering, run epochs, publication leases, and restart semantics. Keep 3D dispatch unchanged.
- Add method/capability metadata and actual uniform stage receipts. Version publication schemas where the wire contract changes; test readers against both methods. Do not manufacture adaptive flux-limiter, rung, or remap receipts to satisfy existing readers.
- Reuse canvas, camera, playback, tracers, source controls, and general field overlays. Publish vertex phi directly for contours; graph/PLIC data must not reposition it. Provide real V/phi disagreement, pressure convergence, extension, work-map and contact-release views.
- Extract a method-neutral stage/view interface from the currently adaptive-owned advance-slice feature. Render uniform stages from the shared 3D/2D Uniform Geometric contract, with dimension-aware descriptions and backend cost readouts. Do not author a second uniform pipeline for the lab. Update page title and explanatory text.
- Hide refinement-region editing and adaptive-SDF controls for uniform; expose uniform resolution and appropriate pressure-cycle/tolerance controls. Keep legacy refinement state isolated so switching back restores it without applying it to uniform.

Exit: scalar and SIMD Wasm fixtures agree within the contract, publications remain immutable across subsequent steps, commands are ordered correctly, and browser stage inspection describes the computation that actually ran. Retain the current single-worker SIMD preference until matched uniform measurements justify changing it; verify threaded parity if that artifact is supported.

## 5. Cut over defaults and persistence

Make bare `/advance-lab`, fresh runs, scene changes, and resets select Uniform Geometric consistently. The lab registry chooses the default method identity; all of that method's parameter defaults come from the shared contract, including URL parsing, store initialization, controller options, and controls. Keep the default scene unless a separate decision changes it.

Persist explicit method identity in shared links, including new uniform links, so later defaults do not reinterpret them. Legacy links with an explicit adaptive transport value should select the matching adaptive method. An old link that omitted its then-default transport is inherently ambiguous: absent method and absent explicit transport now means the new default. Document that migration rather than claiming those links can all be preserved. Ignore incompatible adaptive controls on uniform with a visible explanation when needed; do not silently change method because a refinement region is present.

Retain the adaptive method as an explicit comparison/fallback option through acceptance. Switching method rebuilds the world from the authored scene; transferring a running adaptive state is outside this plan. Old PCG budgets must not become uniform cycle counts by accident.

Exit: URL round trips, explicit legacy links, bare entry, scene switch, reset, and artifact selection all load the expected engine and resolved defaults.

## 6. Acceptance and performance gate

Add a short dedicated uniform-2D regression command, with separate native Rust, Wasm, and source-oracle lanes. Required cases:

- Empty/full domain, initialization and stationary hydrostatic pool.
- Translation and prescribed rotation; stage-level phi, weights, V and velocity parity.
- Mirrored dam breaks, symmetric collapse, long thin-film/far-wall evolution, and pool impact.
- Closed box and embedded ceiling release; both signs of side-wall release at zero gravity; curved-container reduction; open and symmetry boundaries.
- Dust-floor accounting, zero-donor fallback, normalization, sharpening on/off, and the retained phi-feedback controls.
- Default fine/coarse transition, reach limits, odd grid sizes and non-square hierarchy planning; solve-window and pressure-window controls and fallbacks.
- Source start/stop, zero flow, liquid insertion near walls, reset, and supported moving-solid displacement.

Measure pressure residual/divergence, conserved V, phi-implied area, phi/V mismatch, front location, symmetry, release masks, and finite fields. Fail on unexplained numerical differences; do not weaken the oracle to make the port pass. Keep known source defects visible in the baseline manifest.

Add contract tests proving studio/lab default resolution and reset results agree, option serialization round-trips through Rust, generated definitions are current, and every retained dimension-applicable UI option reaches both backends with the declared effect. Instrumented stage traces must match the shared schedule. Add fixtures for runtime option changes as well as startup configurations.

Benchmark existing adaptive Rust and new uniform Rust at the same finest spacing, timestep, scene, simulated duration, and visible diagnostics. Include small lab scenes, large full domains, and mostly-empty/tall domains. Report native and Wasm advance median/p95, stage costs, initialization, memory, publication cost, and end-to-end lab playback separately. Uniform's 3D GPU advantage does not establish a Rust 2D advantage. Establish an explicit acceptable speed/memory budget from these baselines before default cutover; optimization cannot alter accepted behavior.

Run focused Rust tests, existing advance-controller/view and lab persistence/playback tests, Wasm build/load/artifact checks, and TypeScript checks. Use the frozen source's focused uniform boundary Dawn suite when refreshing oracle fixtures. When implementation changes shared presentation, terrain, live editing, or sparse topology, run the repository's canonical `npm run test:dawn:sparse-cm12` gate. Run Dawn serially under the WebGPU lease with no browser session; distinguish reproduced baseline failures from newly introduced ones and never raise ceilings to pass.

## Delivery sequence

1. Delete unused liquid capacity balancing; extract shared options, defaults, UI and stage contracts; verify unchanged retained 3D behavior.
2. Freeze reference manifest, reduction specification, stage capture/comparison harness.
3. Dense uniform Rust state, initialization, transport and geometry, with stage tests.
4. Velocity extension, CM11a pressure, boundary release, and full-step parity.
5. Retained numerical option coverage, default two-level/work-map behavior, commands and Wasm/publication integration.
6. Connect lab to the shared pipeline and controls; add diagnostics and explicit selectable uniform mode.
7. Acceptance/performance evidence, then default and URL cutover in the same reviewed release.

## Continuing workflow: 2D improvements become 3D improvements

The migration establishes a continuing development path, not a one-time port. A numerical experiment starts with a small reproducible 2D fixture and a shared option/variant definition, uses the lab's stage diagnostics to explain the effect, and carries the same parameter bundle into an applicable 3D scene. Store experiment artifacts with contract/reference versions, dimension, scene, timestep, resolved options and measurements so results can be reproduced in either backend.

An experimental implementation may temporarily exist only in Rust, but must be explicitly marked as such and must not become a shared default before the WGSL counterpart and 3D validation exist. Promote an improvement by updating both implementations, adding its cross-dimensional regression fixtures, then changing the default once in the shared schema. A 2D speedup or visual improvement alone is not evidence of a 3D improvement: validate GPU performance and genuinely 3D cases before promotion. Improvements to shared UI, configuration and host policy should reach both dimensions immediately through shared code.

Default cutover is complete only when the actual default profile passes parity, dimension-applicable options and defaults are shared, the lab's stages and controls describe the same method, live/reset/persistence flows pass, and performance/memory results are recorded. Planning alone changes no runtime behavior.
