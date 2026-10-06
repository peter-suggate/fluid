# Uniform Geometric: dynamic 4h-first implementation handoff

3 October 2026. **Implementation plan, not an implemented solver.** This is the controlling plan after Peter's clarification: “no fast paths. a fully dynamic 4h-first + opt-in dynamic detail approach.” It supersedes the dense-endpoint recommendations in the earlier investigation.

## Outcome and fixed decisions

Replace the maintained `uniform-volume` implementation with a genuinely compact 4h-first simulation and optional, live h detail. Ship it through the normal production renderer and controls as the default Uniform Geometric implementation, so Peter can test ordinary scenes without a probe, environment variable or alternate method selector.

There is **one dynamic architecture from zero h detail to full h detail**. Do not add an occupancy-triggered dense mode, legacy fallback, separate all-fine solver, alternate high-occupancy kernels or a hidden h-first backing allocation. Full detail is a request covering the domain; it uses the same allocator, resource interfaces, operators, scheduling and publication as partial detail. Old code may remain temporarily as an offline comparison oracle during development, never as a production execution route.

The base spacing is H = 4h, where h retains the scene's authored finest physical spacing. With no detail requests, the simulation—including transport, free surface, forces, pressure and solid contact—runs at H. A free surface, visible surface, wet solid or high speed must not implicitly restore the current global h-first policy. Mandatory numerical support can expand an **existing** detail request; it must be separately accounted for.

Opting into detail means h transport, momentum/velocity, surface representation and pressure correction, not merely a finer rendering mesh. Activity and user proximity are request generators behind explicit controls. A newly promoted region cannot recover lost fine-scale history; promotion should anticipate approaching motion where enabled.

The existing full-fidelity performance requirement remains: the same dynamic architecture must approach current performance at matched full-detail work. No fast path is available to hide an expensive endpoint. A 5% regression budget is a proposed engineering target, not a relaxed test ceiling or a measured result for the new design. Getting an integrated build into Peter's hands must not wait for a new research programme; completing the default replacement does require the correctness and coverage gates below.

## Evidence sufficient to start

The [broad investigation](uniform-coarse-first-2026-10-03.md) and [addressing experiment](uniform-full-fidelity-addressing-2026-10-03.md) provide the relevant constraints:

- Existing 4h ownership can reduce work substantially, but leaves the dense h allocation floor: a 256³ case still reserves 4.09 GB. Changing selection policy is not the architecture change.
- A generic per-texture-access atlas costs +10.8% for arithmetic placement and +13.7% for directory placement in an identical accepted all-h frame. The experiment allocates every page and has no residency/halo costs, so that implementation is already unsuitable for the endpoint target.
- A native specialization matches production throughput and fields. It was a diagnostic control; **Peter has ruled it out as the new architecture's escape route**.
- Blindly preferring coarse cells failed the resting-pond quality check. Coarse contact and force/pressure balance are implementation dependencies, not tuning to defer until after the default flip.
- Long dynamic trajectories can diverge after shader adaptation even with identity addresses. Compare frozen work for infrastructure cost and owner-aware physical measurements for trajectory quality; do not call a changed trajectory a matched-work speedup.
- The existing forced-all-h mini dam rejects pressure on frame 3. Keep that failure visible. The successful first-frame replay is not a substitute for a stable full-detail trajectory.

**Stop broad exploration here.** Use the existing tools to answer at most the three bounded questions embedded in the milestones: patch-address cost, coarse-contact correctness, and the occupancy/transition curve. Do not add 2h, multirate timestepping, a new transport formulation, sleeping physics or a new composite-pressure algorithm before the first testable default.

## Architecture and invariants

### Compact base and optional detail

Keep the authored domain and physical units unchanged. Allocate base volume, staggered face velocity, vertex phi, solid/open-volume data and base scratch at H dimensions. The base pressure grid must be at the intended physical H spacing; simply passing a quarter-resolution scene to today's constructor also coarsens its root hierarchy and is not an equivalent implementation.

Use a directory over fixed world-space h patches and allocate h field/scratch slots only for admitted detail plus its certified support. Begin with a 32³ h patch allocation unit, retaining 4h tile granularity for ownership/request masks within each allocated patch. This is a starting choice, not an ABI promised forever: one bounded 16³-versus-32³ comparison may change it before integration. Report wasted allocated cells from partly occupied patches separately from active h cells.

A coarse cell owns mass/flux outside detail; its fine children own inside detail. Covered coarse entries contain restricted data for coupling and queries, never a second independent transported copy of the water. Restriction is still needed where covered; full-coverage operation must not advect and sharpen both complete levels. This is the same ownership rule at every occupancy, not an all-fine special case.

No h-sized per-cell ownership map, donor arena, phi volume or renderer expansion may remain hidden in the zero-detail configuration. Directory/ownership metadata may scale with coarse cells or patch count. Static fine solid geometry and independent scene rendering are separate assets; their memory must be reported separately rather than mislabelled as h fluid state.

Give shared same-resolution faces/vertices one canonical writer, keyed by world lattice coordinate rather than slot order. Coarse/fine boundary samples use the accepted ownership/reconstruction rule, and all consumers use that same rule. Duplicate halo copies, if introduced later, are derived data with a generation, never competing authorities. Include outer-domain planes and neighbouring patches promoted/retired in opposite orders in the fixtures.

Start with texture-backed field pools, matching the maintained field formats; do not combine the port with reduced precision. A cell field slot contains P³ cells. Vertex fields may use (P+1)³ samples per slot, with shared boundary samples copied from the canonical writer before consumers run; count that duplication explicitly. This avoids treating a vertex plane as a separate fluid owner. Address metadata carries field stride and pool extent, so adding capacity changes bindings/descriptors rather than recompiling shaders.

WebGPU textures cannot grow in place. Allocate no fine pool until needed, then grow geometrically within a device-derived byte budget; copy live slots into a candidate pool and adopt it transactionally. Plan the old+new pool memory peak before allocating. Retain logical IDs across the move, and release the old resources after all readers finish. Full coverage is simply a filled pool under this same policy. Do not reserve maximum-h textures at initialization in the name of avoiding later allocation.

Suggested new modules under `lib/methods/uniform/`:

| Module | Responsibility |
|---|---|
| `uniform-coarse-first-domain.ts` | Physical lattice, base fields, directory, immutable accepted-generation resource views. |
| `uniform-detail-requests.ts` | Merge authored and transient requests, priority, expiry and requested masks. |
| `uniform-detail-planner.ts` | Numerical support closure, admitted masks, allocation/churn budgets and diagnostics. |
| `uniform-detail-pool.ts` | Stable logical patch IDs, physical slots, generations, capacities and deferred reclamation. |
| `uniform-coarse-first-fields.wgsl.ts` | One coarse/detail sampling ABI and patch-local addressing helpers. |
| `uniform-coarse-first-frame.ts` | Ordered frame transaction using the existing numerical stages as they are ported. |
| `uniform-coarse-first-publication.ts` | Accepted coarse/detail resources for the renderer and scientific views. |

These names are proposed implementation boundaries. Reuse existing resource/acceptance utilities where they fit; do not create a second generic sparse framework or revive the retired Sparse CM12/Losasso methods.

Start the port at these existing dependencies:

| Current code | Change required |
|---|---|
| [Host constructor / mixed initialization](../../lib/methods/uniform/webgpu-uniform-reference.ts) | Remove dense-first field and arena reservation from the new factory path; borrow only numerical helpers whose resource assumptions have been ported. |
| [Transport](../../lib/methods/uniform/uniform-mixed-transport.ts) | Replace `tiles * 64` scratch capacity and fine-coordinate backing assumptions with admitted-owner/patch resources. |
| [Frame transaction](../../lib/methods/uniform/uniform-mixed-frame.ts) | Port every stage's field view and generation dependency while retaining health receipts and publication ordering. |
| [Dynamic classifier](../../lib/methods/uniform/uniform-mixed-dynamic.ts) | Separate optional detail requests from numerical support. Do not obtain the new default by raising the retired coarse-surface-tolerance switch. |
| [Layout and solid promotion](../../lib/methods/uniform/uniform-mixed-layout.ts) | Replace blanket wet-solid h promotion with valid coarse solid operators; retain explicit authored-tier semantics. |
| [Ownership and remap](../../lib/methods/uniform/uniform-mixed-remap.ts) | Keep conservative transfer rules while replacing packed-rank identity with stable patch/generation identity. |
| [Pressure band](../../lib/methods/uniform/uniform-pressure-band.ts) | Allocate/dispatch from admitted fine correction rows and support; eliminate empty fine envelopes without a separate solver path. |

### One execution and sampling model

Dispatch work from accepted coarse/detail work lists, with resolution and physical bounds carried in descriptors. At all occupancies, an h patch uses the same operator family and address rules. Fixed patch/workgroup geometry may be compile-time constants; occupancy must not select an alternate algorithm or representation.

Resolve the executing patch and its fixed-stencil neighbours once per workgroup/stencil build, and reuse those addresses across repeated accesses. Stage local stencil data where profitable. This is a property of the universal patch implementation, not a native-interior bypass. Do not put the tested generic wrapper around every production texture access.

Semi-Lagrangian departures, cubic interpolation and redistance searches can leave a patch. Resolve each sample footprint's participating patches once, then reuse the descriptors across its taps. A 27-neighbour cache is not a certificate for an arbitrarily long departure. The planner must provide actual time-step/stencil support or the query must traverse the directory correctly. Absent fine data means the authoritative coarse reconstruction, not zero and not an arbitrary stale slot.

Avoid permanently duplicating full ghost volumes in the first layout. Workgroup-local staging and explicit cross-patch accesses keep the initial memory model simple. If persistent halos become necessary, add their exact footprint/update cost to every occupancy measurement; do not introduce a different dense path to hide it.

### Dynamic transactions and temporal reuse

Use a request descriptor such as:

```ts
type DetailRequest = {
  id: string;
  bounds_m: { min: Vec3; max: Vec3 };
  targetSpacing: "h";
  source: "region" | "focus" | "activity" | "full";
  priority: number;
  expiresAfterStep?: number;
};
```

Do not persist physical slot IDs in scene documents or URLs. A patch's logical world key stays stable; its slot has a generation. Packed execution-list rank is not identity.

At an accepted frame boundary:

1. Consume the newest request revision and relevant scene/body changes. Compute support closure from the upcoming dt and stencil requirements.
2. Build a candidate directory/list generation; reserve slots before modifying authority. Keep the old accepted generation usable by in-flight simulation and rendering.
3. Promote conservatively from base data; retire by restriction. Transfer extensive liquid volume and compatible momentum/face-flux quantities, reconstruct bounded phi, then project. Do not independently interpolate V and phi and assume conservation follows.
4. Advance the candidate owner set, apply the existing health/pressure acceptance transaction, and publish one matching field/directory generation.
5. Reclaim retired slots only after all readers of the old generation are complete. Two in-flight frames are an existing supported case, not permission to reuse slots after an arbitrary one-frame delay.

Account for transfer mass, momentum/energy change, discarded dust and sources separately. Failed preparation leaves the last accepted generation intact. A missing support page must reject/defer the candidate before publication; it must not silently run a partly updated simulation or invoke another solver.

Use separate promotion/retirement thresholds, minimum residence time and a bounded churn budget. Start with immediate admission of feasible manual/full requests, predictive focus/activity admission, and delayed automatic retirement. Initial automatic-policy constants: retire only below 0.7× the promotion threshold for eight accepted steps; limit automatic membership changes to 5% of patch capacity per step. These are tunable starting values, not validated physics constants. Required support is admitted with its request; partial unsafe closure is never scheduled.

For the first activity source, evaluate a cheap census on the restricted accepted H state: dimensionless strain `dt * ||sym(grad u)||`, curvature `H * abs(kappa)` where the interface is represented, and predicted solid contact within two timesteps. Start with strain threshold 0.1 and curvature threshold 0.5, scaled by `2^(2 - 4 * detailSensitivity)`; impending contact is a high-priority request. Use a max-normalized score, stable world-key tie breaks and the closed-request budget. Add reliable thin-feature evidence when available from initialization/current geometry; do not pretend a coarse field can detect already-erased sheets. This is an initial opt-in selector, not a change to physics or a claim that these thresholds are optimal.

A paused edit runs a topology/remap transaction without transport or force advancement and without changing simulation time. If remapping requires a velocity constraint projection, make that an explicit transfer projection with a valid retained numerical timestep; do not pass dt=0 through pressure coefficients. Published fields and coverage must belong to the same accepted revision before the pause view updates.

Cache static solid geometry, neighbour descriptors and unchanged work lists by accepted generation/dependency revision. A moved body or changed patch invalidates its actual dependency neighbourhood. Pressure warm starting may be reused only with a newly evaluated residual. No sleeping or slower-rate fine evolution in version 1.

### Numerical scope for the first default

Port the maintained conservative geometric transport, force, surface-volume and pressure acceptance behavior. Preserve physical units, timestep controls and existing tolerances. Do not combine this storage rewrite with refluxing/new transport, new pressure tolerances or a new multilevel solver.

Reuse the current global H pressure solve and requested h pressure correction as the initial numerical reference. Build fine pressure capacity from admitted wet/detail/support rows, not dense h capacity or only a visual surface request. Zero detail encodes no fine pressure envelope. Full detail still runs this same coupled pipeline; the covered base supplies restriction/global pressure, not independent base transport.

The current split solve has limited same-step fine-to-global feedback. Record that limitation and validate interface flux consistency against the maintained reference; do not claim a new composite solve. A later composite defect-correction change is a separate numerical project and is not a prerequisite for testing this representation in the UI.

Coarse cells must handle coarse free surfaces and solids themselves. Compute H-cell liquid/open volumes, face apertures, boundary velocity and force/pressure geometry consistently from authoritative solids. Hydrostatic pressure and gravity must cancel under that geometry. Cover fractional waterlines, thin coarse films, resting ponds, sloping terrain, moving bodies and edits. Blanket promotion of all wet cut cells would violate the default's purpose and is not an acceptable substitute.

## Production and UI contract

Keep method id `uniform-volume`; the user still sees **Uniform Geometric**. When integrated, its factory constructs the new dynamic implementation. The previous implementation is not offered as a UI mode and is not selected by scene, occupancy, timing or allocation failure.

Default policy is **4h base with requested detail**: no automatically generated h requests. Existing explicitly authored Fine regions remain opt-ins. Add runtime controls through the existing method-parameter/controller path:

| Proposed control | Default and behavior |
|---|---|
| `detailPolicy`: Requested / Dynamic / Full | **Requested**. Authored Fine regions apply. Dynamic additionally enables selected automatic request sources. Full creates one whole-domain h request through the same planner. |
| `detailNearFocus`: On / Off | **Off** (2026-10-03: no camera-focused detail for now; it holds h around the target with or without liquid). Inactive until Dynamic is selected. Focus is the orbit/interaction target, not the camera eye. |
| `detailActivity`: On / Off | On as a preference, inactive until Dynamic is selected. Uses strain, interface thickness/curvature, acceleration and impending impact; kinetic energy alone is insufficient. |
| `detailBudgetPercent` | Start at 25% for automatic requests, measured after support closure. Manual/Full requests are not silently clipped to this soft budget. Actual allocation limits still apply and requested versus admitted coverage is visible. |
| `detailFocusRadiusPercent` | Start at 20% of the domain's longest physical extent. Show corresponding metres in the UI. |
| `detailSensitivity` | Start at 0.5 on a normalized 0–1 scale. Centralize the mapping to documented estimator thresholds; avoid unrelated hidden scene presets. |

These defaults are proposed product choices for the first test build. They do not change the existing physical acceptance tolerances. Keep detailed estimator/hysteresis settings in one typed policy object initially rather than flooding the UI.

Update `components/UniformCoarseControl.tsx` into the primary **Simulation detail** control in the existing SIM panel. Show accepted coarse/fine coverage, requested/admitted detail, allocated memory, and a pending-change indicator. Expandable diagnostics can show support overhead and request reasons. Do not expose atlas slots, generations, backend names or compilation variants in the ordinary user flow.

The Region tool already has Fine/Coarse held tiers and defaults new Uniform regions to Fine. Reuse it. Change copy from “draw a region to use larger cells” to drawing areas that need finer simulation. Preserve existing region bounds and Fine/Coarse meanings. For overlapping policy: Full wins; explicit Fine wins over explicit Coarse; explicit Coarse suppresses automatic requests; numerical support of accepted fine requests may extend beyond authored bounds and is shown separately. Document and test that precedence instead of relying on list order. Removing the last Fine region in Requested mode returns to the coarse base without resetting time.

Interpret older min/max region records through the existing allowed-tier semantics before creating requests. A lower bound of 1 with an automatic/unbounded ceiling is not by itself an explicit h request. Audit existing tooltips, including the old suggestion of automatic 2h grading; the new contract contains only H and h simulation tiers.

Focus positions are transient per-pane input to the solver worker, coalesced to the latest revision per accepted frame. Add a typed request/input seam alongside `applyRuntimeValues`/`applySceneUniforms`; do not mutate the persisted scene on every camera movement. Scalar controls and authored regions round-trip through the existing scene/URL mechanisms. Compare panes maintain independent focus and detail state.

### Runtime wiring and persistence

- Declare runtime detail controls in `uniform-geometric-parameters.ts`, normalize them in the shared resolver, map them in `uniform-geometric-options.ts`, and include them in `runtimeParamKeys` through the existing parameter mechanism. Resetting parameters must restore the new requested-detail default.
- `lib/core/stores/method-store.ts` resolves defaults → preset → app defaults → user override. Put the architecture/default-policy decision in one documented place; ensure app defaults do not silently undo an explicit saved value.
- Route control changes through `lib/core/simulation/controller.ts`; update the worker/client protocol for transient focus requests. Ensure `structuralMethodValues` in `webgpu-renderer.ts` excludes runtime detail settings, so moving detail or toggling Full does not rebuild/reseed the solver.
- Retire the old `coarsening` control rather than interpreting its old `dynamic` value as permission to enable new automatic detail. Migrate `regions` to Requested; old `dynamic` or absent values also open Requested unless the new `detailPolicy` was explicitly saved. Preserve authored regions. Add a one-time migration notice for imported configurations whose policy semantics change, and serialize only the new keys thereafter.
- Update `lib/core/url-state.ts` and region persistence tests for old/new keys, Fine/Coarse overlap, clear-last-region, reload and compare panes. Update method description/detail text so it no longer promises fine cells outside coarse boxes.
- Keep GPU-only detail controls out of `UNIFORM_GEOMETRIC_NATIVE_PARAMS`; do not accidentally change the separate 2D Rust lab's parameter contract.

### Renderer and publication are part of the first usable slice

Add a typed coarse/detail fluid source at `lib/core/levelset-consumer-abi.ts` and expose it through `GPUSolverInstance` in `method-contract.ts`. It carries base fields, admitted fine fields, directory/ownership, physical lattice and a matching accepted generation. Use the maintained renderer's consumption seams, without routing the new solver through a retired method's sparse-world lifecycle.

Update `webgpu-renderer.ts`, `webgpu-water-pipeline.ts` and `webgpu-grid-overlay.ts` to read that source directly. Adapt surface extraction, normals, V/phi views, pressure/velocity diagnostics and fine-region overlays to the same authority. Resolve coarse/fine boundary samples consistently so contours do not crack, double-draw or pop when a page is retired. Include SVO fluid-coverage consumers that currently assume a dense source.

A full h volume reconstructed every frame for rendering would recreate the cost/memory floor and is disallowed. Coarse texture bindings remain coarse-sized; fine extraction/publication visits admitted/changed patches. Rebuild or resample only the affected contour-support neighbourhood. Scientific slices may use explicit bounded diagnostic readback, never silently add permanent dense fluid fields to ordinary rendering.

## Ordered implementation milestones

Each milestone must leave a runnable commit and a short evidence note. Do not stop after a standalone simulator: the delivery endpoint is the normal app with this implementation selected by default.

| Milestone | Concrete work and exit condition |
|---|---|
| **1. Resource/owner contract** | Introduce the compact base, optional patch pool and generation interfaces. Port field access in one representative heavy stage using the universal patch addressing model. Run the bounded addressing comparison at full occupancy (16³ and 32³ only). Select the layout; do not salvage a failed result with a native bypass. Record remaining cost if above the 5% target. |
| **2. Coarse simulation end to end** | Port initialization, conservative transport, momentum, free surface, forces, root pressure, sources and solid contact to actual H storage. No h allocations/dispatches at zero detail. Pass coarse hydrostatic/contact tests and produce an accepted coarse fluid source visible in the production renderer. At this milestone the app can display the base, but this alone is not the final requested handoff implementation. |
| **3. Stationary requested h detail** | Allocate only one requested h region plus certified support; run h transport/surface/pressure through the same frame transaction. Validate transfers, seam sampling and pressure/flux consistency. Sweep 0%, 12.5%, 25%, 50%, 75%, 100% with the same code path. Include fragmented requests. |
| **4. Fully dynamic lifecycle** | Move/resize/add/remove requests while running and paused, promote/retire conservatively, reuse caches, respect in-flight generations, grow capacity safely and report deferred admission. Add focus/activity request sources and hysteresis. Verify zero→partial→full→zero without restart, NaNs, missing reads or monotonically retained fine allocations. |
| **5. Default production/UI integration** | Complete direct rendering/scientific views, runtime controls, persistence migration and per-pane focus. Switch `uniformVolumeMethod.createSolverAsync` to the new implementation; default Requested mode, no environment flag or alternate method. Precompile the fixed operator family; request changes must not compile shaders. Peter can now test from the ordinary app. |
| **6. Default readiness and cleanup** | Run all existing gates, the new dynamic lifecycle coverage and production browser acceptance below. Record total frame time and memory across maintained scenes. Remove temporary integration flags and production references to the old execution path. Keep offline reference evidence without shipping a fallback. |

If milestone 1 exposes an address-cost problem, optimize the same descriptor/staging implementation. If milestone 2 fails a pond/contact case, repair the coarse operator. If a later scene exceeds the frame budget, profile that same architecture. Do not reopen the rejected dense-path proposal or paper over a failure with relaxed thresholds.

## Bounded validation and performance contract

Keep `npm run check:types`, `npm run test:unit` and **the complete** `npm run test:dawn` green. Dawn is serial and must not overlap the browser. Preserve all existing Uniform assertions and timing ceilings.

Add tests where the new invariants live:

- CPU: stable IDs/generations, region precedence, request expiry/hysteresis, exact support closure fixtures, allocation accounting, parameter migration and runtime-versus-structural keys.
- GPU: zero-detail allocation/dispatch absence; coarse pond and manufactured pressure/solid balance; conservative promotion/retirement; cross-page sampling/departures; missing-page rejection; failed-candidate publication isolation; in-flight reuse; all-h through the same dynamic pipeline; live solids, sources and rigid bodies.
- End-to-end UI: fresh default load, draw/move/delete Fine regions, enable Dynamic, move focus, switch Full and back, pause/edit/resume, save/reload/share, reset controls, and independent compare panes. Verify simulation time does not reset, accepted detail changes visibly and memory is reclaimed after readers finish.

For every meaningful benchmark retain the solver/source fingerprint, exact configuration, physical time, requested/admitted coverage, interface/support counts, memory, mass/centroid/phi-volume consistency, pressure residual and health receipt. Compare source/mass conservation with explicit source/dust accounting. Numerical tolerances come from the existing physical fixtures or an independently justified manufactured problem, not from fitting the candidate's error.

Separate three comparisons:

1. **Infrastructure:** freeze matched accepted work, then compare current and new storage. The supplied [frozen replay tool](../../tools/probe-uniform-frozen-frame-dawn.ts) is the starting harness; extend it for the new source and resource lifecycle. Its first-frame timing excludes CPU planning/restoration/rendering and is not an FPS claim.
2. **Dynamic simulation:** matched initial scene, dt, numerical settings and physical duration; record a stable full-detail trajectory and the occupancy sweep. If shader changes affect the trajectory, report owner-aware physical differences and do not call wall-time deltas pure overhead. Fix or replace the failing all-h mini reference with a supported, documented comparison; never loosen pressure acceptance.
3. **Actual app:** warmed production build on target hardware, physics+publication+render+UI. Report mean/p90, transition worst case, dropped presentation frames, allocation high-water mark and initialization/compilation separately. A 60 Hz target means a complete frame budget of 16.67 ms; a 3–5 ms coarse simulation target is an allocation of that budget, not an established result.

Use the maintained mini dam, Figure 9, 256³ falling drop, resting pond, solid-contact/live-edit and rigid-body cases, plus a manufactured interface/pressure problem. Run longer only when a failure or unresolved drift requires it. The report must identify scene misses; “default and testable” is not a claim of realtime across every scene.

## Peter's test-build checklist and handoff completion

The implementation is ready for Peter's direct test when opening the normal app selects Uniform Geometric's new architecture, no Fine regions means an actual compact coarse simulation, and he can create or automatically request moving h detail live. Full must fill the same detail system, and turning it off must release detail after safe retirement. The fluid, diagnostic overlays and accepted coverage must agree.

Provide the exact build/commit, launch instructions using the existing app scripts, representative scene links with the new serialized controls, the completed validation commands, a concise performance/quality table, and known limitations. Keep rollback at the **commit/release** level; no runtime alternative solver or fast path. Do not describe this planning change as that implementation being available now.

Current repository evidence is in [full-fidelity-evidence.json](uniform-coarse-first-2026-10-03/full-fidelity-evidence.json). This task added research probes, CPU/GPU fixtures and the handoff documents; the new default engine/UI remains the work described above.


Handoff baseline validation: `npm run check:types` passed; `npm run test:unit` passed (852 pass, 52 skip); `npm run test:dawn` passed all 48 files (88 tests). These results cover the current repository and research additions. Each implementation milestone must validate its own changes; none of the planned engine/UI work is claimed complete by these results.
