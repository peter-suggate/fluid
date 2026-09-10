# Feature ownership cleanup on main

This follow-up completes shared persistence/control/lifecycle infrastructure and applies it to topology freeze, surface display, pressure inspection and the SVO settings already exposed by feature UI.

## Ownership

| Concern | Owner | Host responsibility |
| --- | --- | --- |
| Numeric control metadata and normalization | `lib/framework/controls.ts` | Adapt domain commands to shared controls |
| Configuration change impact | `lib/framework/lifecycle.ts` | Validate before mutation; execute the resulting live/rebuild/reset action |
| URL value codecs, defaults and key ownership | `lib/framework/persistence.ts` plus each feature's codec | Compose codecs and preserve unrelated URL keys |
| Topology freeze defaults, persistence and UI | `lib/features/topology-freeze` | Runtime store holds pane-local state; simulation reset applies feature reset policy |
| Surface display choices, state type, persistence and UI | `lib/features/surface-display` | Renderer consumes selected mode; UI host projects feature placements |
| Pressure journal ABI, visualization and filmstrip | `lib/features/pressure-inspection` | Diagnostics and renderer transport the feature publication |
| Sparse CM12 capture settings, WGSL, decoding, tests and probe | `lib/methods/adaptive-mass/features/pressure-inspection` | Resident solver composes capture kernels and schedules their dispatches |
| SVO lighting, primary visibility, construction, diagnostics and radiance URL settings | Respective `lib/svo/features/*/persistence.ts` | `lib/svo/pipeline/persistence.ts` combines their state, including nested tuning |

The toolstrip no longer contains pressure-film behavior or topology/surface-specific control definitions. Slots such as `fluid.inspection`, `sim.inspection`, `scene.surface` and `sim.topology` resolve through the same application view registry. Advertising an additional placement does not create a new panel; any host can render that slot. Capabilities keep pressure inspection and topology freezing limited to supporting methods.

Feature defaults initialize both UI and runtime stores. URL hydration applies the composed state. Reset policy clears topology freeze while preserving surface-display preferences. Effective method changes determine lifecycle impact before announcing GPU work. Removing an explicit override equal to its default still removes the override, without resetting the timeline.

Static publication ports continue to validate representation and provider compatibility. The unused `AcceptedPublication`/`readPublication` runtime abstraction was removed: existing domain publication acceptance, GPU completion and ownership mechanisms remain authoritative.

## Cutover and extension rules

Old pressure-journal module paths were removed, and all consumers, test discovery and explicit GPU scripts now use the feature paths. There are no forwarding modules. Capture WGSL extraction expands to exactly the original resident source; numerical thresholds, dispatch order and GPU acceptance rules were not changed by this cleanup.

To add a concern, define its state/codec, controls and placements with its implementation. Install its codec in the appropriate application or pipeline composition and bind its view in the application registry. Method-specific algorithms, captures and their tests belong below the method's feature directory. Executors retain resource allocation and dispatch sequencing; shared framework modules must not import a method, SVO or React application implementation.

## Verification

Progress commits: `9a7070ed` (persistence, metadata and lifecycle) and `ed5a9539` (pressure inspection and host state). Only this task's patches were staged; concurrent editor, scenery and numerical changes were left to their owners.

- Production build passes.
- Focused checks: 40 passed, 8 GPU cases skipped in the CPU run, no failures.
- Full CPU suite: 1,160 passed, 24 failed, 168 skipped. All 24 failing names match the recorded pre-cleanup baseline.
- No TypeScript errors were reported in the new feature/lifecycle code. The shared checkout still has pre-existing errors and a concurrent scenery-test error; global typecheck is not green.
- Module boundaries retain the same three pre-existing diagnostic-route violations in `app/cm12-hole-probe/page.tsx`.
- GPU gate results are recorded separately after the exclusive WebGPU lease becomes available. Passing CPU checks are not a GPU acceptance claim.
