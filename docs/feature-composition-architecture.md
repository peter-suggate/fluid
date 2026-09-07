# Feature composition architecture

Implementation worktree: `codex/feature-composition`, based on `e894aa04` plus the source checkout's working changes copied before implementation. The source checkout is not the implementation destination.

## Ownership

A feature owns its settings and control semantics, CPU/GPU implementations, diagnostics, reference calculations, tests, and probes. A method or SVO pipeline owns composition and numerical execution. A UI host projects the selected composition into named slots.

The shared kernel is `lib/framework/composition.ts`. It resolves feature dependencies, variant selections, control references and placements, and explicit publication port connections before the host changes state. There is no dynamic plugin loader, runtime registration order, or generic callback scheduler.

`lib/framework/ports.ts` describes typed payloads, representations, units and lifetimes. Matching a port name alone does not make two representations interchangeable. Consumers name their providers explicitly. The domain's existing publication machinery continues to own actual buffers, generation acceptance and lifetime; the kernel does not copy GPU resources or turn timestep feedback into an acyclic execution schedule.

## Package map

- `lib/framework`: domain-independent composition, publication contracts, generic React slot host and verification.
- `lib/features/gravity`: shared gravity commands, persistent remembered vector, query paths, metadata, controls and tests.
- `lib/features/ui`: explicit application composition and bindings between feature control IDs and React implementations.
- `lib/methods/<method>/composition.ts`: each method's supported configuration and surface publication contracts.
- `lib/methods/adaptive-mass/features/adaptivity`: policy, settings, live availability rules, packing, CPU algorithms, WGSL, UI, verification and probes.
- `lib/svo/contracts`: scene and rendering publication contracts.
- `lib/svo/features`: construction, primary visibility, lighting visibility, radiance, materials, scene publication and diagnostics ownership.
- `lib/svo/pipeline`: SVO orchestration, composition, tuning and graph presentation.
- `lib/core/render-frame-stages.ts`: the shared presentation-frame timing ABI. Water, scene construction and SVO all close seams against this common contract.

Shared code stays shared when it is an actual cross-domain contract. Feature ownership and execution stages are not one-to-one: fused GPU work stays fused.

## Feature definition and UI

A `FeatureDefinition` declares its stable ID, capabilities, optional publication inputs/outputs, controls, placements, and variants. A variant belongs to a variation point and declares its requirements, capabilities and update impact. A composition must select one supported variant for each declared point, explicitly or through a unique default.

A control's identity is `(feature ID, control ID)`. Multiple placements reference that identity. A placement specifies an arbitrary slot, order, and compact or expanded presentation. `ComposedFeatureSlot` renders through an explicit binding catalog and rejects missing bindings. Features can implement specialized visualization or grouped controls using the same shared primitives.

The application binds the active method and SVO selections; it does not render default-only configurations while the runtime uses something else. SVO owns its own binding catalog so its UI does not depend on application composition.

Gravity's remembered vector is scene data, not component state. Compact toggling and expanded editing call the same pure commands. History, pane isolation, serialization and URL state retain the remembered vector. UI collapse/open state remains local because it does not change feature semantics.

## Runtime configuration

Method stores and controller commands preflight configuration before publishing mutations or announcing GPU rebuilds. The method descriptor owns `resolveComposition(values)`; generic value resolution also validates the resulting composition. SVO selection setters validate the complete primary/lighting configuration before publishing a new state.

Hosts retain their existing rebuild/reset and resource readiness mechanisms. Composition metadata makes compatibility explicit; it does not replace GPU completion, publication acceptance, cache invalidation, or method-specific timestep sequencing.

## Verification

`npm run test:unit` discovers `*.test.ts` and `*.test.tsx` recursively under both `tests` and `lib`. `tests` holds cross-feature integration tests; feature-local tests stay with their implementation. Production compilation-policy checks distinguish colocated tests from production files. GPU lanes remain explicit and retain the repository-wide WebGPU lease.

Required checks for this cutover:

- Production build.
- Feature composition, UI binding, gravity persistence and transaction tests.
- Method/SVO reference and source-contract tests after relocation.
- Typecheck compared with the inherited checkout errors.
- `npm run test:dawn:sparse-cm12`, without a browser or another Dawn process.

The source checkout already has failing CPU tests and TypeScript errors. Validation results must distinguish inherited failures from introduced failures. Never change an oracle, numerical threshold or timing ceiling merely to obtain a green cutover.

## Deliberate boundaries

This is an internal composition system, not an external plugin marketplace. Static imports and explicit host composition remain intentional. Simulation and SVO share contracts and UI infrastructure, while retaining distinct executors. Old relocated modules are deleted rather than retained as forwarding exports.

The method bootstrap installs the immutable method identity catalog at application/worker/tool entry points. Each method carries its resolved feature composition; the catalog does not register features independently. Existing low-level domain resource and publication APIs remain authoritative; new feature declarations do not introduce competing resource owners.

## Validation result

The production build passes. The final focused architecture/configuration/UI group passes 27/27 tests. The full discovery run reports 1,028 passing, 24 failing and 158 skipped tests; its failing names exactly match the 24 failures in the source checkout. The typecheck still reports inherited errors, with no new file/error-code categories. Module-boundary violations fall from four to three; the remaining three belong to the existing `app/cm12-hole-probe` diagnostic route.

Dawn acceptance is **not green**. Both the worktree and source checkout fail D4 symmetry and time out in multiple lanes, exhausting the unchanged 180-second suite budget. The initial worktree performance probes also lacked an ignored pinned baseline artifact. That exact artifact was copied from the source checkout; isolated mini32 and mini64 performance retries then hit their unchanged 20/30-second timeouts. Later lanes skipped by the suite budget remain unverified. No numerical tolerance, oracle, timing ceiling, or lane was relaxed.

See `feature-composition-validation.json` for receipts and inherited failing test names, and `feature-composition-shader-equivalence.json` for shader identity evidence. See `lib/methods/VARIATION_POINTS.md` for supported method dimensions and the boundary between algorithm metadata and domain-owned solver implementations.

The worktree is reviewable, but these results are not a clean merge/acceptance certification. No commit, merge or deployment was performed.

## Follow-up on main

Shared persistence, control metadata and lifecycle integration, plus pressure inspection ownership, are described in [feature-cleanup-followup.md](feature-cleanup-followup.md). That record supersedes the initial worktree status for these areas.
