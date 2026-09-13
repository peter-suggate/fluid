# Rust/Wasm CPU implementation

This is the implementation companion to [the research plan](advance-lab-rust-wasm-plan.md). The 2D runtime cutover and selectable 3D CPU backend are implemented. The verification and performance limits below remain explicit.

## UI testing handoff

Performance work is paused at the user's request. The local production preview uses `http://localhost:3003`. Choose **CPU · Rust/Wasm** in the simulation pipeline's physics backend setting for authored 3D scenes; GPU remains the default.

The frame-13 water-box failure is fixed. The exact UI-default scene now passes 30 native frames and 60 complete threaded-Wasm frames with finite bounded density, signed finite RDF, visible-field evolution, nonzero velocity, and closed-scene liquid conservation within the existing `32 × f32::EPSILON` bound. The full Wasm flow also passes its wet-sphere, body-free, and live-command lanes. Moving-solid frames retain their established capacity-change refinement and explicitly report that the static closed-air and secondary air constraints are skipped. This preserves the existing moving-body contract; it does not claim a new moving-air formulation.

The exact native flux census identifies a trapped singleton air cell at finest-grid bounds `[21,2,2]..[22,3,3]`. Its six liquid interface faces receive a net rate of `+174.965145111`, while the surrounding 4,469-cell air region receives `-174.964857467`. Exterior, wall, source, and capacity-rate contributions are all zero. The primary physical divergence sum equals its algebraic divergence sum (`-0.000287645`); this is componentwise air exchange, not a primary operator mismatch or an oversized timestep. The census and original artifact hashes are recorded in [the defect receipt](advance-lab-rust-wasm-closed-air-fix.json). A zero-pressure warm-start experiment only delayed the failure and was rejected. The correction adds one uniform pressure unknown per sealed air region to the primary solve, including singleton pockets. The secondary solve preserves the resulting primary-owned interface rates. The formulation follows the per-region pressure constraint in [Goldade and Batty, Constraint Bubbles (2017)](https://cs.uwaterloo.ca/~c2batty/papers/Goldade2017.pdf).

UI testing also exposed a stale surface-cache revision: the CPU adapter uploaded changing density and RDF textures, but the renderer retained its initial extracted mesh because the adapter did not publish a render revision. A headless comparison confirmed substantial physical evolution by frame 11 (density centroid `[5.50, 6.50, 4.50]` to `[8.96, 2.09, 7.25]`, density L1 change 1939, conserved total within `4e-4`). The adapter now publishes the Rust frame count and a separate surface revision; the renderer keys mesh extraction by the surface revision when supplied, so paused edits can invalidate the mesh without inventing a solver step. The production rebuild passed, the real threaded adapter test passed with paused surface-revision assertions, and a browser integration check visibly confirmed collapse and spreading at 0.4 seconds. After the closed-air fix, a fresh production browser check advanced beyond the original fault and visibly confirmed collapsed, spreading water. The CPU preview is now left paused at 1.0333 seconds with 5,248 active sparse cells and no runtime halt. All nine production isolation-header checks passed on port 3003.

## Runtime boundary

`fluid-core` owns scene sampling, sparse topology, physical fields, pressure, geometric transport, source accounting, rigid coupling, adaptive transactions, tracers, and surface reconstruction. `fluid-wasm` is the serialization and ownership boundary. The browser worker owns one mutable world; the UI receives immutable revision-tagged publications and sends ordered commands.

The 2D production constructor accepts the canonical authored `SceneDocument`. The host can materialize the scene catalogue's authored scenery graph, but it does not rasterize liquid, solids, capacity, or topology for the solver. Scalar, SIMD, and threaded builds all execute Rust physics. There is no JavaScript numerical fallback in the new host.

The 3D GPU implementation remains the default backend. The Rust backend now has an owned 3D numerical world, scene lifecycle, adaptive policy, dynamic rigid state, sources, and dense renderer publication adapter. The host CPU selector is enabled, while GPU remains the default choice.

## Determinism and parallel work

- Wasm SIMD uses `simd128`; native Apple Silicon checks exercise NEON. Arithmetic does not use relaxed SIMD or fused multiply-add.
- Pointwise pressure operations process contiguous active spans, including sparse pressure membership. Inactive words are preserved.
- Pressure operator rows and fixed 64-lane reduction groups have disjoint outputs. Final reduction order remains the production order, independent of the Rayon pool size.
- The threaded artifact rebuilds the Rust standard library with atomics and uses imported shared memory. The host requires cross-origin isolation before selecting it.
- A topology generation owns one spatial lookup index. The 2D index uses a finest-cell owner table; 3D uses sparse lattice ownership. Lookup is not a linear cell scan on compiled production graphs.

The initial six-lane baseline was measured before strict instruction validation and before the transfer and dispatch work below. Its artifact labelled scalar still contained explicit SIMD instructions, which the later `wasmparser` check detected. The retained `*-before.json` receipts therefore remain valid exact-output baselines, but they cannot support a scalar-versus-SIMD speedup claim.

The current performance changes preserve source-ID arithmetic order and serial commit order. Three-dimensional conservative transfer queries a sparse dyadic owner index for overlapping source cells, precomputes source and target face boxes, and uses an exact coplanar face index instead of testing every source/target cell and face pair. A mixed-width topology clipped at the domain boundary is checked bit for bit against a test-only brute-force oracle. World advance no longer reconstructs RDF twice: a topology transition keeps the reconstruction it produced, while an unchanged topology constructs it once. Production 2D and 3D face-flux evaluation dispatches pure 1,024-face chunks through Rayon at 8,192 physical subfaces, then faults and outputs commit in canonical order. Small workloads stay serial to avoid worker-pool overhead. The other current thresholds are 64 fixed-64 pressure reduction groups (4,096 active entries), 8,192 pointwise entries with 2,048-entry chunks, and 512 rows/cells for the heavier independent 3D preparation and reconstruction stages.

## Accepted-generation transactions

Candidate topology and conservative transfer are validated before replacing accepted state. Missing target coverage must be explicitly admitted as new air. The fixed leaf arena does not replenish its budget each frame. Dynamic leaf IDs become reusable only after a successful presentation publication.

During moving-solid frames, the beginning-of-frame capacity plane owns the accepted scalar image until transport commits final apertures. A projected-support regrid transfers and validates that capacity epoch conservatively, then restores the candidate's final capacity and derived capacity rate. This fixed the first-frame wet-sphere case without relaying liquid, clamping density, or weakening source and target capacity checks. When refinement children are covered by a merged parent, their obsolete dynamic IDs enter the delayed release queue without being misclassified as discarded wet leaves. Coverage is measured in finest-cell coordinates and clipped to the physical domain, so partial boundary pages release correctly; authored IDs remain reserved.

Projected transport support is accepted before microstep zero. Source requests are frozen before transport, and the source ledger commits at each successful volume microstep. Scalar publication follows transport. Rigid reaction loads are consumed on the following advance.

Live 3D drag constraints use `set-rigid-constraints` records containing an ID, held flag, pose, linear velocity, and angular velocity. Rust validates the complete batch before mutation and updates only the named live bodies; other autonomous poses are retained. The paused command stages old/new geometry, conservative scalar remap metadata, PLIC reconstruction, RDF reconstruction, and field/surface revisions before committing. Held bodies do not integrate under gravity or contacts. Releasing a body resumes integration from the supplied velocities, with world-space angular momentum reconstructed from the normalized pose and body inertia.

Historical lens data uses the fixed finest canvas lattice so regridding does not change the meaning of an earlier observation. In particular, pre-transport density and pre-force face velocities are retained independently of the final topology. Numerical fields are published as aligned typed binary planes, not JSON arrays.

## Verification strategy

Use native and Node tests for normal development. Browser checks are reserved for module loading, cross-origin isolation, worker integration, and visible publication. Browser sessions and Dawn runs must never overlap.

The JavaScript 2D numerical engine and its transitional parity generators have been removed. Rust tests retain frozen source-derived scene, atlas, pressure, transport, scalar, RDF, and transfer evidence. Four production scenes compare generation zero and three complete advances word for word. A separate pressure-budget-eight fixture compares every published physical field at two frames. The old test's expected automatic rerung was stale: an isolated copy of the untouched JavaScript implementation also stayed at generation one. The frozen actual outputs replace that invalid assertion.

Direct production GPU comparisons cover six signed/rotated 3D PLIC/RDF planes and 48 signed-axis, mixed-width, aperture flux cases. The maximum observed high-flux difference was 6.97e-8; the largest center-RDF difference was 1.26e-6. These stage fixtures do not establish full-frame 3D parity. The policy fixtures capture the GPU mask ABI and curvature primitives; complete identical-input policy decision receipts remain a separate verification item.

The accepted 3D geometric-transport compatibility stage projects physical subface rates over the complete accepted non-pressure air domain. It forms deterministic smallest-root components, admits open physical boundary faces as component vents, removes only the representable weighted mean from closed components, solves the weighted graph Laplacian, and applies bounded tree/open-face postconditioning until the normalized residual reaches the fixed target. The final 3D authority stores corrected physical subface rates in `f64`: a single `f32` correction or rate can have a representable increment larger than a small cell’s two-epsilon allowance. Postconditioning adjusts those rates directly; transport multiplies rate by timestep in `f64` before crossing the existing `f32` geometric material-flux boundary. Velocity corrections are derived diagnostics. The secondary solve includes only air cells with a correctable incidence; fixed primary-pressure faces retain primary ownership. Frame, topology, timestep, runtime, geometry, and injection changes invalidate the rate plane. The exact UI-default water box passes thirty complete native advances and sixty threaded-Wasm advances with the unchanged tolerance. Its receipt reports the initial, recursive, measured, and pre/postconditioning residuals plus the maximum projected volume error. Moving-solid frames currently report an explicit accepted `skippedForSolidMotion` receipt because their interpolated capacity/aperture transport has a separate contract; this is a recorded limitation rather than evidence of a compatibility solve.

Useful focused commands:

```sh
cargo +1.96.1 test --manifest-path rust/Cargo.toml -p fluid-core --features parallel
node --import tsx --test lib/physics-wasm/*.test.ts advance-lab/*.test.ts
npm run build:physics-wasm
npm run test:physics-wasm:world
npm run test:physics-wasm:world3d
npm run test:physics-wasm:world3d:sustained
npm run test:physics-wasm:adapter-world3d
npm run test:physics-wasm:threaded
node --import tsx tools/wasm/world-benchmark.ts
```

Toolchain installation, artifact architecture, cross-origin headers, and headless checks are documented in [the Rust workspace README](../../rust/README.md).

The required isolated `npm run test:dawn:sparse-cm12` run is recorded in [advance-lab-rust-wasm-dawn-regression.json](advance-lab-rust-wasm-dawn-regression.json). It completed within the unchanged 480-second suite budget in 217.472 seconds, but failed overall: 3 of 17 lanes passed, 12 exited with failure, and the two live-edit lanes exceeded their unchanged 25-second timeouts. The captured context identifies a production GPU registry request for the absent `certifyGeometricCompatibilityResidualTargets` WGSL entry point. No gate criterion or timing ceiling was changed. The retained log SHA-256 is `35d1aed25ca37a622ed95f43d266feb095850702717ad68b742fe8a51bd4f205`. This is a required gate result and is not a passing release receipt.

The subsequent full gate is [recorded here](advance-lab-rust-wasm-dawn-current.json). The current GPU shader contained an immutable `let sweep` that its supported-boundary branch assigned to; changing only that declaration to `var` restored compilation without changing the arithmetic. The full rerun then reached numerical checks and failed in 351.854 seconds: 3/17 lanes passed, nine failed, and five hit their unchanged lane timeouts. Failures include first-step symmetric expansion mass loss, the authored topology budget assertion, and a hydrostatic geometric transport halt. The suite remains within its 480-second budget, but the GPU regression gate is **not green**. The intermediate compilation-failure run is retained in [this receipt](advance-lab-rust-wasm-dawn-before-compile-fix.json).

After the closed-air CPU fix, the required isolated GPU gate was run again without a browser: [latest receipt](advance-lab-rust-wasm-dawn-closed-air.json). It completed in 351.643 seconds with 4/17 lanes passing, eight failures, and five unchanged lane timeouts. GPU transport halts remain in hydrostatic, mini32, mini64, terrain, and rigid-body lanes. No acceptance threshold or timeout was changed. The CPU tests and final browser integration pass independently; this shared-workspace GPU gate remains red.

An isolated Figure 3 owner-query benchmark improved 200,000 queries from 135.3 ms to 0.5 ms on the development machine. This measures lookup only. Whole-frame speed, scaling with worker count, initialization cost, memory consumption, and publication cost must be measured separately before making a production performance claim.


## Runtime and measurement evidence

The 2D browser integration passed cross-origin isolation, threaded module loading, initial publication, step, play/pause, reset, and scene change. Subsequent routine checks are headless. The final threaded Node 3D flow passes 60 UI-default water-box advances plus held constraint/release, wet-sphere moving-capacity transfer, body-free advance, live injection, scene edits, and renderer publication decoding. The Node 2D flow and real threaded adapter gate also pass. The latest complete native release suite passes 118 unit tests and 17 integration tests, with one manual throughput test and one manual flux census ignored. The original browser frame-13 defect and stale mesh were separate failures; the closed-air census explains the numerical failure, while independent surface revisions fix mesh updates.

The static post-transition pressure refinement now uses a coupled liquid/closed-air system. Its single-precision warm solve and double-precision true-residual refinement each have an explicit 4,096-iteration ceiling; the initial pressure solve still uses the user's iteration cap. The double-precision operator and published physical rates use the same coefficients and arithmetic, and acceptance checks every liquid cell, sealed-air component, and pinned gauge equation against `2 × f32::EPSILON`. There is no arbitrary face-flux repair. The `coupledPressure` receipt reports sealed-component count, warm-solve and double-precision iteration counts, pre-refinement and final physical residuals, target, and moving-solid skip status. Total pressure work includes the double-precision iterations. Moving-solid frames keep the prior 64-iteration refinement instead of applying the static constraint to interpolated geometry.

Production worker bootstrap requires the isolation headers on hashed simulation/render worker assets as well as HTML and Wasm. `npm start` now applies these before Vinext’s static-asset fast path; the deployment `_headers` file carries the same policy. Stable Wasm glue, binaries, and Rayon helpers revalidate rather than retaining a one-hour cache entry. `npm run check:physics-wasm:serving -- http://localhost:3003` checks these actual responses without a browser.

Builds record source and binary SHA-256 fingerprints for scalar, SIMD, and threaded artifacts. The whole-frame benchmark rejects mixed-source builds and compares complete numerical publications and receipts across scalar, SIMD, and 1/2/4/8-worker lanes. Final measured results replace the earlier unisolated smoke measurements below.

The CPU 3D topology currently uses B8 pages. The CPU controls adopt B8 explicitly; the GPU retains its B4/B8/B16 options and remains the default. Method policy, timestep, pressure controls, initial atlas settings, and live numerical updates are interpreted in Rust. The host admits one bounded step toward the requested clock using Rust’s published timestep, so accumulated wall-clock debt cannot become one oversized solve. It tracks admitted host time separately from the rounded numerical timestamp. Paused edits publish their completed Rust fields and poses immediately, and pending command publications fence the next advance. The renderer uploads completed dense density, RDF, velocity, and rigid-pose publications. This dense presentation copy is measured separately from the sparse solver.

## Final build verification

The Sites build wrapper completed the production Vinext build with all three newly compiled Wasm artifacts. Strict `wasmparser` validation counted zero SIMD and zero atomic instructions in scalar, 13,334 SIMD and zero atomic instructions in SIMD, and 13,470 SIMD plus 1,219 atomic instructions in threaded. Relaxed SIMD is rejected in every artifact. The final native release suite passed 118 unit tests and 17 integration tests; the two ignored tests are a manual throughput benchmark and a manual flux census. The real rebuilt Node 2D flow, 3D command flow, and initialized four-worker pressure parity all passed.

The final 3D command flow also passes sixty UI-default advances on the rebuilt threaded artifact with eight workers. Its command scenario includes two advances, runtime pressure/timestep changes, tracers, spherical and disk water insertion, authored scene and body edits, paused hold/move/release constraints, and renderer binary decoding. Separate wet-body and body-free pressure-12 regressions remain in that gate. A separate real threaded-Wasm adapter gate drives four bounded steps from a one-second target jump, then verifies a paused timestep change, liquid insertion, held-pose publication, all nine publication releases, and presentation resource destruction using recorded GPU uploads. The 34 host tests also pass. These checks establish the tested flows, not every long-duration scene or full-frame GPU numerical equivalence.

## M1 Max whole-frame measurements before the final 3D precision and closed-air fixes

These recorded measurements ran sequentially in Node 22 on this M1 Max with no concurrent project builds/tests or browser automation. Ordinary desktop applications remained running. The 2D run uses 5 warmup plus 20 measured frames for three scenes; the 3D run uses 1 warmup plus 4 measured frames for the 24×16×16 water box. Both benchmark dimensions use a pressure budget of 28, relative tolerance 1e-5, B8, macro span 1, one fine surface ring, a 1/30-second step, and a tracer budget of 1,000. These differ from the UI defaults (128 pressure iterations and tolerance 1e-6). These are bounded scene measurements, not sustained thermal or browser frame-rate guarantees.

| Scene | Scalar advance median | SIMD | 4 workers | 8 workers |
| --- | ---: | ---: | ---: | ---: |
| 2D water box | 3.47 ms | 3.41 ms | 4.28 ms | 4.22 ms |
| 2D quarter pool | 6.96 ms | 6.88 ms | 7.89 ms | 8.07 ms |
| 2D Figure 3 | 190.26 ms | 188.83 ms | 216.75 ms | 234.61 ms |
| 3D water box | 504.24 ms | 501.24 ms | 282.43 ms | 237.23 ms |

The 3D eight-worker median is 237.23 ms, about 2.11× faster than SIMD alone and 2.13× faster than scalar. Publication is measured separately (~0.8–1.0 ms for this small 3D scene); renderer upload is not included. The eight-worker p95 is 374.39 ms, so this scene is not yet real-time. The tested 2D cases do not demonstrate useful thread scaling. Advance Lab therefore defaults to single-worker SIMD (scalar where SIMD is unavailable); the controller still accepts an explicit threaded override for experiments. The 3D CPU backend defaults to the capability-selected threaded artifact, up to eight workers.

A Node CPU profile identified canonical topology ordering, hashing, and heap allocation as the main 2D Figure 3 costs. Caching each canonical sort key once per generation reduced its SIMD median from 251.74 ms to 188.83 ms. Pressure/transport work remains parallel where useful, but more workers are not assumed to improve the whole frame. [Profile summary](advance-lab-rust-wasm-profile-2d.json).

At the recorded source fingerprints, all six lanes (scalar, SIMD, threaded 1/2/4/8) match every published numerical plane and complete receipt exactly, excluding allocation-byte accounting. Those outputs also match the pre-optimization snapshots exactly. Performance work was then stopped at the user’s request; the subsequent 3D precision fix is checked for correctness rather than rebenchmarked. The 2D numerical implementation is unchanged by that fix. Raw results: [2D scaling](advance-lab-rust-wasm-scaling-2d.json), [3D scaling](advance-lab-rust-wasm-scaling-3d.json). Before files retain their original evidence and explicitly note that the earlier artifact labelled scalar accidentally contained SIMD; they must not be used as a scalar/SIMD speed comparison.
