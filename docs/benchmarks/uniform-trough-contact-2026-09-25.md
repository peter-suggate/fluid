# Voxel trough wall release

The real `uniform-trough-dam-break` scene, Uniform Geometric, balanced defaults,
was run for 180 steps (six seconds) on native Metal Dawn. All GPU runs were
sequential under the repository lease, with Fluid Lab browser tabs closed.

## Cause

The earlier pond-rest fix made embedded-wall continuation min-only. This
protects submerged stair vertices: their inward sample can lie above the water
plane, so unconditional replacement would cut air into the resting pond.
However, it also prevents a wet wall vertex from drying after the water leaves.
Closed-wall tracing has no normal escape, and repeated surface correction can
amplify the stale negative field. The visible back-wall film is not retained
liquid volume or physical viscosity/surface tension (both are zero here).

At one second, the old rule had only 0.000126 cell-volumes in the upper back
region but 139 wet wall vertices without incident liquid evidence; their
minimum phi was -0.773 m. At six seconds, that region had effectively zero
volume and still had 49 wet vertices, with phi reaching -504.081 m.
Unconditional replacement cleared the region, confirming the contact rule as
the cause, but discards the pond safeguard.

## Change

Only a previously wet vertex without liquid evidence in its incident open
cells accepts replacement from the interior. Other contact keeps the existing
min rule, and explicit pressure-solved separation still wins. The evidence
threshold is the existing `UV_LIQUID_EVIDENCE` (0.05), shared with ghost draining;
this adds no scene-specific parameter or pressure-tolerance change.

Retaining the old rule on already-dry vertices matters. An intermediate guard
that also replaced dry values changed the initial dam pressure solve and hit
the unchanged tolerance limit on step two (5.016 > 5). It was rejected.

The accepted control clears the upper back wall at one and two seconds. It
allows the returning wave to wet it at three seconds and clears it again at
four, five, and six seconds. The first three steps match the old baseline's
reported mass, velocity maxima, and pressure residuals.

## Reproduction and regression checks

`tools/probe-uniform-trough-contact-dawn.ts` defaults to production at UI
tolerance 5. `--contact=pinned` restores the old rule in compiled shader source;
`--contact=assign` removes the pond safeguard. Both controls assert that the
shader replacement actually occurred. `--tolerance=0` selects the full solve.
JSON/log artifacts are written under `artifacts/trough-contact/`.

The new `tests/uniform-trough-contact-dawn.test.ts` checks sustained back-wall
drying, bounds unsupported negative phi, and verifies no volume enters solids.
It runs the authored scene through draining and rebound, rather than asserting
the contact implementation's arithmetic.

The unchanged `npm run test:dawn:uniform-pond-rest` gate passes. At three seconds
with the hose off and the existing full-solve configuration, hero interior RMS
surface error is 0.00493 mm, maximum whole-surface error is 0.0600 mm, residual is
0.0000616, and recovery count is zero. This does not claim still-water stability
for the loose global pressure tolerance; the established hero oracle uses zero.
All 13 recorded hero sample objects exactly equal the saved pre-change
`artifacts/pond-rest/production-after.json` samples, including surface, mass,
velocity, and pressure diagnostics (not merely the final error bounds).

The new trough test and existing geometric-boundary test also pass (8 tests):

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx --test --test-concurrency=1 tests/uniform-trough-contact-dawn.test.ts tests/uniform-geometric-boundary-dawn.test.ts
```

The production trough result exactly matches the accepted shader control's
reported upper-back wet counts: 0 at steps 30 and 60, 560 during rebound at 90,
and 0 at 120, 150, and 180. Unsupported wet vertices during rebound stay within
0.01545 m of zero (less than one 0.025 m cell), instead of the old runaway field.

`npx tsc --noEmit --incremental false` still reports errors in existing test
files (including Sparse CM12 typed-array/implicit-any errors). It reports none
in the contact shader, new probe, or new regression test. See
`artifacts/trough-contact/typecheck-final.log`.

The required `npm run test:dawn:sparse-cm12` ran to completion in 472.3 seconds:
5 lanes passed and 12 failed. Eight lanes timed out; mini64 measured 214.76 ms
against its unchanged 110 ms ceiling. Long Dam halted in
`addWholeFrameUncoveredDonorFallbacks` at frame 29; Tall Cells Hills halted in
`validateAndAuthorizeShadowTopology` at frame 17; outside-tank collapse halted in
`certifyGeometricTopologyFaces` at frame 1. The latter two report missing
compiled topology faces, and Long Dam reports a conservative transport contract
failure. The full report is `artifacts/trough-contact/sparse-gate.log`.

This is not a clean repository-wide gate. These lanes exercise the separate
Sparse CM12 implementation, which does not import the changed Uniform contact
shader. No Sparse CM12 code, lane, timeout, or performance ceiling was changed
for this fix; the shared checkout also contains other ongoing work.
