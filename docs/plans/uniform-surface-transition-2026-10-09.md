# Rectangular surface artifacts at the adaptive NB handoff

The adaptive controls were committed as `1d0a8d6b`. The initial smooth-bowl
investigation found a small render interpolation shoulder, but did not reproduce
the user's Letters screenshot. Its proposed cubic rendering workaround was
removed: it did not repair the canonical trenches and made warm extraction
roughly 4–6 times slower in the 32³ moving-pool probe (0.4–0.7 ms became
2.5–2.9 ms). The final change leaves surface extraction unchanged.

## Actual reproduction

The reported UI is the authored 256 × 192 × 128 Letters scene, adaptive budget
50%, retirement 0.5 seconds, dt = 1/60 second, at t = 0.7 seconds.
Dawn reproduces rectangular trenches in canonical phi, before rendering.
At 0.6 seconds the outer rectangular ring falls to 34.802h against a resting
pool height of 37h; at 0.7 seconds a larger ring falls to 34.950h.
A smaller motionless 32³ pool shows the cause without impact dynamics:
activating 25% of its requesting tiles cuts a trench over 4h deep in six steps.

```bash
npm run test:dawn -- uniform-narrow-band-handoff-surface
```

The file includes both the reduced pool and full Letters scene. To export the
canonical height maps at 0.6 and 0.7 seconds:

```bash
FLUID_TRANSITION_REPORT=/tmp/fluid-handoff \
  npm run test:dawn -- uniform-narrow-band-handoff-surface
```

## Cause and correction

`markSurfaceTiles` expanded both gather coverage and particle heat by 2h.
Trilinear interpolation expanded that influence further, while `clamp(heat,
0,1)` assigned full particle authority to the heat-one overlap collar.
The reconstruction consequently shrank the Eulerian surface outside the complete
seeded footprint. A nonempty gather is not proof of complete particle coverage.

The original expanded heat raster remains for velocity transfer and adding
particle-carried liquid. A separate occupied-tile heat raster controls erasure:
`clamp(heat - 1, 0, 1)` fades inside the seeded collar before its outer edge.
This costs one additional float per 4h tile, not additional h tiles or particles.
Air outside a sheet or droplet is excluded from the erasure heat's interpolation
weight: air is not uncovered liquid. Cold liquid remains in the denominator,
so the transition still fades before an unseeded pool patch. This distinction
preserves the existing moving thin-sheet regression.

During retirement, coarse vertices adjacent to fine owners recover their
metric distances once erasure influence reaches zero. Particle sphere unions
preserve the fine zero set but distort interior phi magnitudes; using those
magnitudes directly as a 4h chord moved an otherwise resting surface. The
already-computed distance field supplies the correction. Adaptive end-step
membership rebuilds its search and metric cache afterward because coarse
normalization can move hanging zero crossings. Fixed-band NB and
cold all-4h regions away from fine owners keep their previous behavior.

All activity still comes from the existing criteria, source heat and transported
particle heat. The percentage denominator, ranking, admission, retirement time,
and particle-support requirements are unchanged. Surface-erasure influence fades
earlier than velocity/particle-addition influence, and the control hint now
states that surface influence fades before remaining support retires.

## Regression evidence

- Initial partial activation, live retirement to zero, and reactivation preserve
  the flat pool across all columns and all 20 steps; the test requires 0.01h.
- The full Letters reproduction samples the two offending rectangular rings.
  The regression permits 0.5h sub-cell ripple (the default shape tolerance),
  rejecting the original trenches exceeding 2h.
- All seven existing adaptive activity tests pass unchanged, including live
  quota counts, retirement timing, resting-pool cycles, quiet thin features,
  source insertion, and a translating thin sheet.
- A fresh Chrome worker replayed the same scene to exactly 0.7000 seconds,
  with Adaptive on, 50% budget, 16.7 ms steps, 256 × 192 × 128 cells, and
  Simple shading. The rectangular terraces are absent with the original
  renderer. The capture is `docs/verification/narrow-band-letters-ui-fixed.png`
  (close-up: `narrow-band-letters-ui-fixed-detail.png`). Diagnostic slice
  overlays were hidden to inspect the actual water mesh.

## Validation

The full Dawn run and affected-file rerun cover all 83 files: **80 pass**.
Every maintained Uniform lane passes. The three remaining failures reproduce
pre-existing narrow-band failures:

- `uniform-narrow-band-contract-dawn.test.ts`: hydrostatic drift
  0.007553498 m (baseline approximately 0.0075535 m).
- `uniform-narrow-band-settling-dawn.test.ts`: settling roughness
  0.0727533h and translating-sheet represented volume below the existing limit.
- `uniform-narrow-band-volume-dawn.test.ts`: Figure 2 volume bounds at
  dt 0.017 and 0.05 seconds (−11.543% and −15.069% at approximately 0.9 s).

The first full run also exposed a missing `nbSurfaceTheta` definition in the
isolated fixed-band gather fixture. Supplying its unchanged fixed-band weight
of one makes all six tests in that file pass. Both particle-overlay files were
rerun after concurrent UI edits and passed. No existing numerical assertion,
maintained Uniform tolerance or timing ceiling was relaxed.

`npm run check:types` passes. `npm run test:unit` passes: 913 passed,
118 GPU-gated tests skipped. The final fresh-worker UI replay also reaches
0.7000 seconds without rectangular terraces or browser console errors.
