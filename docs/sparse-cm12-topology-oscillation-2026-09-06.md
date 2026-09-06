# Sparse CM12 topology round-trip conservation

## Experiment

`tools/probe-topology-oscillation-dawn.ts` runs a tiny 16³ dam, symmetric across z, with a whole-domain width-two enforcement region. Both arms start with the same two width-two steps. The control remains min2/max2. The oscillating arm alternates min1/max1 and min2/max2 on every following frame, asserting an accepted topology change each time. Explicit width-one forcing makes the experiment independent of whether the automatic selector happens to ask for refinement. `TOPOLOGY_ARM=allow` also supports the originally proposed max2 / min2-max2 policy oscillation, but that policy alone does not guarantee a topology change.

Read-only GPU state copies immediately after velocity projection and immediately after candidate publication isolate the topology interval. Scalar mass, cell momentum, cell kinetic energy and staggered face quadrature kinetic energy are measured using each snapshot's accepted cells and rows. Refinement changes the face quadrature: comparing its energy directly across different rungs can show an increase or decrease even for a correct normal-linear reconstruction. The decisive comparison therefore returns to the same final width-two grid.

A second, near-zero-timestep variant seeds a D4-compatible divergence-free staggered mode in a full tank. Gravity, gamma diffusion and sharpening are disabled; dt is one microsecond. Its fixed-grid control accounts for ordinary transport/projection drift. It runs 16 frames, including 14 actual topology changes. The initial state and first two frames match exactly. This is a transfer stress test, not a dam-break physical energy measurement.

## Demonstrated fault and correction

`transferCandidateCellsWork` already preserved mass and collocated cell momentum within its floating-point receipts. However, `publishCandidateShadowFaces` replaced changed staggered faces by weighted means of the transferred **collocated** velocities. The exact exterior flux calculations were only receipts, not the published face authority. Consequently each topology change averaged faces into cells and then averaged cells back into faces. Repeated refinement/coarsening filtered a velocity mode that the coarse grid could already represent.

Candidate validation now remaps the authoritative old staggered field before any old rung or cell field is overwritten. An existing accepted row keeps its exact face value. A new patch integrates the old field, linear in the face-normal direction and piecewise constant tangentially within each donor cell. The integration visits donor rectangles, not finest-grid voxels, so macro cells remain useful. The validated values are cached in transient candidate storage and published only after the topology transaction is authorized. Scalar transfer and topology selection are unchanged.

The additional scratch bound reserves one float per resident row after the candidate fields; the existing scratch buffer is reused. The normal-linear reconstruction preserves integrated divergence under refinement and area-averaged normal flux under coarsening. It does not invent missing fine-scale velocity detail.

## Same-final-grid result

Artifacts are under `artifacts/topology-oscillation/` (ignored by git). Energy units below omit the common physical cell-width and density factors; the ratios are dimensionless.

| 16-frame near-zero-dt arm | Final face kinetic energy | Relative to fixed width two |
| --- | ---: | ---: |
| Fixed min2/max2 | 681.353402 | 100% |
| Oscillating, previous collocated face publication | 4.451317 | 0.6533% |
| Oscillating, conservative staggered remap | 679.909272 | 99.7880% |

All three have total density mass exactly 4096 fine-cell equivalents and zero density reflection error. The corrected oscillating arm's collocated kinetic energy is 532.342896 versus 533.405458 for the control. Every requested oscillation commits and returns to the same final grid, without transaction faults. This isolates a topology-specific loss; it does not imply all remaining coarse-grid advection/projection losses have been eliminated.

The ordinary 30-frame dam also completed all 28 forced topology changes with no faults. Before the fix, late topology-only face quadrature reductions were approximately 3–5% per change. Density mass and cell momentum receipts already closed; the face filtering was not caught by those receipts. Dynamic dam trajectories differ because alternate frames use different resolution, so their raw final energy is not an isolated measure of the remap.

## Verification

Passed locally on Dawn Metal:

- `tests/sparse-cm12-topology-face-transfer-dawn.test.ts`: production WGSL subface flux, inserted normal face, and coarse patch integration on all three axes.
- `tests/sparse-cm12-topology-oscillation-dawn.test.ts`: actual solver fixed-grid versus 14 topology changes, exact warm-up match, exact mass and symmetry, real accepted transitions, same final grid, and kinetic energy within 2% of the fixed control. The previous publication retains only 0.65% and fails this bound.
- Existing `tests/sparse-cm12-clipped-transfer-dawn.test.ts`: clipped-domain coarsening/refinement retains conservative transfer receipts.

The final combined pool-impact suite passes all 13 tests, including both topology regressions above. Every canonical correctness lane has also passed: the four lanes interrupted or skipped by the full suite's 180-second budget passed in separate diagnostic runs. The full canonical gate remains unsuccessful because mini64 takes 107.6101 ms against its 50 ms limit and the suite exceeds its total time budget. Mini32 passes at 35.8482 ms against 40 ms. No existing thresholds were relaxed. See the combined validation in `docs/coarse-first-pool-impact-realism-2026-09-06.md`.

Run an arm with:

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal TOPOLOGY_ARM=oscillate TOPOLOGY_TRANSFER_ONLY=1 TOPOLOGY_STEPS=16 node --import tsx tools/probe-topology-oscillation-dawn.ts
```

Set `TOPOLOGY_CAPTURE_FIELDS=1` to save the template, state layout, pre/post topology GPU states and their corresponding accepted rung maps, plus each frame's dense density field. Those snapshots can distinguish surface publication changes from actual scalar transfer changes. Run Dawn sequentially under the repository WebGPU lease, with the browser unloaded.
