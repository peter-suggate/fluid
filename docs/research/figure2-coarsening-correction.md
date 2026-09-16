# Figure 2 free-fall coarsening correction

The `cm12-figure-2` scene is an extruded 2D disk running through the 3D
adaptive-volume solver. Its surface should coarsen while falling without
substantial deformation; impact may request finer cells again.

## Cause and changes

The new thin-feature tracing had accepted any second sign crossing within the
thin-width threshold, including short tangential chords near a broad curved
surface. It also denied surface proofs when exploratory traces left the sparse
represented field, even if the accepted cell itself had valid samples. Finally,
the planner interpreted a thin-feature veto as an unconditional finest-rung
request. Together these suppressed ordinary circle-surface coarsening.

- Require opposing normals at the two crossings (dot product below -0.25).
  Symmetric differences handle lattice creases; one-sided differences handle
  domain boundaries. The extra normal samples are taken only for short-chord
  candidates.
- Missing support outside the inspected cell terminates that exploratory trace.
  Missing support in the accepted cell still returns unresolved; the independent
  candidate restriction proof continues to require its own samples.
- Thin features retain their current resolution, with the existing final
  coarsening/retirement and macro-merge vetoes. They can still refine for actual
  deformation or transport demand.

No coarsening tolerance, timing ceiling or proof-epoch count was relaxed.

## Equal-time measurements

Thirty steps at 1/30 s, default scene and adaptive-volume options:

| Checkpoint | Before: accepted cells | After: accepted cells | Before/after B8 bricks |
|---|---:|---:|---:|
| Initial | 2704 | 2704 | 4 / 4 |
| Frame 5 | 6160 | 4144 | 10 / 6 |
| Frame 10 | 5328 | 1944 | 8 / 2 |
| Frame 20 | not captured | 332 | not captured / 0 |
| Frame 30 (impact) | 5286 | 6104 | 9 / 10 |

At frame 20 all active bricks are B1/B2. Refinement returns at impact. Median
frame time in these short runs changed from 117.5 ms to 85.6 ms. These timings
include topology changes and are not a general performance benchmark.

Both runs completed without WebGPU validation errors. Initial volume was 4888
fine-cell volumes; final volumes were 4888.0011 before and 4887.6003 after
(0.0082% lower than initial after correction; existing residue deletion remains
active). Negative-phi fine samples at frame 30 were 4297 before and 4266 after.
The measurements establish coarsening behaviour, not exact shape preservation.

Receipts: `artifacts/level-set-volume/figure2-coarsening-before.json` and
`figure2-coarsening-after.json`. The probe now includes per-brick adaptivity
records and rung histograms at checkpoints.

Focused validation: three Dawn tests passed, including sheets, air gaps,
corner droplets, broad-circle grazing cuts, missing exterior support and
unresolved accepted-cell support. Fifteen CPU control/generation/policy tests
passed. Repository typechecking still reports pre-existing errors outside the
changed files.

The canonical Dawn suite completed in 386.1 seconds with 6/17 lanes passing.
Mini64 performance passed at 78.6 ms against the unchanged 110 ms ceiling.
Six previously failing lanes timed out again; min8 surface, long-dam, terrain
far-wall and outside-collapse assertions also remain failing. Live rigid-body
coupling additionally exceeded its 25-second timeout in this run. No lane or
timing threshold was changed. Full receipt:
`artifacts/level-set-volume/figure2-coarsening-regression.json`.

The dedicated production Figure 2 regression also passed (20-frame child-process
run). It checks that accepted cell count falls during free fall, no B8 bricks
remain at frame 20, transported-volume drift stays below 0.01%, and at least 90%
of the initial negative-phi sample count remains represented.
