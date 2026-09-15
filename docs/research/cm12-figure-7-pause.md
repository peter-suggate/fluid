# Figure 7: runtime topology recovery causes long pauses

## Conclusion

Dawn reproduces the first pause when attempting frame 7, after six completed
frames (simulation time 0.200 s). A geometric face coverage check rejects a
host/dynamic-page seam. The solver treats this as a request to rebuild the
resident, rather than a terminal structural failure. Full resource preparation
and pipeline compilation then block advancement for tens of seconds.

The required contract is fail closed: runtime execution selects already compiled
topology. Missing connectivity must stop the step with a diagnostic, preserving
the last accepted state. It must not silently generate topology, rebuild the
resident, or repair connectivity to continue. The initial investigation made no solver
changes; the implementation below subsequently enforces this failure contract.

## Reproduction and timing

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
FLUID_STAGE_PROBE_DEBUG=1 \
node --import tsx tools/probe-sparse-cm12-stage-cost.ts \
  --scene=cm12-figure-7 --production-defaults=1 \
  --warmup=0 --frames=20 --final-qa=0 \
  --enforce-pressure-receipts=0 --quiet=1 \
  --out=artifacts/level-set-volume/figure7-pause-stage-cost.json
```

This measures production-default B8/P8 at dt=1/30. Pressure receipt enforcement
and final QA are disabled for this timing probe; this is not a correctness gate.

| Attempted frame | Wall time |
| --- | ---: |
| 7 | 24.10 s |
| 9 | 29.40 s |
| 11 | 31.22 s |
| 15 | 30.86 s |
| 17 | 31.49 s |
| 19 | 32.82 s |

Median GPU frame time was 83.56 ms. During the first pause, diagnostics report
`framePending: false`, `topologyGenerationPending: true`, and 21 requested
leaves. This is a host-side generation wait, not one unusually long GPU frame.

A second, instrumented run with `--preparation-worker=1` reproduces frame 7 at
33.49 s wall time. Replacement preparation takes 33.32 s: 9.65 s preparing the
worker recipe and approximately 23.68 s realizing resources and pipelines.
There are 260 compute-pipeline requests/completions during replacement. Planning
and transfer captures take only 5.23 ms and 3.26 ms respectively.

The replacement changes hardcoded arena offsets and capacities in shader source.
For example, sharpening shader `LSV_CELL_CAPACITY` changes from 237440 to 280032.
The full-source compilation key consequently changes, defeating reuse of the
previous resident's pipelines. The 23.68 s includes resource realization and is
not a separately measured compiler-only duration.

## First failing face

Diagnostic instrumentation records the lowest failing cell/axis at certification
and stops before automatic replacement:

- Host brick 214, coordinate `[9,7,8]`, scheduled B4 → B8.
- Candidate cell 165060, center `[72.5,56.5,68.5]`, unit widths.
- Missing negative Y face at `[72.5,56,68.5]`.
- Neighbour leaf 422, coordinate `[9,6,8]`, dynamic page 2, scheduled activation.
- Host incidence still contains its one-sided exterior row 505981.
- Matching dynamic row 985436 is also still one-sided: its only term is dynamic
  cell 247036 with coefficient -1; it does not yet reference host cell 165060.

The dispatch and selection logic explain this state:

1. Frontier pages are synthesized, then resolution and activation are planned.
2. `certifyGeometricTopologyFaces` scans the host's existing incidence list.
3. `shadowRowScheduled` suppresses its exterior row because
   `hostExteriorRowSupersededAt` sees a scheduled dynamic neighbour and a B8 host.
   That dynamic-neighbour branch does not prove replacement connectivity exists.
4. `connectSparseWorldFrontierPages` would install the two-sided seam and patch
   host incidence, but runs later, in the authorized publication tail. It cannot
   supply the candidate connectivity required by the earlier certification.

Thus certification combines scheduled membership with connectivity that has not
yet been installed. The recorded failure is not evidence that the GPU needs more
time or a larger distance search. Merely reordering mutation is not established
as a safe solution: the code deliberately protects single-buffered accepted
incidence until authorized publication. Candidate certification needs a complete,
consistent compiled authority before publication.

All 21 initial requests carry `ACTIVITY_GEOMETRIC_FACE_BACKING`; 16 already have
compiled candidate slots. The predicted solver-cell count is 70556, well below
the phi arena's 80% trigger (189952 of 237440), ruling out phi capacity as the
first trigger.

## Why rejection becomes a pause

`requestGeometricTopologyFaceBacking` sets support word 33 and generation request
bits. `sealGeometricTopologyFaces` cancels candidate scheduling, retaining the
accepted image, but does not record a terminal simulation failure.
`captureSimulationFailure` passes the backing flag to the solver, which sets
`geometricBackingPending`. `advanceTo` then schedules topology generation and
returns without advancing until replacement preparation finishes.

Retaining the accepted image is appropriate; automatically rebuilding and
resuming violates the required fail-closed performance contract. The defect in
candidate connectivity and the recovery policy are separate issues.

## Evidence files

Ignored diagnostic artifacts under `artifacts/level-set-volume/`:

- `figure7-pause-stage-cost.json`: uninstrumented 20-frame reproduction.
- `figure7-pause-worker-stage-cost.json`: worker-path timing reproduction.
- `figure7-pause-detail-events.json`: replacement/pipeline timing events.
- `figure7-first-missing-face.json`: cell, incidence, dynamic row, and plan capture.
- `figure7-face-diagnostic-stop.json`: intentional stop before replacement.

Instrumentation was applied in memory by temporary scripts, not production
source. The last run intentionally exits with a diagnostic-stop error; it is
not a passing correctness test.

## Fail-closed implementation

Missing face coverage now records `MISSING_COMPILED_TOPOLOGY_FACE` in the sticky
GPU failure receipt, with cell ownership and integer operands for brick, axis,
accepted resolution and scheduled resolution. Certification still rejects the
incomplete candidate; it does not assume that a later connection pass will repair
it. Both ordinary and zero-time topology paths copy the failure gate before the
publication tail. The host backing-request callback, pending flag and retry loop
have been removed.

The production-default Figure 7 reproduction now halts during frame 6, before
attempting frame 7 or constructing a replacement. This run's first reporter was
cell 186820, brick 275, Y axis, B2 → B8, generation 7. The first reporter can vary
with GPU workgroup scheduling; the earlier diagnostic deliberately selected the
lowest failing cell instead. The log is `/tmp/fluid-figure7-fail-closed.log`.

The fail-closed change alone enforced rejection without resolving the candidate
seam. The subsequent root-cause fix is described below.

## Frame-6 candidate seam fix

`compiledHostDynamicSeam` now binds the immutable host exterior incidence to the
already prepared B8 page template's reserved row and term slots. It checks the
scheduled host rung, unit geometry, selected authored row, active neighbour,
page ownership and completion receipt, capacity, row address and face geometry.
Certification consumes that descriptor instead of requiring canonical incidence
to have been mutated ahead of publication. Both publication orientations consume
the same descriptor when selecting the host incidence slot. Missing or mismatched
backing still triggers the terminal failure above.

No new GPU buffers or dispatches are added. The descriptor binds existing slots;
it does not generate a resident or install an early mutation in the accepted
graph. Publication's former axis/sign scan could match another candidate row in
the host incidence catalogue; matching the exact compiled exterior slot also
removes that ambiguity.

The production-default Figure 7 run now completes frame 6 at roughly 102–103 ms
GPU time, without resident replacement. Focused Dawn coverage exercises all six
orientations at signed coordinates and rejects incomplete receipts, inactive
neighbours, wrong addresses, mismatched geometry and unsupported host rungs.

A run through frame 7 exposes a separate `lsvAdvectPhi` missing-sample fault
(mask 64). That guard remains enabled. Passing the original seam transition does
not yet imply that the scene can run indefinitely. The broad CPU catalogue QA
was stopped after the successful sixth frame; focused seam readback is used for
this change instead.

The focused frame-6 readback confirms cell 165060 uses accepted row 985436
with terms `(247036, -1)` and `(165060, +1)`; cell 186820 uses accepted row
992312 with terms `(248828, -1)` and `(186820, +1)`. Each has exactly one
accepted negative-Y seam. Capture:
`artifacts/level-set-volume/figure7-frame6-published-seams.json`.

Final targeted checks: 3 Dawn tests and 10 failure/UI unit tests pass.
Type-checking still reports errors outside the modified files. The full Dawn
suite was not rerun for the seam fix, per the user's direction.

## Continuation to 3 seconds

The next phi failure was caused by rejecting non-metric boundary support
unconditionally in `lsvExtendFromOwner`. Deep support is not a usable distance
slope, but it is a signed clearance certificate. The boundary sample can cover
an exterior departure even when the original vertex cannot cover the entire
characteristic. The extension now subtracts the full boundary-to-departure
Euclidean distance and accepts only strictly positive remaining clearance. It
retains the phase sign and never marks this extension metric. The advection band
promotion applies only to directly sampled departures; it cannot promote an
extrapolated clearance into a contour distance. Insufficient, absent or nonfinite
support still fails closed.

After phi advection passed, frame 7 exposed host/dynamic retirement and frame 20
exposed dynamic/dynamic retirement. These were the converse of the activation
mismatch: accepted incidence overrides still named a retiring seam while the
scheduled image selected the prepared exterior. Host certification now selects
its immutable exterior when appropriate. Dynamic boundary certification reads
the fixed unit-cell page template, verifies backing, and resolves the scheduled
neighbour binding. No candidate proof mutates the accepted incidence graph.

Final production-default Dawn reproduction:

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
FLUID_STAGE_PROBE_DEBUG=1 \
node --import tsx tools/probe-sparse-cm12-stage-cost.ts \
  --scene=cm12-figure-7 --production-defaults=1 \
  --warmup=0 --frames=90 --final-qa=0 \
  --enforce-pressure-receipts=0 --quiet=1 \
  --out=artifacts/level-set-volume/figure7-3s-final.json
```

Result: 90 frames / 3.000 s, diagnostic passed, zero WebGPU validation errors,
and zero resident replacements across all 90 authority receipts. Median GPU
frame time was 227.8 ms; median wall frame was 236.1 ms and maximum wall frame
809.1 ms. This establishes successful advancement without the 24–33 second
replacement pauses, not a real-time performance or full numerical-accuracy gate.

Four focused Dawn tests pass, including all six host seam orientations, rejection
of missing backing, sticky failure publication gates, and both signs of deep
clearance. Clearance tests also reject insufficient/absent support and verify
that advection preserves phase-only classification even when the source vertex
was metric. The 11 phi core unit tests pass. Type-checking reports errors outside
the changed files; the full regression suite was not rerun, as requested.
