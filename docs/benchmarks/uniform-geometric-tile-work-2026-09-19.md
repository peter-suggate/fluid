# Uniform Geometric: first 4h work-map experiment

## Result

The experiment skips expensive sharpening work using a GPU-owned 4h map.
It leaves the h lattice, transport, velocity extension and pressure unchanged.
It is now the Uniform Geometric default through the `sharpeningWorkMap`
parameter (see "Productised" below); selecting Dense passes
`geometricTileWork: false` and restores the map-free dense schedule, with no
map construction, lookup, or altered shader overhead.

Dawn/Metal, current production method values: semi-Lagrangian, liquid capacity
balancing off, sharpening on, pressure tolerance 0.0001. Four fresh runs per
scene in dense/tiled/tiled/dense order; 30 frames each, first three excluded.
Compilation and final diagnostic readbacks are outside timing. Frame wall time
fences the queue. Sharpening uses hardware timestamps and includes classification.

| Scene | Dense sharpening | Tiled sharpening | Dense whole frame | Tiled whole frame |
| --- | ---: | ---: | ---: | ---: |
| mini64 dam | 5.112 ms | 1.540 ms | 84.748 ms | 81.467 ms |
| large-power dam | 1.868 ms | 0.524 ms | 47.623 ms | 43.992 ms |

Entries average the two run medians. Mini64 sharpening is 3.32× faster; whole
frames are 3.9% faster. The sparse dam's sharpening is 3.56× faster; frames are
7.6% faster. Individual full-frame run medians are noisy: mini64 dense
83.829–85.666 ms versus tiled 79.647–83.288 ms; sparse dam dense 45.280–49.966
ms versus tiled 43.503–44.480 ms. Treat whole-frame percentages as observations
from this machine, not portable promises. The benchmark asserts mini64's tiled
whole-frame median does not exceed its dense control; no ceiling was raised.

The earlier six-frame pilot and first 30-frame run are diagnostic only; the
adjacent JSON contains the final implementation's ABBA results and samples.

## Mechanism and correctness

One 4h record corresponds to a 4³ fine workgroup. A classification dispatch
evaluates the exact sharpening admission-band predicate for its cells. If no
cell can be admitted, that tile has zero surplus and need, independent of V.
All fluxes touching it must therefore be zero. Phi remains fixed across all
eight sharpening sweeps, so this classification is valid for the entire stage.

The experiment skips prepare/propose/limit on inactive tiles, treats inactive
neighbor fluxes as zero, and still copies V during commit. It does not discard
positive V outside phi, use stale stencil scratch, or require a velocity/CFL
estimate. The map reuses the third conditioning plane after balancing ends;
it adds no allocation and makes no GPU-memory saving. Pressure and non-sharpening
pipelines compile from the unchanged shader source.

Validation:

- Existing uniform numerical suite plus the focused scheduling test: 12 tests
  pass under Dawn, including the existing 90-frame mini32 impact check.
- Synthetic scheduling cases match dense output bit-for-bit across changing
  tile activity, partial boundary workgroups, three band widths, deliberately
  poisoned scratch, an entirely inactive map and conserved V far outside phi.
- Replaying identical real transported V/phi/target fields into dense and tiled
  sharpening matches bit-for-bit on every one of 30 mini64 and 30 sparse-dam
  frames. Phi is untouched; synthetic volume conservation also passes.
- Separate 30-frame full simulations are not bitwise matched. Mini64 repeats
  of the dense control itself diverge substantially, and tiled versus dense
  sparse-dam discrepancies can exceed dense-repeat discrepancies. Do not call
  this full-trajectory equivalence. The same-input replay is the decisive
  scheduling check; long-run numerical sensitivity remains an existing solver
  concern and the reason the dense schedule is retained as a selectable control.
- Repository typecheck still reports errors in unrelated sparse tests/probes;
  none remain in the files changed for this experiment. `git diff --check` passes.
- The separate canonical Sparse CM12 `mini64-performance` lane was also run:
  **failed**, 264.176 ms versus its unchanged 110 ms ceiling. That runner requires
  the `adaptive-volume` implementation and does not exercise this experimental
  uniform sharpening path. Its failure is recorded, not waived or relabelled
  as a pass; this work does not certify the canonical sparse regression gate.

## Evaluating extension, sharpening and conservative transport

Interpreting “volume extension” as the existing velocity extension into air,
the shortlist is well supported. A separate dense stage census, using observable
GPU timestamp markers with copy barriers, measured:

| Stage | mini64 dam | Large-power dam | Assessment |
| --- | ---: | ---: | --- |
| Velocity extension, including authority and hierarchy | 7.733 ms | 3.015 ms | Largest measured cost of these three; strong next target |
| Sharpening | 5.571 ms | 1.966 ms | Proven low-risk work reduction; 32 passes amortize one map build |
| Conservative transport, edge construction through gather | 4.260 ms | 1.376 ms | Promising, but donor/receiver dependency closure is mandatory |
| Phi advection + redistance | 0.590 ms | 0.197 ms | Smaller measured target in these configurations |
| Pressure | 32.440 ms | 12.714 ms | Untouched; limits total frame improvement |

Stage-census timing includes small marker overhead and belongs to a separate
run; do not subtract these medians from the ABBA wall medians or interpret
their sum as a complete GPU frame time. Invalid unobservable-marker captures
were discarded; the tool now rejects missing or reversed timestamps.

**Extension:** the current solver seeds and resolves a dense front, uses GPU
convergence-gated FIM dispatches, then restricts/prolongs a hierarchy and packs
the full transport shell. It already stops converged front iterations; the
remaining opportunity is spatial work, especially fine air fill that no
downstream characteristic consumes. A 4h map can select necessary fine velocity
tiles while preserving coarse hierarchy support. Keep interpolation ancestors,
neighbor support and actual forward/reverse sampling footprints. Do not simply
replace distant air velocity with zero. Its six rgba32 scratch fields also
make extension a substantial later memory target.

**Sharpening:** strongest immediate evidence. Its fixed phi predicate proves
an entire tile's work irrelevant without a predictive halo or graph traversal.
This experiment is stage-specific eligibility, not yet a universal occupancy
map. It can skip deep-liquid tiles as well as distant-air tiles because neither
is in the sharpening band.

**Conservative transport:** the 80-byte-per-cell stencil arena and repeated
full-grid edge scans make sparse work/storage attractive. However, the three
always-on geometric row/donor normalization rounds propagate influence through
zero-liquid donors. The map should first bound candidate receivers, then retain
the full backward dependency closure for all rounds. A wet-cell mask alone
changes the operator. Optional liquid balancing may increase the payoff, but
its 64-round benchmarks do not describe the current balancing-off default.

Recommended order: retain this sharpening experiment, investigate a work map
for extension's fine output/sampling support next, then tackle conservative
transport with an explicit dependency census. Pressure remains future work.

## Reproduction

Run sequentially, under the tools' repository-wide WebGPU lease:

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx --test --test-concurrency=1 tests/uniform-volume-dawn.test.ts tests/uniform-volume-tile-work-dawn.test.ts
node --import tsx tools/benchmark-uniform-geometric-tile-work-dawn.ts
node --import tsx tools/benchmark-uniform-geometric-tile-work-dawn.ts --stages
```

The second command is the experimental mini64 no-slowdown gate. `--quick`
performs a short pilot without enforcing that statistical performance gate.
The existing canonical `mini64-performance` lane exercises Sparse CM12, a
different method; it is not a substitute for the Uniform Geometric A/B above.

## Productised (2026-09-19)

The work map is no longer an experiment behind a solver flag:

- It is **default-on** for Uniform Geometric through the new `sharpeningWorkMap`
  method parameter (fine tier, runtime update), which maps to the solver's
  `geometricTileWork`. Because it is a runtime parameter it toggles on the
  attached solver rather than rebuilding it, so the map and the dense schedule
  can be compared live at the same simulated time. The dense schedule is
  retained as the `Dense` choice, not deleted.
- The tile-work shader code is first-class source in the method's shader module
  rather than a string patch applied to the dense source at compile time.
- The map moved behind an 8-word header at scratch word `2N+8`. It therefore no
  longer overwrites the liquid-balancing diagnostics at `2N+5`/`2N+6`, which the
  earlier third-conditioning-plane reuse described above did; the map still adds
  no allocation and makes no GPU-memory saving.
- The active-tile count is published on the solver info record
  (`uniformSharpenWorkMap`, `uniformSharpenTilesActive`,
  `uniformSharpenTilesTotal`) and surfaced on the SIM tab's pipeline panel: the
  Volume sharpening stage's chip reads `4h work map · <pct>% tiles` and its
  `Active tiles` readout reads `active / total (pct%)`, beside the schedule
  choice itself. The chip reads `dense finest lattice` on the dense control.

The medians above were measured before these changes, on the same tile schedule
this default now selects. Productisation moved where the map lives and how its
shader is assembled, not what it classifies; it has not been re-measured.
