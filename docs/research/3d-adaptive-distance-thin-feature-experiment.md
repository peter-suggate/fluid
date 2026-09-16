# 3D adaptive-distance return and thin-feature veto

## Controls and cost

The volume transport stage now exposes **Adaptive distance sweeps** (default 8)
and **Far-volume return passes** (default 4), both integers in [0,16]. These are
runtime controls. Setting either to zero disables the auxiliary return while
retaining existing local sharpening. Turning sharpening off also disables it.

A distance sweep comprises two ping-pong neighbour relaxations. A return pass
comprises prepare, propose, gather and commit, and moves volume at most one
accepted-cell neighbour. The default adds 34 dispatches (reset + seed + 16
relaxations + 16 budget/transfer stages). Disabled return retains only its receipt
reset. Thin-feature checks add work inside existing census/proof passes, with no
new dispatch. Neighbouring phi stencils are cached during thin-feature traces, and redundant
axis-aligned traces are skipped. No extra persistent GPU allocation is needed: return reuses the
transport edge arena after its final consumer.

## Auxiliary distance

Local sharpening runs first. Near-interface liquid cells seed a signed graph
distance on the accepted adaptive solver cells. Open, fully fluid faces connect
cells; extension is restricted to certified pure air, bounded by eight finest
cell widths. This is a graph-distance approximation, not an exact Euclidean SDF,
and does not change the public phi field or claim new metric support for it.

Each seed owns a surface patch. Equal-distance collisions between different
owners are marked ambiguous. Transfer requires matching, unambiguous ownership
and decreasing distance. This deliberately errs toward withholding transfer:
owners are patches, not connected-component labels, so even patches of the same
body can block one another. Missing support, partial solid cells, and closed
faces cannot provide a path. Fewer sweeps leave distant cells unavailable.

Only air surplus is donated. Liquid patches accept their actual phi-derived
volume deficit; air cells relay inward on later passes. Existing shared donor
and receiver budgets bound all competing faces and conserve volume. Auxiliary
work is active only when seeds identify outlying volume (non-metric air, or air
beyond twice the local minimum cell width). No new interface component is made.
Receipts report seeds, far donors, proposed faces, participating cell-rounds,
maximum reached distance and ambiguous cell-rounds.

## Thin features

The accepted trilinear phi is inspected at all twelve crossing edges. Traces in
both normal directions and along the crossing edge look for a second crossing
within the configured physical thin-feature width, with opposing surface normals.
A short tangential chord through a broad curved surface is not sufficient.
Edge traces catch small
corner droplets where a one-sided normal at a lattice crease is unreliable.
This detects represented liquid sheets/filaments and thin air gaps independently
of centre density. It cannot recover a feature absent from the accepted phi.

Detected features retain the existing thin-feature reason and their accepted
resolution. Thinness alone does not request the finest rung; deformation and
transport demand may still request refinement.
Surface coarsening proofs independently inspect accepted geometry; unresolved
support within an accepted cell withholds a proof. A trace leaving represented
support is not itself evidence of a thin feature; the independent candidate
restriction proof still checks the samples it needs. Final GPU candidate validation prevents thin-feature
coarsening or retirement after region limits and grading. Host macro-generation
planning also checks final physical widths, preventing forced region merges
from bypassing the veto. These vetoes take precedence over coarseness requests.

## Validation and initial measurements

Focused Dawn tests cover far return, mixed cell widths, blocked faces, ambiguous
patches, near-only no-op, configurable sweep/pass budgets, oblique sheets, air
gaps, corner droplets and unavailable support. The existing local sharpening
budget/air-gap test is retained. CPU tests cover control bounds and macro-merge
vetoes.

A 10-frame `water-box-dam-break-slab` run compared against the pre-change receipt:

| Metric at frame 10 | Before | After (8 sweeps, 4 passes) |
|---|---:|---:|
| Volume beyond phi > one fine cell | 70.3944 | 68.9519 |
| Volume on positive phi side | 137.4022 | 135.7393 |
| Total accepted volume | 1344.000085 | 1344.000251 |
| Negative-phi fine samples | 1326 | 1326 |
| Median frame time (ms) | 38.32 | 62.24 |

These are short experimental runs, not a broad quality or performance claim.
The combined detector and return cost is material; lower sweep/pass settings
allow experimentation. The production run completed without WebGPU validation
errors. Receipts: `artifacts/level-set-volume/3d-adaptive-distance-baseline.json`
and `3d-adaptive-distance-configurable.json`.

Reproduce with `WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import
 tsx tools/probe-sparse-geometric-sharpening-dawn.ts --steps=10
 --distance-sweeps=8 --return-passes=4 --output=<receipt.json>` (one shell line).

### Regression status

- Focused Dawn: 3/3 tests passed, including the configurable transfer fixtures.
- Controls, generation policy and redistance CPU tests: 14/14 passed.
- The full canonical Dawn gate completed in 369.5 seconds: 6/17 passed. The six
  timeouts and four assertion failures were also present in the earlier 2D-era
  run. Mini64 performance additionally missed its 110 ms ceiling at 112.2 ms.
- After caching neighbouring phi stencils, the focused tests were rerun and the
  mini64 performance lane passed at 102.6 ms, with the same 110 ms ceiling.
  The complete gate was not rerun after this optimization. Full-gate and lane
  receipts are `3d-adaptive-distance-regression.json` and
  `3d-adaptive-distance-mini64-performance.json` under the artifact directory.
- The surface-proof source-contract test still fails its B4-to-B8 reconstruction
  assertion; its regex also fails against the staged source preceding this work.
- Repository typechecking remains failing in unrelated existing files; it reports
  no errors in the files added or changed by this implementation.

The broad regression gate is not green; these results establish the focused
mechanics and short-run conservation, not full-scene readiness.
