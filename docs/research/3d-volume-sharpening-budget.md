# 3D sharpening: spend the eligible face budget

Review checkpoint, 2026-09-15. Prior work, including the 2D adaptivity update,
is committed as `7b658835`. This follow-up is intentionally left for review.

## Follow-up: immediate inward relay and volume-to-phi feedback

The budget-only result was insufficient at 5.3 s in `water-box-dam-break`.
The follow-up keeps fixed local work:

- Cache cell-centre metric phi during sharpening preparation. Air-side surplus
  can cross an immediate face toward decreasing phi when the face sample lies
  between endpoint values. A sampled air crest still rejects the transfer.
- Pure-air cells can temporarily receive up to their remaining capacity, then
  relay inward on the next frame. An immediate phi deficit is no longer needed
  at every hop. Relay capacity cannot pull liquid outward from the interior.
- Retain common donor/receiver limiters, symmetric face transfers and the
  existing no-new-excess bound on each sharpening operation.
- Reduce eligible-cell `V - phi target` and an approximate interface area in
  the preparation pass: a 64-cell workgroup reduction followed by two float
  CAS additions per group. Same-phase nonmetric samples contribute full/empty
  target estimates, but never provide directions or interface area. Cut cells
  and unresolved mixed cells remain excluded.
- Apply `clamp(0.25 * residual / area, -0.1, 0.1)` in finest-cell units to phi.
  The offset is uniform within one finest spacing of the contour, tapers to
  zero by two spacings, and leaves phase-only vertices untouched. Existing
  descending-width constraint projection follows the update. This is one
  damped correction, not an iterative volume solve or a full redistance.

The four volume passes remain. Phi feedback adds one vertex pass and a
constraint-projection sweep (one setup dispatch, then two dispatches per
supported width level). No new buffers or arena planes are allocated: two
unused control words store the reduction and the dead low-volume scratch
caches centre phi. Face proposals do extra endpoint reads; preparation does
one extra centre sample. No longer-range search or additional relocation
round is introduced.

The offset uses a global sum over eligible cells. It does **not** conserve
individual disconnected components' phi-enclosed volumes. The midpoint gate
is a local anti-bridging heuristic, not component labeling. Target integration
and interface area are approximate. These limitations matter for separate
small drops, thin gaps and cut solids.

### Full dam-break result at 5.3 s

Compared with the reviewed budget-only version, both starting from reset:

| Measure | Budget-only | Inward relay + phi feedback |
| --- | ---: | ---: |
| Air-side volume, finest-cell units | 717.003 | 376.579 |
| Volume with published phi > one finest spacing | 305.731 | 98.703 |
| Total conserved volume | 1680.0043 | 1680.0042 |
| Negative-phi finest centre samples | 1374 | 1662 |
| Maximum fill | 1.072 | 1.276 |
| Total excess above capacity | 0.997 | 15.860 |
| Accepted solver cells | 3456 | 3456 |
| Mean completed-frame wall time, ms | 57.917 | 50.868 |

Air-side volume decreases 47.5%, and the phi>h measure decreases 67.7%.
The contour now encloses approximately the conserved volume, measured by
finest centre classification. This is not exact surface integration.
Mass drift is about 0.00025%, with no transport fault or invalid/nonfinite
cells. Wall times are single runs with evolving states, not an isolated
shader-cost comparison.

A visible band remains: 22.4% of total volume is still in phi-positive centre
samples. Aggregate excess rises to about 0.94% of total volume. Although each
sharpening step is bounded, the corrected state changes later transport and
pressure. This is a review checkpoint, not a claim of complete 2D parity.

Receipts: `artifacts/level-set-volume/sharpening-dam-5p3s-fields.json` and
`artifacts/level-set-volume/sharpening-dam-feedback-inward-fields.json`.

### Follow-up verification

- CPU reference/topology tests: 14 passed.
- GPU kernel tests: eight transfer fixtures, including one-hop relay and
  rejecting outward relay; six phi-offset fixtures, including both signs,
  the cap, zero area, disabled strength and phase-only support preservation.
- New production 159-frame dam-break regression passed: conserved mass,
  finite fields, air-side fraction <25%, phi>h fraction <7.5%, and enclosed
  centre-count volume within 5% of conserved volume.
- The existing zero-motion test fails on the same vertex-count change
  (2601 to 3645) both with this change and with committed `7b658835` solver
  sources. A separate current-state probe reports zero correction residual,
  zero trace displacement and zero cells moved by sharpening.
- Full canonical Dawn gate: eight lanes passed, the same nine lanes failed as
  before (six timeouts and three assertions), in 425.4 s. No ceilings or
  assertions were weakened. Receipt:
  `artifacts/level-set-volume/sharpening-feedback-dawn-gate.json`.
- Repository type checking still reports existing errors outside the changed
  files; the changed files have no reported type errors in the final run.

## Initial budget-only change and cost

The old 3D sharpening divided each cell's surplus/deficit by its total face
degree, even when only one face could transfer liquid toward the phi target.
A six-face cell could therefore spend only one sixth of its available budget.

The new proposal uses the opposing cells' full surplus and deficit. The gather
pass totals actual outgoing/incoming proposals and computes a limiter for each
cell. The commit pass applies the minimum donor/receiver limiter to each face;
both endpoints use the same transfer. This preserves conservation and bounds
each cell's total transfer by its prepared budget.

The schedule remains four passes: prepare, propose, gather, commit. Existing
budget planes become limiter planes after proposal. There are no new buffers,
iterations or phi samples. Commit now walks the existing cell-face adjacency,
adding one bounded adjacency walk and endpoint limiter reads.

This brings 3D closer to 2D's practice of spending eligible relocation budgets,
but retains the existing 3D face-local operator and liquid-connection gate.
It does not copy 2D's component labeling, multi-cell searches or four rounds of
redistribution. Phi itself is unchanged by sharpening.

## Slab comparison

Production `water-box-dam-break-slab`, ten steps at dt=1/30 with the default
Dawn solver configuration, before/after this change:

| Step-10 measure | Before | After |
| --- | ---: | ---: |
| Volume on phi-positive side, finest-cell volume units | 158.137 | 118.727 |
| Volume more than one finest spacing into air | 75.328 | 49.246 |
| Fractional finest samples | 2,184 | 1,960 |
| Accepted solver cells | 1,224 | 1,224 |
| Total volume, finest-cell volume units | 1,344.000249 | 1,344.000211 |
| Maximum fill | 2.0391 | 1.9749 |
| Total excess above capacity, finest-cell volume units | 36.597 | 39.224 |
| Mean completed-frame wall time, ms | 35.909 | 35.840 |

Air-side volume drops 24.9%; deeper air-side volume drops 34.6%. These are
volume/contour disagreement measures, not geometric error against a known
solution. The ten-frame wall times include CPU scheduling and GPU completion;
they are a short sanity check, not an isolated GPU performance benchmark.
Both runs have zero transport faults, invalid cells and nonfinite cells.

Sharpening creates no new excess within its individual operation, but it
changes subsequent dynamics. At step ten aggregate excess increases by 7.2%
even though maximum fill decreases. Existing over-capacity fluid remains an
unresolved limitation. This is a bounded halo improvement, not full 2D parity.

Reproduce with:

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/probe-sparse-geometric-sharpening-dawn.ts --output=artifacts/level-set-volume/sharpening-slab.json
```

The ignored before/after receipts are
`artifacts/level-set-volume/sharpening-slab-before.json` and
`artifacts/level-set-volume/sharpening-slab-budget.json`.

## Validation

- CPU reference and compiled-topology tests: 14 passed. Includes one eligible
  face, competing donors/receivers and face orientation/order independence.
- Production GPU proposal/gather/commit kernels: passed four synthetic
  fixtures, checking the CPU oracle, conservation and positive-phi gap isolation.
  Geometry and topology accessors are fixture substitutes; the production slab
  probe exercises the full solver.
- Full canonical Dawn regression gate: 8 lanes passed and 9 failed in 428.2 s
  (within the 480 s suite budget), run without concurrent browser GPU work,
  another Dawn process, compilation or other test suites. The same nine lanes
  failed in the pre-sharpening run: symmetric-expansion, topology-page-budget,
  mini32-correctness, min8-region-surface, mini32-performance,
  mini64-performance, long-dam-far-wall, tall-cells-hills-far-wall and
  outside-tank-symmetric-collapse. Six failures were timeouts and three were
  assertions. No thresholds were changed. This is not a clean regression gate
  or proof of unchanged performance. Receipt:
  `artifacts/level-set-volume/sharpening-dawn-gate.json`.
