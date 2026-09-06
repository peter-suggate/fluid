# Coarse surface reconstruction

Native Dawn reproduced the reported terraces in `minimal-power-dam-break-32`
with a whole-domain minimum cell width of four finest cells. The clearest
capture is step 8 (0.2667 s). Three publication choices contributed:

- A limited linear density polynomial was evaluated independently inside each
  coarse cell. Independent face values are conservative for refinement but do
  not define a continuous display scalar.
- A separate column-height override repeated each native column's height over
  its footprint. Interpolating density alone therefore left the lower terraces.
- Height-receipt validity switched between two different scalar fields. Even
  interpolated column receipts left seams at this switch.

Deep coarse interfaces now use one continuous scalar: integrate a compact
five-node vertical volume stencil at native cell centers, express it in world
length units, and interpolate those nodes. The floor continuation contract
remains separate. The conservative density prolongation used by topology
transfer is unchanged. The reconstruction does not demand finer cells.

The `before-volume` and `native-volume` captures have byte-identical density,
velocity, pressure and open-fraction arrays at steps 1, 8, 16 and 32: sixteen
physical field files. Earlier density-only and height-interpolation ablations
are retained separately under `artifacts/mini32-min4-display/`.

## Reset and mixed-rung waterlines

The half-pool reset also exposed fine dry apron samples published as uniform
`+4h`, beside coarse samples expressed in a different distance scale. Accepted
pool density was already exactly full below y=16 and empty above it. The
published surface nevertheless dipped by 0.388889 finest cells (19.44 mm).
Fine interface aprons now publish their actual scalar, and native volume nodes
use consistent length units. All 4,096 pool columns publish exactly 0.8 m;
adjacent scalar samples are ±0.024993896 m after binary16 quantization. Accepted
physical fields are byte-identical to the reset baseline.

The canonical calm-pool tests caught two further shortcuts. Calm partially
filled pages need interface-support evidence even without a sharp local
feature, or the first step treats them as uniform bulk. At a mixed-rung seam,
a finer display stencil must integrate a coarse partial cell's reconstructed
interface instead of copying its mean as a diffuse slab. A separate display
restriction does this for an enclosed monotone top-interface bracket; raw
finite-volume restriction remains authoritative for topology transfer. Both
`hydrostatic-adaptivity` and `min8-region-surface` pass their unchanged gates
following these fixes.

## Temporal continuity and limits

A zero-gravity, near-zero-time 16³ probe alternates enforced widths 2 and 1.
Its off-grid flat surface and a curved bowl isolate topology changes from wave
transport. Total accepted mass remains constant. The flat fixture initially
exposed a 0.006075-cell height shift with every accepted column volume unchanged:
limited conservative prolongation spreads the interface enough that the fine
five-node support truncates its volume. Fine publication now uses a seven-node
endpoint correction with weights summing to four and zero first moment. This
preserves the original affine-density scalar and exact planar height. Continuous
inner-bracket connection factors prevent a distant liquid body beyond an air
gap from erasing a detached sheet. The actual repeated flat-rung regression now
passes, as do 1,025 refined planar heights and detached-gap fixtures. The curved fixture still
changes its individual column volumes by up to 0.06 finest cells during
prolongation/restriction; the maximum published height shift after the fine bracket fix is
0.07204 finest cells (3.60 mm). This remaining non-flat representation change
is not claimed as eliminated. Root's later half-pool runtime trace found no solve rollback. The curved fixture
uses enforced rung changes, which bypass geometric admission, so it is not itself
evidence of a violated admitted-coarsening displacement tolerance. The admitted probe mode validates the geometric decision: a 0.01-cell budget
returns displacement failure bits 5, while 0.125 cells admits the geometric
proof on eligible leaves. Other eligibility/history constraints retained the
fine topology in that probe, so it is not claimed as an executed admitted
coarsening transition.

The focused fine/reset/actual-rung test run passes all three tests. The new
native-volume test checks 4,100 fractional planar heights over widths
1, 2, 4 and 8. The existing B8–B4–B8 continuity threshold is unchanged. A full
canonical run caught the calm-pool issues above and performance failures;
those correctness lanes have since passed individually. A final full gate is
still required after all shared changes. The accepted geometry proof now reads
committed presentation samples rather than resampling the same density field:
a matched mini64 stage-cost run reduced median publication from 30.34 to
17.24 ms. Overall mini64 timing remains above the canonical ceiling and is an
outstanding performance issue, not a waived gate.

A further native-phi workgroup cache experiment was rejected and removed.
Its physical fields were byte-identical, and maximum crossing change was only
0.000002861 finest cells from intermediate rounding, but it did not improve
matched mini64 publication timing (40.44 ms cached versus 33.69 ms original).
The retained implementation uses the original native-node evaluation.

## Reproduction

Do not run alongside another Dawn process or the Fluid browser:

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal \
  DISPLAY_OUTPUT=artifacts/mini32-min4-display/native-volume \
  node --import tsx tools/probe-mini32-min4-display-dawn.ts
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal \
  RESET_OUTPUT=artifacts/pool-reset-surface/native-volume \
  node --import tsx tools/probe-pool-reset-surface-dawn.ts
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal \
  node --import tsx --test --test-concurrency=1 tests/sparse-cm12-native-volume-surface-dawn.test.ts \
  tests/sparse-cm12-pool-reset-surface-dawn.test.ts \
  tests/sparse-cm12-surface-rung-continuity-dawn.test.ts
```

`tools/analyze-mini32-min4-display.py` produces the density slices and 3D
reconstruction comparison under `artifacts/mini32-min4-display/`.
