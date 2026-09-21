# Long-dam multigrid review after MiniDam32 parity

Measurements target Uniform Geometric on `sparse-cm12-long-dam-break`, the
192×96×32 authored domain. Browser was at the scene library, no simulation
running. Dawn measurements ran sequentially under the exclusive GPU lease.
Branch revision: `91de0807`; main: `5d4d31f2`. Each capture has 44 advances,
four warmups, asynchronous cycle-demand feedback, and hardware stage timings.
The UI “Multigrid cycles” row sums Full-Cycles and V-Cycles; setup, recovery/
final residual, projection and rendering are outside this row.

Production defaults: main 8.060928 ms in cycles / 46.004188 ms per advance;
branch 29.753344 ms in cycles / 82.275271 ms per advance. Main is 3.69× faster
in the cycles row. These are median samples, not the exact browser trajectory.

The branch encodes one cycle on 43/44 frames. Frame 20 misses tolerance and
frame 21 expands its budget to three; only one cycle executes on frame 21.
Main normally encodes two, executes two on frame 20, and meets tolerance there.
The lagged minimal-budget policy therefore changes that transient's numerical
trajectory; the default captures must not be described as bit-identical.

The matched control forces one Full-Cycle, zero V-Cycles, full-domain work,
6/6 sweeps and tolerance 10 on both revisions. Hierarchy dimensions match:

| Configuration | Main cycles | Branch cycles | Ratio |
|---|---:|---:|---:|
| Production defaults | 8.061 ms | 29.753 ms | 3.69× |
| Matched one-cycle/full-domain | 6.029 ms | 30.966 ms | 5.14× |

The controlled pair matches configured work, not replayed identical fields.
Final fine residuals differ (main 3.354240, branch 2.718385); both report one
coarse iteration and no rejection/recovery at the final frame. Therefore the
5.14× ratio is not a measured attribution solely to address arithmetic. The
next pressure-only experiment must replay identical topology/RHS inputs.

Raw captures, source hashes, cycle evidence and work censuses are retained in
`long-dam-pressure-review-measurements.json`. Reproduce with the existing
`benchmark-uniform-long-dam-paging-dawn.ts`, `FRAMES=44 ASYNC_DEMAND=1`, and
`ONE_CYCLE=1 FULL_DOMAIN=1` for the controlled pair.

## Established added work

- `uniformPressurePagedShader` wraps every pressure/topology/RHS/minimum/residual
  load and store in `mgPageAddress`. Logical dimensions arrive through uniform
  metadata. Repeated stencil taps repeat bounds tests, slot linearization and
  atlas row divisions/remainders. This is arithmetic mapping, not a hash lookup.
- `uniformPressurePageWorkgroups` rounds every level into full 16³ physical
  tiles. The 194×98×34 finest level launches 17,472 workgroups rather than 11,025;
  14×8×4 launches 64 rather than 8; 5×5×4 launches 64 rather than 4. This concerns
  ordinary hierarchy passes, not the separate one-workgroup coarsest solver.
- Finest pressure storage is 128×128×80 = 1,310,720 texels rather than the logical
  646,408. That larger footprint and relocated seams can affect cache locality;
  cache-miss counters have not been measured, so no fraction is attributed to it.
- Projection already consumes a dense pressure publication copied from the
  atlas after the solve. That copy is outside the cycles row, but belongs in
  any end-to-end cost accounting.

The MiniDam32 native-coordinate specialization applies only to one domain page.
The long dam has 18 domain pages and still uses generic paged pressure storage.

## Proposed next implementation and controlled attribution

Use a contiguous native multigrid execution workspace for the accepted solve
region. Page ownership remains responsible for residency and growth; pressure
scratch does not need to reproduce the persistent field atlas. Compile the
workspace extents, per-level layouts, dispatches and binding groups when the
accepted topology changes. The repeated cycles use native loads/stores and
ordinary stencils. This removes address translation, restores native locality,
and eliminates tile-padding launches. Projection can consume the native result
directly, removing the existing paged-pressure publication copy as well.

For the currently all-resident long dam, this is a smaller workspace with the
same logical cells. Future domain growth must build/remap the new workspace
before publication; it is not a proposal to reserve the entire future world.
A disconnected sparse domain may require multiple execution blocks and seam
transfers. Those have real costs and must not be called zero-cost addressing.

Replay identical pressure topology/RHS inputs first. Then implement and measure
controlled steps: exact logical launches while retaining
atlas addressing, then native pressure workspace at those same logical extents.
Keep tolerance, sweeps, cycle count, stencil arithmetic and acceptance/recovery
checks fixed. Compare fields as well as stage and full-frame time. Separately
restore/verify main's active-work selection; matching hierarchy capacity is not
proof of matching visited cells.

The immediate target is the measured main single-cycle, full-domain stage;
final acceptance must use production defaults and count setup, publication,
projection, allocation and topology-change costs rather than move work outside
the cycles row. No simulation implementation was changed in this review.
