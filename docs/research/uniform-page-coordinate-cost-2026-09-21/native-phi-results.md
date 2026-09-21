# Native phi execution

The next stage after pressure was vertex phi transport and redistance: 9.60 ms
on the paged branch versus 2.95 ms on main in the preceding production captures.
Main's full-domain control was 5.31 ms, so both execution representation and
work coverage needed attention.

## Design and implementation

Complete rectangular residency is proved once from the accepted catalogue:
page count, dimensions, unique coordinates and bounds must all agree. Native
persistent fields then replace atlas backing for that execution domain. This
includes the input fields shared with phi advection, rather than packing or
copying them before every phi pass. Pressure was already native. Transport
record pages and their existing work lists are unchanged.

The dedicated phi pipelines are compiled at initialization. Advection retains
literal algorithmic loop bounds and uses native texture coordinates without
per-tap membership checks. Redistance retains its previous runtime loop form:
fully unrolling the closest-point iteration changed a near-degenerate injected
surface case by 0.00183 m on identical input, so that variant was rejected.
A narrower attempt that unrolled only interpolation failed the same frozen
input. Unrolling only the outer iteration passed but saved only half a timestamp
quantum in one capture, so the simpler original loop form was retained. Native
fixed-field accesses now bypass the adapter entirely in these two pipelines.

Iteration counts, convergence predicates, wall handling and interpolation
formulas are unchanged.

The phi work window has its own header, bind groups and GPU dispatch record.
It reuses main's support census and motion/stencil closure, without changing
pressure or transport scheduling. A complete census retains remote sources and
live insertions. Disabling the dust floor live selects a prebuilt full-domain
header; it never continues using a work window whose predicate no longer holds. Its three passes are included in whole-advance timing (charged
to the first existing stage); they are not free setup hidden outside the timer.
Both vertex dispatch dimensions and shader coordinates include the six-vertex
read margin. The inherited indirect count omitted that margin; it now matches
the direct-dispatch closure.

Native phi is consumed directly by correction and presentation. Correction's
scratch texture inherits its source's layout, avoiding incompatible atlas/native
copies. No new per-step field packing, copy or allocation was introduced.

## Controlled observations

Individual kernel timings use dedicated timestamp queries, with aggregate trace
instrumentation disabled to avoid overlapping query ownership. Earlier split
captures overwrote the aggregate pass timestamps: only their individual kernel
and whole-advance numbers are meaningful, not their other aggregate stage rows.

| Experiment, full domain | Advection | Redistance |
|---|---:|---:|
| Native phi only, existing loops and audit | 8.356 ms | 0.655 ms |
| Native phi only, literal loops and audit | 10.322 ms | 0.459 ms |
| Native phi only, existing loops, no per-tap audit | 6.947 ms | 0.393 ms |
| Native input fields, existing loops, no audit | 6.652 ms | 0.393 ms |
| Native input fields, literal loops, no audit | 5.341 ms | 0.131 ms |

The final implementation retains runtime loops for redistance for the numerical
reason above. These controls establish that neither phi storage alone nor blind
unrolling addresses the regression. Native inputs and a compact compiled shader
are needed together, followed by restoring the required work region.

## Scope

The current production catalogue is immutable and all-resident. A changed
catalogue must compile and publish a new execution plan before these native
kernels can consume it. This change does not implement velocity-driven physical
page allocation, retirement, or disconnected sparse execution blocks. It does
not claim that topology changes can be processed at zero cost.

## Final production comparison

Sequential exclusive Dawn/Metal runs, 44 advances at 1/30 s, four warmups,
production pressure demand, rendering excluded. Aggregate hardware tracing was
used for these three captures; no per-kernel query override was installed.

| Production arm | Phi stage median | Entire advance median | Reported allocation |
|---|---:|---:|---:|
| Former atlas on this branch, native pressure retained | 9.503 ms | 56.650 ms | 465.7 MB |
| Native fields and separate phi window | 3.342 ms | 48.472 ms | 295.6 MB |
| Main (`5d4d31f2`) | 2.949 ms | 44.968 ms | 295.5 MB |

Whole-advance throughput is 92.8% of main, meeting the overall 90% target for
this capture. The phi stage is 64.8% faster than the atlas arm, but remains
13.3% slower than main (88.2% of its throughput). **Stage parity is not claimed.**
The numerically sensitive redistance loop is retained; neither tolerance nor
iteration count was reduced to manufacture a timing result. These production
runs are separate trajectories, not a pure identical-input attribution.

The native-field consumer changes also reduce the extension front and hierarchy
costs. Their operators and traversal were not changed; they share the backing
fields. Phi support discovery adds approximately 0.5–0.6 ms to the first stage,
already included above. No claim is made that work selection has zero cost.

Raw timings, source hashes, work records and adapter metadata are preserved in
`native-phi-measurements.json`. The discarded compiler experiments are marked
in the capture keys; their timing does not describe final production code.

## Verification

- Nine focused CPU checks pass for coverage proofs, native field layout and
  pressure-page addressing.
- Frozen-input replay covers partial pages and the 192×96×32 long dam, twelve
  snapshots each, including live insertion on frame seven. Inputs are checked
  before each comparison. Advection allows 1/10000 cell of compiler rounding;
  redistance uses identical advected input and a one-micrometre absolute bound.
  Largest observed difference is 1.103 micrometres in advection.
- Window versus full-domain replay compares the complete dependent cell box.
  Both fixtures pass with zero observed difference; long-dam work contracts
  below the full domain. The test does not replace the windowed advected field
  with the full-domain result before redistance, so stale halo reads can fail it.
- Eight reported integration checks pass: domain/embedded wall separation,
  zero-gravity side walls, Figure 8 release, Figure 12 rebound closure, and
  MiniDam32 pressure prefixes/escalation.
- Type checking continues to report 15 existing errors outside the changed files.
- Canonical Sparse CM12 gate: **4/17 lanes pass**, 405.0 s. The same lane
  pass/fail statuses recur as the preceding pressure-change run; some failure
  modes switch between assertion and timeout. Mini64 reports 212.2 ms against
  its unchanged 110 ms ceiling. See `native-phi-sparse-regression.json`. This
  separate adaptive-volume gate remains failing; no lane or ceiling was weakened.

Reproduce the final production capture with `FRAMES=44 ASYNC_DEMAND=1` and the
existing `benchmark-uniform-long-dam-paging-dawn.ts`; add `PHI_STORAGE=paged` for
the former field atlas. `FULL_DOMAIN=1` disables the separate phi window for
controls. `PHI_TIMING=1` records individual kernels instead of aggregate stages.
Run `npm run test:dawn:uniform-phi` for the frozen-input and window oracles.

A final repeat with a reopened browser scene rendering at 55.9 FPS measured
76.04 ms despite identical cycle counts and final residual. It is retained as
`excluded-browser-contention`, not used for the A/B. After the scene tab closed,
the exclusive final-source repeat returned to 48.472 ms / 3.342 ms, as above.
The final frozen-input suite passed all four cases, including the live dust-floor
fallback, before this timing repeat.
