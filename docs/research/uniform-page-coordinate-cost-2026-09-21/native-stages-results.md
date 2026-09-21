# Native execution across the remaining stages

Complete rectangular residency is proved before compilation. Root cell stages
now use logical grid coordinates and exact ceil(dimensions/4) launches, with
native texture instructions instead of per-tap page-membership audits and atlas
address wrappers. The same specialization covers velocity extension, its
hierarchy (including integer nearest-source fields), and surface-volume
correction. Numerical loop bounds return to their original literal form.
The dedicated phi redistance variant retains runtime loop bounds for the
previously measured near-degenerate rounding sensitivity.

The immutable accepted catalogue is published once at initialization. Its
unchanged metadata is no longer rebuilt every advance. No pressure cycles,
pressure cells, transport stencil entries, or convergence tolerances were
removed to obtain these savings.

## Correctness

`tests/uniform-native-stages-dawn.test.ts` compares native execution to the
page-domain-free solver on partial pages and the long dam, including insertion
at step 7. Each advance starts with identical canonical fields to isolate
code generation from accumulated floating-point trajectory differences.
Redistance is disabled in this test: its independent same-input/window oracle
is `tests/uniform-phi-layout-dawn.test.ts`.

Across 12 advances per fixture, volume and phi matched exactly. Maximum
normalized errors were 2.295e-6 for pressure and 3.458e-6 for velocity. The
normalization denominator is max(1, abs(reference)), and the assertion limit
is 1e-5. Volume and phi have exact-equality assertions.

## Timing interpretation

Preliminary 44-advance queue-fenced medians were 41.92 ms after removing
addressing/audits, then 37.42 ms after restoring literal numerical loops.
These runs did not produce usable stage partitions and are not used as
stage-by-stage evidence.

The cause was verified with raw timestamps: surface publication, which does
not depend on projected velocity, completed before pressure projection.
Per-tap audit atomics had accidentally imposed a shared resource dependency.
Removing that dependency permits useful GPU overlap. The previous monotonically
ordered boundary timer correctly rejected the now nonmonotonic sequence.

A per-pass completion-frontier experiment was rejected: its query/blit
instrumentation inflated the 1e-3-floor median from 39.74 ms to 78.97 ms.
It is not enabled in production. The original lightweight boundary timer is
retained, with projection and its overlapping surface publication measured
as one combined interval ending on diagnostics. With rigid coupling enabled,
publication joins that final stage instead. This introduces no artificial
numerical dependency and the panel recognizes the combined labels.
End-to-end comparisons also include instrumentation-off runs.

## Allocation remains separate

This change removes numerical execution overhead. Production fields still
cover the authored domain. It does not enable aggressive allocation and must
not be described as doing so. A high volume dust floor alone does not prove a
page is retireable: independent phi-liquid/interface support still participates
in pressure. Allocation must preserve that support, cover source/velocity and
stencil reach, and atomically replace generation-owned fields and consumer
bindings before releasing old resources. The existing buffer-based page pool
is not the production texture authority.

## Selected dust floor

The user selected 1e-3 cell volumes. The WebGPU defaults and generated native
Rust contract now agree on that value; zero and explicit smaller overrides
remain available. At 44 long-dam advances, the native branch measured:

| Floor | Uninstrumented median advance | Net conservative volume loss | Final transport workgroups |
|---|---:|---:|---:|
| 1e-6 | 38.17 ms | 0.000248% | 2,912 |
| 1e-3 | 39.74 ms | 0.049031% | 4,480 |

A higher floor changes the trajectory as well as removing residue. It did not
reduce the final active work in this fixture; this is not evidence of aggressive
residency, nor a speedup attributable to the dust-floor change.

The canonical Sparse CM12 gate completed in 409.3 seconds with 4/17 lanes
passing, exactly the previous run's lane pass/fail statuses. No timing ceiling
or assertion was relaxed. Type checking still reports the same 15 errors
outside changed implementation files. The expanded CPU suite has three
pre-existing label-expectation failures (`dense finest lattice` versus
`page domain`) in `uniform-volume-initial.test.ts`; the coordinate/storage
checks pass.


At the selected 1e-3 floor, matched uninstrumented main measured 46.38 ms
against the branch's 39.74 ms (16.7% higher branch throughput). Both used 44
advances, four warmup frames, balanced defaults, and exclusive Dawn/Metal.
The main checkout is `5d4d31f2`; only the benchmark harness was copied there.

With the previous 1e-6 floor held fixed, lightweight branch stage medians were:
velocity advection 1.05 ms, extension front 0.98 ms, extension hierarchy 0.85 ms,
phi 3.05 ms, geometric coupling 4.33 ms, gather 2.23 ms, pressure cycles 6.03 ms,
and combined projection/publication 1.18 ms. These are work-matched to the
previous floor; the selected higher floor evolves a different trajectory.


## Domain page state visualization

The existing single Domain Pages layer now reads residency and last-step volume
work separately. Teal marks transport pages (including pages also needed for
sharpening), amber marks sharpening-only pages, and faint purple marks resident
pages without either volume-work flag. Nonresident pages emit no fill, edge or
interface contour, even if an activity record is stale. The controls show the
same colour legend. Pressure/interface work is not inferred from these volume
flags; purple does not imply a page is safe to retire.

Transport metadata is snapshotted before sharpening reuses the scratch flags;
this copies only the small page header/flags/slots, not numerical fields. The
overlay copies the activity records only when visible. Dawn pixel checks cover
all three colours, hidden missing records and hidden nonresident pages with
stale activity. Five CPU visual-layer checks pass.

The observed literal-advection phi difference of 1.535e-6 m is approximately
0.000123 cell widths on long dam. It exceeds the frozen-input test's 0.0001-cell
threshold, but that alone is not evidence of a physical error. Runtime loops did
not eliminate it (3.491e-6 m in a later frame), so literal advection remains
the production default; iterative redistance retains runtime loops. No inference about interface sign
changes or accumulated trajectory error can be made from the maximum norm alone;
those require separate measurements. The failed strict oracle is retained and
not presented as a passing correctness result.

After the page-state visual change, the canonical gate ran again in 414.4 s: 4/17
lanes passed, with no lane pass/fail changes from the preceding run. Full report:
`page-states-sparse-regression.json`. No gate or timing ceiling was relaxed.

The final native-stage oracle passed 12 same-input advances on both partial pages
and long dam, including live liquid insertion. V and phi were bit-identical to
the domain-free arm; pressure and velocity satisfied the unchanged normalized
1e-5 bound. This is a separate check from the unresolved atlas/native phi oracle.
