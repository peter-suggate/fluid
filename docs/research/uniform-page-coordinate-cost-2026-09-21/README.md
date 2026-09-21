# Compiled page coordinates and the single-tile performance gate

For the subsequent production MiniDam32 pressure-budget change and its A/B
results, see [MiniDam32 pressure budget](minidam32-pressure-budget.md). The
coordinate-only measurements below predate that scheduling change.

## Contract

Paging is a storage/residency change. It must not change the pressure unknowns,
active work selection, numerical operators, iteration schedule, or convergence
criteria. The current acceptance case is a scene contained in one tile, at the
same performance as main with matched solver settings. Setup and generation
compilation may do infrastructure work; repeated cell and stencil operations
must not pay for it.

The production domain page is 32³. Physical texture tiles in the previous
adapter were 16³. Both a partial 16³ payload and a full 32³ payload are tested.
Halo/vertex planes are part of the field payload, not additional residency pages.

## What the previous implementation added

The original long-dam A/B used main `5d4d31f2` and branch `fddb0977`, the authored
192×96×32 scene, balanced quality, and 24 advances of 1/30 s. Four warmup frames
were excluded. GPU stage timestamps and queue-fenced wall time exclude setup,
rendering, and diagnostics readback. Browser simulations were absent and all
Dawn runs held the exclusive GPU lease. Stage medians are not additive.

| Configuration | Main, ms/advance | Original pages, ms/advance |
|---|---:|---:|
| Production defaults, first capture | 46.22 | 118.86 |
| Production defaults, repeated census capture | 47.72 | 121.98 |
| One Full-Cycle, zero V-Cycles | 36.06 | 83.81 |
| One Full-Cycle, main also forced to full-domain work | 39.49 | 83.81 |

The final row removes main's active-work selection for a workload-matched
control. It demonstrates that cycle count and windowing do not explain away the
regression. With this control, Full-Cycle GPU time is 6.36 ms on main versus
30.28 ms with the old pages; advection is 1.25 versus 5.96 ms.

### 1. More launches for the same pressure cells

The logical pressure hierarchy has the same capacities on both revisions.
Main can additionally clip work using its active-region records; capacity alone
must not be confused with the actual processed region. The following comparison
uses full-domain launches, independent of that clipping:

| Haloed level dimensions | Main workgroups | Old paged workgroups | Ratio |
|---|---:|---:|---:|
| 194×98×34 | 11,025 | 17,472 | 1.58× |
| 98×50×18 | 1,625 | 3,584 | 2.21× |
| 50×26×10 | 273 | 512 | 1.88× |
| 26×14×6 | 56 | 128 | 2.29× |
| 14×8×4 | 8 | 64 | 8× |
| 8×5×4 | 4 | 64 | 16× |
| 5×5×4 | 4 | 64 | 16× |

These are ordinary per-level dispatches, not the separate single-workgroup
coarsest solve. Padded invocations return without updating extra unknowns, but
launching them is extra runtime work. Coarse levels do not need full 16³ blocks.

### 2. Address translation was placed inside every load/store

`uniform-pressure-pages.ts` wraps texture accesses in `mgPageAddress`. It
checks logical bounds, extracts the logical tile, linearizes its slot, divides
that slot by atlas row dimensions, and constructs a physical coordinate.
Pressure hierarchy dimensions are runtime uniforms, so the compiler cannot
assume that every division has a constant divisor.

`uniform-texture-pages.ts` repeats this transformation across the other fields.
Known root bindings have specialized address constants, but extension fields
use generic metadata, including a runtime backing-kind branch. Repeated stencil
and interpolation taps repeatedly pay these costs. Several numerical helpers
have already clamped/validated the input before the adapter checks it again.

The root membership audit adds a dependent storage-buffer membership read for
every tap. Its atomic writes happen only on missing reads; it is incorrect to
attribute the healthy-path cost to an atomic increment on every read.

### 3. Scratch uses actual dependent indirection

Persistent texture fields currently use arithmetic slot assignment, not a
hashed page directory. By contrast, `uvEdgeAddress` decodes a logical cell ID,
computes its page, atomically reads the page-to-slot map, then computes the
record offset. Kernels such as `uvNormalizeRows` and `uvGather` call it repeatedly
inside donor loops even though the owning cell and page are unchanged. Many
callers already have the cell coordinate before converting it to a linear ID
and decoding it again.

### 4. Layout and cache locality

Cell, vertex, and haloed fields have different logical tile-grid dimensions,
so the same world neighborhood maps differently across fields. Locality within
a 16³ physical tile remains; neighbors crossing tile boundaries are relocated
according to the atlas row packing. The transport work list is appended using
an atomic counter, so spatial workgroup order is not guaranteed.

The finest pressure texture grows from 646,408 to 1,310,720 texels. A cell field
of 192×96×32 grows to 128×128×48; a haloed field becomes 128×128×80. These are
allocation/footprint facts, not proof that all padding is loaded. No hardware
cache-miss counter was captured. Poorer locality is a plausible additional
cost, not a measured percentage of the regression.

### 5. Other added work

The branch encodes all configured cycles and indirect gates rather than main's
lagged budget. It also rebuilds the unchanged accepted catalogue every advance,
constructs scratch work lists, and copies paged fields into dense presentation
textures. Removing or reducing solver cycles is not the coordinate solution.

Early exploratory changes were saved outside the checkout in
`/tmp/uniform-paging-exploratory-optimizations.patch` and reverted. They combined
multiple factors and are not used as individual causal measurements. The
unvalidated smoother fusion was removed completely.

## Implemented coordinate compilation

`uniform-page-execution.ts` proves whether the accepted generation consists of
one complete origin page. A partial payload still qualifies. An empty,
translated, or multi-page catalogue does not.

For that proven case, construction selects the native field representation:

- The cell field is `nx×ny×nz`, phi is `(nx+1)×(ny+1)×(nz+1)`, and pressure and
  transport retain their ordinary halos. The halo does not create atlas tiles.
- The existing native numerical shader is selected before pipeline compilation.
  No field-address adapter, membership wrapper, or runtime loop rewrite is added.
- Transport scratch uses its native linear record index. No transient page map,
  mark/compact passes, or scratch slot lookup is needed.
- Cell, vertex, extension, and pressure launches use native rectangular counts.
- Render/diagnostic consumers receive the actual fields, so no paged-to-dense
  publication copies are required.
- The page catalogue and overlay are built once during initialization. The
  immutable catalogue is not rebuilt during advances.

This is a layout specialization, not a scene-name check or a numerical shortcut.
The accepted page remains the residency unit and retains its overlay. All field
operations and pressure schedules remain unchanged. The specialization also
applies to other one-page shapes such as 24×16×16.

### Generation lifetime

The proof applies to an immutable accepted generation. Current production
catalogues are all-resident and do not mutate during stepping. A future growth,
retirement, or relocation transaction must construct the next execution plan,
resources, and pipelines **before** publishing that generation. It must not
change a live catalogue underneath a native-coordinate plan.

This change does not implement domain growth, fix multi-page performance, or
promise scattered-page addressing for free. Those remain subsequent work. For
multiple pages the design requirement is to resolve placement and neighbor
relationships outside the numerical inner loops, while retaining the same
work selection. A growable contiguous execution image can keep native accesses
but uses dense bounding-box storage; physically scattered pages require either
boundary/halo transfers or some addressing cost. That tradeoff must be measured
and explicit, not hidden in a generic texture-load wrapper.

## Validation and reproduction

The fixture keeps the 0.4 m cubic tank, a 0.2×0.2×0.4 m reservoir, and changes
only lattice resolution between 16³ and 32³. Both revisions use one Full-Cycle,
zero V-Cycles, 6/6 sweeps, tolerance 10, fixed cycle budget, and full-domain work.
This is a matched coordinate-infrastructure gate, not a claim that their
unmatched production scheduling defaults have identical performance.

```
TILE_EDGE=16 ONE_CYCLE=1 FULL_DOMAIN=1 FRAMES=44 \
  WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
  UNIFORM_BENCH_OUTPUT=/tmp/single-tile.json \
  node --import tsx tools/benchmark-uniform-long-dam-paging-dawn.ts
```

Run the identical tool in a detached main checkout, using the same dependencies
and GPU, sequentially. `TILE_EDGE=32` covers the full domain tile. The report
records source hashes, revision, raw frame/stage samples, field dimensions, and
pressure-plan metadata. Timing uses 40 samples after four warmups in the final
repeat captures. Initial exploratory captures used 20 samples.

Focused tests passed: 10 CPU tests and 17 Dawn checks. The new Dawn test compares
volume, vertex phi, velocity, and pressure bit-for-bit against native operators
at both sizes, through 24 advances including liquid insertion. Existing physical
boundary and corrupt-pressure rejection/recovery tests also pass.

The first 16³ measurement is 14.13 ms on main, 21.82 ms before coordinate
compilation, and 13.82 ms after it. Final ABBA captures (40 samples per arm per run) give:

| Fixture | Main pooled median | Compiled pages pooled median | Main-relative throughput |
|---|---:|---:|---:|
| 16³ | 13.844 ms | 13.903 ms | 99.6% |
| 32³ | 16.682 ms | 16.263 ms | 102.6% |

The 16³ final ABBA uses captures 3 and 4 of each arm; 32³ uses captures 1 and 2.
An earlier matched 44-frame pair measured 14.20 ms main / 15.31 ms compiled
(92.7% throughput), so the single-tile cost should be described as parity within
observed run variation, not a guaranteed speedup. Every capture is retained in
`measurements.json`. The comparison tool checks fixture dimensions, hierarchy,
schedule-mode flags, frame counts, and the original 90% throughput floor.

Type checking reports 15 errors outside the changed files, matching the known
repository error set; no new error names a changed file. An exploratory run of
`uniform-volume-pages-dawn.test.ts` failed its old `32 !== 16` overlay-edge
expectation; its garden-hose insertion subtest passed. This unrelated assertion
was not edited to obtain a passing result. The dedicated new native-coordinate,
boundary, and pressure-safety run passes all 17 checks.

The required canonical gate completed in 396.2 s within its 480 s budget:
**4/17 lanes passed; the suite failed**. See `sparse-regression.json`.
Failures include sparse-expansion and authored re-rung assertions, performance
ceilings, correctness timeouts, far-wall lanes, and outside-tank collapse. No
threshold or assertion was changed. Similar failures were already documented
in `../uniform-persistent-field-pages.md`; this run alone does not establish
baseline attribution for every failing lane. The full log remains at
`/tmp/uniform-native-page-sparse-gate.log`.

