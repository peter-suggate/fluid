# Uniform Geometric volume record pages in the application

The production WebGPU solver now has an opt-in `Volume record storage` control
in SIM / transport: Dense, 16³ pages, or 32³ pages. Changing it rebuilds the
solver. `Volume pages` in the visual layer panel shows actual GPU page demand,
with page boundaries and physical-slot tinting. Select a slice through the
liquid; a slice through empty space can correctly have no resident pages.

This is a production integration of paged **transport and sharpening scratch**,
not yet the requested unbounded sparse fluid domain. Volume, vertex phi,
velocity, pressure, donor sums and presentation remain dense. The logical page
table still covers the finite scene dimensions. No solver-domain expansion
beyond those dimensions is implemented. The earlier signed, unbounded host
layout experiment is not used to store these production fields.

## Allocation and scheduling

Each record is 80 bytes. GPU marking uses the existing 4³ work classification
and dispatch window, then compacts resident pages to physical slots. Logical
donor identities and transport arithmetic are unchanged. Scratch is cleared
between phases; it does not carry physical state across topology changes.

For grids with padded volume at most 64³, reserve complete page capacity and
keep the single-submission path. Larger grids read exact demand before each
phase, grow the arena when necessary, and retain its high-water capacity.
Per-stage GPU tracing now shares one timestamp query chain across all page
continuation submissions. Stage elapsed times include gaps between submissions;
they are not pure compute durations. The Page receipt waits readout reports host
wall waits plus allocation separately (including preceding GPU completion, so
these values must not be added to stage totals). The renderer holds presentation
during incomplete split frames. Runtime
controls are applied at the next complete frame boundary. Device-limit failure
is surfaced rather than silently truncating page demand. There is no eviction
or shrinking of the high-water allocation.

Dense remains the default. The option is WebGPU-only and excluded from the
2D WASM options contract.

## Verification

- Production Dawn test compares dense, 16³ and 32³ fields through five mini32
  steps: volume and vertex phi match exactly.
- Garden hose at 144 × 96 × 96 compares dense and 32³ pages through twelve steps,
  including two live liquid insertions and switching dense/tiled work scheduling: volume and vertex phi match exactly.
- Garden record arena high-water allocation: 78,643,200 bytes versus 106,168,320
  bytes dense (25.9% less for this buffer, not total solver memory). The final
  sharpening stage grew from three to six of 45 logical pages; transport demand
  can be larger, explaining the larger reserved high-water arena.
- Visual-layer Dawn test draws every layer and checks composition and absence
  of page records. CPU layout/visual/initial-volume tests: 18 passed.

The full production mini64 benchmark writes a sibling JSON report. It measures
complete simulation steps including queue completion, excludes rendering, and
uses fresh identical scenes in dense/paged/paged/dense order. This is distinct
from the earlier seven-point stencil benchmark.

Mini64 full-step result: dense median 66.65 ms, 32³ pages 67.17 ms, or
99.2% of dense throughput (24 measured steps per arm). This clears the 95%
target in this short local run, but does not certify other scenes or rendering.

The shared Sparse CM12 gate failed: multiple lane timeouts, a 309.33 ms
mini64 result above its 110 ms ceiling, and three topology/transport failures.
The unchanged revision also fails the long-dam assertion (792 vs 106),
terrain transport, outside-drop timeout, and mini64 timing (209.58 ms).
The baseline checks do not establish that every timeout in the full suite
is pre-existing. Raw gate and baseline receipts are saved alongside this report.
`check:types` reports the same 15 pre-existing errors in unrelated tests/tools.

## Remaining work for the original goal

Convert persistent V/phi/MAC velocity fields and all pressure hierarchy levels
to resident pages; define numerical support/air-boundary closure; share a
signed world-coordinate directory with terrain and presentation; perform
GPU frontier growth and safe reclamation; benchmark both large hose scenes
and compact scenes at matched numerical settings. Until those are implemented,
this option does not satisfy endless sparse-domain growth.

## Application verification

`vinext build` passed. A production server runs locally on port 3001.
Production SSR exposed a method-store initialization-order problem; the default
store now initializes on first use after the entry point installs the registry.
Five configuration/initialization tests pass, and `/scene` returns HTTP 200.

Browser verification of the production build: garden hose ran to 1.8 s with
32³ storage selected, purple page regions and boundaries visible on a Z slice,
and no browser error logs. The scene was left paused with the layer enabled.
The first browser shader compilation took approximately 2.5 minutes.

## Restored SIM timing

The initial paging integration disabled `shouldTracePhysics`, leaving SIM at
“no trace yet”. Removed that gate and wrapped the forwarding encoder with one
shared stage recorder so every continuation submission writes the same query
chain. Queue-wall fallback now begins before the first paged submission.

Verified in the production browser with 32³ pages: stage timings appeared with
no console errors. At 8.3 simulated seconds, the rolling window showed 163.91 ms
per advance: phi 48.5 ms, transport 31.1 ms, pressure setup 17.2 ms, projection
16.1 ms, sharpening 13.3 ms, extension 12.9 ms. These are browser observations,
not isolated benchmark samples. Stage elapsed time includes submission gaps.
The page receipt wall waits (64.1 / 38.1 ms in the latest frame) include preceding
GPU completion and overlap stage timings; they must not be added to the total.
