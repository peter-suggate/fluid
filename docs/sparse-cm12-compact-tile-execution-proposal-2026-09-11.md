# Compact tile execution for Sparse CM12

The Figure 7 radius-0.1 audit found 64 liquid-bearing cells, 18,432 accepted
cells including air support, and capacity for 242,244 cells / 816,276 rows.
The proposal is to make routine execution follow required work rather than
reserved capacity. This extends existing GPU structures; it does not replace
the numerical method or introduce a second topology authority.

## Existing foundations

TEI2 already supplies 4³ cell packets, addresses, strides, physical scale and
spatial mappings. Transport packet authority already compacts selected packets.
Final-scalar masks identify field changes. Transport stages a 27-leaf directory
window in workgroup memory. Hot topology and interned boundary operators carry
physical connectivity, including mixed-resolution interfaces.

The missing consistency is that many other stages still dispatch over all
accepted air or all reserved capacity. A new texture atlas alone cannot fix it.

## Target representation

Extend the existing tile descriptors with precompiled neighboring tile/boundary
references and regular-interior versus boundary classification. Keep static
geometry, invalidated by topology changes, separate from dynamic field masks.
Use GPU-authored compact lists for extension, supported faces, diffusion,
excess-density repair, pressure-membership repair and presentation updates.

Each stage needs its own source-and-receiver support closure. A wet-only list
would omit dry receivers that can acquire mass. Retired entries must explicitly
clear old publications. Lists, field masks and geometry must agree on generation.

For local regular stencils a 4³ tile plus one-cell halo has 216 samples: 864 bytes
per scalar or 3,456 bytes per vec4. Cooperative loads can reuse those values
within a workgroup. Long characteristic traces need neighboring-tile traversal
or fallback; mixed-resolution halos must retain the established physical
operators. Benefits must exceed list construction and halo maintenance costs.

Storage buffers are the first implementation target. Textures may help selected
read-heavy sampling later, but do not remove capacity scans, repeated passes or
the fixed pressure dispatch tail. Conservative fixed-point scatter still needs
writable receipts. No speedup is claimed before measurement.

## First implementation: compact pressure-row membership repair

Current row publication dispatches one lane per allocated row and reconstructs
all membership words. Figure 7 therefore visits 816,276 slots to maintain a
pressure problem with tens of cells.

Preserve the accepted membership bitmap and its population. Compile a compact
list of 64-row tiles needing publication. Mark from the accepted-row stream
when scalar/flip facts, topology, gravity or solid conditions invalidate the
existing classification. On a topology change also schedule tiles containing
previous pressure members, so retired rows are explicitly cleared even though
they no longer occur in the accepted-row stream. Classify selected tiles using
the existing row predicate and update population by the difference between old
and new words. Validate publication coverage against the compiled worklist.

The bounded first version may scan tile metadata to compact the list. For the
audited scene that is 12,755 tile entries rather than 816,276 full row lanes.
It still scans accepted rows to discover changes; deriving the same complete
closure directly from dirty scalar tiles is a later optimization. Preserve
unchanged-row theta/cache behavior and keep a full-domain oracle for testing.

Required cases: bootstrap, unchanged frame, scalar change without membership
flip, liquid appearance/disappearance, topology replacement and retirement,
gravity changes, solids, partial final tile, and empty worklist. Compare the
compact publication to the old full-domain predicate on the GPU. Run focused
pressure tests and the unchanged canonical Dawn regression gate. Report list
maintenance cost and changed tile counts as well as end-to-end timings.

## Subsequent steps

1. Supported-face tile lists derived from extension validity.
2. Excess-density tile lists with complete receiver closure between relay rounds.
3. Compact scalar/presentation metadata updates and explicit retirement records.
4. Precompiled neighbor references and field halos where profiling demonstrates
   repeated lookup/load cost.
5. Separately measure support-band sizing and pressure command-tail overhead;
   neither is solved by a new data layout alone.

See [the scene audit](cm12-figure-7-radius-01-work-audit-2026-09-11.md) for counts,
code anchors and the distinction between reset census and browser timing means.

## Implemented first step

The following records the first compact implementation. Its subsequent
[cache reuse](sparse-cm12-pressure-row-cache-reuse-2026-09-11.md)
retains this compact layout and shares tile loads across lanes.

The resident pipeline now marks invalidated accepted rows, compacts stamped
64-row tiles, seals indirect arguments, and publishes only those tiles. The
existing full-domain classifier remains available as the numerical control.
No CPU readback is needed to size the dispatch. These are contiguous row-ID
tiles, not geometric 4³ cell tiles; this step introduces the worklist mechanism
before extending spatial stencil caching.

For the reset census, discovery dispatches 912 groups / 58,368 accepted-row
lanes. List compilation dispatches 200 groups / 12,800 lanes, of which 12,755
are valid tile entries. Sealing uses one invocation. Publication then dispatches
one 64-lane group per selected tile and writes two membership words per full
tile. The last capacity tile has 20 valid rows and only one word.
Previously publication dispatched 12,755 groups / 816,320 lanes, of which
816,276 were valid row slots, and rewrote 25,509 words every advance. Counts
change with topology; these reset counts are not asserted for later frames.

Two 12,755-word arrays (stamps and compact tile IDs), each padded to 12,800
words, plus a 64-word-aligned control block add 102,656 bytes. The compact
publisher preserves bitmap words and cached theta for untouched rows and
maintains total membership with signed popcount differences. On topology
changes, the metadata scan also selects tiles containing old active members,
including retired tiles absent from the accepted stream. Gravity and static
solid cases conservatively invalidate all accepted rows.

Implementation anchors:

- `lib/methods/adaptive-mass/sparse-cm12-canonical-membership.ts`: row arena layout.
- `lib/methods/adaptive-mass/sparse-cm12-canonical-membership.wgsl.ts`: stamps,
  compaction, indirect sealing, persistent word publication and population.
- `lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts`: accepted-row
  invalidation, shared row classifier and compact/full dispatch entry points.
- `lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts`: ordered GPU stages
  and indirect-argument copy.
- `tests/sparse-cm12-pressure-row-repair-dawn.test.ts`: persistent compact/full
  GPU comparison across 11 epochs, including retirement and an empty list.
- `tools/probe-sparse-cm12-stage-cost.ts`: Figure 7 radius/tolerance overrides
  and per-frame dirty-tile/publication-word receipts.

## Dawn measurement and validation

Measured on this machine's Metal backend, serially, with the browser unloaded.
Both arms used Figure 7, radius 0.1 m, B8/P8, coarse-first, scene timestep
1/30 s, pressure relative tolerance 0.194, eight warmup advances and 24 measured
advances. The full-domain control is an isolated source copy with only the row
dispatch sequence replaced by the retained full-domain entry point. Both arms
retain the new arena layout and delta-popcount publisher, so this comparison
isolates execution scheduling rather than reproducing every historical byte.
The exact control patch is saved with the receipts.

| Hardware median, milliseconds | Full-domain control | Compact |
|---|---:|---:|
| Row publication, including list construction/seal/copy | 0.2621 | 0.1966 |
| Entire pressure-topology stage | 0.7209 | 0.6554 |
| Pressure solve | 3.9977 | 4.0632 |
| Non-pressure work | 8.8474 | 8.9129 |
| Entire advance | 12.9106 | 12.9761 |

Row publication improves by one 0.065536 ms timestamp tick (about 25%).
**There is no demonstrated end-to-end speedup in this sample.** The compact
frame median is 0.0655 ms higher; other stage variation outweighs this small
local saving. The prior browser's approximately 15 ms rolling average is a
different sampling context and must not be used as the A/B baseline. Separate
stage medians need not sum to the median of per-frame sums.

The control publishes exactly 25,509 membership words per measured advance.
Compact publication selects **336–890 tiles**, median **565**, and writes
**672–1,780 words**, median **1,130**: a **95.57% median reduction in word
publications**. Those tiles launch 21,504–56,960 row lanes (median 36,160),
compared with the full-domain 816,320 lanes. These numbers are direct GPU
worklist receipts, not estimates from the sphere volume. Each selected tile
includes adjacent unchanged/dry row slots; it is not a tile containing only
pressure members. The measured pressure problem has **16–24 cells** and
**68–128 rows**. End-frame accepted topology contains **6,400–16,672 cells**
and **19,904–52,912 rows**, including air support. Pressure publication uses
its predecessor topology; do not divide by a same-frame end-topology count
and call the quotient exact occupancy. The receipt records topology attribution.

This first version still scans every accepted row for invalidation and every
capacity tile's stamp. Scalar-change masks, topology changes and conservative
invalidation can select many tiles for a pressure problem this small. It does
not yet provide a geometric dirty-cell-to-incident-row closure or eliminate
support-air traversal. The next optimization should target that discovery
cost and the larger non-pressure stages, guided by the audit.

Final density and gamma SHA-256 hashes match exactly between arms, as do all
24 end-frame topology commit counts. Both runs report fault-free pressure
receipts and no WebGPU validation errors. The focused compact/full GPU test
passes all 11 epochs, and the existing gravity/cache GPU test also passes.
The canonical `npm run test:dawn:sparse-cm12` gate passes all 17 lanes in
280.7 seconds; mini32 median is 29.622 ms (40 ms ceiling), mini64 87.491 ms
(110 ms ceiling). No baseline or ceiling was changed for this implementation.
`git diff --check` passes. `npm run check:types` still fails on unrelated
files, including the hole-probe BlobPart type, coarse-first fixture optional
device accesses, and stale retained-scene-density imports; no errors are
reported in this implementation's files.

Reproduce the compact run from the repository root:

```sh
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" FLUID_WEBGPU_BACKEND=metal \
node --import tsx tools/probe-sparse-cm12-stage-cost.ts \
  --scene=cm12-figure-7 --sphere-radius=0.1 --pressure-relative-tolerance=0.194 \
  --production-defaults=1 --time-step=scene --warmup=8 --frames=24 \
  --final-scalar-hash=1 --final-qa=0 --quiet=1 \
  --out=artifacts/cm12-figure-7-radius-01/pressure-row-compact.json
```

Unload the browser simulation before Dawn, as required by repository guidance.
For the control, apply the saved patch to an isolated source copy with the
same dependencies and rerun the same command to a separate output path.
The initially captured paper-timestep pair is retained as supplementary data;
for this scene its actual dt is also 1/30 s, and the scalar hashes agree.

Receipts:

- [Compact scene-timestep run](../artifacts/cm12-figure-7-radius-01/pressure-row-compact.json)
- [Full-domain scene-timestep control](../artifacts/cm12-figure-7-radius-01/pressure-row-full-control.json)
- [Exact control dispatch patch](../artifacts/cm12-figure-7-radius-01/pressure-row-full-control.patch)
- [Canonical regression result](../artifacts/cm12-figure-7-radius-01/pressure-row-regression.json)
