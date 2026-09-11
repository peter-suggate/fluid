# Cheap coarse-surface trial

Two changes are active in production on top of the restored September 6 hot
pipeline. The compiled mixed-patch evaluator is a tested experiment and is not
connected to resident transport yet.

## Production changes

`prepareTransportFaceRow` now reads the frozen support flags of every cell in
its accepted row. This replaces the two patch-centre owner queries, which could
miss a supported fine child. The existing face-support regression now passes
for all four child placements. No geometry search, new pass or allocation is
introduced.

The same read obtains the finest incident physical width. That width remains
fixed through this face's RK trace and terminal interpolation. Explicit coarse
regions can still enlarge it. Uniformly coarse faces therefore respond to small
physical displacements instead of sampling repeated coarse values on a unit
lattice. A production-WGSL quarter-cell wave-translation fixture at widths
1/2/4/8 passes; constants, dry receivers, blocked support and solid blending
also pass.

This does not repair all mixed-width interpolation. Different physical donor
centres can still disagree with virtual lattice positions at a junction. It
also does not restore native staggered-face identity. Those remain separate
work from this first production increment.

## Performance: corrected after external GPU contention

The first three captures (`cheap-surface-baseline`, `cheap-surface-support`,
`cheap-surface-width`) are **invalid for performance comparison**: the user
reported concurrent GPU use. In particular, the apparent 84 ms result is not
evidence against physical-width sampling. The invalidation is recorded in
`artifacts/long-dam-ab-2026-09-10/cheap-surface-timing-status.json`.

After the user explicitly confirmed GPU availability, four sequential runs
used control/candidate/candidate/control order. Control is restored transport
plus complete face support. Candidate also uses physical-width face sampling.
Both use long dam B8/P8, the paper timestep, eight warmup advances and sixteen
measured advances. They are complete evolving simulations, not immutable-state
kernel replays. Each receipt includes configuration and source provenance.

| Run | Median GPU advance (ms) |
| --- | ---: |
| Control A | 61.9315 |
| Candidate A | 63.4388 |
| Candidate B | 62.4558 |
| Control B | 61.9971 |

The average of run medians is **61.9643 → 62.9473 ms**, a **1.59%** increase.
This meets the proposed 10% incremental budget for physical-width sampling.
There is no uncontaminated same-session isolated timing of the support change
against the original two-point gate; do not claim its overhead is proven zero.
All four clean captures report no GPU validation errors and successful frame
receipts. These short runs do not establish long-term settling quality.

Receipts: `artifacts/long-dam-ab-2026-09-10/cheap-surface-exclusive-{control-a,width-a,width-b,control-b}.json`.

## Compiled-patch experiment

`sparse-cm12-interpolation-patch.ts` classifies eight actual donor centres into
axis-aligned rectangular-section geometry: boxes, wedges, pyramids and frusta.
Its evaluator computes the normal fraction followed by the two tangential
fractions directly. It retains the trilinear shape-function weights of these
geometries without Newton iteration or a cell-centre query. Unsupported shapes
and queries outside a patch return explicit misses.

The topology descriptor occupies 64 bytes. Eight donor IDs would add 32 bytes
if not shared with existing topology storage. Field values and query-dependent
weights are not cached. Regular interiors should use existing leaf descriptors
rather than allocating this record per cell.

CPU tests cover geometry reconstruction, positive partition, exact nodes and
boundaries under axis permutations and reflections, plus rejection of warped
and degenerate geometry. The GPU fixture covers **96 patches / 2,016 queries**,
including near-apex samples. Maximum weight disagreement with the CPU reference
is **1.20e-7**; maximum coordinate moment error is **2.29e-6** fine-cell units.

An offline census uses the actual accepted leaf roster from the saved frozen
adaptive-pool capture `mechanism-final-max2/24-activity.json`, with 6,656 cells.
It reconstructs unclipped interior geometry from that roster:

| Interior patch type | Count |
| --- | ---: |
| Regular | 5,202 |
| Recognized mixed | 1,125 |
| Unsupported mixed | 124 |

Thus the direct formula covers **90.07% of interior mixed patches** in this
capture. Geometry plus donor IDs for the recognized mixed records total
**108,000 bytes**. Another 1,490 vertices touch boundaries or absent support and
are excluded. This is a topology-count census, not a runtime query distribution
or a claim that 90% of difficult queries are solved.

The experiment deliberately has no production buffer or dispatch. It does not
yet implement a locator, boundary continuation, topology-delta cache updates,
or generic junction coverage. These costs are the next performance question;
the inexpensive evaluator alone cannot establish an inexpensive sampler.
Mixing it with an inaccurate fallback also needs a continuity check before use.

## Verification

- Production support and face-transport Dawn fixtures passed.
- Effective transport CPU contracts: 12/12 passed.
- Patch CPU/GPU fixtures passed; targeted ESLint passed.
- Repository TypeScript checking remains red in existing files; no reported
  error names the new patch module, its tests, or the census tool.
- Ran the unchanged `npm run test:dawn:sparse-cm12` gate. **9/17 lanes passed**:
  failure halt, mixed topology, clipped transfer, generation storage,
  hydrostatic adaptivity, mini32 correctness, min8 region surface,
  mini32 performance, and mini64 min8 surface.
- Mini32: **31.064 ms**, below the unchanged 40 ms ceiling.
- Symmetric expansion: **0.304505**, above 0.006; symmetry remains set aside by
  user instruction. The preceding restored baseline was 0.296483.
- Topology page-budget and mini64 performance exceeded their 30-second limits.
  Long-dam far-wall exhausted the remaining 5.06 seconds of suite time.
  The four subsequent terrain/live-edit/open-world lanes did not run before
  the unchanged 180-second total budget. They are unverified, not passes.

Gate receipt: `artifacts/long-dam-ab-2026-09-10/cheap-surface-regression.json`.
No timing ceilings or numerical thresholds were changed.
