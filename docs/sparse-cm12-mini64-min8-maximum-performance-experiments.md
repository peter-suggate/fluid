# Sparse CM12 mini64/min8 maximum-performance experiments

Status: active investigation ledger

Last updated: 2026-09-03

## Goal and constraints

Find the maximum performance available from the adaptive-mass approach by using
mini-dam 64 with a full-domain minimum cell size of 8 as a microscope for work
which scales with the physical domain rather than the accepted numerical graph.
The smallest mini-dam scene, mini16, is the reference: during its initial stable
topology epoch it has almost the same accepted cell and row counts as the
mini64/min8 case.

The admissible changes are general algorithms and data structures. They must not
contain scene names, region predicates, special coordinates, or selected-rung
shortcuts. Dispatch-size tuning is outside this investigation. Work counts may
be recorded to identify the representation being traversed, but launch geometry
is not an optimization variable.

The pressure solve is frozen. Do not alter its algorithm, kernels, iteration
policy, reductions, hierarchy, tolerances, or numerical ordering. Pressure
solve time remains in receipts only to close the frame and detect accidental
changes. This ledger also treats pressure RHS and velocity projection as frozen
unless that scope is explicitly reopened. Pressure *topology publication* may
be changed because it is construction of the solve's immutable input, but the
published cell IDs, row IDs, coefficients, ordering, and generation must remain
exact.

## How to read the benchmark

There are two separate targets:

1. **Domain scalability.** At nearly equal accepted work, mini64/min8 should
   approach mini16. A stage ratio materially above the 1.12x cell-count ratio is
   evidence of domain-, template-, brick-, or presentation-shaped work.
2. **Absolute cost.** A stage can be equally expensive in both scenes and still
   dominate the frame. Surface sharpening is the important example. Removing
   only the mini64/mini16 gap is not the maximum-performance result.

Stage medians do not add exactly because each stage is median-reduced
independently. One Metal timestamp quantum in these captures is 65.536 us.

There are also two different clocks in this ledger. **Production frame time**
means uninstrumented queue-complete throughput at the browser's production
queue depth of two. The hardware boundary-chain probe exists to attribute cost
to stages; its `medianAdvance_ms` is an instrumented chain total and is not a
production frame time. Never compare one clock against the other.

## E0p: corrected production-frame baseline

Status: **recorded; three same-source repeats**

The earlier comparison incorrectly called the stage probe's instrumented
boundary-chain total a whole-frame time. A new production benchmark now runs
the shipping solver with performance instrumentation off, excludes construction
and terminal diagnostics, and amortizes queue completion across the browser's
actual `BROWSER_GPU_THROUGHPUT_DEPTH = 2`. This reproduces the observed mini16
average of about 15 ms.

Both arms used source fingerprint
`0fa391b40a1957811db8e1cf22137be91e6a574a4fe175e48e126660cedb899d`,
Metal, B8/P8, scene cadence (`dt = 0.004 s`), 8 warmup advances, and 10
measured advances arranged as five complete depth-two batches. Each result
below aggregates three fresh-process runs (30 advances / 15 batches per arm).

| Production measurement | mini16 stable epoch | mini64 + full-domain min8 | ratio / delta |
|---|---:|---:|---:|
| Mean frame time | **14.7979 ms** | **17.0286 ms** | **1.1507x / +2.2306 ms** |
| Median depth-two batch, per frame | 14.7345 ms | 17.0895 ms | 1.1598x / +2.3550 ms |
| Fresh-process run means | 14.9887 / 14.7319 / 14.6732 ms | 17.0642 / 17.0885 / 16.9330 ms | — |
| Accepted cells | 456 | 512 | 1.12x |
| Accepted rows | 1,524 | 1,728 | 1.13x |
| Pressure cells / rows | 119 / 357 | 175 / 525 | 1.47x |
| Terminal topology commits | 0 | 0 | matched stable epoch |

All six runs had zero WebGPU validation errors. The corrected conclusion is
that mini64/min8 has a real production-frame penalty of about **2.23 ms**, even
though represented cell and row counts are nearly matched. Stage receipts below
remain useful for locating that penalty, but their absolute totals are not the
production frame clock.

## E0: paired stage-attribution baseline

Status: **recorded; attribution only**

Both arms used the same dirty-tree source fingerprint
`a19616c5b9310ada6bb2e4a982ecbc05289558052049d4915a58d654f951e064`
at commit `ffad90ca8e29a18fdbc5244c11f247538573d252`. They used Metal,
B8/P8, scene cadence (`dt = 0.004 s`), gamma diffusion and sharpening enabled,
8 warmups, 11 hardware-timestamped frames, and final readback QA disabled.

Eleven samples are intentional. Mini16 remains at generation 1 with no topology
commit over this interval. In longer captures it refines and ceases to be the
cell-matched reference. Both arms ended with zero validation errors and passed
diagnostic closure.

| Measurement | mini16 stable epoch | mini64 + full-domain min8 | ratio |
|---|---:|---:|---:|
| Finest-domain cells | 4,096 | 262,144 | 64.00x |
| Logical / packed bricks | 8 | 512 | 64.00x |
| Accepted cells | 456 | 512 | 1.12x |
| Accepted rows | 1,524 | 1,728 | 1.13x |
| Pressure cells | 119 | 175 | 1.47x |
| Pressure rows | 357 | 525 | 1.47x |
| Instrumented non-pressure chain median | 15.4010 ms | 16.6461 ms | 1.08x |
| Instrumented full boundary-chain median | 22.9376 ms | 23.0031 ms | 1.00x |
| Pressure solve median (frozen) | 7.0779 ms | 6.3570 ms | 0.90x |

The boundary-chain coincidence is not a production-frame result: mini64/min8
has an extra 1.2451 ms of attributed non-pressure work while its frozen pressure
solve and sharpening medians are lower. Only the stage breakdown is used from
this instrumented capture.

### Stage comparison

| Stage | mini16 | mini64/min8 | delta | ratio | disposition |
|---|---:|---:|---:|---:|---|
| Surface sharpening | 6.4225 | 5.7016 | -0.7209 | 0.89x | optimize absolute cost |
| Conservative transport | 1.0486 | 2.5559 | +1.5073 | 2.44x | first target |
| Transport velocity extension | 1.4418 | 1.5729 | +0.1311 | 1.09x | later absolute target |
| Face preparation | 1.1796 | 1.1796 | 0.0000 | 1.00x | later absolute target |
| Pressure topology | 1.0486 | 1.3107 | +0.2621 | 1.25x | topology only; solve frozen |
| Resolution planning | 0.9830 | 1.2452 | +0.2622 | 1.27x | domain-shaped |
| Presentation publication | 0.4588 | 1.3107 | +0.8519 | 2.86x | domain-shaped |
| Activity measurement | 0.3932 | 0.5243 | +0.1311 | 1.33x | domain-shaped |
| Gamma diffusion | 0.1966 | 0.1966 | 0.0000 | 1.00x | low priority |
| Candidate transfer | 0.3277 | 0.3277 | 0.0000 | 1.00x | low priority |
| Symmetry authority | 0.1311 | 0.1311 | 0.0000 | 1.00x | low priority |
| Body forces | 0.0655 | 0.0655 | 0.0000 | 1.00x | low priority |
| Pressure RHS | 0.5243 | 0.3932 | -0.1311 | 0.75x | frozen |
| Velocity projection | 0.3932 | 0.3932 | 0.0000 | 1.00x | frozen |
| Pressure solve | 7.0779 | 6.3570 | -0.7209 | 0.90x | frozen |

All times are milliseconds. The largest attributable work-chunk gaps are more
specific:

| Work chunk | mini16 | mini64/min8 | delta | ratio |
|---|---:|---:|---:|---:|
| Transport trace | 0.5243 | 1.2452 | +0.7209 | 2.38x |
| Transport deficit scatter | 0.4588 | 1.1796 | +0.7208 | 2.57x |
| Presentation publication | 0.4588 | 1.3107 | +0.8519 | 2.86x |
| Sharpening transform | 1.7695 | 2.2938 | +0.5243 | 1.30x |
| PCM row publication | 0.1966 | 0.6554 | +0.4588 | 3.33x |
| Refinement-policy classification | 0.0655 | 0.2621 | +0.1966 | 4.00x |
| Dirty-brick-mask publication | 0.1966 | 0.3277 | +0.1311 | 1.67x |
| VEX sweeps | 1.0486 | 1.1796 | +0.1310 | 1.12x |

### Representation traversed

The accepted graph is almost matched, but several physical structures are not:

| Capacity | mini16 | mini64/min8 | ratio |
|---|---:|---:|---:|
| Multi-rung template cells | 53,832 | 561,664 | 10.43x |
| Multi-rung template rows | 181,452 | 1,856,064 | 10.23x |
| Accepted-row membership bytes | 22,684 | 232,008 | 10.23x |
| Mass-departure cache bytes | 3,014,592 | 31,453,184 | 10.43x |
| Face-address program bytes | 10,752 | 475,136 | 44.19x |
| Explicit face seam addresses | 2,556 | 114,624 | 44.85x |
| Presentation pages | 8 | 512 | 64.00x |
| Resident allocation bytes | 31,035,544 | 444,295,820 | 14.32x |

The clearest smoking gun is PCM row publication. With unchanged topology,
mini64/min8 scans 58,002 canonical row-membership words every frame to publish
525 pressure rows. Mini16 scans 5,671 words to publish 357 rows. The current
host schedule explicitly launches `compileCanonicalPressureRows` at canonical
membership capacity in
[`webgpu-sparse-cm12-resident.ts`](../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts).

There is also a receipt bug to fix separately: `readWorkShapeQA()` reports zero
conditioning clear bytes, while surface sharpening clears two template-cell
`f32` planes. For mini64/min8 that is 4,493,312 bytes per frame. Fixing the
receipt is instrumentation, not a performance experiment.

## Target representation

The long-term representation should be a **canonical current-generation
block-sparse execution image**, not another cache beside the current template
supergraph.

- Partition same-rung accepted space into small spatial microtiles (initially
  4x4x4 cells) which may cross ownership-brick boundaries.
- Store one current-generation directory entry per tile: rung, state base,
  logical origin, validity mask, and exception-range offsets.
- Derive regular interior cell IDs, centers, widths, interpolation neighbours,
  face rows, row metrics, and common stencil terms arithmetically.
- Store explicit records only for rung transitions, physical boundaries,
  sparse-world lifecycle boundaries, embedded/moving solids, and other cut
  geometry.
- Allocate mutable state for current accepted tiles plus topology-transaction
  delta, rather than for every cell and row in every rung of every brick.
- Keep stable generation and deterministic tile/lane ordering so all consumers
  share one authority and numerical replays remain reproducible.

The staged experiments below prove pieces of this representation before a full
state-layout migration.

## Experiment ledger

Statuses are `planned`, `running`, `accepted`, `rejected`, or `inconclusive`.
Never delete a rejected result; retain its artifacts and the reason it failed.

| ID | Status | Experiment | Primary metric | Promotion condition |
|---|---|---|---|---|
| E0p | recorded | Uninstrumented production-frame baseline | depth-two queue throughput | three paired replays agree |
| E0 | recorded | Same-source stable-epoch stage attribution | paired stage medians | attribution only; never frame time |
| E1 | accepted | Implicit authored-owner arithmetic in transport | trace + scatter, non-pressure | exact fields/receipts; stable win |
| E2 | planned | Shared current-generation tile directory | allocation + consumers | exact generation/state; multi-stage win |
| E3a | rejected | Implicit authored-owner arithmetic in sharpening | sharpening transform | exact conservative result; stable win |
| E3b | rejected | Alternating capacity-repair receipt planes | capacity repair | exact conservative result; stable win |
| E3c | inconclusive | Deterministic two-pass capacity gather | capacity repair | deterministic conservative result; stable win |
| E4 | planned | Implicit pressure-topology row publication | PCM row publication | exact solve input; solve untouched |
| E5 | planned | Adaptive-native presentation image | presentation publication | exact representability/surface receipts |
| E6 | planned | Event-authored activity and planning | activity + planning | exact next topology decisions |
| E7 | planned | Implicit regular face preparation | face preparation | remove traces/loads, not launch filtering |
| E8 | planned | Tile-resident VEX halo reuse | VEX sweeps | exact velocity extension |
| E9 | planned | Current-generation compact mutable state | allocation + bandwidth + frame | exact topology lifecycle and fields |

## E1: implicit regular-interior transport arithmetic

This is the first implementation experiment because transport is the largest
positive matched-work anomaly: 2.44x total, with equal 0.72 ms gaps in trace
and scatter.

The current path materializes and reloads a 56-byte mass-departure stencil per
template cell and repeatedly resolves adaptive ownership. The A/B should retain
current stable template IDs at first, but add a common-case arithmetic path:

1. Given the current tile/rung and local coordinate, derive the eight
   interpolation neighbours and weights without reading explicit neighbour IDs.
2. Cross a same-rung tile or brick boundary through the current-generation
   brick/tile directory and continue arithmetically.
3. Fall back to the existing explicit path for rung seams, domain boundaries,
   sparse-air boundaries, and solid/cut geometry.
4. Keep the same corner order, fixed-point contribution order, and destination
   atomics. This experiment is a representation change, not a numerical method
   change.

Measure regular/fallback counts, bytes read/written per accepted cell, transport
trace, deficit scatter, gather, and the complete non-pressure frame. Do not tune
dispatch width. A win only in mini64/min8 is insufficient: mini16 and at least
one mixed-rung scene must not regress materially.

## E2: shared current-generation tile directory

Promote the E1 arithmetic context into the single structural authority consumed
by transport, sharpening, face work, topology publication, activity, planning,
and presentation. The directory must be rebuilt only with an accepted topology
generation or its candidate delta. This is the bridge away from the all-rung
template supergraph and duplicated stage-specific compact lists.

First prove that the directory reconstructs the exact accepted cell sequence,
accepted row sequence, ownership, and exception set. Then migrate one consumer
at a time. Do not create a second independently maintained active-set cache.

## E3: tile-local surface sharpening

Sharpening is the largest non-pressure stage even though it is not slower in
mini64/min8 than mini16. The transform is domain-sensitive, and the eight-round
capacity repair is the absolute-cost floor.

Load a tile plus its maximum required halo once, compute the common regular
sharpening stencil arithmetically, and emit only cross-tile/exception
contributions globally. Evaluate fusing prepare/scatter only after proving the
same read-before-write dependencies. For capacity repair, test a tile-local
frontier/fixed-point representation which preserves the current eight-hop
conservative result and deterministic contribution ordering.

Do not repeat the rejected branch-only early-exit arm documented in
[`sparse-cm12-ocean-min8-sharpening-capacity-early-exit-experiment.md`](./sparse-cm12-ocean-min8-sharpening-capacity-early-exit-experiment.md).
It reduced no useful work and regressed the frame.

## E4: implicit pressure-topology row publication

Replace the capacity-wide canonical membership scan with row construction from
current accepted tiles plus explicit exception rows. Same-rung regular rows are
derived in deterministic canonical order; seams and cut geometry retain
explicit records. This may change pressure-topology data structures only.

The A/B must prove byte-for-byte identity of pressure cell IDs, row IDs,
membership, coefficients, ordering, topology generation, and all pressure
authority receipts before timing. The pressure solve implementation and inputs
after publication are otherwise untouched. The immediate target is the 58,002
word scan which currently produces only 525 rows.

## E5: adaptive-native presentation image

Presentation owns 512 P8 pages in mini64/min8 versus 8 in mini16 and costs
2.86x as much despite nearly matched physics. Publish coarse accepted tiles in
their native resolution and allocate fine payload only for surface/interface
exceptions. Planning, closure, verification, representability proof, and
retirement should consume the current-generation tile directory rather than
the complete ownership-brick domain.

This is not a mini64 or min8 shortcut. Mixed-rung topology must produce the same
public surface and representability decisions, including live insertion and
retirement.

## E6: event-authored activity and resolution planning

Final scalar and face producers already know which tiles changed. Publish one
generation-stamped per-tile summary (mass, surface interval, velocity/detail
bounds, topology/solid causes) as part of those writes. Activity history and
planning consume changed summaries and the necessary neighbour closure. On a
stable topology with unchanged summaries, the work frontier should naturally
be empty; no scene or quiescence predicate is allowed.

The oracle is exact equality of activity history, planned rung per leaf,
grading closure, candidate delta, and commit generation over evolving scenes.

## E7: implicit regular face preparation

Face preparation is already cell-proportional in the paired benchmark, so it
comes after the domain-shaped wins. Test arithmetic construction of regular
same-rung interior faces and retain the explicit address program only for seam,
boundary, and solid exceptions. Success requires eliminating complete
characteristic traces or their memory traffic.

Do not repeat seam-family dispatch filtering; it cut launched work in half but
regressed face preparation by 7%. See
[`sparse-cm12-face-preparation-seam-family-experiment-2026-09-02.md`](./sparse-cm12-face-preparation-seam-family-experiment-2026-09-02.md).

## E8: tile-resident velocity extension

VEX is close to accepted-cell scaling in the paired result. After E2 exists,
test loading each tile's velocity/mask halo once and performing the eight direct
sweeps locally, publishing only cross-tile boundary changes between dependent
steps. Preserve sweep order and the fused commit result exactly.

## E9: compact current-generation mutable state

Once several consumers use E2, allocate dynamic scalar/vector state by current
accepted tile ordinal plus candidate-delta capacity. The existing resident
layout sizes every dynamic field by template cell/row capacity; the paired
mini64/min8 instance therefore allocates 444 MB to advance 512 cells. Keep
immutable per-rung geometry templates if useful, but stop giving inactive rungs
full mutable runtime state.

This is the highest-ceiling change and the riskiest. It must prove topology
commit, rollback, live liquid/rigid insertion, retirement, and stable-token
translation before replacing the production layout.

## Results

### E1: implicit authored-owner arithmetic in transport / 2026-09-03

Status: **accepted and enabled in production**

- Hypothesis: the regular authored-domain ownership lookup can use the compact
  immutable LOD1 arithmetic directory, retaining WDR1 only for dynamically
  grown SparseWorld pages, instead of resolving every regular point through
  the materialized world directory.
- General change: append the already-constructed immutable LOD1 directory to
  the topology image for this construction-only A/B and use it in packed
  coarse transport trace and scatter. Cell identity, stencil corner order,
  fixed-point contributions, and gather are unchanged.
- Three strict current-source candidate artifacts are
  `/tmp/e1-current-candidate-r{1,2,3}.json`; their same-source controls are
  `/tmp/e3b-revised-control-r{1,2,3}.json`, all at fingerprint
  `2262d9f9...` (full value is stored in every receipt).
- mini64/min8 conservative transport fell from 2.5559 ms to 1.9661 ms in
  every replay: **-0.5898 ms / -23.1%**. Trace fell 1.2452 -> 0.9830 ms,
  scatter 1.1796 -> 0.9175 ms, and gather remained 0.1311 ms.
- Same-source raw Phase-1 receipts in `/tmp/e1-current-hash-control.json` and
  `/tmp/e1-current-hash-candidate.json` match for every captured plane, with
  packet fault zero and packet count 512. This includes departure coordinates,
  stencil IDs/weights, beta, deficit atomics, gathered density/gamma, packet
  IDs/lanes, and the downstream sharpening inputs.
- Terminal accepted work, pressure counts, convergence receipts, topology
  receipts, and validation receipts matched in the strict timing pair.
- Focused CPU check passed: `tests/sparse-cm12-rung-major-execution-image.test.ts`
  (12/12).
- Generality controls: mini16 total transport was unchanged at 0.4588 ms
  (456 cells); ordinary mixed-rung mini64 improved 1.6384 -> 1.5729 ms with
  identical 54,916 accepted cells, 161,145 rows, topology evolution, and
  pressure receipts.
- Focused regression lanes passed: `symmetric-expansion`,
  `hydrostatic-adaptivity`, `mini32-correctness`, `mini32-performance` on an
  isolated retry (39.8459 ms < 40 ms), and `mini64-performance`
  (38.4041 ms < 50 ms). An immediately preceding mini32 performance attempt
  failed during a machine-wide timing excursion at 53.674 ms; no ceiling was
  changed.
- Full `npm run test:dawn:sparse-cm12` passed all 12 lanes in 144.4 s.
  In-suite performance medians were mini32 24.0517 ms and mini64 37.5521 ms.
- Decision: accepted and promoted to the ordinary resident constructor.
  Authored pages now use immutable LOD1 ownership arithmetic; dynamically
  grown signed-world pages retain WDR1 as the general exception path.

### E3a: implicit authored-owner arithmetic in sharpening / 2026-09-03

Status: **rejected**

- Hypothesis: replace TEI's staged 27-leaf sharpening directory with the same
  compact immutable LOD1 arithmetic lookup used by E1.
- Strict same-source artifacts: `/tmp/e3-control-r3.json` and
  `/tmp/e3-candidate-r3.json`, fingerprint
  `4474d319...` (full value is stored in both receipts).
- mini64/min8 sharpening regressed from 5.6361 ms to 6.4225 ms:
  **+0.7864 ms / +14.0%**. The transform regressed 2.2938 -> 3.1457 ms.
  Transport remained exactly 2.5559 ms, confirming the experiment was
  isolated to sharpening.
- Raw Phase-1 field hashes also matched between the diagnostic arms, although
  those diagnostic receipts were captured across an unrelated source edit.
- Decision: reject. The staged TEI neighbourhood has useful locality for the
  many lookups in a sharpening trace; direct LOD arithmetic is not a universal
  replacement. No regression promotion was run for a rejected arm.

### E3b: alternating capacity-repair receipt planes / 2026-09-03

Status: **rejected**

- Hypothesis: the eight conservative relay rounds need eight scatter/finalize
  dependencies, but do not need to clear their receipt plane eight times.
- General change: alternate two conditioning planes whose previous consumers
  are complete. Sharpening finalize consumes and clears plane 6; plane 5 is
  cleared once. Each capacity finalizer atomically consumes and clears its
  plane, readying it for reuse two rounds later. All eight relay rounds,
  accepted-cell traversal, neighbour order, integer division, fixed-point
  atomics, and state updates are retained. Seven full accepted-cell clear
  passes are removed; no dispatch size was changed.
- An initial fused-clear form appeared faster but changed the compiled
  sharpening-finalize path and diverged from control after the first frame. It
  was discarded, not treated as a numerical relaxation.
- Corrected strict same-source timing artifacts are
  `/tmp/e3b-revised-{control,candidate}-r{1,2,3}.json`, fingerprint
  `2262d9f9...` (full value is stored in every receipt).
- Control capacity-repair medians were 2.9491, 2.2282, and 2.4248 ms;
  candidate medians were 2.4248, 2.2282, and 2.4904 ms. Both
  median-of-medians are **2.4248 ms**. Surface median-of-medians regressed
  5.1118 -> 5.1773 ms and non-pressure regressed 16.9083 -> 16.9738 ms, one
  Metal timestamp quantum in each case.
- Both arms retained 512 accepted cells, 1,728 rows, 175 pressure cells,
  525 pressure rows, 24 pressure iterations, zero topology faults, zero
  validation errors, and exact timestamp closure.
- The corrected form clears planes 5 and 6 together after sharpening, then uses
  the production atomic load and arithmetic before clearing each receipt after
  consumption. Same-source control/candidate hashes are bit-exact at steps 1,
  2, and 19, including density, gamma, scalar summaries, pressure residuals,
  topology receipts, and validation results. Repeated candidate runs are also
  deterministic.
- Decision: reject. Seven fewer clearing passes do not improve stable timing;
  dispatch removal alone is not the surface-sharpening ceiling. No regression
  promotion was run for a rejected arm. Pursue tile-local data reuse or a
  conservative gather representation instead.

### E3c: deterministic two-pass capacity gather / 2026-09-03

Status: **inconclusive; target win, not production-safe yet**

- Hypothesis: capacity repair is limited by its contended global receipt
  scatter, not by clearing alone. Publish each source cell's equal integer
  share once, then have each destination gather neighbour shares in canonical
  incidence order.
- General change: each of the same eight rounds uses two passes. The first
  computes the production excess-mass quantization, neighbour count, and
  integer share once per source. The second subtracts that share once per
  outgoing incidence and adds each neighbour's published share. It performs
  the same conservative integer debit/credit graph but avoids global atomic
  adds and reduces 24 passes to 16. No dispatch dimensions or pressure code
  change.
- The final form caches each cell's stage-stable active-neighbour degree once
  in dead conditioning plane 5, avoiding seven further degree graph walks.
- Same-source control/candidate density and gamma hashes, scalar summaries,
  pressure residuals, topology receipts, and validation results are bit-exact
  at steps 1 and 19. Candidate repeats are deterministic.
- Strict mini64/min8 artifacts are
  `/tmp/e3c-degree-{control,candidate}-r{1,2,3}.json`, fingerprint
  `2ee814a7...`. Capacity repair improves in all three pairs; its
  median-of-medians falls 3.0802 -> 2.0316 ms (**-1.0486 ms / -34.0%**).
  Surface sharpening falls 5.8327 -> 4.7841 ms (**-18.0%**) and non-pressure
  falls 17.3670 -> 16.8428 ms.
- mini16 also wins: capacity 4.2598 -> 2.6870 ms, surface 6.2259 ->
  4.5875 ms, and non-pressure 16.1874 -> 13.5004 ms.
- A strict ordinary-adaptive mini64 repeat at fingerprint `b12a8e03...`, with
  54,916 accepted cells, ties total surface at 6.0948 ms but regresses capacity
  3.4079 -> 3.6045 ms and non-pressure 28.5082 -> 29.0324 ms. The earlier
  mixed-rung pair was discarded because its source fingerprints differed.
- Decision: do not promote universally. The gather representation proves a
  large deterministic win for small accepted graphs, including both matched
  microscopes, but has not cleared the non-regression requirement for large
  mixed-rung graphs. Next test a topology-derived cost selection or share the
  cached degree/tile image with other stages; do not key selection to scene,
  region, or named rung. No regression promotion was run for this QA-only arm.

## Measurement and acceptance protocol

### Timing protocol

- Run Dawn serially; never overlap the suite, a probe, or the browser's Dawn
  process.
- Use same-source paired arms. Record commit, dirty status hash, and source
  content fingerprint.
- Use B8/P8, scene cadence, gamma and sharpening on, 8 warmups, 11 measured
  stable-epoch frames for the mini16/mini64 pair.
- Repeat each arm three times in alternating order. Report all replay medians,
  the median of medians, p95, and timestamp quantum.
- Keep mini64 ordinary-adaptive and a mixed-rung scene as generality controls.
- Treat changes smaller than two timestamp quanta as noise unless replay
  distributions clearly separate.
- Record stage and work-chunk timing, accepted work, traversed capacity,
  allocation bytes, and explicit-exception frequency.

### Correctness protocol

Before considering a result successful, require:

- deterministic raw-field hashes across repeated runs of each arm; a small
  intentional sharpening difference from control may be acceptable, but
  run-to-run variation is not;
- exact accepted cells/rows and topology generations;
- exact density, gamma, momentum/velocity, and pressure-input hashes at the
  selected checkpoints;
- exact authority, conservation, symmetry, and validation receipts;
- exact exception-path coverage accounting;
- no new validation errors or fallback faults;
- no pressure-solve source changes and no unintended pressure-input changes.

If a proposed arithmetic path intentionally changes floating-point evaluation
order, it is a separate numerical experiment and is out of scope until the
bit-exact structural form has been exhausted.

### Regression promotion

A timing result is not `accepted` until regression coverage passes.

1. During development, run focused CPU/unit checks and the experiment's
   bit-exact Dawn oracle.
2. For a promising candidate, run relevant canonical lanes with
   `npm run test:dawn:sparse-cm12 -- --lane=<id>`.
3. At minimum, structural/transport changes should cover
   `symmetric-expansion`, `hydrostatic-adaptivity`, `mini32-correctness`,
   `mini32-performance`, and `mini64-performance`. Add
   `mini64-min8-surface` and `min8-region-surface` for sharpening,
   presentation, or min8-visible work. Add live insertion, rigid coupling,
   terrain, and far-wall lanes when the changed authority can affect them.
4. After a large accepted simulation, topology, presentation, boundary, or
   live-edit change, run the complete canonical gate:

   ```bash
   npm run test:dawn:sparse-cm12
   ```

Do not raise a timing ceiling or weaken a correctness lane to accept an
experiment.

## Reproduction commands

Run these sequentially, never concurrently:

```bash
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
node --import tsx tools/probe-sparse-cm12-stage-cost.ts \
  --scene=mini16 --brick-fine=8 --presentation-page=8 \
  --warmup=8 --frames=11 --capture-gap-ms=110 --time-step=scene \
  --final-qa=0 --quiet=1 --out=artifacts/<experiment>-mini16.json
```

```bash
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
node --import tsx tools/probe-sparse-cm12-stage-cost.ts \
  --scene=mini64 --brick-fine=8 --presentation-page=8 \
  --minimum-cell-size=8 --region-scope=domain \
  --warmup=8 --frames=11 --capture-gap-ms=110 --time-step=scene \
  --final-qa=0 --quiet=1 --out=artifacts/<experiment>-mini64-min8.json
```

## Result entry template

Append one section per implementation attempt:

```markdown
### E#: short name / YYYY-MM-DD

Status: running | accepted | rejected | inconclusive

- Hypothesis:
- General algorithm/data-structure change:
- Files/source fingerprint:
- Control artifacts:
- Candidate artifacts:
- Exactness receipts:
- Work removed or transformed:
- mini16 medians:
- mini64/min8 medians:
- ordinary-adaptive/mixed-rung controls:
- Focused regression lanes:
- Full regression suite:
- Decision and reason:
- Follow-up:
```
