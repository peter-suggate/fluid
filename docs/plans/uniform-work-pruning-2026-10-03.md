# Uniform work pruning

This follow-up targets the work outside pressure in the dynamic 4h-first Uniform Geometric method. The retained changes are coarse transport row batching, sharpening launch budgets from the actual compacted work list, and removal of empty seam staging from regular sharpening jobs. Larger changes should remove passes or field publication, with matched-field evidence before promotion. Keep the same architecture at zero, partial and full h detail, direct buffered launches, current solver tolerances and current fidelity controls.

## Measured cost

The 128³ dam break with 32,768 coarse owners took 8.09 ms of GPU simulation time in the instrumented 120-frame run. These are mean stage costs, after eight warmup frames. They exclude rendering. The trace included the subsequently discarded transport edge-packing experiment, and the Uniform source fingerprint changed during that run. Use this snapshot to prioritize stages, not as the final implementation's speed claim.

| Work | GPU time |
| --- | ---: |
| Transport normalization and gather | 0.805 ms |
| Transport dependency lists | 0.126 ms |
| Extension and hierarchy continuation | 0.652 ms |
| Sharpening sweeps | 0.534 ms |
| Sharpening geometry | 0.133 ms |
| Surface redistance | 0.531 ms |
| Surface advection | 0.393 ms |
| Global surface volume correction | 0.349 ms |

The timing source is `artifacts/uniform-coarse-performance-2026-10-03/work-pruning-trace-fixed.json`. Paired throughput checks alternate execution order between two solvers on one GPU, with two frames in flight. They compare final volume, velocity and phi hashes. Report both the aggregate mean and median paired change: occasional timing spikes materially move the mean. GPU execution is serialized through the repository lease.

## Current implementation

Transport previously assigned 64 lanes and two barriers to each coarse row, including the common nine-edge row. Four rows now share a workgroup, with sixteen lanes each. Every row uses its actual donor count, including the 126-edge case at an h interface. Shared storage preserves the original float32 rounding boundary and ascending sum order. Exact integer donor accumulation is unchanged. There is no occupancy-selected operator.

The ownership dispatch helper accepts the number of jobs packed into a group. Its budget remains evidence-based, with 25 percent headroom, bounded decay and at least one group; while a GPU relayout owns the layout it does not decay below a reserve of 1,024 groups (`UNIFORM_WORK_RELAYOUT_RESERVE`), since the census can admit tiles two frames ahead of any evidence. Attaching the GPU relayout discards the evidence: budgets go to their ceilings and receipts from before the attach are ignored, since a budget left at one group runs the census's first h tiles serially for the two frames of receipt lag. The kernel strides the current GPU count, including partial batches and sudden detail growth. The reference shader and dispatch are confined to the QA helper in `tests/helpers/uniform-transport-workgroup-reference.ts`.

Sharpening now returns two work counts through eight extra bytes in the existing frame receipt: active regular owners and coarse seam owners. They size subsequent sweep launches using the existing deterministic two-frame lag. Geometry keeps the ownership budget. All current jobs are still processed when the estimate is too small; the receipt is never an admission limit. The first coarse comparison reduced the eventual sweep launch from 214 to 33 groups, with exact fields, but only a 0.52 percent median throughput improvement. Treat that as a small scheduling gain.

The sweep wrapper now selects regular-owner or coarse-seam processing from the current GPU job count. Previously every regular job first staged an empty seam array; limit and commit each paid a barrier before returning from that empty routine. The partition removes those sixteen empty seam barriers per regular job over eight sweeps. It retains the trailing job barrier, all three dispatches per sweep, every face proposal and the original arithmetic. There is no host occupancy switch and no new work admission rule. The old wrapper is preserved in `tests/helpers/uniform-sharpen-job-reference.ts` for paired checks.

The [retained benchmark evidence](uniform-work-pruning-2026-10-03.json) includes every timed block, hashes and source fingerprints. All five paired runs below kept a stable Uniform source fingerprint and matched final fields exactly. Positive changes mean faster execution.

| Change and fixture | Mean reference → candidate | Mean reduction | Median paired reduction | Faster blocks |
| --- | ---: | ---: | ---: | ---: |
| Four-row transport, coarse 128³ | 7.511 → 7.274 ms | 3.16% | 3.52% | 11/12 |
| Four-row transport, dynamic 32³ | 6.862 → 7.009 ms | −2.14% | 0.88% | 7/8 |
| Four-row transport, dynamic 32³ repeat | 7.077 → 6.955 ms | 1.73% | 0.66% | 6/8 |
| Sharpening receipt, coarse 128³ | 7.577 → 7.408 ms | 2.23% | 0.52% | 10/12 |
| Sharpening receipt, full h 32³ | 4.568 → 4.403 ms | 3.61% | 1.68% | 5/8 |

These changes are measured separately; do not add their percentages. The first mixed transport run has one large candidate timing spike, so its median and mean disagree. The repeat has spikes in both directions, with a 0.66 percent median gain. Across both runs, aggregate time is essentially neutral; the typical block is slightly faster. Treat this as avoiding the scalar version's mixed-detail regression, not as a substantial mixed-detail speedup. Sharpening's benefit is modest and noisy; the useful infrastructure result is that its launch width follows actual work, including shrinkage to one group in the quiet full-h fixture.

## Sharpening follow-up results

The job-partition change passed exact stage parity and the full repository gate. Its benefit is small. All four paired runs below kept stable Uniform source fingerprints and exact final fields. Positive changes mean faster execution.

| Fixture | Mean reference → candidate | Mean reduction | Median paired reduction | Faster blocks |
| --- | ---: | ---: | ---: | ---: |
| Coarse 128³, 12 × 30 steps | 7.273 → 7.127 ms | 2.02% | 0.56% | 10/12 |
| Dynamic 32³, 12 × 30 steps | 6.670 → 6.968 ms | −4.47% | −0.98% | 2/12 |
| Full h 32³, 12 × 30 steps | 4.399 → 4.227 ms | 3.92% | 0.21% | 7/12 |
| Dynamic 32³ repeat, 16 × 60 steps | 6.538 → 6.480 ms | 0.88% | 0.19% | 11/16 |

Timing spikes again distort the means. The initial dynamic run was typically about one percent slower; the longer repeat was essentially neutral, slightly faster in eleven of sixteen blocks. Full detail was also effectively neutral. Retain this as a small removal of redundant work, not as a substantial throughput gain. These fixtures do not establish rendered frame rates across scenes.

The strict stage fixture compares both ping-pong outputs and dust accounting after every pair of sweeps. It covers all coarse, one coarse island, mixed, scattered, all h and mixed-again ownership; all three sharpening policies; solids absent/present; dirty buffers; and an intentionally undersized one-workgroup launch. The production regression test also compares against the older unpacked seam traversal, so partition changes cannot hide behind the current seam implementation.

Follow-up validation passed: types, 863 CPU tests (54 GPU skips), and all 50 Dawn files. Logs are `types-sharpen-jobs-final.log`, `unit-sharpen-jobs.log`, `dawn-sharpen-jobs.log` and `sharpen-jobs-parity.log` under the artifact directory. No thresholds or numerical policies changed. The earlier validation section below records the preceding transport/receipt change separately.

## Experiments excluded from production

- Compact positive-edge lists and packed coarse records preserved fields but produced no reliable throughput gain. Their additional indexing and allocation logic was removed.
- One lane per coarse row, with explicit storage to preserve rounding, improved the typical coarse block by about 3.1 percent but regressed the mixed-detail fixture by about 5.9 percent. It was replaced by four-row batching. The earlier version without the rounding boundary also failed exact parity and eventually rejected pressure; it is not a valid speed comparison.
- Smaller pressure hierarchy, reserve-cycle and surface deferred-launch grids did not establish a repeatable overall win. Those runtime experiments were removed.
- Exact owner-level transport closure remains a research tool. In a frozen evolving mixed-detail fixture, it removed only 254 of 9,206 owners from the broad normalization closure, about 2.76 percent. A stationary one-wet-owner example improves from 64 owners to one, but that synthetic case does not justify a general speed claim. The census uses one common frozen volume/departure input for both graphs; it is not a reconstruction of the actual preceding transport dispatch.

## Next implementation targets

### Sharpening proposal/limit fusion: parked after parity failure

The QA experiment in `tools/uniform-sharpen-fusion-experiment.ts` recomputes canonical proposals during limiting, stages them in shared float32 storage, and publishes positive faces for the unchanged commit. It reduces eight sweeps from 24 to 16 dispatches. The coarse 128³ paired trajectory matched final fields exactly, but its median improvement was only 0.43 percent (five of eight faster blocks); one slow reference block inflated the mean reduction to 3.31 percent.

The stronger dirty-scratch fixture failed in clear coarse ownership, orphan policy `[0, 2]`, by the second sweep. Negative dust can yield a negative give budget. A quiet lower owner whose proposal was explicitly initialized to zero can therefore produce a nonzero proposal if evaluated again against that negative neighbour. The fused limiter then sees a different proposal from the one commit reads. A shared rounding boundary does not solve this membership mismatch.

Exact fusion needs persistent per-owner producer activity, established during geometry preparation and retained for all sweeps. Checking only current budgets is insufficient. The current compacted active list does not provide an inexpensive arbitrary-owner membership lookup; a bitmap would add construction, storage and reads. Given the weak measured gain, the fusion remains QA-only and must not be promoted or cited as a validated speedup. The failure log is `artifacts/uniform-coarse-performance-2026-10-03/sharpen-fusion-parity.log`.

The smaller follow-up uses the existing GPU job partition to avoid entering seam staging for regular-owner jobs. This changes scheduling only; proposal, limiter and commit arithmetic stay intact. Its acceptance and timing results are tracked separately from fusion.

### Publish extension only where consumers need it

First measure the faces actually sampled by surface tracing, momentum and other extension consumers. The current extension constructs the coarse nearest-source hierarchy and publishes canonical faces. A proposed consumer ABI could retain physical supported faces and query the same hierarchy on demand elsewhere, avoiding unused publication and intermediate writes.

The request footprint must include actual departures, interpolation stencils, negative wall planes and refinement seams. A fixed neighbourhood is insufficient for long travel. Carry field and ownership generations so an absent or retired detail slot resolves to the authoritative coarse continuation rather than stale storage. Preserve source bounds, tied-nearest-source averaging and closed-wall values.

Prototype one consumer against captured extension inputs before changing every consumer. Count omitted publication, extra hierarchy taps and directory resolution together. If coarse consumers touch nearly the whole base grid, optimize reuse of those taps before adding another compacted list. No speedup is established for this proposal yet.

### Prune surface work using its actual search predicate

Split the redistance measurement into retirement evidence, distance transforms, window staging and searches. The current fine kernel already keeps off-band vertices and avoids staging a window when no lane searches. A new list must improve on that existing pruning.

Use the current `umRebuildBand` predicate and the full stencil footprint to construct search jobs. Temporal reuse must be invalidated by changed phi, volume, velocity, sources, solids, ownership and the affected halo. Specify how untouched output vertices retain valid values across ping-pong fields; skipping writes alone leaves stale data. Start with same-frame search compaction, then consider reuse only if the invalidation census shows enough stable work. Validate promotion, retirement, long characteristics and moving contact explicitly.

### Revisit owner-level transport closure only with stronger evidence

Keep tile workgroups for cooperative fine donor reduction, but consider owner masks inside listed tiles if sparse fine fixtures show a substantial removable fraction. Masked lanes must still reach every workgroup barrier. Preserve the self fallback, source seeds and sealed-cell reservoirs. The graph-building and mask traffic must be included in timing. The measured mixed fixture does not currently support prioritizing this over extension publication or redistance search compaction.

## Reproduction and acceptance

Run `node --import tsx tools/benchmark-uniform-buffered-work-dawn.ts --reference=transport --blocks=12 --out=/tmp/transport.json` for the all-coarse 128³ trajectory. Add `--scene=minimal-power-dam-break-32 --policy=dynamic` for changing detail. Use `--reference=sharpen --policy=full --scene=minimal-power-dam-break-32` to isolate the sharpening receipt. `--reference=sharpen-jobs` compares the current sweep wrapper against the previous unconditional seam staging. `--reference=shared` is a same-policy noise control. `--work-census` adds the frozen source-free graph comparison outside timing.

The targeted GPU test uses dirty scratch across coarse, mixed and full h layouts, partial four-row batches, fractional and out-of-domain departures, and a deliberately stale one-group budget. The frozen reference must actually compile; a missing shader match is a test failure.

The required repository gate remains `npm run check:types`, `npm run test:unit`, and `npm run test:dawn`. Run Dawn serially and never with a browser or another GPU process. Full-render 60 fps across scenes is still unproven; simulation-only measurements cannot establish it.

## Previous transport and receipt validation

This pruning change passed all three repository gates: `npm run check:types`; `npm run test:unit` with 863 passing tests, 54 skipped GPU-gated tests and no failures; and `npm run test:dawn` with all 50 files passing. The GPU run includes the new partial-batch and stale-budget transport parity test, dynamic coarsening, sharpening seams, geometric boundaries, live solid edits, rigid bodies, strict pond rest, numerical invariants and production publication checks. No solver tolerance or timing ceiling was relaxed.

The retained JSON contains per-file Dawn results as well as the paired timing evidence. Detailed local logs are under `artifacts/uniform-coarse-performance-2026-10-03/` (`types-final.log`, `unit-final.log`, `dawn-final.log`).

A concurrent edit to `uniform-detail-fields.ts` landed near the end of the full suite. Types, all unit tests and transport GPU parity were subsequently rerun and passed. The paired timing evidence predates that edit; the full 50-file suite was not repeated for the other session's late edit.
