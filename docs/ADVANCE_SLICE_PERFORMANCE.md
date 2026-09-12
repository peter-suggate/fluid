# Advance Slice CPU performance

`tools/benchmark-advance-slice-performance.ts` measures the CPU implementation
used by the 2-D Adaptive Lab. Each recorded repetition creates a fresh scene,
advances two untimed warmup frames, and times four frames with 16 pressure
iterations. The default matrix covers an adaptive open-top pool, a closed-wall
dam collision, and the 128 x 128 CM12 pressure embedding. The report also hashes
all reachable typed-array state and relevant scalar receipts at frames 1, 3,
and 6.

Run the standard five-repetition measurement with:

```bash
node --import tsx tools/benchmark-advance-slice-performance.ts \
  --warmup=2 --frames=4 --repetitions=5 --pressure-iterations=16 \
  --output=/tmp/advance-slice-performance.json
```

Run an alternating-order comparison of eager diagnostic publication against
the normal deferred path with:

```bash
node --import tsx tools/benchmark-advance-slice-performance.ts \
  --paired-diagnostics --warmup=2 --frames=4 --repetitions=5 \
  --pressure-iterations=16 --output=/tmp/advance-slice-paired.json
```

For the net simulation A/B, two immutable copies of the same source snapshot
were used. The control copy restored unconditional intermediate publication and
transport receipts, per-value typed-array bitcasts, and the pressure-row cell
scans. The optimized copy retained all simulator changes. The copies differed
only in `slice-solver.ts` and `slice-pressure-embedding.ts`. Three fresh Node
process pairs ran in B/A, A/B, B/A order; each process used two warmup frames,
four measured frames, and 16 pressure iterations.

On Node 22.22.1 (Darwin arm64), that combined comparison measured:

| Scene | Control samples (ms/frame) | Optimized samples (ms/frame) | Median control | Median optimized | Ratio |
| --- | --- | --- | ---: | ---: | ---: |
| coarse-first-pool-impact-half | 172.087, 173.583, 173.270 | 169.194, 171.544, 168.527 | 173.270 | 169.194 | 1.024x |
| cm12-figure-3 | 789.261, 754.758, 758.749 | 737.916, 710.940, 740.155 | 758.749 | 737.916 | 1.028x |

The ending state hashes matched across all six runs for each scene. This is the
net measurement for deferred publication and transport receipts together with
the pressure scan and bitcast changes.

The separate alternating comparison below isolates diagnostic publication;
the pressure optimization is enabled on both sides:

| Scene | Eager diagnostics | Deferred diagnostics | Ratio |
| --- | ---: | ---: | ---: |
| coarse-first-pool-impact-half | 168.710 ms/frame | 165.403 ms/frame | 1.020x |
| twin-dam-collision | 12.692 ms/frame | 12.354 ms/frame | 1.027x |
| cm12-figure-3 | 822.779 ms/frame | 824.304 ms/frame | 0.998x |

The five diagnostic pairs alternated which mode ran first. Both modes ran in
the same process from the same loaded source and fresh identical seeds. The
ending state hash was identical between modes for every scene. CM12 had large
process noise and is dominated by pressure work, so its 0.998x ratio is no
measured diagnostic-publication change.

The retained changes remove dense public-state publication from intermediate
stages when there is no observer, avoid building transport receipt snapshots
without a transport observer, remove unused lattice sums from each render, and
replace repeated pressure-row cell scans and per-value bitcast allocations with
one preparation scan and one reusable four-byte view.

Removal of unused lattice sums is render-loop cleanup and is not included in
the simulation timings above.
