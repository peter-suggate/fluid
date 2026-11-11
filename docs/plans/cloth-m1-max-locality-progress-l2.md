# Cloth locality, lane L2: Experiment 2 (compiled collision topology and round geometry)

Plan: `docs/plans/cloth-m1-max-locality-2026-11-11.md`, Experiment 2. Running log; newest section last.

## 1. Attribution of the query kernels as they stand (before any code)

Tree read: HEAD 67cb29fd with lane C2's uncommitted lambda-at-x_n hunks in `ogc-detector-wgsl.ts`.
Counters: the 2026-11-11 profile (`fluid-scratch/sphere-profile-2026-11-11/simulation.json`), 150^2 rotating sphere.

Per step, mean of 10 steps:

| tick | rounds | scanned | key mismatch (hash collision) | box rejected | candidates | DAT pairs | contacts (vf+ee) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 4.3 | 52.7 M | 14.3 M (27%) | 31.9 M (61%) | 3.6 M (7%) | 0.1 M | 0 |
| 35 | 23.3 | 352 M | 65.7 M (19%) | 236 M (67%) | 41.1 M (12%) | 2.4 M | ~10 k |
| 120 | 30.1 | 753 M | 158 M (21%) | 502 M (67%) | 85.6 M (11%) | 6.9 M | ~63 k |

The rest (about 1%) are records that pass the box test and yield no candidate (a face owning no edge, an adjacent
primitive). `candidates` counts owned edges in the edge role, so it is not one per accepted record.

The grid: H = 2^18 buckets, at most 1.09 M entries of 8 words (32 bytes, 35 MB) a round, so a bucket holds about 4
records. A query thread scans about 250 records a round (753 M / (30 rounds x ~100 k query threads)), so it visits on
the order of 60 cells: the per-cell cost matters as much as the per-record cost.

What each outcome loads today (queryEdge / edgeEdge, the dominant role; hullEdge, sampleFacet and hullFacet alike):

- Per visited cell: `bucketStart(h)` and `bucketStart(h + 1)`, each three loads (`local1[h]`, a 1 MB table read at a
  hashed index; `local2[h >> 10]`, 256 words; `local3[h >> 20]`, 1 word): 6 loads, 2 of them on a line the hash picks.
  Then the bucket's first record, wherever the prefix sum puts it in the 35 MB entry array. With about 4 records a
  bucket this is 6 loads beside 4 to 32 record words: the cell lookup is roughly a fifth of the scan's loads.
- Key mismatch (21%): 1 word of the record (the key), no dependent load.
- Box reject (67%): the record's 8 words (face, key, box lo, box hi), one contiguous 32-byte record. No dependent
  load: the entries already carry their boxes. The second half of the test, `cellOf(max(q.lo, box.lo)) != c` (the
  duplicate-ownership rule: a pair is processed in the one cell holding the min corner of the boxes' intersection),
  is arithmetic on the same words.
- queryVertex only: `triVerts(t)` (3 topology words, dependent on the record's face) is loaded BEFORE the box test, for
  the `hasVertex` skip. So in the vertex role every record that passes the key costs one dependent topology load.
  Reordering would change `boxRejected` (an adjacent face that also fails the box is not counted today), so the order
  is kept; with a face record the load is the same single one.
- Box pass in the edge role: `te` = `topo[triEdges + 4t + 3]` (dependent on t), then per owned edge (1.5 a face on
  average) `e2` = `topo[triEdges + 4t + k]`, `edgeRec(e2)` (4 words, dependent on e2), the endpoint-sharing test, then
  `xd(r2.x)`, `xd(r2.y)` (dependent on the edge record): a chain of depth 4 (entry -> triEdges -> edge -> x_det).
- Candidate in planar mode with e < e2 (half of them): `datPairEE` reads, before its first distance test, 4 `predOf`
  (each with two range branches), 4 `strayOf` (with a moving solid: `work[heldOff + i]` and `work[bodyOff + 8i + 7]`
  each), 2 `envOf` (`work[envOff + i]`): per endpoint the data sits in four places (x_det, pred, and two or three
  separate regions of `work`, an atomic array), about 20 loads over 4 vertices, and the own edge's two endpoints are
  re-read for every candidate (the bindings are read_write or atomic, so the compiler cannot keep them).

So: the REJECTION path carries no dependent chain except the vertex role's `triVerts`. What rejections cost is the
cell lookup (6 loads, 2 random) and a 32-byte stream per record. The chain the plan names
(entry -> face -> owned edge ids -> endpoints -> positions) is on the box-PASS path only: about 12% of scanned
records, but each costs 15 to 40 scattered loads and the segment-segment arithmetic, against 1 to 8 contiguous words
for a rejection. Which side dominates the time is not decidable by reading; it is what variants a/b (pass path) and c
(per-cell path) separate. Inferred, not measured: threads of a 64-wide workgroup diverge between the two paths, so the
pass path's latency may also be paid by neighbours that are rejecting.

Design consequence:

- (c) final bucket offsets attacks the per-cell cost, which rejections share: 6 loads -> 2 adjacent loads.
- (a) face records and (b) round geometry records attack the pass path: chain depth 4 -> 2, and 4-5 scattered
  per-vertex loads -> one 32-byte record from a read-only binding (which also lets the compiler keep the own edge's
  values across candidates).
- Nothing additive shortens the 32-byte rejection stream itself; shrinking or splitting the entry would change the grid
  entry layout, which lane C2's verification pass reads (ruled out by main for this experiment).

## 2. Design (all additive; coordinator constraints)

No param or counter word is taken, no shared WGSL string or buffer changes meaning, the grid and its entry layout are
untouched. Two new buffers on two new bindings, used only by kernels the switch selects:

- binding 24, `ogc locality` (u32): `[0, H]` the round's final bucket offsets (H + 1 words), then at word `H + 8`
  the face records, 8 words a face (static; written at build, so a cut, load or undo, which rebuild the detector,
  rebuild them).
- binding 25, `ogc round geometry` (vec4<u32>, two per id): one record per vertex and per solid sample.

Switch: `OgcContactConfig.locality` (document `solver.contactLocality`), a `+`-joined subset of
`topology`, `geometry`, `offsets`, or `off` (default). Unknown parts fail at validation and at detector construction.

Face record (a), 8 words: `v0 v1 v2 flags e0 e1 e2 0`. flags bits 0-2: the face owns edge k (it is the edge's
face0, the existing mask); bits 3-5: edge k's table order is reversed against the face's (its v0 is the face's
vertex k + 1), so the endpoints come out in exactly the edge table's order (canonical pair order and bitwise results
depend on it); bits 6-8: the face is the owner of its vertex k (the first of the vertex's ascending face list:
hullFacet's rule). 1.42 MB for 44,402 faces.

Round geometry record (b), 32 bytes per id (N vertices, then the static and the moving solid's samples):
`(X.xyz, envelope) (w.xyz, sigma | moving-with-solid << 31)`: x_det, the envelope, the prediction, the stray and
held bit 6, copied bit for bit after prep by one pass (`locGeometry`, on prep's indirect slot). 22,500 vertices:
0.72 MB plus the samples. Per vertex, not per edge or face: a face record already gives the endpoint ids, so the
depth is the same and the hot data stays 0.7 MB instead of 4.3 MB (edges) duplicated.

Final offsets (c): `locOffsets`, one pass on scan1's indirect slot after scan3, writes
`off[h] = local1[h] + local2[h >> 10] + local3[h >> 20]` and `off[H] = total`; the queries read `off[h]`,
`off[h + 1]`. 1.05 MB. The prefix tables stay, and the insert pass still uses them.

## 3. What was built

Switch: `OgcContactConfig.locality` (`ClothSystemOptions.contactLocality`, `createClothSceneContact(..., locality)`,
benchmark `--locality`), default `off`; `parseOgcLocality` rejects anything but `off` or a `+`-joined subset of
`topology`, `geometry`, `offsets`. With the switch off the kernel sources, bind groups and buffers are the ones before
this lane (asserted: `ogcQueryWGSL(role)` and the binding list are unchanged, no buffer is allocated).

- (a) `topology`: `buildOgcFaceRecords` (`ogc-topology.ts`), written once at detector construction at word `H + 8` of
  the locality buffer (binding 24). The detector is rebuilt on a cut, load and undo, so the records are too.
- (b) `geometry`: `OGC_LOC_GEOMETRY_WGSL`, one pass a round on prep's indirect slot, writes binding 25.
- (c) `offsets`: `OGC_LOC_OFFSETS_WGSL`, one pass a round on scan1's indirect slot after scan3, writes `[0, H]` of
  binding 24.

Bytes at 150^2 (N 22,500, T 44,402, 14,767 solid sample slots): offsets 1,048,580; face records 1,420,864; round geometry
1,192,544. (`topology` alone also allocates the unused H + 8 offset words, 1.05 MB, since both share binding 24.)

Two deviations from "every per-vertex read goes through the record", both forced by bit equality (Metal contracts
and folds float expressions by their source form, so equal bits loaded from a different buffer can give a different
last bit):

- The least-shift bisection (`shiftGap`) reads the live `x_det` / prediction buffers under `geometry`. Through the
  record, 4 of 220 vertices in a test case came out 1-2 ulp apart in the shift (prediction, envelope, cap).
- `resolvePlane`'s support projection `dot(n, (x + w) - o)` reads them live too (`o` is read from `x_det` there).
  Through the record a conflict test flipped at a tie: on the 16 x 16 H_vv rotating sphere (bit-deterministic with
  the switch off) tick 4's mandatory round took one redo fewer, the trajectory left the baseline and failed as
  heldConflict at tick 46 (two ladder tests failed under the switch). Bisected by putting one read at a time back on
  the live buffers: x alone or the prediction alone restores equality; envelope, sigma and the with-solid bit do not
  matter; the call site is this projection alone. Storing the record as f32 instead of bitcast u32 did not help.

Both run per DAT pair with a moving vertex (or per held vertex), not per scanned record, so binding 14 stays bound and
the per-record path is untouched. With both, the 60-tick trajectory of that scene is bit-identical to off.

## 4. Equivalence

A finding first: the BASELINE is not bit-reproducible in its contact buffer, off against off, at a fixed state. The
DAT pair buffer is filled by racing appends, its order differs run to run, and the plane records and sums follow that
order. What is reproducible at a fixed state: every counter, the pairs as a multiset (with their plane bits), the
contact entries as a multiset, and the whole `work` buffer. So equivalence is asserted on those, on single detections
and on a real step's mandatory rounds (the round log up to the first round with `filters > 0`), not on trajectories.

`tests/cloth-contact-locality.test.ts` (GPU; run in clean exports of 67cb29fd, da0acf8e and 81a179a6, all pass):

1. Detector level, 4 fixed states (a held stack; isotropic truncation with an armed re-detection; crossing inertial
   predictions with redo rounds and 646 conflicts; a sheet over a box and a ball with solid contacts and sample-facet
   pairs) x 5 variants (`topology`, `geometry`, `offsets`, `topology+geometry`, all three): counters, sorted contact
   entries, DAT pair multisets with plane bits, x_det / prediction / envelope / held bits all equal to off.
2. System level with a moving solid: one real step from a saved endpoint per variant, mandatory rounds equal to off.
3. Cut, load, redo, undo under the full switch: the face records read back from the GPU equal
   `buildOgcFaceRecords(det.topology)` after each rebuild; the next step's mandatory rounds equal off.
4. A whole trajectory where the baseline has one: the 16 x 16 H_vv rotating sphere (bit-deterministic off, with redos
   and moving-solid contacts) is bit-identical in positions and every integer counter over 60 ticks under `geometry`
   and under the full switch.

At scale (150^2 rotating sphere, saved states at ticks 2 / 35 / 120, one real step each; exports of 67cb29fd and of
81a179a6, the latter with both live-read fixes): the
mandatory rounds and the whole-step counters are equal to off for off-again and all five variants, 0 GPU errors.
Tick 120: 14 mandatory rounds, scanned 480,966,347, key collisions 103,705,326, box-rejected 322,893,205, candidates
51,620,593, DAT pairs 3,801,413.

In the timing runs below every variant's per-step counters (scanned, collisions, box-rejected, candidates, DAT pairs,
detections, re-detections, redos, solid/body contacts, vf/ee pairs, pair entries, held) equal off's for 10 of 10 steps
at all three ticks.

## 5. Timing (export of HEAD 81a179a6 plus only this lane's hunks; one run each, back to back; 10 steps a tick)

150^2 rotating sphere, `tools/benchmark-cloth-sphere.ts --ticks 2,35,120 --steps 10 --profile 1 --locality <v>`.
Step = wall median of the 10 timed steps; query = the profiled step's queryVertex + queryEdge + querySample (ms).
Every variant's per-step counters equal off's for 10 of 10 steps at every tick.

| tick | variant | step ms | qV | qE | qS | query sum | prep pass(es) ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | off | 140.1 | 1.2 | 4.1 | 1.1 | 6.5 | - |
| 2 | topology | 146.5 | 2.1 | 4.1 | 1.0 | 7.2 | - |
| 2 | geometry | 140.5 | 1.7 | 4.8 | 0.8 | 7.3 | 0.33 |
| 2 | topology+geometry | 151.9 | 1.6 | 3.0 | 0.7 | 5.4 | 0.39 |
| 2 | offsets | 152.2 | 1.9 | 5.5 | 1.0 | 8.4 | 0.33 |
| 2 | all three | 164.2 | 1.3 | 3.2 | 0.9 | 5.4 | 0.39 + 0.52 |
| 35 | off | 532.9 | 46.1 | 157.9 | 27.4 | 231.4 | - |
| 35 | topology | 540.0 | 45.7 | 159.1 | 27.1 | 231.9 | - |
| 35 | geometry | 460.8 | 33.2 | 110.0 | 21.8 | 164.9 | 0.72 |
| 35 | topology+geometry | 461.5 | 33.6 | 103.4 | 19.9 | 156.9 | 0.33 |
| 35 | offsets | 821.0 (*) | 44.9 | 157.9 | 27.8 | 230.6 | 0.52 |
| 35 | all three | 460.5 | 33.8 | 105.3 | 18.9 | 158.0 | 0.72 + 0.98 |
| 120 | off | 866.3 | 87.9 | 318.2 | 55.9 | 462.1 | - |
| 120 | topology | 874.8 | 91.0 | 318.9 | 54.8 | 464.7 | - |
| 120 | geometry | 721.3 | 64.7 | 220.9 | 43.9 | 329.4 | 0.79 |
| 120 | topology+geometry | 723.0 | 65.6 | 206.6 | 38.9 | 311.2 | 0.46 |
| 120 | offsets | 882.5 | 88.6 | 319.8 | 55.8 | 464.1 | 0.46 |
| 120 | all three | 713.9 | 66.1 | 208.6 | 38.7 | 313.4 | 0.59 + 0.39 |

(*) An outlier of the wall clock, not of the variant: its profiled kernels (query 230.6 ms, profile split sum 559.9
ms) equal off's (231.4, 557.7). Not repeated (one run per variant). Tick 2 (query 6 to 8 ms of 140) is within noise.
The same table on 67cb29fd (before C2's verification landed; geometry alone not timed there) agreed within 2%:
topology+geometry 456.3 / 706.5 ms at ticks 35 / 120 against off 537.2 / 862.8.

## 6. Conclusions

- (b) round geometry records pay, alone: query time -29% at both ticks 35 and 120 (231 -> 165, 462 -> 329 ms), step
  -14% / -17% (533 -> 461, 866 -> 721 ms), for one pass of 0.3-0.8 ms a round and 1.19 MB. The pass path's scattered
  per-vertex loads (x_det, prediction, envelope, held, sigma from four places, the last two in an atomic array)
  were where the time went, not the dependent topology chain.
- (a) face records alone do not pay (231.9 / 464.7 against 231.4 / 462.1 ms): with the per-vertex reads still
  scattered, removing triEdges -> edge record hops changes nothing measurable. On top of (b) they add a further
  -5% of query time (qE 110 -> 103, 221 -> 207 ms; sum 165 -> 157, 329 -> 311): small but consistent in both ticks
  and in both exports. 1.42 MB (2.47 MB as allocated alone, since the offsets' words are reserved too).
- (c) final bucket offsets do not pay, alone or on top (230.6 / 464.1 alone; all three 158.0 / 313.4 against
  156.9 / 311.2 for a+b). The 3-level prefix lookups were not on the critical path: the per-cell cost is the
  32-byte record stream, which the hash spreads over 35 MB, not the 6 prefix loads (two of them in a 1 MB table that
  stays cached). Recommend dropping (c).
- Recommendation for main: `topology+geometry` (or `geometry` alone if 1.4 MB and the face-record path are not
  worth 5%). Step time at tick 120: 866 -> 723 ms. Nothing in the rejection stream itself changed (it cannot,
  additively, without touching the entry layout); that is what remains of the query time.

## 7. Tests (export of 81a179a6 plus this lane's hunks)

- `npm run check:types` clean; `npm run test:unit` 2150 tests, 1753 pass, 0 fail, 397 skipped.
- `tests/cloth-contact-locality.test.ts`: 7 of 7 pass (3 CPU, 4 GPU).
- Cloth GPU contact files (`cloth-contact-gpu`, `-body`, `-scene-solids`, `-solid`, `cloth-system`, `-ladder`,
  `--test-concurrency=1`), switch off and again with every detector under the full switch (a preload sets
  `OgcContactDetector.defaultLocality`): off 91 pass / 0 fail, on 84 pass / 0 fail. The one `not ok` in both is the
  pre-existing TODO "a sheet lands on the moving solid and rides it" (C2, bodySqueeze at 0.45 s). Before the
  resolvePlane fix (67cb29fd) the on pass failed the two H_vv ladder sphere tests (section 3); after it they pass.

## 8. Ruling (main, 2026-11-11): (b) and (a) kept, (c) removed

Accepted: the round geometry records are the win; the face records stay for their extra ~5% of query time. The final
bucket offsets (c) are removed from the code: the `locOffsets` pass, its words in binding 24 and the `offsets` token,
which now fails like any other unknown value. Their measurement above (section 5: 230.6 / 464.1 ms of query alone
against off's 231.4 / 462.1 at ticks 35 / 120; 158.0 / 313.4 on top of a+b against 156.9 / 311.2) stands as the
finding: the three prefix-table loads per cell are not on the queries' critical path. With the offsets gone, binding
24 holds the face records alone, from word 0 (8T words, 1.42 MB at 150^2).
