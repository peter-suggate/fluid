# Cloth locality, lane L1: compiled contact linearizations (Experiment 1)

Plan: `cloth-m1-max-locality-2026-11-11.md`, Experiment 1. Running log; newest section last.

## 1. Audit: inputs of the contact product and when they change (before any code)

Tree audited: HEAD 67cb29fd, with C2's uncommitted hunks in `contact/ogc-detector-wgsl.ts`, `contact/ogc-detector.ts`.

The product `(H_c p)_v = sum_entries wr (Mn q) + wrl (Mf ql)` (`ogcVertexContactHvLanes`, ogc-wgsl.ts) is called from
three places, all in gpu-newton-wgsl.ts: `pcgStep` (every PCG dispatch), `pcgSolve` (the one-patch form), `nwModel`
(the trust-region model's `s.H s`, hybrid only). Per entry it calls `ogcEntryLin(v, ogcX(v), entry, params)`.

One Newton iteration encodes, in order (gpu-solver.ts `encodeStep`):

```
nwFaces, nwGradient, pcgStep x (cap + 1) | pcgSolve, [hook filter], nwProject, nwTrial, [nwModel], nwChoose,
nwApply, [hook afterNewtonStep: decide, detection round (indirect), truncate/apply], nwEdgeDuals
```

So the product's first use in an iteration is the first pcgStep and its last is nwModel; the window is
[after nwGradient, before nwApply].

| Input of `ogcEntryLin` / the product | Where it lives | Written by | Changes inside the window? |
| --- | --- | --- | --- |
| iterate x (`ogcX`, with `nwTrialMove == 0` in all three callers) | `positions` | `frame` (step start), `nwApply`, the detector's `apply` (truncation: afterFrame, afterNewtonStep) | No. The line search's trial points are never applied to `positions` before nwApply; the product never sees `nwTrialMove != 0` |
| step-start x_t (`ogcXt`) | `state` segment S_XT | `frame`/`finishStep` (once per step) | No |
| entry words x, y, w (ids, feature, EE parameters) and the per-vertex counts | contact buffer, `countsOff`, `entriesBase` | detection-round kernels only: `prep` (count reset, solid/body entries), the query roles (`pushEntry`), `bound` (sort, `ownSolids`) | No. Rounds run in afterFrame (mandatory) and afterNewtonStep (re-detection, redo). The filter hook's five kernels write plane records, segment words, project records, level words and `alpha` only |
| entry z: lagged friction `mu lambda` (and `params.z`, the cloth-cloth mu that scales a pair entry's z) | entry word z; `ogcContacts[1]` | z with its entry by a detection round; params by the CPU at setup | No. (C2 is moving z's definition to the step start for every entry kind; the record is built from the entry word as it arrives through `ogcEntryLin`, never re-derived, so that change needs nothing here) |
| barrier params r, kappa, friction eps (`ogcContacts[1]`) | contact buffer header | CPU at setup | No |
| static solid field, its samples and hull facets | contact buffer | CPU (`setSolidField`, edit time) | No |
| moving solid poses (`ogcPose(0/1)`), its field, samples, facets | contact buffer pose words | `stepBegin` (afterFrame) copies the step's pose pair; CPU `setBodyPoses` per frame | No |
| topology (`ogcTri`, `ogcEdgeVerts`), patch slot of a vertex (`pcgVertexSlot`) | `topo` | CPU at topology build | No (a topology edit rebuilds the solver) |
| active-set projector P (`nwProjectP`) | `nw` | `nwGradient` (`nwActiveSet`) | Not an input of the contact product: pcgStep applies P to the whole row sum after it. Not compiled |
| the vector p (`ogcDir` -> `pcgDir`/`pcgUGlobal`; `pcgSrc` 1 in nwModel) | workgroup staging / `nw` | every dispatch | Yes: the only thing that changes. Stays a load |

Things that could have changed the linearization mid-solve and do not:

- Line search / Armijo trials: evaluate energies at trial points (`nwTrial`, `ogcContactEnergyOwnedLanes`), never the product.
- Rejected trust-region step, stall: the iterate stays, but the next iteration's nwGradient runs again anyway.
- Re-detection, redo rounds, levelled rounds, truncation: all inside afterFrame / afterNewtonStep, outside the window.
- BDF2 contact restart: decided in `beginStep`/`restartScan`, before the first iteration.
- Newton done: every kernel of the remaining capped iterations exits; the records are not read.

**Invalidation rule.** The coefficients are a function of (positions, x_t, contact buffer, topology). Positions and
the contact buffer's entries can change only at nwApply and in the afterFrame / afterNewtonStep hooks, which are all
outside [nwGradient, nwApply). So: rebuild unconditionally once per Newton iteration, immediately after nwGradient and
under nwGradient's own Newton-done gate; the records are then valid for every pcgStep/pcgSolve dispatch and the nwModel
of that iteration, and are never read in another. No dirty tracking: any event that could invalidate them is followed
by the next iteration's rebuild before the next read. A harness that re-arms Newton and replays kernels (the
benchmark's `replay`) must replay the build after nwGradient too.

## 2. What the existing profile already bounds

`fluid-scratch/sphere-profile-2026-11-11/simulation.log` (older tree): pcgStep per working dispatch was 141 us at tick 2
(no contact entries at all: 0 pairs, 0 solid contacts), 176 us at tick 35, 204 us at tick 120. If the whole growth
over tick 2 is the contact product, contact is at most ~35 us (20%) of a tick-35 iteration and ~63 us (31%) of a
tick-120 one: at ~1580 iterations a step, at most ~55 ms (tick 35) and ~100 ms (tick 120) of the 530 / 827 ms steps.
That is the ceiling for this experiment before any measurement of mine.

## 3. Design

Constraint found: `ogcEntryLin` returns `OgcPairLin` (ids, wc, wlc, wr, wrl, Mn, Mf), not the rank-one factors
(n, kappa'', nl, mu lambda f1/|u|). Storing those would mean editing or duplicating `ogcEntryLin` while C2 is changing
its friction. The records are therefore built only from `OgcPairLin`, so they follow whatever `ogcEntryLin` computes.

Per-vertex capacity is 128 entries (CLOTH_SCENE_CONTACT.contactCapacity), so a fixed slot per (vertex, entry) would be
22,500 x 128 x 96 B = 276 MB at 150^2: not materialized. Records are packed: a count pass (one workgroup per PCG
patch, a workgroup prefix sum over its 256 rows) and a build pass (patch base = sum of earlier patches' totals) give
every row a contiguous run in a pool of authored size. A row whose run does not fit keeps the matrix-free product
(header marks it), so an undersized pool costs time, never correctness.

Two record forms behind `newton.contactLinearization` ('off' default | 'factorized' | 'blocks'):

- factorized, 6 vec4 (96 B) per entry: partner columns (patch lane, or global bit | vertex), wc, wlc, the symmetric
  wr Mn (6 floats) and wrl Mf (6 floats). Product per entry: q = sum wc_k p_k, ql = sum wlc_k p_k, (wr Mn) q + (wrl Mf) ql.
- blocks, 2 vec4 (32 B) per partner term: the symmetric block B_k = wc_k (wr Mn) + wlc_k (wrl Mf) and its column.
  A solid/body field entry is one term (32 B, 2 loads); a cloth-cloth pair entry up to four (128 B, 8 loads).

Both keep entry order within a row; within an entry the association differs from the baseline at rounding level
(wr (Mn q) vs (wr Mn) q; blocks also distribute Mn over the partners).

## 4. Implementation (as built)

- Switch: `newton.contactLinearization: 'off' | 'factorized' | 'blocks'` (default 'off') and
  `newton.contactLinearizationRecords` (pool, vec4 records per vertex, default 32). Unknown values fail in
  `validateClothDocument`, `packClothGPUTables` and the solver constructor. Off: the Newton WGSL and the table layout
  are the same bytes as before (asserted in tests/cloth-contact-linearization.test.ts).
- Storage: all in the existing `nw` buffer, before the control block (which stays last): a header per vertex
  (first record, items, records asked, 0; u32 bits in a vec4f), a total per PCG patch, the pool. No new binding, no
  OGC param or counter word, no control word.
- Kernels `nwLinCount`, `nwLinBuild` (one 256-thread workgroup per PCG patch, every patch a full workgroup), encoded
  right after nwGradient under the same `ctl.x != 0.0` exit. Count: entries and record demand per row, a workgroup
  prefix sum, the patch total. Build: patch base = sum of earlier patches' totals; each row calls `ogcEntryLin` once
  per entry and writes its records; a row whose run would pass the pool end is marked (`0x7f000000`) and
  `nwLinHv` sends it to `ogcVertexContactHvLanes` as before.
- Product `nwLinHv(v)` replaces the matrix-free call at the three sites (pcgStep, pcgSolve, nwModel). Partner columns
  are pretranslated to what `pcgU` takes (patch lane, or global bit | vertex).
- nwLinBuild is not idempotent on its own (it turns the count's headers into pool addresses): always count, then build.

## 5. Measurements

Tree for every number here: clean export of HEAD 67cb29fd plus only this lane's hunks
(`fluid-scratch/cloth-locality/l1/export-67cb29fd`, patch `l1/l1.patch`); no peer hunks. Off and on from the same
export, back to back, one run each. Logs in `fluid-scratch/cloth-locality/l1/` (`bench-*.log`, `replay-*.log`,
`fixed-n*-*.log`, `op-test-2.log`). 150^2 rotating sphere, states from `cloth-locality/states`. The 'off' run
reproduces the coordinator's baseline (GPU span 133.0 / 524.1 / 848.8 ms vs 133.2 / 525.6 / 846.2).

### 5.1 Operator equivalence (GPU, tests/cloth-contact-linearization.test.ts)

Two stacked sheets (819 vertices, 4 patches) over a static box and a sliding, turning plate, friction on, 14 real
steps; then the same pcgStep dispatch 0 at the fixed state through the records, with every row forced back to the
matrix-free product, and with contact emptied.

| run | entries (solid / body / VT / EE / sample) | rows | max abs dw | worst dw / (contact share + row product) |
| --- | --- | --- | --- | --- |
| factorized | 191 / 187 / 768 / 2228 / 474 | 539 | 1.4e-9 | 2.6e-7 |
| blocks | 194 / 190 / 780 / 2376 / 582 | 532 | 1.9e-9 | 3.2e-7 |
| factorized, pool of 1 record a vertex (20 rows compiled, 537 matrix-free) | 184 / 177 / 906 / 2476 / 438 | 557 | 4.7e-10 | 6.6e-8 |

eps32 is 1.19e-7: the worst row is under 3 eps of its own magnitude; no row past the 64 eps bound the test asserts.
Every entry kind carries lagged friction on some rows. Rows without entries are bit-identical. No hull-facet (w = 2)
entry arose in this scene: that kind is not covered (it goes through the same `ogcEntryLin` output).

### 5.2 Fixed-state step equivalence (one step from the saved state, Newton capped at 1 and at 2 iterations)

| tick, cap | mode | PCG iterations | g^2 first / last iteration | candidates | pairs | rounds | solid contacts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 35, 1 | off | 188 | 6.4065e5 / 6.4065e5 | 20076556 | 3370 | 11 | 84839 |
| 35, 1 | factorized | 187 | same | 20076656 | 3370 | 11 | 84839 |
| 35, 1 | blocks | 187 | same | 20076686 | 3370 | 11 | 84839 |
| 120, 1 | off | 200 | 1.3886e5 / 1.3886e5 | 59496753 | 30732 | 17 | 109237 |
| 120, 1 | factorized | 200 | same | 59497705 | 30728 | 17 | 109237 |
| 120, 1 | blocks | 200 | same | 59496214 | 30732 | 17 | 109237 |
| 35, 2 | off | 388 | 6.4065e5 / 3.5363e5 | 23236928 | 3754 | 14 | 100012 |
| 35, 2 | factorized | 387 | 6.4065e5 / 3.5363e5 | 23237803 | 3734 | 14 | 100008 |
| 35, 2 | blocks | 387 | 6.4065e5 / 3.5363e5 | 23236078 | 3734 | 14 | 100010 |
| 120, 2 | off | 400 | 1.3886e5 / 3.8894e4 | 62967771 | 32138 | 20 | 121587 |
| 120, 2 | factorized | 400 | 1.3886e5 / 3.8880e4 | 62968668 | 32152 | 20 | 121587 |
| 120, 2 | blocks | 400 | 1.3886e5 / 3.8901e4 | 62967187 | 32146 | 20 | 121586 |

After one Newton iteration the gradient norm agrees to the 5 digits printed and the counts to 1e-4; the differences
grow with each iteration (3e-4 in g^2 after two at tick 120). Explanation: every one of these PCG solves stops at or
near its 200-iteration cap unconverged, so the direction carries the rounding of 200 products, and re-detection then
turns a sub-micron difference into a different candidate here and there. Over full 8-iteration steps the
trajectories separate (first step, off / factorized / blocks: PCG 1490 / 1548 / 1384 at tick 35, 1600 / 1600 / 1600
at tick 120; pairs 7532 / 8304 / 7924 and 57742 / 57394 / 57408; rounds 32 / 32 / 32 and 38 / 38 / 38). All three use
all 8 Newton iterations in every step. Not shown beyond this: that the later-step divergence is rounding only; it is
inferred from 5.1 and the one-iteration agreement.

### 5.3 Size

| tick | factorized: entries, used | blocks: partner terms, used (allocated) |
| --- | --- | --- |
| 35 | 15-16k entries, 1.44-1.54 MB | 26-29k terms, 0.84-0.94 MB (0.99-1.10 MB) |
| 120 | 18-19k entries, 1.77-1.87 MB | 44-47k terms, 1.40-1.52 MB (1.57-1.69 MB) |

Pool at the default 32 records a vertex: 720,000 records, 11.52 MB at 150^2; peak demand seen 116,724 records (5.2 a
vertex). No row fell back. 'blocks' allocates 8 records for every pair entry and uses 2 per actual partner term.

### 5.4 Time

Replay (64 dispatches in one pass, state after one step from the tick; us per dispatch). Preparation is
nwLinCount + nwLinBuild as a pair, once per Newton iteration (8 a step).

| tick | pcgStep i=2 off | factorized | blocks | preparation factorized | blocks |
| --- | --- | --- | --- | --- | --- |
| 2 | 53.2 | 46.1 | 44.0 | 20.5 | 20.5 |
| 35 | 148.5 | 88.1 | 73.7 | 73.7 | 76.8 |
| 120 | 177.2 | 92.2 | 112.6 | 93.2 | 99.3 |

Step (10 steps, GPU span median / fenced median, ms):

| tick | off | factorized | blocks |
| --- | --- | --- | --- |
| 2 | 133.0 / 141.4 | 100.6 / 115.6 | 98.7 / 112.8 |
| 35 | 524.1 / 535.2 | 371.2 / 379.2 | 360.9 / 387.8 |
| 120 | 848.8 / 871.4 | 697.4 / 725.0 | 695.1 / 725.0 |

Split-pass profile of the first step, pcgStep total: 285.5 -> 152.0 (factorized) / 136.1 ms (blocks) at tick 35,
341.6 -> 193.7 / 190.7 ms at tick 120; the two build kernels together under 1.2 ms a step. Nothing else in the
profile moved: the whole gain is the product.

Reading: the matrix-free contact product was 64% (tick 35) and 70% (tick 120) of a PCG iteration, far above the bound
of section 2 (that bound came from an older tree and from a no-contact iteration that is now 53 us, not 141). The
records ('factorized') remove 56% of it at tick 35 and 63% at tick 120; preparation costs under one product per Newton iteration. Step: -29% at tick
35, -18% at tick 120 (the rest of the tick-120 step is detection). The tick-2 medians are not a step-for-step
comparison (the runs' later steps differ in PCG iterations, 9554 vs 10185 in total); its first step, with no entry
at all, profiles the same (97.1 vs 96.3 ms).

factorized vs blocks: blocks is smaller and faster where field entries dominate (tick 35), slower per iteration where
pair entries do (tick 120: four terms of two records each, half of them global loads, against six records and four
vector loads), and equal on the step. factorized is the one whose cost per entry does not depend on the entry mix.

## 6. What failed or misled

- The first replay of the compiled modes showed no gain (153.6 / 179.2 us). Cause: the harness repeated nwLinBuild
  64 times without nwLinCount; the second build read pool addresses as prefix sums and marked every row for the
  matrix-free product. The step path always runs the pair. The replay now times them as a pair; the numbers above are
  from the corrected run.
- The test's first blocks run left 84 rows matrix-free at the default pool: that two-sheet stack asks 37 records a
  vertex in 'blocks' (8 allocated per pair entry). The equivalence still held; the test now authors its pool (48).
- Not done: tightening the blocks allocation to the actual term count; a smaller authored pool for the sphere scene;
  a hull-facet entry in the operator test.

## 7. Phase A: what stands between 'factorized' and default-on (HEAD da0acf8e export)

All runs from `export-da0acf8e` (clean archive plus this lane's hunks), 150^2 rotating sphere, one run each.
Tool: `tools/probe-cloth-linearization.ts` (modes iterates / quality / compare).

### 7.1 The full-step divergence, measured

Method: from the saved state at ticks 35 and 120, one step is rerun under a (Newton cap, PCG cap) ladder: PCG caps
1, 2, 5, 10, 20, 50, 100 at one Newton iteration, then Newton 1..8 at the scene's cap (200). The positions after each
are compared between off, factorized, blocks (a reordered-but-equal sum: the same terms assembled per column), and
two controls: off from the same state with every saved position coordinate moved by -1/0/+1 f32 ulp (seeds 1, 2).

Off vs factorized, tick 35 (tick 120 in brackets):

| Newton, PCG cap | max dx (m) | rms dx (m) | vertices past 1e-6 m |
| --- | --- | --- | --- |
| 1, 1 | 0 (0) | 0 | 0: bit-identical |
| 1, 2 | 6.1e-8 | 1.2e-9 | 0 |
| 1, 20 | 1.6e-7 | 1.0e-8 | 0 |
| 1, 50 | 5.2e-6 | 6.3e-8 | 6 |
| 1, 100 | 1.35e-3 | 1.3e-5 | 5223 (151) |
| 1, 200 (one full Newton iteration) | 1.9e-6 | 3.1e-7 (3.5e-6) | 256 |
| 8, 200 (the step) | 2.0e-3 | 3.5e-4 (1.4e-4) | all |

The first PCG iteration is exact; the difference then grows from one ulp (6e-8 m) smoothly through twenty iterations
and reaches the micrometre between iterations 20 and 50. The 1.35 mm single-vertex jumps are discrete contact
decisions flipping (a truncation or a hold of one vertex); the ulp control shows the same 1.35 mm jump after ONE PCG
iteration (174 vertices past 1e-6 m), so they are a property of the step, not of the records.

Against the controls, rms dx after the step (8 Newton iterations), tick 35 / tick 120:

| pair | rms dx (m) | PCG iterations (tick 35) | pairs (tick 35) |
| --- | --- | --- | --- |
| off - factorized | 3.5e-4 / 1.4e-4 | 1490 / 1548 | 7532 / 8304 |
| off - blocks | 4.6e-4 / 1.1e-4 | 1490 / 1384 | 7532 / 7924 |
| off - off+ulp (seed 1) | 4.6e-4 / 1.85e-4 | 1490 / 1548 | 7532 / 8018 |
| off - off+ulp (seed 2) | 3.8e-4 / 2.0e-4 | 1490 / 1541 | 7532 / 7996 |
| factorized - blocks | 3.3e-4 / 1.6e-4 | | |
| off+ulp 1 - off+ulp 2 | 3.3e-4 / 1.4e-4 | | |

After one Newton iteration the records differ from off by LESS than a one-ulp input perturbation does (rms 3.1e-7 vs
3.3e-5 at tick 35, 3.5e-6 vs 1.8e-5 at tick 120); after the step every pair, including off against itself from a
one-ulp different state, is at the same 1e-4 m scale with the same spread of PCG iterations (1384..1548) and pairs
(7532..8304). At tick 120 all five runs use 1600 PCG iterations (the cap, every Newton iteration) and 57108..57742
pairs. Verdict: the step's sensitivity (a capped, unconverged PCG and discrete contact decisions) turns ANY one-ulp
difference into 1e-4 m in one step; the records' difference is inside that and starts later. No bug found.

### 7.2 Trajectory quality over 120 steps (ticks 35 -> 155, about 2 minutes of GPU a run)

Sixteen runs: off and factorized unperturbed, off with ulp seeds 1..7, factorized with ulp seeds 1..6, blocks
unperturbed. The seeds measure the simulation's own run-to-run spread, against which on is judged.

| | off (8 runs) | factorized (7 runs) | blocks (1) |
| --- | --- | --- | --- |
| halted (`heldConflict`) | 1 (seed 4, tick 153) | 1 (unperturbed, tick 140) | 0 |
| cloth-cloth intersections, faces in solid | 0 | 0 | 0 |
| held / travel-held / overflow-held vertices | 0 / 0 / 0 | 0 / 0 / 0 | 0 |
| fallback rows, record overflow | n/a | 0 | 0 |
| Newton iterations | 8 every step | 8 every step | 8 |
| PCG iterations a step, mean (range over runs) | 1538..1564 | 1552..1572 | 1558 |
| detection rounds, mean / max | 33.8..36.1 / 41..52 | 33.5..34.7 / 39..57 | 33.5 / 39 |
| pairs, mean | 46.8k..63.1k | 45.1k..55.4k | 51.7k |
| g / mg at the last Newton iteration, median | 152..208 | 158..182 | 164 |
| same, 90th percentile | 356..1920 | 269..1700 | 1680 |
| max edge strain over the run | 0.274..0.402 | 0.263..0.458 | 0.369 |
| steps with an edge past 1.05 x limit | 41..69 | 54..70 | 72 |
| plane-record peak | 394k..711k | 394k..854k | 394k |
| least true distance, runs with a negative sample | 6 of 8 (to -5.4e-4 m) | 2 of 7 (to -5.8e-4 m) | 0 |
| status flags | envelopeCap, failSafe | same | same |

Reading. Every distribution of factorized lies inside or overlaps the off seeds'; halts are 1 of 8 against 1 of 7.
The unperturbed factorized halt (strain 0.458 and 57 rounds in its last three ticks) has the same signature as off
seed 4's (52 rounds): the scene itself halts with `heldConflict` in roughly one run in eight on this range, with the
records or without. That is a pre-existing robustness defect of the scene/solver at 150^2 (C2's area), not something
the records add; the sample is too small to say the rates are equal to better than "1 in 8 vs 1 in 7".

Caveats, stated plainly: g is the gradient at the START of the last Newton iteration (the solver's word), not a
recomputed end-of-step residual; "least true distance" is this tool's metric (scene solids at cloth time, vertices
plus a 4-per-side face lattice) and goes slightly negative in off as often as in on; the large g/mg (median ~160)
is the unconverged 8 x 200 solve, common to all runs.

### 7.3 Operator-test gaps closed

`tests/cloth-contact-linearization.test.ts` now covers: the trust region's model (`nwModel` under `trustRegion:
'hybrid'`), compiled against matrix-free on the same direction; the one-patch solve (`pcgSolve`) through the records;
and a hull-facet (w = 2) entry, from a sheet set 1.2 mm off the edge of a box turned 30 degrees. Results in 7.5.

### 7.4 readContactLinearization().entries

Removed (it was only right for 'factorized'); the reader returns records, items, rows, fallback rows and bytes.
The pool stays at its default; a per-scene authored pool from the measured peak is a follow-up.

### 7.5 Operator equivalence, new forms (export-da0acf8e, `check`, one run, 3/3 pass)

| scenario | mode | entries (kinds) | worst relative difference |
| --- | --- | --- | --- |
| stack, pcgStep 0 | factorized / blocks / factorized pool 1 (537 rows fall back) | solid, body, VT, EE, sample | 2.6e-7 / 3.2e-7 / 6.6e-8 of (contact + w) |
| stack, hybrid: pcgStep 0 | factorized / blocks | same | 1.8e-7 / 2.3e-7 |
| stack, hybrid: nwModel s.H s per patch | factorized / blocks | same | 1.1e-7 / 1.5e-8 of the largest s.H s |
| facet sheet, pcgSolve (one patch, 3 iterations) | factorized / blocks | 24 hull-facet (w = 2), 23 sample, 21 solid | 2.9e-7 / 1.4e-7 |

(The facet sheet's field needed a band of 0.1 m: the detector refuses a band below r + 1.5 L_max rqMax.)

### 7.6 Evidence for the default, summarized

- Rounding only: the first PCG iterate is bit-identical; the difference then grows like rounding in an unconverged
  capped PCG and is, after one Newton iteration, 10-100x smaller than a one-ulp input perturbation's; after the step
  on-vs-off equals off-vs-off(ulp).
- Trajectory: over 120 steps and 15 seeded runs, every quality measure of factorized lies within the off seeds' spread;
  halts 1/7 vs 1/8, both heldConflict (a pre-existing scene/solver robustness defect).
- Every product form (pcgStep, pcgSolve, nwModel) and every entry kind is covered by an operator test.
- Not decided here: the default (coordinator's call). Cost: pool 32 records a vertex = 11.5 MB at 150^2 (512 B a
  vertex); used peak 1.77 MB at tick 120. A per-scene authored pool from the measured peak is the follow-up.

## 8. Phase B: where pcgStep's time goes, and the split

### 8.1 Attribution (150^2 sphere, replay of pcgStep i = 1, `timing`, clean exports, us)

Ablations are scratch WGSL edits of the factorized build (export-ablate, `L1_ABLATE`); each removes one part of the
iteration and leaves the rest, so differences attribute cost. Ticks 35 | 120.

| build | pcgStep i=1 |
| --- | --- |
| off (matrix-free contact) | 149 \| 177 |
| factorized (records, matrix-free fallback branch present) | 89 \| 93 |
| factorized, fallback branch removed | 54 \| 55 |
| no contact product | 31 \| 31 |
| contact: header loads only | 31 \| 32 |
| contact: records, no partner gathers | 39 \| 39 |
| contact: records, no global (out-of-patch) columns | 46 \| 50 |
| contact-free: diagonal block only | 18 \| 18 |
| contact-free: no bends | 28 \| 28 |
| contact-free: no halo | 29 \| 29 |
| empty dispatch | about 4-5 |

Reading: the never-taken matrix-free fallback branch costs 35 us an iteration by its presence (registers, occupancy).
Without it the contact product is 23-24 us: the record stream about 8, in-patch partner gathers 7-11, global
columns 5-8. The contact-free 31 us is dispatch and own rows and reduction 18 (dispatch about 5), the off-diagonal
stencil 9, bends 3, halo 2. The gathers are small shares, so Experiment 3 (PCG state in patch order) cannot pay for
itself and is not built. Experiment 4 (transposed records) is bounded by the about 8 us record stream.

### 8.2 The split: pcgStep (records only) + pcgStepOverflow (matrix-free)

With compiled records, pcgStep's contact product is the records alone (`nwLinRecordsHv`); a patch with any row the
pool could not hold returns after the lead's scalars and leaves its rows to `pcgStepOverflow`, dispatched after
pcgStep in the same pass with the same record index. nwLinBuild marks the patch (`LIN_PATCH + p` .y = LIN_NONE iff
its total does not fit; rows take records in prefix order, so a row of the patch overflows iff the patch's last
does). pcgStepOverflow returns at once on an unmarked patch (one uniform load), and on a marked one runs the same
iteration matrix-free with `lead = false`. Every value it reads (partials i-1, i-2, the alpha records, r/w/s of
parity i-1) is one pcgStep does not write in dispatch i, so the order is safe. pcgSolve and nwModel keep the per-row
fallback (`nwLinHv`). Off's WGSL is byte-identical (32 flag combinations). The empty overflow dispatch is a full
early-exit grid (no indirect dispatch: that would need a new binding); its cost is measured in 8.3.

Gate (export of 66ccc687 + the split, default 'off'; the linearization test and check:types again at 815f138a):
check:types clean; test:unit 1753 pass, 0 fail (CPU only, WEBGPU_NODE_MODULE unset); tests/cloth-contact-linearization
3/3. Its forced-overflow cases: pool 12 records a vertex sends 3 of 4 patches (293 rows) through pcgStepOverflow, pool 1
all 4; every row over the pool is asserted bit-identical to the all-matrix-free reference, which holds only if its
patch took pcgStepOverflow (a row left to pcgStep's records with the LIN_NONE header would read outside the pool).

### 8.3 Time (150^2 sphere, `timing`, one run each, --lin factorized; HEAD 66ccc687 without the split vs with it)

| tick | pcgStep i=1, without | pcgStep + pcgStepOverflow i=1 (pcgStep, overflow, i=2) | fenced ms/step without -> with |
| --- | --- | --- | --- |
| 35 | 89.1 us | 61.4 (54.3 + 6.1) | 399.2 -> 370.3 |
| 120 | 94.2 | 62.5 (55.3 + 5.1) | 575.6 -> 545.5 |

The same Newton and PCG counts (8 / 1548, 8 / 1600) and |g|^2; no patch overflowed (pool 32). The split takes 28-32 us
off every PCG iteration (about 30 ms a step at 1550-1600 iterations); the empty overflow dispatch keeps 5-6 us of it.
Off (149 | 177 us an iteration, 8.1) against the split: 2.4-2.8x on pcgStep.

## 9. The factorized-default gate and ladder 1e (failure 4)

### 9.1 The gate (fddccb2f + the flip, `check`, every tests/cloth-*.test.ts, concurrency 1)

Every file passes as at the baseline (off at fddccb2f), except four tests, all C1/C2-owned and left unchanged:

| test | file | off (baseline) | factorized |
| --- | --- | --- | --- |
| contact ladder 6: least true frustum distance | tests/cloth-contact-ladder.test.ts | +1.675e-4 m (t44) | -6.06e-5 m (t59) |
| contact ladder 7 (H_vv) | tests/cloth-contact-ladder.test.ts | least +1.465e-4 m | heldConflict halt at tick 46 |
| contact ladder 13 (H_vv determinism) | tests/cloth-contact-ladder.test.ts | passes | the same halt |
| ladder 1e: every vertex over 2 floors certificate-held | tests/cloth-ladder.test.ts | cap, \|g\|/\|mg\| 4.32e-4, 2 over (both held) | cap, 5.58e-1, 14 over, worst 357.7 floors |

The flip was not landed (coordinator's ruling: C1/C2 failures stop it). The default stays 'off' until ladder 1e passes
deterministically; the split lands with it 'off' so the flip is then the one default in CLOTH_NEWTON_DEFAULTS.

### 9.2 Ladder 1e diagnosis (tools/l1-ladder1e.ts, scratch; fddccb2f + flip)

Off and factorized from the same fixture state, Newton capped at k iterations:

| k | max \|x_off - x_fac\| | \|g\|^2 off / fac |
| --- | --- | --- |
| 1 | 5.96e-8 m | 1.641e4 / 1.641e4 |
| 5 | 2.98e-7 | 352.7 / 352.7 (PCG counts first differ, 178 / 175) |
| 11 | 1.06e-5 | 61.00 / 60.92 |
| 15 | 1.57e-4 | 325.1 / 326.6 |
| 20 | 2.11e-3 | 965.4 / 361.3 |
| 45 | 1.90e-3 | 6.49 / 361.4 |
| 80 | 1.87e-3 | 3.27e-3 / 1.51 |
| 100 | 2.03e-3 | 2.39e-7 / 0.514 (factorized's last step 1/8) |
| 150 | 2.03e-3 | 2.28e-7 / 0.381 (step 1/8) |

The difference grows about 10x every 4 iterations from the first iterate's rounding (6e-8 m): exponential
amplification, not a jump at one iterate. By k = 20 the two runs are 2 mm apart and take different contact
histories; off reaches a stationary point, factorized stalls at a 1/8 step with 3 active vertices.

At the start of iteration 11 (where they first part past 1e-5 m) and of iteration 100 (inside the stall), on
identical vectors (pcgStep dispatch 0, the same u):

- records rebuilt fresh vs the matrix-free product: max \|dw\| 9.5e-7 of 6.2 (k = 11), 6.3e-8 of 0.59 (k = 100),
  about 1e-7 relative: f32 rounding;
- records as built inside the run vs rebuilt fresh at the same iterate: 0 header words and 0 pool words differ;
- no row over the pool (40 and 47 rows with entries; demand at most 90 records a row of 32 x 1024).

So none of the three suspects holds at the parting iterate or in the stall: (a) no staleness, the as-built records
are bit-identical to fresh ones; (b) the entry kinds in play (field, bodyField, sampleFace) give the matrix-free
operator to rounding; (c) no row is over capacity. The build's place answers C2's checklist: nwLinCount and
nwLinBuild run after nwGradient in every Newton iteration, under its gate, which is after every detection round
(afterFrame's mandatory detection; afterNewtonStep's verify, decide and re-detection, which follow nwApply and so
precede the next iteration's nwGradient) and before the PCG; nothing runs between the build and pcgStep. The records
hold the entries' linearization only (ids, weights, Mn, Mf with the lagged friction of the entries): no projector,
active set, strain duals, restart flag or trust radius, which the PCG reads live as in 'off'.

### 9.3 The decisive controls: one-ulp perturbed 'off', and the operator switched mid-step

The same tests (scratch copies; `L1_PERTURB` moves every start position by +-1 f32 ulp, mulberry32; seed 0 is
unperturbed). Ladder 1e:

| run | ladder 1e | \|g\|/\|mg\| at the 150 cap | vertices over 2 floors (certificate-held) |
| --- | --- | --- | --- |
| off, seed 0 | pass | 4.32e-4 | 2 (2) |
| off, seed 1 | FAIL | 1.19e-1 | 27 (0) |
| off, seed 2 | pass | 4.59e-4 | 2 (2) |
| off, seed 3 | FAIL | 4.02 | 13 (0) |
| factorized, seed 0 | FAIL | 5.58e-1 | 14 (2) |
| factorized, seed 1 | FAIL | 4.26 | 18 (0) |
| factorized, seed 2 | pass | 4.74e-4 | 2 (2) |
| blocks, seed 0 | pass | 4.17e-4 | 1 (1) |

The operator switched inside the one step (scratch `L1_SWITCH` in nwLinBuild: from Newton iteration K0 on, or before
it, every row to the matrix-free product; '0:off' in the factorized build is bit-identical to true off at k = 20-150):
- factorized's stall state at k = 100 continued with off stays in the stall (k = 150: step 1/8, \|g\|^2 0.381, active
  3/3/2; true off from the fixture reaches 2.3e-7);
- off's state at k = 20 continued with factorized converges as off does (k = 150: \|g\|^2 2.5e-7, step 1).

So ladder 1e's failure is a non-stationary trap of the step that rounding chooses, not the operator: routed by the
coordinator to fluid-60 (state: fluid-scratch/cloth-locality/l1/p/switch-100-off-k*.json).

Contact ladder tests 6 and 7 (CLOTH_LADDER_ONLY=sphere; 13 is 7's determinism twin):

| run | test 6: least true frustum distance | test 7 (H_vv) |
| --- | --- | --- |
| off, seed 0 | pass, +1.675e-4 m (t44) | pass, +1.465e-4 m (t32) |
| off, seed 1 | pass, +1.984e-4 (t56) | pass, +3.958e-5 (t45) |
| off, seed 2 | FAIL, -2.162e-4 (t39) | FAIL, heldConflict halt at tick 46, -7.761e-4 (t36) |
| factorized, seed 0 | FAIL, -6.061e-5 (t59) | FAIL, heldConflict halt at tick 46 |
| blocks, seed 0 | pass, +1.518e-4 (t47) | FAIL, heldConflict halt at tick 46, -9.875e-6 (t40) |

A one-ulp perturbed off run reproduces both failures, deeper than factorized's: pre-existing sensitivity of the scene
(C1/C2), as Phase A's 1-in-8 heldConflict halts of off on the 150^2 sphere already suggested.
