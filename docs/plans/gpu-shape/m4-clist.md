# GPU-shape M4: the C-list passes reshaped (lane B2, 2026-11-24)

**Result: three exact reshapes landed** (uTransfer pages-first k=2, uExt4Seed k4×128, pressureRowsC k2×256). Each one
writes the same words as its parent's kernel on both frozen states. In situ, their sum is **−0.82 ms on paddleboats and
−0.29 ms on dam**. Scene-level gates, against HEAD with the M4 commits reverted:
- **stepab** (frozen whole step, A/A ≤ 0.05 ms): changed families Σ Δ seep −0.20, dam −0.33, letters −0.93 and
  paddleboats −0.49 ms (−1.0 to −2.1%). The changed placements are 3 of 3 identical on every scene.
- **Timing round**, order-balanced: seep −2.0%, dam −1.7%, letters −1.9%, paddleboats −0.7%. Drift is unchanged.

Two more passes were diagnosed but have no exact shape win:
- **lcRetire**: its tail is serial per-lane distance projection.
- **censusDecide**: it is throughput-bound on paddleboats. trig64 is exact, but its dam timing is inconsistent, so it is
  parked as a patch.

The design rule held where the cause was shape: **no group much longer than the median, and ≥ 2 groups a core in
flight**.
- On the three landed passes, the long groups were whole pages, and splitting a page cut them.
- On lcRetire, the long groups are single lanes' serial chains, so no workgroup shape changes them.

Plan: docs/plans/gpu-shape-2026-11-23.md (M4).
Tool: tools/experiments/gpu-shape/m4.mts.
Data:
- docs/plans/gpu-shape/m4/<scene>/:
  - r1–r3.log: the variant rounds at c12d19cd.
  - land1/land2.log: the landed kernels against the parent's.
  - land*-m4.json: HEAD, `git status` and every sample.
  - clock-*.tsv: the per-workgroup traces of the landed passes and their parents.
- docs/plans/gpu-shape/m4/censusDecide-trig64.patch.
- The per-lane diagnostic traces, lanes-*.tsv (1–3 MB), stay in ~/code/me/fluid-scratch/gpu-shape/m4/<scene>/.

## Method

- Every GPU job ran through `tools/gpu/fluid-gpu-run.sh timing`. All returned rc 0, with no gpuError.
- Clock probe: 76.1–80.2 µs throughout, apart from one 113 µs and one 146.6 µs round on dam.
- Frozen state: step 100, fixed dt 1/60. Each variant is a text substitution of the schedule's own kernel, compiled and
  bound as the engine does it.
- A split variant's args word is k·n, written by an untimed one-lane kernel.
- **exact**: one in-situ snapshot runs base, base again, then every variant, and compares every read-write binding word
  for word. Base against base is 0 words in every case.
- **time**: min (and median) of R = 5–7 interleaved in-situ samples, with base in the same rounds.
- **clock**: per-workgroup start and end ticks (M3's clock). The clock inflates heavy kernels, so its absolute µs are
  for shape only.
- **landing check** (REF_DIR): the parent's dumped kernel text and args word run as a "ref" variant against the built
  source, on a frozen state of the built tree.

## Per pass (in situ, µs: parent's kernel → landed; tail = span − time at which 95% of groups had ended)

| pass | variant | paddleboats | dam | clock: max group / tail / in flight | exact |
|---|---|---|---|---|---|
| uTransfer (FCA) | C pages first, a page over 2 groups (FCA2_G1) | 1738 → 1404 (**−334**) | 1569 → 1322 (**−247**) | pb max 515 → 208, tail 369 → 118; dam max 290 → 219, tail 357 → 126 | 0 words, both |
| uExt4Seed (C_ACT) | a page over 4 groups of 128 lanes, TG arrays sized (C_ACT4_G1) | 810 → 422 (**−388**) | 377 → 329 (**−47**) | pb max 446 → 252, tail 391 → 195, in flight 64 → ~155; dam max 382 → 237, in flight 64 → ~125 | 0 words, both |
| pressureRowsC (C_ALL) | a page over 2 groups of 256 lanes, TG arrays sized (C_ALL2_G1) | 479 → 377 (**−101**) | 253 → 257 (+4, flat) | in flight 32 → 64 on both; pb max 199 → 226 | 0 words, both |

Notes on the landed passes:
- **uTransfer.** M3 approved this as a sub-threshold option. On the landed tree it is worth more than M3 measured
  (M3: −0.27 pb / −0.50 dam per step).
- **uExt4Seed variants on paddleboats**, base 808 µs (r2):

  | k2 | k2s | k4w128 | k8w64 |
  |---|---|---|---|
  | −113 | −265 | −354 | −364 |

  - k4w128 was picked as the best on both scenes. Dam: k2s −118, k4w128 −88, k8w64 −84.
  - k4 and k8 at 256 lanes lost: lanes sat idle, and the cap of 64 groups in flight dominated.
- **pressureRowsC variants**, r1/r2:

  | | k2 (512-lane arrays) | k2s | k4s |
  |---|---|---|---|
  | paddleboats | −107 / −102 | −73 | −48 |
  | dam | −27 / −17 | −73 | −77 |

  - k2s was landed because it has half the TG memory: 64 groups in flight instead of ~47–63.
  - On the landed tree, dam is flat. Its parent ran 112 groups for 23–27 in flight, so dam's rowsC was never
    occupancy-bound.

## Diagnosed, not landed

**lcRetire** (C_ACT X8, 64 lanes).
- Base time: 930–943 µs on paddleboats, 637–664 µs on dam.
- Clock:
  - paddleboats: p50 61, max 915, tail 629 µs. 510 groups are in flight, then the count drops to under 40 by 40% of
    the span.
  - dam: max 656, tail 219 µs.
- Per-lane counters (lrDiag: path, projection iterations, nbVertexPhi calls, lcSample calls; paddleboats):
  - The 8 slowest simdgroups, at 770–1324 µs (clocked), all run lcBuildDistance's coarse or reach path (paths 2/3).
    Each has 4–5 projection iterations and 40–56 nbVertexPhi calls per lane.
  - Simdgroup time follows the max vphi count:

    | simdgroup time (µs) | < 50 | 100–200 | 400–800 | > 800 |
    |---|---|---|---|---|
    | mean max vphi calls | 4.8 | 37 | 46 | 53 |

  - The slow simdgroups cluster in the k8 parts of a few pages (g718/719 and g360/361, for example), in lanes 40–47:
    one row of the plane.
- **The tail is one lane's serial chain.** Splitting groups cannot shorten it. (Inference: the next step is a distance
  build that spreads a cell's projection iterations or corner φ over lanes, or calls nbVertexPhi fewer times. That is
  an algorithmic change, not a shape change.)
- Variants tried:
  - **gcache** caches the 8 corners in lcGradient. It gained −77/−143 µs on paddleboats but is **not exact** there:
    104–114 words of C_PHI1. It was exact on dam.
  - The **gfetch** control has the same code shape, refetching every call, and is exact on both scenes. So the
    difference comes from reusing the cache, not from codegen. Inference: a lane's corner φ is not invariant within the
    pass.
  - **Stencil staging** (owner bits in TG memory) is exact, but gains only −9 µs, and 510 → 350 groups are in flight.
    No-go.

**censusDecide** (C_ALL X8, 64 lanes).
- paddleboats: the groups are uniform (p50 233–240, p90 252–263 µs) with ~255 in flight, so the pass is
  throughput-bound, not shape-bound.
- dam: p50 9–11, p90 394–492, max 856–974 µs.
- Per-lane counters (cdDiag, paddleboats and dam) rule out ncInterfaceAny's whole-list walk and ncRangeKeys' box walks
  as the cost. The 10–50% lanes average ~2 interface calls and ~1 range walk; the slowest lanes show 0–36 cells.
- Variants tried:
  - **trig64** spreads TRIG's 8-lane × 8-tile loop over 64 lanes. It is exact on both scenes, with these timings over
    four rounds:

    | | r1 | r3 | land1 | land2 (landed vs ref) |
    |---|---|---|---|---|
    | paddleboats | −35 | −113 | −126 | −95 |
    | dam | +56 | −237 | −33 | +119 |

    Not landed: the dam result is inconsistent. It is parked as m4/censusDecide-trig64.patch.
  - **count** gathers the six calm and candidate counters per workgroup instead of one global atomic per lane. It is
    exact but neutral (+45 / −22 µs). No-go.

**Not attempted this round:** lcAdvect, censusCoarse and the smaller C passes. The round went to the four largest passes
whose clock showed a tail or an occupancy cap.

## Landed (lib/), one family per commit

Every commit passed `npm run check:types` and `npm run test:unit` (2584 tests, 0 failures). Exactness was re-verified
from the built source (land1).

- effaa990: uTransfer pages-first k=2.
  - ARGS.FCA2_G1 = 421; setArgs writes nF + 2 nC.
  - `stagedHead(..., pagesFirst)`; `vcTransferPage(slot, lid, sub)`.
- 12d01bee: uExt4Seed over four 128-lane workgroups a page (ARGS.C_ACT4_G1 = 424).
- c1d5f360: pressureRowsC over two 256-lane workgroups a page.
  - ARGS.C_ALL2_G1 = 427; `ROWS_C_PAGES`.
  - The GPU pressure tests dispatch rowsC from ROWS_C_PAGES.

ARGS_LIST_END is now 430, below LANE_P (512).

Metal thread caps (brick-tune/regcap.sh, size attribute stripped; parent → landed):

| uTransfer | uExt4Seed | pressureRowsC |
|---|---|---|
| 512 → 576 | 640 → 640 | 576 → 576 |

Every cap is above its workgroup size (256, 128, 256).

## Scene-level gates

Base: an export of 06d81562 (for stepab) or 67c09e96 (for the round), with lib/ patched back to the M4 parent.
- stepab's base keeps the ABI's words and setArgs: m4/round/revert-m4-kernels.patch.
- The round's base reverts all of M4: m4/round/revert-m4.patch.
- The candidate is the unpatched export. Provenance and order: m4/round/provenance.log and order.log.

**stepab.** tools/experiments/gpu-shape/m4-stepab.mts runs lane A2's stepab.mts with two generated changes, leaving
stepab itself untouched. Its shape check accepts an indirect dispatch for an indirect one, and a changed placement runs
with the candidate's args word. Step 100, one run a scene (m4/round/stepab-<scene>.log).

| scene | exact | uTransfer | uExt4Seed | pressureRowsC | changed Σ Δ (A/A) | unchanged Σ Δ (A/A) | GPU wall min (A/A) |
|---|---|---|---|---|---|---|---|
| seep | 3 of 3 | 0.891 → 0.743 | 0.275 → 0.258 | 0.257 → 0.222 | **−0.199 ms, −1.02%** (−0.003) | −0.017 (+0.021) | −0.280 ms, −1.44% (+0.101) |
| dam | 3 of 3 | 1.506 → 1.263 | 0.386 → 0.299 | 0.329 → 0.326 | **−0.334 ms, −1.09%** (−0.008) | +0.029 (+0.000) | −0.485 ms, −1.58% (−0.301) |
| letters | 3 of 3 | 2.863 → 2.195 | 0.575 → 0.396 | 0.604 → 0.523 | **−0.929 ms, −2.11%** (−0.047) | +0.110 (+0.069) | −1.045 ms, −2.35% (−0.099) |
| paddleboats | 3 of 3 | 1.510 → 1.390 | 0.790 → 0.432 | 0.410 → 0.395 | **−0.492 ms, −1.46%** (−0.013) | −0.003 (−0.026) | −0.533 ms, −1.58% (−0.038) |

Kernel cells are family minima, in ms, base → candidate. censusDecide moves +0.025 to +0.112 ms, with an A/A of
+0.022 to +0.067. It is unchanged text, and it moves the same way in its A/A.

**Timing round** (tools/nbflip-timing-round.sh, balanced order; m4/round/timing.md):
- All 16 runs returned rc 0, with 0 Metal aborts.
- The round's table step hit the zsh quoting slip fixed in 06d81562, so the table was rerun with nbflip-gates.ts over
  the same runs.

| scene | fenced median | pipelined ms/step | order-balanced | volume drift, base → candidate |
|---|---|---|---|---|
| seep (1200 steps) | 25.97 → 25.96 | 25.92 → 24.91 | **−2.0%** | −0.606% → −0.578% (fenced), −0.615% → −0.592% (pipelined) |
| dam (165 steps) | 33.08 → 32.79 | 35.63 → 34.74 | **−1.7%** | |
| letters (240 steps) | 53.08 → 51.74 | 43.53 → 42.96 | **−1.9%** | +0.622% → +0.619% (fenced) |
| paddleboats (300 steps) | 36.38 → 36.00 | 34.73 → 34.60 | **−0.7%** | |

Peak GPU MB is unchanged on every scene.

Construction on seep and letters went from 4.7/3.0 s to 12.9/11.4 s for the candidate's fenced run; dam and
paddleboats did not change. (Inference: a cold Metal pipeline cache for the new kernel texts. Dam's and paddleboats'
new texts were compiled earlier by the m4.mts runs. This is not a steady-state cost.)
