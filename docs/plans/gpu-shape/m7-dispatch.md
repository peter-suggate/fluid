# GPU-shape M7: the whole-step dispatch census and the PCG slot (lane B2)

Measured on 4b564b0d (dispatch structure unchanged through 87cf07cd), M1 Max, Dawn/Metal, fixed dt 1/60 after a 100-step
warm-up, frozen state. Tools (scratch, `~/code/me/fluid-scratch/gpu-shape/m7/tree-p`): `m4.mts MODE=census` (every
placement as executed, its indirect args copied just before it), `m4.mts MODE=slots` (each PCG slot's placements timed as
one pass in situ), `dfloor.mts` (in-stream cost per extra dependent dispatch). Raw: `m7/census/`, `m7/slots-*.txt`,
`m7/dfloor*.txt`, `m7/sv/`.

## 1. What a dispatch costs (measured)

| case (per extra dependent dispatch in one pass) | default | `skip_validation` |
|---|---|---|
| indirect, 0 groups | 5.6–6.6 µs | 0.85 µs |
| indirect, 1×512 | 5.0–11.1 µs | 5.6 µs |
| direct, 1×512 | 5.2 µs | 5.2 µs |
| direct at capacity, every group exits: 125×64 / 400×64 / 1000×64 | 5.9 / 6.0 / 7.1 µs | |
| direct at capacity, every group exits: 125×512 / 400×512 / 1000×512 | 6.4 / 9.6 / 15.8 µs | |
| direct at capacity, every group exits: 4913×512 / 10240×512 | 23.2 / 34.9 µs | |

- **Dawn validates every indirect dispatch** with one extra serial 1×1×1 dispatch before it. The validation shader's text
  is in the Dawn binary: it clamps args to `maxComputeWorkgroupsPerDimension`. It costs about one dispatch floor
  (≈ 5 µs on dfloor). In the browser it cannot be turned off; `skip_validation` is a Node-only toggle.
- **Whole step, paddleboats, 300 steps pipelined:** 31.35 ms with `skip_validation` against 32.79 / 32.22 ms with
  validation, so −1.0 to −1.4 ms (3–4%). With 269 indirect dispatches, that is ≈ 4–5 µs each in situ. This is the size of
  the validation tax; it is not a lever.
- **Dead PCG slots, in situ** (`MODE=slots`): a converged slot costs 64–66 µs on dam (21 dispatches, ≈ 3.1 µs each) and
  71–83 µs on paddleboats (23 dispatches, ≈ 3.5 µs each). That is less than the dfloor figure; inference: higher clocks
  in situ. A live slot costs 0.82–0.91 ms on dam and 1.26–1.35 ms on paddleboats; the whole PCG is 5.2 ms (dam) and
  4.3 ms (paddleboats).

## 2. Whole-step dispatch census

C = indirect dispatches by group count. Sub-slab means fewer than 32 groups (fewer than one per core).

| scene | dispatches | direct (sub-slab) | indirect | C: 0 groups | C: 1–31 | C: 32–511 | C: ≥ 512 | PCG dispatches | PCG slots live/dead |
|---|---|---|---|---|---|---|---|---|---|
| dam | 283 | 29 (22) | 254 | 40 | 9 | 120 | 85 | 171 | 6/2 |
| seep | 290 | 29 (22) | 261 | 40 | 11 | 161 | 49 | 171 | 6/2 |
| letters | 283 | 29 (22) | 254 | 80 | 6 | 65 | 103 | 171 | 4/4 |
| paddleboats | 307 | 38 (31) | 269 | 104 | 8 | 65 | 92 | 187 | 3/5 |

- **The PCG is 60% of all dispatches.**
- **Dead slots per step**, measured: dam 0.13 ms (0.45%), paddleboats 0.40 ms (1.3%). Estimated: letters ≈ 0.28 ms (0.7%),
  seep ≈ 0.13 ms (0.7%).
- **The validation tax** is ≈ 254–269 × 4–5 µs ≈ 1.1–1.4 ms a step (3–5%) on every scene. It scales with the number of
  indirect dispatches, so each indirect dispatch removed saves its kernel's floor plus one validation dispatch.

## 3. Runs of adjacent passes and whether they can fuse

Runs are adjacent dispatches on the same args word, or the same list. "Dependency" is what the second pass reads of the
first's writes.

| rank | run | per step | dependency | fusable exactly? |
|---|---|---|---|---|
| 1 | pmgDown1C > pmgDown2 > pmgDown3 (C pages) | ×8 | the residual at L2 / L3 reads neighbour pages' EPRE (pV2/pV3) | only by recomputing the neighbour face blocks: down2 → down3 recomputes 6 × 32 L3 cells and 1536 L2 residuals a page (4× the pass) |
| 2 | pmgUp3 > pmgUp2 > pmgNativePost1 (C pages) | ×8 | up2's halo is the neighbours' EPRE + parent EPOST (up3's output) | up3 → up2 by recomputing the 24 neighbour face L3 blocks in up2's spare lanes; no new workgroup memory |
| 3 | pmgDirection > pmgUpdate (F+C) | ×8 | update's Ap reads the neighbours' new p | no: grid-wide |
| 4 | pmgCheck > the next pmgPreS | ×8 | preS's args are written by check | no: preS would have to be direct at the S capacity (4 × F cap) |
| 5 | pmgTopCycle (1 group) and pmgUp3 / pmgDown3 (64 lanes, C cap 125–400) | ×8 each | n/a | not a fusion: make them **direct**, so they pay no validation (§4 c1) |
| 6 | lNear ×3 > lDistance; lDilate ×3 (×2) | 1, 2 | each pass reads neighbours' fcell | no: grid-wide (level-set owner) |
| 7 | setTransferFaces > setTransferCells > setTransferCommit | 1 | Faces → Cells: no shared writes; Cells → Commit: phi | Faces + Cells are a candidate (owner to confirm) |
| 8 | censusWant > setTransferList; pTiles > setMark; pJoin > pDelta; pLiquid > pSeed | 1 each | set only (lists and counters) | candidates if the set words are per-brick, not list builds (owners to confirm; pSeed and pDelta belong to lane C) |
| 9 | pmgGuess > pmgWallsG > pmgSetupF; pmgFinishC > pressureProjectC > pmgPressureDelta | 1 each | pres / pmgF neighbour reads | setup and finish run once a step: small |

Every run inside the PCG slot has a true neighbour (grid) dependency at each boundary. No pair is fusable with only a
workgroup barrier.

## 4. The PCG slot

**What a slot dispatches today:** 21–23 dispatches (24 with walls in slot 0):

- the fine passes: preS, preU, down0, down1, up1, postU, postS, postZ;
- the coarse page passes: nativePre0/1, down1C, down2, down3, up3, up2, nativePost1/0;
- top;
- direction and update;
- check.

**The options, weighed:**

- **A single slot-head gate** that skips the slot's later passes is not possible in WebGPU. Every encoded dispatch is
  paid for: indirect dispatches pay validation even at 0 groups, and direct ones pay a floor of ≥ 5 µs. A dead slot
  therefore costs (its dispatch count) × ≈ 3 µs in situ, and the only lever is fewer dispatches per slot. Skipping
  encoded slots would need the CPU to know convergence (no CPU stages) or a lower slot cap (which changes the
  iteration counts on capped states).
- **c1, exact, built:** make three passes direct.
  - pmgTopCycle becomes direct [1,1,1]; when PH_OPEN is 0 it returns, flagged through wN5 = NB_NONE, so no new
    workgroup memory.
  - pmgDown3 and pmgUp3 (64 lanes) are dispatched direct at the C capacity; pageHead keeps no page when PH_OPEN is 0.
  - Estimate from dfloor: −5 µs a live slot per pass (no validation), +0.4 µs a dead slot. Dam: −3 × 6 × 5 ≈ −90 µs a
    step (−0.3%); paddleboats ≈ −45 µs.
  - This only generalizes while the C capacity is ≤ about 1000. Above that, a direct 64-lane dispatch at capacity costs
    more than the validation. The 512-lane passes stay indirect: 400 × 512 empty groups cost 9.6 µs.
  - Measured with m7ab, not stepab: see §5.
- **c2, exact, to build:** fuse up3 into up2 (rank 2 in §3). −1 dispatch a slot: ≈ −11 µs live and −3 µs dead.
- **c3, exact but costly:** fuse down2 into down3 with recompute (rank 1 in §3). −1 dispatch a slot, but the pass does
  about 4× the work. It is latency-bound (about 35 of 98 groups busy on dam), so the result is unknown until a prototype.
- **c4, not exact; needs the coordinator's OK:** page-local coarse levels, i.e. a zero halo in the L2/L3 residuals as
  well as in the smoothing. Then down1C + down2 + down3 fuse into one page workgroup, and up3 + up2 into another: −3
  dispatches a slot and shorter chains. This changes the preconditioner, so the iteration counts may change.

Even with c1 + c2 + c3 the slot drops from 21 to 19 dispatches. The PCG's dispatch floor is about 18% of a live slot
(≈ 150 µs of about 830 µs on dam). The remaining structural cost is that each coarse pass is a single-group dependent
chain (about 1 busy group per core): the slot is latency-bound, not launch-bound.

## 5. Results

**The tool.** stepab cannot judge these changes: a dispatch changing from indirect to direct, or a placement removed,
is a structural change, and it fails explicitly. In its place, `m7ab.mts` (scratch, `m7/tree-c6`) does a structural
A/B in one process:

- It builds engine A with HEAD's schedule (`PMG_M7=0`) and engine B with the candidate, from the same tree.
- A warms up 100 steps. S_a is A's state just before the PCG range (the first pmgPreS to the last pmgCheck).
- A runs the range twice from S_a; B runs it from S_a copied into B's buffers by name.
- Every buffer is compared word for word.
- The range is timed A / B / A again, interleaved, R = 7, one pass each.
- Control: the candidate built with the flag at 0 gives every buffer identical and Δ −0.6 µs (A/A +11.6).

**c1, landed:** pmgTopCycle direct, plus pmgDown3 and pmgUp3 direct at the C capacity while it is ≤ 1024. Bit-identical
on all four scenes: every compared buffer identical, 25–28 buffers.

| scene | PCG range base (µs) | Δ min (µs) | A/A (µs) | share of the step |
|---|---|---|---|---|
| dam | 5318 | −69.6 | +4.1 | −0.24% |
| seep | 3375 | −60.2 | +8.6 | −0.32% |
| letters | 5384 | −66.8 | −17.5 | −0.16% |
| paddleboats | 4261 | −46.5 | −21.8 | −0.15% |

On dam, flag by flag, top alone is −31.6 µs (A/A +5.1) and down3 alone is −28.2 µs (A/A +4.4).

**c2, no-go (up3 fused into up2):** not exact as built (pmgC differs from the first fused up2 on) and not faster: dam
+41.3 µs (A/A −34.1). The fused 512-lane pass's extra L3 phase (4 barriered passes, about 256 lanes of recompute) costs
about the dispatch it saves. I did not chase the inexactness, since there is no time win to keep.

## 6. One live slot, attributed by pass (slot 0, in situ)

Measured: each pass is timed alone in situ, plus a per-workgroup clock. The data is the M6 audit at b786aff5
(`m7/a-dam.txt`, `m7/a-paddleboats.txt`); the PCG F kernels have not changed since. Bytes are compulsory f32 traffic
(the brick's channels, contiguous by slot), not counted DRAM traffic.

| class | dam (µs, share of the Σ) | paddleboats (µs, share) |
|---|---|---|
| F passes (preS/U, down0/1, up1, postU/S/Z, direction, update, wallsF, bodyZ) | 706 (69%) | 1056 (72%) |
| C passes (native ×4, down1C/2/3, up3/2, wallsC) | 257 (25%) | 340 (23%) |
| top | 56 (6%) | 67 (5%) |

| F pass (dam) | µs | groups | bytes / group | GB/s | groups in flight avg / peak | group p50 | tail |
|---|---|---|---|---|---|---|---|
| pmgUpdate (r = b − A(x+αp), max \|r\|, the first smoothing into W0) | 134 | 591 | ≈ 25.6 KB | ≈ 90 | 45 / 63 | 12.7 µs | 29 µs |
| pmgDirection (x += αd, d = z + βd, q = Az + βq, p.q) | 120 | 591 | ≈ 30 KB | ≈ 115 | 53 / 64 | 11.2 µs | 13.5 µs |
| pmgPreS / pmgPostS (shifted 8³ blocks, 6 red-black passes) | 70 / 69 | 712 | ≈ 22.4 KB | ≈ 230 | 72–76 / 96 | 6.9–7.5 µs | 8–13 µs |
| pmgPostZ (+ the z.r dot) | 71 | 476 | ≈ 22.4 KB | ≈ 150 | 82 / 96 | 7.8 µs | 1.6 µs |
| pmgDown0 (residual, restrict, L1 smoothing) | 67 | 476 | ≈ 18 KB | ≈ 130 | 91 / 96 | 11.3 µs | 0.7 µs |
| pmgPreU / pmgPostU | 47 / 55 | 476 | ≈ 22.4 KB | ≈ 190–225 | 85 / 96 | 7.1 µs | 1.8 µs |

Paddleboats shows the same shape. Direction and update are 392 µs of a 1463 µs Σ (27%), with tails of 16 and 52 µs.

**Verdict: latency- and tail-bound, not bandwidth-bound.**

- The best pass reaches about 58% of 400 GB/s by compulsory bytes; most reach 25–55%.
- The smoothers sit at the occupancy cap (96 groups of 512 lanes, 72–91 in flight). Each group is a dependent chain:
  stage → slot → tile loads → barrier → the row's 7 operator loads → 6 barriered red-black passes → store.
- Direction and update are capped at 64 groups in flight (registers), with long tails.
- f16 for the preconditioner's internal vectors only is out. Those vectors are the W0/W1 tile loads and the smoother
  write: about 8 of the smoothers' ≈ 44 B a cell, and none of direction/update's. Even with perfect bandwidth scaling,
  that is ≤ about 6% of the PCG range, under the 10% gate. Not built.

**The matching levers**, proposals only:

1. pmgUpdate and pmgDirection occupancy, from 64 to 96 in flight, via a register cap (the regcap.sh loop) or by
   splitting the reductions out of the hot loop. Plus longest-first group order against the 13–52 µs tails.
2. The smoothers' load chain: issue the row's operator loads (diag, K, RKU, rhs) before the tile barrier, so the two
   global-load phases overlap. Exact.

## 7. Levers built (round 1 and 2, m7ab, PCG range)

m7ab, with A as HEAD and B as the candidate in one process. Every buffer is compared word for word after the range, and
the range is timed interleaved. Δ is the change in the minimum, against A's own A/A spread. Flags are in the scratch
tree `fluid-scratch/gpu-shape/m8/tree-c8`. Register headroom comes from `m8/regprobe.py`: R pinned registers are
injected, and headroom is the largest R at which psoinfo's thread cap stays 1024. The calibration: ≥ 26 gives 3 groups of
512 lanes a core (96 in flight); ≤ 22 gives 2 (64).

| variant | dam | paddleboats | verdict |
|---|---|---|---|
| update: x+αp tile, coupling-only tests (+ bodies' W staged in 960 B wgmem) | exact, −277 µs (A/A −9) | exact, −145 µs (A/A −9) | landed; seep exact −84 µs (A/A −8), letters exact −298 µs (A/A 5) |
| direction: F/C split dispatch + z tile, coupling-only tests | exact, +11 µs (A/A 39) | exact, +41 µs (A/A −11) | no-go |
| direction split alone | exact, +11 µs | exact, +64 µs | no-go |
| smoothers: row operator and rhs loads before the tile barrier | exact, +295 µs | exact, +169 µs | no-go |
| update tile, no neighbour tests | exact, −200 µs | **inexact** | no-go (below) |

- **Update.**
  - HEAD's limiter was registers, not wgmem. The eager 21-load gather left 6 registers of headroom, so 64 in flight.
    The tile stencil (x+αp on the 10³ halo, the couplings loaded beside it and reused by the first smoothing) has 26 on dam.
  - Paddleboats' mbApply cost 8–12 more registers. Staging the bodies' W_x + αW_d and centres in workgroup memory, with
    the body term formed after the barrier, recovers most of it: headroom 24 offline, 22–24 with the full neighbour
    tests.
  - The pApplyR eager form is gone. Every remaining fineApply caller is the tested form.
  - In situ on paddleboats (slot audit, `m8/p1-paddleboats.txt` vs `h0`): pmgUpdate peaks at 96 in flight, against
    HEAD's 63. That is 81 on average, against 47. Timed alone, the pass takes 140 µs, against 189, and its clock tail is
    9 µs, against 49. So headroom 24 does give 3 groups a core; the ≥ 26 calibration line was conservative.
- **Direction.** Its limiter is the C half's seam loop (headroom 16). The F half reaches 28 with a z tile and
  coupling-only tests, but only once split into its own dispatch, and the split costs more than the tile wins.
- **Smoothers.** Already at 96 in flight (headroom 26–30). Hoisting the row loads above the barrier drops them to 18–22,
  which is 64 in flight, and costs more than the overlap gains. Hoisting the rhs alone still costs 4–6 registers.
- **Tails.** Much of the 13–52 µs per-workgroup clock "tail" is an artifact of clock mode. Dam's update spans 149 µs on
  the clock against 126 µs timed alone; stragglers start late only with the clock on (inference). Longest-first
  ordering was not built.
- **Top cycle.** Already red-black (parity colouring, 1024 lanes, rows ≤ 512 in workgroup memory). Its 25 µs is about
  20 dependent barriered passes at about 1.3 µs each, which is a barrier floor, not serial GS. Not worth a
  multi-colour or Chebyshev rebuild.

**Latent hazard: non-row cells' iterate (report only, not fixed).**
- HEAD's residual and A z read neighbour values only behind tests: kl ≠ 0, ku ≠ 0, and the neighbour existing.
- HEAD's former eager update form loaded every existing neighbour's x+αd, including diag-0 (non-row) cells, and relied
  on the coupling test alone to keep them out of the sum.
- Measured: dropping the tests (0 × value for a zero coupling) left dam exact but made paddleboats diverge, with
  differences up to about 1.7e9 ulp in pres, pmgF and pmgC after one range.
- Inferred: diag-0 cells inside row bricks, or native non-rows, hold non-finite x or d in normal runs.
  pressureRowsZero clears only the zero-list bricks, and nothing found keeps a diag-0 cell's X finite.
- The landed update keeps the coupling tests for this reason. Any future test-free stencil must zero or keep finite
  those cells first.
