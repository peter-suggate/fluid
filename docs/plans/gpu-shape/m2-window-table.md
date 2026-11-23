# GPU-shape M2: a per-step window table, go/no-go (lane B2, 2026-11-24)

**Verdict: NO-GO.** The owner-window staging that the table would replace costs **0.27 ms (dam) and 0.17 ms
(paddleboats) a step in total** over all 13 staging placements, so even removing it outright cannot reach the 1 ms /
4% go line. The prototype is exact (0 words differ in every kernel's written buffers, both scenes) and one build a
step suffices (verified), but its net is **−0.03 to −0.18 ms a step (0.1–0.5%)**. No landing design is proposed.

Plan: docs/plans/gpu-shape-2026-11-23.md (M2). Tool: tools/experiments/gpu-shape/wintable.mts. Data:
docs/plans/gpu-shape/m2/ (wintable JSON per run with HEAD and `git status`, logs, caps.txt).
Phase attribution of the four staged kernels: m1b-staged.md.

## Hypothesis tested

fgWindowStageAt (lib/methods/narrow-band/fluid-grid.wgsl.ts) fills, per F brick and per kernel, 27 brick slots
(fgBrickSlotOfDir: page then child, 2 dependent loads), 216 tile words (page C slot + h bit: page, then CMASK) and
7 coarse-bit words into workgroup memory. A table written once a step (one record per F slot) would turn that into
coalesced loads in every staging kernel. Lane B's M1b numbers suggested ~0.7 ms (uTransfer) + 0.4 ms (uAdvect) of it.

## Runs

All `tools/gpu/fluid-gpu-run.sh timing`, HEAD 734dd9e2, rc 0, no GPU errors; frozen step 100 (census settings).
dam: dump, ceiling (`m2/dam-ceiling.*`), proto + step (`m2/dam-proto.*`), exactness re-run (`m2/dam-exact.*`);
paddleboats: one run of all modes (`m2/paddleboats.*`). Caps: `m2/caps.txt` (regcap.sh / psoinfo). Clock probe after
every round: 75.8–77.2 µs (top clock) except paddleboats' ceiling rounds 1 and 6 (153, 504 µs) and proto round 1
(178 µs); the minima used come from top-clock rounds (min of 7). Other sessions' uncommitted files were as listed in
m1b-staged.md; none is a brick kernel.

## 1. Ceiling: every staging placement (13 a step on both scenes)

The step's placements whose entry calls fgWindowStage / fgWindowStageAt / fgRingStage (from the schedule's own
kernel text; no helper calls them): lResolve ×3, uCache, pLiquid (part 1, after the head), lTiles, lClear, lResolveD,
lDistance, uAdvect, pDeposit, uTransfer, uExtend (part 2). Generic truncation cuts per placement, in situ: `pre`
ends the entry just before the window call, `win` after the barrier that follows it, `ring` after the ring's barrier.
window = win − pre, ring = ring − win (µs, min of 7):

| placement | dam full | dam window | dam ring | paddle full | paddle window | paddle ring |
|---|---|---|---|---|---|---|
| lResolve (p1) | 246 | 19.3 | | 188 | 14.5 | |
| uCache | 460 | 19.3 | | 431 | 15.2 | |
| pLiquid | 162 | 17.5 | 78.5 | 129 | 14.3 | 60.7 |
| lResolve (p2, ×2) | 246 ×2 | 19.5 + 19.3 | | 191 ×2 | 15.3 + 15.3 | |
| lTiles | 184 | 19.5 | 90.0 | 160 | 15.6 | 67.0 |
| lClear | 284 | 19.7 | 86.8 | 221 | 15.2 | 67.3 |
| lResolveD | 166 | 9.8 | | 70 | 2.8 | |
| lDistance | 932 | 19.4 | | 734 | 15.6 | |
| uAdvect | 1130 | 64.0 † | 61.0 | 1337 | 8.9 | 156.0 |
| pDeposit | 675 | 7.8 | 56.2 | 489 | 4.8 | 43.1 |
| uTransfer | 1962 | 19.3 | 266.0 | 1846 | 12.4 | 168.8 |
| uExtend | 1591 | 16.3 | | 1969 | 20.1 | |
| **sum** | 8285 | **270.7** | 638.5 | 7954 | **170.1** | 563.0 |

† staged.mts's dedicated cut gives 11 µs for the same phase (m1b-staged.md): on the heavy kernels a difference of two
in-situ minima carries ~±50 µs of run-to-run noise (the head's atomic slot claims make each prefix replay differ).

- The window is ~15–20 µs per placement (≈ one F_ALL-wide 2-load walk; the build kernel below costs 28–31 µs alone,
  launch included). Lane B's 0.71 / 0.42 ms "window" in uTransfer / uAdvect was their **C pages** (0.53–0.70 /
  0.41–0.43 ms; `pre` here, `0-cpages` in m1b-staged.md).
- Ceiling of a window table: **0.27 ms dam, 0.17 ms paddleboats, minus its build**. Below the 1 ms go line on both.
- The φ rings (0.56–0.64 ms) are not table material: φ changes between ring users (lTiles, lClear, lResolve write
  PHI; uTransfer writes it), so a ring cache would have to be per producer–consumer pair (inference; uAdvect → pDeposit
  → uTransfer read rings of the same φ, at most ~0.3 ms of their ring phases, also under the go line).

## 2. Prototype (no lib/ edits)

- **Table** `m2win`: 256 words an F slot (27 slots, 216 tile words, 7 coarse words, exactly the workgroup arrays),
  4.8 MB dam (4913 F slots), 10 MB paddleboats (10240). An extra read-only binding on each variant (device limit 10
  storage buffers; the engine's own cap is 9, bypassed in the tool).
- **m2Build**: one workgroup (256 lanes) an F_ALL brick, fgWindowStageAt's own walks once, into the slot's record.
  Bound with the next window user's indirect dispatch. Cap 1024.
- **tabA**: fgWindowStageAt's body replaced by table loads into the same workgroup arrays (no new workgroup memory).
  **tabB**: arrays dropped; fgBrickSlot / fgTileCSlot / fgTileH / fgWinCoarseBit read the record in place (device
  loads). The slot comes from the entry's `slot` (every window is centred on the workgroup's own F_ALL brick).

**Exactness (measured): bit-identical.** Per placement: restore, ring, the step's placements before it with the single
build before the first window user, snapshot every buffer; from the snapshot run base (reference), base again
(control), tabA, tabB, comparing every read-write binding of the kernel word for word (GPU compare): **0 words differ
for control, tabA and tabB at all 13 placements on both scenes**, and the snapshot's table equals a fresh build at
every later placement (0 words). A whole-step bitwise compare is not usable: two replays of the frozen step differ
(base vs base 6.9 M words dam, 7.1 M paddleboats: the head claims slots by atomics), so exactness is per kernel from
one snapshot.

**Caps (measured, Metal thread cap without the size attribute):** tabA = base for all 11 kernels; tabB = base except
uExtend 768 → 704. m2Build 1024.

**Timing (in situ, min of 7, interleaved, table built untimed just before):**

| µs | dam base | tabA Δ | tabB Δ | paddle base | tabA Δ | tabB Δ |
|---|---|---|---|---|---|---|
| lResolve ×3 | 234 / 233 / 233 | −12.5 / −12.3 / −9.7 | −18.0 / −17.2 / −16.5 | 190 / 190 / 190 | −14.0 / −11.0 / −10.3 | −12.4 / −11.5 / −13.8 |
| uCache | 440 | −6.8 | −0.3 | 428 | −3.1 | +8.8 |
| pLiquid | 152 | −7.4 | −9.2 | 127 | −4.0 | −6.8 |
| lTiles | 176 | −8.2 | −12.7 | 159 | −5.1 | −11.4 |
| lClear | 259 | −1.8 | −4.5 | 218 | −5.8 | −10.3 |
| lResolveD | 138 | −1.8 | −6.5 | 70 | −0.2 | −1.4 |
| lDistance | 897 | −14.6 | −8.3 | 738 | −21.8 | −21.1 |
| uAdvect | 1146 | +25.0 | +4.5 | 1320 | −30.0 | +8.3 |
| pDeposit | 683 | −17.8 | −9.0 | 446 | −0.9 | +4.4 |
| uTransfer | 1704 | +8.8 | −13.8 | 1777 | −103.8 | −91.6 |
| uExtend | 1500 | +3.4 | −25.4 | 1959 | −1.7 | +25.0 |
| **sum** | 7795 | **−55.7** | **−136.9** | 7811 | **−211.7** | **−133.7** |
| m2Build alone | | 30.9 | | | 28.5 | |
| **net (1 build)** | | **−25** | **−106** | | **−183** | **−105** |

The light kernels gain 5–18 µs each, about their window phase (tabB slightly more: no array stores). The heavy
kernels' deltas (−104 … +25 µs) are within the ±50–100 µs in-situ noise of those kernels (paddleboats' uTransfer
−104 µs is 8× its 12 µs window phase: inference, mostly noise). **Net per step: −0.03 to −0.18 ms (0.1–0.5%).**

**Whole step (both parts, engine passes, min / median of 7, ms):** dam base 31.294 / 31.600, tabA ×1 build
30.928 / 31.631, tabB ×1 31.046 / 31.459, tabA ×2 31.064 / 31.659, tabB ×2 31.389 / 31.602; paddleboats base
36.642 / 36.932, tabA ×1 36.234 / 36.541, tabB ×1 36.017 / 37.011, tabA ×2 35.379 / 37.984, tabB ×2 36.934 / 37.532.
Run-to-run spread is ±0.5–1 ms, so the wall cannot resolve a 0.1–0.2 ms change; no configuration moves the median
by more than its spread.

## 3. When the table must be rebuilt

The window reads three directory regions: NB_D_PAGE (page → C slot), NB_D_CHILD (F slots) and NB_D_CMASK (h bits).
Placements binding `dir` read-write (dam schedule): regionStill 3, resMark 10/13, resSource 11/14,
setClaimPages 12, setClaimBricks 15, setLeave 16, setLeavePages 17, setRecords 18, setLists 20, setCommitMask 28
(part 1's head), setMark 48, setMarkPages 49, setRecords 50, setLists 51 (part 2's set stage). Paddleboats: the same
head, plus the stamp stage's bodyStamp 32/33 and setKRelease 34/35, and part 2's set stage at 52–55.
By their text (brick-set.wgsl.ts, brick-transfer.wgsl.ts, brick-residency.wgsl.ts, brick-region-heat.wgsl.ts,
bodies-stage.wgsl.ts): the PAGE/CHILD/CMASK writers are the residency claims and leaves (10–17) and the tier commit
setCommitMask (28); bodyStamp, setKRelease, regionStill and part 2's set stage write only F/C records (class, cut,
key, neighbour words, C liquid tiles). The first window user (lResolve, 37 /
41) follows the head. **One build a step**, after the head (before lResolve), ~30 µs; measured: the table built there
equals a fresh build at every window user of the step on both scenes. Invalidation in a landing would be "rebuild
after any PAGE/CHILD/CMASK writer before the next window user"; today that is once.

## 4. Go line

Net ≥ 1 ms a step (or ≥ 4%) on dam and paddleboats: measured net −0.03 … −0.18 ms; ceiling 0.17–0.27 ms. **NO-GO**;
no lib/ change and no landing design. The tool and data stay for reference.

Where the staged kernels' time is instead (measured, m1b-staged.md): uTransfer + uAdvect C pages 0.94–1.11 ms a step
(occupancy 6–13%), uExtend reach + a-sweeps ~1.2–1.5 ms and its barriers 0.22–0.29 ms, pDeposit scatter 0.26–0.42 ms,
φ rings 0.56–0.64 ms over six kernels.
