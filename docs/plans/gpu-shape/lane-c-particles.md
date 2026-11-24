# GPU-shape lane C: the particle kernels (2026-11-24)

Scope: uTransfer (P2G gather and forces), pSeed, pMove (G2P, drift, merge, split; and pSolid, which is pMove's body
for the records it defers near statics) and pDeposit (P2G scatter). Plan: docs/plans/gpu-shape-2026-11-23.md.
Scratch and every run's log and jsonl: ~/code/me/fluid-scratch/gpu-shape/lane-c/ (variants.py holds every variant
below as text substitutions; mk.py builds a CANDTEXT from the HEAD dumps in h/; run.sh runs stepab through
`fluid-gpu-run.sh timing`; fam.py prints the family table).

## Method

stepab (tools/experiments/gpu-shape/stepab.mts) on all four gate scenes at step 100, S=7, family minima in the step.
Ablations guard code with `cfg.x==0xffffffffu` (false at run time, kept by the compiler), so the cost of a block is
the family's drop without it; they are inexact and timing only. Families are timed separately, so one CANDTEXT
carried one variant a family. Candidates were gated for exactness with an empty NOEXACT, then with TOL=ulp4 where
relaxed math moved floats.

## Where the time goes

Family minima at HEAD 5dbad0bd (ms in the step):

| family | seep | dam | letters | paddleboats |
|---|---|---|---|---|
| uTransfer | 0.76 | 1.27 | 2.21 | 1.40 |
| pSeed | 0.40 | 1.08 | 2.04 | 0.59 |
| pMove | 0.16 | 1.00 | 1.88 | 0.51 |
| pDeposit | 0.20 | 0.62 | 1.09 | 0.45 |

Ablations (Δ of the family, dam / letters / paddleboats):

| kernel | block removed | Δ |
|---|---|---|
| pSeed | the cell mask count (pSeedCellMask, and so the write) | −72 / −80 / −60% |
| pSeed | the site loop | −36 / −42 / −29% |
| pSeed | the seam test (psSeam; changes control flow) | −13 / −33 / −5% |
| pSeed | the write pass alone | +15 / +28 / −7% (codegen) |
| pSeed | pUncovered, pCellMass | ≈ −2, −3% |
| pMove | the merge (run, cell-centre depth, mass scan) | −32 / −31 / −9% |
| pMove | the G2P sample | −8.5 / −17 / −1% |
| pMove | the split block | ≈ −10% |
| uTransfer | the face loop | −31 / −27 / −36% |
| uTransfer | the near-surface depth term | −13 / −10 / −10% |
| uTransfer | the staging; the forces; the liquid store | −9; −5 / −4 / −7; −3% |
| pDeposit | the scatter (nbfP2GScatterAxes) | ≈ −60% |
| pDeposit | the staging | +5–12% (codegen) |

So pSeed is the cell count and the site loop, pMove the merge's reads, uTransfer the face loop and pDeposit the
scatter's arithmetic and window traffic.

## Landed: e30b1527, pMove reads the merge's depth vertices once

The merge (pMassDepth at the cell centre) and G2P (fgDepth(q)) read the same cell's reach word, bricks and eight
FC_DEPTH vertices. These are atomic loads, which the compiler never merges. pMassDepth2 reads them once for both
sums, and G2P reads again only when q was outside a merge run or the record merged. The split block now runs only for
a coupled record of mass over 1, the only kind nbfSplitWanted can pass; this part is exact on its own.

| scene | pMove | exactness |
|---|---|---|
| letters | 1.897 → 1.610 ms (−15.1%) | set and fmoved within ulp4 |
| dam | 0.993 → 0.914 ms (−7.9%) | fmoved: 50 words within 2 ulp |
| paddleboats | 0.512 → 0.456 ms (−11.0%) | bit-identical |
| seep | 0.153 → 0.150 ms (−2.2%); pSolid −10% | bit-identical, pSolid too |

- **GPU wall:** dam −0.49%, letters −0.70%, seep −0.69%. Paddleboats was +0.43%, but its unchanged families moved
  +0.045 ms against an A/A of +0.048, so that is noise.
- **pSolid:** it shares pMove's body, so it took the change as well. Measured in reverse on seep (HEAD against the
  old text), the old text is slower: pSolid 0.161 → 0.180 ms and pMove 0.160 → 0.172 ms, all bit-identical.
- **Relaxed math:** Tint emits relaxed fp, so recompiling pMove moves G2P's pic and a few positions by 1 ulp on
  letters and dam, and the set's speed bounds with them.
- **stepab change:** the ulp tier now classes the set's BOUND_F..BIN_START words as f32, with vel's 1e-6 floor.
  These words are atomicMax of bitcast non-negative speeds. Without this, letters fails on 1–5 bound words, each a
  few ulp off. The tier had treated all of set as exact. This is a gate-policy change, landed in the same commit so
  that it is easy to revert.

## Ruled out (measured)

| candidate | result |
|---|---|
| M_dc alone, M_dc2 (vertices into an array, two sums) | same timing; neither bit-exact on letters (1-ulp set and fmoved) |
| M_heat (pHeat(q) once) | adds nothing on top of the landed change |
| M_pref (merge scan stops at i on uncrowded cells) | exact; pMove +3.6% dam |
| M_noscan-type register diets | +2–5% (codegen) |
| S_ws, S_ws2 (seeds listed in workgroup memory, then written a site a lane) | exact; pSeed +37% dam |
| S_seam (10³ seam bits staged once) | exact; pSeed +15% dam, +2% letters, +79% seep |
| S_brk (site loop breaks at the 8-mass cap) | exact; −0.8% |
| T_reach (reach bit read once a cell) | exact; ≈ 0 |
| T_split (lane A2's one-cell-a-lane shape: uTransfer as two 256-lane groups a brick over F_ALL X8) | uTransfer +73% letters: each half redoes the brick's staging, which is a third of the kernel |
| D_perm (SIMD lanes take records count/32 apart, so they do not scatter into one cell's faces together) | +3% dam, ≈ 0 elsewhere |

uExt4 and uCache: lane A2 measured uExt4's reshapes (m1-distinct-once.md, "uExt4 and the uExtend pattern"); all are
no-go because its time is a tail of long pages. uCache's loop is in uCache4 (C pages, two tiles a lane), which is
small; not tried.

## Atomic contention bound for the scatters (coordinator item)

Frozen-state variants in which an atomic RMW becomes an atomic store (the sums are wrong; this is a timing bound
only):
- D_lst: the per-record workgroup scatter (wMom, 27 faces an axis, momentum and weight).
- D_gst: the brick's flush to fdep, which is the global atomicAdd that a brick colouring would replace.

| scene | D_lst (workgroup) | D_gst (global flush) |
|---|---|---|
| dam | 0.617 → 0.599 (−2.9%, A/A −2.1%) | 0.630 → 0.630 (0.0%) |
| letters | 1.091 → 1.072 (−1.7%) | 1.086 → 1.089 (+0.3%) |
| paddleboats | 0.433 → 0.438 (+1.1%) | 0.447 → 0.439 (−1.7%) |
| seep | 0.216 → 0.212 (−1.8%) | 0.186 → 0.183 (−1.9%, A/A −2.2%) |

uTransfer has no scatter atomics: it gathers fdep with plain loads. Its bound is zero by construction.

**Conflicts per word** (from the shapes, not counted):
- **Workgroup scatter:** a face word takes one add from every record in the 3×3×3 cells about it, so about 27 × the
  records a cell.
- **Global flush:** a brick flushes its 11×10×10 window an axis (1100 words) over 512 owned faces. That is at most
  2.15 writers a word on average, and 8 at a corner face, with each pair of colliding writes coming from different
  workgroups at different times.

The bound is at most 3% everywhere (within A/A on most), far below the 15% threshold, so graph colouring is not built. An 8-colour
scatter would add 7 dispatches at about 3 µs each (about 21 µs) to save at most about 0 µs from the flush. The
D_perm result above agrees: spreading the SIMD's lanes across cells, which removes same-word collisions within a
SIMD, did not help either. pDeposit's time is the scatter's arithmetic and window traffic, not serialisation.

## pSeed: does it pay for cells that cannot seed? (coordinator item, 2026-11-24)

**Counters.** Counters were added to set words 202–211 at the frozen state. stepab's exactness mode prints the base
and candidate values, and their difference is the count (variants SC1 and SC2). The table is for the step's main
seed (z0, cfg.w = 0). "Launched" cells are 512 for each live F_ALL brick.

| scene | live bricks | (a) go bricks | (b) past psCellGate | n0 < goal, held < 8 | (c) site loop | (d) ≥ 1 seed | SIMD rounds with a loop cell |
|---|---|---|---|---|---|---|---|
| dam | 2063 | 35% | 6.2% | 4.2% | 1.03% | 0.68% | 23% of go bricks' |
| letters | 3217 | 31% | 7.5% | 4.6% | 1.59% | 1.21% | 31% |
| paddleboats | 1695 | 28% | 5.0% | 3.1% | 0.76% | 0.55% | 20% |
| seep | 1258 | 19% | 1.9% | 1.8% | 0.09% | 0.04% | 4% |

The cells that pass n0 < goal and held < 8 but do not reach the loop are the covered shallow cells. Each of them
takes pUncovered (27 run walks) and psSeam before it returns.

**Time split** (ms in the step; deltas of S_noloop and S_nocount, each measured against its own base):

| scene | site loop | gates and early-outs | the rest (prologue, staging, go = 0 bricks) |
|---|---|---|---|
| dam | 0.446 | 0.312 | 0.268 |
| letters | 0.820 | 0.771 | 0.400 |
| paddleboats | 0.164 | 0.188 | 0.244 |
| seep | 0.053 | 0.151 | 0.185 |

**Prototype.** (c) is far under the 25% go line on every scene, so the compaction was prototyped in-workgroup
(S_list), which keeps the brick's staging and needs no new pass:
- Phase A runs every cell's gates up to the loop and pushes a looping cell to a workgroup list (local index, n0,
  held, d and the known, uncovered and covered bits; at most 256 entries, and any overflow loops in place).
- After a barrier, phase B runs the list one cell a lane, then counts and writes that cell.

Result: exact on all four scenes (set identical, parts identical up to order), but slower:

| scene | pSeed |
|---|---|
| dam | +49.6% |
| letters | +33.4% |
| paddleboats | +49.6% |
| seep | +57.3% |

The same text with nothing pushed (S_list0, dam) is +43.5%. So the restructure costs about 44% on its own (cap 512 →
448; the list, the barrier and the split loop), and the compaction itself adds +0.013 ms on top of it.

Packing the looping cells into fewer SIMDs buys nothing. The idle lanes of a sparse SIMD are not what the loop
costs: the loop's cost is each looping lane's serial chain of sites, which stretches its brick's workgroup.

**Verdict: no-go** for packed seed groups. Untested: a split kernel that moves the site loop out of the brick
workgroups altogether. The loop would run over a global list in its own dispatch, so the brick groups end sooner,
which is a latency bet. It needs a new kernel, a list region and an args word (an ABI change), and stepab cannot
prototype a new placement. Its bound is the site-loop column above minus about 6 µs for the two extra dispatches,
and it holds only if the global loop's own chain is short. A brick-level go list would address part of "the rest"
(65–81% of launched bricks have go = 0); that was not measured.

### The restructure tax, attributed offline (regprobe and psoinfo)

The +43.5% is register pressure. HEAD's pSeed sits at Metal's 512-thread cap with under 4 registers of headroom
(regprobe: 4 pinned registers drop it). S_list0 compiles to 448: for 256-lane groups that is one group a core instead
of two. Each piece alone (the list's declarations, the split loop, the barrier, the push) stays at 512, so no one
piece is the cause: the combined two-phase body crosses the step. Not the barrier, and not workgroup memory (HEAD
holds about 3.3 KB, the list 3 KB more; psoinfo's static length reads 0 because Dawn passes it as an argument).

## Landed: a73871bb, pSeed tests a site a lane

The site loop's serial chain moved to one (cell, j) a lane. Phase A runs every cell's gates (two cells a lane) and
lists a looping cell with its goal − n0 sites (one packed atomicAdd: cells low 16 bits, pairs high). Phase B runs one
site a lane and sets its bit in the cell's workgroup mask with atomicOr; the mask is order-free and the held skip
depends on j alone, so it is exact. The count and write passes read the masks. The deep cell's uncovered is taken up
front, where nbfSeedSiteOk would have read the loop's value. The register cap rises 512 → 576, and workgroup memory
grows by about 10 KB.

stepab against af235dd9 (pSeed in step, min over S=7): set identical and parts identical up to order on all four.

| scene | pSeed | GPU wall |
|---|---|---|
| dam | 1.034 → 0.964 ms (−6.8%) | |
| letters | 2.079 → 1.905 ms (−8.4%) | −0.79% |
| paddleboats | 0.615 → 0.556 ms (−9.7%) | −0.42% |
| seep | 0.408 → 0.383 ms (−6.2%) | −0.11% |

A lazy-uncovered variant (S_site2: uncovered computed in phase B) was faster still (dam −17%, letters −18%) but not
exact: z0 records differ (360–112754 f32 words). The deep cell's uncovered must be the loop's value, so it is not
landed.

## pSeparate's heavy groups: how often (coordinator item, 2026-11-24)

pSeparate and pMove run one record a lane, 128 records a group, over the cell-sorted bin. A "heavy brick" is a
crowded cell: each of its records scans the 8 runs of its 2x2x2 cells in order (sepSum), so the per-lane serial
chain is the neighbour count w, and the cell's few groups form the tail.

**Crowding is persistent on dam.** freq.mts (fluid-scratch/gpu-shape/lane-c/freq) read every step's bins over 300
steps. Dam's largest cell run is 309 records at the median (mean 361, max 1242). Cells over 32 records: 150 at the
median (max 1884). Letters: 157 / 87. Paddleboats: 26 / 1. Seep: 49 / 5.

**The bound, at frozen states.** stepab P_cut128: a record with w > 128 skips its scan (timing only). The cut adds the
8-run w scan to every record, about +0.04 ms on dam, so the figures below understate the bound by that much.

| dam step | 30 | 45 | 60 | 80 | 100 | 120 | 150 | 180 | 240 | 300 |
|---|---|---|---|---|---|---|---|---|---|---|
| pSeparate (ms) | 2.75 | 1.33 | 1.88 | 2.28 | 1.17 | 3.27 | 3.49 | 1.57 | 0.86 | 0.88 |
| heavy records' share (ms) | 0.85 | −0.05 | 0.69 | 1.03 | 0.16 | 2.17 | 2.49 | 0.58 | −0.05 | −0.06 |

That is a mean of 0.78 ms over the 10 states, held through the splash (steps 30–180) and gone by step 240. A2's 72%
tail (step 100) matches the heaviest states (66–71%). States differ run to run: the record order is atomic, so the
splash's crowding is chaotic, and stepab's base at step 100 ranged 0.94–1.53 ms across runs.

Elsewhere the same cut does not pay at step 100: letters +6.8%, paddleboats +9.5%, seep +2.7%. The w scan costs more
than the skipped work saves.

Per-step in-run replays were not usable as a bound. A replay of the next step from a saved state, base against cut,
moved ±1–2 ms single-shot and ±1 ms as a minimum of 4. Their run means (−0.17 / +0.09 ms) are noise. The min-of-4 dam
run also overlapped another lane's tsc.

**Tiling: no-go.** P_tile stages the union of a heavy group's 2x2x2 cells, 128 records at a time, in workgroup memory,
in (z, y, x) order: that is every base's c order, so the result is exact. Results, from one gate each:

| variant | dam | letters |
|---|---|---|
| P_tile64 (heavy groups staged) | +17% | +20% |
| P_tile0 (never heavy: the restructure alone) | +7.6% | +18% |

The barriers, workgroup memory and uniform control flow cost more than the shorter latency saves, and a heavy group
still runs the full chain of w adds.

**The shape that remains: spread the chain.**
1. pSeparate skips records over a threshold and lists them, using the record's own cell run as the test (one fgRun,
   not the 8-run w scan the cut paid for).
2. pSeparateHeavy runs one heavy record a subgroup. The 32 lanes stride the concatenated 8 runs in order, and each
   lane computes one neighbour's term.
3. The terms are added in j order. A ballot of the passing lanes, then subgroupShuffle adds in lane order: the same
   nbfSeparationAdd terms in the same order, so it is exact.

This needs a list region and an indirect args word (the ABI change the coordinator OK'd at ≥0.5 ms average on dam;
the bound measured here is 0.78 ms). Not built yet.

### Why dam's cells crowd (diagnosis, no change; crowd/diag.mts)

Read from records at frozen states, one run per engine. The run is chaotic, so the exact cells differ from run to run.

**Who is in the top cells.** Brick dam's top cells (steps 60 / 120 / 150) hold 296–1088 records each. All are unit
mass, coupled (ballistic ≤0.3%), with a surface depth: median −0.2 to −0.8, at most 0.4. They sit on walls and
corners (x=0, z=0, x≈123, z≈120–127) and in surface clumps.

| | cells over 32 records | their records | their records with depth < −1 |
|---|---|---|---|
| step 60 | 737 | 43.7k | 21% |
| step 120 | 287 | 18.8k | 5% |
| step 150 | 250 | 15.9k | 6.5% |

**Why merge does not bound them: the depth gate.** pMove's merge and its cell-mass cut (Ferstl et al. 2016 §3.3, the
2n = 16 unit bound) run only where the cell-centre depth is under −1. Every surface cell is unbounded. The old engine
has the same gate (uniform-narrow-band-flip.ts), so the brick port is faithful.

**Why separation does not spread them.** pSurfaceShift keeps only the tangential part of the push in a surface cell.
The constraint averaging divides by the overlap count (500–1000 here), so inside a pile the push cancels. The piles
are not compressive: radial velocity about the centroid is within ±0.2, for both the records' own velocity and the
grid's. They travel as coherent clumps instead. One clump went 911 → 919 → 1088 records over steps 118–120, moving
about a cell a step, at its records' velocity: the centroid moved (0.8, 2.0, 0.3) and (1.1, 1.1, −0.2) cells a step
at v ≈ (0.5, 0.5, 0.3) m/s, with dt/h ≈ 2.3 cells per m/s. A corner sheet at (127, y, 0) climbs 4.1 then 3.4 cells a
step at v_y = 1.75 → 1.59 → 1.41, which is exactly −g·dt a step.

Correction (2026-11-24): an earlier note here read before.xyz as the grid's velocity and called the piles
"decoupled". before.xyz at a frozen state is the last projection's FLIP delta d (nbfG2P), and d_y ≈ −0.16 = −g·dt.
Coupled records drift with the grid's velocity (nbfDrift takes pic), and the piles move at v·dt/h, so the grid
carries them. They are free-surface sheets and clumps with no pressure response, which is physical. Their density is
passively advected: a divergence-free flow neither builds a pile nor relaxes one.

**Against the old engine (dam, same steps).** Both engines crowd at the splash. The old engine relaxes; the brick
engine holds and grows its piles. Both share the depth gate and the surface projection; the difference is one run
each and not yet explained. The old engine's crowded cells hold more heavy and deep records (1911 heavy, 21k deep at
step 60, against the brick's 12 and 9k), so more of its piles reach the merge.

| | old engine: largest / cells over 32 | brick: largest / cells over 32 |
|---|---|---|
| step 60 | 357 / 1174 | 530 / 737 |
| step 120 | 169 / 94 | 816 / 287 |
| step 150 | 78 / 42 | 436 / 250 |

Letters (brick) shows the same thing on a smaller scale (steps 150 / 200: largest 295 / 360, 82 / 95 cells over 32,
86–92% at surface depth).
