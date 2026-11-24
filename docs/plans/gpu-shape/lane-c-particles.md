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
