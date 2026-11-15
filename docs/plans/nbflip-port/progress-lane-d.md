# NB-FLIP port, lane D: the dam residency defect (progress)

Brief: the baseline 128^3 dam break (`high-resolution-dam-break`) stops in about half its fenced
300-step runs with `GPU failure record: invalid support (layout build N: residency audit:
liquid outside the resident pages, or a reader outside the closure)` at steps 178-247
(plan §4.0). Find the root cause and fix it in the general path; no widened audit, no clamp,
no runtime pool growth.

## 2026-11-15 STARTED (lane D)

HEAD 33b78ebb. Lane A profiles on the check slot; lanes P1 and P2 edit the main checkout, so
every lane D GPU run uses a read-only export (`tools/nbflip-export.sh`) of HEAD, with the fix
as a recorded patch for candidates.

## 2026-11-15 ROOT CAUSE

Diagnostic (scratch only, `fluid-scratch/lane-d/diag-dynamic.patch`): the census recorded,
for the first absent-near tiles, the tile, its flags and page, its velocity keys, the nearest
previous seed, its particles, and the 8 corner phi values of the 4h tile with the holder of
each corner in the wide vertex walk (umWideVertexSlot). 16 baseline runs, 2 failed (steps 179
and 248); both showed the same picture:
- the failing tiles are 4h tiles at y = 24 tiles (cell 96, a block boundary), flags 11111,
  in pages not resident; velocity 0, no particles; the nearest previous seed 1-5 tiles
  beyond its radii, so no reader had reached them;
- their lower corners read phi -0.1169 m (about -18.7 cells: a deep-column value) where
  far air holds AIR, 0.1 m;
- those corners' own block (the page below) was not stored, and the walk took them from the
  next page's low skirt (holder bits 1110, chosen 1).

Mechanism:
1. At t = 0 the column starts as BULK16 regions. A demote hands every vertex the demoted
   block held to its next holder (uniform-bulk-page-transfer.ts), so the page above's
   skirt took the deep phi of the column's top plane.
2. A later promote copied those vertices back into the reclaimed block, and the skirt kept
   its copy: harmless while the block below was stored, since the walk reads the vertex's
   own page first.
3. Near step 180 the block below went idle and retired. A retire wrote nothing: the vertex's
   holder moved to the skirt with no copy (uniform-page-initialise.ts: "never touched when
   it retires"). The census then read phi -0.117 m there, liquid in far air outside every
   resident page, and the absent-near audit fired.

The audit was right. The defect is that a retire, unlike a demote, broke the invariant that a
vertex's value does not change when its holder moves.

## 2026-11-15 FIX

`lib/methods/uniform/uniform-page-initialise.ts`: the retire handoff. In the commit's page
initialise pass, after the claims' AIR image, a workgroup per block of the retired list
(still quarantined, texels intact) walks every texel on the block's faces:
- the block held the vertex before the commit when no page ahead of it in the slot walk was
  stored before the commit (stored now and not claimed by this commit, or retired by it);
- its next holder is the first page of the walk stored now; the handoff copies the texel
  there, in the base (4h) and, where the holder has an h slot, the h store (the field's AIR
  where the retired block had none);
- nothing is written to a holder this commit claimed (its AIR image stands, as before) or for
  a block a demote retired (the demote already handed its vertices).
A vertex has one holder before a commit, so no two workgroups write one texel. This is the
demote's handoff applied to every retire: no numerical change, so no paper; the invariant is
that a vertex keeps its value across a change of holder.

Host twin `uniformPageRetireNext`; regression `tests/uniform-page-retire-handoff.test.ts`:
the dam's stale-skirt case at bits 3 and 5 (fails without the handoff), and a randomized
claim/retire property test (40 commits on a 3^3-page world, bits 3 and 5): after every commit
each vertex reads its previous value, or AIR exactly where a fresh block or nothing now holds
it, and no texel takes two writes. Without the handoff the property test fails on its first
round.

## 2026-11-15 RESULTS

Every GPU run is a check-slot fenced run of a read-only export (`tools/nbflip-export.sh`): base
HEAD (lib/methods/uniform tree bef61da162, the frozen baseline's), candidate the same plus
`fluid-scratch/lane-d/retire-handoff.patch`. Runs in `fluid-scratch/lane-d/runs/`. Every
run rc 0 unless stated.

**Dam completion (300 steps).**
- Candidate: 15 of 15 complete (`runs/fix/dam-fix1..15`).
- Baseline: lane G 6 of 11; lane D diagnostic builds 2 of 16 and the counter build below 4 of
  8 failed, at steps 178-248. Pooled, 12 of 35 failed (34%); 15 clean runs by chance at that
  rate: p = 0.0002.

**Mechanism check (counter builds, scratch only).** A build that counts, for every handoff,
the texels whose holder value differs from the retired block's and those where the holder
held phi < 0 (a stale liquid value about to surface):
- baseline behaviour (counting, no writes), 8 dam runs: the 4 that failed (steps 180, 201,
  245, 248) are exactly the 4 with stale negative holder texels (144-200 base, 72-128 h); the 4
  that completed have none;
- with the handoff, 5 dam runs: none, and all complete.

**Iterate change by scene (counter build with the handoff, one run each).**
| scene | retired blocks | handoff texels (4h base / h) | changed values (4h base / h, per field) | stale phi < 0 |
|---|---|---|---|---|
| paddleboats (300) | 576, all demoted | 0 / 0 | 0 / 0 | 0 |
| seep (1200) | 8704 | 4385 / 40737 | 153 / 0 | 0 |
| letters (240) | 14144 | 9424 / 73056 | 5993 / 1446 and 1876 | 0 |
| dam (300) | 2624-7104 | 1256-5147 / 16288-31107 | 644-1564 / 0-486 | 0 |
Paddleboats retires only through BULK16 demotes, which the handoff skips: it is iterate-
identical by construction. The seep, letters and dam are iterate-changing: after a retire a
vertex now keeps its value (positive far-air phi in all these runs) instead of reading the
value its next holder's skirt last held, so they are judged by the gates.

**Gates (tools/nbflip-gates.ts vs artifacts/nbflip-port/ref, GATED_STEPS.dam raised to 300).**
| scene | candidate runs | result | gated rows | worst margin (fraction of tolerance) |
|---|---|---|---|---|
| seep | 1 | PASS | 13 | 0.73 (G7) |
| dam | 15 | PASS, 15 of 15 | 16 | 0.17-0.45 (D3, D4, D5p, D5e, D8, D7t) |
| letters | 1 | PASS | 6 | 0.22 (L2) |
| paddleboats | 1 | PASS | 5 | 0.33 (P1) |
The dam's 180-300 rows stay informational. One of 15 runs (dam-fix6) exceeds the
informational D4+ tail drift (window mean +0.0013 against -0.0100 +- 0.0023): the 4 complete
references understate that window's spread. The baseline has the same bimodal tail (window
180-255 means of the 14 complete diagnostic runs: two at +0.0021 and 0.0000, the rest -0.009
to -0.012; candidate: -0.0010 to -0.0112).

A letters counter run returned rc 77: at 12:52:54 two other node processes (pids 52048 and
52497, not lane D's) caused GPU hangs, and the run's own process (53015) lost two command
buffers as an innocent victim. Invalid; rerun clean (the table's letters row).

**Timing (tools/nbflip-timing-round.sh, exclusive timing slot, base export HEAD 5763fa6f,
candidate the same uniform tree plus the patch; 0 Metal aborts; dam over its 165 timed steps).**
| scene | fenced median ms | fenced mean ms | pipelined ms/step |
|---|---|---|---|
| seep | 34.65 -> 34.26 (-1.1%) | 33.91 -> 33.60 (-0.9%) | 32.55 -> 32.38 (-0.5%) |
| dam | 45.06 -> 45.08 (+0.1%) | 50.97 -> 50.76 (-0.4%) | 50.14 -> 50.44 (+0.6%) |
| letters, round 1 | 83.44 -> 85.40 (+2.3%) | 69.88 -> 72.56 (+3.8%) | 70.84 -> 70.11 (-1.0%) |
| letters, round 2 | 84.12 -> 83.86 (-0.3%) | 70.55 -> 71.67 (+1.6%) | 70.06 -> 70.21 (+0.2%) |
| paddleboats | 49.31 -> 47.01 (-4.7%) | 46.79 -> 45.14 (-3.5%) | 43.57 -> 44.75 (+2.7%) |
Paddleboats does no handoff work at all (every retire there is a demote), so its +-5% is the
round's noise. Letters, which retires the most blocks (about 59 a step), was repeated: over the
two rounds its fenced median moved +1.0% and pipelined -0.4%, within noise. Its fenced mean
moved +2.7%, which may be a real cost of about 1 ms a step at that retire rate.

## 2026-11-15 OUT OF SCOPE (noted)

Low-face rows have the same hole: a retire hands no face row to its neighbour. No reader
outside the support reads those rows, so no audit or gate can see it today. A face handoff
mirroring the demote's high-plane faces would close it.
