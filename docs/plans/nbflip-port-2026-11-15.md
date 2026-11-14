# Port the seep rewrite's speed wins into Narrow-band FLIP (2026-11-15)

## 0. Status board

| item | state |
|---|---|
| Phase 0: close out the seep rewrite (S2q gate round, technique catalogue) | done: S2q 5.06x fenced / 5.28x pipelined, all gates (9fef519b); catalogue `nbflip-port/seep-techniques.md` (dd896664); seep PRESSURE_CAP failure bit (24b48036) |
| Phase 1A: NB-FLIP cost attribution across scenes, ranked port list (lane A) | in progress (2026-11-15) |
| Phase 1G: cross-scene gates and NB-FLIP references, dam break added (lane G) | done (2026-11-15): §4 gates, 5 refs a scene at 33b78ebb, self-test passed; dam residency defect found (§4.0) |
| Lane D: baseline dam residency-audit failure (6 of 11 fenced 300-step runs stop at steps 178–247); dam gates stop at step 165 until it is fixed | in progress |
| Phase 2: port lanes P1 (pressure) and P2 (submission, dispatch, kernels) | todo |
| Combined cross-scene round | todo |

## 1. Goal and contract

The user chose to port the seep rewrite's improvements into the general method
instead of writing another single-scene solver. The high-resolution dam break joins
as the sloshy acceptance scene.

Success means **NB-FLIP gets substantially faster on every gate scene while its
physics stays the same within measured tolerance**. In detail:
- **Gate scenes:**
  - `reservoir-seep` (1200 steps);
  - `high-resolution-dam-break` (128³, 4096-brick ceiling, 4 particles per fine cell;
    the step count is set by lane G from the scene's 20 s duration and GPU budget);
  - letters (`nbflip-figure-8-letters`, 240 steps; see `tools/benchmark-nbflip-letters.ts`);
  - paddleboats (`docs/plans/paddleboats-60fps-2026-11-10.md`), timing and drift.
- **Physics:** for each scene, the observables of NB-FLIP after the port match NB-FLIP
  before it, within the before/before repeat spread S. NB-FLIP is nondeterministic,
  so every scene's reference runs twice. This is the same method as the seep
  rewrite's §4.
- **Speed:** step time, fenced and pipelined, measured on a quiet GPU by lane G's
  timing round. It is reported per scene against frozen baselines. No single-scene
  win counts: a change that helps one scene and hurts another is reverted, not
  compensated (memory: cross-scene gates, improvements generalize).
- **Unchanged rules:**
  - memory and work stay proportional to the fluid (no domain-sized allocation or
    dispatch);
  - no substeps (dt 1/60);
  - all work on the GPU, with no per-step CPU stages or readbacks;
  - no hacks or clamps;
  - no runtime pool growth;
  - every numerical change cites a paper.
- **Per optimisation:** a fixed-state, one-step equivalence check (bitwise, or a
  stated ulp bound) before any trajectory comparison, wherever the change is meant
  to be exact. Pressure-solver changes that alter the iterate instead pass the same
  residual tolerance, and then the trajectory gates.

## 2. What transfers from the seep rewrite (catalogue)

These are measured in `docs/plans/seep-rewrite/progress-lane-s.md`, S2c–S2q. Lane S
writes the definitive per-technique catalogue as Phase 0. The likely transfers:

1. **Dispatch economics under Dawn.**
   - Every encoded dispatch costs about 3.5 µs.
   - Dawn validates every indirect dispatch with an injected one-workgroup dispatch,
     so an empty indirect slot costs about 50 µs per PCG cycle slot. Command
     translation at submit is CPU time that fenced mode waits on.
   - Levers:
     - fewer dispatches;
     - a convergence check placed before the V-cycle, so the converged iteration
       skips its down sweep;
     - fewer cycle slots;
     - a two-part submit, so the GPU starts while Dawn translates the rest.
   - NB-FLIP issues about 300 dispatches a step, so this applies at full scale.
2. **Pressure preconditioner.**
   - A staggered block Schwarz L0 smoother: the brick partition, then an xz shift by
     half a brick, with exact reverse post-order (Smith, Bjørstad & Gropp 1996). It
     took mean cycles from 5.17 to 3.32 and the maximum from 12 to 7.
   - An exact coarsest solve (Gauss-Jordan once per step) instead of 41 sweeps.
   - A Jacobi warm start (Saad 2003).
   - Block-Jacobi on a coarse level, with restriction fused into pre-smoothing.
   - Symmetry is proven by a CPU Float64 unit test (`seep-smoother.test.ts`).
   - Additive/BPX cycles were measured and rejected: 30–40 iterations against 7–10.
3. **Kernel shape, guided by occupancy and register pressure.**
   - Split register-heavy traces out of merged kernels, compact the work so full SIMD
     groups trace, and use 64-cell slabs per brick instead of 512-thread workgroups on
     lightly loaded bricks.
   - Spread one-thread-per-item loops (for example, interface faces) over 16 lanes.
   - Pack 8 sparse bricks per 64-lane workgroup.
4. **Level-set maintenance.**
   - Skip tiles that are a fixed point: all air, all −PHI_MAX, or no unknown face.
   - Use packed sign-code tiles for cut/frozen tests.
   - Compact true cut cells before the tetrahedral fill.
5. **Particle transfer.**
   - Transposed lanes in P2G, to avoid same-word shared atomics.
   - Flag stores instead of contended atomicAdds.
   - Skip redundant atomicOr.
6. **Physics detail.** Emit at an outlet with the face's own normal velocity and the
   donor cell's tangential velocity. This is relevant wherever NB-FLIP seeds
   particles from grid velocity next to closed faces.

**Seep-only items that do not transfer:**
- dense brick-major storage of the whole box (about 1.05 GB), which violates memory
  scaling;
- baked BULK rows and the throat block;
- particles kept to one chamber;
- the scene-fixed 32×32 coarsest level, which needs a general equivalent.

## 3. Phase 1 (parallel)

### Lane A: attribution and ranked port list
Profile current NB-FLIP on seep, high-resolution dam break, letters and paddleboats.
For each scene, report:
- per-kernel GPU time (timestamp profile) and dispatch counts (direct and indirect);
- empty or zero-workgroup indirect dispatches per step;
- CPU encode and submit time;
- the pressure cycle distribution and cap;
- xctrace occupancy and limiters for the top 10 kernels, using the
  `gpu-frame-profile` skill on a quiet GPU.

Map every cost centre against §2, then produce `docs/plans/nbflip-port/attribution.md`.
It ranks candidate ports by expected ms saved, per scene, with an implementation
sketch and risk for each, and names the NB-FLIP files each port touches. No changes
to `lib/`.

### Lane G: cross-scene gates
Generalise the seep harness into a per-scene gate runner for NB-FLIP before and after:
- **seep:** reuse G1–G8. The candidate is now NB-FLIP itself, so every gate is
  NB-new against NB-reference within the repeat spread.
- **high-resolution dam break:**
  - front position x_f(t) and residual column height h(t) (Martin & Moyce 1952
    form);
  - total volume drift;
  - kinetic and potential energy over time;
  - wet-floor cells;
  - a coarse surface-height profile at fixed times;
  - particle count;
  - step time.
- **letters** (240 steps): drift and step time per the cross-scene table, plus a
  shape observable.
- **paddleboats:** drift and step time.
- Two NB-FLIP reference runs per scene at a frozen baseline commit, giving S.
- Gates of the form |new − ref| ≤ max(rel%, 3·S), written down **before** any port
  lands.
- A timing round script with the quiet-GPU gate, generalised from lane H's
  `timing-round.sh`.

Outputs:
- `tools/nbflip-gates.ts` (or an extension of the seep tools);
- references under `artifacts/nbflip-port/ref/`;
- gate definitions in §4 of this document.

## 4. Gates
Written by lane G before Phase 2 lands anything. Once written, they are never
loosened after the fact. A measurement defect found in the reference, as with
seep §4.1, may be corrected only with evidence, and the original row stays printed.

Lane G wrote these on 2026-11-15, before any port landed. The referee is `tools/nbflip-gates.ts`.

### 4.0 Baseline, references and rule

**Baseline.** Commit 33b78ebb. Its NB-FLIP is identical to 46a2ba94: `lib/methods/uniform`
tree bef61da162 and `lib/core` tree e398dc5508. It runs from the read-only export
`~/code/me/fluid-scratch/lane-g/tree-33b78ebb`, which has a PROVENANCE file.

**References.** Five check-slot fenced runs per scene, all at the baseline, stored in
`artifacts/nbflip-port/ref/<scene>-ref-{a,b,c,d,e}.jsonl` (with `.log` and `-summary.json`).
- The dam has four references: a, b, d and e. Its run c failed twice; see "known baseline
  defect" below.
- Runs d and e carry the meta names `*-ref-d` and `*-base-e`.
- Every run returned rc 0, and the system log shows no Metal abort in its window.

| scene | run | samples | gated steps |
|---|---|---|---|
| seep | `tools/seep-bench.ts --method=nbflip --steps=1200 --every=60 --nb-stats` | every 60 | 1200 |
| dam | `tools/nbflip-gate-bench.ts --scene=dam`: 300 steps, every 15, dense 0–16 | every 15 | 165 (C0); 180–300 informational |
| letters | `tools/nbflip-gate-bench.ts --scene=letters`: 240 steps | every 30 | 240 |
| paddleboats | `tools/nbflip-gate-bench.ts --scene=paddleboats`: 300 steps, moored hulls, wheels at 1/3 throttle | every 30 | 300 |

**Rule.** A scalar row passes when |candidate − mean(refs)| ≤ max(floor, 3·S), where S is the
range of the references (max − min). A distance row (a profile) passes when
d(candidate, mean profile) ≤ max(floor, 3·S), where S is the largest pairwise d between
references. A scene passes when C0 and every gated row pass. Rows marked *info* print but do
not gate.

**Why five references, not two.** The brief's two runs were measured and found too few.
- With two references, S is a single difference. If an observable is pure noise, a third
  baseline run then falls outside 3·S in **18%** of rows (normal model, 200k trials). With n
  references the rates are: 3 → 3.8%, 4 → 1.0%, 5 → 0.3%.
- The data matched the model. Judged on refs a and b only, baseline run c failed paddleboats
  (P2x), and b judged against a failed the dam and letters.
- So S is the range of five runs. Rows that were droplet- or phase-chaotic were also
  rebuilt (below), not given wider floors.

**C0.** The candidate must reach GATED_STEPS (seep 1200, dam 165, letters 240, paddleboats 300)
with no solver failure record. Otherwise the scene fails and nothing else is judged.
`C0f` (all 300 dam steps) is informational until the defect below is fixed. After that,
GATED_STEPS.dam is raised to 300, which tightens the gate and never loosens it.

**Known baseline defect (dam).**
- At baseline physics, 6 of 11 fenced 300-step dam runs stopped with `GPU failure record:
  invalid support (layout build N: residency audit: liquid outside the resident pages, or a
  reader outside the closure)`. They stopped at steps 178, 178, 179, 182, 192 and 247.
- Those 11 runs are:
  - refs a, b, d and e and run f, all complete;
  - run c twice, which failed;
  - the two --set runs of §4.3 that the method ignored;
  - the two timing-slot fenced runs. These had an exclusive GPU, so contention is not the
    cause.
- Every failure had rc 1 and no Metal abort in its process tree.
- Both 300-step pipelined dam runs completed. A pipelined run may not read the failure record
  per step.
- NB-FLIP at 33b78ebb therefore does not reliably finish 300 dam steps, so the dam's gated
  rows end at 165. The dam is timed over 165 steps too.
- This needs an owner who may touch `lib/methods/uniform`. Lane G may not.
- Failed runs are kept as `artifacts/nbflip-port/selftest/dam-base-c-failed-179.*`,
  `dam-base-c2-failed-178.*` and `artifacts/nbflip-port/timing-base/{base,candidate}/dam-timing-fenced.*`.

**Dam step count: 300 steps (5 s), gated to 165.**
- The collapse takes about 11 steps: the front reaches x = 128 at step 11.
- Wall run-up and the ceiling hit follow by about step 30, then the return slosh by 45–60.
- A second wall impact comes by 105–120, then decay.
- At 165 the gated rows have seen the collapse, the impact and two sloshes.
- Steps 180–300 add the decay tail, informational until the defect is fixed.
- A run takes about 30 s at 40–60 ms a step.

### 4.1 Gate table

S is the five-reference value (four for the dam). Where several checkpoints exist, it is
the largest S over them, or a range of S. Each floor is set so that a change a port
should not make fails, while the baseline's own repeats pass.

**Seep.** G1–G8 of the seep rewrite (seep-rewrite-2026-11-14.md §4 and §4.1, including
§4.1's corrected G3 and G8 measurements). Here they are re-expressed as NB-new against
NB-reference.

| gate | observable, checkpoints | floor | S (5 refs) | floor justification |
|---|---|---|---|---|
| C0 | 1200 steps, no failure | required | – | – |
| G1 | q_left, mean over 60-step windows 120–1200 | 5% | 0.00025 L/s (0.09%) | seep §4 G1 |
| G1w | q_left, each of 18 windows | 10% | ≤ 0.0025 L/s | seep §4 G1 |
| G2 | left-tank loss at 300/600/900/1200 | 5% | ≤ 0.016 L | seep §4 G2 |
| G3 | right-tank volume at 300/600/900/1200 | 5% | 0.07–0.17 L | seep §4 G3; NB against NB, so the plain band |
| G4 | throat u, median over 120–1200 | 10% | 0.004 m/s | seep §4 G4 |
| G5 | step at which wet floor reaches 90% | 25% | 4 steps | seep §4 G5 |
| G5f | wet floor cells at 1200 | 5% | 72 cells | seep §4 G5 |
| G6c | right-chamber centroid x at 600/1200 | 16 cells | 2.7 / 6.6 cells | seep §4 G6 |
| G6 | right 16-bin profile, normalised L1, at 600/1200 | 0.15 | 0.14 / 0.10 | seep §4 G6 |
| G7 | total drift at 1200 | 0.001 (absolute fraction) | 0.0004 | NB's own drift is −0.51%; a port may move it by at most 0.1 point |
| G8 | jet impact x at 300/600 | 8 cells | 3.5 cells | seep §4.1 G8 |
| G8h | jet mean height per 8-cell slice, x 256–352, RMS, at 300/600 | 3 cells | 1.3 / 1.1 cells | seep §4.1 G8 |
| Gp *info* | particles at 300–1200 | 10% | ≤ 14k (≈4%) | cost proxy only |

**High-resolution dam break.** A 128³ box, 6.25 mm cells, and an 80 × 117.8 × 80-cell
corner column of 184.0 L. E0 is 669 J.

| gate | observable, checkpoints | floor | S (4 refs) | floor justification |
|---|---|---|---|---|
| C0 | 165 steps, no failure | required | – | see the defect above |
| D1 | front x_f, furthest wet floor x, steps 2–10 (Z = x_f / 80, Martin & Moyce) | 4 cells | 0–1 cell | 5% of the 80-cell dam width (Z ± 0.05); the baseline repeats to a cell |
| D2 | residual column height h at the back wall, steps 4–16 (eta = h / 117.8) | 4 cells | 0 | 3.4% of the column |
| D4c | volume drift at 16 | 0.01 | 0.008 | 1 point; S dominates (ref a reads −5.0% where the others read −4.2%) |
| D5c | KE, steps 4–16 | max(10%, 2 J) | ≤ 2.2 J | 2 J is 0.3% of E0 |
| D5ce | E = KE + PE, steps 4–16 | 2% | ≤ 4.2 J | collapse dissipation |
| D6 | wet floor cells, steps 4/8/12 | 5% | ≤ 9 cells | – |
| D7t | highest wet top per 8-cell x bin, RMS, steps 8/12/16 (the collapse wave's surface) | 4 cells | ≤ 1.7 cells | the wave front's height |
| D8c | particles at 16 | 10% | 9k (2.6%) | – |
| D3 | back-wall column mean depth, window means 30–90 and 105–165 | max(5%, 2 cells) | 1.0 / 1.4 cells | the slosh's return |
| D4 | volume drift, window means | 0.01 | 0.006 / 0.0014 | drift swings −8% to −1% through the splash |
| D5k | KE, window means | max(10%, 2 J) | 2.4 / 2.7 J | – |
| D5p | PE, window means | 2% | 2.3 / 4.2 J | – |
| D5e | E, window means | 2% | 2.7 / 6.8 J | energy lost at impact |
| D7d | mean depth per 8-cell x bin (the surface-height profile), window mean, normalised L1 | 0.05 | 0.006 / 0.008 | – |
| D8 | particles, window means | 10% | 42k / 30k | – |
| D3+ … D8+ *info* | the same rows over 180–300, on complete runs | – | – | – |

Why the dam rows are shaped this way:
- **Collapse rows are instantaneous.** The baseline repeats them to a cell.
- **Rows after the impact are window means of the 15-step samples.**
  - The baseline is not repeatable sample by sample after the impact. Energy lost at the
    impact differs by up to 3% run to run: at step 60, E is 323.8 J in one run and 331.6–333.8 J
    in the others. The runs also fall into two groups by sloshing phase: KE at 75 is 18 J
    against 23–25 J.
  - An instantaneous row at 60 failed a leave-one-out test. The windows answer the
    physical question (how much energy, depth and volume, and where) without gating the
    phase noise.
- **Rejected after measurement: "highest wet top" profiles after step 16.** They are a
  maximum over droplets. Pairwise RMS between baseline runs was 10–40 cells for the dam and
  letters, and up to 2 cells for paddleboats. They were replaced by mean-depth (volume over
  footprint) maps, which repeat to 0.5–1.3 cells (dam, letters) and 0.04 cells
  (paddleboats).

**Letters** (nbflip-figure-8-letters, 240 steps). Drop i lands at step 25i + 3; 1106 L are
dropped by 240.

| gate | observable, checkpoints | floor | S (5 refs) | floor justification |
|---|---|---|---|---|
| C0 | 240 steps, no failure | required | – | – |
| L1 | volume drift against the pool plus the drops, at 120/180/240 | 0.005 | ≤ 0.0018 | the baseline drifts −1.1%; a port may move it by 0.5 point |
| L2 | shape: mean depth per 16 × 16-cell (x, z) bin, RMS, at 60/90/120/180/240 | 1 cell | 0.07 → 1.34 cells | about 2.5% of the pool depth |
| L3 | KE at 60/120/180/240 | 10% | 14–197 J | – |
| L4 | liquid centroid height at 60/120/180/240 | 0.25 cells | ≤ 0.23 cells | – |
| L5 | particles at 60/120/180/240 | 10% | ≤ 22k (2.5%) | – |
| Ln *info* | NB-FLIP's own volume drift (readStats) | 0.005 | ≤ 0.0018 | – |

**Paddleboats** (moored hulls; wheels at 1/3 throttle; 300 steps).

| gate | observable, checkpoints | floor | S (5 refs) | floor justification |
|---|---|---|---|---|
| C0 | 300 steps, no failure | required | – | – |
| P1 | volume drift at 150/300 | 0.0025 | ≤ 0.0007 | the baseline drifts −0.97% by 300 |
| P2 | surface: mean depth per 32 × 32-cell (x, z) bin, RMS, at 150/300 | 0.25 cells | 0.02 / 0.04 cells | 1% of the 24-cell pool |
| P3 | KE (the wheels' wake) at 60–300 | 10% | 1.3–4.0 J | – |
| P4 | particles at 150/300 | 10% | 12k / 24k | – |
| Pn *info* | NB-FLIP's own volume drift | 0.0025 | ≤ 0.0007 | reads −1.5% where the observer reads −0.97% |

**Step time** prints beside every verdict (the timing-slot table, §4.4). It is not a pass or
fail row. Check-slot step times are informational because the slot is shared.

### 4.2 Running the gates

```
# one candidate tree (a commit, plus an optional patch), read-only, in the scratch area
tools/nbflip-export.sh <commit> ~/code/me/fluid-scratch/<lane>/tree-<commit> [patch]
# four check-slot fenced runs from that tree (each through tools/gpu/fluid-gpu-run.sh check):
#   seep: tools/seep-bench.ts --method=nbflip --steps=1200 --every=60 --mode=fenced --slot=check --nb-stats --out=OUT --name=seep-X
#   box:  tools/nbflip-gate-bench.ts --scene=dam|letters|paddleboats --mode=fenced --slot=check --out=OUT --name=<scene>-X
node --import tsx tools/nbflip-gates.ts --ref=artifacts/nbflip-port/ref --candidate=OUT/seep-X.jsonl,OUT/dam-X.jsonl,OUT/letters-X.jsonl,OUT/paddleboats-X.jsonl
# timing (exclusive timing slot, quiet GPU, baseline and candidate back to back):
tools/nbflip-timing-round.sh OUT ~/code/me/fluid-scratch/lane-g/tree-33b78ebb ~/code/me/fluid-scratch/<lane>/tree-<commit>
```

The referee exits 1 on any FAIL and writes NAME.md and NAME.json (default directory
`artifacts/nbflip-port/gates`). rc 76, 77 or 78 from the runner means the run is invalid:
rerun it, and never judge it.

### 4.3 Self-test (2026-11-15)

Reports are in `artifacts/nbflip-port/selftest/` and `~/code/me/fluid-scratch/lane-g/gates/`.
- **Leave-one-out.** Each of refs a–e was judged against the other four (three for the
  dam). All pass on every scene.
- **Sixth baseline run f** (`selftest/*-base-f`), judged against a–e: **PASS on all four
  scenes**. The worst margin was 0.47 (dam D5p); other scenes ≤ 0.25.
- **Perturbed tree, residual 0.5** (`UNIFORM_PRESSURE_ARENA_RETURNS_RESIDUAL` 0.05 → 0.5,
  `patches/residual-0.5.patch`): **FAIL**.
  - Seep fails G1 (margin 2.35), G1w (late windows, up to 3.5), G2 at 1200 (2.15) and G7
    (1.56).
  - Paddleboats fails P1 at 300 (drift −0.60% against −0.98%, margin 1.53).
  - The dam and letters pass: this constant barely touches them.
- **One-parameter perturbations.** These used `--set` on the baseline export, for the dam,
  letters and paddleboats.

  | perturbation | dam | letters | paddleboats |
  |---|---|---|---|
  | `volumeControlSeconds=1` (0.1) | **FAIL** (D4 drift −18% against −4%, margin 8.9; D5p 10.6; D5e 10.0; D3 2.4) | pass (L1 0.64) | pass |
  | `pressureResidualTolerance=50` (5) | pass (≤ 0.20) | pass (≤ 0.19) | pass (≤ 0.38) |
  | `particleCellCap=8` (20) | pass (≤ 0.44) | pass (≤ 0.33) | pass (≤ 0.45) |

  The last two are indistinguishable from baseline run f: the solver does not stop on the
  tolerance, and the cap seldom binds. They show what the gates cannot see, not a hole in
  them.
- **Ignored `--set` runs.** `pressureFullCycles=1` is a fixed parameter: NB-FLIP holds it at
  3. `phiCubicAdvection=off` is derived from `surfaceAdvection`. The bench silently ignored
  both, so those runs were at baseline physics: they passed and were discarded. The bench now
  refuses any `--set` that the method does not resolve to the value given.

**What the gates catch.** On every scene, the gates catch volume and conservation changes
that reach about 0.5–1 point of drift, and changes in seep flux. The dam also catches
energy and depth changes of about 2% or more. Pressure-tolerance and particle-cap changes
that leave the physics unchanged pass, as they should.

### 4.4 Baseline step time (timing slot, quiet GPU)

Timing round on 2026-11-15, baseline tree 33b78ebb against itself (an A/A round).
`tools/nbflip-timing-round.sh` ran it on the exclusive timing slot, each run after under 15%
GPU utilisation for 5 s. Every run returned rc 0.
- The four-scene round is in `artifacts/nbflip-port/timing-base/`.
  - Its dam fenced runs stopped at steps 182 and 247 (the defect above).
  - One GPU recovery hit another process at 12:00:00, during baseline letters fenced. It was
    not this round's process tree.
- The dam was rerun over 165 steps in `artifacts/nbflip-port/timing-base-dam/`, with no
  Metal abort.

| scene (steps) | fenced median ms, A / A | fenced p95 ms | fenced mean ms | pipelined ms/step | peak GPU MB |
|---|---|---|---|---|---|
| seep (1200) | 34.23 / 35.00 | 38.2 / 41.0 | 33.8 / 35.0 | 33.12 / 32.86 | 1187 |
| dam (165) | 45.12 / 46.25 | 72.8 / 74.5 | 51.1 / 51.5 | 50.29 / 49.18 | 1922 |
| letters (240) | 83.05 / 85.10 | 95.9 / 94.5 | 71.1 / 71.4 | 68.42 / 69.26 | 2023 |
| paddleboats (300) | 49.26 / 50.22 | 55.6 / 53.1 | 47.5 / 47.2 | 43.66 / 44.46 | 3228 |

Further timing detail:
- **A/A spread.** It was about 2.5% (fenced median) and 2.2% (pipelined), so a timing claim
  below about 3% is noise for one round.
- **Dam thirds.** Fenced thirds ran 65.3, 44.7 and 43.4 ms. The splash peak (about 1M
  particles at step 45) dominates the first third.
- **Dam over 300 steps.** In the first round, pipelined dam ran 45.4 / 44.7 ms/step.

## 5. Phase 2: port lanes
Two implementation lanes take items from lane A's ranked list:
- **P1, pressure:** the smoother, coarsest solve, check placement and cycle slots,
  in `uniform-pressure-arena.ts` and its relatives. A CPU Float64 symmetry test is
  required.
- **P2, submission, dispatch count and kernel shape:** two-part submit, fewer
  dispatches, register splits, tile skips and particle atomics.

Each landing needs:
- the equivalence check;
- the lane G gates on every scene;
- an interleaved A/B on every scene;
- a commit through the private index guard.

A coordinator-approved round by lane G times the combined result on a quiet GPU.

## 6. Process
- Agents implement; the coordinator plans and judges.
- GPU work only through `tools/gpu/fluid-gpu-run.sh`; rc 76, 77 or 78 means the run is
  invalid.
- Run `npm run check:types` (only the 3 pre-existing lane-n errors) and
  `npm run test:unit` before every commit.
- Use a private index for every commit, then `git reset -q`.
- Each lane writes `docs/plans/nbflip-port/progress-lane-<x>.md`.
