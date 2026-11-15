# Port the seep rewrite's speed wins into Narrow-band FLIP (2026-11-15)

## 0. Status board

| item | state |
|---|---|
| Phase 0: close out the seep rewrite (S2q gate round, technique catalogue) | done: S2q 5.06x fenced / 5.28x pipelined, all gates (9fef519b); catalogue `nbflip-port/seep-techniques.md` (dd896664); seep PRESSURE_CAP failure bit (24b48036) |
| Phase 1A: NB-FLIP cost attribution across scenes, ranked port list (lane A) | done (6b06703f): `nbflip-port/attribution.md`; no hot spot (8 areas of 6–22%), occupancy 12–17%, ranked items 1–12 total about 10–20% a step |
| Phase 1G: cross-scene gates and NB-FLIP references, dam break added (lane G) | done (2026-11-15): §4 gates, 5 refs a scene at 33b78ebb, self-test passed; dam residency defect found (§4.0) |
| Lane G: §7.1 drift amendment; references re-taken at lane D's fix | done (2026-11-15): §4.5 one-sided drift and restored-volume bands, pre-registered; §4.6 default references `ref-2761350a` (5 a scene, dam to 300), new S, self-test passed, timing baselines at 2761350a |
| Lane D: baseline dam residency-audit failure (6 of 11 fenced 300-step runs stop at steps 178–247); dam gates stop at step 165 until it is fixed | done (2026-11-15): root cause a retire with no vertex handoff (stale skirt phi); retire handoff in uniform-page-initialise.ts, 15 of 15 dam runs complete, GATED_STEPS.dam 300; `nbflip-port/progress-lane-d.md` |
| Phase 2: port lanes P1 (pressure: items 13, 3, 10, 8, 5) and P2 (move, submit, kernel shape: items 1, 2, 4+, 6, 7, 11, 9, 12) | **redirected** (2026-11-15): the user asked for NB-FLIP to be restructured toward seep-m1max's architecture, not tweaked item by item. P2's tweak list is stopped; P1 continues on a seep-structured multigrid. |
| Lane X: structural redesign of NB-FLIP (`docs/plans/nbflip-structural-2026-11-15.md`) | done (2f29b9d1), approved (§7.1): a new brick engine in `lib/methods/narrow-band/` behind the NB-FLIP id; milestones M0–M6; target about 2.5–2.9x on all four scenes |
| Structural lanes E (engine core), F (fluid kernels), P (pressure and bodies, continuing P1) | M0 done: ABI frozen (7bc7514e), pools from measured peaks (142ee18e), p1b explained as a wrong-scene run (7371f768, a22d512d), P1 item 13 (5532136b). M1 skeleton done (ac2fce26; 12/12 tests, hydrostatic within 1.9e-5 of Float64). M2 (the dam) in progress. |
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
Since lane D's fix the default set is §4.6's, at 2761350a; this one is kept.
- The dam has four references: a, b, d and e. Its run c failed twice; see "known baseline
  defect" below.
- Runs d and e carry the meta names `*-ref-d` and `*-base-e`.
- Every run returned rc 0, and the system log shows no Metal abort in its window.

| scene | run | samples | gated steps |
|---|---|---|---|
| seep | `tools/seep-bench.ts --method=nbflip --steps=1200 --every=60 --nb-stats` | every 60 | 1200 |
| dam | `tools/nbflip-gate-bench.ts --scene=dam`: 300 steps, every 15, dense 0–16 | every 15 | 300 (C0, since lane D's fix; 165 before); 180–300 rows informational |
| letters | `tools/nbflip-gate-bench.ts --scene=letters`: 240 steps | every 30 | 240 |
| paddleboats | `tools/nbflip-gate-bench.ts --scene=paddleboats`: 300 steps, moored hulls, wheels at 1/3 throttle | every 30 | 300 |

**Rule.** A scalar row passes when |candidate − mean(refs)| ≤ max(floor, 3·S), where S is the
range of the references (max − min). A distance row (a profile) passes when
d(candidate, mean profile) ≤ max(floor, 3·S), where S is the largest pairwise d between
references. A scene passes when C0 and every gated row pass. Rows marked *info* print but do
not gate. Drift rows and volume-coupled rows were amended before any landing (§4.5): drift is
one-sided, and volume-coupled rows also accept the references with their own loss restored.

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
`C0f` (all 300 dam steps) was informational until the defect below was fixed. Lane D fixed it
(the retire handoff, `nbflip-port/progress-lane-d.md`) and GATED_STEPS.dam is now 300, which
tightens the gate. Timing runs still cover the dam's first 165 steps (TIMED_STEPS), so a
baseline from before the fix stays comparable.

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

### 4.5 Amendment (2026-11-15, pre-registered per structural §7.1, before any port or engine landing)

The coordinator ruled this in `nbflip-structural-2026-11-15.md` §7.1(1) before any code of the
rebuild landed. Lane G implemented it in `tools/nbflip-gate-rules.ts` (the rules, tested in
`tests/nbflip-gate-rules.test.ts`) and `tools/nbflip-gates.ts` (the rows). It is not a loosening
after a landing: nothing has been judged under the old rule since §4 was written.

**Why.** NB-FLIP loses volume: the dam about 4–5% through the collapse and impact, letters about
1.1% by 240, paddleboats about 1.0% by 300, and the seep about 0.5% by 1200. That loss is a
defect of the reference, not physics a candidate must copy. This is the seep rewrite's §4
principle, applied there to G6 as NB_restored (70351e00). Under the two-sided §4 rule, a
candidate that conserved volume better would fail every drift row. It would also fail every
row whose value follows the volume: depths, column height, PE and the right-chamber profile.

**(1) Drift rows are one-sided on |drift|.** They are G7, L1, P1, D4c and D4 (window means).
A row passes when |c| ≤ |mean(refs)| + max(floor, 3S), with the floor and S unchanged. The
margin is the excess over |mean(refs)| divided by the tolerance. A candidate that drifts less,
or the other way by no more than the reference does, passes.

**(2) Volume-coupled rows pass between two bands.** These are the rows whose value moves with
the liquid volume:
- seep: G3, G6c and G6;
- dam: D2, D3, D5p, D5e, D5ce, D7t and D7d;
- letters: L2 and L4;
- paddleboats: P2.

KE, front, wet floor and particle rows are not volume-coupled to first order, so they stay as
written.

Each reference is restored with its own measured loss (below). This gives a restored mean, and
a restored S as the spread of the restored references. Then:
- the reference band is mean ± max(floor, 3S);
- the restored band is restored mean ± max(floor, 3 S_restored);
- the candidate passes anywhere on the segment ref + α (restored − ref), α in [0, 1], within the
  tolerance interpolated linearly between the two ends;
- profiles use their distance to the interpolated mean profile.

The rule is evaluated at 201 values of α. The candidate is never restored.

*Note for the coordinator:* this is the hull of the two bands. It is a superset of the ruled
"either band" (the union): it also accepts a candidate between the two bands when they do not
overlap, for example one that loses half NB-FLIP's volume. That follows seep §4.1's G6 centroid
interval between NB and NB_restored. The report marks any such row "between the bands, outside
both", and the JSON carries `unionMargin`. No self-test row (§4.5 and §4.6 below) needed the
hull: every band row that passes also passes the strict union. If the union is preferred, the
change is one line in `segmentMargin`.

**Restoration, per scene** (`restoreBox` and `restoreLostVolume`, both unit-tested against the
observer's Float64 oracle):
- **Box scenes (dam, letters, paddleboats).**
  - lost = max(0, expected_L − total_L) at that sample. A run that gained volume is not
    restored.
  - The wet footprint is the (x, z) bins of mean depth d_b ≥ 1 cell: 8-cell bins (dam),
    16-cell (letters) and 32-cell (paddleboats).
  - The loss goes back as one uniform depth δ = lost / (cell litres × wet footprint area).
  - Each wet bin's litres grow by δ·A_b, and its mean depth and highest wet top by δ. The x
    profiles (mean depth and top) are recomputed from the restored bins.
  - The dam's back-wall column gains δ in depth and top when its own depth is ≥ 1 cell.
  - The layer lies on each wet bin's mean surface, so PE (and E) gains ρ g h⁴ Σ_wet A_b δ
    (d_b + δ/2). The centroid height takes the same moment. KE is unchanged: the restored liquid
    is at rest.
  - Typical sizes at 2761350a:
    - dam: 7.8–9.0 L (δ ≈ 2.0–2.3 cells over about 253 bins) at 16; 8.6–10.1 L at 60;
      0.5–2.5 L at 135; 0–2.5 L at 240;
    - letters: about 107 L at 120 and 140 L at 240 (δ 0.34 and 0.44 cells; ΔPE ≈ 870 and
      1180 J);
    - paddleboats: about 113 L at 150 and 390 L at 300 (δ 0.07 and 0.25 cells).
  - The footprint's d_b counts liquid from the floor up. In bins that hold solids (paddleboat
    hulls, wheels) the layer is placed a little low. PE there is understated by about δ × the
    solid fraction: negligible against the tolerances.
- **Seep.** This is seep §4.1's NB_restored exactly (`restoreLostVolume`, the code
  tools/seep-compare.ts uses).
  - lost = max(0, left loss − open-wall change − right volume).
  - It is added uniformly over the right chamber's 16 bins, so the restored right volume is
    right + lost. The centroid is volume-weighted toward the chamber centre, wall.high + 128
    cells.
  - At 2761350a this is about 0.54 L at 600 (right 2.1 L) and 1.85 L at 1200 (right 3.5 L).

**(3) The §4 originals stay printed.** Each amended row is followed by "ID (as written)", the
two-sided row of §4.1, marked not gated. The dam's 180–300 window rows are now gated, because
GATED_STEPS.dam is 300 and all 20 references reach it. That is a tightening.

**Self-test under the amendment, against the 33b78ebb references** (`artifacts/nbflip-port/ref`;
reports in `artifacts/nbflip-port/selftest/gates-amended/`):
- Leave-one-out a–e: pass on every scene. The worst gated margin is 0.81 (seep G3, ref a).
- Run f: PASS on all four scenes (worst 0.25).
- `tol50` and `cellcap8` pass, as before.
- **residual 0.5** still FAILs the seep, on G1 (2.35), G1w (3.54) and G2 (2.15).
  - G7 no longer fails: the candidate's drift is −0.31% against the reference's −0.51%. That is
    a smaller loss, which the one-sided rule accepts by design. The original row prints
    "(as written) fail 1.56".
  - **Paddleboats now passes.** Its only failure was P1: drift −0.60% against −0.98%, again a
    smaller loss. The as-written row still prints its 1.53.
  - The perturbation is still caught, on the seep's flux rows, which is where it acts.
- **volctl1** still FAILs the dam: D4 8.85, D5p 10.55, D5e 7.41 and D3 2.36. Its drift is −18%
  against −4%, more loss, and it falls outside both bands. Letters and paddleboats pass, as
  before.
- No passing band row relied on the hull.

### 4.6 References at 2761350a (default since 2026-11-15)

Lane D's retire handoff (2761350a) changes iterates on the seep, letters and the dam, so the
references were re-taken there. 2761350a is exported read-only at
`~/code/me/fluid-scratch/lane-g/tree-2761350a`, with `lib/methods/uniform` tree 1abb07aafa and
`lib/core` tree a7863039f6.
- Five check-slot fenced runs a scene are in `artifacts/nbflip-port/ref-2761350a/`
  (`<scene>-ref-{a..e}`). The dam has all five, each to 300 steps.
- All 20 runs completed with no failure record.
- Three runs returned rc 77 (runner guard) on the first try and were rerun: letters a and d, and
  seep c. Each rerun returned rc 0.
- The system log has one GPU hang in the window, at 13:42:00, caught by the guard on seep c's
  first try. seep-base-f's runner logged it as another process's (its tree clean, rc 0).
- **The old set stays** in `artifacts/nbflip-port/ref` and is used with `--ref=artifacts/nbflip-port/ref`.
- The referee defaults to the new set.
- Every floor is as written in §4.1. Only S changes.

**S at 2761350a** (five refs; the largest over checkpoints, or a range), against 33b78ebb's:

| row | S, 33b78ebb | S, 2761350a | binding term at 2761350a |
|---|---|---|---|
| G1 / G1w | 0.00025 / ≤ 0.0025 L/s | 0.0020 / ≤ 0.0044 L/s | floor |
| G2 | ≤ 0.016 L | ≤ 0.036 L | floor |
| G3 | 0.07–0.17 L | 0.03–0.20 L (restored 0.005–0.036) | 3S (as before) |
| G5 / G5f | 4 steps / 72 cells | 18 steps / 12 cells | floor |
| G6c | 2.7 / 6.6 cells | 4.0 / 4.1 cells | floor (3S bound at 1200 before) |
| G6 | 0.14 / 0.10 | 0.08 / 0.16 | 3S (as before) |
| G7 | 0.0004 | 0.00065 | 3S (as before): tol 0.0020 |
| G8 / G8h | 3.5 / 1.3 cells | 4.3 / 1.5 cells | 3S (as before) |
| D4c | 0.008 | 0.007 | 3S (as before) |
| D2 / D5ce | 0 / ≤ 4.2 J | ≤ 0.06 cells / ≤ 0.9 J | floor (3S bound D5ce at 16 before) |
| D7t | ≤ 1.7 cells | ≤ 2.3 cells | 3S at 16 (as before) |
| D3 | 1.0 / 1.4 cells | 0.95 / 0.73 / 0.52 cells | 3S at 30–90 only (before: every window) |
| **D4** | 0.006 / 0.0014 | **0.009 / 0.008 / 0.015** | **3S in every window (new at 105–165 and 180–300): tol 0.027 / 0.025 / 0.045** |
| D5k | 2.4 / 2.7 J | 1.2 / 2.5 / 4.3 J | 3S at 105–300 |
| D5p | 2.3 / 4.2 J | 4.5 / 7.6 / 10.4 J (restored 1.2 / 4.6 / 5.3) | 3S (as before), tol up to 31 J |
| D5e | 2.7 / 6.8 J | 3.9 / 11.9 / 14.4 J (restored 2.1 / 8.9 / 9.7) | 3S (as before), tol up to 43 J |
| D7d | 0.006 / 0.008 | ≤ 0.011 | floor |
| D8 | 42k / 30k | 17k / 30k / 30k | 3S at 105–300 |
| L1 | ≤ 0.0018 | ≤ 0.0014 | floor (3S bound at 120 before) |
| L2 | 0.07 → 1.34 cells | 0.04 → 1.14 cells | 3S at 120–240 (as before) |
| L3 / L4 | 14–197 J / ≤ 0.23 cells | 2–219 J / ≤ 0.14 cells | 3S at 120–180 / 120 |
| **P1** | ≤ 0.0007 | **0.0003 / 0.0018** | **3S at 300 (new): tol 0.0053** |
| P2 / P3 | 0.02 / 0.04 cells, 1.3–4.0 J | 0.02 / 0.06 cells, 1.2–2.4 J | floor |
| P4 | 12k / 24k | 26k / 80k | 3S (as before) |

Where the new S makes 3S binding where the floor bound before:
- **D4 at 105–165 and 180–300** (and the wider D5p and D5e). Dam ref c conserves far better
  late than the others: its drift at 135 / 240 / 300 is −0.3% / +0.5% / +0.8%, against −1.2 to
  −1.4% / −0.7 to −1.4% / −0.8 to −1.0% for a, b, d and e.
  - At 33b78ebb all four dam refs agreed late. At 2761350a the late dam drift is bimodal run to
    run.
  - The 180–300 D4 tolerance is therefore 4.5 points, and a candidate may lose up to 5.2% there.
    D5p's and D5e's tolerances grow with it, since PE follows the volume.
  - The restored bands are tighter (restored S of D5p 5.3 J against 10.4) because restoration
    removes the volume-coupled part of the spread.
  - volctl1 (−18%) still fails it.
  - With ref c left out, the as-written D4, D5p and D5e rows fail ref c itself (LOO-c below). So
    the spread is real baseline variation, not one bad run.
- **P1 at 300.** Paddleboats ref e drifts −0.90% against −1.00 to −1.07%, so S is 0.0018 and the
  tolerance is 0.53 points against the floor's 0.25.

**Self-test against the 2761350a set** (`artifacts/nbflip-port/selftest-2761350a/`, reports in
`selftest-2761350a/gates/`):
- **Leave-one-out a–e: pass on every scene.** The worst gated margin is 0.68 (dam D5k, LOO-c).
  - LOO-c's dam passes only under the amendment. Its as-written D4 (1.36), D5p (1.18) and D5e
    (1.07) fail, because ref c lost less late. The one-sided and band rows accept it, as §4.5
    intends.
- **Run f** (`*-base-f`, a sixth run at 2761350a): **PASS on all four scenes**. The worst margin
  is 0.43 (dam D8); the seep's is 0.42 (G1w).
- **residual 0.5** (tree-2761350a with `patches/residual-0.5-2761350a.patch`; the old patch's
  context moved): **FAIL**.
  - The seep fails G1 (2.12), G1w (3.03) and G2 (1.92), as at 33b78ebb.
  - G7 and P1 pass. Each is a smaller loss: seep −0.34% against −0.48%; paddleboats −0.58%
    against −1.02% at 300. They pass even as written now, at margins 0.71 and 0.83, because the
    new S is larger.
  - The dam passes. Its D3 (as written) prints 1.01, with the column 2.4 cells deeper at
    105–165, which is inside the restored band.
  - Letters passes.
- **volctl1** (`volumeControlSeconds=1`): **FAIL** on the dam: D4 5.20, D5p 5.46, D5e 6.91,
  D3 2.67 and D5k 1.62. Letters and paddleboats pass, as before.

### 4.7 Step-time baselines at 2761350a

These rounds ran on the timing slot, with baseline 33b78ebb and candidate 2761350a back to back
per scene (`tools/nbflip-timing-round.sh`). Every run returned rc 0.
- **The four-scene round** is in `artifacts/nbflip-port/timing-2761350a/`. Its seep pair is
  void:
  - another agent's unit-test run loaded the CPU from 13:51 to 13:53, during the baseline fenced
    run;
  - another process's Metal aborts came at 13:53:40, as the candidate fenced run started;
  - the result was a 38.35 ms baseline median, a 32 s construction and a 48.8 ms candidate
    pipelined.
- **The seep was re-run alone** in `timing-2761350a-seep2/`, clean, with no Metal abort.
- **Letters was repeated** in `timing-2761350a-letters2/`, with no Metal abort.
- Other lanes' check jobs queued on the lock between runs. None ran during a timed run.

**Baselines at 2761350a** (the candidate column; 33b78ebb in brackets, from the same round):

| scene (steps) | fenced median ms | fenced p95 ms | fenced mean ms | pipelined ms/step | peak GPU MB |
|---|---|---|---|---|---|
| seep (1200) | 35.09 (35.19) | 39.0 (39.9) | 34.45 (34.70) | 32.30 (32.42) | 1187 |
| dam (165) | 45.45 (44.50) | 76.9 (67.8) | 51.49 (49.20) | 50.61 (49.66) | 1922 |
| letters (240), round 1 / round 2 | 84.81 / 83.44 (86.50 / 88.60) | 96.7 / 94.1 | 71.81 / 70.70 (72.93 / 74.25) | 69.57 / 69.18 (67.82 / 71.18) | 2023 |
| paddleboats (300) | 50.50 (50.37) | 55.5 (53.6) | 48.78 (48.15) | 45.22 (44.15) | 3228 |

What the rounds show:
- **Lane D's letters +1 ms (fenced mean) is noise.** The two rounds give −1.1 and −3.6 ms against
  33b78ebb. Across four runs, 33b78ebb's own fenced mean spans 71.1–74.3 ms (§4.4's A/A pair
  and these two rounds), a 4.5% range. 2761350a's two runs (71.8 and 70.7) sit inside it.
  Pipelined letters is 69.6 / 69.2 against 67.8 / 71.2: no change.
- **The other scenes match within the §4.4 A/A spread** (about 2.5%):
  - seep −0.3% fenced, −0.4% pipelined;
  - dam +2.1% fenced median, +1.9% pipelined;
  - paddleboats +0.2% fenced median, +2.4% pipelined.
- **Dam p95 and its first third** (76.9 against 67.8 ms; 67.1 against 61.8 ms) are the splash
  peak. 33b78ebb's p95 has read 67.8–74.5 ms over three runs, so 76.9 is 2.4 ms above its
  highest. That is a single run, and the median and pipelined figures do not move.
- **Lane D's handoff costs no measurable step time.** These numbers are the step-time baseline
  for the rebuild's gates from now on.

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
