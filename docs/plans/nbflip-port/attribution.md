# NB-FLIP cost attribution and ranked port list (lane A, 2026-11-15)

Brief: `docs/plans/nbflip-port-2026-11-15.md` §3, lane A. Measured at b2316869 (NB-FLIP is unchanged since
the plan). Progress log: `progress-lane-a.md`. Raw data is in `artifacts/nbflip-port/attribution/`, one JSON
per run. The split phase keeps the last step's ordered dispatch list, and `pressure/` holds the convergence
probes. Numbers in this document come from those files and the scripts in `~/code/me/fluid-scratch/lane-a/`
(`summary.mjs`, `analyze.mjs`, `cycle-stages.mjs`, `cputree.mjs`).

## 0. Headlines

1. **NB-FLIP has no seep-style hot spot.** Its step splits into eight areas of 6–22% each, on every scene (§2.1).
   The particle move is the one kernel worth a whole area, at 11–19% of the step in a single launch.
   - The seep rewrite's 5x came mostly from seep-only representation (catalogue §8).
   - The transferable items ranked here (1–12) add up to about 10–20% a step on NB-FLIP. Fenced, that is −5 to −7.5
     ms on seep 600 and the dam at 200, and −8 to −13 ms on letters. Pipelined is 1.3–1.8 ms less, since items 2
     and 12 are fenced-only.
2. **The Dawn economics differ from the seep's.**
   - NB-FLIP issues 445–477 dispatches a step, but only 1–2 of them are indirect, so indirect validation is worth
     25 µs.
   - Its cost is the direct early-exit launch (7–25 µs each, 147–163 launches under 20 µs a step).
   - Its 74–78 logical passes are already merged into shared native passes once the frame plan is sealed
     (`uniform-compute-blit.ts`).
   - Fenced waits 1.4–1.8 ms of encode before the GPU starts (3.1 ms on paddleboats, whose rigid-body records take
     1.5 ms of CPU first).
3. **Pressure is 12–18% of the step and latency-bound.** At 43 dispatches a cycle, fine-level smoothing is only
   14–20% of a cycle, and the one-workgroup 8h–32h tail is the largest stage on every scene.
   - **Seep and the dam break are cap-limited.** 26–70% of their steps end above the 0.05 target at the 4-cycle
     cap. That is an accuracy defect, not a speed one, and the slow mode sits on the band/native seam.
   - **tailCycles 3 → 1 is not free.** It is free on seep and paddleboats, but it costs 0.15–0.2 decades of final
     residual on the dam and letters.
4. **Top of the port list** (§6):
   - item 1: separation out of the move (P2, bitwise; −1.4 to −5.4 ms);
   - item 2: a two-part submit at the census (P2, bitwise, fenced; −1.0 to −1.3, up to −2.6 on paddleboats);
   - item 3: a one-cycle pressure tail with a stronger coarsest correction (P1; about −1.0, behind a go/no-go
     probe);
   - item 5: arena dispatch fusion (P1; −0.35 to −0.6);
   - item 6: level-set fixed-point tile skips (P2; −0.3 to −2.5, behind a skip-fraction probe).

   Item 4 (splitting the rest of the move) is an upper bound that may measure zero.

   P1 and P2 share no files (§7).
5. **Defect.** The high-resolution dam break fails a residency audit nondeterministically past step 150, in 2 of 5
   runs (§8). That blocks a full-length dam gate.
6. **Occupancy.** No kernel on any scene is near a hardware limit (xctrace, §5). Time-weighted occupancy is 12–17%,
   ALU 17–22%, and the highest limiter is 44%. NB-FLIP is latency-bound, so the port list favours fewer dependent
   launches and less serial work per thread over bandwidth savings.

## 1. Method

- **Tool.** `tools/nbflip-attribution.ts` (committed with this document).
  - It builds NB-FLIP the way `seep-bench` does: through the method registry, with the scene's methodProfile
    over the app defaults and dt equal to the method's effective step (1/60 on every scene).
  - It runs fenced to the sample step, then takes measurement windows of 10 steps each, in this order:
    1. **clean:** no instrumentation. Wall time, CPU time inside advanceTo, CPU time inside `queue.submit`, and
       the CPU time at which each submit returns.
    2. **counted:** one step with every dispatch recorded.
    3. **pass:** a timestamp pair on every compute pass.
    4. **split:** every dispatch in its own timestamped pass, with each indirect dispatch's 12 argument bytes
       copied out just before it runs.
    5. **pipelined:** 10 steps submitted the way the app submits them, then one fence.
    6. **pipepass:** the same as pipelined, with a timestamp pair on every pass and the passes grouped by step. It
       gives the GPU idle time between steps.
    7. **cpuprof:** the V8 sampling profiler at 100 µs, as a call tree under advanceTo.
  - Timestamp quantization is off (`disable-dawn-features=timestamp_quantization`). With it on, every reading
    was a multiple of 65.5 µs.
  - Since the late runs, each phase waits until device utilisation has stayed under 15% for 3 s (at most 300 s). It
    records the wait and the other harness processes alive before and after (`contention` in the JSON).
  - **Timestamps break pass sharing.** Once the frame plan is sealed, `UniformComputeBlit` merges consecutive
    logical passes into one native compute pass, unless a pass carries `timestampWrites`. So the pass, split and
    pipepass phases run every logical pass as its own Metal encoder, and their idle gaps are an upper bound for
    the clean step's.
- **Split overhead is small.** On seep 600 the arena solve's single pass measured 5.69 ms, against 5.99 ms as
  178 split passes (about 1.7 µs a dispatch). Per-dispatch split times are therefore real costs, including the
  10–40 µs floors in §3.
- **Phases see different states.** The windows are consecutive steps, so where the particle count moves fast
  (the dam break at 45, letters), fenced, split and pipelined measure different states. Every row below gives
  its own step range and particle counts. Fenced is compared against split for the same scene, not against
  pipelined.
- **Contention.** Lanes A and G share the three check slots. Runs that overlapped lane G's reference benches
  were thrown away (paddle-300, paddle-300b: fenced 99.7 ms against 49.9 pipelined), and §2 notes which runs
  are gated.

## 2. Per-scene summary

All times are ms a step, medians of a 10-step window. GPU busy is the sum of per-pass timestamps. "Gated" runs
waited for a quiet GPU before every phase (§1); the ungated seep 1100 and letters 220 rows came from rounds whose
other check slots were idle when they ran (`contention.log`).

| scene | steps | particles (split) | fenced | pipelined (30 steps later; §3 has matched steps) | GPU busy (pass) | split sum | dispatches (direct / indirect) | empty indirect | advanceTo CPU | in submit | first submit at | pressure cycles ran / encoded | capped above target | gated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| seep 600 | 600–640 | 136k–138k | 34.5 | 35.5 | 31.8 | 31.6 | 454 (453 / 1) | 0 | 3.15 | 0.80 | 2.38 | 3.56 / 4.00 | 40% | yes |
| seep 1100 | 1100–1140 | 219k–221k | 38.3 | 35.2 | 34.8 | 36.1 | 454 (453 / 1) | 0 | 3.41 | 0.86 | - | 3.58 / 4.00 | 26% | no |
| dam 45 | 45–85 | 443k–402k | 68.6 | 45.4 | 53.8 | 41.4 | 445 (444 / 1) | 1 | 3.16 | 0.76 | 2.52 | 3.95 / 4.00 | 52% | yes |
| dam 200 | 200–240 | 358k–346k | 37.3 | 36.3 | 33.7 | 33.9 | 445 (444 / 1) | 1 | 3.29 | 0.84 | 2.25 | 3.95 / 4.00 | 26% | yes |
| letters 120 | 120–160 | 881k–899k | 84.1 | 93.6 | 82.8 | 84.5 | 451 (450 / 1) | 1 | 3.19 | 0.81 | 2.19 | 3.05 / 3.54 | 12% | yes |
| letters 220 | 220–260 | 876k–846k | 93.1 | 80.1 | 85.7 | 80.5 | 451 (450 / 1) | 1 | 3.56 | 0.93 | 2.55 | 3.38 / 3.72 | 9% | no |
| paddleboats 300 | 300–340 | 272k–266k | 49.8 | 51.0 | 45.2 | 45.9 | 477 (475 / 2) | 1 | 4.78 | 0.83 | 4.23 | 2.95 / 4.00 | 1% | yes |

- **Reading the rows.** The four columns of time see different steps (§1).
  - **Dam break at 45.** The particle count falls from 891k to 402k across the window: the fenced phase averaged
    891k–741k, pass 711k–443k, split 443k–402k. Its fenced and split figures are not comparable with each other.
  - **Letters.** The count oscillates between 846k and 993k through seeding and retirement.
  - **Pressure columns.** "ran / encoded" is the mean over every step up to the sample point. "Capped above target"
    is the share of the last 200 steps that ran all 4 encoded cycles and ended above the 0.05 target.
  - **Paddleboats.** It ran with the moored motor bodies (`--bodies`, as `seep-bench`).
- **Pipelined against fenced.** In these runs the pipelined window came 30 steps after the fenced one, so the
  column sees a different state (on letters 120, 937k particles against 894k). At matched steps (the `*p` runs,
  §3), pipelined wall / 10 is 34.5 (seep 600), 39.0 (dam 200), 87.1 (letters 120) and 54.5 (paddleboats 300). That
  is at or under fenced's 10-step mean, and the GPU runs the steps back to back.

**Top 5 kernels** (split ms a step; the number of launches in brackets when more than one):

| scene | kernels |
|---|---|
| seep 600 | `advectParticlesDeferred` 3.70, `remapFaces` 0.97 (4), `advectMomentum` 0.95, `coupleFine` 0.87 (2), `classify` 0.83 |
| seep 1100 | `advectParticlesDeferred` 5.57, `pRelax` 0.99 (25), `advectMomentum` 0.97, `coupleFine` 0.93 (2), `arenaDistance` 0.85 |
| dam 45 | `advectParticlesDeferred` 7.93, `advectMomentum` 1.47, `arenaDistance` 1.30, `remapFaces` 1.27 (4), `transferSeams` 1.17 |
| dam 200 | `advectParticlesDeferred` 6.00, `arenaDistance` 1.03, `pRelax` 0.91 (24), `transferSeams` 0.86, `advectMomentum` 0.80 |
| letters 120 | `advectParticlesDeferred` 14.12, `classify` 3.46, `arenaDistance` 3.09, `advectMomentum` 3.01, `remapFaces` 2.13 (4) |
| letters 220 | `advectParticlesDeferred` 13.81, `arenaDistance` 2.88, `advectMomentum` 2.83, `classify` 2.50, `advectDeferred` 2.10 (5) |
| paddleboats 300 | `advectParticlesDeferred` 5.12, `arenaDistance` 1.23, `advectMomentum` 1.18, `classify` 1.00, `buildRegions` 1.00 |


### 2.1 Where the time goes (split phase, ms a step, share of the split sum)

Each cell is ms (share of the split sum, dispatches). The pass labels are grouped by `areas.mjs`.

| area | seep 600 | seep 1100 | dam 45 | dam 200 | letters 120 | letters 220 | paddleboats 300 |
|---|---|---|---|---|---|---|---|
| particle move | 3.76 (11%, 1) | 5.57 (15%, 1) | 7.93 (19%, 1) | 6.00 (18%, 1) | 14.12 (17%, 1) | 13.81 (17%, 1) | 5.12 (11%, 1) |
| pressure solve | 5.99 (18%, 178) | 6.28 (17%, 178) | 6.41 (15%, 177) | 6.08 (18%, 177) | 10.48 (12%, 177) | 9.85 (12%, 177) | 6.05 (13%, 178) |
| level set / surface | 6.29 (19%, 49) | 6.70 (19%, 49) | 8.02 (19%, 49) | 6.38 (19%, 49) | 18.08 (21%, 49) | 17.32 (22%, 49) | 9.10 (20%, 57) |
| grid velocity (momentum, extension, remap) | 4.97 (15%, 38) | 5.05 (14%, 38) | 6.69 (16%, 38) | 4.99 (15%, 38) | 13.93 (16%, 38) | 12.93 (16%, 38) | 9.13 (20%, 49) |
| particle transfer / seeding | 2.93 (9%, 44) | 3.06 (8%, 44) | 5.06 (12%, 38) | 3.91 (12%, 38) | 9.79 (12%, 44) | 9.42 (12%, 44) | 4.60 (10%, 44) |
| surface (phi) advection | 3.49 (10%, 21) | 3.47 (10%, 21) | 2.83 (7%, 21) | 2.07 (6%, 21) | 5.76 (7%, 21) | 5.66 (7%, 21) | 4.10 (9%, 21) |
| layout / census / residency | 3.60 (11%, 100) | 3.56 (10%, 100) | 4.37 (11%, 98) | 3.58 (11%, 98) | 8.06 (10%, 98) | 6.51 (8%, 98) | 6.41 (14%, 104) |
| pressure setup + projection | 2.18 (7%, 23) | 2.23 (6%, 23) | 2.24 (5%, 23) | 2.06 (6%, 23) | 4.93 (6%, 23) | 4.71 (6%, 23) | 3.25 (7%, 23) |
| split sum | 33.39 | 36.06 | 41.44 | 33.85 | 84.45 | 80.51 | 45.89 |

- The seep 600 column is the ungated `seep-600` run (step 620–630), whose last split step ran all four cycles. In
  the gated `seep-600c` it converged at 2, which makes the solve look 1.1 ms cheaper.
- **No area is over 22% anywhere.** NB-FLIP has no single seep-style hot spot, so the port has to be broad.
- **The move is the one kernel worth a whole area.** It is 11–19% of the step in a single launch.


## 3. Dispatch and submission economics in NB-FLIP

**Counts (counted step).**

| scene | dispatches | direct | indirect | empty indirect | kernels | passes | submits | command encoders |
|---|---|---|---|---|---|---|---|---|
| seep 600 / 1100 | 454 | 453 | 1 | 0 | 267 | 74 | 4 | 4 |
| dam break 45 / 200 | 445 | 444 | 1 | 1 | 263 | 74 | 4 | 4 |
| letters 120 / 220 | 451 | 450 | 1 | 1 | 265 | 74 | 4 | 4 |
| paddleboats 300 | 477 | 475 | 2 | 1 | 281 | 78 | 4 | 4 |

**What this rules out.**
- NB-FLIP is already almost wholly direct. The only indirect launches are the bulk tier selection's
  (`Uniform bulk tier selection`, one or two a step, one of them empty) and, on paddleboats, one in
  `NB-FLIP surface pressure authority`.
- So the seep's dominant Dawn lever, an injected validation dispatch ahead of every indirect slot (12.8 µs for an
  empty one), is worth at most about 25 µs a step here.
- The equivalent cost in NB-FLIP is the **direct launch that early-exits**. Its width comes from fixed-lag
  evidence, and its readers stride the GPU's counts, returning at once when converged or empty.

**What a dependent dispatch costs on the GPU.** This is the last split step's distribution, per dispatch:

| scene | n | < 10 µs | 10–20 µs | 20–50 µs | 50–100 µs | 0.1–0.5 ms | > 0.5 ms |
|---|---|---|---|---|---|---|---|
| seep 600 | 454 | 52 / 0.34 ms | 95 / 1.36 | 184 / 5.82 | 49 / 3.36 | 66 / 13.73 | 8 / 8.93 |
| dam 45 | 445 | 62 / 0.42 | 87 / 1.25 | 172 / 5.48 | 45 / 3.02 | 62 / 13.12 | 17 / 20.66 |
| letters 120 | 451 | 45 / 0.33 | 67 / 0.93 | 159 / 4.99 | 68 / 5.13 | 75 / 17.76 | 37 / 59.29 |

- **The floor is about 7–10 µs a dispatch**, not lane S's 3.5 µs. An early-exit launch of about 240 workgroups
  costs 7–25 µs on the GPU timeline. In paddleboats' converged fourth cycle (band 3 of 4 encoded), 35 dispatches
  that return at once took 0.45 ms, about 13 µs each.
- Lane S's 3.5–5.2 µs was for 1–64 workgroups with no workgroup memory. NB-FLIP's early-exit kernels launch
  hundreds of workgroups, with large workgroup arrays, behind a serialising barrier.
- **These floors are split-mode numbers.** In the clean step, sealed stages share one Metal encoder (§1), where
  `uniform-compute-blit.ts` cites about 5 µs a dependent launch. The clean wall bounds the difference: fenced equals
  the first submit, plus the split-mode GPU time, plus 1.2–2.0 ms (below). So the split floors overstate the clean
  step by at most that much over all 450 launches. Items that save launches (§6, items 5, 8, 9) are given at the
  split floors, and could be up to half that in the clean step.
- **Seep is the dispatch-bound scene.** 331 of its 454 dispatches run under 50 µs, which is 7.5 ms of 33.4 (22%).
  On letters the 37 dispatches over 0.5 ms are 70% of the step, so letters is kernel-bound.

**Submission** (CPU ms a step; medians of the 10 clean steps of the gated runs; the counted step's first submit,
whose dispatch-recording proxy slows encoding, is given for comparison):

| scene | advanceTo CPU | in queue.submit | first submit returns at (clean / counted) | last submit returns at |
|---|---|---|---|---|
| seep 600 | 3.15 | 0.80 | 1.51 / 2.38 | 3.20 |
| dam 45 | 3.16 | 0.76 | 1.54 / 2.52 | 3.17 |
| dam 200 | 3.29 | 0.84 | 1.59 / 2.25 | 3.29 |
| letters 120 | 3.20 | 0.81 | 1.56 / 2.19 | 3.20 |
| paddleboats 300 | 4.78 | 0.83 | 3.14 / 4.23 | 4.78 |

- **The first submit carries most of the first stage.** The step goes out as 4 command buffers: after the move,
  after the distance, after the advection correction, and the rest (`uniform-mixed-frame.ts` flush calls).
- The first buffer holds 135–137 dispatches (census, layout, remap, frame plan, momentum taps, move). That is
  9.9 ms of GPU work on seep, 15.4 on the dam and 29.7 on letters.
- In fenced mode the GPU waits idle for the first 1.4–1.8 ms of CPU (2.8–3.5 ms on paddleboats). By then the host
  has encoded the census and everything after it up to the move.
- The census prefix (the first 18 dispatches) is 2.0 ms of GPU on seep, 2.1 on the dam and 4.5 on letters. That
  is enough to cover encoding the rest, which is the condition S2q's two-part submit needs.
- **Where advanceTo's CPU goes** (V8 profile, seep 60 and paddleboats 300; `cputree.mjs`):
  - It is spread over about 40 encode functions, roughly 3.5 µs a dispatch (lane S's figure), plus about 0.9 ms
    inside `queue.submit`.
  - Biggest items: `encodeRelayoutHead` 0.70–0.92 ms, `flush`/`submit` 0.69–0.80 ms, and `encodeSolve` 0.40–0.43
    ms (the arena's 178 dispatches).
  - Paddleboats adds `syncBodies` → `uploadBodies` 1.0 ms (of which `rigidMotorRecords` is 0.58 ms) and
    `syncMixedLayout` → `mixedBodyTiles` 0.48 ms. That 1.5 ms of CPU sits in front of the first submit and
    explains its later first submit (3.1 ms against 1.5).
- **What fenced waits on** (clean fenced wall against the pass-timestamp GPU busy time of the next window):

  | scene | fenced | first submit | GPU busy | remainder (gaps and fence) |
  |---|---|---|---|---|
  | seep 600 | 34.50 | 1.51 | 31.76 | 1.23 |
  | dam 200 | 37.26 | 1.59 | 33.70 | 1.97 |
  | paddleboats 300 | 49.85 | 3.14 | 45.18 | 1.53 |

  - The first-submit latency is the larger part. The remainder (1.2–2.0 ms) is GPU idle between passes inside the
    step, plus the fence's completion latency.
  - With a timestamp pair on every pass, the GPU span of a step exceeds its busy time by 2.5–3.7 ms. The
    timestamps unshare the passes (§1), which is why the clean remainder is smaller.
  - The submission levers below are fenced-only.
- **What pipelined waits on** (`pipepass`: 10 steps submitted as the app does, one fence, a timestamp pair on every
  pass; medians over steps 2–10 of the window. The `*p` runs take their pipelined window first, at the same steps as
  the `*c` runs' fenced window, then the pipepass window right after it):

  | scene | fenced median / mean (`*c`, same steps) | pipelined wall / 10 (`*p`) | pipepass steps | particles | GPU busy | GPU span | idle inside the step | gap between steps | period | first step's span |
  |---|---|---|---|---|---|---|---|---|---|---|
  | seep 600 | 34.50 / 37.21 | 34.46 | 610–619 | 135k–136k | 31.76 | 34.63 | 2.87 | 0.05 | 34.69 | 51.4 |
  | dam 200 | 37.26 / 40.37 | 38.96 | 210–219 | 350k–366k | 34.72 | 37.44 | 2.73 | 0.05 | 37.49 | 57.9 |
  | letters 120 | 84.12 / 86.95 | 87.09 | 130–139 | 919k–879k | 83.23 | 86.25 | 3.03 | 0.05 | 86.30 | 106.0 |
  | paddleboats 300 | 49.85 / 55.87 | 54.52 | 310–319 | 291k–295k | 47.28 | 51.16 | 3.88 | 0.11 | 51.27 | 89.0 |

  - **The GPU never waits for the host between steps.** The gap between one step's last pass and the next step's
    first is 0.05 ms (0.11 on paddleboats). So pipelined submission is GPU-bound: its period is the GPU busy time
    plus the idle time inside each step.
  - **The idle time inside a step is 2.7–3.9 ms with every pass timestamped.** That is dependent-launch latency
    between the step's 450 dispatches, the cost the split floors measure (above), and it is what items 5 and 9
    attack in pipelined mode. Pass sharing removes part of it in the clean step. How much is not resolved here,
    since an untimestamped pipelined step has no per-step GPU clock in Dawn.
  - **The window's first step is an outlier** (51 against 35 ms on seep, 89 against 51 on paddleboats). The pipelined
    figure is wall / 10 with one fence, so it carries that step, as fenced's mean carries its own slow steps. Compare
    it with fenced's mean, not its median. Then pipelined is 2.8 ms faster on seep, 1.4 on the dam and paddleboats,
    and level on letters.
  - **The steady state.** The period with timestamps (34.7, 37.5, 86.3, 51.3) is 0.2–2.2 ms above fenced's median,
    which is about what unsharing the passes adds. So the clean pipelined steady state is at, or a little under,
    fenced's median. These windows cannot resolve the 1.5 ms first-submit wait that pipelined should hide.
  - **§2's pipelined column measured later steps.** In the `*c` runs the pipelined window came after split, 30
    steps on (letters: 937k particles and 9.8k fine tiles against 894k and 8.0k). Its "pipelined is slower than
    fenced" was the state changing, not the submission. The states also differ between runs at equal step numbers
    (paddleboats at step 300: 309k particles in `paddle-300p`, 285k in `paddle-300c`), so fenced and pipelined are
    compared within ±3 ms here, not better.
  - **Consequence for the port list.** Items 2, 7 and 12 change only fenced time. In pipelined (app) mode, the step is
    the GPU's busy time plus about 2–3 ms of launch latency inside the step.

## 4. Pressure

**Structure** (`uniform-pressure-arena.ts`, `uniform-pressure-arena-launches.ts`).
- Schedule: `UNIFORM_PRESSURE_ARENA_SCHEDULE` = cycles 4 (`bandCycles`), sweeps 3, tailCycles 3, tail "auto"
  (pages), nativeSweeps 1 (`nativeSmoothing` composite), krylov on.
- The solve encodes `min(4, wideCycles + 1)` cycles. wideCycles is the most cycles any of the last 4 receipts ran.
- Each cycle is 43 direct dispatches (39 in cycle 0):

  | stage | dispatches | what runs |
  |---|---|---|
  | L0 h band, red-black tiles | 6 | `pRelax`, 3 pre and 3 post |
  | native 4h | 4 | `pNativeRelax` |
  | measure and restrict | 3 | `pResidual2`, `pRestrict0`, `pRebaseC` |
  | L1 2h | 3 | `pDownA`, `pDownR`, `pUpA` |
  | L2 4h | 5 | `pNativeOperator`, `pDownB`, `pDown2`, `pBulkLift`, `pUpB` |
  | 8h/16h/32h tail, 3 times | 16 | `pDown3`, `pDown4`, `pTail` (one workgroup), `pUp4`, `pUp3`; the last pass doubles `pUp4` |
  | prolong | 1 | `pProlong0` |
  | GCR(1) | 5 | `kApplyT`, `kApplyNB`, `kReduce` (one workgroup), `kBound`, `kStep` |

- **Convergence is measured after pre-smoothing.** Each cycle's `pResidual2`/`pRestrict0` measure runs after the
  pre-smoothing and writes history word k. Later kernels gate on `bConvergedBefore(pCycle)`/`bConverged()`. A
  step that converges at cycle k therefore runs cycle k's pre-smoothing, native sweeps, measure and rebase in
  full (about 0.37–0.5 ms), and every later launch of the encoded slots early-exits.

**Cost a cycle** (last split step, cycle 1, ms):

| scene | cycle | L0 smooth | native | measure+restrict | L1 | L2 | tail (16) | prolong | GCR |
|---|---|---|---|---|---|---|---|---|---|
| seep 600 | 1.54 | 0.22 | 0.12 | 0.12 | 0.11 | 0.25 | **0.43** | 0.02 | 0.27 |
| seep 1100 | 1.52 | 0.25 | 0.09 | 0.12 | 0.10 | 0.25 | **0.44** | 0.02 | 0.25 |
| dam 45 | 1.59 | 0.30 | 0.13 | 0.16 | 0.15 | 0.17 | **0.37** | 0.03 | 0.28 |
| letters 120 | 2.45 | 0.50 | 0.16 | 0.34 | 0.26 | 0.27 | **0.40** | 0.04 | 0.47 |
| paddleboats 300 | 1.66 | 0.24 | 0.15 | 0.17 | 0.12 | 0.25 | **0.38** | 0.02 | 0.33 |

- The arena solve is latency-bound: 43 dispatches at a mean of 36–57 µs.
- The fine-level work (L0 smoothing) is only 14–20% of a cycle. The 8h–32h tail is the largest stage on every
  scene (24–28%), and it is mostly one-workgroup `pTail` (37–38 µs, 3 a cycle) plus five-dispatch legs of
  14–40 µs each.
- Per step, the tail is 1.48 (dam), 1.62 (seep), 1.66 (letters) and 1.81 (paddleboats) ms of a 6–11 ms solve.
- One-workgroup launches: 36–39 a step, 0.8–1.1 ms. Of that, `pTail` is 12 launches (0.41–0.46 ms) and
  `kReduce` 3 launches (0.12–0.15 ms).

**Cycle distribution, cap and convergence.** Residual target 0.05 1/s.
- The encoded/ran histograms are from the attribution runs.
- The uncapped columns are from `pressure-classes.mts --set=bandCycles:8` (diagnostics on, same scene and
  steps).

| scene | ran (cap 4) | encoded | steps capped above target | uncapped mean | uncapped histogram |
|---|---|---|---|---|---|
| seep 600 | 3: 259, 4: 368 (mean 3.58) | always 4 | 43% | 4.17 | 3: 16, 4: 8, 5: 11, 6: 6 |
| seep 1100 | 3: 469, 4: 658 (mean 3.58) | always 4 | 26% | | |
| dam 30–90 | 4 in 72 of 75 | always 4 | 70% | 4.36 | 4: 39, 5: 22 |
| letters 100–240 | mean 3.07–3.38 | mean 3.5–3.7 | 9–10% | 4.09 | 3: 15, 4: 119, 7: 1, 8: 6 |
| paddleboats 300 | 2: 29, 3: 272, 4: 29 (mean 3.0) | 4 | 1–3% | 2.81 | 2: 8, 3: 33 |

- **Seep and the dam break are cap-limited, not slot-limited.** Over a quarter of their steps stop at the cap with
  the residual above target. The capped residual's median is 0.21 (seep 600) and 0.10 (dam), and the worst is
  1.5–3.3, against 0.05.
- Paddleboats never needs the fourth slot, which is pure headroom there. Letters sits just over the cap (uncapped
  4.09), and 6 of its 141 steps did not converge in 8 cycles.

**Contraction a cycle** (mean log10 of the largest residual after each cycle's pre-smoothing, uncapped probe):

| scene | c0 | c1 | c2 | c3 | c4 | c5 | factor a cycle |
|---|---|---|---|---|---|---|---|
| seep 600 | 2.26 | 0.92 | −0.12 | −0.84 | −1.08 | −1.66 | 0.05, 0.09, 0.19, 0.57 |
| dam 30–90 | 2.21 | 0.99 | 0.19 | −0.63 | −1.40 | −1.75 | 0.06, 0.16, 0.15, 0.17 |
| letters 100–240 | 1.90 | 0.78 | −0.11 | −0.97 | −1.67 | | 0.08, 0.13, 0.14, 0.20 |
| paddleboats 300 | 1.16 | 0.00 | −1.01 | −1.76 | | | 0.07, 0.10, 0.18 |

The target is log10 0.05 = −1.30.

**Where the late residual sits** (row class of the step's largest residual at cycles 2–4; class 1 is the
band/native seam, 7 the band free surface):
- **Dam:** seam in 60, 49 and 60 of 61 steps.
- **Letters:** seam in 134, 99 and 113 of 141.
- **Seep:** free surface in 29 of 41 and seam in 12, at cycle 3.
- **Paddleboats:** seam and interior, split evenly.

At cycle 3, the rows still above target are:
- seam 58, wall 22 and free surface 65 (seep);
- seam 4976 against interior 2340 (dam);
- interior 997 and seam 880 (letters).

The slow mode is at the h/4h composite seam and, on seep, the free surface, not inside the bricks. It is the
NB-FLIP analogue of the seep's brick-face films (S2m), but on the seam between the band and the native rows.

**tailCycles 3 → 1** (`pressure-classes.mts --tailCycles=1` against the default, same scene, start step and
frames; final residual after the step's last cycle, 1/s; "final log10" is the mean log10 of that residual):

| scene, frames | run | mean cycles | histogram | geo-mean | median | worst | final log10 |
|---|---|---|---|---|---|---|---|
| seep 600–640 (41) | default (3) | 3.585 | 3: 17, 4: 24 | 0.0850 | 0.0433 | 1.30 | −0.66 |
| | tailCycles 1 | 3.585 | 3: 17, 4: 24 | 0.0825 | 0.0399 | 1.48 | −0.65 |
| dam 30–90 (61) | default (3) | 4.00 | 4: 61 | 0.0536 | 0.0530 | 5.35 | −1.27 |
| | tailCycles 1 | 4.00 | 4: 61 | **0.0859** | **0.0868** | 0.87 | −1.07 |
| letters 100–240 (141) | default (3) | 3.851 | 3: 21, 4: 120 | 0.0280 | 0.0281 | 0.077 | −1.57 |
| | tailCycles 1 | 3.872 | 3: 18, 4: 123 | **0.0379** | **0.0369** | 0.135 | −1.42 |
| paddleboats 300–340 (41) | default (3) | 2.805 | 2: 9, 3: 31, 4: 1 | 0.0207 | 0.0211 | 0.049 | −1.73 (c3) |
| | tailCycles 1 | 2.854 | 2: 6, 3: 35 | 0.0214 | 0.0220 | 0.049 | −1.72 (c3) |

- **The extra tail V-cycles are not free everywhere.** On seep and paddleboats, one tail cycle converges as well
  as three. On the dam break and letters, it leaves the capped final residual 60% and 35% higher (−0.20 and −0.15
  decades, about a quarter of a late outer cycle). Those are the two scenes whose late residual sits on the
  band/native seam, which the 8h–32h tail corrects.
- The schedule's comment ("one leaves the natives at a twin's 0.5 a cycle; three at 0.15") is borne out on the
  seam-dominated scenes. On the cap-limited dam, the lost accuracy cannot be bought back with a cycle.
- So tailCycles 1 alone is an iterate-changing loss on two of four scenes and fails "improvements generalize".
  The time it would save (about 62% of the tail, 0.9–1.1 ms a step) is real, but it needs a cheaper exact
  coarsest correction (§6, item 4), not fewer sweeps.

## 5. Occupancy and limiters (xctrace)

**Method.** `~/code/me/fluid-scratch/lane-a/tools/nbflip-xctrace.mts` (with its `-worker.mts`, not committed) drives a worker that warms the scene to the sample step
on the check slot, then records about 1.6 s of steps under `xctrace` (Metal System Trace with the GPU counter
profile), and reduces the trace with the `gpu-frame-profile` skill's limiter join: per encoder label, GPU time,
compute occupancy, ALU utilisation, read and write bandwidth (GB/s) and the top limiters, in percent.
- **Every pass carries `timestampWrites`.** Without them, the sealed schedule merges about 85% of the step into
  one encoder labelled "Uniform shared frame schedule" (§1). The first seep capture did that and is kept only for
  its aggregate (`artifacts/nbflip-port/attribution/xctrace/seep-600-merged-summary.txt`): occupancy 12.7%, ALU 19.3%, read bandwidth 8.1 GB/s,
  last-level cache 12.1%.
- **What this costs.** With split encoders the window runs slower than the clean step (a split-median of 50 ms on
  seep, 161 ms on letters). Occupancy and limiter percentages are per kernel and unaffected. The "share" column is
  the kernel's share of the captured kernel time, not of the clean step.
- **The captures.** All four waited for a quiet GPU and ran alone in the check slot. The windows are seep 601–632,
  letters 121–129, dam 46–65 and paddleboats 301–322 (`--bodies`).

**Top 10 kernels per scene** (share | occupancy % | ALU % | read / write GB/s | top limiter %):

| seep 600 | share | occ | ALU | rd / wr | top limiter |
|---|---|---|---|---|---|
| `advectParticlesDeferred` | 6.9% | 11.1 | 17.5 | 10.2 / 3.7 | last-level cache 23.7 |
| `pRelax` | 3.4% | 14.5 | 11.3 | 32.1 / 1.5 | last-level cache 17.7 |
| `arenaPublishCanonical` | 3.3% | 10.3 | 10.1 | 1.5 / 0.2 | ALU 10.3 |
| `arenaDistance` | 3.1% | 16.8 | 30.3 | 4.8 / 0.1 | ALU 33.6 |
| `classify` (census) | 2.8% | 12.9 | 24.9 | 2.1 / 0.1 | ALU 25.8 |
| `geometryRegional` | 2.3% | 7.4 | 14.0 | 20.0 / 0.7 | last-level cache 19.3 |
| `resolve` (phi) | 2.2% | 23.4 | 33.1 | 4.0 / 0.6 | ALU 37.3 |
| `advectStagedTail` | 2.2% | 8.6 | 16.5 | 6.1 / 0.8 | last-level cache 18.0 |
| `advectDeferred` | 2.0% | 8.2 | 14.4 | 7.0 / 3.7 | last-level cache 19.1 |
| `ufxH` | 1.8% | 15.8 | 26.5 | 5.2 / 2.1 | ALU 29.6 |
| time-weighted, all 151 stages | | 11.8 | 16.5 | 7.9 | |

| dam 45 | share | occ | ALU | rd / wr | top limiter |
|---|---|---|---|---|---|
| `advectParticlesDeferred` | 25.1% | 11.9 | 22.0 | 14.3 / 4.5 | last-level cache 23.2 |
| `arenaDistance` | 3.4% | 17.1 | 33.6 | 5.2 / 0.1 | ALU 37.6 |
| `transferSeams` | 3.0% | 9.9 | 15.3 | 5.1 / 0.8 | ALU 15.7 |
| `transferDeposit` | 2.9% | 14.7 | 31.8 | 8.9 / 9.0 | ALU 35.0 |
| `classify` | 2.7% | 13.9 | 24.0 | 3.1 / 0.1 | ALU 24.7 |
| `arenaPublishCanonical` | 2.6% | 11.0 | 14.0 | 2.2 / 0.4 | ALU 14.3 |
| `pRelax` | 2.6% | 25.6 | 22.8 | 68.0 / 5.1 | last-level cache 44.1 |
| `advectDeferred` | 2.4% | 8.0 | 15.5 | 5.4 / 1.0 | ALU 15.9 |
| `resolve` | 2.1% | 25.2 | 35.9 | 5.1 / 1.4 | ALU 40.0 |
| `ufxH` | 1.9% | 16.6 | 26.7 | 8.9 / 3.5 | ALU 30.0 |
| time-weighted, all 143 stages | | 14.0 | 22.1 | 12.5 | |

| letters 120 | share | occ | ALU | rd / wr | top limiter |
|---|---|---|---|---|---|
| `advectParticlesDeferred` | 15.7% | 12.4 | 18.3 | 16.1 / 5.6 | last-level cache 27.3 |
| `resolve` | 4.6% | 25.3 | 35.2 | 5.5 / 2.1 | ALU 39.0 |
| `arenaDistance` | 3.7% | 17.6 | 33.9 | 5.4 / 0.1 | ALU 37.9 |
| `pRelax` | 3.6% | 34.1 | 22.7 | 63.9 / 4.8 | last-level cache 43.0 |
| `classify` | 3.3% | 13.9 | 23.5 | 3.1 / 0.1 | ALU 24.1 |
| `arenaPublishCanonical` | 3.0% | 11.4 | 13.5 | 2.3 / 0.3 | ALU 13.7 |
| `advectMomentum` | 3.0% | 16.3 | 20.7 | 3.9 / 1.4 | ALU 21.5 |
| `pRows` | 2.9% | 12.1 | 22.5 | 6.4 / 14.1 | ALU 23.1 |
| `ufxH` | 2.4% | 16.4 | 26.6 | 7.6 / 2.6 | ALU 29.6 |
| `advectDeferred` | 2.4% | 9.7 | 16.4 | 7.6 / 1.2 | last-level cache 19.4 |
| time-weighted, all 116 stages | | 16.7 | 22.2 | 13.8 | |

| paddleboats 300 | share | occ | ALU | rd / wr | top limiter |
|---|---|---|---|---|---|
| `advectParticlesDeferred` | 7.0% | 12.3 | 20.7 | 14.5 / 5.2 | last-level cache 27.3 |
| `ufxH` | 3.8% | 17.1 | 26.5 | 5.5 / 2.0 | ALU 29.6 |
| `arenaDistance` | 3.1% | 17.1 | 32.3 | 5.7 / 0.1 | ALU 35.8 |
| `arenaPublishCanonical` | 2.6% | 10.8 | 11.4 | 2.0 / 0.3 | ALU 11.6 |
| `classify` | 2.4% | 13.4 | 21.0 | 2.8 / 0.1 | ALU 21.6 |
| `resolve` | 2.4% | 24.9 | 36.8 | 4.9 / 1.6 | ALU 41.5 |
| `transferSeams` | 2.2% | 12.9 | 18.8 | 5.4 / 1.1 | ALU 19.3 |
| `pRelax` | 2.1% | 16.2 | 12.9 | 37.9 / 2.8 | last-level cache 25.2 |
| `ufx4` | 2.1% | 19.1 | 16.8 | 15.8 / 11.2 | ALU 18.3 |
| `geometryRegional` | 1.9% | 8.6 | 14.9 | 28.0 / 0.9 | last-level cache 24.6 |
| time-weighted, all 174 stages | | 13.9 | 20.8 | 10.5 | |

On letters, a few stages overlapped others in the trace (exclusive time 2–31% for `project`, `arenaNearest` and
the tail legs), so its shares are approximate. The counters are unaffected.

**Reading.**
- **Nothing is near a hardware limit.** The highest top limiter of any listed kernel is 44% (`pRelax`'s
  last-level cache on the dam). The time-weighted occupancy is 12–17% and ALU 17–22% on every scene. NB-FLIP is
  latency- and occupancy-bound everywhere, so bandwidth tricks (packing, f16 storage) are not where its time is.
  What pays is fewer dependent launches (items 5, 9), less serial work per thread (items 1, 6), or work removed.
- **The move** runs at 11–12% occupancy with last-level cache 23–27% and ALU 17–22% on all four scenes. It is the
  register-capped, latency-bound kernel of `one-pass-m1max-2026-11-02.md` §10, unchanged. That supports item 1:
  the separation loop's dependent reads at this occupancy are what a standalone pass at full occupancy hides. It
  also bounds item 4 (+33% occupancy at most).
- **`arenaPublishCanonical`** is the least utilised large kernel: 10–11% occupancy, ALU 10–14%, bandwidth about 2
  GB/s. Its threads spend their time waiting, which is what item 6's tile skips remove.
- **`pRelax`** is the one kernel that approaches a memory limit (letters and dam: 64–68 GB/s, last-level cache
  43–44%, occupancy 26–34%). Fusing more work into it (S2g's restriction fused into smoothing, item 5) should be
  costed against that.
- **The pressure tail confirms item 3.** `pTail` runs at 0.6–1.6% occupancy (one workgroup), and `pDown3` at 3–4%.
  On seep, `pTail`, `pUp4` and `pDown3` together are 4.7% of the captured time, more than `pRelax`.
- **`geometryRegional`** has MMU as a top limiter (41% on the dam), a sign of scattered page access. It is not ranked
  (1.7–2.3% of the captured time).

## 6. Ranked port list

Expected ms are per step, at the sample points of §2, as a range. The derivation is given for each. "Fenced" items
change only fenced step time (the GPU does the same work); all others change both. Scenes: seep (600 / 1100), dam
(45 / 200), letters (120 / 220), paddle (300).

**Summary**

| # | candidate | §2 item | lane | class | seep | dam | letters | paddle | risk |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Separation out of the move into its own pass | 3 (kernel shape), 5 | P2 | bitwise | −1.4 to −2.5 | −1.3 to −3.5 | −3.5 to −5.4 | −1.2 to −2.1 | low |
| 2 | Two-part submit at the census | 1 | P2 | bitwise, fenced | −1.0 to −1.3 | −1.0 to −1.3 | −1.0 to −1.3 | −1.0 to −2.6 | low |
| 3 | Pressure tail: one 8h–32h cycle with a stronger coarsest correction | 2 (exact coarsest) | P1 | iterate-changing | −0.9 to −1.0 | −0.9 | −0.9 to −1.0 | −1.0 to −1.1 | medium: needs the go/no-go probe |
| 4 | Move kernel: split the register-heavy trace from the ticket/append loop | 3 | P2 | bitwise | 0 to −0.5 | 0 to −1.0 | 0 to −1.8 | 0 to −0.6 | medium: an upper bound, may be zero |
| 5 | Pressure arena dispatch fusion (43 → about 37 a cycle) | 1, 2 (fused restriction) | P1 | ulp-bounded to iterate-changing | −0.35 to −0.6 | −0.35 to −0.6 | −0.35 to −0.6 | −0.35 to −0.6 | low |
| 6 | Level-set fixed-point tile skips in arenaDistance / arenaPublishCanonical | 4 | P2 | bitwise | −0.3 to −0.8 | −0.4 to −1.0 | −0.8 to −2.5 | −0.4 to −1.0 | medium: skip fraction unmeasured |
| 7 | Rigid-body CPU records cached across steps | (not in §2) | P2 | bitwise, fenced | 0 | 0 | 0 | −1.0 to −1.5 | low |
| 8 | Convergence test before the V-cycle | 1 | P1 | iterate-changing (stopping point) | −0.15 to −0.2 | 0 | −0.1 to −0.2 | −0.35 to −0.4 | low |
| 9 | Small-dispatch fusion outside the pressure solve | 1 | P2 | bitwise | −0.3 to −0.6 | −0.3 to −0.6 | −0.3 to −0.6 | −0.3 to −0.6 | low |
| 10 | Seam-aware L0 / native smoother (staggered Schwarz analogue) | 2 (S2m) | P1 | iterate-changing | accuracy; time only if cycles drop | accuracy | 0 to −1.5 | 0 to −1.5 | high |
| 11 | P2G transposed lanes and flag stores (transferDeposit, transferSeams) | 5 | P2 | bitwise | −0.1 to −0.3 | −0.2 to −0.5 | −0.3 to −0.8 | −0.1 to −0.4 | low |
| 12 | Encode-ahead (on top of 2) | 1 (S2f) | P2 | bitwise, fenced | −0.3 to −0.5 | −0.3 to −0.5 | −0.3 to −0.5 | −0.3 to −0.5 | medium |
| 13 | PRESSURE_CAP failure bit | catalogue §10 | P1 | no time | 0 | 0 | 0 | 0 | none |

**1. Separation out of the move (P2, bitwise).**
- **Evidence.** The move is the largest kernel on every scene (5.1–14.4 ms, 11–19%), and it is one launch.
  `sampleSeparation:off` takes it from 14.43 to 7.11 ms on letters 120 (15.8 → 7.9 ns a particle) and from 25.3 to
  11.9 ns a particle on seep 1100. So separation plus the surface shift is about half the move: 7.2 ms on letters and
  2.9 ms on seep 1100.
- **Why it is expensive there.** `advectParticlesDeferred` is register-capped at 384 threads, 13.2% occupancy
  on letters (`docs/plans/one-pass-m1max-2026-11-02.md` §10). This round's seep 600 capture (§5) gives 11.1% occupancy, ALU 17.5% and last-level cache 23.7%: no limiter is near its ceiling. `nbSeparation`'s candidate loop is
  latency-bound and runs at that occupancy. Lane S measured the same loop shape standalone at 1.7 ns a particle
  (catalogue §5).
- **Derivation.** Saving = the separation share of the move − the cost of a standalone pass at 2–4 ns a particle
  (lane S's 1.7 plus the surface shift's 7 depth samples). Where no off-run exists (dam, paddle), the share is
  taken as 45–53% of the move, the seep and letters range.
- **Sketch.**
  - `nbSeparation(i,q,m)` and `nbSurfaceShift(q,d)` read only the pre-move bank (`particles[]`, which the move does
    not write) and the level set at q. So a pre-pass can compute `end += nbSurfaceShift(q, nbSeparation(...))`'s
    vector for every record into a vec3 scratch (12 B a particle, sized by the particle pool), and the move adds it
    at the same point (`uniform-narrow-band-flip.ts`, the `sampleSeparation==="pairs"` site in `nbAdvect`, about
    line 1480).
  - The `!ballistic` condition is decided inside the move. The pre-pass computes the vector for every record and the
    move adds it only when `!ballistic`, so the arithmetic and its order are unchanged.
  - The pre-pass runs over the same slab job list as the move, one thread a record, with no workgroup memory.
  - The depth sampler alone caps a kernel at 384 threads (same §10). So the pre-pass should hold `nbSeparation`
    only. `nbSurfaceShift` (7 depth samples, needed only where depth > −1) stays in the move, which already has
    that sampler live, or goes in a second launch over the near-surface records.
- **Equivalence.** Bitwise by construction (same function, same inputs, same addition order). Check: a fixed-state
  one-step particle-bank compare.
- **Risk.** Low. One more launch (about 10–25 µs), and 12 B a particle of scratch.

**2. Two-part submit at the census (P2, bitwise, fenced only).**
- **Evidence.** In fenced mode the GPU idles until the first command buffer is submitted, 1.4–1.8 ms into
  advanceTo in clean steps (2.8–3.5 ms on paddleboats; §3). That buffer then holds 135–137 dispatches, 9.9–29.7 ms of
  GPU work. The census prefix (the first 18 dispatches) is 2.0–4.5 ms of GPU work, which covers encoding the rest of
  the first buffer. Fenced minus GPU busy is 2.7 ms (seep 600), 3.6 ms (dam 200) and 4.7 ms (paddle 300).
- **Derivation.** Saving = first-submit time − the time to encode and submit the census prefix (about 0.3–0.5 ms of
  the 1.5). On paddleboats, `syncBodies` and `syncMixedLayout` (1.47 ms of CPU) run before any encoding. They
  stay in front of the GPU unless the census can be submitted before them, hence the wider range (−1.0 if they must stay first, −2.6 if the census can go before them).
- **Sketch.** In `uniform-mixed-frame.ts`, add one `flush` after the census pass (the "Uniform dynamic ownership
  census" pass, dispatch 18 of the step) alongside the existing flushes at the move, the distance and the advection
  correction. Check that no `queue.writeBuffer` issued between the census and the move is meant to land before the
  census (writeBuffer is ordered at the queue, so a write issued after the first submit lands after it).
- **Equivalence.** Bitwise: the same commands in the same order, in more command buffers.
- **Risk.** Low. Catalogue §1.2: the prefix must be short to translate and long enough on the GPU. A 60-dispatch
  prefix gave nothing on the seep.

**3. Pressure tail: one cycle with a stronger coarsest correction (P1, iterate-changing).**
- **Evidence.** The 8h–32h tail (3 V-cycles, 16 dispatches a cycle, `pTail` one workgroup) is the largest stage of
  every pressure cycle: 0.36–0.45 ms a cycle, 1.5–1.8 ms a step (§4). tailCycles 1 saves about 62% of that. But
  it raised the final residual 60% on the dam and 35% on letters at equal cycles (§4), where the late residual sits
  on the seam that the tail corrects. As it stands, it fails "improvements generalize".
- **Derivation.** 10 of the 16 tail dispatches go, at the measured tail cost a dispatch (23–28 µs): −0.9 to −1.1
  ms a step, if one cycle with a better coarsest correction matches three cycles' residual.
- **Sketch.**
  - Replace the 32-sweep red-black `pTail` (`uniform-pressure-arena.ts`, `fn pTail`, TAIL_SWEEPS 32) with an exact
    coarsest solve: a factorisation of the 32h operator once per step, then a solve per cycle (catalogue §2, S2i:
    Gauss-Jordan once per step; Trottenberg et al. 2001). The page tail holds at most TAIL_ROWS = 4096 rows. The
    active 32h row count on these scenes is not yet measured: if it is in the hundreds, a dense factor fits, and
    above that a bound (fall back to the sweeps) is needed.
  - Or keep the sweeps and strengthen the 8h/16h legs (pDown3/pDown4/pUp4/pUp3) as block Jacobi with restriction
    fused into pre-smoothing (S2g).
- **Go/no-go (cheap).** On the `pressure-classes.mts` windows of §4 (dam 30–90, letters 100–240), tailCycles 1 plus
  the candidate must reach the default's final residual (log10 −1.27 and −1.57) at equal or fewer cycles.
- **Risk.** Medium. The tail sees every scene's coarse topology, and the symmetry oracle is required
  (`seep-smoother.test.ts` pattern).

**4. Move kernel: split the register-heavy trace from the ticket loop (P2, bitwise).**
- **Evidence.** The move runs at 11–13% occupancy with no limiter above 24–28% (§5). The sampler,
  `particleDepthIn`, `nbFinish` and `nbRigidContact` each cap it at 384 threads on their own, and the body alone at
  512 (`docs/plans/one-pass-m1max-2026-11-02.md` §10). So a split can raise occupancy by at most a third, and only if
  all four leave the trace kernel. The solid-free twin, which removed one of the four, measured nothing (10.68 against
  11.02 ms, within noise; rejected in e4041e9a).
- **Derivation.** It is an upper bound, applied to what is left of the move after item 1 (about half of it). At most
  25% of the remainder (+33% occupancy on a latency-bound kernel): seep 600 / 1100 1.9–2.6 ms → −0.5 to −0.7; dam 45 /
  200 3.0–4.0 → −0.75 to −1.0; letters 7.1 → −1.8; paddle 2.6 → −0.6. The split's own cost (the trace state handed
  between launches through memory, 12–24 B a particle) comes off that, so the lower end is zero.
- **Sketch.** After item 1, split `nbAdvect` at the end of the trace: launch 1 runs the sampler and `particleDepthIn`
  and writes end position, velocity and depth to scratch. Launch 2 runs `nbFinish`, `nbRigidContact` and the
  ticket/append loop. Each launch's register footprint must then be checked in Xcode's shader profiler before any
  timing.
- **Equivalence.** Bitwise (the same arithmetic, staged through memory in f32).
- **Risk.** Medium. Of all the items, this is the one most likely to measure zero. It sits here because it shares
  item 1's kernel and should be decided in the same change: first profile the registers, and stop if launch 1 still
  caps at 384.

**5. Pressure arena dispatch fusion (P1).**
- **Evidence.** The solve is latency-bound: 43 dispatches a cycle at a mean of 36–57 µs, and the fine-level work is
  only 14–20% of a cycle (§4). Each dependent dispatch costs 7–25 µs of floor.
- **Candidates a cycle.** measure + restrict + rebase (`pResidual2`, `pRestrict0`, `pRebaseC`) 3 → 1; `pDownA` +
  `pDownR` 2 → 1; `pNativeOperator` + `pDownB` 2 → 1; `kApplyT` + `kApplyNB` 2 → 1; `kBound` + `kStep` 2 → 1. That is 6
  fewer, 43 → 37.
- **Derivation.** 6 dispatches × 4 encoded cycles × 15–25 µs = −0.35 to −0.6 ms a step, on every scene.
- **Equivalence.** Fusing where the second kernel reads only its own workgroup's rows of the first is bitwise.
  Where it changes a reduction order, ulp-bounded. Where it changes what a halo sees (restriction fused into
  smoothing, S2g), iterate-changing.
- **Risk.** Low. Per-cycle savings multiply by the encoded cycles (4), not the mean.

**6. Level-set fixed-point tile skips (P2, bitwise).**
- **Evidence.** `arenaDistance` + `arenaPublishCanonical` are 1.7 (seep), 2.3 (dam), 5.2 (letters) and 2.2
  (paddleboats) ms a step, and the whole level-set area is 19–22% of the step on every scene.
- **Derivation.** Unknown skippable fraction. The range assumes 20–50% of the two kernels' tiles are a fixed point
  (all air, all −PHI_MAX, or unchanged since the last step), against the seep's larger deep reservoir.
- **Sketch.** In `uniform-surface-brick-arena.wgsl.ts` (`arenaDistance`, about line 306; `arenaPublishCanonical`,
  about 729), skip tiles whose inputs are a fixed point, using packed sign-code tiles for the cut test (catalogue
  §4, S2o).
- **Go/no-go.** First a probe counting the tiles that would skip, per scene.
- **Risk.** Medium. The NB-FLIP band is narrow, so the fraction may be small.

**7. Rigid-body CPU records cached across steps (P2, bitwise, fenced; paddleboats only).**
- **Evidence.** On paddleboats, `syncBodies` → `uploadBodies` costs 1.0 ms of CPU a step, of which
  `rigidMotorRecords` → `rigidAxleLanes` is 0.57 ms. `syncMixedLayout` → `mixedBodyTiles` costs 0.43–0.48 ms. All of
  it runs before the first encode.
- **Sketch.** `lib/core/webgpu-rigid-body.ts` (`rigidMotorRecords`, `uploadBodies`) and
  `lib/methods/uniform/webgpu-uniform-reference.ts` (`mixedBodyTiles`): cache what depends only on the assembly
  topology (axle lanes) and recompute only the pose-dependent words.
- **Derivation.** 1.0–1.5 ms off the CPU in front of the first submit. Fenced only: in pipelined mode the CPU is not
  the bottleneck.
- **Risk.** Low. It holds for every body scene and costs nothing elsewhere.

**8. Convergence test before the V-cycle (P1, iterate-changing in the stopping point).**
- **Evidence.** Convergence is measured after each cycle's pre-smoothing, so a converged step still runs that
  cycle's L0 pre-smoothing, native sweeps, measure and rebase (0.37–0.5 ms) in the next encoded slot (§4).
- **Derivation.** 0.37–0.4 ms × the mean number of spare encoded slots a step: 0.4 on seep (41% of steps
  converge at 3 of 4), 0 on the dam (always 4), 0.3–0.5 on letters (3.05 ran of 3.54 encoded), 1.05 on paddleboats
  (2.95 of 4).
- **Sketch.** Test the residual after `kStep` of cycle k−1 (a residual norm is already formed there by GCR), and gate
  cycle k's first launches on it (`bConvergedBefore`). This is standard PCG order (Saad 2003, Alg. 9.1), as S2q.
- **Risk.** Low. It changes which residual is compared with the target, so it is checked with the residual
  tolerance, then the gates.

**9. Small-dispatch fusion outside the pressure solve (P2, bitwise).**
- **Evidence.** 107–162 dispatches a step run under 20 µs, 1.2–1.8 ms in all. About 270 of the 450 dispatches lie
  outside the arena, in the census, layout, page and frame-plan passes (§3).
- **Derivation.** Fusing about 20–30 adjacent tiny launches at 10–20 µs each: −0.3 to −0.6 ms.
- **Sketch.** Merge adjacent launches that read only their own workgroup's output of the previous one. Candidates
  are in the page-pool commit, BULK16 transfer and detail-slot passes (dispatches 18–75 in the step's order).
- **Risk.** Low, but each merge needs its own read-set argument.

**10. Seam-aware smoother (P1, iterate-changing).**
- **Evidence.**
  - Seep and the dam break stop at the cap with the residual above target in 26–70% of steps. Uncapped, they need
    4.2–4.4 cycles (§4).
  - The late residual sits on the h/4h band–native seam (dam, letters) and on the seep's free surface.
  - Contraction slows to 0.15–0.57 a cycle after cycle 2.
- **Sketch.** The NB-FLIP analogue of the staggered Schwarz smoother (S2m; Smith, Bjørstad & Gropp 1996): a second
  block partition that straddles the seam rows (the composite level's interface), in `pRelax`/`pNativeRelax`, with
  exact reverse post-order. The CPU Float64 symmetry oracle is mandatory.
- **Derivation.** On seep and the dam, mainly accuracy (fewer capped steps). Time falls only where the encoded
  slot count falls, `min(4, wideCycles + 1)`: on letters and paddleboats a cycle less is −1.5 to −2.5 ms.
- **Risk.** High. It is the largest change to the solver.

**11. Transfer atomics (P2, bitwise).** `transferDeposit` (0.75–1.61 ms) and `transferSeams` (0.72–1.27 ms):
transposed lanes in the shared-atomic splat and flag stores for counters read only as nonzero (catalogue §5, S2n,
which cut p2g by 30%). The files are `uniform-particle-brick.wgsl.ts` (`particleBrickDepositWGSL`) and
`uniform-narrow-band-flip.ts`. Derivation: 10–30% of the two kernels.

**12. Encode-ahead (P2, bitwise for the GPU, fenced).** After item 2, what is left in front of the GPU is the
census encode plus Dawn's translation of the prefix, about 0.3–0.5 ms. Encode-ahead (S2f) removes it by encoding
step n+1 while step n runs. NB-FLIP sizes many launches from fixed-lag CPU evidence (receipts, `wideCycles`,
bandTiles written with `writeBuffer`), so those inputs would be one step older. That is still bitwise if every
width has early-exit headroom, but it must be audited launch by launch. Medium risk for a small fenced-only gain.

**13. PRESSURE_CAP failure bit (P1, correctness).** Latch a failure when the arena stops at the cycle cap above
target, as `24b48036` did for seep. 26–70% of seep and dam steps would latch today (§4), so the bit must be a
reported state, not a hard failure, until item 3 or 10 lands.

**Considered and not ranked**
- **Indirect-dispatch validation.** At most 25 µs a step, since only 1–2 launches are indirect (§3).
- **Lowering the encoded cycle cap.** Ruled out (catalogue §1.3). The empty fourth slot costs 0.8 ms on paddleboats
  (0.37 of it is pre-smoothing, which item 8 removes), but a capped step leaves the pressure unconverged.
- **Phi advection of vertices the reconstruction overwrites** (catalogue §6, S2c). Surface advection is 2.1–5.8 ms
  a step, and `coupleFine` stores reconstructed values at warm vertices (`nbStoreReconstructedWarm`) after it. The
  crossing search between them reads the advected field, though, so the dead fraction needs an audit before it is
  a candidate.


## 7. P1 / P2 independence

The two lanes can start together. Their files and kernels do not overlap, and each item's equivalence check
stands alone.

| item (§6 number) | lane | touches | depends on | can land alone |
|---|---|---|---|---|
| 1 separation pre-pass | P2 | `uniform-narrow-band-flip.ts` (move) | nothing | yes |
| 2 two-part submit | P2 | `uniform-mixed-frame.ts` (flush points) | nothing | yes |
| 3 tail: one cycle plus a stronger coarsest | P1 | `uniform-pressure-arena.ts` tail kernels, schedule | the go/no-go probe | yes |
| 4 move kernel split | P2 | `uniform-narrow-band-flip.ts` (move) | after 1 (same kernel) | yes, after 1 |
| 5 arena dispatch fusion | P1 | `uniform-pressure-arena.ts`, `uniform-pressure-arena-launches.ts` | after 3's decision (the tail legs move) | yes |
| 6 level-set tile skips | P2 | `uniform-surface-brick-arena.wgsl.ts` | a skip-fraction probe | yes |
| 7 body CPU caching | P2 | `lib/core/webgpu-rigid-body.ts`, `webgpu-uniform-reference.ts` | nothing | yes |
| 8 check before the V-cycle | P1 | `uniform-pressure-arena.ts` (gates, kStep) | nothing; easier after 5 | yes |
| 9 small-dispatch fusion outside pressure | P2 | census, layout and frame-plan kernels | nothing | yes |
| 10 seam-aware smoother | P1 | `uniform-pressure-arena.ts` (pRelax, pNativeRelax) | the symmetry oracle | yes |
| 11 transfer atomics | P2 | `uniform-particle-brick.wgsl.ts` (particleBrickDepositWGSL), transferSeams in `uniform-narrow-band-flip.ts` | nothing | yes |
| 12 encode-ahead | P2 | `uniform-mixed-frame.ts`, receipt readers | after 2 (same code path) | yes, after 2 |
| 13 PRESSURE_CAP failure bit | P1 | `uniform-pressure-arena.ts` verdict | nothing | yes |

- **Shared surfaces.** The P1 items share no file with P2 except through the frame's flush order. The arena's
  dispatches sit inside the last command buffer, so item 2's split point (at the census) is upstream of the
  solve.
- **What must be sequenced.**
  - Items 1 and 4 both rewrite the move, so 4 starts from 1.
  - Items 3 and 5 both rewrite the tail legs, so 5 lands after 3's decision.
  - Item 12 builds on 2.
- **Gates.**
  - Items 1, 2, 4, 6, 7, 9, 11 and 12 are bitwise by construction: a fixed-state one-step check, then the timing
    round.
  - Items 3, 5, 8 and 10 change the iterate (5 only where it reorders a reduction or a halo). Each passes the
    residual tolerance on the frozen states of §4 (the `pressure-classes.mts` windows), then lane G's trajectory
    gates on every scene.


## 8. Defects found while measuring

1. **The high-resolution dam break fails a residency audit under the app defaults, nondeterministically.** Two of
   five runs past step 150 failed:
   - **Run dam-300** (attribution tool, default schedule) failed at frame 246;
   - **pc-dam300-c8** (`bandCycles` 8) failed at frame 178;
   - dam-200 (to step 250), pc-dam-c8 (to 90) and the two dam-45 runs passed.
   - The failure message was the same both times:
     ```
     Uniform mixed frame N: GPU failure record: invalid support (layout build N+1: residency audit: liquid
     outside the resident pages, or a reader outside the closure) ... residency audit 1 (1: absent near,
     2: closure); projection withheld
     ```
   - The logs are `artifacts/nbflip-port/attribution/dam-300.log` and `pressure/pc-dam300-c8.log`.
   - **A third dam run died on the way to step 200** (`dam-200p`) with a different error at frame 196:
     `Departure travel gives a closure of 255 tiles, over the cap of 32 ... a sample is moving faster than the
     residency certificate covers` (`dam-200p-invalid-rc77.log`).
     - The guard saw Metal aborts in the run's own process: at 12:00:00.652 an `InnocentVictim` (00000005, with
       WindowServer also a victim of the same GPU recovery), then at 12:00:01.509 `ImpactingInteractivity`
       (0000000e).
     - By the GPU policy the run is invalid (rc 77), and the closure error is what corrupted state after a GPU reset
       looks like. So it is not counted above.
     - The GPU recovery's culprit is not in the log.
     - The rerun (same command, 12:22) passed through step 220. So this is either a GPU recovery from outside, or a
       third face of the nondeterministic defect.
   - This blocks a 20 s dam gate (1200 steps). Lane G and the coordinator need it before Phase 2 can gate on the
     dam. It is not a performance item, and lane A did not investigate it (no lib changes).
2. **Capped, unconverged pressure on seep and the dam (§4).** NB-FLIP has no failure bit for this: the capped
   residual reaches the info fields but is not latched. Lane S's open item for P1 (catalogue §10) applies as
   written.
3. **Shared check slots.** Lane G's reference benches ran on check slots 2 and 3 alongside lane A's runs. Two
   paddleboats windows came out roughly 2x slow (fenced 99.7 ms against 49.9 pipelined), and lane A's runs
   equally perturb lane G's references. From the gated runs on, lane A's tool waits for a quiet GPU before every
   phase. Lane G's reference timings should be checked the same way (for example, with `contention.sh` in the
   lane A scratch directory).
