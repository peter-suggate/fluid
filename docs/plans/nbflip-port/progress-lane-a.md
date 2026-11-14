# NB-FLIP port, lane A: progress log

Brief: docs/plans/nbflip-port-2026-11-15.md §3 "Lane A". Deliverable: docs/plans/nbflip-port/attribution.md.
No changes under lib/. Tools: tools/nbflip-attribution.ts (committed), scratch in ~/code/me/fluid-scratch/lane-a/.

## 2026-11-15 STARTED

- Read the plan §1-§3, lane S S2c-S2q, and the prior NB-FLIP performance notes (nbflip-step-time,
  hot-kernels, kernel-audit, ten-experiments, one-pass-m1max, paddleboats-60fps).
- Wrote `tools/nbflip-attribution.ts`: NB-FLIP through the method registry on any catalog scene, fenced to a
  chosen step, then four instrumented windows (clean wall/CPU/submit; per-pass timestamps; per-dispatch split
  timestamps with indirect-argument capture; pipelined throughput). Timestamp quantization is disabled
  (`disable-dawn-features=timestamp_quantization`; with it on, every reading was a multiple of 65.5 us).
- Smoke run, seep at step 20 (check slot): 454 dispatches a step, **453 direct and 1 indirect**, 267 kernels,
  74 passes, 4 submits. advanceTo 3.3 ms of CPU, of which queue.submit 0.78 ms. Clean wall 23.5 ms, pass-timestamp
  GPU busy 22.0 ms. NB-FLIP's arena already made every pressure launch direct
  (uniform-pressure-arena-launches.ts), so seep's indirect-validation tail does not apply as such.

## 2026-11-15 measurement rounds 1–8

- **Runs** (all on the check slot through `fluid-gpu-run.sh check`). They are listed in
  `artifacts/nbflip-port/attribution/` and `pressure/`, and the scratch logs are `round1.log`–`round8.log`.
  - Attribution: seep 600 and 1100, dam 45 and 200, letters 120 and 220, paddleboats 300.
  - Gated reruns `-c`, which wait for a quiet GPU before every phase and record what they saw (`contention`).
  - Separation-off runs (letters 120, seep 1100).
  - Pipelined runs with pass timestamps `-p` (the new `pipepass` phase).
  - Pressure probes: uncapped (`bandCycles:8`) and `tailCycles` 1 against 3 on all four scenes.
- **Invalid or discarded runs.**
  - dam-45 round 1: rc 77, a Metal hang abort in our process; the rerun was valid.
  - paddle-300 and paddle-300b: contended by lane G's benches on the other check slots (fenced 99.7 ms against
    49.9 pipelined).
  - The first xctrace round: the scratch tool was `.ts` outside the repo, so tsx compiled it as CJS and rejected
    top-level await. It was renamed to `.mts` and requeued.
- **Findings** (details in `attribution.md`):
  - NB-FLIP is wholly direct (1–2 indirect launches a step). The Dawn cost is the direct early-exit launch at 7–25
    µs each, not indirect validation.
  - The first command buffer goes out 1.4–1.8 ms into advanceTo in clean steps (2.8–3.5 ms on paddleboats,
    whose body sync is 1.5 ms of CPU in front of the first encode). The counted step's 2.2–2.5 ms is inflated by the
    dispatch-recording proxy.
  - The pressure tail (8h–32h, 3 cycles) is the largest pressure stage on every scene. Cutting it to one cycle is
    free on seep and paddleboats, but it loses 0.15–0.2 decades of final residual on the dam and letters, so it
    needs a stronger coarsest correction, not just fewer cycles.
  - Separation plus the surface shift is about half of the move (7.2 of 14.4 ms on letters).
  - The high-resolution dam break fails a residency audit nondeterministically past step 150 (2 of 5 runs). This
    is a defect for lane G and the coordinator.

## 2026-11-15 rounds 9–11: pipelined pass timestamps and per-kernel xctrace

- **pipepass** (`*p` runs: pipelined at the same steps as the `*c` fenced windows, then 10 pipelined steps with a
  pass timestamp pair, grouped by step). It now uses one pooled 4096-query set: one set per encoder, with 10 steps
  in flight, ran Metal out of sample buffers ("Cannot allocate sample buffer"), so the first paddle-300p and
  seep-600p were rerun.
  - The GPU runs steps back to back: there is 0.05 ms between steps (0.11 on paddleboats). Idle inside a step is
    2.7–3.9 ms with every pass timestamped.
  - The steady period is 34.7 (seep 600), 37.5 (dam 200), 86.3 (letters 120) and 51.3 (paddleboats 300) ms.
  - The window's first step is an outlier (+17 to +38 ms), and it inflates both wall / 10 figures. Compared with
    fenced's mean rather than its median, pipelined is 0–2.8 ms faster.
  - The earlier "pipelined slower than fenced" came from the `*c` runs measuring pipelined 30 steps later (a
    larger state).
- **xctrace per kernel.** The worker now puts `timestampWrites` on every split pass, so the sealed schedule cannot
  merge them (the first seep capture had 85% under "Uniform shared frame schedule"). There are captures for seep 600,
  letters 120, dam 45 and paddleboats 300, and the summaries are in `artifacts/nbflip-port/attribution/xctrace/`.
  - Time-weighted occupancy is 12–17% and ALU 17–22%. No listed kernel's top limiter is above 44%.
  - The move: 11–12% occupancy, last-level cache 23–27%.
  - `arenaPublishCanonical`: 10–11% occupancy, ALU 10–14%.
  - `pTail`: 0.6–1.6% occupancy.
- **Item 4 (move split) is demoted to an upper bound** (0 to −1.8 ms). Four functions each cap the move at 384
  threads, so a split raises occupancy by at most a third.
- **dam-200p, first attempt: rc 77.** The run's own process saw an InnocentVictim abort, then an
  ImpactingInteractivity abort, then the departure-travel closure error at frame 196. The rerun passed. This is
  recorded under §8.1.
- `npm run check:types`: only the 3 pre-existing lane-n errors. `npm run test:unit`: 1789 pass, 0 fail.
