# NB-FLIP port, lane G: cross-scene gates (progress)

Brief: docs/plans/nbflip-port-2026-11-15.md §3 "Lane G" and §4.

## 2026-11-15 STARTED (lane G)

HEAD 46a2ba94, tree clean. Lane A profiles NB-FLIP on the check slot at the same time.

## 2026-11-15 MILESTONE G1: box observer and cross-scene bench

Files:
- `tools/nbflip-observables.ts`: the box schema for the dam break, letters and paddleboats
  (totals, energies, (x, z) bins with liquid and highest wet top, floor front, wet floor,
  the dam's back-wall column), the shared WGSL accumulator and its Float64 oracle.
- `tools/nbflip-observe.ts`: NB-FLIP's owner traversal (the seep observer's, h and 4h,
  canonical phi, six-tetrahedron fill), face-mean velocities, and BULK16 regions decoded
  on the host with their (x, y, z) page and face-mean velocity.
- `tools/nbflip-gate-bench.ts`: the seep-bench loop for `--scene=dam|letters|paddleboats`
  (fenced or pipelined, check or timing slot). The seep keeps running through
  `tools/seep-bench.ts --method=nbflip`, unchanged.
- `tests/nbflip-observer.test.ts`: oracle against closed forms (a level pool: total, PE,
  bins, tops, front, column; a dam box; a BULK16 region as 64 4h owners; the letters' drop
  schedule, steps 25i+3), and the WGSL accumulator against the oracle (GPU, passed under
  `fluid-gpu-run.sh check`).

Smoke checks (check slot, HEAD 46a2ba94 lib):
- Dam break, t = 0: 184.00 L, the analytic 80 x 118 x 80 cells at 6.25 mm (184.4 L by the
  0.92 x 128 height); 24 BULK16 regions decoded; front 80 cells (Z = 1); column top
  117.75 cells (eta = 1.00). The front reaches the far wall (128) at step 11.
- `readStats` returns the volume pass of the last step that took one, so without a request a
  sample's nbDrift is one sample old (dam: nbDrift at 45 = observer at 30). The bench now
  requests the pass before each sample step's advance, as tools/probe-reservoir-bulk.ts does.
- The first dam run was invalid: another process's GPU hang (kIOGPUCommandBufferCallbackErrorHang,
  11:19:31) discarded its command buffers, and it then failed a residency closure at step 30.
  The rerun was clean and passed step 60.

## 2026-11-15 MILESTONE G2: references, gates, self-test

Files:
- `tools/nbflip-gates.ts`: the referee. It judges per-scene gate tables (C0, then plan §4.1),
  prints the timing-slot table beside each verdict, offers `--timing-only` for the timing
  round, and exits 1 on FAIL.
- `tools/nbflip-timing-round.sh`: the timing round for all four scenes. It runs baseline
  and candidate, fenced and pipelined, back to back on the timing slot, each after a quiet
  GPU (under 15% for 5 s). It logs provenance, utilisation and contention, checks Metal, and
  writes OUT/timing.md.
- `tools/nbflip-export.sh`: makes a read-only export of a commit, with an optional patch and
  a PROVENANCE file.
- `tools/nbflip-gate-bench.ts`: now refuses a `--set` that the method does not resolve to
  the given value.

References: five runs per scene at 33b78ebb (four for the dam), in
`artifacts/nbflip-port/ref/`. Two references were not enough:
- With two, a pure-noise row falls outside 3S in 18% of cases (model), and the data agreed.
- Run c judged on a and b failed paddleboats P2x.
- b judged against a failed the dam (D5k, D7t) and letters (L2, L2t).

Gates rebuilt from measurement, before any port landed:
- **Highest-wet-top profiles were dropped after step 16.** They measure a droplet maximum:
  RMS between baseline runs was 10–40 cells. They were replaced by mean-depth maps.
- **Dam rows after the impact became window means** over 30–90 and 105–165. At step 60 a
  baseline run's E was 2.7% low, and the sloshing phase splits the runs into two groups.
- **KE floors gained an absolute floor** of 2 J (0.3% of E0), because slosh minima are
  2–5 J.

**Baseline defect.** The NB-FLIP dam at 33b78ebb fails a residency audit ("liquid outside the
resident pages, or a reader outside the closure") at steps 178–192:
- 2 of 7 baseline runs failed this way, and so did the ignored-`--set` run;
- the GPU was clean.

So the dam's gated rows end at 165 (C0), and C0f (300 steps) is informational until the
defect is fixed. It needs an owner who may touch `lib/methods/uniform`.

Self-test:
- Leave-one-out over a–e: all PASS.
- Sixth baseline run f against a–e: PASS on all four scenes; worst margin 0.47.
- residual 0.5 tree: FAIL. The seep fails G1, G1w, G2 and G7; paddleboats fails P1. The dam
  and letters are insensitive to it.
- `--set=pressureFullCycles=1` was a no-op: NB-FLIP fixes the value at 3. The runs were
  discarded and the bench was fixed.

Invalid runs:
- Letters and paddleboats on the residual tree returned rc 77: another process's GPU hang,
  and the guard caught it. They were rerun one at a time and kept as `*-rc77` evidence.

More perturbations, all `--set` on the baseline export:
- `volumeControlSeconds=1` FAILs the dam: drift −18%, D4 margin 8.9.
- `pressureResidualTolerance=50` and `particleCellCap=8` match the baseline.
- `pressureFullCycles=1` and `phiCubicAdvection=off` were ignored by the method. The bench now
  refuses such a `--set`.

The dam defect is worse than first counted:
- 6 of 11 fenced 300-step runs at baseline physics stopped, at steps 178–247.
- Both timing-slot fenced runs were among them, so contention is not the cause.
- The dam is therefore timed over its 165 gated steps.

## 2026-11-15 MILESTONE G3: baseline timing round (A/A, timing slot, quiet GPU)

Results are in `artifacts/nbflip-port/timing-base/` and `timing-base-dam/`. Fenced median and
pipelined ms/step, A / A:

| scene (steps) | fenced median ms | pipelined ms/step |
|---|---|---|
| seep (1200) | 34.23 / 35.00 | 33.12 / 32.86 |
| dam (165) | 45.12 / 46.25 | 50.29 / 49.18 |
| letters (240) | 83.05 / 85.10 | 68.42 / 69.26 |
| paddleboats (300) | 49.26 / 50.22 | 43.66 / 44.46 |

The A/A spread is about 2.5%. A first round failed because the script wrote OUT through a
relative path from inside the tree; OUT and the trees are now absolute.

## 2026-11-15 MILESTONE G4: §7.1 drift amendment and references at 2761350a

Plan §4.5 (amendment), §4.6 (references) and §4.7 (timing).
- **Rules.** `tools/nbflip-gate-rules.ts` holds the rules, tested by
  `tests/nbflip-gate-rules.test.ts`:
  - one-sided drift;
  - the restored-volume band, as the hull of the two bands, reporting `unionMargin`;
  - `restoreBox`, checked against `boxOracle` on pools raised by exactly the lost volume.
- **Rows.** Drift rows G7, L1, P1, D4c and D4 are one-sided. Volume-coupled rows G3, G6c, G6, D2,
  D3, D5p, D5e, D5ce, D7t, D7d, L2, L4 and P2 use the band. The §4 originals print as
  "(as written)", not gated. The dam's 180–300 window is gated now.
- **Default references.** The referee defaults to `artifacts/nbflip-port/ref-2761350a`. The old
  set stays at `artifacts/nbflip-port/ref`.
- **Old references, amended rule.**
  - LOO a–e, run f, tol50 and cellcap8 pass.
  - residual 0.5 still fails the seep (G1, G1w, G2). Its G7 and P1 pass now: both are smaller
    losses.
  - volctl1 still fails the dam.
- **New references** (20 runs, all complete; reruns after rc 77 for letters a and d and seep c).
  - LOO a–e and run f pass; the worst margin is 0.68.
  - residual 0.5 fails the seep (G1 2.12, G1w 3.03, G2 1.92).
  - volctl1 fails the dam (D4 5.20, D5p 5.46, D5e 6.91, D3 2.67, D5k 1.62).
- **Finding.** Dam ref c at 2761350a conserves late volume (+0.8% at 300, against about −0.9% for
  the others).
  - D4's late S is 0.015, so its 180–300 tolerance is 4.5 points. 3S also binds P1 at 300
    (tol 0.53 points).
  - LOO-c's dam passes only under the amendment; as written it fails D4, D5p and D5e.

## 2026-11-15 MILESTONE G5: step-time baselines at 2761350a (plan §4.7)

- **Rounds.** Timing slot, 33b78ebb against 2761350a, in `artifacts/nbflip-port/timing-2761350a{,-seep2,-letters2}/`.
  - The first round's seep pair is void: another agent's CPU-heavy unit tests ran during it,
    plus another process's Metal aborts at 13:53:40.
  - The seep was re-run clean in `timing-2761350a-seep2/`.
- **Baselines at 2761350a**, fenced median and pipelined:
  - seep: 35.09 and 32.30 ms;
  - dam (165): 45.45 and 50.61 ms;
  - letters: 84.81 / 83.44 and 69.57 / 69.18 ms, with fenced mean 71.81 / 70.70;
  - paddleboats: 50.50 and 45.22 ms.
- **Letters' +1 ms is noise.** Two rounds give −1.1 and −3.6 ms fenced mean. 33b78ebb itself
  spans 71.1–74.3 ms over four runs.
- **Other scenes.** All are within the about 2.5% A/A spread.

## 2026-11-15 MILESTONE G6: second drift ruling, drift bounded by NB-FLIP's envelope (plan §4.5 (1a))

- **Rule.** Drift rows now pass when |c| ≤ max_i |ref_i| + floor (`driftGate`). The first
  ruling's |mean| + max(floor, 3S) prints as "(first ruling)" (`driftMeanGate`), not gated.
  The hull ruling is accepted.
- **Bounds at 2761350a.**
  - D4: 5.46 / 2.19 / 2.14%, down from 6.74 / 3.42 / 5.22%.
  - D4c: 5.91%, down from 6.42%.
  - G7: 0.61%, down from 0.67%.
  - P1 at 300: 1.32%, down from 1.54%.
- **Self-test.** LOO a–e and f pass on both sets; the worst drift margin is 0.73. residual 0.5
  fails the seep only (G1, G1w, G2). volctl1 fails the dam (D4 13.6).
