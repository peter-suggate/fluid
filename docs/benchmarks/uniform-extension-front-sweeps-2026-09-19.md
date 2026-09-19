# Uniform Geometric: how many Sec. 3.3 front sweeps are worth paying for

Measurement date 2026-09-19. Tool `tools/benchmark-uniform-extension-front-sweeps-dawn.ts`
(measurement only; nothing under `lib/` changed). Raw data:
`docs/benchmarks/uniform-extension-front-sweeps-2026-09-19.json`.
Method values are the shipped defaults: `resolveMethodValues(uniformVolumeMethod,"balanced",{})`,
so `extensionFrontSweeps = 16`, liquid capacity balancing off, semi-Lagrangian velocity
transport, paper `dt = 1/30 s`.

Scenes: `minimal-power-dam-break-64` (64×64×64, h = 12.5 mm, ~94 k liquid cells — a full,
violent box) and `large-power-dam-break` (64×20×64, h = 50 mm, 1,472 cells of liquid in a
room-sized tank — a thin sheet in mostly air).

---

## Verdict

**(i) The smallest bit-identical budget is 16 on both scenes — i.e. there isn't one.**
The FIM front does *not* converge inside the shipped budget. `executedPasses` (the number of
update sweeps that ran with a non-empty active list) hits the 16 ceiling on **58 of 60 frames**
on mini64, and on 57 of those 60 frames the terminal active count is still non-zero (up to 220
faces). On large it is gentler — median 12, max 16, and only 3 of 60 frames end with faces still
active — but the maximum is still 16. So there is no budget below 16 that reproduces 16 over a
whole run, and 16 itself is a truncation rather than the converged JRW07 solve. The budget is
clamped to `min(max(dims),16)`, so what full convergence would give cannot be measured from
outside `lib/`.

Per frame the rule is exact and was verified directly: **a budget equal to that frame's
`executedPasses` is bit-identical to 16**. On large, frame 30 converged in 8 and N=8 reproduced
the reference step bit-for-bit (shell *and* phi); frame 60 converged in 4 and N=4 did.

There is also a protocol fact worth knowing: JRW07 does not update a newly activated node until
the next iteration, so **values advance one cell every two sweeps**. Sweeps pair up — N=1 and
N=2 produce bit-identical fields, and so do N=3 and N=4 (large frame 60: N=3 is bit-identical to
N=16, N=2 is not). Half of every sweep budget is spent activating, not solving. The nominal
"two-cell accurate band" therefore needs *four* sweeps in the straight-line case; everything
above that is paying for cut-cell detours and the long, snaking paths the 2 h band takes around
walls and thin liquid sheets — which is why the count runs to 16.

**(ii) How big the one-step surface perturbation is, and where the velocity differs.**
From an identical input state, one step, `|Δphi|` at vertices inside `|phi| < 2h`, in cells
(median over the 5 snapshots / worst snapshot):

| budget | mini64 RMS Δphi (h) | large RMS Δphi (h) |
|---|---|---|
| 8 | 0.0059 / 0.039 | 0.0000002 / 0.0000016 |
| 4 | 0.163 / 0.305 | 0.00022 / 0.00072 |
| 3 | 0.279 / 0.451 | 0.0040 / 0.026 |
| 2 | 0.509 / 0.969 | 0.0146 / 0.033 |
| 1 | 0.523 / 0.995 | 0.0146 / 0.033 |

So on mini64, **N=1 moves the free surface by about half a cell RMS in a single step** (worst
band vertex 25 h, but that is a handful of vertices where a cell flips wet/dry); N=4 moves it by
about a fifth of a cell; N=8 by under a hundredth. On large the same budgets move it by
0.015 h, 0.0002 h and ~0 — two to three orders of magnitude less, because that scene's liquid is
a thin sheet whose band converges quickly.

Spatially (mini64 frame 30, `|Δu|` normalised by the largest seed speed that frame):

| distance from liquid | N=8 RMS | N=4 RMS | N=1 RMS |
|---|---|---|---|
| 0–1 cells | 1.5e-5 | 4.8e-4 | 1.8e-3 |
| 1–2 | 1.3e-4 | 5.7e-3 | 1.9e-2 |
| 2–3 | 2.6e-4 | 7.9e-3 | 2.3e-2 |
| 3–5 | 7.8e-5 | 4.6e-3 | 4.1e-2 |
| 5–10 | 3.8e-5 | 3.2e-3 | 1.2e-2 |
| >10 | 9.7e-6 | 4.2e-3 | 1.1e-2 |

The story that table tells: at N=8 the difference is confined to the accurate band (1–3 cells)
and is ~1e-4 of the flow speed — 5 faces out of 774 k differ by more than 1 % of the max speed.
At N=4 the *far field* lights up as well: the hierarchy restricts from a smaller known set, so
the velocity everywhere in the air changes, not just the second ring. At N≤2 the far field is as
wrong as the band (on large it is *worse* than the band: 4.8e-2 RMS beyond 10 cells). Faces
inside the liquid are seeds and are exact at every budget.

**(iii) Frame-time saving.** The extension stage is 10.9 % of the mini64 frame and 6.4 % of the
large frame at N=16, so that is the whole prize. GPU-timestamped extension stage (median of 27
frames), and whole-frame queue-fenced wall:

| budget | mini64 extension | Δ vs 16 | large extension | Δ vs 16 | large wall Δ vs 16 |
|---|---|---|---|---|---|
| 16 | 9.31 ms | — | 2.884 ms | — | — |
| 8 | 5.64 ms | −3.67 ms (−39 %) | 1.704 ms | −1.18 ms (−41 %) | −0.6 ms (−1.3 %) |
| 4 | 3.74 ms | −5.57 ms (−60 %) | 0.983 ms | −1.90 ms (−66 %) | −1.90 ms (−4.2 %) |
| 2 | 1.38 ms | −7.93 ms (−85 %) | 0.590 ms | −2.29 ms (−80 %) | −2.25 ms (−5.0 %) |
| 1 | 1.05 ms | −8.26 ms (−89 %) | 0.459 ms | −2.43 ms (−84 %) | −2.55 ms (−5.7 %) |

Cost is linear in the budget: ~0.55 ms per sweep on mini64, ~0.162 ms per sweep on large, over a
fixed ~0.5 / ~0.3 ms of authority + seed + resolve + hierarchy + pack. On large the whole-frame
saving tracks the stage saving almost exactly, and the two N=16 arms agree to three decimals
(2.884 vs 2.884 ms; wall 44.98 vs 44.65 ms) — that lane is trustworthy. The mini64 wall lane is
not: its two N=16 arms differ by 2 ms and the reduced arms are not monotone in N (N=1 is 8 ms
*slower* than N=2). Use the mini64 extension-stage column and the large wall column; treat the
mini64 wall column as direction-only.

**(iv) Recommendation: lower the default to 8, not below.**

- 16 → 8 buys 39–41 % of the extension stage (−3.7 ms of an 85 ms mini64 frame, −1.2 ms of a
  45 ms large frame) for a field that is within ~2e-5 RMS of the shipped one, with a handful of
  faces (0–5 out of 774 k) differing by more than 1 % of the flow speed and the free surface
  moving less than a hundredth of a cell in a step. On the 60-frame census only 5 of 60 frames
  on large and (nominally) all frames on mini64 would clip — but the clipped tail is worth
  ~1e-5, which is four orders of magnitude below what the next step of the physics does.
- Going to 4 is a real change of the numerics, not a free optimisation: it is one accurate ring
  short, the hierarchy fill changes everywhere in the air, and the surface moves ~0.2 cell per
  step differently on mini64. It buys only a further 1.9 ms (2 % of the mini64 frame).
- N=1 (what the app is currently set to) is not "the same simulation, cheaper". It keeps one
  accurate ring, hands everything else to the hierarchy, and shifts the surface by half a cell
  per step on mini64. That it looks fine is a statement about a chaotic dam break's tolerance,
  not about equivalence. The extra saving over N=8 is 4.6 ms on mini64 (5.4 % of frame) and
  1.2 ms on large (2.8 %).
- Deleting the FIM front entirely is not supported by this data. At N=1 the front is what makes
  the first ring — and therefore the hierarchy's entire known set — correct; the far-field
  numbers show what happens when that set shrinks.

### What a one-step test can and cannot say

It bounds the **per-step** surface displacement caused by the sweep budget, from a genuinely
identical input state, with a measured noise floor of exactly zero (see below). That is the
right instrument for "is this change free?" and it answers it: at N=8, yes; at N≤4, no.

It says **nothing** about how those per-step differences compound over seconds of splash. This
solver is chaotic (its own repeats diverge), so a half-cell per-step difference may amount to a
visually identical splash with different droplets, or to a different splash. Nothing here can
distinguish those, and no trajectory comparison on this solver could either. The only evidence
that N=1 "looks the same" is the user's own eye, and this measurement neither confirms nor
refutes it — it only shows that the fields genuinely differ, and by how much.

---

## Method

One reference trajectory per scene at N=16, 60 frames. After every step the extrapolator's four
convergence words are read back outside timing. At frames 5, 15, 30, 45 and 60 the state the
step starts from is copied (GPU-side, every persistent texture and buffer) into a second solver
that never advances, and a third solver replays **one** step from that identical state at each
budget. Full trajectories are never compared.

**Fidelity assert (passed).** Every N=16 replay reproduced the reference step's packed transport
shell *and* its vertex phi bit-for-bit, on all 10 snapshot replays across both scenes. The state
copy is therefore faithful and phi is fully deterministic. Conservative transport's CAS float
sums make V itself irreproducible, and that is the entire noise floor:

| | mini64 | large |
|---|---|---|
| shell, N=16 vs N=16 | bit-identical | bit-identical |
| phi, N=16 vs N=16 | bit-identical (0 h) | bit-identical (0 h) |
| max cell \|ΔV\|, N=16 vs N=16 | 4.8e-7 … 7.6e-6 | 7.6e-8 … 3.6e-7 |
| total ΔV | ≤ 3.4e-5 of 94,208 (3.6e-10) | ≤ 1.3e-6 of 1,472 (9e-10) |

Cost: six 30-frame arms in one process, ordered 16, 1, 8, 2, 4, 16, first 3 frames dropped, one
fresh solver per arm; GPU timestamps bracket the whole `encodeVelocityExtrapolation` (authority
pass + front + hierarchy + pack) via a marker pass, frame wall is `performance.now()` around a
queue-fenced `advanceTo`.

---

## Tables

### 1. Sweeps actually executed at N=16 (60 frames)

| scene | executed-pass histogram | min / median / max | frames ending with faces still active | worst terminal active |
|---|---|---|---|---|
| mini64 | 4×2, 16×58 | 4 / 16 / 16 | 57 / 60 | 220 |
| large | 4×2, 5×2, 7×1, 8×5, 9×3, 10×9, 11×5, 12×15, 13×8, 14×2, 15×3, 16×5 | 4 / 12 / 16 | 3 / 60 | 27 |

Frames 1–2 of both scenes converge in 4 sweeps (still liquid, no cut-cell detours). From frame 3
mini64 never converges inside the budget again.

### 2. One-step sensitivity, packed transport shell (normalised by the largest seed speed)

`RMS` and `max` over all open faces; `>1 %` is the number of open faces differing by more than
1 % of that frame's largest seed speed. Median over the 5 snapshots / worst snapshot.

| scene | budget | RMS \|Δu\| med / worst | max \|Δu\| worst | faces >1 % worst | open faces |
|---|---|---|---|---|---|
| mini64 | 8 | 2.2e-5 / 5.8e-5 | 2.5e-2 | 5 | 774,144 |
| mini64 | 4 | 2.0e-3 / 3.4e-3 | 3.4e-1 | 9,752 | 774,144 |
| mini64 | 3 | 4.3e-3 / 6.2e-3 | 3.5e-1 | 27,043 | 774,144 |
| mini64 | 2 | 2.0e-2 / 1.0e-1 | 9.6e-1 | 348,249 | 774,144 |
| mini64 | 1 | 2.1e-2 / 1.0e-1 | 9.7e-1 | 356,967 | 774,144 |
| large | 8 | 9.0e-8 / 9.7e-6 | 2.0e-3 | 0 | 239,104 |
| large | 4 | 1.5e-4 / 8.6e-4 | 9.3e-2 | 114 | 239,104 |
| large | 3 | 2.7e-3 / 4.2e-2 | 2.0e-1 | 150,283 | 239,104 |
| large | 2 | 8.1e-3 / 4.6e-2 | 4.5e-1 | 194,842 | 239,104 |
| large | 1 | 8.5e-3 / 4.6e-2 | 4.5e-1 | 190,433 | 239,104 |

### 3. One-step consequence: surface and volume

`Δphi` at vertices with `|phi| < 2h`, in cells. `ΔV` is per cell; `ordinary` excludes cells where
either field exceeds 1.5 (see caveat).

| scene | budget | RMS Δphi med / worst (h) | max Δphi worst (h) | band vertices >0.1 h worst | max ordinary \|ΔV\| worst | worst total ΔV |
|---|---|---|---|---|---|---|
| mini64 | 8 | 0.0059 / 0.0385 | 3.8 | 9 | 0.98 | 6.8e-5 |
| mini64 | 4 | 0.163 / 0.305 | 17.4 | 589 | 1.43 | 9.7e-5 |
| mini64 | 3 | 0.279 / 0.451 | 24.6 | 1,382 | 1.43 | 9.7e-5 |
| mini64 | 2 | 0.509 / 0.969 | 24.6 | 2,743 | 1.49 | 6.7e-5 |
| mini64 | 1 | 0.523 / 0.995 | 24.6 | 3,027 | 1.49 | 6.3e-5 |
| large | 8 | 1.6e-7 / 1.6e-6 | 2.4e-5 | 0 | 1.0e-4 | 3.3e-6 |
| large | 4 | 2.2e-4 / 7.2e-4 | 1.4e-2 | 0 | 0.26 | 4.0e-6 |
| large | 3 | 4.0e-3 / 2.6e-2 | 4.7e-1 | 57 | 0.26 | 3.5e-6 |
| large | 2 | 1.5e-2 / 3.3e-2 | 4.7e-1 | 71 | 1.00 | 4.8e-6 |
| large | 1 | 1.5e-2 / 3.3e-2 | 4.7e-1 | 75 | 1.00 | 3.8e-6 |

Total V is conserved to 1e-9 relative in every arm at every budget, as expected: conservation is
a property of the transport operator, not of the velocity it is handed.

### 4. Cost, six arms in one process, order 16 · 1 · 8 · 2 · 4 · 16

30 frames per arm, first 3 dropped; median [p25–p75].

| scene | arm | budget | extension stage (ms) | whole frame wall (ms) |
|---|---|---|---|---|
| mini64 | 0 | 16 | 8.454 [6.816–12.124] | 84.00 [78.18–87.33] |
| mini64 | 1 | 1 | 1.049 [0.918–2.359] | 73.59 [68.90–79.60] |
| mini64 | 2 | 8 | 5.636 [4.194–7.799] | 72.06 [65.67–82.12] |
| mini64 | 3 | 2 | 1.376 [1.180–3.277] | 65.84 [62.33–77.13] |
| mini64 | 4 | 4 | 3.736 [2.228–5.439] | 71.15 [64.67–76.38] |
| mini64 | 5 | 16 | 10.158 [8.192–13.631] | 86.05 [79.84–88.19] |
| large | 0 | 16 | 2.884 [2.621–3.670] | 44.98 [42.57–47.85] |
| large | 1 | 1 | 0.459 [0.393–1.311] | 42.27 [39.87–44.70] |
| large | 2 | 8 | 1.704 [1.638–3.015] | 44.22 [42.69–45.57] |
| large | 3 | 2 | 0.590 [0.590–2.228] | 42.57 [39.96–44.60] |
| large | 4 | 4 | 0.983 [0.918–2.621] | 42.92 [40.66–44.85] |
| large | 5 | 16 | 2.884 [2.425–3.473] | 44.65 [41.61–46.23] |

---

## Caveats and things deliberately not claimed

- **The mini64 whole-frame wall lane did not resolve these differences.** Its two N=16 arms
  differ by 2 ms, the reduced arms are non-monotone in N, and the 16→8 wall drop (13 ms) is
  3.5× the measured stage drop. Reported, not used. The large wall lane is consistent with its
  own stage timings and with itself, and is used.
- Cost arms each run their own trajectory, so a whole-frame difference also contains "a
  different flow was simulated" (different pressure work, different active sets). This is
  another reason to trust the stage timestamps over the wall.
- The reference trajectory is not reproducible run-to-run (CAS float sums), so the sensitivity
  and cost phases ran different trajectories. Every sensitivity number is one step from one
  shared input state inside one process, which is the only comparison this solver supports.
- At these defaults (liquid capacity balancing **off**) the N=16 reference itself carries a
  population of grossly overfull cells — up to V = 118 in ~2,650 cells on mini64. They are
  present identically in the reference, so they are not caused by the sweep budget, but they
  dominate a raw max `|ΔV|` (up to 200 per cell). The `ordinary` column excludes them. This is
  a separate pre-existing finding, not a result of this experiment.
- No budget above 16 was measured: `setFrontPasses` clamps to `min(max(dims),16)`, which cannot
  be raised from outside `lib/`. The distance between 16 and full convergence is therefore
  unknown; the geometric decay of the N=4 → N=8 → N=16 differences (≈2 orders of magnitude per
  doubling) suggests it is ≲1e-5, but that is an extrapolation, not a measurement.
- The distance buckets use `max(0, phi)` at the MAC face centre from the input vertex phi. The
  kernel's own source test reads the `rho' = rho/V` authority field, not phi, so the 0–1 bucket
  contains both seeds and near-surface extended faces. Normalisation uses the largest component
  of the N=16 packed shell, which is exactly the largest seed speed (every extended value is a
  positive-weight average of seeds).
