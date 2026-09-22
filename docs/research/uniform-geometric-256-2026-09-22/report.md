# Uniform Geometric: Figure 7 at 256³

Follow-up: [memory reduction to below 3 GB, with fresh before/after timings](memory-report.md).

Captured 2026-09-22 with Dawn/Metal on Apple M1 Max, macOS 26.6.2. Repository base: `407a11f372a2fce54e713bbf8f8b41c2a51bf529`, with the new scene and profiling tool in the working tree. No solver changes or parameter ablations.

## Scene and measurement

`cm12-figure-7-256` is available in the scene catalog under **Method comparisons** and selects Uniform Geometric (`uniform-volume`, balanced, default values). It keeps Figure 7’s 6.4 × 6.4 × 6.4 m closed, free-slip tank, 1 m radius liquid ball centered at y=4.5 m, gravity 10 m/s², and 1/30 s step. Cell size changes from 0.05 m to 0.025 m: 16,777,216 cells. The boundary shell is compiled at the new resolution.

Each resolution ran in a separate process under the exclusive WebGPU lease: 60 advances / 2 simulated seconds, one substep per advance. Summaries exclude frames 1–4. Frames 5–24 cover free fall; frames 25–60 cover the expected first floor contact and spreading (analytic first contact is about 0.837 s). These windows classify the intended motion, not a visual validation.

All 60 samples in each run use hardware GPU timestamps. Stage values are contiguous solver seam intervals, including intervening work, rather than isolated kernel durations. Stage sums exactly close to total GPU time. Wall times are queue-fenced simulation advances, with rendering, stats readbacks and the 115 ms recorder-cadence gaps excluded. Timeline instrumentation is enabled; these are not uninstrumented throughput measurements. One trajectory per resolution, so p10/p90 describe evolving frame costs, not repeated-run uncertainty.

## Timings

Values below are arithmetic means in milliseconds per advance. Means make the stage table additive.

| Window | 128³ GPU | 256³ GPU | GPU ratio | 128³ wall | 256³ wall |
|---|---:|---:|---:|---:|---:|
| Free fall, 5–24 | 44.53 | 201.82 | 4.53× | 50.70 | 210.90 |
| Impact / spread, 25–60 | 52.28 | 330.83 | 6.33× | 59.40 | 340.12 |
| All measured, 5–60 | 49.51 | 284.76 | 5.75× | 56.29 | 293.97 |

| GPU stage, frames 5–60 | 128³ ms | 256³ ms | 256³ share | Ratio |
|---|---:|---:|---:|---:|
| CM11a Full-Cycles | 7.75 | 66.76 | 23.4% | 8.61× |
| Dense geometric volume coupling | 6.90 | 46.42 | 16.3% | 6.73× |
| Dense vertex phi transport and redistance | 7.35 | 39.97 | 14.0% | 5.44× |
| Dense conservative volume gather | 5.01 | 36.00 | 12.6% | 7.19× |
| Sec. 3.3 interface authority | 4.68 | 24.20 | 8.5% | 5.17× |
| CM11a topology + RHS pyramid | 2.57 | 17.32 | 6.1% | 6.75× |
| Velocity advection + body forces | 1.81 | 11.84 | 4.2% | 6.54× |
| Sec. 3.3 hierarchy fill + transport shell | 3.92 | 10.14 | 3.6% | 2.59× |
| Dense conservative volume sharpening | 2.13 | 9.42 | 3.3% | 4.42× |
| Sec. 3.3 narrow-band FIM front | 4.76 | 9.03 | 3.2% | 1.90× |
| CM11a parity copy + fine residual | 1.43 | 6.23 | 2.2% | 4.37× |
| Pressure projection + surface publication | 0.88 | 4.87 | 1.7% | 5.54× |
| Diagnostics reduction | 0.33 | 2.56 | 0.9% | 7.78× |

## Interpretation

- Doubling each axis costs **5.75× GPU time** overall for **8× cells**. The 256³ measured wall average is 293.97 ms, about 3.4 simulation advances/s; rendering is additional.
- Pressure Full-Cycles is the largest individual stage: **66.76 ms / 23.4%**. Including topology/RHS and parity/residual work, the pressure stages total **90.32 ms / 31.7%**. Pressure-cycle mean scaling is **8.61×**.
- The geometric stages also matter: volume coupling **46.42 ms**, vertex phi transport/redistance **39.97 ms**, conservative volume gather **36.00 ms**. Together they account for **43.0%** of measured GPU time.
- Impact changes the balance: coupling rises from **20.39 to 60.88 ms**, phi transport/redistance from **13.29 to 54.80 ms**, and pressure Full-Cycles from **51.24 to 75.38 ms**. Use the impact window when assessing improvements, as a free-fall-only profile misses these costs.
- In the 256³ run, the lagged pressure budget encodes 1–3 cycles and executes 1–2. This is the production budget, not a fixed one-cycle ablation. The larger pressure cost combines more spatial work and a different convergence trajectory; the stage profile alone cannot separate them.
- Solver-reported allocation grows from **0.953 GiB to 7.521 GiB** (7.89×). The large scene is a useful stress case, but active-page execution has not removed its dense backing allocation. This figure is the solver’s accounting, not measured process peak RSS.

## Validation and caveats

- Both runs completed 60 advances with zero uncaptured WebGPU validation errors and fresh hardware timestamp samples for every frame.
- The 256³ catalog document was checked for exactly 256³ cells, identical physical tank/fluid/numerics to the original, the correctly regenerated boundary shell, and the Uniform Geometric method profile.
- 256³ volume-cell sum changes from 268065.88 at frame 1 to 267723.16 at frame 60 (**−0.128%**). This is a timing run, not a full conservation or visual correctness qualification.
- Setup, including compilation and initialization, took 16.84 s at 128³ and 60.20 s at 256³. These one-off setup observations are excluded from stage results; cache state was not controlled.
- Repository-wide TypeScript checking reports existing errors in unrelated Sparse CM12 tests/tools; no errors remain in the new scene or profiler.
- No Sparse CM12 simulation, topology, publication, boundaries, or live editing implementation changed, so the large-change Sparse CM12 regression gate is outside this change’s scope.

## Reproduce

Run sequentially with no browser simulation or other Dawn process active:

```bash
node --import tsx tools/profile-uniform-geometric-dawn.ts --scene=cm12-figure-7-256 --frames=60 --out=docs/research/uniform-geometric-256-2026-09-22/figure7-256.json
node --import tsx tools/profile-uniform-geometric-dawn.ts --scene=cm12-figure-7 --frames=60 --out=docs/research/uniform-geometric-256-2026-09-22/figure7-128.json
```

Raw reports contain exact resolved parameters, full scene documents, per-frame GPU and CPU traces, pressure/page work counters, and window means/medians/p10/p90: [256³](figure7-256.json), [128³](figure7-128.json).
