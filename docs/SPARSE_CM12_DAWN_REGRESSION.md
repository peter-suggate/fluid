# Sparse CM12 Dawn regression suite

Run the canonical post-refactor gate with:

```bash
npm run test:dawn:sparse-cm12
```

The suite is deliberately small and fail-closed. Each lane runs in its own
process so Dawn resources and the large Tall Cells diagnostic field are
released between scenes. The full run has a 180-second wall-clock budget. List
the exact executable matrix without touching the GPU with:

```bash
npm run test:dawn:sparse-cm12 -- --list
```

## Coverage matrix

| Lane | Authority | Baseline |
| --- | --- | --- |
| `symmetric-expansion` | D4 field/topology symmetry, corner residency, mass | Existing 20-step Dawn property gate |
| `hydrostatic-adaptivity` | Exact UI reset has a calm B4 surface pinned at its density-derived 15.25-cell waterline through step one; non-liquid halos stay out of structure view; stable deep-water 4/2/1 support | Reset plus two simulated seconds |
| `mini32-correctness` | Finite fields and liquid-volume retention | Four simulated seconds, at least 99.5% retained |
| `min8-region-surface` | Partial min8 reconstruction across a B2/B1 boundary | Reset planar top has at least 16 boundary samples, at most 0.05-cell detrended boundary bump, 0.125-cell neighbour step, and 0.02-cell mean curvature |
| `mini32-performance` | Production B8/P8 frame cost | 24.576 ms reference; 40 ms median ceiling |
| `mini64-performance` | Production B8/P8 frame cost | 33.4889 ms reference; 50 ms median ceiling |
| `mini64-min8-surface` | Production min8 presentation reconstruction | Seven paper steps; evolved top-sheet neighbour jump at most 12 fine cells |
| `long-dam-far-wall` | Sparse-world simulation and renderer publication | Material front reaches far-wall page 23 |
| `tall-cells-hills-far-wall` | Terrain cut-cell capacity and bounded mapping | Front reaches far-wall brick 30 |
| `live-rigid-body-coupling` | First rigid roster added to running water | Clock retained, finite motion, buoyancy ordering, mass retained |
| `live-liquid-injection` | Liquid ball added to a running scene | One world generation, added mass survives following step |
| `outside-tank-symmetric-collapse` | Floor-only open-world fluid | Horizontal spread aspect ratio no worse than 2:1 |

The performance gates use Dawn hardware timestamp queries: three warm-up frames
and twelve measured frames. The checked-in reference and ceilings live in
`benchmarks/results/sparse-cm12-dawn-regression-baselines.json`. They were
captured on Apple M1 Max with Metal and include roughly 50% noise headroom.
Update them only from a reviewed clean-tree capture on that reference machine;
never rebaseline in the same change that regresses performance.

## Focused use

Run a single lane or one half of the matrix while diagnosing:

```bash
npm run test:dawn:sparse-cm12 -- --lane=long-dam-far-wall
npm run test:dawn:sparse-cm12 -- --kind=performance
npm run test:dawn:sparse-cm12 -- --kind=correctness
```

These selections are diagnostic conveniences. A large Sparse CM12 change is
accepted only by the unfiltered full command.
