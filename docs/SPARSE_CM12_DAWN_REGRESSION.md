# Sparse CM12 Dawn regression suite

Run the canonical post-refactor gate with:

```bash
npm run test:dawn:sparse-cm12
```

The suite is deliberately small and fail-closed. Each lane runs in its own
process so Dawn resources and the large Tall Cells diagnostic field are
released between scenes. The full run has a 480-second wall-clock budget. List
the exact executable matrix without touching the GPU with:

```bash
npm run test:dawn:sparse-cm12 -- --list
```

## Coverage matrix

| Lane | Authority | Baseline |
| --- | --- | --- |
| `simulation-failure-halt` | Production incidence validator, deficit-support validator and entry guard | Healthy work executes; corrupt input latches first provenance and blocks later stages and frames; empty air may leave resident support but any nonzero donor mass still faults |
| `symmetric-expansion` | Bounded D4 field and topology errors, corner residency, mass | Eight steps; density error ≤0.55, velocity error ≤2.1 m/s, pressure error ≤1400 Pa; topology-resolution mismatch ≤4; complete corner orbit on step one |
| `mixed-ratio-topology` | BTI1 GPU cell, point-owner, and row services plus the BFP1 CPU partition at 8\|2, 8\|1, and 8/4/2/1 | GPU results match the exhaustive CPU mirrors; BFP1 covers every row exactly once |
| `topology-page-budget` | Authored re-rung and WDR page ownership remain independent | Zero, one and 32 growth pages; conservative rung transfers, no borrowed dynamic identities, matching UI/QA allocator receipts |
| `clipped-topology-transfer` | Compact clipped cells survive live coarsening and refinement | 13×10×9 domain, live B1/B8 edits under production defaults; mass, gamma, momentum and transaction receipts |
| `topology-generation-storage` | GPU topology generation storage remains bounded and lease-safe | Stage/cancel/commit, retained old consumers, allocation rollback, stale requests, deferred retry after reclamation; storage component only, not resident adoption |
| `hydrostatic-adaptivity` | B1 surface, halo classification, and bounded deep refinement under production defaults | Two seconds; no deep B4/B8 samples; reset height error ≤0.4 cell and first-step column motion ≤0.04 cell |
| `mini32-correctness` | Production defaults, per-step failure receipts and liquid-volume retention | Four simulated seconds, at least 99.5% retained; includes the coarse-first frame-33 empty-air deficit regression |
| `min8-region-surface` | Authored B2/B1 region boundary with default coarse-first reconstruction | At 1.6 s: left/right height split ≤0.01 cell, mean height drift ≤0.001 cell, published column motion ≤0.02 cell, boundary bump ≤0.11 cell |
| `mini32-performance` | Production B8/P8 frame cost | 24.576 ms reference; 40 ms median ceiling |
| `mini64-performance` | Production B8/P8 frame cost | 83.5584 ms reference; 110 ms median ceiling |
| `mini64-min8-surface` | Production min8 presentation reconstruction | Seven paper steps; evolved top-sheet neighbour jump at most 12 fine cells |
| `long-dam-far-wall` | Sparse-world simulation and renderer publication | Material front reaches far-wall page 23 |
| `tall-cells-hills-far-wall` | Terrain cut-cell capacity and bounded mapping | At three seconds, front ≥brick 24 and allocation failures ≤2200; old brick-30 arrival remains a historical target |
| `live-rigid-body-coupling` | First rigid roster added to running water | Clock retained, finite motion, buoyancy ordering, mass retained |
| `live-liquid-injection` | Liquid ball added to a running scene | One world generation, added mass survives following step |
| `outside-tank-symmetric-collapse` | Floor-only open-world fluid | Horizontal spread aspect ratio no worse than 2:1 |

The performance gates use Dawn hardware timestamp queries: three warm-up frames
and twelve measured frames. The checked-in reference and ceilings live in
`benchmarks/results/sparse-cm12-dawn-regression-baselines.json`. They were
captured on Apple M1 Max with Metal; the September 2026 mini64 refresh includes
about 32% headroom.
Update them only from a reviewed clean-tree capture on that reference machine;
never silently rebaseline in the same change that regresses performance. The
explicitly requested September 11 working-tree refresh below is an approved
exception to the clean-tree policy.

## Focused use

Run a single lane or one half of the matrix while diagnosing:

```bash
npm run test:dawn:sparse-cm12 -- --lane=long-dam-far-wall
npm run test:dawn:sparse-cm12 -- --kind=performance
npm run test:dawn:sparse-cm12 -- --kind=correctness
```

These selections are diagnostic conveniences. A large Sparse CM12 change is
accepted only by the unfiltered full command.

Every simulation lane resolves the balanced adaptive-mass production defaults,
including coarse-first selection, the paper timestep, gamma diffusion,
sharpening, and the pressure policy. Scene method profiles do not override
these defaults. The performance probe uses those same defaults and the checked-in B8/P8
references and timing ceilings.

Authored geometry, refinement regions, live edits, and page-budget fixtures
remain test inputs. Transfer lanes use authored rung edits instead of forcing
QA initial resolutions or accelerating the coarsening policy. The partial-region
surface lane authors a B2/B1 boundary so it still exercises mixed-resolution
reconstruction with coarse-first enabled. Kernel and topology-storage unit
lanes have no simulation selector and continue to test their explicit inputs.

Run `npm run test:dawn:sparse-cm12:coarse-first` for the additional default-policy
still pool, elevated-ball impact and moving-surface coarsening checks.

Run `npm run test:dawn:sparse-cm12:trace-gravity` for dilute donor momentum
retention and the coarse-first half-pool drop's upper-air residue regression.
See [reproduction and cause](sparse-cm12-trace-gravity-2026-09-11.md).

## Accepted behavior baseline — 2026-09-11

The user explicitly requested moving the baseline to the current working tree
so existing deviations do not repeatedly block unrelated work. The measured
limits live in `benchmarks/results/sparse-cm12-dawn-behavior-baseline.json`,
with repeated observations, source fingerprint, and the previous ideal targets.
Passing now means staying within those accepted errors; it does not mean the
original symmetry and hydrostatic targets have been restored. New nonfinite
metrics and values above the recorded ceilings still fail.

The wall budget is now 480 seconds (previously 180). Mini64 has 60 seconds for
construction and sampling (previously 30), and the hydrostatic lane has 45
seconds (previously 30). Mini32 retains its 40 ms frame-time ceiling; mini64 is rebaselined from 50 to 110 ms against a measured 83.5584 ms median. The performance
probe checks GPU counter succession within each resident generation and also
checks the global simulation step, because resident replacement restarts its
local counters. It continues to reject actual GPU faults and stalls.

After visual review, the positive-density velocity-seeding experiment was
withdrawn. The accepted configuration again uses liquid-only velocity
extension (rho > 0.5). The baseline below that configuration restores the
original surface-drift checks and deep-water coarsening checks. Only the
pre-existing boundary ridge has a measured 0.11-cell allowance. The hillside
window is restored to 90 paper steps. Withdrawn measurements are retained as
history, not as acceptance criteria for future algorithm changes.

The restored bounded hillside configuration also shows failed page-allocation
attempts (1537–1768 in focused captures) and reaches brick 26 at three seconds.
Its accepted baseline is at most 2200 attempts and a front at least at brick
24. This is an explicit exception to the former zero-capacity-failure and
brick-30-arrival targets. Frame fault flags, host incidences, finite density,
and terrain-solid exclusion remain hard checks.
