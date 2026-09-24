# Uniform Geometric splash retirement

Figure 9 (`128 × 128 × 64`), production defaults, Dawn/Metal, 180 steps at
1/30 s. The primary cause of post-splash work retention was positive phi with
no volume, not failure of the ordinary dust floor. At five seconds the baseline
had 7,462 upper-half phi-only seed tiles and 16,350/16,384 fine tiles.

The ghost drain stops at +h/2. Newton redistancing can leave that plateau
unchanged when it cannot find a contour, while the work census seeds phi <4h.
The fix checks the entire 4h vertex neighbourhood after an unsuccessful
positive-phi Newton search. If every vertex is positive, no trilinear zero
crossing can exist there and phi can be set to the band edge. Existing surfaces,
negative phi and buried vertices retain their existing handling. Nonzero V
continues to seed work independently. Disabling ghost draining also disables
this retirement; disabling redistance prevents the retirement pass from running.

The additional **Orphan dust floor** defaults to 0.01 cell volumes (ordinary
dust remains 0.001). It runs once after transport and before global surface
correction. It removes only positive, fully open cells below the floor whose
3³ neighbourhood is outside the 4h phi band, contains no cell with V ≥0.05,
and holds less than 0.25 cell volumes in total. It protects resolved surfaces,
compact droplets, cut cells and resting-pool tails. It also reaches dilute
wall-adjacent residue which cannot qualify for airborne momentum. Zero disables
the extra floor; disabling ordinary dust disables it too. It is exposed only
on the WebGPU parameter contract, not the native/WASM backend.

Both floors discard mass. Global surface correction matches phi to current V;
it does **not** replenish discarded V. Separate orphan counters use the orphan
threshold's units to avoid overflow when the ordinary threshold is tiny. The
existing aggregate dust readout includes both floors. Counts/mass are per step;
mass counters round down in 1/64-threshold units, so the census measures exact
cumulative mass loss directly from V.

| At frame 180 (6 s) | Before | Phi retirement | Phi + orphan dust |
|---|---:|---:|---:|
| Upper phi-only seed tiles | 7,063 | 210 | 270 |
| Fine tiles | 16,372 | 8,521 | 8,936 |
| Total V, cell volumes | 335,045.44 | 335,124.78 | 334,603.91 |
| Median step, preceding 30 steps | 43.69 ms | 30.65 ms | 33.33 ms |

Initial V was 335,902. The combined implementation loses 0.386% over six
seconds, versus 0.255% before and 0.231% with phi retirement alone. The extra
floor therefore costs approximately 0.155% of initial V relative to phi-only.
Most speed benefit comes from phi retirement; stronger dust primarily removes
the dilute haze and is adjustable independently. The trajectories diverge
through rebound, so these are scene-level comparisons, not bitwise equivalence
claims. Timings are single-run queue-fenced medians, not renderer FPS or isolated
kernel timestamps. See the adjacent JSON for all checkpoints.

An extended 360-step run retained the improvement at 12 seconds: 125 upper
phi-only tiles, 7,981 fine tiles and 31.87 ms median over the preceding 60 steps.
Cumulative mass loss was 0.469%, with no GPU validation errors.

This contracts work tiles and active scratch pages. The physical domain page
catalogue remains all-resident; this change does not implement memory eviction.

## Reproduction and validation

Run sequentially, with no browser GPU run active. Each probe owns the repository
WebGPU lease:

```sh
FLUID_UNIFORM_AB_OFF=phiretire node --import tsx tools/probe-uniform-splash-retirement-dawn.ts --frames=180 --values='{"orphanDustThreshold":0}' --out=/tmp/retirement-before.json
node --import tsx tools/probe-uniform-splash-retirement-dawn.ts --frames=180 --values='{"orphanDustThreshold":0}' --out=/tmp/retirement-phi.json
node --import tsx tools/probe-uniform-splash-retirement-dawn.ts --frames=180 --verify --out=/tmp/retirement-after.json
```

`--verify` budgets less than 0.5% total mass loss through the six-second rebound,
fewer than 1,000 upper phi-only tiles and fewer than 11,500 fine tiles at frames
150 and 180. The old phi-retention path fails the work budgets.
The combined acceptance run is also available as
`npm run test:dawn:uniform-splash-retirement`.

The airborne-momentum Dawn suite includes empty-plateau retirement, preservation
of nearby zero crossings, remote and wall-adjacent dust removal, compact droplet
protection, unchanged resting-surface tails and orphan mass diagnostics. Existing
free-flight, gravity and pressure-support checks remain in the same suite.

Validation also includes the six-second Figure 9 acceptance probe, the geometric
boundary suite (including Figures 8 and 12), the surface-correction equivalence
suite, page-backed/dense equivalence through drops and moved inflow, and 13 CPU
page/layout tests. The page-equivalence test had a stale forbidden
`pressureCycleBudget` method override; it now retains that fixed budget only
in the explicit solver QA options. No thresholds were relaxed.

The existing symmetric-expansion suite is not green on the control either:
airborne-off fails at frame 17 with mean volume symmetry error
0.00011976467874319496 in both the baseline and changed code. Airborne-on fails
the frame-24 0.0001 budget in both (baseline 0.00022830487142755373; changed
0.00022830746641488986). The baseline also reaches peak cell V=4.2617 by frame
90, beyond that suite's bound of 4. Those unrelated failures remain visible.
Repository-wide TypeScript checking likewise reports the same pre-existing
Sparse CM12 diagnostics at the same locations; no new diagnostic locations
were introduced by this change.
