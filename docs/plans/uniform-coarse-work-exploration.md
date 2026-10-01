# Uniform Geometric: coarse work and transport exploration

1 October 2026: disabled in production at Peter's request after review.
The smooth-surface UI toggle and runtime parameter have been removed, and saved
experimental settings are ignored. Dynamic coarsening keeps surface tiles fine.
The results below describe the earlier experiment, not the current default.

The target is approximately 2x end-to-end simulation throughput on
`sparse-cm12-ladder-long-dam`, with dynamic coarsening. Bit-exact trajectories
are not required. Peter requested periodic pauses for visual review and
suggested moving more work to 4h without large quality degradation.

## Properties to preserve

These are the experiment's comparison dimensions, not permission to weaken
existing tests. Where no independent physical oracle exists, the current
trajectory is a reference, not ground truth.

| Property | Evidence |
| --- | --- |
| Conservation | Canonical owner-weighted mass; account separately for sources, boundaries and explicit dust removal |
| Valid states | Finite fields, no material negative volume, accepted pressure receipts; report overfill rather than hide it by clipping |
| Bulk motion | Center of mass and 95/99/99.9% cumulative-mass fronts at matched physical times; these avoid a phi-only toe or one stray droplet defining the front |
| Impact and sheets | Front arrival, wall climb, return flow and survival of thin sheets, supported by field views and user review |
| Surface/mass agreement | Liquid in positive center-phi regions, alongside represented surface; conserved mass alone is insufficient |
| Resolution independence | h/4h seams and live refine/coarsen must conserve transfers; no unphysical impulses or blocked fronts |
| Boundaries | No new solid leakage; preserve sources, contacts and motion coupling in their dedicated lanes |
| Cost | Full advance including classification, remap, reconstruction and publication; compare identical durations and separate throughput from instrumented stages |

For initial screening, treat front progression and sheets as important and
small ripple differences as potentially acceptable, subject to Peter's review.
Do not invent a passing tolerance after seeing a result. Capture the deltas
first; any new perceptual acceptance policy should be agreed explicitly.

## Experiment 1: smooth surface ownership at 4h

The live classifier already estimates the error of replacing h surface
samples with 4h interpolation. Before this experiment production forced
`surfaceTolerance=0` because simulation ownership also chooses pressure/surface
resolution. The probe varies this shape tolerance, with `fastTravel=0`, retaining the impact
protection, timestep, pressure thresholds, sharpening and volume correction.

Compare 0, 0.125h and 0.5h on the same scene for 120 steps at 1/60 s. Save
canonical quality at step 1 and every 12 steps. This is a coupled-resolution
experiment, not proof that independent h detail over 4h flow works. Historical
overlay experiments lost front motion; do not silently revive those paths.

`tools/probe-uniform-stage-scaling-dawn.ts` accepts `--surface-tolerance` and
`--quality-every`. Quality copies happen after the measured interval. It reads
GPU ownership rather than the obsolete host mirror. `tools/uniform-quality-census.ts`
weights each owner by its physical cell volume and reconstructs diagnostic
phi slices from canonical corners, not stale fine texels inside coarse cells.

`tools/review-uniform-coarsening.ts` builds a local interactive comparison of
the captures. Its depth-mean liquid and center-plane phi are diagnostic views,
not a claim about production rendering quality. Rendering/GPU simulations must
not run concurrently with Dawn.

## Next experiments, chosen from the result

The reviewed 0.5 h arm completed 120 steps with the existing full pressure
allowance (4 V + 3 F cycles), still gated by convergence and with unchanged
pressure tolerance. The lagged two-slot plan failed at step 79 during impact.
The first production implementation used that full allowance. The pre-impact
owner/work reduction was substantial, but encoding the full allowance erased
the whole-run speed benefit.

A subsequent 120-frame A/B adds one reserve slot to the lagged planner (which
already has a spare), capped at the existing configured limits. This completes
the impact with three encoded/executed cycles and residual 3.067. Instrumented
GPU mean falls from 13.324 to 10.235 ms. Captured fields match through step 72;
the largest sampled 99%-mass-front difference is below 0.02 h and final center
of mass differs by less than 0.09 h. Positive surface tolerance now selects
this additional reserve; zero retains the original plan. A stalled V phase
stays dropped. The production reserve also completes 300 steps (five seconds).

Two alternating-order uninstrumented runs per arm, each with eight warmup
steps followed by 120 timed steps, give:

| Policy | Mean wall ms/step |
| --- | ---: |
| Fine surface, original lagged schedule | 11.831 |
| 0.5 h, full pressure allowance | 12.703 |
| 0.5 h, one additional reserve | 10.118 |

These use two frames in flight, no timestamps or quality readbacks in the
timed interval, and exclude rendering. The reserve takes 14.5% less time than
the original (1.17x throughput), or 20.4% less than the full allowance. This is
a measured improvement, not the requested 2x yet.

Compatibility limit: the authored `hero-garden-hose-x10` still-pond profile
retains surface tolerance zero in both UI and harness. Coarse partial surface
owners create a 0.285 m/s first-step hydrostatic impulse there even with the
full pressure solve. This is a real coarse-resolution balance limitation,
not a pressure-budget failure; the passing-at-HEAD still-water oracle and all
its tolerances remain. This profile exception leaves the measured long-dam
policy unchanged. Its probe now reads canonical owner volume and reconstructed
phi so stale fine backing texels cannot masquerade as physical state.

Further screening kept the same pressure tolerance and 120-step duration:

* Direct 4h velocity extension reduced that stage from about 1.17 to 0.44 ms,
  but the mass front fell more than 100 h behind during the run. Rejected.
* Four sharpening sweeps instead of eight cost 10.189 GPU ms versus the
  matched reserve capture's 10.362; sampled COM differed by up to 2.20 h.
  The small gain does not justify changing the default from this evidence.
* One surface-volume round instead of two cost 10.304 GPU ms, with COM
  differences up to 0.78 h. Kept two rounds.

Raw captures and the clean-HEAD failure audit are under
`artifacts/uniform-coarsening/uniform-next-*.json` and
`artifacts/uniform-coarsening/head-test-retirement.json` (generated, ignored).

The production regression checks 120 steps, accepted pressure at impact,
finite nonnegative fields, less than 0.5% mass loss, early mass-front travel
and substantial coarse ownership. Historical probe captures predate the
pressure reserve; use their source fingerprints and explicit policy flags
when interpreting the failed lagged-plan arms.

Validate broader scenes with the live control. If pressure or transport
changes materially, keep h authority where needed and investigate stage-specific
4h work rather than expanding the same ownership shortcut.

Shared-face transport remains the structural experiment: first compare its
conservation, full-cell capacity error, boundedness, diffusion and CFL work on
frozen states; then assess a coupled trajectory. Stable blocks and a compact
interface band are enabling data structures, not assumed speedups. Replacing
the geometric transport/correction chain requires its own quality evidence.

Continue pausing for visual review when proposing further policy changes.
Retained tests keep their assertions and timing ceilings. At Peter's explicit
request, tests also failing at clean HEAD `9022ae06` are removed after
reproduction in an isolated source checkout. Passing subtests are retained.

## Final validation

Type checking passes. Unit tests: 828 passed, 46 skipped. All 43 retained
Dawn files passed at the final source state across serial runs: the pond
first, then the other 42. The latter run had one native SIGSEGV in
`scene-shape-parity.test.ts`; its isolated retry passed all eight subtests.
The crash and retry logs are preserved, rather than reported as a clean
single full-suite run. No assertions or timing ceilings were relaxed.

Production browser verification: Dynamic and 0.500 h appear in Setup. The
long-dam ran past 13 simulated seconds with no console errors, then was reset
and paused at 2.15 s for visual review. Screenshot:
`artifacts/uniform-coarsening/production-coarse-ui.jpg`. Browser frame rate
is not part of the simulation-only throughput claim above.
