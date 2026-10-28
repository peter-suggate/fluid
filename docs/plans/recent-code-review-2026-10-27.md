# Recent code review — 27 October 2026

Scope: the two latest code commits, `dc9e9602` and `af0e5dea`, plus the existing uncommitted solver and probe changes. This was a targeted review of correctness, diagnostics, resource lifetime and the new particle paths; it is not an exhaustive audit of the solver. Existing work was preserved.

Historical follow-up: extended the targeted review into October 23–26, comparing the older diffs with current code. Areas inspected included the repeat-spread gate, moving-window diagnostics, initial liquid/velocity construction, full-vector gravity, sculpt picking and ellipsoid bounds, adaptive AO/primary normals, and imported-surface rendering. Representative commits: `4416c17e`, `92379820`, `eabf093b`, `0d8c6a32`, `545d3a98`, `b7a715ef`, `0d42eff0`, `c8b25c39`, `079502c1`, `585e4e2d`, and `af7db171`. This is a selection of older changes, not a claim to have audited every intervening commit.

## Fixed

| Issue | Change | Regression coverage |
| --- | --- | --- |
| Comparison panes overwrote each other's in-flight startup history, because runs were keyed only by plugin. | Scope active runs by pane, keep completed timing records shared, and reject estimates for a different run/context. | `tests/progress-history.test.ts`: overlapping pane runs and estimates. |
| The allocation cap was enforced after creating the GPU resource; unsupported texture rules could also leave an allocated texture behind. | Compute requested bytes and validate the cap/rule before calling the device. Normalize iterable texture extents once. | `tests/uniform-gpu-allocation-audit.test.ts`: rejected calls never reach the device; destroy accounting; texture defaults and iterators. |
| Kernel profiling split passes that already owned timestamp queries, replaying the same query indices and replacing whole-pass timing with the last dispatch. | Preserve caller-timestamped passes intact and report them as untimed by the kernel profiler. | `tests/gpu-kernel-timer.test.ts`: multiple dispatches stay in the original pass. |
| Copy-only profiling encoders allocated unused query sets; a failed readback abandoned the rest of the pending resources. | Allocate query sets lazily and release the entire collected batch on success or failure. | `tests/gpu-kernel-timer.test.ts`: copy-only encoders and failed mapping. |
| The particle-mass probe treated a float stride as bytes, scanned stale bank tails and decoded APIC mass as FLIP bits. | Read complete records, use the receipt's current slab extent plus spill count, and decode each transfer format correctly. | `tests/particle-mass-readback.test.ts`: FLIP/APIC, dead slots, current spill, stale tail, and unit-mass mode. |
| The zone probe stored box-origin Y/Z over words 34/35, which were also written by the point-phi diagnostic. The race could turn a float's bit pattern into a coordinate. | Give the box origin its own three words, written by one invocation, and use those words for particle zoning. | Type check and source review; GPU execution of this probe remains to be verified. |
| The older repeat-spread gate could pass with NaN/Infinity measurements, JSON-null measurements, no shared frames, or an absent comparison arm. | Fail with a diagnostic for invalid measurements, incomplete instance lists, fewer than two repeats per arm, empty frame intersections, and overflowed window statistics. Keep valid lagged endpoints and the existing tolerance unchanged. | Five added regression cases in `tests/gate-repeat-spread-verdict.test.ts`. Missing `readStats().volumeDrift` explicitly becomes NaN in the capture tool, so this is a reachable failure path. |
| The moving-window zone probe wrapped the box origin modulo 64 pages (`0d8c6a32`), even though it needs a signed displacement. | Use `32 * (umWorldLow() - umFramePage())`, matching the signed runtime wall calculation before its clamp. | Arithmetic reproduction: a -1-page displacement was reported as +2016 cells instead of -32; +64 pages became zero instead of +2048. GPU execution remains unverified. |

## Remaining

### R1 — P2: mass-dependent separation does not search its full radius

Location: `lib/methods/uniform/uniform-narrow-band-flip.ts`, `nbSeparation` (around lines 1099–1113).

The search always visits the eight cells around `floor(q - 0.5)`, but the interaction radius is `0.25 * (cbrt(m_i) + cbrt(m_j))`, up to one cell. It therefore misses valid neighbours of merged samples, and the omission can be asymmetric.

Concrete counterexample with identical Y/Z coordinates: a mass-eight sample at X=1.45 and a unit sample at X=2.05 are 0.60 cells apart, inside their 0.75-cell radius. The heavy sample visits X cells 0/1 and misses the unit sample in cell 2. The unit sample visits cells 1/2 and receives a +0.1333-cell correction from the heavy sample. The pair's mass-weighted centre moves even though the intended paired corrections would cancel. This geometry was checked on the CPU; no claim is made about its frequency in a live scene.

Next: make the neighbourhood cover the mass-dependent radius symmetrically, then verify mixed-mass pairs across cell/tile boundaries and rerun film tests. Left unchanged because a wider gather changes both solver behavior and GPU cost.

### R2 — P2: dense cells still have quadratic particle work

Locations: `nbSeparation` and `nbMergeAbsorbed` in `lib/methods/uniform/uniform-narrow-band-flip.ts`.

The new canonical merge sort removes the old quadratic ranking, but every separated sample still scans its neighbouring runs. A cell with N noncoincident samples can require N*(N-1) neighbour visits. The merge path also repeatedly walks earlier ranks, and mass-saturated samples cannot be absorbed. The existing CPU test `tests/reservoir-particle-cost.test.ts` demonstrates the latter scaling. The new default `sampleSeparation: "pairs"` makes the neighbour scan relevant to normal runs.

Next: profile concentrated cells using clean frame timings and design a bounded spatial/pair processing path that preserves the particle multiset and separation behavior. A sample cap or early return would change the physics, so neither was applied as a quick fix.

### R3 — P1: re-establish the conservation evidence before accepting the new defaults

Locations: `docs/plans/nbflip-seep-volume-2026-10-27.md`, experiments 5–8; `tools/probe-tier-budget.ts`, `--particleMass`; `lib/methods/uniform/uniform-near-settled.ts`, new defaults.

The existing notes report 3.99 L in the right chamber versus 5.62 L delivered at step 1200, and identify cooling/retirement/reheating as an unresolved handoff. This is prior recorded evidence, not a run reproduced during this review. The particle-mass and particle-position arguments in those notes also need fresh measurements: outputs from the reviewed probe can be wrong because of the stride, stale-tail and coordinate-word bugs fixed above. The separately measured level-set volume is not automatically invalidated by the particle readback bug.

Next: rerun a bounded baseline and candidate with the corrected diagnostics; compare particle mass, independent level-set volume, throat flux and heat retirement through the late-time regime. Reassess conclusions such as particle/level-set mass agreement and the fraction of samples packed against the floor. Preserve old reports as historical evidence rather than silently replacing their numbers. No fade-time tuning or conservation correction was introduced here.

### R4 — P2: the repeat-spread gate can silently discard late evidence

Locations: `tools/gate-repeat-spread-verdict.ts`, common-frame intersection; `tools/gate-repeat-spread.ts`, lagged receipt collection and `--rescore`.

The verdict scores only frames present in every repeat and has no required horizon or coverage threshold. Rejecting an empty intersection fixes the zero-evidence case, but a single early shared frame can still pass while later differences are excluded. CPU reproduction: six repeats all have `bandTiles=100` at frame 1; at frame 70 the baseline is 100 and candidate 1000, but one baseline repeat lacks frame 70. The metric reports one window, zero deviation and a passing verdict. Other metrics were held equal. This was reproduced against the corrected scorer.

Next: pass the intended run horizon and metric sampling schedule into the verdict; report missing coverage and require late-window evidence with an explicit allowance for lagged GPU receipts. Saved-run rescoring needs the same contract. Left unchanged because arbitrarily requiring identical frame sets would reject legitimate lagged endpoints, while an invented minimum percentage would still allow gaps in the period under investigation.

## Validation

After the historical follow-up, `npm run check:types` passed. `npm run test:unit` passed: 1,253 passed, 132 skipped, zero failures. `git diff --check` passed. Thirteen added regression cases run on CPU/fake-device boundaries (eight in the initial pass, five in the follow-up). GPU-dependent tests are among the skips; no browser visual check, large GPU simulation or physical-conservation run was performed as part of this review.
