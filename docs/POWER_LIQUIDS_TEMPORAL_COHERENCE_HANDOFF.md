# Temporal coherence and aggressive per-advance work reduction — handoff

Date: 2026-07-29. Branch: `perf/structured-cutover`.
Scope: what the 2026-07-29 xctrace captures say about where the octree lane's
per-advance time goes, which temporal-coherence mechanisms already ship, which
are defeated or dormant, and a risk-ordered attack plan. All stage timings
below are xctrace hardware-counter attribution from the artifacts listed at
the end.

## TL;DR

The epoch machinery makes aggressive temporal coherence sound: fail-closed A/B
topology commit, GPU-resident change fingerprints, and an epoch validator that
already tolerates zero-page deltas as "valid GPU reuse". Two of the four
biggest wins are the *existing* machinery being defeated (double SPGrid
candidate build; air-support `existingReady` wired only to the failure path),
one is dispatch shaping (capacity-shaped launches), and only one — the JFA
closest-point carry — is genuinely new coherence with real soundness risk.
Do NOT attempt to add a pressure warm start: it already ships, unconditionally
(see "Already claimed" below).

Rough ceiling if everything lands: ~10–14 ms of the mini dam's 48.5 ms
attributed GPU per advance (on top of band 1's validated 5.1% clean win), and
proportionally more on ocean where JFA, air support, and V-cycle count
dominate.

## Implementation result (2026-07-29)

- The redundant accepted-mode SPGrid capture is removed. The prior tail's
  candidate is committed at the next head, and the next tail remains the sole
  producer of the following candidate.
- Changed-face velocity transfer now consumes a class-4 compact workset and an
  exact indirect dispatch. The list is mirrored into the already-bound
  candidate transaction so Metal remains at ten storage buffers.
- Recurring JFA-CPT is warm-started by default. Fine topology carries seeds by
  logical brick identity, rejects stale/recycled physical indices, refreshes
  subcell closest-point codes from current transported phi, and uses the
  displacement ladder `8,4,2,1,+1,+1`. A GPU-resident miss gate conditionally
  restores the other three cold collar repairs; no host readback or decision is
  involved. `FLUID_FINE_JFA_WARM_START=0` retains the cold oracle.
- The initial warm implementation exposed two regressions: sparse collar gaps
  caused topology rollbacks, and stale subcell codes caused reverse-depth
  crossings. The conditional repair gate and pre-seed code refresh fix those
  failure modes without making the cold schedule the product path.
- A 500-step warm run completed 500/500 advances with zero validation errors,
  zero rejected advances, zero topology rollbacks, and zero unresolved
  redistance cells. Its remaining late-time disconnection/ceiling failures are
  also present in the cold oracle. Paired 62-step runs measured 2362/2359 ms
  cold versus 2206/2204 ms warm (about 6.6% lower simulation wall time).
- Air-support site-2 deduplication remains intentionally unimplemented: the
  same-step dependency proof in `POWER_LIQUIDS_ULTIMATE_M1MAX.md` still blocks
  the proposed ordering. The pressure secant remains an existing opt-in rather
  than being rebuilt or silently enabled.

## Trace picture: two different regimes

### Mini dam (band sweep, `artifacts/xctrace-band-sweep-2026-07-29`)

Band 4 reference: 44.692 ms/advance clean (500-step control), 48.5 ms
attributed GPU across 255 stages in the fully label-isolated representative
advance.

- **~19.2 ms (40%) of GPU time runs below 10% compute occupancy.** V-cycle
  smoothers at levels 1–3: 0.0–0.7% occupancy, ~55 µs × 11 calls each —
  dispatch overhead over near-empty grids. "Transfer accepted velocity to
  changed topology faces": 1.88 ms at 0.2% occupancy. Hierarchy "candidate"
  stages: ~5.0 ms total at ≈0% occupancy.
- Genuinely busy work: redistance/JFA 12.4 ms (25%, ~50% occupancy, ~67% ALU),
  fine transport 4.7 ms, pressure/multigrid 17.5 ms (mostly low-occupancy).
- Band endpoints: band 0 (perf probe, not quality-valid) shows the fine-band
  ceiling — 23.0% faster clean, 43.4% less attributed GPU. Band 1 is the
  validated setting: 5.1% clean gain, all 500-step quality gates pass
  (`artifacts/band-0-vs-4-full-mini-2026-07-29/comparison.md`).

### Ocean seiche first frame (`artifacts/xctrace-ocean-first-frame-lean-20260729`)

6.7 s attributed GPU; occupancy healthy (50–90%) — this scene is work-bound,
not launch-bound:

- 2.4 s in V-cycle level-0 restrict/pre/post-smooth across 11 cycles.
- 1.2 s of JFA cooperative floods.
- 210 ms "Publish structured air-support identities", 155 ms structured
  extrapolation.
- 505 ms "Structured dynamics report advected kinetic energy" is a
  single-workgroup diagnostic reduction (`dispatchWorkgroups(1)`, gated by
  `projectionEnergyProbe`, `lib/webgpu-octree-structured-dynamics.ts:489`) —
  a capture-config artifact. Keep the probe off in profiled lanes, or give it
  a hierarchical reduction if it must stay on.

Consequence: mini needs *fewer, fatter dispatches*; ocean needs *less work
per advance*. Coherence addresses both, differently.

### Trace-mode caveat

Stage labels only exist under `FLUID_GPU_ISOLATE_PASS_LABELS` — production
`staged()` returns the already-open pass and discards the label
(`lib/webgpu-octree-spgrid-vcycle.ts:1521-1527`). Per-stage attribution is
valid; absolute totals differ from production because isolation adds pass
boundaries (traced 73.6 ms vs clean 44.7 ms on mini). The clean 500-step
controls are authoritative for shipping wall time.

## Already claimed — do not rebuild

- **Pressure warm start ships, unconditionally.**
  `lib/webgpu-octree-pipelined-mgpcg.ts:296-341` documents the chain
  (`mergeFrontierRows` → `rowDeltaNewToOld` → `emitLeaves` remap →
  `commitCandidateRows` epoch-gated copy → `r0 = -rhs - A·p_seed`) and warns
  that multiple sessions have tried to "add" it. The measured 4–5 executed CG
  iterations of an encoded 10 **is** the warm number.
  `FLUID_OCTREE_PRESSURE_COLD_START=1` is a measurement arm only.
- **SPGrid hierarchy rebuild is fingerprint-gated on the GPU.**
  `probeCandidateSkip`/`applyCandidateSkip`
  (`lib/webgpu-octree-spgrid-vcycle.ts:3073-3094`) retire per-level dirty
  flags when the 4-word-per-row fingerprint in `committedInputs` is unchanged;
  published fail-closed only after epoch acceptance. Level 0's stencil is
  always refreshed (free-surface fractions change without topology change).
  **This is the template for all new coherence**: GPU-resident fingerprint +
  fail-closed carry gated on the accepted-epoch commit token. Never a
  host-side skip.
- **Persistent MGPCG megakernel (Part D)** is implemented and measured 15 ms
  *slower*; deliberately not selected (`lib/webgpu-octree.ts:335-350`,
  `FLUID_OCTREE_PERSISTENT_MGPCG`). Do not resurrect it as an answer to the
  small-dispatch problem.
- The epoch validator already tolerates a zero-page SPGrid delta as valid
  reuse (`lib/webgpu-octree-topology-epoch.ts:243-247`) — the precedent for
  extending "unchanged ⇒ no work" to other families.

## Findings, ordered by risk-adjusted value

### 1. Measure topology-change frequency first (no code)

`FLUID_OCTREE_ROW_DELTA_CENSUS=1` (`lib/webgpu-octree.ts:3123-3171`) reports
per-generation `meanAdded/meanRetired/meanDirty/meanAffected` plus
`membershipChangedGenerations` and `zeroAffectedGenerations` — literally "how
often does topology change per advance". Run on mini and ocean. Every sizing
estimate below sharpens or collapses on this number.
`FLUID_WORKSET_CENSUS=1` adds fine-transport page and structured workgroup
censuses.

### 2. Double SPGrid candidate build (encode-path bug, ~2.5 ms mini)

The setup runs twice per advance:

- Tail "candidate" mode: `encodeCandidateSetup`
  (`lib/webgpu-octree-spgrid-vcycle.ts:1499`) from
  `lib/webgpu-octree.ts:3213`.
- Mid-solve "accepted" mode: `encodeSetup` (`:1605`) from
  `lib/webgpu-octree-section43-preconditioner.ts:302` via
  `lib/webgpu-octree-pipelined-mgpcg.ts:794`. Its skip at `:1606` requires
  `!preparedCaptureSource`, but `lib/webgpu-octree.ts:3318` calls
  `encodeCapture` every advance immediately before the solve, so **the skip
  never fires on the recurring path** and the full 21-phase build +
  `encodeReadySetupCommit` re-runs inside the solve.

On a clean fingerprint the accepted-mode build's only required output is the
level-0 stencil refresh. Verify first: the isolated trace should show each
`SPGrid V-cycle - candidate …` label twice per advance. Candidate stages total
5.05 ms in band 4, so up to ~2.5 ms is on the table. No soundness question.

Related known-slow kernel: `buildCandidateLevelDeltas` is one workgroup
looping all levels serially (`:3459`, dispatched `[1,1,1]` at `:1540`) —
0.625 ms at 0.5% occupancy; doc C6
(`docs/POWER_LIQUIDS_ULTIMATE_M1MAX.md:798-827`) targets it.

### 3. Capacity-shaped launches → compacted indirect (doc E4, Gate A)

"Transfer accepted velocity to changed topology faces"
(`lib/webgpu-octree-structured-dynamics.ts:566-576`, encoded from
`lib/webgpu-octree.ts:3207`) is already per-lane dirty-gated (marker check at
`:1311`), but launches `structuredSlotDispatch(slotCapacity)` — work scales
with changed faces, launch scales with capacity: 1.88 ms at 0.2% occupancy on
mini. Convert to the compacted-count indirect form
(`docs/POWER_LIQUIDS_ULTIMATE_M1MAX.md:1053-1064`). Same pattern applies to
the JFA support dispatches (halo radii saturate at mini domain sizes, so
"support" ≈ the whole resident band). No behavior change.

### 4. Air-support publication (~5 ms mini, ~365 ms ocean) — B2, sanctioned form only

`lib/webgpu-octree-air-velocity-support-gpu.ts` encodes all three heavy
stages (publish identities 1.94 ms, extrapolate/reconstruct 2.22 ms,
Section 5 march 0.85 ms on mini) in one `encode()` (`:524`) — and it is
encoded **twice per advance**: `lib/webgpu-octree.ts:3250`
(`encodeReadyTopologyFlip`) and `:3479` (`encode`). Identity/clear/directory
stages dispatch `ceil(domainVolume/256)` — full domain, not band.

The `existingReady` epoch predicate (`:869-882`) already compares layout
version, accepted epoch/bank, and boundary state, and already zeroes the
indirect args — **but only on the failure branch**; the healthy path always
rebuilds.

**Read `docs/POWER_LIQUIDS_ULTIMATE_M1MAX.md:529-615` (B2, "BLOCKED — DO NOT
ATTEMPT") before touching this.** The naive host-side dedupe is unsound: the
second encode is insurance against the *next* step's candidate rejection, and
the verdict lands at the tail of the same step. The sanctioned change is:
(1) move site 2 after candidate validation, (2) have the prepare stage zero
the indirect args when the GPU verdict is clean.

### 5. Temporal-secant pressure predictor — dormant, aimed at ocean

`FLUID_OCTREE_PRESSURE_TEMPORAL_PREDICTOR=1|current-operator` enables the
`p[n]−p[n−1]` row-remapped secant (`lib/webgpu-octree-pressure-history.ts:104-120`,
carried across epoch flips at `lib/webgpu-octree-topology-epoch.ts:289-293`;
α default 0.25 via `FLUID_OCTREE_PRESSURE_TEMPORAL_ALPHA`). Implemented,
default off. Ocean's dominant cost is 11 V-cycles of level-0 work per advance
— cycle *count* is the only lever there, and this is the mechanism built for
it. Trial on the ocean lane; measure executed iterations. Caveat from the
warm-start docs: the seed scales as 1/dt, so fixed-dt lanes only.

### 6. JFA closest-point warm start (Gate B, doc E1) — the real new coherence

Redistance is narrow-band (indirect off compacted support/dirty page lists)
but **cold every advance**: `seedClosestPoints` invalidates every support
sample and the flood re-reaches every band cell from the interface. The file
documents why (`lib/webgpu-octree-fine-levelset-redistance.ts:42-79`): CP
channels are not carried in the fine-page transaction, and a carried seed
naming a recycled physical page fails *open*.

The plan is Gate B (`docs/POWER_LIQUIDS_ULTIMATE_M1MAX.md:967-994`, E1):
carry CP channels through the page transaction with recycle safety; ladder
shrinks from 10 floods (16,8,4,2,1 + five stride-1 collar repairs) to 6.
Worth ~40% of the JFA family: ~5 ms mini, ~470 ms ocean. Build it on the
pressure-seed carry pattern (total row-identity map + fail-closed carry gated
on the accepted commit token). This is the highest-risk item; do it last,
after the census says how often the band actually moves.

### 7. Small-dispatch overhead on mini (not a coherence problem)

The ~10 ms of near-zero-occupancy smoother calls at levels 1–3 is launch
overhead over tiny grids. With Part D off the table, the options are fusing
coarse levels into one dispatch or truncating V-cycle depth for small scenes.
Lower priority than 2–4; revisit after them.

## Coherence coverage map (what the epoch gates today)

Gated fail-closed by `lib/webgpu-octree-topology-epoch.ts` (validate at tail
of N, flip at head of N+1): owner pages, power descriptor/topology,
structured velocity + topology transfer + candidate reconstruction, solid
vertices, structured boundary, SPGrid hierarchy, leaf headers, pressure seed,
pressure history.

Ignores the coupled epoch entirely: the whole fine level-set lane (transport,
fine topology, redistance, volume correction — own `globalFineGeneration`,
own A/B banks, `lib/webgpu-octree.ts:3710-3749`); air-velocity support (reads
the epoch to validate, never to skip); coarse phi correction (own generation
word). The fine transport worksets (classify → scan → publish → compact) are
rebuilt from scratch every advance with no unchanged-page carry
(`lib/webgpu-octree-fine-levelset-transport.ts:397-424`).

The conductor has the hook but no users yet:
`lib/physics-step-program.ts:133-137` — `predicates: Object.freeze([])`,
"P0.5 lands the mechanism, Part B … declares the predicates".

## Suggested order

1. Census runs (item 1) — one run per lane, sizes everything.
2. Double SPGrid build (item 2) — verify in trace, then fix; no gates.
3. E4 dispatch shaping (item 3) — Gate A, no behavior change.
4. Air-support B2 sanctioned reorder + GPU-side indirect zeroing (item 4).
5. Secant predictor trial on ocean (item 5) — flag flip + measurement.
6. JFA CP carry (item 6) — Gate B, design doc first.

Validation: mini dam 500-step clean control for wall time (44.692 ms/advance
band-4 baseline, 42.430 ms band 1); the full gate list from the band sweep
(topology, transport, redistance, pressure, volume, connectivity, raster,
impact) for quality. Ocean-seiche octree bringup state and its generation-3
frontier caveats are in the project memory note `ocean-seiche-octree-bringup`.

## Artifacts

- `artifacts/xctrace-band-sweep-2026-07-29/` — bands 0/1/4 mini captures:
  `comparison.md` (narrative), `comparison.json` (global/family/stage
  records), `stage-comparison.csv` (all 255 stages), per-band
  `summary.json` (per-pass counters: occupancy, ALU, LLC, bandwidth),
  `report.html`, `mini-dam.trace`.
- `artifacts/xctrace-ocean-first-frame-lean-20260729/` — ocean first-frame
  capture: `summary.json`, `report.html`, `mini-dam.trace`, raw ndjson.
- `artifacts/band-0-vs-4-full-mini-2026-07-29/` — band quality validation:
  `comparison.md` is the authoritative band-1-vs-4 quality table.
