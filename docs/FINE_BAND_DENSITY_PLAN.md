# Fine-band density remediation plan

## The finding

At both production lane resolutions the fine level-set "narrow band" is not
narrow. `planGlobalFineNarrowBandBrickCapacity` (`lib/webgpu-octree.ts:557`)
resolves to **100% of the logical brick lattice**:

| lane | grid | brickDims | logical bricks | capacity | occupancy | samples |
| --- | --- | --- | --- | --- | --- | --- |
| mini / moving-interface | 16³, band 3, factor 4 | 16³ | 4096 | 4096 | **100%** | **262,144** |
| UI | 24×18×16, band 4, factor 4 | 24×18×16 | 6912 | 6912 | **100%** | 442,368 |
| (32³, band 4) | — | 32³ | 32768 | 19200 | 59% | 1,228,800 |
| (64³, band 4) | — | 64³ | 262144 | 76800 | 29% | 4,915,200 |

Derivation for the mini lane (`lib/webgpu-octree.ts:1440-1462`):

```
transportBandFineCells  = interfaceRefinementBandCells × fineFactor = 3 × 4 = 12
redistanceBandFineCells = 12 + fineFactor + 3                              = 19
requiredFineCells       = max(backtrace 4 + interpolation 1, 19)           = 19
dilationBrickRings      = ceil(19 / 4) + 1 safety                          = 6
bandLayers              = 2 × 6 + 1                                        = 13
bandBrickCount          = min(4096, maxFaceArea 256 × 13 = 3328)           = 3328
capacity                = ceil(3328 × 1.25) = 4160 → clamped to logical    = 4096
```

Six brick rings is 24 fine cells of dilation on a lattice 64 fine cells across.
The area×width model in the docstring at `:548-556` is asymptotically correct,
but at production sizes the band radius exceeds the domain half-width, so it
materializes the cubic lattice exactly.

## What the paper actually specifies

From `docs/papers/aanjaneya-2017-power-liquids.txt`:

1. **Page allocation is interface blocks plus one ring.** "For efficiency, we
   first allocate all pages corresponding to blocks that either contain the
   interface, or lie in the 1-ring of a block that contains the interface.
   After this step, cells that lie within the narrow band can be safely marked
   in parallel without any data hazards." (§5, Dynamic topology.) We use six
   rings, and we mark them from a single workgroup.

2. **The band width constraint is per-step interface displacement, nothing
   else.** "Of course, we maintain a wide enough band so that the free surface
   still falls inside this band after advection." (§5, Level set advection.)
   The time step is set by the finest effective octree resolution and the
   backtrace is `m` forward-Euler substeps of `Δt/m` with `m = fineFactor`, so
   the displacement bound in fine cells is `m` at CFL ≈ 1 — that is
   `maximumBacktraceFineCells`, which we already compute correctly. It is not
   `interfaceRefinementBandCells × fineFactor`, and it is not that plus
   `fineFactor + 3`.

3. **Four channels.** "we use only 4 channels for interface tracking" —
   1 flags, 1 level set, 2 for fast marching (§5, and §2 on SPGrid). We hold
   ten channel-equivalents at capacity: `flags/phi/workA/workB/rollbackPhi`
   across two generations (`webgpu-octree-fine-levelset-bricks.ts:93`), plus
   two `transportedPhiSnapshot` arenas (`fine-levelset-topology.ts:556`).

4. **Target occupancy.** The paper's fine level set is 543M active voxels at
   2048³ (8.09 GB, Fig. 2) — 8.6G logical, so **6.3% occupancy**. Ours is 100%.

5. **The fine band is expected to dominate the step even when done right.**
   Table 2, Fig. 2 column: level-set advection 26s + reinitialization 20s of an
   80s step = 57%. This is why the constant factor matters: there is no version
   of this pipeline where the fine band is cheap enough to ignore.

## Measured state

`benchmark:power-dam-moving-interface` (16³, band 3, factor 4, 62 steps),
clean tree, no concurrent GPU session: **230.02 ms/advance**, 708 dispatches,
120 compute passes per advance.

Per-stage attribution was not obtainable. `--profile` SIGSEGVs under Dawn
`skip_validation`, and `test:webgpu:minimal-power-dam-two-step` dies at
`tools/run-webgpu-smoke.ts:357` (`copyBufferToBuffer` given a non-object
binding) before printing diagnostics. Both look like harness faults rather than
solver faults. This is why P0 below is blocking.

---

## P0 — make the band measurable (blocking, no numerical change)

> **Status 2026-07-26: items 1–3 landed and verified; item 4 partially captured.**
> 67 CPU tests green, `npx tsc --noEmit` clean, `test:power-liquids-structure`
> 32/32.
>
> **The occupancy number now exists, and it confirms the diagnosis: 98.2% —
> 4,024 of 4,096 logical bricks active on the mini lane** (`fine band occupancy:
> 98.2% (4024 active / 4096 logical bricks, capacity 4096)`). Predicted 100%,
> measured 98.2%. Before the P0.1 fix this field printed the generation counter.
>
> The wall-time half of the baseline is **not** trustworthy yet and is
> deliberately not recorded. Two samples disagree by 4.7× (230.02 → 1077.19
> ms/advance) with pressure iterations 0/7 → 7/7, and the working tree was being
> edited during the second. Three subsequent runs crashed at construction. See
> "Baseline capture is blocked" below.

Nothing downstream can be ranked or attributed until these land.

1. **Fix the active-brick counter.** `lib/webgpu-uniform-eulerian.ts:751` reads
   `value.worklistHeader[0]`, which is the *generation*; the count is word 1
   (`lib/octree-fine-levelset-bricks.ts:499`). A 62-step run currently prints
   "64 active fine bricks" because it is printing generation 64. This single
   wrong word is why band density has never shown up in a benchmark.
2. **Publish fine occupancy per advance** — `activeBricks / logicalBricks`,
   plus `dirtyPages` and `supportPages` from `pageDelta` — through
   `lib/webgpu-octree-work-accounting.ts` as scheduled/active lanes for the five
   fine stages (topology, transport, redistance, volume, restriction). This also
   unblocks the `activeScheduledRatio` artifact blocker recorded in
   `WEBGPU_OCTREE_M1_MAX_IMPLEMENTATION_PLAN.md` item 1.
3. **Widen the source guard.** `SHADER_SOURCE_CHECKS`
   (`lib/webgpu-octree-work-accounting.ts:1010-1019`) does carry a WGSL
   `loop {` matcher, but it only fires when `lookup`, `search`, or `probe`
   appears within 512 characters of the loop opening. `publicationRow`
   (`webgpu-octree-fine-levelset-transport.wgsl.ts:116`) wraps a nested binary
   search using none of those words, so it passes. Nothing catches a
   bounded-but-quadratic scan such as `changedNeighborRadii`. Add a structural
   matcher for a `while`/`loop` nested inside a loop over a capacity-derived
   bound.

**Exit:** mini and UI lanes report a true occupancy ratio; a committed baseline
exists; the guard fires on the two known offenders.

### What the new guard surfaced

`npm run audit:octree-production-source` went from *60 sources clean* to five
violations. Two are the predicted targets; three are pre-existing debt this
plan does not own. The corpus tool now exits non-zero. It is not part of
`test:power-liquids-structure` and the repository audit test asserts only that
violations belong to discovered sources, so no acceptance gate broke.

| site | rule | owner |
| --- | --- | --- |
| `fine-levelset-topology.ts:1574` `changedNeighborRadii` | `capacity-scan` | **P1c fixes this** |
| `fine-levelset-transport.wgsl.ts:116` `publicationRow` (outer `loop`) | `capacity-scan` | not this plan — owner-row directory probe |
| `fine-levelset-transport.wgsl.ts:116` `publicationRow` (inner search) | `nested-capacity-scan` | not this plan |
| `fine-levelset-topology.ts:1349` `desiredContains` | `capacity-scan` | cold-bootstrap expansion only |
| `fine-levelset-topology.ts:1131` `externalSeedTaggedValue` | `capacity-scan` | cold-seed insertion only |

The three non-target sites are binary searches, which the M1 Max plan's Phase 1
exit gate already forbids in hot loops ("zero binary/hash probes in regular
fine-brick and pressure-page hot loops"). `publicationRow` runs once per brick
rather than once per sample, so it is bounded but still a directory probe on a
recurring path. They are recorded here rather than suppressed: the audit module
deliberately does not support inline suppressions.

### Baseline capture is blocked by tree instability, not by the GPU lock

The lock was free and the benchmark path already acquires it correctly through
`acquireWebGPUExclusiveLock`. Four attempts produced one result and three
construction crashes:

```
TypeError: no overload matched for createBindGroup
  ... while converting member 'resource' for array element 2
  at WebGPUOctreeAirVelocitySupportProducer (webgpu-octree-air-velocity-support-gpu.ts:356)
```

`OCTREE_AIR_SUPPORT_GPU_ENTRY_BINDINGS.markFineBandAirSupportDemand` declares
bindings `[0,7,25,26,27,28]` unconditionally (`:82`), but the `buffers` map
supplies 25–28 only when `inputs.fineSources` is present (`:354`,
`...(inputs.fineSources ? [...] : [])`). Without it, `buffers.get(25)!` yields
`undefined` — array element 2 of that entry list, exactly as reported. The file
carries +113 uncommitted lines and was modified seconds before the failing run,
so this is in-flight work rather than a standing defect.

Two consequences for anyone capturing a baseline:

1. **Do not benchmark a tree that is being edited.** The 230 → 1077 ms/advance
   spread across two samples cannot be attributed to anything while the sources
   move between runs. The documented post-stall regime
   (`POWER_LIQUIDS_PERF_HANDOFF.md`) is a second, independent reason a single
   sample is not a baseline: it reports 30–100× nondeterministic spread once the
   coarse topology publication stalls, and the 1077 ms sample showed the
   matching signature.
2. **A hard crash poisons every later run.** `run-webgpu-exclusive.ts` releases
   the lock in a `finally`, but a worker that dies on a native fault never
   reaches it, leaving `/tmp/fluid-webgpu-exclusive.lock` behind so the next
   invocation fails with `EEXIST`. Recovery is to confirm the recorded owner PID
   is gone and then call `releaseWebGPUExclusiveLock()`. A stale-lock reaper that
   checks liveness of the recorded PID would remove this failure mode.

---

## P1 — delete dead and serialized work (immediate wall-clock)

These are the wins available at *current* lane sizes. P1b–P1d are pure
refactors: same values, same order, fewer lanes. **P1a is not** — it changes
far-field phi and needs its own parity gate. Land it separately from the
refactors so an IoU or volume regression is attributable.

### P1a. Implement or delete the dead transport band gate

`transportBandCells` is plumbed from `webgpu-octree.ts:3001` through
`fine-levelset-transport.ts:287` into the uniform, and the shader declares
`bandCells:u32` at `transport.wgsl.ts:12` — **and never reads it**. Every valid
sample in every active page is traced, each with `m = fineFactor` velocity
reinterpolations. At 16³ that is 262,144 traces and ~1M velocity samples per
advance.

Implement the gate: samples with `|phi| > bandCells × h` copy forward unchanged
instead of tracing. Paper basis — the narrow band only requires valid signed
distance inside the band; outside it phi carries sign, not distance, and the
coarse octree level set is the authority there (§5, "we use the fine level set
to correct the coarse level set wherever we have valid ϕ-values").

### P1b. Parallelize the band dilation

`publishRecurringSparseBand` (`fine-levelset-topology.ts:1270`) is dispatched
`dispatchWorkgroups(1, 1)` at `:849-850` — one workgroup, 256 lanes — and each
lane runs `recurringScatterMembership` (`:1255`), a `(2r+1)³ = 13³ = 2197`
iteration `atomicOr` loop per interface seed brick.

The mask is already a dense logical-key array and `atomicOr` is already
idempotent, which is exactly the paper's "can be safely marked in parallel
without any data hazards". Dispatch one workgroup per interface seed with the
`(2r+1)³` stencil spread across lanes.

### P1c. Replace the quadratic support classifier

`changedNeighborRadii` (`fine-levelset-topology.ts:1570-1583`) is an
O(desired × changed) linear scan — up to 4096 × 8192 = 33.5M iterations per
advance, with an early-out that only fires for pages near a change.

Invert it: scatter each changed key's dirty/support Chebyshev halo into the same
dense mask P1b already uses, then each desired page reads one word. Cost becomes
O(changed × r³) and shares P1b's kernel shape.

### P1d. Remove the single-lane sweeps from transport

Three `@compute @workgroup_size(1)` kernels serially walk the page arena every
advance:

| kernel | file:line | serial iterations |
| --- | --- | --- |
| `publishStructuredFineTransportWorksets` | `transport.wgsl.ts:123` | 2 × live ≤ 8192 |
| `summarizeStructuredFineTransport` | `transport.wgsl.ts:194` | live ≤ 4096 |
| `publishStructuredFineDelta` | `transport.wgsl.ts:198` | `8 + 2×pageCapacity` = 8200, **capacity-shaped**, plus live |

~20k dependent single-lane iterations per advance. Replace with mark → prefix
rank → scatter, the pattern already implemented in this repo's identity chain
(`fine-levelset-topology.ts:876-890`). Make `publishStructuredFineDelta`
active-shaped rather than capacity-shaped.

**Exit:** no single-lane loop over `pageCapacity` remains in the recurring fine
path; no scan is O(pages × changes); wall-time delta attributed on mini and UI.

---

## P2 — restore the paper's band width (scaling, carries regression risk)

### P2a. Give the fine surface band its own control

`lib/methods/octree.ts:9` already documents `interfaceRefinementBandCells` to
the user as *"distinct from the separate high-resolution surface-tracking
band"* — but `webgpu-octree.ts:1440-1441` and `:3001` derive the surface band
from it. Introduce `globalFineSurfaceBandCells`, in fine cells, defaulting to
`fineFactor + 1` (the paper's displacement bound plus one interpolation cell).
`interfaceRefinementBandCells` then governs octree pressure refinement only,
matching its documented meaning.

### P2b. Stop stacking radii

`redistanceBandFineCells = transportBand + fineFactor + 3` (`:1452`) sums three
alternative radii into one. `planFineLevelSetTopologyBand` (`:130`) already
takes the max of `backtrace + interpolation` against the redistance band —
which is correct — so once P2a supplies the band directly, the `+ fineFactor + 3`
is double-counting and goes away.

### P2c. Halve the JFA support halo

`supportHaloRings = 2 × dirtyHaloRings` (`:694`). JFA needs a seed-and-landing
halo of one band width, not two. Set them equal and let the fail-closed
`unresolved` count in the redistance control prove it.

### P2d. Make the band adaptive and fail closed

`control.maxDisplacement` is already published by
`summarizeStructuredFineTransport` (`transport.wgsl.ts:194`) and
`control.outside` already counts departures leaving the band. Size the next
epoch's band from measured displacement plus hysteresis, and reject the epoch
when `outside > 0`. This converts "is the band wide enough" from a static
guess into a measured, self-correcting invariant — which is what makes it safe
to narrow it.

### Projected effect

| lane | capacity now | capacity after | JFA passes | support halo |
| --- | --- | --- | --- | --- |
| 16³ band 3 | 4096 (100%) | 2240 (**55%**) | 7 → 5 | 10 → 2 rings |
| 24×18×16 band 4 | 6912 (100%) | 3780 (**55%**) | 7 → 5 | 12 → 2 rings |
| 32³ | 19200 (59%) | 8960 (**27%**) | 7 → 5 | 12 → 2 rings |
| 64³ | 76800 (29%) | 35840 (**14%**) | 7 → 5 | 12 → 2 rings |

**Exit:** occupancy at or below the table; zero-crossing hash, volume, and IoU
parity unchanged on mini and UI; `departureOutsideBand` and redistance
`unresolved` remain zero across the 500-step mini lane.

**Risk:** a band that is too narrow lets the surface leave it. P2d is the
mitigation and must land in the same change, not after it. This phase lands
after P0 baselines exist so a regression is attributable.

---

## P3 — active-shaped dispatch and the channel diet

1. **Indirect the identity chain.** `classifyIdentity`, `compactIdentity`, and
   `assignIdentity` (`fine-levelset-topology.ts:876-889`) dispatch at
   `ceil(capacity/64)`. Drive them from the published desired count instead.
2. **Cut capacity-shaped scratch.** `rollbackPhi` (2 × capacity × 64 × 4 B) and
   `transportedPhiSnapshot` (2 × capacity × 64 × 4 B) are ours, not the paper's
   four channels. Both are transaction state whose scope is the dirty page list,
   which `pageDelta` already carries — shape them to it. At 16³ that is ~3 MB of
   the ~11.5 MB fine allocation.
3. **Record the dense logical directory as a deliberate choice.** Each
   generation carries a `logicalBrickCount`-word logical→physical directory
   (`fine-levelset-bricks.ts:90-92`) and the recurring scan/scatter runs
   `ceil(logicalBrickCount/256)` blocks (`fine-levelset-topology.ts:851`). This
   is the plan document's own sanctioned "bounded logical bitset, prefix rank"
   trade — it buys an O(1) probe in every fine interpolation tap. It is correct
   at ≤64³ brick domains and becomes the binding constraint above that. Record
   it rather than change it now.

**Exit:** allocated fine bytes track active pages; bytes moved per active sample
recorded for transport, redistance, and restriction.

---

## P4 — prove it at a size where a band is a band

16³ and 24×18×16 cannot demonstrate compact-band scaling: even at three brick
rings the band covers 55% of a 16-brick-wide domain. Every conclusion about
sparsity drawn from those lanes is measuring a dense grid.

Add a 64³ lane (14% occupancy under P2) as the scaling gate. This is the same
argument the M1 Max plan makes for the Section 6.3 operator work — that a lane
large enough to make the difference measurable must be added alongside it, or
the work is done with no way to prove it paid.

---

## Ordering and honest expectations

**P0 → P1 → P2 → P3, with P4 added alongside P2.**

- **P1 is the near-term wall-clock win** on current lanes and carries no
  numerical risk: one ungated per-sample stage, one single-workgroup dilation,
  one 33.5M-iteration scan, and ~20k dependent single-lane iterations, all per
  advance.
- **P2 is the scaling win.** It roughly halves fine per-sample traffic at 16³
  and gives ~2× at 32–64³, growing with resolution. It is also the phase that
  can break the surface, which is why it goes behind P0's baselines and ships
  with P2d.
- **Neither reaches the paper's envelope alone.** The Section 6.3 operator
  layout (item 3 of `OCTREE_M1_MAX_IMPLEMENTER_HANDOFF.md`) is the other half.
  This plan is the fine-band half and is independent of it.

No ms figure is projected for any phase. The 230 ms/advance measured above
could not be attributed by stage because both profiling paths are currently
broken, and predicting a split without that data would be a guess. P0 exists to
remove that excuse before P1 starts.
