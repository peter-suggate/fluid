# Agent handoff — fine-band P0 (instrumentation) and P1 (serial-work removal)

You are working in `/Users/petersuggate/code/me/fluid`, a TypeScript + WGSL
WebGPU fluid solver implementing Aanjaneya et al. 2017, *"Power Diagrams and
Sparse Paged Grids for High Resolution Adaptive Liquids"*. The paper text is at
`docs/papers/aanjaneya-2017-power-liquids.txt`.

Your scope is **P0 and P1 only** from `docs/FINE_BAND_DENSITY_PLAN.md`. Read
that document first — it contains the full diagnosis, the paper citations, and
the phases you are *not* doing. Do not start P2 (band-width changes), P3, or P4.
Do not touch the Section 6.3 operator work described in
`docs/OCTREE_M1_MAX_IMPLEMENTER_HANDOFF.md`.

---

## Background you need

The fine level set is a sparse narrow-band 4³-brick lattice at `fineFactor`×
the finest octree resolution, independent of the octree pressure hierarchy
(paper §5). A static audit found that at both production lane resolutions the
"narrow band" resolves to **100% of the logical brick lattice** — 4096/4096
bricks (262,144 samples) on the 16³ mini lane, 6912/6912 on the 24×18×16 UI
lane. Separately, several recurring fine-band kernels are serialized onto a
single workgroup or a single lane, and one per-sample gate is dead code.

P0 makes the problem measurable. P1 removes the serial and dead work. Band
widths (the cause of the 100% occupancy) are deliberately **out of scope** —
they carry surface-correctness risk and land later, after your baselines exist.

Measured starting point: `npm run benchmark:power-dam-moving-interface` reports
**230.02 ms/advance**, 708 dispatches, 120 compute passes per advance.

---

## Critical environment constraints

**Read these before running anything.**

1. **The GPU lock is exclusive.** Never run a Dawn smoke/benchmark while a
   `vinext dev` server, a browser WebGPU tab, or another Dawn process is live.
   Check with `ps aux | grep -iE "vinext|vite|next dev|dawn"` first. The test
   scripts print a SAFETY banner about this; it is real, not decorative.
   `tools/run-webgpu-exclusive.ts` takes the lock for the test lanes.
2. **Two profiling paths are currently broken.** Do not spend time debugging
   them unless a task below asks you to:
   - `npm run profile:power-dam-mini` / any `--profile` benchmark SIGSEGVs
     under Dawn with `FLUID_WEBGPU_DAWN_FEATURES=skip_validation`.
   - `npm run test:webgpu:minimal-power-dam-two-step` dies at
     `tools/run-webgpu-smoke.ts:357` — `copyBufferToBuffer` receives a
     non-object binding — before printing diagnostics.
   Both look like harness faults rather than solver faults. If P0 task 4 turns
   out to require one of them, say so and stop rather than rewriting the
   harness.
3. **Verification cost preference.** For checking that a rename or deletion left
   no stale references, prefer targeted `grep -rn <symbol> lib/ tests/ tools/`
   over a full `npx tsc --noEmit`. Note that `zsh` rejects `--include=*.ts`;
   use `grep -rn --include='*.ts'` with quotes, or just scope by directory.
   Run the full typecheck once at the end via `npm run test:power-liquids-structure`.

---

## P0 — make the fine band measurable

**Nothing in P1 can be attributed until P0 lands. Do P0 first and commit it
separately.**

### P0.1 — Fix the active-brick counter (one word)

`lib/webgpu-uniform-eulerian.ts:751`:

```ts
this.info.globalFineActiveBricks = value.worklistHeader[0] ?? 0;
```

The workset header layout is written at `lib/octree-fine-levelset-bricks.ts:499`:

```
[generation, activeCount, capacity, flags, dispatchX, dispatchY, dispatchZ]
```

Word 0 is the **generation**; the count is word **1**. A 62-step run currently
prints `terminal active set: ... 64 active fine bricks` — that is generation 64,
not a brick count. This single wrong word is why band density has never appeared
in a benchmark.

Fix it, and check whether any other consumer of `worklistHeader` makes the same
mistake (`grep -rn "worklistHeader" lib/ tools/ tests/`).

**Acceptance:** `npm run benchmark:power-dam-moving-interface` prints an active
brick count that is plausible (in `[1, 4096]`, changing across the run), not a
step-count-shaped integer.

### P0.2 — Publish fine-band occupancy per advance

Add to the work-accounting surface (`lib/webgpu-octree-work-accounting.ts`,
consumed from `lib/webgpu-octree.ts`):

- `activeBricks` and `logicalBricks` for the fine lattice, and the derived
  occupancy ratio;
- `dirtyPages` and `supportPages`, both already present in the page-delta
  buffer (`lib/webgpu-octree-fine-levelset-topology.ts`, offsets
  `dirtyPagesOffsetWords` / `supportPagesOffsetWords` on `pageDeltaLayout`);
- `scheduledLanes` / `activeLanes` for the five fine stages: **fine topology,
  fine transport, fine redistance, fine volume, fine restriction**.

The accounting module already models `scheduledLanes`/`activeLanes` as
`number | null` per stage (`:23-24`, `:142-147`) and flags an artifact blocker
when either is `null` (see `docs/OCTREE_REGRESSION_ATTRIBUTION.md`). Populate
them rather than inventing a parallel mechanism.

**Acceptance:** the mini and UI lanes report a real occupancy ratio; no fine
stage reports a `null` scheduled or active counter.

### P0.3 — Widen the source guard

`SHADER_SOURCE_CHECKS` (`lib/webgpu-octree-work-accounting.ts:1010-1019`)
already has a WGSL `loop {` matcher, but it only fires when `lookup`, `search`,
or `probe` appears within 512 characters of the loop opening:

```ts
expression: /\bloop\s*\{(?=[\s\S]{0,512}\b(?:lookup|search|probe)\b)/gi,
```

Two known offenders slip through:

- `publicationRow` (`lib/webgpu-octree-fine-levelset-transport.wgsl.ts:116`) —
  a `loop {}` wrapping a nested binary search that uses none of those three
  words.
- `changedNeighborRadii` (`lib/webgpu-octree-fine-levelset-topology.ts:1570`) —
  a bounded `for` loop over a capacity-derived count, so no existing form
  matches it at all.

Add a structural matcher for *a `while`/`loop` nested inside a loop whose bound
is capacity-derived*. `CAPACITY_AUTHORITY` (`:1021`) already enumerates the
capacity identifiers; reuse it. Keep the existing keyword matcher — do not
replace it.

Expect this to light up existing code. That is the point, but it means you must
decide per hit: fix it in P1, or add a narrowly-scoped, commented allowlist
entry justifying why it is bounded and cheap. Do not widen the guard and then
blanket-suppress it.

**Acceptance:** `node --import tsx --test tests/webgpu-octree-work-accounting.test.ts`
passes with new cases covering both offenders above; every surviving hit is
either fixed or has a written justification.

### P0.4 — Capture a baseline

With a clean tree and no concurrent GPU session, run and record:

```
npm run benchmark:power-dam-moving-interface
```

Record ms/advance, dispatches/advance, passes/advance, per-stage pass counts,
and the newly-correct occupancy figures. This is the before-number every P1 task
is measured against. Commit it as a note in the plan doc or under
`docs/baselines/` following whatever convention already exists there.

---

## P1 — remove dead and serialized work

**Land P1a separately from P1b–P1d.** P1b, P1c, and P1d are pure refactors:
identical values, identical ordering, fewer lanes. P1a changes far-field phi
values and needs its own parity gate.

### P1b — Parallelize the band dilation (do this one first; biggest single win)

`publishRecurringSparseBand` (`lib/webgpu-octree-fine-levelset-topology.ts:1270`)
is dispatched as **one workgroup**:

```ts
runIdentity(this.publishRecurringSparseBandPipeline, discoverEntries,
  1, 1, [0, 6, 7, 14, 16, 21, 23]);   // :849-850 — dispatchWorkgroups(1, 1)
```

Its 256 lanes stride over the interface seed list, and for each seed call
`recurringScatterMembership` (`:1255`), which runs a `(2r+1)³` triple loop of
`atomicOr` — with the current `dilationBrickRings = 6`, that is **2,197 atomics
per seed brick**, on one core group of an M1 Max, every advance.

Restructure to one workgroup per interface seed, with the `(2r+1)³` stencil
spread across the workgroup's lanes. The target buffer (`topologyErrors`) is
already a dense logical-key-indexed array and `atomicOr` is already idempotent,
so overlapping halos deduplicate at the point of insertion with no sort, hash,
or ordering requirement.

This is explicitly sanctioned by the paper, §5 *Dynamic topology*: *"After this
step, cells that lie within the narrow band can be safely marked in parallel
without any data hazards."*

Note the kernel also does two preamble jobs before the scatter — a
`pageCapacity`-strided reset of `topologyErrors` (`:1291-1293`) and a
`livePages`-strided reset keyed off the current worklist (`:1295-1300`). Those
are already lane-parallel within the workgroup but become wrong if you simply
fan the kernel out per seed. Split them into their own dispatch, or keep a
prologue dispatch and make only the scatter per-seed.

**Acceptance:** published desired-brick set is bit-identical to the pre-change
set for the first 62 steps of the mini lane; the fine-topology stage's
scheduled-lane count now scales with seed count.

### P1c — Replace the quadratic support classifier

`changedNeighborRadii` (`lib/webgpu-octree-fine-levelset-topology.ts:1570-1583`)
is called once per desired page from `classifyFineAffectedPages` (`:1584`), and
each call linearly scans up to `2 × pageCapacity = 8192` changed keys:

```wgsl
for(var item=0u;item<total&&(dirty==0u||support==0u);item+=1u){ ... }
```

Worst case 4096 × 8192 = **33.5M iterations per advance**. The early-out only
fires for pages that are near a change; distant pages scan the entire list.

Invert it into a scatter-mark, reusing P1b's kernel shape: for each *changed*
key, scatter its `dirtyHaloRings` and `supportHaloRings` Chebyshev halos into
two bits of the dense logical mask with `atomicOr`; then each desired page reads
one word. Cost becomes `O(changed × r³)` and shares P1b's structure.

Watch the buffer lifetime: `topologyErrors` is reused for lifecycle error
records in the same encode (`:1596`, `:783-799`), and `scatterRecurringSparseBand`
(`:1340`) clears it as it consumes it. Either use distinct bits in the same
array or add a separate mask; do not let the two uses alias.

**Acceptance:** `dirtyPages` and `supportPages` sets are bit-identical to the
pre-change sets across 62 mini-lane steps; the classify stage no longer scales
with `desired × changed`.

### P1d — Remove the single-lane sweeps from transport

Three `@compute @workgroup_size(1)` kernels serially walk the page arena every
advance in `lib/webgpu-octree-fine-levelset-transport.wgsl.ts`:

| kernel | line | serial iterations |
| --- | --- | --- |
| `publishStructuredFineTransportWorksets` | `:123` | two loops over `live` ≤ 4096 each |
| `summarizeStructuredFineTransport` | `:194` | one loop over `count` ≤ 4096 |
| `publishStructuredFineDelta` | `:198` | `8 + 2×pageCapacity` = 8200 words, **capacity-shaped regardless of active count**, plus a second loop over `liveCount` |

~20,000 dependent single-lane iterations per advance.

Replace each with mark → prefix rank → scatter. **Do not invent the pattern** —
this repo already implements it as the identity chain in
`lib/webgpu-octree-fine-levelset-topology.ts`:

- block scan helper: `scanIdentityBlock` (`:1413`) — a Blelloch scan over 256
  lanes;
- three-tier pipeline: `scanIdentityRecords` → `scanIdentityGroups` →
  `scanIdentitySuperGroups` → `offsetIdentityGroups` → `offsetIdentityRecords`,
  encoded at `:876-889`.

Reuse that shape. `summarizeStructuredFineTransport` is a pure reduction (sums
and a max) and should become a standard two-tier workgroup reduction, not a
scan. Make `publishStructuredFineDelta` active-shaped rather than
`2 × pageCapacity`-shaped.

Ordering matters for reproducibility: the summarize reduction accumulates f32-
free integer counters and a `max`, so tree order is safe. Confirm that before
assuming it for any counter you touch.

**Acceptance:** the four transport worksets contain the same page IDs in the
same order as before; `control` words (`outside`, `nonfinite`, `processed`,
`extended`, `maxDisplacement`, `committed`) are identical across 62 mini-lane
steps; the compacted delta is identical.

### P1a — Implement the dead transport band gate (separate commit, parity-gated)

`transportBandCells` is plumbed from `lib/webgpu-octree.ts:3001` through
`lib/webgpu-octree-fine-levelset-transport.ts:287` (`u[22] = band`) into the
uniform, and the shader declares it at
`lib/webgpu-octree-fine-levelset-transport.wgsl.ts:12`:

```wgsl
physical:f32,dt:f32,bandCells:u32,closed:u32,openTop:u32, ...
```

**It is never read.** `grep -c bandCells` on that file returns 1 — the struct
declaration. Every valid sample in every active page is traced, each with
`m = fineFactor` velocity reinterpolations. At 16³ that is 262,144 characteristic
traces and roughly 1M velocity samples per advance.

Implement the gate: a sample with `|phi| > bandCells × h` copies its value
forward unchanged instead of tracing. Paper basis, §5: the fine grid owns the
zero crossing within the band, and *"we use the fine level set to correct the
coarse level set wherever we have valid ϕ-values"* — outside the band the fine
value carries sign, not distance, and the coarse octree level set is the
authority.

**This changes numbers.** Before committing, verify:

1. the redistance seed identity still holds — `seedClosestPoints`
   (`lib/webgpu-octree-fine-levelset-redistance.ts`) seeds from sign changes,
   so a skipped far-field sample must not fabricate or destroy a crossing;
2. `control.outside` (`departureOutsideBand`) and the redistance `unresolved`
   count both stay at zero across the full 500-step mini lane;
3. zero-crossing hash, volume, and IoU parity hold on mini and UI.

If (1) cannot be established, stop and report — do not ship the gate on a
hunch. Deleting the dead plumbing instead is an acceptable fallback outcome,
and is strictly better than leaving a parameter that lies about what it does.

---

## Verification protocol

Run after each of P0, P1b–P1d, and P1a, with no concurrent GPU session:

```bash
# CPU-side: types, shader compile, structural guards
npm run test:power-liquids-structure

# GPU-side, exclusive lock: fine level-set and power correctness
npm run test:webgpu:octree-power

# GPU-side: 62-step wall-clock lane — the before/after number
npm run benchmark:power-dam-moving-interface

# GPU-side: 500-step correctness gate — required before declaring P1 done
npm run test:webgpu:minimal-power-dam-break
```

Targeted tests worth running while iterating:

```
tests/webgpu-octree-work-accounting.test.ts        # P0.3
tests/octree-fine-levelset-bricks.test.ts          # header layout, P0.1
tests/webgpu-octree-fine-levelset-*.test.ts        # topology / transport / rollback / endurance
tests/power-liquids-recurring-structure.test.ts    # recurring-dispatch structure
```

---

## Constraints

- **Match the surrounding style.** This codebase writes dense single-line WGSL
  and comments that explain *why*, not *what*. Read the file you are editing
  before adding to it.
- **Fail closed.** Every publication path here validates and rejects rather than
  degrading. Preserve that: a restructured kernel must still produce the same
  error/flag words on the same malformed input.
- **No new capacity-shaped recurring dispatches.** That is what P1 is removing.
- **Commit granularity:** P0 as one commit; P1b/P1c/P1d as one or three; P1a
  strictly on its own.
- Do not run `git push` or open a PR unless explicitly asked.

## Deliverable

A short report containing:

1. before/after ms per advance, dispatches per advance, and passes per advance
   on `benchmark:power-dam-moving-interface`, attributed per P1 task;
2. the true fine-band occupancy ratio on the mini and UI lanes, now that P0.1
   makes it readable — this number is the entire justification for P2 and
   nobody has ever seen it;
3. every source-guard hit P0.3 exposed, with fixed-or-justified for each;
4. for P1a: shipped or not, and the evidence either way;
5. anything you found that contradicts `docs/FINE_BAND_DENSITY_PLAN.md`. The
   plan is a static audit with one live benchmark behind it. If a measurement
   disagrees with it, the measurement wins — say so plainly.
