# Octree M1 Max — implementer handoff

Audit date: 2026-07-26. Audits `docs/WEBGPU_OCTREE_M1_MAX_IMPLEMENTATION_PLAN.md`
against the working tree and `docs/papers/aanjaneya-2017-power-liquids.txt`.

**Read this first.** The tree was being edited by another session throughout the
audit — `lib/webgpu-octree-spgrid-vcycle.ts` was rewritten several times during
the read, once mid-`sed`. Findings are therefore anchored on **symbol names, not
line numbers**. Every finding below was re-verified against the tree state at
11:29 on the audit date. Before acting on any item, re-grep for the symbol and
confirm it still exists in the form described.

No lane was benchmarked. The GPU exclusive lock was **free** at audit time
(no `vinext dev`, no smoke run), so nothing blocks capture — it simply had not
been run. Every cost figure below is derived from encoded dispatch counts and
the measured link taxes in `docs/EXECUTION_MULTIPLIERS_HANDOFF.md`.

---

## Do not redo — these landed and are correct

Confirmed against the paper. Listed so no one re-opens them.

### Section 6.3 operator layout — landed the paper's way

The previous audit's deepest gap is closed. The catalog is no longer
uploaded-but-unbound:

- `catalogCoefficients` is **bound and read** in `lib/webgpu-octree-spgrid-vcycle.ts`.
- Neighbour addresses are **derived, not stored**: `section63Direction` supplies
  the 18 canonical offsets and `section63WorldDirection` applies the 6-bit
  transform code (8 sign flips × 6 permutations). The apply computes
  `vec3i(q) + section63WorldDirection(section63Direction(channel), m.transformAndFlags & 63u)`.
- Cross-level couplings became **ghost cells**, as in the paper.
  `rebuildCandidateGhostsFor` walks the 18 channels and inserts a `GHOST` owned
  by the coarse row wherever the channel target is not same-level.
- `lib/octree-section63-operator.ts` is the CPU oracle and declares
  `storedNeighbourIndices: 0`, `PAGE_SHAPE = [8,8,4]`.

The 36-slot resolved-row form still exists but is **not** in the hot apply.
`resolvedNeighbor` is reachable only from `dilateBandAtoB` / `dilateBandBtoA`,
i.e. band classification at topology publication — explicitly allowed by the
plan ("sorted or hashed lookup is restricted to topology publication").

### Solve tail — landed, and better than specified

`lib/octree-solve-tail-policy.ts` derives the encoded outer budget (4–10) from
**immutable scene facts** — refinement depth, inflow, terrain, moving bodies,
aspect ratio — not from the previous step's observed iteration count. That keeps
the command graph independent of simulation history. Preserve this property: do
not "improve" it into a feedback loop on last frame's count.

The four band class-applies collapsed into one (`applyMergedBand`,
`encodedMergedBandDispatchCount: 1`).

Working `countOctreePressureCommands` with the live values
(`fullOperatorDispatches: 5`, `firstOrderSetupDispatches: 9`,
`firstOrderCorrectionDispatches: l + 5 + (l-1)*4`, shell depth 2, `l = 4`):

**encoded pressure dispatches ≈ 71 + 46·E**, about **393 at E = 7**, against
~1,900 before. Derived from the formulas, not measured.

### Fine summary — merge ladder gone

`lib/webgpu-octree-fine-levelset-summary.ts` is now a one-line re-export of
`lib/webgpu-octree-fine-levelset-summary-direct.ts`. No sort, no merge ladder,
no binary search (verified by grep for all three). Access is "direct hierarchy
key to active-mip rank".

`FINE_LEVELSET_SUMMARY_CONSUMERS` enumerates consumers and classifies them —
two `simulation-critical` (`pressureRefinementEvidence`, `currentPressureOwnerWet`)
and two `diagnostics-only`. That enumeration was the prerequisite for the
replacement and it exists.

### Dead modules — resolved

`webgpu-octree-page-pool.ts` deleted. `webgpu-octree-topology-epoch.ts` now has
a production importer. The remainder became **named oracles with explicit
declarations** — e.g. `webgpu-octree-structured-velocity.ts` states "not imported
by production and intentionally contains no GPU executor or shader." The
"green module, red production path" category is gone.

---

## Work items

Ordered. Item 1 unblocks the measurement of every other item.

### 1. Capture the four regression baselines

**State:** `docs/baselines/octree-regression/` and `artifacts/octree-regression/`
do not exist. Everything else is wired: all four `capture:octree-regression-*`
scripts exist in `package.json`, `--artifact=` already forces
`FLUID_STABILITY_ENVELOPE` and `FLUID_PERFORMANCE_TRACES`, and the comparator is
implemented.

**Why it is first:** the Section 6.3 operator, the ~4.8× encoded-dispatch
reduction, and the summary replacement have all landed with **no attributed
number**. Items 2, 3 and 5 below all resolve to measurements rather than
argument the moment a quiescent lane exists.

**What to do:**

1. Confirm no concurrent GPU session and no concurrent editing session. Both
   invalidate the run; the second is currently active.
2. Run all four captures.
3. Confirm **zero `blockers`** in each artifact. Two metrics need explicit
   confirmation rather than assumption:
   - `energyRatio` — now sourced from the stability envelope, not the deleted
     energy-ledger module.
   - `activeScheduledRatio` — a `null` scheduled/active counter on any stage
     blocks the whole artifact; the accounting deliberately never coerces
     `null` to zero.
4. Capture `mini` as the exact 500-step / 2.0-second run. The contract accepts
   no other form.

**Done when:** four baselines with empty `blockers` are committed and
`npm run compare:octree-regression` is green against a re-capture.

**Decide before freezing:** the artifact has no field for the quantity the
deleted energy ledger used to own. A baseline defines what the project is blind
to. If those per-stage energy taps were load-bearing for catching dissipation
regressions, add the replacement metric *before* freezing, not after.

---

### 2. The MG hierarchy candidate rebuild is single-threaded

**Symptom.** Two entry points in `lib/webgpu-octree-spgrid-vcycle.ts`:

| Entry point | Workgroup size | Dispatch | Contains |
| --- | --- | --- | --- |
| `buildCandidateLevelSetsAndGhosts` | `1` | `[1,1,1]` | `rebuildCandidateLevelSetFor`, `rebuildCandidateGhostsFor` |
| `buildCandidateLevelDeltas` | `1` | `[1,1,1]` | `rebuildCandidateTransferFor`, `rebuildCandidateDirectoryFor`, `rebuildCandidatePageWorksetFor`, `rebuildCandidateStencilFor`, `publishCandidateSpectralBoundFor` |

Both run the **entire MG candidate hierarchy rebuild on one GPU thread**.

Note: these two were a single entry point at the start of the audit and were
split during it. The split moved work between them and changed **no** serial
cost — both halves are still `@workgroup_size(1)` at `[1,1,1]`. Do not read the
split as a fix.

**The dominant term** is `rebuildCandidateGhostsFor`: `levels × rows() × 18
channels`, each channel doing up to two `cLookup` calls, and `cLookup` is a
**256-probe linear hash**:

```wgsl
fn cLookup(l:u32,q:vec3u)->u32{ ... var slot=insertionHash(wanted,l);
 for(var probe=0u;probe<256u;probe+=1u){ ... slot=(slot+1u)&(levelCapacity(l)-1u);} return INVALID;}
```

At ~1,372 rows and 4 levels that is ~200K serial iterations at one probe each,
and far worse when the table loads. Serially, on one lane.

`rebuildCandidateLevelSetFor` additionally clears `levelCapacity(l)` slots and
`p.capacity.x` rowMap entries per level, then inserts every row through the same
hash — all serial.

**Plan violations.** Three, all explicit:

- Verification matrix: *"no hot-loop binary/hash lookup"*.
- Worksets section: *"Every recurring kernel consumes a workset or a proven live
  prefix. Direct dispatch over `dims³`, row capacity, face capacity, or maximum
  page capacity is for bootstrap and diagnostics only."*
- Phase 7 changes: *"Enforce grading with bounded mark/dilate passes over compact
  frontier lists."*

This is also the exact shape of the 59.68 ms redistance from the earlier
serial-kernel audit.

**Done when:** candidate hierarchy construction runs as parallel workset passes
with the per-level dependency order preserved by pass boundaries rather than by
serial statement order; no recurring `[1,1,1]` dispatch performs per-row or
per-capacity work; hash probing is confined to insertion during publication.

**Verify with:** the quiescent lane from item 1, plus `dispatchesPerAdvance`
by owning stage.

---

### 3. The topology delta comparison is dead code — item 2 runs every step

**Symptom.** In `beginL1CapturePlan`'s delta planner in the same file:

```wgsl
var topologyChanged=true;
var stencilChanged=true;
if(!topologyChanged){let captured=capturedGeometry[old];topologyChanged=!sameL1Topology(source,captured);
 if(topologyChanged){pageFirst=min(...);}}
else{pageFirst=0u;}
if(topologyChanged){pageFlags|=4u;stencilChanged=true;}
```

Both flags are unconditionally `true`, so the `if(!topologyChanged)` branch —
the actual comparison against captured geometry — is **unreachable**. Every row
in every dirty page is marked topology- and stencil-changed.

`topologyDirty(l)` is the **only** guard on all seven rebuild passes in item 2.
So item 2's serial cost is paid whenever any page is dirty, which for a moving
free surface is every step.

**This may be deliberate.** The adjacent comment argues a dirty fixed row can
change a neighbour handle without moving its cell, so the sparse suffix must be
rebuilt rather than compared against a second adjacency representation. If that
reasoning is sound, the correct fix is a cheaper *sufficient* dirty predicate —
not restoring a comparison that is genuinely unsound.

**Plan violation.** Phase 7 exit gate: *"unchanged topology performs no row/page
rebuild work."* As written this gate cannot hold.

**Done when:** an unchanged topology performs no candidate rebuild work, proven
on the quiescent lane; and if the unconditional flags are retained, the reason is
recorded and the dead `if(!topologyChanged)` branch is deleted rather than left
as misleading code.

**Sequencing note:** fix this before item 2. If the dirty predicate stops firing
every step, item 2's cost may drop below the threshold that justifies a
parallel rewrite — measure before doing the larger work.

---

### 4. Five reproducible test failures on the M1 cutover suite

`npm run test:webgpu:octree-m1-cutover` → **51 pass, 5 fail**. Identical across
two consecutive runs, so not mid-edit noise.

| # | Test | Error |
| --- | --- | --- |
| 25 | Dawn Metal compiles transactional structured boundary update | storage buffers (12) exceeds max per-stage limit (10) |
| 35 | Dawn Metal compiles all structured dynamics variants | storage buffers (12) exceeds max per-stage limit (10) |
| 38 | Dawn Metal compiles every direct structured publication stage | storage buffers (12) exceeds max per-stage limit (10) |
| 19 | Dawn executes a manufactured one-row solve through the merged recurrence | `binding index 2 not present in the bind group layout` — `Pipelined MGPCG · formInitialResidual` |
| 39 | rejected packed A/B publication preserves every accepted byte | accepted bytes not preserved |

**The three limit failures are one issue.** The repo declares
`PORTABLE_STORAGE_BUFFER_LIMIT = 10` (`tests/webgpu-power-ui-construction.test.ts`)
and these tests deliberately request
`maxStorageBuffersPerShaderStage: Math.min(10, adapter.limits...)` to enforce it.
`lib/gpu-startup.ts` requests no raised limit. Production shaders in
`webgpu-octree-structured-boundary.ts`, `webgpu-octree-structured-dynamics.ts`
and the direct structured publication path now need 12.

**Decision required — this is a contract question, not a bug fix.** Either:

- the portable floor moves to 12, `gpu-startup.ts` requests it, the M1 Max limit
  contract fails closed when absent (consistent with Phase 8's
  *"unsupported subgroup configurations are rejected before allocation"*), and
  the tests are updated to request the production limit; **or**
- the three shaders come back under 10 bindings.

Do not resolve it by loosening the tests alone — that deletes the gate.

**#19 and #39 are separate** and look like ordinary defects: a bind-group layout
that omits binding 2 while the bind group supplies it, and an A/B publication
rollback that does not preserve accepted bytes. #39 in particular is a
correctness gate on epoch publication and should not ship red.

**Done when:** the cutover suite is 56/56, with the portability decision recorded
in the plan.

---

### 5. Section 4.3 shell depth is below the paper, and the parity rule is invented

**Symptom.** `lib/webgpu-octree-section43-contract.ts`:

```ts
return Math.max(2, Math.min(16, Math.round(requested / 2) * 2));
```

forces **even** k, justified as "Section 4.3 shell depth must be even for
matching halves". `lib/octree-solve-tail-policy.ts` sets
`OCTREE_SECTION43_PRODUCTION_SHELL_DEPTH = 2`.

**The paper does not require even k.** §4.3 reads:

> (1) … execute **k** iterations of the damped Jacobi method … (3) Repeat **k
> additional** iterations of the same Jacobi method …

The total is 2k; k itself is unconstrained. The paper's k is **3** for normal
scenes and **10** for the thin river canyon ("10 Jacobi smoothing iterations, as
opposed to 3 in the other examples").

Production therefore sits at k = 2, **below the paper's baseline**, and the
invented parity rule is exactly what makes k = 3 unreachable.

**Why it matters:** under-smoothing the shell pushes work onto the outer PCG,
which is now capped at 10 encoded iterations. If the shell is too shallow the
solve can silently under-converge within the encoded budget.

**What to do:** drop the parity constraint, then sweep k ∈ {2,3,4} against
executed outer-iteration count and residual on the mini and quiescent lanes.
k multiplies directly into the correction dispatch count
(`2k + 4 + fullOperator + (2k-1)·mergedBand`), so this is a real
accuracy-vs-dispatch trade and should be chosen from measurement.

**Done when:** k is chosen by the fixture sweep with the measurement recorded,
and the even-k constraint is either removed or justified by something other than
"matching halves".

---

### 6. Minor — source guard misses the WGSL loop form

`lib/webgpu-octree-work-accounting.ts`, rule `unbounded-lookup`:

```js
expression: /while\s*\(\s*true\s*\)|for\s*\(\s*;\s*;\s*\)/g,
```

Matches neither WGSL `loop {}` — the form the shaders actually use — nor bounded
probe loops like `cLookup`'s 256-iteration scan. The guard did not catch
item 2. Worth widening while item 2 is being fixed, so the fix stays fixed.

---

## Suggested order

**1 → 3 → 2 → 4 → 5.**

Baselines first: nothing else can be ranked without them, and the lock is free.
Then item 3, because it is small and may shrink item 2 below the threshold that
justifies a parallel rewrite. Then item 2 with a measurement in hand. Item 4 is
independent and can run in parallel with any of these — it needs a decision more
than it needs work. Item 5 last, since it is a tuning sweep that wants the
baselines to exist.

## Standing constraints

From `docs/EXECUTION_MULTIPLIERS_HANDOFF.md` §1 rule 2 — **no accuracy-affecting
default flips in a perf commit.** Item 5 changes convergence behaviour and must
land on its own, behind its own differential gate.

From `docs/POWER_LIQUIDS_PERF_HANDOFF.md` — batching cycles behind extra pass
boundaries **regressed** on M1 (4-cycle 11.60 ms, 8-cycle 11.21 ms, against
10.88 ms direct). More pass structure is not the answer to dispatch count;
shortening the encoded chain is.
