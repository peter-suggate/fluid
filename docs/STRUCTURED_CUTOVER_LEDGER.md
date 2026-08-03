# Structured cutover ledger

Status: static audit, 2026-08-03. Branch `perf/structured-cutover`, tree at
`97a6fa7` (`feat(render): cut over to live sparse scene publication`).
Companion: `POWER_LIQUIDS_SCENARIO_LEAP_PLAN.md` (see §4 for its errata).

**No GPU was used to produce this.** Every claim below is a static read of the
tree plus the artifacts already in `artifacts/scene-size-overhead/`. Where a
question needs a device, it says so.

**Anchor on symbol names, not line numbers.** `lib/webgpu-octree-air-velocity-
support-gpu.ts` and the SPGrid/transport files were being edited while this was
written; line numbers moved twice. They are correct as of `97a6fa7`.

## 0. What this is

Every `FLUID_*` env toggle read in `lib/`, classified into one of five buckets:

| bucket | meaning | count |
|---|---|---:|
| 1 INSTRUMENT | diagnostic / census / oracle / audit / tripwire / fault injection. **Not** cutover candidates — they must stay gated. | 29 |
| 2 LANDED, hatch live | default ON, the `=0` path is real code | 24 |
| 3 LANDED, hatch vestigial | default ON (or OFF) but the other path is unreachable/overridden — **the flag does nothing** | 7 |
| 4 NOT TAKEN | default OFF with a real implementation behind it | 8 |
| 5 DEAD | flag referenced but the feature is gone, or the name is a constant, not an env read | — |

Two facts that apply to the whole table and change how you should read it:

- **No toggle's non-default value is exercised by any GPU test anywhere in the
  tree.** `npm run test:unit` (`tests/*.test.ts`) is pure CPU: every hit is a
  predicate unit test (`assert.equal(fooEnabled({FLUID_X:"0"}), false)`) or a
  regex over a WGSL/host source string. No npm script and no smoke lane in
  `package.json` sets any of these flags to a non-default value — verified by
  grepping the whole `scripts` block. The single exception is the GPU *tool*
  `tools/measure-power-liquids-reason-cones.ts:28`. Therefore **every doc
  comment of the form "a zero restores the former behaviour exactly" is an
  unverified claim**, with exactly two exceptions noted in §2.
- **`npx tsc --noEmit` is clean** on this tree (exit 0), so any deletion in §2
  can be type-checked before it goes near a device.

---

## 1. The full toggle table

Default column: **ON** = `!== "0"`, **OFF** = `=== "1"`, **num** = numeric.

### Bucket 1 — INSTRUMENT (leave gated; not cutover candidates)

| flag | file:line | default | what the enabled path does |
|---|---|---|---|
| `FLUID_ALGORITHM_DIAGNOSTICS` | `lib/octree-algorithm-diagnostics.ts:15` | OFF | Closes compute passes at hypothesis boundaries for Dawn attribution. Header (`:7-8`) asserts it "never changes dispatch counts, buffers, shader inputs, or GPU/CPU synchronization". |
| `FLUID_STRUCTURED_ENERGY_PROBE` | `lib/webgpu-octree-structured-dynamics.ts:154` | forced 1/0 | Forces the projection-energy probe either way; otherwise implied by the three below. |
| `FLUID_STABILITY_ENVELOPE` | `…structured-dynamics.ts:158` | OFF | Enables four `@workgroup_size(128)` energy summarizers over the class 5–8 face workset. |
| `FLUID_POWER_GENERATION_AUDIT` | `…structured-dynamics.ts:159` | OFF | Same probe; also the smoke gate's audit-snapshot trigger. |
| `FLUID_ENERGY_EVERY_STEPS` | `…structured-dynamics.ts:160` | num 0 | `>0` logs the four stage energies through `readStats`. |
| `FLUID_WORKSET_CENSUS` | `…fine-levelset-transport.ts:35`, `…structured-dynamics.ts:273` | OFF | Per-class workset population census. |
| `FLUID_SYMMETRY_STAGE_AUDIT` | `…persistent-mgpcg.ts:553`, `…structured-dynamics.ts:434`, `webgpu-octree.ts:2978`, `:5748` | OFF | Pins §6.3 diagonal, case ids, rhs and preconditioner stages at tolerance 0. **See the trap in §4.** |
| `FLUID_PERSISTENT_MGPCG_STAGE_CAPTURE` | `…persistent-mgpcg.ts:547`, `webgpu-octree.ts:5749` | unset | Selects which of 9 named solver stages the audit captures; throws on an unknown name. |
| `FLUID_PERSISTENT_MGPCG_DIAGNOSTIC_OUTPUT` | `…persistent-mgpcg.ts:74` | unset | Names a diagnostic readback channel. |
| `FLUID_POWER_HYBRID_FULL_ADDRESS_ORACLE` | `…persistent-mgpcg.ts:532` | OFF | Compile-time WGSL oracle variant of the row addressing. |
| `FLUID_POWER_HYBRID_FULL_ADJOINT_ORACLE` | `…persistent-mgpcg.ts:534` | OFF | Ditto, adjoint. |
| `FLUID_POWER_HYBRID_ALL_CHANNELS_ORACLE` | `…persistent-mgpcg.ts:536` | OFF | Ditto, all 18 channels. |
| `FLUID_POWER_HYBRID_SECTION63_BAND_ORACLE` | `…persistent-mgpcg.ts:538` | OFF | Ditto, §6.3 band. |
| `FLUID_POWER_HYBRID_FULL_APPLY_ORACLE` | `…persistent-mgpcg.ts:540` | OFF | Ditto, full apply. |
| `FLUID_POWER_HYBRID_LINEAR_APPLY_ORACLE` | `…persistent-mgpcg.ts:542` | OFF | Ditto, linear apply. |
| `FLUID_POWER_INIT_TRACE` | `lib/webgpu-octree.ts:2632` | OFF | Console trace of power initialisation. |
| `FLUID_TRACE_DEBUG` | `lib/performance-trace.ts:622` | OFF | Warns on trace-decode failure. |
| `FLUID_WATER_DIAGNOSTICS` | `lib/webgpu-water-pipeline.ts:1363` | OFF | Surface diagnostics encode (also on via `?panel=diagnostics`). |
| `FLUID_GPU_ISOLATE_PASS_LABELS` | `lib/webgpu-pass-broker.ts:27` | OFF | Fences on every compute-label change so xctrace can attribute per label. Self-documents its own cost as ~neutral (39.07 vs 39.09 ms/adv) but **does** change the command stream. |
| `FLUID_GPU_ISOLATE_PASS_LABEL_PREFIXES` | `lib/webgpu-pass-broker.ts:36` | "" | Scopes the above to named stage prefixes. |
| `FLUID_SPGRID_TOUCHED_RADIX_TRIPWIRE` | `lib/webgpu-octree-spgrid-vcycle.ts:1175` | OFF | Runs the dense and compact candidate builders **concurrently** and reports 4 differential mismatch codes. Inert unless `FLUID_SPGRID_TOUCHED_RADIX_SORT=1`. Strict superset of the default work. |
| `FLUID_SPGRID_ROW_SCHEDULE_ORACLE` | `…spgrid-vcycle.ts:1247` | `"capacity"` | Forces `probeCandidateSkip`/`publishCommittedInputs` to launch at `plan.rowDispatch` instead of the GPU-published live count. Both kernels are internally row-bounded, so it can only add no-op invocations: a bit-identity oracle. |
| `FLUID_SPGRID_DIRECT_BY_CHASE` | `…spgrid-vcycle.ts:1213` | OFF | Swaps the operator-image staging for the older per-row page-chase. WGSL header: "Production never encodes it… it exists so `tests/webgpu-octree-operator-image-differential.test.ts` can apply BOTH addressings". **Also production-unreachable — see §2.** |
| `FLUID_SPGRID_ADJOINT_BY_CHASE` | `…spgrid-vcycle.ts:1220` | OFF | Same for the merged-band adjoint. **Zero test coverage of any kind. Also production-unreachable.** |
| `FLUID_ENGINE_SPLIT` | `lib/webgpu-octree.ts:360` | `!== "collapsed"` | Selects 14-name fine-phase vs 7-bucket engine-phase trace attribution. Reachable **only** when `productionBoundary` is supplied, which happens at exactly one site (`lib/webgpu-uniform-eulerian.ts`, trace-gated). Zero production effect. |
| `FLUID_STRUCTURED_DEBUG_WRITE_MASK` | `…structured-dynamics.ts:470` | num 0 | Debug write channel, gated on the symmetry audit. |
| `FLUID_PROBE_STRUCTURED_PUBLICATION_REPEAT` | `…structured-velocity-gpu.ts:120` | 1 | Repeats the publication 2–8× to separate encode from drain. Comment: "Never set in production". |
| `FLUID_TEST_INJECT_RECURRING_BAND_REJECTION_GENERATION` | `…fine-levelset-topology.ts:52` | unset | Fault injection for the band-rejection fail-closed path. |
| `FLUID_OCTREE_AIR_SUPPORT_FRONTIER_WAVES` | `…air-velocity-support-gpu.ts:82` | num 12 | Wide-wave prefix length. Not a correctness bound — the persistent tail remains the exact authority; a wave with an empty frontier dispatches zero workgroups. Pure measurement lever. |

### Bucket 2 — LANDED, escape hatch live

| flag | file:line | default | what the `=0` path does | legacy size | WGSL text changes on delete? | round-trip Δ ⇒ gate |
|---|---|---|---|---|---|---|
| `FLUID_OCTREE_AIR_SUPPORT_TOPOLOGY_REUSE` | `…air-velocity-support-gpu.ts:273` | ON | Clears uniform word 63 (`P.reuseTopology`), which flips five WGSL `select()`s (`:1673,:1676,:1682,:1683,:1687`) back to `p.supportCapacity`-shaped dispatch records. | 0 host lines; 5 `select()` terms | **No** — WGSL text is identical either way, only the uniform differs | none ⇒ A |
| `FLUID_OCTREE_AIR_SUPPORT_CHANGED_FRONTIER` | `…air-velocity-support-gpu.ts:333` | ON | Selects the dense 12-wave prefix + 12-wave + exact-tail domain-wide oracle instead of the sparse changed-frontier march. | 4 WGSL entry points (`extendAirSupportFacesAtoB/BtoA` `:2675`, `advanceAirSupportMarchWave`, `marchAirSupportFacesToFixedPoint` `:2696`) + ~24 encode lines (`:1204-1211`, `:1274-1292`) | Yes (removes 4 entry points from the module) | none within a kernel; different march schedule ⇒ **B** |
| `FLUID_OCTREE_AIR_SUPPORT_INDIRECT_FRONTIER_GATE` | `…air-velocity-support-gpu.ts:343` | **ON** | Falls back from the GPU-published wave gate (indirect offsets 0/12/24) to the host-authored face schedule (offset 48). Also requires `fineFactor===1`, so it is inert on the factor-4/8 lanes. | ~8 ternaries in `encode` | No | none ⇒ A |
| `FLUID_OCTREE_AIR_SUPPORT_CANDIDATE_BOUND_RESET` | `…air-velocity-support-gpu.ts:325` | ON | Moves one WGSL statement `sw(30u,0u);` from before the candidate bound (`:1639`) to after it (`:1655`). **This is a known-broken path** — commit `3348b8a` fixed exactly this ordering; the doc comment describes the corruption it produced. | 1 WGSL statement | No (ON emits exactly the post-deletion text) | none ⇒ A |
| `FLUID_AIR_SUPPORT_RECONSTRUCT_COMPACT_PASS` | `…air-velocity-support-gpu.ts:373` | ON | Adds one `broker.fence("Section 5 closest-face fixed point published")` between the march and reconstruction (`:383`, called `:1296`). | 1 fence | No | pass boundary only ⇒ A |
| `FLUID_OCTREE_AIR_SUPPORT_MARCH_FASTPATH` | `…air-velocity-support-gpu.ts:104` | ON | Restores the authored source for three exact work removals in `extendFace`: branch-vs-`select` bank load (`:2557`), settled-seed frontier skip (`:2568`), loop-invariant patch centre (`:2651`, `:2664`). Captured at module load into a `const` (`:106`). | ~6 WGSL lines | Yes (4 template sites) | none ⇒ A |
| `FLUID_POWER_HYBRID_DRY_IDENTITY` | `…structured-boundary.ts:49` | ON | Emits `if(false)` instead of `if(dryIdentity)` in `dynamicRowClass` (`:614`), deleting row class 4. Dry rows fall into classes 2/3, widening the row union and losing the MGPCG `applyIdentityRow` arm (`…persistent-mgpcg.wgsl.ts:160`). Read at **module load**. | 0 lines (token substitution) | **No** — deletion leaves the ON WGSL byte-identical. The only flag in the tree with that property. | none ⇒ A |
| `FLUID_STRUCTURED_BOUNDARY_COMPACT_PASS` | `…structured-boundary.ts:340` | ON | Adds 5 `broker.fence()` calls (`:357`) around classify/resolve/resolveSolid/commit/rebuild. | 1 line | No | none ⇒ A (expect bit-exact) |
| `FLUID_STRUCTURED_BOUNDARY_ADVECT_FLAT` | `…structured-dynamics.ts:53` | ON | Drops `classifyStructuredBoundaryDryProbes` (1 pipeline, 1 indirect dispatch) and binds `advectStructuredFamilies` instead of `advectStructuredFamiliesFlattenedBoundary`. Classes 5/6 use the serial walk on **both** arms. | 1 entry point + 1 dispatch | Yes (entry-point roster) | **ON adds** a cache round-trip ⇒ **B** (though the reduction is an order-free boolean OR, oracle-pinned on CPU) |
| `FLUID_STRUCTURED_DEEP_IDENTITY_CARRY` | `…structured-dynamics.ts:61` | ON | Clears the pipeline-override constant `deepIdentityCarryEnabled` (`:965`, injected `:518-523`), disabling the exact-carry early-out at `:2236` so every deep-interior face takes the full 2nd-order midpoint trace. | 0 lines (override const) | No (specialized program unchanged) | none, but the OFF arm is **not** a clean cost control: the dry-probe dispatch is still encoded and its output is never read ⇒ **B** |
| `FLUID_STRUCTURED_DYNAMICS_COMPACT_PASS` | `…structured-dynamics.ts:72` | ON | Adds 2 `broker.fence()` calls (`:807`, `:877`). The genuine storage→indirect fence at `:814` is unconditional in both arms. | 2 fence sites | No | none ⇒ A (expect bit-exact) |
| `FLUID_STRUCTURED_IDENTITY_CARRY` | `…structured-velocity-gpu.ts:411` | ON | Clears uniform word 42 (`P.identityCarryEnabled`), read at `:780`. Turns four zero-workgroup identity dispatches (records 0/3 → `countFamilies`, `classify`, `scatter`, `section63`) into full row/slot-width launches on **every** step. **The single largest work lever in the tree.** | 0 lines | No (uniform word) | none, but the OFF arm flips `control.activeBank`, so re-derived ≠ carried by address ⇒ **B** |
| `FLUID_OCTREE_PERSISTENT_MGPCG_COMPACT_ROWS` | `…persistent-mgpcg.ts:157` | ON | Substitutes `capacity()` for `wRows` as the arena channel stride (`…persistent-mgpcg.wgsl.ts:137`, used `:369-375`). Produces a second shader variant (distinct cache key `:584`). | 1 token | Yes (source-level const) | none — single dispatch, single buffer ⇒ A |
| `FLUID_FINE_TRANSPORT_B4_ADDRESSING` | `…fine-levelset-transport.ts:51-57` | ON | Clears override `b4FineAddressing`, restoring integer division in `localCoord`/`sampleIndex`. **Geometrically dead**: `planFineLevelSetBricks` throws for anything but 4³/64 (`lib/octree-fine-levelset-bricks.ts:170`), so only the env var can select the generic arm. | 4 WGSL lines | Yes (removes the override + branches) | none — integer address arithmetic ⇒ A |
| `FLUID_FINE_TRANSPORT_QUIESCENCE` | `…fine-levelset-transport.ts:87` | ON | Writes `-1` instead of the epsilon into float word 73 (`P.inflowTiming.y`, `:585`). Since `displacementCells >= 0`, `displacementCells <= -1` is identically false and no brick can ever sleep. | **0 lines** | No (byte-identical WGSL, identical pipelines and pass graph) | none ⇒ A |
| `FLUID_FINE_REASON_CONES` | `…fine-levelset-topology.ts:486` | ON | Interpolates `const REASON_CONES:bool=false` (`:1702`), so the delta-radius scatter runs unconditionally (wider halo dispatch, `:2089`, `:2116`) and `interfaceNeighborRadii` always reads the broad scattered mask. | 5 WGSL branch sites + halo dispatch width | Yes (source-string interpolation) | none (u32 masks + atomics); changes the dirty/support **set** ⇒ **B** |
| `FLUID_FINE_DELTA_RADIUS_MASK` | `…fine-levelset-topology.ts:1696` | ON | Interpolates `const DELTA_RADIUS_MASK:bool=false` (`:1703`), replacing the O(1) mask read at `:2430` with an O(producers) linear scan (`:2443+`). Also changes the error-array reset extent (`:2025`). No named predicate — it is a bare parameter default with **no doc comment**. | ~15 WGSL lines | Yes | none ⇒ A |
| `FLUID_FINE_JFA_DIRTY_FRONTIER` | `…fine-levelset-topology.ts:1236` **and** `…fine-levelset-redistance.ts:327` | ON | Restores support-wide JFA seeding: `params.recurringDelta` 2→1 and the seed pass's indirect record switches from `dirtyOffsetBytes` to the frontier record. **Two read sites**: topology re-reads per encode, redistance captures once at construction — value-identical, temporally desynchronizable. | 2 uniform words + a dispatch-record selection | No | coverage change, not rounding ⇒ **B** |
| `FLUID_FINE_JFA_B4_ADDRESSING` | `…fine-levelset-redistance.ts:332` | ON | Selects `fineLevelSetJFACPTWGSL` (generic) instead of `fineLevelSetJFACPTB4AddressingWGSL`. Two fully materialized module-level sources differing in exactly 4 lines. Geometrically dead, same as the transport twin. | 4 lines × 2 sources | No (default already `buildFineLevelSetJFACPTWGSL(true)`) | none ⇒ A |
| `FLUID_OCTREE_COARSE_REDISTANCE` | `…power-coarse-levelset.ts:446` | ON | Skips **8 indirect dispatches** (4× `redistanceAtoB` / 4× `redistanceBtoA`, `:447-456`) — the paper's coarse eikonal sweeps. Only reachable on the coarse-only (factor-1) lane; the kernel also self-disables on `params.hasFine!=0u` (`:602`). No named predicate, no doc comment on the hatch. | 8 dispatches, ~35 WGSL lines + 6 helpers | No | **8 storage round-trips** (scratch bank ping-pong) ⇒ **B**, never A |
| `FLUID_FINE_VOLUME_CADENCE` | `…fine-levelset-volume.ts:41` | num 1 | `>1` skips the whole volume-correction encode (~10 dispatches) on off-cadence steps (`:312-319`). Malformed values **throw**, they do not fail closed. | ~8 host lines | No | removes round-trips on skipped steps; `applyPipeline` mutates `source.phi` ⇒ deliberate accuracy trade, **B** |
| `FLUID_OCTREE_SECTION43_SHELL_DEPTH` | `lib/octree-solve-tail-policy.ts:294` | 8 (4 mini) | Even integer in [2,16] → `Params.shape.z`, the §4.3 shell smoothing trip count. Changes the preconditioner. | uniform word | No | **physics/iteration knob, not a cutover** |
| `FLUID_OCTREE_VCYCLE_TAIL_CELLS` | `lib/octree-solve-tail-policy.ts:81` | 64 | Sets how many V-cycle levels run as wide per-level dispatches vs fold into the one-workgroup `coarseVcycleTail`. Floor `OCTREE_VCYCLE_MINIMUM_PARALLEL_LEVELS = 2`. | plan-shape only | No | **moves** a round-trip (dispatch boundary vs `storageBarrier`) ⇒ **B** despite the source's Gate-A claim at `:98-103` |
| `FLUID_COARSE_VELOCITY_STENCIL_COVERAGE` | `…coarse-summary.ts:25` | num 1 → 0.999 | Lowers the stencil-coverage acceptance threshold in `velocityAtCentered` (`:454`), renormalising by covered weight. The source itself (`:452-453`) says "That is a physics change, not a launch shape". | uniform float | No | **physics knob, not a cutover** |

### Bucket 3 — VESTIGIAL. See §2.

### Bucket 4 — NOT TAKEN. See §3.

### Bucket 5 — DEAD

- **`FLUID_OCTREE_ROW_DELTA_CENSUS`** — referenced by `POWER_LIQUIDS_SCENARIO_LEAP_PLAN.md:134` (P0.4 tells the team to run with it), `POWER_LIQUIDS_10X_DISCOVERY_EXPERIMENTS.md:77`, `POWER_LIQUIDS_FINE_BAND_10X.md:29,60,500`, and `POWER_LIQUIDS_TEMPORAL_COHERENCE_HANDOFF.md:134` (which even cites `lib/webgpu-octree.ts:3123-3171`). **It exists nowhere in `lib/`, `tools/`, or `tests/`.** Removed by commit `9de199a` "refactor: remove legacy fluid and rendering paths".
- **Not env reads** (they grep as `FLUID_*` but are ABI/layout constants):
  `FLUID_BLAST_RADIUS_ROUNDS`/`_UNREACHED` (`lib/webgpu-fluid-blast-radius.ts:30,36`),
  `FLUID_M1_MAX_REDUCTION_LANES` (`lib/webgpu-device-limits.ts:57`),
  `FLUID_BODY_SELECTION_ID`/`_MINIMUM_CELLS` (`lib/editor-fluid-body.ts:43,45`),
  `OCTREE_PRESSURE_FLUID_FOOTPRINT_HEADROOM` (`lib/webgpu-octree.ts:986`),
  and the whole `FLUID_CELL_TRACE_*` / `FLUID_BRICK_*` / `FLUID_TILE_*` /
  `FLUID_RESIDENCY_*` / `FLUID_COVERAGE_*` / `FLUID_GPU_ACTIVITY_*` families.

---

## 2. The free-deletion list (bucket 3)

Seven flags whose non-default path is **unreachable, overridden, or unwired**.
Removing them is provably behaviour-preserving: the flag cannot change a single
GPU instruction today. Ordered by confidence.

### 2.1 `FLUID_OCTREE_AIR_SUPPORT_COMPACT_FINE_DEMAND` — **DELETED** (2026-08-03)

Was `lib/webgpu-octree-air-velocity-support-gpu.ts`, predicate
`octreeAirSupportCompactFineDemandEnabled`, plus the `compactFineDemand`
default parameter of `parameterData`.

**Evidence, re-verified against the post-consolidation file.** The predicate had
exactly one caller in the tree: that default parameter. `parameterData` in turn
had exactly one caller — `encode` — and `encode` passed an explicit `true`
(`const compactFineDemand = true;`), so the default expression, and therefore
the environment read, was **never evaluated**. Independently, `compactFineDemand`
was **not referenced anywhere inside `parameterData`'s body**: it was absent from
the packed `capturePreceding` control word (bits `publicationCount>0`=1,
`changedFrontier`=2, `compactFineCells`=4, `indirectFrontierGate`=8,
`retainedGraph`=16). Dead twice over. Its two uses inside `encode` were
constant-folds of `true`:
`fineSlot !== undefined && compactFineDemand && this.fineDemandScheduleGroups`
and `(fineSlot === undefined || !compactFineDemand ? 4 : 5) * 12`.

Removed: the 9-line exported predicate, the `compactFineDemand` parameter, the
`const compactFineDemand = true;` local, and both folds. The three-line
"Production has one scheduling contract" comment was retained and moved onto the
fine-demand schedule dispatch it actually explains.

Vestigialized by commit `a56ddd0` "perf(power-liquids): revive the coarse-only
tracker and land Bet 1/4 scaffolding" (`git log -S 'const compactFineDemand = true;'`).

### 2.2 `FLUID_OCTREE_AIR_SUPPORT_COMPACT_FINE_CELLS` — **DELETED** (2026-08-03)

Was `…air-velocity-support-gpu.ts`, predicate
`octreeAirSupportCompactFineCellsEnabled`, plus the `compactFineCells` default
parameter of `parameterData`.

**Evidence, re-verified.** Same shape: the predicate's only caller was that
default parameter, and `encode` — the sole caller of `parameterData` — always
passed the explicit local `const compactFineCells = fineSlot !== undefined;`.
The env value could not reach the control word. Bit 4 is set from the local.

Removed: the 9-line exported predicate. The `compactFineCells` **parameter was
kept** — it still carries bit 4 and gates the three-dispatch closure/emission
block in `encode` — with its default changed from the env read to
`fineSlot !== undefined`, which is exactly what the sole call site passes.

**New finding — bit 4 is written but never read.** The WGSL helper that reads it,
`compactFineDemandActive()` (misnamed: bit 4 carries `compactFineCells`, not fine
demand), is **declared once and called from nowhere** in the 135,461-byte
generated module — verified by grepping the emitted WGSL, where `capturePreceding`
is only ever masked with `1u`, `8u` and `16u` outside that one dead declaration.
So the host writes bit 4 for no GPU consumer. This was **not** fixed here: both
the ledger's suggested rename and removing the dead helper would perturb the
generated WGSL, and the deletion above was required to leave it byte-identical.
It is also an ABI change (a hole in `capturePreceding`) rather than a flag
deletion. Left as a follow-up, currently locked by
`tests/webgpu-octree-air-velocity-support-gpu.test.ts:416`.

**Tests (both flags).** The two predicate-only tests
(*"fine demand defaults to a GPU-authored live-page launch and retains the
capacity A/B"* and *"factor-4/8 compact fine-cell listing defaults on with a
dense A/B"* — both titles asserting an A/B `a56ddd0` had already removed) checked
nothing once the predicates were gone. They were replaced by one test,
*"fine demand and compact fine-cell listing are unconditional in the encoder"*,
which regexes `encode.toString()` for the properties that are actually true: no
`compactFineDemand` token anywhere, the schedule dispatch gated on `fineSlot`
alone, `compactFineCells` derived from `fineSlot`, and no `COMPACT_FINE_*`
environment name in the encoder.

**Verification.** `npx tsc --noEmit` exit 0 before and after.
`tests/webgpu-octree-air-velocity-support-gpu.test.ts` 29/30 → 28/29 pass
(1 `WEBGPU_NODE_MODULE` skip both times, 0 fail both times; the count drops
because two tests became one) and `tests/octree-air-support-compact-authority.test.ts`
5/5 before and after. *"no Section 5 producer entry point reaches the portable
storage-buffer ceiling"* still passes. The generated
`octreeAirVelocitySupportPublicationWGSL` is **byte-identical** — 135,461 bytes,
sha256 `50ec3bd7…4337cc85` before and after — because neither flag ever reached
WGSL at all.

### 2.3 `FLUID_COARSE_PHI_BFECC` — unreachable today, but **KEPT** (2026-08-03 re-audit)

`lib/webgpu-octree-fine-levelset-transport.ts:70`:

```ts
return fineFactor === 1 && environment?.[FLUID_COARSE_PHI_BFECC_ENV] === "1";
```

**Unreachability re-verified, with a corrected proof.** The original evidence
above was wrong on the mechanism: `if (!this.coarseOnlySurfaceTracking) {`
(`lib/webgpu-octree.ts:2156`) **closes at `:2273`**, and the two
`new WebGPUFineLevelSetTransport(...)` calls (`:3080`/`:3083`) are in a *later
method entirely*, guarded by `if (this.globalFineSourceA && this.globalFineSourceB)`
(`:3068`). The conclusion survives by a different route: `globalFineSourceA/B`
are assigned at exactly one place each — `:2230`/`:2231`, inside the `:2156`
block — so a transport still only exists when `globalFineLevelSetFactor !== 1`,
and `plan.fineFactor ∈ {4, 8}` (the option type is `1 | 4 | 8`).
`coarsePhiBFECCEnabled` therefore cannot return true.

**But do not delete it.**

1. `docs/POWER_LIQUIDS_COARSE_ONLY_PLAN.md` names this exact flag twice: §4.2
   (line 160, "Add optional BFECC/MacCormack behind `FLUID_COARSE_PHI_BFECC`
   for the quality ladder") and §4.6 (line 220, an acceptance gate: "if it
   exceeds this, BFECC goes default-on before proceeding"). Its Stage 1 is a
   factor-**1 band instance**, i.e. precisely `plan.fineFactor === 1`.
2. The flag was **reachable three days ago**. `coarsePhiBFECCEnabled` landed in
   `8600088` (2026-07-31), when `webgpu-octree.ts` had no
   `coarseOnlySurfaceTracking` at all and `globalFineFactor` could be 1.
   `6b14f96` (08-01) / `a56ddd0` (08-02) repointed factor 1 at the coarse-only
   tracker. This is collateral of an in-flight pivot on this branch, not a
   retirement — no commit message retires BFECC.
3. **Deleting it frees zero GPU resources today.** `reversePhi` is only created
   when `bfeccEnabled` (`…transport.ts:379-382`), the reverse/correction
   pipelines are only compiled when `bfeccEnabled` (`:479-486`), and the
   `layout: "auto"` bind groups that mention binding 10 (`:442-445`) are only
   built from those pipelines. No buffer, no pipeline, no dispatch, and **no
   pressure on Dawn's ten-storage-buffer ceiling** is being spent on it.

**What must become true for it to be reachable.** `WebGPUOctree` must be able to
construct a fine band whose `plan.fineFactor === 1`. Today `:1638` collapses
"factor 1" onto "coarse-only, no band", so the band-construction block at `:2156`
is skipped and `globalFineSourceA/B` stay undefined. Reachability needs
surface-tracking *mode* separated from *factor* — e.g. coarse-only-tracker vs
band-at-octree-resolution as distinct options — which is exactly the coarse-only
plan's Stage 1, and is required outright by its Stage 2 (a factor-1 baseline band
under a factor-4/8 regional overlay). Then `FLUID_COARSE_PHI_BFECC=1`.

*Related dead-by-the-same-construction code, to be revived or removed together,
never piecemeal:* the WGSL factor-1 RK2 family (`midpointDeparture`,
`reverseMidpointDeparture`, and the four `if(p.segments==1u)` early-outs at
`…transport.wgsl.ts:556-559`, plus `:224`), and `lib/webgpu-octree.ts:3024-3028`,
which sizes air-support demand for `plan.fineFactor === 1`.

### 2.4 `FLUID_FINE_DELTA_NEIGHBOR_QUERY` — **DELETED** (2026-08-03)

Was `lib/webgpu-octree-fine-levelset-topology.ts:1698` (bare parameter default,
no doc comment, no helper).

**Evidence, re-verified.** Its only consumer was `interfaceNeighborRadii`
(`:2422`). Reading the body in order:

```wgsl
:2430   if(DELTA_RADIUS_MASK){let membership=atomicLoad(&topologyErrors[key]);
:2431     return vec2u(select(0u,1u,(membership&DELTA_DIRTY)!=0u),
:2432       select(0u,1u,(membership&DELTA_SUPPORT)!=0u));}
:2434   if(DELTA_NEIGHBOR_QUERY){ … }
```

`DELTA_RADIUS_MASK` is default **true** (`FLUID_FINE_DELTA_RADIUS_MASK`, §1
bucket 2), so control never reached `:2434`. `FLUID_FINE_DELTA_NEIGHBOR_QUERY=1`
alone was a no-op; it also needed `FLUID_FINE_DELTA_RADIUS_MASK=0`, and nothing
in `package.json`, `tools/`, or any lane sets that. The coupling was documented
nowhere.

Removed: the 4th parameter of `makeFineLevelSetTopologyWGSL`, the interpolated
`const DELTA_NEIGHBOR_QUERY`, and the 8-line `if(DELTA_NEIGHBOR_QUERY){…}` arm
— 638 bytes of generated WGSL, all of it statically dead. The full text diff of
the emitted module is exactly those two regions and nothing else; the surviving
`interfaceNeighborRadii` arms (the `DELTA_RADIUS_MASK` mask read and the
O(producers) fallback at what is now `:2435+`) are byte-identical. The
`DELTA_RADIUS_MASK=0` ablation of §3's Q5 keeps its control arm.

One test went with it: `tests/webgpu-octree-fine-levelset-bricks.test.ts`
*"opt-in fine delta neighbor query uses the producer's dense changed marker"*
— it was the only caller passing four arguments, and it correctly set
`deltaRadiusMask=false` itself, which is how the coupling was discovered.
`producerChangedContains` survives: it still has three live callers
(`:2051`, `:2384`, `:2482`).

### 2.5 `FLUID_STRUCTURED_IMAGE_IDENTITY_CARRY` — its consumers are never encoded

`lib/webgpu-octree-structured-velocity-gpu.ts:133`, written to uniform word 43,
read once at `:780`:

```wgsl
if(identity){ publishExactRowDispatch(0u,0u); publishExactRowDispatch(3u,0u);
  if(p.imageIdentityCarryEnabled!=0u){ publishExactRowDispatch(21u,0u); publishExactRowDispatch(24u,0u); } }
```

**Evidence.** Records 21/24 are byte offsets 84/96 of the structured class
dispatch buffer, and their *only* consumers in the tree are
`lib/webgpu-octree-spgrid-vcycle.ts:2072-2079` (`buildAccurateOperatorRows`,
`buildAccurateAdjointRows`). Those two dispatches sit inside
`if (this.hierarchicalExecutorCompiled) {` at `…spgrid-vcycle.ts:2027`, and:

- `hierarchicalExecutorCompiled = options.compileHierarchicalExecutor !== false`
  (`…spgrid-vcycle.ts:1427`);
- the **only** production construction of `WebGPUOctreeSPGridVCycle` is
  `lib/webgpu-octree.ts:2768`, and it passes **`compileHierarchicalExecutor: false`**
  at `:2775`. Every other construction site is in `tests/`.

So in production the two image compiles are never encoded, and word 43 has no
observable effect. The flag is dead unless someone re-enables the hierarchical
executor.

### 2.6 `FLUID_SPGRID_PERSISTENT_IMAGES` — same gate

`lib/webgpu-octree-spgrid-vcycle.ts:1228`. Both of its effects —
`imageBanks = persistentImageCarry ? 2 : 1` (`:1645`) sizing
`accurateOperatorRows`/`accurateAdjointRows`, and the `persistentImageCarry`
pipeline-override constant — live entirely inside the
`if (this.hierarchicalExecutorCompiled) {` block that spans `:1615`–`:1731`.
Production-dead by §2.5's evidence. It is *also* a refuted arm on its own terms:
the comment at `:1647-1650` records that the remap "was slower than rebuilding
compact live rows on the large lane".

**The same gate makes `FLUID_SPGRID_DIRECT_BY_CHASE` (`:1213`) and
`FLUID_SPGRID_ADJOINT_BY_CHASE` (`:1220`) production-unreachable too** — their
pipelines are created at `:1687`/`:1697`, inside the block. They remain useful
as differential-harness switches (the harness constructs the V-cycle itself), so
they belong in bucket 1, not here; but nobody should expect them to do anything
in a benchmark run.

### 2.7 `FLUID_OCTREE_FACTOR1_PREDICTED_SOLVE_TAIL` — unwired, but **KEPT: wire it** (2026-08-03 re-audit)

`lib/octree-solve-tail-policy.ts:185`.

**Evidence, re-verified.** `octreeFactorOnePredictedSolveTailEnabled` and the
function it feeds, `selectOctreeFactorOneEncodedSolveTail` (`:197-262`, five
documented refusal reasons), have **no non-test caller anywhere** in `lib/`,
`tools/`, `components/`, or `app/` — the only references outside the defining
file are `tests/octree-solve-tail-policy.test.ts`. Production sets the tail
unconditionally from `planOctreeSolveTail`'s hard-coded
`OCTREE_SOLVE_TAIL_MAXIMUM_ENCODED_OUTER_ITERATIONS = 10`
(`lib/webgpu-octree.ts:1607`, consumed at `:2068`, `:2965`, `:3937`, `:5059`).

**It is not a dead feature — it is the landed half of an active, quantified plan
item.** `docs/POWER_LIQUIDS_COARSE_5X_WALLCLOCK_PLAN.md` §1.3 fallback tier item
**(a) "Stop encoding the dead iteration tail"** (lines 149-165) specifies this
selector almost verbatim: an encode-side, stale-safe predicate fed by "the
**previous advance's published iteration count + safety margin** (predict high,
clamp to the envelope on any topology change) — never a lowered cap, and the GPU
residual gate remains the convergence authority". That is
`observation.publishedIterationCount + OCTREE_FACTOR1_PREDICTED_SOLVE_TAIL_SAFETY_MARGIN`,
the `topology-change` refusal, and the `:191-195` doc comment, respectively.
The prize it quantifies on the coarse-only mini lane: the profiled advance
executed 6 of 10 encoded outer iterations (the 500-advance control, 4), so
**4 dead applies ≈ 388 zero-workgroup dispatches ≈ 31% of lane launches**.

**That prize is dead on this branch (2026-08-03).** It is hierarchical-MGPCG
accounting, and the hierarchical executor is gone:
`compileHierarchicalExecutor: false` is hard-coded on the only production
`WebGPUOctreeSPGridVCycle` construction (`lib/webgpu-octree.ts:2790`) and
test-locked (`tests/gpu-initialization.test.ts:153`). The persistent kernel runs
the entire outer loop **inside one workgroup**, with
`storageBarrier()`/`workgroupBarrier()` where the dispatch boundaries used to be
(`…persistent-mgpcg.wgsl.ts:1236`), and it already breaks out on convergence
(`:1257`, `:1274`). The `accountZeroAll`/`accountZeroRemaining` counters at
`:1004-1011` say it in their own comment — *"The persistent path has no indirect
outer records to zero; these keep the GPU-authored `zeroedDispatches` accounting
word identical to the hierarchical run so a lockstep A/B compares the full
control record."* They **simulate** the old zeroed-dispatch count. There are no
dispatches there to remove.

Consequently a lowered `encodedOuterIterations` removes no dispatch and shortens
no converged solve. On the persistent path it moves exactly one thing:
`p.shape.x`, the non-convergence trip at `:1066-1068` — i.e. it *tightens a
fail-closed threshold*. It cannot even do that per step: the loop trip count is
a WGSL literal baked at shader build (`octreePersistentMGPCGWGSL`'s
`maximumIterations`, `…persistent-mgpcg.wgsl.ts:136`) and the params buffer is
written once, in the constructor (`…persistent-mgpcg.ts:495`), from
`this.solveTailPolicy.encodedOuterIterations` fixed at `webgpu-octree.ts:1607`.

**Verdict: still KEEP the selector as written spec, but it is not a launch win
and must not be queued as one.** Two blockers precede any revival:

1. `selectOctreeFactorOneEncodedSolveTail` returns `not-factor-one` unless
   `globalFineLevelSetFactor === 1` — every shipping mini/dam lane runs factor 4.
2. It returns `non-adjacent-history` unless `observation.step + 1 === nextStep`,
   but the browser polls `readStats` on a **250 ms cadence**
   (`lib/webgpu-renderer.ts:2042`), so a step-adjacent observation never
   arrives and the selector would refuse on every call.

The plumbing gap is still real: `lib/physics-step-program.ts` declares the P0.5
launch-shape carve-out in its driver contract (`:127`) and still has
`predicates: Object.freeze([])` (`:137`), and nothing captures an
`OctreeFactorOneSolveTailObservation`. That work only becomes worth doing if the
hierarchical executor returns or the coarse-only factor-1 track ships.

### Summary of §2

| flag | verdict | what goes with it |
|---|---|---|
| `FLUID_OCTREE_AIR_SUPPORT_COMPACT_FINE_DEMAND` | **DELETED** 2026-08-03 | 1 predicate, 1 parameter, 1 `= true` local, 2 constant folds; WGSL byte-identical |
| `FLUID_OCTREE_AIR_SUPPORT_COMPACT_FINE_CELLS` | **DELETED** 2026-08-03 | 1 predicate; parameter kept (defaults to `fineSlot !== undefined`). WGSL rename **not** done — see §2.2 |
| `FLUID_COARSE_PHI_BFECC` | **KEEP** (§2.3) — the coarse-only plan's Stage 1/2 needs it; deleting frees no GPU resource today | — |
| `FLUID_FINE_DELTA_NEIGHBOR_QUERY` | **DELETED** 2026-08-03 | 638 bytes of dead WGSL, 1 interpolated const, 1 parameter, 1 test |
| `FLUID_STRUCTURED_IMAGE_IDENTITY_CARRY` | **KEEP** — Bet 3.2 wants the hierarchical path A/B-able | 1 predicate, 1 uniform word, 1 WGSL `if` |
| `FLUID_SPGRID_PERSISTENT_IMAGES` | **KEEP** — same gate, same bet | 1 field, `imageBanks`, 1 override const, `persistentImagePredecessor` |
| `FLUID_OCTREE_FACTOR1_PREDICTED_SOLVE_TAIL` | **KEEP: wire it** (§2.7) — it is 5×-plan §1.3(a) | — |

Of the original seven, three were free deletions (§2.1, §2.2, §2.4); four are
wanted by a live plan track. The re-audit's lesson: "unreachable today" and
"dead" are different claims, and on a branch this active the gap between them is
usually two days of someone else's work.

---

## 3. The A/B queue (bucket 4)

Eight flags are default-OFF with a real implementation. Three of them are
already refuted or superseded and should never be queued; five are live
candidates. Ordered by expected value per GPU-minute.

### The lanes (four-cell matrix, per plan §2)

| | small scene | large scene |
|---|---|---|
| **still** | `--lane=hydrostatic-tiny` (`hydrostatic-power-two-level`, 16³, 240 steps) | `--lane=large-hydrostatic` (`large-power-hydrostatic`, 64×20×64, 240 steps) |
| **churn** | `--lane=mini` (`minimal-power-dam-break`, 16³, 500 steps) | `--lane=large` (`large-power-dam-break`, 64×20×64, 500 steps) |

All four are defined in `tools/power-dam-lane-environment.ts:13-90` and wired to
npm scripts at `package.json:78-81`. `tools/benchmark-power-dam.ts:88-89` sets
`WEBGPU_NODE_MODULE` and the Metal backend itself, so the direct `node` form
below is self-contained.

**Both large cells now capture (2026-08-03), with one caveat each.**
`--lane=large-hydrostatic` runs 240 steps clean (143.6 ms/adv,
`artifacts/measurement-floor/large-hydrostatic-240.json`, `clean: true`) — the
old restriction bootstrap loop is gone. `--lane=large` **dies at t=0** with
*"Initial sparse authority cold-topology published no liquid-row frontier"*, and
that is a **lane-table bug, not a physics red**: the smoke catalog applies
`largePowerDamOverrides` (`lib/scene-webgpu-smoke-catalog.ts:349-355`,
`pressureRowCapacity: 8_192` and `globalFineLevelSetMaximumBricks: 32_768`) and
the benchmark lane table sets neither. Work around it with

```sh
FLUID_PRESSURE_ROW_CAPACITY=8192 node --import tsx tools/benchmark-power-dam.ts \
  --lane=large --steps=200 --artifact=…
```

which runs green through 200 steps (0 validation errors) — but the artifact then
reports `clean: false` with `FLUID_PRESSURE_ROW_CAPACITY` as its contaminant, so
it is not a baseline. **`globalFineLevelSetMaximumBricks` has no env override at
all** (`lib/methods/octree.ts:58-62` reads it from authored method values only),
so the lane cannot express its scene's authored brick capacity even in
principle. The real fix is in `tools/power-dam-lane-environment.ts`. Past step
200 the lane still enters the alternating empty-band/rejected-rebuild regime
around step 413; that is a separate, genuine physics question (air-support
seeding of the settled thin film). **Do not treat a red large lane as a null A/B
result — and name which of the three configurations you ran.**

### The protocol (plan §5 + `POWER_LIQUIDS_ULTIMATE_M1MAX.md:191-198`)

Because these are env toggles on one binary, an A/B needs **no second worktree**
— which removes the biggest source of the blocked-design error that doc warns
about. Interleave in one session:

```sh
mkdir -p artifacts/cutover
L=mini                       # or hydrostatic-tiny | large | large-hydrostatic
F=FLUID_FINE_TOPOLOGY_INDIRECT_ASSIGN

# --- A/A first. This must be ~zero before any A/B means anything. ---
node --import tsx tools/benchmark-power-dam.ts --lane=$L --artifact=artifacts/cutover/$L-AA-0.json
node --import tsx tools/benchmark-power-dam.ts --lane=$L --artifact=artifacts/cutover/$L-AA-1.json

# --- A/B, 4 interleaved rounds, paired median of per-round deltas. ---
for r in 0 1 2 3; do
  node --import tsx tools/benchmark-power-dam.ts --lane=$L --artifact=artifacts/cutover/$L-A-$r.json
  env $F=1 node --import tsx tools/benchmark-power-dam.ts --lane=$L --artifact=artifacts/cutover/$L-B-$r.json
done
```

Discard rounds where either arm exceeds 1.3× its own minimum. Never compare
across sessions. Never run a second WebGPU client concurrently — per plan §5 it
silently changes physics.

#### The environment record (new 2026-08-03 — read this before comparing anything)

C7 below was caused by two artifacts that described **different simulations**
and said nothing about it. Both benchmark artifacts now carry a resolved
`environment` record, produced by `tools/power-dam-run-environment.ts` and
attached to the diagnostic artifact (now `schemaVersion: 2`) and to the octree
regression artifact. Fields, and how to use them:

| field | use |
|---|---|
| `resolved` | every `FLUID_*`/`WEBGPU_*` variable the run actually received. This is the run's identity; the lane name is not. |
| `inherited` | which of those came from the **shell** rather than from the lane definition. A non-empty `inherited` means someone's exported variable is in your measurement. |
| `overridden` | lane values the command line replaced. |
| `contaminants` | wall-affecting knobs sitting off their measurement-clean value, each with `{name, resolved, clean}`. A tripwire or audit flag here can cost ~27% on its own. |
| `clean` | `true` only when `contaminants` is empty. **A `clean: false` arm is not a baseline.** |
| `comparisonKey` | digest over the scene + solver configuration. **Two artifacts with different `comparisonKey`s are not comparable, full stop** — that is exactly the leaf-2-vs-leaf-32 trap, and the key changes when the leaf changes. |

Worked example already in the tree:
`artifacts/measurement-floor/mini-240-clean-a.json` (leaf 32,
`comparisonKey 752e800d…`, `clean: false`) versus
`artifacts/measurement-floor/mini-240-leaf2.json` (leaf 2,
`comparisonKey 1d87d090…`, `clean: true`). Different keys — so the 245.3 vs
211.1 ms/adv difference is a **deliberate A/B on the leaf axis**, not an A/A,
and the `clean: false` arm's contaminant list must be read before either number
is quoted.

**Leaf size is now an explicit axis**, not an accident:

```sh
node --import tsx tools/benchmark-power-dam.ts --lane=mini --leaf-size=32 \
  --artifact=artifacts/cutover/mini-leaf32.json
node --import tsx tools/benchmark-power-dam.ts --lane=mini --leaf-size=2 \
  --artifact=artifacts/cutover/mini-leaf2.json
```

`--leaf-size=` takes a power of two and throws otherwise. Hold it fixed for any
A/B that is not about discretization, and vary only it when it is.

**Gate commands.**

- *Gate A (bit-exact)* — run both arms and diff the readback streams:
  `npm run test:webgpu:symmetric-expansion` (250 steps, factor 1, checkpoints
  every step) and `npm run test:webgpu:minimal-power-dam-break` (500 steps,
  checkpoints every 0.1 s). The D4 window in
  `docs/SYMMETRIC_EXPANSION_ORACLE.md:88` must stay at step 68 / spread 0.
- *Gate B (physics)* — the above plus
  `npm run test:webgpu:large-power-dam-runtime` (the `runtime-150` lane,
  `lib/scene-webgpu-smoke-catalog.ts:1113`, volume drift ≤ 0.01) and the staged
  audits `npm run test:webgpu:symmetric-expansion:{one,two,three}-step`
  (`FLUID_SYMMETRY_STAGE_AUDIT=1`).
  **Do not use the symmetry audit as the control for
  `FLUID_STRUCTURED_BOUNDARY_ADVECT_FLAT`** — see §4.

### Results so far (2026-08-03)

Interleaved on the `mini` lane, 120 steps, 3 rounds, paired median of per-round
deltas. **A/A noise floor 5.54 ms**; control median **251.4 ms/advance**. A
result inside the noise floor is inconclusive, not null.

| arm | Δ ms/adv | verdict |
|---|---:|---|
| `FLUID_OCTREE_SECTION43_SHELL_DEPTH=4` | **−25.26 (−10.0%)** | **CONCLUSIVE FASTER** |
| `FLUID_SPGRID_TOUCHED_RADIX_SORT=1` (Q2) | +2.72 | **INCONCLUSIVE** — inside the noise floor |

Two things follow.

**Q2 is not "the biggest structural prize" on mini.** It removes the dense
directory sweep and does not move the wall — which is §5's own trap #2
(*"dispatch-count deletion alone does not move the wall"*) confirmed with
numbers on the exact item the queue ranked first for structural value. **Re-rank
it below Q1 and Q3.** Caveat before retiring it: mini's `plan.brickCount` is
small, so the dense sweep it deletes is cheap there; the large lane has 20× the
bricks and may behave differently. Re-run on `--lane=large` before concluding.

**`FLUID_OCTREE_SECTION43_SHELL_DEPTH` is a physics/iteration knob, not a
cutover** (§1 bucket 2 says so), and it is the largest measured win in the set.
That is consistent with the plan §2 matrix finding that the wall tracks the
persistent solve: the §4.3 shell trip count is solve work, and shortening it
shortens the frame. It changes the preconditioner, so it needs the Gate-B
battery and a convergence check across all four lanes before it can be an
authored default — a 10% wall win bought by slower convergence on a harder lane
is not a win. Queue it as its own item, not as a flag flip.

### The queue

**Q1 — `FLUID_FINE_TOPOLOGY_INDIRECT_ASSIGN=1`** · `…fine-levelset-topology.ts:1689` · **Gate A** · lanes: `large`, then `hydrostatic-tiny`

Highest EV/minute in the set. It is a pure dispatch relocation: the same
`assignIdentityPipeline`, same bind entries, same `used` mask, moved from a
direct `ceil(maximumResidentBricks/64)` launch (`:1476`) to an indirect
`ceil(desiredCount/64)` launch after the identity fence (`:1482`). Statically
provable bit-identical: `assignDesiredPageIdentities` (`:2326-2337`) and
`finalizeDesiredPageIdentityAssignment` (`:2338-2350`) write disjoint word sets,
and the kernel early-returns on `work >= desiredCount`, so the narrower launch
writes exactly the same words. This is precisely Bet 1 residue (b) — the
`maximumResidentBricks` recurring scans.

**Corrected 2026-08-03: the win is ~2.3–4×, not 10×.** The dispatch is shaped by
`plan.maximumResidentBricks` (`Math.ceil(plan.maximumResidentBricks / 64)`,
`…fine-levelset-topology.ts:1476-1477`), and that capacity has since halved on
the large lane. At `recheck-persistent-1.json` (07-30) the lane's
`terminalCounters` recorded `fineBrickCapacity 81,920` against
`desiredFineBricks 8,126` → 1,280 vs 127 workgroups, so "10×" was right *then*.
The lane now carries an **authored** budget:
`LARGE_POWER_DAM_FINE_BRICK_CAPACITY = 32_768` (`lib/scenes.ts:73`, applied at
`:86` and `lib/scene-webgpu-smoke-catalog.ts:351`, consumed as
`options.globalFineLevelSetMaximumBricks` at `lib/webgpu-octree.ts:2199`) → 512
workgroups, against 227 for the 14,474 desired fine pages recorded in
`artifacts/scene-size-overhead/large-current/traced.log`
(`finePlan.maximumResidentBricks = 32768`, `logicalBrickCount = 81920`,
`candidate-build.activePages = 14474`). So ~2.3× at that lane state and ~4× at
the sparser one. Note the mechanism is the **authored scene budget**, not a
capacity planner change — the fine-band default still comes from
`planFluidFootprintFineNarrowBandBrickCapacity` (`webgpu-octree.ts:881-907`),
and `lib/webgpu-octree-owner-pages.ts` plans *owner* pages, a different arena.
The item stays queued: a 2.3× narrowing of a recurring capacity-shaped launch is
still the cheapest Bet-1 proof in the set, but do not budget against 10×.

Cost: ~20 GPU-minutes for A/A + 4 rounds on two
lanes. Risk: it adds one `broker.fence()`; that is a pass boundary, not a
storage round-trip, so Gate A should hold — but if the diff is non-zero, fall
back to Gate B rather than assuming contamination.

**Q2 — `FLUID_SPGRID_TOUCHED_RADIX_SORT=1`** · `lib/webgpu-radix-sort-u32.ts:23` · **Gate B** · lanes: `large` only now

> **Measured 2026-08-03 on mini: +2.72 ms, inconclusive (noise floor 5.54 ms).**
> Re-ranked below Q1 and Q3. The text below is why it *looked* like the biggest
> prize; the only remaining reason to spend GPU minutes on it is that mini's
> `plan.brickCount` is small and the large lane has 20× the bricks.

The former "biggest structural prize", and the highest variance. It removes six
dense-directory dispatches (`markCandidateBrickOccupancy`,
`rankCandidateBricks`, `scatterCandidateRankedSlots`,
`markCandidatePageOccupancy`, `compactCandidatePages`,
`linkCandidatePageNeighbours`) — of which **four** are actually capacity-shaped
(the other two run off `CANDIDATE_SCHEDULE.topologyLevels`, which is level-count
shaped; see §6) — and replaces them with live-run-shaped compact
equivalents driven by two 18-dispatch radix sorts. The author's own accounting
at `…spgrid-vcycle.ts:1824-1827` gives **net +30 setup dispatches** — and plan
§5's own trap says dispatch-count deletion does not move the wall, so judge this
on *launch volume*, not dispatch count. It adds real inter-kernel storage
round-trips (`touchedBrickKeys`/`touchedBrickHeader` → `prepareRadixSort` →
`runs`/`control` → the four compact builders) ⇒ **Gate B, never A**.

Run it in two stages: first one correctness pass with
`FLUID_SPGRID_TOUCHED_RADIX_SORT=1 FLUID_SPGRID_TOUCHED_RADIX_TRIPWIRE=1` on
`--lane=mini --steps=125` and require zero differential codes (the tripwire runs
both builders and reports mismatches); only then the clean perf A/B. Budget ~45
GPU-minutes. **Correct the plan doc first** — see §4.

**Q3 — `FLUID_FINE_TRANSPORT_STAGED_ADDRESSING=1`** · `…fine-levelset-transport.ts:45` · **Gate A** · lanes: `mini`, `large`

Stages page ids and air-owner records into workgroup memory ahead of the
terminal gather in the hot transport kernel (~40 WGSL lines, 8 workgroup
variables, zero new kernels or dispatches). The bit-exactness argument is
structural and credible — `…transport.wgsl.ts:405-410` states "No phi or flags
are staged, preserving the exact float evaluation and storage-rounding path",
and the staged payloads are `u32` page indices and integer `AirOwner` records.
Nothing verifies it. Note the ON path retains a per-call runtime escape
(`|| finePageWindowEnabled == 0u`), so it is a best-effort halo, not a hard
specialization — a null result is a plausible outcome. ~20 GPU-minutes.

⚠ Before running: the in-flight pipeline-deferral cutover deleted the
synchronous-compile `constants: { … }` literal that
`tests/webgpu-octree-fine-levelset-production-transport.test.ts:327` and `:345`
regex. Fix those two assertions first or the unit gate is already red.

**Q4 — `FLUID_COARSE_SUMMARY_INDIRECT_DISPATCH=1`** · `…coarse-summary.ts:45` · **Gate B** · lane: coarse-only tracker only

Lower EV than the plan implies, and **unreachable on every matrix lane**, so it
should not be run at all until the coarse-only track is live.

*Unreachability, confirmed:* `WebGPUOctreeCoarseSummary` is constructed at
exactly one place, `lib/webgpu-octree.ts:2825`, inside
`if (this.coarseOnlySurfaceTracking) {` (`:2819`), and
`this.coarseOnlySurfaceTracking = options.globalFineLevelSetFactor === 1`
(`:1638`). All eleven other `coarseOnlySummary` references are optional-chained
(`:2680`, `:2854`, `:4190`, `:4373`, `:4385`, `:4510`, `:6082`), so on a factor
4/8 lane the object simply does not exist. The factor default is `"4"`
(`lib/methods/octree.ts:7`, `:108`) and the construction path reads
`options.globalFineLevelSetFactor ?? 4` (`webgpu-octree.ts:2155`).

*Corrected size of the prize.* The flag re-routes **16 of the module's 17
capacity-shaped dispatches** — every `dispatch(entry, groups, record)` call with
a record (`…coarse-summary.ts:249-274`); only `resetSummary` (`:248`, no record)
is deliberately left capacity-shaped, so the encode does not become fully
live-shaped either way. Of the 16, **twelve are record 0**, and in production
`ownerDirectoryCellCapacity` **is** the domain's finest-cell count
(`lib/webgpu-octree-air-velocity-support.ts:184`, passed at
`lib/webgpu-octree.ts:2817`), so the published record equals
`ceil(domainVolume/256)` and **saves nothing on those twelve whenever air
support is published**. The real saving is (a) all twelve zeroed when air
support is *unpublished* (`:638`, `select(0u, …, airPublished())`), and (b) the
**4** dispatches on records 1/2 (`ensureSummaryPages`, `ensureSummaryRanks`,
`correctCoarseDirectory`, `finalizeSummaryEntries`) going from
`rowCapacity`/`entryCapacity` to live counts. Adds a storage→indirect round-trip
⇒ Gate B. ~15 GPU-minutes, and only worth it if the coarse-only lane is on the
critical path.

Note that the `select(0u, …, airPublished())` record 0 is now a
`capacity-indirect-args` violation in its own right
(`…coarse-summary.ts:638`): a live predicate wrapped around
`(p.domainVolume + 255u) / 256u` is still a domain-shaped launch on the step it
fires. See §6.

**Q5 — `FLUID_FINE_DELTA_RADIUS_MASK=0` (an *ablation*, not a cutover)** · **Gate B** · lane: `large`

Not a candidate to flip — it is already ON. Queue it only if you want to size
what the delta-radius mask is buying, since the OFF arm is an O(producers) scan
and nothing measures the difference. Deprioritise unless the fine-topology stage
shows up in an xctrace capture.

### Do not queue

| flag | why |
|---|---|
| `FLUID_STRUCTURED_PUBLICATION_COMPACT_PASS` (`…velocity-gpu.ts:526`) | Already measured: **−0.36 ms/advance regression** (`POWER_LIQUIDS_10X_DISCOVERY_RESULTS.md:298-303`). |
| `FLUID_STRUCTURED_RECONSTRUCTION_COMPACT_PASS` (`…velocity-gpu.ts:591`) | Already measured: neutral, ~0.06 ms within noise (same doc). |
| `FLUID_STRUCTURED_WORKSET_SPLIT` (`…structured-boundary.ts:380`) | Self-declared superseded by `FLUID_GPU_ISOLATE_PASS_LABELS` (`…structured-boundary.ts:375-378`), and a conditional no-op under it. |
| `FLUID_OCTREE_AIR_SUPPORT_RETAINED_GRAPH` (`…air-velocity-support-gpu.ts:307`) | Refuted on symmetry, with numbers: the doc comment at `:279-301` records that enabling it moves the bitwise-D4 window from **99 → 18** (velocity/pressure/rhs) and **105 → 41** (diagonal/topology) on `symmetric-expansion` at 110 steps, because `betterFace` orders equidistant seeds by a *live* magnitude that the retained refresh never re-evaluates. Same-receipt does not imply same-winner. Costs ~18% of the advance to give up; the correct fix is to re-run only the magnitude tie-break over the retained equidistant set. |
| `FLUID_SPGRID_PERSISTENT_IMAGES` | Refuted *and* production-unreachable (§2.6). |

---

## 4. Doc corrections for `POWER_LIQUIDS_SCENARIO_LEAP_PLAN.md`

Every item below is a claim in the plan that the code or the in-tree artifacts
contradict. Ordered by how much damage the error can do.

**C1 — §3 Bet 1, item 2: "`FLUID_OCTREE_AIR_SUPPORT_INDIRECT_FRONTIER_GATE`
(GPU-published wave gate, currently default-off)" (line 211-212).**
It is **default ON**: `lib/webgpu-octree-air-velocity-support-gpu.ts:343` reads
`resolved?.FLUID_OCTREE_AIR_SUPPORT_INDIRECT_FRONTIER_GATE !== "0"`, and
`encode` consumes it at `:1071`. Test-locked at
`tests/webgpu-octree-air-velocity-support-gpu.test.ts:67-75`. The A/B the plan
asks for has already been taken.

**C2 — §3 Bet 1, item 2: "The compact-demand indirect mechanism already exists
(`FLUID_OCTREE_AIR_SUPPORT_COMPACT_FINE_DEMAND`, default on) — finish the
family" (line 207-209).**
The *mechanism* exists and is unconditional; the *flag* was dead and is now
**gone** (deleted 2026-08-03, §2.1). `encode` gates the fine-demand schedule on
`fineSlot` alone. Reading this line as "there is a switch here" will waste GPU
time — there is no switch, and setting the env var does nothing because nothing
reads it.

**C3 — §3 Bet 1 status: "the multi-workgroup **stable LSD radix sort** over
touched u32 identities — `sortSparseCandidates` is a single-workgroup bitonic,
not it" is named as *the one missing primitive* (line 189-192).**
Half right. `sortSparseCandidates` **is** a single-workgroup bitonic
(`…fine-levelset-topology.ts:2030`, `@workgroup_size(256)`, no `workgroup_id`
builtin, dispatched `runIdentity(…, 1, 1, …)` at `:1523`) — that part is
correct. But the primitive is **not missing**: `lib/webgpu-radix-sort-u32.ts` is
exactly it — 4 passes of 8-bit digits, stable, genuinely multi-workgroup
(`countRadixDigits`/`scatterRadixDigits` dispatched via
`dispatchWorkgroupsIndirect` off a GPU-authored block count with a folded 2-D
`foldedBlock`), Dawn-validated against the CPU oracle
`lib/octree-spgrid-touched-directory.ts` by a real GPU test
(`tests/webgpu-radix-sort-u32.test.ts:105`, 40 000 random u32, duplicate-heavy
runs, block boundaries, three fail-closed rejections). It is wired into the
SPGrid candidate chain behind `FLUID_SPGRID_TOUCHED_RADIX_SORT`
(`…spgrid-vcycle.ts:1516-1521`, `:2020`, `:2025`). Landed in `d577016`
"feat(fluid): advance structured cutover".
**Suggested rewrite:** "landed and wired behind
`FLUID_SPGRID_TOUCHED_RADIX_SORT=1`, Dawn-validated against the CPU oracle,
awaiting the Gate-B A/B."

**C4 — §2, P0.4: "one run each with `FLUID_OCTREE_ROW_DELTA_CENSUS=1`" (line 134).**
That flag does not exist. Grep of `lib/ tools/ tests/` returns nothing; it
survives only in five doc files. It was removed in commit `9de199a`. Anyone
following P0.4 literally gets a silently ignored env var and a census-free
artifact they will believe is a census.

**C5 — §1, item 1: "Air support dispatches four kernels at
`ceil(domainVolume/256)` (`webgpu-octree-air-velocity-support-gpu.ts`,
`plan.domainVolume`), at two encode sites per advance" (lines 56-59).**
Stale, and **contradicted by the plan's own §3 status block** ("the four
`domainVolume` air-support dispatches are gone (test-locked)", line 186). In the
current file `domainVolume` appears only in capacity planning (`:629-635`,
`:149-155`) and WGSL bounds checks (`:1493`, `:1518`, `:1545`, `:1743`, `:2247`)
— there is no host `dispatchWorkgroups(ceil(domainVolume/256))`. Also: there are
**four** air-support encode call sites, not two and not three —
`lib/webgpu-octree.ts:3774`, `:4004`, `:4342`, `:4380` (re-checked 2026-08-03;
this file moves, anchor on `this.airVelocitySupport.encode(`). Two of them run
per steady-state advance, which the artifacts show directly: every Section 5
pass label now appears twice, suffixed `- topology-commit` and `- settled-fine`
(diff `baseline-mini.json` against `measurement-floor/mini-240-clean-a.json`).
That doubling is most of the +61 dispatches/advance between those two captures.

**C6 — §3 Bet 1, item 3: "support membership is still ≈ the whole air partition
(~94k face patches marched on mini…)" (lines 219-220).**
The plan's own status block already retracts the 94k figure ("the '~94k faces'
figure is retracted by the in-tree censuses", line 190) but the bet text still
carries it. The in-tree number is
`artifacts/scene-size-overhead/fresh-20260802-mini-a.json`
`terminalCounters.airSupportFaceItems = 31,584` (seed faces 17,564; support rows
1,450; cells 1,182). **The stronger argument for the corridor bet is in the same
artifact set and is not in the doc:** the *still* scene marches **more** faces
than the churn scene — `fresh-20260802-hydrostatic-tiny-a.json` records
`airSupportFaceItems = 43,776`, `airSupportSeedFaces = 30,528`,
`airSupportRows = 2,624` on a domain of the same 16³ size. Air support is
scaling with the air partition, not with change, exactly as Bet 1.3 predicts.

**C7 — §1 table (line 43-47) is four days stale relative to the baselines §2
tells you to use.** The table's mini row (80 passes / 442 dispatches / 252
indirect / **1** MGPCG dispatch) reproduces `artifacts/scene-size-overhead/baseline-mini.json`,
captured `2026-07-29T23:56Z`. The interleaved fresh capture of `2026-08-02` that
§2's status block names as the current baseline records, for the same mini lane:

| artifact | passes/adv | dispatches/adv | indirect | MGPCG disp | ms/adv |
|---|---:|---:|---:|---:|---:|
| `baseline-mini.json` (07-29, 240 steps) | 80.03 | 442.08 | 252 | **1** | 69.60 |
| `fresh-20260802-mini-a.json` (500 steps) | 80.00 | **503.02** | **328** | **26** | **241.26** |
| `fresh-20260802-mini-b.json` (500 steps) | 80.00 | 503.02 | 328 | 26 | 240.03 |
| `fresh-20260802-hydrostatic-tiny-a.json` (240 steps) | 80.00 | 499.03 | 328 | 26 | **352.14** |
| `recheck-persistent-1.json` (large, 07-30, 120 steps) | 80.05 | 470.17 | 281 | 1 | 122.67 |

Two consequences:
- §1's "The pressure *solve* is now one dispatch (`encodeSolve`); it is no
  longer the launch problem" is **TRUE everywhere, and this ledger was wrong to
  doubt it** — see the resolution below.
- §2's "today (approx)" column — mini churn "~38–44", tiny hydrostatic
  "≈ mini wall (~38–44)", large 20× dam "~38–47" — matches **no artifact in the
  tree**. The fresh A/A pairs are tight (241.26 vs 240.03, i.e. 0.5%; 352.14 vs
  352.08, i.e. 0.02%) and interleaved by capture time (10:00, 10:02, 10:05,
  10:07), so they are clean-protocol runs, not noise.

#### C7 resolution status, 2026-08-03 — **PARTIAL. Do not close this item.**

**(a) The 1 → 26 "MGPCG dispatches" step is a label-attribution artifact.
RESOLVED.** `mgpcgDispatchesPerAdvance` is computed by `mgpcgSolveDispatches`
over `audit.dispatchesByPassLabel`, keyed on the *owning stage* a pass label
resolves to, and `POWER_DAM_MGPCG_SOLVE_STAGE`
(`tools/power-dam-performance-report.ts:476-482`) deliberately folds
`SPGrid V-cycle`, `Section 4.3 preconditioner`, `SPGrid accurate A2` and
`SPGrid Section 6.3 apply` into the solve. The persistent solve is one dispatch:
`WebGPUOctreePersistentMGPCG.encodedDispatchCount = 1`
(`lib/webgpu-octree-persistent-mgpcg.ts:292-293`), and **every** capture in the
table above — including `baseline-mini.json` — records
`computePassesByLabel["Octree persistent MGPCG - whole solve in one workgroup"]
= 1`. The step came from pass *labels*, not from work: `baseline-mini.json`
carries **no** `SPGrid V-cycle` label at all, while the current captures carry
two (`… · capture plan L1 delta`, `… · candidate commit changed L1`), and with
`FLUID_GPU_ISOLATE_PASS_LABELS` off the ~24 candidate-rebuild dispatches of
`encodeCaptureDelta` + `encodeSetupCandidate` all land inside those two passes.
So 26 ≈ 1 solve + 25 SPGrid candidate rebuild.
*Recommended (not done — that file is owned elsewhere): split the metric, e.g.
`pressureStageDispatchesPerAdvance` alongside a
`persistentSolveDispatchesPerAdvance`, so it stops reading as the solve.*

**(b) The missing environment. RESOLVED.** See the environment-record section in
§3 — `tools/power-dam-run-environment.ts`, `environment` on both artifacts,
`contaminants`/`clean`/`comparisonKey`, and `--leaf-size=`.

**(c) The wall. OPEN, and larger than it looked.** The comparison was invalid in
a way nobody had recorded: commit `065219a` (2026-08-01, "fix(octree): restore
factor-one scene parity") changed `FLUID_MAXIMUM_LEAF_SIZE` in
`tools/power-dam-lane-environment.ts` from **2 → 32** on mini and **16 → 32** on
large (also 16→32 hydrostatic, 2→32 on two more lanes). Every pre-`065219a`
artifact is a different discretization from every post-`065219a` one. Measured
this session on the same 240-step mini lane at HEAD:

| capture | leaf | ms/adv | passes | disp/adv |
|---|---:|---:|---:|---:|
| `artifacts/measurement-floor/mini-240-clean-a.json` | 32 | **245.3** | 80.00 | 503.0 |
| `artifacts/measurement-floor/mini-240-leaf2.json` | 2 | **211.1** | 79.00 | 469.0 |
| `artifacts/scene-size-overhead/baseline-mini.json` (07-29) | 2 | **69.6** | 80.03 | 442.1 |

The leaf change accounts for ~14% (245.3 → 211.1). **211.1 vs 69.6 at the same
leaf size is a real ~3× regression, still unexplained**, and nothing else in the
tree is watching it. It is the single most valuable thing on this list. Bisect
it with `--leaf-size=2` held fixed and `comparisonKey` checked on both arms.

**C8 — §2, P0.2 status: "P0.2 (`large-power-hydrostatic`) exist in
`tools/power-dam-lane-environment.ts`" (lines 102-104).** True but the key is
`large-hydrostatic`, not `large-power-hydrostatic` (that is the *scene* id).
`--lane=large-power-hydrostatic` throws. npm script:
`benchmark:power-dam-large-hydrostatic`.

**C9 — §3 Bet 1 status credits commit `a0a2247`** (line 184). On the current
history that commit is `a56ddd0` "perf(power-liquids): revive the coarse-only
tracker and land Bet 1/4 scaffolding" (rebased). Anchor on the message, not the
hash.

**Claims I verified as TRUE** (so nobody re-checks them): §1's
`copyBufferToBuffer`/`clearBuffer` fence claim (`lib/webgpu-pass-broker.ts:166`,
`:181`, `:198` — every one calls `fence()`); §1's fine-topology
`maximumResidentBricks`-shaped recurring scans (the identity chain at
`…fine-levelset-topology.ts:1464-1517` is shaped by
`pageDeltaLayout.identityScanBlockWords = ceil(pageCapacity/256)` where
`pageCapacity` derives from `maximumResidentBricks` at `:984`); §1's
symmetric-expansion window 68/spread 0 (`docs/SYMMETRIC_EXPANSION_ORACLE.md:88`
— P0.5 did land); the `CANDIDATE_SCHEDULE`/`runCandidateIndirect` schedule
(`…spgrid-vcycle.ts:723`, `:1939`); `octree-spgrid-touched-directory.ts` being
CPU-oracle/test-only; and every symbol the plan names —
`encodeSetupCandidate`, `encodeSolve`, `probeCandidateSkip`,
`applyCandidateSkip`, `committedInputs`, `rowDeltaNewToOld`,
`buildCandidateLevelDeltas`, `supportPublicationValid`,
`planOctreePressureCapacity`, `planGlobalFineNarrowBandBrickCapacity`,
`globalFineLevelSetMaximumBricks`, `createLargePowerDamBreakScene`,
`marchAirSupportFacesToFixedPoint`, `octree-pipelined-pcg.ts`, the
`runtime-150` lane, the `hydrostatic-tiny` lane, and `FLUID_MAXIMUM_LEAF_SIZE=32`
on the large lane.

### One more trap, not currently in the plan

**`FLUID_SYMMETRY_STAGE_AUDIT=1` is not a valid control for
`FLUID_STRUCTURED_BOUNDARY_ADVECT_FLAT`.** In
`lib/webgpu-octree-structured-dynamics.ts`, `advect` writes
`debugAdvectionWord(p.ownerOffset, handle, …)` and
`(p.neighborOffset, handle, …)` at `:2183-2184`, addressed
`(1u-bank())*p.authorityWords + offset + handle` — **the same two words** the
dry-probe cache uses (`:2107-2108`, written at `:1964-1969`). Those debug writes
are gated on `p.padB & 2u`, i.e. `words[39]`, which is set to 63 whenever
`FLUID_SYMMETRY_STAGE_AUDIT=1` (`:434-435`, `:471-472`). Under the audit the
lane therefore clobbers its own cache to a nonzero bitcast before reading it at
`:2239`, and `rowTouchesDryCached` returns true unconditionally — the flattened
arm silently degrades to trace-all. This is read off the addressing arithmetic;
**confirming it needs a device.** Until then, gate that toggle with the
non-audit lanes.

---

## 5. Recommendation

**Superseded by the 2026-08-03 re-audit — see the revised §2 summary table.**
The original recommendation ("do the seven deletions in §2 now") was tested one
candidate at a time and did not survive. Only
`FLUID_FINE_DELTA_NEIGHBOR_QUERY` was a free deletion and it has landed.
`FLUID_COARSE_PHI_BFECC` (§2.3) and `FLUID_OCTREE_FACTOR1_PREDICTED_SOLVE_TAIL`
(§2.7) are both wanted by live plan tracks and are kept; the air-support two
remain deferred until the agent editing that file lands. Note especially that
§2.3's original claim — that the BFECC arm "makes
`lib/webgpu-octree-fine-levelset-transport.ts` look like it owns a second
advection scheme it can never run" — is backwards: the coarse-only plan intends
it to run, and the honest fix is the §2.3 reachability note, not a deletion.

**Keep every one of the 24 escape hatches in bucket 2.** The instinct to delete
them to "save a branch" is wrong here, and the evidence is in the table itself:

- Sixteen of the 24 cost **nothing at runtime**. Nine flip a single uniform word
  or override constant (`TOPOLOGY_REUSE` word 63, `STRUCTURED_IDENTITY_CARRY`
  word 42, `FINE_TRANSPORT_QUIESCENCE` float word 73, `DEEP_IDENTITY_CARRY`,
  `INDIRECT_FRONTIER_GATE`, …) — the WGSL is byte-identical either way and the
  specialized program the GPU runs is unchanged. Seven are compile-time source
  substitutions that never coexist in one binary. There is no branch to save.
- Five of the remaining eight cost one host-side `if` around a `broker.fence()`
  or a pipeline selection — not a measurable cost on a 40–240 ms advance. Only
  three own a real alternate dispatch family (`CHANGED_FRONTIER`'s dense march,
  `OCTREE_COARSE_REDISTANCE`'s 8 eikonal sweeps, `FINE_VOLUME_CADENCE`'s ~10
  skipped dispatches), and all three are code you would have to write again the
  first time a result looks wrong.
- Against that: these are the **only** differential instruments the team has.
  `FLUID_FINE_JFA_B4_ADDRESSING` is the one flag whose exactness claim is
  actually proven in-tree (three CPU tests: 4-line diff bound, byte-identical
  float expression body, control-flow/barrier invariance) — and it is proven
  *because* both arms still exist as materialized sources. Delete the arm and
  you delete the proof. `FLUID_SPGRID_DIRECT_BY_CHASE` is what makes
  `tests/webgpu-octree-operator-image-differential.test.ts` possible at all.
  Plan §5's own list of traps is a list of things that were only caught because
  a control arm existed.

**Two exceptions where deletion is the right call.** Delete
`FLUID_OCTREE_AIR_SUPPORT_CANDIDATE_BOUND_RESET`: its `=0` path is not an
alternative behaviour, it is the bug that commit `3348b8a` fixed — a stale fine
block inflating the candidate extent, unscattered tail records, and a declined
publication. An escape hatch to a path that corrupts a publication is not an
A/B; it is a loaded gun with a doc comment. (The second exception named here was
`FLUID_OCTREE_FACTOR1_PREDICTED_SOLVE_TAIL`; the 2026-08-03 re-audit resolves
that "either/or" in favour of **wire it** — it is 5×-plan §1.3(a), worth ~31% of
coarse-only lane launches. See §2.7.)

**Fix the measurement floor before the queue.** The single most valuable thing
on this list is still not a cutover: it is finishing C7. Two of its three parts
are now closed — the "26 vs 1 MGPCG dispatches" contradiction was a pass-label
attribution artifact (the solve is 1 dispatch in every capture), and artifacts
now carry a resolved environment with `clean`/`comparisonKey`. The third part
got **worse** on inspection: the leaf-size change in `065219a` invalidated every
cross-date comparison, and correcting for it leaves a **real ~3× mini
regression** (211.1 ms/adv at leaf 2 today vs 69.6 on 07-29 at the same leaf)
that nothing in the tree is watching. Every accept/reject threshold in the
program is denominated in those numbers. **Bisect that first**, with
`--leaf-size=2` pinned and `comparisonKey` checked on both arms.

**Then run Q1.** `FLUID_FINE_TOPOLOGY_INDIRECT_ASSIGN` is the only queued item
that is both statically provable and directly on the Bet 1 critical path
(capacity-shaped launch → live-count launch, **~2.3–4×** narrower on the large
lane — the 10× figure is corrected in Q1 above). It is a Gate-A change with a
written-down proof and no test. If it lands clean, it also validates the pattern
for the rest of the `maximumResidentBricks` residue. Q2
(`FLUID_SPGRID_TOUCHED_RADIX_SORT`) is the larger prize but needs the tripwire
correctness pass first and a Gate-B battery after; C3 is now corrected in the
plan, so the next reader will not re-derive a primitive that already exists and
is already GPU-tested.

**Finally, note what the whole table says about confidence.** Not one of the 39
non-instrument toggles has its non-default arm executed by a GPU test. Every
"a zero restores the former behaviour exactly" in this tree is a claim resting
on a CPU predicate assertion and a regex. That is fine for an A/B lever — you
find out when you run it — but it means **no escape hatch in this repo should be
trusted as a rollback path without first running it.** If a landing goes wrong
at 2 a.m., `FLUID_X=0` is a hypothesis, not a fix.

---

## 6. The Bet-1 source gate, 2026-08-03

`npm run audit:octree-production-source` now scans **61** sources and reports
**119** violations (previously 108 in 59). The delta is entirely new *visibility*
— no dispatch changed.

| change | effect |
|---|---|
| Scope widened past the octree-prefix filter | `lib/webgpu-fluid-brick-residency.ts`: **6** `capacity-dispatch`. `lib/webgpu-sparse-brick-topology-mutation.ts`: **clean**. |
| New rule `capacity-indirect-args` | **4** hits, all real (below). |
| Capacity vocabulary `\w*Capacity` → `\w*[Cc]apacity` | +5, of which 4 are brick-residency launches spelled `this.capacity`. |

**Brick residency, the newly-scoped module.** All six are genuine and all recur:
four `dispatchWorkgroups(bricks)` at `:1403-1407` where
`bricks = Math.ceil(this.capacity / 64)` (`:1401`); one
`emitTopologyTiles` at `ceil(this.tileCapacity / 64)` (`:1412`); and one
`commitFineSeedCandidates` at
`ceil(max(worklistByteLength, tileWorklistByteLength, stateBytes, tileStateBytes) / 4 / 64)`
(`:1477`) — a launch shaped by the **byte size of its own arenas**, which is
O(capacity) in the most literal form the rule can express. None of them had ever
been inside the gate.

**The third blindness.** The gate exempted `dispatchWorkgroupsIndirect` on the
reasoning that "an indirect launch is shaped by whatever the GPU published into
the args buffer, which is the compliant form". That is only true if the
*publisher* used a live count.

- `webgpu-octree-spgrid-vcycle.ts:3884-3886` — `prepareCandidateSchedules`
  (`:3843`) sets `brickItems = p.totals.y`, `logicalPageItems =
  physicalPageItems = p.totals.z` (`:3878-3879`) and publishes them as candidate
  schedule records 4/5/6. The host writes `p.totals = [totalLevelSlots,
  plan.brickCount, plan.pageDirectoryBytes/4, …]` (`:1598-1599`), and
  `plan.brickCount` is a pure O(domain) sum over levels (`:887-891`). **Four**
  recurring launches consume those records: `markCandidateBrickOccupancy` and
  `scatterCandidateRankedSlots` off `.bricks`, `markCandidatePageOccupancy` off
  `.logicalPages`, `linkCandidatePageNeighbours` off `.physicalPages`
  (`:1971-1982`). The other two of the six the Q2 entry names —
  `rankCandidateBricks`, `compactCandidatePages` — run off `.topologyLevels`,
  which is level-count shaped and therefore fine.
- `webgpu-octree-coarse-summary.ts:638` — record 0's
  `select(0u, (p.domainVolume + 255u) / 256u, airPublished())`. Factor-1-only, so
  it does not fire on a matrix lane, but it is the same shape.

The rule (`capacityIndirectArgumentViolations` in
`lib/webgpu-octree-work-accounting.ts`) treats a published item count as
host-authored when it derives from the capacity vocabulary **or from any uniform
block** — a uniform is host-written and the host may not read back a live count
(`hostSchedulingUsesReadback: false`), so a uniform-derived count is an
allocation-time constant by construction. That is what catches `p.totals.y`,
which no capacity vocabulary contained. It suppresses on a storage/atomic read
(GPU-published), on a function parameter (judged at the call site), and on
literals; it reads *through* `select`'s predicate, because a live predicate
around a capacity is still a capacity on the step it fires. Pinned by four
regressions with negative cases in
`tests/webgpu-octree-work-accounting.test.ts`.

**This is the gate's third blindness, and a green result from it has twice been
quoted as evidence while wrong.** `docs/BET1_DISPATCH_SHAPE_AUDIT.md` now
carries a "what the gate still cannot see" list — host-written indirect args,
mixed live/capacity assignment paths, capacity laundered through an
unrecognised accessor name, per-invocation work, out-of-scope modules, and
deliberate exemptions like `resetSummary` that are reported identically to
accidents. Read it before quoting a clean run.
