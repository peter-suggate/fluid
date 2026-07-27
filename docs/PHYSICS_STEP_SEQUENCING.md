# Physics step sequencing: the formal procedure and the two drivers

Status: implemented on `perf/structured-cutover` (2026-07-27), amended by
P0.4/P0.5 of `docs/POWER_LIQUIDS_ULTIMATE_M1MAX.md` (items A4 and A5). This
document is the contract; `lib/physics-step-program.ts` is the machine-readable
form, persisted at `docs/physics-step-program.json` and enforced at runtime.
§6 is a specification for work that has not landed.

## 1. Why this exists

The mini-dam-break browser runs diverged from Dawn in ways that took weeks to
forensically unwind (docs/STRUCTURED_DAM_BREAK_HANDOFF.md). The root pattern,
confirmed by a full mapping of both drivers, was never the step itself — both
drivers execute the same `advanceTo` and the octree lane's topology adoption
is entirely GPU-encoded — but the *procedure around the step*:

1. **Scheduling depended on instrumentation.** `presentationHasPhysicsSlack`
   and `presentationPhysicsQueueDepth` consumed `info.physicsTrace`, which
   exists only while the profiler runs. Toggling profiling changed whether
   physics was admitted before or after presentation encoding and moved the
   in-flight advance ceiling between the bootstrap 2 and a measured 1..8.
   Profiling literally changed the simulation's driver.
2. **Diagnostics raced the pipeline.** `readStats` submitted independent
   copy encoders against live control buffers while up to eight advances were
   in flight. A legal mid-pipeline sample read cleared counters ("0 live
   pressure rows") and — worse — the "authority lag" metric compared a
   queue-fenced GPU epoch word against the *host* fine-generation mirror read
   at map-completion time, so measured lag was inflated by exactly the
   pipeline depth profiling had just changed. Phantom 11-generation lag
   readings and the WATER-UPDATE-REJECTED presentation rollback both grew
   from this: observation was entangled with scheduling, and the observed
   "damping" tracked the profiler state.

## 2. The canonical step (what both drivers execute)

One advance of the octree structured lane is ONE command buffer, ONE
`queue.submit`, encoded by `WebGPUUniformEulerianSimulation.advanceTo`:

| # | stage id | consumes | produces |
|---|----------|----------|----------|
| 1 | `ready-topology-flip` | previous step's candidate epoch (carried) | accepted epoch + structured velocity/boundary receipt, Section 5 air support |
| 2 | `surface-transport` | accepted epoch/velocity, previous projected velocity (carried) | fine + coarse level set, fine generation |
| 3 | `pressure-projection` | accepted receipt, level sets | projected velocity, pressure, solve stats, projection energy |
| 4 | `inactive-topology-candidate` | level set, projected state | next step's candidate epoch (validated or poisoned) |
| 5 | `rigid-exchange` (optional: bodies present) | projected velocity, pressure | rigid state |
| 6 | `sparse-brick-world` | level set, projected velocity | render world |
| 7 | `step-snapshot` | accepted receipt, worklist header, stats, epoch state, SPGrid level delta, air-support failure words, transport governor | step-coherent diagnostics record |

Key structural facts, verified in code:

- The active epoch is immutable for the whole step; adoption happens only at
  stage 1 from the previous tail's candidate, and a poisoned candidate leaves
  the accepted receipt on the old epoch (stale-but-valid: transport keeps
  consuming the old velocity bank — the genuine damping mode). Recovery is
  re-attempted every step by construction, with **no host round-trip**
  (`encodeInactiveTopologyCandidate` → `encodeReadyTopologyFlip`).
- `hostSchedulingUsesReadback: false` — nothing inside the step consults a
  GPU readback.
- For the power scenes `fixedDt_s === maxDt_s`, so the UI's collapsed target
  clock yields dt = maxDt on every advance, identical to Dawn's checkpoints.
  Dawn's free-run is bit-identical to lockstep (phiBitXor-verified), so
  *pipeline depth does not change physics* — only observation ever did.

The machine-readable program (ids, reads/writes, carried reads, driver
contract) lives in `lib/physics-step-program.ts`; `advanceTo` records the
stage ids it actually encodes and validates them against the program after
every submit (`StepSequenceRecorder`). A deviation publishes
`info.stepSequenceDeviations`, raises the `step-sequence-deviation` stability
flag, and logs `[step-sequence]` once. This holds for BOTH drivers, traced or
untraced — the executable form of "the exact same sequence always occurs".

## 3. The driver contract

Also serialized in the program JSON. A driver (Dawn harness, web renderer, or
any future one) may decide only:

1. **when** to request an advance,
2. **how many** advances are in flight, and
3. (since P0.5) the **launch shape** of an advance, within the carve-out in
   §3.1 — never the values the physics reads.

Everything else is the solver's. Specifically:

- one advance = one command buffer = one submit; drivers never split steps;
- the target clock advances in whole fixed steps (controller collapse /
  harness checkpoints), so dt sequences are identical across drivers;
- diagnostics consume the end-of-step snapshot record; no consumer may race
  live solver buffers with an independent readback encoder;
- scheduling inputs must exist identically with instrumentation on or off;
- instrumentation must be sequence-neutral (boundaries ride real passes; a
  traced advance submits the same command graph as an untraced one).
  **Known false today**: tracing threads `productionBoundary` through
  `webgpu-octree.ts`, inserting extra fences and swapping the `PassBroker` at
  ~11 sites. Either the hooks become no-ops that ride existing pass breaks or
  this clause needs amending; it is recorded here rather than silently
  believed (docs/POWER_LIQUIDS_ULTIMATE_M1MAX.md A5, closing note).

### 3.1 Launch-shape carve-out (P0.5)

The original text said drivers "are forbidden from affecting step content".
That was too strong in one direction and too weak in another. The precise
rule:

> A driver may delete an encode it can prove does **zero work**. It may never
> delete a live encode, and it may never change a value the physics reads.

Mechanically:

1. A stage that a driver may delete declares `condition: <predicate-id>` in
   `lib/physics-step-program.ts`, and the predicate is declared in the
   program's `predicates` list. `optional` keeps its old meaning (a condition
   the driver does **not** control, e.g. rigid bodies in the scene); the two
   are different things and are validated separately.
2. `StepSequenceRecorder.recordCondition(id, value, sourceStep)` states, per
   step, which predicates the driver evaluated, what they evaluated to, and
   **which snapshot step the evidence came from**. The sequence check accepts
   a conditional stage's absence only when that step recorded its predicate
   true. An unrecorded predicate reads false: silence never authorizes
   deleting an encode.
3. Every predicate must be **conservative under staleness**. Predicates are
   read from a snapshot that is up to `maxPendingAdvances` steps old, so the
   only sound shape is one where a stale read makes the driver encode *more*,
   never less. "The last observed step did zero solve iterations" is
   conservative for the settled tier only when combined with a synchronously
   known host-input test (see B5a) — entering the tier late merely costs
   savings; exiting is immediate.
4. Every predicate declares a `zeroWorkWitness`: the GPU counter in that
   step's **own** snapshot record that must read zero. This is what makes the
   rule enforceable rather than aspirational.

`hostSchedulingUsesReadback: false` still holds for *values*: nothing inside
the step consults a readback to decide a physical quantity. What P0.5 permits
is a readback deciding whether an encode that would have written nothing is
emitted at all.

### 3.2 Predicted work and the lag-k check (P0.5)

Shaping an encode from a stale observation is only safe if the shape is
audited against reality. Each step therefore records a **prediction** —
`PhysicsStepPrediction` in `lib/physics-step-program.ts`:

- the outer solve budget the command buffer actually encoded;
- the conductor-conditional stages it deleted;
- for each, the predicate value and the snapshot step it was read from.

Predictions are held in a `PhysicsStepPredictionLedger` sized past the
pipeline depth. When step N's own snapshot record maps — k steps later, k =
in-flight depth — `physicsStepPredictionFailures` compares the prediction
against the GPU's counters for that same step: executed solve iterations and
convergence, the topology-epoch flip verdict, the per-level SPGrid setup
deltas, the fine active-brick count, the air-support error word. A counter
whose record segment was not copied is `undefined`, and `undefined` is never
evidence of zero work — an absent witness fails the check exactly like a
non-zero one.

A mismatch means a predicate deleted live work. It is a **hard failure** in
harness lanes and an **ALERT** on the browser stability card
(`step-prediction-mismatch`, from `info.stepPredictionFailures`). It is
latched per run: one dishonest step invalidates the predicate, so later
conforming steps must not clear it.

### How each driver satisfies it

| concern | Dawn (`tools/run-webgpu-smoke.ts`) | Web UI (`lib/webgpu-renderer.ts`) |
|---|---|---|
| advance trigger | loop to `FLUID_TARGET_S`, retry on reject | rAF admission + fence-completion refills |
| in-flight bound | fence every `FLUID_AWAIT_EVERY_STEPS` (30) | `presentationPhysicsQueueDepth` (1..8) |
| step cost input | n/a (throughput-bound) | `GPUAdvanceWallEstimator` (always on) |
| step-coherent reads | audit snapshot copies after each accepted step | `StructuredStepSnapshotRing`, same ABI |
| dt | `maxDt` exactly | `maxDt` exactly (fixed-step collapse) |

## 4. What changed in this pass

1. **`lib/gpu-advance-pacing.ts` — `GPUAdvanceWallEstimator`.** Scheduling
   (slack test + queue depth, and now presentation cost too) reads a
   wall-clock estimate derived from the per-advance and per-presentation
   `onSubmittedWorkDone` fences the renderer already registers: zero GPU
   work, exists in every mode. `observedGPUAdvanceTime_ms` (the hardware
   trace total) is telemetry-only. Toggling the profiler can no longer change
   admission ordering or pipeline depth. Side benefit: uninstrumented
   sessions previously never satisfied the slack test and sat at bootstrap
   depth 2 forever; they now get the measured depth and pre-presentation
   admission — the "send work to the GPU earlier" utilization win, in both
   modes.
2. **`lib/structured-authority-audit.ts`** — the Dawn audit record ABI
   (`STRUCTURED_GENERATION_AUDIT_SNAPSHOT`) and validators, promoted
   from `tools/webgpu-smoke-structured-audit.ts` (now a re-export shim) so
   browser and harness share one layout and one decode. **P0.4 widened the
   record from 336 B to 808 B**:

   | offset | bytes | segment | source |
   |---|---|---|---|
   | 0 | 24 | structured velocity control | `structuredVelocity.control` |
   | 24 | 28 | structured boundary control | `structuredBoundary.control` |
   | 52 | 28 | fine workset publication header | `globalFineLevelSetSource.worklist` |
   | 80 | 64 | MGPCG control (executed / converged / error) | `pipelinedMGPCG.control` |
   | 144 | 64 | fine volume control | `globalFineVolumeA.control` |
   | 208 | 128 | structured projection energy | `structuredProjectionEnergyStats` |
   | 336 | 64 | topology-epoch state (16-word `Epoch`) | `WebGPUOctreeTopologyEpoch.state` |
   | 400 | 384 | SPGrid `levelDelta`, 12 reserved levels × 8 words | SPGrid V-cycle |
   | 784 | 8 | air-support `scratch[0..1]` failure words | air-support producer |
   | 792 | 16 | fine-transport `governor` words 0–3 | `WebGPUFineLevelSetTransport.governor` |
   | | **808** | stride | |

   Every copy still rides the step's own encoder as its last commands, so
   each word is that step's final value. Appending a segment is legal only
   while that holds — `step-snapshot` must stay the final program stage, and
   a new source must be appended after all of its producers.

   Two decode rules the record now enforces:

   - the fine **active-brick count is worklist header word 1**. Word 0 is the
     generation; decoding it as a count reported a monotonically rising
     generation as band occupancy and hid the `0xFFFFFFFF` capacity-overflow
     sentinel, which no-ops the solver while the run still prints PASS
     (`structuredAuthorityStepHealth.activeFineBricks` /
     `.fineBandCapacityOverflow`).
   - an **absent segment is not a zero counter**. Optional sources leave
     their segment all-zero, which is exactly what a skip predicate wants to
     see, so `encodeStructuredAuditRecordCopies` returns the segments it
     copied and the record carries them as `presentSegments`. Every
     zero-work witness requires presence.
3. **`lib/structured-step-snapshot.ts` — `StructuredStepSnapshotRing`.** The
   step's own encoder copies the accepted structured velocity/boundary
   controls, fine worklist header, MGPCG/volume controls, projection energy,
   and (since P0.4) the topology-epoch state, the SPGrid per-level setup
   delta, the air-support failure words, and the fine-transport governor
   schedule into a small MAP_READ ring as the last commands of the advance
   (`step-snapshot` stage). `readStats` consumes the freshest record:
   - authority lag is now `publishedFineGeneration − acceptedEpoch − 1` with
     BOTH words copied at the same step boundary — exact whole-step
     staleness, immune to pipeline depth and readback cadence
     (`info.structuredAuthorityLagSteps`, flagged at ≥2);
   - receipt validity and projection energy decode from the record, not from
     racing copies;
   - the legacy live-sample path remains only as fallback when no record
     exists yet.

   **P0.4 slot policy.** The ring was 3 slots. The browser keeps up to 8
   advances in flight and a record's map resolves only after its step
   completes, so a per-step consumer could find every slot mapping — the
   producer then *skips* the record, the required `step-snapshot` stage is
   missing, and that latches a permanent step-sequence fault. The slot count
   is now `structuredStepSnapshotSlotCount(maxPendingAdvances)` =
   `maxPendingAdvances + 1` (9), derived from
   `MAXIMUM_PENDING_PHYSICS_ADVANCES`, which is the same ceiling
   `presentationPhysicsQueueDepth` clamps to. The producer asserts it never
   skips: `StructuredStepSnapshotRing.skippedRecords` must stay zero, and the
   first skip logs `[step-snapshot]` naming the cause.
4. **`lib/physics-step-program.ts` + `docs/physics-step-program.json`** — the
   declarative program, its validator, JSON persistence (golden-tested), and
   the per-step sequence conformance check wired into `advanceTo`. P0.5 adds
   conductor-conditional stages, the predicate declarations they name, the
   predicted-work payload, and the lag-k prediction check (§3.1, §3.2). The
   validator additionally rejects a program where a non-guarded stage reads a
   skippable stage's output inside the same step — that consumer would
   silently receive last step's value.

Tests: `tests/gpu-advance-pacing.test.ts`,
`tests/physics-step-program.test.ts`,
`tests/structured-step-snapshot.test.ts`,
`tests/webgpu-smoke-structured-audit.test.ts` (the last pins the record
stride; P0.4 moved it from 336 B to 808 B).

## 5. Remaining sanctioned differences and open items

- **Wall-time cadence** still differs (Dawn saturates; the UI paces to
  presentation). This is the drivers' one degree of freedom and does not
  change per-step physics.
- **Real authority lag** (GPU candidate poisoning under violent flow) is now
  measured exactly. When `structured-authority-lag N gen` fires with the
  snapshot source, it is a genuine solver event to fix — not a sampling
  artifact to argue with.
- The performance-instrumentation default was flipped to `enabled: true`
  (uncommitted, concurrent instrumentation work). With scheduling decoupled
  this no longer perturbs admission, but trace passes/readbacks still ride
  the default path, and `tests/webgpu-renderer-lifecycle.test.ts` ("default
  lean UI path") fails against that default. Decision needed before merge.
- Migrating the remaining `readStats` QA readbacks (brick residency, global
  fine QA sweep) onto snapshot slots is possible follow-up; they are
  display-only and fenced, so they were left as-is.
- Future utilization work the dependency capture enables: stage 4
  (`inactive-topology-candidate`) feeds only the NEXT step's stage 1, and
  stage 6 feeds only presentation — a scheduler could overlap them with
  presentation encoding once measured worthwhile.
- P0.4's record widening is **half-landed**: the fine-transport governor is
  copied today (reached via `workAccountingBuffers.fineTransportGovernor`),
  but the topology-epoch state, the SPGrid `levelDelta`, and the air-support
  `scratch` are private fields of `WebGPUOctreeProjection`. Three one-line
  getters in `lib/webgpu-octree.ts` — `topologyEpochState` →
  `this.topologyEpoch?.state`, `spgridLevelDelta` → the V-cycle's `levelDelta`
  (itself private in `webgpu-octree-spgrid-vcycle.ts`), `airSupportScratch` →
  `this.airVelocitySupport?.scratch` — complete it. Until they exist those
  three segments report absent, and every zero-work witness that depends on
  them correctly refuses to authorize a skip.

---

## 6. UNIMPLEMENTED — settled-tier audit ABI (design only, lands with P1.5b)

This section is a **specification, not a description**. Nothing below exists
in code. It is written now because B5b (the frozen settled tier) is blocked on
it and because relaxing a safety net deserves a reviewed design rather than an
in-flight patch.

### 6.1 What blocks the frozen tier

`exactStructuredGenerationAuditFailures` in `lib/structured-authority-audit.ts`
requires, for every accepted step:

    publishedFineGeneration === expectedStructuredEpoch + 1
    publishedFineGeneration >  previousFineGeneration        // must ADVANCE
    expectedStructuredEpoch >  previousStructuredEpoch        // must ADVANCE

The two `>` clauses are the block. A frozen settled step deliberately leaves
`[forces + divergence + solve + projection]` and the fine-surface chain
unencoded, so no buffer is written and no generation advances. Under today's
ABI that is indistinguishable from the failure it was written to catch: a
silently rolled-back topology publication, which also freezes generations and
also still prints PASS. That ambiguity is precisely why the audit is strict,
and why it cannot simply be loosened.

### 6.2 The record kind

Introduce an **authored record kind** so the audit checks a *different*
invariant rather than a *weaker* one.

    type StructuredAuditRecordKind = "advancing" | "settled";

A step's kind is authored by the driver at encode time — it is a claim, not an
observation — and travels with the record:

- **`advancing`** — today's semantics, unchanged. Both generations must
  strictly advance and every existing clause applies. This stays the default,
  so nothing becomes laxer by omission.
- **`settled`** — the driver declares the step frozen. The audit then requires
  generations to be **unchanged**, not advanced:

      structured.epoch      === previousStructuredEpoch
      publishedFineGeneration === previousFineGeneration
      publishedFineGeneration === structured.epoch + 1   // still coherent
      structured.flags === 0 && structured.firstError === 0xffffffff
      boundary coherence clauses: unchanged from `advancing`

  plus the settled-specific evidence clauses in §6.3. A settled step that
  *does* advance a generation is a failure, symmetric with an advancing step
  that does not: in both directions the audit fails when the run disagrees
  with the claim, which is the property that makes the relaxation safe.

### 6.3 Required snapshot evidence

A settled claim is only accepted when the step's own record carries the
witnesses that make it checkable. Absent segments fail closed (§4.2's
`presentSegments` rule):

| witness | required value | segment |
|---|---|---|
| executed solve iterations | 0 | `mgpcg` |
| SPGrid per-level dirty words | all 0 | `spgridLevelDelta` |
| topology-epoch `ready` (flip token) | false | `topologyEpoch` |
| topology-epoch `error` | 0 | `topologyEpoch` |
| air-support `scratch[0]` | 0 | `airSupportFailure` |
| fine active-brick count | unchanged from the previous step | `fine` |
| fine band capacity overflow | false | `fine` |

The last row is why the count must be decoded from worklist header word 1: an
unchanged *generation* proves nothing about occupancy, and the word-0 bug
would have made "settled" trivially satisfiable.

### 6.4 The host-side claim and its provenance

The settled claim is a conductor predicate under §3.1 and carries the same
obligations: it is recorded per step with the snapshot step it was read from,
and it must be conservative under staleness. B5a's formulation is the one that
qualifies — executed iterations 0 AND fine page-delta 0 from the latest mapped
snapshot, AND no host-side input changed this step (bodies and inflow are
`advanceTo` arguments, compared synchronously), AND `movingRigidBodyCount === 0`
in the authored scene. Entering late merely costs savings; exiting is immediate
because host-input changes are known without any readback.

### 6.5 Landing conditions

1. The settling experiment first: a 60-step settled window with the unit
   frozen must be **bit-identical** to the same window encoded. If it is not,
   the fixed point is not stable and the tier stays at B5a (encode everything,
   solve budget 1) — the ABI change is then unnecessary, not merely deferred.
2. `advancing` remains the default everywhere; `settled` is only reachable
   through an explicit authored claim.
3. The gates must show a settled run failing when the claim is falsified —
   inject a step that claims settled while the solve executes an iteration and
   confirm the audit rejects it. A relaxation that has never been observed to
   fail is not a safety net.
