# Physics step sequencing: the formal procedure and the two drivers

Status: implemented on `perf/structured-cutover` (2026-07-27). This document
is the contract; `lib/physics-step-program.ts` is the machine-readable form,
persisted at `docs/physics-step-program.json` and enforced at runtime.

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
| 7 | `step-snapshot` | accepted receipt, worklist header, stats | step-coherent diagnostics record |

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

1. **when** to request an advance, and
2. **how many** advances are in flight.

Everything else is the solver's. Specifically:

- one advance = one command buffer = one submit; drivers never split steps;
- the target clock advances in whole fixed steps (controller collapse /
  harness checkpoints), so dt sequences are identical across drivers;
- diagnostics consume the end-of-step snapshot record; no consumer may race
  live solver buffers with an independent readback encoder;
- scheduling inputs must exist identically with instrumentation on or off;
- instrumentation must be sequence-neutral (boundaries ride real passes; a
  traced advance submits the same command graph as an untraced one).

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
   (`STRUCTURED_GENERATION_AUDIT_SNAPSHOT`, 336 B) and validators, promoted
   from `tools/webgpu-smoke-structured-audit.ts` (now a re-export shim) so
   browser and harness share one layout and one decode.
3. **`lib/structured-step-snapshot.ts` — `StructuredStepSnapshotRing`.** The
   step's own encoder copies the accepted structured velocity/boundary
   controls, fine worklist header, MGPCG/volume controls, and projection
   energy into a small MAP_READ ring as the last commands of the advance
   (`step-snapshot` stage). `readStats` consumes the freshest record:
   - authority lag is now `publishedFineGeneration − acceptedEpoch − 1` with
     BOTH words copied at the same step boundary — exact whole-step
     staleness, immune to pipeline depth and readback cadence
     (`info.structuredAuthorityLagSteps`, flagged at ≥2);
   - receipt validity and projection energy decode from the record, not from
     racing copies;
   - the legacy live-sample path remains only as fallback when no record
     exists yet.
4. **`lib/physics-step-program.ts` + `docs/physics-step-program.json`** — the
   declarative program, its validator, JSON persistence (golden-tested), and
   the per-step sequence conformance check wired into `advanceTo`.

Tests: `tests/gpu-advance-pacing.test.ts`,
`tests/physics-step-program.test.ts`,
`tests/structured-step-snapshot.test.ts`; the stale stride expectation in
`tests/webgpu-smoke-structured-audit.test.ts` was updated to the 336-byte /
32-word projection-energy ABI the working tree already uses.

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
