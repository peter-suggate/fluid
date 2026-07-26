# Mini dam break 2.0 s — fresh-session handoff

Date: 2026-07-26

## Goal

Get the exact WebGPU/Dawn mini dam break to complete **500 steps and 2.0 s**
without a regression:

```sh
npm run test:webgpu:minimal-power-dam-break
```

Do not expand the task into all remaining optimization work. Read
`docs/OCTREE_M1_MAX_IMPLEMENTER_HANDOFF.md`, but implement an open item only if
runtime evidence shows it blocks this exact run. The current audit conclusion
is that none of its remaining items blocks the present one-step failure.

## Non-negotiable constraints

- This is an immediate cutover. Do not restore or retain a fallback, legacy
  shader, alternate transport authority, Galerkin RAP path, or retired face
  path.
- Fail closed on invalid or incomplete live topology. Do not turn a rejected
  sample into zero, the owner value, or another interpolant merely to pass the
  smoke test.
- FLIP/APIC is not part of this cutover and must not be introduced.
- Run only one Dawn/WebGPU process at a time. A prior concurrent GPU run crashed
  macOS WindowServer.
- Keep browser WebGPU tabs closed during Dawn runs.
- Every GPU worker must own `/tmp/fluid-webgpu-exclusive.lock`. If the lock
  exists, verify that no GPU process is alive before removing a stale lock.
- The worktree is intentionally very dirty and contains the full cutover. Do
  not reset, clean, restore deleted files, or discard unrelated changes.
- Do not commit unless the user asks.

## Last verified state

Green:

```sh
npx tsc --noEmit
npx tsx --test tests/webgpu-octree-structured-dynamics.test.ts tests/webgpu-bringup-stages.test.ts
npm run test:water-shaders -- --json
FLUID_BRINGUP_STAGE=sparse-t0 npm run test:webgpu:bringup-stage
```

The sparse t=0 checkpoint publishes the dam-break-ui authority on a
`24x18x16` grid. The latest allocation was about 144.6 MB. Dawn/Metal shader
submission and validation are green through the first positive-time command
buffer; the failure is now a logical fail-closed authority rejection.

Latest serialized one-step command:

```sh
FLUID_BRINGUP_STAGE=one-step npm run test:webgpu:bringup-stage
```

Latest result:

```text
structuredVelocityControl = [64, 16777479, 1352, 1, 0, 4550]
```

Decode:

- flags `64`: `OCTREE_STRUCTURED_GPU_ERROR.carry`
- packed first error `16777479 == 0x01000107`
- stage `1`: the initial full-vector sample in structured advection was invalid
- family handle `263`
- accepted authority remains generation 1, bank 0, 1352 rows, 4550 slots
- accepted boundary remains valid at generation 1
- later generation-2 candidate controls are empty/rejected and are secondary
- MGPCG failure state is secondary because structured advection invalidated the
  accepted authority before positive-time pressure work could be admitted

Before the latest fix the same fault was `stage 1 / handle 0`. Moving to handle
263 is positive evidence that the prescribed closed-wall case was fixed rather
than the gate being weakened.

## Current blocker

`lib/webgpu-octree-structured-dynamics.ts` has four family classes:

- 5: regular interior
- 6: transition interior
- 7: regular boundary
- 8: transition boundary

The original implementation aliased classes 7/8 to the interior samplers.
That was structurally incomplete: physical-boundary rows deliberately have no
exterior owner, and boundary tetrahedra can deliberately retain exterior
virtual selectors.

The following production corrections now exist:

1. A fully prescribed `aperture == 0` family handle does not run characteristic
   sampling. It publishes the exact solid normal velocity and finite transport
   metrics. Binding 22 is explicitly present on advection.
2. Regular cell-centred sampling clamps the query to physical cell-centre
   support and skips only exactly zero-weight cube corners. Any missing
   non-zero-weight live neighbor still rejects.
3. A transition selector may use the constant boundary extension only when its
   reconstructed selector centre is proven outside the physical domain. A
   missing selector whose centre is inside the domain still rejects.
4. `encodeClasses` dispatches each destination-owned class exactly once. A
   duplicate dispatch was found and removed; leaving it would apply gravity and
   pressure twice.

Handle 263 still rejects at the first sample. Do not guess whether it is a
regular cube, a transition selector, an invalid row velocity, or a dynamic
free-surface boundary. The bring-up worker now contains a failure-only helper,
`readStructuredHandleFailure`, and includes `structuredHandleFailure` in the
one-step exception. It reports:

- active control and bank
- handle family/orientation, owner and neighbor
- value, area, inverse distance, aperture and pressure scale
- normal, centroid and prescribed solid normal velocity
- owner geometry, full-vector velocity and topology metric
- six O(1) axis neighbors and the packed row-neighbor list

This helper was added after the latest GPU run. It is type-checked but has not
yet been exercised. The next action should be one serialized one-step run to
obtain this record.

## Recommended next sequence

1. Confirm no WebGPU/Dawn process is active and the exclusive lock is absent.
2. Run exactly one:

   ```sh
   FLUID_BRINGUP_STAGE=one-step npm run test:webgpu:bringup-stage
   ```

3. Read `power.structuredHandleFailure` from the exception.
4. Fix the exact production interpolation contract indicated by that record.
   Keep the distinction between expected boundary support and missing live
   topology explicit. Do not add a generic substitute sample.
5. Run, in order and never concurrently:

   ```sh
   npx tsc --noEmit
   npx tsx --test tests/webgpu-octree-structured-dynamics.test.ts tests/webgpu-bringup-stages.test.ts
   npm run test:water-shaders -- --json
   FLUID_BRINGUP_STAGE=one-step npm run test:webgpu:bringup-stage
   npm run test:webgpu:minimal-power-dam-two-step
   npm run test:webgpu:minimal-power-dam-break
   ```

6. The terminal success condition is the exact smoke contract: 500 accepted
   steps, 2.0 simulated seconds, no authority rejection, no validation error,
   no non-finite/stability regression, and all required spatial/raster/fine
   generation checks present.
7. Only if the exact run reaches a measured timeout or a later pressure failure
   should the deferred handoff items be reconsidered.

## Already-fixed defects worth preserving

- Candidate/accepted control and bank ABIs were corrected across SPGrid and the
  structured boundary.
- Analytic t=0 boundary bootstrap is an explicit primary selector, then retires
  and fails closed.
- Boundary 19-channel mapping uses centroid plus half-normal correctly.
- Section 4.3, direct summary and fine-seed bind groups/uniforms use their exact
  layouts.
- t=0 structured advection is skipped as the exact identity operation.
- Structured divergence writes exact zero RHS for `dt == 0`, but rejects
  invalid or negative time steps.
- The production pressure path consumes only the accepted six-word authority
  control ABI; retired MGPCG authority selection was removed.
- Scalar-packed structured dynamics dimensions preserve the host/WGSL uniform
  offsets; a `vec3u` alignment mismatch was fixed.
- Power topology commit uses the required 256-lane coverage.
- Candidate validation no longer contaminates the live accepted MGPCG control.
- Accurate A2 and finer-adjoint page accesses validate native/ghost page bounds
  before page-record reads.
- Fine level-set transport no longer binds one buffer as writable storage and
  indirect in the same synchronization scope; it uses a dedicated copied
  indirect buffer.
- Failure-only stage/index attribution exists in SPGrid and structured
  dynamics. Do not move those readbacks into the recurring path.
- Legacy/fallback shader modules and Galerkin production paths were deleted.

## Deferred work

The following items from `docs/OCTREE_M1_MAX_IMPLEMENTER_HANDOFF.md` are not
current blockers and should remain deferred until runtime evidence says
otherwise:

- capturing/freezing all four regression baselines
- parallelizing the single-threaded MG hierarchy candidate rebuild
- repairing the topology-delta dirty predicate
- further pressure-shell/k=3 or page-shape performance work

If the exact 500-step run becomes throughput-bound after correctness is green,
measure first. Do not infer that one of these must be implemented merely because
it remains open in the broader handoff.

## Useful files

- `docs/WEBGPU_OCTREE_M1_MAX_IMPLEMENTATION_PLAN.md`
- `docs/OCTREE_M1_MAX_IMPLEMENTER_HANDOFF.md`
- `lib/webgpu-octree-structured-dynamics.ts`
- `lib/webgpu-octree-structured-boundary.ts`
- `lib/webgpu-octree-structured-velocity-gpu.ts`
- `lib/webgpu-octree.ts`
- `lib/webgpu-uniform-eulerian.ts`
- `tools/run-webgpu-bringup-stage-worker.ts`
- `tools/run-webgpu-bringup-stage.ts`
- `tools/run-webgpu-smoke-isolated-worker.ts`
- `tests/webgpu-octree-structured-dynamics.test.ts`
- `tests/webgpu-bringup-stages.test.ts`

## Fresh-session kickoff prompt

```text
Continue the exact goal in docs/MINI_DAM_BREAK_2S_FRESH_SESSION_HANDOFF.md.
Read that file first, then docs/OCTREE_M1_MAX_IMPLEMENTER_HANDOFF.md only for
context. Get npm run test:webgpu:minimal-power-dam-break to pass exactly 500
steps / 2.0 s. Do not restore fallbacks or legacy shaders; fail closed on real
topology errors. Keep all Dawn/WebGPU tests strictly serialized behind
/tmp/fluid-webgpu-exclusive.lock because concurrent GPU runs previously crashed
WindowServer. Preserve the dirty worktree and do not commit unless asked. Start
with one FLUID_BRINGUP_STAGE=one-step run and use the new
power.structuredHandleFailure record for handle 263 before changing code.
```
