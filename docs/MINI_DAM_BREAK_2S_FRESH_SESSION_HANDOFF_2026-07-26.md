# Mini dam break 2.0 s — fresh-session handoff (2026-07-26)

## Goal

Get the exact Dawn/WebGPU mini dam break to finish **500 accepted steps and
2.0 simulated seconds**:

```sh
npm run test:webgpu:minimal-power-dam-break
```

The implementation must remain supported by
`docs/papers/aanjaneya-2017-power-liquids.txt`, especially Section 4.3 for the
pressure solve, Section 5 for fine-surface transport and velocity
extrapolation, and Section 6 for topology/operator storage. Do not weaken a
failed authority check, substitute zero/owner velocity, restore a legacy path,
or add FLIP/APIC.

## Non-negotiable operating rules

- The worktree is intentionally very dirty. Do not reset, clean, restore
  deleted files, or discard unrelated edits. Do not commit unless asked.
- Run exactly one Dawn/WebGPU process at a time. Every GPU worker must own
  `/tmp/fluid-webgpu-exclusive.lock`.
- A stale lock currently names PID `61840`; the latest process check found no
  live Dawn/WebGPU process. Re-check before removing the stale lock.
- Keep browser WebGPU tabs closed during Dawn runs.
- If a pipeline reaches the portable limit of ten storage buffers, first
  remove or coalesce existing buffers. Do not reflexively add an eleventh
  binding.
- Use subagents for independent read-only audits or bounded fixes, but never
  run Dawn concurrently.

## What changed in this session

### UI import error is closed

The UI error about an invalid import of
`require(identity, STRUCTURED_AIR_SUPPORT_RECORD_FLAGS.interfaceSource, ...)`
was caused by a local helper named `require`, which Vite treated as CommonJS.
It is now named `requireAcceptedCell` in
`lib/octree-structured-air-support.ts`. Repository search found no remaining
offending local `require`. `npm run build`, `npx tsc --noEmit`, and the focused
air-support tests passed.

### Pressure curvature failure is repaired

The pressure solver now uses standard matrix-free PCG with direct positive
curvature `d^T A d`, rather than the f32-fragile Chronopoulos-Gear subtractive
denominator. It reuses existing direction/partial buffers, adds no storage
buffer, zeroes future indirect work after convergence, and keeps true
nonpositive curvature fail-closed. The Section 4.3 shell depth is `k=8`, which
matches the paper's reported adequate Jacobi sweep count.

Relevant files:

- `lib/octree-pipelined-pcg.ts`
- `lib/webgpu-octree-pipelined-mgpcg.ts`
- `lib/webgpu-octree-work-accounting.ts`
- `tests/octree-pipelined-pcg.test.ts`
- `tests/webgpu-octree-pipelined-mgpcg.test.ts`

Independent review approved the CPU/GPU recurrence and schedule. The terminal
converged iteration's gated-off `A(direction)` is no longer overcounted.
Focused pressure validation reported 33 passes and one optional Dawn skip;
TypeScript and diff checks passed.

### Section 5 positive-air support is integrated

The previous positive-air miss and the old averaging extension path were
replaced with a generation-coherent, GPU-resident Section 5 transaction:

- Air-support ABI version 2 appends a dense per-base-cell owner record
  `{tag, originCell, size, case|transform}` to the already-bound shared arena.
  No storage buffer was added.
- Fine transport no longer binds `rowGeometry`; both live common/rare kernels
  use nine storage buffers, below Dawn's portable ten-buffer ceiling.
- The exact selected A/B fine generation is validated in the worklist header,
  every live page, and air-support control word 15.
- Every valid fine destination inside the authored transport band marks its
  complete maximum-backtrace cube.
- A scratch-only `QUERY_FINE` bit separates query owners from nonrecursive
  value-only interpolation closure, avoiding a cross-workgroup race.
- Regular owners publish the exact 27-point logical closure; transition owners
  publish every selector actually referenced by the case's tetrahedra.
- Per-trajectory sampling now calls `transitionSample`; case 0 immediately
  falls through to regular trilinear sampling. This avoids unsound page-anchor
  specialization when a trajectory crosses a T-junction.
- Four fine transport page classes were collapsed to common/rare, removing two
  zero/unsafe specialized dispatches.
- Ordinary-face adjacency is resolved once at publication. The six relaxation
  waves do indexed gathers only; the old 24-dispatch/six-fence averaging path
  is gone.
- Face relaxation now propagates weighted face-centre distance with a stable
  source tie-break. Final publication requires the last relaxation wave to
  make no changes and every demanded vector to reconstruct. This implements
  the paper's closest-interface-face rule and fails closed if six waves are
  insufficient.
- The invalid comparison between fine-cell displacement and face-graph wave
  count was removed. Fine-cell displacement is used only to rasterize the
  trajectory demand; graph convergence is a separate invariant.
- Support is refreshed after pending fine settlement, so it consumes the newly
  settled A/B source before surface transport.

Relevant files:

- `lib/webgpu-octree-air-velocity-support.ts`
- `lib/webgpu-octree-air-velocity-support-gpu.ts`
- `lib/webgpu-octree-fine-levelset-transport.ts`
- `lib/webgpu-octree-fine-levelset-transport.wgsl.ts`
- `lib/webgpu-octree.ts`
- `tests/webgpu-octree-air-velocity-support-gpu.test.ts`
- `tests/webgpu-octree-fine-levelset-production-transport.test.ts`

Latest validation from the implementing agent:

```text
npx tsc --noEmit                                      PASS
npm run test:water-shaders                            PASS
focused Section 5/fine transport suite                42 pass, 2 Dawn skips
```

The shader validator now includes the air-support WGSL. No Dawn run was made
after this integration.

## One small correctness strengthening before Dawn

`markFineResolvedOwner` in
`lib/webgpu-octree-air-velocity-support-gpu.ts` currently records whichever
accepted owner contains `floor(probe)`. Fine transport later verifies the exact
owner centre and size and therefore still fails closed, but support publication
should reject the mismatch earlier.

Strengthen the producer so closure publication proves the same identity the
consumer needs:

- Regular closure: clamp the expected same-size centre exactly as
  `regularSample` does, then require the resolved owner to have that size and
  centre.
- Transition closure: compute `selectorSize = round(owner.size * v.w)` and the
  exact transformed selector centre. When the centre is inside the valid
  selector-centre domain, require the resolved owner size and centre to match;
  only a geometrically proven physical exterior may use the constant boundary
  extension.
- On mismatch, set the producer topology/catalog error. Do not defer it to a
  zero value or alternate interpolant.

This is a clearer publication invariant directly supported by Section 5's
cube/tetrahedron vertex interpolation. Re-run TypeScript, focused tests, and
all shader validation after the edit.

Other reviewed limitations are safe for this exact run but should remain
explicit:

- Six face-relaxation waves are not a universal graph-diameter proof. The
  final no-change gate makes the current implementation safe: it rejects
  rather than committing a non-nearest result. Increase the bound only if
  runtime evidence shows nonconvergence, and retain the final fixed-point gate.
- Direct domain-wide publication dispatches assume the current small domain.
  Larger domains need 2-D dispatching or explicit rejection.
- Position-to-owner lookup assumes the production fine-domain origin is zero.
  General non-zero origins need an explicit coordinate transform.

## Exact next validation sequence

Do not run any two GPU commands concurrently.

1. Make the exact selector-centre/size publication check above.
2. Run CPU/static validation; independent commands may run in parallel:

   ```sh
   npx tsc --noEmit
   npx tsx --test \
     tests/webgpu-octree-air-velocity-support.test.ts \
     tests/webgpu-octree-air-velocity-support-gpu.test.ts \
     tests/webgpu-octree-fine-levelset-production-transport.test.ts \
     tests/webgpu-octree-fine-levelset-transport.test.ts \
     tests/octree-pipelined-pcg.test.ts \
     tests/webgpu-octree-pipelined-mgpcg.test.ts \
     tests/webgpu-octree-work-accounting.test.ts
   npm run test:water-shaders -- --json
   npm run build
   git diff --check
   ```

3. Check for a live GPU worker. If none exists, remove only the verified stale
   `/tmp/fluid-webgpu-exclusive.lock` directory.
4. Run one serialized one-step checkpoint:

   ```sh
   FLUID_BRINGUP_STAGE=one-step npm run test:webgpu:bringup-stage
   ```

5. If the support transaction rejects, inspect its flags/first error and the
   fine-generation fields. A nonzero final relaxation-change count means the
   fixed six-wave ceiling did not converge; increase the encoded wave bound,
   retain metric relaxation and the final no-change proof, then retest. Do not
   reinterpret waves as fine cells.
6. Once one-step is green, run:

   ```sh
   npm run test:webgpu:minimal-power-dam-two-step
   ```

7. Once two-step is green, run the exact terminal command:

   ```sh
   npm run test:webgpu:minimal-power-dam-break
   ```

Success means exactly 500 accepted steps, 2.0 simulated seconds, no authority
rejection, no WebGPU validation error, finite/stable fields, and all required
spatial/raster/fine-generation checks present.

## Paper contract to preserve

From `docs/papers/aanjaneya-2017-power-liquids.txt`:

- Section 4.3: matrix-free CG with a linear symmetric positive-definite hybrid
  preconditioner; the paper reports about eight Jacobi sweeps and satisfactory
  convergence in six to ten iterations.
- Section 5: the fine SPGrid is a sparse narrow band; each characteristic uses
  `m` velocity-resampled substeps and assigns one final phi sample back to every
  starting cell; cube interpolation is trilinear, transition interpolation is
  tetrahedral/barycentric; velocity is mapped from power faces to ordinary
  faces, copied from the face closest to the free surface, then interpolated
  back to power faces.
- Section 6: topology is encoded compactly and immutable geometry/operator data
  is cache-oriented. Do not introduce a larger per-row streamed operator under
  the claim that it is the paper's LUT design.

The current pressure and Section 5 changes are backed by those contracts. The
earlier attempted Section 6 descriptor/LUT shortcut was reverted because
dynamic aperture/theta rows did not have an exact immutable-row proof.

## Performance issues to keep ranked, but do not guess before the exact run

- E1: the second-order `applyRow` address chase remains a major candidate; the
  shared-memory `smoothPage` pattern is the intended GPU direction, but any
  numerical change must retain the exact Section 4.3 operator.
- E2: several O(N) single-workgroup publishers/reductions remain. Convert them
  with deterministic count/scan/scatter only when runtime evidence warrants.
- E3: recurring diagnostics should be gated when no readback is requested.
- E4: fixed MGPCG encoding remains dispatch-heavy, but prior measurements found
  dispatch-count cuts wall-neutral; measure before changing it.
- E5: the old six pass-boundary extension path is removed. The new support
  transaction uses two required storage-to-indirect boundaries.
- E6: fine transport specialization is now two common/rare dispatches instead
  of four. Other cheap four-class stages remain candidates.
- E7: fine transport uses dense owner records and loops exactly over the chosen
  substep count. The dynamics-side sampler still deserves measurement.
- E8: transition catalog locality remains transition-only and unresolved.
- Section 6 bandwidth: dynamic rows still materialize 19 f32 coefficients.
  Geometry comes from the catalog, but an exact 4-byte descriptor/LUT cutover
  requires proving which BC-scaled rows are immutable; do not use the reverted
  unsafe fast path.

## Copy-paste kickoff prompt

```text
Continue the exact goal in
docs/MINI_DAM_BREAK_2S_FRESH_SESSION_HANDOFF_2026-07-26.md. Read that file in
full first. Get npm run test:webgpu:minimal-power-dam-break to pass exactly 500
steps / 2.0 s. Preserve the paper contracts in
docs/papers/aanjaneya-2017-power-liquids.txt, fail closed, do not restore legacy
or fallback paths, and do not commit/reset/clean the intentionally dirty tree.

First strengthen markFineResolvedOwner so the air-support producer proves the
exact regular/transition centre and size required by the consumer. Then run
TypeScript, focused tests, all shader validation, and the build. After checking
that no WebGPU process is alive, remove only the verified stale exclusive lock
and run Dawn strictly serially: one-step, two-step, then the exact 500-step
command. Diagnose the first concrete failure only; do not weaken an authority
gate. If the face march rejects because its final wave still changes labels,
increase the relaxation bound while retaining weighted closest-face relaxation
and the final fixed-point check.

Use subagents for independent read-only audits and bounded fixes where useful.
If a pipeline reaches ten storage buffers, first remove or coalesce an existing
buffer rather than adding another.
```
