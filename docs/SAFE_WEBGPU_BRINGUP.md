# Safe WebGPU bring-up after an AGX/WindowServer watchdog failure

Two browser WebGPU attempts on 2026-07-20 caused machine-wide WindowServer/AGX
failures that required a reboot. Until the fault is localized, treat GPU
validation as a system-safety operation rather than an ordinary test run.

> **Never run a Dawn test and a browser WebGPU simulation concurrently. Close
> every browser tab using WebGPU before starting Dawn, and close Dawn before
> opening the simulation UI. Do not run two Dawn stages concurrently.**

`tools/run-webgpu-bringup-stage.ts` launches exactly one checkpoint in an
isolated child process. The parent enforces a wall-clock timeout (120 seconds by
default), and the child holds `/tmp/fluid-webgpu-exclusive.lock` so two Dawn
bring-up processes cannot overlap. A stale lock must be removed only after
checking that no Dawn or browser GPU workload is active.

Run one command, inspect its final JSON record, and let the process exit before
advancing to the next command:

```sh
FLUID_BRINGUP_STAGE=adapter-device npm run test:webgpu:bringup-stage
FLUID_BRINGUP_STAGE=compute-sentinel npm run test:webgpu:bringup-stage
FLUID_BRINGUP_STAGE=solver-resources npm run test:webgpu:bringup-stage
FLUID_BRINGUP_STAGE=sparse-t0 npm run test:webgpu:bringup-stage
FLUID_BRINGUP_STAGE=one-step npm run test:webgpu:bringup-stage
```

The default is the deliberately small balanced dam-break grid: 384 finest
columns, or exactly `24 x 18 x 16` cubic cells for the default box. It still
uses compact face transport, authoritative power projection, MGPCG, and the
factor-4 global fine lattice. Override `FLUID_SURFACE_COLUMNS` only after all
five stages pass at 384.

Those settings are also the normal balanced browser product preset, not hidden query
overrides. `leafSolver=auto` admits the paper's Section 4.3 hybrid only after
power authority passes its topology, spacing, geometry, transfer, solve, and
publication gates. Terrain and imported/seeded geometry keep the documented
axis/Chebyshev compatibility path. High and ultra keep the older bounded
rollback preset until their memory and endurance gates pass.

The checkpoints are real execution boundaries:

- `adapter-device` requests the Metal adapter and device, then exits without a
  shader or queue submission.
- `compute-sentinel` compiles and reads back one known compute word.
- `solver-resources` performs the solver allocation and asynchronous pipeline
  task graph, then stops at the named `solver.warmup` boundary. It does not
  submit the sparse t=0 publication.
- `sparse-t0` runs that warmup, waits its queue fence, and requires
  `initialSparseAuthorityReady`. Within this stage, Section 5 face-band work
  is submitted and fenced in paper dependency order: band-row topology,
  catalog-Delaunay transition adjacency plus regular-face emission,
  face-centered fast marching, and regular-face to power-face publication.
  Each initialization JSON record carries `boundary: "starting"` or
  `boundary: "completed"`; the last starting record without its matching
  completion localizes a timeout or device loss without another GPU run.
  The corresponding task IDs are
  `solver.warmup.section5-face-band-topology`,
  `solver.warmup.section5-face-band-transitions`,
  `solver.warmup.section5-face-band-fast-march`, and
  `solver.warmup.section5-face-band-power-publication`.
- `one-step` performs exactly one fixed 0.004-second advance, waits diagnostics,
  and rejects validation errors or an incorrect submission count.

The factor-4 transport still performs all four Section 5 piecewise-linear
trajectory segments, including a fresh Stage-B velocity query and face-band
override on every segment. Its query scratch is now batched from the adapter's
storage-binding, buffer-size, dispatch, and offset-alignment limits, with a
conservative target of 65,536 queries rather than the old fixed 4,096. On the
`24 x 18 x 16` scene this changes 442,368 samples from 108 chunks / 2,810
encoded compute passes to 7 chunks / 184 passes. The shared prepass scratch is
82,837,504 bytes at that target, below the portable 128 MiB storage-binding
limit; a more constrained adapter selects the largest smaller aligned batch.
The target is a cap, so a larger domain does not turn this scratch arena into a
domain-global allocation.

Use `FLUID_BRINGUP_TIMEOUT_MS` to select a per-process limit from 1,000 to
600,000 milliseconds. A timeout exits with status 124. A timeout, device loss,
WindowServer reset, display corruption, or sentinel mismatch is a hard stop:
do not advance to the next stage and do not open the browser simulation.

Only after the staged Dawn sequence is green should browser testing resume.
Close the Dawn process and confirm its command has exited before opening the
browser. The portable browser application has no trusted local-filesystem
broker and cannot inspect Dawn's `/tmp/fluid-webgpu-exclusive.lock` lease from
the browser sandbox. Therefore Dawn/browser mutual exclusion is still an
operator precondition, not an automated guarantee. Do not add a server route
that exposes or removes the `/tmp` lease: that would turn a local safety signal
into an unauthenticated remote mutation surface.

Use this exact bounded browser query (shown across lines only for readability):

```text
?gpu=safe&method=octree&scene=water-box-dam-break&quality=balanced&render=raster&voxels=smooth&param.octree.surfaceColumns=384&param.octree.faceVelocityTransport=on&param.octree.globalFineLevelSetFactor=4&param.octree.powerDiagramProjection=authoritative&param.octree.secondaryParticles=off&param.octree.maximumLeafSize=16
```

Keep the tab visible and focused because solver scheduling and presentation use
`requestAnimationFrame`; a background tab may be throttled and is not a valid
bring-up result. Safe mode validates the authored scene and these settings
before requesting an adapter. It also takes the exclusive
`fluid-lab:webgpu-exclusive` Web Lock. A second Fluid Lab tab fails immediately,
and browsers without Web Locks are refused in safe mode. Normal modes retain
their existing compatibility behavior when Web Locks are unavailable.

The browser procedure is deliberately one-shot:

1. Confirm all Dawn processes have exited and no other browser tab is using
   Fluid Lab WebGPU.
2. Open the exact query above and press **START WEBGPU**.
3. Wait until the status is ready and the fenced
   `initialSparseAuthorityReady` flag has unlocked **STEP**.
4. Press **STEP** exactly once and wait for the queue-confirmed time to reach
   `0.0040 s`.
5. Press **STOP GPU**. Wait for “device released — safe to close this tab”
   before closing or navigating away.

After STOP, the status first changes to **STOPPING**. The cross-tab lease stays
held while an in-flight adapter/device request, presentation-pipeline compile,
optional pipeline compile, or solver initialization settles. Only after the
renderer has destroyed the device and those host transactions have drained
does the UI report “device released” and relinquish the lease.

Safe mode disables Play, reset, recording, timestep edits, and scene import.
The controller independently rejects continuous running and a second step, so
the limit does not depend on button state alone. Changing the pinned scene,
method, quality, any resolved numerical method value, diagnostics, overlay,
renderer, right-side panel, or stage-capture state after startup begins an
awaited shutdown. A reset/rebuild attempt also invalidates the session even
when it resolves to identical values; it cannot silently construct a second
solver. Unapproved URL flags are rejected before adapter access. Do not use
screenshots, DevTools performance capture, diagnostic readbacks, timestamp
queries, automatic device recovery, or a second WebGPU tab during this pass.

## Isolated one-step comparison

The exact power/fine-versus-tall comparison also runs through a bounded child
process and the same exclusive lock. Run it only after the staged sequence is
green and all browser WebGPU tabs are closed:

```sh
FLUID_WEBGPU_SMOKE_TIMEOUT_MS=120000 npm run test:webgpu:dam-power-fine-compare-one-step
```

The script pins both methods to `24 x 18 x 16`, `maxDt=0.004`, and exactly one
accepted and encoded step. After the queue fence it records and checks both the
submitted and completed clocks at `0.004 s`, refreshes solver diagnostics, and
requires nontrivial sampled motion (`peak speed >= 0.01 m/s`; the first-step
gravity impulse is about `0.039 m/s`). The octree compact-face maximum is not
claimed to be liquid-restricted; the existing `[0.5, 2]` tall-relative ratio
remains an independent parity gate. The octree side
does not use the old aggregate placeholder: QA reads the current factor-4 fine
hash/pages and same-generation compact coarse-phi directory, averages the
`4^3` fine occupancies into each of the 6,912 comparison cells, and fails closed
on a stale or missing publication. Acceptance additionally requires nonzero
fine and coarse sample contributions, a nonempty current worklist, finite valid
samples on both sides of the interface, and a clean published topology; a
plausible coarse-only field cannot pass. Tall-cell continues to read its real packed
level-set field. The final IoU/centroid comparison therefore uses two spatial
fields of identical dimensions.

`FLUID_WEBGPU_SMOKE_TIMEOUT_MS` accepts only 60,000–240,000 milliseconds. The
launcher reports the child PID, and the worker records that PID in
`/tmp/fluid-webgpu-exclusive.lock/owner.json` before importing Dawn. A timeout
sends SIGTERM, waits two seconds, then sends SIGKILL and exits with status 124.
Normal success or an ordinary test error removes the lock in `finally`; a
wedged or signalled process deliberately leaves its owner evidence in place.
Remove such a lock only after confirming that PID is gone and no Dawn or
browser WebGPU workload is active.
