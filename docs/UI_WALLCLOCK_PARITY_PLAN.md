# UI ↔ Dawn wall-clock parity, and timings that cost nothing

The Dawn smoke lane and the browser UI run the same solver and report very
different ms/advance. This plan separates the two causes, deletes the ones that
are pure loss, and replaces the measurement machinery with hardware timestamps
that add no commands to a frame.

## 1. Where the divergence comes from

### 1.1 Dawn's loop keeps the queue saturated

`tools/run-webgpu-smoke.ts:3192` is a bare `while` over `advanceTo` with no
fence between advances and no presentation. The host runs ahead of the GPU, so
the queue is never empty and `simulationWall_ms / steps` measures execution.

### 1.2 The UI throttles physics on a signal that only exists while measuring

`presentationHasPhysicsSlack` (`lib/webgpu-renderer.ts:77`) returns `false` when
`physics_ms` is undefined, and every caller computed `observedStep_ms` as

```ts
measurementInstrumentationEnabled ? observedGPUAdvanceTime_ms(trace) : undefined
```

`physicsTrace` is only ever populated by the measurement path. So on the default
UI path (MEASUREMENT LOAD off, `performance-instrumentation-store.ts:18`):

- `renderBeforePhysics` is permanently `true` — physics is never submitted before
  the presentation encode;
- `presentationPhysicsQueueDepth(undefined)` returns the bootstrap constant `2`,
  so exactly two advances are admitted per presentation regardless of their cost;
- `continuePreparedGPUWork` (`:1165`) bails on the same slack test, so the queue
  is never refilled between animation frames.

Net effect: **≤ 2 advances per 16.7 ms rAF tick, and the GPU drains to idle after
the second one and waits for a callback.** Turning measurement *on* is what gives
the scheduler its estimate — the opposite of the intent.

### 1.3 The presentation fence stalls physics submission

`draw()` sets `presentationPending` and returns early at `:1274` on every frame
while a presentation is in flight. `queue.onSubmittedWorkDone()` is queue-global,
so it resolves only after the presentation *and* the physics queued behind it
complete. Until then no work at all is admitted.

### 1.4 Measurement-only taxes (only with MEASUREMENT LOAD on)

- `DynamicGPUPerformanceTraceRecorder` emits, **per boundary**, a
  `copyBufferToBuffer` plus a whole extra compute pass with a 1-workgroup
  dispatch, deliberately to defeat Dawn/Metal pass folding
  (`lib/performance-trace.ts:567`). The octree physics lane arms dozens of
  boundaries per advance.
- `GPUSegmentedQueueWallPerformanceTraceRecorder` submits **one command buffer
  per phase** with `onSubmittedWorkDone()` between each (`:664`). Its own doc
  comment says the numbers are not execution time.
- `GPUQueueWallPerformanceTraceRecorder` adds one more queue fence per sample.

## 2. What lands

### Phase 1 — schedule physics from evidence the frame already has

Both completion callbacks the renderer already owns are turned into free
estimates: `submitPreparedGPUFluid` stamps the submit instant and the existing
advance fence yields a per-batch wall EMA; the existing presentation fence
yields a presentation EMA. No new fences, no measurement dependency.

- `physicsStepEstimate_ms()` prefers a live physics trace, else the fence EMA.
- The `measurementInstrumentationEnabled ? … : undefined` guards at `:1275`,
  `:1608` and `:1169` are deleted.
- `continuePreparedGPUWork` submits unconditionally when nothing is in flight —
  keeping the queue non-empty is the invariant, not a budget optimisation — and
  uses the slack test only to decide whether to go deeper than one advance.
- `draw()` tops physics up before the `presentationPending` early return.

### Phase 2 — `GPUStageTimestampRecorder`

A boundary chain that writes `timestampWrites.beginningOfPassWriteIndex` onto the
**first real pass of each stage** instead of injecting a marker pass. A `Proxy`
over `GPUCommandEncoder` intercepts `beginComputePass`/`beginRenderPass` and
splices the armed boundary into the descriptor — the same technique already
proven on Dawn/Metal by `auditCommandEncoder` in the smoke tool.

Cost per traced advance: **N counter samples and one trailing marker pass**,
versus N marker passes + N blits + N forced encoder breaks today. Phases whose
stage encoded no pass collapse to zero duration by sharing a boundary slot, so
the partition still closes exactly and `measurementSource` stays
`gpu-hardware-timestamp`.

### Phase 3 — adopt it in the two UI lanes

`webgpu-uniform-eulerian.ts` (physics) and `webgpu-renderer.ts` (presentation)
switch to the new recorder. The segmented queue-wall probe is deleted outright:
it is exactly the fencing this work removes, and
`GPUQueueWallPerformanceTraceRecorder` remains as the non-invasive fallback when
hardware timestamps are unavailable. The offline benchmark tools keep the
marker-based recorders, where added passes cost nothing that matters.

## 3. Measured result

`benchmark-power-dam --lane=moving-interface`, Apple M1 Max, 62 advances, with
and without `--profile`:

| | ms/advance | dispatches/advance | compute passes/advance |
|---|---|---|---|
| untraced | 570.82 | 1218.0 | 171.0 |
| traced | 581.98 | 1218.0 | 171.1 |

Zero added dispatches, +0.1 compute passes, +1.9 % wall — and the physics lane
reports `gpu-hardware-timestamp` with exact accounting across all fourteen
semantic stages. The old marker chain added one pass, one blit and one dispatch
*per boundary*; here the only added pass is the one that closes the chain.

(The absolute numbers reflect a solver mid-change on this branch, not a
regression from this work: the encoded command graph is byte-identical.)

## 4. Acceptance

- `npm run test:unit` gains no failures over the 14 already red on this branch.
  Verified; the octree failures that appeared during the session track a
  concurrent editor's in-flight changes to `lib/webgpu-octree*.ts`.
- With MEASUREMENT LOAD off, no `createQuerySet`, marker pass, encoder break or
  measurement fence is reachable on the frame path.
- Still to verify in the browser: the physics-admission change is renderer-only
  and the Dawn harness does not exercise it.
