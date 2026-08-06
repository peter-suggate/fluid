# Identity memos across the render-worker boundary — implementation plan

**Goal:** the garden study renders at vsync, not 25 fps. Concretely: worker
`draw()` CPU ≤ 2 ms on `garden-svo-lighting` at any zoom, against a measured
**38.3 ms today**, with sculpted-terrain caustics still updating when the ground
actually changes.

**Context.** Measured 2026-08-05 in the browser at commit `c81628f` (dirty),
M1 Max, `?scene=garden-svo-lighting&camera.distance=12`. The cost is *not* the
renderer: an xctrace of the identical scene and camera through
`tools/profile-svo-render-xctrace.ts` puts the whole SVO frame at 4.9 ms GPU /
9.6 ms wall at 800×460. See `[[identity-memos-die-at-worker-boundary]]`.

---

## 1. Evidence

Rendering runs in a dedicated worker on an OffscreenCanvas
(`lib/webgpu-render-worker.ts`). The main thread posts one `draw` per frame
**carrying the scene document**, so `postMessage` structured-clones the whole
graph every frame and every consumer sees fresh objects.

| Probe | Value | Reading |
|---|---|---|
| draws sent / frames returned | 25.6/s / 25.6/s | 1:1 — no double-render |
| idle between frames (worker) | 0.57 ms | pipeline ~98% duty; UI is not the limiter |
| worker `draw()` | **38.2 ms** synchronous | all of it is CPU inside the worker |
| canvas 1.41 → 0.426 Mpx | 25.2 → 25.1 fps | **pixel-independent** |
| `setSceneOptics(...)` | **37.4 of 38.3 ms** | one statement |

Bisected by timing `draw()` in the worker, then accumulate-since-last-mark
probes down the body. The headless lane cannot express any of this: it publishes
the scene **once** at setup and asserts it goes idle, so it measured the same
scene at 8.3 ms and was blind to the bug.

## 2. Root cause

`lib/webgpu-water-pipeline.ts:1423`

```ts
if (key === this.receiverKey && terrain === this.receiverSource) return;
```

`terrain === this.receiverSource` is an object-identity test on a value that is
structured-cloned every frame, so it is never true. The guard never returns
early and each frame re-runs 384² = **147,456 `terrainHeightAt` calls**, a
`writeTexture`, and `rebuildBindGroups()` — the exact cost the comment above it
says would "cost more than the bake did".

The identity conjunct is load-bearing, which is why this is not a one-line
delete: `key` covers `width`/`depth`/`baseHeight_m`/`features.length` and the
grid's `nx`/`nz`/`spacing`/`origin`, but **not** feature contents and **not**
`grid.heights_m`. A brush stroke, or a mound dragged without resizing, returns a
new grid of identical shape and the same key. Dropping identity alone trades a
perf bug for a staleness bug — measured 0.91 ms / 60 fps, but wrong.

Same defect, second site: `sceneRevision` (`lib/model.ts:257`) is a
`WeakMap<SceneDescription, number>`, and a WeakMap entry cannot survive a
structured clone, so `publishRenderScene` always takes the `canonicalScene()`
deep-clone fallback. Only 0.31 ms on the 24 KB garden document — a
correctness-of-intent bug that scales with document size, not today's wall.

## 3. Work packages

### W1 — serializable content stamp for terrain *(fixes the 37.4 ms)*

Replace the identity conjunct with a stamp that survives serialization. Compute
it **on the main thread**, where document identity *is* stable, and memoize it
by identity there; ship it alongside the scene.

- `lib/terrain.ts`: `terrainContentStamp(terrain): string` — cheap over
  `features` (≤ 8, hash contents not just length), and over `grid` by hashing
  `heights_m` (≤ 262,144 samples, one pass).
- Main thread: `WeakMap<TerrainDescription, string>` memo, so the hash is paid
  once per authored change, never per frame.
- `lib/webgpu-water-pipeline.ts`: fold the stamp into `key`, delete
  `receiverSource` and the `terrain ===` conjunct.

Fails **closed**: a producer that mutates a grid in place without replacing it
already violates the documented immutability contract, and the hash catches it
anyway.

### W2 — stop shipping the scene document every frame *(structural)*

W1 makes the guard correct; W2 removes the reason it is hot. The main thread
owns the scene store and knows when the document changes.

- Post `set-simulation-scene` **on change only** (currently ~60/s), and carry a
  scene *revision number* in `draw` instead of the document.
- Worker resolves the revision against its retained document.

Removes one structured clone per frame in each direction and closes the whole
class of "identity memo silently misses" defects at the seam rather than one
call site at a time.

### W3 — `sceneRevision` across the seam

Once W2 lands, the worker holds a stable document object and the existing
WeakMap works as designed. Until then, `publishRenderScene`'s `canonicalScene`
fallback stays — cheap on today's documents, and a size-scaling trap worth a
comment naming the reason.

## 4. Verification

- **Gate:** `?scene=garden-svo-lighting&camera.distance=12`, worker `draw()`
  ≤ 2 ms (from 38.3), fps vsync-limited. Measure by timing `runtime.draw()` in
  `webgpu-render-worker.ts` — the browser round-trip and the app's FPS chip both
  agree with it to within noise.
- **Correctness (the point of W1):** sculpt a grid with the brush, keep the
  container and grid dimensions fixed, confirm the caustic receiver updates.
  This is exactly the case the naive delete breaks, so it needs a regression
  test, not a manual check.
- **No renderer regression:** `npm run benchmark:svo-fixed-overhead`; the
  headless lane cannot see W1/W2 either way, which is itself the point.

## 5. Risk

The one real risk is W1's hash missing a mutation path that the identity check
caught. Enumerate the producers of `TerrainDescription`/`TerrainGrid` before
landing; if any mutates in place, fix that instead of widening the hash.
