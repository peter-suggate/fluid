# Incremental voxel-mesh rebuilds — what landed, what is verified, what is not

Date: 2026-09-09. Implemented and validated on CPU tests, naga and two
synthetic Dawn tests. Not yet committed. Not yet checked in the app.

The question asked: when the user edits a scene it takes a long time for the
rasterized primary path (`svoPrimary=mesh`) to come back. Was the approach
optimal? It was not. Every publication rebuilt the whole mesh from scratch in
fixed 128-brick batches while the whole frame fell back to tracing, so an edit
of a dozen bricks cost the same as loading the scene, and the fallback trace
(not the extraction) was most of that wall (see
`mesh-build-wall-is-the-fallback-trace` in memory and the numbers in
`docs/voxel-surface-rasterization.md`).

Three levers were agreed and all three are in:

1. keep the drawn mesh during rebuilds and only cull/trace the edited region;
2. pace the host by presentation, not by a fixed batch;
3. re-extract per brick from the voxelizer's dirty-brick list.

---

## Part 1 — Files

Everything below is uncommitted in the working tree. Two other diffs in the
same tree are **not** part of this work and should not be bundled with it:
`lib/svo/features/shading/program.ts` and `tools/benchmark-svo-dry-frame-gpu.ts`
(the geometric/shading normal split, see `mesh-filtered-detail-toggle`), and
the untracked surface-band files under `tests/` and `docs/HANDOFF_FLUID_SURFACE_*`.

| File | Change |
|---|---|
| `lib/svo/features/primary-visibility/svo-surface-mesh.ts` | Rewritten. All WGSL kernels, the state-word map, the receipt decoder, the pacing ramp. |
| `lib/svo/pipeline/webgpu-svo-dry-scene.ts` | Host orchestration rewritten: two arenas, work buffer, receipt handling, pacing, dispatch encoding. |
| `lib/core/webgpu-voxel-debug.ts` | `SparseVoxelStructuralRenderSource.sceneMaintenance` (optional): the voxelizer's maintenance buffer and dirty-brick list layout. |
| `lib/svo/features/construction/webgpu-svo-sparse-bricks.ts` | Structural source now carries `proxyVoxelizer.maintenanceBinding`. |
| `lib/svo/pipeline/render-pipeline-graph.ts` | Tip summary; chip says "mesh updating · edited bricks traced" while pending and drawn. |
| `lib/core/webgpu-renderer.ts` | Status equality includes `drawn` and `liveQuads`. |
| `tests/svo-surface-mesh-build-pacing.test.ts` | Rewritten: ramp + receipt decoder. |
| `tests/svo-surface-mesh-detail.test.ts` | Regex updates for the new kernel names. |
| `tests/svo-mesh-live-fallback-dawn.test.ts` | Background pass now tested against dirty boxes. |
| `tests/svo-surface-mesh-scheduler-dawn.test.ts` | **New.** Two-brick synthetic octree through the real kernels. |
| `tools/probe-svo-bounded-mesh-dawn.ts` | Removed; the scheduler test replaces it. |
| `docs/voxel-surface-rasterization.md`, `docs/svo-depth3-rendering.md` | Build/overflow/pacing sections rewritten; probe reference updated. |

A copy of the pre-rewrite `svo-surface-mesh.ts` is at
`/private/tmp/claude-501/-Users-petersuggate-code-me-fluid/3bf893ac-2879-42d9-8fbc-778d5f038f67/scratchpad/svo-surface-mesh.before.ts`
while that scratchpad survives; `git show HEAD:...` is the durable copy.

---

## Part 2 — The design in one pass

### Storage

- **Quad arena** (32 B per `SurfaceQuad`: origin, identity, extent, face word).
  A quad with an all-zero extent is dead. Two arena slots, `surfaceMeshArenas[0|1]`;
  the drawn one is the **front** (state word 26). An unbound slot holds a 64 B
  placeholder.
- **Per-leaf table** in the work buffer (`surfaceMeshWorkBytes(leafCapacity)`),
  4 words per leaf after a 512-word prefix of dirty boxes: `base`,
  `count | depth<<27`, `keyX`, `keyY` (the brick's octree address). A slot whose
  key does not match the leaf currently in it is stale and gets re-extracted.
- **Scratch** (4 words per leaf) holds the pre-batch table entry for rollback.
- **Worklist** (1 word per leaf) built by the mark pass.
- **State** (192 B, `SVO_SURFACE_MESH_STATE`): draw args, cursors, flags,
  dispatch triples, host words. Words 32 and 38-45 are written by the host.

### Kernel order per presentation (`encodeSurfaceMesh`)

`prepare` → if pending: `boxes`, `mark` (indirect), `schedule`, `count`,
`allocate`, `emit` → `publish` → `background` → `cull` → draw → 192 B readback.
Indirect triples live in a 64 B dispatch buffer at 0 (mark), 16 (extract),
32 (allocate), 48 (cull).

### Modes (state word 27)

| Mode | When | Writes into | Mask |
|---|---|---|---|
| 1 incremental | dirty list trusted | front, past the cursor | dirty boxes |
| 2 initial | nothing drawn | front | whole frame traced |
| 3 replacement | untrusted list, or compaction | back arena, flips at the end | none; keeps drawing |

**Trusted dirty list**: `requested == completed`, `completed == consumed + 1`,
no `SPARSE_SCENE_MAINTENANCE_INCOMPLETE_OVERFLOW` bits. `finalizeScene` bumps
the geometry revision only on a completed maintenance revision, so a mesh
revision change is exactly one completed maintenance revision. `consumed`
(word 33) is set to `completed` on every answered publication.

**Compaction** triggers when dead holes exceed `live + 65536`. It is a
replacement build with restart reason 5.

### Dirty boxes and the mask

Boxes are closed lattice boxes from the dirty-brick list: up to 63 individual
slots plus their union in slot 63; more than 63 collapses to the union alone.

- **Mark** includes bricks whose box *overlaps or touches* a dirty box (their
  faces against the edit may change).
- **Cull** removes only quads *wholly contained* in a box.
- **Background** traces rays that *touch* any box (closed slab test).

Containment for cull and touch for trace are deliberately asymmetric. Every
culled quad lies in a box, and every ray that would have hit it touches that
box, so nothing goes missing; culling by overlap would have opened holes under
neighbour quads that merely touch the edit.

### Overflow

`allocate` sets error bit 0 and `emit` is skipped. `publish` rolls the cursor
back to the checkpoint, restores the table from scratch, restores `liveQuads`,
records `overflowLength`. The host doubles the target arena (front for
incremental, back for replacement) with a prefix copy and rebinds; `prepare`
resumes when `arrayLength(arena) > overflowLength`. Dead-marking of a brick's
old range is deferred to emit job 0, which is why a rolled-back batch leaves
the drawn mesh intact.

### Back arena handshake

A replacement build sets `needBack` and `wantedBackQuads`. The host allocates
`max(min(64 MiB, max), min(max, wanted × 32))` in the back slot, bumps
`surfaceMeshBackGeneration` and writes word 40. The GPU starts when word 40 ≠
word 41 (`backGenerationConsumed`), which it sets at the flip and on state
reset. On flip the host swaps the old front for the placeholder and destroys
it after `onSubmittedWorkDone`.

### Pacing

Word 32 is bricks per batch, written every presentation:
`surfaceMeshBuildBricks(pendingPresentations, drawn)` = 512 on the first,
doubling, ceiling 2,048 while drawn and 16,384 while undrawn. Reset on
`publishScene`, on state reset, and on a ready→pending receipt. Build passes
are encoded only while the host status is not "ready".

---

## Part 3 — Invariants worth knowing before touching it

- `markPending` (flag bit 2) is set by `prepare` on a new build and cleared by
  `schedule` **only after all of its early-return waits**. Clearing it before
  the back-arena wait made the flip draw nothing; the scheduler test covers this.
- `writeSurfaceMeshHostWords` must never write word 41. Rebinding an arena
  during a pending replacement would otherwise deadlock the handshake.
- `target` is a reserved identifier in naga; the WGSL uses `arena`.
- Tint uniformity: every barrier in the count/emit kernels is in uniform
  control flow. Select by count, never branch around a barrier.
- Header word semantics the benchmark tool reads (1, 4, 5, 6, 7, 12, 13, 15)
  were retained.
- The receipt decoder `interpretSurfaceMeshState` is pure; the host acts only
  on flips, `needBackBytes` and `grow`. A stale receipt can neither restart nor
  skip a build because the GPU worklist and cursor are authoritative.

---

## Part 4 — Verification

| Check | Command | Result |
|---|---|---|
| Ramp + receipt decoder, kernel regexes, naga on every bundle variant | `node --import tsx --test tests/svo-surface-mesh-build-pacing.test.ts tests/svo-surface-mesh-detail.test.ts` | 14 pass |
| Scheduler on Dawn: initial 12 quads, reuse, incremental (worklist 2, live 22, cursor 34), overflow rollback in a 40-quad arena, growth to 80, replacement with an untrusted list, back arena generation 1, flip (front 1, cursor 22, builds 4) | `WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx --test tests/svo-surface-mesh-scheduler-dawn.test.ts` | pass |
| Background pass on Dawn: box crossing, missing, `boxCount` 0, undrawn | same form, `tests/svo-mesh-live-fallback-dawn.test.ts` | pass |
| `tsc --noEmit` on every touched file | | clean (pre-existing reds elsewhere, see `cpu-suite-is-red-at-head`) |

Run the Dawn tests directly. They acquire the GPU lock themselves, and wrapping
them in `tools/run-webgpu-exclusive.ts` fails with "Refusing concurrent GPU
execution".

### Not verified

- **No in-app run.** Nobody has edited a scene with the mesh path on and watched
  it. That is the acceptance test and it is Peter's to run.
- **No full-renderer Dawn render** with the mesh path. Headless Dawn renders are
  rationed (they can hang the Mac). If one is wanted, the benchmark tool with
  `FLUID_SVO_DRY_FRAME_SURFACE_MESH=1` is the lane; keep it to one run.
- **Real dirty lists.** The scheduler test synthesises the maintenance buffer.
  Whether the voxelizer's list is trusted on a real edit (requested ==
  completed, no overflow bit) has only been reasoned from `finalizeScene`.
  If the app shows every edit as a replacement build, that is the first
  thing to check: read state words 38 and 33 against the maintenance state.

### What to look for in the app

1. Edit a scene. The mesh should keep drawing; only the edited bricks should
   go to the traced fallback, and the chip should read "mesh updating · edited
   bricks traced".
2. The edit should be back on the rasterized path within a presentation or
   two, not tens.
3. No holes at the edges of the edit and no stale quads inside it.
4. After many edits the panel should show one "compaction" restart rather than
   ever-growing arena use.

---

## Part 5 — Open items

- Compaction as a replacement build still traces nothing and draws the old
  mesh, so it is cheap on screen, but it re-extracts every brick. A copy-compact
  pass (move live ranges, rewrite table bases) would avoid that. Not needed
  until compactions are observed to matter.
- More than 63 dirty bricks collapses the mask to one union box. A large
  sculpt stroke therefore traces its whole bounding box while pending. If that
  reads as a visible flash, raise `SVO_SURFACE_MESH_BOX_INDIVIDUAL_CAPACITY`
  (cost: the cull and background read more boxes per quad/pixel).
- The old `probe-svo-bounded-mesh-dawn.ts` numbers quoted in
  `docs/svo-depth3-rendering.md` are historical and describe the removed
  128-brick scheduler.
