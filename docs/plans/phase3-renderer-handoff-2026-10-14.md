# Phase 3 renderer hand-over — 14 October 2026

Implemented serially in `/Users/petersuggate/code/me/fluid`. No commits, worktrees, Git mutations, browser runs, or unit-suite runs. Other sessions' edits were retained. The final `npm run check:types` passes, and `git diff --check` is clean.

## What landed and which hunks I own

`/private/tmp/claude-501/handoff/phase3-owned.patch` records my exact edits as chronological, zero-context diffs. It includes intermediate versions followed by their refinements: it is an ownership audit, **not a patch to apply again**. The two newly created tools named below as wholly owned have their entire initial content owned by this task, in addition to subsequent edits recorded in that audit. This hand-over is wholly owned too.

- `lib/core/webgpu-water-pipeline.ts`: K9 window capacity, block/local list ownership and packing, direct collection launch of `8 * (blocks - 2)` workgroups, fixed extraction launch over that list, block-based header and shared samples, and source adoption/bindings. Deleted the box-keyed collection path. Window cells, wall closure, incident vertices, normal caches and corner sampling use the shared page address and SOLID-page holds. Low y closure uses `umFaceLow` / `umLowFaceAbove`. Surface positions read the solver's GPU cell size and frame origin. The optical contact sampler also reads that frame. The full classifier retains its original bounds and behavior for non-mixed methods. Included K9's asynchronous mesh-overflow diagnostic/throw. Geometry and list storage remain fixed while a pool is held.
- `lib/methods/uniform/uniform-mixed-presentation.wgsl.ts`: removed the box dimensions from the renderer's `uniformLatticeAddressWGSL` construction. Presentation now uses the solver's page-table/block-record address directly.
- `lib/core/uniform-detail-abi.ts`: removed the box clamp from packed cell and face loads; raw-texture loads are unchanged.
- `lib/core/grid-overlay-levelset-volume.wgsl.ts`: mixed phi/volume sampling no longer rejects coordinates using the seed-box dimensions; the raw-texture bounds remain.
- `lib/core/levelset-consumer-abi.ts`: added `mixedBlocks` and `mixedFrame`. The latter binds the first 48 bytes of solver params: grid, cell size, frame origin.
- `lib/methods/uniform/webgpu-uniform-reference.ts`: publishes those fields and a stable `mixedPageSource`, combining pool publication with the current GPU params prefix. These are constructed at mixed-frame creation, not per advance.
- `lib/methods/uniform/uniform-page-pool.ts`: re-keyed overlay publication by block, including full signed coordinates and resident state. Header version 2 has 32 words, then four words per block. Resident bounds are computed in the existing publish launch. The source separately exposes the page directory from pool state. Publication now exists even without a seed box.
- `lib/core/method-view-records.ts`: documents the new block-record ABI and its directory/frame bindings.
- `lib/core/domain-pages.wgsl.ts`: reader resolves a page through the directory and checks the block's signed coordinate/state. The 3D DDA covers published resident bounds, with the current GPU frame. AIR stays invisible, resident stays purple and SOLID stays grey. No per-pixel block scan.
- `lib/core/webgpu-grid-overlay.ts`: copies the directory and GPU frame into the existing combined overlay record buffer when pages are visible; slice page lookup uses world-to-frame conversion. Removed its packed tile-read box clamp. Only the exact relevant snippets were replaced.
- `tools/check-surface-window-ownership.ts`: **entire new file owned**. Deterministic random resident sets, an independent tile/window incidence oracle, block/local ownership twin, capacity check, publication count/coordinate checks, and a signed-coordinate case.
- `tools/check-surface-window-list.ts`: **entire new file owned**. Runs a scene, checks fatal status and block publication, compares full/window meshes, or counts outside-seed triangles. Full comparison scans the final resident bounding rectangle, including AIR gaps, through `latticeValue`; it does not use the window list to select cubes. Hashes canonicalize triangle order and quantize positions to 1e-5 cells. The tool checks mesh overflow, including the production seed-size mesh capacity. Diagnostic timeout: 115 seconds.
- `tools/check-surface-window-compile.ts`: new compile-only oracle, adapted from K9; checks the two windows kernels against 5 seconds, validates the full overlay shader and compiles its page fragment pipeline.
- `tools/probe-domain-pages.ts`: retained the existing render probe, adapted its missing-source binding, added residents on both sides of its old seed box (including a negative coordinate), and added a numbered PASS summary.
- `tools/probe-uniform-page-pool.ts`: adapted the publication audit and corruption offset to block records. Other pool checks are unchanged.
- `tests/uniform-page-pool.test.ts`: deleted whole under `common.md`'s changed-subject rule. It asserted the deleted eight-word, box-indexed publication, including invariance under block permutation and no publication without a box. No replacement unit tests were added, and the suite was not run.

## Decisions made here

1. The reference patch's check passed, reporting 42 edits and one new tool, despite the report describing an earlier 35-edit version. I re-cut the reviewed replacements and adapted them to current addressing rather than running the stale apply script. The reference's later far-stencil handling was retained.
2. Window capacity is `min(address-window windows, 729 * (blocks - 2))`. The extent passed is 2048 cells on each axis. Using the initial seed-box extent would underallocate as liquid leaves it. This reserves more than the old box bound on some closed scenes, upfront.
3. The actual page-record publisher is `UniformPagePool`, not a shader in `webgpu-uniform-reference.ts`; the latter exposes the source and frame. The directory is copied only for a visible overlay (1 MiB with the 64³ table), not added to every pool commit. The existing publish launch writes block records and computes a small resident-bounds reduction. No simulation pass or solver readback was added.
4. Frame origin is bound directly from GPU params, not cached as a host constant. Surface extraction uses binding 17; optical contact uses binding 20. Overlay encoding copies the same prefix. This follows subsequent GPU updates of those params; integration with the other session's moving-window implementation remains a gate.
5. The production window collection/extraction launches are direct and fixed. Existing downstream polygonise dispatch machinery was retained; this task adds no indirect launch or deferred pipeline build.
6. The full oracle scans resident bounds rather than the original seed box, otherwise it would wrongly report legitimate outside geometry as extra. Its diagnostic classifier specialization is compiled outside the advance path.

## Exact oracle commands

Run from `/Users/petersuggate/code/me/fluid`. Device commands below are for **bash**, under the shared GPU lock. Peter's main session owns the gate. Both the trough and water-slide comparisons have now been run; their independent results are recorded below.

Host ownership/publication twin:

```sh
node --import tsx tools/check-surface-window-ownership.ts
```

Full against windows, both requested scenes:

```sh
/usr/bin/lockf /private/tmp/claude-501/fluid-gpu.lock env WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/check-surface-window-list.ts uniform-trough-dam-break 40
/usr/bin/lockf /private/tmp/claude-501/fluid-gpu.lock env WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/check-surface-window-list.ts nbflip-water-slide 60
```

Outside geometry and overlay publication together:

```sh
/usr/bin/lockf /private/tmp/claude-501/fluid-gpu.lock env WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/check-surface-window-list.ts nbflip-open-ground 120 --outside
```

Overlay publication alone, if needed instead of that combined run:

```sh
/usr/bin/lockf /private/tmp/claude-501/fluid-gpu.lock env WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/check-surface-window-list.ts nbflip-open-ground 120 --overlay-only
```

Compile timing/validation:

```sh
/usr/bin/lockf /private/tmp/claude-501/fluid-gpu.lock env WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/check-surface-window-compile.ts
```

Existing synthetic overlay render probe, adapted but not run here:

```sh
/usr/bin/lockf /private/tmp/claude-501/fluid-gpu.lock env WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/probe-domain-pages.ts
```

The existing broader pool probe was updated for the ABI but not run; its command remains `node --import tsx tools/probe-uniform-page-pool.ts`, through the same GPU lock if the main session elects to run it. It is not needed to repeat the short renderer smoke checks.

## What ran and printed

All logs below are in `/private/tmp/claude-501/handoff/`.

- Reference applicability: `python3 /private/tmp/claude-501/handoff/k9-ref/patch.py --check /Users/petersuggate/code/me/fluid` — `check passed: 42 edits to 1 files and 1 new files apply; nothing written`.
- Host oracle (`phase3-host.log`): **PASS, 32 cases, 320,923 windows**, zero missing/extra/duplicate in every case. Publication count equals resident count in each case; signed publication check passes.
- Trough, one run at 40 steps and 1/60 s (`phase3-trough.log`): **26,304 full triangles, 26,304 window triangles**. Both position hashes: `75f15ee80b67cb1c6ea5c43bcb1bb7e1a8b87f0c4f010047ddff8d52ff1c47e4`. Overlay publication: **22 drawn/readable, 22 resident, 0 missing**. Fatal none. PASS.
- Water slide, one follow-up run explicitly requested by Peter, at 60 steps and 1/60 s (`phase3-water-slide.log`): **15,520 full triangles, 15,520 window triangles**. Both position hashes: `394c35e3c7f35e77d5d45d1b7b6716975e4764608faa5a2af70f74455846eee3`. Overlay publication: **66 drawn/readable, 66 resident, 0 missing**. Fatal none. **PASS**, process exit 0. The production-capacity assertion also passed. Exact command, executed under bash:

  ```sh
  /usr/bin/lockf -t 45 /private/tmp/claude-501/fluid-gpu.lock env WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/check-surface-window-list.ts nbflip-water-slide 60 > /private/tmp/claude-501/handoff/phase3-water-slide.log 2>&1
  ```

- Open ground, one run at 120 steps and 1/60 s (`phase3-outside.log`): **39,394 triangles; 16,114 outside the seed box**. Overlay publication: **69 drawn/readable, 69 resident, 0 missing**. Fatal none. PASS.
- Initial compilation (`phase3-compile.log`): collection 381 ms, extraction 1,679 ms.
- Final compilation (`phase3-compile-final.log`): **collection 331 ms, extraction 1,702 ms, page overlay 52 ms**, all PASS. A preceding nonwaiting lock attempt found the GPU busy; the successful retry used `lockf -t 45`. No simulation rerun.
- `npm run check:types` (`phase3-types.log`): **PASS** on the final source. An intervening run caught another session's `webgpu-live-svo-scene.ts` progress phase mismatch; I left it alone, waited, and the other session corrected its type. The retry passes.
- `git diff --check`: clean.

The original trough/open-ground smoke commands used `lockf -t 0`; the requested water-slide follow-up used `lockf -t 45`, with its exact command above. Each scene was run once; no soak or full Dawn run was performed. The production-capacity assertion added after the original smoke runs was exercised successfully by the water-slide follow-up. The host FAIL formatting was type-checked; no deliberate host failure was injected. This follow-up changed only this hand-over, added the run log, and made no commit.

## Unverified / left for the main session

- Browser picture, frame time, window arrival-order performance, and normals at every mixed seam or concave SOLID junction. The position hash checks geometry, not normals.
- Actual overlay raster visibility/pixels: compiled and its real pool records checked by a host reader, but the synthetic render probe and app view were not run. “Drawn/readable” above means a resident record is reachable through the overlay's directory/coordinate rule, not a pixel count.
- Moving-window integration, especially frame updates/rebases and topology publication ordering; the binding reads the live params prefix but no forced recenter run was made here.
- Mesh-overflow diagnostic behavior under deliberate exhaustion and the broader adapted pool probe.
- Unit suite deliberately skipped under the shared handoff rules. No assertion was weakened or new unit test added.

The work is ready for Peter's main gate. Stop here; the main session decides the commit after testing.
