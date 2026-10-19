import assert from 'node:assert/strict';
import test from 'node:test';
import { SparseVoxelDrySceneRenderer, SVO_DRY_SCENE_PARAMS_LAYOUT } from '../lib/svo/pipeline/webgpu-svo-dry-scene';
import { DEFAULT_SVO_RENDER_TUNING, SvoSurfaceTuningCache } from '../lib/svo/pipeline/svo-render-tuning';
import { SVO_SURFACE_MESH_STATE_BYTES } from '../lib/svo/features/primary-visibility/svo-surface-mesh';

test('surface tuning reuses storage but notices in-place edits and surface-style changes', () => {
  const cache = new SvoSurfaceTuningCache();
  const input = { ...DEFAULT_SVO_RENDER_TUNING };
  const first = cache.resolve(input, true), key = cache.key;
  assert.equal(cache.resolve({ ...input }, true), first);
  input.resolutionScale = .5;
  assert.notEqual(cache.resolve(input, true), first);
  assert.notEqual(cache.key, key);
  assert.equal(cache.resolve(input, false).surfaceMeshing, input.surfaceMeshing);
});

test('moving rigid bounds update only their 16-byte parameter block and memoized words', () => {
  const upload = new Float32Array(4);
  const writes: { offset: number; data: number[] }[] = [];
  const renderer = Object.assign(Object.create(SparseVoxelDrySceneRenderer.prototype), {
    shadingPath: 'inline', rigidBodyCount: 0, rigidBounds: [0, 0, 0, -1],
    rigidBoundsUpload: upload, rigidBoundsUploadWords: new Uint32Array(upload.buffer),
    paramsWords: new Uint32Array(256), paramsBuffer: {}, rasterPrimary: false,
    device: { queue: { writeBuffer(_buffer: unknown, offset: number, data: Float32Array) { writes.push({ offset, data: Array.from(data) }); } } },
    rigidBodiesRequested: () => true,
    writeParams() { assert.fail('body motion must not repack the scene or rebuild shadow metadata'); },
  });
  renderer.setRigidBodyCount(1, { centre_m: [1, 2, 3], radius_m: 4 });
  renderer.setRigidBodyCount(1, { centre_m: [1, 2, 3], radius_m: 4 });
  renderer.setRigidBodyCount(1, { centre_m: [2, 2, 3], radius_m: 4 });
  renderer.setRigidBodyCount(0);
  assert.deepEqual(writes, [[1, 2, 3, 4], [2, 2, 3, 4], [0, 0, 0, -1]].map(data => ({ offset: SVO_DRY_SCENE_PARAMS_LAYOUT.rigidBoundsWordOffset * 4, data })));
  assert.equal(renderer.rigidBoundsUpload, upload);
  assert.deepEqual(Array.from(renderer.paramsWords.slice(SVO_DRY_SCENE_PARAMS_LAYOUT.rigidBoundsWordOffset, SVO_DRY_SCENE_PARAMS_LAYOUT.rigidBoundsWordOffset + 4)), Array.from(new Uint32Array(upload.buffer)));
});

function meshFixture() {
  let bundles = 0, draws = 0, writes = 0, receipts = 0;
  const compute: GPUComputePassDescriptor[] = [], render: GPURenderPassDescriptor[] = [];
  const commands = { setPipeline() {}, setBindGroup() {}, setVertexBuffer() {}, setIndexBuffer() {}, draw() {}, drawIndirect() {}, drawIndexedIndirect() {}, finish() { return {}; } };
  const renderer = Object.assign(Object.create(SparseVoxelDrySceneRenderer.prototype), {
    device: { queue: { writeBuffer() { writes++; } }, createRenderBundleEncoder() { bundles++; return commands; } },
    source: { structural: { domain: { brickSize: 8 }, planarBoundaries: { count: 2 }, sceneMaintenance: { activity: 0 } } },
    surfaceMeshState: {}, surfaceMeshArena: { size: 4096 }, surfaceMeshCullIndirect: {}, surfaceMeshReadback: {},
    surfaceMeshPipelines: Object.fromEntries(['prepare', 'publish', 'select', 'candidates', 'cullDispatch', 'cull', 'ingest', 'schedule', 'count', 'allocate', 'emit', 'boxes', 'background', 'planar', 'draw'].map(key => [key, {}])),
    surfaceMeshComputeGroup: {}, surfaceMeshDrawGroup: {}, bindGroup: {},
    surfaceMeshStatus: { state: 'ready', drawn: true }, surfaceMeshSeenActivity: 0,
    surfaceMeshPresentation: 0, surfaceMeshBuildPresentations: 0,
    surfaceMeshBatchWords: new Uint32Array(2), surfaceMeshReadbackAt_ms: performance.now(),
    surfaceMeshWorkLeafCapacity: 64, experiments: {}, renderTuning: DEFAULT_SVO_RENDER_TUNING,
    importedSurfaceInstances: [], disabledStages: new Set(), rigidBodyCount: 0,
    splitGeometryView: {}, splitOpaqueIdentityView: {}, ensureSurfaceMeshWork() {},
  });
  const encoder = {
    beginComputePass(descriptor: GPUComputePassDescriptor) { compute.push(descriptor); return { ...commands, dispatchWorkgroups() {}, dispatchWorkgroupsIndirect() {}, end() {} }; },
    beginRenderPass(descriptor: GPURenderPassDescriptor) { render.push(descriptor); return { executeBundles() { draws++; }, end() {} }; },
    copyBufferToBuffer(_a: unknown, _b: number, target: unknown, _d: number, size: number) { if (target === renderer.surfaceMeshReadback) { assert.equal(size, SVO_SURFACE_MESH_STATE_BYTES); receipts++; } },
  };
  const views = { hardwareDepth: {}, packedSurface: {}, identityMedia: {} };
  return { renderer, views, frame: () => renderer.encodeSurfaceMesh(encoder, views, false, 1), compute, render,
    counts: () => ({ bundles, draws, writes, receipts }) };
}

test('warm primary replays commands and descriptors, still culls every frame, and never waits on readback', () => {
  const f = meshFixture();
  f.frame(); const first = f.counts(); const passes = f.compute.length;
  f.frame();
  assert.deepEqual(f.counts(), { ...first, draws: first.draws + 1 });
  assert.equal(f.render[0], f.render[1]);
  assert.equal(f.compute.length, 2 * passes);
  for (let i = 0; i < passes; i++) assert.equal(f.compute[i], f.compute[i + passes]);
  assert.ok(f.compute.some(p => p.label === 'Voxel mesh culling'));
  f.renderer.surfaceMeshReadbackAt_ms = performance.now() - 251;
  f.frame(); assert.equal(f.counts().receipts, 1);
  // An in-flight receipt never blocks the next draw or starts a duplicate map.
  f.renderer.surfaceMeshReadbackCopied = false; f.renderer.surfaceMeshReadbackPending = true;
  f.frame(); assert.equal(f.counts().receipts, 1);
  f.renderer.source.structural.sceneMaintenance.activity++;
  f.frame();
  assert.equal(f.renderer.surfaceMeshStatus.state, 'pending');
  assert.ok(f.compute.some(p => p.label === 'Voxel surface mesh quad emission'), 'known edits start in this frame, even while a receipt is pending');
});

test('primary commands invalidate on bindings, assets, variants and rigid inspection; resize replaces attachments', () => {
  const f = meshFixture(); f.frame();
  for (const key of ['bindGroup', 'surfaceMeshDrawGroup', 'importedSurfaceLodRecords', 'importedSurfaceDrawArgs', 'rasterRigidInputBindGroup']) {
    const before = f.counts().bundles; f.renderer[key] = {}; f.frame(); assert.equal(f.counts().bundles, before + 1, key);
  }
  f.renderer.source.structural.planarBoundaries.count++; f.frame();
  const before = f.counts().bundles;
  f.views.hardwareDepth = {}; f.frame(); assert.equal(f.counts().bundles, before, 'bundles contain no viewport or attachment views');
  assert.notEqual(f.render.at(-1), f.render.at(-2));
  f.renderer.rasterRigidActive = true; f.frame(); assert.equal(f.counts().bundles, before + 1);
  f.renderer.identityPlanesInspected = true; f.frame(); assert.equal(f.counts().bundles, before + 2);
});

test('cached G-buffer views retain identity until resize and retire on destruction', async () => {
  const { SparseVoxelGBufferTargetArena } = await import('../lib/svo/features/primary-visibility/webgpu-svo-gbuffer-targets');
  const globals = globalThis as unknown as Record<string, unknown>, saved = globals.GPUTextureUsage;
  globals.GPUTextureUsage ??= { RENDER_ATTACHMENT: 1, TEXTURE_BINDING: 2, STORAGE_BINDING: 4, COPY_SRC: 8 };
  try {
    const arena = new SparseVoxelGBufferTargetArena({ createTexture: () => ({ createView: () => ({}), destroy() {} }) } as unknown as GPUDevice);
    assert.equal(arena.views, undefined);
    arena.ensureSize(10, 20); const before = arena.views;
    assert.equal(arena.views, before); assert.equal(arena.ensureSize(10, 20), false);
    arena.ensureSize(20, 20); assert.notEqual(arena.views, before);
    arena.destroy(); assert.equal(arena.views, undefined);
  } finally { if (saved === undefined) delete globals.GPUTextureUsage; else globals.GPUTextureUsage = saved; }
});
