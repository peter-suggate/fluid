import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { FineLevelSetBrickOracle, packFineLevelSetBrickKey,
  planFineLevelSetBricks } from "../lib/octree-fine-levelset-bricks";
import { WebGPUFineLevelSetBricks } from "../lib/webgpu-octree-fine-levelset-bricks";
import { FINE_TO_COARSE_LEVELSET_ERROR, fineToCoarseLevelSetWGSL,
  fineToCoarseLevelSetActivityShader, unpackFineToCoarseGPUControl,
  WebGPUFineToCoarseLevelSet } from "../lib/webgpu-octree-fine-to-coarse-levelset";
import { PassBroker } from "../lib/webgpu-pass-broker";
import { createGPULogicalActivityAdoptionContext,
  gpuLogicalActivityTaskDescriptions } from "../lib/gpu-logical-activity-adoption";

test("fine-to-coarse activity is conditional and describes all three stages", () => {
  const disabled = createGPULogicalActivityAdoptionContext({
    moduleId: "octree/fine-to-coarse-levelset",
    profile: { enabled: false, generation: 0x4300 },
  });
  assert.equal(fineToCoarseLevelSetActivityShader(disabled), fineToCoarseLevelSetWGSL);

  const generation = 0x4301;
  const enabled = createGPULogicalActivityAdoptionContext({
    moduleId: "octree/fine-to-coarse-levelset",
    profile: { enabled: true, generation },
  });
  for (const [task, id, label] of [
    ["prepare-restriction", "gpu.physics.fine-restriction.prepare", "Fine restriction · prepare rows"],
    ["restrict-coarse-rows", "gpu.physics.fine-restriction.rows", "Fine restriction · restrict coarse rows"],
    ["publish-restriction", "gpu.physics.fine-restriction.publish", "Fine restriction · publish"],
  ] as const) enabled.describeTask(task, { id, label, phaseId: "adaptive-publication" });
  const shader = enabled.module(fineToCoarseLevelSetActivityShader(enabled)).code;
  assert.match(shader, /@group\(3\) @binding\(0\)/);
  assert.equal((shader.match(/fluidGpuLogicalActivityWorkgroup\(/g) ?? []).length, 4,
    "one helper definition and three stage checkpoints are generated");
  const labels = Object.values(gpuLogicalActivityTaskDescriptions(generation)).map(({ label }) => label);
  assert.deepEqual(labels.sort(), [
    "Fine restriction · prepare rows",
    "Fine restriction · publish",
    "Fine restriction · restrict coarse rows",
  ]);
});

test("fine-to-coarse restriction rejects an unpublished or stale fine source", () => {
  const shader = fineToCoarseLevelSetWGSL.replace(/\s+/g, "");
  const encode = WebGPUFineToCoarseLevelSet.prototype.encode.toString().replace(/\s+/g, "");
  assert.match(encode, /^encode\(broker,fine,input\)/);
  assert.doesNotMatch(encode, /newPassBroker|broker\.finish/,
    "restriction must retain the caller-owned encoder");
  assert.match(encode, /run\("publish"\);broker\.fence/,
    "restriction must close its final compute pass before caller readback or downstream copies");
  assert.equal(FINE_TO_COARSE_LEVELSET_ERROR.unpublishedSource, 8);
  assert.match(shader,
    /letrollback=\(topologyControl\[0\]&DOWNSTREAM_ROLLBACK\)!=0u&&topologyControl\[4\]==1u&&topologyControl\[5\]==1u&&topologyControl\[7\]!=0u/,
    "restriction accepts the exact prior field retagged by a completed downstream rollback, including combined failure flags");
  assert.match(shader,
    /topologyReady=\(topologyControl\[0\]==0u&&\(committed\|\|provisional\)&&topologyControl\[5\]==0u&&topologyControl\[7\]==0u\)\|\|rollback/,
    "restriction rejects arbitrary stale input while allowing committed, provisional, or exact rollback authority");
  assert.match(encode, /input\.allowValidatedProvisional\?1:0/,
    "pre-force correction must opt into provisional input rather than weakening the default publication gate");
  assert.match(shader,
    /fnpublishRestriction[\s\S]*if\(control\.flags==0u\)[\s\S]*else\{control\.count=0xffffffffu/,
    "a rejected fine source must poison the downstream coarse correction rather than publish an empty correction");
  assert.match(shader,
    /control\.rowCount=min\(rowCountSource\[0\],p\.rowCapacity\);if\(control\.rowCount<arrayLength\(&rowOffsets\)\)\{rowOffsets\[control\.rowCount\]=control\.rowCount;\}else\{control\.flags\|=CAPACITY;\}/,
    "the CSR sentinel must terminate the live row prefix consumed by the coarse correction");
  assert.doesNotMatch(shader, /rowOffsets\[p\.rowCapacity\]=p\.rowCapacity/,
    "allocation capacity is not the terminal offset when fewer rows are live");
  assert.match(encode, /prepare:\[0,2,7,9,13,14,15\]/,
    "the scalar prepare pass must bind the fine worklist and topology transaction it validates");
  assert.doesNotMatch(encode, /finalizeRestrictionRows|run\("finalize"/,
    "row owners publish accepted aggregates directly without a second capacity-wide pass");
  assert.doesNotMatch(encode, /diagnose|unowned/i,
    "the removed capacity-wide diagnostic path must not remain in production encoding");
});

test("fine-to-coarse restriction evaluates phi at the octree cell center", () => {
  const shader = fineToCoarseLevelSetWGSL.replace(/\s+/g, "");
  const encode = WebGPUFineToCoarseLevelSet.prototype.encode.toString().replace(/\s+/g, "");
  assert.match(shader, /centerDelta=abs\(abs\(d\)-vec3f\(\.5\*p\.fineWidth\)\)/,
    "the eight samples surrounding the cell center define the correction");
  assert.match(shader, /mask\|=1u<<corner[\s\S]*rowCorners\[lid\*8u\+corner\]=value/,
    "the eight center-corner samples occupy workgroup-owned deterministic slots");
  assert.match(shader, /rowCombinedMasks\[0\]==255u[\s\S]*centerPhi\+=\.125\*cornerPhi/,
    "cell-center phi must be a fixed-order trilinear average, not a nearest-sample tie break");
  assert.match(shader, /rowCombinedMasks\[0\]==255u/,
    "a partial fine stencil must not claim a valid exact cell-center correction");
  assert.doesNotMatch(shader, /nearestDistance|nearestLogical|NearestPhi/);
  assert.doesNotMatch(encode, /selectRestrictionLogicalId|emitRestrictionNearestPhi/);
});

test("fine-to-coarse restriction publishes lost coverage instead of dropping it silently", () => {
  const shader = fineToCoarseLevelSetWGSL.replace(/\s+/g, "");
  // An unaccepted row raises no flag and writes no contribution, so a fine band
  // that no longer reaches a coarse row deletes that row's correction without a
  // trace. The count is the only signal a band-width change has lost coverage.
  assert.match(shader,
    /fnpublishRestriction[\s\S]*unaccepted\+=select\(1u,0u,aggregates\[r\]\.valid!=0u\)/,
    "publication must count the live rows whose eight-corner stencil was incomplete");
  assert.match(shader,
    /fnpublishRestriction[\s\S]*diagnosticCounts\[lid\]\+=diagnosticCounts\[lid\+width\][\s\S]*control\.unacceptedRows=/,
    "the unaccepted-row count must be reduced in the existing publication workgroup");
  assert.match(shader, /letsourceRejected=control\.flags!=0u[\s\S]*control\.unacceptedRows=select\(diagnosticCounts\[0\],0u,sourceRejected\)/,
    "a rejected source dispatches no row workgroups, so its stale aggregates must not be counted");
  const publish = shader.slice(shader.indexOf("fnpublishRestriction"));
  assert.deepEqual(publish.match(/control\.flags\|=\w+/g), ["control.flags|=diagnosticErrors"],
    "uncorrected dry rows are legitimate and must never poison the coarse transaction");
  assert.equal(unpackFineToCoarseGPUControl([0, 1, 0, 7, 9, 0]).unacceptedRows, 7,
    "the count occupies the already-read-back control word so no new readback is needed");
});

test("fine-to-coarse restriction is row-owned and synchronization-atomic-free", () => {
  const shader = fineToCoarseLevelSetWGSL.replace(/\s+/g, "");
  const encode = WebGPUFineToCoarseLevelSet.prototype.encode.toString().replace(/\s+/g, "");
  assert.doesNotMatch(shader,
    /\batomic(?:Add|And|CompareExchangeWeak|Exchange|Load|Max|Min|Or|Store|Sub|Xor)\s*\(|atomic<u32>/,
    "the deleted sample-scatter aggregate must not survive behind an atomic fallback");
  assert.match(shader,
    /fnrestrictCoarseRows[\s\S]*letr=fineLinearWorkgroup\(w,n\)[\s\S]*aggregates\[r\]=Aggregate/,
    "one workgroup exclusively owns each coarse-row reduction");
  assert.match(shader,
    /fnfinePage\(key:u32\)[\s\S]*directoryBase=7u\+p\.pageCapacity[\s\S]*worklist\[directoryBase\+key\]/,
    "row-owned gathers use the generation-validated direct fine-page publication");
  assert.doesNotMatch(shader, /fnfinePage\(key:u32\)[\s\S]*while\(low<high\)/,
    "row-owned gathers must not binary-search the compact worklist for every covered brick");
  assert.match(encode,
    /run\("prepare"\)[\s\S]*dispatchWorkgroupsIndirect\(this\.dispatch,0\)[\s\S]*run\("publish"\)/,
    "production schedules only singleton preparation/publication around the exact row-owned indirect dispatch");
  assert.doesNotMatch(encode, /planFineLevelSetDispatch2D|this\.plan\.rowCapacity,\s*this\.device\.limits/,
    "recurring restriction must never schedule the row-capacity bound from the host");
  assert.doesNotMatch(encode, /aggregateRestriction|fallback/i,
    "the legacy atomic scatter and fallback selector are deleted");
});

test("Dawn builds deterministic factor-4/factor-8 cell-center aggregates with O(rows) storage", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  assert.ok(adapter.limits.maxStorageBuffersPerShaderStage >= 10,
    "production fine restriction requires the M1-class ten-storage binding budget");
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 10 } });
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  for (const factor of [4, 8] as const) {
    const brickDimensions = factor / 4; const residentBricks = brickDimensions ** 3;
    const plan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [1, 1, 1],
      finestCellWidth: 1, fineFactor: factor, brickResolution: 4, maximumResidentBricks: residentBricks });
    const oracle = new FineLevelSetBrickOracle(plan);
    const keys = Array.from({ length: residentBricks }, (_, key) => packFineLevelSetBrickKey(plan,
      [key % brickDimensions, Math.floor(key / brickDimensions) % brickDimensions,
        Math.floor(key / (brickDimensions * brickDimensions))]));
    oracle.publishInterfaceAndRing(keys, ([x]) => x - 0.5);
    const owner = new WebGPUFineLevelSetBricks(device, plan);
    const source = owner.uploadGeneration(oracle.exportGPUGeneration());
    // Keep allocation capacity above the single live row. The CSR terminal is
    // rowOffsets[liveRows], not rowOffsets[rowCapacity].
    const restriction = new WebGPUFineToCoarseLevelSet(device, 2, plan.maximumResidentBricks * plan.samplesPerBrick);
    const headers = device.createBuffer({ size: 48, usage: storage });
    const header = new Uint32Array(12); header[3] = 1; device.queue.writeBuffer(headers, 0, header);
    const rowCount = device.createBuffer({ size: 4, usage: storage }); device.queue.writeBuffer(rowCount, 0, new Uint32Array([1]));
    const topologyControl = device.createBuffer({ size: 32, usage: storage });
    device.queue.writeBuffer(topologyControl, 0, new Uint32Array([0, 0, 0, 0, 1, 0, 0, 0]));
    const expected = factor ** 3, readBytes = 32;
    const readback = device.createBuffer({ size: readBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    const result = restriction.encode(new PassBroker(encoder), source, { headers, rowCount, topologyControl,
      dimensions: [1, 1, 1], physicalCellSize: 1, maximumLeafSize: 1 });
    encoder.copyBufferToBuffer(result.counts, 0, readback, 0, 8);
    encoder.copyBufferToBuffer(result.rowOffsets, 0, readback, 8, 8);
    encoder.copyBufferToBuffer(result.contributions, 0, readback, 16, 16);
    device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ); const bytes = readback.getMappedRange().slice(0); readback.unmap();
    const words = new Uint32Array(bytes); assert.deepEqual([...words.slice(0, 4)], [1, 1, 0, 1]);
    const values = new Float32Array(bytes, 16); const [center, minimum, maximum] = values;
    assert.ok(Number.isFinite(center)); assert.ok(Math.abs(center) <= 1e-6,
      `factor-${factor} restriction must evaluate the plane at cell center, got ${center}`);
    assert.equal(words[7], 1);
    assert.ok(minimum < 0 && maximum > 0, "restriction interval must retain the plane zero crossing");
    readback.destroy(); topologyControl.destroy(); rowCount.destroy(); headers.destroy(); restriction.destroy(); owner.destroy();
  }
  {
    // The fresh fine generation includes an air safety ring. Restriction is
    // row-owned, so resident samples outside compact wet/live rows are neither
    // scanned nor reported as skipped work.
    const plan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [2, 1, 1],
      finestCellWidth: 1, fineFactor: 4, brickResolution: 4, maximumResidentBricks: 2 });
    const oracle = new FineLevelSetBrickOracle(plan);
    oracle.publishInterfaceAndRing([packFineLevelSetBrickKey(plan, [0, 0, 0])], ([x]) => x - 0.5);
    const owner = new WebGPUFineLevelSetBricks(device, plan);
    const source = owner.uploadGeneration(oracle.exportGPUGeneration());
    const restriction = new WebGPUFineToCoarseLevelSet(device, 1, 2 * plan.samplesPerBrick);
    const headers = device.createBuffer({ size: 48, usage: storage });
    const header = new Uint32Array(12); header[3] = 1; device.queue.writeBuffer(headers, 0, header);
    const rowCount = device.createBuffer({ size: 4, usage: storage });
    device.queue.writeBuffer(rowCount, 0, new Uint32Array([1]));
    const topologyControl = device.createBuffer({ size: 32, usage: storage });
    device.queue.writeBuffer(topologyControl, 0, new Uint32Array([0, 0, 0, 0, 1, 0, 0, 0]));
    const readback = device.createBuffer({ size: 24, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    const result = restriction.encode(new PassBroker(encoder), source, { headers, rowCount, topologyControl,
      dimensions: [2, 1, 1], physicalCellSize: 1, maximumLeafSize: 1 });
    encoder.copyBufferToBuffer(result.counts, 0, readback, 0, 24);
    device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ); const words = new Uint32Array(readback.getMappedRange().slice(0)); readback.unmap();
    // count, maximumPerRow, flags, unacceptedRows, rowCount, valid. The single
    // live row's eight corners lie inside the published interface brick, so the
    // fine correction is complete and the coverage counter must stay at zero.
    assert.deepEqual([...words], [1, 1, 0, 0, 1, 0x8000_0000]);
    readback.destroy(); topologyControl.destroy(); rowCount.destroy(); headers.destroy(); restriction.destroy(); owner.destroy();
  }
  device.destroy();
});
