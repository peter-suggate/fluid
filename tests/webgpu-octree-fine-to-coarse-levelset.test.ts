import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { FineLevelSetBrickOracle, packFineLevelSetBrickKey,
  planFineLevelSetBricks } from "../lib/octree-fine-levelset-bricks";
import { WebGPUFineLevelSetBricks } from "../lib/webgpu-octree-fine-levelset-bricks";
import { FINE_TO_COARSE_LEVELSET_ERROR, fineToCoarseLevelSetWGSL,
  WebGPUFineToCoarseLevelSet } from "../lib/webgpu-octree-fine-to-coarse-levelset";
import { PassBroker } from "../lib/webgpu-pass-broker";

test("fine-to-coarse restriction rejects an unpublished or stale fine source", () => {
  const shader = fineToCoarseLevelSetWGSL.replace(/\s+/g, "");
  const encode = WebGPUFineToCoarseLevelSet.prototype.encode.toString().replace(/\s+/g, "");
  assert.match(encode, /^encode\(broker,fine,input\)/);
  assert.doesNotMatch(encode, /newPassBroker|broker\.fence/,
    "restriction must not terminate its caller-owned publication pass");
  assert.equal(FINE_TO_COARSE_LEVELSET_ERROR.unpublishedSource, 8);
  assert.match(shader,
    /if\(arrayLength\(&worklist\)<5u\|\|arrayLength\(&topologyControl\)<8u\)[\s\S]*topologyControl\[0\]!=0u\|\|topologyControl\[4\]!=1u\|\|topologyControl\[5\]!=0u\|\|topologyControl\[7\]!=0u/,
    "restriction must consume only an accepted, non-rollback fine transaction");
  assert.match(shader,
    /fnpublishRestriction[\s\S]*if\(control\.flags==0u\)[\s\S]*else\{control\.count=0xffffffffu/,
    "a rejected fine source must poison the downstream coarse correction rather than publish an empty correction");
  assert.match(encode, /prepare:\[0,2,7,9,13,14\]/,
    "the scalar prepare pass must bind the fine worklist and topology transaction it validates");
  assert.doesNotMatch(encode, /finalizeRestrictionRows|run\("finalize"/,
    "row owners publish accepted aggregates directly without a second capacity-wide pass");
  assert.match(encode, /if\(input\.diagnoseUnownedFineSamples\)run\("diagnose",1\)/,
    "the capacity-wide unowned-sample scan must remain an explicit QA opt-in");
});

test("resident fine-band samples may lead the compact liquid-row set", () => {
  const diagnose = fineToCoarseLevelSetWGSL.match(
    /fn diagnoseUnownedFineSamples[\s\S]*?control\.flags\|=diagnosticErrors\[0\];}/,
  )?.[0] ?? "";
  assert.match(diagnose, /unowned\+=1u/,
    "one deterministic workgroup keeps unowned fine samples observable");
  assert.match(diagnose, /maximumUnownedLiquidMagnitude/,
    "advancing non-positive misses retain a magnitude diagnostic");
  assert.doesNotMatch(diagnose, /control\.flags\|=UNOWNED/,
    "a resident fine sample does not require a compact pressure-row fallback");
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
    /fnfinePage\(key:u32\)[\s\S]*directoryBase=5u\+p\.pageCapacity[\s\S]*worklist\[directoryBase\+key\]/,
    "row-owned gathers use the generation-validated direct fine-page publication");
  assert.doesNotMatch(shader, /fnfinePage\(key:u32\)[\s\S]*while\(low<high\)/,
    "row-owned gathers must not binary-search the compact worklist for every covered brick");
  assert.match(encode,
    /run\("restrict",rows\.x,rows\.y\)[\s\S]*if\(input\.diagnoseUnownedFineSamples\)run\("diagnose",1\)/,
    "production schedules only row-owned restriction; the deterministic capacity scan is opt-in");
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
  const adapter = await gpu.requestAdapter(); assert.ok(adapter); const device = await adapter.requestDevice();
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
    const restriction = new WebGPUFineToCoarseLevelSet(device, 1, plan.maximumResidentBricks * plan.samplesPerBrick);
    const headers = device.createBuffer({ size: 48, usage: storage });
    const header = new Uint32Array(12); header[3] = 1; device.queue.writeBuffer(headers, 0, header);
    const rowDirectory = device.createBuffer({ size: 32, usage: storage });
    device.queue.writeBuffer(rowDirectory, 0, new Uint32Array([1, 1, 0, 0]));
    const rowCount = device.createBuffer({ size: 4, usage: storage }); device.queue.writeBuffer(rowCount, 0, new Uint32Array([1]));
    const topologyControl = device.createBuffer({ size: 32, usage: storage });
    device.queue.writeBuffer(topologyControl, 0, new Uint32Array([0, 0, 0, 0, 1, 0, 0, 0]));
    const expected = factor ** 3, readBytes = 32;
    const readback = device.createBuffer({ size: readBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    const result = restriction.encode(new PassBroker(encoder), source, { headers, rowDirectory, rowCount, topologyControl,
      dimensions: [1, 1, 1], physicalCellSize: 1, maximumLeafSize: 1, rowDirectoryCapacity: 1 });
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
    readback.destroy(); topologyControl.destroy(); rowCount.destroy(); rowDirectory.destroy(); headers.destroy(); restriction.destroy(); owner.destroy();
  }
  {
    // The fresh fine generation includes an air safety ring.  Positive samples
    // in that ring can legitimately lie beyond the compact wet/live rows; they
    // are counted but must not invalidate restriction of the owned interface.
    const plan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [2, 1, 1],
      finestCellWidth: 1, fineFactor: 4, brickResolution: 4, maximumResidentBricks: 2 });
    const oracle = new FineLevelSetBrickOracle(plan);
    oracle.publishInterfaceAndRing([packFineLevelSetBrickKey(plan, [0, 0, 0])], ([x]) => x - 0.5);
    const owner = new WebGPUFineLevelSetBricks(device, plan);
    const source = owner.uploadGeneration(oracle.exportGPUGeneration());
    const restriction = new WebGPUFineToCoarseLevelSet(device, 1, 2 * plan.samplesPerBrick);
    const headers = device.createBuffer({ size: 48, usage: storage });
    const header = new Uint32Array(12); header[3] = 1; device.queue.writeBuffer(headers, 0, header);
    const rowDirectory = device.createBuffer({ size: 32, usage: storage });
    device.queue.writeBuffer(rowDirectory, 0, new Uint32Array([1, 1, 0, 0]));
    const rowCount = device.createBuffer({ size: 4, usage: storage });
    device.queue.writeBuffer(rowCount, 0, new Uint32Array([1]));
    const topologyControl = device.createBuffer({ size: 32, usage: storage });
    device.queue.writeBuffer(topologyControl, 0, new Uint32Array([0, 0, 0, 0, 1, 0, 0, 0]));
    const readback = device.createBuffer({ size: 24, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    const result = restriction.encode(new PassBroker(encoder), source, { headers, rowDirectory, rowCount, topologyControl,
      dimensions: [2, 1, 1], physicalCellSize: 1, maximumLeafSize: 1, rowDirectoryCapacity: 1 });
    encoder.copyBufferToBuffer(result.counts, 0, readback, 0, 24);
    device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ); const words = new Uint32Array(readback.getMappedRange().slice(0)); readback.unmap();
    assert.deepEqual([...words], [1, 1, 0, plan.samplesPerBrick, 1, 0x8000_0000]);
    readback.destroy(); topologyControl.destroy(); rowCount.destroy(); rowDirectory.destroy(); headers.destroy(); restriction.destroy(); owner.destroy();
  }
  device.destroy();
});
