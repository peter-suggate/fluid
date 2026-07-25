import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { FineLevelSetBrickOracle, packFineLevelSetBrickKey,
  planFineLevelSetBricks } from "../lib/octree-fine-levelset-bricks";
import { WebGPUFineLevelSetBricks } from "../lib/webgpu-octree-fine-levelset-bricks";
import { PassBroker } from "../lib/webgpu-pass-broker";
import { planGlobalFineNarrowBandBrickCapacity, resolveGlobalFineBrickCapacity } from
  "../lib/webgpu-octree";
import { WebGPUFineLevelSetRedistance } from "../lib/webgpu-octree-fine-levelset-redistance";
import { planFineLevelSetSummaryLeafLookup } from "../lib/webgpu-octree-fine-levelset-summary";
import { WebGPUFineLevelSetLeafSeeds, WebGPUFineLevelSetTopology,
  fineLevelSetLeafSeedWGSL, makeFineLevelSetTopologyWGSL, planFineLevelSetLeafBrickBounds,
  unpackFineLevelSetGPUTopologyControl } from "../lib/webgpu-octree-fine-levelset-topology";

test("factor-8 B4 topology pre-dilates and clips while redistance remains fixed-resident", () => {
  const plan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [60, 45, 40],
    finestCellWidth: 1, fineFactor: 8, brickResolution: 4, maximumResidentBricks: 64 });
  assert.deepEqual(plan.brickDimensions, [120, 90, 80]);
  assert.deepEqual(planFineLevelSetLeafBrickBounds(plan, [17, 12, 9], 1), {
    first: [34, 24, 18], last: [35, 25, 19], bricksPerFinestCell: 2, brickCount: 8,
  });
  assert.match(fineLevelSetLeafSeedWGSL,
    /fn leafFirst\(leaf:FineSeedLeaf\)->vec3u\{return leafOrigin\(leaf\)\*params\.header\.x\/params\.header\.y;\}[\s\S]*return min\(high\/params\.header\.y/);
  assert.doesNotMatch(fineLevelSetLeafSeedWGSL, /@workgroup_size\(1\)/,
    "fine seed classification, sorting, and run compaction must not retain a single-lane hot path");
  assert.match(fineLevelSetLeafSeedWGSL,
    /classifySourceBlocks[\s\S]*classifyEndpointBlocks[\s\S]*scanSourceBlocks[\s\S]*emitSourceRecords[\s\S]*emitEndpointRecords[\s\S]*sortSeedRecords[\s\S]*classifySeedRuns[\s\S]*scanSeedRuns[\s\S]*emitSeedRuns/,
    "fixed leaf records use one bounded cooperative ordering transaction followed by run-boundary compaction");
  assert.doesNotMatch(fineLevelSetLeafSeedWGSL,
    /histogramOwner|prefixOwner|scatterOwner|histogramKey|prefixKey|scatterKey|radixDigit|radixTotals/,
    "the old owner/key radix schedule and helpers must stay deleted");
  assert.match(fineLevelSetLeafSeedWGSL,
    /fn endpointRecordCount[\s\S]*classifyEndpointBlocks[\s\S]*endpointRecordCount\(face\)[\s\S]*emitEndpointRecords[\s\S]*endpointRecordCount\(face\)/,
    "power endpoints must reserve and emit only exact in-domain trilinear records");
  assert.doesNotMatch(fineLevelSetLeafSeedWGSL,
    /16u\*endpointCount\(\)|validateEndpointCapacity/,
    "the deleted worst-case endpoint reservation must not reject a sparse exact endpoint set");

  const encode = WebGPUFineLevelSetRedistance.prototype.encode.toString().replace(/\s+/g, "");
  assert.match(encode, /fineFactor!==4&&this\.source\.plan\.fineFactor!==8/);
  assert.match(encode, /this\.encodeJFA\(broker,bytes,options\.bandCells,tolerance\)/,
    "factor-8 uses the same mandatory fixed-resident JFA-CPT path as factor-4");
  assert.doesNotMatch(encode, /method===|fast-sweeping/,
    "the retired redistance selector must stay deleted");
  assert.doesNotMatch(encode, /requestPipeline|finishActivationPipeline/,
    "redistance no longer interleaves topology allocation with distance propagation");
  assert.deepEqual(planFineLevelSetSummaryLeafLookup(plan.brickDimensions,
    plan.finestCellDimensions, [17, 12, 9], 1, plan.samplesPerBrick), {
    level: 1, key: 864_000 + 17 + 60 * (12 + 45 * 9), brickSide: 2,
    expectedBrickCount: 8, expectedSampleCount: 512,
  });
  const topology = makeFineLevelSetTopologyWGSL(
    "fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.x;}",
  );
  assert.match(topology,
    /fn initializeDesiredSamples[\s\S]*if\(any\(q>=params\.sampleDimensions\)\)\{targetA\[index\]=0u;targetB\[index\]=0u;\}/,
    "topology owns factor-8 B4 domain clipping before fixed-resident redistance begins");
  assert.match(topology,
    /redistanceValid=arrayLength\(&redistanceControl\)>=4u&&redistanceControl\[0\]==0u&&\(redistanceControl\[2\]>0u\|\|pageDelta\[2\]==0u\)&&redistanceControl\[3\]!=0u/,
    "factor-8 topology remains provisional until redistance commits, while an exact empty dirty set needs no work");

  const projection = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
  const construction = projection.match(/this\.globalFineVelocityPrepass = new WebGPUOctreePowerVelocityPrepass[\s\S]*?this\.globalFineTransportB = new WebGPUFineLevelSetTransport[\s\S]*?\);/)?.[0];
  assert.ok(construction, "production fine-velocity construction block must exist");
  assert.doesNotMatch(construction, /fineFactor\s*===\s*4/,
    "factor-8 must construct the regular/transition face publication rather than bypassing it");
  assert.match(construction,
    /new WebGPUOctreeFaceClosestPointExtension\([\s\S]*?this\.powerFaces\.plan\.faceCapacity/,
    "the factor-8 closest-point extension must retain the bounded generalized-face capacity");
  assert.match(projection,
    /this\.globalFineFaceExtension\.encodePhase\([\s\S]*?powerFaces:\s*this\.powerFaces\.source/,
    "the production factor-8 publication must receive the authoritative power-face source");
  assert.match(fineLevelSetLeafSeedWGSL,
    /endpointValid[\s\S]*powerFaceControl\[3\]==0u&&powerFaceControl\[8\]==0x80000000u/,
    "only a clean published power-face generation may extend fine residency");
  assert.match(fineLevelSetLeafSeedWGSL,
    /let local=ordinal&7u;let lattice=\(position-params\.fineDomain\.xyz\)\/params\.fineDomain\.w-vec3f\(0\.5\)/,
    "both endpoint samples retain every trilinear lattice contributor");
  assert.match(fineLevelSetLeafSeedWGSL,
    /sortSeedRecords[\s\S]*fn runHasEndpoint[\s\S]*fn emitSeedRuns/,
    "owner-first stable sorting preserves affine records before support-only endpoint duplicates");
  assert.match(fineLevelSetLeafSeedWGSL,
    /seeds\[seedTagBase\(\)\+output\]=output\|select\(0u,RECURRING_SUPPORT,runHasEndpoint\(index,record\.x\)\)/,
    "an endpoint duplicate preserves the cold affine plane while marking the key for recurring residency");
  assert.match(fineLevelSetLeafSeedWGSL,
    /else\{seeds\[base\]=0u;[\s\S]*seeds\[base\+4u\]=bitcast<u32>\(3\.402823e38\)/,
    "support-only seeds use the strict-invalid sentinel and therefore initialize phi from the coarse level set");
  assert.match(topology,
    /var value=sampleCoarseOctreePhi\(position\);let seeded=externalSeedPhi[\s\S]*if\(finite\(seeded\)\)\{value=seeded;\}/,
    "endpoint support preserves the paper's coarse phi unless a real interface affine seed is finite");
  assert.match(topology,
    /fn exactAnalyticSeedPhi[\s\S]*heightFraction=max\(0\.92,fill\)[\s\S]*length\(max\(q,vec3f\(0\.0\)\)\)\+min\(max\(q\.x,max\(q\.y,q\.z\)\),0\.0\)/,
    "the cold authored dam is sampled exactly on the independent fine SPGrid instead of flattened to a coarse leaf plane");
  assert.match(topology,
    /exposedMaximum=params\.domainOrigin\+vec3f[\s\S]*let q=point-exposedMaximum/,
    "cold fine phi must not create interfaces on the three closed tank-contact planes");
  assert.match(topology,
    /fn externalSeedPhi[\s\S]*currentFinePopulated\(\)[\s\S]*exactAnalyticSeedPhi/,
    "exact analytic initialization is cold-start only; recurring generations preserve transported fine phi");
  assert.match(topology,
    /let first=vec3f\(brick\*params\.brickResolution\)\/f32\(params\.fineFactor\);let last=vec3f\(\(brick\+vec3u\(1u\)\)\*params\.brickResolution\)\/f32\(params\.fineFactor\)/,
    "an analytic interface aligned with an SPGrid page boundary must seed the adjacent page supports");
});

test("Dawn endpoint seeds cover exact factor-4/factor-8 trilinear support without replacing interface phi", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU fine-levelset checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create(["backend=metal"]); const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  assert.ok(adapter.limits.maxStorageBuffersPerShaderStage >= 10);
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 10 } });
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const upload = (data: ArrayBufferView<ArrayBuffer>) => {
    const buffer = device.createBuffer({ size: Math.max(4, data.byteLength), usage: storage });
    device.queue.writeBuffer(buffer, 0, data); return buffer;
  };
  for (const factor of [4, 8] as const) {
    const plan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [4, 4, 4],
      finestCellWidth: 1, fineFactor: factor, brickResolution: 4, maximumResidentBricks: 64 });
    const owner = new WebGPUFineLevelSetBricks(device, plan); const target = owner.initializeEmptyGPUGeneration(1);
    const leafBytes = new ArrayBuffer(64); const leafWords = new Uint32Array(leafBytes);
    const leafFloats = new Float32Array(leafBytes);
    leafWords.set([1, 1, 1, 1, 2]); leafFloats[8] = -0.25; leafFloats[9] = 0.5;
    const leaves = upload(new Uint8Array(leafBytes)); const rowCount = upload(new Uint32Array([1, 0]));
    const queryFloats = new Float32Array([1.5, 1.5, 1.5, 1, 2.5, 1.5, 1.5, 1]);
    const queries = upload(queryFloats); const controlWords = new Uint32Array(16);
    controlWords[1] = 1; controlWords[8] = 0x8000_0000; const control = upload(controlWords);
    const seeds = new WebGPUFineLevelSetLeafSeeds(device, target);
    const readback = device.createBuffer({ size: seeds.buffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    const broker = new PassBroker(encoder);
    seeds.encodeFromAllInterfaceLeaves(broker, { buffer: leaves }, { buffer: rowCount },
      { queries: { buffer: queries }, control: { buffer: control } });
    broker.copyBufferToBuffer(seeds.buffer, 0, readback, 0, seeds.buffer.size);
    device.queue.submit([broker.finish()]); await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ); const words = new Uint32Array(readback.getMappedRange());
    const count = words[0], keys = Array.from(words.slice(4, 4 + count));
    assert.equal(words[1], 0); assert.equal(count, factor === 4 ? 2 : 16);
    const tagBase = 4 + plan.maximumResidentBricks;
    const planeBase = tagBase + plan.maximumResidentBricks;
    for (const [index, key] of keys.entries()) {
      let valueIndex = words[tagBase + index];
      const recurringSupport = (valueIndex & 0x8000_0000) !== 0;
      valueIndex &= 0x7fff_ffff;
      const phi = new Float32Array(new Uint32Array([words[planeBase + valueIndex * 8 + 4]]).buffer)[0];
      const brickX = key % plan.brickDimensions[0];
      assert.equal(recurringSupport, brickX >= factor / 4 && brickX <= 3 * factor / 4 - 1,
        "only exact endpoint pages carry recurring support");
      if (brickX < factor / 2) assert.equal(phi, -0.25,
        "an endpoint duplicate must retain the interface leaf's affine phi");
      else assert.ok(phi > 3e38,
        "a support-only endpoint must retain the coarse-initialization sentinel");
    }
    readback.unmap(); readback.destroy(); seeds.destroy(); leaves.destroy(); rowCount.destroy(); queries.destroy(); control.destroy(); owner.destroy();
  }
  device.destroy();
});

test("Dawn recurring endpoint support enters a published topology without recurring affine phi", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU fine-levelset checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  assert.ok(adapter.limits.maxStorageBuffersPerShaderStage >= 10);
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 10 } });
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const upload = (data: ArrayBufferView<ArrayBuffer>) => {
    const buffer = device.createBuffer({ size: Math.max(4, data.byteLength), usage: storage });
    device.queue.writeBuffer(buffer, 0, data); return buffer;
  };
  const run = async (validFaceControl: boolean) => {
    const plan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [8, 1, 1],
      finestCellWidth: 1, fineFactor: 4, brickResolution: 4, maximumResidentBricks: 8 });
    const oracle = new FineLevelSetBrickOracle(plan);
    oracle.publishInterfaceAndRing([packFineLevelSetBrickKey(plan, [2, 0, 0])], ([x]) => x - 2.5);
    const owner = new WebGPUFineLevelSetBricks(device, plan);
    const current = owner.uploadGeneration(oracle.exportGPUGeneration());
    const next = owner.prepareGPUGeneration(2);

    // Key 0 is an ordinary affine CORE seed and must remain cold-only. Key 4
    // has the same affine seed but is also an exact power endpoint, so the
    // recurring marker must retain it without applying the -77 affine value.
    const leafBytes = new ArrayBuffer(2 * 64); const leafWords = new Uint32Array(leafBytes);
    const leafFloats = new Float32Array(leafBytes);
    for (const [row, x] of [0, 4].entries()) {
      const base = row * 16; leafWords.set([x, 0, 0, 1, 2], base); leafFloats[base + 8] = -77;
    }
    const leaves = upload(new Uint8Array(leafBytes)); const rowCount = upload(new Uint32Array([2, 0]));
    const queries = upload(new Float32Array([3.5, 0.5, 0.5, 1, 4.5, 0.5, 0.5, 1]));
    const controlWords = new Uint32Array(16); controlWords[1] = 1;
    if (validFaceControl) controlWords[8] = 0x8000_0000;
    else controlWords[3] = 8;
    const faceControl = upload(controlWords);
    const seeds = new WebGPUFineLevelSetLeafSeeds(device, next);
    const topology = new WebGPUFineLevelSetTopology(device, current, next,
      "fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.x-2.5;}");

    const controlBytes = 36; const metadataBytes = plan.maximumResidentBricks * 10 * 4;
    const worklistBytes = (5 + plan.maximumResidentBricks) * 4;
    const phiBytes = plan.maximumResidentBricks * plan.samplesPerBrick * 4;
    const totalBytes = controlBytes + metadataBytes + worklistBytes + phiBytes;
    const readback = device.createBuffer({ size: totalBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    const broker = new PassBroker(encoder);
    const seedSource = seeds.encodeFromAllInterfaceLeaves(broker, { buffer: leaves }, { buffer: rowCount },
      { queries: { buffer: queries }, control: { buffer: faceControl } });
    topology.encode(broker, seedSource);
    encoder.copyBufferToBuffer(topology.control, 0, readback, 0, controlBytes);
    encoder.copyBufferToBuffer(next.metadata, 0, readback, controlBytes, metadataBytes);
    encoder.copyBufferToBuffer(next.worklist, 0, readback, controlBytes + metadataBytes, worklistBytes);
    encoder.copyBufferToBuffer(next.phi, 0, readback, controlBytes + metadataBytes + worklistBytes, phiBytes);
    device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ); const bytes = readback.getMappedRange().slice(0); readback.unmap();
    const topologyControl = unpackFineLevelSetGPUTopologyControl(new Uint32Array(bytes, 0, 9));
    assert.equal(topologyControl.flags, 0); assert.equal(topologyControl.published, true);
    const metadata = new Uint32Array(bytes, controlBytes, metadataBytes / 4);
    const worklist = new Uint32Array(bytes, controlBytes + metadataBytes, worklistBytes / 4);
    const phi = new Float32Array(bytes, controlBytes + metadataBytes + worklistBytes, phiBytes / 4);
    const idsByKey = new Map<number, number>();
    for (let work = 0; work < worklist[0]; work += 1) {
      const id = worklist[5 + work]; idsByKey.set(metadata[id * 10 + 1], id);
    }
    assert.equal(idsByKey.has(0), false, "ordinary affine CORE keys stay cold-only");
    if (validFaceControl) {
      assert.ok(idsByKey.has(4), "the exact endpoint key must recur");
      assert.ok(idsByKey.has(5), "the endpoint's paper Section 5 one-ring must recur");
      for (const key of [4, 5]) {
        const id = idsByKey.get(key)!; const expected = key + 0.125 - 2.5;
        assert.ok(Math.abs(phi[id * plan.samplesPerBrick] - expected) < 1e-6,
          `support page ${key} must initialize from coarse phi rather than the -77 affine seed`);
      }
    } else {
      assert.equal(idsByKey.has(4), false, "an invalid/unpublished face record cannot extend residency");
      assert.equal(idsByKey.has(5), false, "an invalid endpoint cannot contribute a forward ring");
    }
    readback.destroy(); topology.destroy(); seeds.destroy(); leaves.destroy(); rowCount.destroy();
    queries.destroy(); faceControl.destroy(); owner.destroy();
  };
  await run(true); await run(false); device.destroy();
});

test("Dawn production-width factor-8 topology publishes the complete twelve-ring support within planned capacity", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU fine-levelset checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create(["backend=metal"]); const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  assert.ok(adapter.limits.maxStorageBuffersPerShaderStage >= 10);
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 10 } });
  const brickDimensions = [120, 90, 80] as const;
  const capacityPlan = planGlobalFineNarrowBandBrickCapacity(brickDimensions, 12);
  const maximumResidentBricks = resolveGlobalFineBrickCapacity(
    capacityPlan.maximumResidentBricks, undefined,
    device.limits.maxComputeWorkgroupsPerDimension, 64,
    Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize), 64,
  );
  assert.equal(maximumResidentBricks, 337_500,
    "the production physical-band plan reserves 25 layers plus 25% surface-growth headroom");
  const plan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [60, 45, 40],
    finestCellWidth: 1, fineFactor: 8, brickResolution: 4, maximumResidentBricks });
  const owner = new WebGPUFineLevelSetBricks(device, plan);
  const current = owner.initializeEmptyGPUGeneration(1); const next = owner.prepareGPUGeneration(2);

  // One compact FineSeedLeaf per x/y column along the domain's largest planar
  // interface. This stresses the same maximum-area orientation used by the
  // production physical-band capacity planner.
  // At factor eight each leaf covers 2^3 globally keyed bricks; the topology
  // pass must deduplicate these and add the complete production support: the
  // factor-8 default has max(8 backtrace + 1 interpolation, 43 redistance)
  // cells, rounded to eleven B4 rings plus one publication-safety ring.
  const leafCount = 60 * 45; const leafBytes = new ArrayBuffer(leafCount * 64);
  const words = new Uint32Array(leafBytes); const floats = new Float32Array(leafBytes);
  for (let y = 0; y < 45; y += 1) for (let x = 0; x < 60; x += 1) {
    const row = x + 60 * y, base = row * 16;
    words[base] = x; words[base + 1] = y; words[base + 2] = 10; words[base + 3] = 1; words[base + 4] = 2;
    floats[base + 8] = 0; floats[base + 11] = 1;
  }
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const leaves = device.createBuffer({ size: leafBytes.byteLength, usage: storage });
  const rowCount = device.createBuffer({ size: 8, usage: storage });
  device.queue.writeBuffer(leaves, 0, leafBytes); device.queue.writeBuffer(rowCount, 0, new Uint32Array([leafCount]));
  const seeds = new WebGPUFineLevelSetLeafSeeds(device, next);
  const topology = new WebGPUFineLevelSetTopology(device, current, next,
    "fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.z-10.5;}");
  const headerReadback = device.createBuffer({ size: 56, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  const broker = new PassBroker(encoder);
  topology.encode(broker, seeds.encodeFromAllInterfaceLeaves(broker, { buffer: leaves }, { buffer: rowCount }), [], {
    maximumBacktraceFineCells: 8,
    interpolationSupportFineCells: 1,
    redistanceBandFineCells: 43,
    safetyBrickRings: 1,
  });
  encoder.copyBufferToBuffer(topology.control, 0, headerReadback, 0, 36);
  encoder.copyBufferToBuffer(next.worklist, 0, headerReadback, 36, 20);
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
  await headerReadback.mapAsync(GPUMapMode.READ);
  const header = new Uint32Array(headerReadback.getMappedRange().slice(0)); headerReadback.unmap();
  const control = unpackFineLevelSetGPUTopologyControl(header);
  assert.equal(control.flags, 0, JSON.stringify({ header: [...header], control }));
  assert.equal(control.published, true); assert.equal(control.rolledBack, false);
  assert.equal(control.interfaceBricks, 21_600,
    "the cold checkpoint must retain the exact externally seeded interface-brick population");
  assert.equal(control.desiredBricks, 280_800,
    "a maximum-area x/y factor-8 plane dilated by twelve rings occupies exactly 120 x 90 x 26 bricks");
  assert.equal(control.requiredDesiredBricks, control.desiredBricks);
  assert.equal(control.requiredDesiredBricksExact, true);
  assert.equal(control.dilationBrickRings, 12);
  assert.equal(maximumResidentBricks - control.desiredBricks, 56_700);
  assert.ok(control.desiredBricks <= maximumResidentBricks);
  assert.equal(header[8], control.interfaceSeedBricks);
  assert.equal(header[9], control.desiredBricks);
  assert.equal(header[10], 2, "published worklist must carry generation 2");

  headerReadback.destroy(); topology.destroy(); seeds.destroy(); leaves.destroy(); rowCount.destroy();
  owner.destroy(); device.destroy();
});
