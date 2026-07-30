import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { planFineLevelSetBricks } from "../lib/octree-fine-levelset-bricks";
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
    /classifySourceBlocks[\s\S]*scanSourceBlocks[\s\S]*emitSourceRecords[\s\S]*sortSeedRecords[\s\S]*classifySeedRuns[\s\S]*scanSeedRuns[\s\S]*emitSeedRuns/,
    "fixed leaf records use one bounded cooperative ordering transaction followed by run-boundary compaction");
  assert.doesNotMatch(fineLevelSetLeafSeedWGSL,
    /histogramOwner|prefixOwner|scatterOwner|histogramKey|prefixKey|scatterKey|radixDigit|radixTotals/,
    "the old owner/key radix schedule and helpers must stay deleted");
  assert.doesNotMatch(fineLevelSetLeafSeedWGSL,
    /endpoint|boundaryQueries|powerFaceControl|RECURRING_SUPPORT/,
    "fine residency must not retain the retired generalized-face endpoint path");

  const encode = WebGPUFineLevelSetRedistance.prototype.encode.toString().replace(/\s+/g, "");
  assert.match(encode, /fineFactor!==4&&this\.source\.plan\.fineFactor!==8/);
  assert.match(encode,
    /this\.encodeJFA\(broker,bytes,options\.bandCells,warmStart\?maximumDisplacementFineCells:options\.bandCells,warmStart,tolerance\)/,
    "factor-8 uses the same mandatory JFA-CPT path as factor-4, including recurring warm-start bounds");
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
    /fn initializeDesiredSamples[\s\S]*if\(any\(q>=params\.sampleDimensions\)\)\{targetA\[index\]=0u;targetB\[index\]=0u;\}[\s\S]*fn initializeDesiredWorkSamples/,
    "topology owns factor-8 B4 domain clipping before fixed-resident redistance begins");
  assert.match(topology,
    /redistanceValid=arrayLength\(&redistanceControl\)>=4u&&redistanceControl\[0\]==0u&&\(redistanceControl\[2\]>0u\|\|pageDelta\[2\]==0u\)&&redistanceControl\[3\]!=0u/,
    "factor-8 topology remains provisional until redistance commits, while an exact empty dirty set needs no work");

  const projection = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
  const construction = projection.match(/const structured = new WebGPUDirectStructuredVelocityAuthority[\s\S]*?this\.globalFineTransportB = new WebGPUFineLevelSetTransport[\s\S]*?\);/)?.[0];
  assert.ok(construction, "production direct structured velocity and fine-transport construction must exist");
  assert.match(construction,
    /const fineTransportResources = \{[\s\S]*?structured: structuredSource,[\s\S]*?airSupport: \{[\s\S]*?arena: this\.airVelocitySupport\.source\.arena/,
    "factor-8 transport must consume the committed direct-row and suffix face authority");
  assert.doesNotMatch(construction, /closestPointSeeds|Structured closest-point extension seeds/,
    "the retired row-average extension cannot remain as a fallback authority");
  assert.doesNotMatch(construction,
    /WebGPUOctreePowerVelocity|WebGPUOctreeFaceClosestPoint|powerFaces|fineFactor\s*===\s*4/,
    "factor-8 must not select or construct a retired face-based velocity path");
  assert.doesNotMatch(projection,
    /globalFineFaceExtension|globalFineVelocityPrepass|globalFinePowerVelocity|powerFaceControl/,
    "production orchestration must contain no retired face-band authority");
  assert.match(topology,
    /var value=sampleCoarseOctreePhi\(position\);let seeded=externalSeedPhi[\s\S]*if\(finite\(seeded\)\)\{value=seeded;\}/,
    "new fine pages preserve coarse phi unless a real interface affine seed is finite");
  assert.match(topology,
    /fn exactAnalyticSeedPhi[\s\S]*heightFraction=max\(0\.92,fill\)[\s\S]*length\(max\(q,vec3f\(0\.0\)\)\)\+min\(max\(q\.x,max\(q\.y,q\.z\)\),0\.0\)/,
    "the cold authored dam is sampled exactly on the independent fine SPGrid instead of flattened to a coarse leaf plane");
  assert.match(topology,
    /damOffset=4u\+10u\*params\.pageCapacity[\s\S]*damDimensions=select\(fallback,authored[\s\S]*exposedMaximum=params\.domainOrigin\+damDimensions/,
    "the global fine cold seed reads authored absolute reservoir extents from its ABI tail");
  assert.match(topology,
    /exposedMaximum=params\.domainOrigin\+damDimensions[\s\S]*let q=point-exposedMaximum/,
    "cold fine phi must not create interfaces on the three closed tank-contact planes");
  assert.match(topology,
    /fn externalSeedPhi[\s\S]*currentFinePopulated\(\)[\s\S]*exactAnalyticSeedPhi/,
    "exact analytic initialization is cold-start only; recurring generations preserve transported fine phi");
  assert.match(topology,
    /let first=vec3f\(brick\*params\.brickResolution\)\/f32\(params\.fineFactor\);let last=vec3f\(\(brick\+vec3u\(1u\)\)\*params\.brickResolution\)\/f32\(params\.fineFactor\)/,
    "an analytic interface aligned with an SPGrid page boundary must seed the adjacent page supports");
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
  assert.equal(maximumResidentBricks, 405_000,
    "the production physical-band plan reserves 25 layers plus 50% surface-growth headroom");
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
  assert.equal(header[9], 2, "published worklist must carry generation 2");
  assert.equal(header[10], control.desiredBricks);

  headerReadback.destroy(); topology.destroy(); seeds.destroy(); leaves.destroy(); rowCount.destroy();
  owner.destroy(); device.destroy();
});
