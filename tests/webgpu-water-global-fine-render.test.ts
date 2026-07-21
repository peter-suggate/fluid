import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  FineLevelSetBrickOracle,
  packFineLevelSetBrickKey,
  planFineLevelSetBricks,
  type FineLevelSetFactor,
} from "../lib/octree-fine-levelset-bricks";
import { createGlobalFineLevelSetConsumerSource } from "../lib/octree-consumer-sampling";
import { WebGPUFineLevelSetBricks } from "../lib/webgpu-octree-fine-levelset-bricks";
import {
  extractionPrepareShader,
  globalFineFallbackMaySeedRenderer,
  globalFineSurfaceDispatch,
  RasterWaterPipeline,
  surfaceExtractionShader,
  WATER_INTERFACE_CULL_MODES,
  waterSurfaceGeometrySource,
} from "../lib/webgpu-water-pipeline";
import { globalFineClassifiedCountShader, globalFineClassifiedEmitShader, globalFineClassifiedEmitShaders, globalFineClassifiedScanShader } from "../lib/webgpu-water-global-fine-tetra";
import { globalFineSurfaceClassificationShader } from "../lib/webgpu-water-global-fine-classify";

const modulePath = process.env.WEBGPU_NODE_MODULE;

function initializedBuffer(device: GPUDevice, data: ArrayBufferView, usage: GPUBufferUsageFlags): GPUBuffer {
  const buffer = device.createBuffer({ size: Math.max(4, data.byteLength), usage, mappedAtCreation: true });
  new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  buffer.unmap();
  return buffer;
}

type V3 = readonly [number, number, number];
const tetrahedra = [
  [0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
  [0, 7, 4, 6], [0, 5, 6, 4], [0, 5, 1, 6],
] as const;
const cubeCorners: readonly V3[] = [
  [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
];
const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const subtract = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale = (value: V3, amount: number): V3 => [value[0] * amount, value[1] * amount, value[2] * amount];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V3, b: V3): V3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const normalize = (value: V3): V3 => scale(value, 1 / Math.hypot(...value));

function tetraSurface(points: readonly V3[], values: readonly number[], normal: V3): V3[][] {
  const edge = (a: number, b: number): V3 => {
    const t = Math.max(0.02, Math.min(0.98, (0.5 - values[a]) / (values[b] - values[a])));
    return add(scale(points[a], 1 - t), scale(points[b], t));
  };
  const raw: V3[][] = [];
  const tri = (a: V3, b: V3, c: V3) => {
    raw.push(dot(cross(subtract(b, a), subtract(c, a)), normal) >= 0 ? [a, b, c] : [a, c, b]);
  };
  const mask = values.reduce((bits, value, index) => bits | (value >= 0.5 ? 1 << index : 0), 0);
  if (mask === 1 || mask === 14) tri(edge(0, 1), edge(0, 2), edge(0, 3));
  else if (mask === 2 || mask === 13) tri(edge(1, 0), edge(1, 3), edge(1, 2));
  else if (mask === 4 || mask === 11) tri(edge(2, 0), edge(2, 1), edge(2, 3));
  else if (mask === 8 || mask === 7) tri(edge(3, 0), edge(3, 2), edge(3, 1));
  else if (mask === 3 || mask === 12) {
    const ac = edge(0, 2), ad = edge(0, 3), bc = edge(1, 2), bd = edge(1, 3);
    tri(ac, bc, bd); tri(ac, bd, ad);
  } else if (mask === 5 || mask === 10) {
    const ab = edge(0, 1), ad = edge(0, 3), cb = edge(2, 1), cd = edge(2, 3);
    tri(ab, cb, cd); tri(ab, cd, ad);
  } else if (mask === 6 || mask === 9) {
    const ba = edge(1, 0), bd = edge(1, 3), ca = edge(2, 0), cd = edge(2, 3);
    tri(ba, ca, cd); tri(ba, cd, bd);
  }
  return raw;
}

test("global fine planar surfaces publish outward CCW triangles for front-face culling", () => {
  const sources = [...globalFineClassifiedEmitShaders, globalFineClassifiedEmitShader];
  for (const source of sources) {
    assert.match(source, /n=-normalize\(gradient\)/,
      "negative-inside phi must turn the decreasing occupancy gradient into an outward normal");
    assert.match(source,
      /if\(dot\(cross\(y\.position\.xyz-x\.position\.xyz,z\.position\.xyz-x\.position\.xyz\),n\)>=0\.\)\{out\[first\+1u\]=y;out\[first\+2u\]=z;\}else\{out\[first\+1u\]=z;out\[first\+2u\]=y;\}/,
      "every emitted triangle must be reordered against its outward world-space normal");
  }
  assert.deepEqual(WATER_INTERFACE_CULL_MODES, { front: "back", back: "front" });
  const initialize = RasterWaterPipeline.prototype.initialize.toString();
  assert.match(initialize, /frontFace:"ccw",cullMode/);
  assert.match(initialize, /surfaceFrontPipeline=.*WATER_INTERFACE_CULL_MODES\.front/,
    "the production entry-interface pass must retain outward CCW faces");

  const cell: V3 = [0.7, 1.3, 2.1];
  const worldCorners = cubeCorners.map((q): V3 => [q[0] * cell[0], q[1] * cell[1], q[2] * cell[2]]);
  const centre: V3 = [0.5 * cell[0], 0.5 * cell[1], 0.5 * cell[2]];
  const directions = [-1, 0, 1].flatMap(x => [-1, 0, 1].flatMap(y => [-1, 0, 1]
    .flatMap(z => x === 0 && y === 0 && z === 0 ? [] : [normalize([x, y, z])])));
  let triangles = 0;
  for (const outward of directions) for (const offset of [-0.35, -0.1, 0, 0.1, 0.35]) {
    const occupancy = worldCorners.map(point => 0.5 - (dot(outward, subtract(point, centre)) - offset) / 10);
    if (occupancy.every(value => value < 0.5) || occupancy.every(value => value >= 0.5)) continue;
    const gx = 0.25 * ((occupancy[1] + occupancy[2] + occupancy[5] + occupancy[6])
      - (occupancy[0] + occupancy[3] + occupancy[4] + occupancy[7]));
    const gy = 0.25 * ((occupancy[2] + occupancy[3] + occupancy[6] + occupancy[7])
      - (occupancy[0] + occupancy[1] + occupancy[4] + occupancy[5]));
    const gz = 0.25 * ((occupancy[4] + occupancy[5] + occupancy[6] + occupancy[7])
      - (occupancy[0] + occupancy[1] + occupancy[2] + occupancy[3]));
    const normal = normalize([-gx / cell[0], -gy / cell[1], -gz / cell[2]]);
    for (const ids of tetrahedra) {
      const points = ids.map(index => worldCorners[index]);
      const values = ids.map(index => occupancy[index]);
      for (const triangle of tetraSurface(points, values, normal)) {
        const geometric = cross(subtract(triangle[1], triangle[0]), subtract(triangle[2], triangle[0]));
        assert.ok(dot(geometric, normal) > 1e-10,
          "a nondegenerate planar interface triangle must face its published outward normal");
        triangles += 1;
      }
    }
  }
  assert.ok(triangles > 500, "the oracle must cover many orientations, offsets, and tetrahedron cases");
});

function compactCoarsePlane(device: GPUDevice): GPUBuffer {
  const capacity=8, bytes=new ArrayBuffer(32+capacity*32), u32=new Uint32Array(bytes), f32=new Float32Array(bytes);
  u32.set([0x80000000,1,capacity,1,2,2,2],0);f32[7]=1;
  const hash=(cell:number)=>{let value=(cell^Math.imul(1,0x9e3779b9))>>>0;value=Math.imul((value^(value>>>16))>>>0,0x7feb352d)>>>0;value=Math.imul((value^(value>>>15))>>>0,0x846ca68b)>>>0;return (value^(value>>>16))>>>0;};
  for(let cell=0;cell<8;cell+=1){let slot=hash(cell)&(capacity-1);while(u32[8+slot*8]!==0)slot=(slot+1)&(capacity-1);const at=8+slot*8;u32[at]=cell+1;u32[at+1]=1;f32[at+2]=4;f32[at+3]=4;f32[at+4]=4;u32[at+5]=9;u32[at+6]=cell;}
  return initializedBuffer(device,new Uint8Array(bytes),GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
}

test("global fine extraction has a bounded two-dimensional dispatch", () => {
  assert.deepEqual(globalFineSurfaceDispatch(8, 4 ** 3), [2, 1, 1]);
  assert.deepEqual(globalFineSurfaceDispatch(65_536, 8 ** 3), [65_535, 3, 1]);
  assert.throws(() => globalFineSurfaceDispatch(0, 64), /positive integers/);
  assert.match(surfaceExtractionShader, /sparseActivePages\[1u\]!=sparseParams\.brickDims\.w/,
    "a stale worklist generation must fail closed");
  assert.match(globalFineSurfaceClassificationShader, /sampleCoarseOctreePhi/,
    "missing fine samples must query compact coarse-octree phi");
  assert.doesNotMatch(globalFineSurfaceClassificationShader, /textureLoad|texture_3d/,
    "global fine classification must not reach dense phi texture authority");
  assert.match(globalFineSurfaceClassificationShader, /fineCubeFullyValid\(base,i32\(scale\)\)/,
    "coarse extraction may disappear only when fine validity covers all eight coarse corners");
  assert.doesNotMatch(globalFineSurfaceClassificationShader, /fineValid\(centre\)/,
    "a single valid centre sample must not suppress a partially covered coarse leaf");
  assert.match(globalFineClassifiedScanShader, /vertexAllocator\)==0xffffffffu\)\{return;/,
    "an unpublished A/B generation must retain the previous surface draw count");
  assert.match(globalFineSurfaceClassificationShader,
    /if\(lo>=0\.5\|\|hi<0\.5\)\{return;\}atomicStore\(&drawArgs\.globalFineAuthorityLatch,1u\);atomicMin\(&drawArgs\.vertexAllocator,0u\)/,
    "only an actual fine/coarse crossing may claim global renderer authority");
  assert.doesNotMatch(globalFineSurfaceClassificationShader, /atomicStore\(&drawArgs\.firstInstance/,
    "renderer-private authority must never make an indirect draw non-portable");
  assert.doesNotMatch(globalFineSurfaceClassificationShader,
    /if\(slot==0u\)\{atomicMin\(&drawArgs\.vertexAllocator,0u\);\}/,
    "a merely published compact directory must not replace a visible mesh with an empty one");
  assert.match(RasterWaterPipeline.prototype.encode.toString(),
    /writeBuffer\(this\.indirectBuffer,4,new Uint32Array\(\[1,0,0,0,(?:0xffff_ffff|4294967295),0\]\)\)/,
    "global extraction must preserve the mesh while resetting firstInstance and the private authority latch");
  assert.doesNotMatch(RasterWaterPipeline.prototype.setGlobalFineLevelSet.toString(), /geometryKey\s*=\s*["']{2}/,
    "a same-shaped unpublished B source must not destroy A's retained geometry allocation");
  const initialize = RasterWaterPipeline.prototype.initialize.toString();
  assert.match(initialize, /module:globalClassify,entryPoint:"extractGlobalFineMain"/,
    "production global extraction must compile the binding-minimal fine/coarse classifier");
  assert.match(initialize, /binding:16,visibility:GPUShaderStage\.COMPUTE,buffer:\{type:"read-only-storage"\}/,
    "production layout must bind compact coarse phi with its exact read-only ABI");
  assert.match(initialize, /binding:17,visibility:GPUShaderStage\.COMPUTE,buffer:\{type:"read-only-storage"\}/,
    "production layout must bind the transaction that published the selected fine slot");
  assert.match(initialize, /binding:8,visibility:GPUShaderStage\.COMPUTE,buffer:\{type:"read-only-storage"\}/,
    "production layout must bind the selected fine slot's publication worklist");
  assert.match(globalFineSurfaceClassificationShader,
    /fineTopologyControl\[0\]==0u&&fineTopologyControl\[4\]==1u&&fineTopologyControl\[5\]==0u&&fineTopologyControl\[7\]==0u/,
    "global publication requires a clean current-slot transaction");
  assert.match(globalFineSurfaceClassificationShader,
    /fineWorklist\[2\]==\(count\+63u\)\/64u&&fineWorklist\[3\]==1u&&fineWorklist\[4\]==1u/,
    "global publication requires the selected fine worklist to be complete and published");
  assert.match(globalFineSurfaceClassificationShader,
    /\(powerCoarseSamples\.generation&0x3fffffffu\)==generation/,
    "global publication requires exact same-generation compact coarse authority");
  assert.doesNotMatch(globalFineSurfaceClassificationShader, /rollback|coarseGeneration\+1u/,
    "a rejected rollback transaction must retain the prior mesh, not become a mixed render epoch");
  assert.equal((globalFineSurfaceClassificationShader.match(/@group\(0\)@binding\(8\)/g) ?? []).length, 1,
    "the publication worklist consumes the classifier's tenth and final storage binding");
  assert.doesNotMatch(globalFineSurfaceClassificationShader, /atomicLoad\(&powerCoarseSamples/,
    "the read-only compact coarse binding must never be consumed through an atomic pointer");
  assert.doesNotMatch(initialize, /globalFineField:\s*1/,
    "production must not revive the legacy dense-texture global override");
  const encode = RasterWaterPipeline.prototype.encode.toString();
  const globalStart = encode.indexOf("if(globalFine){");
  const adaptiveStart = encode.indexOf("else if(surfaceExtractionRepresentation", globalStart);
  assert.ok(globalStart >= 0 && adaptiveStart > globalStart,
    "the single global fine/coarse representation must take authority before leaf-page extraction");
  assert.doesNotMatch(encode.slice(globalStart, adaptiveStart), /mapAsync|copyBufferToBuffer|textureLoad|volume/,
    "global extraction must neither read back nor reach a dense/leaf-page compatibility field");
  const fallbackRetentionGuards = surfaceExtractionShader.match(
    /globalFineFallback&&\(atomicLoad\(&drawArgs\.globalFineAuthorityLatch\)!=0u\|\|atomicLoad\(&drawArgs\.vertexCount\)!=0u\)/g,
  ) ?? [];
  assert.equal(fallbackRetentionGuards.length, 2,
    "both resident-page and nonresident-leaf fallback classifiers must retain a published mesh");
  assert.ok((globalFineSurfaceClassificationShader.match(/var<storage/g) ?? []).length <= 10,
    "the global classifier must remain within the conservative Metal storage-binding limit");
  assert.match(extractionPrepareShader,
    /if \(drawArgs\.globalFineAuthorityLatch != 0u\) \{[\s\S]*DispatchArgs\(0u, 1u, 1u\)/,
    "global authority must also suppress the downstream adaptive polygonise dispatch");
  assert.match(surfaceExtractionShader,
    /atomicCompareExchangeWeak\(&drawArgs\.vertexAllocator,\s*SPARSE_INVALID,\s*0u\)/,
    "adaptive fallback must transactionally replace, rather than append to, retained geometry");
  assert.match(encode,
    /extractAdaptiveLeafFallbackPipeline[\s\S]*extractAdaptiveFallbackPipeline[\s\S]*polygoniseAdaptivePipeline/,
    "an empty global publication must fall through to the existing adaptive page renderer");
});

test("renderer diagnostics distinguish simulation publication from presentation fallback", () => {
  assert.equal(waterSurfaceGeometrySource(true, 600, 1, 600), "global-fine-coarse");
  assert.equal(waterSurfaceGeometrySource(true, 600, 0, 600), "adaptive-fallback");
  assert.equal(waterSurfaceGeometrySource(true, 600, 0, 0xffff_ffff), "retained-previous");
  assert.equal(waterSurfaceGeometrySource(true, 0, 0, 0xffff_ffff), "empty");
  assert.equal(waterSurfaceGeometrySource(false, 600, 0, 600), "adaptive-octree");
  assert.equal(waterSurfaceGeometrySource(false, 0, 0, 0), "empty");
});

test("adaptive presentation fallback seeds only a newly attached empty renderer", () => {
  assert.equal(globalFineFallbackMaySeedRenderer(0, 0), true);
  assert.equal(globalFineFallbackMaySeedRenderer(918, 0), false,
    "an unpublished B generation must retain A instead of replacing it with adaptive geometry");
  assert.equal(globalFineFallbackMaySeedRenderer(0, 1), false,
    "a current global crossing must retain global ownership even before polygonisation publishes vertices");
});

test("Dawn polygonises tagged global factor-4/factor-8 bricks and retains A when B is unpublished", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU global-fine water rendering checks",
}, async () => {
  const dawn = await import(pathToFileURL(modulePath!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const adapter = await dawn.create(["backend=metal"]).requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 10 } });
  device.pushErrorScope("validation");
  const errors: string[] = [];
  device.addEventListener("uncapturederror", (event: unknown) => {
    errors.push((event as { error: { message: string } }).error.message);
  });

  const extractLayout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ...[7, 8, 9, 11, 12].map((binding) => ({
      binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const },
    })),
    { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 16, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 17, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
  ] });
  const prepareLayout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
  ] });
  const polygonLayout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
  ] });
  const countLayout = device.createBindGroupLayout({ entries: [
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
  ] });
  const extractModule = device.createShaderModule({ code: globalFineSurfaceClassificationShader });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [extractLayout] });
  const classify = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: extractModule, entryPoint: "extractGlobalFineMain" },
  });
  const polygonPipelineLayout=device.createPipelineLayout({bindGroupLayouts:[polygonLayout]});
  const count=device.createComputePipeline({layout:device.createPipelineLayout({bindGroupLayouts:[countLayout]}),compute:{module:device.createShaderModule({code:globalFineClassifiedCountShader}),entryPoint:"countGlobalFineTriangles"}});
  const scan=device.createComputePipeline({layout:polygonPipelineLayout,compute:{module:device.createShaderModule({code:globalFineClassifiedScanShader}),entryPoint:"scanGlobalFineTriangles"}});
  const emit=globalFineClassifiedEmitShaders.map((code,index)=>device.createComputePipeline({layout:polygonPipelineLayout,compute:{module:device.createShaderModule({code}),entryPoint:`emitGlobalFineTetra${index}`}}));
  const emitAll=device.createComputePipeline({layout:polygonPipelineLayout,compute:{module:device.createShaderModule({code:globalFineClassifiedEmitShader}),entryPoint:"emitGlobalFineTetrahedra"}});
  const prepare = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [prepareLayout] }),
    compute: { module: device.createShaderModule({ code: extractionPrepareShader }), entryPoint: "prepareMain" },
  });

  for (const factor of [4, 8] as const satisfies readonly FineLevelSetFactor[]) {
    const plan = planFineLevelSetBricks({
      domainOrigin: [0, 0, 0], finestCellDimensions: [2, 2, 2], finestCellWidth: 1,
      fineFactor: factor, brickResolution: factor, maximumResidentBricks: 8,
    });
    const oracle = new FineLevelSetBrickOracle(plan);
    oracle.publishInterfaceAndRing([packFineLevelSetBrickKey(plan, [0, 0, 0])], ([, y]) => (y - 0.75) * plan.fineCellWidth);
    const owner = new WebGPUFineLevelSetBricks(device, plan);
    const brickSource = owner.uploadGeneration(oracle.exportGPUGeneration());
    const source = createGlobalFineLevelSetConsumerSource(brickSource);
    const coarseDirectory=compactCoarsePlane(device);
    const topologyControl=initializedBuffer(device,new Uint32Array([0,1,1,1,1,0,1,0]),GPUBufferUsage.STORAGE);

    const uniformData = new Float32Array(28);
    uniformData.set([96, 96, 0, 0], 0);
    uniformData.set([0, 1, 4, 0], 4);
    uniformData.set([0, 0.75, 0, 0], 8);
    uniformData.set([2, 2, 2, 0], 12);
    uniformData.set([2, 2, 2, 3], 20);
    const uniform = initializedBuffer(device, uniformData, GPUBufferUsage.UNIFORM);
    const renderParamBytes = new ArrayBuffer(112);
    const paramU32 = new Uint32Array(renderParamBytes), paramF32 = new Float32Array(renderParamBytes);
    paramU32.set([...source.sampleDimensions, source.brickResolution], 0);
    paramU32.set([...source.brickDimensions, source.samplesPerBrick], 4);
    paramU32.set([source.hashCapacity, source.maximumHashProbes, source.pageCapacity, source.generation], 8);
    paramF32.set([...source.domainOrigin, source.fineCellWidth], 12); paramF32[16] = source.fineFactor;
    const renderParams = initializedBuffer(device, new Uint8Array(renderParamBytes), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const volume = device.createTexture({
      size: { width: 2, height: 2, depthOrArrayLayers: 2 }, dimension: "3d", format: "r32float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    const coarsePhi = new Float32Array(64 * 4);
    for (let z = 0; z < 2; z += 1) for (let y = 0; y < 2; y += 1) {
      coarsePhi[z * 128 + y * 64] = y === 0 ? -0.25 : 0.75;
      coarsePhi[z * 128 + y * 64 + 1] = y === 0 ? -0.25 : 0.75;
    }
    device.queue.writeTexture({ texture: volume }, coarsePhi,
      { bytesPerRow: 256, rowsPerImage: 2 }, [2, 2, 2]);
    const vertices = device.createBuffer({ size: 2 * 1024 * 1024, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const classifyVertexDummy = device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE });
    const drawArgs = initializedBuffer(device, new Uint32Array([0, 1, 0, 0, 0, 0, 0]),
      GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    const cubes = device.createBuffer({ size: 256 * 1024, usage: GPUBufferUsage.STORAGE });
    const cubeValues = device.createBuffer({ size: 1024 * 1024, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const cubeOffsets = device.createBuffer({ size: 6 * 128 * 1024, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const dispatch = device.createBuffer({ size: 12, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT });
    const extractGroup = device.createBindGroup({ layout: extractLayout, entries: [
      { binding: 0, resource: { buffer: uniform } },
      { binding: 3, resource: { buffer: classifyVertexDummy } }, { binding: 4, resource: { buffer: drawArgs } },
      { binding: 5, resource: { buffer: cubes } }, { binding: 7, resource: source.hash },
      { binding: 6, resource: { buffer: cubeValues } },
      { binding: 8, resource: source.worklist },
      { binding: 9, resource: source.phi },
      { binding: 10, resource: { buffer: renderParams } }, { binding: 11, resource: source.flags },
      { binding: 12, resource: source.metadata },
      { binding: 16, resource: { buffer: coarseDirectory } },
      { binding: 17, resource: { buffer: topologyControl } },
    ] });
    const polygonGroup = device.createBindGroup({ layout: polygonLayout, entries: [
      { binding: 0, resource: { buffer: uniform } }, { binding: 3, resource: { buffer: vertices } },
      { binding: 4, resource: { buffer: drawArgs } }, { binding: 5, resource: { buffer: cubes } },
      { binding: 6, resource: { buffer: cubeValues } }, { binding: 7, resource: { buffer: cubeOffsets } },
      { binding: 10, resource: { buffer: renderParams } },
    ] });
    const countGroup = device.createBindGroup({ layout: countLayout, entries: [
      { binding: 4, resource: { buffer: drawArgs } }, { binding: 5, resource: { buffer: cubes } },
      { binding: 6, resource: { buffer: cubeValues } }, { binding: 7, resource: { buffer: cubeOffsets } },
    ] });
    const prepareGroup = device.createBindGroup({ layout: prepareLayout, entries: [
      { binding: 0, resource: { buffer: drawArgs } }, { binding: 1, resource: { buffer: cubes } },
      { binding: 2, resource: { buffer: dispatch } },
    ] });
    const readback = device.createBuffer({ size: 120, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    const classifyPass = encoder.beginComputePass();
    classifyPass.setPipeline(classify); classifyPass.setBindGroup(0, extractGroup);
    classifyPass.dispatchWorkgroups(...globalFineSurfaceDispatch(source.pageCapacity, source.samplesPerBrick));
    classifyPass.setBindGroup(0,polygonGroup);classifyPass.setPipeline(scan);classifyPass.dispatchWorkgroups(1);
    classifyPass.setPipeline(emitAll);classifyPass.dispatchWorkgroups(512,6);classifyPass.end();void emit;void prepare;void prepareGroup;
    void countGroup;void count;
    encoder.copyBufferToBuffer(drawArgs, 0, readback, 0, 24);
    encoder.copyBufferToBuffer(vertices, 0, readback, 24, 96);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ);
    const bytes = readback.getMappedRange().slice(0); readback.unmap();
    const counters = new Uint32Array(bytes, 0, 6);
    const firstTriangle = new Float32Array(bytes, 24, 24);
    assert.ok(counters[4] > 0, `factor ${factor} must classify at least one fine surface cube`);
    assert.ok(counters[0] >= 3, `factor ${factor} must polygonise visible geometry`);
    assert.ok([...firstTriangle].every(Number.isFinite), `factor ${factor} geometry must remain finite`);
    assert.ok(firstTriangle[3] === 1 && firstTriangle[11] === 1 && firstTriangle[19] === 1,
      `factor ${factor} must emit three live positions`);
    // Production A/B cutover: make the fine generation stale and invalidate
    // the compact coarse directory.  The classifier must leave generation A's
    // finite mesh draw count intact instead of publishing an empty B frame.
    paramU32[11] = source.generation + 1;
    device.queue.writeBuffer(renderParams, 0, new Uint8Array(renderParamBytes));
    device.queue.writeBuffer(coarseDirectory, 0, new Uint32Array([0]));
    device.queue.writeBuffer(drawArgs, 4, new Uint32Array([1, 0, 0, 0, 0xffff_ffff, 0]));
    const invalidEncoder = device.createCommandEncoder();
    const invalidPass = invalidEncoder.beginComputePass();
    invalidPass.setPipeline(classify); invalidPass.setBindGroup(0, extractGroup);
    invalidPass.dispatchWorkgroups(...globalFineSurfaceDispatch(source.pageCapacity, source.samplesPerBrick));
    invalidPass.setBindGroup(0, polygonGroup); invalidPass.setPipeline(scan); invalidPass.dispatchWorkgroups(1);
    invalidPass.setPipeline(emitAll); invalidPass.dispatchWorkgroups(512, 6); invalidPass.end();
    const retainedReadback = device.createBuffer({ size: 24, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    invalidEncoder.copyBufferToBuffer(drawArgs, 0, retainedReadback, 0, 24);
    device.queue.submit([invalidEncoder.finish()]); await device.queue.onSubmittedWorkDone();
    await retainedReadback.mapAsync(GPUMapMode.READ);
    const retained = new Uint32Array(retainedReadback.getMappedRange().slice(0)); retainedReadback.unmap();
    assert.equal(retained[0], counters[0], `factor ${factor} invalid B must retain generation A geometry`);
    assert.equal(retained[4], 0, `factor ${factor} invalid B must classify no stale cubes`);
    assert.equal(retained[5], 0xffff_ffff, `factor ${factor} invalid B must remain unpublished`);
    retainedReadback.destroy();
    for (const buffer of [uniform, renderParams, vertices, classifyVertexDummy, drawArgs, cubes,
      cubeValues, cubeOffsets, dispatch, readback, coarseDirectory, topologyControl]) buffer.destroy();
    volume.destroy(); owner.destroy();
  }
  const scopedError=await device.popErrorScope();
  assert.equal(scopedError,null,"global fine extraction must not produce a scoped WebGPU validation error");
  assert.deepEqual(errors, [], "global fine extraction must pass WebGPU validation for both factors");
  device.destroy();
});
