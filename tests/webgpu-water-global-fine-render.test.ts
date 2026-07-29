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
  globalFineCoarseSurfaceDispatch,
  globalFineSurfaceDispatch,
  RasterWaterPipeline,
  surfaceExtractionShader,
  WATER_INTERFACE_CULL_MODES,
  waterSurfaceGeometrySource,
} from "../lib/webgpu-water-pipeline";
import { globalFineClassifiedEmitShader, globalFineClassifiedEmitShaders, globalFineClassifiedScanShader } from "../lib/webgpu-water-global-fine-tetra";
import {
  GLOBAL_FINE_SHARP_CORNER_HALF_CELL_EPSILON,
  globalFineSurfaceClassificationShader,
} from "../lib/webgpu-water-global-fine-classify";

const modulePath = process.env.WEBGPU_NODE_MODULE;

test("hard-clipped sharp corner caps admit only four half-cell zero crossings", () => {
  assert.ok(GLOBAL_FINE_SHARP_CORNER_HALF_CELL_EPSILON <= 1e-3);
  const eligible = (inside: number, outside: number) =>
    inside < 0 && outside > 0
    && Math.abs(-inside / (outside - inside) - 0.5)
      <= GLOBAL_FINE_SHARP_CORNER_HALF_CELL_EPSILON;
  assert.equal(eligible(-0.5, 0.5), true, "the exact t=0 box corner keeps its two caps");
  assert.equal(Math.abs(-0.55 + 0.45) <= 0.15, true,
    "the retired signed-value tolerance admitted a visibly off-center crossing");
  assert.equal(eligible(-0.55, 0.45), false,
    "an evolved 0.55-cell crossing must use the ordinary conforming tetra mesh");
  const compact = globalFineSurfaceClassificationShader.replace(/\s+/g, "");
  const helper = compact.slice(compact.indexOf("fnhalfCellCrossing("),
    compact.indexOf("//Asigneddistancetoanaxis-alignedliquidbox"));
  assert.match(helper,
    /inside<0\.0&&outside>0\.0&&denominator>0\.0&&abs\(\(-inside\/denominator\)-0\.5\)<=SHARP_CORNER_HALF_CELL_EPSILON/);
  const classifier = compact.slice(compact.indexOf("fnclassifySharpInteriorXZCorner("),
    compact.indexOf("fnclassifyScaled(", compact.indexOf("fnclassifySharpInteriorXZCorner(")));
  assert.equal(classifier.match(/halfCellCrossing\(/g)?.length, 4,
    "bottom/top x/z crossings must all match the polygonizer's hardcoded half-cube clips");
  assert.doesNotMatch(classifier, /letsymmetric=/,
    "a broad signed-value tolerance must not admit visibly off-center moving corners");
});

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

test("global fine side-wall caps meet at the exact x/z tank edge", () => {
  const cap = (axis: 0 | 2, normal: V3) => tetrahedra.flatMap(ids => {
    const points = ids.map(index => cubeCorners[index]);
    const values = ids.map(index => cubeCorners[index][axis]);
    return tetraSurface(points, values, normal);
  });
  const xCap = cap(0, [-1, 0, 0]);
  const zCap = cap(2, [0, 0, -1]);
  const tolerance = 1e-12;
  assert.ok(xCap.length > 0 && zCap.length > 0);
  assert.ok(xCap.flat().every(point => Math.abs(point[0] - 0.5) <= tolerance),
    "the x-wall owner must remain on the x-wall plane instead of forming a diagonal chamfer");
  assert.ok(zCap.flat().every(point => Math.abs(point[2] - 0.5) <= tolerance),
    "the z-wall owner must remain on the z-wall plane instead of forming a diagonal chamfer");
  assert.ok(xCap.flat().some(point => Math.abs(point[0] - 0.5) <= tolerance && Math.abs(point[2] - 0.5) <= tolerance)
    && zCap.flat().some(point => Math.abs(point[0] - 0.5) <= tolerance && Math.abs(point[2] - 0.5) <= tolerance),
  "the two independently owned caps must share the exact tank-corner edge");
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
  assert.deepEqual(globalFineCoarseSurfaceDispatch(8_192), [8_192, 1, 1]);
  assert.deepEqual(globalFineCoarseSurfaceDispatch(65_536), [65_535, 2, 1]);
  assert.match(surfaceExtractionShader, /sparseActivePages\[1u\]!=sparseParams\.brickDims\.w/,
    "a stale worklist generation must fail closed");
  assert.match(globalFineSurfaceClassificationShader, /sampleCoarseOctreePhi/,
    "missing fine samples must query compact coarse-octree phi");
  assert.doesNotMatch(globalFineSurfaceClassificationShader, /textureLoad|texture_3d/,
    "global fine classification must not reach dense phi texture authority");
  assert.match(globalFineSurfaceClassificationShader, /fineOwnsCube\(candidate\)/,
    "each unit cube must have exactly one fine-or-coarse lower-anchor owner");
  assert.doesNotMatch(globalFineSurfaceClassificationShader, /classifyScaled\([^;]*i32\(scale\)\)/,
    "coarse fallback must not place scaled tetrahedra beside unit fine tetrahedra");
  assert.doesNotMatch(globalFineSurfaceClassificationShader, /fineValid\(centre\)/,
    "a single valid centre sample must not suppress a partially covered coarse leaf");
  assert.match(globalFineSurfaceClassificationShader,
    /let lowX=select\(0u,1u,origin\.x==0u\);let lowY=select\(0u,1u,origin\.y==0u\);let lowZ=select\(0u,1u,origin\.z==0u\)/,
    "coarse fallback must enumerate low-wall and floor lattice bases");
  assert.match(globalFineSurfaceClassificationShader,
    /let sx=scale\+lowX;let sy=scale\+lowY;let sz=scale\+lowZ;let total=sx\*sy\*sz/,
    "coarse fallback must use a unit-lattice Cartesian product at edges and corners");
  assert.match(globalFineSurfaceClassificationShader,
    /if\(xWall&&zWall\)\{classifyScaledForWall\(base,scale,1u\);classifyScaledForWall\(base,scale,2u\);return;\}/,
    "an x/z edge cube must publish two wall-owned caps rather than one diagonal scalar-field chamfer");
  assert.match(globalFineSurfaceClassificationShader,
    /fn classifySharpInteriorXZCorner[\s\S]*extruded&&sharp&&halfClipped[\s\S]*emitClassifiedCubeTagged\(base,scale,xlo,xhi[\s\S]*emitClassifiedCubeTagged\(base,scale,zlo,zhi/,
    "an exact half-cell axis-aligned liquid corner must retain its two independently owned faces");
  assert.match(globalFineClassifiedEmitShader,
    /fn clipped[\s\S]*mode==1u[\s\S]*r\.z=base\.z\+scale\*select\(\.5\*q\.z,\.5\+\.5\*q\.z,high\)[\s\S]*mode==2u[\s\S]*r\.x=base\.x\+scale\*select\(\.5\*q\.x,\.5\+\.5\*q\.x,high\)/,
    "separate corner owners must terminate on their shared half-cell edge rather than occlude each other");
  assert.match(globalFineSurfaceClassificationShader,
    /if\(p\.x<=0\|\|p\.x>=dims\.x\+1\)\{[\s\S]*wallMode!=2u[\s\S]*if\(p\.z<=0\|\|p\.z>=dims\.z\+1\)\{[\s\S]*wallMode!=1u/,
    "each edge cap must mirror only through its tangential wall and retain air across its owned normal wall");
  assert.match(globalFineSurfaceClassificationShader,
    /@builtin\(workgroup_id\)group:vec3u,@builtin\(local_invocation_index\)local:u32[\s\S]*for\(var index=local;index<total;index\+=256u\)/,
    "one workgroup must cooperatively subdivide a coarse leaf instead of risking a scalar scale-cubed loop");
  assert.match(globalFineClassifiedScanShader, /let published=atomicLoad\(&args\.vertexAllocator\)!=0xffffffffu/,
    "an unpublished A/B generation must retain the previous surface draw count");
  assert.match(globalFineClassifiedScanShader,
    /if\(published\)\{[\s\S]*atomicStore\(&args\.vertexCount[\s\S]*atomicStore\(&args\.meshPublicationGeneration,p\.table\.w\)/,
    "the GPU scan that publishes the mesh must commit its exact fine-source generation in the same branch");
  assert.match(globalFineClassifiedScanShader,
    /var<workgroup>laneOffsets:array<u32,256>[\s\S]*@workgroup_size\(256\)[\s\S]*let begin=lid\*base\+min\(lid,extra\)[\s\S]*workgroupBarrier\(\)[\s\S]*offsets\[i\*6u\+/,
    "global-fine mesh allocation must use a deterministic cooperative prefix scan, not one invocation over every cube");
  assert.doesNotMatch(globalFineClassifiedScanShader, /@workgroup_size\(1\)/);
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
    "global extraction must preserve the mesh generation while resetting firstInstance and the private authority latch");
  const completeDiagnostics = RasterWaterPipeline.prototype.completeSurfaceDiagnostics.toString();
  assert.match(completeDiagnostics,
    /meshPublicationGeneration=\(surfaceGeometrySource==="global-fine-coarse"\|\|surfaceGeometrySource==="retained-previous"\)&&words\[7\]!==(?:0xffffffff|4294967295)\?words\[7\]:(?:undefined|void 0)/,
    "presentation diagnostics must report the generation committed by the GPU mesh publication");
  assert.doesNotMatch(completeDiagnostics, /lastRasterMeshPublicationGeneration/,
    "presentation diagnostics must not infer retained mesh authority from host attachment history");
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
    /let clean=topologyFlags==0u&&fineTopologyControl\[4\]==1u&&fineTopologyControl\[5\]==0u&&topologyReason==0u/,
    "global publication recognizes a clean current-slot transaction");
  assert.match(globalFineSurfaceClassificationShader,
    /fineWorklist\[2\]==params\.table\.z&&\(fineWorklist\[3\]&3u\)==3u[\s\S]*fineWorklist\[4\]==\(count\+63u\)\/64u&&fineWorklist\[5\]==1u&&fineWorklist\[6\]==1u/,
    "global publication requires the selected fine worklist to be complete and published");
  assert.match(globalFineSurfaceClassificationShader,
    /\(clean&&coarseGeneration==generation\)\|\|retained/,
    "a clean publication requires equal generations while an explicit rollback retains the independent octree");
  assert.match(globalFineSurfaceClassificationShader,
    /let retained=fineTopologyControl\[4\]==1u&&fineTopologyControl\[5\]==1u[\s\S]*topologyFlags&~0x1fu[\s\S]*topologyReason&~0x0fu/,
    "the Section-5 two-mesh fallback accepts only a known, explicit rollback transaction");
  assert.equal((globalFineSurfaceClassificationShader.match(/@group\(0\)@binding\(8\)/g) ?? []).length, 1,
    "the publication worklist consumes the classifier's tenth and final storage binding");
  assert.doesNotMatch(globalFineSurfaceClassificationShader, /atomicLoad\(&powerCoarseSamples/,
    "the read-only compact coarse binding must never be consumed through an atomic pointer");
  assert.doesNotMatch(initialize, /globalFineField:\s*1/,
    "production must not revive the legacy dense-texture global override");
  const encode = RasterWaterPipeline.prototype.encode.toString();
  const globalStart = encode.indexOf("if(globalFine){");
  assert.ok(globalStart >= 0, "the single global fine/coarse representation must be encoded");
  assert.doesNotMatch(`${encode}\n${surfaceExtractionShader}`,
    /extractAdaptive|polygoniseAdaptive|adaptiveField|globalFineFallback|adaptiveArena|adaptiveLeaves/,
    "the deleted row-attached page renderer must not remain as a fallback");
  assert.ok((globalFineSurfaceClassificationShader.match(/var<storage/g) ?? []).length <= 10,
    "the global classifier must remain within the conservative Metal storage-binding limit");
});

test("renderer diagnostics distinguish current global publication from retained geometry", () => {
  assert.equal(waterSurfaceGeometrySource(true, 600, 1), "global-fine-coarse");
  assert.equal(waterSurfaceGeometrySource(true, 600, 0), "retained-previous");
  assert.equal(waterSurfaceGeometrySource(true, 0, 0), "empty");
  assert.equal(waterSurfaceGeometrySource(false, 600, 0), "volume");
  assert.equal(waterSurfaceGeometrySource(false, 0, 0), "empty");
});

test("Dawn cooperatively scans global-fine cubes in stable cube/tetra order", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU global-fine scan checks",
}, async () => {
  const dawn = await import(pathToFileURL(modulePath!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const adapter = await dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]).requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice();
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const args = initializedBuffer(device, new Uint32Array([99, 1, 0, 0, 2, 0, 0, 0]), storage);
  const cubes = initializedBuffer(device, new Uint32Array(4), storage);
  const values = initializedBuffer(device, new Float32Array([
    1, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1,
  ]), storage);
  const offsets = initializedBuffer(device, new Uint32Array(12), storage);
  const output = device.createBuffer({ size: 4096, usage: storage });
  const parameterWords = new Uint32Array(28);
  parameterWords[11] = 73;
  const parameters = initializedBuffer(device, parameterWords, GPUBufferUsage.UNIFORM);
  const module = device.createShaderModule({ code: globalFineClassifiedScanShader });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "scanGlobalFineTriangles" },
  });
  const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
    { binding: 3, resource: { buffer: output } },
    { binding: 4, resource: { buffer: args } },
    { binding: 5, resource: { buffer: cubes } },
    { binding: 6, resource: { buffer: values } },
    { binding: 7, resource: { buffer: offsets } },
    { binding: 10, resource: { buffer: parameters } },
  ] });
  const readback = device.createBuffer({
    size: 80,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(1);
  pass.end();
  encoder.copyBufferToBuffer(args, 0, readback, 0, 32);
  encoder.copyBufferToBuffer(offsets, 0, readback, 32, 48);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await readback.mapAsync(GPUMapMode.READ);
  const words = new Uint32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  assert.equal(words[0], 36);
  assert.equal(words[5], 36);
  assert.equal(words[7], 73, "scan publication must commit the exact GPU-visible source generation");
  assert.deepEqual([...words.slice(8)], [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33]);
  for (const buffer of [args, cubes, values, offsets, output, parameters, readback]) buffer.destroy();
  device.destroy();
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
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ...[8, 9, 11, 12].map((binding) => ({
      binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const },
    })),
    { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 16, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 17, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
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
  const extractModule = device.createShaderModule({ code: globalFineSurfaceClassificationShader });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [extractLayout] });
  const classify = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: extractModule, entryPoint: "extractGlobalFineMain" },
  });
  const classifyCoarse = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: extractModule, entryPoint: "extractGlobalCoarseMain" },
  });
  const polygonPipelineLayout=device.createPipelineLayout({bindGroupLayouts:[polygonLayout]});
  const scan=device.createComputePipeline({layout:polygonPipelineLayout,compute:{module:device.createShaderModule({code:globalFineClassifiedScanShader}),entryPoint:"scanGlobalFineTriangles"}});
  const emitAll=device.createComputePipeline({layout:polygonPipelineLayout,compute:{module:device.createShaderModule({code:globalFineClassifiedEmitShader}),entryPoint:"emitGlobalFineTetrahedra"}});

  for (const factor of [4, 8] as const satisfies readonly FineLevelSetFactor[]) {
    const plan = planFineLevelSetBricks({
      domainOrigin: [0, 0, 0], finestCellDimensions: [2, 2, 2], finestCellWidth: 1,
      fineFactor: factor, brickResolution: 4, maximumResidentBricks: 8,
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
    paramU32.set([source.pageCapacity, 7, source.pageCapacity, source.generation], 8);
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
    const drawArgs = initializedBuffer(device, new Uint32Array([0, 1, 0, 0, 0, 0, 0, 0xffff_ffff]),
      GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    const cubes = device.createBuffer({ size: 256 * 1024,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const cubeValues = device.createBuffer({ size: 1024 * 1024, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const cubeOffsets = device.createBuffer({ size: 6 * 128 * 1024, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const extractGroup = device.createBindGroup({ layout: extractLayout, entries: [
      { binding: 0, resource: { buffer: uniform } },
      { binding: 4, resource: { buffer: drawArgs } },
      { binding: 5, resource: { buffer: cubes } },
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
    const cubeReadbackOffset = 128;
    const readback = device.createBuffer({ size: cubeReadbackOffset + cubes.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    device.pushErrorScope("validation");
    const encoder = device.createCommandEncoder();
    const classifyPass = encoder.beginComputePass();
    classifyPass.setPipeline(classify); classifyPass.setBindGroup(0, extractGroup);
    classifyPass.dispatchWorkgroups(...globalFineSurfaceDispatch(source.pageCapacity, source.samplesPerBrick));
    classifyPass.setPipeline(classifyCoarse);
    classifyPass.dispatchWorkgroups(...globalFineCoarseSurfaceDispatch(8));
    classifyPass.end();
    const meshPass = encoder.beginComputePass();
    meshPass.setBindGroup(0,polygonGroup);meshPass.setPipeline(scan);meshPass.dispatchWorkgroups(1);
    meshPass.setPipeline(emitAll);meshPass.dispatchWorkgroups(512,6);meshPass.end();
    encoder.copyBufferToBuffer(drawArgs, 0, readback, 0, 32);
    encoder.copyBufferToBuffer(vertices, 0, readback, 32, 96);
    encoder.copyBufferToBuffer(cubes, 0, readback, cubeReadbackOffset, cubes.size);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const validationError = await device.popErrorScope();
    assert.equal(validationError, null, validationError?.message);
    await readback.mapAsync(GPUMapMode.READ);
    const bytes = readback.getMappedRange().slice(0); readback.unmap();
    const counters = new Uint32Array(bytes, 0, 8);
    const firstTriangle = new Float32Array(bytes, 32, 24);
    assert.ok(counters[4] > 0, `factor ${factor} must classify at least one fine surface cube`);
    assert.ok(counters[0] >= 3, `factor ${factor} must polygonise visible geometry`);
    assert.equal(counters[7], source.generation,
      `factor ${factor} scan must publish the exact GPU-visible fine generation`);
    assert.ok([...firstTriangle].every(Number.isFinite), `factor ${factor} geometry must remain finite`);
    assert.ok(firstTriangle[3] === 1 && firstTriangle[11] === 1 && firstTriangle[19] === 1,
      `factor ${factor} must emit three live positions`);
    assert.ok(counters[4] * 8 <= cubes.size, `factor ${factor} classified cube worklist must not clip`);
    const classified = new Uint32Array(bytes, cubeReadbackOffset, counters[4] * 2);
    const baseOwnerCounts = new Map<string, number>();
    for (let cube = 0; cube < counters[4]; cube += 1) {
      const packedXZ = classified[cube * 2], packedYScale = classified[cube * 2 + 1];
      const descriptor = packedYScale >>> 16;
      assert.equal(descriptor & 0xff, 1, `factor ${factor} coarse/fine cube ${cube} must use unit scale`);
      const x = packedXZ & 0xffff, y = packedYScale & 0xffff, z = packedXZ >>> 16;
      const key = `${x},${y},${z}`;
      const ownerCount = (baseOwnerCounts.get(key) ?? 0) + 1;
      const dualWallCorner = (x === 0 || x === source.sampleDimensions[0])
        && (z === 0 || z === source.sampleDimensions[2]);
      const sharpCornerOwner = ((descriptor >> 8) & 3) !== 0;
      assert.ok(ownerCount === 1 || (ownerCount === 2 && (dualWallCorner || sharpCornerOwner)),
        `factor ${factor} cube base ${key} has ${ownerCount} owners outside an explicit x/z corner`);
      baseOwnerCounts.set(key, ownerCount);
    }
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
    const retainedReadback = device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    invalidEncoder.copyBufferToBuffer(drawArgs, 0, retainedReadback, 0, 32);
    device.queue.submit([invalidEncoder.finish()]); await device.queue.onSubmittedWorkDone();
    await retainedReadback.mapAsync(GPUMapMode.READ);
    const retained = new Uint32Array(retainedReadback.getMappedRange().slice(0)); retainedReadback.unmap();
    assert.equal(retained[0], counters[0], `factor ${factor} invalid B must retain generation A geometry`);
    assert.equal(retained[4], 0, `factor ${factor} invalid B must classify no stale cubes`);
    assert.equal(retained[5], 0xffff_ffff, `factor ${factor} invalid B must remain unpublished`);
    assert.equal(retained[7], source.generation,
      `factor ${factor} invalid B must retain generation A's GPU publication word`);
    retainedReadback.destroy();
    for (const buffer of [uniform, renderParams, vertices, drawArgs, cubes,
      cubeValues, cubeOffsets, readback, coarseDirectory, topologyControl]) buffer.destroy();
    volume.destroy(); owner.destroy();
  }
  const scopedError=await device.popErrorScope();
  assert.equal(scopedError,null,"global fine extraction must not produce a scoped WebGPU validation error");
  assert.deepEqual(errors, [], "global fine extraction must pass WebGPU validation for both factors");
  device.destroy();
});
