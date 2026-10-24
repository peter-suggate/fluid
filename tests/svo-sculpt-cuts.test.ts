import { referenceDmcCentreNormals } from "./helpers/reference-dmc-centre-normals";
import assert from "node:assert/strict";
import test from "node:test";
import { packSvoSculptCuts, sampleSvoSculptCuts, intersectsSvoSculptCuts,
  intersectsSvoSculptPrimitiveBounds, SVO_SCULPT_CUT_MAXIMUM_RECORDS, svoSculptCutsWGSL,
  SVO_SCULPT_CUT_ARENA_WORDS, SVO_SCULPT_OWNER_MASK_OFFSET_WORDS } from "../lib/svo/features/scene-publication/svo-sculpt-cuts";
import { SparseSceneProxyVoxelizer, sparseSceneProxyVoxelizationShaderFor } from "../lib/core/webgpu-sparse-scene-proxies";
import { PublishedWords } from "../lib/svo/features/scene-publication/svo-table-upload";
import { SCENE_ENVIRONMENT_OWNER_BASE } from "../lib/core/webgpu-rigid-body";
import type { SolidWorldVoxelPatch } from "../lib/core/solid-world";
import { subcellEllipsoidPatch } from "../lib/core/solid-patch/ellipsoid-bounds";
import { withUniformDevice } from "./helpers/uniform-geometric";

const lattice = { origin_m: [-2, 0, -1] as const, cellSize_m: [.25, .5, .125] as const };
const cuts: SolidWorldVoxelPatch[] = [
  { operation: "clear", affectsScenery: true, minimum: [0, 0, 0], maximumExclusive: [8, 4, 8] },
  { operation: "clear", affectsScenery: true, kind: "ellipsoid", minimum: [8, 0, 0], maximumExclusive: [16, 4, 8] },
];

test("scenery cuts are explicit, bounded and expressed in the physical voxel lattice", () => {
  const words = packSvoSculptCuts(cuts, lattice);
  assert.equal(words[0], 2);
  assert.equal(sampleSvoSculptCuts(words, [-1, 1, -.5]).distance, -.5);
  assert.equal(sampleSvoSculptCuts(words, [1, 1, -.5]).distance, -.5);
  assert.equal(sampleSvoSculptCuts(words, [2, 1, -.5]).distance, 0);
  assert.deepEqual(sampleSvoSculptCuts(words, [2, 1, -.5]).normal, [1, 0, 0]);
  assert.equal(intersectsSvoSculptCuts(words, [-.1, .9, -.6], [.1, 1.1, -.4]), true);
  assert.equal(intersectsSvoSculptPrimitiveBounds(words, [10, 1, 0], [.1, .1, .1]), false);
  assert.equal(packSvoSculptCuts([{ ...cuts[0]!, affectsScenery: undefined }], lattice)[0], 0);
  assert.equal(packSvoSculptCuts([{ ...cuts[0]!, operation: "fill" }], lattice)[0], 0);
  assert.throws(() => packSvoSculptCuts(Array(SVO_SCULPT_CUT_MAXIMUM_RECORDS + 1).fill(cuts[0]), lattice), /capacity/);
  assert.throws(() => packSvoSculptCuts([{ ...cuts[0]!, kind: "cylinder" }], lattice), /does not support/);
  assert.throws(() => packSvoSculptCuts([{ ...cuts[0]!, maximumExclusive: [0, 0, 0] }], lattice), /nonempty/);
});

test("voxelizer cut publication retains its arena and restores affected native owners on undo", () => {
  const offset = 12, image = new Uint32Array(offset + SVO_SCULPT_CUT_ARENA_WORDS);
  const owner = SCENE_ENVIRONMENT_OWNER_BASE + 2;
  const queue = { writeBuffer(_buffer: unknown, at: number, data: Uint32Array, from = 0, count = data.length) {
    image.set(data.subarray(from, from + count), at / 4);
  } };
  const voxelizer = Object.assign(Object.create(SparseSceneProxyVoxelizer.prototype), {
    options: { solidWorldLattice: lattice }, device: { queue }, maintenanceArena: {}, sculptCutOffsetWords: offset,
    publishedSculptCuts: new PublishedWords(), publishedSculptOwners: new PublishedWords(),
    publishedPrimitives: [{ kind: "field-program", center: [-1, 1, -.5], envelopeRadii: [.2, .2, .2],
      ownerId: owner, program: { ops: [], result: 0, sampledVolume: { asset: "honed-travertine-paver" } } }],
  });
  voxelizer.setSculptCuts(cuts);
  assert.equal(image[offset], 2);
  const mask = offset + SVO_SCULPT_OWNER_MASK_OFFSET_WORDS + (owner >>> 5);
  assert.equal((image[mask]! >>> (owner & 31)) & 1, 1);
  const published = image.slice();
  assert.throws(() => voxelizer.setSculptCuts(Array(SVO_SCULPT_CUT_MAXIMUM_RECORDS + 1).fill(cuts[0])), /capacity/);
  assert.deepEqual(image, published, "refused edits must not change the arena");
  voxelizer.setSculptCuts([]);
  assert.equal(image[offset], 0);
  assert.equal(image[mask], 0, "undo restores native visibility and unmirrored materials together");
  assert.ok(image.every(word => word === 0), "retired records are cleared");
});

test("native sculpt union distances and cut normals match CPU for box/ellipsoid CSG", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice("scenery sculpt distance parity", async device => {
    const words = packSvoSculptCuts([...cuts, { ...subcellEllipsoidPatch(
      [15.125, 3.125, 6.125], [20.125, 8.125, 11.125], "clear"), affectsScenery: true }], lattice);
    const points = Array.from({ length: 513 }, (_, i) => [
      -2.5 + 5 * ((i * 73 % 521) / 520), -.5 + 3 * ((i * 137 % 521) / 520),
      -1.5 + 2 * ((i * 197 % 521) / 520), 0,
    ].map(Math.fround));
    const records = device.createBuffer({ size: words.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const input = device.createBuffer({ size: points.length * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const output = device.createBuffer({ size: points.length * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: output.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      device.queue.writeBuffer(records, 0, words); device.queue.writeBuffer(input, 0, new Float32Array(points.flat()));
      const module = device.createShaderModule({ code: `
@group(0) @binding(0) var<storage,read> cuts:array<u32>;
@group(0) @binding(1) var<storage,read> points:array<vec4f>;
@group(0) @binding(2) var<storage,read_write> results:array<vec4f>;
fn loadCut(word:u32)->u32{return cuts[word];}
${svoSculptCutsWGSL("loadCut")}
@compute @workgroup_size(64) fn check(@builtin(global_invocation_id) id:vec3u){
  if(id.x>=arrayLength(&points)){return;}let p=points[id.x].xyz;let sample=svoSculptCutSample(p);
  results[id.x*2u]=vec4f(sample.normal,sample.distance);
  results[id.x*2u+1u]=vec4f(svoSculptApplyDistance(-.25,p),0,0,0);
}` });
      const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "check" } });
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [records, input, output].map((buffer, binding) => ({ binding, resource: { buffer } })) });
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(points.length / 64)); pass.end();
      encoder.copyBufferToBuffer(output, 0, read, 0, read.size); device.queue.submit([encoder.finish()]); await read.mapAsync(GPUMapMode.READ);
      const actual = new Float32Array(read.getMappedRange());
      points.forEach((point, i) => {
        const expected = sampleSvoSculptCuts(words, point.slice(0, 3) as [number, number, number]);
        assert.ok(Math.abs(actual[i * 8 + 3]! - expected.distance) < 2e-6);
        expected.normal.forEach((v, a) => assert.ok(Math.abs(actual[i * 8 + a]! - v) < 3e-5));
        assert.ok(Math.abs(actual[i * 8 + 4]! - Math.max(-.25, -expected.distance)) < 2e-6);
      });
      read.unmap();
      // Compile all production field consumers: coverage, contour and DMC.
      for (const [contours, dual] of [[false, false], [true, false], [false, true]]) {
        const production = device.createShaderModule({ code: sparseSceneProxyVoxelizationShaderFor(
          "dry", "f16-unorm8", "dense", undefined, undefined, contours, dual ? 100000 : 0, dual,
          undefined, undefined, undefined, 0, 0) });
        assert.deepEqual((await production.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
        await device.createComputePipelineAsync({ layout: "auto", compute: { module: production, entryPoint: "rebuildDirtyBrickPayload" } });
      }
    } finally { records.destroy(); input.destroy(); output.destroy(); read.destroy(); }
  });
});

test("native imported paver voxelization carves cells, preserves palette, bakes cut normals and restores undo", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  const { readFileSync } = await import("node:fs"), { gunzipSync } = await import("node:zlib");
  const { registerMeshAsset, sampledVolumeExtent } = await import("../lib/core/imported-assets/asset-library");
  const { decodeMeshVolume } = await import("../lib/core/imported-assets/mesh-volume");
  const { SparseBrickOctreeGPU, planSparseBrickOctree, packSparseBrickPlan } = await import("../lib/svo/features/construction/sparse-brick-octree");
  const { packSparseSceneFieldProgramArena } = await import("../lib/core/webgpu-sparse-scene-proxies");
  const { SVO_FIELD_PROGRAM_BLOCK_WORDS } = await import("../lib/svo/features/scene-publication/svo-field-program");
  const { decodeSvoGBufferNormalOct8 } = await import("../lib/svo/contracts/svo-gbuffer");
  const asset = "honed-travertine-paver";
  registerMeshAsset(asset, decodeMeshVolume(gunzipSync(readFileSync(new URL(`../public/assets/pool-party/${asset}.fvx.gz`, import.meta.url)))));
  const primitives = [{ kind: "field-program" as const, center: [0, .025, 0] as const,
    envelopeRadii: sampledVolumeExtent({ asset }), materialId: 100, ownerId: SCENE_ENVIRONMENT_OWNER_BASE,
    program: { ops: [], result: 0, sampledVolume: { asset, materialBase: 100 } } }];
  const packedField = packSparseSceneFieldProgramArena(primitives, 1);
  await withUniformDevice("imported paver sculpt voxelization", async device => {
    let dmcImages: Uint32Array[] | undefined;
    for (const mode of ["coverage", "dmc", "dmc-reference"] as const) {
    const dmc = mode !== "coverage";
    const shaderDevice = mode === "dmc-reference" ? referenceDmcCentreNormals(device) : device;
    const tree = new SparseBrickOctreeGPU(device, { brickSize: 8, nodeCapacity: 1, leafCapacity: 1, payloadProfile: "dry" });
    const packed = packSparseBrickPlan(planSparseBrickOctree([{ x: 0, y: 0, z: 0 }], { brickSize: 8 }));
    device.queue.writeBuffer(tree.structure, tree.nodeOffsetBytes, packed.nodes);
    device.queue.writeBuffer(tree.structure, tree.leafOffsetBytes, packed.leaves);
    device.queue.writeBuffer(tree.control, 0, packed.counts.subarray(0, 4));
    const localLattice = { origin_m: [-.04, -.02, -.04] as const, cellSize_m: [.01, .01, .01] as const };
    const voxelizer = new SparseSceneProxyVoxelizer(shaderDevice, tree, { cellSize: localLattice.cellSize_m, worldOrigin: localLattice.origin_m,
      finestLevel: 0, primitiveCapacity: 1, dirtyRegionCapacity: 2, dirtyBrickCapacity: 1, candidatesPerDirtyBrick: 1,
      fieldProgramCapacity: 1, fieldProgramExtraWords: packedField.length - SVO_FIELD_PROGRAM_BLOCK_WORDS,
      solidWorldLattice: localLattice, surfaceDualMarchingCubes: dmc });
    const read = device.createBuffer({ size: tree.payload.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    let revision = 0;
    const publish = async (patches: readonly SolidWorldVoxelPatch[], identityOnly = false,
      geometry?: { minimum: readonly [number, number, number]; maximum: readonly [number, number, number] }) => {
      voxelizer.setSculptCuts(patches);
      voxelizer.publish({ primitives, dirtyRegions: [{ minimum: [-.04, -.02, -.04], maximum: [.04, .06, .04], identityOnly }, ...(geometry ? [geometry] : [])], revision: ++revision, budgeted: false });
      const encoder = device.createCommandEncoder(); voxelizer.encodeMaintenance(encoder);
      encoder.copyBufferToBuffer(tree.payload, 0, read, 0, read.size); device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ); const image = new Uint32Array(read.getMappedRange().slice(0)); read.unmap(); return image;
    };
    const readDual = async () => {
      if (!dmc) return new Uint32Array();
      const buffer = device.createBuffer({ size: 4096, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      try {
        const encoder = device.createCommandEncoder(), binding = voxelizer.maintenanceBinding;
        encoder.copyBufferToBuffer(binding.buffer, binding.surfaceVertexOffsetBytes!, buffer, 0, 4096);
        device.queue.submit([encoder.finish()]); await buffer.mapAsync(GPUMapMode.READ);
        const words = new Uint32Array(buffer.getMappedRange().slice(0)); buffer.unmap(); return words;
      } finally { buffer.destroy(); }
    };
    try {
      await voxelizer.initializePipelines();
      const baseline = await publish([]), baselineDual = await readDual();
      const base = tree.sceneMaterialOwnerOffsetBytes / 4;
      const distantCut: SolidWorldVoxelPatch = { operation: "clear", affectsScenery: true, minimum: [24, 0, 0], maximumExclusive: [28, 8, 8] };
      const mirrored = await publish([distantCut], true);
      const expectedMirror = baseline.slice();
      for (let i = base; i < base + tree.voxelCapacity; i++) if ((expectedMirror[i]! & 65535) !== 0) expectedMirror[i]! += 8192;
      assert.deepEqual(mirrored, expectedMirror, "distant native transition changes only existing palette identities, preserving every geometry and normal word");
      assert.deepEqual(await readDual(), baselineDual, "identity-only transition retains every DMC point and field value byte");
      assert.deepEqual(await publish([distantCut], true), expectedMirror, "repeated identity repair is idempotent");
      assert.deepEqual(await publish([], true), baseline, "identity-only undo restores the original palette exactly");
      const carved = await publish([{ operation: "clear", affectsScenery: true, minimum: [4, 0, 0], maximumExclusive: [8, 8, 8] }], true,
        { minimum: [0, -.02, -.04], maximum: [.04, .06, .04] });
      const at = (x: number, y: number, z: number) => base + x + 8 * (y + 8 * z);
      const removed = at(6, 3, 3), retained = at(1, 3, 3), cap = at(dmc ? 4 : 3, 3, 3);
      assert.ok((baseline[removed]! & 65535) > 0, "the original imported paver occupies the cut interior");
      assert.equal(carved[removed]! & 65535, 0, "the marked cut erases imported solid cells");
      assert.equal(carved[retained]! & 65535, (baseline[retained]! & 65535) + 8192, "retained imported cells keep their PBR palette identity in the sculpt mirror");
      const normal = decodeSvoGBufferNormalOct8(carved[cap]! >>> 16);
      assert.ok(normal[0] > .98, `cut cap faces into cavity: ${JSON.stringify(normal)}`);
      if (dmc) {
        const { unpackDualGridPoint } = await import("../lib/svo/features/meshing/dual-grid-layout");
        const dualRead = device.createBuffer({ size: 4096, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        try {
          const binding = voxelizer.maintenanceBinding, encoder = device.createCommandEncoder();
          encoder.copyBufferToBuffer(binding.buffer, binding.surfaceVertexOffsetBytes!, dualRead, 0, 4096);
          device.queue.submit([encoder.finish()]); await dualRead.mapAsync(GPUMapMode.READ);
          const range = dualRead.getMappedRange(), points = new Uint32Array(range), values = new Float32Array(range);
          const capVoxel = cap - base, point = unpackDualGridPoint(points[capVoxel * 2]!);
          assert.ok(Math.abs(point[0] * .01) < 1e-4, "the actual dual vertex lies on the carved x=0 plane");
          assert.ok(Math.abs(values[capVoxel * 2 + 1]!) < 1e-5, "the cap dual vertex is a surface zero");
          assert.ok(values[(removed - base) * 2 + 1]! > 0, "DMC evaluates removed imported volume as exterior");
          assert.ok(values[(retained - base) * 2 + 1]! < 0, "DMC retains the source interior");
          dualRead.unmap();
        } finally { dualRead.destroy(); }
      }
      const images = [baseline, baselineDual, mirrored, carved, await readDual()];
      if (mode === "dmc") dmcImages = images;
      if (mode === "dmc-reference") assert.deepEqual(images, dmcImages,
        "discarding occupied centre normals preserves imported geometry, materials, normals and DMC fits");
      assert.deepEqual(await publish([]), baseline, "undo restores every geometry, material and normal word");
    } finally { read.destroy(); voxelizer.destroy(); tree.destroy(); }
    }
  });
});

test("native renderer primitiveHit finds cavity caps before source exits and respects clipped segments", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  const { createSvoDrySceneFragmentWGSL } = await import("../lib/svo/features/shading/program");
  const production = createSvoDrySceneFragmentWGSL(1, "raster-primary", "off", "split", 0, false, false, false, {});
  const start = production.indexOf("fn primitiveHit(record:");
  const end = production.indexOf("// The patch half", start);
  assert.ok(start >= 0 && end > start);
  // Execute the actual renderer function against a simple analytic box oracle;
  // the full production shader is compiled separately by imported-surface-gpu.
  const rayFunction = production.slice(start, end);
  const words = packSvoSculptCuts([{ operation: "clear", affectsScenery: true, minimum: [-2, -20, -20], maximumExclusive: [2, 20, 20] }],
    { origin_m: [0, 0, 0], cellSize_m: [.1, .1, .1] });
  await withUniformDevice("renderer carved cavity rays", async device => {
    const records = device.createBuffer({ size: words.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const output = device.createBuffer({ size: 7 * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: output.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const module = device.createShaderModule({ code: `
@group(0) @binding(0) var<storage,read> cuts:array<u32>;
@group(0) @binding(1) var<storage,read_write> results:array<vec4f>;
fn loadCut(word:u32)->u32{return cuts[word];}
${svoSculptCutsWGSL("loadCut")}
struct SvoPrimitiveRecord{orientation:vec4f}
struct Exact{status:u32,t_m:f32,normal:vec4f,featureId:u32}
struct DryHit{t:f32,normal:vec3f,material:u32,owner:u32,feature:u32,field:u32,motion:u32,valid:u32,extra:f32,padding:vec3u}
struct Surface{material:u32,normal:vec3f}
const SVO_PRIMITIVE_RAY_HIT=1u;const SVO_KIND_FIELD_PROGRAM=12u;
const DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET=0u;const DRY_SCENE_FIELD_PROGRAM_BLOCK_WORDS=4u;
const DRY_GBUFFER_FIELD_ANALYTIC=1u;const DRY_GBUFFER_MOTION_STATIC=0u;
fn missHit()->DryHit{return DryHit(1e20,vec3f(0),0,0,0,0,0,0,0,vec3u(0));}
fn dryClusterPacking(r:SvoPrimitiveRecord)->u32{return 0u;}
fn svoPrimitiveMaterialId(r:SvoPrimitiveRecord)->u32{return 7u;}
fn svoPrimitiveOwnerId(r:SvoPrimitiveRecord)->u32{return 9u;}
fn svoPrimitiveKind(r:SvoPrimitiveRecord)->u32{return 0u;}
fn svoPrimitiveFieldProgramReference(r:SvoPrimitiveRecord)->u32{return 0u;}
fn svoPrimitiveLocalPoint(r:SvoPrimitiveRecord,p:vec3f)->vec3f{return p;}
fn dryFieldProgramBlockSurface(i:u32,p:vec3f)->Surface{return Surface(7u,vec3f(1,0,0));}
fn svoQuaternionRotate(q:vec4f,n:vec3f)->vec3f{return n;}
fn svoPrimitiveDistance_m(r:SvoPrimitiveRecord,p:vec3f,c:u32)->f32{return abs(p.x)-1.;}
fn svoEvaluatePrimitive(r:SvoPrimitiveRecord,p:vec3f,c:u32)->Exact{return Exact(1u,0.,vec4f(sign(p.x),0,0,0),0u);}
fn svoIntersectPrimitiveExact(r:SvoPrimitiveRecord,ro:vec3f,rd:vec3f,lo:f32,hi:f32,c:u32)->Exact{
 let a=(-1.-ro.x)/rd.x;let b=(1.-ro.x)/rd.x;let enter=min(a,b);let exit=max(a,b);
 let t=select(exit,enter,enter>=lo);return Exact(select(0u,1u,t>=lo&&t<=hi),t,vec4f(sign(ro.x+rd.x*t),0,0,0),0u);
}
${rayFunction}
@compute @workgroup_size(1) fn check(){
 let record=SvoPrimitiveRecord(vec4f(0,0,0,1));
 let origins=array<f32,7>(0.,0.,-2.,0.,-.1,-.5,-.5);let limits=array<f32,7>(2.,.5,4.,.1,.5,2.,.6);
 for(var i=0u;i<7u;i+=1u){let hit=primitiveHit(record,vec3f(origins[i],0,0),vec3f(1,0,0),0.,limits[i]);results[i]=vec4f(hit.normal,hit.t);}
}` });
    try {
      device.queue.writeBuffer(records, 0, words);
      const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "check" } });
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [records, output].map((buffer, binding) => ({ binding, resource: { buffer } })) });
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
      encoder.copyBufferToBuffer(output, 0, read, 0, read.size); device.queue.submit([encoder.finish()]); await read.mapAsync(GPUMapMode.READ);
      const actual = new Float32Array(read.getMappedRange());
      [.2, .2, 1, 1e20, .3, .3, .3].forEach((t, i) => assert.ok(Math.abs(actual[i * 4 + 3]! - t) < Math.max(1e-5, t * 1e-6), `ray ${i}: ${actual[i * 4 + 3]} vs ${t}`));
      assert.equal(actual[0], -1, "right cavity cap points back into the cavity");
      assert.equal(actual[4], -1); assert.equal(actual[20], 1, "retained-solid ray sees the cavity entrance"); read.unmap();
    } finally { records.destroy(); output.destroy(); read.destroy(); }
  });
});

test("sculpt ownership uses rotated boxes without withdrawing distant flat or tall native meshes", () => {
  const at = (p: readonly number[]) => packSvoSculptCuts([{ operation: "clear", affectsScenery: true,
    minimum: p.map(v => v - .0001) as [number, number, number],
    maximumExclusive: p.map(v => v + .0001) as [number, number, number] }], { origin_m: [0, 0, 0], cellSize_m: [1, 1, 1] });
  const groundCut = at([0, 0, 0]);
  assert.equal(intersectsSvoSculptPrimitiveBounds(groundCut, [0, 4, 0], [3, 3, 3]), false,
    "ground cut must not replace a canopy whose lowest point is one metre above it");
  assert.equal(intersectsSvoSculptPrimitiveBounds(at([0, .2, 0]), [0, 0, 0], [1, .03, 1]), false,
    "a flat paver does not inherit its diagonal radius as vertical reach");
  const rotation = [0, Math.SQRT1_2, 0, Math.SQRT1_2] as const;
  assert.equal(intersectsSvoSculptPrimitiveBounds(at([0, 0, 1.8]), [0, 0, 0], [2, .1, .1], rotation), true);
  assert.equal(intersectsSvoSculptPrimitiveBounds(at([1.8, 0, 0]), [0, 0, 0], [2, .1, .1], rotation), false);
  // Every corner of arbitrarily rotated source bounds remains conservative.
  for (let i = 1; i <= 20; i++) {
    const raw = [Math.sin(i), Math.cos(i * 1.7), Math.sin(i * 2.3), Math.cos(i * .3)];
    const length = Math.hypot(...raw), q = raw.map(v => v / length) as [number, number, number, number];
    const [x, y, z, w] = q, extent = [.2, 3, .7] as const;
    for (let corner = 0; corner < 8; corner++) {
      const p = extent.map((v, axis) => v * ((corner >> axis) & 1 ? 1 : -1));
      const cross = [y * p[2]! - z * p[1]!, z * p[0]! - x * p[2]!, x * p[1]! - y * p[0]!];
      const twice = [y * cross[2]! - z * cross[1]!, z * cross[0]! - x * cross[2]!, x * cross[1]! - y * cross[0]!];
      const rotated = p.map((v, axis) => v + 2 * (w * cross[axis]! + twice[axis]!));
      assert.equal(intersectsSvoSculptPrimitiveBounds(at(rotated), [0, 0, 0], extent, q), true);
    }
  }
});
