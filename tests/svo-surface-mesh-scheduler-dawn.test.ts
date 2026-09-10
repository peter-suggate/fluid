import assert from "node:assert/strict";
import test from "node:test";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createDawnRenderDevice } from "../tools/svo-dry-frame-harness";
import {
  SVO_SURFACE_MESH_BOX_UNION_SLOT,
  SVO_SURFACE_MESH_BOX_WORDS,
  SVO_SURFACE_MESH_FLAGS,
  SVO_SURFACE_MESH_MODE,
  SVO_SURFACE_MESH_QUAD_BYTES,
  SVO_SURFACE_MESH_STATE,
  SVO_SURFACE_MESH_STATE_BYTES,
  interpretSurfaceMeshState,
  surfaceMeshWorkBytes,
  svoSurfaceMeshWGSL,
} from "../lib/svo/features/primary-visibility/svo-surface-mesh";
import { SPARSE_SCENE_MAINTENANCE_STATE_WORDS } from "../lib/core/webgpu-sparse-scene-proxies";

/**
 * The production build kernels over a two-brick synthetic octree: brick 0 at
 * lattice [0,2)^3 and brick 1 at [2,4)x[0,2)x[0,2), each 2x2x2 voxels. The
 * host sequence mirrors the dry scene's per-presentation passes.
 */
(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("the mesh scheduler builds, re-extracts dirty bricks in place, rolls an overflow back, and flips a replacement", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "tests/svo-surface-mesh-scheduler-dawn.test.ts");
  let device: GPUDevice | undefined;
  try {
    const initialized = await createDawnRenderDevice(); device = initialized.device;
    const W = SVO_SURFACE_MESH_STATE;
    const M = SPARSE_SCENE_MAINTENANCE_STATE_WORDS;
    const source = svoSurfaceMeshWGSL(1, 1);
    const kernels = source.slice(0, source.indexOf("// Which of a brick's levels this camera draws"));
    const module = device.createShaderModule({ code: `
      struct Mapping { worldOrigin:vec3f, brickSize:u32, cellSize:f32, maximumDepth:u32 }
      struct Dry { materialPublication:vec4u, mapping:Mapping, lod:vec4f, meshFilter:vec4f }
      const dry=Dry(vec4u(0u,0u,0u,1u),Mapping(vec3f(0.0),2u,1.0,1u),vec4f(0.0),vec4f(0.0));
      struct View { cameraPosition:vec4f, viewport:vec2f }
      const uniforms=View(vec4f(-5.0,0.5,0.5,0.0),vec2f(800.0,460.0));
      const REQUIRED_FIELDS:u32=1u;
      const SVO_INVALID:u32=0xffffffffu;
      const SCENE_IDENTITY_NO_NORMAL:u32=0xffffu;
      // Words: valid, fields, topology revision, geometry revision, node count, leaf count.
      @group(0) @binding(1) var<storage,read> publication:array<u32>;
      @group(0) @binding(2) var<storage,read> nodes:array<vec4u>;
      @group(0) @binding(3) var<storage,read> leaves:array<vec4u>;
      @group(0) @binding(4) var<storage,read> voxels:array<u32>;
      fn dryPublicationWord(i:u32)->u32{return publication[i];}
      fn svoControlLoad(i:u32)->u32{return publication[4u+i];}
      struct Leaf { topology:vec4u }
      struct Node { address:vec4u, links:vec4u }
      fn svoLeafLoad(i:u32)->Leaf{return Leaf(leaves[i]);}
      fn svoNodeLoad(i:u32)->Node{return Node(nodes[i*2u],nodes[i*2u+1u]);}
      fn svoBrickLifecycleDecode(w:u32)->u32{return w;}
      fn svoBrickLifecycleCurrent(l:u32)->bool{return l==1u;}
      fn svoDecodeMorton(x:u32,y:u32,depth:u32)->vec3u{return vec3u(x,y,0u);}
      fn svoBrickVoxelIndex(payload:u32,local:vec3u,n:u32)->u32{return payload+local.x+local.y*n+local.z*n*n;}
      fn dryVoxelCapacity()->u32{return arrayLength(&voxels);}
      fn sceneIdentityAt(v:u32)->u32{return voxels[v];}
      fn sceneIdentitySolid(i:u32)->bool{return (i&0xffffu)!=0u;}
      fn sceneIdentityMaterial(i:u32)->u32{return i&0xffffu;}
      fn sceneIdentityHasNormal(i:u32)->bool{return false;}
      fn sceneIdentityNormal(i:u32)->vec3f{return vec3f(0.0);}
      fn svoGBufferPackNormalOct8(n:vec3f)->u32{return 0u;}
      ${kernels}
    ` });
    const sceneLayout = device.createBindGroupLayout({ entries: [1, 2, 3, 4].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const } })) });
    const meshLayout = device.createBindGroupLayout({ entries: [
      ...[30, 32, 38, 34, 40].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } })),
      { binding: 41, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const } },
    ] });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [sceneLayout, meshLayout] });
    const names = ["surfaceMeshPrepare", "surfaceMeshBoxes", "surfaceMeshMark", "surfaceMeshSchedule", "surfaceMeshCount", "surfaceMeshAllocate", "surfaceMeshEmit", "surfaceMeshPublish"] as const;
    const pipelines = Object.fromEntries(await Promise.all(names.map(async (entryPoint) =>
      [entryPoint, await device!.createComputePipelineAsync({ label: entryPoint, layout, compute: { module, entryPoint } })] as const))) as Record<typeof names[number], GPUComputePipeline>;

    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const publication = device.createBuffer({ size: 32, usage: storage });
    const nodes = device.createBuffer({ size: 6 * 16, usage: storage });
    const leaves = device.createBuffer({ size: 2 * 16, usage: storage });
    const voxels = device.createBuffer({ size: 16 * 4, usage: storage });
    // Root with two children in octants 0 and +x; each child is a current voxel brick.
    device.queue.writeBuffer(nodes, 0, new Uint32Array([
      0, 0, 0, 0b11, /* links */ 1, 0, 0xffffffff, 0,
      0, 0, 1, 0, /* links */ 0, 0, 0, 1,
      1, 0, 1, 0, /* links */ 0, 0, 1, 1,
    ]));
    device.queue.writeBuffer(leaves, 0, new Uint32Array([1, 0, 0, 0, 2, 8, 0, 0]));
    const setVoxel = (index: number, material: number) => device!.queue.writeBuffer(voxels, index * 4, new Uint32Array([material]));
    const publish = (topology: number, geometry: number) => device!.queue.writeBuffer(publication, 0, new Uint32Array([1, 1, topology, geometry, 3, 2]));
    const sceneGroup = device.createBindGroup({ layout: sceneLayout, entries: [
      { binding: 1, resource: { buffer: publication } }, { binding: 2, resource: { buffer: nodes } },
      { binding: 3, resource: { buffer: leaves } }, { binding: 4, resource: { buffer: voxels } }] });

    const state = device.createBuffer({ size: SVO_SURFACE_MESH_STATE_BYTES, usage: storage | GPUBufferUsage.INDIRECT });
    const dispatch = device.createBuffer({ size: 64, usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });
    const work = device.createBuffer({ size: surfaceMeshWorkBytes(2), usage: storage });
    const visible = device.createBuffer({ size: 1024, usage: storage });
    const maintenanceRecords = 24;
    const maintenance = device.createBuffer({ size: (maintenanceRecords + 4 * 4) * 4, usage: storage });
    const empty = device.createBuffer({ size: 2 * SVO_SURFACE_MESH_QUAD_BYTES, usage: storage });
    const arenaQuads = 40;
    const arenas: [GPUBuffer, GPUBuffer | undefined] = [device.createBuffer({ size: arenaQuads * SVO_SURFACE_MESH_QUAD_BYTES, usage: storage }), undefined];
    const bind = () => device!.createBindGroup({ layout: meshLayout, entries: [
      { binding: 30, resource: { buffer: state } }, { binding: 32, resource: { buffer: arenas[0] } },
      { binding: 38, resource: { buffer: arenas[1] ?? empty } }, { binding: 34, resource: { buffer: visible } },
      { binding: 40, resource: { buffer: work } }, { binding: 41, resource: { buffer: maintenance } }] });
    let meshGroup = bind();
    // Host words: 512 bricks a batch, a bound maintenance list of two leaf slots, the list's layout.
    device.queue.writeBuffer(state, W.bricksPerBatch * 4, new Uint32Array([512]));
    device.queue.writeBuffer(state, W.hostMaintenance * 4, new Uint32Array([1, 2, 0]));
    device.queue.writeBuffer(state, W.hostMaintenanceStateWords * 4, new Uint32Array([0, maintenanceRecords, 4]));
    const maintain = (requested: number, completed: number, dirtyLeaves: number[]) => {
      const words = new Uint32Array(maintenanceRecords + 16);
      words[M.dirtyBrickCount] = dirtyLeaves.length; words[M.requestedRevision] = requested; words[M.completedRevision] = completed;
      dirtyLeaves.forEach((leaf, index) => { words[maintenanceRecords + index * 4] = leaf; });
      device!.queue.writeBuffer(maintenance, 0, words);
    };

    const readback = device.createBuffer({ size: SVO_SURFACE_MESH_STATE_BYTES + 2 * 128 * SVO_SURFACE_MESH_QUAD_BYTES, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const frame = async () => {
      const encoder = device!.createCommandEncoder();
      const compute = (pipeline: GPUComputePipeline, indirectOffset?: number) => {
        const pass = encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, sceneGroup); pass.setBindGroup(1, meshGroup);
        if (indirectOffset === undefined) pass.dispatchWorkgroups(1); else pass.dispatchWorkgroupsIndirect(dispatch, indirectOffset);
        pass.end();
      };
      compute(pipelines.surfaceMeshPrepare);
      compute(pipelines.surfaceMeshBoxes);
      encoder.copyBufferToBuffer(state, W.markDispatch * 4, dispatch, 0, 12);
      compute(pipelines.surfaceMeshMark, 0);
      compute(pipelines.surfaceMeshSchedule);
      encoder.copyBufferToBuffer(state, W.extractDispatch * 4, dispatch, 16, 12);
      encoder.copyBufferToBuffer(state, W.allocateDispatch * 4, dispatch, 32, 12);
      compute(pipelines.surfaceMeshCount, 16);
      compute(pipelines.surfaceMeshAllocate, 32);
      compute(pipelines.surfaceMeshEmit, 16);
      compute(pipelines.surfaceMeshPublish);
      encoder.copyBufferToBuffer(state, 0, readback, 0, SVO_SURFACE_MESH_STATE_BYTES);
      for (const slot of [0, 1] as const) {
        const arena = arenas[slot]; if (!arena) continue;
        encoder.copyBufferToBuffer(arena, 0, readback, SVO_SURFACE_MESH_STATE_BYTES + slot * 128 * SVO_SURFACE_MESH_QUAD_BYTES, Math.min(arena.size, 128 * SVO_SURFACE_MESH_QUAD_BYTES));
      }
      device!.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const bytes = readback.getMappedRange().slice(0); readback.unmap();
      const words = new Uint32Array(bytes, 0, SVO_SURFACE_MESH_STATE_BYTES / 4);
      const arena = (slot: 0 | 1, count: number) => {
        const quads = new Uint32Array(bytes, SVO_SURFACE_MESH_STATE_BYTES + slot * 128 * SVO_SURFACE_MESH_QUAD_BYTES, count * 8);
        const alive: number[] = []; const dead: number[] = [];
        for (let i = 0; i < count; i += 1) {
          const extent = [quads[i * 8 + 4]!, quads[i * 8 + 5]!, quads[i * 8 + 6]!];
          (extent.every((value) => value === 0) ? dead : alive).push(i);
        }
        return { alive, dead };
      };
      const receipt = interpretSurfaceMeshState(words, { arenaBytes: [arenas[0].size, arenas[1]?.size ?? empty.size], maximumBytes: 1 << 20 });
      return { words, arena, receipt };
    };

    // Brick 0 holds one solid voxel at lattice x in [1,2): six exact faces
    // and six coarse faces for its one level-1 cell.
    setVoxel(1, 3);
    publish(1, 1);
    maintain(1, 1, []);
    let result = await frame();
    assert.equal(result.receipt.status.state, "ready");
    assert.equal(result.words[W.usable], 1);
    assert.equal(result.words[W.frontCursor], 12);
    assert.equal(result.words[W.liveQuads], 12);
    assert.equal(result.words[W.builds], 1);
    assert.equal(result.words[W.consumedMaintenanceRevision], 1);
    assert.deepEqual(result.arena(0, 12).dead, []);
    result = await frame();
    assert.equal(result.words[W.frontCursor], 12, "an unchanged publication extracts nothing");
    assert.equal(result.words[W.builds], 1);

    // Brick 1 gains a solid voxel touching brick 0's across their shared
    // face. The dirty list names brick 1 alone; brick 0 is re-extracted as
    // its neighbour, both lose the face between them, and the old ranges are
    // freed in place while the mesh stays drawn.
    setVoxel(8, 3);
    maintain(2, 2, [1]);
    publish(1, 2);
    result = await frame();
    assert.equal(result.receipt.status.state, "ready");
    assert.equal(result.words[W.builds], 2);
    assert.equal(result.words[W.restartReason], 3, "a geometry revision");
    assert.equal(result.words[W.worklistCount], 2, "the dirty brick and its neighbour");
    assert.equal(result.words[W.liveQuads], 22);
    assert.equal(result.words[W.frontCursor], 34);
    assert.equal(result.words[W.consumedMaintenanceRevision], 2);
    assert.equal(result.words[W.boxCount], 0, "the mask lifts when the build completes");
    assert.deepEqual(result.arena(0, 34).dead, Array.from({ length: 12 }, (_, i) => i), "the first build's range is freed");
    assert.equal(result.arena(0, 34).alive.length, 22);

    // A second voxel in brick 1 (its side faces merge with the first's, so
    // the brick still needs eleven quads) puts the batch's 22 quads past a
    // cursor of 34 in a 40-quad arena: it rolls back whole and waits for growth.
    setVoxel(9, 3);
    maintain(3, 3, [1]);
    publish(1, 3);
    result = await frame();
    assert.equal(result.words[W.errorFlags] & 1, 1);
    assert.equal(result.words[W.frontCursor], 34, "the cursor returns to the checkpoint");
    assert.equal(result.words[W.liveQuads], 22, "the batch's bricks keep their previous ranges");
    assert.equal(result.arena(0, 34).alive.length, 22, "nothing was freed by the rolled-back batch");
    assert.deepEqual(result.receipt.grow, { slot: 0, overflowQuads: arenaQuads });
    assert.equal(result.receipt.status.buildPhase, "capacity");
    const paused = await frame();
    assert.equal(paused.words[W.frontCursor], 34, "a paused build does not advance");
    assert.equal(paused.words[W.builds], 3);
    const grown = device.createBuffer({ size: 2 * arenaQuads * SVO_SURFACE_MESH_QUAD_BYTES, usage: storage });
    const copy = device.createCommandEncoder(); copy.copyBufferToBuffer(arenas[0], 0, grown, 0, arenas[0].size); device.queue.submit([copy.finish()]);
    arenas[0].destroy(); arenas[0] = grown; meshGroup = bind();
    result = await frame();
    assert.equal(result.words[W.errorFlags], 0);
    assert.equal(result.words[W.builds], 3, "growth does not restart the build");
    assert.equal(result.words[W.frontCursor], 56);
    assert.equal(result.words[W.liveQuads], 22);
    assert.equal(result.receipt.status.state, "ready");
    assert.equal(result.arena(0, 56).alive.length, 22);
    assert.equal(result.arena(0, 56).dead.length, 34);

    // A publication whose dirty list is still being rewritten (requested
    // ahead of completed) cannot be trusted: the mesh is rebuilt beside the
    // drawn one, into a back arena the host provides, and flips.
    maintain(5, 4, [0, 1]);
    publish(1, 4);
    result = await frame();
    assert.equal(result.words[W.mode], SVO_SURFACE_MESH_MODE.replacement);
    assert.equal(result.words[W.flags] & SVO_SURFACE_MESH_FLAGS.needBack, SVO_SURFACE_MESH_FLAGS.needBack);
    assert.equal(result.words[W.usable], 1, "the front stays drawn while waiting");
    assert.equal(result.words[W.frontCursor], 56);
    assert.equal(typeof result.receipt.needBackBytes, "number");
    assert.equal(result.receipt.status.buildPhase, "capacity");
    arenas[1] = device.createBuffer({ size: 2 * arenaQuads * SVO_SURFACE_MESH_QUAD_BYTES, usage: storage });
    device.queue.writeBuffer(state, W.hostBackGeneration * 4, new Uint32Array([1]));
    meshGroup = bind();
    result = await frame();
    assert.equal(result.words[W.front], 1, "the replacement flipped to the back arena");
    assert.equal(result.words[W.frontCursor], 22, "the compact arena holds only live quads");
    assert.equal(result.words[W.liveQuads], 22);
    assert.equal(result.words[W.backCursor], 0);
    assert.equal(result.words[W.backGenerationConsumed], 1);
    assert.equal(result.words[W.mode], SVO_SURFACE_MESH_MODE.idle);
    assert.equal(result.receipt.front, 1);
    assert.equal(result.receipt.status.state, "ready");
    assert.deepEqual(result.arena(1, 22).dead, []);
    assert.equal(result.words[W.builds], 4);
    result = await frame();
    assert.equal(result.words[W.frontCursor], 22, "the flipped mesh is reused");

    assert.deepEqual(initialized.validationErrors, []);
    for (const buffer of [publication, nodes, leaves, voxels, state, dispatch, work, visible, maintenance, empty, arenas[0], arenas[1]!, readback]) buffer.destroy();
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
