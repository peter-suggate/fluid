import assert from "node:assert/strict";
import test from "node:test";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { SculptWallMesh, sculptWallMeshWGSL, SCULPT_WALL_VERTEX_CAPACITY } from "../lib/svo/features/primary-visibility/sculpt-wall-mesh";
import { importedSurfaceCsgWGSL } from "../lib/svo/features/primary-visibility/imported-surface-csg.wgsl";
import { WebgpuSculptSubstrate } from "../lib/svo/features/scene-publication/webgpu-sculpt-substrate";
import { SCULPT_ARENA_WORDS } from "../lib/svo/features/scene-publication/svo-sculpt-substrate";
import { packSvoSculptCuts, svoSculptCutsWGSL } from "../lib/svo/features/scene-publication/svo-sculpt-cuts";
import type { SolidWorldVoxelPatch } from "../lib/core/solid-world";

const fixture = `struct Primitive { id:u32 }
struct Dry { metadata:vec4u }
const dry=Dry(vec4u(1,0,0,0));
struct DryHit { t:f32,normal:vec3f,material:u32,owner:u32,feature:u32,field:u32,motion:u32,valid:u32,extra:f32,padding:vec3u }
struct Surface { material:u32,normal:vec3f }
const SVO_KIND_FIELD_PROGRAM=12u;const DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET=0u;const DRY_SCENE_FIELD_PROGRAM_BLOCK_WORDS=4u;
const SVO_FEATURE_SMOOTH=0u;const DRY_GBUFFER_FIELD_ANALYTIC=1u;const DRY_GBUFFER_MOTION_STATIC=0u;
fn missHit()->DryHit{return DryHit(1e20,vec3f(0),0,0,0,0,0,0,0,vec3u(0));}
fn dryPrimitive(i:u32)->Primitive{return Primitive(i);}
fn svoPrimitiveKind(p:Primitive)->u32{return SVO_KIND_FIELD_PROGRAM;}
fn svoPrimitiveFieldProgramReference(p:Primitive)->u32{return 0u;}
fn drySceneWord(i:u32)->u32{return 0xfffffffeu;}
fn dryFieldProgramBlockVolumeDistance(base:u32,w:vec3f)->f32{return length(w)-1.0;}
fn svoPrimitiveLocalPoint(p:Primitive,w:vec3f)->vec3f{return w;}
fn dryFieldProgramBlockSurface(i:u32,p:vec3f)->Surface{return Surface(7u,normalize(p));}
fn svoPrimitiveMaterialId(p:Primitive)->u32{return 7u;}
fn svoPrimitiveOwnerId(p:Primitive)->u32{return 9u;}
`;
test("GPU wall triangles cache geometry, clip to source, and restore undo/redo without camera rebuilds", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice("cached sculpt wall triangles", async device => {
    const arena = device.createBuffer({ size: SCULPT_ARENA_WORDS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const field = new WebgpuSculptSubstrate(device), mesh = new SculptWallMesh(device);
    await field.initialize();
    const module = device.createShaderModule({ code: `
@group(0) @binding(0) var<storage,read> cuts:array<u32>;
fn drySculptCutWord(i:u32)->u32{return cuts[i];}
${svoSculptCutsWGSL("drySculptCutWord")}
${fixture}
${importedSurfaceCsgWGSL}
${sculptWallMeshWGSL}` });
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error").map(m => m.message), []);
    const layout = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }] });
    const group = device.createBindGroup({ layout, entries: [{ binding: 0, resource: { buffer: arena } }] });
    const pipelines = await mesh.compile(module, layout);
    const lattice = { origin_m: [0, 0, 0] as const, cellSize_m: [.1, .1, .1] as const };
    const sphere: SolidWorldVoxelPatch = { operation: "clear", kind: "ellipsoid", affectsScenery: true, minimum: [-5,-5,5], maximumExclusive: [5,5,15] };
    const box: SolidWorldVoxelPatch = { operation: "clear", affectsScenery: true, minimum: [-2,-2,2], maximumExclusive: [2,2,8] };
    async function read(buffer: GPUBuffer, bytes = buffer.size) {
      const target = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(buffer, 0, target, 0, bytes); device.queue.submit([encoder.finish()]);
      await target.mapAsync(GPUMapMode.READ); const result = target.getMappedRange().slice(0); target.unmap(); target.destroy(); return result;
    }
    async function update(patches: SolidWorldVoxelPatch[]) {
      field.publish(arena, 0, packSvoSculptCuts(patches, lattice));
      const encoder = device.createCommandEncoder(); mesh.encode(encoder, field.bakePlan, "source", group, pipelines); device.queue.submit([encoder.finish()]);
      const vertices: string[] = [];
      for (const held of mesh.pages.values()) {
        const args = new Uint32Array(await read(held.args));
        assert.equal(args[0], args[4], "ordinary cuts must use cached triangles, not overflow proxies");
        assert.equal(args[0]! % 3, 0);
        if (!args[0]) continue;
        const data = await read(held.vertices, args[0] * 32), f = new Float32Array(data), u = new Uint32Array(data);
        for (let i = 0; i < f.length; i += 8) {
          assert.ok(Math.hypot(f[i]!,f[i+1]!,f[i+2]!) <= 1.00001, "wall vertices are clipped inside the source sphere");
          assert.equal(u[i+3],8199); assert.equal(u[i+7],9);
          assert.ok(Math.abs(Math.hypot(f[i+4]!,f[i+5]!,f[i+6]!)-1)<1e-5);
          vertices.push(Array.from(u.subarray(i,i+8)).join(","));
        }
      }
      return vertices.sort();
    }
    try {
      const first = await update([sphere]); assert.ok(first.length > 100);
      const built = mesh.builtPages, held = [...mesh.pages.values()].map(p => p.vertices);
      const encoder = device.createCommandEncoder(); mesh.encode(encoder, field.bakePlan, "source", group, pipelines); device.queue.submit([encoder.finish()]);
      assert.equal(mesh.builtPages,built,"camera-only frames perform no extraction");
      assert.deepEqual([...mesh.pages.values()].map(p => p.vertices),held,"camera frames retain GPU buffers");
      const second = await update([sphere,box]); assert.notDeepEqual(second,first);
      assert.deepEqual(await update([sphere]),first,"undo restores the complete first mesh");
      assert.deepEqual(await update([sphere,box]),second,"redo restores positions, normals, material and owner");
      // Exercise bounded overflow publication explicitly, without manufacturing
      // pathological geometry or reading a count in the production path.
      const page = [...mesh.pages.values()][0]!;
      device.queue.writeBuffer(page.args,16,new Uint32Array([SCULPT_WALL_VERTEX_CAPACITY+3]));
      const overflow = device.createCommandEncoder(), pass = overflow.beginComputePass();
      pass.setPipeline(pipelines.publish); pass.setBindGroup(0,group); pass.setBindGroup(1,page.group); pass.dispatchWorkgroups(1); pass.end(); device.queue.submit([overflow.finish()]);
      assert.equal(new Uint32Array(await read(page.args))[0],36);
      const proxy = new Uint32Array(await read(page.vertices,36*32));
      for(let i=0;i<36;i++){assert.equal(proxy[i*8+3],0xffffffff);assert.equal(proxy[i*8+7],page.page.slot);}
      assert.deepEqual(await update([]),[]); assert.equal(mesh.pages.size,0,"undo-all removes every cached page");
      assert.deepEqual(await update([sphere]),first,"branching after undo-all reconstructs fresh geometry");
    } finally { mesh.destroy(); field.destroy(); arena.destroy(); }
  });
});
