import assert from "node:assert/strict";
import test from "node:test";
import { packSvoSculptCuts, svoSculptCutsWGSL } from "../lib/svo/features/scene-publication/svo-sculpt-cuts";
import { planSculptBake, SculptPickingField, SCULPT_ARENA_WORDS, SCULPT_PAGE_CAPACITY } from "../lib/svo/features/scene-publication/svo-sculpt-substrate";
import { WebgpuSculptSubstrate } from "../lib/svo/features/scene-publication/webgpu-sculpt-substrate";
import type { SolidWorldVoxelPatch } from "../lib/core/solid-world";
import { subcellEllipsoidPatch } from "../lib/core/solid-patch/ellipsoid-bounds";
import { withUniformDevice } from "./helpers/uniform-geometric";
const lattice = { origin_m: [0, 0, 0] as const, cellSize_m: [.01, .01, .01] as const };
const box = (x: number): SolidWorldVoxelPatch => ({ operation: "clear", affectsScenery: true,
  minimum: [x, 0, 0], maximumExclusive: [x + 4, 4, 4] });
const packed = (cuts: readonly SolidWorldVoxelPatch[]) => packSvoSculptCuts(cuts, lattice);

test("packing reuses immutable prefixes without losing undo, branches or lattice metadata", () => {
  const a = box(0), b = box(40), c = box(-40), fill = { ...box(3), operation: "fill" as const };
  const base = [a, fill], extended = [...base, b];
  const first = packed(base), second = packed(extended);
  assert.equal(packed(extended), second, "one publication is packed once across consumers");
  assert.deepEqual(second.subarray(4, 12), first.subarray(4, 12));
  assert.deepEqual(packed([a, fill]), first, "undo truncates only the command stream");
  assert.deepEqual(packed([a, fill, c]).subarray(12, 20), packed([c]).subarray(4, 12));
  const growing = [a]; packed(growing); growing.push(b);
  assert.equal(packed(growing)[0], 2, "an appended array cannot reuse a stale record count");
  const loaded = JSON.parse(JSON.stringify(extended)) as SolidWorldVoxelPatch[];
  assert.deepEqual(packed(loaded), second, "saved history rehydrates the same field operands");
});

test("sculpt append bakes only new operands, independently of history length", () => {
  const repeated = Array.from({ length: 4200 }, () => box(0));
  const before = planSculptBake(undefined, packed(repeated));
  const next = planSculptBake(before, packed([...repeated, box(1)]));
  assert.ok(next.jobs.length > 0);
  assert.ok(next.jobs.every(job => !job.reset && job.cuts.length === 1 && job.cuts[0] === 4200));
  assert.equal(next.pages.size, before.pages.size);
  assert.ok(next.proxies[0]! < 20, "rendering draws baked pages, not thousands of historical stamp proxies");
});

test("sculpt undo/redo/branch rebuild only changed pages and preserve carved emptiness", () => {
  const initial = planSculptBake(undefined, packed([box(0)]));
  const more = planSculptBake(initial, packed([box(0), box(40)]));
  const undone = planSculptBake(more, packed([box(0)]));
  assert.equal(undone.pages.size, initial.pages.size);
  assert.equal(undone.jobs.length, 0, "removing distant pages needs no rebake of retained pages");
  const redo = planSculptBake(undone, packed([box(0), box(40)]));
  assert.ok(redo.jobs.every(job => job.reset));
  const a = new SculptPickingField(initial), b = new SculptPickingField(more, a);
  const c = new SculptPickingField(undone, b), d = new SculptPickingField(redo, c);
  for (const field of [a, b, c, d]) assert.ok(field.sample([.015, .015, .015]).distance < 0);
  assert.ok(b.sample([.415, .015, .015]).distance < 0);
  assert.ok(c.sample([.415, .015, .015]).distance > 0);
  assert.deepEqual(d.sample([.415, .015, .015]), b.sample([.415, .015, .015]));
  const branch = planSculptBake(undone, packed([box(0), box(-40)]));
  assert.equal(branch.slots, more.slots, "branching reuses freed GPU pages");
  assert.equal(planSculptBake(branch, packed([])).pages.size, 0);
});

test("picking uses baked samples, keeps old snapshots immutable and handles one-cell cuts", () => {
  const small: SolidWorldVoxelPatch = { ...box(0), maximumExclusive: [1, 1, 1] };
  const first = planSculptBake(undefined, packed([small]));
  const a = new SculptPickingField(first), centre = [.005, .005, .005] as const;
  assert.ok(Math.abs(a.sample(centre).distance + .005) < 1e-7);
  const before = a.sample(centre);
  const second = planSculptBake(first, packed([small, box(0)]));
  const b = new SculptPickingField(second, a);
  assert.deepEqual(a.sample(centre), before);
  assert.ok(b.sample([.025, .025, .025]).distance < 0);
  assert.ok(a.sample([.025, .025, .025]).distance > 0);
  const copy = JSON.parse(JSON.stringify([small, box(0)])) as SolidWorldVoxelPatch[];
  assert.equal(planSculptBake(second, packed(copy)).jobs.length, 0, "save/load preserves the field's content identity");
});

test("baking respects translated and anisotropic voxel grids, including single-cell cuts", () => {
  const translated = { origin_m: [.003, -.007, .002] as const, cellSize_m: [.01, .02, .005] as const };
  const small = { ...box(0), maximumExclusive: [1, 1, 1] as const };
  const plan = planSculptBake(undefined, packSvoSculptCuts([small], translated));
  const field = new SculptPickingField(plan);
  const centre = translated.origin_m.map((v, a) => v + translated.cellSize_m[a]! * .5) as [number, number, number];
  assert.ok(Math.abs(field.distance(centre) + .0025) < 1e-7);
  assert.ok(Math.abs(field.distance([centre[0], centre[1], translated.origin_m[2]])) < 1e-7);
});

test("substrate refuses spatial capacity atomically instead of imposing an edit-history limit", () => {
  const prior = planSculptBake(undefined, packed([box(0)]));
  assert.throws(() => planSculptBake(prior, packed([{ ...box(0), maximumExclusive: [100000, 100000, 100000] }])),
    new RegExp(`capacity.*${SCULPT_PAGE_CAPACITY}`));
  assert.equal(planSculptBake(prior, prior.words).jobs.length, 0);
});

test("GPU baked sculpt pages match picking, stay local as history grows, and restore undo", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice("baked sculpt substrate", async device => {
    const arena = device.createBuffer({ size: SCULPT_ARENA_WORDS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const points = Array.from({ length: 193 }, (_, i) => [-.025 + (i % 11) * .01, -.015 + (Math.floor(i / 11) % 7) * .01, .015 + (i % 3) * .01, 0]);
    const input = device.createBuffer({ size: points.length * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const output = device.createBuffer({ size: points.length * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: output.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const field = new WebgpuSculptSubstrate(device); await field.initialize();
      device.queue.writeBuffer(input, 0, new Float32Array(points.flat()));
      const module = device.createShaderModule({ code: `
@group(0) @binding(0) var<storage,read> arena:array<u32>;
@group(0) @binding(1) var<storage,read> points:array<vec4f>;
@group(0) @binding(2) var<storage,read_write> result:array<vec4f>;
fn load(i:u32)->u32{return arena[i];}
${svoSculptCutsWGSL("load")}
@compute @workgroup_size(64) fn check(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=arrayLength(&points)){return;}let s=svoSculptCutSample(points[id.x].xyz);result[id.x]=vec4f(s.normal,s.distance);
}` });
      assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error").map(m => m.message), []);
      const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "check" } });
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [arena, input, output].map((buffer, binding) => ({ binding, resource: { buffer } })) });
      const query = async () => {
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(points.length / 64)); pass.end();
        encoder.copyBufferToBuffer(output, 0, read, 0, output.size); device.queue.submit([encoder.finish()]);
        await read.mapAsync(GPUMapMode.READ); const result = new Float32Array(read.getMappedRange()).slice(); read.unmap(); return result;
      };
      const round = { ...subcellEllipsoidPatch([3.125, -2.125, 0], [8.125, 2.875, 5], "clear"), affectsScenery: true };
      const first = packed([box(0)]), second = packed([box(0), round]);
      field.publish(arena, 0, first); const before = await query();
      field.publish(arena, 0, second); const after = await query();
      const cpu = new SculptPickingField(planSculptBake(undefined, second));
      points.forEach((p, i) => {
        const expected = cpu.sample(p.slice(0, 3) as [number, number, number]);
        assert.ok(Math.abs(after[i * 4 + 3]! - expected.distance) < 2e-7, `distance ${i}`);
        // At a lattice knot the derivative is discontinuous; float rounding
        // can choose either neighboring cell. Distances remain continuous.
      });
      assert.notDeepEqual(after, before);
      field.publish(arena, 0, first); assert.deepEqual(await query(), before, "undo restores sampled geometry exactly");
      field.publish(arena, 0, second); assert.deepEqual(await query(), after, "redo restores sampled geometry exactly");
      let history: SolidWorldVoxelPatch[] = [box(0), round];
      for (let i = 0; i < 80; i++) {
        history = [...history, box(0)]; field.publish(arena, 0, packed(history));
        assert.equal(field.lastUpdate.records, field.lastUpdate.pages, "GPU evaluates only one new operand per affected page");
      }
      assert.deepEqual(await query(), after, "repeated strokes stay baked, without cumulative sampling error");
      history = [...history, ...Array.from({ length: 4200 }, () => box(0))];
      field.publish(arena, 0, packed(history));
      assert.deepEqual(await query(), after, "bounded load batches retain all earlier page samples");
      field.publish(arena, 0, packed([...history, box(0)]));
      assert.equal(field.lastUpdate.records, field.lastUpdate.pages, "long history still appends only one operand per page");
      const translated = packSvoSculptCuts([box(0), round], { origin_m: [.003, -.007, .002], cellSize_m: [.01, .02, .005] });
      field.publish(arena, 0, translated);
      const shifted = await query(), shiftedCpu = new SculptPickingField(planSculptBake(undefined, translated));
      points.forEach((p, i) => assert.ok(Math.abs(shifted[i * 4 + 3]!
        - shiftedCpu.distance(p.slice(0, 3) as [number, number, number])) < 2e-7, `translated distance ${i}`));
      field.publish(arena, 0, packed([]));
      const empty = await query(); for (let i = 3; i < empty.length; i += 4) assert.ok(empty[i]! > 0);
    } finally { arena.destroy(); input.destroy(); output.destroy(); read.destroy(); }
  });
});

test("baked imported cavity visibility follows the substrate through overlap, undo and redo", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  const { importedSurfaceCsgWGSL } = await import("../lib/svo/features/primary-visibility/imported-surface-csg.wgsl");
  await withUniformDevice("baked imported cavity visibility", async device => {
    const arena = device.createBuffer({ size: SCULPT_ARENA_WORDS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const output = device.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const field = new WebgpuSculptSubstrate(device); await field.initialize();
      const code = `
@group(0) @binding(0) var<storage,read> cuts:array<u32>;
@group(0) @binding(1) var<storage,read_write> results:array<vec4f>;
fn drySculptCutWord(i:u32)->u32{return cuts[i];}
${svoSculptCutsWGSL("drySculptCutWord")}
struct Primitive { id:u32 }
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
${importedSurfaceCsgWGSL}
fn nearest(ro:vec3f,rd:vec3f,end:f32)->DryHit{
 var hit=missHit();for(var page=0u;page<svoSculptCutCount();page++){
  let candidate=importedCutHit(page,ro,rd,0.0,end);if(candidate.t<hit.t){hit=candidate;}}
 return hit;
}
@compute @workgroup_size(1) fn check(){
 let a=nearest(vec3f(0,0,3),vec3f(0,0,-1),10.0);results[0]=vec4f(a.t,a.normal.z,f32(a.material),f32(a.owner));
 let b=nearest(vec3f(0,0,.7),vec3f(0,0,-1),10.0);results[1]=vec4f(b.t,b.normal.z,0,0);
 results[2]=vec4f(select(0.,1.,importedSurfaceRemoved(vec3f(0,0,1),0xffffffffu)),select(0.,1.,importedSurfaceRemoved(vec3f(.8,0,.6),0xffffffffu)),0,0);
 results[3]=vec4f(nearest(vec3f(0,0,3),vec3f(0,0,-1),2.7).t,nearest(vec3f(2,0,3),vec3f(0,0,-1),10.0).t,0,0);
}`;
      const module = device.createShaderModule({ code });
      assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error").map(m => m.message), []);
      const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "check" } });
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [arena, output].map((buffer, binding) => ({ binding, resource: { buffer } })) });
      const run = async (patches: SolidWorldVoxelPatch[]) => {
        field.publish(arena, 0, packSvoSculptCuts(patches, { origin_m: [0, 0, 0], cellSize_m: [.1, .1, .1] }));
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
        encoder.copyBufferToBuffer(output, 0, read, 0, 64); device.queue.submit([encoder.finish()]);
        await read.mapAsync(GPUMapMode.READ); const values = new Float32Array(read.getMappedRange()).slice(); read.unmap(); return values;
      };
      const sphere: SolidWorldVoxelPatch = { operation: "clear", kind: "ellipsoid", affectsScenery: true, minimum: [-5, -5, 5], maximumExclusive: [5, 5, 15] };
      const deeper: SolidWorldVoxelPatch = { operation: "clear", affectsScenery: true, minimum: [-2, -2, 2], maximumExclusive: [2, 2, 8] };
      const first = await run([sphere]);
      assert.ok(Math.abs(first[0]! - 2.5) < .02); assert.ok(first[1]! > .99);
      assert.deepEqual(Array.from(first.subarray(2, 4)), [8199, 9]);
      const overlap = await run([sphere, deeper]);
      assert.ok(Math.abs(overlap[0]! - 2.8) < .001, `deeper wall ${overlap[0]}`);
      assert.ok(Math.abs(overlap[4]! - .5) < .001, "camera inside cavity sees the retained wall");
      assert.deepEqual(Array.from(overlap.subarray(8, 10)), [1, 0]);
      assert.ok(overlap[12]! > 1e19 && overlap[13]! > 1e19);
      assert.deepEqual(await run([sphere]), first);
      assert.deepEqual(await run([sphere, deeper]), overlap);
    } finally { arena.destroy(); output.destroy(); read.destroy(); }
  });
});
