import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import test from "node:test";
import {
  materializeOctreeTerrainVertexSdf,
  octreeSolidVertexSdfShader,
  planOctreeSolidVertexSdf,
  WebGPUOctreeSolidVertexSdf,
} from "../lib/webgpu-octree-solid-vertex-sdf";
import { PassBroker } from "../lib/webgpu-pass-broker";

test("solid vertex-SDF storage scales with compact owner rows", () => {
  const plan = planOctreeSolidVertexSdf(37);
  assert.equal(plan.sampleCapacity, 37 * 8);
  assert.equal(plan.sdfBytes, 37 * 8 * 4);
  assert.equal(plan.allocatedBytes, 37 * 8 * 4 + 64 + 64 + 12);
  assert.throws(() => planOctreeSolidVertexSdf(0), /positive integer/);
});

test("terrain SDF is materialized at the eight actual vertices of each sparse owner", () => {
  const heights = new Float32Array(4 * 4).fill(1);
  const values = materializeOctreeTerrainVertexSdf([
    { cell: 0, size: 2 },
    { cell: 2 + 4 * 2 + 16 * 2, size: 2 },
  ], heights, [4, 4, 4], [0.5, 0.5, 0.5]);
  assert.equal(values.length, 16, "no finest-box vertex lattice is materialized");
  assert.deepEqual(Array.from(values.slice(0, 8)), [-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5]);
  assert.deepEqual(Array.from(values.slice(8, 16)), [0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5]);
});

test("cell-centred terrain heights are bilinearly sampled at owner vertices", () => {
  const values = materializeOctreeTerrainVertexSdf(
    [{ cell: 0, size: 2 }],
    new Float32Array([0, 2, 0, 2]),
    [2, 2, 2],
    [1, 1, 1],
  );
  // x=0 clamps to the first cell centre; x=2 clamps to the second. The four
  // upper corners differ only by owner y, never by an invented square face.
  assert.deepEqual(Array.from(values), [0, -2, 2, 0, 0, -2, 2, 0]);
});

test("terrain vertex publication rejects malformed sparse owners", () => {
  assert.throws(() => materializeOctreeTerrainVertexSdf(
    [{ cell: 1, size: 2 }], new Float32Array(16), [4, 4, 4], [1, 1, 1],
  ), /canonical octree owner/);
  assert.throws(() => materializeOctreeTerrainVertexSdf(
    [{ cell: 0, size: 2 }], new Float32Array([Number.NaN, 0, 0, 0]), [2, 2, 2], [1, 1, 1],
  ), /non-finite/);
});

test("GPU publication is generation-tagged and fail-closed", () => {
  assert.match(octreeSolidVertexSdfShader, /solid vertex SDF|publishSolidVertexSdf/i);
  assert.match(octreeSolidVertexSdfShader, /arena\.control\[4\]=generation/);
  assert.match(octreeSolidVertexSdfShader, /arena\.control\[3\]=count\*8u/);
  assert.match(octreeSolidVertexSdfShader, /arena\.control\[5\]=VALID/);
  // The retry generation stays GPU-authored. It is header word 7 of the
  // row-delta control block -- the candidate epoch the descriptor publishes --
  // so a rollback that happens after the host wrote its attempt stamp is still
  // witnessed here. The former predicate read the ACCEPTED structured control
  // and required its word 6 (the projected flag) to equal this module's VALID
  // magic, which is constant false; the arena published invalid on every step
  // of every scene with terrain or a rigid body.
  assert.match(octreeSolidVertexSdfShader,
    /fn candidateGeneration\(\)->u32\{let at=params\.rowDelta\.x\+7u;/);
  assert.match(octreeSolidVertexSdfShader, /let generation=candidateGeneration\(\);/);
  assert.match(octreeSolidVertexSdfShader, /let rollbackValid=generation!=0u&&count!=0u;/);
  assert.doesNotMatch(octreeSolidVertexSdfShader, /generation=params\.publication\.x/,
    "retry generation is GPU-derived from the candidate row delta, never a host attempt stamp");
  // The row count is the same row-delta header the structured publication
  // takes; the general compaction arena is concurrently reused for dirty-tile
  // work and is cleared to zero there.
  assert.match(octreeSolidVertexSdfShader,
    /fn candidateRowCount\(\)->u32\{let at=params\.rowDelta\.x;/);
  assert.doesNotMatch(octreeSolidVertexSdfShader, /rowCountSource\[0\]/,
    "word 0 of the compaction arena is a dirty-tile count, not a row count");
  assert.doesNotMatch(octreeSolidVertexSdfShader,
    /atomic(?:Load|Store|Add|Or|Min|Max|CompareExchange)|atomic<u32>/,
    "each row owns its eight samples and singleton publication reduces row sentinels deterministically");
  assert.doesNotMatch(octreeSolidVertexSdfShader, /dims\.x\*params\.dims\.y\*params\.dims\.z\*8u/,
    "storage must remain compact-row-scaled");
  assert.doesNotMatch(octreeSolidVertexSdfShader, /bitcast<f32>\(0x7fc00000u\)/,
    "portable WGSL must not construct a constant NaN in an unreachable texture guard");
  assert.match(octreeSolidVertexSdfShader, /if\(any\(extent<=vec2i\(0\)\)\)\{return 0\.0;\}/);
  assert.match(octreeSolidVertexSdfShader,
    /fn rigidSdf\(body:RigidBody,world:vec3f\)[\s\S]*shape==0[\s\S]*shape==1[\s\S]*shape==2/,
    "dynamic sphere, box, capsule, and cylinder SDFs share the sparse vertex authority");
  assert.match(octreeSolidVertexSdfShader,
    /for\(var body=0u;body<params\.publication\.z;body\+=1u\)[\s\S]*sdf=min\(sdf,rigidSdf/,
    "moving rigid geometry must participate without a host pose readback");
});

test("solid vertex publication fences its GPU-authored active-row dispatch", () => {
  const source = readFileSync(new URL("../lib/webgpu-octree-solid-vertex-sdf.ts", import.meta.url), "utf8");
  const encode = source.slice(source.indexOf("encode(broker: PassBroker"), source.indexOf("get source:"));
  assert.ok(encode.length > 0);
  assert.doesNotMatch(encode, /new PassBroker|beginComputePass/,
    "solid vertex observation remains within the caller's command graph");
  assert.match(encode, /preparePipeline[\s\S]*broker\.fence[\s\S]*dispatchWorkgroupsIndirect/,
    "the active row count must publish an indirect dispatch before materialization");
  assert.doesNotMatch(encode, /dispatchWorkgroups\(Math\.ceil\(this\.plan\.rowCapacity/,
    "materialization must not schedule the row-capacity domain");
});

test("Dawn validates the active-row indirect solid SDF publication", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU publication validation",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice();
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const headers = device.createBuffer({ size: 48, usage: storage });
  const rowCount = device.createBuffer({ size: 4, usage: storage });
  const rigidBodies = device.createBuffer({ size: 128, usage: storage });
  const terrain = device.createTexture({
    size: [2, 2],
    format: "r32float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeBuffer(headers, 0, new Uint32Array([0, 0, 0, 1]));
  device.queue.writeBuffer(rowCount, 0, new Uint32Array([1]));
  device.pushErrorScope("validation");
  const publication = new WebGPUOctreeSolidVertexSdf(
    device, 1, headers, rowCount, terrain, rigidBodies, 0,
  );
  const encoder = device.createCommandEncoder();
  const broker = new PassBroker(encoder);
  publication.encode(broker, {
    dimensions: [2, 2, 2],
    physicalSpacing: [1, 1, 1],
    generation: 1,
    terrainEnabled: true,
    bodyCount: 0,
  });
  device.queue.submit([broker.finish()]);
  await device.queue.onSubmittedWorkDone();
  const error = await device.popErrorScope();
  assert.equal(error, null, error?.message);
  publication.destroy();
  headers.destroy();
  rowCount.destroy();
  rigidBodies.destroy();
  terrain.destroy();
  device.destroy();
});
