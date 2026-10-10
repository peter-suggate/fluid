import assert from "node:assert/strict";
import test from "node:test";
import { createSolidWorld, sampleSolidWorld, type SolidWorldVoxelPatch } from "../lib/core/solid-world";
import { createWebgpuSolidWorldPageLayout, packWebgpuSolidWorldPages } from "../lib/core/webgpu-solid-world-pages";
import { solidWorldProxyWGSL } from "../lib/core/webgpu-sparse-scene-proxies";
import { createDawnRenderDevice } from "../tools/svo-dry-frame-harness";

// A two-cell shell tapering one cell in x for every four in y, across page
// boundaries and negative coordinates: the bath wall's staircase in miniature.
const RUN = 4;
const step = (y: number) => Math.floor(y / RUN);
const patches: SolidWorldVoxelPatch[] = [];
for (let y = -10; y < 30; y += 1) patches.push({ operation: "fill", materialId: 7,
  minimum: [step(y) - 2, y, -12], maximumExclusive: [step(y), y + 1, 22] });
const world = createSolidWorld(patches);
const origin = [-0.3, 0.1, -0.2] as const, cell = 0.025;
const fraction = (x: number, y: number, z: number) => sampleSolidWorld(world, [x, y, z]).solidFraction;

function referenceField(point: readonly number[]): number {
  const p = point.map((value, axis) => (value - origin[axis]!) / cell - 0.5);
  const base = p.map(Math.floor), weights = p.map((value, axis) => {
    const t = value - base[axis]!, u = 1 - t;
    return [u * u * u / 6, (3 * t * t * t - 6 * t * t + 4) / 6, (-3 * t * t * t + 3 * t * t + 3 * t + 1) / 6, t * t * t / 6];
  });
  let solid = 0;
  for (let k = 0; k < 4; k += 1) for (let j = 0; j < 4; j += 1) for (let i = 0; i < 4; i += 1) {
    solid += weights[0]![i]! * weights[1]![j]! * weights[2]![k]! * fraction(base[0]! + i - 1, base[1]! + j - 1, base[2]! + k - 1);
  }
  return solid <= 0 ? 1e20 : (0.5 - solid) * cell * 4 / 3;
}

function referenceNormal(point: readonly number[]): number[] {
  const p = point.map((value, axis) => (value - origin[axis]!) / cell - 0.5);
  const centre = p.map(Math.round), away = [0, 0, 0];
  for (let k = -4; k <= 4; k += 1) for (let j = -4; j <= 4; j += 1) for (let i = -4; i <= 4; i += 1) {
    const q = [centre[0]! + i, centre[1]! + j, centre[2]! + k], solid = fraction(q[0]!, q[1]!, q[2]!);
    if (solid <= 0) continue;
    const d = p.map((value, axis) => value - q[axis]!);
    const weight = solid * Math.max(Math.exp(-0.125 * (d[0]! ** 2 + d[1]! ** 2 + d[2]! ** 2)) - 0.0795595, 0);
    d.forEach((value, axis) => { away[axis]! += weight * value; });
  }
  const length = Math.hypot(...away);
  return length < 1e-6 ? [0, 0, 0] : away.map(value => value / length);
}

(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("GPU page solids reconstruct a continuous field and a staircase's slope", async () => {
  // Surface points on the shell's +x side: riser faces and ledge tops.
  const points: number[][] = [];
  for (let y = -4; y < 24; y += 1) for (const z of [-3.5, 4.5, 8.5]) {
    points.push([step(y), y + 0.5, z].map((value, axis) => origin[axis]! + value * cell));
    if ((y + 1) % RUN === 0) points.push([step(y) + 0.5, y + 1, z].map((value, axis) => origin[axis]! + value * cell));
  }
  const surfaceCount = points.length;
  let seed = 0x2545f491;
  const random = () => { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return (seed >>> 0) / 2 ** 32; };
  for (let index = 0; index < 1500; index += 1) {
    points.push([-8 + 16 * random(), -14 + 48 * random(), -16 + 42 * random()].map((value, axis) => origin[axis]! + value * cell));
  }

  // The true wall rises RUN cells for each cell it leans.
  const slope = [RUN, -1, 0].map(value => value / Math.hypot(RUN, 1));
  const angle = (normal: readonly number[]) => Math.acos(Math.min(1, normal[0]! * slope[0]! + normal[1]! * slope[1]! + normal[2]! * slope[2]!)) * 180 / Math.PI;
  for (let index = 0; index < surfaceCount; index += 1) {
    assert.ok(Math.abs(referenceField(points[index]!)) < 0.3 * cell, "the smoothed surface stays within a fraction of a cell of the voxel faces");
    assert.ok(angle(referenceNormal(points[index]!)) < 12, `riser and ledge shade as one slope (${angle(referenceNormal(points[index]!)).toFixed(1)} degrees)`);
  }

  const layout = createWebgpuSolidWorldPageLayout({ baseWords: 0, authoredPageCount: world.pages.length, includesMaterial: true });
  const image = packWebgpuSolidWorldPages(layout, world, [0, 0, 0], { origin_m: origin, cellSize_m: [cell, cell, cell] });
  let device: GPUDevice | undefined;
  try {
    device = (await createDawnRenderDevice()).device;
    const module = device.createShaderModule({ code: /* wgsl */ `
@group(0) @binding(0) var<storage,read_write> maintenance:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read> points:array<vec4f>;
@group(0) @binding(2) var<storage,read_write> results:array<vec4f>;
const INVALID_INDEX:u32=0xffffffffu;
${solidWorldProxyWGSL(layout)}
@compute @workgroup_size(64) fn evaluate(@builtin(global_invocation_id) id:vec3u){
  if(id.x>=arrayLength(&points)){return;}
  let point=points[id.x].xyz;
  results[2u*id.x]=vec4f(swSmoothNormal(point),swSmoothField(point));
  results[2u*id.x+1u]=vec4f(f32(swNearestMaterial(point)),sampleSolidWorld(point,vec3f(${cell})).fraction,0.0,0.0);
}` });
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(message => message.type === "error")
      .map(message => `${message.lineNum}:${message.linePos} ${message.message}`), []);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "evaluate" } });
    const buffer = (data: Uint32Array | Float32Array, usage: number) => {
      const created = device!.createBuffer({ size: data.byteLength, usage: usage | GPUBufferUsage.COPY_DST });
      device!.queue.writeBuffer(created, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
      return created;
    };
    const arena = buffer(image, GPUBufferUsage.STORAGE);
    const input = buffer(new Float32Array(points.flatMap(point => [...point, 0])), GPUBufferUsage.STORAGE);
    const output = device.createBuffer({ size: points.length * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const receipt = device.createBuffer({ size: points.length * 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
      entries: [arena, input, output].map((resource, binding) => ({ binding, resource: { buffer: resource } })) }));
    pass.dispatchWorkgroups(Math.ceil(points.length / 64));
    pass.end();
    encoder.copyBufferToBuffer(output, 0, receipt, 0, points.length * 32);
    device.queue.submit([encoder.finish()]);
    await receipt.mapAsync(GPUMapMode.READ);
    const results = new Float32Array(receipt.getMappedRange().slice(0));
    receipt.unmap();

    let finite = 0, directed = 0, filled = 0;
    points.forEach((point, index) => {
      const [nx, ny, nz, field, material, solid] = results.subarray(8 * index, 8 * index + 6);
      const expectedField = referenceField(point), expectedNormal = referenceNormal(point);
      const where = `point ${index} (${point.map(value => value.toFixed(4)).join(", ")})`;
      if (expectedField > 1e12) assert.ok(field! > 1e12, `${where} is outside every stencil`);
      else { finite += 1; assert.ok(Math.abs(field! - expectedField) < 2e-4 * cell, `${where} field ${field} != ${expectedField}`); }
      const magnitude = Math.hypot(nx!, ny!, nz!);
      if (Math.hypot(...expectedNormal) === 0) assert.ok(magnitude < 1e-3, `${where} has no solid in reach`);
      // A direction nearly cancelled between the shell's two faces is not compared.
      else if (Math.abs(expectedField) < 0.6 * cell) {
        directed += 1;
        assert.ok(nx! * expectedNormal[0]! + ny! * expectedNormal[1]! + nz! * expectedNormal[2]! > 0.999, `${where} normal ${[nx, ny, nz]} != ${expectedNormal}`);
      }
      // Every cell the smoothing can fill has a voxel's material beside it.
      if (expectedField < 0 && solid === 0) { filled += 1; assert.equal(material, 7, `${where} borrows its neighbour's material`); }
    });
    assert.ok(finite > 300 && directed > 150, `the sample covers the surface band (${finite} finite, ${directed} directed, ${filled} filled)`);
  } finally { device?.destroy(); }
});
