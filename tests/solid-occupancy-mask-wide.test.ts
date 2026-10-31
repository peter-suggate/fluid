import assert from "node:assert/strict";
import test from "node:test";
import { SolidOccupancyMask, solidOccupancyMaskWGSL } from "../lib/core/solid-occupancy-mask";
import { SolidWorldDirectory, type SolidWorld, type SolidWorldVoxelPatch } from "../lib/core/solid-world";
import { uniformLatticeAddress } from "../lib/methods/uniform/uniform-lattice-address";
import { uniformSolidMaskPages } from "../lib/methods/uniform/uniform-mixed-solid.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";

// A closed lattice wider than the mask's 64-page flat directory on x and on z,
// with solids past page 63, across the flat directory's edge, across
// super-block edges (every fourth page), a full page, a cleared hole and the
// last page of the box. A cell is solid where the last region over it fills
// it, and every cell off the box is a wall.
const PAGES = [72, 2, 68] as const, CELLS = PAGES.map(n => 32 * n) as [number, number, number];
const box = (operation: "fill" | "clear", minimum: [number, number, number], maximumExclusive: [number, number, number]): SolidWorldVoxelPatch => ({ operation, minimum, maximumExclusive });
const REGIONS = [
  box("fill", [2000, 0, 10], [2100, 40, 50]), // pages 62..65 on x: across the flat edge and super-blocks 15|16
  box("clear", [2040, 10, 20], [2060, 30, 30]), // a hole across pages 63|64
  box("fill", [2240, 0, 2112], [2272, 32, 2144]), // page (70, 0, 66) whole: FULL, past 63 on x and z
  box("fill", [2290, 50, 2170], [2304, 64, 2176]), // the box's last page
  box("fill", [120, 30, 2040], [136, 34, 2056]), // pages 3|4 on x, 0|1 on y, 63|64 on z
  box("fill", [2048, 0, 0], [2050, 2, 2]), // page (64, 0, 0) alone: page (0, 0, 0), 64 pages off, stays open
  box("fill", [500, 60, 2100], [520, 64, 2110]), // pages 15|16 on x, page 65 on z
];
const world = (regions: readonly SolidWorldVoxelPatch[]): SolidWorld => ({ pages: [], directory: new SolidWorldDirectory([]), patches: [], regions });
function truth(x: number, y: number, z: number, regions = REGIONS): boolean {
  if (x < 0 || y < 0 || z < 0 || x >= CELLS[0] || y >= CELLS[1] || z >= CELLS[2]) return true;
  let solid = false;
  for (const r of regions) if (x >= r.minimum[0] && y >= r.minimum[1] && z >= r.minimum[2] && x < r.maximumExclusive[0] && y < r.maximumExclusive[1] && z < r.maximumExclusive[2]) solid = r.operation === "fill";
  return solid;
}
// The pages read: the halo, the flat directory's edge, super-block edges and the box's far pages.
const SAMPLE: [number, number, number][] = [];
for (const z of [-1, 0, 3, 4, 63, 64, 65, 66, 67, 68]) for (const y of [-1, 0, 1, 2]) for (const x of [-1, 0, 3, 4, 15, 16, 62, 63, 64, 65, 67, 68, 70, 71, 72]) SAMPLE.push([x, y, z]);
/** A page's cells as 1024 words, bit x of word y + 32 z. */
function pageBits(page: readonly number[], solid: (x: number, y: number, z: number) => boolean, out: Uint32Array, at: number): void {
  for (let row = 0; row < 1024; row++) {
    let bits = 0;
    for (let x = 0; x < 32; x++) if (solid(32 * page[0]! + x, 32 * page[1]! + (row & 31), 32 * page[2]! + (row >> 5))) bits |= 1 << x;
    out[at + row] = bits >>> 0;
  }
}
const wideAddress = () => {
  const air: [number, number, number][] = [];
  for (let z = 0; z < PAGES[2]; z++) for (let y = 0; y < PAGES[1]; y++) for (let x = 0; x < PAGES[0]; x++) air.push([x, y, z]);
  return uniformLatticeAddress(CELLS, false, { air });
};
function firstDifference(got: Uint32Array, want: Uint32Array): string | undefined {
  for (let i = 0; i < want.length; i++) if (got[i] !== want[i]) {
    const page = SAMPLE[i >> 10]!, row = i & 1023, bits = (got[i]! ^ want[i]!) >>> 0, x = 31 - Math.clz32(bits & -bits);
    return `cell ${32 * page[0] + x},${32 * page[1] + (row & 31)},${32 * page[2] + (row >> 5)} of page ${page.join(",")} reads ${((got[i]! >>> x) & 1) ? "solid" : "open"}`;
  }
  return undefined;
}

test("a solid mask wider than 64 pages holds every cell past page 63 and refuses an edit that needs a new super-block", () => {
  const mask = new SolidOccupancyMask(uniformSolidMaskPages(wideAddress()));
  mask.update(world(REGIONS));
  const want = new Uint32Array(1024 * SAMPLE.length), got = new Uint32Array(want.length);
  SAMPLE.forEach((page, i) => { pageBits(page, truth, want, 1024 * i); pageBits(page, (x, y, z) => mask.solid(x, y, z), got, 1024 * i); });
  assert.equal(firstDifference(got, want), undefined);
  // An edit inside the super-blocks the build gave tables to is taken; one past them is refused.
  const shrunk = REGIONS.map((r, i) => i === 2 ? box("fill", [2240, 0, 2112], [2250, 32, 2144]) : r);
  assert.ok(mask.update(world(shrunk)));
  assert.equal(mask.solid(2245, 5, 2120), true); assert.equal(mask.solid(2260, 5, 2120), false);
  assert.throws(() => mask.update(world([...shrunk, box("fill", [2180, 5, 330], [2190, 9, 340])])), /super-block/);
});

test("the GPU reads a solid mask wider than 64 pages exactly as the host holds it", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  const pages = uniformSolidMaskPages(wideAddress()), mask = new SolidOccupancyMask(pages);
  mask.update(world(REGIONS));
  const want = new Uint32Array(1024 * SAMPLE.length);
  SAMPLE.forEach((page, i) => pageBits(page, truth, want, 1024 * i));
  await withUniformDevice("wide solid mask read", async device => {
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const image = device.createBuffer({ label: "wide mask image", size: mask.words.byteLength, usage });
    const list = device.createBuffer({ label: "wide mask pages", size: 16 * SAMPLE.length, usage });
    const out = SAMPLE.map((_, i) => i).length * 4096;
    const moving = device.createBuffer({ label: "wide mask moving read", size: out, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const fixed = device.createBuffer({ label: "wide mask fixed read", size: out, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const staging = device.createBuffer({ size: 2 * out, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      device.queue.writeBuffer(image, 0, mask.words);
      device.queue.writeBuffer(list, 0, new Int32Array(SAMPLE.flatMap(p => [...p, 0])));
      const module = device.createShaderModule({ code: /* wgsl */ `
@group(0) @binding(0) var<storage,read> mask:array<u32>;
@group(0) @binding(1) var<storage,read> pages:array<vec4i>;
@group(0) @binding(2) var<storage,read_write> movingOut:array<u32>;
@group(0) @binding(3) var<storage,read_write> fixedOut:array<u32>;
// The solver's frame and world box: a closed tank at window page 0.
fn umFramePage()->vec3i{return vec3i(0);}
fn umWorldSolid(q:vec3i)->bool{return any(q<vec3i(0))||any(q>=vec3i(${PAGES.join(",")}));}
${solidOccupancyMaskWGSL("mask", "movingPage", "movingVoxel", true)}
${solidOccupancyMaskWGSL("mask", "fixedPage", "fixedVoxel", false, mask.walls)}
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) g:vec3u,@builtin(local_invocation_index) i:u32){
 let page=pages[g.x].xyz;
 for(var k=0u;k<4u;k++){
  let row=4u*i+k;var a=0u;var b=0u;
  for(var x=0u;x<32u;x++){
   let c=32*page+vec3i(i32(x),i32(row&31u),i32(row>>5u));
   if(movingVoxel(c)){a|=1u<<x;}
   if(fixedVoxel(c)){b|=1u<<x;}
  }
  movingOut[1024u*g.x+row]=a;fixedOut[1024u*g.x+row]=b;
 }
}` });
      const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [image, list, moving, fixed].map((buffer, binding) => ({ binding, resource: { buffer } })) });
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(SAMPLE.length); pass.end();
      encoder.copyBufferToBuffer(moving, 0, staging, 0, out); encoder.copyBufferToBuffer(fixed, 0, staging, out, out);
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const read = new Uint32Array(staging.getMappedRange().slice(0)); staging.unmap();
      assert.equal(firstDifference(read.subarray(0, want.length), want), undefined, "the solver's reader (umSolidVoxel)");
      assert.equal(firstDifference(read.subarray(want.length), want), undefined, "the bake's reader");
    } finally { for (const b of [image, list, moving, fixed, staging]) b.destroy(); }
  });
});
