import assert from "node:assert/strict";
import { uniformMixedCertifiedEntriesWGSL } from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import { narrowBandLocalNearestWGSL, narrowBandNearestEntries, narrowBandNearestWGSL } from "../lib/methods/uniform/uniform-narrow-band-nearest.wgsl";

export const globalNearestEntries = narrowBandNearestEntries;

/** Late diagnostic compilation follows the enclosing module's integer-root
 * ABI, whether its scene constants are values or runtime accessors. */
function adaptRoots(extra: string, code: string): string {
  for (const match of code.matchAll(/\bfn\s+([A-Z][A-Z_0-9]*)\s*\(/g)) {
    extra = extra.replace(new RegExp(`\\b${match[1]}\\b`, "g"), `${match[1]}()`);
  }
  return extra;
}
export function globalNearestSearchWGSL(code: string): string {
  assert.ok(code.includes("fn depthNearestJob("));
  assert.ok(!code.includes("fn depthNearestXJob("));
  const extra = uniformMixedCertifiedEntriesWGSL(narrowBandNearestWGSL, globalNearestEntries, "(atomicLoad(&bins[NB_BAND+1u])+3u)/4u");
  return code + adaptRoots(extra, code);
}
export function localNearestSearchWGSL(code: string): string {
  assert.ok(code.includes("fn depthNearestZJob("));
  assert.ok(!code.includes("fn depthNearestJob("));
  const extra = uniformMixedCertifiedEntriesWGSL(narrowBandLocalNearestWGSL, ["depthNearest"], "atomicLoad(&bins[NB_BAND+1u])");
  return code + adaptRoots(extra, code);
}

/** A few synthetic crossing sets cover empty reach, all domain faces,
 * positive/negative tie breaks, packed-word seams and a dense crossing set. */
export function globalNearestFixtures(dimensions: readonly number[]): { name: string; masks: Uint32Array; reach: Uint32Array; search: Uint32Array }[] {
  const [nx, ny, nz] = dimensions as readonly [number, number, number];
  assert.ok(dimensions.every(n => Number.isInteger(n) && n >= 12 && n % 4 === 0));
  const tx = nx / 4, ty = ny / 4, tz = nz / 4, tiles = tx * ty * tz;
  const fixtures: ReturnType<typeof globalNearestFixtures> = [];
  const add = (name: string, points: readonly (readonly number[])[], dense = false) => {
    const masks = new Uint32Array(2 * tiles), reach = new Uint32Array(tiles);
    if (dense) { masks.fill(0xa5a5a5a5); reach.fill(3); }
    else for (const [x, y, z] of points as readonly (readonly [number, number, number])[]) {
      assert.ok(x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz);
      const cx = x >> 2, cy = y >> 2, cz = z >> 2, tile = cx + tx * (cy + ty * cz), bit = (x & 3) + 4 * (y & 3) + 16 * (z & 3);
      masks[2 * tile + (bit >>> 5)]! |= 1 << (bit & 31);
      for (let dz = -2; dz <= 2; dz++) for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
        const qx = cx + dx, qy = cy + dy, qz = cz + dz;
        if (qx >= 0 && qx < tx && qy >= 0 && qy < ty && qz >= 0 && qz < tz) reach[qx + tx * (qy + ty * qz)] = 3;
      }
    }
    const selected: number[] = []; for (let tile = 0; tile < tiles; tile++) if (reach[tile]! & 1) selected.push(tile);
    const search = Uint32Array.from(selected);
    fixtures.push({ name, masks, reach, search });
  };
  add("empty", []);
  add("domain-corners", Array.from({ length: 8 }, (_, k) => [k & 1 ? nx - 1 : 0, k & 2 ? ny - 1 : 0, k & 4 ? nz - 1 : 0]));
  const c = dimensions.map(n => 4 * Math.floor(n / 8));
  add("symmetric-ties", [0, 1, 2].flatMap(axis => [-2, 2].map(delta => c.map((n, a) => n + (a === axis ? delta : 0)))));
  add("packed-seams", [[3, 3, 1], [0, 0, 2], [4, 3, 1], [3, 4, 2], [7, 7, 3], [8, 7, 4], [nx - 1, ny - 2, nz - 1]]);
  add("dense-checker", [], true);
  return fixtures;
}

/** Exercise the same integer kernels on a small unequal-stride lattice.
 * Scratch starts poisoned, so an omitted reach guard cannot hide behind zero.
 */
export async function checkGlobalNearestStrides(device: GPUDevice, membership: string): Promise<unknown[]> {
  const dimensions = [12, 16, 20] as const, cells = 12 * 16 * 20, tiles = cells / 64, vertices = 13 * 17 * 21;
  const depthA = tiles, depthB = depthA + vertices, band = depthB + cells, mask = band + 4, reach = mask + 2 * tiles, search = reach + 2 * tiles;
  const words = band + 4 + 7 * tiles;
  const headerStart = membership.indexOf("const NB_DEPTH_A"), headerEnd = membership.indexOf("// A 4h tile among");
  const kernelStart = membership.indexOf("fn nbBandRow("), kernelEnd = membership.indexOf("fn bulkDepth(");
  assert.ok(headerStart >= 0 && headerEnd > headerStart && kernelStart > headerEnd && kernelEnd > kernelStart);
  let source = /* wgsl */ `
const UM_D:vec3u=vec3u(12u,16u,20u);const UM_T:vec3u=UM_D/4u;const NB_CELLS:u32=12u*16u*20u;const NB_SURFACE_TILES:u32=0u;
@group(0) @binding(0) var<storage,read_write> bins:array<atomic<u32>>;
fn umCorner(i:u32,n:u32)->vec3u{return vec3u(i%n,(i/n)%n,i/(n*n));}
fn umTileCoord(i:u32)->vec3u{return vec3u(i%UM_T.x,(i/UM_T.x)%UM_T.y,i/(UM_T.x*UM_T.y));}
fn cellIndex(p:vec3i)->u32{return u32(p.x)+UM_D.x*(u32(p.y)+UM_D.y*u32(p.z));}
` + membership.slice(headerStart, headerEnd) + membership.slice(kernelStart, kernelEnd);
  source = globalNearestSearchWGSL(uniformMixedCertifiedEntriesWGSL(source, ["depthNearest"], "atomicLoad(&bins[NB_BAND+1u])"));
  const module = device.createShaderModule({ label: "Unequal-stride nearest fixture", code: source });
  const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
  assert.deepEqual(errors, [], errors.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("\n"));
  const bindings = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }] });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bindings] });
  const pipelines = await Promise.all(["depthNearest", ...globalNearestEntries].map(entryPoint => device.createComputePipelineAsync({ layout, compute: { module, entryPoint } })));
  const bins = device.createBuffer({ size: 4 * words, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const read = device.createBuffer({ size: 4 * cells, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const group = device.createBindGroup({ layout: bindings, entries: [{ binding: 0, resource: { buffer: bins } }] });
  const results: unknown[] = [];
  try {
    for (const fixture of globalNearestFixtures(dimensions)) {
      const data = new Uint32Array(words); data.fill(0xf00df00d, depthA, depthB); data.fill(0xdeadbeef, depthB, band);
      data[band + 1] = fixture.search.length; data.set(fixture.masks, mask); data.set(fixture.reach, reach); data.set(fixture.search, search);
      const capture = async (baseline: boolean) => {
        device.queue.writeBuffer(bins, 0, data);
        const e = device.createCommandEncoder(), pass = e.beginComputePass(); pass.setBindGroup(0, group);
        for (const pipeline of baseline ? [pipelines[0]!] : pipelines.slice(1)) { pass.setPipeline(pipeline); pass.dispatchWorkgroups(4); }
        pass.end(); e.copyBufferToBuffer(bins, 4 * depthB, read, 0, 4 * cells); device.queue.submit([e.finish()]);
        await read.mapAsync(GPUMapMode.READ); const output = new Uint32Array(read.getMappedRange().slice(0)); read.unmap(); return output;
      };
      const a = await capture(true), b = await capture(false); let mismatches = 0; const first: unknown[] = [];
      for (let i = 0; i < cells; i++) if (a[i] !== b[i]) { mismatches++; if (first.length < 8) first.push({ word: i, baseline: a[i], candidate: b[i] }); }
      results.push({ kind: `nearest-unequal-${fixture.name}`, dimensions, words: cells, searchTiles: fixture.search.length, mismatches, first });
    }
    return results;
  } finally { bins.destroy(); read.destroy(); }
}
