/** Narrow-band FLIP's box observer (tools/nbflip-observables.ts): the seep
 * observer's traversal (tools/seep-observe-nbflip.ts) over every mixed owner,
 * h and 4h, with the same measure. An owner's liquid is the six-tetrahedron
 * fill of its eight corner phi (UniformMixedDiagnostics' canonical load) times
 * its cells; solids are not subtracted. Its velocity is, an axis, the mean of
 * its two faces, each the area-weighted mean of its MAC patches (as
 * UniformMixedDiagnostics reads them). Every open BULK16 region adds 16^3
 * liquid cells at the mean of its two face means an axis (read on the host).
 *
 * Reads only after the caller fenced the frame. Issues no solver call. */
import type { SceneDescription } from "../lib/core/model";
import { UNIFORM_DETAIL_CANONICAL_LOAD as C } from "../lib/core/uniform-detail-abi";
import { levelSetFillWGSL } from "../lib/core/level-set-fill.wgsl";
import { uniformDetailBindLayout, uniformDetailGroup, uniformDetailModule, uniformDetailPipeline } from "../lib/methods/uniform/uniform-detail-fields";
import { UNIFORM_MIXED_COUNTED, uniformMixedCountedEntriesWGSL, uniformMixedTopologyWGSL } from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import type { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { UNIFORM_WINDOW_HEADER } from "../lib/methods/uniform/uniform-window-address";
import { BULK16_STATUS, type UniformBulkPages } from "../lib/methods/uniform/uniform-bulk-pages";
import { BULK16_PAGE, BULK16_PAGE_WORDS } from "../lib/methods/uniform/uniform-bulk-pages-reference";
import type { GPUSolverInstance } from "../lib/core/method-contract";
import { boxAddRegion, boxObserverWGSL, boxWords, decodeBoxWords, type BoxGeometry, type BoxSums } from "./nbflip-observables";

interface NBInternals {
  device: GPUDevice;
  vertexPhiField: GPUTexture; velocityA: GPUTexture;
  narrowBandFlipInfo?: { particles?: number };
  mixedFrame: { ownership: UniformMixedOwnership; bulkPages?: UniformBulkPages };
}

export interface BoxObserver {
  readonly geometry: BoxGeometry;
  /** The current, fenced state's box sums; `bulkRegions` the open BULK16 regions added. */
  observe(): Promise<BoxSums & { bulkRegions: number }>;
  destroy(): void;
}

export async function createNBFlipBoxObserver(device: GPUDevice, solver: GPUSolverInstance, _scene: SceneDescription, g: BoxGeometry): Promise<BoxObserver> {
  const nb = solver as unknown as NBInternals;
  const frame = nb.mixedFrame;
  if (!frame?.ownership || !nb.vertexPhiField || !nb.velocityA) throw new Error("NB-FLIP box observer: the solver has no mixed-ownership frame");
  const WORDS = boxWords(g);
  const layout = uniformDetailBindLayout(device, { label: "Box observer (NB-FLIP)", entries: [
    ...[0, 1].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" as const, viewDimension: "3d" as const } })),
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } }] });
  const code = uniformMixedCountedEntriesWGSL(uniformMixedTopologyWGSL(frame.ownership.capacity, 0) + /* wgsl */`
@group(1) @binding(0) var phi:texture_3d<f32>;
@group(1) @binding(1) var velocity:texture_3d<f32>;
@group(1) @binding(2) var<storage,read_write> boxWords:array<atomic<u32>>;
fn corner(p:vec3u)->f32{return ${C}textureLoad(phi,vec3i(p),0).x;}
${levelSetFillWGSL}
${boxObserverWGSL(g, 64)}
fn ownerFill(q:vec3u,w:u32)->f32{
 var c:array<f32,8>;
 for(var k=0u;k<8u;k++){c[k]=corner(q+umCorner(k,2u)*w);}
 var lo=1e30;var hi=-1e30;for(var k=0u;k<8u;k++){lo=min(lo,c[k]);hi=max(hi,c[k]);}
 let cells=f32(w*w*w);
 if(hi<0.0){return cells;}if(lo>=0.0){return 0.0;}
 return fill(c)*cells;
}
// An axis's face mean: its patches' velocities weighted by area (UniformMixedDiagnostics).
fn faceMean(owner:UMOwner,axis:u32,sign:i32)->f32{
 let first=umFace(owner,axis,sign,0u);var sum=0.0;
 for(var part=0u;part<first.count;part++){let face=umFace(owner,axis,sign,part);sum+=${C}textureLoad(velocity,face.anchor,0)[axis]*f32(face.width*face.width);}
 return sum/f32(owner.width*owner.width);
}
fn boxOrigin()->vec3i{return 32*(umWorldLow()-umFramePage());}
@compute @workgroup_size(64) fn observe(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 boxBegin(lane);
 let owner=umAllOwner(gid);
 if(owner.width!=0u){
  let window=umOrigin(owner);let q=vec3u(vec3i(window)-boxOrigin());let w=owner.width;
  let liquid=ownerFill(window,w);
  var u=vec3f(0.0);
  if(liquid>0.0){for(var a=0u;a<3u;a++){u[a]=0.5*(faceMean(owner,a,-1)+faceMean(owner,a,1));}}
  boxAdd(q,w,liquid,u);
 }
 boxEnd(lane);
}`, ["observe"]);
  const module = uniformDetailModule(nb.device, { label: "Box observer (NB-FLIP)", code });
  const messages = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
  if (messages.length) throw new Error(`NB-FLIP box observer WGSL: ${messages.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("; ")}`);
  const pipeline = await uniformDetailPipeline(device, frame.ownership, {
    layout: device.createPipelineLayout({ bindGroupLayouts: [frame.ownership.bindLayout, layout] }),
    compute: { module, entryPoint: "observe", constants: { umDispatchX: frame.ownership.dispatchX, umCountedJobs: UNIFORM_MIXED_COUNTED.all } } });
  const words = device.createBuffer({ label: "Box observer words (NB-FLIP)", size: 4 * WORDS, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const read = device.createBuffer({ label: "Box observer readback (NB-FLIP)", size: 4 * WORDS, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const group = uniformDetailGroup(device, { layout, entries: [{ binding: 0, resource: nb.vertexPhiField }, { binding: 1, resource: nb.velocityA }, { binding: 2, resource: { buffer: words } }] });
  const zero = new Uint32Array(WORDS);
  const seed = frame.ownership.capacity.address.seed, at = seed.layout.tail.window;
  const worldLow = [0, 1, 2].map(a => seed.tail[at + UNIFORM_WINDOW_HEADER.solidLow + a]! | 0);
  const bulk = frame.bulkPages;
  if (bulk && worldLow.some(v => v === -0x80000000)) throw new Error("NB-FLIP box observer: BULK16 pages in a box without a solid low bound");
  /** Open BULK16 regions (bit r = x + 2y + 4z of a page's 2^3), their low corners in box cells and face-mean velocities. */
  const readBulk16 = async () => {
    const regions: { low: [number, number, number]; u: [number, number, number] }[] = [];
    if (!bulk) return regions;
    const { buffer, offset = 0, size } = bulk.binding as GPUBufferBinding & { size: number };
    const copy = device.createBuffer({ label: "Box observer bulk16 readback", size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const e = device.createCommandEncoder(); e.copyBufferToBuffer(buffer, offset, copy, 0, size); device.queue.submit([e.finish()]);
      await copy.mapAsync(GPUMapMode.READ); const w = new Uint32Array(copy.getMappedRange().slice(0)); copy.unmap();
      const f = new Float32Array(w.buffer), L = bulk.layout, n = w[BULK16_STATUS.live]!;
      for (const slot of w.subarray(L.live, L.live + n)) {
        const base = L.records + BULK16_PAGE_WORDS * slot, open = w[base + BULK16_PAGE.open]!;
        const page = [0, 1, 2].map(a => (w[base + BULK16_PAGE.page + a]! | 0) - worldLow[a]!);
        for (let r = 0; r < 8; r++) {
          if (!((open >> r) & 1)) continue;
          const c = [r & 1, (r >> 1) & 1, r >> 2];
          const u = [0, 1, 2].map(a => {
            const t = c[(a + 1) % 3]! + 2 * c[(a + 2) % 3]!;
            return 0.5 * (f[base + BULK16_PAGE.velocity + 12 * a + 4 * c[a]! + t]! + f[base + BULK16_PAGE.velocity + 12 * a + 4 * (c[a]! + 1) + t]!);
          }) as [number, number, number];
          regions.push({ low: [32 * page[0]! + 16 * c[0]!, 32 * page[1]! + 16 * c[1]!, 32 * page[2]! + 16 * c[2]!], u });
        }
      }
      return regions;
    } finally { copy.destroy(); }
  };
  return {
    geometry: g,
    async observe() {
      device.queue.writeBuffer(words, 0, zero);
      const encoder = device.createCommandEncoder({ label: "Box observer (NB-FLIP)" });
      const pass = encoder.beginComputePass({ label: "Box observer (NB-FLIP)" });
      pass.setBindGroup(0, frame.ownership.bindGroup); pass.setBindGroup(1, group.group);
      frame.ownership.dispatchAllCounted(pass, pipeline); pass.end();
      encoder.copyBufferToBuffer(words, 0, read, 0, 4 * WORDS);
      device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ); const u = new Uint32Array(read.getMappedRange().slice(0)); read.unmap();
      const sums = decodeBoxWords(g, u);
      const regions = await readBulk16();
      for (const r of regions) {
        if (r.low.some((v, a) => v < 0 || v + 16 > [g.nx, g.ny, g.nz][a]!)) throw new Error(`NB-FLIP box observer: a BULK16 region at ${r.low} lies outside the box`);
        boxAddRegion(g, sums, r.low, r.u);
      }
      sums.particles = nb.narrowBandFlipInfo?.particles ?? 0;
      return { ...sums, bulkRegions: regions.length };
    },
    destroy() { words.destroy(); read.destroy(); },
  };
}
