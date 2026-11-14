/** Narrow-band FLIP's seep observer: the zone sums of tools/seep-observables.ts
 * over every mixed owner (h and 4h) and the BULK16 pages, factored from
 * tools/probe-tier-budget.ts's zone probe with the same measure: an owner's
 * liquid is the six-tetrahedron fill of its eight corner phi (loaded as
 * UniformMixedDiagnostics' canonical form) times its cells; solids are not
 * subtracted (the wall zone holds them); an open BULK16 region is 16^3 liquid
 * cells. The baseline (artifacts/seep-rewrite/baseline) shows every load form
 * of the probe agreeing to 0.004 cells, so the canonical form alone is read.
 *
 * Reads only after the caller fenced the frame (awaitFrameCompletion). Issues
 * no solver call that changes what the next step encodes: readStats (which
 * asks the next step for NB-FLIP's volume pass) is the caller's choice. */
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
import { ZONE_WORD, ZONE_WORDS, decodeZoneWords, seepGeometry, seepZoneWGSL, type SeepGeometry, type ZoneSums } from "./seep-observables";

interface NBInternals {
  device: GPUDevice;
  vertexPhiField: GPUTexture; velocityA: GPUTexture;
  narrowBandFlipInfo?: { particles?: number };
  mixedFrame: { ownership: UniformMixedOwnership; bulkPages?: UniformBulkPages };
}

export interface ZoneObserver {
  readonly geometry: SeepGeometry;
  /** The current, fenced state's zone sums. */
  observe(): Promise<ZoneSums>;
  destroy(): void;
}

export async function createNBFlipZoneObserver(device: GPUDevice, solver: GPUSolverInstance, scene: SceneDescription): Promise<ZoneObserver> {
  const g = seepGeometry(scene);
  const nb = solver as unknown as NBInternals;
  const frame = nb.mixedFrame;
  if (!frame?.ownership || !nb.vertexPhiField || !nb.velocityA) throw new Error("NB-FLIP observer: the solver has no mixed-ownership frame");
  const layout = uniformDetailBindLayout(device, { label: "Seep observer (NB-FLIP)", entries: [
    ...[0, 1].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" as const, viewDimension: "3d" as const } })),
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } }] });
  const code = uniformMixedCountedEntriesWGSL(uniformMixedTopologyWGSL(frame.ownership.capacity, 0) + /* wgsl */`
@group(1) @binding(0) var phi:texture_3d<f32>;
@group(1) @binding(1) var velocity:texture_3d<f32>;
@group(1) @binding(2) var<storage,read_write> zoneWords:array<atomic<u32>>;
fn corner(p:vec3u,w:u32)->f32{return ${C}textureLoad(phi,vec3i(p),0).x;}
fn faceU(a:vec3i)->f32{return ${C}textureLoad(velocity,a,0).x;}
${levelSetFillWGSL}
${seepZoneWGSL(g)}
fn ownerFill(q:vec3u,w:u32)->f32{
 var c:array<f32,8>;
 for(var k=0u;k<8u;k++){c[k]=corner(q+umCorner(k,2u)*w,w);}
 var lo=1e30;var hi=-1e30;for(var k=0u;k<8u;k++){lo=min(lo,c[k]);hi=max(hi,c[k]);}
 let cells=f32(w*w*w);
 if(hi<0.0){return cells;}if(lo>=0.0){return 0.0;}
 return fill(c)*cells;
}
// Owner origins are window cells; the box's low cell is its world page less the frame's.
fn boxOrigin()->vec3i{return 32*(umWorldLow()-umFramePage());}
@compute @workgroup_size(64) fn observe(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 zoneBegin(lane);
 let owner=umAllOwner(gid);
 if(owner.width!=0u){
  let window=umOrigin(owner);let q=vec3u(vec3i(window)-boxOrigin());let w=owner.width;
  zoneAdd(q,w,ownerFill(window,w));
  if(w==1u&&all(vec3i(q)==ZONE_THROAT)){
   atomicStore(&zoneWords[${ZONE_WORD.throatU}],bitcast<u32>(faceU(umFace(owner,0u,1,0u).anchor)));atomicStore(&zoneWords[${ZONE_WORD.throatFound}],1u);
  }
 }
 zoneEnd(lane);
}`, ["observe"]);
  // The detail rewrite finds field storage by the device that made it (the solver's scoped device).
  const module = uniformDetailModule(nb.device, { label: "Seep observer (NB-FLIP)", code });
  const messages = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
  if (messages.length) throw new Error(`NB-FLIP observer WGSL: ${messages.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("; ")}`);
  const pipeline = await uniformDetailPipeline(device, frame.ownership, {
    layout: device.createPipelineLayout({ bindGroupLayouts: [frame.ownership.bindLayout, layout] }),
    compute: { module, entryPoint: "observe", constants: { umDispatchX: frame.ownership.dispatchX, umCountedJobs: UNIFORM_MIXED_COUNTED.all } } });
  const words = device.createBuffer({ label: "Seep observer words (NB-FLIP)", size: 4 * ZONE_WORDS, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const read = device.createBuffer({ label: "Seep observer readback (NB-FLIP)", size: 4 * ZONE_WORDS, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const group = uniformDetailGroup(device, { layout, entries: [{ binding: 0, resource: nb.vertexPhiField }, { binding: 1, resource: nb.velocityA }, { binding: 2, resource: { buffer: words } }] });
  const zero = new Uint32Array(ZONE_WORDS);
  const seed = frame.ownership.capacity.address.seed, at = seed.layout.tail.window;
  const bulk = frame.bulkPages;
  /** BULK16 pages hold no owners: each open region (bit r = x + 2y + 4z) is 16^3 liquid cells. */
  const readBulk16 = async () => {
    if (!bulk) return [] as number[];
    const { buffer, offset = 0, size } = bulk.binding as GPUBufferBinding & { size: number };
    const copy = device.createBuffer({ label: "Seep observer bulk16 readback", size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const e = device.createCommandEncoder(); e.copyBufferToBuffer(buffer, offset, copy, 0, size); device.queue.submit([e.finish()]);
      await copy.mapAsync(GPUMapMode.READ); const w = new Uint32Array(copy.getMappedRange().slice(0)); copy.unmap();
      const L = bulk.layout, n = w[BULK16_STATUS.live]!, frameX = seed.tail[at + UNIFORM_WINDOW_HEADER.frame]! | 0;
      const lows: number[] = [];
      for (const slot of w.subarray(L.live, L.live + n)) {
        const x = (w[L.records + BULK16_PAGE_WORDS * slot + BULK16_PAGE.page]! | 0) - frameX, open = w[L.records + BULK16_PAGE_WORDS * slot + BULK16_PAGE.open]!;
        for (let r = 0; r < 8; r++) if ((open >> r) & 1) lows.push(32 * x + 16 * (r & 1));
      }
      return lows;
    } finally { copy.destroy(); }
  };
  return {
    geometry: g,
    async observe() {
      device.queue.writeBuffer(words, 0, zero);
      const encoder = device.createCommandEncoder({ label: "Seep observer (NB-FLIP)" });
      const pass = encoder.beginComputePass({ label: "Seep observer (NB-FLIP)" });
      pass.setBindGroup(0, frame.ownership.bindGroup); pass.setBindGroup(1, group.group);
      frame.ownership.dispatchAllCounted(pass, pipeline); pass.end();
      encoder.copyBufferToBuffer(words, 0, read, 0, 4 * ZONE_WORDS);
      device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ); const u = new Uint32Array(read.getMappedRange().slice(0)); read.unmap();
      const sums = decodeZoneWords(u);
      sums.wallAll = undefined;
      for (const low of await readBulk16()) {
        // Only the deep reservoir coarsens to 16h; a region at or right of the
        // wall would need its y (floor band, wet floor) and bin, which this
        // observer does not decode.
        if (low + 16 > g.wall.low) throw new Error(`NB-FLIP observer: a BULK16 region at x ${low} reaches the wall; its zone split is not decoded`);
        sums.total += 4096; sums.left += 4096;
      }
      sums.particles = nb.narrowBandFlipInfo?.particles ?? 0;
      return sums;
    },
    destroy() { words.destroy(); read.destroy(); },
  };
}
