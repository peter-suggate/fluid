/** The Seep solver's observer: the zone sums of tools/seep-observables.ts over
 * the dense lattice, through the solver's SeepObservationSource
 * (lib/methods/seep/seep-observation.ts). A cell's liquid is the
 * six-tetrahedron fill (lib/core/level-set-fill.wgsl.ts) of its eight
 * seepVertexPhi corners; only open cells (seepCellOpen) count, and the wall
 * zone's fill over every cell, solid included, is reported apart (wallAll)
 * for comparison with NB-FLIP, whose owners do not subtract solids.
 * One lane a cell, every cell of the box: an observation, not a step. */
import { levelSetFillWGSL } from "../lib/core/level-set-fill.wgsl";
import type { SeepObservationSource } from "../lib/methods/seep/seep-observation";
import { ZONE_WORD, ZONE_WORDS, decodeZoneWords, seepZoneWGSL, type SeepGeometry } from "./seep-observables";
import type { ZoneObserver } from "./seep-observe-nbflip";

const GROUP = 0, WORKGROUP = 256, MAX_X = 32768;

export function seepObserverWGSL(source: SeepObservationSource, g: SeepGeometry, dispatchX: number): string {
  return /* wgsl */`
${source.declarations(GROUP)}
${source.wgsl}
@group(${GROUP + 1}) @binding(0) var<storage,read_write> zoneWords:array<atomic<u32>>;
${levelSetFillWGSL}
${seepZoneWGSL(g)}
const OBS_DIMS:vec3u=vec3u(${g.nx}u,${g.ny}u,${g.nz}u);
@compute @workgroup_size(${WORKGROUP}) fn observe(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 zoneBegin(lane);
 let i=gid.x+gid.y*${dispatchX * WORKGROUP}u;
 if(i<OBS_DIMS.x*OBS_DIMS.y*OBS_DIMS.z){
  let c=vec3i(vec3u(i%OBS_DIMS.x,(i/OBS_DIMS.x)%OBS_DIMS.y,i/(OBS_DIMS.x*OBS_DIMS.y)));
  var v:array<f32,8>;var lo=1e30;var hi=-1e30;
  for(var k=0u;k<8u;k++){v[k]=seepVertexPhi(c+vec3i(vec3u(k&1u,(k>>1u)&1u,k>>2u)));lo=min(lo,v[k]);hi=max(hi,v[k]);}
  var liquid=0.0;if(hi<0.0){liquid=1.0;}else if(lo<0.0){liquid=fill(v);}
  let q=vec3u(c);
  if(q.x+1u>ZONE_WALL_LOW&&q.x<ZONE_WALL_HIGH){zoneAddWallAll(liquid);}
  if(liquid>0.0&&seepCellOpen(c)){zoneAdd(q,1u,liquid);}
 }
 if(all(gid==vec3u(0u))){
  atomicStore(&zoneWords[${ZONE_WORD.throatU}],bitcast<u32>(seepFaceU(ZONE_THROAT,0u)));atomicStore(&zoneWords[${ZONE_WORD.throatFound}],1u);
  atomicStore(&zoneWords[${ZONE_WORD.particles}],seepParticleCount());
 }
 zoneEnd(lane);
}`;
}

export async function createSeepZoneObserver(device: GPUDevice, source: SeepObservationSource, g: SeepGeometry): Promise<ZoneObserver> {
  const [nx, ny, nz] = source.dimensions;
  if (nx !== g.nx || ny !== g.ny || nz !== g.nz) throw new Error(`Seep observer: source lattice ${source.dimensions} is not the scene's ${[g.nx, g.ny, g.nz]}`);
  if (Math.abs(source.cellSize_m - g.cell_m) > 1e-12) throw new Error(`Seep observer: source cell ${source.cellSize_m} m is not the scene's ${g.cell_m} m`);
  const groups = Math.ceil(nx * ny * nz / WORKGROUP), dispatchX = Math.min(groups, MAX_X), dispatchY = Math.ceil(groups / dispatchX);
  const module = device.createShaderModule({ label: "Seep observer (Seep)", code: seepObserverWGSL(source, g, dispatchX) });
  const messages = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
  if (messages.length) throw new Error(`Seep observer WGSL: ${messages.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("; ")}`);
  const sourceLayout = device.createBindGroupLayout({ label: "Seep observation source", entries: source.buffers.map((_, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const } })) });
  const wordsLayout = device.createBindGroupLayout({ label: "Seep observer words", entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } }] });
  const pipeline = await device.createComputePipelineAsync({ label: "Seep observer (Seep)", layout: device.createPipelineLayout({ bindGroupLayouts: [sourceLayout, wordsLayout] }), compute: { module, entryPoint: "observe" } });
  const words = device.createBuffer({ label: "Seep observer words (Seep)", size: 4 * ZONE_WORDS, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const read = device.createBuffer({ label: "Seep observer readback (Seep)", size: 4 * ZONE_WORDS, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  // The source's buffers are bound as they are now; a solver that replaces one must keep it (contract: stable buffers).
  const sourceGroup = device.createBindGroup({ label: "Seep observation source", layout: sourceLayout, entries: source.buffers.map((b, binding) => ({ binding, resource: { buffer: b.buffer } })) });
  const wordsGroup = device.createBindGroup({ label: "Seep observer words", layout: wordsLayout, entries: [{ binding: 0, resource: { buffer: words } }] });
  const zero = new Uint32Array(ZONE_WORDS);
  return {
    geometry: g,
    async observe() {
      device.queue.writeBuffer(words, 0, zero);
      const encoder = device.createCommandEncoder({ label: "Seep observer (Seep)" });
      const pass = encoder.beginComputePass({ label: "Seep observer (Seep)" });
      pass.setPipeline(pipeline); pass.setBindGroup(0, sourceGroup); pass.setBindGroup(1, wordsGroup);
      pass.dispatchWorkgroups(dispatchX, dispatchY); pass.end();
      encoder.copyBufferToBuffer(words, 0, read, 0, 4 * ZONE_WORDS);
      device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ); const u = new Uint32Array(read.getMappedRange().slice(0)); read.unmap();
      const sums = decodeZoneWords(u);
      sums.particles = u[ZONE_WORD.particles]!;
      return sums;
    },
    destroy() { words.destroy(); read.destroy(); },
  };
}
