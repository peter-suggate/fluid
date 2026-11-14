/** Cell dumps and the S2g audit reduction (lane H; docs/plans/seep-rewrite/progress-lane-h.md).
 *
 * Diagnostics outside the gated schema of tools/seep-observables.ts. Each dump
 * lists every owner (NB-FLIP: h and 4h mixed owners; Seep: dense cells) at or
 * right of x = wall.low holding liquid, with the same six-tetrahedron fill
 * (lib/core/level-set-fill.wgsl.ts) of the same corner phi the observers read,
 * and the owner's mean x-face velocity. `auditCells` reduces a dump in Float64:
 *  - the wall zone split into open throat cells (the scene's clear patch) and
 *    solid cells (NB-FLIP's owners do not subtract solids; Seep's solid cells
 *    carry extrapolated phi), plus any 4h owner overlapping the wall zone;
 *  - the jet trajectory: 8-cell x slices of the right chamber, each with its
 *    liquid at y >= 8 (above the floor band), that liquid's weighted mean y
 *    (cell centres) and mean u (x faces), beside the ballistic parabola from
 *    the throat exit velocity.
 * A dump is an observation, not a step: one lane an owner, after the caller
 * fenced the frame. */
import type { SceneDescription } from "../lib/core/model";
import { UNIFORM_DETAIL_CANONICAL_LOAD as C } from "../lib/core/uniform-detail-abi";
import { levelSetFillWGSL } from "../lib/core/level-set-fill.wgsl";
import type { GPUSolverInstance } from "../lib/core/method-contract";
import { uniformDetailBindLayout, uniformDetailGroup, uniformDetailModule, uniformDetailPipeline } from "../lib/methods/uniform/uniform-detail-fields";
import { UNIFORM_MIXED_COUNTED, uniformMixedCountedEntriesWGSL, uniformMixedTopologyWGSL } from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import type { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import type { SeepObservationSource } from "../lib/methods/seep/seep-observation";
import { SEEP_FLOOR_BAND_CELLS, SEEP_JET_REACH_CELLS, type SeepGeometry } from "./seep-observables";

/** One owner: low cell (box), width, liquid in cells, mean x-face u (m/s), and the solver's own open flag where it has one. */
export interface CellRecord { x: number; y: number; z: number; w: number; liquid: number; u: number; open?: boolean }
export interface CellDump { dump(): Promise<CellRecord[]>; destroy(): void }

const RECORD_WORDS = 8, HEADER_WORDS = 4, CAPACITY = 1 << 20;
const DUMP_WGSL = /* wgsl */`
fn dumpEmit(q:vec3u,w:u32,liquid:f32,u:f32,open:u32){
 let slot=atomicAdd(&dumpWords[0],1u);if(slot>=${CAPACITY}u){return;}
 let b=${HEADER_WORDS}u+${RECORD_WORDS}u*slot;
 atomicStore(&dumpWords[b],q.x);atomicStore(&dumpWords[b+1u],q.y);atomicStore(&dumpWords[b+2u],q.z);atomicStore(&dumpWords[b+3u],w);
 atomicStore(&dumpWords[b+4u],bitcast<u32>(liquid));atomicStore(&dumpWords[b+5u],bitcast<u32>(u));atomicStore(&dumpWords[b+6u],open);
}`;

function dumpBuffers(device: GPUDevice, label: string) {
  const size = 4 * (HEADER_WORDS + RECORD_WORDS * CAPACITY);
  const words = device.createBuffer({ label, size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const read = device.createBuffer({ label: `${label} readback`, size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const run = async (encode: (pass: GPUComputePassEncoder) => void, ownOpen: boolean): Promise<CellRecord[]> => {
    device.queue.writeBuffer(words, 0, new Uint32Array(HEADER_WORDS));
    const encoder = device.createCommandEncoder({ label });
    const pass = encoder.beginComputePass({ label }); encode(pass); pass.end();
    encoder.copyBufferToBuffer(words, 0, read, 0, size);
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const u = new Uint32Array(read.getMappedRange().slice(0)); read.unmap();
    const f = new Float32Array(u.buffer), n = u[0]!;
    if (n > CAPACITY) throw new Error(`${label}: ${n} owners exceed the dump's ${CAPACITY}`);
    const records: CellRecord[] = [];
    for (let i = 0; i < n; i++) {
      const b = HEADER_WORDS + RECORD_WORDS * i;
      records.push({ x: u[b]!, y: u[b + 1]!, z: u[b + 2]!, w: u[b + 3]!, liquid: f[b + 4]!, u: f[b + 5]!, ...(ownOpen ? { open: u[b + 6] !== 0 } : {}) });
    }
    return records;
  };
  return { words, run, destroy() { words.destroy(); read.destroy(); } };
}

interface NBInternals {
  device: GPUDevice; vertexPhiField: GPUTexture; velocityA: GPUTexture;
  mixedFrame: { ownership: UniformMixedOwnership };
}

/** NB-FLIP: every mixed owner at x + w > wall.low with liquid; BULK16 pages lie in the left reservoir (the zone observer asserts it). */
export async function createNBFlipCellDump(device: GPUDevice, solver: GPUSolverInstance, g: SeepGeometry): Promise<CellDump> {
  const nb = solver as unknown as NBInternals, frame = nb.mixedFrame;
  if (!frame?.ownership || !nb.vertexPhiField || !nb.velocityA) throw new Error("NB-FLIP cell dump: the solver has no mixed-ownership frame");
  const layout = uniformDetailBindLayout(device, { label: "Seep audit dump (NB-FLIP)", entries: [
    ...[0, 1].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" as const, viewDimension: "3d" as const } })),
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } }] });
  const code = uniformMixedCountedEntriesWGSL(uniformMixedTopologyWGSL(frame.ownership.capacity, 0) + /* wgsl */`
@group(1) @binding(0) var phi:texture_3d<f32>;
@group(1) @binding(1) var velocity:texture_3d<f32>;
@group(1) @binding(2) var<storage,read_write> dumpWords:array<atomic<u32>>;
fn corner(p:vec3u)->f32{return ${C}textureLoad(phi,vec3i(p),0).x;}
fn faceU(a:vec3i)->f32{return ${C}textureLoad(velocity,a,0).x;}
${levelSetFillWGSL}
${DUMP_WGSL}
fn ownerFill(q:vec3u,w:u32)->f32{
 var c:array<f32,8>;var lo=1e30;var hi=-1e30;
 for(var k=0u;k<8u;k++){c[k]=corner(q+umCorner(k,2u)*w);lo=min(lo,c[k]);hi=max(hi,c[k]);}
 let cells=f32(w*w*w);
 if(hi<0.0){return cells;}if(lo>=0.0){return 0.0;}
 return fill(c)*cells;
}
fn boxOrigin()->vec3i{return 32*(umWorldLow()-umFramePage());}
@compute @workgroup_size(64) fn dumpOwners(@builtin(global_invocation_id) gid:vec3u){
 let owner=umAllOwner(gid);
 if(owner.width==0u){return;}
 let window=umOrigin(owner);let q=vec3u(vec3i(window)-boxOrigin());let w=owner.width;
 if(q.x+w<=${g.wall.low}u){return;}
 let liquid=ownerFill(window,w);if(liquid<=0.0){return;}
 let u=0.5*(faceU(umFace(owner,0u,-1,0u).anchor)+faceU(umFace(owner,0u,1,0u).anchor));
 dumpEmit(q,w,liquid,u,0u);
}`, ["dumpOwners"]);
  const module = uniformDetailModule(nb.device, { label: "Seep audit dump (NB-FLIP)", code });
  const messages = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
  if (messages.length) throw new Error(`NB-FLIP cell dump WGSL: ${messages.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("; ")}`);
  const pipeline = await uniformDetailPipeline(device, frame.ownership, {
    layout: device.createPipelineLayout({ bindGroupLayouts: [frame.ownership.bindLayout, layout] }),
    compute: { module, entryPoint: "dumpOwners", constants: { umDispatchX: frame.ownership.dispatchX, umCountedJobs: UNIFORM_MIXED_COUNTED.all } } });
  const buffers = dumpBuffers(device, "Seep audit dump (NB-FLIP)");
  const group = uniformDetailGroup(device, { layout, entries: [{ binding: 0, resource: nb.vertexPhiField }, { binding: 1, resource: nb.velocityA }, { binding: 2, resource: { buffer: buffers.words } }] });
  return {
    dump: () => buffers.run(pass => { pass.setBindGroup(0, frame.ownership.bindGroup); pass.setBindGroup(1, group.group); frame.ownership.dispatchAllCounted(pass, pipeline); }, false),
    destroy: () => buffers.destroy(),
  };
}

/** Seep: every dense cell at x >= wall.low with liquid, open or solid (its own seepCellOpen). */
export async function createSeepCellDump(device: GPUDevice, source: SeepObservationSource, g: SeepGeometry): Promise<CellDump> {
  const WORKGROUP = 256, x0 = g.wall.low, sx = g.nx - x0, cells = sx * g.ny * g.nz;
  const groups = Math.ceil(cells / WORKGROUP), dispatchX = Math.min(groups, 32768), dispatchY = Math.ceil(groups / dispatchX);
  const code = /* wgsl */`
${source.declarations(0)}
${source.wgsl}
@group(1) @binding(0) var<storage,read_write> dumpWords:array<atomic<u32>>;
${levelSetFillWGSL}
${DUMP_WGSL}
@compute @workgroup_size(${WORKGROUP}) fn dumpCells(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x+gid.y*${dispatchX * WORKGROUP}u;if(i>=${cells}u){return;}
 let c=vec3i(i32(${x0}u+i%${sx}u),i32((i/${sx}u)%${g.ny}u),i32(i/${sx * g.ny}u));
 var v:array<f32,8>;var lo=1e30;var hi=-1e30;
 for(var k=0u;k<8u;k++){v[k]=seepVertexPhi(c+vec3i(vec3u(k&1u,(k>>1u)&1u,k>>2u)));lo=min(lo,v[k]);hi=max(hi,v[k]);}
 var liquid=0.0;if(hi<0.0){liquid=1.0;}else if(lo<0.0){liquid=fill(v);}
 if(liquid<=0.0){return;}
 let u=0.5*(seepFaceU(c-vec3i(1,0,0),0u)+seepFaceU(c,0u));
 dumpEmit(vec3u(c),1u,liquid,u,select(0u,1u,seepCellOpen(c)));
}`;
  const module = device.createShaderModule({ label: "Seep audit dump (Seep)", code });
  const messages = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
  if (messages.length) throw new Error(`Seep cell dump WGSL: ${messages.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("; ")}`);
  const sourceLayout = device.createBindGroupLayout({ label: "Seep audit source", entries: source.buffers.map((_, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const } })) });
  const wordsLayout = device.createBindGroupLayout({ label: "Seep audit words", entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } }] });
  const pipeline = await device.createComputePipelineAsync({ label: "Seep audit dump (Seep)", layout: device.createPipelineLayout({ bindGroupLayouts: [sourceLayout, wordsLayout] }), compute: { module, entryPoint: "dumpCells" } });
  const buffers = dumpBuffers(device, "Seep audit dump (Seep)");
  const sourceGroup = device.createBindGroup({ layout: sourceLayout, entries: source.buffers.map((b, binding) => ({ binding, resource: { buffer: b.buffer } })) });
  const wordsGroup = device.createBindGroup({ layout: wordsLayout, entries: [{ binding: 0, resource: { buffer: buffers.words } }] });
  return {
    dump: () => buffers.run(pass => { pass.setPipeline(pipeline); pass.setBindGroup(0, sourceGroup); pass.setBindGroup(1, wordsGroup); pass.dispatchWorkgroups(dispatchX, dispatchY); }, true),
    destroy: () => buffers.destroy(),
  };
}

export const AUDIT_SLICE_CELLS = 8;
export const AUDIT_SLICE_END = 448;

export interface AuditSlice {
  x0: number; x1: number;
  /** Liquid at y >= 8 (L), its weighted mean y (cells, cell centres) and mean u (m/s); all liquid in the slice (L). */
  jet_L: number; meanY: number | null; meanU: number | null; all_L: number;
  /** The ballistic centreline at the slice centre, y0 - g x^2 / (2 u0^2), in cells. */
  ballisticY: number;
}

export interface AuditRow {
  kind: "audit"; step: number;
  /** Wall zone (wall.low <= x < wall.high), litres: open throat cells, solid cells, 4h owners overlapping the zone. */
  wall_L: { open: number; solid: number; coarse: number; total: number };
  /** Seep only: owners whose own open flag disagrees with the scene's throat patch. */
  openMismatch?: number;
  throatU_m_s: number; y0_cells: number;
  slices: AuditSlice[];
  /** Jet liquid within wall.high + 64 (the G8 column, from the dump) and its ballistic estimate q * 64h / u0 (L). */
  jetColumn_L: number; ballisticColumn_L: number;
  /** x where the ballistic centreline reaches the top of the floor band, and where the measured jet mean y first falls below 10 cells (interpolated). */
  ballisticImpactX: number; measuredImpactX: number | null;
  owners: { h: number; coarse: number };
}

/** The scene's open throat: the clear patch through the wall (seepGeometry's throat is its low y, z). */
export const throatOpen = (g: SeepGeometry, x: number, y: number, z: number) =>
  x >= g.wall.low && x < g.wall.high && y >= g.throat[1] && y < g.throat[1] + g.throatSide && z >= g.throat[2] && z < g.throat[2] + g.throatSide;

export function auditCells(g: SeepGeometry, step: number, records: readonly CellRecord[], throatU_m_s: number): AuditRow {
  const L = g.cell_m ** 3 * 1000, high = g.wall.high, band = SEEP_FLOOR_BAND_CELLS;
  const wall = { open: 0, solid: 0, coarse: 0, total: 0 };
  let openMismatch = 0, jetColumn = 0, hOwners = 0, coarseOwners = 0;
  const n = Math.ceil((AUDIT_SLICE_END - high) / AUDIT_SLICE_CELLS);
  const acc = Array.from({ length: n }, () => ({ jet: 0, y: 0, u: 0, all: 0 }));
  for (const r of records) {
    if (r.w === 1) hOwners++; else coarseOwners++;
    const inWall = r.x < high && r.x + r.w > g.wall.low;
    if (inWall) {
      if (r.w > 1) wall.coarse += r.liquid;
      else if (throatOpen(g, r.x, r.y, r.z)) wall.open += r.liquid; else wall.solid += r.liquid;
      if (r.w === 1 && r.open !== undefined && r.open !== throatOpen(g, r.x, r.y, r.z)) openMismatch++;
      continue;
    }
    if (r.x < high) continue;
    const above = r.y >= band;
    if (above && r.x < high + SEEP_JET_REACH_CELLS) jetColumn += r.liquid;
    // A w-wide owner spreads its liquid evenly over its w columns of x; its height is its centre.
    const yc = r.y + 0.5 * r.w;
    for (let dx = 0; dx < r.w; dx++) {
      const s = Math.floor((r.x + dx - high) / AUDIT_SLICE_CELLS);
      if (s < 0 || s >= n) continue;
      const l = r.liquid / r.w, a = acc[s]!;
      a.all += l;
      if (above) { a.jet += l; a.y += l * yc; a.u += l * r.u; }
    }
  }
  wall.total = wall.open + wall.solid + wall.coarse;
  const y0 = g.throat[1] + 0.5 * g.throatSide, u0 = throatU_m_s, h = g.cell_m, gr = g.gravity_m_s2;
  const ballistic = (x: number) => y0 - gr * (x * h) ** 2 / (2 * u0 * u0) / h;
  const slices: AuditSlice[] = acc.map((a, s) => {
    const x0 = high + s * AUDIT_SLICE_CELLS, x1 = Math.min(x0 + AUDIT_SLICE_CELLS, AUDIT_SLICE_END);
    return { x0, x1, jet_L: a.jet * L, meanY: a.jet > 0 ? a.y / a.jet : null, meanU: a.jet > 0 ? a.u / a.jet : null, all_L: a.all * L, ballisticY: ballistic(0.5 * (x0 + x1) - high) };
  });
  let measuredImpactX: number | null = null;
  for (let s = 0; s < slices.length && measuredImpactX === null; s++) {
    const y = slices[s]!.meanY, prev = s > 0 ? slices[s - 1]!.meanY : null, xc = 0.5 * (slices[s]!.x0 + slices[s]!.x1);
    if (y === null || y >= 10) continue;
    measuredImpactX = prev === null || prev < 10 ? xc : xc - AUDIT_SLICE_CELLS * (10 - y) / (prev - y);
  }
  // Delivery q at the throat: u0 over the throat's side^2 open face.
  const q_m3s = u0 * (g.throatSide * h) ** 2;
  return {
    kind: "audit", step, wall_L: { open: wall.open * L, solid: wall.solid * L, coarse: wall.coarse * L, total: wall.total * L },
    ...(records.some(r => r.open !== undefined) ? { openMismatch } : {}),
    throatU_m_s, y0_cells: y0, slices,
    jetColumn_L: jetColumn * L,
    ballisticImpactX: high + u0 * Math.sqrt(2 * (y0 - band) * h / gr) / h, measuredImpactX,
    // The ballistic column: the throat's flux over the time its constant horizontal u takes to cross 64 cells.
    ballisticColumn_L: q_m3s * (SEEP_JET_REACH_CELLS * h / u0) * 1000,
    owners: { h: hOwners, coarse: coarseOwners },
  };
}
