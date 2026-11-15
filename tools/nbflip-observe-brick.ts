/** The box observer (tools/nbflip-observables.ts) on NB-FLIP's brick engine
 * (lib/methods/narrow-band/brick-solver.ts), with the same measure as the old
 * engine's observer (tools/nbflip-observe.ts):
 * - an owner is an h cell of a live F brick (w = 1; M2 has no 4h tiles);
 * - its liquid is the six-tetrahedron fill of its eight corner phi times its
 *   cells (level-set-fill.wgsl.ts);
 * - its velocity is, an axis, the mean of its two faces. Faces at the
 *   lattice walls are closed (0). A face in a brick that is not resident
 *   (beyond the band) drops out, and the cell takes the other face.
 *
 * Work follows the pool: one workgroup of 64 a pool slot, 8 cells a lane;
 * free slots exit. Reads only after the caller fenced the frame. Issues no
 * solver call. When 4h tiles land, a 4h-owned tile adds as one owner of
 * width 4 from its C brick. */
import { brickAbiWGSL } from "../lib/methods/narrow-band/brick-abi";
import type { BrickNBFlipSolver } from "../lib/methods/narrow-band/brick-solver";
import { levelSetFillWGSL } from "../lib/core/level-set-fill.wgsl";
import { boxObserverWGSL, boxWords, decodeBoxWords, type BoxGeometry } from "./nbflip-observables";
import type { BoxObserver } from "./nbflip-observe";

export async function createBrickBoxObserver(device: GPUDevice, solver: BrickNBFlipSolver, g: BoxGeometry): Promise<BoxObserver> {
  const e = solver.brickEngine, L = e.layout, n = L.lattice;
  if (g.nx !== n[0] || g.ny !== n[1] || g.nz !== n[2]) throw new Error(`Brick box observer: the box ${g.nx}x${g.ny}x${g.nz} is not the engine's lattice ${n.join("x")}`);
  const WORDS = boxWords(g);
  const code = /* wgsl */`
${brickAbiWGSL(L, { set: { name: "nbSet", atomic: false }, dir: { name: "dir", atomic: false } })}
@group(0) @binding(0) var<storage,read> nbSet:array<u32>;
@group(0) @binding(1) var<storage,read> dir:array<u32>;
@group(0) @binding(2) var<storage,read> phi:array<u32>;
@group(0) @binding(3) var<storage,read> vel:array<f32>;
@group(0) @binding(4) var<storage,read_write> boxWords:array<atomic<u32>>;
${levelSetFillWGSL}
${boxObserverWGSL(g, 64)}
fn vertexAt(v:vec3i)->f32{
  if(any(v<vec3i(0))||any(v>NB_LATTICE)){return NB_PHI_AIR;}
  let s=nbFSlotOfBrick(v>>vec3u(3u));
  if(s>=NB_SEAM){return NB_PHI_AIR;}
  return bitcast<f32>(phi[NB_PHI_F_PHI+NB_STRIDE*s+nbLocal(v)]);
}
// The lower face of c along a, and whether it is stored (the walls are closed and stored as 0).
fn faceAt(c:vec3i,a:u32)->vec2f{
  if(any(c<vec3i(0))||any(c>=NB_LATTICE)){return vec2f(0.0,1.0);}
  let s=nbFSlotOfCell(c);
  if(s>=NB_SEAM){return vec2f(0.0,0.0);}
  return vec2f(vel[NB_VEL_F_U0+NB_STRIDE*(s+a*NB_FCAP)+nbLocal(c)],1.0);
}
@compute @workgroup_size(64) fn observe(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  boxBegin(lane);
  let slot=wid.x;let key=nbFRec(slot,NB_REC_KEY);
  if(key!=NB_NONE){
    let o=8*nbKeyCoord(key);
    for(var j=0u;j<8u;j++){
      let c=o+nbLocalCoord(lane+64u*j);
      if(!nbInLattice(c)){continue;}
      var k8:array<f32,8>;var lo=1e30;var hi=-1e30;
      for(var k=0u;k<8u;k++){k8[k]=vertexAt(c+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)));lo=min(lo,k8[k]);hi=max(hi,k8[k]);}
      var liquid=0.0;
      if(hi<0.0){liquid=1.0;}else if(lo<0.0){liquid=fill(k8);}
      var u=vec3f(0.0);
      if(liquid>0.0){
        for(var a=0u;a<3u;a++){
          var e=vec3i(0);e[a]=1;
          let l=faceAt(c,a);let h=faceAt(c+e,a);
          u[a]=(l.x+h.x)/max(l.y+h.y,1.0);
        }
      }
      boxAdd(vec3u(c),1u,liquid,u);
    }
  }
  boxEnd(lane);
}`;
  const module = device.createShaderModule({ label: "Box observer (brick engine)", code });
  const messages = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
  if (messages.length) throw new Error(`Brick box observer WGSL: ${messages.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("; ")}`);
  // An explicit layout: "auto" would drop the bindings this entry never reads.
  const C = GPUShaderStage.COMPUTE;
  const bindLayout = device.createBindGroupLayout({ label: "Box observer (brick engine)", entries: [
    ...[0, 1, 2, 3].map(binding => ({ binding, visibility: C, buffer: { type: "read-only-storage" as const } })),
    { binding: 4, visibility: C, buffer: { type: "storage" as const } }] });
  const pipeline = await device.createComputePipelineAsync({ label: "Box observer (brick engine)", layout: device.createPipelineLayout({ bindGroupLayouts: [bindLayout] }), compute: { module, entryPoint: "observe" } });
  const words = device.createBuffer({ label: "Box observer words (brick engine)", size: 4 * WORDS, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const read = device.createBuffer({ label: "Box observer readback (brick engine)", size: 4 * WORDS, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const group = device.createBindGroup({ label: "Box observer (brick engine)", layout: bindLayout, entries: [
    { binding: 0, resource: { buffer: e.buffer("set") } }, { binding: 1, resource: { buffer: e.buffer("dir") } },
    { binding: 2, resource: { buffer: e.buffer("phi") } }, { binding: 3, resource: { buffer: e.buffer("vel") } },
    { binding: 4, resource: { buffer: words } }] });
  const zero = new Uint32Array(WORDS);
  return {
    geometry: g,
    async observe() {
      device.queue.writeBuffer(words, 0, zero);
      const encoder = device.createCommandEncoder({ label: "Box observer (brick engine)" });
      const pass = encoder.beginComputePass({ label: "Box observer (brick engine)" });
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(L.cap.f); pass.end();
      encoder.copyBufferToBuffer(words, 0, read, 0, 4 * WORDS);
      device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ); const u = new Uint32Array(read.getMappedRange().slice(0)); read.unmap();
      const sums = decodeBoxWords(g, u);
      sums.particles = solver.info.brickEngine?.particles ?? 0;
      return { ...sums, bulkRegions: 0 };
    },
    destroy() { words.destroy(); read.destroy(); },
  };
}
