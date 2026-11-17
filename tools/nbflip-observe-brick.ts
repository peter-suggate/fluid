/** The box observer (tools/nbflip-observables.ts) on NB-FLIP's brick engine
 * (lib/methods/narrow-band/brick-solver.ts), with the same measure as the old
 * engine's observer (tools/nbflip-observe.ts):
 * - owners are the engine's tiers, vGauge's split (velocity-kernels.wgsl.ts), so nothing counts twice:
 *   - an h owner is a cell of a live F brick whose tile is h-owned (every
 *     cell all-h; brick-owner.wgsl.ts nbTileOwnerH);
 *   - a 4h owner is a 4h-owned tile of a live C page (w = 4), the old
 *     observer's width-4 owner. An F brick still resident over it (reach,
 *     margins) holds stale φ and is not read;
 * - an owner's liquid is the six-tetrahedron fill of its eight corner phi
 *   times its cells (level-set-fill.wgsl.ts): F's vertices for an h cell,
 *   the C slot's corners for a 4h tile (vFill4's corner rule);
 * - its velocity is, an axis, the mean of its two faces. For an h cell
 *   these are its F faces. For a 4h tile a face plane is its C face, or,
 *   where the tile beside is h-owned, the area mean of the 16 F faces on
 *   that plane (the old observer's patch mean). Faces at the box's walls
 *   are closed (0). An h face in a brick that is not resident (beyond the
 *   band) drops out, and the cell takes the other face.
 *
 * Work follows the pools: one workgroup of 64 a pool slot (F slots, then C
 * slots), 8 cells or tiles a lane; free slots exit. Reads only after the
 * caller fenced the frame. Its only solver call is readStats (a fence and a
 * header read, no side effects), for the live bank's particle count.
 *
 * Until 2026-11-17 the observer read F cells only, so a tiered scene's 4h
 * interior was missing (the letters' calm pool: drift -97% at step 30 while
 * the gauge held the budget to 0.1%). */
import { BOX_TOKEN, boxWGSL, brickAbiWGSL, resolveBox } from "../lib/methods/narrow-band/brick-abi";
import { brickOwnerWGSL } from "../lib/methods/narrow-band/brick-owner.wgsl";
import type { BrickNBFlipSolver } from "../lib/methods/narrow-band/brick-solver";
import { levelSetFillWGSL } from "../lib/core/level-set-fill.wgsl";
import { boxObserverWGSL, boxWords, decodeBoxWords, type BoxGeometry } from "./nbflip-observables";
import type { BoxObserver } from "./nbflip-observe";

export async function createBrickBoxObserver(device: GPUDevice, solver: BrickNBFlipSolver, g: BoxGeometry): Promise<BoxObserver> {
  // The measure is over the box (layout.box: the container's cells). A padded lattice's cells past it hold no liquid and
  // are not read; its walls are at the box (brick-abi.ts resolveBox).
  const e = solver.brickEngine, L = e.layout, n = L.box;
  if (g.nx !== n[0] || g.ny !== n[1] || g.nz !== n[2]) throw new Error(`Brick box observer: the box ${g.nx}x${g.ny}x${g.nz} is not the engine's box ${n.join("x")}`);
  const WORDS = boxWords(g);
  const code = resolveBox(L, /* wgsl */`
${brickAbiWGSL(L, { set: { name: "nbSet", atomic: false }, dir: { name: "dir", atomic: false } })}
@group(0) @binding(0) var<storage,read> nbSet:array<u32>;
@group(0) @binding(1) var<storage,read> dir:array<u32>;
@group(0) @binding(2) var<storage,read> phi:array<u32>;
@group(0) @binding(3) var<storage,read> vel:array<f32>;
@group(0) @binding(4) var<storage,read_write> boxWords:array<atomic<u32>>;
const OBS_TIERED:bool=${e.stages.tiered === true};
${brickOwnerWGSL({ phi: "phi", vel: "vel", box: boxWGSL(L) })}
${levelSetFillWGSL}
${boxObserverWGSL(g, 64)}
fn vertexAt(v:vec3i)->f32{
  if(any(v<vec3i(0))||any(v>${BOX_TOKEN})){return NB_PHI_AIR;}
  let s=nbFSlotOfBrick(v>>vec3u(3u));
  if(s>=NB_SEAM){return NB_PHI_AIR;}
  return bitcast<f32>(phi[NB_PHI_F_PHI+NB_STRIDE*s+nbLocal(v)]);
}
// The lower face of c along a, and whether it is stored (the walls are closed and stored as 0).
fn faceAt(c:vec3i,a:u32)->vec2f{
  if(any(c<vec3i(0))||any(c>=${BOX_TOKEN})){return vec2f(0.0,1.0);}
  let s=nbFSlotOfCell(c);
  if(s>=NB_SEAM){return vec2f(0.0,0.0);}
  return vec2f(vel[NB_VEL_F_U0+NB_STRIDE*(s+a*NB_FCAP)+nbLocal(c)],1.0);
}
// The liquid of an owner of width w from its eight corners (cells).
fn ownerLiquid(k8:array<f32,8>,w:f32)->f32{
  var lo=1e30;var hi=-1e30;for(var k=0u;k<8u;k++){lo=min(lo,k8[k]);hi=max(hi,k8[k]);}
  if(hi<0.0){return w*w*w;}if(lo>=0.0){return 0.0;}
  return fill(k8)*w*w*w;
}
// The unit vector along axis a (a select: no write through a dynamic index).
fn obsUnit(a:u32)->vec3i{return select(vec3i(0),vec3i(1),vec3<bool>(a==0u,a==1u,a==2u));}
// A 4h tile T's face plane along a, 4 (T[a] + up) (up 0 or 1): 0 on the box's walls; the C face where the tile
// beside is 4h-owned too; else the area mean of the plane's 16 F faces (an h-owned tile beside stores them in F).
// The wall test 4 P[a] >= box[a] holds the box to whole tiles where a 4h tile meets it: a whole-page box (the
// only padded box, brick-abi.md §2) is.
fn tilePlane(T:vec3i,a:u32,up:i32)->f32{
  let e=obsUnit(a);let P=T+up*e;
  if(P[a]<=0||4*P[a]>=${boxWGSL(L)}[a]){return 0.0;}
  let beside=T+(2*up-1)*e;
  if(!nbTileOwnerH(beside)){return nbCFace(P,a);}
  let eb=obsUnit((a+1u)%3u);let ed=obsUnit((a+2u)%3u);var sum=0.0;
  for(var i=0;i<16;i++){sum+=nbFFace(4*P+(i&3)*eb+(i>>2)*ed,a);}
  return sum/16.0;
}
@compute @workgroup_size(64) fn observe(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  boxBegin(lane);
  if(wid.x>=NB_FCAP){
    // A C page: its 4h-owned tiles, eight a lane.
    let cs=wid.x-NB_FCAP;
    if(OBS_TIERED&&nbCRec(cs,NB_REC_KEY)!=NB_NONE){
      let o=nbCOrigin(cs)>>vec3u(2u);
      for(var j=0u;j<8u;j++){
        let i=lane+64u*j;let T=o+vec3i(i32(i&7u),i32((i>>3u)&7u),i32(i>>6u));
        if(!nbTileInWallBox(T)||nbTileOwnerH(T)){continue;}
        var k8:array<f32,8>;
        for(var k=0u;k<8u;k++){k8[k]=nbCVertex(T+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)));}
        let liquid=ownerLiquid(k8,4.0);
        var u=vec3f(0.0);
        if(liquid>0.0){for(var a=0u;a<3u;a++){u+=0.5*(tilePlane(T,a,0)+tilePlane(T,a,1))*vec3f(obsUnit(a));}}
        boxAdd(vec3u(4*T),4u,liquid,u);
      }
    }
    boxEnd(lane);
    return;
  }
  let slot=wid.x;let key=nbFRec(slot,NB_REC_KEY);
  if(key!=NB_NONE){
    let o=8*nbKeyCoord(key);
    for(var j=0u;j<8u;j++){
      let c=o+nbLocalCoord(lane+64u*j);
      if(!nbInWallBox(c)||(OBS_TIERED&&!nbCellOwnerH(c))){continue;}
      var k8:array<f32,8>;
      for(var k=0u;k<8u;k++){k8[k]=vertexAt(c+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)));}
      let liquid=ownerLiquid(k8,1.0);
      var u=vec3f(0.0);
      if(liquid>0.0){
        for(var a=0u;a<3u;a++){
          let l=faceAt(c,a);let h=faceAt(c+obsUnit(a),a);
          u+=(l.x+h.x)/max(l.y+h.y,1.0)*vec3f(obsUnit(a));
        }
      }
      boxAdd(vec3u(c),1u,liquid,u);
    }
  }
  boxEnd(lane);
}`);
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
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(L.cap.f + L.cap.c); pass.end();
      encoder.copyBufferToBuffer(words, 0, read, 0, 4 * WORDS);
      device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ); const u = new Uint32Array(read.getMappedRange().slice(0)); read.unmap();
      const sums = decodeBoxWords(g, u);
      sums.particles = (await solver.readStats()).brickEngine?.particles ?? 0;
      return { ...sums, bulkRegions: 0 };
    },
    destroy() { words.destroy(); read.destroy(); },
  };
}
