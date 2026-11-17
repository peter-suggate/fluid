/** The seep's zone observer (tools/seep-observables.ts) on NB-FLIP's brick engine
 * (lib/methods/narrow-band/brick-solver.ts), with the old engine's measure
 * (tools/seep-observe-nbflip.ts) and the brick box observer's owners
 * (tools/nbflip-observe-brick.ts), so nothing counts twice:
 * - an h owner is a cell of a live F brick whose tile is h-owned; its liquid is
 *   the six-tetrahedron fill of its eight F vertices;
 * - a 4h owner is a 4h-owned tile of a live C page (w = 4), its liquid from the
 *   C slot's corners (vFill4's corner rule);
 * - only the container's cells count (an open top's headroom page above it is in the box, not in the zones);
 * - solids are not subtracted (the wall zone holds them), as the old observer;
 * - u, read only for the jet slices (right of the wall, above the floor band),
 *   is the mean of an owner's two x faces: an h cell's F faces (a face in a brick
 *   that is not resident drops out), a 4h tile's face planes (its C face, or the
 *   16 F faces where the tile beside is h-owned; 0 at the box's walls);
 * - the throat's u is the +x face of the throat cell (an h cell; a throat under
 *   a 4h tile reports none, and the bench fails on it as on the old engine).
 *
 * Work follows the pools: one workgroup of 64 a pool slot (F slots, then C
 * slots), 8 cells or tiles a lane; free slots exit. Reads only after the
 * caller fenced the frame. Its only solver call is readStats (a fence and a
 * header read, no side effects), for the live bank's particle count. */
import { BOX_TOKEN, boxWGSL, brickAbiWGSL, resolveBox } from "../lib/methods/narrow-band/brick-abi";
import { brickOwnerWGSL } from "../lib/methods/narrow-band/brick-owner.wgsl";
import type { BrickNBFlipSolver } from "../lib/methods/narrow-band/brick-solver";
import { levelSetFillWGSL } from "../lib/core/level-set-fill.wgsl";
import { ZONE_WORD, ZONE_WORDS, decodeZoneWords, seepZoneWGSL, type SeepGeometry } from "./seep-observables";
import type { ZoneObserver } from "./seep-observe-nbflip";

export async function createBrickZoneObserver(device: GPUDevice, solver: BrickNBFlipSolver, g: SeepGeometry): Promise<ZoneObserver> {
  const e = solver.brickEngine, L = e.layout, n = L.box;
  // An open top puts a page of headroom above the container in the box (brick-scene.ts BRICK_OPEN_TOP_HEADROOM); the
  // zones are the container's cells, as the old engine's observer reads only the lattice.
  if (g.nx !== n[0] || g.ny > n[1] || g.nz !== n[2]) throw new Error(`Brick seep observer: the seep's ${g.nx}x${g.ny}x${g.nz} does not fit the engine's box ${n.join("x")}`);
  const code = resolveBox(L, /* wgsl */`
${brickAbiWGSL(L, { set: { name: "nbSet", atomic: false }, dir: { name: "dir", atomic: false } })}
@group(0) @binding(0) var<storage,read> nbSet:array<u32>;
@group(0) @binding(1) var<storage,read> dir:array<u32>;
@group(0) @binding(2) var<storage,read> phi:array<u32>;
@group(0) @binding(3) var<storage,read> vel:array<f32>;
@group(0) @binding(4) var<storage,read_write> zoneWords:array<atomic<u32>>;
const OBS_TIERED:bool=${e.stages.tiered === true};
const OBS_NY:i32=${g.ny};
${brickOwnerWGSL({ phi: "phi", vel: "vel", box: boxWGSL(L) })}
${levelSetFillWGSL}
${seepZoneWGSL(g, 64)}
fn vertexAt(v:vec3i)->f32{
  if(any(v<vec3i(0))||any(v>${BOX_TOKEN})){return NB_PHI_AIR;}
  let s=nbFSlotOfBrick(v>>vec3u(3u));
  if(s>=NB_SEAM){return NB_PHI_AIR;}
  return bitcast<f32>(phi[NB_PHI_F_PHI+NB_STRIDE*s+nbLocal(v)]);
}
// The lower x face of c, and whether it is stored (the walls are closed and stored as 0).
fn faceX(c:vec3i)->vec2f{
  if(any(c<vec3i(0))||any(c>=${BOX_TOKEN})){return vec2f(0.0,1.0);}
  let s=nbFSlotOfCell(c);
  if(s>=NB_SEAM){return vec2f(0.0,0.0);}
  return vec2f(vel[NB_VEL_F_U0+NB_STRIDE*s+nbLocal(c)],1.0);
}
fn ownerLiquid(k8:array<f32,8>,w:f32)->f32{
  var lo=1e30;var hi=-1e30;for(var k=0u;k<8u;k++){lo=min(lo,k8[k]);hi=max(hi,k8[k]);}
  if(hi<0.0){return w*w*w;}if(lo>=0.0){return 0.0;}
  return fill(k8)*w*w*w;
}
// Whether an owner at q of width w is in a jet slice and needs its u (seepZoneWGSL reads u only there).
fn jetOwner(q:vec3u)->bool{return q.x>=ZONE_WALL_HIGH&&q.y>=ZONE_FLOOR_BAND;}
// A 4h tile T's x face plane 4 (T.x + up): 0 on the box's walls; the C face where the tile beside is 4h-owned too;
// else the area mean of the plane's 16 F faces (the brick box observer's tilePlane on x).
fn tilePlaneX(T:vec3i,up:i32)->f32{
  let P=T+vec3i(up,0,0);
  if(P.x<=0||4*P.x>=${boxWGSL(L)}.x){return 0.0;}
  if(!nbTileOwnerH(T+vec3i(2*up-1,0,0))){return nbCFace(P,0u);}
  var sum=0.0;
  for(var i=0;i<16;i++){sum+=nbFFace(4*P+vec3i(0,i&3,i>>2),0u);}
  return sum/16.0;
}
@compute @workgroup_size(64) fn observe(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  zoneBegin(lane);
  if(wid.x>=NB_FCAP){
    let cs=wid.x-NB_FCAP;
    if(OBS_TIERED&&nbCRec(cs,NB_REC_KEY)!=NB_NONE){
      let o=nbCOrigin(cs)>>vec3u(2u);
      for(var j=0u;j<8u;j++){
        let i=lane+64u*j;let T=o+vec3i(i32(i&7u),i32((i>>3u)&7u),i32(i>>6u));
        if(!nbTileInWallBox(T)||nbTileOwnerH(T)||4*T.y>=OBS_NY){continue;}
        var k8:array<f32,8>;
        for(var k=0u;k<8u;k++){k8[k]=nbCVertex(T+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)));}
        let liquid=ownerLiquid(k8,4.0);let q=vec3u(4*T);
        var u=0.0;if(liquid>0.0&&jetOwner(q)){u=0.5*(tilePlaneX(T,0)+tilePlaneX(T,1));}
        zoneAdd(q,4u,liquid,u);
      }
    }
    zoneEnd(lane);
    return;
  }
  let slot=wid.x;let key=nbFRec(slot,NB_REC_KEY);
  if(key!=NB_NONE){
    let o=8*nbKeyCoord(key);
    for(var j=0u;j<8u;j++){
      let c=o+nbLocalCoord(lane+64u*j);
      if(!nbInWallBox(c)||c.y>=OBS_NY||(OBS_TIERED&&!nbCellOwnerH(c))){continue;}
      var k8:array<f32,8>;
      for(var k=0u;k<8u;k++){k8[k]=vertexAt(c+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)));}
      let liquid=ownerLiquid(k8,1.0);let q=vec3u(c);
      var u=0.0;
      if(liquid>0.0&&jetOwner(q)){let l=faceX(c);let h=faceX(c+vec3i(1,0,0));u=(l.x+h.x)/max(l.y+h.y,1.0);}
      zoneAdd(q,1u,liquid,u);
      if(all(c==ZONE_THROAT)){
        let h=faceX(c+vec3i(1,0,0));
        if(h.y>0.0){atomicStore(&zoneWords[${ZONE_WORD.throatU}],bitcast<u32>(h.x));atomicStore(&zoneWords[${ZONE_WORD.throatFound}],1u);}
      }
    }
  }
  zoneEnd(lane);
}`);
  const module = device.createShaderModule({ label: "Seep observer (brick engine)", code });
  const messages = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
  if (messages.length) throw new Error(`Brick seep observer WGSL: ${messages.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("; ")}`);
  // An explicit layout: "auto" would drop the bindings this entry never reads.
  const C = GPUShaderStage.COMPUTE;
  const bindLayout = device.createBindGroupLayout({ label: "Seep observer (brick engine)", entries: [
    ...[0, 1, 2, 3].map(binding => ({ binding, visibility: C, buffer: { type: "read-only-storage" as const } })),
    { binding: 4, visibility: C, buffer: { type: "storage" as const } }] });
  const pipeline = await device.createComputePipelineAsync({ label: "Seep observer (brick engine)", layout: device.createPipelineLayout({ bindGroupLayouts: [bindLayout] }), compute: { module, entryPoint: "observe" } });
  const words = device.createBuffer({ label: "Seep observer words (brick engine)", size: 4 * ZONE_WORDS, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const read = device.createBuffer({ label: "Seep observer readback (brick engine)", size: 4 * ZONE_WORDS, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const group = device.createBindGroup({ label: "Seep observer (brick engine)", layout: bindLayout, entries: [
    { binding: 0, resource: { buffer: e.buffer("set") } }, { binding: 1, resource: { buffer: e.buffer("dir") } },
    { binding: 2, resource: { buffer: e.buffer("phi") } }, { binding: 3, resource: { buffer: e.buffer("vel") } },
    { binding: 4, resource: { buffer: words } }] });
  const zero = new Uint32Array(ZONE_WORDS);
  return {
    geometry: g,
    async observe() {
      device.queue.writeBuffer(words, 0, zero);
      const encoder = device.createCommandEncoder({ label: "Seep observer (brick engine)" });
      const pass = encoder.beginComputePass({ label: "Seep observer (brick engine)" });
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(L.cap.f + L.cap.c); pass.end();
      encoder.copyBufferToBuffer(words, 0, read, 0, 4 * ZONE_WORDS);
      device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ); const u = new Uint32Array(read.getMappedRange().slice(0)); read.unmap();
      const sums = decodeZoneWords(u);
      sums.wallAll = undefined;
      sums.particles = (await solver.readStats()).brickEngine?.particles ?? 0;
      return sums;
    },
    destroy() { words.destroy(); read.destroy(); },
  };
}
