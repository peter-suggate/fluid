/**
 * The brick engine's procedural boundary (M3, the static solid's field): a scene's solidBoundary sampled as a field,
 * ported verbatim from uniform-solid-boundary.wgsl.ts (umBoundaryRecord, umBoundarySampleAt, umBoundarySample,
 * umBoundaryBoxOpen, umBoundaryTouchesBox) and uniform-particle-contact.wgsl.ts (nbBoundaryCubicHit, nbBoundaryHit,
 * nbProceduralContact, nbParticleBlocked's record lane, nbVoxelBlocked), for a closed or open-top scene: the frame page
 * is 0, so a brick's key is its lattice brick (the old engine's boundaryBrickOffset, 4 x its box page, is 0 here).
 *
 * The source is the old engine's own image: proceduralSolidField(scene).pack() (lib/core/procedural-solid.ts), whose
 * validity bits are the voxel edits after the generated prefix (a later geometric edit owns each cell it touches, so a
 * live edit repacks the image and the field and the edit compose, AGENTS.md). The image rides in the static `solid`
 * buffer after the mask's words and the terrain's columns (BrickStaticWalls.boundary.base): no extra binding.
 *
 * Units: lattice cells. The brick lattice is cubic (brickSceneSetup), so the old engine's metre scalings (params.hDt in
 * the contact) cancel exactly; the forms below are those with h = 1. The field-cut map the old record build keeps
 * (umSolidFieldCutAt, an acceleration) is the record-free variant's: every record cell may be cut.
 */
import { geometricPlaneBoxWGSL } from "../../core/geometric-plane-box.wgsl";
import { PROCEDURAL_SOLID_RECORD_WORDS, proceduralSolidField, sampleSolidNodes } from "../../core/procedural-solid";
import { sceneLatticeDimensions } from "../../core/scene-lattice-dimensions";
import type { SceneDescription } from "../../core/model";
import { planeBoxFraction } from "./brick-terrain.wgsl";

/** Where the kernels read the image: an array<u32> binding and the word its header is at. */
export interface BrickBoundarySource {
  readonly buffer: string;
  readonly base: number;
}

/**
 * The scene's boundary image as the old engine packs it, or undefined when the scene has none or collides with its
 * voxels (collision "voxels": the mask's half-solid voxels are the collision, the field is drawn only). A field whose
 * lattice is not the container's fails, as the old engine's build does.
 */
export function brickBoundaryImage(scene: SceneDescription): Uint32Array<ArrayBuffer> | undefined {
  const b = scene.solidBoundary;
  if (!b || b.collision === "voxels") return undefined;
  const field = proceduralSolidField(scene);
  if (!field) throw new Error("Brick engine boundary: the procedural solid's generated patches were replaced; regenerate it");
  const cells = sceneLatticeDimensions(scene);
  const h = [scene.container.width_m / cells[0]!, scene.container.height_m / cells[1]!, scene.container.depth_m / cells[2]!];
  if (b.dimensions.some((v, a) => v !== cells[a]) || b.cellSize_m.some((v, a) => Math.abs(v - h[a]!) > 1e-9))
    throw new Error("Brick engine boundary: the procedural boundary's lattice is not the container's; regenerate it at this resolution");
  return field.pack([0, 0, 0]);
}

/** geometricPlaneBoxFraction as nbBoundaryPlaneBoxFraction (the old umTerrainPlaneBoxFraction copy), its one dynamic
 * vector component write as a select (P's compile rule, as brick-terrain.wgsl.ts's copy). The same values. */
function boundaryPlaneBoxWGSL(): string {
  const text = geometricPlaneBoxWGSL.replaceAll("geometric", "nbBoundary"), write = "spans[dimensions]=value;";
  if (text.split(write).length !== 2) throw new Error("Brick engine boundary: geometricPlaneBoxWGSL changed shape");
  return text.replace(write, "spans=select(spans,vec3f(value),vec3u(dimensions)==vec3u(0u,1u,2u));");
}

/**
 * The boundary's WGSL, after brick-static.wgsl.ts's box rule and voxel reader (nbStaticInBox, nbStaticVoxel):
 * nbBoundaryRecord (the record of a cell, 0 none), nbBoundaryBrick (a brick the image holds, its cells' validity
 * aside), nbBoundarySampleAt / nbBoundarySample (gradient, value: positive in fluid), nbBoundaryBoxOpen,
 * nbBoundaryTouchesBox, nbBoundaryBlocked (nbParticleBlocked's record lane), nbVoxelBlocked and nbProceduralContact.
 */
export function brickBoundaryWGSL(src: BrickBoundarySource): string {
  if (!Number.isInteger(src.base) || src.base < 0) throw new Error(`Brick engine boundary: base word ${src.base}`);
  const w = src.buffer;
  return /* wgsl */`
const NB_BOUNDARY_BASE:u32=${src.base}u;
fn nbBoundaryHash(b:vec3i)->u32{return (u32(b.x)*73856093u)^(u32(b.y)*19349663u)^(u32(b.z)*83492791u);}
// The record word (header offset r) of brick b, 0 where the image holds none: umBoundaryRecord's probe.
fn nbBoundaryBrick(b:vec3i)->u32{
 let base=NB_BOUNDARY_BASE;let mask=${w}[base];var slot=nbBoundaryHash(b)&mask;
 for(var probe=0u;probe<=mask;probe++){
  let at=base+4u+4u*slot;let r=${w}[at+3u];if(r==0u){return 0u;}
  if(all(vec3i(bitcast<i32>(${w}[at]),bitcast<i32>(${w}[at+1u]),bitcast<i32>(${w}[at+2u]))==b)){return r;}
  slot=(slot+1u)&mask;
 }
 return 0u;
}
// umBoundaryRecord: the word of cell p's 9^3 node values, 0 outside the box, in a brick the image lacks, or in a cell a
// later voxel edit owns (its validity bit clear).
fn nbBoundaryRecord(p:vec3i)->u32{
 if(!nbStaticInBox(p)){return 0u;}
 let r=nbBoundaryBrick(p>>vec3u(3u));if(r==0u){return 0u;}
 let base=NB_BOUNDARY_BASE;let c=vec3u(p)&vec3u(7u);let i=c.x+8u*(c.y+8u*c.z);
 if((${w}[base+r+(i>>5u)]&(1u<<(i&31u)))==0u){return 0u;}
 return base+r+16u;
}
// umBoundarySampleAt: derivatives and value, the gradient in lattice coordinates.
fn nbBoundarySampleAt(q:vec3f,cell:vec3i,record:u32)->vec4f{
 let c=vec3u(cell)&vec3u(7u);let f=q-vec3f(cell);var result=vec4f(0.0);
 for(var k=0u;k<8u;k++){
  let b=vec3u(k&1u,(k>>1u)&1u,k>>2u);let n=c+b;let wt=select(1.0-f,f,b!=vec3u(0u));
  let v=bitcast<f32>(${w}[record+n.x+9u*(n.y+9u*n.z)]);
  let sign=select(vec3f(-1.0),vec3f(1.0),b!=vec3u(0u));
  result+=v*vec4f(sign.x*wt.y*wt.z,sign.y*wt.x*wt.z,sign.z*wt.x*wt.y,wt.x*wt.y*wt.z);
 }
 return result;
}
// umBoundarySample: a cell with no record is solid (-1e6) if a voxel (the box rule's outside included), else open.
fn nbBoundarySample(q:vec3f)->vec4f{
 let c=vec3i(floor(q));let r=nbBoundaryRecord(c);
 if(r==0u){return vec4f(0.0,0.0,0.0,select(1e6,-1e6,nbStaticVoxel(c)));}
 return nbBoundarySampleAt(q,c,r);
}
${boundaryPlaneBoxWGSL()}
// umBoundaryBoxOpen (umUnitBoxOpen's boundary lane): eight octant planes.
fn nbBoundaryBoxOpen(q:vec3f,width:vec3f)->f32{
 var open=0.0;
 for(var k=0u;k<8u;k++){
  let bit=vec3u(k&1u,(k>>1u)&1u,k>>2u);let at=q+select(vec3f(-0.25),vec3f(0.25),bit!=vec3u(0u))*width;
  let s=nbBoundarySample(at);
  open+=0.125*nbBoundaryPlaneBoxFraction(s.xyz,s.w,width*0.5);
 }
 return open;
}
// umBoundaryTouchesBox: a record under one of the box's eight octant samples.
fn nbBoundaryTouchesBox(q:vec3f,width:vec3f)->bool{
 for(var k=0u;k<8u;k++){
  let b=vec3u(k&1u,(k>>1u)&1u,k>>2u);
  if(nbBoundaryRecord(vec3i(floor(q+select(vec3f(-0.25),vec3f(0.25),b!=vec3u(0u))*width)))!=0u){return true;}
 }
 return false;
}
// nbParticleBlocked's static part with a record: the field's sign in a record cell, else the voxel.
fn nbBoundaryBlocked(q:vec3f)->bool{
 let c=vec3i(floor(q));let record=nbBoundaryRecord(c);
 if(record!=0u){return nbBoundarySampleAt(q,c,record).w<0.0;}
 return nbStaticVoxel(c);
}
// nbVoxelBlocked: a voxel with no record (the box rule's outside included).
fn nbVoxelBlocked(q:vec3f)->bool{let c=vec3i(floor(q));return nbStaticVoxel(c)&&nbBoundaryRecord(c)==0u;}
struct NbBoundaryContact{position:vec3f,velocity:vec3f}
// nbBoundaryCubicHit: the earliest zero of a cubic in Bernstein form (depth 14), a depth-first walk of the halvings.
fn nbBoundaryCubicHit(initial:vec4f)->f32{
 var b=initial;var code=0u;var depth=0u;
 loop{
  if(min(min(b.x,b.y),min(b.z,b.w))>0.0){
   loop{if(depth==0u||(code&1u)==0u){break;}code>>=1u;depth--;}
   if(depth==0u){return -1.0;}
   code|=1u;b=initial;
   for(var k=depth;k>0u;k--){
    let a=0.5*(b.xyz+b.yzw);let c=0.5*(a.xy+a.yz);let mid=0.5*(c.x+c.y);
    b=select(vec4f(b.x,a.x,c.x,mid),vec4f(mid,c.y,a.z,b.w),((code>>(k-1u))&1u)!=0u);
   }
   continue;
  }
  if(b.x<=0.0||depth==14u){return f32(code)/f32(1u<<depth);}
  let a=0.5*(b.xyz+b.yzw);let c=0.5*(a.xy+a.yz);let mid=0.5*(c.x+c.y);
  b=vec4f(b.x,a.x,c.x,mid);code<<=1u;depth++;
 }
}
// nbBoundaryHit: the segment's first crossing of the field, cell by cell (a half-cell stride crosses at most three
// faces). No dynamic component write: the faces read the axis by select.
fn nbBoundaryHit(start:vec3f,end:vec3f)->f32{
 let delta=end-start;var t=0.0;
 for(var step=0u;step<5u&&t<1.0;step++){
  let c=vec3i(floor(mix(start,end,t)+sign(delta)*1e-6));var stop=1.0;
  for(var a=0u;a<3u;a++){
   let da=select(select(delta.x,delta.y,a==1u),delta.z,a==2u);
   if(abs(da)<1e-12){continue;}
   let ca=select(select(c.x,c.y,a==1u),c.z,a==2u);let sa=select(select(start.x,start.y,a==1u),start.z,a==2u);
   let face=f32(ca)+select(0.0,1.0,da>0.0);
   let crossing=(face-sa)/da;if(crossing>t+1e-7){stop=min(stop,crossing);}
  }
  let record=nbBoundaryRecord(c);
  if(record!=0u){
   let p=mix(start,end,t);let d=(stop-t)*delta;
   let f0=nbBoundarySampleAt(p,c,record).w;let f1=nbBoundarySampleAt(p+d/3.0,c,record).w;
   let f2=nbBoundarySampleAt(p+2.0*d/3.0,c,record).w;let f3=nbBoundarySampleAt(p+d,c,record).w;
   let hit=nbBoundaryCubicHit(vec4f(f0,3.0*f1-1.5*f2-5.0*f0/6.0+f3/3.0,3.0*f2-1.5*f1-5.0*f3/6.0+f0/3.0,f3));
   if(hit>=0.0){return mix(t,stop,hit);}
  }
  t=stop;
 }
 return -1.0;
}
// nbProceduralContact, in cells: a segment that crosses the surface slides along it from the hit; then, crossed or
// not, a centre nearer the surface than its radius (inset) moves out along the normal, losing its velocity into the
// surface. A correction that would leave the box stops at the hit (umPointPresent).
fn nbProceduralContact(start:vec3f,end:vec3f,velocity:vec3f,inset:f32)->NbBoundaryContact{
 let hit=nbBoundaryHit(start,end);var at=end;var q=end;var v=velocity;
 if(hit>=0.0){
  at=mix(start,end,hit);let s=nbBoundarySample(at);let gradient=s.xyz;
  if(dot(gradient,gradient)<1e-12){return NbBoundaryContact(start,velocity);}
  let n=normalize(gradient);var travel=end-at;
  travel-=min(dot(travel,n),0.0)*n;q=at+travel;
  v=velocity-min(dot(velocity,n),0.0)*n;
 }
 for(var i=0u;i<4u;i++){
  let sample=nbBoundarySample(q);let g=sample.xyz;let lengthG=length(g);
  if(lengthG<1e-12){break;}let normal=g/lengthG;
  let depth=inset-sample.w/lengthG;
  if(depth<=0.0){break;}q+=depth*normal;v-=min(dot(v,normal),0.0)*normal;
 }
 if(!nbStaticInBox(vec3i(floor(q)))){return NbBoundaryContact(at,v);}
 return NbBoundaryContact(q,v);
}
`;
}

/** The bricks the boundary can cut (solidStamp's rule: a cell's box or a lower face's box reaches a record cell): each
 * image brick and the bricks just past its upper faces, in [0, box) bricks. K's static count (brickStaticCutBricks). */
export function brickBoundaryCutBricks(image: Uint32Array, box: readonly number[], add: (x: number, y: number, z: number) => void): void {
  const B = box.map(n => Math.ceil(n / 8)), slots = image[0]! + 1;
  for (let s = 0; s < slots; s++) {
    const at = 4 + 4 * s; if (image[at + 3] === 0) continue;
    const b = [image[at]! | 0, image[at + 1]! | 0, image[at + 2]! | 0];
    for (const d of [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]) {
      const q = b.map((v, a) => v + d[a]!);
      if (q.every((v, a) => v >= 0 && v < B[a]!)) add(q[0]!, q[1]!, q[2]!);
    }
  }
}

/** The cells whose validity bit differs between two images of the same field (a live edit's), in lattice cells. */
export function brickBoundaryChangedCells(before: Uint32Array, after: Uint32Array, add: (x: number, y: number, z: number) => void): void {
  if (before.length !== after.length || before[0] !== after[0]) throw new Error("Brick engine boundary: the two images hold different bricks");
  const slots = before[0]! + 1;
  for (let s = 0; s < slots; s++) {
    const at = 4 + 4 * s, r = before[at + 3]!; if (r === 0) continue;
    if (after[at + 3] !== r || after[at] !== before[at] || after[at + 1] !== before[at + 1] || after[at + 2] !== before[at + 2]) throw new Error("Brick engine boundary: the two images hold different bricks");
    for (let w = 0; w < 16; w++) {
      let moved = (before[r + w]! ^ after[r + w]!) >>> 0;
      while (moved) {
        const bit = 31 - Math.clz32(moved); moved = (moved & ~(1 << bit)) >>> 0;
        const i = 32 * w + bit;
        add(8 * (before[at]! | 0) + (i & 7), 8 * (before[at + 1]! | 0) + ((i >> 3) & 7), 8 * (before[at + 2]! | 0) + (i >> 6));
      }
    }
  }
}

/** The Float64 mirror over the same image words (the test oracle): record, sample, boxOpen, touchesBox, blocked.
 * `inBox` and `voxel` are the static mirror's box rule and voxel reader. */
export function brickBoundaryMirror(image: Uint32Array, inBox: (c: readonly number[]) => boolean, voxel: (c: readonly number[]) => boolean) {
  const f32 = new Float32Array(image.buffer, image.byteOffset, image.length), mask = image[0]!;
  const hash = (x: number, y: number, z: number) => (Math.imul(x, 73856093) ^ Math.imul(y, 19349663) ^ Math.imul(z, 83492791)) >>> 0;
  const brick = (b: readonly number[]) => {
    let slot = hash(b[0]!, b[1]!, b[2]!) & mask;
    for (let probe = 0; probe <= mask; probe++) {
      const at = 4 + 4 * slot, r = image[at + 3]!; if (r === 0) return 0;
      if ((image[at]! | 0) === b[0] && (image[at + 1]! | 0) === b[1] && (image[at + 2]! | 0) === b[2]) return r;
      slot = (slot + 1) & mask;
    }
    return 0;
  };
  const record = (p: readonly number[]) => {
    if (!inBox(p)) return 0;
    const r = brick(p.map(v => v >> 3)); if (r === 0) return 0;
    const c = p.map(v => v & 7), i = c[0]! + 8 * (c[1]! + 8 * c[2]!);
    if ((image[r + (i >> 5)]! & (1 << (i & 31))) === 0) return 0;
    return r + 16;
  };
  const sampleAt = (q: readonly number[], cell: readonly number[], r: number) => {
    const values = f32.subarray(r, r + PROCEDURAL_SOLID_RECORD_WORDS - 16);
    return sampleSolidNodes(values, q.map((v, a) => v - (cell[a]! - (cell[a]! & 7))) as [number, number, number]);
  };
  const sample = (q: readonly number[]): readonly number[] => {
    const c = q.map(Math.floor), r = record(c);
    if (r === 0) return [0, 0, 0, voxel(c) ? -1e6 : 1e6];
    return sampleAt(q, c, r);
  };
  const boxOpen = (q: readonly number[], width: readonly number[]) => {
    let open = 0;
    for (let k = 0; k < 8; k++) {
      const at = q.map((v, a) => v + (((k >> a) & 1) ? 0.25 : -0.25) * width[a]!), s = sample(at);
      open += 0.125 * planeBoxFraction([s[0]!, s[1]!, s[2]!], s[3]!, width.map(v => 0.5 * v));
    }
    return open;
  };
  const touchesBox = (q: readonly number[], width: readonly number[]) => {
    for (let k = 0; k < 8; k++) if (record(q.map((v, a) => Math.floor(v + (((k >> a) & 1) ? 0.25 : -0.25) * width[a]!))) !== 0) return true;
    return false;
  };
  const blocked = (q: readonly number[]) => { const c = q.map(Math.floor), r = record(c); return r !== 0 ? sampleAt(q, c, r)[3]! < 0 : voxel(c); };
  return { record, sample, boxOpen, touchesBox, blocked };
}
