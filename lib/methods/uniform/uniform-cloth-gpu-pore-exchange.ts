import type { ClothWaterLedger } from '../../cloth/definition';
import type { ClothFrameIdentity } from '../../cloth/types';
import { CLOTH_PORE_MATERIAL_FLOATS, ClothGPUPoreTransport, clothPoreMaterialTable, clothTimestampInterval_ms, type ClothPoreGPUResult, type ClothPoreGPUStep } from '../../cloth/gpu-pore-transport';
import type { UniformClothGridFields } from './uniform-cloth-grid-coupling';
import { uniformDetailBindLayout, uniformDetailGroup, uniformDetailModule, uniformDetailPipeline } from './uniform-detail-fields';
import { UniformGPURadixSort, UniformGPUScan, type UniformGPUOrderingKernels } from './uniform-gpu-radix-sort';
import { uniformMixedSolidWGSL } from './uniform-mixed-solid.wgsl';
import { uniformCompensatedSumWGSL } from './uniform-compensated-sum.wgsl';
import { NB_VOLUME_COMPENSATION_FLOAT } from './uniform-mixed-surface-volume';
import { uniformMixedTopologyWGSL } from './uniform-mixed-topology.wgsl';

/**
 * The cloth's per-step water exchange with the production NB-FLIP fields,
 * on the GPU, around the GPU manifold transport, in one submit and one
 * readback: the native support sample at the accepted positions, Darcy
 * capture from the local free reservoir, the manifold transport, release
 * of compressed overflow, the material capacity table NB-FLIP reads, and
 * the paired water-budget commit.
 *
 * UniformClothWaterExchange's sampleSupport, captureFromSample,
 * releaseFromSample and commitPoreWater (with prepareClothPorousCapacity)
 * are the f64 oracles, kept for parity tests. Pore water stays in the
 * transport's int32 quanta throughout: capture and release move whole
 * quanta, so the bound inventory and the budget delta are exact integers.
 *
 * Each vertex reads the 32 cells of its capture support: its eight
 * trilinear cells and their face neighbours (cells b + o, o in -1..2 with
 * at most one coordinate outside 0..1). Competing capture requests on a
 * shared cell are summed exactly in a hash table of 64-bit fixed-point
 * quanta, so the donor limiter is order independent. The capacity table is
 * sorted z, y, x, members summed in vertex-major order, as the oracle.
 */
export const CLOTH_EXCHANGE_STATUS = { unowned: 1, coverage: 2, noOpenSupport: 4, fillsCell: 8, hashFull: 16, quantum: 32, permeability: 64 } as const;
const STATUS_MESSAGES: readonly [number, string][] = [
  [CLOTH_EXCHANGE_STATUS.unowned, 'Cloth support requires h ownership at its sampled cells'],
  [CLOTH_EXCHANGE_STATUS.coverage, 'Cloth support hard-geometry integration did not converge'],
  [CLOTH_EXCHANGE_STATUS.noOpenSupport, 'Cloth has no open local cell support; collision/admission must be resolved'],
  [CLOTH_EXCHANGE_STATUS.fillsCell, 'Cloth material fills a cell: refine the actual porous support'],
  [CLOTH_EXCHANGE_STATUS.hashFull, 'Cloth capture cell table is full'],
  [CLOTH_EXCHANGE_STATUS.quantum, 'Captured pore water exceeds the int32 quantum range'],
  [CLOTH_EXCHANGE_STATUS.permeability, 'Fiber permeability fit is not positive at this solid fraction'],
];
const SLOTS = 32, STATUS_BYTES = 32, ALIGN = 64;
const pad = (words: number) => Math.ceil(Math.max(1, words) / ALIGN) * ALIGN;
/** The capture support offsets, z-, y-, x-major. */
const SLOT_OFFSETS: readonly [number, number, number][] = (() => {
  const out: [number, number, number][] = [];
  for (let z = -1; z <= 2; z++) for (let y = -1; y <= 2; y++) for (let x = -1; x <= 2; x++)
    if ([x, y, z].filter(a => a < 0 || a > 1).length <= 1) out.push([x, y, z]);
  if (out.length !== SLOTS) throw new Error('Cloth capture support must have 32 cells');
  return out;
})();
const SLOT_OF = Array.from({ length: 64 }, (_, i) => { const o = [i % 4 - 1, (i >> 2) % 4 - 1, (i >> 4) - 1]; const s = SLOT_OFFSETS.findIndex(p => p.every((a, k) => a === o[k])); return s < 0 ? 255 : s; });

export function clothExchangeSections(V: number) {
  const C = pad(8 * V), R = pad(SLOTS * V);
  return { weight: 0, porosity: C, request: 2 * C, keep: 2 * C + R, positions: 3 * C + R, key: 4 * C + R, starts: 5 * C + R, words: 6 * C + R };
}

const params = /* wgsl */`
struct Params {
 frame:vec4i,   // key frame origin xyz, vertex count
 dims:vec4u,    // key frame dims xyz, hash mask
 scalars:vec4f, // water quantum, cell volume, smallest cell edge, dt
 flags:vec4f,   // capture-rate multiplier, water density, absorption, solveSolid
};
const SLOT=array<vec3i,${SLOTS}>(${SLOT_OFFSETS.map(o => `vec3i(${o.join(',')})`).join(',')});
const SLOT_OF=array<u32,64>(${SLOT_OF.map(s => `${s}u`).join(',')});
fn slotOf(o:vec3i)->u32{return SLOT_OF[u32(o.x+1)+4u*u32(o.y+1)+16u*u32(o.z+1)];}
fn corner(c:u32)->vec3i{return vec3i(i32(c&1u),i32((c>>1u)&1u),i32((c>>2u)&1u));}
// Trilinear weight, in the oracle's product order.
fn cornerWeight(f:vec3f,c:u32)->f32{let o=corner(c);return select(1.0-f.x,f.x,o.x==1)*select(1.0-f.y,f.y,o.y==1)*select(1.0-f.z,f.z,o.z==1);}
`;

// Native support sample: one thread a (vertex, support cell).
const sampleCode = /* wgsl */`
${params}
@group(1) @binding(0) var<uniform> params:Params;
@group(1) @binding(1) var<storage,read> cells:array<vec4f>;
@group(1) @binding(2) var<storage,read_write> samples:array<vec4f>;
@group(1) @binding(3) var<storage,read_write> status:array<atomic<u32>>;
@group(1) @binding(4) var phi:texture_3d<f32>;
@group(1) @binding(5) var velocity:texture_3d<f32>;
fn tetraFill(x:vec4f)->f32{
 var neg=array<f32,4>(0.0,0.0,0.0,0.0);var pos=array<f32,4>(0.0,0.0,0.0,0.0);var n=0u;var m=0u;
 for(var k=0u;k<4u;k++){if(x[k]<0.0){neg[n]=x[k];n++;}else{pos[m]=x[k];m++;}}
 if(n==0u){return 0.0;}if(m==0u){return 1.0;}
 if(n==1u){var p=1.0;for(var k=0u;k<m;k++){p*=-neg[0]/(pos[k]-neg[0]);}return p;}
 if(n==3u){var p=1.0;for(var k=0u;k<3u;k++){p*=pos[0]/(pos[0]-neg[k]);}return 1.0-p;}
 let a=-neg[0]/(pos[0]-neg[0]);let b=-neg[0]/(pos[1]-neg[0]);let c=-neg[1]/(pos[0]-neg[1]);let d=-neg[1]/(pos[1]-neg[1]);
 return max(0.0,min(1.0,a*b+b*c*(1.0-a)+c*d*(1.0-b)));
}
// The production surface-volume authority's six tetrahedra (tetrahedralWaterFill).
fn cellFill(v:array<f32,8>)->f32{
 return (tetraFill(vec4f(v[0],v[1],v[3],v[7]))+tetraFill(vec4f(v[0],v[1],v[5],v[7]))+tetraFill(vec4f(v[0],v[2],v[3],v[7]))
  +tetraFill(vec4f(v[0],v[2],v[6],v[7]))+tetraFill(vec4f(v[0],v[4],v[5],v[7]))+tetraFill(vec4f(v[0],v[4],v[6],v[7])))/6.0;
}
fn fail(bit:u32,v:u32){atomicOr(&status[0],bit);atomicMax(&status[1],0xffffffffu-v);}
// (aperture, tetrahedral fill, occupied material fraction, live) and the
// cell's MAC velocity. Cells no live corner reaches are not sampled.
@compute @workgroup_size(64) fn sampleSupport(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;let V=u32(params.frame.w);if(i>=${SLOTS}u*V){return;}let v=i/${SLOTS}u;let s=i%${SLOTS}u;
 let base=bitcast<vec4i>(cells[2u*v]).xyz;let f=cells[2u*v+1u].xyz;let o=SLOT[s];
 var live=false;for(var c=0u;c<8u;c++){let d=abs(o-corner(c));if(d.x+d.y+d.z<=1&&cornerWeight(f,c)>0.0){live=true;}}
 samples[2u*i]=vec4f(0.0);samples[2u*i+1u]=vec4f(0.0);if(!live){return;}
 let p=base+o-32*umFramePage();
 var corners:array<f32,8>;for(var c=0u;c<8u;c++){let at=p+corner(c);corners[c]=/*c*/textureLoad(phi,at,0).x;}
 let fill=cellFill(corners);let u=/*c*/textureLoad(velocity,p,0).xyz;
 // A cell beyond the world box's walls is solid: closed, not unowned.
 var open=0.0;
 if(!umWorldSolid((p>>vec3u(5u))+umFramePage())){
  let tile=512u*umBlockOf(p>>vec3u(5u))+umTileLocal(vec3u(p>>vec3u(2u)));
  if(!umKeyStored(tile)||umTileWidth(tile)!=1u){fail(${CLOTH_EXCHANGE_STATUS.unowned}u,v);return;}
  open=umCellOpen(p);if(umRigidCoverageFailure.x!=0u){fail(${CLOTH_EXCHANGE_STATUS.coverage}u,v);return;}
 }
 samples[2u*i]=vec4f(open,fill,dot(umPorousFractions(p),vec2f(1.0)),1.0);samples[2u*i+1u]=vec4f(u,0.0);
}`;

const exchangeCode = (V: number) => {
  const s = clothExchangeSections(V);
  return /* wgsl */`
${params}
${uniformCompensatedSumWGSL}
@group(0) @binding(0) var<uniform> params:Params;
// Two vec4 a vertex: (base cell, material) bits, (cell fraction, sub-quantum water).
@group(0) @binding(1) var<storage,read> cells:array<vec4f>;
@group(0) @binding(2) var<storage,read> samples:array<vec4f>;
// The transport's vertex records: (position, control volume), pressure and
// solid acceleration, (dry mass, solid volume, solid fraction, fixed).
@group(0) @binding(3) var<storage,read_write> vertices:array<vec4f>;
@group(0) @binding(4) var<storage,read_write> water:array<i32>;
@group(0) @binding(5) var<storage,read_write> velocities:array<vec4f>;
@group(0) @binding(6) var<storage,read_write> work:array<u32>;
@group(0) @binding(7) var<storage,read_write> table:array<atomic<u32>>;
// Status bits, first failing vertex, captured and released quanta (u64 lo, hi).
@group(0) @binding(8) var<storage,read_write> status:array<atomic<u32>>;
@group(0) @binding(9) var<storage,read> materials:array<f32>;
@group(0) @binding(10) var<storage,read_write> sortKeys:array<u32>;
@group(0) @binding(11) var<storage,read_write> sortValues:array<u32>;
@group(0) @binding(12) var<storage,read_write> counts:array<atomic<u32>>;
@group(0) @binding(13) var<storage,read_write> capacity:array<vec4f>;
@group(0) @binding(14) var<storage,read_write> budget:array<f32>;
const W_=${s.weight}u;const P_=${s.porosity}u;const R_=${s.request}u;const KEEP_=${s.keep}u;const POS_=${s.positions}u;const KEY_=${s.key}u;const START_=${s.starts}u;
fn V()->u32{return u32(params.frame.w);}
fn q()->f32{return params.scalars.x;}
fn cellVolume()->f32{return params.scalars.y;}
fn fail(bit:u32,v:u32){atomicOr(&status[0],bit);atomicMax(&status[1],0xffffffffu-v);}
fn baseOf(v:u32)->vec3i{return bitcast<vec4i>(cells[2u*v]).xyz;}
fn materialOf(v:u32)->u32{return bitcast<vec4u>(cells[2u*v]).w;}
fn residual(v:u32)->f32{return cells[2u*v+1u].w;}
fn controlVolume(v:u32)->f32{return vertices[4u*v].w;}
fn dryMass(v:u32)->f32{return vertices[4u*v+3u].x;}
fn solidVolume(v:u32)->f32{return vertices[4u*v+3u].y;}
fn solidFraction(v:u32)->f32{return vertices[4u*v+3u].z;}
fn fixedVertex(v:u32)->bool{return vertices[4u*v+3u].w!=0.0;}
fn poreCapacity(v:u32)->f32{return max(0.0,controlVolume(v)-solidVolume(v));}
fn waterVolume(v:u32)->f32{return f32(water[v])*q()+residual(v);}
fn sampleAt(v:u32,s:u32)->vec4f{return samples[2u*(${SLOTS}u*v+s)];}
fn velocityAt(v:u32,s:u32)->vec3f{return samples[2u*(${SLOTS}u*v+s)+1u].xyz;}
fn material(m:u32,k:u32)->f32{return materials[${CLOTH_PORE_MATERIAL_FLOATS}u*m+k];}
fn getF(at:u32)->f32{return bitcast<f32>(work[at]);}
fn setF(at:u32,x:f32){work[at]=bitcast<u32>(x);}
fn frameKey(g:vec3i)->u32{let l=vec3u(g-params.frame.xyz);return l.x+params.dims.x*(l.y+params.dims.y*l.z);}
fn moving()->bool{return params.flags.w!=0.0;}
fn add64(at:u32,x:u32){let old=atomicAdd(&status[at],x);if(old+x<old){atomicAdd(&status[at+1u],1u);}}
// Shared-cell demand: open addressing on (frame key + 1), 64-bit sums.
fn demandSlot(key:u32,insert:bool,v:u32)->u32{
 let mask=params.dims.w;var h=(key*0x9e3779b1u)&mask;
 for(var n=0u;n<=mask;n++){
  let found=atomicLoad(&table[3u*h]);
  if(found==key+1u){return h;}
  if(found==0u){
   if(!insert){return 0xffffffffu;}
   let r=atomicCompareExchangeWeak(&table[3u*h],0u,key+1u);
   if(r.exchanged||r.old_value==key+1u){return h;}
   if(r.old_value==0u){n--;continue;}
  }
  h=(h+1u)&mask;
 }
 fail(${CLOTH_EXCHANGE_STATUS.hashFull}u,v);return 0xffffffffu;
}
fn demandOf(h:u32)->f32{return f32(atomicLoad(&table[3u*h+2u]))*4294967296.0+f32(atomicLoad(&table[3u*h+1u]));}
// porousCaptureFraction: across-fiber permeability and suction, grid h.
fn captureFraction(v:u32,f:f32)->f32{
 let m=materialOf(v);let d=material(m,2u);let mu=material(m,3u);let logTerm=-log(f)-1.476+2.0*f;
 let along=(logTerm-0.5*f*f)*d*d/(16.0*f);let across=(logTerm-1.774*f*f+4.078*f*f*f)*d*d/(32.0*f);
 if(!(along>0.0&&across>0.0)){fail(${CLOTH_EXCHANGE_STATUS.permeability}u,v);return 0.0;}
 let suction=0.5*material(m,4u)*f/(1.0-f);let h=params.scalars.z;
 return min(1.0,params.flags.x*params.scalars.w*across*max(0.0,suction)/(mu*h*h));
}
// Fill-weighted cell-centred MAC velocity over the control's kernel.
fn fluidVelocity(v:u32,fallback:vec3f)->vec3f{
 var u=vec3f(0.0);var weight=0.0;
 for(var c=0u;c<8u;c++){
  let wc=getF(W_+8u*v+c);if(!(wc>0.0)){continue;}let s=slotOf(corner(c));let w=wc*sampleAt(v,s).y;if(!(w>0.0)){continue;}weight+=w;
  let own=velocityAt(v,s);
  for(var a=0u;a<3u;a++){var low=corner(c);low[a]-=1;u[a]+=w*0.5*(own[a]+velocityAt(v,slotOf(low))[a]);}
 }
 return select(fallback,u/weight,weight>0.0);
}
fn freeVolume(v:u32,s:u32)->f32{let x=sampleAt(v,s);return x.y*x.x*cellVolume()*max(0.0,1.0-x.z);}

// clothCellWeights at the accepted position, then the capture requests:
// rate times the MAC reservoir weight times the free reservoir, scaled to
// the room, deposited as whole (rounded up) quanta on the shared cell.
@compute @workgroup_size(64) fn vertexPlan(@builtin(global_invocation_id) gid:vec3u){
 let v=gid.x;if(v>=V()){return;}let f=cells[2u*v+1u].xyz;
 var w:array<f32,8>;var hard:array<f32,8>;var normalization=0.0;
 for(var c=0u;c<8u;c++){w[c]=cornerWeight(f,c);hard[c]=sampleAt(v,slotOf(corner(c))).x;if(w[c]>0.0){normalization+=w[c]*hard[c];}}
 for(var s=0u;s<${SLOTS}u;s++){setF(R_+${SLOTS}u*v+s,0.0);}
 let supported=normalization>1e-12;if(!supported){fail(${CLOTH_EXCHANGE_STATUS.noOpenSupport}u,v);}
 for(var c=0u;c<8u;c++){let keep=supported&&w[c]>0.0&&hard[c]>0.0;
  setF(W_+8u*v+c,select(0.0,w[c]*hard[c]/normalization,keep));setF(P_+8u*v+c,select(0.0,w[c]/normalization,keep));}
 if(!supported||params.flags.z==0.0){return;}
 let room=poreCapacity(v)-waterVolume(v);let fraction=solidFraction(v);
 if(!(room>0.0)||!(fraction>0.0&&fraction<1.0)){return;}
 let rate=captureFraction(v,fraction);if(!(rate>0.0)){return;}
 var reservoir:array<f32,${SLOTS}>;for(var s=0u;s<${SLOTS}u;s++){reservoir[s]=0.0;}
 for(var c=0u;c<8u;c++){let wc=getF(W_+8u*v+c);if(!(wc>0.0)){continue;}
  for(var a=0u;a<3u;a++){for(var k=-1;k<=1;k++){var o=corner(c);o[a]+=k;reservoir[slotOf(o)]+=0.25*wc*select(1.0,2.0,k==0);}}}
 var request:array<f32,${SLOTS}>;var total=0.0;
 for(var s=0u;s<${SLOTS}u;s++){request[s]=0.0;if(!(reservoir[s]>0.0)){continue;}let volume=rate*reservoir[s]*freeVolume(v,s);if(volume>0.0){request[s]=volume;total+=volume;}}
 let scale=select(1.0,room/total,total>room);let base=baseOf(v);
 for(var s=0u;s<${SLOTS}u;s++){
  if(!(request[s]>0.0)){continue;}let r=request[s]*scale;setF(R_+${SLOTS}u*v+s,r);
  let h=demandSlot(frameKey(base+SLOT[s]),true,v);if(h==0xffffffffu){continue;}
  let n=u32(min(ceil(r/q()),4294967040.0));let old=atomicAdd(&table[3u*h+1u],n);if(old+n<old){atomicAdd(&table[3u*h+2u],1u);}
 }
}
// planPoreAbsorption: donor-limited by each shared cell's free reservoir,
// then receiver-limited by the room; whole quanta captured, with the
// captured water's fluid momentum.
@compute @workgroup_size(64) fn capture(@builtin(global_invocation_id) gid:vec3u){
 let v=gid.x;if(v>=V()||params.flags.z==0.0){return;}let base=baseOf(v);
 var receiver=0.0;
 for(var s=0u;s<${SLOTS}u;s++){
  let r=getF(R_+${SLOTS}u*v+s);if(!(r>0.0)){continue;}let h=demandSlot(frameKey(base+SLOT[s]),false,v);if(h==0xffffffffu){continue;}
  let limited=r*min(1.0,(freeVolume(v,s)/q())/max(demandOf(h),1e-30));setF(R_+${SLOTS}u*v+s,limited);receiver+=limited;
 }
 if(!(receiver>0.0)){return;}
 let before=waterVolume(v);let room=max(0.0,poreCapacity(v)-before);let factor=min(1.0,room/receiver);
 var accepted=0.0;for(var s=0u;s<${SLOTS}u;s++){let r=getF(R_+${SLOTS}u*v+s);if(r>0.0){accepted+=r*factor;}}
 let n=i32(floor(accepted/q()));if(n<=0){return;}
 if(!(f32(water[v])+f32(n)<2147483520.0)){fail(${CLOTH_EXCHANGE_STATUS.quantum}u,v);return;}
 if(moving()&&!fixedVertex(v)){
  let old=velocities[v].xyz;let u=fluidVelocity(v,old);let rho=params.flags.y;let mass=dryMass(v)+rho*before;let added=rho*f32(n)*q();
  let next=(mass*old+added*u)/(mass+added);velocities[v]=vec4f(next,velocities[v].w);
  // The pore frame's solid acceleration includes the captured momentum, as the oracle's.
  vertices[4u*v+2u]=vec4f(vertices[4u*v+2u].xyz+(next-old)/params.scalars.w,vertices[4u*v+2u].w);
 }
 water[v]+=n;add64(2u,u32(n));
}
// Compressed overflow seeps into the wetted kernel's free liquid, in whole
// quanta never more than the vertex holds, with its momentum.
@compute @workgroup_size(64) fn release(@builtin(global_invocation_id) gid:vec3u){
 let v=gid.x;if(v>=V()){return;}
 let held=i32(ceil(max(0.0,poreCapacity(v)-residual(v))/q()));let overflow=water[v]-held;if(overflow<=0){return;}
 var wet=0.0;for(var c=0u;c<8u;c++){let wc=getF(W_+8u*v+c);if(wc>0.0){wet+=wc*sampleAt(v,slotOf(corner(c))).y;}}
 let share=min(1.0,max(0.0,wet));
 // The share is below one in f32, so the product never exceeds the overflow.
 let n=select(i32(floor(f32(overflow)*share)),overflow,share>=1.0);if(n<=0){return;}
 if(moving()&&!fixedVertex(v)){
  let old=velocities[v].xyz;let u=fluidVelocity(v,old);let rho=params.flags.y;let mass=dryMass(v)+rho*waterVolume(v);let removed=rho*f32(n)*q();
  velocities[v]=vec4f((mass*old-removed*u)/(mass-removed),velocities[v].w);
 }
 water[v]-=n;add64(4u,u32(n));
}
// prepareClothPorousCapacity: every open corner of a vertex with material.
@compute @workgroup_size(64) fn capacityEntries(@builtin(global_invocation_id) gid:vec3u){
 let v=gid.x;if(v>=V()){return;}let material=solidVolume(v)+waterVolume(v)>0.0;let base=baseOf(v);
 for(var c=0u;c<8u;c++){let e=8u*v+c;let keep=material&&getF(P_+e)>0.0;
  work[KEY_+e]=frameKey(base+corner(c));work[KEEP_+e]=select(0u,1u,keep);work[POS_+e]=select(0u,1u,keep);}
}
@compute @workgroup_size(64) fn compactEntries(@builtin(global_invocation_id) gid:vec3u){
 let e=gid.x;let n=8u*V();if(e>=n){return;}let at=work[POS_+e];let keep=work[KEEP_+e]!=0u;
 if(keep){sortKeys[at]=work[KEY_+e];sortValues[at]=e;}
 if(e==n-1u){atomicStore(&counts[0],at+select(0u,1u,keep));}
}
fn cellStart(j:u32)->bool{return j==0u||sortKeys[j]!=sortKeys[j-1u];}
@compute @workgroup_size(64) fn markCells(@builtin(global_invocation_id) gid:vec3u){
 let j=gid.x;if(j>=8u*V()){return;}work[START_+j]=select(0u,1u,j<atomicLoad(&counts[0])&&cellStart(j));
}
// The sorted sparse table (uniformPorousCapacityWGSL): header count, then
// a cell's coordinate bits and (fiber, bound) fractions.
@compute @workgroup_size(64) fn writeCells(@builtin(global_invocation_id) gid:vec3u){
 let j=gid.x;let n=atomicLoad(&counts[0]);if(j==0u&&n==0u){capacity[0]=vec4f(0.0);}if(j>=n){return;}
 let cell=work[START_+j]+select(0u,1u,cellStart(j))-1u;
 if(j==n-1u){atomicStore(&counts[1],cell+1u);capacity[0]=bitcast<vec4f>(vec4u(cell+1u,0u,0u,0u));}
 if(!cellStart(j)){return;}
 let key=sortKeys[j];let volume=cellVolume();var solid=0.0;var bound=0.0;
 for(var k=j;k<n&&sortKeys[k]==key;k++){let e=sortValues[k];let v=e/8u;let w=getF(P_+e);solid+=solidVolume(v)*w/volume;bound+=waterVolume(v)*w/volume;}
 let dims=params.dims.xyz;let p=vec3i(vec3u(key%dims.x,(key/dims.x)%dims.y,key/(dims.x*dims.y)))+params.frame.xyz;
 if(solid+bound>=1.0){fail(${CLOTH_EXCHANGE_STATUS.fillsCell}u,sortValues[j]/8u);}
 capacity[1u+2u*cell]=bitcast<vec4f>(vec4i(p,0));capacity[2u+2u*cell]=vec4f(solid,bound,0.0,0.0);
}
// The paired budget commit (released minus captured quanta), into the NB
// budget's compensated (high, low) pair as source emission commits.
@compute @workgroup_size(1) fn commitBudget(){
 let capturedLo=atomicLoad(&status[2]);let capturedHi=atomicLoad(&status[3]);let releasedLo=atomicLoad(&status[4]);let releasedHi=atomicLoad(&status[5]);
 var lo=releasedLo-capturedLo;var hi=releasedHi-capturedHi-select(0u,1u,releasedLo<capturedLo);
 if(lo==0u&&hi==0u){return;}
 // Convert the magnitude: a negative two's-complement pair would cancel in f32.
 let negative=(hi&0x80000000u)!=0u;
 if(negative){lo=~lo+1u;hi=~hi+select(0u,1u,lo==0u);}
 let magnitude=f32(hi)*4294967296.0+f32(lo);
 let delta=select(magnitude,-magnitude,negative)*q()/cellVolume();
 // A clamp at zero leaves the pair's sum unchanged: high's deficit goes to low.
 var next=umCompensatedAdd(vec2f(budget[0],budget[${NB_VOLUME_COMPENSATION_FLOAT}]),delta);if(next.x<0.0){next=vec2f(0.0,next.y+next.x);}
 budget[0]=next.x;budget[${NB_VOLUME_COMPENSATION_FLOAT}]=next.y;budget[3]=0.0;
}`;
};

type Kernel = 'vertexPlan' | 'capture' | 'release' | 'capacityEntries' | 'compactEntries' | 'markCells' | 'writeCells' | 'commitBudget';
const USED = new Map<Kernel, readonly number[]>([
  ['vertexPlan', [0, 1, 2, 3, 4, 6, 7, 8, 9]],
  ['capture', [0, 1, 2, 3, 4, 5, 6, 7, 8]],
  ['release', [0, 1, 2, 3, 4, 5, 6, 8]],
  ['capacityEntries', [0, 1, 3, 4, 6]],
  ['compactEntries', [0, 6, 10, 11, 12]],
  ['markCells', [0, 6, 10, 12]],
  ['writeCells', [0, 1, 3, 4, 6, 8, 10, 11, 12, 13]],
  ['commitBudget', [0, 8, 14]],
]);

export interface ClothGPUPoreExchangeStep extends ClothPoreGPUStep {
  readonly identity: Pick<ClothFrameIdentity, 'generation' | 'tick'>;
  readonly vertexMaterial?: ArrayLike<number>;
}
export interface ClothGPUPoreExchangeResult {
  readonly waterVolume_m3: Float64Array; readonly velocities_m_s: Float64Array;
  /** Whole quanta moved from free to bound water, and back, this step (exact). */
  readonly captured_m3: number; readonly released_m3: number;
  readonly ledger: ClothWaterLedger;
  readonly transport?: ClothPoreGPUResult;
  /** Device time from the support sample to the budget commit (timestamp-query). */
  readonly gpu_ms?: number;
}

/** One cloth's GPU pore exchange, sharing the transport's water and velocity buffers. */
export class UniformClothGPUPoreExchange {
  readonly id = 'physics.cloth.gpu-pore-exchange';
  readonly capacityTexture: GPUTexture;
  private readonly device: GPUDevice;
  private readonly owned: GPUBuffer[] = [];
  private readonly params: GPUBuffer; private readonly cells: GPUBuffer; private readonly samples: GPUBuffer; private readonly work: GPUBuffer;
  private readonly table: GPUBuffer; private readonly status: GPUBuffer; private readonly materials: GPUBuffer; private readonly counts: GPUBuffer;
  private readonly capacity: GPUBuffer; private readonly readback: GPUBuffer;
  private readonly sort: UniformGPURadixSort; private readonly scans: readonly [UniformGPUScan, UniformGPUScan];
  private readonly hashEntries: number; private readonly tableWidth: number; private readonly tableHeight: number;
  private readonly timestamps?: { readonly set: GPUQuerySet; readonly resolve: GPUBuffer };
  private samplePipeline?: GPUComputePipeline; private sampleGroup?: GPUBindGroup;
  private readonly pipelines = new Map<Kernel, GPUComputePipeline>();
  private readonly groups = new Map<Kernel, GPUBindGroup>();
  private busy = false; private disposed = false;
  constructor(readonly fields: UniformClothGridFields, readonly transport: ClothGPUPoreTransport, kernels: UniformGPUOrderingKernels) {
    const d = this.device = fields.device, V = transport.topology.V, sections = clothExchangeSections(V);
    if (!fields.solid) throw new Error('Cloth GPU pore exchange needs the porous solid library');
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const make = (label: string, bytes: number) => { const b = d.createBuffer({ label: `Cloth GPU pore exchange ${label}`, size: Math.max(16, Math.ceil(bytes / 16) * 16), usage }); this.owned.push(b); return b; };
    this.params = d.createBuffer({ label: 'Cloth GPU pore exchange parameters', size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); this.owned.push(this.params);
    this.cells = make('vertex cells', 32 * V); this.samples = make('support samples', 32 * SLOTS * V); this.work = make('work', 4 * sections.words);
    // Load at most one half: every support cell of every vertex distinct.
    this.hashEntries = 2 ** Math.ceil(Math.log2(Math.max(64, 2 * SLOTS * V)));
    this.table = make('capture cells', 12 * this.hashEntries); this.status = make('status', STATUS_BYTES); this.counts = make('counts', 16);
    this.materials = make('materials', 4 * CLOTH_PORE_MATERIAL_FLOATS * Math.max(1, transport.topology.M));
    this.tableWidth = Math.min(1024, d.limits.maxTextureDimension2D); this.tableHeight = Math.ceil((1 + 2 * 8 * V) / this.tableWidth);
    if (this.tableHeight > d.limits.maxTextureDimension2D) throw new Error('Porous material capacity exceeds the device texture limit');
    this.capacity = make('capacity table', 16 * this.tableWidth * this.tableHeight);
    this.capacityTexture = d.createTexture({ label: 'Cloth GPU physical free capacity', size: [this.tableWidth, this.tableHeight], format: 'rgba32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    this.readback = d.createBuffer({ label: 'Cloth GPU pore exchange readback', size: STATUS_BYTES + transport.resultBytes + 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }); this.owned.push(this.readback);
    this.sort = new UniformGPURadixSort(kernels, 8 * V, 32, this.counts, 'Cloth capacity sort');
    this.scans = [new UniformGPUScan(kernels, this.work, 8 * V, 4 * sections.positions), new UniformGPUScan(kernels, this.work, 8 * V, 4 * sections.starts)];
    if (d.features.has('timestamp-query')) {
      const resolve = d.createBuffer({ label: 'Cloth GPU pore exchange timestamps', size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }); this.owned.push(resolve);
      this.timestamps = { set: d.createQuerySet({ label: 'Cloth GPU pore exchange timestamps', type: 'timestamp', count: 2 }), resolve };
    }
  }
  async initialize(): Promise<void> {
    const d = this.device, fields = this.fields, solid = fields.solid!, V = this.transport.topology.V;
    const most = Math.max(...[...USED.values()].map(used => used.filter(b => b !== 0).length));
    if (d.limits.maxStorageBuffersPerShaderStage < most) throw new Error(`Cloth GPU pore exchange needs ${most} storage buffers a stage; the device allows ${d.limits.maxStorageBuffersPerShaderStage}`);
    const b = (binding: number, type: GPUBufferBindingType): GPUBindGroupLayoutEntry => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
    const texture = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '3d' } });
    const layout = uniformDetailBindLayout(d, { entries: [b(0, 'uniform'), b(1, 'read-only-storage'), b(2, 'storage'), b(3, 'storage'), texture(4), texture(5)] });
    const sample = uniformDetailModule(d, { label: 'Cloth GPU pore exchange support sample', code: uniformMixedTopologyWGSL(fields.ownership.capacity, 0) + uniformMixedSolidWGSL(2, solid.coarse?.count) + sampleCode });
    const exchange = d.createShaderModule({ label: 'Cloth GPU pore exchange', code: exchangeCode(V) });
    for (const module of [sample, exchange]) {
      const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error');
      if (errors.length) throw new Error(errors.map(m => `Cloth GPU pore exchange ${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));
    }
    this.samplePipeline = await uniformDetailPipeline(d, fields.ownership, { layout: d.createPipelineLayout({ bindGroupLayouts: [fields.ownership.bindLayout, layout, solid.tileLayout] }), compute: { module: sample, entryPoint: 'sampleSupport' } });
    this.sampleGroup = uniformDetailGroup(d, { layout, entries: [{ binding: 0, resource: { buffer: this.params } }, { binding: 1, resource: { buffer: this.cells } }, { binding: 2, resource: { buffer: this.samples } },
      { binding: 3, resource: { buffer: this.status } }, { binding: 4, resource: fields.phi }, { binding: 5, resource: fields.projectedVelocity }] }).group;
    await Promise.all([...USED.keys()].map(async entryPoint => this.pipelines.set(entryPoint, await d.createComputePipelineAsync({ label: `Cloth GPU pore exchange ${entryPoint}`, layout: 'auto', compute: { module: exchange, entryPoint } }))));
    const bindings: GPUBuffer[] = [this.params, this.cells, this.samples, this.transport.vertexBuffer, this.transport.waterBuffer, this.transport.velocityBuffer, this.work, this.table, this.status,
      this.materials, this.sort.keys, this.sort.values, this.counts, this.capacity, fields.waterBudget];
    for (const [kernel, used] of USED) this.groups.set(kernel, d.createBindGroup({ label: `Cloth GPU pore exchange ${kernel}`, layout: this.pipelines.get(kernel)!.getBindGroupLayout(0), entries: used.map(binding => ({ binding, resource: { buffer: bindings[binding]! } })) }));
  }
  /**
   * One step at the accepted cloth state: sample, capture, transport,
   * release, capacity and budget on the device, then a single readback of
   * the water, velocities and ledger. The capacity table becomes NB-FLIP's.
   */
  async step(step: ClothGPUPoreExchangeStep): Promise<ClothGPUPoreExchangeResult> {
    if (this.disposed) throw new Error('Cloth GPU pore exchange was disposed');
    if (this.busy) throw new Error('Cloth GPU pore exchange steps must be serialized');
    if (!this.samplePipeline || !this.sampleGroup) throw new Error('Cloth GPU pore exchange is not initialized');
    const d = this.device, fields = this.fields, t = this.transport.topology, V = t.V, q = this.transport.quantum_m3;
    const staged = this.transport.stage(step, true)!;
    const h = fields.cellSize_m, o = fields.origin_m, records = new ArrayBuffer(32 * V), ints = new Int32Array(records), floats = new Float32Array(records);
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let v = 0; v < V; v++) {
      const m = step.vertexMaterial?.[v] ?? 0;
      if (!Number.isInteger(m) || m < 0 || m >= t.M) throw new Error('Invalid cloth vertex material');
      for (let a = 0; a < 3; a++) {
        const x = (step.positions_m[3 * v + a]! - o[a]!) / h[a]! - .5;
        if (!Number.isFinite(x)) throw new Error('Invalid cloth capacity position');
        const base = Math.floor(x); ints[8 * v + a] = base; floats[8 * v + 4 + a] = x - base;
        lo[a] = Math.min(lo[a]!, base - 1); hi[a] = Math.max(hi[a]!, base + 2);
      }
      ints[8 * v + 3] = m; floats[8 * v + 7] = staged.residual_m3[v]!;
    }
    const dims = lo.map((l, a) => hi[a]! - l + 1);
    if (dims[0]! * dims[1]! * dims[2]! >= 2 ** 32 - 1) throw new Error('Cloth support spans more cells than a 32-bit key frame addresses');
    const header = new ArrayBuffer(64), hu = new Uint32Array(header), hi32 = new Int32Array(header), hf = new Float32Array(header);
    const rate = step.physics?.captureRate ?? 1, rho = step.materials[0]!.waterDensity_kg_m3;
    if (!Number.isFinite(rate) || rate < 0) throw new Error('Invalid cloth capture-rate multiplier');
    hi32.set([lo[0]!, lo[1]!, lo[2]!, V], 0); hu.set([dims[0]!, dims[1]!, dims[2]!, this.hashEntries - 1], 4);
    hf.set([q, h[0]! * h[1]! * h[2]!, Math.min(...h), step.dt_s, rate, rho, step.physics?.absorption === false ? 0 : 1, step.physics?.solveSolid === false ? 0 : 1], 8);
    d.queue.writeBuffer(this.params, 0, header); d.queue.writeBuffer(this.cells, 0, records);
    d.queue.writeBuffer(this.materials, 0, clothPoreMaterialTable(step.materials, step.physics));
    const e = d.createCommandEncoder({ label: 'Cloth GPU pore exchange' }), ts = this.timestamps;
    e.clearBuffer(this.table); e.clearBuffer(this.status); e.clearBuffer(this.counts);
    const run = (pass: GPUComputePassEncoder, kernel: Kernel, n: number) => { pass.setPipeline(this.pipelines.get(kernel)!); pass.setBindGroup(0, this.groups.get(kernel)!); pass.dispatchWorkgroups(Math.max(1, Math.ceil(n / 64))); };
    let pass = e.beginComputePass({ label: 'Cloth GPU support sample', ...(ts ? { timestampWrites: { querySet: ts.set, beginningOfPassWriteIndex: 0 } } : {}) });
    pass.setPipeline(this.samplePipeline); pass.setBindGroup(0, fields.ownership.bindGroup); pass.setBindGroup(1, this.sampleGroup); pass.setBindGroup(2, fields.solid!.tileGroup);
    pass.dispatchWorkgroups(Math.ceil(SLOTS * V / 64)); pass.end();
    pass = e.beginComputePass({ label: 'Cloth GPU capture' }); run(pass, 'vertexPlan', V); run(pass, 'capture', V); pass.end();
    this.transport.encode(e, undefined, staged);
    const [entries, starts] = this.scans;
    pass = e.beginComputePass({ label: 'Cloth GPU release and capacity entries' });
    run(pass, 'release', V); run(pass, 'capacityEntries', V); entries.encode(pass); run(pass, 'compactEntries', 8 * V); pass.end();
    this.sort.encode(e);
    pass = e.beginComputePass({ label: 'Cloth GPU capacity table', ...(ts ? { timestampWrites: { querySet: ts.set, endOfPassWriteIndex: 1 } } : {}) });
    run(pass, 'markCells', 8 * V); starts.encode(pass); run(pass, 'writeCells', 8 * V); pass.setPipeline(this.pipelines.get('commitBudget')!); pass.setBindGroup(0, this.groups.get('commitBudget')!); pass.dispatchWorkgroups(1); pass.end();
    e.copyBufferToTexture({ buffer: this.capacity, bytesPerRow: 16 * this.tableWidth, rowsPerImage: this.tableHeight }, { texture: this.capacityTexture }, [this.tableWidth, this.tableHeight]);
    e.copyBufferToBuffer(this.status, 0, this.readback, 0, STATUS_BYTES);
    const end = this.transport.encodeResult(e, this.readback, STATUS_BYTES);
    if (ts) { e.resolveQuerySet(ts.set, 0, 2, ts.resolve, 0); e.copyBufferToBuffer(ts.resolve, 0, this.readback, end, 16); }
    d.queue.submit([e.finish()]);
    this.busy = true;
    try {
      await this.readback.mapAsync(GPUMapMode.READ);
      const data = this.readback.getMappedRange().slice(0); this.readback.unmap();
      const status = new Uint32Array(data, 0, 8);
      for (const [bit, message] of STATUS_MESSAGES) if (status[0]! & bit) throw new Error(`Cloth GPU pore exchange: ${message} (vertex ${0xffffffff - status[1]!})`);
      const transport = this.transport.parseResult(data, STATUS_BYTES, staged);
      const captured = (status[2]! + 2 ** 32 * status[3]!) * q, released = (status[4]! + 2 ** 32 * status[5]!) * q, freeDelta_m3 = released - captured;
      // NB-FLIP reads the new partition from the next step on.
      fields.setPorousCapacity(this.capacityTexture); fields.invalidateLocalWater();
      return { waterVolume_m3: transport.waterVolume_m3, velocities_m_s: transport.velocities_m_s, captured_m3: captured, released_m3: released,
        ledger: { generation: step.identity.generation, tick: step.identity.tick, freeDelta_m3, boundDelta_m3: -freeDelta_m3, externalDelta_m3: 0 },
        ...(staged.transport ? { transport } : {}), ...(ts ? { gpu_ms: clothTimestampInterval_ms(data, end) } : {}) };
    } finally { if (this.readback.mapState === 'mapped') this.readback.unmap(); this.busy = false; }
  }
  /** Test readback: the published capacity cells, (x, y, z, fiber, bound) a row. */
  async readCapacityTable(): Promise<Float64Array> {
    const out = this.device.createBuffer({ size: this.capacity.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const e = this.device.createCommandEncoder(); e.copyBufferToBuffer(this.capacity, 0, out, 0, this.capacity.size); this.device.queue.submit([e.finish()]);
    await out.mapAsync(GPUMapMode.READ); const raw = out.getMappedRange().slice(0); out.destroy();
    const bits = new Int32Array(raw), values = new Float32Array(raw), n = bits[0]!, rows = new Float64Array(5 * n);
    for (let i = 0; i < n; i++) rows.set([bits[4 * (1 + 2 * i)]!, bits[4 * (1 + 2 * i) + 1]!, bits[4 * (1 + 2 * i) + 2]!, values[4 * (2 + 2 * i)]!, values[4 * (2 + 2 * i) + 1]!], 5 * i);
    return rows;
  }
  destroy(): void {
    if (this.disposed) return; this.disposed = true;
    if (this.fields.porousCapacity === this.capacityTexture) this.fields.setPorousCapacity();
    this.sort.destroy(); for (const scan of this.scans) scan.destroy(); for (const b of this.owned) b.destroy(); this.owned.length = 0; this.timestamps?.set.destroy();
    const texture = this.capacityTexture; void this.device.queue.onSubmittedWorkDone().then(() => texture.destroy(), () => texture.destroy());
  }
}
