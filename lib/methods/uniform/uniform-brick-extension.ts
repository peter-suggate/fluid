/** ubaExtend: the velocity extension of the brick-resident step's B4 pass
 * (docs/plans/brick-resident-step-2026-11-08.md §3.3) as one fused brick
 * stage. It replaces the paired seed, sweep, seam and publish kernels.
 *
 * For each face component in turn, one workgroup a brick does this:
 * 1. Stages the component's values and seed distances with ubaStage, with
 *    a halo of `sweeps` faces. The seed distance is 0 at a source face and
 *    unreached (1e20) elsewhere. A negative seed (UBA_BLOCKED, the default
 *    outside every stored page) is no part of the domain: never swept, never
 *    an upwind neighbour. A seam's values come from ubaStage's seam rule, so
 *    no sweep sees a seam: a seam face is swept like any other.
 * 2. Runs `sweeps` Jacobi sweeps of the Godunov update in threadgroup
 *    memory, ping-ponging between the staged windows and two scratch
 *    windows:
 *    - each sweep solves |grad d| = 1 on the upwind neighbours (Zhao 2005,
 *      fast sweeping; the first-order upwind root of Sethian 1999);
 *    - it carries the value as the extension velocity: the upwind
 *      neighbours' mean, weighted by (root - d_n)/h_n² (Adalsteinsson and
 *      Sethian 1999, grad v . grad d = 0 discretized upwind);
 *    - it cuts off at `reach` cells (Bridson 2015 §4.4 extends a fixed band
 *      of layers).
 *
 *    Sweep s computes the window inset by s+1, which every later sweep's
 *    stencil stays inside. The core after `sweeps` sweeps is therefore
 *    exactly that many global Jacobi sweeps, whatever the neighbours hold.
 *    This is what today's slot arrays do across launches.
 * 3. Hands each core face's (value, distance) to the caller's write fn.
 *
 * A component's windows are reused by the next, so the kernel takes 4
 * windows of (8+2·sweeps)³ words whatever the components. */
import type { UniformBrickArena } from "./uniform-brick-arena";
import { UBA_STAGE_WORDS, ubaStageWGSL, type UbaStageChannel, type UbaStageLevel } from "./uniform-brick-stage";

/** A staged field without its window placement: ubaExtend names and places it. */
export type UbaExtendField=Omit<UbaStageChannel,"name"|"layout"|"axis"|"halo"|"window">;
export interface UbaExtendOptions{
 level:UbaStageLevel;
 workgroup:number;
 /** Jacobi sweeps (the staged halo): 1..4. */
 sweeps:number;
 /** WGSL vec3f expression: the level's cell size. */
 spacing:string;
 /** The band: a root past reach × the least spacing stays unreached. */
 reach:number;
 /** The value and seed distance field of each component, in axis order (1 to 3). */
 components:readonly {axis:0|1|2;value:UbaExtendField;seed:UbaExtendField}[];
 /** fn(c:vec3i,axis:u32,state:vec2f): core face c (0..7 an axis) of the
  * component, state (value, distance), distance >= 0.5e20 unreached. */
 write:string;
 resolve?:string;
 fineOwned?:string;
}
export const UBA_UNREACHED=1e20;
/** The seed distance of a face outside the domain. */
export const UBA_BLOCKED=-1;

export function ubaExtendWGSL(arena:UniformBrickArena,kernel:string,options:UbaExtendOptions):string{
 const {sweeps,components,workgroup}=options,K=kernel;
 if(!Number.isInteger(sweeps)||sweeps<1||sweeps>4)throw new Error(`ubaExtend ${K}: ${sweeps} sweeps (1..4)`);
 if(!(options.reach>0))throw new Error(`ubaExtend ${K}: reach ${options.reach}`);
 if(components.length<1||components.length>3||new Set(components.map(c=>c.axis)).size!==components.length)throw new Error(`ubaExtend ${K}: components name distinct axes`);
 const e=8+2*sweeps,e3=e**3;
 if(4*e3+128>UBA_STAGE_WORDS)throw new Error(`ubaExtend ${K}: ${4*e3} window words past the threadgroup budget`);
 const blocked=`ubaBlocked_${K}`;
 const channels:UbaStageChannel[]=components.flatMap((c,j)=>[
  {...c.value,name:`v${c.axis}`,layout:"face" as const,axis:c.axis,halo:sweeps,...(j?{window:`v${components[0]!.axis}`}:{})},
  {outside:blocked,...c.seed,name:`d${c.axis}`,layout:"face" as const,axis:c.axis,halo:sweeps,...(j?{window:`d${components[0]!.axis}`}:{})},
 ]);
 const first=components[0]!.axis,V=`ubaW_${K}_v${first}`,D=`ubaW_${K}_d${first}`,XV=`ubaXV_${K}`,XD=`ubaXD_${K}`;
 const sweep=(from:[string,string],to:[string,string],s:number)=>{
  const inset=sweeps-s-1,n=8+2*inset;
  return /* wgsl */`
 for(var k=lane;k<${n**3}u;k+=${workgroup}u){
  let c=vec3i(vec3u(k%${n}u,(k/${n}u)%${n}u,k/${n*n}u))-vec3i(${inset});
  let at=ubaExtendAt_${K}(c);let old=vec2f(${from[0]}[at],${from[1]}[at]);var next=old;
  if(old.y>0.0){
   var lo:array<vec2f,3>;var hi:array<vec2f,3>;
   for(var a=0u;a<3u;a++){var d=vec3i(0);d[a]=1;let l=ubaExtendAt_${K}(c-d);let u=ubaExtendAt_${K}(c+d);lo[a]=ubaExtendNeighbour_${K}(${from[0]}[l],${from[1]}[l]);hi[a]=ubaExtendNeighbour_${K}(${from[0]}[u],${from[1]}[u]);}
   next=ubaExtendCombine_${K}(old,lo,hi);
  }
  ${to[0]}[at]=next.x;${to[1]}[at]=next.y;
 }
 workgroupBarrier();`;
 };
 const buffers:[string,string][]=[[V,D],[XV,XD]];
 const final=buffers[sweeps%2]!;
 return ubaStageWGSL(arena,K,{level:options.level,workgroup,channels,resolve:options.resolve,fineOwned:options.fineOwned})+/* wgsl */`
var<workgroup> ${XV}:array<f32,${e3}>;
var<workgroup> ${XD}:array<f32,${e3}>;
fn ${blocked}(p:vec3i)->f32{return ${UBA_BLOCKED}.0;}
fn ubaExtendAt_${K}(c:vec3i)->u32{let q=vec3u(c+vec3i(${sweeps}));return q.x+${e}u*(q.y+${e}u*q.z);}
// A blocked face is never an upwind neighbour.
fn ubaExtendNeighbour_${K}(value:f32,distance:f32)->vec2f{return select(vec2f(value,distance),vec2f(0.0,${UBA_UNREACHED.toExponential()}),distance<0.0);}
// The upwind root over the nearer neighbour of each axis, the band cut-off,
// then the weighted mean of the neighbours below the root (ueCombine at one
// spacing a level).
fn ubaExtendCombine_${K}(old:vec2f,lo:array<vec2f,3>,hi:array<vec2f,3>)->vec2f{
 let h=${options.spacing};var minima:array<f32,3>;var spacing:array<f32,3>;
 for(var a=0u;a<3u;a++){minima[a]=min(lo[a].y,hi[a].y);spacing[a]=h[a];}
 var sorted=minima;var lengths=spacing;
 for(var i=0u;i<2u;i++){for(var j=i+1u;j<3u;j++){if(sorted[j]<sorted[i]){let d=sorted[i];sorted[i]=sorted[j];sorted[j]=d;let s=lengths[i];lengths[i]=lengths[j];lengths[j]=s;}}}
 if(sorted[0]>=${(0.5*UBA_UNREACHED).toExponential()}){return old;}
 var root=sorted[0]+lengths[0];
 for(var count=2u;count<=3u;count++){
  if(sorted[count-1u]>=${(0.5*UBA_UNREACHED).toExponential()}||root<=sorted[count-1u]){break;}
  var a=0.0;var b=0.0;var c=-1.0;
  for(var i=0u;i<count;i++){let w=1.0/(lengths[i]*lengths[i]);a+=w;b+=sorted[i]*w;c+=sorted[i]*sorted[i]*w;}
  root=(b+sqrt(max(b*b-a*c,0.0)))/a;
 }
 let cut=${options.reach}.0*min(h.x,min(h.y,h.z));
 if(root>cut||root>old.y){return old;}
 let epsilon=max(root,cut)*1.1920929e-7;var weighted=0.0;var total=0.0;
 for(var a=0u;a<3u;a++){
  if(minima[a]>=root-epsilon){continue;}var value=0.0;var count=0.0;
  if(abs(lo[a].y-minima[a])<=epsilon){value+=lo[a].x;count+=1.0;}
  if(abs(hi[a].y-minima[a])<=epsilon){value+=hi[a].x;count+=1.0;}
  let weight=(root-minima[a])/(spacing[a]*spacing[a]);weighted+=weight*value/max(count,1.0);total+=weight;
 }
 return vec2f(select(0.0,weighted/total,total>0.0),root);
}
// One brick: every component staged, swept and written. Holds barriers:
// call it from uniform control flow.
fn ubaExtend_${K}(slot:u32,lane:u32){
 ubaRecord_${K}(slot,lane);
${components.map(c=>`
 ubaStage_${K}_v${c.axis}(lane);ubaStage_${K}_d${c.axis}(lane);
${Array.from({length:sweeps},(_,s)=>sweep(buffers[s%2]!,buffers[(s+1)%2]!,s)).join("")}
 for(var k=lane;k<512u;k+=${workgroup}u){let c=vec3i(vec3u(k&7u,(k>>3u)&7u,k>>6u));let at=ubaExtendAt_${K}(c);${options.write}(c,${c.axis}u,vec2f(${final[0]}[at],${final[1]}[at]));}
 workgroupBarrier();`).join("")}
}
`;
}
