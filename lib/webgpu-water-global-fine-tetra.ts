import { makeOctreePowerCoarseLevelSetSampleWGSL } from "./webgpu-octree-power-coarse-levelset";
import { fineLevelSetPackedSampleWGSL } from "./fine-levelset-packed-sample";

export const FLUID_FINE_FILTERED_NORMALS_ENV = "FLUID_FINE_FILTERED_NORMALS";

/** Factor one keeps its established compensator; oversampled bands opt in so
 * existing image baselines remain unchanged until explicitly re-blessed. */
export function fineSurfaceFilteredNormalsEnabled(
  fineFactor: number,
  environment: Record<string, string | undefined> | undefined
    = typeof process !== "undefined" ? process.env : undefined,
): boolean {
  return fineFactor === 1 || ((fineFactor === 4 || fineFactor === 8)
    && environment?.[FLUID_FINE_FILTERED_NORMALS_ENV] === "1");
}

const filteredFineFactorsWGSL = fineSurfaceFilteredNormalsEnabled(4)
  ? "(factor==1u||factor==4u||factor==8u)"
  : "factor==1u";

const TETS = [
  [0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
  [0, 7, 4, 6], [0, 5, 6, 4], [0, 5, 1, 6],
] as const;

const value = (corner: number) => `v${corner}`;
const countCall = (tetra: readonly number[]) => `countTet(${tetra.map(value).join(",")})`;

export const globalFineClassifiedScanShader = /* wgsl */ `
struct V{position:vec4f,normal:vec4f}struct A{vertexCount:atomic<u32>,instanceCount:u32,firstVertex:u32,firstInstance:u32,activeCubeCount:atomic<u32>,vertexAllocator:atomic<u32>,globalFineAuthorityLatch:atomic<u32>,meshPublicationGeneration:atomic<u32>}struct P{sample:vec4u,bricks:vec4u,table:vec4u,settings:vec4f,cell:vec4f,sizing:vec4f,physical:vec4f}
@group(0)@binding(3)var<storage,read_write>out:array<V>;@group(0)@binding(4)var<storage,read_write>args:A;@group(0)@binding(5)var<storage,read>cubes:array<vec2u>;@group(0)@binding(6)var<storage,read>values:array<vec4f>;@group(0)@binding(7)var<storage,read_write>offsets:array<u32>;@group(0)@binding(10)var<uniform>p:P;
fn countTet(a:f32,b:f32,c:f32,d:f32)->u32{let n=select(0u,1u,a>=.5)+select(0u,1u,b>=.5)+select(0u,1u,c>=.5)+select(0u,1u,d>=.5);if(n==0u||n==4u){return 0u;}return select(3u,6u,n==2u);}
var<workgroup>laneOffsets:array<u32,256>;
@compute @workgroup_size(256)fn scanGlobalFineTriangles(@builtin(local_invocation_index)lid:u32){
  let published=atomicLoad(&args.vertexAllocator)!=0xffffffffu;
  let count=select(0u,min(atomicLoad(&args.activeCubeCount),min(arrayLength(&cubes),min(arrayLength(&offsets)/6u,arrayLength(&values)/2u))),published);
  let base=count/256u;let extra=count%256u;let begin=lid*base+min(lid,extra);
  let end=begin+base+select(0u,1u,lid<extra);var localTotal=0u;
  for(var i=begin;i<end;i+=1u){let descriptor=cubes[i].y>>16u;let clipMask=(descriptor>>8u)&7u;let lo=values[i*2u];let hi=values[i*2u+1u];let v0=lo.x;let v1=lo.y;let v2=lo.z;let v3=lo.w;let v4=hi.x;let v5=hi.y;let v6=hi.z;let v7=hi.w;if(clipMask!=0u){localTotal+=6u;}else{localTotal+=${TETS.map(countCall).join("+")};}}
  laneOffsets[lid]=localTotal;workgroupBarrier();
  if(lid==0u){var total=0u;for(var lane=0u;lane<256u;lane+=1u){let subtotal=laneOffsets[lane];laneOffsets[lane]=total;total+=subtotal;}if(published){let capacity=arrayLength(&out)-arrayLength(&out)%3u;atomicStore(&args.vertexCount,min(total,capacity));atomicStore(&args.vertexAllocator,total);atomicStore(&args.meshPublicationGeneration,p.table.w);}}
  workgroupBarrier();var cursor=laneOffsets[lid];
  for(var i=begin;i<end;i+=1u){let descriptor=cubes[i].y>>16u;let clipMask=(descriptor>>8u)&7u;let lo=values[i*2u];let hi=values[i*2u+1u];let v0=lo.x;let v1=lo.y;let v2=lo.z;let v3=lo.w;let v4=hi.x;let v5=hi.y;let v6=hi.z;let v7=hi.w;if(clipMask!=0u){offsets[i*6u]=cursor;cursor+=6u;offsets[i*6u+1u]=cursor;offsets[i*6u+2u]=cursor;offsets[i*6u+3u]=cursor;offsets[i*6u+4u]=cursor;offsets[i*6u+5u]=cursor;}else{${TETS.map((tetra,index)=>`offsets[i*6u+${index}u]=cursor;cursor+=${countCall(tetra)};`).join("")}}}
}
`;

/** Production scan variant that publishes the exact tetrahedron emission grid. */
export const globalFineClassifiedIndirectScanShader = globalFineClassifiedScanShader
  .replace(
    "@group(0)@binding(10)var<uniform>p:P;",
    "@group(0)@binding(10)var<uniform>p:P;@group(0)@binding(11)var<storage,read_write>emitDispatch:array<u32>;",
  )
  .replace(
    "if(published){let capacity=arrayLength(&out)-arrayLength(&out)%3u;",
    "emitDispatch[0]=(count+63u)/64u;emitDispatch[1]=6u;emitDispatch[2]=1u;if(published){let capacity=arrayLength(&out)-arrayLength(&out)%3u;",
  );

const cornerPosition = [
  "vec3f(0,0,0)", "vec3f(1,0,0)", "vec3f(1,1,0)", "vec3f(0,1,0)",
  "vec3f(0,0,1)", "vec3f(1,0,1)", "vec3f(1,1,1)", "vec3f(0,1,1)",
] as const;

const factorOneFilteredNormalShader = /* wgsl */ `
@group(0)@binding(8)var<storage,read>fineWorklist:array<u32>;
@group(0)@binding(9)var<storage,read>fineSamples:array<u32>;
@group(0)@binding(12)var<storage,read>metadata:array<u32>;
${makeOctreePowerCoarseLevelSetSampleWGSL(16)}
const INVALID:u32=0xffffffffu;
${fineLevelSetPackedSampleWGSL("fineSamples", false)}
fn finitePhi(value:f32)->bool{return value==value&&abs(value)<3.402823e38;}
fn finePage(key:u32)->u32{
  if(p.table.y!=7u||arrayLength(&fineWorklist)<7u||fineWorklist[0]!=p.table.w
    ||fineWorklist[2]!=p.table.z||(fineWorklist[3]&3u)!=3u
    ||fineWorklist[5]!=1u||fineWorklist[6]!=1u){return INVALID;}
  let logicalCount=p.bricks.x*p.bricks.y*p.bricks.z;
  let directoryBase=7u+p.table.z;
  if(key>=logicalCount||directoryBase+key>=arrayLength(&fineWorklist)){return INVALID;}
  let id=fineWorklist[directoryBase+key];let base=id*4u;
  return select(INVALID,id,id<p.table.z&&base+2u<arrayLength(&metadata)
    &&metadata[base]==id&&metadata[base+1u]==key&&metadata[base+2u]==p.table.w);
}
fn signedPhi(qi:vec3i)->f32{
  // The optical wall cubes keep their existing normals. For the adjoining free
  // surface, repeat the nearest in-domain signed sample so the smoothing kernel
  // has a stable one-sided extension instead of reading the artificial wall cap.
  let dims=vec3i(p.sample.xyz);
  let q=vec3u(clamp(qi,vec3i(0),dims-vec3i(1)));
  let r=max(1u,p.sample.w);let brick=q/r;let local=q-brick*r;
  if(all(brick<p.bricks.xyz)){
    let key=brick.x+p.bricks.x*(brick.y+p.bricks.y*brick.z);let id=finePage(key);
    if(id!=INVALID){
      let index=id*p.bricks.w+local.x+r*(local.y+r*local.z);
      if(index<arrayLength(&fineSamples)
        &&(finePackedFlags(index)&1u)!=0u&&finitePhi(finePackedPhi(index))){return finePackedPhi(index);}
    }
  }
  return sampleCoarseOctreePhi(p.settings.xyz+(vec3f(q)+vec3f(.5))*p.settings.w);
}
fn filteredNormalAt(lattice:vec3f,fallback:vec3f,filterEnabled:bool)->vec3f{
  // Factor-4/8 filtering is compiled in only under the fidelity flag, keeping
  // the established baseline byte-for-byte when the flag is absent.
  let factor=u32(round(p.cell.x));
  if(!(${filteredFineFactorsWGSL})||!filterEnabled){return fallback;}
  let x=lattice-vec3f(1.0);
  let center=vec3i(round(x));
  var weightSum=0.0;var phiSum=0.0;
  var derivativeWeightSum=vec3f(0.0);var phiDerivativeSum=vec3f(0.0);
  var valid=true;
  // Gradient of a normalized 3x3x3 Gaussian reconstruction evaluated at the
  // emitted vertex. Subtracting the normalization derivative makes constants
  // reproduce exactly even when the vertex is off-centre in the sample stencil.
  for(var oz=0;oz<3;oz+=1){for(var oy=0;oy<3;oy+=1){for(var ox=0;ox<3;ox+=1){
    let q=center+vec3i(ox-1,oy-1,oz-1);
    let delta=vec3f(q)-x;
    let weight=exp(-.5*dot(delta,delta)/(.85*.85));
    let value=signedPhi(q);
    if(!finitePhi(value)){valid=false;}
    weightSum+=weight;phiSum+=weight*value;
    derivativeWeightSum+=weight*delta;
    phiDerivativeSum+=weight*value*delta;
  }}}
  if(!valid||weightSum<=1e-8){return fallback;}
  let derivative=phiDerivativeSum*weightSum-phiSum*derivativeWeightSum;
  let size=vec3f(p.sample.xyz);
  let gradient=vec3f(derivative.x*size.x/u.container.x,
    derivative.y*size.y/u.container.y,derivative.z*size.z/u.container.z);
  return select(fallback,normalize(gradient),length(gradient)>1e-8);
}
`;

/** Shared verbatim by production and the focused Dawn rectangle reproducer. */
export const globalFineDirectSharpPatchWGSL = /* wgsl */ `
fn directPatch(cursor:ptr<function,u32>,base:vec3f,scale:f32,descriptor:u32,n:vec3f){let mask=(descriptor>>8u)&7u;let axis=(descriptor>>14u)&3u;var lx=0.0;var ly=0.0;var lz=0.0;var hx=1.0;var hy=1.0;var hz=1.0;if((mask&1u)!=0u){let high=((descriptor>>11u)&1u)!=0u;lx=select(0.0,0.5,high);hx=select(0.5,1.0,high);}if((mask&2u)!=0u){let high=((descriptor>>12u)&1u)!=0u;ly=select(0.0,0.5,high);hy=select(0.5,1.0,high);}if((mask&4u)!=0u){let high=((descriptor>>13u)&1u)!=0u;lz=select(0.0,0.5,high);hz=select(0.5,1.0,high);}var a=vec3f(0);var b=vec3f(0);var c=vec3f(0);var d=vec3f(0);if(axis==0u){a=vec3f(0.5,ly,lz);b=vec3f(0.5,hy,lz);c=vec3f(0.5,hy,hz);d=vec3f(0.5,ly,hz);}else if(axis==1u){a=vec3f(lx,0.5,lz);b=vec3f(lx,0.5,hz);c=vec3f(hx,0.5,hz);d=vec3f(hx,0.5,lz);}else{a=vec3f(lx,ly,0.5);b=vec3f(hx,ly,0.5);c=vec3f(hx,hy,0.5);d=vec3f(lx,hy,0.5);}a=base+scale*a;b=base+scale*b;c=base+scale*c;d=base+scale*d;tri(cursor,a,b,c,n,false);tri(cursor,a,c,d,n,false);}
`;

function emitShader(tetraIndex: number): string {
  const [a, b, c, d] = TETS[tetraIndex];
  return /* wgsl */ `
struct U{viewport:vec4f,cameraPosition:vec4f,cameraTarget:vec4f,container:vec4f,options:vec4f,gridInfo:vec4f,debug:vec4f}struct V{position:vec4f,normal:vec4f}struct A{vertexCount:atomic<u32>,instanceCount:u32,firstVertex:u32,firstInstance:u32,activeCubeCount:atomic<u32>,vertexAllocator:atomic<u32>,globalFineAuthorityLatch:atomic<u32>,meshPublicationGeneration:atomic<u32>}struct P{sample:vec4u,bricks:vec4u,table:vec4u,settings:vec4f,cell:vec4f,sizing:vec4f,physical:vec4f}
@group(0)@binding(0)var<uniform>u:U;@group(0)@binding(3)var<storage,read_write>out:array<V>;@group(0)@binding(4)var<storage,read_write>args:A;@group(0)@binding(5)var<storage,read>cubes:array<vec2u>;@group(0)@binding(6)var<storage,read>values:array<vec4f>;@group(0)@binding(7)var<storage,read_write>offsets:array<u32>;@group(0)@binding(10)var<uniform>p:P;
${factorOneFilteredNormalShader}
fn world(q:vec3f)->vec3f{let x=clamp((q-.5)/vec3f(p.sample.xyz),vec3f(0),vec3f(1));return vec3f(-.5*u.container.x,0,-.5*u.container.z)+x*u.container.xyz;}fn edge(x:vec3f,y:vec3f,a:f32,b:f32)->vec3f{return mix(x,y,clamp((.5-a)/(b-a),.02,.98));}fn countTet(a:f32,b:f32,c:f32,d:f32)->u32{let n=select(0u,1u,a>=.5)+select(0u,1u,b>=.5)+select(0u,1u,c>=.5)+select(0u,1u,d>=.5);if(n==0u||n==4u){return 0u;}return select(3u,6u,n==2u);}
fn tri(cursor:ptr<function,u32>,a:vec3f,b:vec3f,c:vec3f,n:vec3f,filterEnabled:bool){let first=*cursor;*cursor=first+3u;let limit=min(atomicLoad(&args.vertexCount),arrayLength(&out));if(first+3u>limit){return;}let x=V(vec4f(world(a),1),vec4f(filteredNormalAt(a,n,filterEnabled),0));let y=V(vec4f(world(b),1),vec4f(filteredNormalAt(b,n,filterEnabled),0));let z=V(vec4f(world(c),1),vec4f(filteredNormalAt(c,n,filterEnabled),0));out[first]=x;if(dot(cross(y.position.xyz-x.position.xyz,z.position.xyz-x.position.xyz),n)>=0.){out[first+1u]=y;out[first+2u]=z;}else{out[first+1u]=z;out[first+2u]=y;}}
fn tet(cursor:ptr<function,u32>,pa:vec3f,pb:vec3f,pc:vec3f,pd:vec3f,va:f32,vb:f32,vc:f32,vd:f32,n:vec3f,filterEnabled:bool){let m=select(0u,1u,va>=.5)|select(0u,2u,vb>=.5)|select(0u,4u,vc>=.5)|select(0u,8u,vd>=.5);if(m==1u||m==14u){tri(cursor,edge(pa,pb,va,vb),edge(pa,pc,va,vc),edge(pa,pd,va,vd),n,filterEnabled);}else if(m==2u||m==13u){tri(cursor,edge(pb,pa,vb,va),edge(pb,pd,vb,vd),edge(pb,pc,vb,vc),n,filterEnabled);}else if(m==4u||m==11u){tri(cursor,edge(pc,pa,vc,va),edge(pc,pb,vc,vb),edge(pc,pd,vc,vd),n,filterEnabled);}else if(m==8u||m==7u){tri(cursor,edge(pd,pa,vd,va),edge(pd,pc,vd,vc),edge(pd,pb,vd,vb),n,filterEnabled);}else if(m==3u||m==12u){let ac=edge(pa,pc,va,vc);let ad=edge(pa,pd,va,vd);let bc=edge(pb,pc,vb,vc);let bd=edge(pb,pd,vb,vd);tri(cursor,ac,bc,bd,n,filterEnabled);tri(cursor,ac,bd,ad,n,filterEnabled);}else if(m==5u||m==10u){let ab=edge(pa,pb,va,vb);let ad=edge(pa,pd,va,vd);let cb=edge(pc,pb,vc,vb);let cd=edge(pc,pd,vc,vd);tri(cursor,ab,cb,cd,n,filterEnabled);tri(cursor,ab,cd,ad,n,filterEnabled);}else if(m==6u||m==9u){let ba=edge(pb,pa,vb,va);let bd=edge(pb,pd,vb,vd);let ca=edge(pc,pa,vc,va);let cd=edge(pc,pd,vc,vd);tri(cursor,ba,ca,cd,n,filterEnabled);tri(cursor,ba,cd,bd,n,filterEnabled);}}
fn clipped(base:vec3f,scale:f32,q:vec3f,descriptor:u32)->vec3f{var r=base+scale*q;let mask=(descriptor>>8u)&7u;for(var axis=0u;axis<3u;axis+=1u){if((mask&(1u<<axis))!=0u){let high=((descriptor>>(11u+axis))&1u)!=0u;r[axis]=base[axis]+scale*select(.5*q[axis],.5+.5*q[axis],high);}}return r;}
${globalFineDirectSharpPatchWGSL}
@compute @workgroup_size(64)fn emitGlobalFineTetra${tetraIndex}(@builtin(global_invocation_id)g:vec3u){let i=g.x;let count=min(atomicLoad(&args.activeCubeCount),min(arrayLength(&cubes),arrayLength(&offsets)/6u));if(i>=count||i*2u+1u>=arrayLength(&values)){return;}let packed=cubes[i];let descriptor=packed.y>>16u;let scale=f32(max(1u,descriptor&255u));let base=vec3f(f32(packed.x&0xffffu),f32(packed.y&0xffffu),f32(packed.x>>16u));let lo=values[i*2u];let hi=values[i*2u+1u];let v0=lo.x;let v1=lo.y;let v2=lo.z;let v3=lo.w;let v4=hi.x;let v5=hi.y;let v6=hi.z;let v7=hi.w;let gx=.25*((v1+v2+v5+v6)-(v0+v3+v4+v7));let gy=.25*((v2+v3+v6+v7)-(v0+v1+v4+v5));let gz=.25*((v4+v5+v6+v7)-(v0+v1+v2+v3));let size=vec3f(p.sample.xyz);let gradient=vec3f(gx*size.x/u.container.x,gy*size.y/u.container.y,gz*size.z/u.container.z);var n=vec3f(0,1,0);if(length(gradient)>1e-5){n=-normalize(gradient);}let clipMask=(descriptor>>8u)&7u;let filterEnabled=clipMask==0u&&base.x>0.0&&base.y>0.0&&base.z>0.0&&base.x<f32(p.sample.x)&&base.y<f32(p.sample.y)&&base.z<f32(p.sample.z);var cursor=offsets[i*6u+${tetraIndex}u];if(clipMask!=0u){if(${tetraIndex}u==0u){directPatch(&cursor,base,scale,descriptor,n);}return;}tet(&cursor,clipped(base,scale,${cornerPosition[a]},descriptor),clipped(base,scale,${cornerPosition[b]},descriptor),clipped(base,scale,${cornerPosition[c]},descriptor),clipped(base,scale,${cornerPosition[d]},descriptor),v${a},v${b},v${c},v${d},n,filterEnabled);}
`;
}

export const globalFineClassifiedEmitShaders = TETS.map((_, index) => emitShader(index));

/** One bounded 2-D dispatch: x selects a classified cube, y one of six exact tetrahedra. */
export const globalFineClassifiedEmitShader = emitShader(0).replace(
  /@compute @workgroup_size\(64\)fn emitGlobalFineTetra0[\s\S]*$/,
  `@compute @workgroup_size(64)fn emitGlobalFineTetrahedra(@builtin(global_invocation_id)g:vec3u){let i=g.x;let tetrahedron=g.y;if(tetrahedron>=6u){return;}let count=min(atomicLoad(&args.activeCubeCount),min(arrayLength(&cubes),arrayLength(&offsets)/6u));if(i>=count||i*2u+1u>=arrayLength(&values)){return;}let packed=cubes[i];let descriptor=packed.y>>16u;let scale=f32(max(1u,descriptor&255u));let base=vec3f(f32(packed.x&0xffffu),f32(packed.y&0xffffu),f32(packed.x>>16u));let lo=values[i*2u];let hi=values[i*2u+1u];let samples=array<f32,8>(lo.x,lo.y,lo.z,lo.w,hi.x,hi.y,hi.z,hi.w);let positions=array<vec3f,8>(clipped(base,scale,vec3f(0,0,0),descriptor),clipped(base,scale,vec3f(1,0,0),descriptor),clipped(base,scale,vec3f(1,1,0),descriptor),clipped(base,scale,vec3f(0,1,0),descriptor),clipped(base,scale,vec3f(0,0,1),descriptor),clipped(base,scale,vec3f(1,0,1),descriptor),clipped(base,scale,vec3f(1,1,1),descriptor),clipped(base,scale,vec3f(0,1,1),descriptor));let ids=array<vec4u,6>(vec4u(0,1,2,6),vec4u(0,2,3,6),vec4u(0,3,7,6),vec4u(0,7,4,6),vec4u(0,5,6,4),vec4u(0,5,1,6))[tetrahedron];let v0=lo.x;let v1=lo.y;let v2=lo.z;let v3=lo.w;let v4=hi.x;let v5=hi.y;let v6=hi.z;let v7=hi.w;let gx=.25*((v1+v2+v5+v6)-(v0+v3+v4+v7));let gy=.25*((v2+v3+v6+v7)-(v0+v1+v4+v5));let gz=.25*((v4+v5+v6+v7)-(v0+v1+v2+v3));let size=vec3f(p.sample.xyz);let gradient=vec3f(gx*size.x/u.container.x,gy*size.y/u.container.y,gz*size.z/u.container.z);var n=vec3f(0,1,0);if(length(gradient)>1e-5){n=-normalize(gradient);}let clipMask=(descriptor>>8u)&7u;let filterEnabled=clipMask==0u&&base.x>0.0&&base.y>0.0&&base.z>0.0&&base.x<f32(p.sample.x)&&base.y<f32(p.sample.y)&&base.z<f32(p.sample.z);var cursor=offsets[i*6u+tetrahedron];if(clipMask!=0u){if(tetrahedron==0u){directPatch(&cursor,base,scale,descriptor,n);}return;}tet(&cursor,positions[ids.x],positions[ids.y],positions[ids.z],positions[ids.w],samples[ids.x],samples[ids.y],samples[ids.z],samples[ids.w],n,filterEnabled);}`,
);
