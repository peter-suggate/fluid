import {
  OCTREE_FIRST_ORDER_CHEBYSHEV_DEGREES,
  OCTREE_FIRST_ORDER_CHEBYSHEV_LOWER_FRACTION,
} from "./webgpu-octree-section43-contract";
import type { PassBroker } from "./webgpu-pass-broker";
import {
  FACTOR_ONE_DENSE_GPU_PUBLICATION_WORD,
  FACTOR_ONE_DENSE_GPU_WORKLIST_RECORD_WORDS,
  type FactorOneDenseGpuChannelName,
  type FactorOneDenseGpuPhysicalPlan,
} from "./factor-one-dense-pressure-hierarchy";
import type { FactorOneDenseAcceptedView } from "./webgpu-factor-one-dense-pressure-shadow";

export const FACTOR_ONE_DENSE_CORRECTION_VALID = 0x4431_4d47;
export const FACTOR_ONE_DENSE_CORRECTION_WORKGROUP_SIZE = 64;

export interface FactorOneDenseCorrectionLayout {
  readonly dimensions: readonly [number, number, number];
  readonly rowCapacity: number;
  readonly levelCount: number;
  readonly totalSlots: number;
  readonly levelBases: readonly number[];
  readonly levelVolumes: readonly number[];
  readonly finestCellWidth: number;
  readonly acceptedOccupiedCountStrideWords?: number;
  readonly offsetsWords: Readonly<{
    acceptedFlags: number;
    acceptedOwners: number;
    acceptedDiagonal: number;
    rhs: number;
    a: number;
    b: number;
    spectralUpper: number;
    acceptedOccupiedIndices: number;
    acceptedOccupiedCounts: number;
    publicationValid: number;
    publicationEpoch: number;
    publicationError: number;
  }>;
  readonly publicationValidValue?: number;
}

export interface FactorOneDenseCorrectionSource {
  readonly arena: GPUBuffer;
  /** Accepted structured control: status, ..., row count, epoch, bank. */
  readonly acceptedControl: GPUBuffer;
  /** Raw SPGrid CapturePageState: publishedGeneration=6, error=8. */
  readonly epochControl: GPUBuffer;
  readonly layout: FactorOneDenseCorrectionLayout;
  readonly worklistIndexKind: "level-local";
}

function physicalChannel(
  plan: FactorOneDenseGpuPhysicalPlan,
  name: FactorOneDenseGpuChannelName,
) {
  const result = plan.channels.find((candidate) => candidate.name === name);
  if (!result) throw new RangeError(`Dense correction is missing channel ${name}`);
  return result;
}

/** Narrow adapter from the Stage-B shadow publication into the Stage-C executor. */
export function factorOneDenseCorrectionSource(
  view: FactorOneDenseAcceptedView,
  acceptedControl: GPUBuffer,
  epochControl: GPUBuffer,
  rowCapacity: number,
  finestCellWidth: number,
): FactorOneDenseCorrectionSource {
  const plan = view.physicalPlan;
  const hierarchy = plan.hierarchy;
  const word = (bytes: number) => bytes / 4;
  const publication = word(plan.publicationControl.offsetBytes);
  return Object.freeze({
    arena: view.arena,
    acceptedControl,
    epochControl,
    worklistIndexKind: view.worklistIndexKind,
    layout: Object.freeze({
      dimensions: hierarchy.dimensions,
      rowCapacity,
      levelCount: hierarchy.levelCount,
      totalSlots: hierarchy.totalSlots,
      levelBases: hierarchy.levelBases,
      levelVolumes: hierarchy.levelVolumes,
      finestCellWidth,
      acceptedOccupiedCountStrideWords: FACTOR_ONE_DENSE_GPU_WORKLIST_RECORD_WORDS,
      offsetsWords: Object.freeze({
        acceptedFlags: word(physicalChannel(plan, "acceptedFlags").offsetBytes),
        acceptedOwners: word(physicalChannel(plan, "acceptedOwners").offsetBytes),
        acceptedDiagonal: word(physicalChannel(plan, "acceptedDiagonal").offsetBytes),
        rhs: word(physicalChannel(plan, "rhs").offsetBytes),
        a: word(physicalChannel(plan, "a").offsetBytes),
        b: word(physicalChannel(plan, "b").offsetBytes),
        spectralUpper: word(plan.spectralUpper.offsetBytes),
        acceptedOccupiedIndices: word(plan.acceptedOccupiedIndices.offsetBytes),
        acceptedOccupiedCounts: word(plan.acceptedOccupiedCounts.offsetBytes),
        publicationValid: publication + FACTOR_ONE_DENSE_GPU_PUBLICATION_WORD.acceptedValid,
        publicationEpoch: publication + FACTOR_ONE_DENSE_GPU_PUBLICATION_WORD.acceptedEpoch,
        publicationError: publication + FACTOR_ONE_DENSE_GPU_PUBLICATION_WORD.acceptedError,
      }),
      publicationValidValue: 1,
    }),
  });
}

export function octreeFactorOneDenseMGEnabled(
  environment: Record<string, string | undefined> | undefined =
    typeof process !== "undefined" ? process.env : undefined,
): boolean {
  return environment?.FLUID_OCTREE_FACTOR1_DENSE_MG !== "0";
}

const PARAMS_BYTES = 256;
const MAXIMUM_LEVELS = 12;
const WORDS_PER_LEVEL = 6;
const BYTES_PER_LEVEL = WORDS_PER_LEVEL * 4;

type DensePipelineName = "prepareDenseCorrectionDispatches" | "initializeDenseCorrection"
  | "smoothDenseAtoB0" | "smoothDenseBtoA0"
  | "smoothDenseBtoA0AndPublish"
  | "smoothDenseAtoB1" | "smoothDenseBtoA1"
  | "smoothDenseAtoB2" | "smoothDenseBtoA2"
  | "smoothDenseAtoB3" | "smoothDenseBtoA3"
  | "restrictDenseAggregate" | "denseVcycleTail"
  | "prolongDenseAggregate";

export const FACTOR_ONE_DENSE_CORRECTION_BINDINGS:
Readonly<Record<DensePipelineName, readonly number[]>> = Object.freeze({
  prepareDenseCorrectionDispatches: [0, 1, 2, 3, 6, 7],
  initializeDenseCorrection: [0, 1, 2, 3, 4, 5],
  smoothDenseAtoB0: [0, 1, 3], smoothDenseBtoA0: [0, 1, 3],
  smoothDenseBtoA0AndPublish: [0, 1, 2, 3, 5],
  smoothDenseAtoB1: [0, 1, 3], smoothDenseBtoA1: [0, 1, 3],
  smoothDenseAtoB2: [0, 1, 3], smoothDenseBtoA2: [0, 1, 3],
  smoothDenseAtoB3: [0, 1, 3], smoothDenseBtoA3: [0, 1, 3],
  restrictDenseAggregate: [0, 1, 3],
  denseVcycleTail: [0, 1, 2, 3, 5],
  prolongDenseAggregate: [0, 1, 3],
});

function checkedInteger(value: number, label: string, allowZero = false): number {
  if (!Number.isSafeInteger(value) || value < (allowZero ? 0 : 1) || value > 0xffff_ffff) {
    throw new RangeError(`${label} must be a ${allowZero ? "non-negative" : "positive"} u32`);
  }
  return value;
}

function validateLayout(layout: FactorOneDenseCorrectionLayout): void {
  if (layout.dimensions.length !== 3) throw new RangeError("Dense correction dimensions are invalid");
  layout.dimensions.forEach((value) => checkedInteger(value, "Dense correction dimension"));
  checkedInteger(layout.rowCapacity, "Dense correction row capacity");
  checkedInteger(layout.levelCount, "Dense correction level count");
  checkedInteger(layout.totalSlots, "Dense correction total slots");
  if (layout.levelCount > MAXIMUM_LEVELS
    || layout.levelBases.length !== layout.levelCount
    || layout.levelVolumes.length !== layout.levelCount) {
    throw new RangeError("Dense correction level tables are invalid");
  }
  let end = 0;
  for (let level = 0; level < layout.levelCount; level += 1) {
    const base = checkedInteger(layout.levelBases[level]!, "Dense correction level base", true);
    const volume = checkedInteger(layout.levelVolumes[level]!, "Dense correction level volume");
    if (base !== end) throw new RangeError("Dense correction levels must be contiguous");
    end += volume;
  }
  if (end !== layout.totalSlots) throw new RangeError("Dense correction level extent is invalid");
  if (!(layout.finestCellWidth > 0) || !Number.isFinite(layout.finestCellWidth)) {
    throw new RangeError("Dense correction finest-cell width must be positive");
  }
  checkedInteger(layout.acceptedOccupiedCountStrideWords ?? 1,
    "Dense correction occupied-count stride");
  Object.entries(layout.offsetsWords).forEach(([name, value]) =>
    checkedInteger(value, `Dense correction ${name} offset`, true));
}

export const factorOneDenseCorrectionWGSL = /* wgsl */ `
struct Params{
 rootLevel:vec4u,shape:vec4u,offsets0:vec4u,offsets1:vec4u,offsets2:vec4u,
 numerics:vec4u,levelBases:array<vec4u,3>
}
@group(0)@binding(0)var<uniform>p:Params;
@group(0)@binding(1)var<storage,read_write>arena:array<u32>;
@group(0)@binding(2)var<storage,read>accepted:array<u32>;
@group(0)@binding(3)var<storage,read_write>control:array<atomic<u32>>;
@group(0)@binding(4)var<storage,read>inputRhs:array<f32>;
@group(0)@binding(5)var<storage,read_write>outputCorrection:array<f32>;
@group(0)@binding(6)var<storage,read_write>dispatches:array<u32>;
@group(0)@binding(7)var<storage,read>epochControl:array<u32>;
const ACTIVE=1u;const GHOST=2u;const INVALID=0xffffffffu;
const OVERFLOW=2u;const NONFINITE=4u;const NONPOSITIVE=8u;
fn finite(v:f32)->bool{return v==v&&abs(v)<=3.402823e38;}
fn stopped()->bool{return atomicLoad(&control[0])!=0u||atomicLoad(&control[1])!=0u;}
fn reportAt(flag:u32,stage:u32,index:u32){atomicOr(&control[0],flag);for(var retry=0u;retry<16u;retry+=1u){
 let claim=atomicCompareExchangeWeak(&control[6],0u,stage);if(claim.exchanged){atomicStore(&control[7],index);return;}
 if(claim.old_value!=0u){return;}}}
fn levels()->u32{return p.shape.y;}fn totalSlots()->u32{return p.shape.z;}
fn level()->u32{return p.rootLevel.w;}fn rows()->u32{return min(select(0u,accepted[2],arrayLength(&accepted)>=4u),p.shape.x);}
fn table(l:u32)->vec2u{let q=min(l,11u);return vec2u(q>>2u,q&3u);}
fn levelBase(l:u32)->u32{let t=table(l);return p.levelBases[t.x][t.y];}
fn dims(l:u32)->vec3u{let scale=1u<<l;return(p.rootLevel.xyz+vec3u(scale-1u))>>vec3u(l);}
fn volume(l:u32)->u32{let d=dims(l);return d.x*d.y*d.z;}
fn count(l:u32)->u32{return arena[p.offsets2.x+l*p.numerics.z];}
fn flag(s:u32)->u32{return arena[p.offsets0.x+s];}
fn owner(s:u32)->u32{return arena[p.offsets0.y+s];}
fn loadf(base:u32,s:u32)->f32{return bitcast<f32>(arena[base+s]);}
fn storef(base:u32,s:u32,v:f32){arena[base+s]=bitcast<u32>(v);}
fn workSlot(l:u32,i:u32)->u32{if(i>=count(l)){return INVALID;}let local=arena[p.offsets1.w+levelBase(l)+i];
 if(local>=volume(l)){reportAt(OVERFLOW,101u,local);return INVALID;}let s=levelBase(l)+local;
 if(flag(s)==0u){reportAt(OVERFLOW,102u,s);return INVALID;}return s;}
fn denseReadyFailure()->u32{
 if(arrayLength(&accepted)<3u||accepted[0]!=0u){return 1u;}
 if(arrayLength(&epochControl)<9u||epochControl[6]==0u){return 2u;}
 if(epochControl[8]!=0u){return 3u;}
 if(arena[p.offsets2.y]!=p.offsets2.w){return 4u;}
 if(arena[p.offsets2.z]!=epochControl[6]){return 5u;}
 if(arena[p.numerics.x]!=0u){return 6u;}
 return 0u;
}
fn denseReady()->bool{return denseReadyFailure()==0u;}
fn vectorItem(g:vec3u)->u32{return g.x+g.y*65535u*64u;}
fn parentItem(wg:vec3u)->u32{return wg.x+wg.y*65535u;}
fn writeRecord(at:u32,groups:u32){let x=max(1u,min(65535u,groups));dispatches[at]=select(x,0u,groups==0u);
 dispatches[at+1u]=select((groups+x-1u)/x,1u,groups==0u);dispatches[at+2u]=1u;}
@compute @workgroup_size(1)fn prepareDenseCorrectionDispatches(){var live=!stopped()&&denseReady();var maximum=rows();
 for(var l=0u;l<levels();l+=1u){let n=count(l);if(n>volume(l)){reportAt(OVERFLOW,103u,l);live=false;}maximum=max(maximum,n);
  writeRecord(6u*l,select(0u,(volume(l)+63u)/64u,live));var parents=0u;if(live&&l+1u<levels()){parents=count(l+1u);}
  writeRecord(6u*l+3u,parents);}
 let commonBase=6u*levels();writeRecord(commonBase,select(0u,(maximum+63u)/64u,live));
 if(!live&&!stopped()){let reason=denseReadyFailure();
  let detail=select(epochControl[6],arena[p.offsets2.z],reason==5u);
  reportAt(OVERFLOW,100u+reason,detail);}}
@compute @workgroup_size(64)fn initializeDenseCorrection(@builtin(global_invocation_id)g:vec3u){let i=vectorItem(g);if(stopped()){return;}
 for(var l=0u;l<levels();l+=1u){if(i>=count(l)){continue;}let s=workSlot(l,i);if(s==INVALID){continue;}
  storef(p.offsets0.w,s,0.0);storef(p.offsets1.x,s,0.0);storef(p.offsets1.y,s,0.0);
  if((flag(s)&ACTIVE)!=0u){let encoded=owner(s);if(encoded==0u||encoded>rows()){reportAt(OVERFLOW,104u,s);continue;}
   let row=encoded-1u;outputCorrection[row]=0.0;let v=inputRhs[row];
   if(!finite(v)){reportAt(NONFINITE,73u,row);}else{storef(p.offsets0.w,s,v);}}}}
fn localCoord(l:u32,s:u32)->vec3u{let d=dims(l);let local=s-levelBase(l);let yz=local/d.x;
 return vec3u(local-yz*d.x,yz%d.y,yz/d.y);}
fn directSlot(l:u32,q:vec3u)->u32{let d=dims(l);return levelBase(l)+q.x+d.x*(q.y+d.y*q.z);}
fn sorted6Sum(values:array<f32,6>)->f32{var sorted=values;for(var i=1u;i<6u;i+=1u){let value=sorted[i];var j=i;loop{if(j==0u||sorted[j-1u]<=value){break;}sorted[j]=sorted[j-1u];j-=1u;}sorted[j]=value;}var sum=0.0;for(var i=0u;i<6u;i+=1u){sum+=sorted[i];}return sum;}
fn sorted8Sum(values:array<f32,8>)->f32{var sorted=values;for(var i=1u;i<8u;i+=1u){let value=sorted[i];var j=i;loop{if(j==0u||sorted[j-1u]<=value){break;}sorted[j]=sorted[j-1u];j-=1u;}sorted[j]=value;}var sum=0.0;for(var i=0u;i<8u;i+=1u){sum+=sorted[i];}return sum;}
fn applied(l:u32,s:u32,sourceBase:u32)->f32{let q=localCoord(l,s);let d=dims(l);let c=f32(1u<<l)*bitcast<f32>(p.numerics.y);
 var terms:array<f32,6>;
 let directions=array<vec3i,6>(vec3i(1,0,0),vec3i(-1,0,0),vec3i(0,1,0),vec3i(0,-1,0),vec3i(0,0,1),vec3i(0,0,-1));
 for(var k=0u;k<6u;k+=1u){let other=vec3i(q)+directions[k];if(any(other<vec3i(0))||any(other>=vec3i(d))){continue;}
  let n=directSlot(l,vec3u(other));if(flag(n)!=0u){terms[k]=c*loadf(sourceBase,n);}}
 return loadf(p.offsets0.z,s)*loadf(sourceBase,s)-sorted6Sum(terms);}
fn chebyshevWeight(l:u32,phase:u32)->f32{let upper=loadf(p.offsets1.z,l);let lower=upper*${OCTREE_FIRST_ORDER_CHEBYSHEV_LOWER_FRACTION};
 if(!(lower>0.0)||!(upper>lower)||!finite(upper)){reportAt(NONPOSITIVE,76u,l);return 0.0;}
 let centre=0.5*(upper+lower);let radius=0.5*(upper-lower);
 return 1.0/(centre-radius*cos(3.141592653589793*(2.0*f32(phase)+1.0)/(2.0*f32(p.shape.w))));}
fn relax(l:u32,s:u32,src:u32,dst:u32,phase:u32){let source=loadf(src,s);if((flag(s)&GHOST)!=0u){storef(dst,s,source);return;}
 let diagonal=loadf(p.offsets0.z,s);if(!(diagonal>0.0)){reportAt(NONPOSITIVE,79u,s);storef(dst,s,source);return;}
 let next=source+chebyshevWeight(l,phase)*(loadf(p.offsets0.w,s)-applied(l,s,src))/diagonal;
 if(!finite(next)){reportAt(NONFINITE,80u,s);storef(dst,s,source);}else{storef(dst,s,next);}}
fn smoothOne(g:vec3u,src:u32,dst:u32,phase:u32){let i=vectorItem(g);let l=level();if(i<volume(l)&&!stopped()){
 let s=levelBase(l)+i;if(flag(s)!=0u){relax(l,s,src,dst,phase);}}}
@compute @workgroup_size(64)fn smoothDenseAtoB0(@builtin(global_invocation_id)g:vec3u){smoothOne(g,p.offsets1.x,p.offsets1.y,0u);}
@compute @workgroup_size(64)fn smoothDenseBtoA0(@builtin(global_invocation_id)g:vec3u){smoothOne(g,p.offsets1.y,p.offsets1.x,0u);}
@compute @workgroup_size(64)fn smoothDenseAtoB1(@builtin(global_invocation_id)g:vec3u){smoothOne(g,p.offsets1.x,p.offsets1.y,1u);}
@compute @workgroup_size(64)fn smoothDenseBtoA1(@builtin(global_invocation_id)g:vec3u){smoothOne(g,p.offsets1.y,p.offsets1.x,1u);}
@compute @workgroup_size(64)fn smoothDenseAtoB2(@builtin(global_invocation_id)g:vec3u){smoothOne(g,p.offsets1.x,p.offsets1.y,2u);}
@compute @workgroup_size(64)fn smoothDenseBtoA2(@builtin(global_invocation_id)g:vec3u){smoothOne(g,p.offsets1.y,p.offsets1.x,2u);}
@compute @workgroup_size(64)fn smoothDenseAtoB3(@builtin(global_invocation_id)g:vec3u){smoothOne(g,p.offsets1.x,p.offsets1.y,3u);}
@compute @workgroup_size(64)fn smoothDenseBtoA3(@builtin(global_invocation_id)g:vec3u){smoothOne(g,p.offsets1.y,p.offsets1.x,3u);}
fn publishNative(s:u32,value:f32){if((flag(s)&ACTIVE)==0u){return;}let encoded=owner(s);
 if(encoded==0u||encoded>rows()){reportAt(OVERFLOW,105u,s);return;}
 if(!finite(value)){reportAt(NONFINITE,92u,encoded-1u);}else{outputCorrection[encoded-1u]=value;}}
@compute @workgroup_size(64)fn smoothDenseBtoA0AndPublish(@builtin(global_invocation_id)g:vec3u){let i=vectorItem(g);let l=level();
 if(i<volume(l)&&!stopped()){let s=levelBase(l)+i;if(flag(s)!=0u){relax(l,s,p.offsets1.y,p.offsets1.x,0u);
  if(!stopped()){publishNative(s,loadf(p.offsets1.x,s));}}}}
var<workgroup> childResidual:array<f32,8>;var<workgroup> childFailed:array<u32,8>;
fn restrictParent(l:u32,coarse:u32,lane:u32){var residual=0.0;var failed=0u;if(lane<8u&&coarse!=INVALID){let parent=localCoord(l+1u,coarse);
 let child=parent*2u+vec3u(lane&1u,(lane>>1u)&1u,(lane>>2u)&1u);if(all(child<dims(l))){let fine=directSlot(l,child);
  if(flag(fine)!=0u){let product=applied(l,fine,p.offsets1.x);residual=select(-product,loadf(p.offsets0.w,fine)-product,(flag(fine)&GHOST)==0u);}}}
 if(lane<8u){childResidual[lane]=residual;childFailed[lane]=failed;}workgroupBarrier();
 if(lane==0u&&coarse!=INVALID){var values:array<f32,8>;var anyFailed=false;for(var child=0u;child<8u;child+=1u){values[child]=childResidual[child];anyFailed=anyFailed||childFailed[child]!=0u;}let sum=sorted8Sum(values);
  if(anyFailed||!finite(sum)){reportAt(OVERFLOW,87u,coarse);}else{storef(p.offsets0.w,coarse,loadf(p.offsets0.w,coarse)+sum);}}}
@compute @workgroup_size(64)fn restrictDenseAggregate(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lane:u32){
 let l=level();let i=parentItem(wg);var coarse=INVALID;if(i<count(l+1u)&&!stopped()){coarse=workSlot(l+1u,i);}
 restrictParent(l,coarse,lane);}
fn prolongOne(l:u32,fine:u32){let parent=directSlot(l+1u,localCoord(l,fine)/2u);if(flag(parent)==0u){reportAt(OVERFLOW,84u,fine);return;}
 var value=select(loadf(p.offsets1.x,fine),0.0,(flag(fine)&GHOST)!=0u);value+=loadf(p.offsets1.x,parent);
 if(!finite(value)){reportAt(NONFINITE,91u,fine);}else{storef(p.offsets1.x,fine,value);}}
@compute @workgroup_size(64)fn prolongDenseAggregate(@builtin(global_invocation_id)g:vec3u){let i=vectorItem(g);let l=level();
 if(i<volume(l)&&!stopped()){let fine=levelBase(l)+i;if(flag(fine)!=0u){prolongOne(l,fine);}}}
fn tailSmoothPhase(l:u32,src:u32,dst:u32,phase:u32,publish:bool,lane:u32){let n=count(l);for(var i=lane;i<n;i+=64u){if(stopped()){continue;}
 let s=workSlot(l,i);if(s!=INVALID){relax(l,s,src,dst,phase);
  if(publish&&!stopped()){publishNative(s,loadf(dst,s));}}}}
fn tailSmooth(l:u32,reverse:bool,lane:u32){for(var step=0u;step<p.shape.w;step+=1u){let phase=select(step,p.shape.w-1u-step,reverse);
 let publish=reverse&&step+1u==p.shape.w;
 if((step&1u)==0u){tailSmoothPhase(l,p.offsets1.x,p.offsets1.y,phase,publish,lane);}else{tailSmoothPhase(l,p.offsets1.y,p.offsets1.x,phase,publish,lane);}
 storageBarrier();workgroupBarrier();}}
fn tailRestrict(l:u32,lane:u32){if(l+1u>=levels()){return;}let n=count(l+1u);for(var i=lane;i<n;i+=64u){if(stopped()){continue;}
 let coarse=workSlot(l+1u,i);if(coarse!=INVALID){let parent=localCoord(l+1u,coarse);var values:array<f32,8>;for(var child=0u;child<8u;child+=1u){
  let q=parent*2u+vec3u(child&1u,(child>>1u)&1u,(child>>2u)&1u);if(any(q>=dims(l))){continue;}let fine=directSlot(l,q);
  if(flag(fine)==0u){continue;}let product=applied(l,fine,p.offsets1.x);
  values[child]=select(-product,loadf(p.offsets0.w,fine)-product,(flag(fine)&GHOST)==0u);}let sum=sorted8Sum(values);
 if(!finite(sum)){reportAt(OVERFLOW,87u,coarse);}else{storef(p.offsets0.w,coarse,loadf(p.offsets0.w,coarse)+sum);}}}}
fn tailProlong(l:u32,lane:u32){let n=count(l);for(var i=lane;i<n;i+=64u){if(!stopped()){let fine=workSlot(l,i);if(fine!=INVALID){prolongOne(l,fine);}}}}
fn tailPublish(l:u32,lane:u32){let n=count(l);for(var i=lane;i<n;i+=64u){if(!stopped()){let s=workSlot(l,i);
 if(s!=INVALID){publishNative(s,loadf(p.offsets1.x,s));}}}}
@compute @workgroup_size(64)fn denseVcycleTail(@builtin(local_invocation_index)lane:u32){let first=level();
 for(var l=first;l+1u<levels();l+=1u){tailSmooth(l,false,lane);tailRestrict(l,lane);storageBarrier();workgroupBarrier();}
 if(lane==0u&&!stopped()){let l=levels()-1u;if(count(l)!=1u){reportAt(NONPOSITIVE,88u,l);}else{let s=workSlot(l,0u);
  let diagonal=loadf(p.offsets0.z,s);if(!(diagonal>0.0)){reportAt(NONPOSITIVE,89u,s);}else{let x=loadf(p.offsets0.w,s)/diagonal;
   if(!finite(x)){reportAt(NONFINITE,90u,s);}else{storef(p.offsets1.x,s,x);}}}}
 storageBarrier();workgroupBarrier();tailPublish(levels()-1u,lane);
 for(var signed=i32(levels())-2;signed>=i32(first);signed-=1){let l=u32(signed);
  tailProlong(l,lane);storageBarrier();workgroupBarrier();tailSmooth(l,true,lane);}}
`;

type CachedGroup = {
  readonly control: GPUBuffer;
  readonly rhs?: GPUBuffer;
  readonly correction?: GPUBuffer;
  readonly group: GPUBindGroup;
};

export class WebGPUOctreeFactorOneDenseCorrection {
  readonly encodedPassTransitionCount = 1;
  readonly encodedCorrectionDispatchCount: number;
  readonly allocatedBytes: number;
  readonly smootherContract;
  private readonly params: readonly GPUBuffer[];
  private readonly pipelines: Readonly<Record<DensePipelineName, GPUComputePipeline>>;
  private readonly groups = new Map<string, CachedGroup>();
  private readonly dispatches: GPUBuffer;
  private readonly degree: 2 | 4;
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    readonly source: FactorOneDenseCorrectionSource,
    options: { readonly smoothingIterations?: 2 | 4 } = {},
  ) {
    validateLayout(source.layout);
    if (source.worklistIndexKind !== "level-local") {
      throw new RangeError("Dense correction requires level-local occupied indices");
    }
    const requested = options.smoothingIterations ?? 4;
    if (!OCTREE_FIRST_ORDER_CHEBYSHEV_DEGREES.includes(requested)) {
      throw new RangeError("Dense correction smoothing degree must be exactly 2 or 4");
    }
    this.degree = requested;
    this.smootherContract = Object.freeze({
      kind: "chebyshev" as const,
      degree: this.degree,
      spectralBounds: "transactional-scaled-gershgorin" as const,
      lowerFraction: OCTREE_FIRST_ORDER_CHEBYSHEV_LOWER_FRACTION,
    });
    const layout = source.layout;
    this.dispatches = device.createBuffer({
      label: "Factor-1 dense M1 convergence-gated dispatches",
      size: (WORDS_PER_LEVEL * layout.levelCount + 3) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
    });
    const makeParams = (level: number) => {
      const buffer = device.createBuffer({
        label: `Factor-1 dense M1 level ${level} parameters`,
        size: PARAMS_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const words = new Uint32Array(PARAMS_BYTES / 4);
      words.set([...layout.dimensions, level], 0);
      words.set([layout.rowCapacity, layout.levelCount, layout.totalSlots, this.degree], 4);
      const offsets = layout.offsetsWords;
      words.set([offsets.acceptedFlags, offsets.acceptedOwners,
        offsets.acceptedDiagonal, offsets.rhs], 8);
      words.set([offsets.a, offsets.b, offsets.spectralUpper,
        offsets.acceptedOccupiedIndices], 12);
      words.set([offsets.acceptedOccupiedCounts, offsets.publicationValid,
        offsets.publicationEpoch, layout.publicationValidValue
          ?? FACTOR_ONE_DENSE_CORRECTION_VALID], 16);
      words.set([offsets.publicationError,
        new Uint32Array(new Float32Array([layout.finestCellWidth]).buffer)[0]!,
        layout.acceptedOccupiedCountStrideWords ?? 1, 0], 20);
      for (let l = 0; l < layout.levelCount; l += 1) words[24 + l] = layout.levelBases[l]!;
      device.queue.writeBuffer(buffer, 0, words);
      return buffer;
    };
    this.params = Object.freeze(Array.from(
      { length: layout.levelCount }, (_, level) => makeParams(level),
    ));
    const module = device.createShaderModule({
      label: "Factor-1 dense-indexed M1 correction", code: factorOneDenseCorrectionWGSL,
    });
    const make = (entryPoint: DensePipelineName) => device.createComputePipeline({
      label: `Factor-1 dense M1 · ${entryPoint}`,
      layout: "auto", compute: { module, entryPoint },
    });
    this.pipelines = Object.freeze(Object.fromEntries(
      (Object.keys(FACTOR_ONE_DENSE_CORRECTION_BINDINGS) as DensePipelineName[])
        .map((name) => [name, make(name)]),
    )) as Readonly<Record<DensePipelineName, GPUComputePipeline>>;
    const parallelLevels = Math.min(2, layout.levelCount - 1);
    this.encodedCorrectionDispatchCount = 3
      + parallelLevels * (2 * this.degree + 2);
    this.allocatedBytes = this.dispatches.size
      + this.params.reduce((sum, buffer) => sum + buffer.size, 0);
  }

  encodeCorrection(
    broker: PassBroker,
    input: { rhs: GPUBuffer; correction: GPUBuffer; solverControl: GPUBuffer },
  ): void {
    const pass = broker.compute({ label: "Factor-1 dense M1 · convergence gate" });
    this.encodeCorrectionGate(pass, input);
    broker.fence("Factor-1 dense M1 convergence-gated indirect publication");
    this.encodeCorrectionBody(broker, input);
  }

  encodeCorrectionGate(
    pass: GPUComputePassEncoder,
    input: { rhs: GPUBuffer; correction: GPUBuffer; solverControl: GPUBuffer },
  ): void {
    this.assertInput(input);
    this.bind(pass, "prepareDenseCorrectionDispatches", 0, input);
    pass.dispatchWorkgroups(1);
  }

  encodeCorrectionBody(
    broker: PassBroker,
    input: { rhs: GPUBuffer; correction: GPUBuffer; solverControl: GPUBuffer },
  ): void {
    this.assertInput(input);
    let pass = broker.compute({ label: "Factor-1 dense M1 · initialize correction" });
    this.runIndirect(pass, "initializeDenseCorrection", 0,
      this.source.layout.levelCount * BYTES_PER_LEVEL, input);
    const parallelLevels = Math.min(2, this.source.layout.levelCount - 1);
    for (let level = 0; level < parallelLevels; level += 1) {
      this.smooth(broker, level, false, input);
      this.runIndirect(broker.compute({
        label: `Factor-1 dense M1 · restrict level ${level}`,
      }), "restrictDenseAggregate", level, level * BYTES_PER_LEVEL + 12, input);
    }
    this.runDirect(broker.compute({
      label: `Factor-1 dense M1 · tail levels ${parallelLevels}-bottom`,
    }), "denseVcycleTail", parallelLevels, input);
    for (let level = parallelLevels - 1; level >= 0; level -= 1) {
      this.runIndirect(broker.compute({
        label: `Factor-1 dense M1 · prolong level ${level}`,
      }), "prolongDenseAggregate", level, level * BYTES_PER_LEVEL, input);
      this.smooth(broker, level, true, input);
    }
  }

  private smooth(
    broker: PassBroker,
    level: number,
    reverse: boolean,
    input: { rhs: GPUBuffer; correction: GPUBuffer; solverControl: GPUBuffer },
  ): void {
    for (let step = 0; step < this.degree; step += 1) {
      const phase = reverse ? this.degree - 1 - step : step;
      const direction = (step & 1) === 0 ? "AtoB" : "BtoA";
      const finalNativePublication = reverse && step + 1 === this.degree;
      const name = finalNativePublication
        ? "smoothDenseBtoA0AndPublish"
        : `smoothDense${direction}${phase}` as DensePipelineName;
      this.runIndirect(broker.compute({
        label: `Factor-1 dense M1 · ${reverse ? "post" : "pre"}-smooth level ${level}`,
      }), name, level, level * BYTES_PER_LEVEL, input);
    }
  }

  private runIndirect(
    pass: GPUComputePassEncoder,
    name: DensePipelineName,
    level: number,
    offset: number,
    input: { rhs: GPUBuffer; correction: GPUBuffer; solverControl: GPUBuffer },
  ): void {
    this.bind(pass, name, level, input);
    pass.dispatchWorkgroupsIndirect(this.dispatches, offset);
  }

  private runDirect(
    pass: GPUComputePassEncoder,
    name: DensePipelineName,
    level: number,
    input: { rhs: GPUBuffer; correction: GPUBuffer; solverControl: GPUBuffer },
  ): void {
    this.bind(pass, name, level, input);
    pass.dispatchWorkgroups(1);
  }

  private bind(
    pass: GPUComputePassEncoder,
    name: DensePipelineName,
    level: number,
    input: { rhs: GPUBuffer; correction: GPUBuffer; solverControl: GPUBuffer },
  ): void {
    const key = `${name}:${level}`;
    const cached = this.groups.get(key);
    let group = cached?.group;
    if (!cached || cached.control !== input.solverControl
      || cached.rhs !== input.rhs || cached.correction !== input.correction) {
      const resources = new Map<number, GPUBuffer>([
        [0, this.params[level]!], [1, this.source.arena],
        [2, this.source.acceptedControl], [3, input.solverControl],
        [4, input.rhs], [5, input.correction], [6, this.dispatches],
        [7, this.source.epochControl],
      ]);
      const pipeline = this.pipelines[name];
      group = this.device.createBindGroup({
        label: `Factor-1 dense M1 · ${name} · level ${level}`,
        layout: pipeline.getBindGroupLayout(0),
        entries: FACTOR_ONE_DENSE_CORRECTION_BINDINGS[name].map((binding) => ({
          binding, resource: { buffer: resources.get(binding)! },
        })),
      });
      this.groups.set(key, {
        control: input.solverControl, rhs: input.rhs,
        correction: input.correction, group,
      });
    }
    pass.setPipeline(this.pipelines[name]);
    pass.setBindGroup(0, group!);
  }

  private assertInput(
    input: { rhs: GPUBuffer; correction: GPUBuffer; solverControl: GPUBuffer },
  ): void {
    if (this.destroyed) throw new Error("Factor-1 dense M1 correction is destroyed");
    const bytes = this.source.layout.rowCapacity * 4;
    if (input.rhs.size < bytes || input.correction.size < bytes) {
      throw new RangeError("Factor-1 dense M1 row vectors are smaller than row capacity");
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.params.forEach((buffer) => buffer.destroy());
    this.dispatches.destroy();
    this.groups.clear();
  }
}
