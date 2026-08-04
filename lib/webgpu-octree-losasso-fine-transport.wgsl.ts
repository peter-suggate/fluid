import { fineLevelSetPackedSampleWGSL } from "./fine-levelset-packed-sample";
import { makeFineLevelSetSortedWorklistLookupWGSL } from "./webgpu-octree-fine-levelset-bricks";
import { octreeLosassoVelocitySamplingWGSL } from "./webgpu-octree-losasso-velocity-sampler.wgsl";

export const octreeLosassoFineTransportWGSL = /* wgsl */ `
const INVALID:u32=0xffffffffu;const VALID:u32=1u;const PAGE_INTERFACE:u32=2u;const PAGE_DIRTY:u32=8u;
struct Params{
 brickDimensions:vec3u,brickResolution:u32,
 sampleDimensions:vec3u,samplesPerBrick:u32,
 domainOrigin:vec3f,fineCellWidth:f32,
 pageCapacity:u32,generation:u32,worklistCapacity:u32,worklistHeaderWords:u32,
 velocityDimensions:vec3u,directoryCapacity:u32,
 velocityCellSize:f32,maximumLeafSize:u32,reservedVelocity0:u32,reservedVelocity1:u32,
 bandCells:u32,maxBacktrace:u32,closed:u32,openTop:u32,
 dt:f32,expectedVelocityEpoch:u32,reserved0:u32,reserved1:u32,
}
@group(0)@binding(0)var<uniform>p:Params;
@group(0)@binding(1)var<storage,read_write>metadata:array<u32>;
@group(0)@binding(2)var<storage,read>worklist:array<u32>;
@group(0)@binding(3)var<storage,read_write>samples:array<u32>;
@group(0)@binding(4)var<storage,read_write>nextPhi:array<f32>;
@group(0)@binding(5)var<storage,read_write>control:array<atomic<u32>>;
@group(0)@binding(6)var<storage,read>coarse:array<u32>;
@group(0)@binding(7)var<storage,read>faceGeometry:array<vec4u>;
@group(0)@binding(8)var<storage,read>extendedVelocity:array<f32>;
@group(0)@binding(9)var<storage,read>faceDirectory:array<vec2u>;
@group(0)@binding(10)var<storage,read_write>delta:array<u32>;
fn finite(v:f32)->bool{return v==v&&abs(v)<3.402823e38;}
fn finite3(v:vec3f)->bool{return all(v==v)&&all(abs(v)<vec3f(3.402823e38));}
${fineLevelSetPackedSampleWGSL("samples", true)}
${makeFineLevelSetSortedWorklistLookupWGSL("p", "metadata", "worklist", "losassoFineBrick")}
${octreeLosassoVelocitySamplingWGSL}
fn packBrick(q:vec3u)->u32{return q.x+p.brickDimensions.x*(q.y+p.brickDimensions.y*q.z);}
fn localCoord(index:u32)->vec3u{let z=index/(p.brickResolution*p.brickResolution);
 let rem=index-z*p.brickResolution*p.brickResolution;let y=rem/p.brickResolution;
 return vec3u(rem-y*p.brickResolution,y,z);}
fn sampleIndex(q:vec3u)->u32{let brick=q/p.brickResolution;let page=losassoFineBrick(packBrick(brick));
 if(page==INVALID){return INVALID;}let local=q-brick*p.brickResolution;
 return page*p.samplesPerBrick+local.x+p.brickResolution*(local.y+p.brickResolution*local.z);}
fn sampleFinePhi(position:vec3f)->f32{
 let grid=(position-p.domainOrigin)/p.fineCellWidth-vec3f(.5);
 let base=vec3i(floor(grid));let fraction=fract(grid);var exact:array<i32,36>;
 for(var corner=0u;corner<8u;corner+=1u){let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
  let weights=vec3f(select(1.-fraction.x,fraction.x,(corner&1u)!=0u),
   select(1.-fraction.y,fraction.y,(corner&2u)!=0u),
   select(1.-fraction.z,fraction.z,(corner&4u)!=0u));
  let weight=losassoSymmetricWeight(weights);if(weight==0.){continue;}
  let coordinate=base+offset;if(any(coordinate<vec3i(0))||any(coordinate>=vec3i(p.sampleDimensions))){return 3.402823e38;}
  let index=sampleIndex(vec3u(coordinate));if(index==INVALID||index>=arrayLength(&samples)||(finePackedFlags(index)&VALID)==0u){return 3.402823e38;}
  let value=finePackedPhi(index);if(!finite(value)){return 3.402823e38;}
  losassoExactAdd(&exact,weight*value);
 }
 return losassoExactValue(exact);
}
fn acceptedVelocity()->bool{return arrayLength(&coarse)>=4u&&coarse[3]==1u
 &&(p.expectedVelocityEpoch==0u||coarse[0]==p.expectedVelocityEpoch)
 &&coarse[2]<=arrayLength(&faceGeometry)&&coarse[2]<=arrayLength(&extendedVelocity)
 &&p.directoryCapacity<=arrayLength(&faceDirectory);}
fn acceptedStep()->bool{return atomicLoad(&control[6])==0u&&atomicLoad(&control[7])==0u
 &&atomicLoad(&control[1])==0u&&atomicLoad(&control[0])==0u;}

@compute @workgroup_size(64)
fn prepareLosassoFineTransport(@builtin(local_invocation_index)lane:u32){
 if(lane<16u){atomicStore(&control[lane],select(0u,INVALID,lane==12u));}if(lane<8u){delta[lane]=0u;}workgroupBarrier();
 if(lane==0u){if(!acceptedVelocity()){atomicStore(&control[6],1u);}
  if(arrayLength(&worklist)<7u||worklist[0]!=p.generation||worklist[2]!=p.pageCapacity
    ||worklist[3]!=3u||worklist[5]!=1u||worklist[6]!=1u){atomicStore(&control[7],1u);}}
}

@compute @workgroup_size(64)
fn advectLosassoFinePhi(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){
 let work=group.x;if(!acceptedVelocity()||work>=min(worklist[1],p.pageCapacity)){return;}
 let page=worklist[7u+work];if(page>=p.pageCapacity||metadata[4u*page+2u]!=p.generation){atomicAdd(&control[7],1u);atomicAdd(&control[11],1u);return;}
 let brickKey=metadata[4u*page+1u];let xy=p.brickDimensions.x*p.brickDimensions.y;
 let bz=brickKey/xy;let remainder=brickKey-bz*xy;let by=remainder/p.brickDimensions.x;
 let brick=vec3u(remainder-by*p.brickDimensions.x,by,bz);
 for(var local=lane;local<p.samplesPerBrick;local+=64u){let index=page*p.samplesPerBrick+local;
  if(index>=arrayLength(&samples)||index>=arrayLength(&nextPhi)||(finePackedFlags(index)&VALID)==0u){continue;}
  let old=finePackedPhi(index);nextPhi[index]=old;if(!finite(old)){atomicAdd(&control[1],1u);continue;}
  let q=brick*p.brickResolution+localCoord(local);if(any(q>=p.sampleDimensions)){continue;}
  if(abs(old)>f32(p.bandCells)*p.fineCellWidth){atomicAdd(&control[2],1u);continue;}
  let position=p.domainOrigin+(vec3f(q)+vec3f(.5))*p.fineCellWidth;
  let first=losassoVelocityAt(position);if(first.valid==0u){atomicAdd(&control[7],1u);atomicAdd(&control[8],1u);atomicMin(&control[12],index);continue;}
  var midpoint=position-.5*p.dt*first.value;let low=p.domainOrigin+vec3f(.5*p.fineCellWidth);
  var high=p.domainOrigin+(vec3f(p.sampleDimensions)-vec3f(.5))*p.fineCellWidth;
  if(p.openTop!=0u){high.y=max(high.y,midpoint.y);}
  if(p.closed!=0u){midpoint=clamp(midpoint,low,high);}else if(any(midpoint<low)||any(midpoint>high)){atomicAdd(&control[0],1u);continue;}
  let middle=losassoVelocityAt(midpoint);if(middle.valid==0u){atomicAdd(&control[7],1u);atomicAdd(&control[9],1u);atomicMin(&control[12],index);continue;}
  var departure=position-p.dt*middle.value;if(p.openTop!=0u){high.y=max(high.y,departure.y);}
  if(p.closed!=0u){departure=clamp(departure,low,high);}else if(any(departure<low)||any(departure>high)){atomicAdd(&control[0],1u);continue;}
  let displacement=u32(ceil(length(middle.value)*p.dt/max(p.fineCellWidth,1e-20)));
  atomicMax(&control[5],displacement);if(displacement>p.maxBacktrace){atomicAdd(&control[0],1u);continue;}
  let transported=sampleFinePhi(departure);if(!finite(transported)){atomicAdd(&control[7],1u);atomicAdd(&control[10],1u);atomicMin(&control[12],index);continue;}
  nextPhi[index]=transported;atomicAdd(&control[2],1u);
 }
}

var<workgroup> changed:array<u32,64>;
@compute @workgroup_size(64)
fn commitLosassoFinePhi(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){
 let work=group.x;var page=INVALID;let live=acceptedStep()&&work<min(worklist[1],p.pageCapacity);
 if(live){page=worklist[7u+work];if(page>=p.pageCapacity||metadata[4u*page+2u]!=p.generation){page=INVALID;}}
 // One reduction carries three independent facts: liquid samples, air
 // samples, and a sign change. Recurring topology is seeded from the current
 // zero set, not merely from samples that happened to cross zero this step;
 // a smooth sub-cell translation can move the interface while changing no
 // sample sign at all. The near-zero bit also catches a crossing that lies on
 // a brick boundary, where the neighboring pages can each be single-signed.
 var localChanged=0u;if(page!=INVALID){for(var local=lane;local<p.samplesPerBrick;local+=64u){let index=page*p.samplesPerBrick+local;
  if((finePackedFlags(index)&VALID)==0u){continue;}let old=finePackedPhi(index);let fresh=nextPhi[index];
  if(!finite(fresh)){localChanged|=4u;continue;}
  localChanged|=select(2u,1u,fresh<0.);
  localChanged|=select(0u,4u,(old<0.)!=(fresh<0.));
  localChanged|=select(0u,8u,abs(fresh)<=p.fineCellWidth);}}
 changed[lane]=localChanged;workgroupBarrier();for(var width=32u;width>0u;width>>=1u){if(lane<width){changed[lane]|=changed[lane+width];}workgroupBarrier();}
 if(page!=INVALID&&lane==0u){let key=metadata[4u*page+1u];let prior=metadata[4u*page+3u];
  let dirty=(changed[0]&4u)!=0u;let onInterface=((changed[0]&3u)==3u)||(changed[0]&8u)!=0u;
  metadata[4u*page+3u]=(prior&~(PAGE_INTERFACE|PAGE_DIRTY))
   |select(0u,PAGE_INTERFACE,onInterface)|select(0u,PAGE_DIRTY,dirty);
  delta[8u+page]=select(INVALID,key,onInterface);
  delta[8u+p.pageCapacity+work]=key;delta[8u+2u*p.pageCapacity+page]=select(INVALID,key,dirty);}
 workgroupBarrier();if(page!=INVALID){for(var local=lane;local<p.samplesPerBrick;local+=64u){let index=page*p.samplesPerBrick+local;
  if((finePackedFlags(index)&VALID)!=0u){fineWritePackedPhi(index,nextPhi[index]);}}}
}

@compute @workgroup_size(64)
fn finalizeLosassoFineTransport(@builtin(local_invocation_index)lane:u32){if(lane!=0u){return;}
 let accepted=acceptedStep()&&atomicLoad(&control[2])>0u;let count=select(0u,min(worklist[1],p.pageCapacity),accepted);
 atomicStore(&control[3],select(0u,1u,accepted));delta[0]=count;delta[1]=p.generation;
 delta[2]=select(0u,1u,accepted);delta[3]=min(worklist[1],p.pageCapacity);
 delta[4]=select(0u,(count+63u)/64u,count>0u);delta[5]=1u;delta[6]=1u;delta[7]=atomicLoad(&control[5]);
}
`;
