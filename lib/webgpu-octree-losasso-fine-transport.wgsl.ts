import { fineLevelSetPackedSampleWGSL } from "./fine-levelset-packed-sample";
import { makeFineLevelSetSortedWorklistLookupWGSL } from "./webgpu-octree-fine-levelset-bricks";
import { octreeLosassoVelocitySamplingWGSL } from "./webgpu-octree-losasso-velocity-sampler.wgsl";

export const octreeLosassoFineTransportWGSL = /* wgsl */ `
const INVALID:u32=0xffffffffu;const VALID:u32=1u;const PAGE_INTERFACE:u32=2u;const PAGE_DIRTY:u32=8u;
const PAGE_ACTIVITY_VALID:u32=16u;const PAGE_ACTIVITY_MOVING:u32=32u;const PAGE_WAKE_HALO:u32=64u;
const PAGE_SIZING_CHANGED:u32=128u;const PAGE_SIZING_SHIFT:u32=16u;const PAGE_SIZING_MASK:u32=0xffu<<PAGE_SIZING_SHIFT;
struct Params{
 brickDimensions:vec3u,brickResolution:u32,
 sampleDimensions:vec3u,samplesPerBrick:u32,
 domainOrigin:vec3f,fineCellWidth:f32,
 pageCapacity:u32,generation:u32,worklistCapacity:u32,worklistHeaderWords:u32,
 velocityDimensions:vec3u,directoryCapacity:u32,
 velocityCellSize:f32,maximumLeafSize:u32,reservedVelocity0:u32,reservedVelocity1:u32,
 bandCells:u32,maxBacktrace:u32,closed:u32,openTop:u32,
 dt:f32,expectedVelocityEpoch:u32,substeps:u32,reserved1:u32,
 inflowPositionRadius:vec4f,inflowVelocityStrength:vec4f,
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
@group(0)@binding(11)var<storage,read_write>liveDispatch:array<atomic<u32>>;
@group(0)@binding(12)var<storage,read_write>reversePhi:array<f32>;
fn finite(v:f32)->bool{return v==v&&abs(v)<3.402823e38;}
fn finite3(v:vec3f)->bool{return all(v==v)&&all(abs(v)<vec3f(3.402823e38));}
// Keep characteristic offsets on a lattice finer than 1e-6 m at factor four.
// This is still coarser than the ulp of the largest sample index, so adding an
// offset to q and its reflected opposite produces complementary fractions
// instead of losing a different low bit on the two sides of the tank.
fn quantizeTraceOffset(v:vec3f)->vec3f{return round(v*65536.)/65536.;}
${fineLevelSetPackedSampleWGSL("samples", true)}
${makeFineLevelSetSortedWorklistLookupWGSL("p", "metadata", "worklist", "losassoFineBrick")}
${octreeLosassoVelocitySamplingWGSL}
fn packBrick(q:vec3u)->u32{return q.x+p.brickDimensions.x*(q.y+p.brickDimensions.y*q.z);}
fn unpackBrick(key:u32)->vec3u{let xy=p.brickDimensions.x*p.brickDimensions.y;let z=key/xy;
 let rem=key-z*xy;let y=rem/p.brickDimensions.x;return vec3u(rem-y*p.brickDimensions.x,y,z);}
fn localCoord(index:u32)->vec3u{let z=index/(p.brickResolution*p.brickResolution);
 let rem=index-z*p.brickResolution*p.brickResolution;let y=rem/p.brickResolution;
 return vec3u(rem-y*p.brickResolution,y,z);}
fn sampleIndex(q:vec3u)->u32{let brick=q/p.brickResolution;let page=losassoFineBrick(packBrick(brick));
 if(page==INVALID){return INVALID;}let local=q-brick*p.brickResolution;
 return page*p.samplesPerBrick+local.x+p.brickResolution*(local.y+p.brickResolution*local.z);}
fn sampleFinePhiGrid(grid:vec3f)->f32{
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
fn sampleNextPhiGrid(grid:vec3f)->f32{
 let base=vec3i(floor(grid));let fraction=fract(grid);var value=0.;
 for(var corner=0u;corner<8u;corner+=1u){let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
  let weight=select(1.-fraction.x,fraction.x,(corner&1u)!=0u)
   *select(1.-fraction.y,fraction.y,(corner&2u)!=0u)
   *select(1.-fraction.z,fraction.z,(corner&4u)!=0u);
  let coordinate=base+offset;if(any(coordinate<vec3i(0))||any(coordinate>=vec3i(p.sampleDimensions))){return 3.402823e38;}
  let index=sampleIndex(vec3u(coordinate));if(index==INVALID||index>=arrayLength(&nextPhi)||!finite(nextPhi[index])){return 3.402823e38;}
  value+=weight*nextPhi[index];}
 return value;
}
fn traceFine(position:vec3f,forward:bool)->vec4f{
 let low=vec3f(0);var high=vec3f(p.sampleDimensions)-vec3f(1);let sign=select(-1.,1.,forward);
 let firstGrid=(position+vec3f(.5))*(p.fineCellWidth/p.velocityCellSize);let first=losassoVelocityAtGrid(firstGrid);
 if(first.valid==0u){return vec4f(position,0.);}var midpoint=position+quantizeTraceOffset(sign*.5*(p.dt/p.fineCellWidth)*first.value);
 if(p.openTop!=0u){high.y=max(high.y,midpoint.y);}if(any(midpoint<low)||any(midpoint>high)){return vec4f(position,0.);}
 let middleGrid=(midpoint+vec3f(.5))*(p.fineCellWidth/p.velocityCellSize);let middle=losassoVelocityAtGrid(middleGrid);
 if(middle.valid==0u){return vec4f(position,0.);}let traced=position+quantizeTraceOffset(sign*(p.dt/p.fineCellWidth)*middle.value);
 if(p.openTop!=0u){high.y=max(high.y,traced.y);}return vec4f(traced,select(0.,1.,all(traced>=low)&&all(traced<=high)));
}
fn inflowSample(world:vec3f)->bool{return p.inflowVelocityStrength.w>0.&&p.inflowPositionRadius.w>0.
 &&inflowSourcePhi(world)<=.70710678*p.fineCellWidth;}
fn oldDonorBounds(grid:vec3f)->vec3f{let base=vec3i(floor(grid));var lo=3.402823e38;var hi=-3.402823e38;
 for(var corner=0u;corner<8u;corner+=1u){let coordinate=base+vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
  if(any(coordinate<vec3i(0))||any(coordinate>=vec3i(p.sampleDimensions))){return vec3f(0);}
  let index=sampleIndex(vec3u(coordinate));if(index==INVALID||index>=arrayLength(&samples)||(finePackedFlags(index)&VALID)==0u){return vec3f(0);}
  let value=finePackedPhi(index);if(!finite(value)){return vec3f(0);}lo=min(lo,value);hi=max(hi,value);}
 return vec3f(lo,hi,1.);
}
fn pageSizingScore(page:u32)->f32{if(page>=p.pageCapacity){return 0.;}
 return f32((metadata[4u*page+3u]&PAGE_SIZING_MASK)>>PAGE_SIZING_SHIFT)/255.;}
fn advectedSizingScore(grid:vec3f)->f32{let q=vec3i(floor(grid));if(any(q<vec3i(0))||any(q>=vec3i(p.sampleDimensions))){return 0.;}
 let index=sampleIndex(vec3u(q));return select(0.,pageSizingScore(index/p.samplesPerBrick),index!=INVALID);}
// Ando-Batty section 6 sizing evidence on the fine lattice.  The phi term is
// the discrete Laplacian of the carved field (therefore including rigid
// contact curvature); the velocity term is the diagonal derivative magnitude
// of the accepted extended MAC field.  Both are dimensionless before their
// conservative activation scales are applied.
fn evaluatedSizingScore(grid:vec3f,centerPhi:f32)->f32{
 if(!finite(centerPhi)||abs(centerPhi)>2.*p.fineCellWidth){return 0.;}
 var laplacian=0.;var diagonalDerivative=0.;let ratio=p.fineCellWidth/p.velocityCellSize;
 for(var axis=0u;axis<3u;axis+=1u){var offset=vec3f(0);offset[axis]=1.;
  let lo=sampleFinePhiGrid(grid-offset);let hi=sampleFinePhiGrid(grid+offset);
  if(finite(lo)&&finite(hi)){laplacian+=lo-2.*centerPhi+hi;}
  let velocityLo=losassoVelocityAtGrid((grid-offset+vec3f(.5))*ratio);
  let velocityHi=losassoVelocityAtGrid((grid+offset+vec3f(.5))*ratio);
  if(velocityLo.valid!=0u&&velocityHi.valid!=0u){
   diagonalDerivative+=abs(velocityHi.value[axis]-velocityLo.value[axis])/(2.*p.fineCellWidth);}
 }
 let curvature=abs(laplacian)/max(p.fineCellWidth,1e-9);
 return clamp(max(curvature/.125,p.dt*diagonalDerivative/.04),0.,1.);
}
fn acceptedVelocity()->bool{return arrayLength(&coarse)>=7u&&coarse[3]==1u
 &&(p.expectedVelocityEpoch==0u||coarse[0]==p.expectedVelocityEpoch)
 &&coarse[2]<=arrayLength(&faceGeometry)&&coarse[2]<=arrayLength(&extendedVelocity)
 &&losassoDirectoryCapacity()>=2u*coarse[2]
 &&losassoDirectoryCapacity()<=arrayLength(&faceDirectory);}
fn acceptedStep()->bool{return atomicLoad(&control[6])==0u&&atomicLoad(&control[7])==0u
 &&atomicLoad(&control[1])==0u&&atomicLoad(&control[0])==0u;}
fn pageAwake(flags:u32)->bool{return (flags&PAGE_ACTIVITY_VALID)==0u
 ||(flags&(PAGE_INTERFACE|PAGE_ACTIVITY_MOVING|PAGE_WAKE_HALO|PAGE_DIRTY))!=0u
 ||p.inflowVelocityStrength.w>0.;}
// The nozzle is an authored boundary condition around Section 5 transport.
// Reapply its short downstream plug after every accepted characteristic;
// topology initialization alone only reaches newly allocated pages and lets
// a resident source disappear after its first generation.
fn inflowLatticePosition()->vec3f{let extent=vec3f(p.sampleDimensions)*p.fineCellWidth;
 return p.inflowPositionRadius.xyz+vec3f(.5*extent.x,0.,.5*extent.z);}
fn inflowSourcePhi(position:vec3f)->f32{
 let velocity=p.inflowVelocityStrength.xyz;let speed=length(velocity);if(speed<=1e-6){return 3.402823e38;}
 let direction=velocity/speed;let relative=position-inflowLatticePosition();let axial=dot(relative,direction);
 let radial=length(relative-axial*direction);let depth=2.*p.velocityCellSize;
 return max(radial-p.inflowPositionRadius.w,max(-axial,axial-depth));
}
fn applyInflowPhi(value:f32,position:vec3f)->f32{
 let strength=clamp(p.inflowVelocityStrength.w,0.,1.);let source=inflowSourcePhi(position);
 let enabled=strength>0.&&p.inflowPositionRadius.w>0.&&source<=.70710678*p.fineCellWidth;
 return select(value,min(value,max(source,-.5*p.fineCellWidth*strength)),enabled);
}

@compute @workgroup_size(64)
fn prepareLosassoFineTransport(@builtin(local_invocation_index)lane:u32){
 if(lane<16u){atomicStore(&control[lane],select(0u,INVALID,lane==12u||lane==13u));}if(lane<8u){delta[lane]=0u;}workgroupBarrier();
 if(lane==0u){let velocityValid=acceptedVelocity();let worklistValid=arrayLength(&worklist)>=7u
   &&worklist[0]==p.generation&&worklist[2]==p.pageCapacity
   &&worklist[3]==3u&&worklist[5]==1u&&worklist[6]==1u;
  if(!velocityValid){atomicStore(&control[6],1u);}if(!worklistValid){atomicStore(&control[7],1u);}
  let live=select(0u,min(worklist[1],p.pageCapacity),velocityValid&&worklistValid);
  atomicStore(&liveDispatch[0],(live+63u)/64u);atomicStore(&liveDispatch[1],1u);atomicStore(&liveDispatch[2],1u);
  atomicStore(&liveDispatch[4],0u);atomicStore(&liveDispatch[5],1u);atomicStore(&liveDispatch[6],1u);
  atomicStore(&liveDispatch[8],live);atomicStore(&liveDispatch[9],1u);atomicStore(&liveDispatch[10],1u);}
}

@compute @workgroup_size(64)
fn classifyLosassoFineActivity(@builtin(global_invocation_id)gid:vec3u){
 let work=gid.x;let live=min(worklist[1],p.pageCapacity);if(!acceptedVelocity()||work>=live){return;}
 let page=worklist[7u+work];if(page>=p.pageCapacity||metadata[4u*page+2u]!=p.generation){
  atomicAdd(&control[7],1u);atomicAdd(&control[11],1u);return;}
 let awake=pageAwake(metadata[4u*page+3u]);delta[8u+page]=select(0u,0x80000000u,awake);
 if(awake){let rank=atomicAdd(&control[14],1u);if(rank<p.pageCapacity){delta[8u+2u*p.pageCapacity+rank]=work;}else{atomicAdd(&control[0],1u);}}
 else{atomicAdd(&control[15],1u);}
}

@compute @workgroup_size(1)
fn finalizeLosassoFineActivity(){
 let live=min(worklist[1],p.pageCapacity);let activeCount=atomicLoad(&control[14]);let sleeping=atomicLoad(&control[15]);
 if(activeCount+sleeping!=live){atomicAdd(&control[7],1u);atomicAdd(&control[11],1u);}
 atomicStore(&liveDispatch[4],select(0u,activeCount,acceptedStep()));
 atomicStore(&liveDispatch[5],1u);atomicStore(&liveDispatch[6],1u);
}

var<workgroup> pageMotion:array<u32,64>;var<workgroup> pageSizing:array<u32,64>;
@compute @workgroup_size(64)
fn advectLosassoFinePhi(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){
 let rank=group.x;let activeCount=atomicLoad(&control[14]);var work=INVALID;
 if(rank<activeCount){work=delta[8u+2u*p.pageCapacity+rank];}
 var page=INVALID;if(work<min(worklist[1],p.pageCapacity)){let candidate=worklist[7u+work];
  if(candidate<p.pageCapacity&&metadata[4u*candidate+2u]==p.generation){page=candidate;}
  else{atomicAdd(&control[7],1u);atomicAdd(&control[11],1u);}}
 var localMotion=0u;var localSizing=0u;
 if(page!=INVALID){let brickKey=metadata[4u*page+1u];let xy=p.brickDimensions.x*p.brickDimensions.y;
 let bz=brickKey/xy;let remainder=brickKey-bz*xy;let by=remainder/p.brickDimensions.x;
 let brick=vec3u(remainder-by*p.brickDimensions.x,by,bz);
 for(var local=lane;local<p.samplesPerBrick;local+=64u){let index=page*p.samplesPerBrick+local;
  if(index>=arrayLength(&samples)||index>=arrayLength(&nextPhi)||(finePackedFlags(index)&VALID)==0u){continue;}
  let old=finePackedPhi(index);nextPhi[index]=old;if(!finite(old)){atomicAdd(&control[1],1u);continue;}
  let q=brick*p.brickResolution+localCoord(local);if(any(q>=p.sampleDimensions)){continue;}
  if(abs(old)>f32(p.bandCells)*p.fineCellWidth){atomicAdd(&control[2],1u);continue;}
  let position=vec3f(q);let low=vec3f(0);var high=vec3f(p.sampleDimensions)-vec3f(1);
  let stages=max(1u,p.substeps);let subDt=p.dt/f32(stages);var departure=position;
  var exitCells=0.;var validTrace=true;
  // One explicit-midpoint trace is sufficient for the host time step. The
  // donor bound governs topology residency and rejects an unexpectedly long
  // characteristic; it must not silently increase the number of time stages.
  for(var stage=0u;stage<stages;stage+=1u){
   let firstGrid=(departure+vec3f(.5))*(p.fineCellWidth/p.velocityCellSize);
   let first=losassoVelocityAtGrid(firstGrid);if(first.valid==0u){atomicAdd(&control[7],1u);atomicAdd(&control[8],1u);atomicMin(&control[12],index);atomicMin(&control[13],brickKey);validTrace=false;break;}
   var midpoint=departure+quantizeTraceOffset(-.5*(subDt/p.fineCellWidth)*first.value);if(p.openTop!=0u){high.y=max(high.y,midpoint.y);}
   if(p.closed!=0u){midpoint=clamp(midpoint,low,high);}else if(any(midpoint<low)||any(midpoint>high)){atomicAdd(&control[0],1u);validTrace=false;break;}
   let middleGrid=(midpoint+vec3f(.5))*(p.fineCellWidth/p.velocityCellSize);
   let middle=losassoVelocityAtGrid(middleGrid);if(middle.valid==0u){atomicAdd(&control[7],1u);atomicAdd(&control[9],1u);atomicMin(&control[12],index);atomicMin(&control[13],brickKey);validTrace=false;break;}
   let traced=departure+quantizeTraceOffset(-(subDt/p.fineCellWidth)*middle.value);if(p.openTop!=0u){high.y=max(high.y,traced.y);}
   if(p.closed!=0u){let interior=clamp(traced,low,high);exitCells+=distance(traced,interior);departure=interior;}
   else if(any(traced<low)||any(traced>high)){atomicAdd(&control[0],1u);validTrace=false;break;}
   else{departure=traced;}
  }
  if(!validTrace){continue;}
  let displacementCells=distance(position,departure)+exitCells;
  localMotion=max(localMotion,bitcast<u32>(displacementCells));let displacement=u32(ceil(displacementCells));
  atomicMax(&control[5],displacement);if(displacement>p.maxBacktrace){atomicAdd(&control[0],1u);continue;}
  let transported=sampleFinePhiGrid(departure);if(!finite(transported)){atomicAdd(&control[7],1u);atomicAdd(&control[10],1u);atomicMin(&control[12],index);continue;}
  let sizing=max(advectedSizingScore(departure),evaluatedSizingScore(departure,transported));
  localSizing=max(localSizing,bitcast<u32>(sizing));
  // A characteristic leaving a closed wall originates in solid/air. Adding
  // its distance outside the sample lattice gives phi the unit outward slope
  // needed for a receding interface; clamping alone re-samples a wall film and
  // pins it indefinitely. This is the reduced LoSasso form of the established
  // Power transport boundary extension.
  let world=p.domainOrigin+(vec3f(q)+vec3f(.5))*p.fineCellWidth;
  nextPhi[index]=applyInflowPhi(transported+exitCells*p.fineCellWidth,world);atomicAdd(&control[2],1u);
 }}
 pageMotion[lane]=localMotion;pageSizing[lane]=localSizing;workgroupBarrier();for(var width=32u;width>0u;width>>=1u){if(lane<width){pageMotion[lane]=max(pageMotion[lane],pageMotion[lane+width]);pageSizing[lane]=max(pageSizing[lane],pageSizing[lane+width]);}workgroupBarrier();}
 if(lane==0u&&page!=INVALID){delta[8u+page]=0x80000000u|pageMotion[0];
  let oldFlags=metadata[4u*page+3u];let oldScore=pageSizingScore(page);
  let retention=pow(.9,p.dt/.01);let score=max(bitcast<f32>(pageSizing[0]),retention*oldScore);
  let quantized=u32(round(255.*clamp(score,0.,1.)));let oldQuantized=(oldFlags&PAGE_SIZING_MASK)>>PAGE_SIZING_SHIFT;
  metadata[4u*page+3u]=(oldFlags&~PAGE_SIZING_MASK)|(quantized<<PAGE_SIZING_SHIFT)
   |select(0u,PAGE_SIZING_CHANGED,quantized!=oldQuantized);}
}

// MacCormack runs over the live-page dispatch rather than the compact active
// list. Rechecking the activity bits keeps these entry points independent of
// the control and topology-delta bindings, so each optional pass stays below
// the portable storage-buffer-per-stage limit.
fn bfeccFinePage(work:u32)->u32{if(work>=min(worklist[1],p.pageCapacity)){return INVALID;}
 let page=worklist[7u+work];if(page>=p.pageCapacity||metadata[4u*page+2u]!=p.generation
  ||!pageAwake(metadata[4u*page+3u])){return INVALID;}return page;}
@compute @workgroup_size(64)
fn reverseLosassoFinePhi(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){
 let page=bfeccFinePage(group.x);if(page==INVALID){return;}let brick=unpackBrick(metadata[4u*page+1u]);
 for(var local=lane;local<p.samplesPerBrick;local+=64u){let index=page*p.samplesPerBrick+local;
  if(index>=arrayLength(&reversePhi)||index>=arrayLength(&nextPhi)){continue;}reversePhi[index]=3.402823e38;
  let q=brick*p.brickResolution+localCoord(local);if(any(q>=p.sampleDimensions)){continue;}
  let world=p.domainOrigin+(vec3f(q)+.5)*p.fineCellWidth;if(inflowSample(world)){continue;}
  let traced=traceFine(vec3f(q),true);if(traced.w==0.){continue;}let value=sampleNextPhiGrid(traced.xyz);
  if(finite(value)){reversePhi[index]=value;}}
}
@compute @workgroup_size(64)
fn correctLosassoFinePhi(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){
 let page=bfeccFinePage(group.x);if(page==INVALID){return;}let brick=unpackBrick(metadata[4u*page+1u]);
 for(var local=lane;local<p.samplesPerBrick;local+=64u){let index=page*p.samplesPerBrick+local;
  if(index>=arrayLength(&reversePhi)||(finePackedFlags(index)&VALID)==0u){continue;}
  let old=finePackedPhi(index);let predicted=nextPhi[index];let reversed=reversePhi[index];
  if(!finite(old)||!finite(predicted)||!finite(reversed)||abs(old)>f32(p.bandCells)*p.fineCellWidth){continue;}
  let q=brick*p.brickResolution+localCoord(local);if(any(q>=p.sampleDimensions)){continue;}
  let world=p.domainOrigin+(vec3f(q)+.5)*p.fineCellWidth;if(inflowSample(world)){continue;}
  let departure=traceFine(vec3f(q),false);if(departure.w==0.){continue;}let bounds=oldDonorBounds(departure.xyz);
  if(bounds.z==0.){continue;}let corrected=predicted+.5*(old-reversed);
  if(finite(corrected)){nextPhi[index]=clamp(corrected,bounds.x,bounds.y);}}
}

var<workgroup> changed:array<u32,64>;
@compute @workgroup_size(64)
fn commitLosassoFinePhi(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){
 let rank=group.x;var work=INVALID;if(acceptedStep()&&rank<atomicLoad(&control[14])){work=delta[8u+2u*p.pageCapacity+rank];}
 var page=INVALID;if(work<min(worklist[1],p.pageCapacity)){page=worklist[7u+work];
  if(page>=p.pageCapacity||metadata[4u*page+2u]!=p.generation){page=INVALID;}}
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
 if(page!=INVALID&&lane==0u){let prior=metadata[4u*page+3u];
  let dirty=(changed[0]&4u)!=0u||(prior&PAGE_SIZING_CHANGED)!=0u;let onInterface=((changed[0]&3u)==3u)||(changed[0]&8u)!=0u;
  let moving=(delta[8u+page]&0x7fffffffu)!=0u;
  metadata[4u*page+3u]=(prior&~(PAGE_INTERFACE|PAGE_DIRTY|PAGE_ACTIVITY_MOVING|PAGE_SIZING_CHANGED))|PAGE_ACTIVITY_VALID
   |select(0u,PAGE_INTERFACE,onInterface)|select(0u,PAGE_DIRTY,dirty)
   |select(0u,PAGE_ACTIVITY_MOVING,moving);}
 workgroupBarrier();if(page!=INVALID){for(var local=lane;local<p.samplesPerBrick;local+=64u){let index=page*p.samplesPerBrick+local;
  if((finePackedFlags(index)&VALID)!=0u){fineWritePackedPhi(index,nextPhi[index]);}}}
}

@compute @workgroup_size(64)
fn publishLosassoFineDelta(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){
 let work=group.x;if(!acceptedStep()||work>=min(worklist[1],p.pageCapacity)){return;}
 let page=worklist[7u+work];if(page>=p.pageCapacity||metadata[4u*page+2u]!=p.generation){if(lane==0u){atomicAdd(&control[7],1u);}return;}
 if(lane==0u){let key=metadata[4u*page+1u];var flags=metadata[4u*page+3u];let awake=(delta[8u+page]&0x80000000u)!=0u;
  if(!awake){flags=(flags&~(PAGE_DIRTY|PAGE_ACTIVITY_MOVING))|PAGE_ACTIVITY_VALID;metadata[4u*page+3u]=flags;}
  delta[8u+page]=select(INVALID,key,(flags&PAGE_INTERFACE)!=0u);
  delta[8u+p.pageCapacity+work]=key;delta[8u+2u*p.pageCapacity+page]=select(INVALID,key,(flags&PAGE_DIRTY)!=0u);}
}

@compute @workgroup_size(64)
fn finalizeLosassoFineTransport(@builtin(local_invocation_index)lane:u32){if(lane!=0u){return;}
 let live=min(worklist[1],p.pageCapacity);let activePages=atomicLoad(&control[14]);
 let accepted=acceptedStep()&&(live==0u||activePages==0u||atomicLoad(&control[2])>0u);let count=select(0u,live,accepted);
 atomicStore(&control[3],select(0u,1u,accepted));delta[0]=count;delta[1]=p.generation;
 delta[2]=select(0u,1u,accepted);delta[3]=min(worklist[1],p.pageCapacity);
 delta[4]=select(0u,(count+63u)/64u,count>0u);delta[5]=1u;delta[6]=1u;delta[7]=atomicLoad(&control[5]);
}
`;
