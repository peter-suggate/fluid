import { fineLevelSetPackedSampleWGSL } from "./fine-levelset-packed-sample";
import { makeFineLevelSetSortedWorklistLookupWGSL } from "./webgpu-octree-fine-levelset-bricks";
import { octreeLosassoVelocitySamplingWGSL } from "./webgpu-octree-losasso-velocity-sampler.wgsl";
import { OCTREE_FINE_SEED_STATE } from "./octree-fine-seed-leaves";

export const OCTREE_LOSASSO_COARSE_PHI_MAGIC = 0x4c50_4849;

/** Factor-one transport over compact Losasso rows. It deliberately has no
 * fine-page bindings: the accepted row directory and axis-face sampler are
 * the complete surface/velocity authorities in this construction. */
export const octreeLosassoCoarseOnlyPhiWGSL = /* wgsl */ `
const INVALID:u32=0xffffffffu;const VALID:u32=1u;const FINITE:u32=2u;const INTERFACE:u32=4u;
const MAGIC:u32=${OCTREE_LOSASSO_COARSE_PHI_MAGIC}u;
const FACE_SOLID_NEUMANN:u32=0x08000000u;const FACE_INTERFACE_NEARBY:u32=0x10000000u;
const FACE_SEPARATED:u32=0x20000000u;const FACE_CLOSED_BOUNDARY:u32=0x40000000u;
const FACE_ROW_ON_POSITIVE_SIDE:u32=0x80000000u;
struct Params{velocityDimensions:vec3u,directoryCapacity:u32,domainOrigin:vec3f,dt:f32,
 velocityCellSize:f32,maximumLeafSize:u32,closed:u32,openTop:u32,rowCapacity:u32,faceCapacity:u32,
 arenaDirectoryCapacity:u32,generation:u32,inflowPositionRadius:vec4f,inflowVelocityStrength:vec4f}
struct LeafHeader{cell:u32,entryStart:u32,entryCount:u32,size:u32,diagonal:f32,rhs:f32,pad0:u32,pad1:u32,gradient:vec4f}
struct FineSeedLeaf{originX:u32,originY:u32,originZ:u32,size:u32,flags:u32,pad0:u32,pad1:u32,pad2:u32,phiGradient:vec4f,motion:vec4f}
struct Face{negativeRow:u32,positiveRow:u32,axis:u32,reserved:u32,area:f32,inverseDistance:f32,openFraction:f32,normalVelocity:f32}
@group(0)@binding(0)var<uniform>p:Params;
@group(0)@binding(1)var<storage,read>coarse:array<u32>;
@group(0)@binding(2)var<storage,read>headers:array<LeafHeader>;
@group(0)@binding(3)var<storage,read>seedLeaves:array<FineSeedLeaf>;
@group(0)@binding(4)var<storage,read>previousArena:array<u32>;
@group(0)@binding(5)var<storage,read_write>rowPhi:array<vec4u>;
@group(0)@binding(6)var<storage,read>previousGradient:array<vec4f>;
@group(0)@binding(7)var<storage,read_write>nextGradient:array<vec4f>;
@group(0)@binding(8)var<storage,read>faceGeometry:array<vec4u>;
@group(0)@binding(9)var<storage,read>extendedVelocity:array<f32>;
@group(0)@binding(10)var<storage,read>faceDirectory:array<vec2u>;
@group(0)@binding(11)var<storage,read_write>nextArena:array<atomic<u32>>;
@group(0)@binding(12)var<storage,read_write>faces:array<Face>;
@group(0)@binding(13)var<storage,read_write>ghosts:array<vec4u>;
@group(0)@binding(14)var<storage,read>solidCells:array<u32>;
@group(0)@binding(15)var<storage,read>authority:array<u32>;
@group(0)@binding(16)var<storage,read>coarseTracker:array<u32>;
fn finite(v:f32)->bool{return v==v&&abs(v)<3.402823e38;}
fn finite3(v:vec3f)->bool{return all(v==v)&&all(abs(v)<vec3f(3.402823e38));}
${octreeLosassoVelocitySamplingWGSL}
fn rows()->u32{return min(min(authority[1],p.rowCapacity),arrayLength(&headers));}
fn faceCount()->u32{return min(min(authority[2],p.faceCapacity),arrayLength(&faces));}
fn cellOf(index:u32)->vec3u{return vec3u(index%p.velocityDimensions.x,
 (index/p.velocityDimensions.x)%p.velocityDimensions.y,index/(p.velocityDimensions.x*p.velocityDimensions.y));}
fn hash(cell:u32,size:u32)->u32{return ((cell*0x9e3779b1u)^size)*0x85ebca6bu;}
fn priorRow(cell:u32,size:u32)->u32{if(arrayLength(&previousArena)<20u||previousArena[0]!=MAGIC){return INVALID;}
 let capacity=previousArena[11];if(capacity==0u||(capacity&(capacity-1u))!=0u){return INVALID;}let wanted=hash(cell,size);let mask=capacity-1u;
 for(var probe=0u;probe<32u;probe+=1u){let at=previousArena[10]+4u*((wanted+probe)&mask);if(at+3u>=arrayLength(&previousArena)){return INVALID;}
  let key=previousArena[at];if(key==0u){return INVALID;}if(key==cell+1u&&previousArena[at+1u]==size&&previousArena[at+3u]==wanted){return previousArena[at+2u]-1u;}}
 return INVALID;}
struct PhiSample{value:f32,gradient:vec3f,valid:u32}
fn priorPhi(position:vec3f)->PhiSample{let q=vec3u(clamp(vec3i(floor(position)),vec3i(0),vec3i(p.velocityDimensions)-vec3i(1)));var size=1u;
 loop{let origin=(q/vec3u(size))*vec3u(size);let cell=origin.x+p.velocityDimensions.x*(origin.y+p.velocityDimensions.y*origin.z);let row=priorRow(cell,size);
  if(row!=INVALID&&row<previousArena[2]&&row<arrayLength(&previousGradient)){let entry=previousArena[9]+8u*row;if(entry+5u<arrayLength(&previousArena)&&(previousArena[entry+5u]&(VALID|FINITE))==(VALID|FINITE)&&previousGradient[row].w==1.){
    let centre=vec3f(origin)+vec3f(.5*f32(size));let gradient=previousGradient[row].xyz;let value=bitcast<f32>(previousArena[entry+2u])+dot(gradient,(position-centre)*p.velocityCellSize);
    return PhiSample(value,gradient,select(0u,1u,finite(value)&&finite3(gradient)));}}
  if(size>=p.maximumLeafSize){break;}size*=2u;}return PhiSample(0.,vec3f(0),0u);}
fn seedPhi(row:u32,position:vec3f)->PhiSample{if(row>=arrayLength(&seedLeaves)){return PhiSample(0.,vec3f(0),0u);}let leaf=seedLeaves[row];let gradient=leaf.phiGradient.yzw;
 let centre=vec3f(vec3u(leaf.originX,leaf.originY,leaf.originZ))+vec3f(.5*f32(leaf.size));let value=leaf.phiGradient.x+dot(gradient,(position-centre)*p.velocityCellSize);
 return PhiSample(value,gradient,select(0u,1u,(leaf.flags&${OCTREE_FINE_SEED_STATE.live}u)!=0u&&leaf.size>0u&&finite(value)&&finite3(gradient)));}
fn trackerValue(position:vec3f)->PhiSample{let volume=p.velocityDimensions.x*p.velocityDimensions.y*p.velocityDimensions.z;
 let capacity=(max(arrayLength(&coarseTracker),8u)-8u)/8u;
 if(arrayLength(&coarseTracker)<8u||coarseTracker[0]!=0x80000000u||(coarseTracker[1]&0x40000000u)==0u
  ||capacity<p.rowCapacity+volume||any(vec3u(coarseTracker[4],coarseTracker[5],coarseTracker[6])!=p.velocityDimensions)){
  return PhiSample(0.,vec3f(0),0u);}
 let bounded=clamp(position,vec3f(.5),vec3f(p.velocityDimensions)-vec3f(.5));let low=vec3i(floor(bounded-vec3f(.5)));
 let fraction=bounded-(vec3f(low)+vec3f(.5));var value=0.;var total=0.;
 for(var corner=0u;corner<8u;corner+=1u){let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
  let q=vec3u(clamp(low+offset,vec3i(0),vec3i(p.velocityDimensions)-vec3i(1)));let cell=q.x+p.velocityDimensions.x*(q.y+p.velocityDimensions.y*q.z);
  let base=8u+8u*(p.rowCapacity+cell);let sample=bitcast<f32>(coarseTracker[base+2u]);
  if((coarseTracker[base+5u]&9u)!=9u||!finite(sample)){return PhiSample(0.,vec3f(0),0u);}
  let weight=select(1.-fraction.x,fraction.x,(corner&1u)!=0u)*select(1.-fraction.y,fraction.y,(corner&2u)!=0u)
   *select(1.-fraction.z,fraction.z,(corner&4u)!=0u);value+=weight*sample;total+=weight;}
 return PhiSample(value/max(total,1e-12),vec3f(0),select(0u,1u,total>.999&&finite(value)));}
fn trackerPhi(position:vec3f)->PhiSample{let centre=trackerValue(position);if(centre.valid==0u){return centre;}var gradient=vec3f(0);
 for(var axis=0u;axis<3u;axis+=1u){var low=position;var high=position;low[axis]-=1.;high[axis]+=1.;let a=trackerValue(low);let b=trackerValue(high);
  if(a.valid==0u||b.valid==0u){return PhiSample(0.,vec3f(0),0u);}gradient[axis]=.5*(b.value-a.value)/p.velocityCellSize;}
 return PhiSample(centre.value,gradient,select(0u,1u,finite3(gradient)));}
fn quantize(v:vec3f)->vec3f{return round(v*65536.)/65536.;}
fn inflowPhi(world:vec3f)->f32{let velocity=p.inflowVelocityStrength.xyz;let speed=length(velocity);if(speed<=1e-6||p.inflowPositionRadius.w<=0.){return 3.402823e38;}
 let direction=velocity/speed;let extent=vec3f(p.velocityDimensions)*p.velocityCellSize;let centre=p.inflowPositionRadius.xyz+vec3f(.5*extent.x,0.,.5*extent.z);
 let relative=world-centre;let axial=dot(relative,direction);let radial=length(relative-axial*direction);return max(radial-p.inflowPositionRadius.w,max(-axial,axial-2.*p.velocityCellSize));}
@compute @workgroup_size(64)
fn advectLosassoCoarsePhi(@builtin(global_invocation_id)gid:vec3u){let row=gid.x;if(row>=p.rowCapacity||row>=arrayLength(&headers)){return;}let header=headers[row];if(header.size==0u||header.size>p.maximumLeafSize){rowPhi[row]=vec4u(0);nextGradient[row]=vec4f(0);return;}
 let origin=cellOf(header.cell);let centre=vec3f(origin)+vec3f(.5*f32(header.size));let transported=trackerPhi(centre);
 if(transported.valid==0u){rowPhi[row]=vec4u(0);nextGradient[row]=vec4f(0);return;}
 var value=transported.value;let world=p.domainOrigin+centre*p.velocityCellSize;let sourceValue=inflowPhi(world);if(p.inflowVelocityStrength.w>0.&&finite(sourceValue)){value=min(value,max(sourceValue,-.5*p.velocityCellSize*clamp(p.inflowVelocityStrength.w,0.,1.)));}
 // Compact pressure rows are a restriction cache, not a second transported
 // affine surface.  Keep their centre value exact and derive the interface
 // bit from the dense coarse lattice's six neighbouring row-width samples;
 // renderer/topology consumers read the dense complement directly.
 let gradient=vec3f(0);let minimum=value;let maximum=value;var flags=VALID|FINITE;
 for(var axis=0u;axis<3u;axis+=1u){for(var side=0u;side<2u;side+=1u){var neighbor=centre;
  neighbor[axis]+=select(-f32(header.size),f32(header.size),side!=0u);let sample=trackerValue(neighbor);
  if(sample.valid!=0u&&((value<0.)!=(sample.value<0.))){flags|=INTERFACE;}}}
 rowPhi[row]=vec4u(bitcast<u32>(value),bitcast<u32>(minimum),bitcast<u32>(maximum),flags);nextGradient[row]=vec4f(gradient,1);}
@compute @workgroup_size(64)
fn publishLosassoCoarseOnlyRows(@builtin(global_invocation_id)gid:vec3u){let row=gid.x;if(row>=rows()){return;}let header=headers[row];let value=rowPhi[row];let entry=20u+8u*row;
 atomicStore(&nextArena[entry],header.cell+1u);atomicStore(&nextArena[entry+1u],header.size);atomicStore(&nextArena[entry+2u],value.x);atomicStore(&nextArena[entry+3u],value.y);
 atomicStore(&nextArena[entry+4u],value.z);atomicStore(&nextArena[entry+5u],value.w);atomicStore(&nextArena[entry+6u],row);atomicStore(&nextArena[entry+7u],bitcast<u32>(f32(header.size*header.size*header.size)*p.velocityCellSize*p.velocityCellSize*p.velocityCellSize));
 let wanted=hash(header.cell,header.size);let mask=p.arenaDirectoryCapacity-1u;var installed=false;for(var probe=0u;probe<32u;probe+=1u){let at=20u+8u*p.rowCapacity+4u*((wanted+probe)&mask);var claimed=atomicCompareExchangeWeak(&nextArena[at],0u,header.cell+1u);
  loop{if(claimed.exchanged||claimed.old_value!=0u){break;}claimed=atomicCompareExchangeWeak(&nextArena[at],0u,header.cell+1u);}if(claimed.exchanged){atomicStore(&nextArena[at+1u],header.size);atomicStore(&nextArena[at+2u],row+1u);atomicStore(&nextArena[at+3u],wanted);installed=true;break;}}
 if(!installed){atomicOr(&nextArena[13],4u);}}
fn currentPhi(position:vec3f)->PhiSample{let q=vec3u(clamp(vec3i(floor(position)),vec3i(0),vec3i(p.velocityDimensions)-vec3i(1)));var size=1u;loop{let origin=(q/vec3u(size))*vec3u(size);let cell=origin.x+p.velocityDimensions.x*(origin.y+p.velocityDimensions.y*origin.z);let wanted=hash(cell,size);let mask=p.arenaDirectoryCapacity-1u;
  for(var probe=0u;probe<32u;probe+=1u){let at=20u+8u*p.rowCapacity+4u*((wanted+probe)&mask);let key=atomicLoad(&nextArena[at]);if(key==0u){break;}if(key==cell+1u&&atomicLoad(&nextArena[at+1u])==size&&atomicLoad(&nextArena[at+3u])==wanted){let row=atomicLoad(&nextArena[at+2u])-1u;let entry=20u+8u*row;if(row>=p.rowCapacity||row>=arrayLength(&nextGradient)||entry+5u>=arrayLength(&nextArena)||nextGradient[row].w!=1.){return PhiSample(0.,vec3f(0),0u);}let centre=vec3f(origin)+vec3f(.5*f32(size));let gradient=nextGradient[row].xyz;let value=bitcast<f32>(atomicLoad(&nextArena[entry+2u]))+dot(gradient,(position-centre)*p.velocityCellSize);return PhiSample(value,gradient,select(0u,1u,finite(value)&&finite3(gradient)));}}
  if(size>=p.maximumLeafSize){break;}size*=2u;}return PhiSample(0.,vec3f(0),0u);}
fn currentRowPhi(row:u32,position:vec3f)->PhiSample{
 if(row>=rows()||row>=arrayLength(&nextGradient)||nextGradient[row].w!=1.){return PhiSample(0.,vec3f(0),0u);}
 let header=headers[row];let origin=cellOf(header.cell);let centre=vec3f(origin)+vec3f(.5*f32(header.size));
 let gradient=nextGradient[row].xyz;let value=bitcast<f32>(rowPhi[row].x)+dot(gradient,(position-centre)*p.velocityCellSize);
 return PhiSample(value,gradient,select(0u,1u,(rowPhi[row].w&(VALID|FINITE))==(VALID|FINITE)&&finite(value)&&finite3(gradient)));}
fn solidFraction(cell:vec3i)->f32{if(any(cell<vec3i(0))||any(cell>=vec3i(p.velocityDimensions))){return 0.;}let q=vec3u(cell);let word=2u*(q.x+p.velocityDimensions.x*(q.y+p.velocityDimensions.y*q.z));if(word>=arrayLength(&solidCells)){return 0.;}let value=bitcast<f32>(solidCells[word]);return clamp(select(0.,value,value==value),0.,1.);}
@compute @workgroup_size(64)
fn publishLosassoCoarseOnlyGhosts(@builtin(global_invocation_id)gid:vec3u){let faceId=gid.x;if(faceId>=faceCount()){return;}var face=faces[faceId];face.reserved&=~FACE_SOLID_NEUMANN;face.reserved&=~FACE_INTERFACE_NEARBY;
 let negativeNearby=face.negativeRow<rows()&&(rowPhi[face.negativeRow].w&INTERFACE)!=0u;
 let positiveNearby=face.positiveRow<rows()&&(rowPhi[face.positiveRow].w&INTERFACE)!=0u;
 if(negativeNearby||positiveNearby){face.reserved|=FACE_INTERFACE_NEARBY;}
 var distance=0.;var theta=1.;var airPhi=0.;var flags=0u;if(face.positiveRow==INVALID){let geometry=faceGeometry[faceId];let axis=geometry.x&3u;let span=1u<<(geometry.x>>2u);let origin=geometry.yzw;let boundary=origin[axis]==0u||origin[axis]==p.velocityDimensions[axis];
     if(boundary&&(face.reserved&FACE_CLOSED_BOUNDARY)!=0u&&face.negativeRow<rows()){let liquid=bitcast<f32>(rowPhi[face.negativeRow].x);let towardAir=select(1.,-1.,(face.reserved&FACE_ROW_ON_POSITIVE_SIDE)!=0u);var point=vec3f(origin);for(var tangent=0u;tangent<3u;tangent+=1u){if(tangent!=axis){point[tangent]+=.5*f32(span);}}point[axis]-=towardAir*.5;var sampled=currentPhi(point);if(sampled.valid==0u){sampled=currentRowPhi(face.negativeRow,point);}face.openFraction=0.;flags=2u;
   let contactEnabled=(p.closed&2u)==0u;let separated=contactEnabled&&(face.reserved&FACE_SEPARATED)!=0u;let sampledAir=liquid<0.&&sampled.valid!=0u&&sampled.value>0.;
   if(contactEnabled&&(separated||sampledAir)){airPhi=sampled.value;let dual=max(.5,.5*f32(span)-.5)*p.velocityCellSize;theta=select(1.,clamp(-liquid/(airPhi-liquid),1e-2,1.),sampled.valid!=0u&&airPhi>0.&&liquid<0.);distance=theta*dual;face.inverseDistance=1./max(distance,1e-9);face.openFraction=1.;flags=3u;}}
  else if(boundary){distance=.5*p.velocityCellSize;face.inverseDistance=1./distance;face.openFraction=1.;flags=1u;}
     else if(face.negativeRow<rows()){let liquid=bitcast<f32>(rowPhi[face.negativeRow].x);
      let sign=select(1.,-1.,(face.reserved&FACE_ROW_ON_POSITIVE_SIDE)!=0u);var point=vec3f(origin);
      for(var tangent=0u;tangent<3u;tangent+=1u){if(tangent!=axis){point[tangent]+=.5*f32(span);}}point[axis]+=sign*.5*f32(span);
      var sampled=trackerValue(point);if(sampled.valid==0u){sampled=currentPhi(point);}if(sampled.valid==0u){sampled=currentRowPhi(face.negativeRow,point);}
      // Losasso's p_air value is at the centre of a same-sized bordering air
      // cell.  A coarse row may emit several smaller T-junction subfaces, but
      // their areas do not move that pressure sample closer to the wet-row
      // centre.  Match the factor-four ghost kernel and the staged topology
      // coefficient by using the accepted row width, not the subface span.
      let rowSize=headers[face.negativeRow].size;let dual=f32(rowSize)*p.velocityCellSize;distance=dual;
   if(solidFraction(vec3i(floor(point)))>=.999999){face.reserved|=FACE_SOLID_NEUMANN;flags=4u;}
     else if(liquid<0.&&(p.closed&2u)!=0u){theta=1.;distance=dual;face.inverseDistance=1./dual;flags=1u;}
   else if(liquid<0.&&sampled.valid!=0u&&sampled.value>0.){airPhi=sampled.value;theta=clamp(-liquid/(airPhi-liquid),1e-4,1.);distance=theta*dual;face.inverseDistance=1./distance;face.openFraction=1.;flags=1u;}else{flags=4u;}}}
 faces[faceId]=face;ghosts[faceId]=vec4u(bitcast<u32>(distance),bitcast<u32>(theta),bitcast<u32>(airPhi),flags);}
`;

export const octreeLosassoCoarsePhiWGSL = /* wgsl */ `
const INVALID:u32=0xffffffffu;const VALID:u32=1u;const FINITE:u32=2u;
const INTERFACE:u32=4u;const FINE_RESTRICTED:u32=8u;const MAGIC:u32=${OCTREE_LOSASSO_COARSE_PHI_MAGIC}u;
const FACE_SOLID_NEUMANN:u32=0x08000000u;const FACE_INTERFACE_NEARBY:u32=0x10000000u;const FACE_SEPARATED:u32=0x20000000u;const FACE_CLOSED_BOUNDARY:u32=0x40000000u;const FACE_ROW_ON_POSITIVE_SIDE:u32=0x80000000u;
struct Params{brickDimensions:vec3u,brickResolution:u32,sampleDimensions:vec3u,samplesPerBrick:u32,
 domainOrigin:vec3f,fineCellWidth:f32,pageCapacity:u32,generation:u32,worklistCapacity:u32,worklistHeaderWords:u32,
 dimensions:vec3u,maximumLeafSize:u32,cellSize:f32,directoryCapacity:u32,rowCapacity:u32,faceCapacity:u32,
 schedule:vec4u}
struct LeafHeader{cell:u32,entryStart:u32,entryCount:u32,size:u32,diagonal:f32,rhs:f32,pad0:u32,pad1:u32,gradient:vec4f}
struct Face{negativeRow:u32,positiveRow:u32,axis:u32,reserved:u32,area:f32,inverseDistance:f32,openFraction:f32,normalVelocity:f32}
@group(0)@binding(0)var<uniform>p:Params;
@group(0)@binding(1)var<storage,read>metadata:array<u32>;
@group(0)@binding(2)var<storage,read>worklist:array<u32>;
@group(0)@binding(3)var<storage,read>samples:array<u32>;
@group(0)@binding(4)var<storage,read>headers:array<LeafHeader>;
@group(0)@binding(5)var<storage,read>coarseControl:array<u32>;
@group(0)@binding(6)var<storage,read_write>faces:array<Face>;
@group(0)@binding(7)var<storage,read>faceGeometry:array<vec4u>;
@group(0)@binding(8)var<storage,read_write>arena:array<atomic<u32>>;
@group(0)@binding(9)var<storage,read_write>rowPhi:array<vec4u>;
@group(0)@binding(10)var<storage,read_write>ghosts:array<vec4u>;
@group(0)@binding(14)var<storage,read_write>publishedArena:array<atomic<u32>>;
@group(0)@binding(16)var<storage,read_write>readyDispatch:array<u32>;
@group(0)@binding(17)var<storage,read>solidCells:array<u32>;
fn finite(v:f32)->bool{return v==v&&abs(v)<3.402823e38;}
${fineLevelSetPackedSampleWGSL("samples")}
${makeFineLevelSetSortedWorklistLookupWGSL("p", "metadata", "worklist", "losassoCoarseFineBrick")}
fn rows()->u32{return min(min(coarseControl[1],p.rowCapacity),arrayLength(&headers));}
// The ghost entry point deliberately avoids the leaf-header binding so its
// solid-aware fine-phi classification stays within ten storage buffers. The
// accepted directory has already validated and published this same row count.
fn ghostRows()->u32{return min(coarseControl[1],p.rowCapacity);}
fn faceCount()->u32{return min(min(coarseControl[2],p.faceCapacity),arrayLength(&faces));}
fn originOf(header:LeafHeader)->vec3u{return vec3u(header.cell%p.dimensions.x,
 (header.cell/p.dimensions.x)%p.dimensions.y,header.cell/(p.dimensions.x*p.dimensions.y));}
fn packBrick(q:vec3u)->u32{return q.x+p.brickDimensions.x*(q.y+p.brickDimensions.y*q.z);}
struct FinePhi{value:f32,valid:u32}
fn addExact(limbs:ptr<function,array<i32,36>>,value:f32){let bits=bitcast<u32>(value);let magnitude=bits&0x7fffffffu;if(magnitude==0u){return;}
 let rawExponent=(magnitude>>23u)&0xffu;let fraction=magnitude&0x7fffffu;let significand=select(fraction,0x800000u|fraction,rawExponent!=0u);
 let shift=select(3u,rawExponent+2u,rawExponent!=0u);let firstLimb=shift>>3u;let shifted=significand<<(shift&7u);
 let sign=select(1,-1,(bits&0x80000000u)!=0u);for(var digit=0u;digit<4u;digit+=1u){let limb=firstLimb+digit;
  let byte=i32((shifted>>(digit*8u))&0xffu);if(byte!=0&&limb<36u){(*limbs)[limb]+=sign*byte;}}}
fn floorDiv256(value:i32)->vec2i{let carry=value>>8;return vec2i(carry,value-carry*256);}
fn exactValue(source:ptr<function,array<i32,36>>)->f32{for(var limb=0u;limb+1u<36u;limb+=1u){let normalized=floorDiv256((*source)[limb]);(*source)[limb]=normalized.y;(*source)[limb+1u]+=normalized.x;}
 let negative=(*source)[35]<0;if(negative){for(var limb=0u;limb<36u;limb+=1u){(*source)[limb]=-(*source)[limb];}
  for(var limb=0u;limb+1u<36u;limb+=1u){let normalized=floorDiv256((*source)[limb]);(*source)[limb]=normalized.y;(*source)[limb+1u]+=normalized.x;}}
 var magnitude=0.;for(var limb=0u;limb<36u;limb+=1u){magnitude+=ldexp(f32((*source)[limb]),-152+i32(limb*8u));}return select(magnitude,-magnitude,negative);}
// Sample in coarse-cell lattice coordinates. Reconstructing world positions as
// domainOrigin + q*dx and subtracting domainOrigin here gives reflected cells
// different f32 interpolation weights through cancellation, which perturbs the
// ghost distance and therefore the assembled diagonal. The cell-space form is
// the same physical point without that asymmetric round trip.
fn finePhi(positionCells:vec3f)->FinePhi{let grid=positionCells*(p.cellSize/p.fineCellWidth)-vec3f(.5);
 let base=vec3i(floor(grid));let fraction=fract(grid);var exact:array<i32,36>;
 for(var corner=0u;corner<8u;corner+=1u){let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
  let coordinate=base+offset;if(any(coordinate<vec3i(0))||any(coordinate>=vec3i(p.sampleDimensions))){return FinePhi(0.,0u);}
  let q=vec3u(coordinate);let brick=q/p.brickResolution;let page=losassoCoarseFineBrick(packBrick(brick));
  if(page==INVALID){return FinePhi(0.,0u);}let local=q-brick*p.brickResolution;
  let index=page*p.samplesPerBrick+local.x+p.brickResolution*(local.y+p.brickResolution*local.z);
  if(index>=arrayLength(&samples)||(finePackedFlags(index)&VALID)==0u){return FinePhi(0.,0u);}
  let sample=finePackedPhi(index);if(!finite(sample)){return FinePhi(0.,0u);}
  let weight=select(1.-fraction.x,fraction.x,(corner&1u)!=0u)
   *select(1.-fraction.y,fraction.y,(corner&2u)!=0u)
   *select(1.-fraction.z,fraction.z,(corner&4u)!=0u);addExact(&exact,weight*sample);}
 return FinePhi(exactValue(&exact),1u);}
fn denseSolidFraction(cell:vec3i)->f32{if(any(cell<vec3i(0))||any(vec3u(cell)>=p.dimensions)){return 0.;}
 let q=vec3u(cell);let index=q.x+p.dimensions.x*(q.y+p.dimensions.y*q.z);let word=2u*index;
 if(word>=arrayLength(&solidCells)){return 0.;}let value=bitcast<f32>(solidCells[word]);
 return clamp(select(0.,value,value==value),0.,1.);}
fn denseRigidOwner(cell:vec3i)->i32{if(any(cell<vec3i(0))||any(vec3u(cell)>=p.dimensions)){return -1;}
 let q=vec3u(cell);let index=q.x+p.dimensions.x*(q.y+p.dimensions.y*q.z);let word=2u*index+1u;
 if(word>=arrayLength(&solidCells)){return -1;}return bitcast<i32>(solidCells[word]);}
fn directoryHash(cell:u32,size:u32)->u32{return ((cell*0x9e3779b1u)^size)*0x85ebca6bu;}
fn entryBase(row:u32)->u32{return atomicLoad(&arena[9])+8u*row;}
fn directoryBase(slot:u32)->u32{return atomicLoad(&arena[10])+4u*slot;}
fn exactTopologyReuse()->bool{return p.schedule.x==0u&&arrayLength(&coarseControl)>5u&&coarseControl[5]==1u;}
fn targetLoad(index:u32,usePublished:bool)->u32{
 if(usePublished){return atomicLoad(&publishedArena[index]);}return atomicLoad(&arena[index]);}
fn targetStore(index:u32,value:u32,usePublished:bool){
 if(usePublished){atomicStore(&publishedArena[index],value);}else{atomicStore(&arena[index],value);}}
fn targetOr(index:u32,value:u32,usePublished:bool){
 if(usePublished){atomicOr(&publishedArena[index],value);}else{atomicOr(&arena[index],value);}}

@compute @workgroup_size(256)
fn prepareLosassoCoarsePhi(@builtin(local_invocation_index)lane:u32){
 let reuse=exactTopologyReuse();
 if(lane==0u){readyDispatch[0]=select(p.schedule.y,0u,reuse);readyDispatch[1]=1u;readyDispatch[2]=1u;
  readyDispatch[3]=select(p.schedule.z,0u,reuse);readyDispatch[4]=1u;readyDispatch[5]=1u;}
 if(reuse){if(lane==0u){let valid=atomicLoad(&publishedArena[0])==MAGIC
   &&atomicLoad(&publishedArena[2])==rows()&&atomicLoad(&publishedArena[3])==faceCount()
   &&atomicLoad(&publishedArena[4])==p.maximumLeafSize
   &&atomicLoad(&publishedArena[5])==p.dimensions.x&&atomicLoad(&publishedArena[6])==p.dimensions.y
   &&atomicLoad(&publishedArena[7])==p.dimensions.z
   &&atomicLoad(&publishedArena[8])==bitcast<u32>(p.cellSize)
   &&atomicLoad(&publishedArena[11])==p.directoryCapacity;
   atomicStore(&publishedArena[0],0u);atomicStore(&publishedArena[1],coarseControl[0]);
   atomicStore(&publishedArena[13],select(16u,0u,valid));
   atomicStore(&publishedArena[12],p.generation);atomicStore(&publishedArena[14],p.generation);}
  return;}
 if(lane<20u){atomicStore(&arena[lane],0u);}workgroupBarrier();
 if(lane==0u){atomicStore(&arena[1],coarseControl[0]);atomicStore(&arena[2],rows());atomicStore(&arena[3],faceCount());
  atomicStore(&arena[4],p.maximumLeafSize);atomicStore(&arena[5],p.dimensions.x);atomicStore(&arena[6],p.dimensions.y);
  atomicStore(&arena[7],p.dimensions.z);atomicStore(&arena[8],bitcast<u32>(p.cellSize));
  atomicStore(&arena[10],20u+8u*p.rowCapacity);atomicStore(&arena[11],p.directoryCapacity);
  atomicStore(&arena[12],p.generation);atomicStore(&arena[14],p.generation);
  atomicStore(&arena[15],bitcast<u32>(p.domainOrigin.x));atomicStore(&arena[16],bitcast<u32>(p.domainOrigin.y));
  atomicStore(&arena[17],bitcast<u32>(p.domainOrigin.z));atomicStore(&arena[9],20u);
  if(arrayLength(&coarseControl)<4u||coarseControl[3]!=1u||rows()==0u){atomicOr(&arena[13],1u);}}
 let base=20u+8u*p.rowCapacity;for(var slot=lane;slot<p.directoryCapacity;slot+=256u){
  for(var word=0u;word<4u;word+=1u){atomicStore(&arena[base+4u*slot+word],0u);}}
}

@compute @workgroup_size(1)
fn prepareLosassoCoarsePhiRefresh(){let valid=atomicLoad(&arena[0])==MAGIC
 &&atomicLoad(&arena[1])==coarseControl[0]&&atomicLoad(&arena[2])==rows()
 &&atomicLoad(&arena[3])==faceCount()&&atomicLoad(&arena[4])==p.maximumLeafSize
 &&atomicLoad(&arena[5])==p.dimensions.x&&atomicLoad(&arena[6])==p.dimensions.y
 &&atomicLoad(&arena[7])==p.dimensions.z&&atomicLoad(&arena[8])==bitcast<u32>(p.cellSize)
 &&atomicLoad(&arena[11])==p.directoryCapacity;
 atomicStore(&arena[0],0u);atomicStore(&arena[13],select(16u,0u,valid));
 atomicStore(&arena[12],p.generation);atomicStore(&arena[14],p.generation);}

fn restrictLosassoCoarsePhiRow(row:u32,rebuildDirectory:bool,usePublished:bool){if(row>=rows()){return;}
 let header=headers[row];let origin=originOf(header);let centre=vec3f(origin)+vec3f(.5*f32(header.size));
 let centreFine=finePhi(centre);let seed=rowPhi[row];var centrePhi=bitcast<f32>(seed.x);
 var minimum=bitcast<f32>(seed.y);var maximum=bitcast<f32>(seed.z);var flags=seed.w&(VALID|FINITE|FINE_RESTRICTED);
 if(header.size==0u||header.size>p.maximumLeafSize){rowPhi[row]=vec4u(0u);targetOr(13u,2u,usePublished);return;}
 let entry=20u+8u*row;
 if(!rebuildDirectory&&(targetLoad(entry,usePublished)!=header.cell+1u
   ||targetLoad(entry+1u,usePublished)!=header.size
   ||targetLoad(entry+6u,usePublished)!=row)){rowPhi[row]=vec4u(0u);targetOr(13u,16u,usePublished);return;}
 if(centreFine.valid!=0u){flags=VALID|FINITE;centrePhi=centreFine.value;minimum=centrePhi;maximum=centrePhi;var complete=1u;
  for(var corner=0u;corner<8u;corner+=1u){let offset=vec3u(corner&1u,(corner>>1u)&1u,(corner>>2u)&1u);
   let point=vec3f(origin+offset*header.size);let sample=finePhi(point);
   if(sample.valid==0u){complete=0u;}else{minimum=min(minimum,sample.value);maximum=max(maximum,sample.value);}}
  if(complete!=0u){flags|=FINE_RESTRICTED;}else{let extent=.5*f32(header.size)*p.cellSize;minimum=centrePhi-extent;maximum=centrePhi+extent;}
 }
 if((flags&(VALID|FINITE))==(VALID|FINITE)&&minimum<=0.&&maximum>=0.){flags|=INTERFACE;}rowPhi[row]=vec4u(bitcast<u32>(centrePhi),bitcast<u32>(minimum),bitcast<u32>(maximum),flags);
 targetStore(entry,header.cell+1u,usePublished);targetStore(entry+1u,header.size,usePublished);
 targetStore(entry+2u,bitcast<u32>(centrePhi),usePublished);targetStore(entry+3u,bitcast<u32>(minimum),usePublished);
 targetStore(entry+4u,bitcast<u32>(maximum),usePublished);targetStore(entry+5u,flags,usePublished);
 targetStore(entry+6u,row,usePublished);targetStore(entry+7u,bitcast<u32>(f32(header.size*header.size*header.size)*p.cellSize*p.cellSize*p.cellSize),usePublished);
 if(!rebuildDirectory){return;}let hash=directoryHash(header.cell,header.size);let mask=p.directoryCapacity-1u;var installed=false;
 for(var probe=0u;probe<32u;probe+=1u){let slot=(hash+probe)&mask;let at=20u+8u*p.rowCapacity+4u*slot;
  var claimed=atomicCompareExchangeWeak(&arena[at],0u,header.cell+1u);
  loop{if(claimed.exchanged||claimed.old_value!=0u){break;}claimed=atomicCompareExchangeWeak(&arena[at],0u,header.cell+1u);}
  if(claimed.exchanged){atomicStore(&arena[at+1u],header.size);
   atomicStore(&arena[at+2u],row+1u);atomicStore(&arena[at+3u],hash);installed=true;break;}}
 if(!installed){targetOr(13u,4u,usePublished);}}

@compute @workgroup_size(64)
fn restrictLosassoCoarsePhi(@builtin(global_invocation_id)invocation:vec3u){
 let reuse=exactTopologyReuse();restrictLosassoCoarsePhiRow(invocation.x,!reuse,reuse);}

@compute @workgroup_size(64)
fn refreshLosassoCoarsePhi(@builtin(global_invocation_id)invocation:vec3u){
 restrictLosassoCoarsePhiRow(invocation.x,false,false);}

@compute @workgroup_size(1)
fn finalizeLosassoCoarsePhi(){let reuse=exactTopologyReuse();if(targetLoad(13u,reuse)==0u){targetStore(0u,MAGIC,reuse);}}

@compute @workgroup_size(64)
fn publishLosassoCoarsePhiArena(@builtin(global_invocation_id)invocation:vec3u){let word=invocation.x;
 if(exactTopologyReuse()){return;}
 if(word<min(arrayLength(&arena),arrayLength(&publishedArena))){atomicStore(&publishedArena[word],atomicLoad(&arena[word]));}}

@compute @workgroup_size(64)
fn publishLosassoGhostDistances(@builtin(global_invocation_id)invocation:vec3u){let faceId=invocation.x;if(faceId>=faceCount()){return;}
 let reuse=exactTopologyReuse();
 var face=faces[faceId];face.reserved&=~FACE_SOLID_NEUMANN;face.reserved&=~FACE_INTERFACE_NEARBY;
 let rowBase=targetLoad(9u,reuse);
 if((face.negativeRow<ghostRows()&&(targetLoad(rowBase+8u*face.negativeRow+5u,reuse)&INTERFACE)!=0u)
   ||(face.positiveRow<ghostRows()&&(targetLoad(rowBase+8u*face.positiveRow+5u,reuse)&INTERFACE)!=0u)){face.reserved|=FACE_INTERFACE_NEARBY;}
 var flags=0u;var distance=0.;var theta=1.;var airPhi=0.;
 if(targetLoad(0u,reuse)!=MAGIC){ghosts[faceId]=vec4u(0u,bitcast<u32>(1.),0u,8u);return;}
 if(face.positiveRow==INVALID){let geometry=faceGeometry[faceId];let axis=geometry.x&3u;let span=1u<<(geometry.x>>2u);
  let origin=geometry.yzw;let boundary=origin[axis]==0u||origin[axis]==p.dimensions[axis];
  if(boundary&&(face.reserved&FACE_CLOSED_BOUNDARY)!=0u&&face.negativeRow<targetLoad(2u,reuse)){
   let entry=targetLoad(9u,reuse)+8u*face.negativeRow;let liquid=bitcast<f32>(targetLoad(entry+2u,reuse));
   let rowSize=targetLoad(entry+1u,reuse);let rowFlags=targetLoad(entry+5u,reuse);
   let towardAir=select(1.,-1.,(face.reserved&FACE_ROW_ON_POSITIVE_SIDE)!=0u);var centroid=vec3f(origin);
   for(var tangent=0u;tangent<3u;tangent+=1u){if(tangent!=axis){centroid[tangent]+=.5*f32(span);}}
   // Fine samples are cell centred. Probe the outermost interior centre; a
   // positive value there proves an air gap between the liquid row and wall.
   // The lagged unilateral active set may open it one solve earlier; otherwise
   // the face remains exact zero-aperture Neumann.
   var wallSample=centroid;wallSample[axis]-=towardAir*.5*p.fineCellWidth/p.cellSize;
   let sampled=finePhi(wallSample);face.openFraction=0.;flags=2u;
   let contactEnabled=p.schedule.w==0u;let separated=contactEnabled&&(face.reserved&FACE_SEPARATED)!=0u;
   if(contactEnabled&&(separated||((rowFlags&(VALID|FINITE))==(VALID|FINITE)&&liquid<0.&&sampled.valid!=0u&&sampled.value>0.))){
    let dual=max(.5*p.fineCellWidth,.5*f32(rowSize)*p.cellSize-.5*p.fineCellWidth);
    if(sampled.valid!=0u&&sampled.value>0.&&liquid<0.){airPhi=sampled.value;
     // Match the established ghost-fluid robustness floor used by the Power
     // reference. A 1e-4 cut-cell fraction makes this first-order operator up
     // to 100x stiffer than the 2017 path and exhausts the fixed MGPCG tail
     // before the symmetric dam front has crossed one coarse cell.
     theta=clamp(-liquid/(airPhi-liquid),1e-2,1.);distance=theta*dual;
    }else{distance=max(.5*p.fineCellWidth,.5*f32(rowSize)*p.cellSize);}
    face.inverseDistance=1./distance;face.openFraction=1.;flags=3u;}
   faces[faceId]=face;
  }else if(boundary){
   // An authored open boundary is pressure-Dirichlet at the wall plane.
   let rowSize=select(1u,targetLoad(rowBase+8u*face.negativeRow+1u,reuse),
    face.negativeRow<ghostRows());
   distance=max(.5*p.fineCellWidth,.5*f32(rowSize)*p.cellSize);face.inverseDistance=1./distance;
   face.openFraction=1.;faces[faceId]=face;flags=1u;
  }else if(face.negativeRow<targetLoad(2u,reuse)){let entry=targetLoad(9u,reuse)+8u*face.negativeRow;
   let liquid=bitcast<f32>(targetLoad(entry+2u,reuse));let rowSize=targetLoad(entry+1u,reuse);let rowFlags=targetLoad(entry+5u,reuse);
   let sign=select(1.,-1.,(face.reserved&0x80000000u)!=0u);var centroid=vec3f(origin);
   for(var tangent=0u;tangent<3u;tangent+=1u){if(tangent!=axis){centroid[tangent]+=.5*f32(span);}}
   // A missing pressure neighbor is staged at the same size as this row, so
   // its immutable centre-to-centre dual is exactly one row width. Rebuild it
   // from geometry: face.inverseDistance may already contain this pass's prior
   // theta-conditioned write and is therefore not an admissible input.
   let dual=f32(rowSize)*p.cellSize;distance=dual;let faceToAir=.5*dual;
   var endpoint=centroid;endpoint[axis]+=sign*faceToAir/p.cellSize;
   let sampled=finePhi(endpoint);let cellCenteredAir=p.schedule.w==1u;
   let endpointCell=vec3i(floor(endpoint));let solidFraction=denseSolidFraction(endpointCell);
   // Rigid carving can remove a pressure row before the cell becomes fully
   // occupied. On an analytically blocked, rigid-owned face that missing row
   // is still solid Neumann, never atmospheric air.
   let solidNeighbor=solidFraction>=.999999
    ||(denseRigidOwner(endpointCell)>=0&&face.openFraction<.999999);
   if(solidNeighbor){
    // Carving makes a rigid interior positive in phi, but that is not air.
    // The cut-face aperture owns the solid and this pressure face remains
    // Neumann; only a genuine liquid/air neighbor may install p_air = 0.
    face.reserved|=FACE_SOLID_NEUMANN;flags=4u;
   }else if(cellCenteredAir&&(rowFlags&(VALID|FINITE))==(VALID|FINITE)&&liquid<0.){
    // Losasso 2004 places the p_air=0 Dirichlet value at the neighboring air
    // cell centre.  No subcell theta enters the pressure operator in this A/B.
    theta=1.;distance=dual;face.inverseDistance=1./dual;faces[faceId]=face;flags=1u;
   }else if((rowFlags&(VALID|FINITE))==(VALID|FINITE)&&liquid<0.&&sampled.valid!=0u&&sampled.value>0.){airPhi=sampled.value;
    theta=clamp(-liquid/(airPhi-liquid),1e-4,1.);distance=theta*dual;face.inverseDistance=1./distance;faces[faceId]=face;flags=1u;}
   else{flags=4u;}}
 }faces[faceId]=face;ghosts[faceId]=vec4u(bitcast<u32>(distance),bitcast<u32>(theta),bitcast<u32>(airPhi),flags);}
`;

export const octreeLosassoWarmPhiWGSL = /* wgsl */ `
const MAGIC:u32=${OCTREE_LOSASSO_COARSE_PHI_MAGIC}u;const VALID:u32=1u;const FINITE:u32=2u;
struct Params{brickDimensions:vec3u,brickResolution:u32,sampleDimensions:vec3u,samplesPerBrick:u32,
 domainOrigin:vec3f,fineCellWidth:f32,pageCapacity:u32,generation:u32,worklistCapacity:u32,worklistHeaderWords:u32,
 dimensions:vec3u,maximumLeafSize:u32,cellSize:f32,directoryCapacity:u32,rowCapacity:u32,faceCapacity:u32,
 schedule:vec4u}
struct LeafHeader{cell:u32,entryStart:u32,entryCount:u32,size:u32,diagonal:f32,rhs:f32,pad0:u32,pad1:u32,gradient:vec4f}
@group(0)@binding(0)var<uniform>p:Params;@group(0)@binding(4)var<storage,read>headers:array<LeafHeader>;
@group(0)@binding(5)var<storage,read>coarseControl:array<u32>;@group(0)@binding(9)var<storage,read_write>rowPhi:array<vec4u>;
@group(0)@binding(14)var<storage,read>previousArena:array<u32>;
fn finite(v:f32)->bool{return v==v&&abs(v)<3.402823e38;}
fn hash(cell:u32,size:u32)->u32{return ((cell*0x9e3779b1u)^size)*0x85ebca6bu;}
@compute @workgroup_size(64)fn warmLosassoCoarsePhi(@builtin(global_invocation_id)invocation:vec3u){let row=invocation.x;
 if(arrayLength(&coarseControl)>5u&&coarseControl[5]==1u){return;}
 let count=min(min(coarseControl[1],p.rowCapacity),arrayLength(&headers));if(row>=count){return;}let header=headers[row];
 var value=-.5*f32(header.size)*p.cellSize;var minimum=-f32(header.size)*p.cellSize;var maximum=-1e-4*p.cellSize;
 if(arrayLength(&previousArena)>=20u&&previousArena[0]==MAGIC&&previousArena[4]==p.maximumLeafSize
  &&previousArena[5]==p.dimensions.x&&previousArena[6]==p.dimensions.y&&previousArena[7]==p.dimensions.z){let capacity=previousArena[11];
  if(capacity>0u&&(capacity&(capacity-1u))==0u){let wanted=hash(header.cell,header.size);let mask=capacity-1u;
   for(var probe=0u;probe<32u;probe+=1u){let at=previousArena[10]+4u*((wanted+probe)&mask);if(at+3u>=arrayLength(&previousArena)){break;}
    let key=previousArena[at];if(key==0u){break;}if(key==header.cell+1u&&previousArena[at+1u]==header.size&&previousArena[at+3u]==wanted){let priorRow=previousArena[at+2u]-1u;let entry=previousArena[9]+8u*priorRow;
     if(entry+7u<arrayLength(&previousArena)&&(previousArena[entry+5u]&(VALID|FINITE))==(VALID|FINITE)){let candidate=bitcast<f32>(previousArena[entry+2u]);let lo=bitcast<f32>(previousArena[entry+3u]);let hi=bitcast<f32>(previousArena[entry+4u]);
      if(finite(candidate)&&finite(lo)&&finite(hi)){value=candidate;minimum=lo;maximum=hi;}}break;}}}}
 rowPhi[row]=vec4u(bitcast<u32>(value),bitcast<u32>(minimum),bitcast<u32>(maximum),VALID|FINITE);}
`;

export const octreeLosassoVolumeBridgeWGSL = /* wgsl */ `
const MAGIC:u32=${OCTREE_LOSASSO_COARSE_PHI_MAGIC}u;
struct Params{brickDimensions:vec3u,brickResolution:u32,sampleDimensions:vec3u,samplesPerBrick:u32,
 domainOrigin:vec3f,fineCellWidth:f32,pageCapacity:u32,generation:u32,worklistCapacity:u32,worklistHeaderWords:u32,
 dimensions:vec3u,maximumLeafSize:u32,cellSize:f32,directoryCapacity:u32,rowCapacity:u32,faceCapacity:u32,
 schedule:vec4u}
@group(0)@binding(0)var<uniform>p:Params;@group(0)@binding(8)var<storage,read>arena:array<u32>;
@group(0)@binding(5)var<storage,read>coarseControl:array<u32>;
@group(0)@binding(11)var<storage,read_write>volumeDirectory:array<u32>;
@group(0)@binding(12)var<storage,read_write>volumePublication:array<u32>;
@group(0)@binding(13)var<storage,read_write>physicalVolumes:array<f32>;
@group(0)@binding(14)var<storage,read>previousArena:array<u32>;
@group(0)@binding(15)var<storage,read_write>summaryDelta:array<u32>;
fn denseTrackerFlag()->u32{let volume=p.dimensions.x*p.dimensions.y*p.dimensions.z;
 let capacity=(max(arrayLength(&volumeDirectory),8u)-8u)/8u;
 return select(0u,0x40000000u,capacity>=p.rowCapacity+volume&&(volumeDirectory[1]&0x40000000u)!=0u);}
@compute @workgroup_size(64)
fn publishLosassoVolumeBridge(@builtin(global_invocation_id)invocation:vec3u){let row=invocation.x;let good=arena[0]==MAGIC;
 let reuse=p.schedule.x==0u&&arrayLength(&coarseControl)>5u&&coarseControl[5]==1u;
 if(reuse){let retainedGood=arrayLength(&previousArena)>=20u&&previousArena[0]==MAGIC;
  let retainedCount=select(0u,min(previousArena[2],p.rowCapacity),retainedGood);
  if(row==0u){volumeDirectory[0]=select(0u,0x80000000u,retainedGood);
   volumeDirectory[1]=previousArena[12]|denseTrackerFlag();volumeDirectory[2]=retainedCount;volumeDirectory[3]=p.maximumLeafSize;
   volumeDirectory[4]=p.dimensions.x;volumeDirectory[5]=p.dimensions.y;volumeDirectory[6]=p.dimensions.z;
   volumeDirectory[7]=bitcast<u32>(p.cellSize);volumePublication[0]=0u;volumePublication[2]=retainedCount;
   volumePublication[10]=previousArena[12];volumePublication[11]=select(0u,0x80000000u,retainedGood);
   summaryDelta[0]=retainedCount;summaryDelta[1]=previousArena[12];summaryDelta[2]=0u;
   summaryDelta[3]=select(0u,0x80000000u,retainedGood);}
  if(row<retainedCount){let sourceBase=previousArena[9]+8u*row;let destination=8u+8u*row;
   for(var word=0u;word<8u;word+=1u){volumeDirectory[destination+word]=previousArena[sourceBase+word];}
   volumeDirectory[destination+5u]=9u;physicalVolumes[row]=bitcast<f32>(previousArena[sourceBase+7u]);
   let record=16u+4u*row;summaryDelta[record]=previousArena[sourceBase];summaryDelta[record+1u]=previousArena[sourceBase+1u];
   summaryDelta[record+2u]=row;summaryDelta[record+3u]=1u;}
  return;}
 let count=select(0u,min(arena[2],p.rowCapacity),good);let priorGood=arrayLength(&previousArena)>=20u&&previousArena[0]==MAGIC;
 let priorCount=select(0u,min(previousArena[2],p.rowCapacity),priorGood);if(row==0u){volumeDirectory[0]=select(0u,0x80000000u,good);
  volumeDirectory[1]=arena[12]|denseTrackerFlag();volumeDirectory[2]=count;volumeDirectory[3]=p.maximumLeafSize;
  volumeDirectory[4]=p.dimensions.x;volumeDirectory[5]=p.dimensions.y;volumeDirectory[6]=p.dimensions.z;
  volumeDirectory[7]=bitcast<u32>(p.cellSize);volumePublication[0]=0u;volumePublication[2]=count;
  volumePublication[10]=arena[12];volumePublication[11]=select(0u,0x80000000u,good);
  summaryDelta[0]=select(0u,priorCount+count,good);summaryDelta[1]=arena[12];summaryDelta[2]=0u;
  summaryDelta[3]=select(0u,0x80000000u,good);}
 if(row<count){let sourceBase=arena[9]+8u*row;let destination=8u+8u*row;
  for(var word=0u;word<8u;word+=1u){volumeDirectory[destination+word]=arena[sourceBase+word];}
  volumeDirectory[destination+5u]=9u;physicalVolumes[row]=bitcast<f32>(arena[sourceBase+7u]);}
 if(row<priorCount+count){let prior=row<priorCount;let itemRow=select(row-priorCount,row,prior);
  let sourceArenaBase=select(arena[9],previousArena[9],prior)+8u*itemRow;let record=16u+4u*row;
  summaryDelta[record]=select(arena[sourceArenaBase],previousArena[sourceArenaBase],prior);
  summaryDelta[record+1u]=select(arena[sourceArenaBase+1u],previousArena[sourceArenaBase+1u],prior);
  summaryDelta[record+2u]=itemRow;summaryDelta[record+3u]=select(1u,2u,prior);}}

@compute @workgroup_size(64)
fn refreshLosassoVolumeBridge(@builtin(global_invocation_id)invocation:vec3u){let row=invocation.x;let good=arena[0]==MAGIC;
 let count=select(0u,min(arena[2],p.rowCapacity),good);if(row==0u){volumeDirectory[0]=select(0u,0x80000000u,good);
  volumeDirectory[1]=arena[12]|denseTrackerFlag();volumeDirectory[2]=count;volumeDirectory[3]=p.maximumLeafSize;
  volumeDirectory[4]=p.dimensions.x;volumeDirectory[5]=p.dimensions.y;volumeDirectory[6]=p.dimensions.z;
  volumeDirectory[7]=bitcast<u32>(p.cellSize);volumePublication[0]=0u;volumePublication[2]=count;
  volumePublication[10]=arena[12];volumePublication[11]=select(0u,0x80000000u,good);
  summaryDelta[0]=count;summaryDelta[1]=arena[12];summaryDelta[2]=0u;
  summaryDelta[3]=select(0u,0x80000000u,good);}
 if(row<count){let sourceBase=arena[9]+8u*row;let destination=8u+8u*row;
  for(var word=0u;word<8u;word+=1u){volumeDirectory[destination+word]=arena[sourceBase+word];}
  volumeDirectory[destination+5u]=9u;physicalVolumes[row]=bitcast<f32>(arena[sourceBase+7u]);
  let record=16u+4u*row;summaryDelta[record]=arena[sourceBase];summaryDelta[record+1u]=arena[sourceBase+1u];
  summaryDelta[record+2u]=row;summaryDelta[record+3u]=1u;}}
`;
