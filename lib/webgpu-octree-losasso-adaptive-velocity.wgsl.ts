/** Clean-room compact adaptive nodal-velocity reconstruction and extension. */
export const octreeLosassoAdaptiveVelocityWGSL = /* wgsl */ `
const AV_INVALID:u32=0xffffffffu;
const AV_SEEDED:u32=1u;
const AV_VALID_COMPONENTS:u32=7u;
const AV_ERROR_GENERATION:u32=1u;
const AV_ERROR_GRAPH:u32=2u;
const AV_ERROR_PHI:u32=4u;
const AV_ERROR_FACE:u32=8u;
const AV_ERROR_CAPACITY:u32=16u;
const AV_ERROR_COVERAGE:u32=32u;
const AV_ERROR_CONSTRAINT:u32=64u;
const AV_ERROR_NONFINITE:u32=128u;
const AV_ERROR_UNRESOLVED:u32=256u;
const AV_ERROR_COMMIT:u32=512u;
const AV_DIAGNOSTIC_HEADER:u32=13u;
const AV_DIAGNOSTIC_RECORD_WORDS:u32=24u;
const AV_DIAGNOSTIC_RECORDS:u32=64u;
const AV_DIAGNOSTIC_BANK_WORDS:u32=1549u;

// Graph control ABI: generation,node count,leaf count,valid,error bits.
// A compiled face stencil is one fixed 144-byte record per node/component:
// (unique count, covered finest-face units, coarsest span, valid), sixteen
// canonical face slots, then sixteen integer area weights. The topology-only
// builder may search the face directory; recurring reconstruction only loads
// this fixed record and performs sixteen explicitly unrolled face reads.
struct AVParams{capacity:u32,bankOffset:u32,receiptBase:u32,waves:u32,reach:f32,phiStride:u32,phiOffset:u32,pad0:u32,dimensions:vec3u,maximumSpan:u32,faceCapacity:u32,directoryCapacity:u32,pad1:vec2u}
struct AVFaceStencil{header:vec4u,slots0:vec4u,slots1:vec4u,slots2:vec4u,slots3:vec4u,weights0:vec4u,weights1:vec4u,weights2:vec4u,weights3:vec4u}
@group(0)@binding(0)var<uniform> avp:AVParams;

fn avFinite(v:f32)->bool{return v==v&&abs(v)<=3.402823e38;}
fn avLive(generation:u32,published:u32,errors:u32,count:u32)->u32{return select(0u,min(count,avp.capacity),generation!=0u&&published==generation&&errors==0u);}
fn avComponent(v:vec4u,c:u32)->f32{return bitcast<f32>(v[c]);}
fn avSetComponent(v:vec4u,c:u32,x:f32)->vec4u{var r=v;r[c]=bitcast<u32>(x);return r;}

@group(0)@binding(1)var<storage,read>prepareGraph:array<u32>;
@group(0)@binding(2)var<storage,read>prepareAcceptedGraph:array<u32>;
@group(0)@binding(3)var<storage,read>prepareStencilControl:array<u32>;
@group(0)@binding(4)var<storage,read_write>prepareReceipt:array<atomic<u32>>;
@group(0)@binding(5)var<storage,read_write>prepareDiagnostics:array<atomic<u32>>;
@compute @workgroup_size(1)fn prepareAdaptiveVelocity(){
  for(var i=0u;i<12u;i+=1u){atomicStore(&prepareReceipt[avp.receiptBase+i],0u);}
  let diagnosticBase=(avp.receiptBase/12u)*AV_DIAGNOSTIC_BANK_WORDS;
  for(var i=0u;i<AV_DIAGNOSTIC_HEADER;i+=1u){atomicStore(&prepareDiagnostics[diagnosticBase+i],0u);}
  atomicStore(&prepareDiagnostics[diagnosticBase+10u],AV_INVALID);
  let topologyGeneration=select(0u,prepareGraph[0u],arrayLength(&prepareGraph)>=7u);
  let fieldGeneration=select(0u,prepareGraph[5u],arrayLength(&prepareGraph)>=7u);
  atomicStore(&prepareReceipt[avp.receiptBase],fieldGeneration);
  atomicStore(&prepareReceipt[avp.receiptBase+1u],bitcast<u32>(avp.reach));
  let candidateCold=avp.pad0==1u&&(arrayLength(&prepareAcceptedGraph)<7u
    ||prepareAcceptedGraph[0u]==0u||prepareAcceptedGraph[3u]!=prepareAcceptedGraph[0u]
    ||prepareAcceptedGraph[4u]!=0u||prepareAcceptedGraph[5u]==0u
    ||prepareAcceptedGraph[6u]!=prepareAcceptedGraph[5u]);
  // Ready publication is the only accepted path allowed to synthesize the
  // cold zero field, and only for the initial graph clock where topology and
  // surface generation coincide and no velocity has ever been published.
  // A graph commit that already carried coherent candidate nodal fields uses
  // mode 2 and validates those records without filtering them through faces.
  let readyRetain=avp.pad0==2u&&arrayLength(&prepareGraph)>=7u
    &&fieldGeneration!=0u&&prepareGraph[6u]==fieldGeneration;
  let readyCold=avp.pad0==2u&&arrayLength(&prepareGraph)>=7u
    &&topologyGeneration==fieldGeneration&&prepareGraph[6u]==0u;
  let cold=candidateCold||readyCold;
  let skipFaces=cold||readyRetain;
  atomicStore(&prepareReceipt[avp.receiptBase+11u],select(select(0u,2u,readyRetain),1u,cold));
  var errors=0u;
  if(arrayLength(&prepareGraph)<7u||topologyGeneration==0u
    ||prepareGraph[3u]!=topologyGeneration||prepareGraph[4u]!=0u){errors|=AV_ERROR_GRAPH;}
  if(fieldGeneration==0u){errors|=AV_ERROR_PHI;}
  if(arrayLength(&prepareStencilControl)<8u||prepareStencilControl[0u]!=topologyGeneration
    ||prepareStencilControl[1u]!=prepareGraph[2u]
    ||prepareStencilControl[3u]!=topologyGeneration||prepareStencilControl[4u]!=0u){errors|=AV_ERROR_FACE;}
  if(arrayLength(&prepareGraph)>=3u&&prepareGraph[2u]>avp.capacity){errors|=AV_ERROR_CAPACITY;}
  if(arrayLength(&prepareStencilControl)<1u
    ||prepareStencilControl[0u]!=topologyGeneration){errors|=AV_ERROR_GENERATION;}
  atomicStore(&prepareReceipt[avp.receiptBase+2u],errors);
}

// Candidate-only topology compiler. Searches and variable loops belong here,
// before the graph/face tuple is accepted; no recurring physics entry point
// binds the face geometry or its hash directory.
@group(0)@binding(91)var<storage,read>stencilGraph:array<u32>;
@group(0)@binding(92)var<storage,read>stencilNodes:array<vec4u>;
@group(0)@binding(93)var<storage,read>stencilFaceControl:array<u32>;
@group(0)@binding(94)var<storage,read>stencilFaceGeometry:array<vec4u>;
@group(0)@binding(95)var<storage,read>stencilFaceDirectory:array<vec2u>;
@group(0)@binding(96)var<storage,read_write>stencilRecords:array<AVFaceStencil>;
@group(0)@binding(97)var<storage,read_write>stencilControl:array<atomic<u32>>;
fn avStencilHash(packed:u32,q:vec3u)->u32{var h=(packed+1u)*0x9e3779b1u;h=(h^q.x)*0x85ebca6bu;h=(h^q.y)*0xc2b2ae35u;return(h^q.z)*0x27d4eb2du;}
fn avStencilExactFace(axis:u32,logSpan:u32,q:vec3u)->u32{if(arrayLength(&stencilFaceControl)<7u){return AV_INVALID;}let capacity=stencilFaceControl[6u];if(capacity==0u||(capacity&(capacity-1u))!=0u||capacity>avp.directoryCapacity||capacity>arrayLength(&stencilFaceDirectory)){return AV_INVALID;}let packed=axis|(logSpan<<2u);let hash=avStencilHash(packed,q);let mask=capacity-1u;
  for(var probe=0u;probe<32u;probe+=1u){let d=stencilFaceDirectory[(hash+probe)&mask];if(d.x==0u){return AV_INVALID;}let face=d.x-1u;if(d.y==hash&&face<min(avp.faceCapacity,stencilFaceControl[2u])&&face<arrayLength(&stencilFaceGeometry)&&all(stencilFaceGeometry[face]==vec4u(packed,q))){return face;}}return AV_INVALID;}
fn avStencilContainingFace(axis:u32,q:vec3u)->vec2u{var span=1u;var logSpan=0u;loop{var origin=q;for(var c=0u;c<3u;c+=1u){if(c!=axis){origin[c]=(origin[c]/span)*span;}}let face=avStencilExactFace(axis,logSpan,origin);if(face!=AV_INVALID){return vec2u(face,span);}if(span>avp.maximumSpan/2u){break;}span*=2u;logSpan+=1u;}return vec2u(AV_INVALID,0u);}
@compute @workgroup_size(1)fn prepareAdaptiveVelocityStencils(){
  for(var i=0u;i<8u;i+=1u){atomicStore(&stencilControl[i],0u);}var errors=0u;
  let epoch=select(0u,stencilGraph[0u],arrayLength(&stencilGraph)>=7u);let nodes=select(0u,min(stencilGraph[2u],avp.capacity),arrayLength(&stencilGraph)>=7u);
  if(epoch==0u||stencilGraph[3u]!=epoch||stencilGraph[4u]!=0u||stencilGraph[2u]>avp.capacity){errors|=AV_ERROR_GRAPH;}
  if(arrayLength(&stencilFaceControl)<7u||stencilFaceControl[0u]!=epoch||stencilFaceControl[3u]!=1u||stencilFaceControl[4u]!=0u){errors|=AV_ERROR_FACE;}
  if(arrayLength(&stencilFaceControl)>=7u){let directory=stencilFaceControl[6u];if(directory==0u||(directory&(directory-1u))!=0u||directory>avp.directoryCapacity||directory>arrayLength(&stencilFaceDirectory)||directory<2u*stencilFaceControl[2u]){errors|=AV_ERROR_CAPACITY;}}
  if(nodes>arrayLength(&stencilRecords)/3u){errors|=AV_ERROR_CAPACITY;}
  atomicStore(&stencilControl[0u],select(0u,epoch,errors==0u));atomicStore(&stencilControl[1u],select(0u,nodes,errors==0u));atomicStore(&stencilControl[2u],select(0u,stencilFaceControl[2u],errors==0u));atomicStore(&stencilControl[4u],errors);
}
fn avAddStencilFace(face:u32,area:u32,slots:ptr<function,array<u32,16>>,weights:ptr<function,array<u32,16>>,count:ptr<function,u32>,covered:ptr<function,u32>)->bool{
  if(face==AV_INVALID){return true;}var found=AV_INVALID;for(var i=0u;i<(*count);i+=1u){if((*slots)[i]==face){found=i;break;}}
  if(found==AV_INVALID){if((*count)>=16u){return false;}found=(*count);(*slots)[found]=face;(*count)+=1u;}(*weights)[found]+=area;(*covered)+=area;return true;
}
fn avPackStencil(count:u32,covered:u32,coarse:u32,valid:bool,slots:array<u32,16>,weights:array<u32,16>)->AVFaceStencil{return AVFaceStencil(
  vec4u(count,covered,coarse,select(0u,1u,valid)),vec4u(slots[0],slots[1],slots[2],slots[3]),vec4u(slots[4],slots[5],slots[6],slots[7]),vec4u(slots[8],slots[9],slots[10],slots[11]),vec4u(slots[12],slots[13],slots[14],slots[15]),
  vec4u(weights[0],weights[1],weights[2],weights[3]),vec4u(weights[4],weights[5],weights[6],weights[7]),vec4u(weights[8],weights[9],weights[10],weights[11]),vec4u(weights[12],weights[13],weights[14],weights[15]));}
@compute @workgroup_size(64)fn compileAdaptiveVelocityStencils(@builtin(global_invocation_id)id:vec3u,@builtin(num_workgroups)n:vec3u){
  let node=id.x+id.y*n.x*64u;let live=avLive(stencilGraph[0u],stencilGraph[3u],stencilGraph[4u],stencilGraph[2u]);if(node>=live||atomicLoad(&stencilControl[4u])!=0u){return;}let constrained=node>=arrayLength(&stencilNodes)||stencilNodes[node].w!=AV_INVALID;let item=stencilNodes[node].x;let nd=avp.dimensions+vec3u(1u);let q=vec3u(item%nd.x,(item/nd.x)%nd.y,item/(nd.x*nd.y));
  // A component value at a graph node is restricted from the one MAC face
  // incident on each tangential quadrant. Weighting that face by span^2 is
  // exactly the finest-face-area average. Expanding every quadrant to the
  // largest incident span instead samples faces which do not touch this node;
  // the resulting non-local stencil changes when distant cells refine and was
  // the source of the first graded dam candidate's capacity rejection.
  for(var component=0u;component<3u;component+=1u){var slots:array<u32,16>;var weights:array<u32,16>;for(var i=0u;i<16u;i+=1u){slots[i]=AV_INVALID;weights[i]=0u;}var count=0u;var covered=0u;var coarse=1u;var valid=true;
    if(!constrained){let a=(component+1u)%3u;let b=(component+2u)%3u;
      for(var quadrant=0u;quadrant<4u;quadrant+=1u){var sample=q;sample[a]=select(select(0u,q[a]-1u,q[a]>0u),min(q[a],avp.dimensions[a]-1u),(quadrant&1u)!=0u);sample[b]=select(select(0u,q[b]-1u,q[b]>0u),min(q[b],avp.dimensions[b]-1u),(quadrant&2u)!=0u);let face=avStencilContainingFace(component,sample);if(face.x!=AV_INVALID){coarse=max(coarse,face.y);valid=avAddStencilFace(face.x,face.y*face.y,&slots,&weights,&count,&covered)&&valid;}}}
    if(!valid){atomicOr(&stencilControl[4u],AV_ERROR_CAPACITY);}atomicAdd(&stencilControl[5u],covered);atomicMax(&stencilControl[6u],count);stencilRecords[3u*node+component]=avPackStencil(count,covered,coarse,valid,slots,weights);
  }atomicAdd(&stencilControl[7u],1u);
}
@compute @workgroup_size(1)fn finishAdaptiveVelocityStencils(){let epoch=atomicLoad(&stencilControl[0u]);let complete=epoch!=0u&&atomicLoad(&stencilControl[4u])==0u&&atomicLoad(&stencilControl[7u])==atomicLoad(&stencilControl[1u]);atomicStore(&stencilControl[3u],select(0u,epoch,complete));}

@group(0)@binding(101)var<storage,read>commitStencilCandidateControl:array<u32>;
@group(0)@binding(102)var<storage,read>commitStencilCandidate:array<AVFaceStencil>;
@group(0)@binding(103)var<storage,read>commitStencilAcceptedGraph:array<u32>;
@group(0)@binding(104)var<storage,read>commitStencilAcceptedFaces:array<u32>;
@group(0)@binding(105)var<storage,read_write>commitStencilAcceptedControl:array<u32>;
@group(0)@binding(106)var<storage,read_write>commitStencilAccepted:array<AVFaceStencil>;
fn avStencilCommitReady()->bool{let epoch=commitStencilCandidateControl[0u];return epoch!=0u&&commitStencilCandidateControl[3u]==epoch&&commitStencilCandidateControl[4u]==0u&&commitStencilAcceptedGraph[0u]==epoch&&commitStencilAcceptedGraph[3u]==epoch&&commitStencilAcceptedGraph[4u]==0u&&commitStencilAcceptedGraph[2u]==commitStencilCandidateControl[1u]&&commitStencilAcceptedFaces[0u]==epoch&&commitStencilAcceptedFaces[3u]==1u&&commitStencilAcceptedFaces[4u]==0u;}
@compute @workgroup_size(64)fn commitAdaptiveVelocityStencils(@builtin(global_invocation_id)id:vec3u,@builtin(num_workgroups)n:vec3u){let node=id.x+id.y*n.x*64u;if(!avStencilCommitReady()||node>=commitStencilCandidateControl[1u]){return;}commitStencilAccepted[3u*node]=commitStencilCandidate[3u*node];commitStencilAccepted[3u*node+1u]=commitStencilCandidate[3u*node+1u];commitStencilAccepted[3u*node+2u]=commitStencilCandidate[3u*node+2u];}
@compute @workgroup_size(1)fn finishAdaptiveVelocityStencilCommit(){if(!avStencilCommitReady()){return;}for(var i=0u;i<8u;i+=1u){commitStencilAcceptedControl[i]=commitStencilCandidateControl[i];}}

@group(0)@binding(11)var<storage,read>reconstructGraph:array<u32>;
@group(0)@binding(12)var<storage,read>reconstructNodes:array<vec4u>;
@group(0)@binding(13)var<storage,read>reconstructStencils:array<AVFaceStencil>;
@group(0)@binding(14)var<storage,read>reconstructFaceValues:array<f32>;
@group(0)@binding(15)var<storage,read_write>reconstructVelocity:array<vec4u>;
@group(0)@binding(16)var<storage,read_write>reconstructStatus:array<u32>;
@group(0)@binding(17)var<storage,read_write>reconstructReceipt:array<atomic<u32>>;
@group(0)@binding(18)var<storage,read_write>reconstructSupport:array<u32>;
@group(0)@binding(19)var<storage,read>reconstructFaceStatus:array<u32>;
const AV_EXACT_LIMBS:u32=36u;
fn avFloorDiv256(value:i32)->vec2i{let carry=value>>8;return vec2i(carry,value-carry*256);}
// The stencil compiler produces an invariant weighted multiset, but its slot
// order follows face publication and therefore changes under D4 transforms.
// Accumulate the sixteen local scalar terms exactly so reconstruction depends
// only on that multiset. The small private limb array is arithmetic scratch;
// every topology/face load remains fixed and explicitly unrolled below.
fn avExactAdd(limbs:ptr<function,array<i32,36>>,value:f32){let bits=bitcast<u32>(value);let magnitude=bits&0x7fffffffu;if(magnitude==0u){return;}let exponent=(magnitude>>23u)&0xffu;let fraction=magnitude&0x7fffffu;let significand=select(fraction,0x800000u|fraction,exponent!=0u);let shift=select(3u,exponent+2u,exponent!=0u);let first=shift>>3u;let shifted=significand<<(shift&7u);let sign=select(1,-1,(bits&0x80000000u)!=0u);for(var digit=0u;digit<4u;digit+=1u){let limb=first+digit;let byte=i32((shifted>>(digit*8u))&0xffu);if(byte!=0&&limb<AV_EXACT_LIMBS){(*limbs)[limb]+=sign*byte;}}}
fn avExactValue(input:ptr<function,array<i32,36>>)->f32{for(var limb=0u;limb+1u<AV_EXACT_LIMBS;limb+=1u){let n=avFloorDiv256((*input)[limb]);(*input)[limb]=n.y;(*input)[limb+1u]+=n.x;}let negative=(*input)[AV_EXACT_LIMBS-1u]<0;if(negative){for(var limb=0u;limb<AV_EXACT_LIMBS;limb+=1u){(*input)[limb]=-(*input)[limb];}for(var limb=0u;limb+1u<AV_EXACT_LIMBS;limb+=1u){let n=avFloorDiv256((*input)[limb]);(*input)[limb]=n.y;(*input)[limb+1u]+=n.x;}}var magnitude=0.;for(var limb=0u;limb<AV_EXACT_LIMBS;limb+=1u){magnitude+=ldexp(f32((*input)[limb]),-152+i32(8u*limb));}return select(magnitude,-magnitude,negative);}
fn avAccumulateStencilFace(slot:u32,weight:u32,sum:ptr<function,array<i32,36>>,covered:ptr<function,u32>,valid:ptr<function,bool>){if(weight==0u){return;}if(slot==AV_INVALID||slot>=arrayLength(&reconstructFaceValues)){atomicOr(&reconstructReceipt[avp.receiptBase+2u],AV_ERROR_FACE);(*valid)=false;return;}
  // A candidate topology can cover only part of a coarse transition sheet.
  // Tall Cells 3.3.1 restricts from known fine velocities only, renormalizing
  // their weights. Unknown migration status is therefore absence, not a bad
  // zero and not grounds for discarding the other known face samples.
  if(avp.pad1.x!=0u&&(slot>=arrayLength(&reconstructFaceStatus)||reconstructFaceStatus[slot]==0u)){return;}let value=reconstructFaceValues[slot];let term=f32(weight)*value;if(!avFinite(value)||!avFinite(term)){atomicOr(&reconstructReceipt[avp.receiptBase+2u],AV_ERROR_NONFINITE);(*valid)=false;return;}avExactAdd(sum,term);(*covered)+=weight;}
fn avStencilComponent(node:u32,component:u32)->vec2f{let at=3u*node+component;if(at>=arrayLength(&reconstructStencils)){atomicOr(&reconstructReceipt[avp.receiptBase+2u],AV_ERROR_FACE);return vec2f(0.0);}let s=reconstructStencils[at];if(s.header.w==0u||s.header.x>16u){atomicOr(&reconstructReceipt[avp.receiptBase+2u],AV_ERROR_FACE);return vec2f(0.0);}var sum:array<i32,36>;var covered=0u;var valid=true;
  avAccumulateStencilFace(s.slots0.x,s.weights0.x,&sum,&covered,&valid);avAccumulateStencilFace(s.slots0.y,s.weights0.y,&sum,&covered,&valid);avAccumulateStencilFace(s.slots0.z,s.weights0.z,&sum,&covered,&valid);avAccumulateStencilFace(s.slots0.w,s.weights0.w,&sum,&covered,&valid);
  avAccumulateStencilFace(s.slots1.x,s.weights1.x,&sum,&covered,&valid);avAccumulateStencilFace(s.slots1.y,s.weights1.y,&sum,&covered,&valid);avAccumulateStencilFace(s.slots1.z,s.weights1.z,&sum,&covered,&valid);avAccumulateStencilFace(s.slots1.w,s.weights1.w,&sum,&covered,&valid);
  avAccumulateStencilFace(s.slots2.x,s.weights2.x,&sum,&covered,&valid);avAccumulateStencilFace(s.slots2.y,s.weights2.y,&sum,&covered,&valid);avAccumulateStencilFace(s.slots2.z,s.weights2.z,&sum,&covered,&valid);avAccumulateStencilFace(s.slots2.w,s.weights2.w,&sum,&covered,&valid);
  avAccumulateStencilFace(s.slots3.x,s.weights3.x,&sum,&covered,&valid);avAccumulateStencilFace(s.slots3.y,s.weights3.y,&sum,&covered,&valid);avAccumulateStencilFace(s.slots3.z,s.weights3.z,&sum,&covered,&valid);avAccumulateStencilFace(s.slots3.w,s.weights3.w,&sum,&covered,&valid);
  if(!valid||covered==0u){return vec2f(0.0);}if(covered>s.header.y){atomicOr(&reconstructReceipt[avp.receiptBase+2u],AV_ERROR_FACE);return vec2f(0.0);}let complete=covered==s.header.y;if(!complete){atomicAdd(&reconstructReceipt[avp.receiptBase+9u],1u);}let value=avExactValue(&sum)/f32(covered);if(!avFinite(value)){atomicOr(&reconstructReceipt[avp.receiptBase+2u],AV_ERROR_NONFINITE);return vec2f(0.0);}atomicAdd(&reconstructReceipt[avp.receiptBase+8u],s.header.x);return vec2f(value,select(2.0,1.0,complete));}
@compute @workgroup_size(64)fn reconstructAdaptiveVelocity(@builtin(global_invocation_id)id:vec3u,@builtin(num_workgroups)n:vec3u){
  let node=id.x+id.y*n.x*64u;if(node>=avLive(reconstructGraph[0],reconstructGraph[3],reconstructGraph[4],reconstructGraph[2])){return;}
  reconstructStatus[node]=0u;reconstructSupport[node]=0u;var velocity=vec4u(0u);var mask=0u;
  let constrained=node>=arrayLength(&reconstructNodes)||reconstructNodes[node].w!=AV_INVALID;
  let readyMode=atomicLoad(&reconstructReceipt[avp.receiptBase+11u]);
  if(!constrained&&readyMode==1u){mask=AV_VALID_COMPONENTS;}
  if(!constrained&&readyMode==2u){velocity=reconstructVelocity[2u*node+avp.bankOffset];mask=velocity.w&AV_VALID_COMPONENTS;}
  if(!constrained&&readyMode==0u){let retained=reconstructVelocity[2u*node+avp.bankOffset];if(avp.pad0==0u){velocity=retained;mask=retained.w&AV_VALID_COMPONENTS;}atomicAdd(&reconstructReceipt[avp.receiptBase+10u],1u);
    let x=avStencilComponent(node,0u);if(x.y!=0.0){velocity=avSetComponent(velocity,0u,x.x);}if(x.y==1.0){mask|=1u;}else{atomicAdd(&reconstructReceipt[avp.receiptBase+4u],1u);}
    let y=avStencilComponent(node,1u);if(y.y!=0.0){velocity=avSetComponent(velocity,1u,y.x);}if(y.y==1.0){mask|=2u;}else{atomicAdd(&reconstructReceipt[avp.receiptBase+4u],1u);}
    let z=avStencilComponent(node,2u);if(z.y!=0.0){velocity=avSetComponent(velocity,2u,z.x);}if(z.y==1.0){mask|=4u;}else{atomicAdd(&reconstructReceipt[avp.receiptBase+4u],1u);}
  }
  velocity.w=mask;reconstructVelocity[2u*node+avp.bankOffset]=velocity;reconstructStatus[node]=mask;
  if(mask!=0u){atomicAdd(&reconstructReceipt[avp.receiptBase+3u],1u);}
}

// Candidate topology publication may introduce air-side independent nodes
// whose new pressure-face stencil has no liquid seed. Preserve exact
// coincident values from the still-accepted nodal authority before extending
// onto genuinely new nodes. This sorted-directory search is candidate-only
// topology work; recurring accepted reconstruction never binds this pass.
@group(0)@binding(111)var<storage,read>handoffGraph:array<u32>;
@group(0)@binding(112)var<storage,read>handoffNodes:array<vec4u>;
@group(0)@binding(113)var<storage,read_write>handoffVelocity:array<vec4u>;
@group(0)@binding(114)var<storage,read_write>handoffStatus:array<u32>;
@group(0)@binding(115)var<storage,read>handoffAcceptedGraph:array<u32>;
@group(0)@binding(116)var<storage,read>handoffAcceptedDirectory:array<vec2u>;
@group(0)@binding(117)var<storage,read>handoffAcceptedVelocity:array<vec4u>;
@compute @workgroup_size(64)fn handoffAdaptiveVelocity(@builtin(global_invocation_id)id:vec3u,@builtin(num_workgroups)n:vec3u){
  let node=id.x+id.y*n.x*64u;let live=avLive(handoffGraph[0],handoffGraph[3],handoffGraph[4],handoffGraph[2]);if(node>=live||node>=arrayLength(&handoffNodes)||node>=arrayLength(&handoffStatus)||handoffNodes[node].w!=AV_INVALID){return;}
  if(arrayLength(&handoffAcceptedGraph)<7u||handoffAcceptedGraph[0u]==0u||handoffAcceptedGraph[3u]!=handoffAcceptedGraph[0u]||handoffAcceptedGraph[4u]!=0u||handoffAcceptedGraph[5u]==0u||handoffAcceptedGraph[6u]!=handoffAcceptedGraph[5u]){return;}
  let acceptedLive=min(handoffAcceptedGraph[2u],u32(arrayLength(&handoffAcceptedDirectory)));let item=handoffNodes[node].x;var low=0u;var high=acceptedLive;while(low<high){let middle=low+(high-low)/2u;let record=handoffAcceptedDirectory[middle];if(record.x<item){low=middle+1u;}else{high=middle;}}
  if(low>=acceptedLive){return;}let found=handoffAcceptedDirectory[low];if(found.x!=item||found.y>=handoffAcceptedGraph[2u]){return;}let sourceAt=2u*found.y+avp.bankOffset;let outputAt=2u*node+avp.bankOffset;if(sourceAt>=arrayLength(&handoffAcceptedVelocity)||outputAt>=arrayLength(&handoffVelocity)){return;}let source=handoffAcceptedVelocity[sourceAt];let sourceMask=source.w&AV_VALID_COMPONENTS;var output=handoffVelocity[outputAt];let old=handoffStatus[node]&AV_VALID_COMPONENTS;var next=old;for(var component=0u;component<3u;component+=1u){let bit=1u<<component;if((sourceMask&bit)!=0u){let value=avComponent(source,component);if(avFinite(value)){output=avSetComponent(output,component,value);next|=bit;}}}output.w=next;handoffVelocity[outputAt]=output;handoffStatus[node]=next;
}

@group(0)@binding(61)var<storage,read>supportGraph:array<u32>;
@group(0)@binding(62)var<storage,read>supportBandMask:array<u32>;
@group(0)@binding(63)var<storage,read>supportConstraints:array<u32>;
@group(0)@binding(64)var<storage,read_write>supportStatus:array<atomic<u32>>;
@group(0)@binding(65)var<storage,read_write>supportDiagnostics:array<atomic<u32>>;
@group(0)@binding(66)var<storage,read>supportLeaves:array<vec4u>;
@group(0)@binding(67)var<storage,read>supportIncidentLeaves:array<u32>;
fn markAdaptiveVelocitySupportSlot(slot:u32,live:u32,diagnosticBase:u32){
  if(slot>=live||slot>=arrayLength(&supportStatus)){return;}
  let old=atomicOr(&supportStatus[slot],1u);if(old==0u){atomicAdd(&supportDiagnostics[diagnosticBase+7u],1u);}
}
// The graph validator guarantees that every hanging-node master is
// independent. Resolve the fixed 0/2/4-way record explicitly: this pass is a
// compiled-topology consumer, never a directory search or pointer chase loop.
fn markAdaptiveVelocityNodeClosure(slot:u32,live:u32,diagnosticBase:u32){
  if(slot>=live){return;}let base=12u*slot;if(base+11u>=arrayLength(&supportConstraints)){return;}
  let count=supportConstraints[base+1u];
  if(count==0u){markAdaptiveVelocitySupportSlot(slot,live,diagnosticBase);return;}
  if(count!=2u&&count!=4u){return;}
  markAdaptiveVelocitySupportSlot(supportConstraints[base+4u],live,diagnosticBase);
  markAdaptiveVelocitySupportSlot(supportConstraints[base+5u],live,diagnosticBase);
  if(count==4u){
    markAdaptiveVelocitySupportSlot(supportConstraints[base+6u],live,diagnosticBase);
    markAdaptiveVelocitySupportSlot(supportConstraints[base+7u],live,diagnosticBase);
  }
}
fn markAdaptiveVelocityLeafClosure(leaf:u32,live:u32,leafLive:u32,diagnosticBase:u32){
  let at=4u*leaf;if(leaf>=leafLive||at+3u>=arrayLength(&supportLeaves)||
    (supportLeaves[at+1u].x&1u)==0u){return;}
  let low=supportLeaves[at+2u];let high=supportLeaves[at+3u];
  markAdaptiveVelocityNodeClosure(low.x,live,diagnosticBase);
  markAdaptiveVelocityNodeClosure(low.y,live,diagnosticBase);
  markAdaptiveVelocityNodeClosure(low.z,live,diagnosticBase);
  markAdaptiveVelocityNodeClosure(low.w,live,diagnosticBase);
  markAdaptiveVelocityNodeClosure(high.x,live,diagnosticBase);
  markAdaptiveVelocityNodeClosure(high.y,live,diagnosticBase);
  markAdaptiveVelocityNodeClosure(high.z,live,diagnosticBase);
  markAdaptiveVelocityNodeClosure(high.w,live,diagnosticBase);
}
// Compile the exact interpolation closure of the transport band. Each node's
// eight signed-octant leaves were resolved once at graph publication; fixed
// direct loads now demand every independent leaf corner needed by midpoint
// trilinear sampling, including corners just outside the nominal phi reach.
@compute @workgroup_size(64)fn markAdaptiveVelocitySupport(@builtin(global_invocation_id)id:vec3u,@builtin(num_workgroups)n:vec3u){
  let live=avLive(supportGraph[0],supportGraph[3],supportGraph[4],supportGraph[2]);let node=id.x+id.y*n.x*64u;let base=12u*node;if(node>=live||base+11u>=arrayLength(&supportConstraints)){return;}
  if(node>=arrayLength(&supportBandMask)||supportBandMask[node]==0u){return;}
  let diagnosticBase=(avp.receiptBase/12u)*AV_DIAGNOSTIC_BANK_WORDS;
  markAdaptiveVelocityNodeClosure(node,live,diagnosticBase);
  let incidentBase=8u*node;if(incidentBase+7u>=arrayLength(&supportIncidentLeaves)){return;}let leafLive=supportGraph[1u];
  markAdaptiveVelocityLeafClosure(supportIncidentLeaves[incidentBase],live,leafLive,diagnosticBase);
  markAdaptiveVelocityLeafClosure(supportIncidentLeaves[incidentBase+1u],live,leafLive,diagnosticBase);
  markAdaptiveVelocityLeafClosure(supportIncidentLeaves[incidentBase+2u],live,leafLive,diagnosticBase);
  markAdaptiveVelocityLeafClosure(supportIncidentLeaves[incidentBase+3u],live,leafLive,diagnosticBase);
  markAdaptiveVelocityLeafClosure(supportIncidentLeaves[incidentBase+4u],live,leafLive,diagnosticBase);
  markAdaptiveVelocityLeafClosure(supportIncidentLeaves[incidentBase+5u],live,leafLive,diagnosticBase);
  markAdaptiveVelocityLeafClosure(supportIncidentLeaves[incidentBase+6u],live,leafLive,diagnosticBase);
  markAdaptiveVelocityLeafClosure(supportIncidentLeaves[incidentBase+7u],live,leafLive,diagnosticBase);
}

@group(0)@binding(21)var<storage,read>extendGraph:array<u32>;
@group(0)@binding(22)var<storage,read>extendAdjacency:array<u32>;
@group(0)@binding(23)var<storage,read>extendPhi:array<f32>;
@group(0)@binding(24)var<storage,read_write>extendVelocity:array<vec4u>;
@group(0)@binding(25)var<storage,read>extendStatusIn:array<u32>;
@group(0)@binding(26)var<storage,read_write>extendStatusOut:array<u32>;
@group(0)@binding(27)var<storage,read_write>extendReceipt:array<atomic<u32>>;
@compute @workgroup_size(64)fn extendAdaptiveVelocity(@builtin(global_invocation_id)id:vec3u,@builtin(num_workgroups)n:vec3u){
  let live=avLive(extendGraph[0],extendGraph[3],extendGraph[4],extendGraph[2]);let node=id.x+id.y*n.x*64u;if(node>=live){return;}let old=extendStatusIn[node];extendStatusOut[node]=old;
  let phiAt=avp.phiStride*node+avp.phiOffset;if(phiAt>=arrayLength(&extendPhi)||!avFinite(extendPhi[phiAt])){atomicOr(&extendReceipt[avp.receiptBase+2u],AV_ERROR_NONFINITE);return;}let nodePhi=extendPhi[phiAt];
  // The accurate Jeong/causal solve is intentionally a two-finest-cell shell.
  // Outside it, the following sparse hierarchy closure performs known-only
  // restriction/prolongation over the adaptive graph.
  let accurateReach=min(avp.reach,bitcast<f32>(avp.pad1.y));
  if(abs(nodePhi)>accurateReach||old==AV_VALID_COMPONENTS){return;}
  var value=extendVelocity[2u*node+avp.bankOffset];var next=old;
  for(var component=0u;component<3u;component+=1u){if((old&(1u<<component))!=0u){continue;}
    var weighted=0.;var weightSum=0.;
    for(var direction=0u;direction<6u;direction+=1u){let at=12u*node+direction;if(at+6u>=arrayLength(&extendAdjacency)){continue;}let neighborSlot=extendAdjacency[at];
      let neighborPhiAt=avp.phiStride*neighborSlot+avp.phiOffset;if(neighborSlot==AV_INVALID||neighborSlot>=live||neighborPhiAt>=arrayLength(&extendPhi)||neighborSlot>=arrayLength(&extendStatusIn)){continue;}
      let distance=bitcast<f32>(extendAdjacency[at+6u]);let neighborPhi=extendPhi[neighborPhiAt];
      // Signed-distance plateaus are common on authored planar interfaces.
      // They are one monotone FMM bucket, not local minima: synchronous status
      // still guarantees that only a node accepted by an earlier wave donates.
      // Reject only a strictly farther neighbor so the bucket floods outward
      // from real reconstructed face seeds without a held-value fallback.
      if(distance<=0.||!avFinite(distance)||!avFinite(neighborPhi)||abs(neighborPhi)>abs(nodePhi)||(extendStatusIn[neighborSlot]&(1u<<component))==0u){continue;}
      let neighbor=avComponent(extendVelocity[2u*neighborSlot+avp.bankOffset],component);if(!avFinite(neighbor)){continue;}
      let weight=1./distance;weighted+=weight*neighbor;weightSum+=weight;
    }
    if(weightSum>0.){value=avSetComponent(value,component,weighted/weightSum);next|=1u<<component;}
  }
  value.w=next;extendVelocity[2u*node+avp.bankOffset]=value;extendStatusOut[node]=next;
}

// Permutation-invariant accumulation for at most six directional donors. Sort
// by magnitude before summing so reflected/rotated adjacency enumeration does
// not change a floating reduction. Tied opposite-sign magnitudes cancel
// exactly; tied same-sign terms are identical for addition purposes.
fn avCanonicalSum(values:ptr<function,array<f32,6>>,count:u32)->f32{
  for(var i=1u;i<count;i+=1u){let value=(*values)[i];let magnitude=abs(value);var j=i;while(j>0u&&abs((*values)[j-1u])>magnitude){(*values)[j]=(*values)[j-1u];j-=1u;}(*values)[j]=value;}
  var sum=0.;for(var i=0u;i<count;i+=1u){sum+=(*values)[i];}return sum;
}

// A causal |phi| sweep can strand a legitimate discrete local minimum whose
// incident face stencil is sparse. Close only those still-trial components by
// solving the compact directional graph Laplace relation outward from the
// causal accepted set. This is an elliptic graph extension, not a held, zero,
// or nearest-neighbor fallback: a component remains trial without an accepted
// adjacent Dirichlet value and finalization remains fail-closed.
@group(0)@binding(28)var<storage,read>closeSupport:array<u32>;
@compute @workgroup_size(64)fn closeAdaptiveVelocity(@builtin(global_invocation_id)id:vec3u,@builtin(num_workgroups)n:vec3u){
  let live=avLive(extendGraph[0],extendGraph[3],extendGraph[4],extendGraph[2]);let node=id.x+id.y*n.x*64u;if(node>=live){return;}let old=extendStatusIn[node];extendStatusOut[node]=old;
  let phiAt=avp.phiStride*node+avp.phiOffset;if(phiAt>=arrayLength(&extendPhi)||!avFinite(extendPhi[phiAt])){atomicOr(&extendReceipt[avp.receiptBase+2u],AV_ERROR_NONFINITE);return;}
  if((abs(extendPhi[phiAt])>avp.reach&&(node>=arrayLength(&closeSupport)||closeSupport[node]==0u))||old==AV_VALID_COMPONENTS){return;}
  var value=extendVelocity[2u*node+avp.bankOffset];var next=old;
  for(var component=0u;component<3u;component+=1u){if((old&(1u<<component))!=0u){continue;}var terms:array<f32,6>;var weights:array<f32,6>;var count=0u;
    for(var direction=0u;direction<6u;direction+=1u){let at=12u*node+direction;if(at+6u>=arrayLength(&extendAdjacency)){continue;}let neighborSlot=extendAdjacency[at];if(neighborSlot==AV_INVALID||neighborSlot>=live||neighborSlot>=arrayLength(&extendStatusIn)||(extendStatusIn[neighborSlot]&(1u<<component))==0u){continue;}let distance=bitcast<f32>(extendAdjacency[at+6u]);if(distance<=0.||!avFinite(distance)){continue;}let neighbor=avComponent(extendVelocity[2u*neighborSlot+avp.bankOffset],component);if(!avFinite(neighbor)){continue;}let weight=1./distance;terms[count]=weight*neighbor;weights[count]=weight;count+=1u;}
    if(count>0u){let numerator=avCanonicalSum(&terms,count);let denominator=avCanonicalSum(&weights,count);if(denominator>0.&&avFinite(numerator)&&avFinite(denominator)){value=avSetComponent(value,component,numerator/denominator);next|=1u<<component;}}
  }
  value.w=next;extendVelocity[2u*node+avp.bankOffset]=value;extendStatusOut[node]=next;
}

@group(0)@binding(81)var<storage,read>copySupportGraph:array<u32>;
@group(0)@binding(82)var<storage,read>copySupportIn:array<u32>;
@group(0)@binding(83)var<storage,read_write>copySupportOut:array<u32>;
@compute @workgroup_size(64)fn copyAdaptiveVelocitySupport(@builtin(global_invocation_id)id:vec3u,@builtin(num_workgroups)n:vec3u){let live=avLive(copySupportGraph[0],copySupportGraph[3],copySupportGraph[4],copySupportGraph[2]);let node=id.x+id.y*n.x*64u;if(node<live&&node<arrayLength(&copySupportIn)&&node<arrayLength(&copySupportOut)){copySupportOut[node]=copySupportIn[node];}}

@group(0)@binding(71)var<storage,read>dilateSupportGraph:array<u32>;
@group(0)@binding(72)var<storage,read>dilateSupportAdjacency:array<u32>;
@group(0)@binding(73)var<storage,read>dilateSupportConstraints:array<u32>;
@group(0)@binding(74)var<storage,read>dilateSupportStatus:array<u32>;
@group(0)@binding(75)var<storage,read>dilateSupportIn:array<u32>;
@group(0)@binding(76)var<storage,read_write>dilateSupportOut:array<atomic<u32>>;
@group(0)@binding(77)var<storage,read_write>dilateSupportDiagnostics:array<atomic<u32>>;
// Dilation is synchronous: copy preserves the old frontier, then this pass
// atomically unions exactly one directional-graph hop into the other bank.
// Only support-only independent nodes expand, and only independent nodes are
// admitted, matching the graph's validated non-nested master relation.
@compute @workgroup_size(64)fn dilateAdaptiveVelocitySupport(@builtin(global_invocation_id)id:vec3u,@builtin(num_workgroups)n:vec3u){
  let live=avLive(dilateSupportGraph[0],dilateSupportGraph[3],dilateSupportGraph[4],dilateSupportGraph[2]);let node=id.x+id.y*n.x*64u;let base=12u*node;if(node>=live||node>=arrayLength(&dilateSupportStatus)||node>=arrayLength(&dilateSupportIn)||base+11u>=arrayLength(&dilateSupportConstraints)){return;}
  if(dilateSupportIn[node]==0u||(dilateSupportStatus[node]&AV_VALID_COMPONENTS)==AV_VALID_COMPONENTS||dilateSupportConstraints[base+1u]!=0u){return;}let diagnosticBase=(avp.receiptBase/12u)*AV_DIAGNOSTIC_BANK_WORDS;
  for(var direction=0u;direction<6u;direction+=1u){let at=12u*node+direction;if(at>=arrayLength(&dilateSupportAdjacency)){continue;}let neighbor=dilateSupportAdjacency[at];let neighborBase=12u*neighbor;if(neighbor==AV_INVALID||neighbor>=live||neighbor>=arrayLength(&dilateSupportOut)||neighborBase+11u>=arrayLength(&dilateSupportConstraints)||dilateSupportConstraints[neighborBase+1u]!=0u){continue;}let old=atomicOr(&dilateSupportOut[neighbor],1u);if(old==0u){atomicAdd(&dilateSupportDiagnostics[diagnosticBase+8u],1u);}}
}

@group(0)@binding(31)var<storage,read>constraintGraph:array<u32>;
@group(0)@binding(32)var<storage,read>constraintHeaders:array<u32>;
@group(0)@binding(33)var<storage,read>constraintMasters:array<u32>;
@group(0)@binding(34)var<storage,read_write>constraintVelocity:array<vec4u>;
@group(0)@binding(35)var<storage,read_write>constraintStatus:array<u32>;
@group(0)@binding(36)var<storage,read_write>constraintReceipt:array<atomic<u32>>;
@compute @workgroup_size(64)fn constrainAdaptiveVelocity(@builtin(global_invocation_id)id:vec3u,@builtin(num_workgroups)n:vec3u){
  let live=avLive(constraintGraph[0],constraintGraph[3],constraintGraph[4],constraintGraph[2]);let node=id.x+id.y*n.x*64u;let base=12u*node;if(node>=live||base+11u>=arrayLength(&constraintHeaders)){return;}let count=constraintHeaders[base+1u];let denominator=constraintHeaders[base+2u];if(count==0u){return;}
  if((count!=2u&&count!=4u)||denominator==0u){atomicOr(&constraintReceipt[avp.receiptBase+2u],AV_ERROR_CONSTRAINT);constraintStatus[node]=0u;return;}
  var value=vec4u(0u);var mask=0u;var weightSum=0.;
  for(var i=0u;i<count;i+=1u){let master=constraintMasters[base+4u+i];if(master>=live){atomicOr(&constraintReceipt[avp.receiptBase+2u],AV_ERROR_CONSTRAINT);constraintStatus[node]=0u;return;}weightSum+=f32(constraintMasters[base+8u+i])/f32(denominator);}
  if(!avFinite(weightSum)||abs(weightSum-1.)>0.00001){atomicOr(&constraintReceipt[avp.receiptBase+2u],AV_ERROR_CONSTRAINT);constraintStatus[node]=0u;return;}
  for(var component=0u;component<3u;component+=1u){var sum=0.;var valid=true;for(var i=0u;i<count;i+=1u){let master=constraintMasters[base+4u+i];if((constraintStatus[master]&(1u<<component))==0u){valid=false;}sum+=(f32(constraintMasters[base+8u+i])/f32(denominator))*avComponent(constraintVelocity[2u*master+avp.bankOffset],component);}
    if(valid&&avFinite(sum)){value=avSetComponent(value,component,sum);mask|=1u<<component;}}
  value.w=mask;constraintVelocity[2u*node+avp.bankOffset]=value;constraintStatus[node]=mask;
}

@group(0)@binding(41)var<storage,read>finalGraph:array<u32>;
@group(0)@binding(42)var<storage,read>finalPhi:array<f32>;
@group(0)@binding(43)var<storage,read_write>finalVelocity:array<vec4u>;
@group(0)@binding(44)var<storage,read>finalStatus:array<u32>;
@group(0)@binding(45)var<storage,read_write>finalReceipt:array<atomic<u32>>;
@group(0)@binding(46)var<storage,read_write>finalValidity:array<u32>;
@group(0)@binding(47)var<storage,read>finalSupport:array<u32>;
@group(0)@binding(48)var<storage,read>finalConstraints:array<u32>;
@group(0)@binding(49)var<storage,read>finalAdjacency:array<u32>;
@group(0)@binding(50)var<storage,read_write>finalDiagnostics:array<atomic<u32>>;
@compute @workgroup_size(64)fn finalizeAdaptiveVelocity(@builtin(global_invocation_id)id:vec3u,@builtin(num_workgroups)n:vec3u){
  let node=id.x+id.y*n.x*64u;if(node>=avLive(finalGraph[0],finalGraph[3],finalGraph[4],finalGraph[2])){return;}let mask=finalStatus[node];var v=finalVelocity[2u*node+avp.bankOffset];v.w=mask;finalVelocity[2u*node+avp.bankOffset]=v;let shift=4u*avp.bankOffset;finalValidity[node]=(finalValidity[node]&~(7u<<shift))|(mask<<shift);
  let phiAt=avp.phiStride*node+avp.phiOffset;if(phiAt>=arrayLength(&finalPhi)||!avFinite(finalPhi[phiAt])){atomicOr(&finalReceipt[avp.receiptBase+2u],AV_ERROR_NONFINITE);return;}
  let magnitude=abs(finalPhi[phiAt]);let demanded=node<arrayLength(&finalSupport)&&finalSupport[node]!=0u;if(!demanded&&mask!=AV_VALID_COMPONENTS){let diagnosticBase=(avp.receiptBase/12u)*AV_DIAGNOSTIC_BANK_WORDS;atomicAdd(&finalDiagnostics[diagnosticBase+9u],1u);atomicMin(&finalDiagnostics[diagnosticBase+10u],bitcast<u32>(magnitude));atomicMax(&finalDiagnostics[diagnosticBase+11u],bitcast<u32>(magnitude));let constraintBase=12u*node;if(constraintBase+11u<arrayLength(&finalConstraints)&&finalConstraints[constraintBase+1u]==0u){atomicAdd(&finalDiagnostics[diagnosticBase+12u],1u);}}
  if(demanded&&mask!=AV_VALID_COMPONENTS){atomicAdd(&finalReceipt[avp.receiptBase+5u],1u);atomicOr(&finalReceipt[avp.receiptBase+2u],AV_ERROR_UNRESOLVED);
    let missing=AV_VALID_COMPONENTS&~mask;var validAdjacency=0u;var nonFarAdjacency=0u;var acceptedMissing=0u;var neighborSlots:array<u32,6>;var neighborPhiBits:array<u32,6>;var neighborMasks:array<u32,6>;
    for(var direction=0u;direction<6u;direction+=1u){let at=12u*node+direction;var slot=AV_INVALID;var bits=AV_INVALID;var neighborMask=0u;if(at+6u<arrayLength(&finalAdjacency)){slot=finalAdjacency[at];if(slot!=AV_INVALID&&slot<finalGraph[2u]){validAdjacency+=1u;let neighborPhiAt=avp.phiStride*slot+avp.phiOffset;if(neighborPhiAt<arrayLength(&finalPhi)){let neighborPhi=finalPhi[neighborPhiAt];bits=bitcast<u32>(neighborPhi);if(avFinite(neighborPhi)&&abs(neighborPhi)<=abs(finalPhi[phiAt])){nonFarAdjacency+=1u;}}if(slot<arrayLength(&finalStatus)){neighborMask=finalStatus[slot];if((neighborMask&missing)!=0u){acceptedMissing+=1u;}}}}neighborSlots[direction]=slot;neighborPhiBits[direction]=bits;neighborMasks[direction]=neighborMask;}
    let constraintBase=12u*node;let constraintCount=select(AV_INVALID,finalConstraints[constraintBase+1u],constraintBase+11u<arrayLength(&finalConstraints));var badMasters=0u;if(constraintCount!=0u&&constraintCount!=AV_INVALID){if(constraintCount!=2u&&constraintCount!=4u){badMasters=1u;}else{for(var masterIndex=0u;masterIndex<constraintCount;masterIndex+=1u){let master=finalConstraints[constraintBase+4u+masterIndex];if(master>=finalGraph[2u]||master>=arrayLength(&finalStatus)||(finalStatus[master]&missing)!=missing){badMasters+=1u;}}}}
    var cause=0u;if(validAdjacency==0u){cause|=1u;}if(validAdjacency!=0u&&nonFarAdjacency==0u){cause|=2u;}if(nonFarAdjacency!=0u&&acceptedMissing==0u){cause|=4u;}if(constraintCount!=0u&&constraintCount!=AV_INVALID){cause|=8u;}if(badMasters!=0u){cause|=16u;}
    let diagnosticBase=(avp.receiptBase/12u)*AV_DIAGNOSTIC_BANK_WORDS;let record=atomicAdd(&finalDiagnostics[diagnosticBase],1u);if((cause&1u)!=0u){atomicAdd(&finalDiagnostics[diagnosticBase+1u],1u);}if((cause&2u)!=0u){atomicAdd(&finalDiagnostics[diagnosticBase+2u],1u);}if((cause&4u)!=0u){atomicAdd(&finalDiagnostics[diagnosticBase+3u],1u);}if((cause&8u)!=0u){atomicAdd(&finalDiagnostics[diagnosticBase+4u],1u);}if((cause&16u)!=0u){atomicAdd(&finalDiagnostics[diagnosticBase+5u],1u);}
    if(record<AV_DIAGNOSTIC_RECORDS){atomicMax(&finalDiagnostics[diagnosticBase+6u],record+1u);let out=diagnosticBase+AV_DIAGNOSTIC_HEADER+record*AV_DIAGNOSTIC_RECORD_WORDS;atomicStore(&finalDiagnostics[out],node);atomicStore(&finalDiagnostics[out+1u],AV_INVALID);atomicStore(&finalDiagnostics[out+2u],missing);atomicStore(&finalDiagnostics[out+3u],constraintCount);atomicStore(&finalDiagnostics[out+4u],bitcast<u32>(finalPhi[phiAt]));atomicStore(&finalDiagnostics[out+5u],cause);for(var direction=0u;direction<6u;direction+=1u){atomicStore(&finalDiagnostics[out+6u+direction],neighborSlots[direction]);atomicStore(&finalDiagnostics[out+12u+direction],neighborPhiBits[direction]);atomicStore(&finalDiagnostics[out+18u+direction],neighborMasks[direction]);}}
  }
  if(mask==AV_VALID_COMPONENTS){atomicAdd(&finalReceipt[avp.receiptBase+6u],1u);}
}
@group(0)@binding(51)var<storage,read_write>finishReceipt:array<atomic<u32>>;
@group(0)@binding(52)var<storage,read_write>finishGraph:array<atomic<u32>>;
@group(0)@binding(53)var<storage,read_write>finishLeafLocator:array<atomic<u32>>;
@compute @workgroup_size(1)fn finishAdaptiveVelocity(){let errors=atomicLoad(&finishReceipt[avp.receiptBase+2u]);let ready=select(0u,1u,errors==0u);atomicStore(&finishReceipt[avp.receiptBase+7u],ready);if(avp.bankOffset==1u){let prior=avp.receiptBase-12u;let generation=atomicLoad(&finishGraph[5u]);let both=generation!=0u&&ready==1u&&atomicLoad(&finishReceipt[prior+7u])==1u&&atomicLoad(&finishReceipt[prior])==generation&&atomicLoad(&finishReceipt[avp.receiptBase])==generation;let published=select(0u,generation,both);atomicStore(&finishGraph[6u],published);atomicStore(&finishLeafLocator[13u],select(0u,generation,both));atomicStore(&finishLeafLocator[14u],published);}}

`;

/**
 * Four-binding containing-leaf sampler for LosassoSurfaceGraphBankSource.
 * It deliberately uses the caller's dynamics parameters for dimensions,
 * domain origin and finest physical cell widths.
 */
export function octreeLosassoAdaptiveVelocitySamplerWGSL(bindings: {
  readonly leaves: number; readonly ownerArena: number; readonly leafLocator: number;
  readonly velocityArena: number;
} = { leaves: 12, ownerArena: 13, leafLocator: 14, velocityArena: 15 }): string {
  return /* wgsl */ `
const ADAPTIVE_VELOCITY_INVALID=0xffffffffu;
struct AdaptiveVelocitySample{value:vec3f,lower:vec3f,upper:vec3f,valid:bool}
@group(0)@binding(${bindings.leaves})var<storage,read>adaptiveVelocityLeaves:array<vec4u>;
@group(0)@binding(${bindings.ownerArena})var<storage,read>adaptiveVelocityOwnerArena:array<u32>;
@group(0)@binding(${bindings.leafLocator})var<storage,read>adaptiveVelocityLeafLocator:array<u32>;
@group(0)@binding(${bindings.velocityArena})var<storage,read>adaptiveVelocityArena:array<vec4u>;
// Fixed two-level lookup: the owner authority supplies a direct logical-brick
// directory, and the graph transaction supplies one leaf slot per resident
// physical-page cell. No recurring search or topology-dependent loop exists.
fn adaptiveLocatedLeaf(grid:vec3f,dims:vec3u)->u32{if(arrayLength(&adaptiveVelocityLeafLocator)<16u||arrayLength(&adaptiveVelocityOwnerArena)<11u){return ADAPTIVE_VELOCITY_INVALID;}let epoch=adaptiveVelocityLeafLocator[1u];let pages=adaptiveVelocityLeafLocator[2u];let support=adaptiveVelocityLeafLocator[3u];let leaves=adaptiveVelocityLeafLocator[4u];let nodes=adaptiveVelocityLeafLocator[5u];let capacity=adaptiveVelocityLeafLocator[8u];let logicalCount=adaptiveVelocityLeafLocator[9u];let directoryA=adaptiveVelocityLeafLocator[10u];let payload=adaptiveVelocityLeafLocator[11u];let brickDims=(dims+vec3u(7u))/8u;if(adaptiveVelocityLeafLocator[0u]!=0x4c4f4341u||epoch==0u||adaptiveVelocityLeafLocator[6u]!=epoch||adaptiveVelocityLeafLocator[7u]!=0u||pages>capacity||support!=pages*512u||leaves==0u||4u*leaves>arrayLength(&adaptiveVelocityLeaves)||nodes==0u||2u*nodes>arrayLength(&adaptiveVelocityArena)||logicalCount!=brickDims.x*brickDims.y*brickDims.z||payload<16u||payload+capacity*512u>arrayLength(&adaptiveVelocityLeafLocator)||adaptiveVelocityLeafLocator[13u]==0u||adaptiveVelocityLeafLocator[14u]!=adaptiveVelocityLeafLocator[13u]||adaptiveVelocityOwnerArena[7u]!=epoch){return ADAPTIVE_VELOCITY_INVALID;}let cell=vec3u(min(floor(grid),vec3f(dims)-vec3f(1.)));let brick=cell/8u;let logical=brick.x+brickDims.x*(brick.y+brickDims.y*brick.z);if(logical>=logicalCount){return ADAPTIVE_VELOCITY_INVALID;}let table=adaptiveVelocityOwnerArena[10u]>>31u;let directory=directoryA+table*logicalCount+logical;if(directory>=arrayLength(&adaptiveVelocityOwnerArena)){return ADAPTIVE_VELOCITY_INVALID;}let page=adaptiveVelocityOwnerArena[directory];if(page==0u||page==ADAPTIVE_VELOCITY_INVALID||page>capacity){return ADAPTIVE_VELOCITY_INVALID;}let local=(cell.x&7u)+8u*(cell.y&7u)+64u*(cell.z&7u);let at=payload+(page-1u)*512u+local;if(at>=arrayLength(&adaptiveVelocityLeafLocator)){return ADAPTIVE_VELOCITY_INVALID;}let slot=adaptiveVelocityLeafLocator[at];if(slot>=leaves){return ADAPTIVE_VELOCITY_INVALID;}let record=4u*slot;if(record+3u>=arrayLength(&adaptiveVelocityLeaves)){return ADAPTIVE_VELOCITY_INVALID;}let header=adaptiveVelocityLeaves[record];let info=adaptiveVelocityLeaves[record+1u];if((info.x&1u)==0u||info.y!=slot||header.w==0u||any(cell<header.xyz)||any(cell>=header.xyz+vec3u(header.w))){return ADAPTIVE_VELOCITY_INVALID;}return slot;}
fn adaptiveSampleProduct3(value:vec3f)->f32{let lowPair=min(value.x,value.y);let highPair=max(value.x,value.y);let low=min(lowPair,value.z);let high=max(highPair,value.z);let middle=max(lowPair,min(highPair,value.z));return(low*middle)*high;}
fn adaptiveSampleFloorDiv256(value:i32)->vec2i{let carry=value>>8;return vec2i(carry,value-carry*256);}
fn adaptiveSampleExactAdd(limbs:ptr<function,array<i32,36>>,value:f32){let bits=bitcast<u32>(value);let magnitude=bits&0x7fffffffu;if(magnitude==0u){return;}let exponent=(magnitude>>23u)&0xffu;let fraction=magnitude&0x7fffffu;let significand=select(fraction,0x800000u|fraction,exponent!=0u);let shift=select(3u,exponent+2u,exponent!=0u);let first=shift>>3u;let shifted=significand<<(shift&7u);let sign=select(1,-1,(bits&0x80000000u)!=0u);for(var digit=0u;digit<4u;digit+=1u){let limb=first+digit;let byte=i32((shifted>>(digit*8u))&0xffu);if(byte!=0&&limb<36u){(*limbs)[limb]+=sign*byte;}}}
fn adaptiveSampleExactValue(input:ptr<function,array<i32,36>>)->f32{for(var limb=0u;limb+1u<36u;limb+=1u){let n=adaptiveSampleFloorDiv256((*input)[limb]);(*input)[limb]=n.y;(*input)[limb+1u]+=n.x;}let negative=(*input)[35u]<0;if(negative){for(var limb=0u;limb<36u;limb+=1u){(*input)[limb]=-(*input)[limb];}for(var limb=0u;limb+1u<36u;limb+=1u){let n=adaptiveSampleFloorDiv256((*input)[limb]);(*input)[limb]=n.y;(*input)[limb+1u]+=n.x;}}var magnitude=0.;for(var limb=0u;limb<36u;limb+=1u){magnitude+=ldexp(f32((*input)[limb]),-152+i32(8u*limb));}return select(magnitude,-magnitude,negative);}
fn adaptiveAccumulateSample(slot:u32,weight:f32,fieldBank:u32,nodeCapacity:u32,x:ptr<function,array<i32,36>>,y:ptr<function,array<i32,36>>,z:ptr<function,array<i32,36>>,lower:ptr<function,vec3f>,upper:ptr<function,vec3f>,valid:ptr<function,bool>){if(weight==0.){return;}let record=2u*slot+fieldBank;if(slot>=nodeCapacity||fieldBank>1u||record>=arrayLength(&adaptiveVelocityArena)){(*valid)=false;return;}let packed=adaptiveVelocityArena[record];let value=vec3f(bitcast<f32>(packed.x),bitcast<f32>(packed.y),bitcast<f32>(packed.z));if((packed.w&7u)!=7u||any(value!=value)){(*valid)=false;return;}adaptiveSampleExactAdd(x,weight*value.x);adaptiveSampleExactAdd(y,weight*value.y);adaptiveSampleExactAdd(z,weight*value.z);(*lower)=min((*lower),value);(*upper)=max((*upper),value);}
fn sampleAdaptiveVelocityGrid(gridValue:vec3f,fieldBank:u32)->AdaptiveVelocitySample{let dims=params.dimensionsMaximumLeaf.xyz;let grid=clamp(gridValue,vec3f(0.),vec3f(dims));let leaf=adaptiveLocatedLeaf(grid,dims);if(leaf==ADAPTIVE_VELOCITY_INVALID){return AdaptiveVelocitySample(vec3f(0.),vec3f(0.),vec3f(0.),false);}let at=4u*leaf;let header=adaptiveVelocityLeaves[at];let fraction=clamp((grid-vec3f(header.xyz))/f32(header.w),vec3f(0.),vec3f(1.));let lo=vec3f(1.)-fraction;let nodeCapacity=adaptiveVelocityLeafLocator[5u];let slots0=adaptiveVelocityLeaves[at+2u];let slots1=adaptiveVelocityLeaves[at+3u];var x:array<i32,36>;var y:array<i32,36>;var z:array<i32,36>;var lower=vec3f(3.402823e38);var upper=vec3f(-3.402823e38);var valid=true;
  adaptiveAccumulateSample(slots0.x,adaptiveSampleProduct3(vec3f(lo.x,lo.y,lo.z)),fieldBank,nodeCapacity,&x,&y,&z,&lower,&upper,&valid);adaptiveAccumulateSample(slots0.y,adaptiveSampleProduct3(vec3f(fraction.x,lo.y,lo.z)),fieldBank,nodeCapacity,&x,&y,&z,&lower,&upper,&valid);adaptiveAccumulateSample(slots0.z,adaptiveSampleProduct3(vec3f(lo.x,fraction.y,lo.z)),fieldBank,nodeCapacity,&x,&y,&z,&lower,&upper,&valid);adaptiveAccumulateSample(slots0.w,adaptiveSampleProduct3(vec3f(fraction.x,fraction.y,lo.z)),fieldBank,nodeCapacity,&x,&y,&z,&lower,&upper,&valid);adaptiveAccumulateSample(slots1.x,adaptiveSampleProduct3(vec3f(lo.x,lo.y,fraction.z)),fieldBank,nodeCapacity,&x,&y,&z,&lower,&upper,&valid);adaptiveAccumulateSample(slots1.y,adaptiveSampleProduct3(vec3f(fraction.x,lo.y,fraction.z)),fieldBank,nodeCapacity,&x,&y,&z,&lower,&upper,&valid);adaptiveAccumulateSample(slots1.z,adaptiveSampleProduct3(vec3f(lo.x,fraction.y,fraction.z)),fieldBank,nodeCapacity,&x,&y,&z,&lower,&upper,&valid);adaptiveAccumulateSample(slots1.w,adaptiveSampleProduct3(vec3f(fraction.x,fraction.y,fraction.z)),fieldBank,nodeCapacity,&x,&y,&z,&lower,&upper,&valid);
  if(!valid){return AdaptiveVelocitySample(vec3f(0.),vec3f(0.),vec3f(0.),false);}return AdaptiveVelocitySample(vec3f(adaptiveSampleExactValue(&x),adaptiveSampleExactValue(&y),adaptiveSampleExactValue(&z)),lower,upper,true);}
fn sampleAdaptiveVelocity(position:vec3f,fieldBank:u32)->AdaptiveVelocitySample{return sampleAdaptiveVelocityGrid((position-params.domainOriginDt.xyz)/params.cellWidthDensity.xyz,fieldBank);}
`;
}
