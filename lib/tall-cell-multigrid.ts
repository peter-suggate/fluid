import type { TallCellLayout } from "./tall-cell-grid";

const multigridShader = /* wgsl */ `
struct LevelParams { dims:vec4u, settings:vec4u, cell:vec4f }
@group(0) @binding(0) var pressureIn:texture_3d<f32>;
@group(0) @binding(1) var pressureOut:texture_storage_3d<r32float,write>;
@group(0) @binding(2) var rhsIn:texture_3d<f32>;
@group(0) @binding(3) var phiIn:texture_3d<f32>;
@group(0) @binding(4) var baseIn:texture_2d<f32>;
@group(0) @binding(5) var<uniform> level:LevelParams;
@group(0) @binding(6) var sourcePressure:texture_3d<f32>;
@group(0) @binding(7) var sourceRhs:texture_3d<f32>;
@group(0) @binding(8) var sourcePhi:texture_3d<f32>;
@group(0) @binding(9) var sourceBase:texture_2d<f32>;
@group(0) @binding(10) var<uniform> sourceLevel:LevelParams;
@group(0) @binding(11) var rhsOut:texture_storage_3d<r32float,write>;
@group(0) @binding(12) var phiOut:texture_storage_3d<r32float,write>;
@group(0) @binding(13) var baseOut:texture_storage_2d<r32float,write>;
@group(0) @binding(14) var solidIn:texture_3d<f32>;
@group(0) @binding(15) var sourceSolid:texture_3d<f32>;
@group(0) @binding(16) var<storage,read_write> diagnostics:array<atomic<u32>>;
// Stencil coefficients are geometry (phi, solid, base) which is frozen for a
// whole solve, so they are baked once per level per solve instead of being
// reassembled from ~40 texture loads on every relaxation.  coeffLateral holds
// the x-,x+,z-,z+ pressure coefficients; coeffVertical holds (down, up,
// diagonal, solvable-flag).  Air neighbours contribute to the diagonal only,
// so their pressure coefficient is stored as zero.
@group(0) @binding(17) var coeffLateralIn:texture_3d<f32>;
@group(0) @binding(18) var coeffVerticalIn:texture_3d<f32>;
@group(0) @binding(19) var coeffLateralOut:texture_storage_3d<rgba32float,write>;
@group(0) @binding(20) var coeffVerticalOut:texture_storage_3d<rgba32float,write>;

fn dims()->vec3i{return vec3i(level.dims.xyz);}fn fineY()->i32{return i32(level.dims.w);}
fn validPacked(id:vec3i)->bool{return all(id>=vec3i(0))&&all(id<dims());}
fn baseAt(x:i32,z:i32)->i32{if(x<0||x>=dims().x||z<0||z>=dims().z){return 0;}return i32(round(textureLoad(baseIn,vec2i(x,z),0).x));}
fn activeSample(id:vec3i)->bool{if(!validPacked(id)){return false;}let b=baseAt(id.x,id.z);return select(b>0,b+id.y-2<fineY(),id.y>=2);}
fn sampleY(id:vec3i)->f32{let b=baseAt(id.x,id.z);if(id.y==0){return 0.5;}if(id.y==1){return max(0.5,f32(b)-0.5);}return f32(b+id.y-2)+0.5;}
fn phiCell(q:vec3i)->f32{let air=max(level.cell.x,max(level.cell.y,level.cell.z));if(any(q<vec3i(0))||q.x>=dims().x||q.y>=fineY()||q.z>=dims().z){return air;}let b=baseAt(q.x,q.z);if(q.y<b&&b>0){let t=clamp(f32(q.y)/f32(max(b-1,1)),0.0,1.0);return mix(textureLoad(phiIn,vec3i(q.x,0,q.z),0).x,textureLoad(phiIn,vec3i(q.x,1,q.z),0).x,t);}let py=2+q.y-b;if(py<2||py>=dims().y){return air;}return textureLoad(phiIn,vec3i(q.x,py,q.z),0).x;}
fn pressureCell(q:vec3i)->f32{if(any(q<vec3i(0))||q.x>=dims().x||q.y>=fineY()||q.z>=dims().z){return 0.0;}let b=baseAt(q.x,q.z);if(q.y<b&&b>0){let t=clamp(f32(q.y)/f32(max(b-1,1)),0.0,1.0);return mix(textureLoad(pressureIn,vec3i(q.x,0,q.z),0).x,textureLoad(pressureIn,vec3i(q.x,1,q.z),0).x,t);}let py=2+q.y-b;if(py<2||py>=dims().y){return 0.0;}return textureLoad(pressureIn,vec3i(q.x,py,q.z),0).x;}
fn solidCell(q:vec3i)->f32{if(any(q<vec3i(0))||q.x>=dims().x||q.y>=fineY()||q.z>=dims().z){return 1.0;}let b=baseAt(q.x,q.z);if(q.y<b&&b>0){let t=clamp(f32(q.y)/f32(max(b-1,1)),0.0,1.0);return mix(textureLoad(solidIn,vec3i(q.x,0,q.z),0).x,textureLoad(solidIn,vec3i(q.x,1,q.z),0).x,t);}let py=2+q.y-b;if(py<2||py>=dims().y){return 1.0;}return textureLoad(solidIn,vec3i(q.x,py,q.z),0).x;}
fn ghostFraction(ownPhi:f32,otherPhi:f32)->f32{return clamp(abs(ownPhi)/max(abs(ownPhi)+abs(otherPhi),1e-6),0.05,1.0);}
// (pressure coefficient, diagonal coefficient) of one neighbour term.  Liquid
// neighbours couple with c*open on both; air neighbours are ghost-scaled and
// couple through the diagonal only.  Matches the historical term() exactly.
fn termCoefficients(ownPhi:f32,otherPhi:f32,otherSolid:f32,c:f32)->vec2f{if(otherPhi<=0.0){let open=1.0-clamp(otherSolid,0.0,1.0);return vec2f(c*open,c*open);}return vec2f(0.0,c/ghostFraction(ownPhi,otherPhi));}
@compute @workgroup_size(4,4,4) fn bakeCoefficients(@builtin(global_invocation_id) gid:vec3u){let id=vec3i(gid);if(!validPacked(id)){return;}
  if(!activeSample(id)||textureLoad(phiIn,id,0).x>0.0||textureLoad(solidIn,id,0).x>0.9){textureStore(coeffLateralOut,id,vec4f(0.0));textureStore(coeffVerticalOut,id,vec4f(0.0));return;}
  let ownPhi=textureLoad(phiIn,id,0).x;let h=level.cell.xyz;let base=baseAt(id.x,id.z);var lateral=vec4f(0.0);var diag=0.0;
  if(id.y==0&&base>0){
    // The store is one lateral degree of freedom: couple bottom texel to the
    // neighbouring stores' bottom texels (hydrostatically exact).
    if(id.x>0){let n=vec3i(id.x-1,0,id.z);let t=termCoefficients(ownPhi,textureLoad(phiIn,n,0).x,textureLoad(solidIn,n,0).x,1.0/(h.x*h.x));lateral.x=t.x;diag+=t.y;}
    if(id.x+1<dims().x){let n=vec3i(id.x+1,0,id.z);let t=termCoefficients(ownPhi,textureLoad(phiIn,n,0).x,textureLoad(solidIn,n,0).x,1.0/(h.x*h.x));lateral.y=t.x;diag+=t.y;}
    if(id.z>0){let n=vec3i(id.x,0,id.z-1);let t=termCoefficients(ownPhi,textureLoad(phiIn,n,0).x,textureLoad(solidIn,n,0).x,1.0/(h.z*h.z));lateral.z=t.x;diag+=t.y;}
    if(id.z+1<dims().z){let n=vec3i(id.x,0,id.z+1);let t=termCoefficients(ownPhi,textureLoad(phiIn,n,0).x,textureLoad(solidIn,n,0).x,1.0/(h.z*h.z));lateral.w=t.x;diag+=t.y;}
  } else {
    let q=vec3i(id.x,i32(floor(sampleY(id))),id.z);
    if(q.x>0){let n=q-vec3i(1,0,0);let t=termCoefficients(ownPhi,phiCell(n),solidCell(n),1.0/(h.x*h.x));lateral.x=t.x;diag+=t.y;}
    if(q.x+1<dims().x){let n=q+vec3i(1,0,0);let t=termCoefficients(ownPhi,phiCell(n),solidCell(n),1.0/(h.x*h.x));lateral.y=t.x;diag+=t.y;}
    if(q.z>0){let n=q-vec3i(0,0,1);let t=termCoefficients(ownPhi,phiCell(n),solidCell(n),1.0/(h.z*h.z));lateral.z=t.x;diag+=t.y;}
    if(q.z+1<dims().z){let n=q+vec3i(0,0,1);let t=termCoefficients(ownPhi,phiCell(n),solidCell(n),1.0/(h.z*h.z));lateral.w=t.x;diag+=t.y;}
  }
  // Paper Eq 15/16 vertical couplings: solid below the bottom endpoint
  // (Neumann, dropped), Eq 5 interpolated pressure across the tall span at
  // 1/(distance*h), ordinary 1/(h*h) inside the regular band, and a ghost air
  // term above the highest active sample.
  var vertical=vec2f(0.0);
  if(id.y==0){let n=vec3i(id.x,1,id.z);let distance=max(h.y,f32(base-1)*h.y);let t=termCoefficients(ownPhi,textureLoad(phiIn,n,0).x,textureLoad(solidIn,n,0).x,1.0/(distance*h.y));vertical.y=t.x;diag+=t.y;}
  else if(id.y==1){let distance=max(h.y,f32(base-1)*h.y);let composed=base<=i32(level.settings.x);let bottomCoefficient=select(1.0/(distance*h.y),1.0/(distance*distance),composed);let bandCoefficient=select(1.0/(h.y*h.y),1.0/(distance*h.y),composed);let n0=vec3i(id.x,0,id.z);let t0=termCoefficients(ownPhi,textureLoad(phiIn,n0,0).x,textureLoad(solidIn,n0,0).x,bottomCoefficient);vertical.x=t0.x;diag+=t0.y;if(activeSample(vec3i(id.x,2,id.z))){let n=vec3i(id.x,2,id.z);let t=termCoefficients(ownPhi,textureLoad(phiIn,n,0).x,textureLoad(solidIn,n,0).x,bandCoefficient);vertical.y=t.x;diag+=t.y;}}
  else{let q=vec3i(id.x,i32(floor(sampleY(id))),id.z);if(id.y>2||base>=2){let n=id-vec3i(0,1,0);let t=termCoefficients(ownPhi,textureLoad(phiIn,n,0).x,textureLoad(solidIn,n,0).x,1.0/(h.y*h.y));vertical.x=t.x;diag+=t.y;}if(activeSample(id+vec3i(0,1,0))){let n=id+vec3i(0,1,0);let t=termCoefficients(ownPhi,textureLoad(phiIn,n,0).x,textureLoad(solidIn,n,0).x,1.0/(h.y*h.y));vertical.y=t.x;diag+=t.y;}else if(q.y+1>=fineY()){let t=termCoefficients(ownPhi,h.y,1.0,1.0/(h.y*h.y));diag+=t.y;}}
  textureStore(coeffLateralOut,id,lateral);textureStore(coeffVerticalOut,id,vec4f(vertical.x,vertical.y,diag,1.0));}
// Off-diagonal contribution sum using baked coefficients.  Lateral neighbour
// pressures keep the exact historical access pattern: direct bottom texels
// for a tall store, Eq 5 interpolated world samples otherwise.
fn neighborSum(id:vec3i,lateral:vec4f,vertical:vec4f)->f32{
  var sum=0.0;let base=baseAt(id.x,id.z);
  if(id.y==0&&base>0){
    if(lateral.x!=0.0){sum+=lateral.x*textureLoad(pressureIn,vec3i(id.x-1,0,id.z),0).x;}
    if(lateral.y!=0.0){sum+=lateral.y*textureLoad(pressureIn,vec3i(id.x+1,0,id.z),0).x;}
    if(lateral.z!=0.0){sum+=lateral.z*textureLoad(pressureIn,vec3i(id.x,0,id.z-1),0).x;}
    if(lateral.w!=0.0){sum+=lateral.w*textureLoad(pressureIn,vec3i(id.x,0,id.z+1),0).x;}
  } else {
    let q=vec3i(id.x,i32(floor(sampleY(id))),id.z);
    if(lateral.x!=0.0){sum+=lateral.x*pressureCell(q-vec3i(1,0,0));}
    if(lateral.y!=0.0){sum+=lateral.y*pressureCell(q+vec3i(1,0,0));}
    if(lateral.z!=0.0){sum+=lateral.z*pressureCell(q-vec3i(0,0,1));}
    if(lateral.w!=0.0){sum+=lateral.w*pressureCell(q+vec3i(0,0,1));}
  }
  if(vertical.x!=0.0){let n=select(id-vec3i(0,1,0),vec3i(id.x,0,id.z),id.y==1);sum+=vertical.x*textureLoad(pressureIn,n,0).x;}
  if(vertical.y!=0.0){let n=select(id+vec3i(0,1,0),vec3i(id.x,1,id.z),id.y==0);sum+=vertical.y*textureLoad(pressureIn,n,0).x;}
  return sum;}
@compute @workgroup_size(4,4,4) fn clearPressure(@builtin(global_invocation_id) gid:vec3u){let id=vec3i(gid);if(validPacked(id)){textureStore(pressureOut,id,vec4f(0.0));}}
fn smoothColor(id:vec3i,color:i32){if(!validPacked(id)){return;}
  let vertical=textureLoad(coeffVerticalIn,id,0);
  if(vertical.w==0.0){textureStore(pressureOut,id,vec4f(0.0));return;}
  let old=textureLoad(pressureIn,id,0).x;
  if(((id.x+id.y+id.z)&1)!=color){textureStore(pressureOut,id,vec4f(old));return;}
  let candidate=(neighborSum(id,textureLoad(coeffLateralIn,id,0),vertical)-textureLoad(rhsIn,id,0).x)/max(vertical.z,1e-9);
  textureStore(pressureOut,id,vec4f(mix(old,candidate,0.8),0.0,0.0,0.0));}
@compute @workgroup_size(4,4,4) fn smoothRed(@builtin(global_invocation_id) gid:vec3u){smoothColor(vec3i(gid),0);}
@compute @workgroup_size(4,4,4) fn smoothBlack(@builtin(global_invocation_id) gid:vec3u){smoothColor(vec3i(gid),1);}
// The residual is materialised once into the idle ping-pong texture and the
// restriction gathers it trilinearly, instead of re-deriving the full stencil
// for all eight gather corners of every coarse cell.
@compute @workgroup_size(4,4,4) fn computeResidual(@builtin(global_invocation_id) gid:vec3u){let id=vec3i(gid);if(!validPacked(id)){return;}
  let vertical=textureLoad(coeffVerticalIn,id,0);
  if(vertical.w==0.0){textureStore(pressureOut,id,vec4f(0.0));return;}
  let value=textureLoad(rhsIn,id,0).x-(neighborSum(id,textureLoad(coeffLateralIn,id,0),vertical)-vertical.z*textureLoad(pressureIn,id,0).x);
  textureStore(pressureOut,id,vec4f(value,0.0,0.0,0.0));}

// One persistent workgroup solves the coarsest (<=256 sample) level with all
// state resident in workgroup memory; the barriered iterations never touch
// textures.
var<workgroup> topPressure:array<f32,256>;
var<workgroup> topLateral:array<vec4f,256>;
var<workgroup> topVertical:array<vec4f,256>;
var<workgroup> topRhs:array<f32,256>;
var<workgroup> topColumnBase:array<i32,256>;
fn topIndex(id:vec3i)->u32{return u32(id.x+dims().x*(id.y+dims().y*id.z));}
fn topStored(id:vec3i)->f32{if(!validPacked(id)){return 0.0;}return topPressure[topIndex(id)];}
fn topBaseAt(x:i32,z:i32)->i32{if(x<0||x>=dims().x||z<0||z>=dims().z){return 0;}return topColumnBase[x+dims().x*z];}
fn topPressureCell(q:vec3i)->f32{if(any(q<vec3i(0))||q.x>=dims().x||q.y>=fineY()||q.z>=dims().z){return 0.0;}let b=topBaseAt(q.x,q.z);if(q.y<b&&b>0){let t=clamp(f32(q.y)/f32(max(b-1,1)),0.0,1.0);return mix(topStored(vec3i(q.x,0,q.z)),topStored(vec3i(q.x,1,q.z)),t);}let py=2+q.y-b;if(py<2||py>=dims().y){return 0.0;}return topStored(vec3i(q.x,py,q.z));}
fn topNeighborSum(id:vec3i,localIndex:u32)->f32{
  let lateral=topLateral[localIndex];let vertical=topVertical[localIndex];var sum=0.0;let base=topBaseAt(id.x,id.z);
  if(id.y==0&&base>0){
    if(lateral.x!=0.0){sum+=lateral.x*topStored(vec3i(id.x-1,0,id.z));}
    if(lateral.y!=0.0){sum+=lateral.y*topStored(vec3i(id.x+1,0,id.z));}
    if(lateral.z!=0.0){sum+=lateral.z*topStored(vec3i(id.x,0,id.z-1));}
    if(lateral.w!=0.0){sum+=lateral.w*topStored(vec3i(id.x,0,id.z+1));}
  } else {
    let q=vec3i(id.x,i32(floor(sampleY(id))),id.z);
    if(lateral.x!=0.0){sum+=lateral.x*topPressureCell(q-vec3i(1,0,0));}
    if(lateral.y!=0.0){sum+=lateral.y*topPressureCell(q+vec3i(1,0,0));}
    if(lateral.z!=0.0){sum+=lateral.z*topPressureCell(q-vec3i(0,0,1));}
    if(lateral.w!=0.0){sum+=lateral.w*topPressureCell(q+vec3i(0,0,1));}
  }
  if(vertical.x!=0.0){let n=select(id-vec3i(0,1,0),vec3i(id.x,0,id.z),id.y==1);sum+=vertical.x*topStored(n);}
  if(vertical.y!=0.0){let n=select(id+vec3i(0,1,0),vec3i(id.x,1,id.z),id.y==0);sum+=vertical.y*topStored(n);}
  return sum;}
@compute @workgroup_size(256) fn solveTop(@builtin(local_invocation_index) localIndex:u32){let count=u32(dims().x*dims().y*dims().z);
  if(localIndex<count){let x=i32(localIndex%u32(dims().x));let yz=i32(localIndex/u32(dims().x));let id=vec3i(x,yz%dims().y,yz/dims().y);
    topPressure[localIndex]=0.0;topLateral[localIndex]=textureLoad(coeffLateralIn,id,0);topVertical[localIndex]=textureLoad(coeffVerticalIn,id,0);topRhs[localIndex]=textureLoad(rhsIn,id,0).x;
    if(id.y==0){topColumnBase[id.x+dims().x*id.z]=baseAt(id.x,id.z);}}
  workgroupBarrier();
  for(var iteration=0u;iteration<level.settings.z;iteration+=1u){for(var color=0;color<2;color+=1){
    if(localIndex<count){let x=i32(localIndex%u32(dims().x));let yz=i32(localIndex/u32(dims().x));let id=vec3i(x,yz%dims().y,yz/dims().y);
      if(((id.x+id.y+id.z)&1)==color&&topVertical[localIndex].w!=0.0){
        let old=topPressure[localIndex];
        let candidate=(topNeighborSum(id,localIndex)-topRhs[localIndex])/max(topVertical[localIndex].z,1e-9);
        topPressure[localIndex]=mix(old,candidate,0.8);}}
    workgroupBarrier();}}
  if(localIndex<count){let x=i32(localIndex%u32(dims().x));let yz=i32(localIndex/u32(dims().x));let id=vec3i(x,yz%dims().y,yz/dims().y);
    textureStore(pressureOut,id,vec4f(select(0.0,topPressure[localIndex],topVertical[localIndex].w!=0.0)));}}

fn sourceDims()->vec3i{return vec3i(sourceLevel.dims.xyz);}fn sourceFineY()->i32{return i32(sourceLevel.dims.w);}
fn sourceBaseAt(x:i32,z:i32)->i32{if(x<0||x>=sourceDims().x||z<0||z>=sourceDims().z){return 0;}return i32(round(textureLoad(sourceBase,vec2i(x,z),0).x));}
fn sourcePhiCell(q:vec3i)->f32{let air=max(sourceLevel.cell.x,max(sourceLevel.cell.y,sourceLevel.cell.z));if(any(q<vec3i(0))||q.x>=sourceDims().x||q.y>=sourceFineY()||q.z>=sourceDims().z){return air;}let b=sourceBaseAt(q.x,q.z);if(q.y<b&&b>0){let t=clamp(f32(q.y)/f32(max(b-1,1)),0.0,1.0);return mix(textureLoad(sourcePhi,vec3i(q.x,0,q.z),0).x,textureLoad(sourcePhi,vec3i(q.x,1,q.z),0).x,t);}let py=2+q.y-b;if(py<2||py>=sourceDims().y){return air;}return textureLoad(sourcePhi,vec3i(q.x,py,q.z),0).x;}
fn sourcePressureCell(q:vec3i)->f32{if(any(q<vec3i(0))||q.x>=sourceDims().x||q.y>=sourceFineY()||q.z>=sourceDims().z){return 0.0;}let b=sourceBaseAt(q.x,q.z);if(q.y<b&&b>0){let t=clamp(f32(q.y)/f32(max(b-1,1)),0.0,1.0);return mix(textureLoad(sourcePressure,vec3i(q.x,0,q.z),0).x,textureLoad(sourcePressure,vec3i(q.x,1,q.z),0).x,t);}let py=2+q.y-b;if(py<2||py>=sourceDims().y){return 0.0;}return textureLoad(sourcePressure,vec3i(q.x,py,q.z),0).x;}
fn sourceSolidCell(q:vec3i)->f32{if(any(q<vec3i(0))||q.x>=sourceDims().x||q.y>=sourceFineY()||q.z>=sourceDims().z){return 1.0;}let b=sourceBaseAt(q.x,q.z);if(q.y<b&&b>0){let t=clamp(f32(q.y)/f32(max(b-1,1)),0.0,1.0);return mix(textureLoad(sourceSolid,vec3i(q.x,0,q.z),0).x,textureLoad(sourceSolid,vec3i(q.x,1,q.z),0).x,t);}let py=2+q.y-b;if(py<2||py>=sourceDims().y){return 1.0;}return textureLoad(sourceSolid,vec3i(q.x,py,q.z),0).x;}

@compute @workgroup_size(8,8,1) fn downsampleBase(@builtin(global_invocation_id) gid:vec3u){if(gid.x>=level.dims.x||gid.y>=level.dims.z){return;}let x=i32(gid.x);let z=i32(gid.y);var maximum=0;for(var dz=0;dz<2;dz+=1){for(var dx=0;dx<2;dx+=1){maximum=max(maximum,sourceBaseAt(2*x+dx,2*z+dz));}}let upper=max(0,fineY()-i32(level.settings.x));var b=clamp((maximum+1)/2,0,upper);
  // A base-1 store is degenerate: both endpoint dofs land on the SAME world
  // cell (sampleY 0.5 for each), duplicating unknowns across the level. For
  // shallow tanks Eq 9 halving produces base 1 over entire coarse levels and
  // the coarse correction re-injects a floor-row checkerboard every cycle
  // (2026-07-16 still-tank audit). Mirror the fine layout's rule: a column
  // owns a genuine (h >= 2) tall cell whenever the level can represent one.
  if(upper>=2){b=max(b,2);}
  textureStore(baseOut,vec2i(x,z),vec4f(f32(b)));}
// Phi and solid downsample share the same block sweep; fusing them halves the
// per-level preparation loads.
fn coarseBlockBoth(x:i32,y:i32,z:i32)->vec2f{var total=0.0;var count=0.0;var positive=0.0;var positiveCount=0.0;var negativeCount=0.0;var solidTotal=0.0;for(var dz=0;dz<2;dz+=1){for(var dx=0;dx<2;dx+=1){for(var dy=0;dy<2;dy+=1){let q=vec3i(2*x+dx,2*y+dy,2*z+dz);if(any(q<vec3i(0))||q.x>=sourceDims().x||q.y>=sourceFineY()||q.z>=sourceDims().z){continue;}let value=sourcePhiCell(q);total+=value;count+=1.0;if(value>0.0){positive+=value;positiveCount+=1.0;}else{negativeCount+=1.0;}solidTotal+=sourceSolidCell(q);}}}var phi=total/max(count,1.0);if(level.settings.y>0u&&positiveCount>0.0&&negativeCount>0.0){phi=positive/positiveCount;}return vec2f(phi,solidTotal/max(count,1.0));}
fn coarseTallFitBoth(x:i32,z:i32,b:i32)->vec4f{let air=max(level.cell.x,max(level.cell.y,level.cell.z));if(b<=0){return vec4f(air,air,0.0,0.0);}if(b==1){let both=coarseBlockBoth(x,0,z);return vec4f(both.x,both.x,clamp(both.y,0.0,1.0),clamp(both.y,0.0,1.0));}var sumT=0.0;var sumTT=0.0;var sumPhi=0.0;var sumTPhi=0.0;var sumSolid=0.0;var sumTSolid=0.0;for(var y=0;y<b;y+=1){let t=f32(y)/f32(b-1);let both=coarseBlockBoth(x,y,z);sumT+=t;sumTT+=t*t;sumPhi+=both.x;sumTPhi+=t*both.x;sumSolid+=both.y;sumTSolid+=t*both.y;}let n=f32(b);let denominator=max(n*sumTT-sumT*sumT,1e-6);let phiSlope=(n*sumTPhi-sumT*sumPhi)/denominator;let phiIntercept=(sumPhi-phiSlope*sumT)/n;let solidSlope=(n*sumTSolid-sumT*sumSolid)/denominator;let solidIntercept=(sumSolid-solidSlope*sumT)/n;let solid=clamp(vec2f(solidIntercept,solidIntercept+solidSlope),vec2f(0.0),vec2f(1.0));return vec4f(phiIntercept,phiIntercept+phiSlope,solid.x,solid.y);}
@compute @workgroup_size(4,4,4) fn downsampleColumns(@builtin(global_invocation_id) gid:vec3u){let id=vec3i(gid);if(!validPacked(id)){return;}if(!activeSample(id)){textureStore(phiOut,id,vec4f(max(level.cell.x,max(level.cell.y,level.cell.z))));textureStore(rhsOut,id,vec4f(0.0));return;}let b=baseAt(id.x,id.z);var phiValue=0.0;var solidValue=0.0;if(id.y<2){let fit=coarseTallFitBoth(id.x,id.z,b);phiValue=select(fit.x,fit.y,id.y==1);solidValue=select(fit.z,fit.w,id.y==1);}else{let both=coarseBlockBoth(id.x,b+id.y-2,id.z);phiValue=both.x;solidValue=both.y;}textureStore(phiOut,id,vec4f(phiValue));textureStore(rhsOut,id,vec4f(solidValue));}
// World-space view of the materialised source-level residual: only the tall
// endpoints carry residual inside a tall span, matching the historical
// per-corner stencil evaluation.
fn sourceResidualWorld(q:vec3i)->f32{if(any(q<vec3i(0))||q.x>=sourceDims().x||q.y>=sourceFineY()||q.z>=sourceDims().z){return 0.0;}let b=sourceBaseAt(q.x,q.z);if(q.y<b&&b>0){if(q.y==0){return textureLoad(sourcePressure,vec3i(q.x,0,q.z),0).x;}if(q.y==b-1){return textureLoad(sourcePressure,vec3i(q.x,1,q.z),0).x;}return 0.0;}let py=2+q.y-b;if(py<2||py>=sourceDims().y){return 0.0;}return textureLoad(sourcePressure,vec3i(q.x,py,q.z),0).x;}
fn sourceSampleResidual(p:vec3f)->f32{let q=p-vec3f(0.5);let b=vec3i(floor(q));let f=fract(q);var value=0.0;var weight=0.0;for(var corner=0u;corner<8u;corner+=1u){let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let sample=b+offset;if(any(sample<vec3i(0))||sample.x>=sourceDims().x||sample.y>=sourceFineY()||sample.z>=sourceDims().z){continue;}let w=select(1.0-f.x,f.x,offset.x==1)*select(1.0-f.y,f.y,offset.y==1)*select(1.0-f.z,f.z,offset.z==1);value+=w*sourceResidualWorld(sample);weight+=w;}return select(0.0,value/weight,weight>1e-8);}
@compute @workgroup_size(4,4,4) fn restrictResidual(@builtin(global_invocation_id) gid:vec3u){let id=vec3i(gid);if(!validPacked(id)){return;}if(textureLoad(coeffVerticalIn,id,0).w==0.0){textureStore(rhsOut,id,vec4f(0.0));return;}let p=vec3f((f32(id.x)+0.5)*2.0,sampleY(id)*2.0,(f32(id.z)+0.5)*2.0);textureStore(rhsOut,id,vec4f(sourceSampleResidual(p),0.0,0.0,0.0));}
fn sourceSamplePressure(p:vec3f)->f32{let q=p-vec3f(0.5);let b=vec3i(floor(q));let f=fract(q);var value=0.0;var weight=0.0;for(var corner=0u;corner<8u;corner+=1u){let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let sample=b+offset;if(any(sample<vec3i(0))||sample.x>=sourceDims().x||sample.y>=sourceFineY()||sample.z>=sourceDims().z){continue;}let w=select(1.0-f.x,f.x,offset.x==1)*select(1.0-f.y,f.y,offset.y==1)*select(1.0-f.z,f.z,offset.z==1);value+=w*sourcePressureCell(sample);weight+=w;}return select(0.0,value/weight,weight>1e-8);}
@compute @workgroup_size(4,4,4) fn prolongateCorrection(@builtin(global_invocation_id) gid:vec3u){let id=vec3i(gid);if(!validPacked(id)){return;}if(textureLoad(coeffVerticalIn,id,0).w==0.0){textureStore(pressureOut,id,vec4f(0.0));return;}let p=vec3f((f32(id.x)+0.5)*0.5,sampleY(id)*0.5,(f32(id.z)+0.5)*0.5);textureStore(pressureOut,id,vec4f(textureLoad(pressureIn,id,0).x+sourceSamplePressure(p),0.0,0.0,0.0));}
fn finiteScalar(value:f32)->bool{return value==value&&abs(value)<=3.402823e38;}
fn storeDiagnosticLocation(id:vec3i,slot:u32){atomicStore(&diagnostics[slot],u32(max(id.x,0)));atomicStore(&diagnostics[slot+1u],u32(max(i32(floor(sampleY(id))),0)));atomicStore(&diagnostics[slot+2u],u32(max(id.z,0)));}
fn updateDiagnosticMaximum(value:f32,valueSlot:u32,locationSlot:u32,id:vec3i){if(!finiteScalar(value)||value<0.0){atomicAdd(&diagnostics[20],1u);return;}let bits=bitcast<u32>(value);let previous=atomicMax(&diagnostics[valueSlot],bits);if(bits>previous){storeDiagnosticLocation(id,locationSlot);}}
@compute @workgroup_size(4,4,4) fn reducePressureDiagnostics(@builtin(global_invocation_id) gid:vec3u){let id=vec3i(gid);if(!validPacked(id)){return;}let vertical=textureLoad(coeffVerticalIn,id,0);if(vertical.w==0.0){return;}let pressure=abs(textureLoad(pressureIn,id,0).x);let rhs=abs(textureLoad(rhsIn,id,0).x);let residual=abs(textureLoad(rhsIn,id,0).x-(neighborSum(id,textureLoad(coeffLateralIn,id,0),vertical)-vertical.z*textureLoad(pressureIn,id,0).x));updateDiagnosticMaximum(pressure,21u,24u,id);updateDiagnosticMaximum(residual,22u,27u,id);if(finiteScalar(rhs)){atomicMax(&diagnostics[23],bitcast<u32>(rhs));}else{atomicAdd(&diagnostics[20],1u);}}
`;

/** Structural-conformance handle only; the runtime always compiles the
 * module through TallCellMultigrid. */
export const tallCellMultigridShaderForTests = multigridShader;

interface Level {
  nx:number;packedNy:number;nz:number;fineNy:number;regularLayers:number;
  cell:{x:number;y:number;z:number};pressure:[GPUTexture,GPUTexture];rhs:GPUTexture;phi:GPUTexture;solid:GPUTexture;base:GPUTexture;coeffLateral:GPUTexture;coeffVertical:GPUTexture;params:GPUBuffer;correctionParams:GPUBuffer;owned:boolean;
}

export interface TallCellMultigridFineResources { pressureA:GPUTexture;pressureB:GPUTexture;volume:GPUTexture;solid:GPUTexture;base:GPUTexture;diagnostics:GPUBuffer }
export interface TallCellMultigridEncodeOptions {
  warmStart?:boolean;
  topologyChanged?:boolean;
}

export class TallCellMultigrid {
  readonly fineRhs:GPUTexture;readonly allocatedBytes:number;
  private levels:Level[]=[];private layout:GPUBindGroupLayout;private bakeLayout:GPUBindGroupLayout;
  private diagnostics:GPUBuffer;private clearPipeline!:GPUComputePipeline;private smoothPipelines!:GPUComputePipeline[];private topPipeline!:GPUComputePipeline;private basePipeline!:GPUComputePipeline;private columnsPipeline!:GPUComputePipeline;private bakePipeline!:GPUComputePipeline;private residualComputePipeline!:GPUComputePipeline;private restrictPipeline!:GPUComputePipeline;private prolongatePipeline!:GPUComputePipeline;private residualPipeline!:GPUComputePipeline;
  private shaderModule:GPUShaderModule;private pipelineLayout:GPUPipelineLayout;private bakePipelineLayout:GPUPipelineLayout;
  private dummy3D:[GPUTexture,GPUTexture,GPUTexture,GPUTexture];private dummy2D:GPUTexture;
  private bakeGroups:GPUBindGroup[]=[];
  private bindGroupCache=new Map<string,GPUBindGroup>();private resourceIds=new WeakMap<object,number>();private nextResourceId=1;
  private activePass?:GPUComputePassEncoder;

  constructor(private device:GPUDevice,geometry:TallCellLayout,fine:TallCellMultigridFineResources,readonly refinementCycles=8,deferPipelineCompilation=false){
    if(!deferPipelineCompilation)device.pushErrorScope("validation");
    const usage=GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST;
    const texture3=(nx:number,ny:number,nz:number)=>device.createTexture({size:[nx,ny,nz],dimension:"3d",format:"r32float",usage});
    const coefficientTexture=(nx:number,ny:number,nz:number)=>device.createTexture({size:[nx,ny,nz],dimension:"3d",format:"rgba32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING});
    const texture2=(nx:number,nz:number)=>device.createTexture({size:[nx,nz],format:"r32float",usage});
    this.fineRhs=texture3(geometry.nx,geometry.packedNy,geometry.nz);
    this.diagnostics=fine.diagnostics;
    // The restricted solver owns a true signed-distance field now; use it
    // directly on the finest pressure level instead of rebuilding phi from a
    // VOF proxy before every solve.
    const finePhi=fine.volume;
    const expandedInletBand=geometry.planning.storedRegularLayers>geometry.planning.requestedRegularLayers;
    const fullTopIterations=geometry.planning.ordinaryGridFallback||expandedInletBand?256:geometry.fineNy>geometry.settings.regularLayers*4?192:32;
    const correctionTopIterations=geometry.planning.ordinaryGridFallback||expandedInletBand?64:geometry.fineNy>geometry.settings.regularLayers*4?144:16;
    const makeParams=(nx:number,packedNy:number,nz:number,fineNy:number,layers:number,preserveAir:boolean,cell:{x:number;y:number;z:number},topIterations:number)=>{const buffer=device.createBuffer({size:48,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(buffer,0,new Uint32Array([nx,packedNy,nz,fineNy,layers,preserveAir?1:0,topIterations,0]));device.queue.writeBuffer(buffer,32,new Float32Array([cell.x,cell.y,cell.z,0]));return buffer;};
    const params=(nx:number,packedNy:number,nz:number,fineNy:number,layers:number,preserveAir:boolean,cell:{x:number;y:number;z:number})=>({params:makeParams(nx,packedNy,nz,fineNy,layers,preserveAir,cell,fullTopIterations),correctionParams:makeParams(nx,packedNy,nz,fineNy,layers,preserveAir,cell,correctionTopIterations)});
    this.levels.push({nx:geometry.nx,packedNy:geometry.packedNy,nz:geometry.nz,fineNy:geometry.fineNy,regularLayers:geometry.settings.regularLayers,cell:geometry.cellSize_m,pressure:[fine.pressureA,fine.pressureB],rhs:this.fineRhs,phi:finePhi,solid:fine.solid,base:fine.base,coeffLateral:coefficientTexture(geometry.nx,geometry.packedNy,geometry.nz),coeffVertical:coefficientTexture(geometry.nx,geometry.packedNy,geometry.nz),...params(geometry.nx,geometry.packedNy,geometry.nz,geometry.fineNy,geometry.settings.regularLayers,false,geometry.cellSize_m),owned:false});
    for(let index=1;index<8;index+=1){const previous=this.levels[index-1];if(previous.nx*previous.packedNy*previous.nz<=256)break;const nx=Math.ceil(previous.nx/2),nz=Math.ceil(previous.nz/2),fineNy=Math.ceil(previous.fineNy/2),regularLayers=Math.ceil(previous.regularLayers/2),packedNy=Math.ceil((previous.packedNy-2)/2)+2,cell={x:previous.cell.x*2,y:previous.cell.y*2,z:previous.cell.z*2};this.levels.push({nx,packedNy,nz,fineNy,regularLayers,cell,pressure:[texture3(nx,packedNy,nz),texture3(nx,packedNy,nz)],rhs:texture3(nx,packedNy,nz),phi:texture3(nx,packedNy,nz),solid:texture3(nx,packedNy,nz),base:texture2(nx,nz),coeffLateral:coefficientTexture(nx,packedNy,nz),coeffVertical:coefficientTexture(nx,packedNy,nz),...params(nx,packedNy,nz,fineNy,regularLayers,index<=2,cell),owned:true});}
    if(this.levels.at(-1)!.nx*this.levels.at(-1)!.packedNy*this.levels.at(-1)!.nz>256)throw new Error("Tall-cell multigrid top level exceeds one workgroup");
    this.dummy3D=[texture3(1,1,1),texture3(1,1,1),texture3(1,1,1),texture3(1,1,1)];this.dummy2D=texture2(1,1);
    this.layout=device.createBindGroupLayout({entries:[
      {binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},{binding:1,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float",viewDimension:"3d"}},{binding:2,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},{binding:3,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},{binding:4,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"2d"}},{binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
      {binding:6,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},{binding:7,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},{binding:8,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},{binding:9,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"2d"}},{binding:10,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
      {binding:11,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float",viewDimension:"3d"}},{binding:12,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float",viewDimension:"3d"}},{binding:13,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float",viewDimension:"2d"}},
      {binding:14,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},{binding:15,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},{binding:16,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:17,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},{binding:18,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}}
    ]});
    // The coefficient bake writes rgba32float storage textures; the main
    // layout is already at the four-storage-texture limit, so the bake owns a
    // dedicated layout reusing the same WGSL binding indices.
    this.bakeLayout=device.createBindGroupLayout({entries:[
      {binding:3,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},{binding:4,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"2d"}},{binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:14,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
      {binding:19,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32float",viewDimension:"3d"}},{binding:20,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32float",viewDimension:"3d"}}
    ]});
    this.shaderModule=device.createShaderModule({label:"Restricted tall-cell multigrid",code:multigridShader});this.pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[this.layout]});this.bakePipelineLayout=device.createPipelineLayout({bindGroupLayouts:[this.bakeLayout]});
    this.bakeGroups=this.levels.map(level=>device.createBindGroup({layout:this.bakeLayout,entries:[
      {binding:3,resource:level.phi.createView()},{binding:4,resource:level.base.createView()},{binding:5,resource:{buffer:level.params}},{binding:14,resource:level.solid.createView()},
      {binding:19,resource:level.coeffLateral.createView()},{binding:20,resource:level.coeffVertical.createView()}
    ]}));
    if(!deferPipelineCompilation){this.createPipelinesSync();void device.popErrorScope().then(error=>{if(error)console.error(`Tall multigrid pipeline creation: ${error.message}`);}).catch(()=>{/* Device loss is reported by the renderer. */});}
    const fineBytes=geometry.nx*geometry.packedNy*geometry.nz*4;
    this.allocatedBytes=fineBytes+this.levels.reduce((sum,l,index)=>sum+l.nx*l.packedNy*l.nz*4*(index===0?8:13)+(index===0?0:l.nx*l.nz*4),0)+8;
  }

  get levelCount(){return this.levels.length;}

  private pipelineDescriptor(entryPoint:string):GPUComputePipelineDescriptor{return{label:`Tall multigrid ${entryPoint}`,layout:entryPoint==="bakeCoefficients"?this.bakePipelineLayout:this.pipelineLayout,compute:{module:this.shaderModule,entryPoint}};}
  private createPipelinesSync(){const pipeline=(entryPoint:string)=>this.device.createComputePipeline(this.pipelineDescriptor(entryPoint));this.clearPipeline=pipeline("clearPressure");this.smoothPipelines=[pipeline("smoothRed"),pipeline("smoothBlack")];this.topPipeline=pipeline("solveTop");this.basePipeline=pipeline("downsampleBase");this.columnsPipeline=pipeline("downsampleColumns");this.bakePipeline=pipeline("bakeCoefficients");this.residualComputePipeline=pipeline("computeResidual");this.restrictPipeline=pipeline("restrictResidual");this.prolongatePipeline=pipeline("prolongateCorrection");this.residualPipeline=pipeline("reducePressureDiagnostics");}
  async initializePipelines(onProgress:(label:string,completed:number,total:number)=>void,offset=0,total=11){
    const definitions=[
      ["Clear pressure","clearPressure"],["Smooth red cells","smoothRed"],["Smooth black cells","smoothBlack"],
      ["Solve coarsest level","solveTop"],["Downsample column bases","downsampleBase"],["Downsample level set and solids","downsampleColumns"],
      ["Bake stencil coefficients","bakeCoefficients"],["Compute pressure residual","computeResidual"],
      ["Restrict residual","restrictResidual"],["Prolongate correction","prolongateCorrection"],["Reduce pressure diagnostics","reducePressureDiagnostics"]
    ] as const;
    const compiled:GPUComputePipeline[]=[];
    for(let index=0;index<definitions.length;index+=1){const [label,entryPoint]=definitions[index];onProgress(label,offset+index,total);compiled.push(await this.device.createComputePipelineAsync(this.pipelineDescriptor(entryPoint)));onProgress(label,offset+index+1,total);}
    this.clearPipeline=compiled[0];this.smoothPipelines=[compiled[1],compiled[2]];this.topPipeline=compiled[3];this.basePipeline=compiled[4];this.columnsPipeline=compiled[5];this.bakePipeline=compiled[6];this.residualComputePipeline=compiled[7];this.restrictPipeline=compiled[8];this.prolongatePipeline=compiled[9];this.residualPipeline=compiled[10];
  }

  private resourceId(resource:object){let id=this.resourceIds.get(resource);if(id===undefined){id=this.nextResourceId++;this.resourceIds.set(resource,id);}return id;}
  private group(current:Level,source:Level,pIn:GPUTexture,pOut:GPUTexture,options:{currentRhs?:GPUTexture;currentPhi?:GPUTexture;currentBase?:GPUTexture;currentSolid?:GPUTexture;currentParams?:GPUBuffer;sourcePressure?:GPUTexture;sourcePhi?:GPUTexture;sourceSolid?:GPUTexture;rhsOut?:GPUTexture;phiOut?:GPUTexture;baseOut?:GPUTexture}={}){const resources=[pIn,pOut,options.currentRhs??current.rhs,options.currentPhi??current.phi,options.currentBase??current.base,options.currentParams??current.params,options.sourcePressure??this.dummy3D[3],source.rhs,options.sourcePhi??source.phi,source.base,source.params,options.rhsOut??this.dummy3D[1],options.phiOut??this.dummy3D[2],options.baseOut??this.dummy2D,options.currentSolid??current.solid,options.sourceSolid??source.solid,this.diagnostics,current.coeffLateral,current.coeffVertical];const key=resources.map(resource=>this.resourceId(resource)).join(":");const cached=this.bindGroupCache.get(key);if(cached)return cached;const group=this.device.createBindGroup({layout:this.layout,entries:[{binding:0,resource:pIn.createView()},{binding:1,resource:pOut.createView()},{binding:2,resource:(options.currentRhs??current.rhs).createView()},{binding:3,resource:(options.currentPhi??current.phi).createView()},{binding:4,resource:(options.currentBase??current.base).createView()},{binding:5,resource:{buffer:options.currentParams??current.params}},{binding:6,resource:(options.sourcePressure??this.dummy3D[3]).createView()},{binding:7,resource:source.rhs.createView()},{binding:8,resource:(options.sourcePhi??source.phi).createView()},{binding:9,resource:source.base.createView()},{binding:10,resource:{buffer:source.params}},{binding:11,resource:(options.rhsOut??this.dummy3D[1]).createView()},{binding:12,resource:(options.phiOut??this.dummy3D[2]).createView()},{binding:13,resource:(options.baseOut??this.dummy2D).createView()},{binding:14,resource:(options.currentSolid??current.solid).createView()},{binding:15,resource:(options.sourceSolid??source.solid).createView()},{binding:16,resource:{buffer:this.diagnostics}},{binding:17,resource:current.coeffLateral.createView()},{binding:18,resource:current.coeffVertical.createView()}]});this.bindGroupCache.set(key,group);return group;}
  private dispatch(encoder:GPUCommandEncoder,pipeline:GPUComputePipeline,group:GPUBindGroup,l:Level,columns=false){const pass=this.activePass??encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(l.nx/(columns?8:4)),Math.ceil((columns?l.nz:l.packedNy)/(columns?8:4)),columns?1:Math.ceil(l.nz/4));if(!this.activePass)pass.end();}
  private clear(encoder:GPUCommandEncoder,l:Level,index:number){this.dispatch(encoder,this.clearPipeline,this.group(l,l,l.pressure[1-index],l.pressure[index]),l);}
  private smooth(encoder:GPUCommandEncoder,l:Level,state:number,count:number){for(let i=0;i<count;i+=1){for(const pipeline of this.smoothPipelines){const next=1-state;this.dispatch(encoder,pipeline,this.group(l,l,l.pressure[state],l.pressure[next]),l);state=next;}}return state;}
  private solveTop(encoder:GPUCommandEncoder,l:Level,correction=false){const pass=this.activePass??encoder.beginComputePass();pass.setPipeline(this.topPipeline);pass.setBindGroup(0,this.group(l,l,l.pressure[1],l.pressure[0],correction?{currentParams:l.correctionParams}:undefined));pass.dispatchWorkgroups(1);if(!this.activePass)pass.end();return 0;}
  private prolong(encoder:GPUCommandEncoder,l:Level,state:number,child:Level,childState:number){const next=1-state;this.dispatch(encoder,this.prolongatePipeline,this.group(l,child,l.pressure[state],l.pressure[next],{sourcePressure:child.pressure[childState]}),l);return next;}
  // Writes the level's residual into its idle ping-pong texture; the next
  // prolongation overwrites the same texture, so the scratch costs nothing.
  private residual(encoder:GPUCommandEncoder,l:Level,state:number){this.dispatch(encoder,this.residualComputePipeline,this.group(l,l,l.pressure[state],l.pressure[1-state]),l);return l.pressure[1-state];}
  private restrict(encoder:GPUCommandEncoder,child:Level,level:Level,residual:GPUTexture){this.dispatch(encoder,this.restrictPipeline,this.group(child,level,child.pressure[0],this.dummy3D[0],{currentRhs:this.dummy3D[3],sourcePressure:residual,rhsOut:child.rhs}),child);}
  private cycle(encoder:GPUCommandEncoder,index:number,state:number):number{const level=this.levels[index];if(index===this.levels.length-1)return this.solveTop(encoder,level,true);state=this.smooth(encoder,level,state,2);const child=this.levels[index+1];this.restrict(encoder,child,level,this.residual(encoder,level,state));this.clear(encoder,child,0);const childState=this.cycle(encoder,index+1,0);state=this.prolong(encoder,level,state,child,childState);return this.smooth(encoder,level,state,2);}

  encode(encoder:GPUCommandEncoder,options:TallCellMultigridEncodeOptions={}){
    this.activePass=encoder.beginComputePass({label:options.warmStart?"Tall-cell warm multigrid":"Tall-cell cold FMG"});
    const fine=this.levels[0];
    for(let index=1;index<this.levels.length;index+=1){
      const source=this.levels[index-1],current=this.levels[index];
      if(options.topologyChanged!==false)this.dispatch(encoder,this.basePipeline,this.group(current,source,current.pressure[0],this.dummy3D[0],{currentPhi:this.dummy3D[3],currentBase:this.dummy2D,baseOut:current.base}),current,true);
      this.dispatch(encoder,this.columnsPipeline,this.group(current,source,current.pressure[0],this.dummy3D[0],{currentPhi:this.dummy3D[3],currentSolid:this.dummy3D[3],phiOut:current.phi,rhsOut:current.solid}),current);
    }
    for(let index=0;index<this.levels.length;index+=1)this.dispatch(encoder,this.bakePipeline,this.bakeGroups[index],this.levels[index]);
    let state=0;
    if(!options.warmStart){
      for(const level of this.levels)this.clear(encoder,level,0);
      for(let index=1;index<this.levels.length;index+=1){
        const source=this.levels[index-1],current=this.levels[index];
        this.restrict(encoder,current,source,this.residual(encoder,source,0));
      }
      state=this.solveTop(encoder,this.levels.at(-1)!);
      for(let index=this.levels.length-2;index>=0;index-=1){
        state=this.prolong(encoder,this.levels[index],0,this.levels[index+1],state);
        state=this.cycle(encoder,index,state);
      }
    }
    // Refinement depth is an explicit convergence control. Eight cycles are
    // the validated default for remeshed tall columns; the UI and smoke
    // harness retain an override for convergence/performance probes.
    for(let cycle=0;cycle<this.refinementCycles;cycle+=1)state=this.cycle(encoder,0,state);
    this.activePass.end();this.activePass=undefined;
    if(state!==0)encoder.copyTextureToTexture({texture:fine.pressure[state]},{texture:fine.pressure[0]},[fine.nx,fine.packedNy,fine.nz]);
    this.dispatch(encoder,this.residualPipeline,this.group(fine,fine,fine.pressure[0],fine.pressure[1]),fine);
  }
  destroy(){this.bindGroupCache.clear();for(const l of this.levels){l.params.destroy();l.correctionParams.destroy();l.rhs.destroy();l.coeffLateral.destroy();l.coeffVertical.destroy();if(l.owned){l.pressure[0].destroy();l.pressure[1].destroy();l.phi.destroy();l.solid.destroy();l.base.destroy();}}for(const texture of this.dummy3D)texture.destroy();this.dummy2D.destroy();}
}
