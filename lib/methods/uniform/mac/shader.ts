import { MAC_SCHEDULE_SHADER } from "./schedule";
import { uniformSurfaceFillWGSL } from "../uniform-surface-volume.wgsl";

/** Buffer fields are the simulation authority; textures are the common presentation ABI. */
export const MAC_SHADER = /* wgsl */ `
struct Params { dims:vec4u, hdt:vec4f, gravityNu:vec4f, material:vec4f, solve:vec4f }
struct CG { x:f32, r:f32, d:f32, q:f32 }
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> vin:array<vec4f>;
@group(0) @binding(2) var<storage,read_write> vout:array<vec4f>;
@group(0) @binding(3) var<storage,read> v0:array<vec4f>;
@group(0) @binding(4) var<storage,read> phi:array<f32>;
@group(0) @binding(5) var<storage,read_write> phiOut:array<f32>;
@group(0) @binding(6) var<storage,read> phi0:array<f32>;
@group(0) @binding(7) var<storage,read> solid:array<u32>;
@group(0) @binding(8) var<storage,read_write> cellPhi:array<f32>;
@group(0) @binding(9) var<storage,read_write> matrix:array<vec4f>;
@group(0) @binding(10) var<storage,read_write> cg:array<CG>;
@group(0) @binding(11) var<storage,read_write> partial:array<vec4f>;
@group(0) @binding(12) var<storage,read_write> scalars:array<f32>;
@group(0) @binding(13) var volumeTex:texture_storage_3d<r32float,write>;
@group(0) @binding(14) var phiTex:texture_storage_3d<r32float,write>;
@group(0) @binding(15) var velocityTex:texture_storage_3d<rgba32float,write>;
@group(0) @binding(16) var pressureTex:texture_storage_3d<r32float,write>;
@group(0) @binding(17) var divergenceTex:texture_storage_3d<r32float,write>;
@group(0) @binding(18) var openTex:texture_storage_3d<r32float,write>;
@group(0) @binding(19) var<storage,read_write> rhs:array<f32>;
var<workgroup> sums:array<vec4f,64>;
fn dt()->f32{return scalars[16];}
fn count()->u32{return p.dims.x*p.dims.y*p.dims.z;}
fn vertices()->u32{return (p.dims.x+1u)*(p.dims.y+1u)*(p.dims.z+1u);}
fn coord(i:u32,d:vec3u)->vec3i{return vec3i(vec3u(i%d.x,(i/d.x)%d.y,i/(d.x*d.y)));}
fn index(q:vec3i,d:vec3u)->u32{return u32(q.x)+d.x*(u32(q.y)+d.y*u32(q.z));}
fn inside(q:vec3i)->bool{return all(q>=vec3i(0))&&all(q<vec3i(p.dims.xyz));}
fn closed(q:vec3i)->bool{
 if(!inside(q)){return !(p.material.w>0.0&&q.y>=i32(p.dims.y)&&q.x>=0&&q.z>=0&&q.x<i32(p.dims.x)&&q.z<i32(p.dims.z));}
 return solid[index(q,p.dims.xyz)]!=0u;
}
fn axis(a:u32)->vec3i{var e=vec3i(0);e[a]=1;return e;}
fn blocked(q:vec3i,a:u32)->bool{return closed(q)||closed(q+axis(a));}
fn ph(q:vec3i)->f32{return phi[index(clamp(q,vec3i(0),vec3i(p.dims.xyz)),p.dims.xyz+vec3u(1))];}
fn ph0(q:vec3i)->f32{return phi0[index(clamp(q,vec3i(0),vec3i(p.dims.xyz)),p.dims.xyz+vec3u(1))];}
fn cell(q:vec3i)->f32{
 if(!inside(q)){return max(p.hdt.x,max(p.hdt.y,p.hdt.z));}
 return cellPhi[index(q,p.dims.xyz)];
}
fn liquid(q:vec3i)->bool{return inside(q)&&!closed(q)&&cell(q)<0.0;}
fn face(q:vec3i,a:u32)->f32{
 if(!inside(q)){return 0.0;}return vin[index(q,p.dims.xyz)][a];
}
fn originalFace(q:vec3i,a:u32)->f32{
 if(!inside(q)){return 0.0;}return v0[index(q,p.dims.xyz)][a];
}
fn sampledFace(q:vec3i,a:u32,original:bool)->f32{
 // Mirror tangential velocity across free-slip walls, while normal faces stay zero.
 var r=q;for(var b=0u;b<3u;b++){if(b!=a){r[b]=clamp(r[b],0,i32(p.dims[b])-1);}}
 if(!inside(r)){return 0.0;}
 if(closed(r)&&closed(r+axis(a))){
  var total=0.0;var n=0.0;
  for(var b=0u;b<3u;b++){if(b!=a){for(var s=-1;s<=1;s+=2){let v=r+s*axis(b);if(inside(v)&&!blocked(v,a)){total+=select(face(v,a),originalFace(v,a),original);n+=1.0;}}}}
  return total/max(n,1.0);
 }
 return select(face(r,a),originalFace(r,a),original);
}
fn sampleFace(position:vec3f,a:u32,original:bool)->f32{
 var offset=vec3f(0.5);offset[a]=1.0;
 let x=position/p.hdt.xyz-offset;let q=vec3i(floor(x));let f=fract(x);var value=0.0;
 for(var k=0u;k<8u;k++){
  let o=vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));
  let w=select(vec3f(1)-f,f,o!=vec3i(0));value+=w.x*w.y*w.z*sampledFace(q+o,a,original);
 }return value;
}
fn velocity(x:vec3f)->vec3f{return vec3f(sampleFace(x,0u,true),sampleFace(x,1u,true),sampleFace(x,2u,true));}
fn trace(x:vec3f,dt:f32)->vec3f{
 let middle=x-0.5*dt*velocity(x);let departure=x-dt*velocity(middle);
 // A small-step characteristic must not sample through a voxel wall.
 let grid=departure/p.hdt.xyz;let q=vec3i(floor(grid));var open=!closed(q);
 // A vertex exactly on a high-side wall belongs to its open incident cell too.
 for(var k=1u;k<8u;k++){let o=vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));if(all(select(fract(grid)<vec3f(1e-5),vec3<bool>(true),o==vec3i(0)))){open=open||!closed(q-o);}}
 if(!open){return x;}
 return departure;
}
fn samplePhi(x:vec3f)->f32{
 let g=clamp(x/p.hdt.xyz,vec3f(0),vec3f(p.dims.xyz));let q=vec3i(floor(g));let f=fract(g);var value=0.0;
 for(var k=0u;k<8u;k++){let o=vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));let w=select(vec3f(1)-f,f,o!=vec3i(0));value+=w.x*w.y*w.z*ph(q+o);}return value;
}
@compute @workgroup_size(64) fn advect(@builtin(global_invocation_id) id:vec3u){
 let i=id.x+id.y*65535u*64u;if(i<vertices()){let q=coord(i,p.dims.xyz+vec3u(1));phiOut[i]=samplePhi(trace(vec3f(q)*p.hdt.xyz,dt()));}
 if(i>=count()){return;}let q=coord(i,p.dims.xyz);var result=vec3f(0);
 for(var a=0u;a<3u;a++){if(!blocked(q,a)){var offset=vec3f(0.5);offset[a]=1.0;let x=(vec3f(q)+offset)*p.hdt.xyz;result[a]=sampleFace(trace(x,dt()),a,false);}}
 vout[i]=vec4f(result,0);
}
// The reverse sample and correction are fused; the predictor remains immutable.
@compute @workgroup_size(64) fn correct(@builtin(global_invocation_id) id:vec3u){
 let i=id.x+id.y*65535u*64u;
 if(i<vertices()){
  let q=coord(i,p.dims.xyz+vec3u(1));let x=vec3f(q)*p.hdt.xyz;let departure=trace(x,dt());
  let base=vec3i(floor(departure/p.hdt.xyz));var lo=1e30;var hi=-1e30;
  for(var k=0u;k<8u;k++){let o=vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));let v=ph0(base+o);lo=min(lo,v);hi=max(hi,v);}
  phiOut[i]=clamp(phi[i]+0.5*(phi0[i]-samplePhi(trace(x,-dt()))),lo,hi);
 }
 if(i>=count()){return;}let q=coord(i,p.dims.xyz);var result=vec3f(0);
 for(var a=0u;a<3u;a++){if(!blocked(q,a)){
  var offset=vec3f(0.5);offset[a]=1.0;let x=(vec3f(q)+offset)*p.hdt.xyz;
  let base=vec3i(floor(trace(x,dt())/p.hdt.xyz-offset));var lo=1e30;var hi=-1e30;
  for(var k=0u;k<8u;k++){let o=vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));let v=sampledFace(base+o,a,true);lo=min(lo,v);hi=max(hi,v);}
  result[a]=clamp(vin[i][a]+0.5*(v0[i][a]-sampleFace(trace(x,-dt()),a,false)),lo,hi);
 }}vout[i]=vec4f(result,0);
}
@compute @workgroup_size(64) fn redistance(@builtin(global_invocation_id) id:vec3u){
 let i=id.x+id.y*65535u*64u;if(i>=vertices()){return;}let q=coord(i,p.dims.xyz+vec3u(1));let reference=phi0[i];let value=phi[i];
 var pinned=abs(reference)<1e-12;var norm2=0.0;
 for(var a=0u;a<3u;a++){
  let e=axis(a);pinned=pinned||((reference<0.0)!=(ph0(q-e)<0.0))||((reference<0.0)!=(ph0(q+e)<0.0));
  let backward=(value-ph(q-e))/p.hdt[a];let forward=(ph(q+e)-value)/p.hdt[a];
  let b=select(min(backward,0.0),max(backward,0.0),reference>=0.0);let f=select(max(forward,0.0),min(forward,0.0),reference>=0.0);
  norm2+=max(b*b,f*f);
 }
 let h=min(p.hdt.x,min(p.hdt.y,p.hdt.z));let s=reference/sqrt(reference*reference+h*h);
 phiOut[i]=select(value-0.3*h*s*(sqrt(norm2)-1.0),reference,pinned);
}
@compute @workgroup_size(64) fn geometry(@builtin(global_invocation_id) id:vec3u){
 let i=id.x+id.y*65535u*64u;if(i>=count()){return;}let q=coord(i,p.dims.xyz);var average=0.0;
 for(var k=0u;k<8u;k++){average+=ph(q+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)));}cellPhi[i]=average*0.125;
}
@compute @workgroup_size(64) fn forces(@builtin(global_invocation_id) id:vec3u){
 let i=id.x+id.y*65535u*64u;if(i>=count()){return;}let q=coord(i,p.dims.xyz);var result=vin[i].xyz;
 for(var a=0u;a<3u;a++){
  if(blocked(q,a)){result[a]=0.0;continue;}var laplacian=0.0;
  for(var b=0u;b<3u;b++){
   let e=axis(b);
   // Normal velocity has a zero wall value; tangential velocity has zero normal derivative.
   let lower=select(face(q-e,a),vin[i][a],b!=a&&blocked(q-e,a));let upper=select(face(q+e,a),vin[i][a],b!=a&&blocked(q+e,a));
   laplacian+=(lower-2.0*vin[i][a]+upper)/(p.hdt[b]*p.hdt[b]);
  }
  result[a]+=dt()*(p.gravityNu[a]+p.gravityNu.w*laplacian);
 }vout[i]=vec4f(result,0);
}
fn normal(q:vec3i)->vec3f{
 var r=clamp(q,vec3i(0),vec3i(p.dims.xyz)-1);
 // Do not differentiate buried solid sentinels into a capillary force.
 if(closed(r)){for(var a=0u;a<3u;a++){for(var s=-1;s<=1;s+=2){let candidate=q+s*axis(a);if(inside(candidate)&&!closed(candidate)){r=candidate;}}}}
 let centre=cell(r);var g=vec3f(0);
 for(var a=0u;a<3u;a++){let e=axis(a);let lower=clamp(r-e,vec3i(0),vec3i(p.dims.xyz)-1);let upper=clamp(r+e,vec3i(0),vec3i(p.dims.xyz)-1);
  g[a]=(select(cell(upper),centre,closed(upper))-select(cell(lower),centre,closed(lower)))/(2.0*p.hdt[a]);}
 return g/max(length(g),1e-12);
}
fn boundaryPotential(q:vec3i)->f32{
 if(p.material.y==0.0){return 0.0;}var curvature=0.0;
 for(var a=0u;a<3u;a++){let e=axis(a);curvature+=(normal(q+e)[a]-normal(q-e)[a])/(2.0*p.hdt[a]);}
 return dt()*p.material.y/p.material.x*curvature;
}
fn theta(q:vec3i,r:vec3i)->f32{if(!inside(r)){return 0.5;}return max(0.01,abs(cell(q))/(abs(cell(q))+abs(cell(r))));}
fn divergence(q:vec3i)->f32{
 var d=0.0;for(var a=0u;a<3u;a++){d+=(face(q,a)-face(q-axis(a),a))/p.hdt[a];}return d;
}
@compute @workgroup_size(64) fn buildSystem(@builtin(global_invocation_id) id:vec3u){
 let i=id.x+id.y*65535u*64u;if(i>=count()){return;}let q=coord(i,p.dims.xyz);matrix[i]=vec4f(0);rhs[i]=0.0;
 if(!liquid(q)){return;}var diagonal=0.0;var positive=vec3f(0);var b=-divergence(q);let capillary=boundaryPotential(q);
 for(var a=0u;a<3u;a++){for(var s=-1;s<=1;s+=2){let r=q+s*axis(a);if(closed(r)){continue;}
  var w=1.0/(p.hdt[a]*p.hdt[a]);if(liquid(r)){if(s==1){positive[a]=w;}}else{w/=theta(q,r);b+=w*capillary;}diagonal+=w;
 }}matrix[i]=vec4f(positive,diagonal);rhs[i]=b;
}
fn applyA(q:vec3i,direction:bool)->f32{
 let i=index(q,p.dims.xyz);
 // Air and solid rows are identically zero. Avoid six neighbour gathers for
 // every inactive cell on every pressure iteration.
 if(matrix[i].w==0.0){return 0.0;}
 var own=cg[i].x;if(direction){own=cg[i].r/max(matrix[i].w,1e-30)+scalars[2]*cg[i].d;}var value=matrix[i].w*own;
 for(var a=0u;a<3u;a++){
  let e=axis(a);if(inside(q+e)){let j=index(q+e,p.dims.xyz);var x=cg[j].x;if(direction){x=cg[j].r/max(matrix[j].w,1e-30)+scalars[2]*cg[j].d;}value-=matrix[i][a]*x;}
  if(inside(q-e)){let j=index(q-e,p.dims.xyz);var x=cg[j].x;if(direction){x=cg[j].r/max(matrix[j].w,1e-30)+scalars[2]*cg[j].d;}value-=matrix[j][a]*x;}
 }return value;
}
// Reductions use a fixed tree; no floating point atomics or host dot products.
fn reduceLocal(lane:u32,value:vec4f)->vec4f{
 sums[lane]=value;workgroupBarrier();for(var stride=32u;stride>0u;stride/=2u){if(lane<stride){let a=sums[lane];let b=sums[lane+stride];sums[lane]=vec4f(a.x+b.x,max(a.y,b.y),a.z+b.z,max(a.w,b.w));}workgroupBarrier();}let result=sums[0];workgroupBarrier();return result;
}
fn reduceGlobalOffset(lane:u32,offset:u32)->vec4f{
 var value=vec4f(0);for(var i=lane;i<(count()+63u)/64u;i+=64u){let b=partial[offset+i];value=vec4f(value.x+b.x,max(value.y,b.y),value.z+b.z,max(value.w,b.w));}return reduceLocal(lane,value);
}
fn reduceGlobal(lane:u32)->vec4f{return reduceGlobalOffset(lane,0u);}
@compute @workgroup_size(64) fn initializeCG(@builtin(global_invocation_id) id:vec3u){let i=id.x+id.y*65535u*64u;if(i<count()){cg[i]=CG(select(0.0,cg[i].x*dt()/max(scalars[15],1e-30),matrix[i].w>0.0),rhs[i],0.0,0.0);}}
fn residual(id:vec3u,lane:u32,group:vec3u,restart:bool){
 let i=id.x+id.y*65535u*64u;var value=vec4f(0);
 if(i<count()){
  let r=rhs[i]-applyA(coord(i,p.dims.xyz),false);let z=r/max(matrix[i].w,1e-30);
  if(restart){cg[i].r=r;cg[i].d=0.0;}
  value=vec4f(r*z,abs(r),0,abs(rhs[i]));
 }let reduced=reduceLocal(lane,value);if(lane==0u&&group.x+group.y*65535u<(count()+63u)/64u){partial[group.x+group.y*65535u]=reduced;}
}
@compute @workgroup_size(64) fn trueResidual(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3u){residual(id,lane,group,true);}
@compute @workgroup_size(64) fn checkResidual(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3u){residual(id,lane,group,scalars[3]>0.0);}
@compute @workgroup_size(64) fn restartCG(@builtin(local_invocation_index) lane:u32){
 let r=reduceGlobal(lane);if(lane==0u){scalars[0]=r.x;scalars[2]=0.0;scalars[3]=select(0.0,1.0,r.y<=p.material.z*0.25);scalars[4]=r.y;scalars[5]=r.w;}
}
@compute @workgroup_size(64) fn checkCG(@builtin(local_invocation_index) lane:u32){
 let r=reduceGlobal(lane);
 if(lane==0u&&scalars[21]>0.0&&scalars[20]==0.0){
  scalars[4]=r.y;scalars[5]=r.w;
  if(r.y<=p.material.z){scalars[20]=1.0;launches[3]=noLaunch();launches[4]=noLaunch();}
  else if(scalars[3]>0.0){scalars[0]=r.x;scalars[2]=0.0;scalars[3]=0.0;}
 }
}
@compute @workgroup_size(64) fn multiply(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3u){
 let i=id.x+id.y*65535u*64u;var value=0.0;if(i<count()&&scalars[3]==0.0){let q=applyA(coord(i,p.dims.xyz),true);cg[i].q=q;value=(cg[i].r/max(matrix[i].w,1e-30)+scalars[2]*cg[i].d)*q;}
 let reduced=reduceLocal(lane,vec4f(value,0,0,0));if(lane==0u&&group.x+group.y*65535u<(count()+63u)/64u){partial[group.x+group.y*65535u]=reduced;}
}
@compute @workgroup_size(64) fn alpha(@builtin(local_invocation_index) lane:u32){
 let r=reduceGlobal(lane);if(lane==0u&&scalars[3]==0.0){if(r.x>0.0){scalars[1]=scalars[0]/r.x;}else{scalars[1]=0.0;scalars[3]=1.0;}}
}
@compute @workgroup_size(64) fn updateCG(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3u){
 let i=id.x+id.y*65535u*64u;var value=vec4f(0);
 if(i<count()&&scalars[3]==0.0&&matrix[i].w>0.0){let d=cg[i].r/matrix[i].w+scalars[2]*cg[i].d;cg[i].d=d;cg[i].x+=scalars[1]*d;cg[i].r-=scalars[1]*cg[i].q;let r=cg[i].r;value=vec4f(r*r/matrix[i].w,abs(r),0,0);}
 let reduced=reduceLocal(lane,value);if(lane==0u&&group.x+group.y*65535u<(count()+63u)/64u){partial[group.x+group.y*65535u]=reduced;}
}
@compute @workgroup_size(64) fn beta(@builtin(local_invocation_index) lane:u32){
 let r=reduceGlobal(lane);if(lane==0u&&scalars[3]==0.0){scalars[2]=r.x/max(scalars[0],1e-30);scalars[0]=r.x;scalars[10]+=1.0;if(r.y<=p.material.z*0.25){scalars[3]=1.0;}}
}
@compute @workgroup_size(64) fn project(@builtin(global_invocation_id) id:vec3u){
 let i=id.x+id.y*65535u*64u;if(i>=count()){return;}let q=coord(i,p.dims.xyz);var value=vec3f(0);var valid=0u;
 for(var a=0u;a<3u;a++){
  let r=q+axis(a);if(blocked(q,a)||(!liquid(q)&&!liquid(r))){continue;}var left=0.0;var right=0.0;var distance=p.hdt[a];
  if(liquid(q)){left=cg[i].x;}if(liquid(r)){right=cg[index(r,p.dims.xyz)].x;}
  if(!liquid(r)){distance*=theta(q,r);right=boundaryPotential(q);}else if(!liquid(q)){distance*=theta(r,q);left=boundaryPotential(r);}
  value[a]=vin[i][a]-(right-left)/distance;valid|=1u<<a;
 }vout[i]=vec4f(value,f32(valid));
}
@compute @workgroup_size(64) fn extend(@builtin(global_invocation_id) id:vec3u){
 let i=id.x+id.y*65535u*64u;if(i>=count()){return;}let q=coord(i,p.dims.xyz);var value=vin[i].xyz;var valid=u32(vin[i].w);
 for(var a=0u;a<3u;a++){if(blocked(q,a)||(valid&(1u<<a))!=0u){continue;}var total=0.0;var n=0.0;
  for(var b=0u;b<3u;b++){for(var s=-1;s<=1;s+=2){let r=q+s*axis(b);if(inside(r)){let j=index(r,p.dims.xyz);if((u32(vin[j].w)&(1u<<a))!=0u){total+=vin[j][a];n+=1.0;}}}}
  if(n>0.0){value[a]=total/n;valid|=1u<<a;}
 }vout[i]=vec4f(value,f32(valid));
}
${uniformSurfaceFillWGSL}
@compute @workgroup_size(64) fn publishPhi(@builtin(global_invocation_id) id:vec3u){let i=id.x+id.y*65535u*64u;if(i<vertices()){textureStore(phiTex,coord(i,p.dims.xyz+vec3u(1)),vec4f(phi[i]));}}
@compute @workgroup_size(64) fn measure(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3u){
 let i=id.x+id.y*65535u*64u;var volume=0.0;var speed=0.0;var energy=0.0;var div=0.0;var front=0.0;var pressure=0.0;var fault=0.0;
 if(i<count()){
  let q=coord(i,p.dims.xyz);var corners:array<f32,8>;
  for(var k=0u;k<8u;k++){corners[k]=ph(q+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)));if(!(abs(corners[k])<1e30)){fault=1.0;}}
  if(!closed(q)){volume=fill(corners);}
  var centre=vec3f(0);for(var a=0u;a<3u;a++){centre[a]=0.5*(vin[i][a]+face(q-axis(a),a));}
  // Component maxima conservatively bound the norm used by the next CFL step.
  speed=length(vin[i].xyz);energy=0.5*p.material.x*volume*dot(centre,centre);
  if(!(speed<1e30)){fault=1.0;}
  pressure=cg[i].x*p.material.x/max(dt(),1e-20);if(!(abs(pressure)<1e30)){fault=1.0;}
  if(volume>0.01){front=f32(q.x+1);}
  if(liquid(q)){div=divergence(q);}
 }
 // Rectangular 2D dispatches can include padding workgroups. They must not
 // overwrite the second reduction array with empty first-array results.
 let reduced=reduceLocal(lane,vec4f(volume,speed,energy,abs(div)));if(lane==0u&&group.x+group.y*65535u<(count()+63u)/64u){partial[group.x+group.y*65535u]=reduced;}
 let extra=reduceLocal(lane,vec4f(fault,abs(pressure),0,front));if(lane==0u&&group.x+group.y*65535u<(count()+63u)/64u){partial[(count()+63u)/64u+group.x+group.y*65535u]=extra;}
}
@compute @workgroup_size(64) fn commitVelocity(@builtin(global_invocation_id) id:vec3u){let i=id.x+id.y*65535u*64u;if(i<count()){vout[i]=vin[i];}}
@compute @workgroup_size(64) fn publish(@builtin(global_invocation_id) id:vec3u){
 let i=id.x+id.y*65535u*64u;if(i>=count()){return;}let q=coord(i,p.dims.xyz);var corners:array<f32,8>;
 for(var k=0u;k<8u;k++){corners[k]=ph(q+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)));}
 textureStore(volumeTex,q,vec4f(select(fill(corners),0.0,closed(q))));textureStore(velocityTex,q,vin[i]);
 textureStore(pressureTex,q,vec4f(cg[i].x*p.material.x/max(dt(),1e-20)));
 textureStore(divergenceTex,q,vec4f(select(0.0,divergence(q),liquid(q))));
 textureStore(openTex,q,vec4f(select(1.0,0.0,closed(q))));
}
@compute @workgroup_size(64) fn statistics(@builtin(local_invocation_index) lane:u32){
 let r=reduceGlobal(lane);if(lane==0u){scalars[6]=r.y;scalars[7]=r.x;scalars[8]=r.z*p.hdt.x*p.hdt.y*p.hdt.z;scalars[9]=r.w;}
 let extra=reduceGlobalOffset(lane,(count()+63u)/64u);if(lane==0u){scalars[11]=extra.x;scalars[13]=extra.y;scalars[14]=extra.w;}
}
${MAC_SCHEDULE_SHADER}
`;

// Explicit entry resources keep layouts small and let the compiler enforce read/write separation.
export const MAC_BINDINGS = {
  advect: [0, 1, 2, 3, 4, 5, 7, 12], correct: [0, 1, 2, 3, 4, 5, 6, 7, 12],
  redistance: [0, 4, 5, 6], geometry: [0, 4, 8], forces: [0, 1, 2, 7, 12],
  buildSystem: [0, 1, 7, 8, 9, 12, 19], initializeCG: [0, 9, 10, 12, 19],
  trueResidual: [0, 9, 10, 11, 12, 19], checkResidual: [0, 9, 10, 11, 12, 19], restartCG: [0, 11, 12], checkCG: [0, 11, 12, 20],
  multiply: [0, 9, 10, 11, 12], alpha: [0, 11, 12], updateCG: [0, 9, 10, 11, 12], beta: [0, 11, 12],
  project: [0, 1, 2, 7, 8, 10, 12], extend: [0, 1, 2, 7], commitVelocity: [0, 1, 2], publishPhi: [0, 4, 14],
  measure: [0, 1, 4, 7, 8, 10, 11, 12], statistics: [0, 11, 12],
  publish: [0, 1, 4, 7, 8, 10, 12, 13, 15, 16, 17, 18],
  seedState: [0, 12], beginFrame: [0, 12, 20], prepareStep: [0, 12, 20],
  pressureVerdict: [0, 12, 20], finishStep: [0, 12], endFrame: [0, 12, 20],
} as const;
export type MacEntry = keyof typeof MAC_BINDINGS;
