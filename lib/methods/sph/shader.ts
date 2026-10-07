import { levelSetFillWGSL } from "../../core/level-set-fill.wgsl";

/** Classic Müller poly6 / spiky / viscosity kernels in metres.
 * Separate density, force and motion dispatches prevent neighbor read/write races.
 * Integer linked bins have no fixed per-cell neighbor capacity. */
export const SPH_SHADER = /* wgsl */ `
struct Params { dims:vec4u, h:vec4f, gravityNu:vec4f, material:vec4f, tuning:vec4f, kernel:vec4f }
struct Particle { position:vec4f, velocity:vec4f }
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read_write> particles:array<Particle>;
@group(0) @binding(2) var<storage,read_write> heads:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read_write> links:array<u32>;
@group(0) @binding(4) var<storage,read> solids:array<u32>;
@group(0) @binding(5) var<storage,read_write> density:array<vec2f>;
@group(0) @binding(6) var<storage,read_write> forces:array<vec4f>;
@group(0) @binding(7) var<storage,read_write> phi:array<f32>;
@group(0) @binding(8) var<storage,read_write> fields:array<vec4f>;
@group(0) @binding(9) var<storage,read_write> state:array<f32>;
@group(0) @binding(10) var<storage,read_write> status:array<atomic<u32>>;
@group(0) @binding(11) var phiTex:texture_storage_3d<r32float,write>;
@group(0) @binding(12) var volumeTex:texture_storage_3d<r32float,write>;
@group(0) @binding(13) var velocityTex:texture_storage_3d<rgba32float,write>;
@group(0) @binding(14) var openTex:texture_storage_3d<r32float,write>;
@group(0) @binding(15) var<storage,read_write> partial:array<vec4f>;
fn flat(id:vec3u)->u32{return id.x+id.y*65535u*64u;}
fn cells()->u32{return p.dims.x*p.dims.y*p.dims.z;}
fn binDims()->vec3u{return vec3u(ceil(vec3f(p.dims.xyz)*p.h.xyz/p.h.w));}
fn binCount()->u32{let d=binDims();return d.x*d.y*d.z;}
fn binInside(q:vec3i)->bool{return all(q>=vec3i(0))&&all(q<vec3i(binDims()));}
fn binIndex(q:vec3i)->u32{let d=binDims();return u32(q.x)+d.x*(u32(q.y)+d.y*u32(q.z));}
fn coord(i:u32,d:vec3u)->vec3i{return vec3i(vec3u(i%d.x,(i/d.x)%d.y,i/(d.x*d.y)));}
fn inside(q:vec3i)->bool{return all(q>=vec3i(0))&&all(q<vec3i(p.dims.xyz));}
fn index(q:vec3i)->u32{return u32(q.x)+p.dims.x*(u32(q.y)+p.dims.y*u32(q.z));}
fn closed(q:vec3i)->bool{
 if(!inside(q)){return !(p.material.z>0.0&&q.y>=i32(p.dims.y)&&q.x>=0&&q.z>=0&&q.x<i32(p.dims.x)&&q.z<i32(p.dims.z));}
 return (solids[index(q)]&1u)!=0u;
}
fn healthy()->bool{return atomicLoad(&status[0])==0u;}
fn stepActive()->bool{return healthy()&&state[0]>1e-9;}
fn poly(r2:f32)->f32{let t=max(0.0,1.0-r2/(p.h.w*p.h.w));return p.kernel.x*t*t*t;}
fn boundaryPosition(q:vec3i,k:u32)->vec3f{let n=u32(p.kernel.w);return (vec3f(q)+(vec3f(f32(k%n),f32((k/n)%n),f32(k/(n*n)))+vec3f(0.5))/p.kernel.w)*p.h.xyz;}
fn mass()->f32{return p.material.x*p.kernel.z;}
@compute @workgroup_size(64) fn beginFrame(@builtin(global_invocation_id) id:vec3u){if(flat(id)==0u){state[0]=p.tuning.w;state[2]=0.0;state[1]=0.0;}}
@compute @workgroup_size(64) fn clearBins(@builtin(global_invocation_id) id:vec3u){let i=flat(id);if(i<binCount()){atomicStore(&heads[i],0xffffffffu);}}
@compute @workgroup_size(64) fn clearStepBins(@builtin(global_invocation_id) id:vec3u){let i=flat(id);if(i<binCount()&&stepActive()){atomicStore(&heads[i],0xffffffffu);}}
fn insertParticle(i:u32){if(i>=p.dims.w||!healthy()){return;}let a=particles[i];if(a.position.w<=0.0){return;}
 if(!all(abs(a.position.xyz)<vec3f(1e20))||!all(abs(a.velocity.xyz)<vec3f(1e20))){atomicStore(&status[0],1u);return;}
 let q=vec3i(floor(a.position.xyz/p.h.xyz));if(!inside(q)||closed(q)){atomicStore(&status[0],2u);return;}
 let bin=vec3i(floor(a.position.xyz/p.h.w));links[i]=atomicExchange(&heads[binIndex(bin)],i);
}
@compute @workgroup_size(64) fn binParticles(@builtin(global_invocation_id) id:vec3u){insertParticle(flat(id));}
@compute @workgroup_size(64) fn binStepParticles(@builtin(global_invocation_id) id:vec3u){if(stepActive()){insertParticle(flat(id));}}
fn evaluateDensity(i:u32){
 if(i>=p.dims.w||!healthy()||particles[i].position.w<=0.0){return;}
 let position=particles[i].position.xyz;let q=vec3i(floor(position/p.h.w));var rho=0.0;
 for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
  let bin=q+vec3i(x,y,z);if(!binInside(bin)){continue;}
  var j=atomicLoad(&heads[binIndex(bin)]);while(j!=0xffffffffu){let d=position-particles[j].position.xyz;rho+=mass()*poly(dot(d,d));j=links[j];}
 }}}
 let cell=vec3i(floor(position/p.h.xyz));let reach=vec3i(ceil(vec3f(p.h.w)/p.h.xyz));let n=u32(p.kernel.w);
 if((solids[index(cell)]&2u)!=0u){
 for(var z=-reach.z;z<=reach.z;z++){for(var y=-reach.y;y<=reach.y;y++){for(var x=-reach.x;x<=reach.x;x++){
  let bin=cell+vec3i(x,y,z);if(!closed(bin)){continue;}
  for(var k=0u;k<n*n*n;k++){let d=position-boundaryPosition(bin,k);rho+=mass()*poly(dot(d,d));}
 }}}
 }
 density[i]=vec2f(rho,p.tuning.x*p.tuning.x*max(0.0,rho-p.material.x));
 if(!(rho>0.0&&rho<1e20)){atomicStore(&status[0],1u);}
}
@compute @workgroup_size(64) fn densities(@builtin(global_invocation_id) id:vec3u){if(stepActive()){evaluateDensity(flat(id));}}
@compute @workgroup_size(64) fn finalDensity(@builtin(global_invocation_id) id:vec3u){evaluateDensity(flat(id));}
// Returns pressure/viscosity acceleration and the explicit viscosity row sum.
fn pairForce(d:vec3f,v:vec3f,ri:vec2f,rj:vec2f,wall:bool)->vec4f{
 let r=length(d);if(r>=p.h.w){return vec4f(0);}
 let t=1.0-r/p.h.w;
 var coefficient=p.gravityNu.w*p.material.x*mass()*p.kernel.y*t/(ri.x*rj.x);
 if(wall&&p.material.w==0.0){coefficient=0.0;}
 var a=coefficient*v;
 if(r>1e-8*p.h.w){a+=mass()*(ri.y+rj.y)/(2.0*ri.x*rj.x)*p.kernel.y*p.h.w*t*t*d/r;}
 // Monaghan approaching-pair artificial viscosity damps acoustic shocks while
 // preserving equal-and-opposite fluid impulses. It does not smooth expansion.
 if(r>1e-8*p.h.w&&dot(v,d)>0.0){
  let shock=mass()*p.tuning.z*p.tuning.x*p.h.w/(0.5*(ri.x+rj.x)*(r*r+0.01*p.h.w*p.h.w))*p.kernel.y*p.h.w*t*t*r;
  a+=shock*dot(v,d/r)*d/r;coefficient+=shock;
 }
 return vec4f(a,coefficient);
}
struct Interaction { force:vec4f, normal:vec3f, laplacian:f32 }
fn interaction(i:u32,j:u32)->Interaction{
 var result:Interaction;let a=particles[i];let b=particles[j];if(b.position.w<=0.0){return result;}
 let d=a.position.xyz-b.position.xyz;let r2=dot(d,d);if(r2>=p.h.w*p.h.w){return result;}
 let t=1.0-r2/(p.h.w*p.h.w);
 if(j!=i){result.force=pairForce(d,b.velocity.xyz-a.velocity.xyz,density[i],density[j],false);}
 if(p.material.y>0.0){let volume=mass()/density[j].x;
  result.normal=volume*(-6.0*p.kernel.x/(p.h.w*p.h.w))*t*t*d;
  result.laplacian=volume*(-6.0*p.kernel.x/(p.h.w*p.h.w))*t*(3.0-7.0*r2/(p.h.w*p.h.w));
 }
 return result;
}
fn evaluateAcceleration(i:u32)->vec4f{
 let a=particles[i];let ri=density[i];let q=vec3i(floor(a.position.xyz/p.h.w));var force=vec4f(p.gravityNu.xyz,0);
 var normal=vec3f(0);var laplacian=0.0;
 for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
  let bin=q+vec3i(x,y,z);if(!binInside(bin)){continue;}
  var j=atomicLoad(&heads[binIndex(bin)]);while(j!=0xffffffffu){
   let pair=interaction(i,j);force+=pair.force;normal+=pair.normal;laplacian+=pair.laplacian;
   j=links[j];
  }
 }}}
 let cell=vec3i(floor(a.position.xyz/p.h.xyz));let reach=vec3i(ceil(vec3f(p.h.w)/p.h.xyz));let count=u32(p.kernel.w);
 if((solids[index(cell)]&2u)!=0u){
 for(var z=-reach.z;z<=reach.z;z++){for(var y=-reach.y;y<=reach.y;y++){for(var x=-reach.x;x<=reach.x;x++){
  let bin=cell+vec3i(x,y,z);if(!closed(bin)){continue;}
  for(var k=0u;k<count*count*count;k++){let d=a.position.xyz-boundaryPosition(bin,k);force+=pairForce(d,-a.velocity.xyz,ri,vec2f(p.material.x,ri.y),true);}
 }}}
 }
 let n=length(normal);if(p.material.y>0.0&&n>0.01/p.h.w){force=vec4f(force.xyz-p.material.y*laplacian*normal/(ri.x*n),force.w);}
 forces[i]=force;
 if(!all(abs(force)<vec4f(1e20))){atomicStore(&status[0],1u);}
 return force;
}
var<workgroup> maxima:array<vec4f,64>;
@compute @workgroup_size(64) fn accelerations(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3u){
 let i=flat(id);var m=vec4f(0);
 if(i<p.dims.w&&stepActive()&&particles[i].position.w>0.0){let force=evaluateAcceleration(i);m=vec4f(length(particles[i].velocity.xyz),length(force.xyz),force.w,0);}
 maxima[lane]=m;workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){if(lane<stride){maxima[lane]=max(maxima[lane],maxima[lane+stride]);}workgroupBarrier();}
 let block=group.x+group.y*65535u;if(lane==0u&&block<(p.dims.w+63u)/64u){partial[block]=maxima[0];}
}
@compute @workgroup_size(64) fn chooseStep(@builtin(local_invocation_index) lane:u32){
 var m=vec4f(0);
 for(var i=lane;i<(p.dims.w+63u)/64u;i+=64u){m=max(m,partial[i]);}
 maxima[lane]=m;workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){if(lane<stride){maxima[lane]=max(maxima[lane],maxima[lane+stride]);}workgroupBarrier();}
 if(lane==0u){state[1]=0.0;if(stepActive()){
  let v=maxima[0];var dt=min(state[0],p.tuning.y*p.h.w/(p.tuning.x+v.x));
  dt=min(dt,sqrt(p.tuning.y*p.h.w/max(v.y,1e-30)));
  dt=min(dt,0.25/max(v.z,1e-30));
  if(p.material.y>0.0){dt=min(dt,0.25*sqrt(p.material.x*p.h.w*p.h.w*p.h.w/(3.14159265*p.material.y)));}
  if(p.dims.w==0u){dt=state[0];}
  if(!(dt>=1e-9)){atomicStore(&status[0],3u);}else{state[1]=dt;state[11]=dt;state[2]+=1.0;
   state[0]=max(0.0,state[0]-dt);}
 }}
}
@compute @workgroup_size(64) fn integrate(@builtin(global_invocation_id) id:vec3u){
 let i=flat(id);if(i>=p.dims.w||!healthy()||state[1]<=0.0){return;}var a=particles[i];if(a.position.w<=0.0){return;}
 let dt=state[1];var v=a.velocity.xyz+dt*forces[i].xyz;let travel=dt*v;
 if(!all(abs(travel)<vec3f(1e20))||!all(abs(v)<vec3f(1e20))){atomicStore(&status[0],1u);return;}
 let steps=u32(ceil(4.0*max(abs(travel.x/p.h.x),max(abs(travel.y/p.h.y),abs(travel.z/p.h.z)))))+1u;
 if(steps>64u){atomicStore(&status[0],2u);return;}
 var position=a.position.xyz;var hit=vec3<bool>(false);let extent=vec3f(p.dims.xyz)*p.h.xyz;
 for(var step=0u;step<steps;step++){for(var axis=0u;axis<3u;axis++){
  var trial=position;trial[axis]+=travel[axis]/f32(steps);let low=0.0001*p.h[axis];let high=extent[axis]-low;
  if(trial[axis]<low){trial[axis]=low;hit[axis]=true;}
  if(trial[axis]>high&&!(axis==1u&&p.material.z>0.0)){trial[axis]=high;hit[axis]=true;}
  if(closed(vec3i(floor(trial/p.h.xyz)))){hit[axis]=true;}else{position=trial;}
 }}
 if(position.y>=extent.y&&p.material.z>0.0){a.position.w=0.0;atomicAdd(&status[1],1u);}
 if(any(hit)){v=select(v,vec3f(0),hit);if(p.material.w>0.0){v=vec3f(0);}}
 a.position=vec4f(position,a.position.w);a.velocity=vec4f(v,0);particles[i]=a;
}
// Weighted-centroid implicit surface: a presentation estimate, not density or mass.
@compute @workgroup_size(64) fn surface(@builtin(global_invocation_id) id:vec3u){
 let i=flat(id);let dims=p.dims.xyz+vec3u(1);if(i>=dims.x*dims.y*dims.z||!healthy()){return;}
 let q=coord(i,dims);let x=vec3f(q);let base=vec3i(floor(x*p.h.xyz/p.h.w));let reach=i32(ceil(1.5*max(p.h.x,max(p.h.y,p.h.z))/p.h.w));var sum=vec3f(0);var total=0.0;
 for(var z=-reach;z<=reach;z++){for(var y=-reach;y<=reach;y++){for(var k=-reach;k<=reach;k++){
  let bin=base+vec3i(k,y,z);if(!binInside(bin)){continue;}var j=atomicLoad(&heads[binIndex(bin)]);
  while(j!=0xffffffffu){let centre=particles[j].position.xyz/p.h.xyz;
   var mirror=vec3f(0);var enabled=vec3<bool>(false);
   for(var a=0u;a<3u;a++){
    if(x[a]<1.5){enabled[a]=true;mirror[a]=-centre[a];}
    else if(x[a]>f32(p.dims[a])-1.5&&!(a==1u&&p.material.z>0.0)){enabled[a]=true;mirror[a]=2.0*f32(p.dims[a])-centre[a];}
   }
   for(var mask=0u;mask<8u;mask++){
    let reflected=vec3<bool>((mask&1u)!=0u,(mask&2u)!=0u,(mask&4u)!=0u);
    if((reflected.x&&!enabled.x)||(reflected.y&&!enabled.y)||(reflected.z&&!enabled.z)){continue;}
    let c=select(centre,mirror,reflected);let d=x-c;let t=max(0.0,1.0-dot(d,d)/2.25);let w=t*t*t;sum+=w*c;total+=w;
   }j=links[j];
  }
 }}}
 let dx=min(p.h.x,min(p.h.y,p.h.z));var value=3.0*dx;if(total>1e-20){value=(length(x-sum/total)-0.55)*dx;}phi[i]=value;
}
${levelSetFillWGSL}
@compute @workgroup_size(64) fn cellFields(@builtin(global_invocation_id) id:vec3u){
 let i=flat(id);if(i>=cells()||!healthy()){return;}let q=coord(i,p.dims.xyz);let vd=p.dims.xyz+vec3u(1);var corners:array<f32,8>;
 for(var k=0u;k<8u;k++){let c=vec3u(q)+vec3u(k&1u,(k>>1u)&1u,k>>2u);corners[k]=phi[c.x+vd.x*(c.y+vd.y*c.z)];}
 var velocity=vec3f(0);var weight=0.0;let centre=(vec3f(q)+vec3f(0.5))*p.h.xyz;let base=vec3i(floor(centre/p.h.w));
 for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
  let bin=base+vec3i(x,y,z);if(!binInside(bin)){continue;}var j=atomicLoad(&heads[binIndex(bin)]);
  while(j!=0xffffffffu){let d=centre-particles[j].position.xyz;let w=poly(dot(d,d));velocity+=w*particles[j].velocity.xyz;weight+=w;j=links[j];}
 }}}
 fields[i]=vec4f(velocity/max(weight,1e-30),select(fill(corners),0.0,closed(q)));
}
var<workgroup> sums:array<vec4f,64>;
@compute @workgroup_size(64) fn statistics(@builtin(local_invocation_index) lane:u32){
 var s=vec4f(0);var m=vec4f(0);
 for(var i=lane;i<p.dims.w;i+=64u){let a=particles[i];if(a.position.w>0.0){
  let speed=length(a.velocity.xyz);s.x+=1.0;s.y+=0.5*mass()*speed*speed;m.x=max(m.x,speed);m.y=max(m.y,density[i].x/p.material.x-1.0);
  if(!(speed<1e20)||!all(abs(a.position.xyz)<vec3f(1e20))){atomicStore(&status[0],1u);}
 }}
 for(var i=lane;i<cells();i+=64u){s.z+=fields[i].w;if(!all(abs(fields[i])<vec4f(1e20))){atomicStore(&status[0],1u);}}
 sums[lane]=s;maxima[lane]=m;workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){if(lane<stride){sums[lane]+=sums[lane+stride];maxima[lane]=max(maxima[lane],maxima[lane+stride]);}workgroupBarrier();}
 if(lane==0u){
  if(state[0]>1e-9&&healthy()){atomicStore(&status[0],4u);}
  let r=sums[0];state[4]=maxima[0].x;state[5]=maxima[0].y;state[6]=r.x;state[7]=r.x*p.kernel.z;
  state[8]=f32(atomicLoad(&status[1]))*p.kernel.z;state[9]=r.y;state[10]=r.z;state[12]=f32(atomicLoad(&status[0]));
 }
}
@compute @workgroup_size(64) fn publishPhi(@builtin(global_invocation_id) id:vec3u){let i=flat(id);let d=p.dims.xyz+vec3u(1);if(i<d.x*d.y*d.z&&healthy()){textureStore(phiTex,coord(i,d),vec4f(phi[i]));}}
@compute @workgroup_size(64) fn publishCells(@builtin(global_invocation_id) id:vec3u){let i=flat(id);if(i>=cells()||!healthy()){return;}let q=coord(i,p.dims.xyz);
 textureStore(volumeTex,q,vec4f(fields[i].w));textureStore(velocityTex,q,vec4f(fields[i].xyz,0));textureStore(openTex,q,vec4f(select(1.0,0.0,closed(q))));
}
`;
export const SPH_BINDINGS = {
  clearStepBins: [0,2,9,10], binStepParticles: [0,1,2,3,4,9,10],
  beginFrame: [0,9], clearBins: [0,2], binParticles: [0,1,2,3,4,10],
  densities: [0,1,2,3,4,5,9,10], finalDensity: [0,1,2,3,4,5,10],
  accelerations: [0,1,2,3,4,5,6,9,10,15], chooseStep: [0,9,10,15],
  integrate: [0,1,4,6,9,10],
  surface: [0,1,2,3,7,10], cellFields: [0,1,2,3,4,7,8,10],
  statistics: [0,1,5,8,9,10], publishPhi: [0,7,10,11], publishCells: [0,4,8,10,12,13,14],
} as const;
export type SphEntry = keyof typeof SPH_BINDINGS;
