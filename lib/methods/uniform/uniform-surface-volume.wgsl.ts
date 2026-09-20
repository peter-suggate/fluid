/** Experimental global surface-volume constraint. No cellwise reconstruction. */
export const uniformSurfaceVolumeWGSL = /* wgsl */ `
struct Params { dims:vec4u, h:vec4f }
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var phi:texture_3d<f32>;
@group(0) @binding(2) var volume:texture_3d<f32>;
@group(0) @binding(3) var capacity:texture_3d<f32>;
@group(0) @binding(4) var output:texture_storage_3d<r32float,write>;
@group(0) @binding(5) var<storage,read_write> band:array<atomic<u32>>;
@group(0) @binding(6) var<storage,read_write> nextBand:array<atomic<u32>>;
@group(0) @binding(7) var<storage,read_write> partial:array<vec4f>;
@group(0) @binding(8) var<storage,read_write> reduced:array<vec4f>;
// centre, half-range, desired V, surface volume before correction; iteration.
@group(0) @binding(9) var<storage,read_write> state:array<vec4f>;
var<workgroup> sums:array<vec4f,320>;
fn groupIndex(w:vec3u)->u32{return w.x+w.y*65535u;}
fn cells()->u32{return p.dims.x*p.dims.y*p.dims.z;}
fn vertices()->u32{return (p.dims.x+1u)*(p.dims.y+1u)*(p.dims.z+1u);}
fn point(i:u32,d:vec3u)->vec3i{return vec3i(vec3u(i%d.x,(i/d.x)%d.y,i/(d.x*d.y)));}
fn index(q:vec3i,d:vec3u)->u32{return u32(q.x)+d.x*(u32(q.y)+d.y*u32(q.z));}
fn corner(k:u32)->vec3i{return vec3i(i32(k&1u),i32((k>>1u)&1u),i32((k>>2u)&1u));}
fn value(q:vec3i)->f32{return textureLoad(phi,clamp(q,vec3i(0),vec3i(p.dims.xyz)),0).x;}
@compute @workgroup_size(1) fn begin(){state[0]=vec4f(0,p.h.w,0,0);state[1]=vec4f(0);}
@compute @workgroup_size(64) fn seed(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)l:u32){
 let i=groupIndex(w)*64u+l;if(i>=cells()){return;}let q=point(i,p.dims.xyz);
 if(textureLoad(capacity,q,0).x<=0.0){return;}
 var lo=1e30;var hi=-1e30;
 for(var k=0u;k<8u;k++){let v=value(q+corner(k));lo=min(lo,v);hi=max(hi,v);}
 if(lo<=0.0&&hi>=0.0){for(var k=0u;k<8u;k++){atomicMax(&band[index(q+corner(k),p.dims.xyz+vec3u(1))],5u);}}
}
@compute @workgroup_size(64) fn dilate(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)l:u32){
 let i=groupIndex(w)*64u+l;if(i>=vertices()){return;}let d=p.dims.xyz+vec3u(1);let q=point(i,d);
 var b=atomicLoad(&band[i]);
 for(var a=0u;a<3u;a++){for(var s=-1;s<=1;s+=2){var n=q;n[a]+=s;
  if(all(n>=vec3i(0))&&all(n<vec3i(d))){let v=atomicLoad(&band[index(n,d)]);b=max(b,select(0u,v-1u,v>0u));}
 }}atomicStore(&nextBand[i],b);
}
@compute @workgroup_size(64) fn metric(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)l:u32){
 let i=groupIndex(w)*64u+l;if(i>=vertices()){return;}let q=point(i,p.dims.xyz+vec3u(1));var gradient=vec3f(0);
 for(var a=0u;a<3u;a++){var lo=q;var hi=q;lo[a]=max(0,q[a]-1);hi[a]=min(i32(p.dims[a]),q[a]+1);
  gradient[a]=(value(hi)-value(lo))/(f32(hi[a]-lo[a])*p.h[a]);}
 atomicStore(&band[i],bitcast<u32>(f32(atomicLoad(&band[i]))*0.2*max(0.1,length(gradient))));
}
// Exact negative volume of a linear scalar on a tetrahedron, normalized by
// tetrahedron volume. Crossing-edge ratios avoid repeated-value singularities.
fn tetra(v:vec4f)->f32{
 var n:array<f32,4>;var o:array<f32,4>;var count=0u;var outside=0u;
 for(var k=0u;k<4u;k++){if(v[k]<0.0){n[count]=v[k];count++;}else{o[outside]=v[k];outside++;}}
 if(count==0u){return 0.0;}if(count==4u){return 1.0;}
 if(count==1u){return (-n[0]/(o[0]-n[0]))*(-n[0]/(o[1]-n[0]))*(-n[0]/(o[2]-n[0]));}
 if(count==3u){return 1.0-(o[0]/(o[0]-n[0]))*(o[0]/(o[0]-n[1]))*(o[0]/(o[0]-n[2]));}
 let a=-n[0]/(o[0]-n[0]);let b=-n[0]/(o[1]-n[0]);let c=-n[1]/(o[0]-n[1]);let d=-n[1]/(o[1]-n[1]);
 return clamp(a*b+b*c*(1.0-a)+c*d*(1.0-b),0.0,1.0);
}
fn fill(v:array<f32,8>)->f32{
 return (tetra(vec4f(v[0],v[1],v[3],v[7]))+tetra(vec4f(v[0],v[1],v[5],v[7]))
 +tetra(vec4f(v[0],v[2],v[3],v[7]))+tetra(vec4f(v[0],v[2],v[6],v[7]))
 +tetra(vec4f(v[0],v[4],v[5],v[7]))+tetra(vec4f(v[0],v[4],v[6],v[7])))/6.0;
}
fn sumGroup(l:u32){
 workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){if(l<stride){for(var k=0u;k<5u;k++){sums[l*5u+k]+=sums[(l+stride)*5u+k];}}workgroupBarrier();}
}
@compute @workgroup_size(64) fn measure(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)l:u32){
 let group=groupIndex(w);let i=group*64u+l;var result:array<vec4f,5>;
 if(i<cells()){
  let q=point(i,p.dims.xyz);let cap=textureLoad(capacity,q,0).x;
  result[4].y=textureLoad(volume,q,0).x;
  if(cap>0.0){
   var raw:array<f32,8>;var scale:array<f32,8>;var lo=1e30;var hi=-1e30;
   for(var k=0u;k<8u;k++){let v=q+corner(k);raw[k]=value(v);scale[k]=bitcast<f32>(atomicLoad(&band[index(v,p.dims.xyz+vec3u(1))]));
    lo=min(lo,raw[k]-(state[0].x+state[0].y)*scale[k]);hi=max(hi,raw[k]-(state[0].x-state[0].y)*scale[k]);}
   for(var sample=0u;sample<17u;sample++){
    var fraction=0.0;
    if(hi<0.0){fraction=1.0;}else if(lo<0.0){
     let shift=state[0].x+(f32(sample)/8.0-1.0)*state[0].y;var v:array<f32,8>;
     for(var k=0u;k<8u;k++){v[k]=raw[k]-shift*scale[k];}fraction=fill(v);
    }
    result[sample/4u][sample%4u]=fraction*cap;
   }
  }
 }
 for(var k=0u;k<5u;k++){sums[l*5u+k]=result[k];}sumGroup(l);
 if(l==0u){for(var k=0u;k<5u;k++){partial[group*5u+k]=sums[k];}}
}
@compute @workgroup_size(64) fn reduce(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)l:u32){
 let group=groupIndex(w);let i=group*64u+l;let count=(cells()+63u)/64u;
 for(var k=0u;k<5u;k++){sums[l*5u+k]=vec4f(0);if(i<count){sums[l*5u+k]=partial[i*5u+k];}}sumGroup(l);
 if(l==0u){for(var k=0u;k<5u;k++){reduced[group*5u+k]=sums[k];}}
}
@compute @workgroup_size(64) fn solve(@builtin(local_invocation_index)l:u32){
 let count=(cells()+4095u)/4096u;
 for(var k=0u;k<5u;k++){var sum=vec4f(0);for(var i=l;i<count;i+=64u){sum+=reduced[i*5u+k];}sums[l*5u+k]=sum;}sumGroup(l);
 if(l==0u){
  let desired=sums[4].y;var shift=state[0].x;let width=state[0].y;
  if(state[1].x==0.0){state[0].w=sums[2].x;}
  if(sums[4].x-sums[0].x>1e-6 && abs(sums[2].x-desired)>max(1e-5,1e-7*abs(desired))){
   shift=state[0].x+select(-width,width,desired>sums[4].x);
   for(var k=0u;k<16u;k++){let a=sums[k/4u][k%4u];let b=sums[(k+1u)/4u][(k+1u)%4u];
    if(desired>=a&&desired<=b&&b>a){shift=state[0].x+(f32(k)+clamp((desired-a)/(b-a),0.0,1.0)-8.0)*width/8.0;break;}}
  }
  state[0].x=clamp(shift,-p.h.w,p.h.w);state[0].y=width/8.0;state[0].z=desired;state[1].x+=1.0;
 }
}
@compute @workgroup_size(64) fn apply(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)l:u32){
 let i=groupIndex(w)*64u+l;if(i>=vertices()){return;}let q=point(i,p.dims.xyz+vec3u(1));
 textureStore(output,q,vec4f(value(q)-state[0].x*bitcast<f32>(atomicLoad(&band[i]))));
}
`;
