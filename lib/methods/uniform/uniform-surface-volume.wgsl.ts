import { uniformAbOn } from "./uniform-ab-switch";

const leanMeasure = uniformAbOn("measurelean");
const deadGroups = uniformAbOn("deadgroups");
/** Experimental global surface-volume constraint. No cellwise reconstruction. */
export function createUniformSurfaceVolumeWGSL(compact: boolean): string { return /* wgsl */ `
struct Params { dims:vec4u, h:vec4f }
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var phi:texture_3d<f32>;
@group(0) @binding(2) var volume:texture_3d<f32>;
@group(0) @binding(3) var capacity:texture_3d<f32>;
@group(0) @binding(4) var output:texture_storage_3d<r32float,write>;
// The band lives on cells with capacity, never on vertices: seed and the four
// dilations index cells, and metric folds the eight incident cells into one
// per-vertex scale in the other parity buffer. A solid cell neither carries nor
// passes the band, so the shift can only reach vertices some measured cell
// reads. Dilating the vertex lattice let a surface deficit push the contour
// through a voxel wall, where it added no measured fill and never converged.
@group(0) @binding(5) var<storage,read_write> band:array<atomic<u32>>;
@group(0) @binding(6) var<storage,read_write> nextBand:array<atomic<u32>>;
@group(0) @binding(7) var<storage,read_write> partial:array<vec4f>;
@group(0) @binding(8) var<storage,read_write> reduced:array<vec4f>;
// centre, half-range, desired V, surface volume before correction; iteration.
@group(0) @binding(9) var<storage,read_write> state:array<vec4f>;
${compact ? `// Bounds include every target-volume contribution and every cell with a
// non-positive phi corner. Five cells of padding enclose the four-step band
// dilation, including crossing cells' upper vertices. No liquid is omitted
// from the global constraint, even if it is disconnected or sleeping.
@group(0) @binding(10) var<storage,read_write> work:array<atomic<u32>>;
var<workgroup> bounds:array<atomic<u32>,6>;
fn workCells()->u32{return atomicLoad(&work[11]);}
fn workVertices()->u32{return atomicLoad(&work[15]);}
fn workDims()->vec3u{return vec3u(atomicLoad(&work[12]),atomicLoad(&work[13]),atomicLoad(&work[14]));}
fn workOrigin()->vec3i{return vec3i(i32(atomicLoad(&work[8])),i32(atomicLoad(&work[9])),i32(atomicLoad(&work[10])));}
fn cellPoint(i:u32)->vec3i{return workOrigin()+point(i,workDims());}
fn vertexPoint(i:u32)->vec3i{return workOrigin()+point(i,workDims()+vec3u(1));}
@compute @workgroup_size(1) fn finishWork(){
 var lo=vec3u(0);var d=vec3u(0);var nc=0u;var nv=0u;
 if(atomicLoad(&work[4])>0u){
  for(var a=0u;a<3u;a++){
   lo[a]=u32(max(0,i32(atomicLoad(&work[a]))-5));
   d[a]=min(p.dims[a],atomicLoad(&work[4u+a])+5u)-lo[a];
  }
  nc=d.x*d.y*d.z;nv=(d.x+1u)*(d.y+1u)*(d.z+1u);
 }
 for(var a=0u;a<3u;a++){atomicStore(&work[8u+a],lo[a]);atomicStore(&work[12u+a],d[a]);}
 atomicStore(&work[11],nc);atomicStore(&work[15],nv);
 let counts=array<u32,3>((nc+63u)/64u,(nv+63u)/64u,(nc+4095u)/4096u);
 for(var k=0u;k<3u;k++){
  atomicStore(&work[16u+4u*k],min(counts[k],65535u));
  atomicStore(&work[17u+4u*k],(counts[k]+65534u)/65535u);
  atomicStore(&work[18u+4u*k],1u);
 }
}` : ""}
var<workgroup> sums:array<vec4f,320>;
var<workgroup> liveLanes:atomic<u32>;
fn groupIndex(w:vec3u)->u32{return w.x+w.y*65535u;}
fn cells()->u32{return p.dims.x*p.dims.y*p.dims.z;}
fn vertices()->u32{return (p.dims.x+1u)*(p.dims.y+1u)*(p.dims.z+1u);}
fn point(i:u32,d:vec3u)->vec3i{return vec3i(vec3u(i%d.x,(i/d.x)%d.y,i/(d.x*d.y)));}
fn index(q:vec3i,d:vec3u)->u32{return u32(q.x)+d.x*(u32(q.y)+d.y*u32(q.z));}
fn corner(k:u32)->vec3i{return vec3i(i32(k&1u),i32((k>>1u)&1u),i32((k>>2u)&1u));}
fn value(q:vec3i)->f32{return textureLoad(phi,clamp(q,vec3i(0),vec3i(p.dims.xyz)),0).x;}
@compute @workgroup_size(1) fn begin(){state[0]=vec4f(0,p.h.w,0,0);state[1]=vec4f(0);
 ${compact ? `for(var a=0u;a<3u;a++){atomicStore(&work[a],p.dims[a]);atomicStore(&work[4u+a],0u);}` : ""}
}
@compute @workgroup_size(64) fn seed(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)l:u32){
 ${compact ? `if(l<3u){atomicStore(&bounds[l],p.dims[l]);}workgroupBarrier();
 let i=groupIndex(w)*64u+l;
 if(i<cells()){
  let q=point(i,p.dims.xyz);var live=textureLoad(volume,q,0).x!=0.0;
  if(textureLoad(capacity,q,0).x>0.0){
   var lo=1e30;var hi=-1e30;
   for(var k=0u;k<8u;k++){let v=value(q+corner(k));lo=min(lo,v);hi=max(hi,v);}
   live=live||lo<=0.0;
   if(lo<=0.0&&hi>=0.0){atomicStore(&band[i],5u);}
  }
  if(live){for(var a=0u;a<3u;a++){atomicMin(&bounds[a],u32(q[a]));atomicMax(&bounds[3u+a],u32(q[a])+1u);}}
 }
 workgroupBarrier();
 if(l<3u&&atomicLoad(&bounds[3])>0u){
  atomicMin(&work[l],atomicLoad(&bounds[l]));atomicMax(&work[4u+l],atomicLoad(&bounds[3u+l]));
 }` : `let i=groupIndex(w)*64u+l;if(i>=cells()){return;}let q=point(i,p.dims.xyz);
 if(textureLoad(capacity,q,0).x<=0.0){return;}
 var lo=1e30;var hi=-1e30;
 for(var k=0u;k<8u;k++){let v=value(q+corner(k));lo=min(lo,v);hi=max(hi,v);}
 if(lo<=0.0&&hi>=0.0){atomicStore(&band[i],5u);}`}
}
@compute @workgroup_size(64) fn dilate(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)l:u32){
 ${compact ? `let logical=groupIndex(w)*64u+l;if(logical>=workCells()){return;}
 let q=cellPoint(logical);let i=index(q,p.dims.xyz);`
 : `let i=groupIndex(w)*64u+l;if(i>=cells()){return;}let q=point(i,p.dims.xyz);`}
 var b=0u;
 if(textureLoad(capacity,q,0).x>0.0){
  b=atomicLoad(&band[i]);
  for(var a=0u;a<3u;a++){for(var s=-1;s<=1;s+=2){var n=q;n[a]+=s;
   if(all(n>=vec3i(0))&&all(n<vec3i(p.dims.xyz))){let v=atomicLoad(&band[index(n,p.dims.xyz)]);b=max(b,select(0u,v-1u,v>0u));}
  }}
 }
 atomicStore(&nextBand[i],b);
}
// Per-vertex shift scale, written to the other parity buffer: the cell band it
// reads and the vertex scale it writes do not share an index space.
@compute @workgroup_size(64) fn metric(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)l:u32){
 ${compact ? `let logical=groupIndex(w)*64u+l;if(logical>=workVertices()){return;}
 let q=vertexPoint(logical);let i=index(q,p.dims.xyz+vec3u(1));`
 : `let i=groupIndex(w)*64u+l;if(i>=vertices()){return;}let q=point(i,p.dims.xyz+vec3u(1));`}
 var b=0u;var open=array<bool,6>(false,false,false,false,false,false);
 for(var k=0u;k<8u;k++){let c=q-vec3i(1)+corner(k);
  if(all(c>=vec3i(0))&&all(c<vec3i(p.dims.xyz))){b=max(b,atomicLoad(&band[index(c,p.dims.xyz)]));
   if(textureLoad(capacity,c,0).x>0.0){for(var a=0u;a<3u;a++){open[2u*a+((k>>a)&1u)]=true;}}}}
 // The slope is read by centred differences along axes whose edges are both
 // open. A vertex buried in a solid keeps its construction fill, metres above
 // the redistance band, and differenced raw it scaled a wall vertex's shift by
 // that fill over one cell. No one-sided fallback: that reads the vertex's own
 // value, so its scale would grow with every shift it received.
 var gradient=vec3f(0);
 for(var a=0u;a<3u;a++){var lo=q;var hi=q;lo[a]=max(0,q[a]-1);hi[a]=min(i32(p.dims[a]),q[a]+1);
  if(open[2u*a]&&open[2u*a+1u]){gradient[a]=(value(hi)-value(lo))/(f32(hi[a]-lo[a])*p.h[a]);}}
 atomicStore(&nextBand[i],bitcast<u32>(f32(b)*0.2*max(0.1,length(gradient))));
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
 if(i<${compact ? "workCells()" : "cells()"}){
  let q=${compact ? "cellPoint(i)" : "point(i,p.dims.xyz)"};let cap=textureLoad(capacity,q,0).x;
  result[4].y=textureLoad(volume,q,0).x;
  if(cap>0.0){
   ${leanMeasure ? `// A crossing cell carries the band itself, so all eight of its corners scale
   // above zero; a cell whose corners are all zero has one strict sign and no
   // shift can move it: one phi load decides all 17 samples. That is ~97% of a
   // 128^3 lattice.
   var scale:array<f32,8>;var banded=0u;
   for(var k=0u;k<8u;k++){let bits=atomicLoad(&band[index(q+corner(k),p.dims.xyz+vec3u(1))]);banded|=bits;scale[k]=bitcast<f32>(bits);}
   if(banded==0u){
    let filled=select(0.0,cap,value(q)<0.0);
    for(var sample=0u;sample<17u;sample++){result[sample/4u][sample%4u]=filled;}
   }else{
   let centre=state[0].x;let width=state[0].y;
   var raw:array<f32,8>;var lo=1e30;var hi=-1e30;
   for(var k=0u;k<8u;k++){raw[k]=value(q+corner(k));
    lo=min(lo,raw[k]-(centre+width)*scale[k]);hi=max(hi,raw[k]-(centre-width)*scale[k]);}
   for(var sample=0u;sample<17u;sample++){
    var fraction=0.0;
    if(hi<0.0){fraction=1.0;}else if(lo<0.0){
     let shift=centre+(f32(sample)/8.0-1.0)*width;var v:array<f32,8>;
     for(var k=0u;k<8u;k++){v[k]=raw[k]-shift*scale[k];}fraction=fill(v);
    }
    result[sample/4u][sample%4u]=fraction*cap;
   }
   }` : `var raw:array<f32,8>;var scale:array<f32,8>;var lo=1e30;var hi=-1e30;
   for(var k=0u;k<8u;k++){let v=q+corner(k);raw[k]=value(v);scale[k]=bitcast<f32>(atomicLoad(&band[index(v,p.dims.xyz+vec3u(1))]));
    lo=min(lo,raw[k]-(state[0].x+state[0].y)*scale[k]);hi=max(hi,raw[k]-(state[0].x-state[0].y)*scale[k]);}
   for(var sample=0u;sample<17u;sample++){
    var fraction=0.0;
    if(hi<0.0){fraction=1.0;}else if(lo<0.0){
     let shift=state[0].x+(f32(sample)/8.0-1.0)*state[0].y;var v:array<f32,8>;
     for(var k=0u;k<8u;k++){v[k]=raw[k]-shift*scale[k];}fraction=fill(v);
    }
    result[sample/4u][sample%4u]=fraction*cap;
   }`}
  }
 }
 ${deadGroups ? `// Air, and any 64-cell run with nothing in it, reduces sixty-four +0 records to
 // +0 through six barriers. Bits, not values: a -0 anywhere takes the tree.
 var bits=0u;for(var k=0u;k<5u;k++){let b=bitcast<vec4u>(result[k]);bits|=b.x|b.y|b.z|b.w;}
 if(bits!=0u){atomicStore(&liveLanes,1u);}
 if(workgroupUniformLoad(&liveLanes)==0u){
  if(l==0u){for(var k=0u;k<5u;k++){partial[group*5u+k]=vec4f(0);}}
  return;
 }` : ""}
 for(var k=0u;k<5u;k++){sums[l*5u+k]=result[k];}sumGroup(l);
 if(l==0u){for(var k=0u;k<5u;k++){partial[group*5u+k]=sums[k];}}
}
@compute @workgroup_size(64) fn reduce(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)l:u32){
 let group=groupIndex(w);let i=group*64u+l;let count=(${compact ? "workCells()" : "cells()"}+63u)/64u;
 for(var k=0u;k<5u;k++){sums[l*5u+k]=vec4f(0);if(i<count){sums[l*5u+k]=partial[i*5u+k];}}sumGroup(l);
 if(l==0u){for(var k=0u;k<5u;k++){reduced[group*5u+k]=sums[k];}}
}
@compute @workgroup_size(64) fn solve(@builtin(local_invocation_index)l:u32){
 let count=(${compact ? "workCells()" : "cells()"}+4095u)/4096u;
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
}

/** Dense control, also usable as a standalone shader fixture. */
export const uniformSurfaceVolumeWGSL = createUniformSurfaceVolumeWGSL(false);
