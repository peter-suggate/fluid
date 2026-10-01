/** Shared exact tetrahedral volume curve for the global surface constraint. */
export const uniformSurfaceFillWGSL = /* wgsl */ `
// Adjacent pair, an outside value (not negative) after a negative one.
fn tetraOrder(a:f32,b:f32)->vec2f{return select(vec2f(a,b),vec2f(b,a),!(a<0.0)&&b<0.0);}
fn tetra(v:vec4f)->f32{
 // Stable partition, negatives first, by a fixed odd-even transposition
 // network: s holds the negatives then the outside values in their order,
 // in registers (no dynamically indexed private arrays).
 var s=v;var p=tetraOrder(s.x,s.y);s.x=p.x;s.y=p.y;p=tetraOrder(s.z,s.w);s.z=p.x;s.w=p.y;
 p=tetraOrder(s.y,s.z);s.y=p.x;s.z=p.y;
 p=tetraOrder(s.x,s.y);s.x=p.x;s.y=p.y;p=tetraOrder(s.z,s.w);s.z=p.x;s.w=p.y;
 p=tetraOrder(s.y,s.z);s.y=p.x;s.z=p.y;
 let count=dot(select(vec4u(0u),vec4u(1u),v<vec4f(0.0)),vec4u(1u));
 if(count==0u){return 0.0;}if(count==4u){return 1.0;}
 if(count==1u){return (-s.x/(s.y-s.x))*(-s.x/(s.z-s.x))*(-s.x/(s.w-s.x));}
 if(count==3u){return 1.0-(s.w/(s.w-s.x))*(s.w/(s.w-s.y))*(s.w/(s.w-s.z));}
 let a=-s.x/(s.z-s.x);let b=-s.x/(s.w-s.x);let c=-s.y/(s.z-s.y);let d=-s.y/(s.w-s.y);
 return clamp(a*b+b*c*(1.0-a)+c*d*(1.0-b),0.0,1.0);
}
fn fill(v:array<f32,8>)->f32{
 return (tetra(vec4f(v[0],v[1],v[3],v[7]))+tetra(vec4f(v[0],v[1],v[5],v[7]))
 +tetra(vec4f(v[0],v[2],v[3],v[7]))+tetra(vec4f(v[0],v[2],v[6],v[7]))
 +tetra(vec4f(v[0],v[4],v[5],v[7]))+tetra(vec4f(v[0],v[4],v[6],v[7])))/6.0;
}
`;

import { uniformAbOn } from "./uniform-ab-switch";

const leanMeasure = uniformAbOn("measurelean");
const deadGroups = uniformAbOn("deadgroups");
const measureSamples = uniformAbOn("measuresamples");
/** Experimental global surface-volume constraint. No cellwise reconstruction. */
/** `receiverBase` (compact only): word of the transport receiver list's count
 * inside binding 11. Given, the seed runs over the phi window and those tiles. */
export function createUniformSurfaceVolumeWGSL(compact: boolean, capacityComponent: "x" | "w" = "x", receiverBase?: number): string {
 const windowed = compact && receiverBase !== undefined;
 return /* wgsl */ `
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
${windowed ? `// surfaceseed. A seed-live cell has V != 0 or a phi corner <= 0. Outside the
// receiver tiles V is start-of-step V, zero outside the phi census box
// (work[28..33], copied from the phi region). Phi is written only on the
// census box's vertex window (VERTEX_PHI_REACH = 6) and was at or above the 4h
// band everywhere else. So every live cell is in a receiver tile or touches the
// window with a corner; the tiles pass only visits cells the box pass did not.
@group(0) @binding(11) var<storage,read> receivers:array<u32>;
const RECEIVER_BASE:u32=${receiverBase}u;
fn seedBoxOrigin()->vec3u{return vec3u(atomicLoad(&work[34]),atomicLoad(&work[35]),atomicLoad(&work[36]));}
fn seedBoxDims()->vec3u{return vec3u(atomicLoad(&work[37]),atomicLoad(&work[38]),atomicLoad(&work[39]));}
fn planSeedBox(){
 let o=vec3u(atomicLoad(&work[28]),atomicLoad(&work[29]),atomicLoad(&work[30]));
 let m=vec3u(atomicLoad(&work[31]),atomicLoad(&work[32]),atomicLoad(&work[33]));
 var lo=vec3u(0);var hi=p.dims.xyz;
 if(all(m>o)&&all(m<=p.dims.xyz)){
  let low=o-min(o,vec3u(6u));let high=min(p.dims.xyz,m+vec3u(6u));
  lo=low-min(low,vec3u(1u));hi=min(p.dims.xyz,high+vec3u(1u));
 }
 let d=hi-lo;let n=d.x*d.y*d.z;let groups=(n+63u)/64u;
 for(var a=0u;a<3u;a++){atomicStore(&work[34u+a],lo[a]);atomicStore(&work[37u+a],d[a]);}
 atomicStore(&work[40],n);atomicStore(&work[41],min(groups,65535u));atomicStore(&work[42],(groups+65534u)/65535u);atomicStore(&work[43],1u);
}` : ""}
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
 // balancesupport: the 4³ tiles of this box grown one cell, the cells whose
 // corners apply may move, as origin, dims and an indirect dispatch.
 var to=vec3u(0);var te=vec3u(0);
 if(nc>0u){for(var a=0u;a<3u;a++){
  let c0=lo[a]-min(lo[a],1u);let c1=min(p.dims[a],lo[a]+d[a]+1u);to[a]=c0/4u;te[a]=(c1+3u)/4u-to[a];
 }}
 let tiles=te.x*te.y*te.z;
 for(var a=0u;a<3u;a++){atomicStore(&work[44u+a],to[a]);atomicStore(&work[47u+a],te[a]);}
 atomicStore(&work[50],min(tiles,65535u));atomicStore(&work[51],(tiles+65534u)/65535u);atomicStore(&work[52],1u);
 ${windowed ? `// classifysupport: that box joined with the seed box (cells touching the phi
 // window), in 4³ tiles, at work[53..58] with its dispatch at work[59..61].
 var u0=seedBoxOrigin();var u1=u0+seedBoxDims();
 if(nc>0u){for(var a=0u;a<3u;a++){u0[a]=min(u0[a],lo[a]-min(lo[a],1u));u1[a]=max(u1[a],min(p.dims[a],lo[a]+d[a]+1u));}}
 var unionTiles=1u;
 for(var a=0u;a<3u;a++){let t0=u0[a]/4u;let t1=select(t0,(u1[a]+3u)/4u,u1[a]>u0[a]);
  atomicStore(&work[53u+a],t0);atomicStore(&work[56u+a],t1-t0);unionTiles*=t1-t0;}
 atomicStore(&work[59],min(unionTiles,65535u));atomicStore(&work[60],(unionTiles+65534u)/65535u);atomicStore(&work[61],1u);` : ""}
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
 ${windowed ? "planSeedBox();" : ""}
}
${compact ? `fn seedCell(q:vec3i,valid:bool,l:u32){
 if(l<3u){atomicStore(&bounds[l],p.dims[l]);}workgroupBarrier();
 if(valid){
  let i=index(q,p.dims.xyz);var live=textureLoad(volume,q,0).x!=0.0;
  if(textureLoad(capacity,q,0).${capacityComponent}>0.0){
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
 }
}` : ""}
${windowed ? `@compute @workgroup_size(64) fn seedBox(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)l:u32){
 let i=groupIndex(w)*64u+l;let valid=i<atomicLoad(&work[40]);
 seedCell(vec3i(seedBoxOrigin())+point(select(0u,i,valid),max(seedBoxDims(),vec3u(1))),valid,l);
}
@compute @workgroup_size(64) fn seedTiles(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)l:u32){
 let n=groupIndex(w);var valid=n<receivers[RECEIVER_BASE];
 let tile=select(0u,receivers[RECEIVER_BASE+1u+n],valid);let d=(p.dims.xyz+vec3u(3u))/4u;
 let q=vec3u(tile%d.x,(tile/d.x)%d.y,tile/(d.x*d.y))*4u+vec3u(l%4u,(l/4u)%4u,l/16u);
 let lo=seedBoxOrigin();valid=valid&&all(q<p.dims.xyz)&&!(all(q>=lo)&&all(q<lo+seedBoxDims()));
 seedCell(vec3i(q),valid,l);
}` : ""}
@compute @workgroup_size(64) fn seed(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)l:u32){
 ${compact ? `let i=groupIndex(w)*64u+l;seedCell(point(select(0u,i,i<cells()),p.dims.xyz),i<cells(),l);` : `let i=groupIndex(w)*64u+l;if(i>=cells()){return;}let q=point(i,p.dims.xyz);
 if(textureLoad(capacity,q,0).${capacityComponent}<=0.0){return;}
 var lo=1e30;var hi=-1e30;
 for(var k=0u;k<8u;k++){let v=value(q+corner(k));lo=min(lo,v);hi=max(hi,v);}
 if(lo<=0.0&&hi>=0.0){atomicStore(&band[i],5u);}`}
}
@compute @workgroup_size(64) fn dilate(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)l:u32){
 ${compact ? `let logical=groupIndex(w)*64u+l;if(logical>=workCells()){return;}
 let q=cellPoint(logical);let i=index(q,p.dims.xyz);`
 : `let i=groupIndex(w)*64u+l;if(i>=cells()){return;}let q=point(i,p.dims.xyz);`}
 var b=0u;
 if(textureLoad(capacity,q,0).${capacityComponent}>0.0){
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
   if(textureLoad(capacity,c,0).${capacityComponent}>0.0){for(var a=0u;a<3u;a++){open[2u*a+((k>>a)&1u)]=true;}}}}
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
${uniformSurfaceFillWGSL}
fn sumGroup(l:u32){
 workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){if(l<stride){for(var k=0u;k<5u;k++){sums[l*5u+k]+=sums[(l+stride)*5u+k];}}workgroupBarrier();}
}
@compute @workgroup_size(64) fn measure(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)l:u32){
 let group=groupIndex(w);let i=group*64u+l;var result:array<vec4f,5>;
 if(i<${compact ? "workCells()" : "cells()"}){
  let q=${compact ? "cellPoint(i)" : "point(i,p.dims.xyz)"};let cap=textureLoad(capacity,q,0).${capacityComponent};
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
     ${measureSamples ? `// One strict sign across this sample's eight corners is what fill() returns
     // anyway: six whole tetrahedra sum to exactly 6/6, none to 0.
     var negative=0u;
     for(var k=0u;k<8u;k++){v[k]=raw[k]-shift*scale[k];negative+=select(0u,1u,v[k]<0.0);}
     if(negative==8u){fraction=1.0;}else if(negative!=0u){fraction=fill(v);}` : `for(var k=0u;k<8u;k++){v[k]=raw[k]-shift*scale[k];}fraction=fill(v);`}
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
