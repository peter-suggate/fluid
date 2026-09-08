import { gpuCompilationManagerFor } from "../../lib/core/gpu-compilation-manager";

/** Research prototype. No production solver imports this module.
 *
 * A fixed Cartesian support stores a CURRENT quadratic in metres:
 * phi(x)=c+g·(x-origin)+0.5*(x-origin)^T H (x-origin).
 * All connected valid supports must describe the same global quadratic. We
 * check that stronger contract, including H, rather than pretending unrelated
 * local fits constitute a continuous field. General branches are unsupported.
 *
 * An affine step is a departure map x_old=B*x_new+t. Every cell in a
 * conservative AABB of the complete mapped destination support is checked.
 * Missing coverage, too many donors, or inconsistent coefficients rejects the
 * generation. There is no extrapolation, boundary clamping, mass correction,
 * initial-field resampling, or surface repair.
 *
 * Scope: only prescribed volume-preserving affine maps are admitted. Before
 * publication, every potentially wet source support's full forward footprint
 * must be in the candidate worklist. The conservative range/footprint test can
 * require extra dry halo; a sampled zero integral never authorizes omission.
 * Advance publishes coefficients/coherence and spatial coverage;
 * integrate is a SEPARATE diagnostic operation and may reject afterwards.
 * This is not a combined field-and-mass transaction or a production transport
 * implementation. Integral assertions apply to the independently enclosed
 * plane/sphere/quadric fixtures in the accompanying tests.
 */
export type V3 = readonly [number, number, number];
export type M3 = readonly [number, number, number, number, number, number, number, number, number];
/** c,gx,gy,gz,Hxx,Hyy,Hzz,Hxy,Hxz,Hyz. */
export type Quadratic = readonly [number, number, number, number, number, number, number, number, number, number];
export interface QuadraticSupportGrid { origin: V3; dimensions: V3; h: number }
export interface AffineDeparture { matrix: M3; translation: V3 }
export interface CanonicalAffineDeparture { departure: AffineDeparture; forward: AffineDeparture }
export const IDENTITY_MATRIX: M3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];
export const QUADRATIC_PATCH_WORDS = 16;
const MIN_NORMAL_F32 = 2 ** -126;
const MAX_AFFINE_COEFFICIENT = 8;

/** Validate the actual f32 map uploaded to the GPU, then derive its inverse.
 * This is constant-size operation data, not CPU cell/geometry expansion. */
export function canonicalAffineDeparture(map: AffineDeparture): CanonicalAffineDeparture {
  if (map.matrix.length !== 9 || map.translation.length !== 3) throw new Error("Invalid or unsupported prescribed departure map");
  const m = map.matrix.map(Math.fround) as unknown as M3, t = map.translation.map(Math.fround) as unknown as V3;
  const [a, b, c, d, e, f, g, h, i] = m;
  const determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (![...m, ...t].every(Number.isFinite) || Math.abs(determinant - 1) > 1e-6) {
    throw new Error("Invalid or unsupported prescribed departure map");
  }
  const inverse = [e * i - f * h, c * h - b * i, b * f - c * e,
    f * g - d * i, a * i - c * g, c * d - a * f,
    d * h - e * g, b * g - a * h, a * e - b * d].map(value => Math.fround(value / determinant)) as unknown as M3;
  const translation = [0, 1, 2].map(row => Math.fround(-(inverse[3 * row]! * t[0]
    + inverse[3 * row + 1]! * t[1] + inverse[3 * row + 2]! * t[2]))) as unknown as V3;
  if (![...inverse, ...translation].every(Number.isFinite)
    || [...m, ...inverse].some(value => Math.abs(value) > MAX_AFFINE_COEFFICIENT)) {
    throw new Error("Invalid or unsupported inverse departure map (coefficient envelope8)");
  }
  return { departure: { matrix: m, translation: t }, forward: { matrix: inverse, translation } };
}

export function supportCount(grid: QuadraticSupportGrid): number {
  if (!(grid.h > 0) || !Number.isFinite(grid.h) || !grid.origin.every(Number.isFinite)
    || !grid.dimensions.every(value => Number.isSafeInteger(value) && value > 0)) {
    throw new Error("Invalid quadratic support grid");
  }
  const h = Math.fround(grid.h), volume = Math.fround(h * h * h);
  if (!Number.isFinite(h) || !(volume >= MIN_NORMAL_F32) || !Number.isFinite(volume)
    || grid.origin.some((value, axis) => !Number.isFinite(Math.fround(value))
      || Math.abs(Math.fround(value) / h) + grid.dimensions[axis]! > 1_048_576)) {
    throw new Error("Unsupported f32 quadratic grid envelope");
  }
  const count = grid.dimensions[0] * grid.dimensions[1] * grid.dimensions[2];
  if (count > 1_048_576) throw new Error("Prototype support capacity exceeds 1048576");
  return count;
}
export function supportOrigin(grid: QuadraticSupportGrid, id: number): V3 {
  const [nx, ny] = grid.dimensions;
  return [grid.origin[0] + (id % nx) * grid.h,
    grid.origin[1] + (Math.floor(id / nx) % ny) * grid.h,
    grid.origin[2] + Math.floor(id / (nx * ny)) * grid.h];
}
export function quadraticValue(q: Quadratic, x: V3): number {
  return q[0] + q[1] * x[0] + q[2] * x[1] + q[3] * x[2]
    + .5 * (q[4] * x[0] ** 2 + q[5] * x[1] ** 2 + q[6] * x[2] ** 2)
    + q[7] * x[0] * x[1] + q[8] * x[0] * x[2] + q[9] * x[1] * x[2];
}
export function quadraticGradient(q: Quadratic, x: V3): V3 {
  return [q[1] + q[4] * x[0] + q[7] * x[1] + q[8] * x[2],
    q[2] + q[7] * x[0] + q[5] * x[1] + q[9] * x[2],
    q[3] + q[8] * x[0] + q[9] * x[1] + q[6] * x[2]];
}
export function quadraticDensity(q: Quadratic, local: V3, width: number): number {
  return Math.min(1, Math.max(0, .5 - quadraticValue(q, local) / width));
}
/** Algebraic CPU reference for coefficient transforms; test geometry oracles
 * evaluate the independently authored primitive at the mapped point instead. */
export function pullbackQuadratic(q: Quadratic, B: M3, d: V3): Quadratic {
  const H = [[q[4], q[7], q[8]], [q[7], q[5], q[9]], [q[8], q[9], q[6]]];
  const gradient = quadraticGradient(q, d);
  const g = [0, 0, 0], nextH = Array.from({ length: 3 }, () => [0, 0, 0]);
  for (let i = 0; i < 3; i++) {
    for (let k = 0; k < 3; k++) g[i]! += B[3 * k + i]! * gradient[k]!;
    for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) for (let l = 0; l < 3; l++) {
      nextH[i]![j]! += B[3 * k + i]! * H[k]![l]! * B[3 * l + j]!;
    }
  }
  return [quadraticValue(q, d), g[0]!, g[1]!, g[2]!, nextH[0]![0]!, nextH[1]![1]!,
    nextH[2]![2]!, nextH[0]![1]!, nextH[0]![2]!, nextH[1]![2]!];
}
/** Initial compilation only. No CPU quadrature is part of this prototype. */
export function compileQuadraticSupports(grid: QuadraticSupportGrid, global: Quadratic): Float32Array {
  if (!global.every(Number.isFinite)) throw new Error("Non-finite initial quadratic");
  const records = new Float32Array(supportCount(grid) * QUADRATIC_PATCH_WORDS);
  const words = new Uint32Array(records.buffer);
  for (let id = 0; id < supportCount(grid); id++) {
    records.set(pullbackQuadratic(global, IDENTITY_MATRIX, supportOrigin(grid, id)), 16 * id);
    words[16 * id + 10] = 1;
  }
  return records;
}
export function supportBoxWorklist(grid: QuadraticSupportGrid, lower: V3, upper: V3): Uint32Array {
  const ids: number[] = [], [nx, ny, nz] = grid.dimensions;
  for (let axis = 0; axis < 3; axis++) if (!Number.isInteger(lower[axis]) || !Number.isInteger(upper[axis])
    || lower[axis]! < 0 || upper[axis]! > grid.dimensions[axis]! || lower[axis]! >= upper[axis]!) {
    throw new Error("Invalid requested support box");
  }
  for (let z = lower[2]; z < upper[2]; z++) for (let y = lower[1]; y < upper[1]; y++) {
    for (let x = lower[0]; x < upper[0]; x++) ids.push(x + nx * (y + ny * z));
  }
  void nz; return Uint32Array.from(ids);
}

export const QUADRATIC_PULLBACK_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage,read> source:array<f32>;
@group(0) @binding(1) var<storage,read_write> destination:array<f32>;
@group(0) @binding(2) var<storage,read> p:array<f32>;
@group(0) @binding(3) var<storage,read> work:array<u32>;
@group(0) @binding(4) var<storage,read_write> receipt:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read> queries:array<f32>;
@group(0) @binding(6) var<storage,read_write> output:array<f32>;
const INVALID:u32=0xffffffffu;
fn u(i:u32)->u32{return bitcast<u32>(p[i]);}
fn dims()->vec3u{return vec3u(u(4u),u(5u),u(6u));}
fn coord(id:u32)->vec3u{let n=dims();return vec3u(id%n.x,(id/n.x)%n.y,id/(n.x*n.y));}
fn lower(id:u32)->vec3f{return vec3f(p[0],p[1],p[2])+p[3]*vec3f(coord(id));}
fn cell(q:vec3i)->u32{let n=dims();if(any(q<vec3i(0))||any(q>=vec3i(n))){return INVALID;}
 return u32(q.x)+n.x*(u32(q.y)+n.y*u32(q.z));}
fn owner(x:vec3f)->u32{return cell(vec3i(floor((x-vec3f(p[0],p[1],p[2]))/p[3])));}
fn fail(code:u32,id:u32){atomicOr(&receipt[0],code);atomicMin(&receipt[1],id);}
struct Q{c:f32,g:vec3f,H:mat3x3f}
fn loadQ(id:u32,next:bool)->Q{
 var a:array<f32,10>;for(var i=0u;i<10u;i++){a[i]=source[16u*id+i];
  if(next){a[i]=destination[16u*id+i];}}
 return Q(a[0],vec3f(a[1],a[2],a[3]),
  mat3x3f(vec3f(a[4],a[7],a[8]),vec3f(a[7],a[5],a[9]),vec3f(a[8],a[9],a[6])));
}
fn valid(id:u32,next:bool)->bool{if(id==INVALID){return false;}
 var tag=bitcast<u32>(source[16u*id+10u]);if(next){tag=bitcast<u32>(destination[16u*id+10u]);}
 return tag==u(20u)+select(0u,1u,next);
}
fn finiteQ(q:Q)->bool{return abs(q.c)<3.4e38&&all(abs(q.g)<vec3f(3.4e38))
 &&all(abs(q.H[0])<vec3f(3.4e38))&&all(abs(q.H[1])<vec3f(3.4e38))&&all(abs(q.H[2])<vec3f(3.4e38));}
fn phi(q:Q,x:vec3f)->f32{return q.c+dot(q.g,x)+0.5*dot(x,q.H*x);}
fn grad(q:Q,x:vec3f)->vec3f{return q.g+q.H*x;}
fn density(q:Q,x:vec3f)->f32{return clamp(0.5-phi(q,x)/p[24],0.0,1.0);}
fn translated(q:Q,d:vec3f)->Q{return Q(phi(q,d),grad(q,d),q.H);}
fn equivalent(a:Q,b:Q)->bool{
 let h=p[3];let tol=p[25];
 return abs(a.c-b.c)<=tol*max(1.0,max(abs(a.c),abs(b.c)))
 &&all(abs((a.g-b.g)*h)<=vec3f(tol)*max(vec3f(1.0),max(abs(a.g*h),abs(b.g*h))))
 &&all(abs((a.H[0]-b.H[0])*h*h)<=vec3f(tol)*max(vec3f(1.0),max(abs(a.H[0]*h*h),abs(b.H[0]*h*h))))
 &&all(abs((a.H[1]-b.H[1])*h*h)<=vec3f(tol)*max(vec3f(1.0),max(abs(a.H[1]*h*h),abs(b.H[1]*h*h))))
 &&all(abs((a.H[2]-b.H[2])*h*h)<=vec3f(tol)*max(vec3f(1.0),max(abs(a.H[2]*h*h),abs(b.H[2]*h*h))));
}
fn checkNeighbors(id:u32,next:bool){
 if(!valid(id,next)){return;}let a=loadQ(id,next);if(!finiteQ(a)){fail(8u,id);return;}
 let q=vec3i(coord(id));for(var axis=0u;axis<3u;axis++){
  var neighbor=q;neighbor[axis]+=1;let at=cell(neighbor);
  if(valid(at,next)&&!equivalent(translated(a,lower(at)-lower(id)),loadQ(at,next))){fail(2u,id);}
 }
}
@compute @workgroup_size(64) fn validateSource(@builtin(global_invocation_id)gid:vec3u){
 if(gid.x<u(7u)){checkNeighbors(gid.x,false);}}
fn mapMatrix()->mat3x3f{return mat3x3f(vec3f(p[8],p[12],p[16]),
 vec3f(p[9],p[13],p[17]),vec3f(p[10],p[14],p[18]));}
fn mapTranslation()->vec3f{return vec3f(p[11],p[15],p[19]);}
@compute @workgroup_size(64) fn pullback(@builtin(global_invocation_id)gid:vec3u){
 if(gid.x>=u(21u)){return;}let id=work[gid.x];let B=mapMatrix();
 // Compute the footprint in lattice-relative coordinates. Identity maps must
 // not manufacture tiny slivers across an exact support face through x/h.
 let gridOrigin=vec3f(p[0],p[1],p[2]);
 var start=B*vec3f(coord(id))+(B*gridOrigin-gridOrigin+mapTranslation())/p[3];
 // Even an exactly representable ratio t/h can acquire a sliver from GPU
 // division. The host uploads certified half-integer lattice offsets for
 // identity rows; their integer-coordinate sums need no division at all.
 for(var axis=0u;axis<3u;axis++){if((u(40u)&(1u<<axis))!=0u){start[axis]=f32(coord(id)[axis])-p[44u+axis];}}
 var lo=vec3f(3.4e38);var hi=vec3f(-3.4e38);
 for(var corner=0u;corner<8u;corner++){
  let x=start+B*vec3f(f32(corner&1u),f32((corner>>1u)&1u),f32(corner>>2u));
  lo=min(lo,x);hi=max(hi,x);
 }
 let a=vec3i(floor(lo));let b=vec3i(ceil(hi))-vec3i(1);let span=b-a+vec3i(1);
 if(any(span<=vec3i(0))||any(span>vec3i(i32(u(23u))))){fail(4u,id);return;}
 let count=u32(span.x*span.y*span.z);if(count>u(23u)){fail(4u,id);return;}
 let first=cell(a);if(!valid(first,false)){fail(1u,id);return;}
 let q=loadQ(first,false);if(!finiteQ(q)){fail(8u,id);return;}
 for(var z=a.z;z<=b.z;z++){for(var y=a.y;y<=b.y;y++){for(var x=a.x;x<=b.x;x++){
  let donor=cell(vec3i(x,y,z));if(!valid(donor,false)){fail(1u,id);return;}
  let other=loadQ(donor,false);if(!finiteQ(other)){fail(8u,id);return;}
  if(!equivalent(translated(q,lower(donor)-lower(first)),other)){fail(2u,id);return;}
 }}}
 let d=(start-vec3f(coord(first)))*p[3];let next=Q(phi(q,d),transpose(B)*grad(q,d),transpose(B)*q.H*B);
 if(!finiteQ(next)){fail(8u,id);return;}
 let at=16u*id;destination[at]=next.c;destination[at+1u]=next.g.x;
 destination[at+2u]=next.g.y;destination[at+3u]=next.g.z;
 destination[at+4u]=next.H[0].x;destination[at+5u]=next.H[1].y;destination[at+6u]=next.H[2].z;
 destination[at+7u]=next.H[0].y;destination[at+8u]=next.H[0].z;destination[at+9u]=next.H[1].z;
 destination[at+10u]=bitcast<f32>(u(20u)+1u);destination[at+11u]=bitcast<f32>(count);
 for(var i=12u;i<16u;i++){destination[at+i]=0.0;}
 atomicAdd(&receipt[2],count);atomicAdd(&receipt[3],1u);
}
@compute @workgroup_size(64) fn validateDestination(@builtin(global_invocation_id)gid:vec3u){
 if(gid.x<u(21u)){checkNeighbors(work[gid.x],true);}}

@compute @workgroup_size(64) fn validateWetCoverage(@builtin(global_invocation_id)gid:vec3u){
 let id=gid.x;if(id>=u(7u)||atomicLoad(&receipt[0])!=0u||!valid(id,false)){return;}
 // A zero quadrature result is not a dry certificate. This conservative
 // quadratic range includes coefficient/evaluation roundoff in phi units.
 if(conservativePhiRange(loadQ(id,false)).x>=0.5*p[24]){return;}
 let A=mat3x3f(vec3f(p[28],p[32],p[36]),vec3f(p[29],p[33],p[37]),vec3f(p[30],p[34],p[38]));
 let translation=vec3f(p[31],p[35],p[39]);let gridOrigin=vec3f(p[0],p[1],p[2]);
 let originFine=gridOrigin/p[3];let sourceCoordinate=vec3f(coord(id));
 var start=A*sourceCoordinate+(A*gridOrigin-gridOrigin+translation)/p[3];
 for(var axis=0u;axis<3u;axis++){if((u(40u)&(1u<<axis))!=0u){start[axis]=sourceCoordinate[axis]+p[44u+axis];}}
 var lo=vec3f(3.4e38);var hi=vec3f(-3.4e38);
 for(var corner=0u;corner<8u;corner++){
  let x=start+A*vec3f(f32(corner&1u),f32((corner>>1u)&1u),f32(corner>>2u));lo=min(lo,x);hi=max(hi,x);
 }
 // Half-integer translations on identity rows are exact for this <=2^20
 // lattice. Other rows get a conservative f32 inverse/evaluation envelope.
 // Its possible extra dry halo is deliberate; it never trims a footprint.
 let absoluteA=mat3x3f(abs(A[0]),abs(A[1]),abs(A[2]));
 var roundoff=8e-6*(vec3f(1.0)+absoluteA*(abs(originFine)+abs(sourceCoordinate)+vec3f(1.0))
  +abs(originFine)+abs(translation/p[3]));
 for(var axis=0u;axis<3u;axis++){if((u(40u)&(1u<<axis))!=0u){roundoff[axis]=0.0;}}
 lo-=roundoff;hi+=roundoff;
 if(!all(abs(lo)<=vec3f(16777216.0))||!all(abs(hi)<=vec3f(16777216.0))){fail(4u,id);return;}
 let a=vec3i(floor(lo));let b=vec3i(ceil(hi))-vec3i(1);let span=b-a+vec3i(1);
 if(any(span<=vec3i(0))||any(span>vec3i(i32(u(23u))))||u32(span.x*span.y*span.z)>u(23u)){fail(4u,id);return;}
 for(var z=a.z;z<=b.z;z++){for(var y=a.y;y<=b.y;y++){for(var x=a.x;x<=b.x;x++){
  if(!valid(cell(vec3i(x,y,z)),true)){fail(32u,id);return;}
 }}}
 atomicAdd(&receipt[4],1u);atomicAdd(&receipt[5],u32(span.x*span.y*span.z));
}

// Integrate the clamped quadratic EXACTLY along x after splitting at its
// q=0 and q=1 roots, then bounded adaptive 8-point Gauss in y/z. A nested-rule
// difference is an error ESTIMATE, not a mathematical enclosure; independent
// analytic/numerical oracles remain necessary. No target mass is fitted here.
fn lineIntegral(a:f32,b:f32,c:f32)->f32{
 var cuts:array<f32,6>;cuts[0]=0.0;cuts[1]=1.0;var count=2u;
 for(var level=0u;level<2u;level++){
  let k=c-f32(level);var roots=vec2f(-1.0);var n=0u;
  if(abs(a)<1e-20){if(abs(b)>1e-20){roots.x=-k/b;n=1u;}}
  else{let discriminant=b*b-4.0*a*k;if(discriminant>=0.0){
   let term=-0.5*(b+select(-sqrt(discriminant),sqrt(discriminant),b>=0.0));
   if(abs(term)>1e-20){roots=vec2f(term/a,k/term);n=2u;}
   else{roots.x=-b/(2.0*a);n=1u;}
  }}
  for(var r=0u;r<n;r++){if(roots[r]>0.0&&roots[r]<1.0){cuts[count]=roots[r];count++;}}
 }
 for(var i=1u;i<count;i++){let value=cuts[i];var j=i;
  loop{if(j==0u||cuts[j-1u]<=value){break;}cuts[j]=cuts[j-1u];j--;}
  cuts[j]=value;
 }
 var sum=0.0;for(var i=1u;i<count;i++){let length=cuts[i]-cuts[i-1u];let m=0.5*(cuts[i]+cuts[i-1u]);
  let value=(a*m+b)*m+c;if(value>=1.0){sum+=length;}
  else if(value>0.0){sum+=length*(a*(m*m+length*length/12.0)+b*m+c);}
 }
 return sum;
}
const NODES:array<f32,8> = array<f32,8>(0.0198550717512319,0.101666761293187,0.237233795041836,
 0.408282678752175,0.591717321247825,0.762766204958164,0.898333238706813,0.980144928248768);
const WEIGHTS:array<f32,8> = array<f32,8>(0.0506142681451881,0.111190517226687,0.156853322938944,
 0.181341891689181,0.181341891689181,0.156853322938944,0.111190517226687,0.0506142681451881);
fn integrateYZ(q:Q,origin:vec2f,side:f32)->f32{
 let h=p[3];var sum=0.0;
 for(var z=0u;z<8u;z++){for(var y=0u;y<8u;y++){
  let yz=h*(origin+side*vec2f(NODES[y],NODES[z]));let point=vec3f(0.0,yz);
  sum+=WEIGHTS[y]*WEIGHTS[z]*lineIntegral(-0.5*q.H[0].x*h*h/p[24],
   -grad(q,point).x*h/p[24],0.5-phi(q,point)/p[24]);
 }}return sum*side*side;
}
fn conservativePhiRange(q:Q)->vec2f{
 let r=0.5*p[3];let center=vec3f(r);let value=phi(q,center);
 let variation=r*dot(abs(grad(q,center)),vec3f(1.0))
  +0.5*r*r*(dot(abs(q.H[0]),vec3f(1.0))+dot(abs(q.H[1]),vec3f(1.0))+dot(abs(q.H[2]),vec3f(1.0)));
 let roundoff=8e-6*max(1.0,abs(value)+variation);
 return vec2f(value-variation-roundoff,value+variation+roundoff);
}
@compute @workgroup_size(64) fn integrate(@builtin(global_invocation_id)gid:vec3u){
 if(gid.x>=u(21u)){return;}let id=work[gid.x];if(!valid(id,false)){fail(1u,id);return;}
 let q=loadQ(id,false);let h=p[3];let range=conservativePhiRange(q);
 if(range.x>=0.5*p[24]){output[gid.x]=0.0;return;}
 if(range.y<=-0.5*p[24]){output[gid.x]=h*h*h;return;}
 // Depth-first quadtree: depth<=4 needs at most 1+3*4=13 pending tiles.
 // Every parent compares against its four children; worst case87360 exact-x
 // slices per support. Terminal tiles share the same WHOLE-support2e-6
 // budget, rather than requiring a vanishing area-proportional budget at a
 // tangent. The sum of all accepted/terminal estimates must meet that budget.
 var pending:array<vec4f,16>;pending[0]=vec4f(0.0,0.0,1.0,integrateYZ(q,vec2f(0.0),1.0));
 var count=1u;var sum=0.0;var errorSum=0.0;var slices=64u;
 loop{if(count==0u){break;}count--;let tile=pending[count];let side=0.5*tile.z;
  var children:array<vec4f,4>;var fine=0.0;
  for(var corner=0u;corner<4u;corner++){
   let origin=tile.xy+side*vec2f(f32(corner&1u),f32(corner>>1u));
   let amount=integrateYZ(q,origin,side);children[corner]=vec4f(origin,side,amount);fine+=amount;
  }slices+=256u;let error=abs(fine-tile.w);
  if(error<=p[25]*tile.z*tile.z||tile.z<=0.0625){sum+=fine;errorSum+=error;}
  else{
   for(var corner=0u;corner<4u;corner++){pending[count]=children[corner];count++;}
  }
 }
 atomicMax(&receipt[2],bitcast<u32>(errorSum));atomicMax(&receipt[3],slices);
 if(errorSum>p[25]){fail(64u,id);return;}
 if(!(sum>=-p[25]&&sum<=1.0+p[25])){fail(8u,id);return;}
 output[gid.x]=sum*h*h*h;
}
fn closestRoot(a:f32,b:f32,c:f32,lo:f32,hi:f32)->f32{
 var answer=3.4e38;if(abs(a)<1e-20){if(abs(b)>1e-20){let r=-c/b;if(r>=lo&&r<=hi){answer=r;}}}
 else{let disc=b*b-4.0*a*c;if(disc>=0.0){let root=sqrt(disc);
  let r0=(-b-root)/(2.0*a);let r1=(-b+root)/(2.0*a);
  if(r0>=lo&&r0<=hi){answer=r0;}if(r1>=lo&&r1<=hi){answer=min(answer,r1);}
 }}return answer;
}
@compute @workgroup_size(64) fn sample(@builtin(global_invocation_id)gid:vec3u){
 if(gid.x>=u(22u)){return;}let at=8u*gid.x;let x=vec3f(queries[at],queries[at+1u],queries[at+2u]);
 let direction=vec3f(queries[at+4u],queries[at+5u],queries[at+6u]);let id=owner(x);
 if(!valid(id,false)){fail(16u,gid.x);return;}let q=loadQ(id,false);let local=x-lower(id);
 let value=phi(q,local);let gradient=grad(q,local);let root=closestRoot(0.5*dot(direction,q.H*direction),
  dot(gradient,direction),value,queries[at+3u],queries[at+7u]);var normal=vec3f(0.0);
 if(root<3.4e38){let point=x+root*direction;let rootOwner=owner(point);
  if(!valid(rootOwner,false)){fail(16u,gid.x);return;}
  // A globally coherent polynomial does not authorize evaluation through an
  // absent support. Check the whole ray segment, not just its two endpoints.
  let gridOrigin=vec3f(p[0],p[1],p[2]);
  let a=vec3i(floor((min(x,point)-gridOrigin)/p[3]));
  let b=vec3i(floor((max(x,point)-gridOrigin)/p[3]));let span=b-a+vec3i(1);
  if(any(span<=vec3i(0))||any(span>vec3i(i32(u(23u))))||u32(span.x*span.y*span.z)>u(23u)){
   fail(4u,gid.x);return;
  }
  for(var z=a.z;z<=b.z;z++){for(var y=a.y;y<=b.y;y++){for(var rayX=a.x;rayX<=b.x;rayX++){
   let donor=cell(vec3i(rayX,y,z));if(!valid(donor,false)){fail(16u,gid.x);return;}
   if(!equivalent(translated(q,lower(donor)-lower(id)),loadQ(donor,false))){fail(2u,gid.x);return;}
  }}}
  let rootQ=loadQ(rootOwner,false);let rootLocal=point-lower(rootOwner);
  if(abs(phi(rootQ,rootLocal))>8.0*p[25]){fail(2u,gid.x);return;}
  let g=grad(rootQ,rootLocal);normal=g/max(length(g),1e-20);
 }
 let out=16u*gid.x;output[out]=value;output[out+1u]=density(q,local);
 output[out+2u]=gradient.x;output[out+3u]=gradient.y;output[out+4u]=gradient.z;
 output[out+5u]=root;output[out+6u]=normal.x;output[out+7u]=normal.y;output[out+8u]=normal.z;
 output[out+9u]=q.H[0].x;output[out+10u]=q.H[1].y;output[out+11u]=q.H[2].z;
 output[out+12u]=q.H[0].y;output[out+13u]=q.H[0].z;output[out+14u]=q.H[1].z;output[out+15u]=1.0;
}
`;

export interface QuadraticSample { point: V3; direction?: V3; minimum?: number; maximum?: number }
export interface PullbackReceipt { generation: number; supports: number; donorVisits: number;
  potentiallyWetSourceSupports: number; forwardCoverageVisits: number }

export class GPUQuadraticPullback {
  private bank = 0;
  private epoch = 1;
  private busy = false;
  lastIntegrationStats: { maximumEstimatedMeanError: number; maximumXSlicesPerSupport: number } | undefined;
  private constructor(readonly device: GPUDevice, readonly grid: QuadraticSupportGrid,
    readonly width: number, private readonly banks: readonly [GPUBuffer, GPUBuffer],
    private readonly layout: GPUBindGroupLayout, private readonly pipelines: ReadonlyMap<string, GPUComputePipeline>,
    private readonly tolerance: number) {}

  get generation(): number { return this.epoch; }

  static async create(device: GPUDevice, grid: QuadraticSupportGrid, width: number,
    records: Float32Array, tolerance = 2e-6): Promise<GPUQuadraticPullback> {
    const count = supportCount(grid);
    if (!(Math.fround(width) >= MIN_NORMAL_F32) || !Number.isFinite(Math.fround(width)) || records.length !== 16 * count
      || !(Math.fround(tolerance) >= MIN_NORMAL_F32) || !Number.isFinite(Math.fround(tolerance))) throw new Error("Invalid quadratic field input");
    const compiler = gpuCompilationManagerFor(device);
    const module = compiler.createShaderModule({ label: "Research: coherent current quadratic pullback", code: QUADRATIC_PULLBACK_WGSL });
    const layout = device.createBindGroupLayout({ entries: Array.from({ length: 7 }, (_, binding) => ({
      binding, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: [0, 2, 3, 5].includes(binding) ? "read-only-storage" as const : "storage" as const },
    })) });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const names = ["validateSource", "pullback", "validateDestination", "validateWetCoverage", "integrate", "sample"];
    const pipelines = new Map<string, GPUComputePipeline>();
    try {
      for (const entryPoint of names) pipelines.set(entryPoint, await compiler.compileComputePipeline({
        label: `Research quadratic ${entryPoint}`, layout: pipelineLayout, compute: { module, entryPoint },
      }, { priority: "critical" }));
    } catch (error) {
      const info = await module.getCompilationInfo();
      throw new Error(`Quadratic pullback WGSL: ${info.messages.map(message => `${message.lineNum}:${message.linePos} ${message.message}`).join("\n")}`, { cause: error });
    }
    const banks = [0, 1].map(bank => device.createBuffer({ label: `Research current quadratic bank ${bank}`,
      size: records.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC })) as [GPUBuffer, GPUBuffer];
    device.queue.writeBuffer(banks[0], 0, records.buffer as ArrayBuffer, records.byteOffset, records.byteLength);
    return new GPUQuadraticPullback(device, grid, width, banks, layout, pipelines, tolerance);
  }

  private async run(names: readonly string[], targets: Uint32Array, map: AffineDeparture,
    queries = new Float32Array(), resultWords = 1, maxDonors = 64): Promise<{ receipt: Uint32Array; values: Float32Array }> {
    if (this.busy) throw new Error("Quadratic generation transaction already running");
    this.busy = true;
    const resources: GPUBuffer[] = [];
    try {
      const count = supportCount(this.grid), seen = new Set<number>();
      for (const id of targets) {
        if (id >= count || seen.has(id)) throw new Error("Invalid or duplicate destination support");
        seen.add(id);
      }
      const canonical = canonicalAffineDeparture(map);
      const values = new Float32Array(48), words = new Uint32Array(values.buffer);
      values.set([...this.grid.origin, this.grid.h]); words.set([...this.grid.dimensions, count], 4);
      for (let row = 0; row < 3; row++) {
        values.set([...canonical.departure.matrix.slice(row * 3, row * 3 + 3), canonical.departure.translation[row]!], 8 + 4 * row);
        values.set([...canonical.forward.matrix.slice(row * 3, row * 3 + 3), canonical.forward.translation[row]!], 28 + 4 * row);
        const offset = canonical.forward.translation[row]! / Math.fround(this.grid.h);
        const exactRow = [0, 1, 2].every(column => canonical.forward.matrix[3 * row + column] === Number(row === column)
          && canonical.departure.matrix[3 * row + column] === Number(row === column));
        if (exactRow && Number.isInteger(2 * offset) && Math.abs(offset) <= 1_048_576) {
          words[40]! |= 1 << row; values[44 + row] = offset;
        }
      }
      words.set([this.epoch, targets.length, queries.length / 8, maxDonors], 20);
      values[24] = this.width; values[25] = this.tolerance;
      const buffer = (label: string, size: number, usage: GPUBufferUsageFlags) => {
        const result = this.device.createBuffer({ label, size: Math.max(4, size), usage }); resources.push(result); return result;
      };
      const upload = (label: string, data: Float32Array | Uint32Array, extra = 0) => {
        const result = buffer(label, data.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extra);
        if (data.byteLength) this.device.queue.writeBuffer(result, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
        return result;
      };
      const parameters = upload("Quadratic immutable operation parameters", values);
      const worklist = upload("Quadratic requested fixed supports", targets);
      const queryBuffer = upload("Quadratic field queries", queries);
      const fault = upload("Quadratic publication receipt", new Uint32Array([0, 0xffffffff, 0, 0, 0, 0, 0, 0]), GPUBufferUsage.COPY_SRC);
      const result = buffer("Quadratic diagnostics", 4 * resultWords, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const readback = buffer("Quadratic diagnostics and receipt readback", 32 + 4 * resultWords, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
      const binding = this.device.createBindGroup({ layout: this.layout,
        entries: [this.banks[this.bank]!, this.banks[this.bank ^ 1]!, parameters, worklist, fault, queryBuffer, result]
          .map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
      const encoder = this.device.createCommandEncoder({ label: "Research coherent quadratic operation" });
      // A rejected attempt may have written some epoch+1 records. Retrying
      // with a smaller/disjoint worklist must not publish those old writes.
      if (names.includes("pullback")) encoder.clearBuffer(this.banks[this.bank ^ 1]!);
      for (const name of names) {
        const invocations = name === "validateSource" || name === "validateWetCoverage" ? count
          : name === "sample" ? queries.length / 8 : targets.length;
        if (invocations === 0) continue;
        const pass = encoder.beginComputePass(); pass.setPipeline(this.pipelines.get(name)!); pass.setBindGroup(0, binding);
        pass.dispatchWorkgroups(Math.ceil(invocations / 64)); pass.end();
      }
      encoder.copyBufferToBuffer(fault, 0, readback, 0, 32);
      encoder.copyBufferToBuffer(result, 0, readback, 32, 4 * resultWords);
      this.device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
      const mapped = readback.getMappedRange();
      const receipt = new Uint32Array(mapped, 0, 8).slice();
      const output = new Float32Array(mapped, 32, resultWords).slice(); readback.unmap();
      if (receipt[0]) throw new Error(`Quadratic generation rejected: fault=${receipt[0]} id=${receipt[1]} generation=${this.epoch}`
        + (names.includes("integrate") ? ` maxEstimatedMeanError=${new Float32Array(receipt.buffer)[2]} maxXSlices=${receipt[3]}` : ""));
      return { receipt, values: output };
    } finally { for (const resource of resources) resource.destroy(); this.busy = false; }
  }

  async advance(map: AffineDeparture, targets: Uint32Array, maxDonors = 64): Promise<PullbackReceipt> {
    if (targets.length === 0 || !Number.isInteger(maxDonors) || maxDonors < 1 || maxDonors > 256 || this.epoch >= 1_000_000) {
      throw new Error("Invalid or unsupported prescribed departure map");
    }
    const { receipt } = await this.run(["validateSource", "pullback", "validateDestination", "validateWetCoverage"], targets, map, undefined, 1, maxDonors);
    if (receipt[3] !== targets.length) throw new Error("Incomplete quadratic generation receipt");
    this.bank ^= 1; this.epoch++;
    return { generation: this.epoch, supports: targets.length, donorVisits: receipt[2]!,
      potentiallyWetSourceSupports: receipt[4]!, forwardCoverageVisits: receipt[5]! };
  }

  async integrate(targets: Uint32Array): Promise<Float32Array> {
    const result = await this.run(["validateSource", "integrate"], targets,
      { matrix: IDENTITY_MATRIX, translation: [0, 0, 0] }, undefined, targets.length);
    this.lastIntegrationStats = { maximumEstimatedMeanError: new Float32Array(result.receipt.buffer)[2]!,
      maximumXSlicesPerSupport: result.receipt[3]! };
    return result.values;
  }

  /** Output stride16: phi,density,gradient xyz,ray root t,root normal xyz,
   * Hessian xx yy zz xy xz yz,valid. Missing ray root is 3.4e38. */
  async sample(samples: readonly QuadraticSample[]): Promise<Float32Array> {
    if (samples.length === 0) return new Float32Array();
    const input = new Float32Array(8 * samples.length);
    for (let i = 0; i < samples.length; i++) {
      const sample = samples[i]!;
      const record = [...sample.point, sample.minimum ?? 0, ...sample.direction ?? [0, 0, 0], sample.maximum ?? 0];
      if (!record.every(Number.isFinite)) throw new Error("Non-finite quadratic query");
      input.set(record, 8 * i);
    }
    return (await this.run(["validateSource", "sample"], new Uint32Array(),
      { matrix: IDENTITY_MATRIX, translation: [0, 0, 0] }, input, 16 * samples.length)).values;
  }

  async readCurrentRecordsForQA(): Promise<Float32Array> {
    if (this.busy) throw new Error("Quadratic transaction still running");
    const readback = this.device.createBuffer({ size: this.banks[0].size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const encoder = this.device.createCommandEncoder(); encoder.copyBufferToBuffer(this.banks[this.bank]!, 0, readback, 0, readback.size);
      this.device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(readback.getMappedRange()).slice(); readback.unmap(); return result;
    } finally { readback.destroy(); }
  }
  destroy(): void { for (const bank of this.banks) bank.destroy(); }
}
