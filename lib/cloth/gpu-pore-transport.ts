import type { ClothMaterial, ClothPhysicsOptions, ClothTopologyData, ClothVector } from './types';
import { porousPermeability } from './porous-material';

/** GPU manifold pore transport: the released source's Eq. 18 FEM seepage
 * (advanceClothPoreTransport is its f64 oracle, kept for parity tests).
 *
 * Pore water is fixed-point: int32 quanta of a power-of-two volume. Every
 * accepted edge flux is an integer moved from one control to its neighbour,
 * so the manifold conserves water exactly, and the paired donor limiter
 * works in exact integers, so water never goes negative. The f32 Krylov
 * solve stops at a normwise backward error of 16 units of f32 roundoff. All
 * substeps run on the device without a readback; a failure is a status bit
 * raised at the end of the step. */
export const PORE_GPU_STATUS = { collapsedFace: 1, collapsedRod: 2, permeability: 4, krylov: 8, picard: 16, quantum: 32, predictor: 64 } as const;
const STATUS_MESSAGES: readonly [number, string][] = [
  [PORE_GPU_STATUS.collapsedFace, 'Collapsed pore-flow shell face'], [PORE_GPU_STATUS.collapsedRod, 'Collapsed pore-flow rod edge'],
  [PORE_GPU_STATUS.permeability, 'Porous solid fraction must lie strictly between zero and one with a positive permeability'],
  [PORE_GPU_STATUS.krylov, 'Pore transport Krylov solve failed to reach f32 backward-error precision'],
  [PORE_GPU_STATUS.picard, 'Nonlinear manifold mobility did not converge'],
  [PORE_GPU_STATUS.quantum, 'Pore water exceeds the int32 quantum range'], [PORE_GPU_STATUS.predictor, 'Invalid manifold Gauss force predictor'],
];

/** Full rest control volume spans 2^26 quanta: 32x headroom to int32. */
export const PORE_QUANTUM_BITS = 26;
/** Floats a material in clothPoreMaterialTable. */
export const CLOTH_PORE_MATERIAL_FLOATS = 8;
const MATERIAL_FLOATS = CLOTH_PORE_MATERIAL_FLOATS, GAUSS_VEC4 = 8, TERM_VEC4 = 2;

export function clothPoreQuantum(restControlVolume_m3: ArrayLike<number>): number {
  let largest = 0; for (let i = 0; i < restControlVolume_m3.length; i++) largest = Math.max(largest, restControlVolume_m3[i]!);
  if (!(largest > 0) || !Number.isFinite(largest)) throw new Error('Pore water quantum needs a positive rest control volume');
  return 2 ** (Math.ceil(Math.log2(largest)) - PORE_QUANTUM_BITS);
}

/** Static tables of one topology, in the oracle's assembly order: triangle
 * edge terms (three a triangle), then rod segment terms; edge slots in first
 * appearance; each slot's terms; each vertex's slots (bit 31: the vertex is b). */
export interface ClothPoreTopology {
  readonly V: number; readonly T: number; readonly R: number; readonly C: number; readonly E: number; readonly G: number; readonly M: number;
  readonly words: Uint32Array<ArrayBuffer>; readonly floats: Float32Array<ArrayBuffer>;
  readonly offsets: { readonly tri: number; readonly rod: number; readonly term: number; readonly slot: number; readonly slotStart: number; readonly slotTerms: number; readonly vertStart: number; readonly vertSlots: number; readonly triWeights: number; readonly rodRest: number; readonly materials: number };
  /** Edge slot endpoints (a < b), for tests. */
  readonly slots: Uint32Array;
}

export function clothPoreTopology(data: ClothTopologyData, materials: readonly ClothMaterial[], V: number): ClothPoreTopology {
  const T = data.triangles.length / 3, rest = data.restPositions_m;
  const rods: number[] = [];
  for (let r = 0; r + 1 < data.rodOffsets.length; r++) for (let j = data.rodOffsets[r]!; j + 1 < data.rodOffsets[r + 1]!; j++) rods.push(data.rodIndices[j]!, data.rodIndices[j + 1]!, data.rodMaterials?.[r] ?? 0);
  const R = rods.length / 3, G = T + R, terms: number[] = [], slotOf = new Map<number, number>(), slots: number[] = [];
  const slot = (a: number, b: number) => { const key = a * V + b; let s = slotOf.get(key); if (s === undefined) { s = slots.length / 2; slotOf.set(key, s); slots.push(a, b); } return s; };
  for (let t = 0; t < T; t++) for (let j = 0; j < 3; j++) {
    let a = data.triangles[3 * t + j]!, b = data.triangles[3 * t + (j + 1) % 3]!; if (a > b) [a, b] = [b, a];
    terms.push(a, b, slot(a, b), 3 * t + j);
  }
  for (let r = 0; r < R; r++) { let a = rods[3 * r]!, b = rods[3 * r + 1]!; if (a > b) [a, b] = [b, a]; terms.push(a, b, slot(a, b), (0x80000000 | r) >>> 0); }
  const C = terms.length / 4, E = slots.length / 2, M = materials.length;
  const slotStart = new Uint32Array(E + 1), slotTerms = new Uint32Array(C), vertStart = new Uint32Array(V + 1), vertSlots = new Uint32Array(2 * E);
  for (let k = 0; k < C; k++) slotStart[terms[4 * k + 2]! + 1]!++;
  for (let e = 0; e < E; e++) slotStart[e + 1]! += slotStart[e]!;
  const fill = slotStart.slice(0, E); for (let k = 0; k < C; k++) slotTerms[fill[terms[4 * k + 2]!]!++] = k;
  for (let e = 0; e < E; e++) { vertStart[slots[2 * e]! + 1]!++; vertStart[slots[2 * e + 1]! + 1]!++; }
  for (let v = 0; v < V; v++) vertStart[v + 1]! += vertStart[v]!;
  const at = vertStart.slice(0, V); for (let e = 0; e < E; e++) { vertSlots[at[slots[2 * e]!]!++] = e; vertSlots[at[slots[2 * e + 1]!]!++] = (0x80000000 | e) >>> 0; }
  const sections: [string, number][] = [['tri', 4 * T], ['rod', 4 * R], ['term', 4 * C], ['slot', 2 * E], ['slotStart', E + 1], ['slotTerms', C], ['vertStart', V + 1], ['vertSlots', 2 * E]];
  const o: Record<string, number> = {}; let words = 0; for (const [name, n] of sections) { o[name] = words; words += Math.ceil(Math.max(1, n) / 4) * 4; }
  const w = new Uint32Array(words);
  for (let t = 0; t < T; t++) w.set([data.triangles[3 * t]!, data.triangles[3 * t + 1]!, data.triangles[3 * t + 2]!, data.triangleMaterials?.[t] ?? 0], o.tri! + 4 * t);
  for (let r = 0; r < R; r++) w.set([rods[3 * r]!, rods[3 * r + 1]!, rods[3 * r + 2]!, 0], o.rod! + 4 * r);
  w.set(terms, o.term!); w.set(slots, o.slot!); w.set(slotStart, o.slotStart!); w.set(slotTerms, o.slotTerms!); w.set(vertStart, o.vertStart!); w.set(vertSlots, o.vertSlots!);
  // Rest data: triangle Gauss weights (vertex angle / pi) and rest area; rod rest length.
  const fo = { triWeights: 0, rodRest: 4 * T, materials: 4 * T + Math.ceil(Math.max(1, R) / 4) * 4 }, f = new Float32Array(fo.materials + MATERIAL_FLOATS * Math.max(1, M));
  const p = (v: number) => [rest[3 * v]!, rest[3 * v + 1]!, rest[3 * v + 2]!];
  const sub = (x: number[], y: number[]) => x.map((v, k) => v - y[k]!), cross = (x: number[], y: number[]) => [x[1]! * y[2]! - x[2]! * y[1]!, x[2]! * y[0]! - x[0]! * y[2]!, x[0]! * y[1]! - x[1]! * y[0]!];
  for (let t = 0; t < T; t++) {
    const ids = [0, 1, 2].map(k => data.triangles[3 * t + k]!);
    const weights = ids.map((a, i) => { const e = sub(p(ids[(i + 1) % 3]!), p(a)), g = sub(p(ids[(i + 2) % 3]!), p(a)); return Math.atan2(Math.hypot(...cross(e, g)), e.reduce((s, v, k) => s + v * g[k]!, 0)) / Math.PI; });
    f.set([...weights, .5 * Math.hypot(...cross(sub(p(ids[1]!), p(ids[0]!)), sub(p(ids[2]!), p(ids[0]!))))], fo.triWeights + 4 * t);
  }
  for (let r = 0; r < R; r++) f[fo.rodRest + r] = Math.hypot(...sub(p(rods[3 * r + 1]!), p(rods[3 * r]!)));
  return { V, T, R, C, E, G, M, words: w, floats: f, offsets: { tri: o.tri!, rod: o.rod!, term: o.term!, slot: o.slot!, slotStart: o.slotStart!, slotTerms: o.slotTerms!, vertStart: o.vertStart!, vertSlots: o.vertSlots!, ...fo }, slots: Uint32Array.from(slots) };
}

/** Per-material constants: thickness, rod area pi r rB, fiber diameter,
 * viscosity, suction scale (suction = scale f / (1 - f)), nonlinear drag
 * scale (Ergun term at unit speed times sqrt(K), before (1-f)^-1.5), rest
 * packing, density. */
export function clothPoreMaterialTable(materials: readonly ClothMaterial[], physics: ClothPhysicsOptions | undefined): Float32Array<ArrayBuffer> {
  const exponent = physics?.dragExponent ?? 1.6, nonlinear = physics?.nonlinearDrag !== false, out = new Float32Array(MATERIAL_FLOATS * Math.max(1, materials.length));
  if (!Number.isFinite(exponent) || exponent <= 0) throw new Error('Drag exponent must be positive');
  materials.forEach((m, i) => {
    const radius = porousPermeability(m.solidFraction, m.fiberDiameter_m).capillaryRadius_m;
    if (!Number.isFinite(m.surfaceTension_N_m) || m.surfaceTension_N_m < 0 || !Number.isFinite(m.contactAngle_rad) || m.contactAngle_rad < 0 || m.contactAngle_rad > Math.PI) throw new Error('Invalid water surface tension or contact angle');
    if (!(m.waterViscosity_Pa_s > 0) || !(m.waterDensity_kg_m3 > 0)) throw new Error('Water viscosity and density must be positive');
    const cosine = Math.abs(m.contactAngle_rad - Math.PI / 2) < 1e-14 ? 0 : Math.cos(m.contactAngle_rad);
    const ergun = nonlinear ? (1.75 / Math.sqrt(150)) * m.fiberDiameter_m ** (exponent - 1) * m.waterViscosity_Pa_s ** (1 - exponent) * m.waterDensity_kg_m3 ** exponent : 0;
    out.set([m.thickness_m, Math.PI * m.rodRadius_m * (m.rodRadiusB_m ?? m.rodRadius_m), m.fiberDiameter_m, m.waterViscosity_Pa_s, 2 * m.surfaceTension_N_m * cosine / radius, ergun, m.solidFraction, m.waterDensity_kg_m3], MATERIAL_FLOATS * i);
  });
  return out;
}

/** WG: the transport workgroup, a power of two (the device's widest). */
const code = (topology: ClothPoreTopology, WG: number, subgroups: boolean) => /* wgsl */`
struct Params {
 counts:vec4u,    // V, T, R, C
 counts2:vec4u,   // E, G, substeps, flags (1 pressureOnManifold, 2 rest solid fraction, 4 solveSolid, 8 drag floor)
 physics0:vec4f,  // substep dt, predictor dt, density, drag exponent
 physics1:vec4f,  // gravity xyz, water quantum
 physics2:vec4f,  // drag floor reference volume
};
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> words:array<u32>;
@group(0) @binding(2) var<storage,read> floats:array<f32>;
// Four vec4 a vertex: position + control volume, pressure acceleration,
// solid acceleration, (dry mass, solid volume, solid fraction, fixed).
@group(0) @binding(3) var<storage,read> vertices:array<vec4f>;
@group(0) @binding(4) var<storage,read_write> plan:array<vec4f>;
@group(0) @binding(5) var<storage,read_write> status:array<atomic<u32>>;
@group(0) @binding(6) var<storage,read_write> velocities:array<vec4f>;
@group(0) @binding(7) var<storage,read_write> water:array<i32>;
@group(0) @binding(8) var<storage,read_write> work:array<u32>;
const O_TRI=${topology.offsets.tri}u;const O_ROD=${topology.offsets.rod}u;const O_TERM=${topology.offsets.term}u;const O_SLOT=${topology.offsets.slot}u;
const O_SLOT_START=${topology.offsets.slotStart}u;const O_SLOT_TERMS=${topology.offsets.slotTerms}u;const O_VERT_START=${topology.offsets.vertStart}u;const O_VERT_SLOTS=${topology.offsets.vertSlots}u;
const O_TRI_W=${topology.offsets.triWeights}u;const O_ROD_REST=${topology.offsets.rodRest}u;const O_MAT=${topology.offsets.materials}u;
const PLAN_GAUSS=${Math.ceil(Math.max(1, topology.V) / 4)}u;const PLAN_TERMS=PLAN_GAUSS+${GAUSS_VEC4}u*${Math.max(1, topology.G)}u;
const WG=${WG}u;const NONE=0xffffffffu;
fn V()->u32{return params.counts.x;}fn T()->u32{return params.counts.y;}fn R()->u32{return params.counts.z;}fn C()->u32{return params.counts.w;}
fn E()->u32{return params.counts2.x;}fn G()->u32{return params.counts2.y;}
fn flag(bit:u32)->bool{return (params.counts2.w&bit)!=0u;}
fn fail(bit:u32){atomicOr(&status[0],bit);}
fn position(v:u32)->vec3f{return vertices[4u*v].xyz;}
fn controlVolume(v:u32)->f32{return vertices[4u*v].w;}
fn pressureAcceleration(v:u32)->vec3f{return vertices[4u*v+1u].xyz;}
fn solidAcceleration(v:u32)->vec3f{return vertices[4u*v+2u].xyz;}
fn dryMass(v:u32)->f32{return vertices[4u*v+3u].x;}
fn solidVolume(v:u32)->f32{return vertices[4u*v+3u].y;}
fn solidFraction(v:u32)->f32{return vertices[4u*v+3u].z;}
fn fixedVertex(v:u32)->bool{return vertices[4u*v+3u].w!=0.0;}
fn material(m:u32,k:u32)->f32{return floats[O_MAT+${MATERIAL_FLOATS}u*m+k];}
fn capacity(v:u32)->f32{return plan[v/4u][v%4u];}
fn suction(m:u32,f:f32)->f32{return material(m,4u)*f/(1.0-f);}
fn permeabilityValid(f:f32,d:f32)->bool{
 if(!(f>0.0&&f<1.0)){return false;}let fit=-log(f)-1.476+2.0*f;
 return (fit-0.5*f*f)>0.0&&(fit-1.774*f*f+4.078*f*f*f)>0.0&&d>0.0;
}
fn alongPermeability(f:f32,d:f32)->f32{let fit=-log(f)-1.476+2.0*f;return (fit-0.5*f*f)*d*d/(16.0*f);}
// A Gauss predictor: triangles first, then rod segments. Vertex k of it.
fn gaussCount(g:u32)->u32{return select(2u,3u,g<T());}
fn gaussVertex(g:u32,k:u32)->u32{if(g<T()){return words[O_TRI+4u*g+k];}return words[O_ROD+4u*(g-T())+k];}
fn gaussMaterial(g:u32)->u32{if(g<T()){return words[O_TRI+4u*g+3u];}return words[O_ROD+4u*(g-T())+2u];}
fn gaussWeight(g:u32,k:u32)->f32{if(g<T()){return floats[O_TRI_W+4u*g+k];}return 0.5;}

// One thread a vec4 of four capacities: a component store is a whole-vector
// read-modify-write on some devices, so neighbours must not share one.
@compute @workgroup_size(64) fn planVertices(@builtin(global_invocation_id) gid:vec3u){
 let w=gid.x;if(4u*w>=V()){return;}var c=vec4f(0.0);
 for(var k=0u;k<4u;k++){let v=4u*w+k;if(v<V()){c[k]=max(0.0,controlVolume(v)-solidVolume(v));}}
 plan[w]=c;
}
// Gauss record (8 vec4): (fraction, area or length, 0, 0), suction a vertex,
// gradient a vertex (3), force constant a vertex (3).
@compute @workgroup_size(64) fn planGauss(@builtin(global_invocation_id) gid:vec3u){
 let g=gid.x;if(g>=G()){return;}let n=gaussCount(g);let m=gaussMaterial(g);let base=PLAN_GAUSS+${GAUSS_VEC4}u*g;
 var size=0.0;var gradients=array<vec3f,3>(vec3f(0.0),vec3f(0.0),vec3f(0.0));
 if(g<T()){
  let p0=position(gaussVertex(g,0u));let p1=position(gaussVertex(g,1u));let p2=position(gaussVertex(g,2u));
  let normal=cross(p1-p0,p2-p0);size=0.5*length(normal);if(!(size>0.0)){fail(${PORE_GPU_STATUS.collapsedFace}u);return;}
  let normal2=4.0*size*size;let ps=array<vec3f,3>(p0,p1,p2);
  for(var i=0u;i<3u;i++){gradients[i]=cross(normal,ps[(i+2u)%3u]-ps[(i+1u)%3u])/normal2;}
 }else{
  let a=gaussVertex(g,0u);let b=gaussVertex(g,1u);let d=position(b)-position(a);size=length(d);
  if(!(size>0.0)){fail(${PORE_GPU_STATUS.collapsedRod}u);return;}
  gradients[0]=-d/(size*size);gradients[1]=d/(size*size);
 }
 var fraction=0.0;var suctions=vec4f(0.0);
 for(var k=0u;k<n;k++){
  let v=gaussVertex(g,k);let w=gaussWeight(g,k);let f=solidFraction(v);fraction+=w*f;suctions[k]=suction(m,f);
  let pressure=select(vec3f(0.0),pressureAcceleration(v),flag(1u));
  plan[base+2u+k]=vec4f(gradients[k],0.0);plan[base+5u+k]=vec4f(w*(params.physics1.xyz+pressure-solidAcceleration(v)),0.0);
 }
 if(n==2u){plan[base+4u]=vec4f(0.0);plan[base+7u]=vec4f(0.0);}
 plan[base]=vec4f(fraction,size,0.0,0.0);plan[base+1u]=suctions;
}
// Term record (2 vec4): (A/L (1-psi)^2, linear drag, nonlinear drag at unit
// speed power, gravity-frame drive), (suction a, suction b, open, gauss).
@compute @workgroup_size(64) fn planTerms(@builtin(global_invocation_id) gid:vec3u){
 let k=gid.x;if(k>=C()){return;}let a=words[O_TERM+4u*k];let b=words[O_TERM+4u*k+1u];let aux=words[O_TERM+4u*k+3u];
 var g=0u;var m=0u;var areaOverLength=0.0;
 if((aux>>31u)==0u){
  let t=aux/3u;let j=aux%3u;g=t;m=words[O_TRI+4u*t+3u];let opposite=words[O_TRI+4u*t+(j+2u)%3u];
  let area=0.5*length(cross(position(words[O_TRI+4u*t+1u])-position(words[O_TRI+4u*t]),position(words[O_TRI+4u*t+2u])-position(words[O_TRI+4u*t])));
  var thickness=material(m,0u);if(flag(2u)){thickness*=floats[O_TRI_W+4u*t+3u]/area;}
  // -V grad(N_a).grad(N_b), kept signed; a numerically nil one is no edge.
  let coefficient=thickness*dot(position(a)-position(opposite),position(b)-position(opposite))/(4.0*area);
  areaOverLength=select(coefficient,0.0,abs(coefficient)<1e-12*abs(thickness)||!(area>0.0));
 }else{
  let r=aux&0x7fffffffu;g=T()+r;m=words[O_ROD+4u*r+2u];let len=length(position(b)-position(a));
  areaOverLength=select(0.0,material(m,1u)*select(1.0,floats[O_ROD_REST+r]/len,flag(2u))/len,len>0.0);
 }
 let fa=solidFraction(a);let fb=solidFraction(b);
 let open=capacity(a)>0.0&&capacity(b)>0.0&&areaOverLength!=0.0&&fa>0.0&&fa<1.0&&fb>0.0&&fb<1.0;
 let base=PLAN_TERMS+${TERM_VEC4}u*k;
 if(!open){plan[base]=vec4f(0.0);plan[base+1u]=vec4f(0.0,0.0,0.0,f32(g));return;}
 // The oracle evaluates every member's suction and the Gauss permeability.
 let d=material(m,2u);let fraction=plan[PLAN_GAUSS+${GAUSS_VEC4}u*g].x;var valid=permeabilityValid(fraction,d)&&permeabilityValid(material(m,6u),d);
 for(var i=0u;i<gaussCount(g);i++){valid=valid&&permeabilityValid(solidFraction(gaussVertex(g,i)),d);}
 if(!valid){fail(${PORE_GPU_STATUS.permeability}u);}
 let K=alongPermeability(fraction,d);let linear=material(m,3u)/K;let nonlinear=material(m,5u)/(pow(1.0-fraction,1.5)*sqrt(K));
 let pressure=select(vec3f(0.0),0.5*(pressureAcceleration(a)+pressureAcceleration(b)),flag(1u));
 let drive=params.physics0.z*dot(params.physics1.xyz+pressure-0.5*(solidAcceleration(a)+solidAcceleration(b)),position(b)-position(a));
 plan[base]=vec4f(areaOverLength*(1.0-fraction)*(1.0-fraction),linear,nonlinear,drive);
 plan[base+1u]=vec4f(suction(m,fa),suction(m,fb),1.0,f32(g));
}

// ---- One workgroup: every substep, Picard around Jacobi-BiCGSTAB. ----
${subgroups ? /* wgsl */`
// Subgroup reductions: each subgroup's partial, then every subgroup folds the
// partials; three barriers a reduction instead of log2(WG).
var<private> SID:u32;var<private> SGS:u32;
var<workgroup> partial:array<vec4f,WG/32u>;var<workgroup> total:vec4f;
fn sum4(lid:u32,x:vec4f)->vec4f{
 let s=subgroupAdd(x);if(SID==0u){partial[lid/SGS]=s;}workgroupBarrier();
 var y=vec4f(0.0);for(var k=SID;k<WG/SGS;k+=SGS){y+=partial[k];}let t=subgroupAdd(y);
 if(lid==0u){total=t;}let r=workgroupUniformLoad(&total);workgroupBarrier();return r;
}
fn max4(lid:u32,x:vec4f)->vec4f{
 let s=subgroupMax(x);if(SID==0u){partial[lid/SGS]=s;}workgroupBarrier();
 var y=vec4f(-3.0e38);for(var k=SID;k<WG/SGS;k+=SGS){y=max(y,partial[k]);}let t=subgroupMax(y);
 if(lid==0u){total=t;}let r=workgroupUniformLoad(&total);workgroupBarrier();return r;
}` : /* wgsl */`
var<workgroup> red:array<vec4f,WG>;
fn sum4(lid:u32,x:vec4f)->vec4f{
 red[lid]=x;workgroupBarrier();
 for(var s=WG/2u;s>0u;s>>=1u){if(lid<s){red[lid]+=red[lid+s];}workgroupBarrier();}
 let r=workgroupUniformLoad(&red[0]);workgroupBarrier();return r;
}
fn max4(lid:u32,x:vec4f)->vec4f{
 red[lid]=x;workgroupBarrier();
 for(var s=WG/2u;s>0u;s>>=1u){if(lid<s){red[lid]=max(red[lid],red[lid+s]);}workgroupBarrier();}
 let r=workgroupUniformLoad(&red[0]);workgroupBarrier();return r;
}`}
fn sync(){storageBarrier();workgroupBarrier();}
// A raised status bit, as a workgroup-uniform value.
fn failed(lid:u32,mask:u32)->bool{sync();return max4(lid,vec4f(select(0.0,1.0,(atomicLoad(&status[0])&mask)!=0u),0.0,0.0,0.0)).x>0.0;}
// Work layout: vectors of padded V, then edge arrays of padded E, then Gauss arrays.
const VS=${Math.ceil(Math.max(1, topology.V) / 64) * 64}u;const ES=${Math.ceil(Math.max(1, topology.E) / 64) * 64}u;
const X=0u;const RR=1u;const HAT=2u;const P=3u;const VV=4u;const SS=5u;const Y=6u;const Z=7u;const TT=8u;const DIAG=9u;const B=10u;const GUESS=11u;const CAP=12u;const OLD=13u;const VEL=14u;const MASS=15u;const DM=16u;const OUT=17u;const X0=18u;const GAMMA=19u;
// Momentum triples: x, y, z vectors at k, k+1, k+2.
const MX=20u;const MR=23u;const MHAT=26u;const MP=29u;const MVV=32u;const MS=35u;const MY=38u;const MZ=41u;const MT=44u;const MB=47u;const MVEL=50u;const VECTORS=53u;
const AB=0u;const BA=1u;const SRC=2u;const FLUX=3u;const LIMITED=4u;const MAB=5u;const MBA=6u;const F0=8u;const CUMULATIVE=7u;const EDGES=9u;
// Words, so integer arrays never pass through float registers.
fn vec(k:u32,i:u32)->f32{return bitcast<f32>(work[k*VS+i]);}
fn setVec(k:u32,i:u32,x:f32){work[k*VS+i]=bitcast<u32>(x);}
fn vecWord(k:u32,i:u32)->u32{return work[k*VS+i];}
fn setVecWord(k:u32,i:u32,x:u32){work[k*VS+i]=x;}
fn edge(k:u32,e:u32)->f32{return bitcast<f32>(work[VECTORS*VS+k*ES+e]);}
fn setEdge(k:u32,e:u32,x:f32){work[VECTORS*VS+k*ES+e]=bitcast<u32>(x);}
fn edgeInt(k:u32,e:u32)->i32{return bitcast<i32>(work[VECTORS*VS+k*ES+e]);}
fn setEdgeInt(k:u32,e:u32,x:i32){work[VECTORS*VS+k*ES+e]=bitcast<u32>(x);}
fn gaussSpeed(g:u32)->f32{return bitcast<f32>(work[VECTORS*VS+EDGES*ES+2u*g]);}
fn gaussFloor(g:u32)->f32{return bitcast<f32>(work[VECTORS*VS+EDGES*ES+2u*g+1u]);}
fn slotA(e:u32)->u32{return words[O_SLOT+2u*e];}
fn slotB(e:u32)->u32{return words[O_SLOT+2u*e+1u];}
// Identity rows: a zero-capacity control in the mass solve, a fixed vertex in momentum.
fn identityRow(mode:u32,i:u32)->bool{return select(capacity(i)==0.0,fixedVertex(i),mode==1u);}

// out = A src: diag(CAP) plus the signed edge fluxes ab x_a - ba x_b.
fn apply(lid:u32,mode:u32,dst:u32,src:u32,ab:u32,ba:u32){
 for(var i=lid;i<V();i+=WG){
  let x=vec(src,i);if(identityRow(mode,i)){setVec(dst,i,x);continue;}
  var out=vec(CAP,i)*x;
  for(var j=words[O_VERT_START+i];j<words[O_VERT_START+i+1u];j++){
   let entry=words[O_VERT_SLOTS+j];let e=entry&0x7fffffffu;let flux=edge(ab,e)*vec(src,slotA(e))-edge(ba,e)*vec(src,slotB(e));
   out+=select(flux,-flux,(entry>>31u)!=0u);
  }
  setVec(dst,i,out);
 }
 sync();
}
// The unit scale of a system whose larger magnitude of |b| and ||A|| |x0| is
// m: 2^-e with m 2^-e in [1/2,1), and its inverse. A zero (closed) or
// non-finite m keeps 1; e stays where both powers are normal f32.
fn krylovScale(m:f32)->vec2f{
 var e=0;if(m>0.0&&m<3.4e38){e=clamp(frexp(m).exp,-126,126);}
 return vec2f(ldexp(1.0,-e),ldexp(1.0,e));
}
fn unscaleX(lid:u32,inv:f32){for(var i=lid;i<V();i+=WG){setVec(X,i,inv*vec(X,i));}sync();}
// Jacobi-preconditioned BiCGSTAB on the (power-of-two scaled) system in X,
// B. Stops at the normwise backward error ||r|| <= 16u (||b|| + ||A|| ||x||):
// below it f32 cannot resolve the residual. Returns the iterations.
// It solves the unit-scaled system 2^-e (b, x0) (krylovScale) and returns x
// at the caller's scale. At round-off magnitudes (a transverse free-fall
// momentum near 1e-25, a nearly dry cloth) the inner products of the raw
// system square below f32's range: rho and t.t flushed to zero, a breakdown
// at iteration 0. The power of two is exact, A is unchanged, and the
// backward error, so the stopping rule, is invariant under it.
const KRYLOV_TOLERANCE=16.0*5.9604645e-8;const KRYLOV_BUDGET=2000u;
fn solve(lid:u32,mode:u32,ab:u32,ba:u32)->u32{
 var rowMax=vec4f(0.0);
 for(var i=lid;i<V();i+=WG){
  var diag=1.0;var row=1.0;
  if(!identityRow(mode,i)){diag=vec(CAP,i);row=abs(diag);
   for(var j=words[O_VERT_START+i];j<words[O_VERT_START+i+1u];j++){let entry=words[O_VERT_SLOTS+j];let e=entry&0x7fffffffu;
    diag+=select(edge(ab,e),edge(ba,e),(entry>>31u)!=0u);row+=abs(edge(ab,e))+abs(edge(ba,e));}
  }
  setVec(DIAG,i,diag);rowMax=max(rowMax,vec4f(row,abs(vec(B,i)),abs(vec(X,i)),0.0));
 }
 sync();let scale=max4(lid,rowMax);let normA=scale.x;
 let unit=krylovScale(max(scale.y,normA*scale.z));let normB=unit.x*scale.y;
 for(var i=lid;i<V();i+=WG){setVec(X,i,unit.x*vec(X,i));}
 sync();
 apply(lid,mode,VV,X,ab,ba);
 var m=vec4f(0.0);
 for(var i=lid;i<V();i+=WG){let r=unit.x*vec(B,i)-vec(VV,i);setVec(RR,i,r);setVec(HAT,i,r);setVec(P,i,0.0);setVec(VV,i,0.0);m=max(m,vec4f(abs(r),abs(vec(X,i)),0.0,0.0));}
 sync();m=max4(lid,m);
 var residual=m.x;var tolerance=KRYLOV_TOLERANCE*(normB+normA*m.y);
 var rhoOld=1.0;var alpha=1.0;var omega=1.0;var iterations=0u;
 loop{
  if(residual<=tolerance||iterations>=KRYLOV_BUDGET){break;}
  var local=0.0;for(var i=lid;i<V();i+=WG){local+=vec(HAT,i)*vec(RR,i);}
  let rho=sum4(lid,vec4f(local,0.0,0.0,0.0)).x;
  if(!(abs(rho)>0.0)||!(abs(rho)<3.0e38)){if(lid==0u){fail(${PORE_GPU_STATUS.krylov}u);}unscaleX(lid,unit.y);return iterations;}
  let beta=(rho/rhoOld)*(alpha/omega);
  for(var i=lid;i<V();i+=WG){let p=vec(RR,i)+beta*(vec(P,i)-omega*vec(VV,i));setVec(P,i,p);setVec(Y,i,p/vec(DIAG,i));}
  sync();apply(lid,mode,VV,Y,ab,ba);
  local=0.0;for(var i=lid;i<V();i+=WG){local+=vec(HAT,i)*vec(VV,i);}
  let divisor=sum4(lid,vec4f(local,0.0,0.0,0.0)).x;
  if(divisor==0.0){if(lid==0u){fail(${PORE_GPU_STATUS.krylov}u);}unscaleX(lid,unit.y);return iterations;}
  alpha=rho/divisor;
  var sm=0.0;for(var i=lid;i<V();i+=WG){let s=vec(RR,i)-alpha*vec(VV,i);setVec(SS,i,s);setVec(Z,i,s/vec(DIAG,i));sm=max(sm,abs(s));}
  sync();let sNorm=max4(lid,vec4f(sm,0.0,0.0,0.0)).x;
  if(sNorm<=tolerance){for(var i=lid;i<V();i+=WG){setVec(X,i,vec(X,i)+alpha*vec(Y,i));}sync();residual=sNorm;iterations++;break;}
  apply(lid,mode,TT,Z,ab,ba);
  var dots=vec4f(0.0);for(var i=lid;i<V();i+=WG){let t=vec(TT,i);dots+=vec4f(t*t,t*vec(SS,i),0.0,0.0);}
  dots=sum4(lid,dots);
  if(dots.x==0.0){if(lid==0u){fail(${PORE_GPU_STATUS.krylov}u);}unscaleX(lid,unit.y);return iterations;}
  omega=dots.y/dots.x;if(omega==0.0||!(abs(omega)<3.0e38)){if(lid==0u){fail(${PORE_GPU_STATUS.krylov}u);}unscaleX(lid,unit.y);return iterations;}
  m=vec4f(0.0);
  for(var i=lid;i<V();i+=WG){let x=vec(X,i)+alpha*vec(Y,i)+omega*vec(Z,i);setVec(X,i,x);let r=vec(SS,i)-omega*vec(TT,i);setVec(RR,i,r);m=max(m,vec4f(abs(r),abs(x),0.0,0.0));}
  sync();m=max4(lid,m);residual=m.x;tolerance=KRYLOV_TOLERANCE*(normB+normA*m.y);
  rhoOld=rho;iterations++;
 }
 if(!(residual<=tolerance)&&lid==0u){fail(${PORE_GPU_STATUS.krylov}u);}
 unscaleX(lid,unit.y);return iterations;
}

fn get3(k:u32,i:u32)->vec3f{return vec3f(vec(k,i),vec(k+1u,i),vec(k+2u,i));}
fn set3(k:u32,i:u32,x:vec3f){setVec(k,i,x.x);setVec(k+1u,i,x.y);setVec(k+2u,i,x.z);}
// The momentum operator on three velocity components at once: CAP diagonal
// plus the upwind mass fluxes; a fixed vertex is an identity row.
fn apply3(lid:u32,dst:u32,src:u32){
 for(var i=lid;i<V();i+=WG){
  let x=get3(src,i);if(fixedVertex(i)){set3(dst,i,x);continue;}
  var out=vec(CAP,i)*x;
  for(var j=words[O_VERT_START+i];j<words[O_VERT_START+i+1u];j++){
   let entry=words[O_VERT_SLOTS+j];let e=entry&0x7fffffffu;let flux=edge(MAB,e)*get3(src,slotA(e))-edge(MBA,e)*get3(src,slotB(e));
   out+=select(flux,-flux,(entry>>31u)!=0u);
  }
  set3(dst,i,out);
 }
 sync();
}
fn unscaleMX(lid:u32,inv:vec3f){for(var i=lid;i<V();i+=WG){set3(MX,i,inv*get3(MX,i));}sync();}
// Jacobi BiCGSTAB on the three momentum systems together (one matrix, three
// right-hand sides): every reduction and barrier serves all three. Each axis
// keeps its own scalars and stops at its own normwise backward error.
fn solve3(lid:u32)->u32{
 var rowMax=vec4f(0.0);var xmax=vec3f(0.0);
 for(var i=lid;i<V();i+=WG){
  var diag=1.0;var row=1.0;
  if(!fixedVertex(i)){diag=vec(CAP,i);row=abs(diag);
   for(var j=words[O_VERT_START+i];j<words[O_VERT_START+i+1u];j++){let entry=words[O_VERT_SLOTS+j];let e=entry&0x7fffffffu;
    diag+=select(edge(MAB,e),edge(MBA,e),(entry>>31u)!=0u);row+=abs(edge(MAB,e))+abs(edge(MBA,e));}
  }
  setVec(DIAG,i,diag);rowMax=max(rowMax,vec4f(row,abs(get3(MB,i))));xmax=max(xmax,abs(get3(MX,i)));
 }
 sync();let scale=max4(lid,rowMax);let normA=scale.x;
 // Each axis solves its own unit-scaled system (krylovScale, as solve).
 let m0=max(scale.yzw,normA*max4(lid,vec4f(xmax,0.0)).xyz);
 let ux=krylovScale(m0.x);let uy=krylovScale(m0.y);let uz=krylovScale(m0.z);
 let unit=vec3f(ux.x,uy.x,uz.x);let restore=vec3f(ux.y,uy.y,uz.y);let normB=unit*scale.yzw;
 for(var i=lid;i<V();i+=WG){set3(MX,i,unit*get3(MX,i));}
 sync();
 apply3(lid,MVV,MX);
 var rm=vec3f(0.0);var xm=vec3f(0.0);
 for(var i=lid;i<V();i+=WG){let r=unit*get3(MB,i)-get3(MVV,i);set3(MR,i,r);set3(MHAT,i,r);set3(MP,i,vec3f(0.0));set3(MVV,i,vec3f(0.0));rm=max(rm,abs(r));xm=max(xm,abs(get3(MX,i)));}
 sync();var residual=max4(lid,vec4f(rm,0.0)).xyz;var tolerance=KRYLOV_TOLERANCE*(normB+normA*max4(lid,vec4f(xm,0.0)).xyz);
 var rhoOld=vec3f(1.0);var alpha=vec3f(1.0);var omega=vec3f(1.0);var iterations=0u;
 loop{
  let open=residual>tolerance;
  if(!any(open)||iterations>=KRYLOV_BUDGET){break;}
  var local=vec3f(0.0);for(var i=lid;i<V();i+=WG){local+=get3(MHAT,i)*get3(MR,i);}
  let rho=sum4(lid,vec4f(local,0.0)).xyz;
  if(any(open&(!(abs(rho)>vec3f(0.0))|!(abs(rho)<vec3f(3.0e38))))){if(lid==0u){fail(${PORE_GPU_STATUS.krylov}u);}unscaleMX(lid,restore);return iterations;}
  let on=select(vec3f(0.0),vec3f(1.0),open);
  let beta=select(vec3f(0.0),(rho/rhoOld)*(alpha/omega),open);
  for(var i=lid;i<V();i+=WG){let p=get3(MR,i)+beta*(get3(MP,i)-omega*get3(MVV,i));set3(MP,i,p);set3(MY,i,on*p/vec(DIAG,i));}
  sync();apply3(lid,MVV,MY);
  local=vec3f(0.0);for(var i=lid;i<V();i+=WG){local+=get3(MHAT,i)*get3(MVV,i);}
  let divisor=sum4(lid,vec4f(local,0.0)).xyz;
  if(any(open&(divisor==vec3f(0.0)))){if(lid==0u){fail(${PORE_GPU_STATUS.krylov}u);}unscaleMX(lid,restore);return iterations;}
  alpha=select(vec3f(0.0),rho/divisor,open);
  var sm=vec3f(0.0);for(var i=lid;i<V();i+=WG){let s=get3(MR,i)-alpha*get3(MVV,i);set3(MS,i,s);sm=max(sm,abs(s));}
  sync();let sNorm=max4(lid,vec4f(sm,0.0)).xyz;
  // An axis whose half step converges takes it and stops (omega 0).
  let half=open&(sNorm<=tolerance);let full=open&!half;let onFull=select(vec3f(0.0),vec3f(1.0),full);
  for(var i=lid;i<V();i+=WG){set3(MZ,i,onFull*get3(MS,i)/vec(DIAG,i));}
  sync();apply3(lid,MT,MZ);
  var tt=vec3f(0.0);var ts=vec3f(0.0);for(var i=lid;i<V();i+=WG){let t=get3(MT,i);tt+=t*t;ts+=t*get3(MS,i);}
  tt=sum4(lid,vec4f(tt,0.0)).xyz;ts=sum4(lid,vec4f(ts,0.0)).xyz;
  if(any(full&(tt==vec3f(0.0)))){if(lid==0u){fail(${PORE_GPU_STATUS.krylov}u);}unscaleMX(lid,restore);return iterations;}
  let nextOmega=select(vec3f(0.0),ts/tt,full);
  if(any(full&((nextOmega==vec3f(0.0))|!(abs(nextOmega)<vec3f(3.0e38))))){if(lid==0u){fail(${PORE_GPU_STATUS.krylov}u);}unscaleMX(lid,restore);return iterations;}
  rm=vec3f(0.0);xm=vec3f(0.0);
  for(var i=lid;i<V();i+=WG){
   let x=get3(MX,i)+alpha*get3(MY,i)+nextOmega*get3(MZ,i);set3(MX,i,x);
   let r=select(get3(MR,i),get3(MS,i)-nextOmega*get3(MT,i),full);set3(MR,i,r);rm=max(rm,abs(r));xm=max(xm,abs(x));
  }
  sync();
  // A half-converged axis keeps its s residual; the rest take r.
  residual=select(max4(lid,vec4f(rm,0.0)).xyz,sNorm,half);residual=select(residual,vec3f(0.0),half);
  tolerance=select(tolerance,KRYLOV_TOLERANCE*(normB+normA*max4(lid,vec4f(xm,0.0)).xyz),open);
  rhoOld=select(rhoOld,rho,open);omega=select(omega,nextOmega,full);iterations++;
 }
 if(any(residual>tolerance)&&lid==0u){fail(${PORE_GPU_STATUS.krylov}u);}
 unscaleMX(lid,restore);return iterations;
}

// Saturation-dependent edges at GUESS (Eq. 18): mobility (1-psi) eps_L / C
// with the Gauss-wide nonlinear drag, the full suction difference, and the
// gravity-frame drift; scaled by sigma like CAP.
fn buildEdges(lid:u32,dt:f32,sigma:f32){
 let rho=params.physics0.z;let predictorDt=params.physics0.y;let exponent=params.physics0.w;let floorOn=flag(8u);
 for(var g=lid;g<G();g+=WG){
  let base=PLAN_GAUSS+${GAUSS_VEC4}u*g;var acceleration=vec3f(0.0);var fluid=0.0;
  for(var k=0u;k<gaussCount(g);k++){let v=gaussVertex(g,k);let S=vec(GUESS,v);let pressure=plan[base+1u][k]*(1.0-clamp(S,0.0,1.0));
   acceleration+=plan[base+2u+k].xyz*pressure/rho+plan[base+5u+k].xyz;fluid+=gaussWeight(g,k)*capacity(v)*max(0.0,S);}
  work[VECTORS*VS+EDGES*ES+2u*g]=bitcast<u32>(pow(abs((1.0-plan[base].x)*predictorDt*length(acceleration)),exponent));
  work[VECTORS*VS+EDGES*ES+2u*g+1u]=bitcast<u32>(rho*fluid);
 }
 sync();
 for(var e=lid;e<E();e+=WG){
  var ab=0.0;var ba=0.0;var source=0.0;let a=slotA(e);let b=slotB(e);let Sa=vec(GUESS,a);let Sb=vec(GUESS,b);
  let sa=clamp(Sa,0.0,1.0);let sb=clamp(Sb,0.0,1.0);let clampA=select(1.0,sa/Sa,Sa>0.0);let clampB=select(1.0,sb/Sb,Sb>0.0);
  for(var j=words[O_SLOT_START+e];j<words[O_SLOT_START+e+1u];j++){
   let k=words[O_SLOT_TERMS+j];let t0=plan[PLAN_TERMS+${TERM_VEC4}u*k];let t1=plan[PLAN_TERMS+${TERM_VEC4}u*k+1u];if(t1.z==0.0){continue;}
   let g=u32(t1.w);let speed=gaussSpeed(g);
   if(!(speed<3.0e38)){fail(${PORE_GPU_STATUS.predictor}u);}
   var drag=t0.y+t0.z*speed;if(floorOn){drag=max(drag,0.001*gaussFloor(g)/(predictorDt*params.physics2.x));}
   let base=t0.x/drag;let mobility=base*0.5*(sa+sb);let drift=base*t0.w;
   ab+=dt*(mobility*t1.x*clampA+max(drift,0.0));ba+=dt*(mobility*t1.y*clampB+max(-drift,0.0));source+=dt*mobility*(t1.y-t1.x);
  }
  setEdge(AB,e,sigma*ab);setEdge(BA,e,sigma*ba);setEdge(SRC,e,sigma*source);
  // The edge flux at the substep's start saturation, at these edges.
  setEdge(F0,e,sigma*(ab*vec(X0,a)-ba*vec(X0,b)+source));
 }
 sync();
 // Increment form: A delta = -div F0 (cap x0 is the old water exactly), so
 // f32 resolves the change itself, not the change against the inventory.
 for(var i=lid;i<V();i+=WG){
  var b=0.0;
  if(!identityRow(0u,i)){for(var j=words[O_VERT_START+i];j<words[O_VERT_START+i+1u];j++){let entry=words[O_VERT_SLOTS+j];let f=edge(F0,entry&0x7fffffffu);b+=select(-f,f,(entry>>31u)!=0u);}}
  setVec(B,i,b);
 }
 sync();
}
fn powerOfTwoBelow(x:f32)->f32{return exp2(-floor(log2(x)));}
// Output words after the status bits: Krylov iterations, max Picard
// iterations, then f32 bits: moved, limited volume, support reaction xyz.
@compute @workgroup_size(WG) fn transport(@builtin(local_invocation_index) lid:u32${subgroups ? ',@builtin(subgroup_invocation_id) sid:u32,@builtin(subgroup_size) sgs:u32' : ''}){
${subgroups ? ' SID=sid;SGS=sgs;\n' : ''} let substeps=params.counts2.z;let dt=params.physics0.x;let q=params.physics1.w;let rho=params.physics0.z;
 var capMax=0.0;var wet=0;for(var i=lid;i<V();i+=WG){capMax=max(capMax,capacity(i));wet=max(wet,water[i]);}
 let extent=max4(lid,vec4f(capMax,f32(wet),0.0,0.0));capMax=extent.x;
 // A dry cloth has nothing to transport: the oracle's early exit, including
 // its plan checks (water can arrive on the device, from GPU capture).
 if(!(extent.y>0.0)){if(lid==0u){atomicStore(&status[0],0u);}return;}
 if(!(capMax>0.0)||failed(lid,${PORE_GPU_STATUS.collapsedFace | PORE_GPU_STATUS.collapsedRod | PORE_GPU_STATUS.permeability}u)){return;}
 let sigma=powerOfTwoBelow(capMax);
 for(var e=lid;e<E();e+=WG){setEdgeInt(CUMULATIVE,e,0);}
 var krylov=0u;var picardMax=0u;var moved=0.0;var limitedVolume=0.0;var reaction=vec3f(0.0);
 for(var step=0u;step<substeps;step++){
  // Saturation unknowns; a zero-capacity control keeps its water as an identity row.
  for(var i=lid;i<V();i+=WG){
   let old=f32(water[i])*q;setVec(OLD,i,old);let c=capacity(i);setVec(CAP,i,sigma*c);
   let x=select(old,old/c,c>0.0);setVec(X0,i,x);setVec(GUESS,i,x);setVec(GAMMA,i,0.0);setVec(X,i,0.0);
  }
  sync();
  buildEdges(lid,dt,sigma);krylov+=solve(lid,0u,AB,BA);
  var relaxation=1.0;var previous=3.0e38;var picard=0u;
  loop{
   // X is the saturation increment, GAMMA the increment the edges were built at.
   var local=vec4f(0.0);for(var i=lid;i<V();i+=WG){if(identityRow(0u,i)){continue;}let x=vec(X,i);local=max(local,vec4f(abs(x-vec(GAMMA,i)),abs(x),0.0,0.0));}
   let m=max4(lid,local);
   // The fixed point to the f32 Krylov precision of the increment.
   if(m.x<=PICARD_TOLERANCE*m.y){break;}
   if(picard>=96u){if(lid==0u){fail(${PORE_GPU_STATUS.picard}u);}break;}
   if(m.x>=0.5*previous){relaxation=max(0.125,0.5*relaxation);}previous=m.x;
   for(var i=lid;i<V();i+=WG){let g=relaxation*vec(X,i)+(1.0-relaxation)*vec(GAMMA,i);setVec(GAMMA,i,g);setVec(GUESS,i,vec(X0,i)+g);}
   sync();buildEdges(lid,dt,sigma);krylov+=solve(lid,0u,AB,BA);picard++;
  }
  picardMax=max(picardMax,picard);
  if(failed(lid,${PORE_GPU_STATUS.krylov | PORE_GPU_STATUS.picard | PORE_GPU_STATUS.predictor}u)){break;}
  // Accepted edge fluxes in whole quanta, a->b positive.
  for(var e=lid;e<E();e+=WG){
   let f=(edge(F0,e)+edge(AB,e)*vec(X,slotA(e))-edge(BA,e)*vec(X,slotB(e)))/(sigma*q);
   if(!(abs(f)<2147483520.0)){fail(${PORE_GPU_STATUS.quantum}u);setEdgeInt(FLUX,e,0);continue;}
   setEdgeInt(FLUX,e,i32(round(f)));
  }
  sync();
  // Paired donor limiter in exact integers, as the oracle: only when some
  // control would go negative, every donor whose outgoing quanta exceed its
  // water scales all of its outgoing fluxes together.
  var negative=0.0;
  for(var i=lid;i<V();i+=WG){
   var out=0u;var into=0u;
   for(var j=words[O_VERT_START+i];j<words[O_VERT_START+i+1u];j++){let entry=words[O_VERT_SLOTS+j];let f=edgeInt(FLUX,entry&0x7fffffffu);
    let n=u32(abs(f));let donates=select(f>0,f<0,(entry>>31u)!=0u);
    if(donates){out=select(out+n,0xffffffffu,out+n<out);}else{into=select(into+n,0xffffffffu,into+n<into);}}
   let have=u32(max(water[i],0));let available=select(have+into,0xffffffffu,have+into<have);
   if(out>available){negative=1.0;}
   setVecWord(OUT,i,out);
  }
  sync();
  let limiting=max4(lid,vec4f(negative,0.0,0.0,0.0)).x>0.0;
  var limitedLocal=0.0;var movedLocal=0.0;
  for(var e=lid;e<E();e+=WG){
   let f=edgeInt(FLUX,e);let donor=select(slotB(e),slotA(e),f>=0);let out=vecWord(OUT,donor);let have=u32(max(water[donor],0));
   var accepted=f;
   if(limiting&&out>have){let fraction=f32(have)/f32(out);accepted=i32(floor(f32(abs(f))*fraction*(1.0-9.5367432e-7)))*select(-1,1,f>=0);}
   setEdgeInt(LIMITED,e,accepted);setEdgeInt(CUMULATIVE,e,edgeInt(CUMULATIVE,e)+accepted);limitedLocal+=f32(abs(f-accepted))*q;movedLocal+=f32(abs(accepted))*q;
   let volume=f32(accepted)*q;setEdge(MAB,e,rho*max(volume,0.0));setEdge(MBA,e,rho*max(-volume,0.0));
  }
  sync();
  let totals=sum4(lid,vec4f(limitedLocal,movedLocal,0.0,0.0));limitedVolume+=totals.x;moved+=totals.y;
  var massMax=0.0;
  for(var i=lid;i<V();i+=WG){
   var n=water[i];var gain=0.0;
   for(var j=words[O_VERT_START+i];j<words[O_VERT_START+i+1u];j++){let entry=words[O_VERT_SLOTS+j];let f=edgeInt(LIMITED,entry&0x7fffffffu);
    let into=select(-f,f,(entry>>31u)!=0u);n+=into;gain+=f32(max(into,0));}
   if(!(f32(water[i])+gain<2147483520.0)){fail(${PORE_GPU_STATUS.quantum}u);}
   let change=n-water[i];water[i]=n;let mass=dryMass(i)+rho*f32(n)*q;setVec(MASS,i,mass);setVec(DM,i,rho*f32(change)*q);massMax=max(massMax,mass);
  }
  sync();
  // Transported momentum: the same accepted fluxes, implicit upwind.
  let solveSolid=flag(4u);
  if(!solveSolid){
   var r=vec3f(0.0);for(var i=lid;i<V();i+=WG){r+=vec(DM,i)*velocities[i].xyz;}
   reaction+=sum4(lid,vec4f(r,0.0)).xyz;continue;
  }
  let massScale=powerOfTwoBelow(max4(lid,vec4f(massMax,0.0,0.0,0.0)).x);
  for(var e=lid;e<E();e+=WG){setEdge(MAB,e,massScale*edge(MAB,e));setEdge(MBA,e,massScale*edge(MBA,e));}
  for(var i=lid;i<V();i+=WG){setVec(CAP,i,massScale*vec(MASS,i));}
  sync();
  // Increment form: (M + upwind) dv = -dm v - div(upwind v); fixed rows dv = 0.
  for(var i=lid;i<V();i+=WG){set3(MVEL,i,velocities[i].xyz);set3(MX,i,vec3f(0.0));}
  sync();
  for(var i=lid;i<V();i+=WG){
   var b=vec3f(0.0);
   if(!fixedVertex(i)){b=-massScale*vec(DM,i)*get3(MVEL,i);
    for(var j=words[O_VERT_START+i];j<words[O_VERT_START+i+1u];j++){let entry=words[O_VERT_SLOTS+j];let e=entry&0x7fffffffu;let f=edge(MAB,e)*get3(MVEL,slotA(e))-edge(MBA,e)*get3(MVEL,slotB(e));b+=select(-f,f,(entry>>31u)!=0u);}}
   set3(MB,i,b);
  }
  sync();krylov+=solve3(lid);
  var r=vec3f(0.0);for(var i=lid;i<V();i+=WG){let dv=get3(MX,i);let v=get3(MVEL,i);r+=vec(MASS,i)*dv+vec(DM,i)*v;velocities[i]=vec4f(v+dv,velocities[i].w);}
  sync();reaction+=sum4(lid,vec4f(r,0.0)).xyz;
 }
 if(lid==0u){atomicStore(&status[1],krylov);atomicStore(&status[2],picardMax);atomicStore(&status[3],bitcast<u32>(moved));atomicStore(&status[4],bitcast<u32>(limitedVolume));
  atomicStore(&status[5],bitcast<u32>(reaction.x));atomicStore(&status[6],bitcast<u32>(reaction.y));atomicStore(&status[7],bitcast<u32>(reaction.z));}
}
`;

export interface ClothPoreGPUStep {
  readonly positions_m: ArrayLike<number>; readonly velocities_m_s: ArrayLike<number>; readonly controlVolume_m3: ArrayLike<number>;
  readonly solidVolume_m3: ArrayLike<number>; readonly solidFraction: ArrayLike<number>; readonly dryMass_kg: ArrayLike<number>;
  readonly waterVolume_m3: ArrayLike<number>; readonly fixed?: ArrayLike<number>;
  readonly materials: readonly ClothMaterial[]; readonly physics?: ClothPhysicsOptions;
  readonly dt_s: number; readonly manifoldPredictorDt_s?: number; readonly gravity_m_s2: ClothVector;
  readonly pressureAcceleration_m_s2?: ArrayLike<number>; readonly solidAcceleration_m_s2?: ArrayLike<number>;
}
export interface ClothPoreGPUResult {
  /** Water a vertex: the moved whole quanta (exact in f64) plus the
   * sub-quantum remainder it held, which stays in place: exactly conserved. */
  readonly waterVolume_m3: Float64Array; readonly velocities_m_s: Float64Array;
  readonly iterations: number; readonly picardIterations: number; readonly movedVolume_m3: number; readonly limitedVolume_m3: number; readonly supportReaction_Ns: ClothVector;
  /** Device time of the transport pass; only with the `timing` option. */
  readonly gpu_ms?: number;
}
export interface ClothGPUPoreTransportOptions {
  /**
   * Profiling only: timestamp writes around the transport pass of `advance()`,
   * reported as `gpu_ms`. Production construction carries none.
   * Hazard: on M1 Max (2026-11-07) timestamp attachments, in a submit that
   * also runs the manifold transport, aborted the command buffer with a Metal
   * out-of-memory error (kIOGPUCommandBufferCallbackErrorOutOfMemory). Even an
   * empty end-of-pass timestamp pass after the transport did it, and without
   * them the same submit ran. Dawn reports none of this, and the GPU runner's
   * guard exits 77. A profiler that opts in must treat a failed readback or
   * rc 77 as "timing unavailable", never as a result.
   */
  readonly timing?: boolean;
}

const STATUS_BYTES = 32, EDGE_ARRAYS = 9, WORK_VECTORS = 53, CUMULATIVE_ARRAY = 7;
/** A staged step: the sub-quantum remainders kept on the host, and whether the transport runs. */
export interface ClothPoreGPUStaged { readonly residual_m3: Float64Array; readonly transport: boolean }
/** Milliseconds between two resolved timestamps at `offset` (0 when the device reordered them). */
export function clothTimestampInterval_ms(data: ArrayBuffer, offset: number): number {
  const [a, b] = new BigUint64Array(data.slice(offset, offset + 16)); return b! > a! ? Number(b! - a!) * 1e-6 : 0;
}
/** One topology's GPU pore transport. */
export class ClothGPUPoreTransport {
  readonly quantum_m3: number;
  /** Subgroup reductions where every subgroup holds at least 32 lanes (Apple, as NB-FLIP). */
  readonly subgroups: boolean;
  private readonly buffers: GPUBuffer[] = [];
  private readonly params: GPUBuffer; private readonly words: GPUBuffer; private readonly floats: GPUBuffer; private readonly vertices: GPUBuffer;
  private readonly plan: GPUBuffer; private readonly status: GPUBuffer; private readonly velocities: GPUBuffer; private readonly water: GPUBuffer; private readonly work: GPUBuffer;
  private readonly readback: GPUBuffer;
  private readonly timestamps?: { readonly set: GPUQuerySet; readonly resolve: GPUBuffer };
  private pipelines?: Record<'planVertices' | 'planGauss' | 'planTerms' | 'transport', GPUComputePipeline>;
  private group?: GPUBindGroup;
  private busy = false;
  constructor(private readonly device: GPUDevice, readonly topology: ClothPoreTopology, restControlVolume_m3: ArrayLike<number>, options: ClothGPUPoreTransportOptions = {}) {
    // Before any allocation: a refused construction leaves nothing behind.
    if (options.timing !== undefined && typeof options.timing !== 'boolean') throw new Error('Cloth GPU pore transport timing must be a boolean');
    if (options.timing && !device.features.has('timestamp-query')) throw new Error('Cloth GPU pore transport timing needs the timestamp-query feature');
    this.quantum_m3 = clothPoreQuantum(restControlVolume_m3);
    this.subgroups = device.features.has('subgroups') && ((device as GPUDevice & { adapterInfo?: { subgroupMinSize?: number } }).adapterInfo?.subgroupMinSize ?? 0) >= 32;
    const t = topology, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const make = (label: string, bytes: number, extra = 0) => { const b = device.createBuffer({ label: `Cloth GPU pore ${label}`, size: Math.max(16, Math.ceil(bytes / 16) * 16), usage: usage | extra }); this.buffers.push(b); return b; };
    this.params = device.createBuffer({ label: 'Cloth GPU pore parameters', size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); this.buffers.push(this.params);
    this.words = make('topology', t.words.byteLength); device.queue.writeBuffer(this.words, 0, t.words);
    this.floats = make('rest data', t.floats.byteLength); device.queue.writeBuffer(this.floats, 0, t.floats);
    this.vertices = make('vertices', 64 * t.V);
    this.plan = make('plan', 16 * (Math.ceil(Math.max(1, t.V) / 4) + GAUSS_VEC4 * Math.max(1, t.G) + TERM_VEC4 * Math.max(1, t.C)));
    this.status = make('status', STATUS_BYTES); this.velocities = make('velocities', 16 * t.V); this.water = make('water quanta', 4 * t.V);
    const VS = Math.ceil(Math.max(1, t.V) / 64) * 64, ES = Math.ceil(Math.max(1, t.E) / 64) * 64;
    this.work = make('work', 4 * (WORK_VECTORS * VS + EDGE_ARRAYS * ES + 2 * Math.max(1, t.G)));
    this.readback = device.createBuffer({ label: 'Cloth GPU pore readback', size: STATUS_BYTES + 20 * t.V + 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }); this.buffers.push(this.readback);
    if (options.timing) {
      const resolve = device.createBuffer({ label: 'Cloth GPU pore timestamps', size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }); this.buffers.push(resolve);
      this.timestamps = { set: device.createQuerySet({ label: 'Cloth GPU pore timestamps', type: 'timestamp', count: 2 }), resolve };
    }
  }
  async initialize(): Promise<void> {
    const d = this.device, source = code(this.topology, workgroupWidth(this.device), this.subgroups).replace(/PICARD_TOLERANCE/g, 'PICARD_TOLERANCE_F32');
    const module = d.createShaderModule({ label: 'Cloth GPU pore transport', code: `${this.subgroups ? 'enable subgroups;\n' : ''}const PICARD_TOLERANCE_F32=${PICARD_TOLERANCE};\n${source}` });
    const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error');
    if (errors.length) throw new Error(errors.map(m => `Cloth GPU pore ${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));
    const b = (binding: number, type: GPUBufferBindingType): GPUBindGroupLayoutEntry => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
    const layout = d.createBindGroupLayout({ entries: [b(0, 'uniform'), b(1, 'read-only-storage'), b(2, 'read-only-storage'), b(3, 'read-only-storage'), b(4, 'storage'), b(5, 'storage'), b(6, 'storage'), b(7, 'storage'), b(8, 'storage')] });
    const pipelineLayout = d.createPipelineLayout({ bindGroupLayouts: [layout] });
    const entries = ['planVertices', 'planGauss', 'planTerms', 'transport'] as const;
    const made = await Promise.all(entries.map(entryPoint => d.createComputePipelineAsync({ label: `Cloth GPU pore ${entryPoint}`, layout: pipelineLayout, compute: { module, entryPoint } })));
    this.pipelines = Object.fromEntries(entries.map((e, i) => [e, made[i]!])) as typeof this.pipelines;
    this.group = d.createBindGroup({ layout, entries: [this.params, this.words, this.floats, this.vertices, this.plan, this.status, this.velocities, this.water, this.work].map((buffer, binding) => ({ binding, resource: { buffer } })) });
  }
  /** Upload the accepted state, run every manifold substep, read water and velocity back once. */
  async advance(step: ClothPoreGPUStep): Promise<ClothPoreGPUResult | undefined> {
    if (this.busy) throw new Error('Cloth GPU pore transport steps must be serialized');
    const staged = this.stage(step); if (!staged) return undefined;
    const e = this.device.createCommandEncoder({ label: 'Cloth GPU pore transport' }), ts = this.timestamps;
    this.encode(e, ts ? { querySet: ts.set, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } : undefined, staged);
    const bytes = this.encodeResult(e, this.readback, 0);
    if (ts) { e.resolveQuerySet(ts.set, 0, 2, ts.resolve, 0); e.copyBufferToBuffer(ts.resolve, 0, this.readback, bytes, 16); }
    this.device.queue.submit([e.finish()]);
    this.busy = true;
    try {
      await this.readback.mapAsync(GPUMapMode.READ);
      const data = this.readback.getMappedRange().slice(0); this.readback.unmap();
      const result = this.parseResult(data, 0, staged);
      return ts ? { ...result, gpu_ms: clothTimestampInterval_ms(data, bytes) } : result;
    } finally { if (this.readback.mapState === 'mapped') this.readback.unmap(); this.busy = false; }
  }
  /** Validate and upload one step (vertex frame, materials, water quanta and
   * velocities). Undefined: no transport this step (zero dt, transport off,
   * or a dry cloth, the oracle's early exit). With `wetOnDevice`, water may
   * arrive on the device before the transport (GPU capture), so dryness is
   * decided there instead. */
  stage(step: ClothPoreGPUStep, wetOnDevice = false): ClothPoreGPUStaged | undefined {
    if (!this.pipelines || !this.group) throw new Error('Cloth GPU pore transport is not initialized');
    const t = this.topology, V = t.V, dt = step.dt_s, substeps = step.physics?.manifoldSubsteps ?? 1, predictorDt = step.manifoldPredictorDt_s ?? dt;
    if (!(predictorDt >= 0) || !Number.isFinite(predictorDt)) throw new Error('Invalid manifold force predictor interval');
    if (!Number.isInteger(substeps) || substeps < 1) throw new Error('Manifold substeps must be a positive integer');
    if (!Number.isFinite(dt) || dt < 0 || step.gravity_m_s2.some(v => !Number.isFinite(v))) throw new Error('Invalid pore transport step');
    for (const field of [step.pressureAcceleration_m_s2, step.solidAcceleration_m_s2]) if (field && field.length !== 3 * V) throw new Error('Invalid pore-frame acceleration');
    const reference = step.physics?.manifoldDragFloorReferenceVolume_m3;
    if (reference !== undefined && (!(reference > 0) || !Number.isFinite(reference))) throw new Error('Invalid source manifold regularizer volume');
    const transport = !!dt && step.physics?.manifoldTransport !== false && (wetOnDevice || !C0(step.waterVolume_m3));
    if (!transport && !wetOnDevice) return undefined;
    const rho = step.materials[0]?.waterDensity_kg_m3;
    if (!(rho! > 0) || step.materials.some(m => m.waterDensity_kg_m3 !== rho)) throw new Error('Pore transport requires a common water density');
    if (step.materials.length !== t.M) throw new Error('Cloth GPU pore material count changed');
    // Only position differences enter: centre them in f64 before f32.
    const center = [0, 1, 2].map(a => { let lo = Infinity, hi = -Infinity; for (let v = 0; v < V; v++) { const x = step.positions_m[3 * v + a]!; lo = Math.min(lo, x); hi = Math.max(hi, x); } return .5 * (lo + hi); });
    const q = this.quantum_m3, packed = new Float32Array(16 * V), quanta = new Int32Array(V), residual = new Float64Array(V), velocity = new Float32Array(4 * V);
    for (let v = 0; v < V; v++) {
      const w = step.waterVolume_m3[v]!;if (!Number.isFinite(w) || w < 0) throw new Error('Invalid manifold water inventory');
      const n = Math.floor(w / q); if (n > 2 ** 31 - 1) throw new Error('Pore water exceeds the int32 quantum range'); quanta[v] = n; residual[v] = w - n * q;
      for (let a = 0; a < 3; a++) {
        packed[16 * v + a] = step.positions_m[3 * v + a]! - center[a]!; packed[16 * v + 4 + a] = step.pressureAcceleration_m_s2?.[3 * v + a] ?? 0;
        packed[16 * v + 8 + a] = step.solidAcceleration_m_s2?.[3 * v + a] ?? 0; velocity[4 * v + a] = step.velocities_m_s[3 * v + a]!;
      }
      packed[16 * v + 3] = step.controlVolume_m3[v]!;
      packed.set([step.dryMass_kg[v]!, step.solidVolume_m3[v]!, step.solidFraction[v]!, ((step.fixed?.[v] ?? 0) & 1) ? 1 : 0], 16 * v + 12);
    }
    for (const x of packed) if (!Number.isFinite(x)) throw new Error('Invalid pore-frame acceleration');
    for (const x of velocity) if (!Number.isFinite(x)) throw new Error('Invalid pore-frame velocity');
    const d = this.device, header = new ArrayBuffer(80), u = new Uint32Array(header), f = new Float32Array(header);
    const flags = (step.physics?.pressureOnManifold === false ? 0 : 1) | (step.physics?.varyingSolidFraction === false ? 2 : 0) | (step.physics?.solveSolid === false ? 0 : 4) | (reference === undefined ? 0 : 8);
    u.set([V, t.T, t.R, t.C, t.E, t.G, substeps, flags], 0);
    f.set([dt / substeps, predictorDt, rho!, step.physics?.dragExponent ?? 1.6, step.gravity_m_s2[0], step.gravity_m_s2[1], step.gravity_m_s2[2], q, reference ?? 0, 0, 0, 0], 8);
    d.queue.writeBuffer(this.params, 0, header);
    d.queue.writeBuffer(this.floats, 4 * t.offsets.materials, clothPoreMaterialTable(step.materials, step.physics));
    d.queue.writeBuffer(this.vertices, 0, packed); d.queue.writeBuffer(this.water, 0, quanta); d.queue.writeBuffer(this.velocities, 0, velocity);
    return { residual_m3: residual, transport };
  }
  /** Encode the staged step: every substep in one pass, no readback. */
  encode(e: GPUCommandEncoder, timestampWrites?: GPUComputePassTimestampWrites, staged?: ClothPoreGPUStaged): void {
    if (!this.pipelines || !this.group) throw new Error('Cloth GPU pore transport is not initialized');
    e.clearBuffer(this.status);
    if (staged && !staged.transport) return;
    const t = this.topology, V = t.V, p = this.pipelines;
    const pass = e.beginComputePass({ label: 'Cloth GPU pore transport', ...(timestampWrites ? { timestampWrites } : {}) });
    pass.setBindGroup(0, this.group);
    pass.setPipeline(p.planVertices); pass.dispatchWorkgroups(Math.max(1, Math.ceil(V / 256)));
    pass.setPipeline(p.planGauss); pass.dispatchWorkgroups(Math.max(1, Math.ceil(t.G / 64)));
    pass.setPipeline(p.planTerms); pass.dispatchWorkgroups(Math.max(1, Math.ceil(t.C / 64)));
    pass.setPipeline(p.transport); pass.dispatchWorkgroups(1); pass.end();
  }
  /** Copy status, water quanta and velocities to `target` at `offset`; returns the end offset. */
  encodeResult(e: GPUCommandEncoder, target: GPUBuffer, offset: number): number {
    const V = this.topology.V;
    e.copyBufferToBuffer(this.status, 0, target, offset, STATUS_BYTES);
    e.copyBufferToBuffer(this.water, 0, target, offset + STATUS_BYTES, 4 * V);
    e.copyBufferToBuffer(this.velocities, 0, target, offset + STATUS_BYTES + 4 * V, 16 * V);
    return offset + this.resultBytes;
  }
  /** Bytes encodeResult writes. */
  get resultBytes(): number { return STATUS_BYTES + 20 * this.topology.V; }
  parseResult(data: ArrayBuffer, offset: number, staged: ClothPoreGPUStaged): ClothPoreGPUResult {
    const V = this.topology.V, q = this.quantum_m3, status = new Uint32Array(data, offset, 8), floats = new Float32Array(data, offset, 8);
    for (const [bit, message] of STATUS_MESSAGES) if (status[0]! & bit) throw new Error(`Cloth GPU pore transport: ${message}`);
    const water = new Int32Array(data, offset + STATUS_BYTES, V), velocities = new Float32Array(data, offset + STATUS_BYTES + 4 * V, 4 * V);
    for (let v = 0; v < V; v++) if (water[v]! < 0) throw new Error('Cloth GPU pore water went negative');
    return { waterVolume_m3: Float64Array.from(water, (n, v) => n * q + staged.residual_m3[v]!), velocities_m_s: Float64Array.from({ length: 3 * V }, (_, i) => velocities[4 * Math.floor(i / 3) + i % 3]!),
      iterations: status[1]!, picardIterations: status[2]!, movedVolume_m3: floats[3]!, limitedVolume_m3: floats[4]!, supportReaction_Ns: [floats[5]!, floats[6]!, floats[7]!] };
  }
  /** Device buffers a coupled GPU stage shares: int32 water quanta, vec4
   * velocities, and the four-vec4 vertex records (see the kernel header). */
  get waterBuffer(): GPUBuffer { return this.water; }
  get velocityBuffer(): GPUBuffer { return this.velocities; }
  get vertexBuffer(): GPUBuffer { return this.vertices; }
  /** Parity output: the last step's accepted volume a->b of each edge slot
   * (topology.slots order), summed over substeps. A test readback only. */
  async readEdgeFluxes(): Promise<Float64Array> {
    const t = this.topology, VS = Math.ceil(Math.max(1, t.V) / 64) * 64, ES = Math.ceil(Math.max(1, t.E) / 64) * 64, bytes = 4 * Math.max(1, t.E);
    const out = this.device.createBuffer({ size: Math.ceil(bytes / 16) * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const e = this.device.createCommandEncoder(); e.copyBufferToBuffer(this.work, 4 * (WORK_VECTORS * VS + CUMULATIVE_ARRAY * ES), out, 0, Math.ceil(bytes / 4) * 4); this.device.queue.submit([e.finish()]);
    await out.mapAsync(GPUMapMode.READ); const quanta = new Int32Array(out.getMappedRange().slice(0, 4 * t.E)); out.destroy();
    return Float64Array.from(quanta, n => n * this.quantum_m3);
  }
  destroy(): void { for (const b of this.buffers) b.destroy(); this.buffers.length = 0; this.timestamps?.set.destroy(); }
}
/** The single transport workgroup: the widest power of two the device runs. */
function workgroupWidth(device: GPUDevice): number { const n = Math.min(1024, device.limits.maxComputeInvocationsPerWorkgroup, device.limits.maxComputeWorkgroupSizeX, device.limits.maxComputeWorkgroupStorageSize / 16); return 2 ** Math.floor(Math.log2(n)); }
/** Dry cloth: nothing to transport (the oracle's early exit). */
function C0(water: ArrayLike<number>): boolean { for (let i = 0; i < water.length; i++) if (water[i]! > 0) return false; return true; }
/** The Picard fixed point is resolved to the inner solve's f32 backward-error
 * precision on the saturation scale: 8x the Krylov bound (16u). */
const PICARD_TOLERANCE = 8 * 16 * 2 ** -24;
