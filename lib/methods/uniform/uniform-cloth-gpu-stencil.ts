import type { ClothMaterial, ClothTopologyData } from '../../cloth/types';
import { porousPermeability } from '../../cloth/porous-material';
import { UniformGPURadixSort, UniformGPUScan, type UniformGPUOrderingKernels } from './uniform-gpu-radix-sort';

/**
 * The cloth's h-MAC exchange stencil built on the GPU: the same anchors,
 * links, per-coordinate gather and vertex material moments that
 * prepareUniformClothStencil (the CPU oracle) builds, from GPU-resident
 * cloth state, with no host round trip.
 *
 * Every vertex owns fixed link slots, (3v + axis) * 12 + face, the face in
 * the oracle's insertion order; a slot is live when its face weight exceeds
 * 1e-12. Live slots are compacted in slot order (the oracle's creation
 * order) and stably sorted by (face cell, axis): that is the oracle's link
 * layout, anchor by anchor in z, y, x order, axis-major, creation order
 * within an axis. Counts stay on the GPU (`counts`: live links, anchors,
 * status bits); consumers bound them by those words, not by array length.
 *
 * Buffers are packed in 256-byte aligned sections so no kernel binds more
 * than nine storage buffers (devices may allow only ten). `graph` is the
 * tensor drag's layout: ranges | gather | link faces | link weights | colors.
 */

/** Twelve faces an axis: three along it times two by two across. */
export const CLOTH_STENCIL_FACES = 12;
export const CLOTH_STENCIL_SLOTS = 3 * CLOTH_STENCIL_FACES;
/** counts[2] bits: a failed stencil is fatal, never a silent partial one. */
export const CLOTH_STENCIL_STATUS = { outsideLattice: 1, masslessFreeVertex: 2, noOpenSupport: 4, degenerateElement: 8, anchorCapacity: 16, colorBudget: 32, colorRounds: 64 } as const;
/** A vertex's faces lie in its base cell's [-1, 1] x [0, 1] x [0, 1] neighborhoods: at most 8 + 3 * 4 face cells. */
export const CLOTH_STENCIL_ANCHORS_PER_VERTEX = 20;
/** Conflict-free drag colors (Lie-split block passes): a fixed budget, its overflow a status error. */
export const CLOTH_DRAG_COLORS = 96;
const MASK_WORDS = CLOTH_DRAG_COLORS / 32;
/** Jones-Plassmann rounds encoded a build; vertices left uncolored are a status error. */
export const CLOTH_COLOR_ROUNDS = 160;
/** Vertex coloring priorities: the base cell's class (mod 3 an axis) major,
 * the vertex index minor, so priorities are unique. Two vertices whose base
 * cells differ by 3 or more along any axis share no anchor, so a decreasing
 * priority path stays within one cell between class changes: Jones-Plassmann
 * then needs about as many rounds as colors (a hashed priority needs ~3x). */
export function clothColorPriorities(fields: { readonly origin_m: readonly number[]; readonly cellSize_m: readonly number[] }, positions: ArrayLike<number>): Uint32Array {
  const V = positions.length / 3; if (V >= 2 ** 27) throw new Error('Cloth coloring priorities need fewer than 2^27 vertices');
  return Uint32Array.from({ length: V }, (_, v) => {
    let cls = 0; for (let a = 2; a >= 0; a--) { const b = Math.floor((positions[3*v+a]! - fields.origin_m[a]!) / fields.cellSize_m[a]! - .5); cls = 3 * cls + ((b % 3) + 3) % 3; }
    return (cls * 2 ** 27 + v) >>> 0;
  });
}
/** Section alignment in words: 256 bytes, the largest storage offset alignment WebGPU allows. */
const ALIGN = 64;
const pad = (words: number) => Math.ceil(words / ALIGN) * ALIGN;
/** Word offsets of the packed sections for V vertices. */
export function clothStencilSections(vertices: number) {
  const L = CLOTH_STENCIL_SLOTS * vertices, stride = pad(L), gather = pad(6 * vertices), A = CLOTH_STENCIL_ANCHORS_PER_VERTEX * vertices, colorsBase = 5 * stride + pad(3 * vertices);
  return {
    links: L,
    work: { weights: 0, keys: stride, positions: 2 * stride, anchorStarts: 3 * stride, slotLinks: 4 * stride, owners: 5 * stride,
      colors: colorsBase, anchorMax: colorsBase + pad(vertices), anchorDone: colorsBase + pad(vertices) + pad(A), masks: colorsBase + pad(vertices) + 2 * pad(A),
      priorities: colorsBase + pad(vertices) + 2 * pad(A) + pad(MASK_WORDS * A), words: colorsBase + 2 * pad(vertices) + 2 * pad(A) + pad(MASK_WORDS * A) },
    graph: { ranges: 0, gather, faces: gather + stride, weights: gather + 2 * stride, colors: gather + 3 * stride, colorRanges: gather + 3 * stride + pad(vertices),
      words: gather + 3 * stride + pad(vertices) + pad(2 * CLOTH_DRAG_COLORS) },
  };
}

/** The static per-topology tables: incident elements in the oracle's accumulation order. */
export interface ClothStencilTopology {
  readonly vertices: number;
  readonly elementOffsets: Uint32Array;
  /** Triangle index, or (1 << 31) | rod segment index. */
  readonly elements: Uint32Array;
  readonly triangles: Uint32Array;
  readonly segments: Uint32Array;
  /** Two vec4 a material: (thickness, rod radius, rod radius B, fiber diameter), (viscosity, capillary factor, 0, 0). */
  readonly materials: Float32Array;
}

export function clothStencilTopology(data: ClothTopologyData, materials: readonly ClothMaterial[], vertices: number): ClothStencilTopology {
  const lists: number[][] = Array.from({ length: vertices }, () => []);
  const triangles = new Uint32Array(4 * (data.triangles.length / 3));
  for (let t = 0; t < data.triangles.length / 3; t++) {
    const m = data.triangleMaterials?.[t] ?? 0;
    if (!materials[m]) throw new Error('Missing cloth triangle material');
    triangles.set([data.triangles[3*t]!, data.triangles[3*t+1]!, data.triangles[3*t+2]!, m], 4*t);
    for (let k = 0; k < 3; k++) lists[data.triangles[3*t+k]!]!.push(t);
  }
  const segments: number[] = [];
  for (let rod = 0; rod + 1 < data.rodOffsets.length; rod++) {
    const m = data.rodMaterials?.[rod] ?? 0;
    if (!materials[m]) throw new Error('Missing cloth rod material');
    for (let j = data.rodOffsets[rod]!; j + 1 < data.rodOffsets[rod+1]!; j++) {
      const s = segments.length / 4, a = data.rodIndices[j]!, b = data.rodIndices[j+1]!;
      segments.push(a, b, m, 0); lists[a]!.push((1 << 31 | s) >>> 0); lists[b]!.push((1 << 31 | s) >>> 0);
    }
  }
  const elementOffsets = new Uint32Array(vertices + 1);
  for (let v = 0; v < vertices; v++) elementOffsets[v+1] = elementOffsets[v]! + lists[v]!.length;
  const elements = new Uint32Array(Math.max(1, elementOffsets[vertices]!));
  lists.forEach((list, v) => elements.set(list, elementOffsets[v]!));
  const table = new Float32Array(8 * materials.length);
  materials.forEach((m, i) => {
    const radius = porousPermeability(m.solidFraction, m.fiberDiameter_m).capillaryRadius_m;
    const cosine = Math.abs(m.contactAngle_rad - Math.PI/2) < 1e-14 ? 0 : Math.cos(m.contactAngle_rad);
    table.set([m.thickness_m, m.rodRadius_m, m.rodRadiusB_m ?? m.rodRadius_m, m.fiberDiameter_m, m.waterViscosity_Pa_s, m.surfaceTension_N_m * cosine / radius, 0, 0], 8*i);
  });
  return { vertices, elementOffsets, elements, triangles: triangles.length ? triangles : new Uint32Array(4), segments: segments.length ? Uint32Array.from(segments) : new Uint32Array(4), materials: table };
}

const code = /* wgsl */`
struct Params {origin:vec4f,inverseCell:vec4f,dims:vec4u,misc:vec4f};
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> positions:array<vec4f>;
@group(0) @binding(2) var<storage,read> predictor:array<vec4f>;
@group(0) @binding(3) var<storage,read> scalars:array<vec4f>;
@group(0) @binding(4) var<storage,read> elementOffsets:array<u32>;
@group(0) @binding(5) var<storage,read> elements:array<u32>;
@group(0) @binding(6) var<storage,read> triangles:array<vec4u>;
@group(0) @binding(7) var<storage,read> segments:array<vec4u>;
@group(0) @binding(8) var<storage,read> materials:array<vec4f>;
@group(0) @binding(9) var<storage,read_write> vertexMaterial:array<vec4f>;
@group(0) @binding(10) var<storage,read_write> work:array<u32>;
@group(0) @binding(11) var<storage,read_write> sortKeys:array<u32>;
@group(0) @binding(12) var<storage,read_write> sortValues:array<u32>;
@group(0) @binding(13) var<storage,read_write> counts:array<atomic<u32>>;
@group(0) @binding(14) var<storage,read_write> anchors:array<vec4i>;
@group(0) @binding(15) var<storage,read_write> coefficients:array<vec4f>;
@group(0) @binding(16) var<storage,read_write> occupied:array<f32>;
@group(0) @binding(17) var<storage,read_write> graph:array<u32>;
@group(0) @binding(18) var<storage,read> cornerApertures:array<f32>;
@group(0) @binding(19) var<storage,read> frame:array<vec4i>;
@group(0) @binding(20) var<storage,read_write> colorDispatch:array<u32>;
@group(0) @binding(21) var<storage,read_write> frameBounds:array<atomic<i32>>;
const SLOTS=${CLOTH_STENCIL_SLOTS}u;const FACES=${CLOTH_STENCIL_FACES}u;const ALIGN=${ALIGN}u;
fn vertexCount()->u32{return params.dims.w;}
fn padded(n:u32)->u32{return (n+ALIGN-1u)/ALIGN*ALIGN;}
fn stride()->u32{return padded(SLOTS*vertexCount());}
// Work sections: slot weights, slot keys, slot positions, anchor starts, slot links, owner offsets.
fn slotWeight(s:u32)->f32{return bitcast<f32>(work[s]);}
fn live(s:u32)->bool{return slotWeight(s)>0.0;}
fn gatherBase()->u32{return padded(6u*vertexCount());}
fn fail(bit:u32){atomicOr(&counts[2],bit);}
fn fixedVertex(v:u32)->bool{return params.misc.z==0.0||scalars[v].w!=0.0;}
fn vertexMass(v:u32)->f32{return scalars[v].x+params.misc.x*scalars[v].y;}

// The key frame: every cell a vertex's corners touch lies in [min base,
// max base + 1], its faces one cell lower. The frame spans that with a
// one-cell margin below, so keys are 32-bit in any window placement.
// frameBounds words: origin xyz, dims xyz, min base xyz, max base + 1 xyz (a vec4 each).
fn baseCell(v:u32)->vec3i{return vec3i(floor((positions[v].xyz-params.origin.xyz)*params.inverseCell.xyz-vec3f(0.5)));}
@compute @workgroup_size(1) fn frameReset(){
 for(var a=0u;a<3u;a++){atomicStore(&frameBounds[8u+a],2147483647);atomicStore(&frameBounds[12u+a],-2147483647-1);}
}
@compute @workgroup_size(64) fn frameExtent(@builtin(global_invocation_id) gid:vec3u){
 let v=gid.x;if(v>=vertexCount()){return;}let base=baseCell(v);
 for(var a=0u;a<3u;a++){atomicMin(&frameBounds[8u+a],base[a]);atomicMax(&frameBounds[12u+a],base[a]+1);}
}
@compute @workgroup_size(1) fn frameKeys(){
 var lo=vec3i(0);var hi=vec3i(0);for(var a=0u;a<3u;a++){lo[a]=atomicLoad(&frameBounds[8u+a]);hi[a]=atomicLoad(&frameBounds[12u+a]);}
 let dims=hi-lo+vec3i(2);
 // Four faces a key cell must fit 32 bits; a wider cloth is an explicit failure.
 let span=4.0*f32(dims.x)*f32(dims.y)*f32(dims.z);
 if(!(span<4294967040.0)){atomicOr(&counts[2],${CLOTH_STENCIL_STATUS.outsideLattice}u);}
 for(var a=0u;a<3u;a++){atomicStore(&frameBounds[a],lo[a]-1);atomicStore(&frameBounds[4u+a],select(0,dims[a],span<4294967040.0));}
}

// Vertex moments: geometric volume, capillary coefficient, fiber diameter and
// viscosity, pore tensor, volume-weighted over incident triangles then rods.
@compute @workgroup_size(64) fn vertexMoments(@builtin(global_invocation_id) gid:vec3u){
 let v=gid.x;if(v>=vertexCount()){return;}
 var volume=0.0;var coefficient=0.0;var micro=vec2f(0.0);var tensor=mat3x3f(vec3f(0.0),vec3f(0.0),vec3f(0.0));
 for(var e=elementOffsets[v];e<elementOffsets[v+1u];e++){
  let element=elements[e];var axis=vec3f(0.0);var share=0.0;var m=0u;var shape=1.0;
  if((element>>31u)==0u){
   let t=triangles[element];m=t.w;let a=positions[t.x].xyz;let n=cross(positions[t.y].xyz-a,positions[t.z].xyz-a);let twiceArea=length(n);
   if(!(twiceArea>0.0)){fail(${CLOTH_STENCIL_STATUS.degenerateElement}u);return;}
   axis=n/twiceArea;share=twiceArea*materials[2u*m].x/6.0;
  }else{
   let s=segments[element&0x7fffffffu];m=s.z;let tangent=positions[s.y].xyz-positions[s.x].xyz;let len=length(tangent);
   if(!(len>0.0)){fail(${CLOTH_STENCIL_STATUS.degenerateElement}u);return;}
   axis=tangent/len;shape=0.0;share=0.5*3.141592653589793*materials[2u*m].y*materials[2u*m].z*len;
  }
  // porousTensor(2, 1, axis, shape): tangent I + (axial - tangent) axis axis^T.
  let tangentValue=shape*2.0+(1.0-shape);let axial=shape+(1.0-shape)*2.0;
  let pore=mat3x3f(vec3f(tangentValue,0.0,0.0),vec3f(0.0,tangentValue,0.0),vec3f(0.0,0.0,tangentValue))+(axial-tangentValue)*mat3x3f(axis*axis.x,axis*axis.y,axis*axis.z);
  volume+=share;micro+=share*vec2f(materials[2u*m].w,materials[2u*m+1u].x);coefficient+=share*materials[2u*m+1u].y;tensor+=share*pore;
 }
 if(volume>0.0){coefficient/=volume;micro/=volume;tensor=tensor*(1.0/volume);}
 vertexMaterial[4u*v]=vec4f(volume,coefficient,micro);
 vertexMaterial[4u*v+1u]=vec4f(tensor[0],0.0);vertexMaterial[4u*v+2u]=vec4f(tensor[1],0.0);vertexMaterial[4u*v+3u]=vec4f(tensor[2],0.0);
}

// The vertex's link slots: clothCellWeights' open corners, each axis's
// faces in first-touch order with the averaged half-cell partition.
@compute @workgroup_size(64) fn vertexSlots(@builtin(global_invocation_id) gid:vec3u){
 let v=gid.x;if(v>=vertexCount()){return;}let L=stride();
 for(var s=SLOTS*v;s<SLOTS*(v+1u);s++){work[s]=0u;work[L+s]=0u;work[2u*L+s]=0u;}
 if(!fixedVertex(v)&&!(vertexMass(v)>0.0)){fail(${CLOTH_STENCIL_STATUS.masslessFreeVertex}u);}
 let q=(positions[v].xyz-params.origin.xyz)*params.inverseCell.xyz-vec3f(0.5);let base=vec3i(floor(q));let f=q-floor(q);
 let cls=vec3u(((base%3)+3)%3);work[priorityBase()+v]=(cls.x+3u*cls.y+9u*cls.z)*(1u<<27u)+v;
 if(vertexMaterial[4u*v].x==0.0){return;}
 var weights:array<f32,8>;var normalization=0.0;
 for(var c=0u;c<8u;c++){
  let o=vec3u(c&1u,(c>>1u)&1u,(c>>2u)&1u);let w=select(1.0-f.x,f.x,o.x==1u)*select(1.0-f.y,f.y,o.y==1u)*select(1.0-f.z,f.z,o.z==1u);
  let hard=cornerApertures[8u*v+c];weights[c]=0.0;if(w<=0.0){continue;}normalization+=w*hard;if(hard>0.0){weights[c]=w*hard;}
 }
 if(normalization<=1e-12){fail(${CLOTH_STENCIL_STATUS.noOpenSupport}u);return;}
 let origin=frame[0].xyz;let dims=vec3u(frame[1].xyz);
 for(var axis=0u;axis<3u;axis++){
  var keys:array<vec3i,12>;var faceWeight:array<f32,12>;var faces=0u;
  for(var c=0u;c<8u;c++){
   if(weights[c]<=0.0){continue;}let weight=0.5*weights[c]/normalization;
   let cell=base+vec3i(i32(c&1u),i32((c>>1u)&1u),i32((c>>2u)&1u));
   for(var offset=-1;offset<=0;offset++){
    var face=cell;face[axis]+=offset;var k=0u;loop{if(k>=faces||all(keys[k]==face)){break;}k++;}
    if(k<faces){faceWeight[k]+=weight;}else{keys[faces]=face;faceWeight[faces]=weight;faces++;}
   }
  }
  for(var k=0u;k<faces;k++){
   if(faceWeight[k]<=1e-12){continue;}
   let local=keys[k]-origin;
   if(any(local<vec3i(0))||any(vec3u(local)>=dims)){fail(${CLOTH_STENCIL_STATUS.outsideLattice}u);continue;}
   let slot=SLOTS*v+FACES*axis+k;work[slot]=bitcast<u32>(faceWeight[k]);
   work[L+slot]=4u*(u32(local.x)+dims.x*(u32(local.y)+dims.y*u32(local.z)))+axis;work[2u*L+slot]=1u;
  }
 }
}

// Live slots, compacted in slot (creation) order; the last slot publishes the count.
@compute @workgroup_size(64) fn compactSlots(@builtin(global_invocation_id) gid:vec3u){
 let s=gid.x;let total=SLOTS*vertexCount();if(s>=total){return;}let L=stride();let at=work[2u*L+s];
 if(live(s)){sortKeys[at]=work[L+s];sortValues[at]=s;}
 if(s==total-1u){atomicStore(&counts[0],at+select(0u,1u,live(s)));}
}
fn anchorStart(j:u32)->bool{return j==0u||(sortKeys[j]>>2u)!=(sortKeys[j-1u]>>2u);}
fn anchorOf(j:u32)->u32{return work[3u*stride()+j]+select(0u,1u,anchorStart(j))-1u;}
@compute @workgroup_size(64) fn markAnchors(@builtin(global_invocation_id) gid:vec3u){
 let j=gid.x;if(j>=SLOTS*vertexCount()){return;}work[3u*stride()+j]=select(0u,1u,j<atomicLoad(&counts[0])&&anchorStart(j));
}
// Link records in sorted (final) order.
@compute @workgroup_size(64) fn writeLinks(@builtin(global_invocation_id) gid:vec3u){
 let j=gid.x;let n=atomicLoad(&counts[0]);if(j>=n){return;}
 let anchor=anchorOf(j);let slot=sortValues[j];let v=slot/SLOTS;let axis=(slot/FACES)%3u;let w=slotWeight(slot);
 let mass=select(w*vertexMass(v),0.0,fixedVertex(v));
 coefficients[j]=vec4f(mass,0.0,0.0,predictor[v][axis]);occupied[j]=w*(scalars[v].z+scalars[v].y)*params.misc.y;
 let L=stride();let G=gatherBase();graph[G+L+j]=3u*anchor+axis;graph[G+2u*L+j]=bitcast<u32>(w);work[4u*L+slot]=j;
}
// Anchors at each run's first link: face cell, first link, per-axis counts.
@compute @workgroup_size(64) fn writeAnchors(@builtin(global_invocation_id) gid:vec3u){
 let j=gid.x;let n=atomicLoad(&counts[0]);if(j==0u&&n==0u){anchors[0]=vec4i(0);}if(j>=n){return;}
 let anchor=anchorOf(j);
 if(j==n-1u){atomicStore(&counts[1],anchor+1u);anchors[0]=vec4i(i32(anchor+1u),0,0,0);anchors[1]=vec4i(0);}
 if(!anchorStart(j)){return;}
 if(anchor>=${CLOTH_STENCIL_ANCHORS_PER_VERTEX}u*vertexCount()){fail(${CLOTH_STENCIL_STATUS.anchorCapacity}u);return;}let key=sortKeys[j]>>2u;var perAxis=vec3i(0);var k=j;
 loop{if(k>=n||(sortKeys[k]>>2u)!=key){break;}perAxis[sortKeys[k]&3u]+=1;k++;}
 let dims=vec3u(frame[1].xyz);let local=vec3u(key%dims.x,(key/dims.x)%dims.y,key/(dims.x*dims.y));
 anchors[2u+2u*anchor]=vec4i(vec3i(local)+frame[0].xyz,i32(j));anchors[3u+2u*anchor]=vec4i(perAxis,0);
}
// Jones-Plassmann coloring of the anchor-sharing conflict graph: a vertex
// whose priority is the largest among the uncolored members of every anchor
// it touches takes the least color none of those anchors holds. That is
// sequential greedy in decreasing priority (clothDragColors, the oracle):
// when it colors, exactly its higher-priority neighbors are colored.
const NONE=0xffffffffu;const COLORS=${CLOTH_DRAG_COLORS}u;const MASKS=${MASK_WORDS}u;
fn anchorCapacity()->u32{return ${CLOTH_STENCIL_ANCHORS_PER_VERTEX}u*vertexCount();}
fn colorBase()->u32{return 5u*stride()+padded(3u*vertexCount());}
fn anchorMaxBase()->u32{return colorBase()+padded(vertexCount());}
fn anchorDoneBase()->u32{return anchorMaxBase()+padded(anchorCapacity());}
fn maskBase()->u32{return anchorDoneBase()+padded(anchorCapacity());}
fn priorityBase()->u32{return maskBase()+padded(MASKS*anchorCapacity());}
fn graphColors()->u32{return gatherBase()+3u*stride();}
fn graphColorRanges()->u32{return graphColors()+padded(vertexCount());}
fn priority(v:u32)->u32{return work[priorityBase()+v];}
fn anchorTotal()->u32{return u32(anchors[0].x);}
@compute @workgroup_size(64) fn colorInit(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i<vertexCount()){work[colorBase()+i]=NONE;}
 if(i<anchorTotal()){work[anchorDoneBase()+i]=0u;for(var k=0u;k<MASKS;k++){work[maskBase()+MASKS*i+k]=0u;}}
}
@compute @workgroup_size(64) fn anchorRound(@builtin(global_invocation_id) gid:vec3u){
 let a=gid.x;if(a>=anchorTotal()||work[anchorDoneBase()+a]!=0u){return;}
 let first=u32(anchors[2u+2u*a].w);let n=anchors[3u+2u*a];var best=0u;var any=false;
 for(var j=first;j<first+u32(n.x+n.y+n.z);j++){let u=sortValues[j]/SLOTS;if(work[colorBase()+u]==NONE){any=true;best=max(best,priority(u));}}
 if(!any){work[anchorDoneBase()+a]=1u;}work[anchorMaxBase()+a]=best;
}
@compute @workgroup_size(64) fn vertexRound(@builtin(global_invocation_id) gid:vec3u){
 let v=gid.x;if(v>=vertexCount()||work[colorBase()+v]!=NONE){return;}let own=priority(v);let G=gatherBase();let F=G+stride();
 var used:array<u32,${MASK_WORDS}>;
 for(var o=3u*v;o<3u*v+3u;o++){for(var k=0u;k<graph[2u*o+1u];k++){
  let a=graph[F+graph[G+graph[2u*o]+k]]/3u;if(work[anchorMaxBase()+a]!=own){return;}
  for(var m=0u;m<MASKS;m++){used[m]|=work[maskBase()+MASKS*a+m];}
 }}
 var color=COLORS;for(var m=0u;m<MASKS;m++){if(used[m]!=0xffffffffu){color=32u*m+countTrailingZeros(~used[m]);break;}}
 work[colorBase()+v]=color;if(color>=COLORS){return;}
 // No other vertex colors at these anchors this round: plain writes suffice.
 for(var o=3u*v;o<3u*v+3u;o++){for(var k=0u;k<graph[2u*o+1u];k++){
  let a=graph[F+graph[G+graph[2u*o]+k]]/3u;work[maskBase()+MASKS*a+color/32u]|=1u<<(color%32u);
 }}
}
@compute @workgroup_size(64) fn colorCount(@builtin(global_invocation_id) gid:vec3u){
 let v=gid.x;if(v>=vertexCount()){return;}let color=work[colorBase()+v];
 if(color==NONE){fail(${CLOTH_STENCIL_STATUS.colorRounds}u);return;}if(color>=COLORS){fail(${CLOTH_STENCIL_STATUS.colorBudget}u);return;}
 atomicAdd(&counts[4u+color],1u);atomicMax(&counts[3],color+1u);
}
@compute @workgroup_size(1) fn colorScan(){
 var run=0u;let R=graphColorRanges();
 for(var c=0u;c<COLORS;c++){let n=atomicLoad(&counts[4u+c]);graph[R+2u*c]=run;graph[R+2u*c+1u]=n;run+=n;atomicStore(&counts[4u+COLORS+c],0u);
  colorDispatch[3u*c]=(n+63u)/64u;colorDispatch[3u*c+1u]=1u;colorDispatch[3u*c+2u]=1u;}
}
// Within a color no two vertices interact, so their order is immaterial.
@compute @workgroup_size(64) fn colorScatter(@builtin(global_invocation_id) gid:vec3u){
 let v=gid.x;if(v>=vertexCount()){return;}let color=work[colorBase()+v];if(color>=COLORS){return;}
 let slot=atomicAdd(&counts[4u+COLORS+color],1u);graph[graphColors()+graph[graphColorRanges()+2u*color]+slot]=v;
}
@compute @workgroup_size(64) fn countOwners(@builtin(global_invocation_id) gid:vec3u){
 let o=gid.x;if(o>=3u*vertexCount()){return;}var n=0u;for(var f=0u;f<FACES;f++){if(live(FACES*o+f)){n++;}}work[5u*stride()+o]=n;
}
// Each velocity coordinate's links in increasing final order.
@compute @workgroup_size(64) fn writeGather(@builtin(global_invocation_id) gid:vec3u){
 let o=gid.x;if(o>=3u*vertexCount()){return;}let L=stride();let offset=work[5u*L+o];
 var list:array<u32,12>;var n=0u;
 for(var f=0u;f<FACES;f++){if(live(FACES*o+f)){let j=work[4u*L+FACES*o+f];var k=n;loop{if(k==0u||list[k-1u]<=j){break;}list[k]=list[k-1u];k--;}list[k]=j;n++;}}
 let G=gatherBase();for(var k=0u;k<n;k++){graph[G+offset+k]=list[k];}graph[2u*o]=offset;graph[2u*o+1u]=n;
}`;

type Kernel = 'frameReset' | 'frameExtent' | 'frameKeys' | 'vertexMoments' | 'vertexSlots' | 'compactSlots' | 'markAnchors' | 'writeLinks' | 'writeAnchors' | 'countOwners' | 'writeGather'
  | 'colorInit' | 'anchorRound' | 'vertexRound' | 'colorCount' | 'colorScan' | 'colorScatter';
// The bindings each entry point reads or writes ('auto' layouts carry only those).
const USED = new Map<Kernel, readonly number[]>([
  ['vertexMoments', [0, 1, 4, 5, 6, 7, 8, 9, 13]],
  ['frameReset', [21]],
  ['frameExtent', [0, 1, 21]],
  ['frameKeys', [13, 21]],
  ['vertexSlots', [0, 1, 3, 9, 10, 13, 18, 19]],
  ['compactSlots', [0, 10, 11, 12, 13]],
  ['markAnchors', [0, 10, 11, 13]],
  ['writeLinks', [0, 2, 3, 10, 11, 12, 13, 15, 16, 17]],
  ['writeAnchors', [0, 10, 11, 13, 14, 19]],
  ['countOwners', [0, 10]],
  ['writeGather', [0, 10, 17]],
  ['colorInit', [0, 10, 14]],
  ['anchorRound', [0, 10, 12, 14]],
  ['vertexRound', [0, 10, 17]],
  ['colorCount', [0, 10, 13]],
  ['colorScan', [0, 13, 17, 20]],
  ['colorScatter', [0, 10, 13, 17]],
]);

export interface ClothStencilStep {
  readonly origin_m: readonly number[]; readonly cellSize_m: readonly number[];
  readonly density_kg_m3: number; readonly pressureOnSolid: boolean; readonly solveSolid: boolean;
}

/** One cloth's GPU stencil at a fixed vertex count. */
export class UniformClothGPUStencil {
  /** Inputs, written by the cloth state and the aperture kernel. */
  readonly positions: GPUBuffer; readonly predictor: GPUBuffer; readonly scalars: GPUBuffer;
  readonly cornerApertures: GPUBuffer;
  /** The step's key frame, computed on the GPU from the cloth's cell bounds: origin, dims (a vec4i each), then the bounds. */
  readonly frame: GPUBuffer;
  /** Outputs. `counts`: live links, anchors, status bits. `anchors` is a header (count, 0, 0, 0; 0) then two vec4 an anchor. */
  readonly vertexMaterial: GPUBuffer; readonly counts: GPUBuffer;
  readonly anchors: GPUBuffer; readonly coefficients: GPUBuffer; readonly occupied: GPUBuffer; readonly graph: GPUBuffer;
  /** Indirect dispatch arguments, one (ceil(n / 64), 1, 1) a drag color. */
  readonly colorDispatch: GPUBuffer;
  /** The step's scalar parameters (origin, inverse cell, dims with V in w, misc). */
  readonly params: GPUBuffer;
  /** Each sorted link's slot, (3v + axis) * 12 + face. */
  readonly linkSlots: GPUBuffer;
  /** Scratch sections (`sections.work`), including each vertex's drag color. */
  readonly work: GPUBuffer;
  readonly sections: ReturnType<typeof clothStencilSections>;
  readonly anchorCapacity: number;
  private readonly owned: GPUBuffer[] = [];
  private readonly groups = new Map<Kernel, GPUBindGroup>();
  private readonly scans: UniformGPUScan[];
  private readonly sort: UniformGPURadixSort;
  private readonly device: GPUDevice; private readonly pipelines: ReadonlyMap<Kernel, GPUComputePipeline>;
  constructor(stencilKernels: UniformClothStencilKernels, kernels: UniformGPUOrderingKernels, readonly topology: ClothStencilTopology) {
    const device = this.device = stencilKernels.device; this.pipelines = stencilKernels.pipelines;
    const V = topology.vertices, sections = this.sections = clothStencilSections(V), L = sections.links;
    this.anchorCapacity = CLOTH_STENCIL_ANCHORS_PER_VERTEX * V;
    if (V >= 2 ** 27) throw new Error('Cloth coloring priorities need fewer than 2^27 vertices');
    if (Math.ceil(L / 64) > device.limits.maxComputeWorkgroupsPerDimension) throw new Error('Cloth stencil exceeds one dispatch dimension');
    if (256 % device.limits.minStorageBufferOffsetAlignment) throw new Error('Cloth stencil sections need a storage offset alignment dividing 256 bytes');
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const make = (label: string, bytes: number, data?: ArrayBufferView) => {
      const b = device.createBuffer({ label: `Cloth GPU stencil ${label}`, size: Math.max(16, bytes), usage }); this.owned.push(b);
      if (data) device.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength); return b;
    };
    this.params = device.createBuffer({ label: 'Cloth GPU stencil parameters', size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); this.owned.push(this.params);
    this.positions = make('positions', 16*V); this.predictor = make('predictor', 16*V); this.scalars = make('scalars', 16*V);
    this.cornerApertures = make('corner apertures', 32*V); this.frame = make('key frame', 64);
    const elementOffsets = make('element offsets', topology.elementOffsets.byteLength, topology.elementOffsets), elements = make('elements', topology.elements.byteLength, topology.elements);
    const triangles = make('triangles', topology.triangles.byteLength, topology.triangles), segments = make('segments', topology.segments.byteLength, topology.segments);
    const materials = make('materials', topology.materials.byteLength, topology.materials);
    this.vertexMaterial = make('vertex material', 64*V);
    const work = this.work = make('work', 4*sections.work.words);
    this.counts = make('counts', 16 + 8 * CLOTH_DRAG_COLORS);
    this.anchors = make('anchors', 32 * (1 + this.anchorCapacity)); this.coefficients = make('coefficients', 16*L); this.occupied = make('occupied volume', 4*L);
    this.graph = make('graph', 4*sections.graph.words);
    this.colorDispatch = device.createBuffer({ label: 'Cloth GPU stencil color dispatch', size: 12 * CLOTH_DRAG_COLORS, usage: usage | GPUBufferUsage.INDIRECT }); this.owned.push(this.colorDispatch);
    this.scans = [new UniformGPUScan(kernels, work, L, 4*sections.work.positions), new UniformGPUScan(kernels, work, L, 4*sections.work.anchorStarts), new UniformGPUScan(kernels, work, 3*V, 4*sections.work.owners)];
    this.sort = new UniformGPURadixSort(kernels, L, 32, this.counts, 'Cloth stencil sort');
    this.linkSlots = this.sort.values;
    const bindings: GPUBuffer[] = [this.params, this.positions, this.predictor, this.scalars, elementOffsets, elements, triangles, segments, materials, this.vertexMaterial,
      work, this.sort.keys, this.sort.values, this.counts, this.anchors, this.coefficients, this.occupied, this.graph, this.cornerApertures, this.frame, this.colorDispatch, this.frame];
    for (const [kernel, used] of USED) {
      const layout = this.pipelines.get(kernel)!.getBindGroupLayout(0);
      this.groups.set(kernel, device.createBindGroup({ label: `Cloth GPU stencil ${kernel}`, layout, entries: used.map(binding => ({ binding, resource: { buffer: bindings[binding]! } })) }));
    }
  }
  static async create(device: GPUDevice, kernels: UniformGPUOrderingKernels, topology: ClothStencilTopology): Promise<UniformClothGPUStencil> {
    return new UniformClothGPUStencil(await UniformClothStencilKernels.create(device), kernels, topology);
  }
  /** A section of `graph` as a binding (word offsets from `sections.graph`). */
  graphBinding(offset: number, words: number): GPUBufferBinding { return { buffer: this.graph, offset: 4 * offset, size: 4 * Math.max(1, words) }; }
  /** The step's scalar parameters (a uniform write; no readback). */
  writeParameters(step: ClothStencilStep): void {
    const data = new ArrayBuffer(64), f = new Float32Array(data), u = new Uint32Array(data);
    f.set([step.origin_m[0]!, step.origin_m[1]!, step.origin_m[2]!, 0], 0);
    f.set([1/step.cellSize_m[0]!, 1/step.cellSize_m[1]!, 1/step.cellSize_m[2]!, 0], 4);
    u.set([0, 0, 0, this.topology.vertices], 8);
    f.set([step.density_kg_m3, step.pressureOnSolid ? 1 : -1, step.solveSolid ? 1 : 0, 0], 12);
    this.device.queue.writeBuffer(this.params, 0, data);
  }
  encode(encoder: GPUCommandEncoder): void {
    const V = this.topology.vertices, L = this.sections.links, [slots, anchors, owners] = this.scans as [UniformGPUScan, UniformGPUScan, UniformGPUScan];
    encoder.clearBuffer(this.counts);
    const run = (pass: GPUComputePassEncoder, kernel: Kernel, n: number) => { pass.setPipeline(this.pipelines.get(kernel)!); pass.setBindGroup(0, this.groups.get(kernel)!); pass.dispatchWorkgroups(Math.max(1, Math.ceil(n / 64))); };
    let pass = encoder.beginComputePass({ label: 'Cloth GPU stencil slots' });
    run(pass, 'frameReset', 1); run(pass, 'frameExtent', V); run(pass, 'frameKeys', 1); run(pass, 'vertexMoments', V); run(pass, 'vertexSlots', V); slots.encode(pass); run(pass, 'compactSlots', L); pass.end();
    this.sort.encode(encoder);
    pass = encoder.beginComputePass({ label: 'Cloth GPU stencil links' });
    run(pass, 'markAnchors', L); anchors.encode(pass); run(pass, 'writeLinks', L); run(pass, 'writeAnchors', L);
    run(pass, 'countOwners', 3*V); owners.encode(pass); run(pass, 'writeGather', 3*V);
    run(pass, 'colorInit', this.anchorCapacity);
    for (let round = 0; round < CLOTH_COLOR_ROUNDS; round++) { run(pass, 'anchorRound', this.anchorCapacity); run(pass, 'vertexRound', V); }
    run(pass, 'colorCount', V); run(pass, 'colorScan', 1); run(pass, 'colorScatter', V); pass.end();
  }
  destroy(): void { this.sort.destroy(); for (const scan of this.scans) scan.destroy(); for (const b of this.owned) b.destroy(); this.owned.length = 0; }
}

/** The GPU stencil pipelines: compiled once a device, shared by every cloth. */
export class UniformClothStencilKernels {
  private constructor(readonly device: GPUDevice, readonly pipelines: ReadonlyMap<Kernel, GPUComputePipeline>) {}
  static async create(device: GPUDevice): Promise<UniformClothStencilKernels> {
    const most = Math.max(...[...USED.values()].map(used => used.length - 1));
    if (device.limits.maxStorageBuffersPerShaderStage < most) throw new Error(`Cloth GPU stencil needs ${most} storage buffers a stage; the device allows ${device.limits.maxStorageBuffersPerShaderStage}`);
    const module = device.createShaderModule({ label: 'Cloth GPU stencil', code });
    const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error');
    if (errors.length) throw new Error(errors.map(m => `Cloth GPU stencil ${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));
    const pipelines = new Map<Kernel, GPUComputePipeline>();
    await Promise.all([...USED.keys()].map(async entryPoint => pipelines.set(entryPoint, await device.createComputePipelineAsync({ label: `Cloth GPU stencil ${entryPoint}`, layout: 'auto', compute: { module, entryPoint } }))));
    return new UniformClothStencilKernels(device, pipelines);
  }
}
