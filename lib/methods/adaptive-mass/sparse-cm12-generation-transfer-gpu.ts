import { gpuCompilationManagerFor } from "../../core/gpu-compilation-manager";
import { sparseBrickMaximumFine, sparseBrickSpan, type SparseAdaptiveMassAtlas } from "./sparse-brick-atlas";
import type { CM12CapturedGeometryRecipe } from "./sparse-cm12-captured-geometry";
import { SparseCM12GenerationBudgetDeferred } from "./sparse-cm12-generation-budget";
import { PreparedSparseCM12GenerationTransfer, type SparseCM12GenerationFields,
  type SparseCM12NewAirCoverage } from "./sparse-cm12-generation-transfer";

export interface SparseCM12PackedGenerationTarget extends SparseCM12GenerationFields {
  readonly topology: GPUBuffer;
  readonly atlas: SparseAdaptiveMassAtlas;
  readonly densityOtherOffset: number;
  readonly gammaOtherOffset: number;
  readonly velocityOtherOffset: number;
  readonly faceOtherOffset: number;
}

/** Compact immutable input. Native geometry, hash insertion, overlap search and
 * field accumulation execute on the GPU; there is no host native-cell graph. */
export function packSparseCM12GPUGenerationTransferInput(
  source: CM12CapturedGeometryRecipe, target: Pick<SparseCM12PackedGenerationTarget, "atlas" | "cellIds" | "rowIds">,
  newAirCoverage: readonly SparseCM12NewAirCoverage[] = [],
) {
  if (source.atlas.brickFineResolution !== 8 || target.atlas.brickFineResolution !== 8)
    throw new Error("CM12 GPU generation transfer requires B8 native topology");
  if (source.atlas.dimensions.some((n, axis) => n !== target.atlas.dimensions[axis]))
    throw new Error("CM12 transfer requires the same physical domain");
  const leaves = source.atlas.bricks.length;
  const pages = Math.max(0, ...source.sourcePageCoordinates.keys()) + 1;
  const leafBase = 0, pageBase = 12 * leaves, sourceRows = pageBase + 4 * pages;
  const targetCells = sourceRows + source.rows.length;
  const targetRows = targetCells + target.cellIds.length;
  const targetWidths = targetRows + target.rowIds.length;
  const airBase = targetWidths + target.atlas.bricks.length;
  const words = new Uint32Array(airBase + 6 * newAirCoverage.length);
  const floats = new Float32Array(words.buffer);
  let cellCount = 0, maximumSpan = 1;
  source.atlas.bricks.forEach((brick, leaf) => {
    const at = leafBase + 12 * leaf;
    const width = 8 * sparseBrickSpan(brick) / brick.resolution;
    if (width < 1 || !Number.isInteger(Math.log2(width)))
      throw new Error("CM12 transfer native widths must be aligned powers of two");
    maximumSpan = Math.max(maximumSpan, width);
    let count = 1;
    for (let axis = 0; axis < 3; axis++) {
      const lower = 8 * brick.coordinate[axis]!;
      const maximum = sparseBrickMaximumFine(source.atlas, brick, axis as 0 | 1 | 2);
      const n = Math.min(brick.resolution, Math.ceil((maximum - lower) / width));
      if (n <= 0) throw new Error("CM12 transfer captured an empty source leaf");
      floats[at + axis] = lower;
      floats[at + 8 + axis] = maximum;
      words[at + 4] |= n << (10 * axis);
      count *= n;
    }
    floats[at + 3] = width;
    const first = source.sourceFirst.get(brick.key);
    if (first === undefined) throw new Error("CM12 transfer captured a source leaf without a cell range");
    words[at + 5] = first;
    words[at + 6] = Number(source.active.has(brick.key));
    words[at + 7] = Number(source.dynamicKeys.has(brick.key));
    words[at + 11] = brick.resolution;
    cellCount += count;
  });
  for (const [page, q] of source.sourcePageCoordinates) {
    floats.set(q, pageBase + 4 * page);
    words[pageBase + 4 * page + 3] = 1;
  }
  words.set(source.rows, sourceRows);
  words.set(target.cellIds, targetCells);
  words.set(target.rowIds, targetRows);
  target.atlas.bricks.forEach((brick, leaf) => {
    const width = 8 * sparseBrickSpan(brick) / brick.resolution;
    maximumSpan = Math.max(maximumSpan, width);
    floats[targetWidths + leaf] = width;
  });
  for (let i = 0; i < newAirCoverage.length; i++) {
    floats.set(newAirCoverage[i]!.minimumFine, airBase + 6 * i);
    floats.set(newAirCoverage[i]!.maximumExclusiveFine, airBase + 6 * i + 3);
  }
  const capacity = (count: number) => 2 ** Math.ceil(Math.log2(Math.max(2, 2 * count)));
  return { words, leafBase, pageBase, pageCount: pages, sourceRows, targetCells, targetRows,
    targetWidths, airBase, airCount: newAirCoverage.length, leaves, cellCount,
    sourceRowCount: source.rows.length, cellHashCapacity: capacity(cellCount),
    faceHashCapacity: capacity(source.rows.length), maximumSpan };
}

/** Direct GPU dyadic intersection. The source topology lease fixes the compact
 * geometry while live parity is read at publication, preserving advancing fields. */
export async function prepareSparseCM12GPUGenerationTransfer(
  device: GPUDevice, recipe: CM12CapturedGeometryRecipe,
  source: SparseCM12GenerationFields, target: SparseCM12PackedGenerationTarget,
  maximumTemporaryBytes = Number.POSITIVE_INFINITY,
  newAirCoverage: readonly SparseCM12NewAirCoverage[] = [],
): Promise<PreparedSparseCM12GenerationTransfer> {
  const control = source.liveControl;
  if (!control) throw new Error("CM12 GPU transfer requires the leased source topology and live parity");
  const input = packSparseCM12GPUGenerationTransferInput(recipe, target, newAirCoverage);
  const hashBytes = 4 * (input.cellHashCapacity + input.faceHashCapacity);
  const temporaryBytes = Math.max(4, input.words.byteLength) + hashBytes + 8;
  if (temporaryBytes > maximumTemporaryBytes)
    throw new SparseCM12GenerationBudgetDeferred(temporaryBytes, maximumTemporaryBytes);
  if (hashBytes > device.limits.maxStorageBufferBindingSize
    || input.words.byteLength > device.limits.maxStorageBufferBindingSize)
    throw new SparseCM12GenerationBudgetDeferred(temporaryBytes, device.limits.maxStorageBufferBindingSize);
  const offset = (name: string, pair: readonly [number, number], parity: number) =>
    `fn ${name}()->u32{return select(${pair[0]}u,${pair[1]}u,(ot[${parity}u]&1u)!=0u);}`;
  const code = `
@group(0) @binding(0) var<storage,read> old:array<f32>;
@group(0) @binding(1) var<storage,read_write> next:array<f32>;
@group(0) @binding(2) var<storage,read> m:array<u32>;
@group(0) @binding(3) var<storage,read_write> table:array<atomic<u32>>;
@group(0) @binding(4) var<storage,read_write> fault:atomic<u32>;
@group(0) @binding(5) var<storage,read> ot:array<u32>;
@group(0) @binding(6) var<storage,read> nt:array<u32>;
${offset("oldDensity", control.densityOffsets, control.scalarParityWord)}
${offset("oldGamma", control.gammaOffsets, control.scalarParityWord)}
${offset("oldVelocity", control.velocityOffsets, control.scalarParityWord)}
${offset("oldFace", control.faceOffsets, control.faceParityWord)}
const MAX_SPAN:f32=${input.maximumSpan}.0;
const INVALID:u32=0xffffffffu;
struct Box { lower:vec3f, width:vec3f, span:f32, axis:u32, physical:u32, valid:u32 }
fn mf(at:u32)->f32{return bitcast<f32>(m[at]);}
fn native(at:u32, previous:bool)->u32{if(previous){return ot[at];}return nt[at];}
fn nf(at:u32, previous:bool)->f32{return bitcast<f32>(native(at,previous));}
fn valid(v:f32)->bool{return abs(v)<=3.402823e38;}
fn sourceCell(slot:u32)->Box {
 let leaf=slot/512u;let ordinal=slot%512u;let at=${input.leafBase}u+12u*leaf;
 let n=vec3u(m[at+4u]&1023u,(m[at+4u]>>10u)&1023u,m[at+4u]>>20u);
 if(ordinal>=n.x*n.y*n.z){return Box(vec3f(0),vec3f(0),1,3u,INVALID,0u);}
 let q=vec3u(ordinal%n.x,(ordinal/n.x)%n.y,ordinal/(n.x*n.y));
 let span=mf(at+3u);let lower=vec3f(mf(at),mf(at+1u),mf(at+2u))+vec3f(q)*span;
 let width=min(vec3f(span),vec3f(mf(at+8u),mf(at+9u),mf(at+10u))-lower);
 let resolution=m[at+11u];let local=select(ordinal,q.x+resolution*(q.y+resolution*q.z),m[at+7u]!=0u);
 return Box(lower,width,span,3u,select(INVALID,m[at+5u]+local,m[at+6u]!=0u),1u);
}
fn faceBox(row:u32, previous:bool)->Box {
 let rows=native(3u,previous);let base=native(7u,previous);
 var center=vec3f(0);var axis=0u;var area=1.0;
 if(row<rows){
  axis=native(base+rows+row,previous)>>30u;
  center=vec3f(nf(base+6u*rows+row,previous),nf(base+7u*rows+row,previous),nf(base+8u*rows+row,previous));
  area=nf(base+3u*rows+row,previous);
 }else{
  let relative=row-rows;let page=relative/1728u;let within=relative%1728u;
  if(!previous||page>=${input.pageCount}u||m[${input.pageBase}u+4u*page+3u]==0u){
   atomicOr(&fault,4u);return Box(vec3f(0),vec3f(0),1,0u,INVALID,0u);
  }
  let at=${input.pageBase}u+4u*page;
  center=8.0*vec3f(mf(at),mf(at+1u),mf(at+2u));axis=within/576u;
  let index=within%576u;let uv=index/9u;
  center[axis]+=f32(index%9u);
  center[(axis+1u)%3u]+=f32(uv%8u)+.5;
  center[(axis+2u)%3u]+=f32(uv/8u)+.5;
 }
 let a=(axis+1u)%3u;let b=(axis+2u)%3u;
 var span=1.0;
 loop {
  var lower=center;lower[a]=floor(center[a]/span)*span;lower[b]=floor(center[b]/span)*span;
  var width=vec3f(1);width[a]=2.0*(center[a]-lower[a]);width[b]=2.0*(center[b]-lower[b]);
  if(width[a]>0&&width[b]>0&&width[a]<=span&&width[b]<=span&&abs(width[a]*width[b]-area)<1e-6){
   return Box(lower,width,span,axis,row,1u);
  }
  if(span>=MAX_SPAN){break;}span*=2.0;
 }
 atomicOr(&fault,4u);return Box(vec3f(0),vec3f(0),1,axis,INVALID,0u);
}
fn sourceFace(ordinal:u32)->Box{return faceBox(m[${input.sourceRows}u+ordinal],true);}
fn targetCell(ordinal:u32)->Box {
 let cell=m[${input.targetCells}u+ordinal];let at=nt[6u]+8u*cell;
 let center=vec3f(nf(at,false),nf(at+1u,false),nf(at+2u,false));
 let width=vec3f(nf(at+4u,false),nf(at+5u,false),nf(at+6u,false));
 let leaf=nt[at+7u]>>5u;
 return Box(center-.5*width,width,mf(${input.targetWidths}u+leaf),3u,cell,1u);
}
fn boxKey(lower:vec3f,span:f32,axis:u32)->vec4i {
 return vec4i(vec3i(lower),i32(4u*firstLeadingBit(u32(span))+axis));
}
fn key(box:Box)->vec4i{return boxKey(box.lower,box.span,box.axis);}
fn hash(k:vec4i)->u32 {
 var h=2166136261u;
 for(var axis=0u;axis<4u;axis++){h=(h^bitcast<u32>(k[axis]))*16777619u;}
 h^=h>>16u;h*=2246822519u;h^=h>>13u;return h;
}
fn sourceBox(token:u32,faces:bool)->Box{if(faces){return sourceFace(token-1u);}return sourceCell(token-1u);}
fn tableBase(faces:bool)->u32{return select(0u,${input.cellHashCapacity}u,faces);}
fn tableSize(faces:bool)->u32{return select(${input.cellHashCapacity}u,${input.faceHashCapacity}u,faces);}
fn insert(token:u32,box:Box,faces:bool){
 let k=key(box);let mask=tableSize(faces)-1u;let start=hash(k);let base=tableBase(faces);
 for(var probe=0u;probe<=mask;probe++){
  let at=base+((start+probe)&mask);
  let observed=atomicCompareExchangeWeak(&table[at],0u,token);
  if(observed.exchanged){return;}
  if(observed.old_value==0u){probe-=1u;continue;}
  if(all(key(sourceBox(observed.old_value,faces))==k)){atomicOr(&fault,32u);return;}
 }
 atomicOr(&fault,8u);
}
fn lookup(lower:vec3f,span:f32,axis:u32,faces:bool)->u32 {
 let k=boxKey(lower,span,axis);let mask=tableSize(faces)-1u;let start=hash(k);let base=tableBase(faces);
 for(var probe=0u;probe<=mask;probe++){
  let token=atomicLoad(&table[base+((start+probe)&mask)]);
  if(token==0u){return 0u;}
  if(all(key(sourceBox(token,faces))==k)){return token;}
 }
 atomicOr(&fault,8u);return 0u;
}
fn aligned(lower:vec3f,span:f32,axis:u32)->vec3f{
 var result=floor(lower/span)*span;if(axis<3u){result[axis]=lower[axis];}return result;
}
fn overlap(source:Box,target:Box)->f32{
 var width=max(vec3f(0),min(source.lower+source.width,target.lower+target.width)-max(source.lower,target.lower));
 if(target.axis<3u){width[target.axis]=1;}
 return width.x*width.y*width.z;
}
// Ascend to the next dyadic sibling without per-invocation recursion or stacks.
// x/y/z child order matches the CPU conservative-overlap oracle exactly.
fn successor(position:vec3f,size:f32,target:Box)->vec4f{
 var lower=position;var span=size;
 loop {
  if(span>=target.span){return vec4f(lower,0);}
  let relative=vec3u((lower-target.lower)/span);
  var child=0u;var bit=0u;
  for(var axis=0u;axis<3u;axis++){
   if(axis==target.axis){continue;}child|=(relative[axis]&1u)<<bit;bit++;
  }
  let parent=target.lower+floor((lower-target.lower)/(2.0*span))*(2.0*span);
  if(child+1u<(1u<<bit)){
   lower=parent;bit=0u;child++;
   for(var axis=0u;axis<3u;axis++){
    if(axis==target.axis){lower[axis]=target.lower[axis];continue;}
    lower[axis]+=f32((child>>bit)&1u)*span;bit++;
   }
   return vec4f(lower,span);
  }
  span*=2.0;lower=parent;if(target.axis<3u){lower[target.axis]=target.lower[target.axis];}
 }
}
fn initialAncestor(target:Box,faces:bool)->u32{
 var span=target.span;
 loop {
  let token=lookup(aligned(target.lower,span,target.axis),span,target.axis,faces);
  if(token!=0u){return token;}if(span>=MAX_SPAN){return 0u;}span*=2.0;
 }
}
fn linear(id:vec3u)->u32{return id.x+id.y*${device.limits.maxComputeWorkgroupsPerDimension * 64}u;}
@compute @workgroup_size(64) fn indexCells(@builtin(global_invocation_id) invocation:vec3u){
 let slot=linear(invocation);if(slot>=${512 * input.leaves}u){return;}
 let box=sourceCell(slot);if(box.valid!=0u){insert(slot+1u,box,false);}
}
@compute @workgroup_size(64) fn indexFaces(@builtin(global_invocation_id) invocation:vec3u){
 let ordinal=linear(invocation);if(ordinal>=${input.sourceRowCount}u){return;}
 let box=sourceFace(ordinal);if(box.valid!=0u){insert(ordinal+1u,box,true);}
}
@compute @workgroup_size(64) fn cells(@builtin(global_invocation_id) invocation:vec3u){
 let id=linear(invocation);if(id>=${target.cellIds.length}u){return;}
 let target=targetCell(id);let volume=target.width.x*target.width.y*target.width.z;
 var mass=0.0;var gamma=0.0;var pressure=0.0;var covered=0.0;
 var momentum=vec3f(0);var dryVelocity=vec3f(0);
 var cursor=vec4f(target.lower,target.span);var first=true;
 loop {
  var token=0u;
  if(first){token=initialAncestor(target,false);first=false;}
  else{token=lookup(cursor.xyz,cursor.w,3u,false);}
  if(token!=0u){
   let box=sourceCell(token-1u);let weight=overlap(box,target);covered+=weight;
   if(box.physical==INVALID){gamma+=weight;}
   else{
    let before=box.physical;let rho=old[oldDensity()+before];let g=old[oldGamma()+before];
    let p=old[${source.pressureOffset}u+before];let at=oldVelocity()+4u*before;
    let v=vec3f(old[at],old[at+1u],old[at+2u]);
    if(!valid(rho)||rho<0||!valid(g)||!valid(p)||!valid(v.x)||!valid(v.y)||!valid(v.z)){atomicOr(&fault,1u);}
    mass+=rho*weight;gamma+=g*weight;pressure+=p*weight;
    momentum+=rho*weight*v;dryVelocity+=weight*v;
   }
  }else if(cursor.w>1.0){cursor.w*=.5;continue;}
  cursor=successor(cursor.xyz,cursor.w,target);if(cursor.w==0){break;}
 }
 if(covered>volume+1e-6){atomicOr(&fault,16u);}
 if(covered<volume-1e-6){
  var admitted=false;
  for(var region=0u;region<${input.airCount}u;region++){
   let at=${input.airBase}u+6u*region;
   let lower=vec3f(mf(at),mf(at+1u),mf(at+2u));let upper=vec3f(mf(at+3u),mf(at+4u),mf(at+5u));
   if(all(target.lower>=lower)&&all(target.lower+target.width<=upper)){admitted=true;break;}
  }
  if(!admitted){atomicOr(&fault,16u);}gamma+=volume-covered;
 }
 let dst=target.physical;let rho=mass/volume;let g=gamma/volume;
 var velocity=dryVelocity/volume;if(mass>0){velocity=momentum/mass;}
 next[${target.densityOffset}u+dst]=rho;next[${target.densityOtherOffset}u+dst]=rho;
 next[${target.gammaOffset}u+dst]=g;next[${target.gammaOtherOffset}u+dst]=g;
 next[${target.pressureOffset}u+dst]=pressure/volume;
 for(var axis=0u;axis<3u;axis++){
  next[${target.velocityOffset}u+4u*dst+axis]=velocity[axis];
  next[${target.velocityOtherOffset}u+4u*dst+axis]=velocity[axis];
 }
}
@compute @workgroup_size(64) fn faces(@builtin(global_invocation_id) invocation:vec3u){
 let id=linear(invocation);if(id>=${target.rowIds.length}u){return;}
 let row=m[${input.targetRows}u+id];let target=faceBox(row,false);
 if(target.valid==0u){return;}
 var flux=0.0;var covered=0.0;var cursor=vec4f(target.lower,target.span);var first=true;
 loop {
  var token=0u;
  if(first){token=initialAncestor(target,true);first=false;}
  else{token=lookup(cursor.xyz,cursor.w,target.axis,true);}
  if(token!=0u){
   let box=sourceFace(token-1u);let weight=overlap(box,target);let velocity=old[oldFace()+box.physical];
   if(!valid(velocity)){atomicOr(&fault,2u);}flux+=weight*velocity;covered+=weight;
  }else if(cursor.w>1.0){cursor.w*=.5;continue;}
  cursor=successor(cursor.xyz,cursor.w,target);if(cursor.w==0){break;}
 }
 let rowCount=nt[3u];let rowBase=nt[7u];let packed=nt[rowBase+row];
 let firstTerm=packed&0x7fffffu;let count=packed>>23u;
 var velocity=0.0;var weight=0.0;
 for(var term=0u;term<count;term++){
  let at=nt[8u]+2u*(firstTerm+term);let cell=nt[at];let w=abs(nf(at+1u,false));
  velocity+=w*next[${target.velocityOffset}u+4u*cell+target.axis];weight+=w;
 }
 let area=nf(rowBase+3u*rowCount+row,false);
 if(covered>area+1e-6){atomicOr(&fault,16u);}
 let value=(flux+max(0.0,area-covered)*velocity/max(weight,1e-20))/area;
 next[${target.faceOffset}u+row]=value;next[${target.faceOtherOffset}u+row]=value;
}
`;
  const buffers: GPUBuffer[] = [];
  try {
    const metadata = device.createBuffer({ label: "CM12 compact GPU transfer descriptors", size: Math.max(4, input.words.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); buffers.push(metadata);
    device.queue.writeBuffer(metadata, 0, input.words);
    const hash = device.createBuffer({ label: "CM12 GPU native overlap index", size: hashBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); buffers.push(hash);
    const fault = device.createBuffer({ label: "CM12 GPU generation validation", size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }); buffers.push(fault);
    const readback = device.createBuffer({ label: "CM12 GPU generation receipt", size: 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }); buffers.push(readback);
    const layout = device.createBindGroupLayout({ entries: [0, 1, 2, 3, 4, 5, 6].map(binding => ({
      binding, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: [0, 2, 5, 6].includes(binding) ? "read-only-storage" as const : "storage" as const },
    })) });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const compiler = gpuCompilationManagerFor(device);
    const module = compiler.createShaderModule({ label: "CM12 GPU dyadic generation transfer", code });
    const pipelines = await Promise.all(["indexCells", "indexFaces", "cells", "faces"].map(entryPoint =>
      compiler.compileComputePipeline({ label: `CM12 GPU generation ${entryPoint}`, layout: pipelineLayout,
        compute: { module, entryPoint } }, { priority: "critical" })));
    const bindings = device.createBindGroup({ layout, entries: [source.state, target.state, metadata, hash,
      fault, control.buffer, target.topology].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    return new PreparedSparseCM12GenerationTransfer(device, pipelines, bindings, buffers,
      fault, readback, target.cellIds.length, target.rowIds.length,
      [512 * input.leaves, input.sourceRowCount, target.cellIds.length, target.rowIds.length], [hash]);
  } catch (error) { for (const buffer of buffers) buffer.destroy(); throw error; }
}
