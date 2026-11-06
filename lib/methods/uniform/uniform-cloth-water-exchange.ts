import type { ClothWaterLedger } from '../../cloth/definition';
import { planLocalPoreExchange, planLocalCapacityRemap, remapCapacityMomentum, tetrahedralWaterFill, type LocalWaterCell, type LocalLevelSetExchange } from '../../cloth/grid-water-transfer';
import { porousCaptureFraction } from '../../cloth/porous-material';
import { planPoreAbsorption, type PoreWaterExchangeRequest } from '../../cloth/porous-transport';
import type { UniformClothGridFields, UniformClothStep } from './uniform-cloth-grid-coupling';
import { uniformDetailBindLayout, uniformDetailGroup, uniformDetailModule, uniformDetailPipeline } from './uniform-detail-fields';
import { uniformMixedTopologyWGSL } from './uniform-mixed-topology.wgsl';
import { uniformMixedSolidWGSL } from './uniform-mixed-solid.wgsl';
import { clothPorousCapacityOf, clothCellWeights,clothMacCaptureReservoir, setClothHardApertures, setClothHardAperturesByKey, clothCellKey, prepareClothPorousCapacity, publishClothPorousCapacity, releaseClothPorousCapacity } from './uniform-porous-capacity';
import { NB_VOLUME_COMPENSATION_FLOAT } from './uniform-mixed-surface-volume';
import { writeBufferChunked } from './uniform-chunked-upload';
import {uniformMixedVertexSamplingSource} from './uniform-mixed-vertex-sampling.wgsl';
import {UNIFORM_DETAIL_4H_LOAD} from '../../core/uniform-detail-abi';

type Point=[number,number,number];
interface Cell {p:Point;vertices:number[]}
/** Exact integer cell key (insertion-ordered Maps; no key ordering is used). */
const key=(p:readonly number[])=>clothCellKey(p[0]!,p[1]!,p[2]!);
const ck=(p:readonly number[])=>clothCellKey(p[0]!,p[1]!,p[2]!);
const corner=(i:number):Point=>[i&1,(i>>1)&1,(i>>2)&1];

/** Correctness-first local free/bound water transaction. Reads a compact
 * support, solves the EXACT native tetrahedral fill curve in f32 phi, and
 * commits phi, source momentum and the compensated NB budget together.
 * It never substitutes particles for the surface-volume authority. */
export class UniformClothWaterExchange {
 readonly id='physics.cloth.local-water-exchange';
 private readLayout:GPUBindGroupLayout;
 private writeLayout:GPUBindGroupLayout;
 private readPipeline?:GPUComputePipeline;
 private currentReadPipeline?:GPUComputePipeline;
 private writePipeline?:GPUComputePipeline;
 private readonly phiQueryLayout:GPUBindGroupLayout;
 private phiQueryPipeline?:GPUComputePipeline;
 private readonly device:GPUDevice;
 private busy=false;
 private disposed=false;
 constructor(readonly fields:UniformClothGridFields){
  const d=this.device=fields.device;
  const b=(binding:number,type:GPUBufferBindingType):GPUBindGroupLayoutEntry=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type}});
  this.readLayout=uniformDetailBindLayout(d,{entries:[b(0,'read-only-storage'),b(1,'storage'),
   {binding:2,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float',viewDimension:'3d'}},
   {binding:3,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float',viewDimension:'3d'}}]});
  this.writeLayout=uniformDetailBindLayout(d,{entries:[b(0,'read-only-storage'),b(1,'read-only-storage'),b(2,'storage'),b(3,'uniform'),
   {binding:4,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:'write-only',format:'r32float',viewDimension:'3d'}},
   {binding:5,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:'write-only',format:'rgba32float',viewDimension:'3d'}}]});
  this.phiQueryLayout=uniformDetailBindLayout(d,{entries:[b(0,'read-only-storage'),b(1,'storage'),{binding:2,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float',viewDimension:'3d'}}]});
 }
 async initialize():Promise<void>{
  const d=this.device,topology=uniformMixedTopologyWGSL(this.fields.ownership.capacity,0),solid=this.fields.solid;
  const readCode=/* wgsl */`
@group(1) @binding(0) var<storage,read> sites:array<vec4i>;
@group(1) @binding(1) var<storage,read_write> result:array<vec4f>;
@group(1) @binding(2) var phi:texture_3d<f32>;
@group(1) @binding(3) var velocity:texture_3d<f32>;
@compute @workgroup_size(64) fn sampleWater(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=arrayLength(&sites)){return;}let site=sites[i];let p=site.xyz-32*umFramePage();
 if(site.w==0){result[i]=vec4f(/*c*/textureLoad(phi,p,0).x,0,0,0);return;}
 if(site.w==2){result[i]=/*c*/textureLoad(velocity,p,0);return;}
 // A cell beyond the world box's walls is solid: closed (aperture 0), not an
 // unowned liquid cell. Its dual-face mass is zero; no water is stored there.
 if(umWorldSolid((p>>vec3u(5u))+umFramePage())){result[i]=vec4f(0,0,0,0);return;}
 let tile=512u*umBlockOf(p>>vec3u(5u))+umTileLocal(vec3u(p>>vec3u(2u)));
 if(!umKeyStored(tile)){result[i]=vec4f(0,0,0,2);return;}
 if(umTileWidth(tile)!=1u){result[i]=vec4f(0,0,0,1);return;}
 let open=umCellOpen(p);if(umRigidCoverageFailure.x!=0u){result[i]=vec4f(0,0,0,3);return;}result[i]=vec4f(open,0,0,0);
}`;
  const read=uniformDetailModule(d,{label:'Cloth local water support sample',code:topology+uniformMixedSolidWGSL(solid?2:undefined,solid?.coarse?.count)+readCode});
  const currentRead=uniformDetailModule(d,{label:'Cloth current body-source apertures',code:topology+uniformMixedSolidWGSL(solid?2:undefined,undefined,false,false,true,true)+readCode});
  const phiQuery=uniformDetailModule(d,{label:'Cloth actual mixed-resolution liquid queries',code:topology+/* wgsl */`
struct PhiQuery {cell:vec4i,fraction:vec4f};
@group(1) @binding(0) var<storage,read> queries:array<PhiQuery>;
@group(1) @binding(1) var<storage,read_write> result:array<vec4f>;
@group(1) @binding(2) var phi:texture_3d<f32>;
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
fn umLoadCorner(p:vec3u)->f32{return ${UNIFORM_DETAIL_4H_LOAD}textureLoad(phi,vec3i(p),0).x;}
${uniformMixedVertexSamplingSource('',true,undefined,'umLoadCorner')}
@compute @workgroup_size(64) fn sampleAcceptedPhi(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=arrayLength(&queries)){return;}let query=queries[i];let point=vec3f(query.cell.xyz-32*umFramePage())+query.fraction.xyz;let cell=umPointCell(point);
 // AIR is an explicit native page-table state, whose positive sentinel is
 // the production sampler's accepted background. Read that field value;
 // do not fabricate a distance or turn an unknown liquid owner into air.
 let kind=umCellKind(cell);if(kind==UM_AIR){let air=umLoadCorner(vec3u(cell));result[i]=vec4f(air,0,0,select(3.0,0.0,air>0.0));return;}
 if(kind!=UM_RESIDENT){result[i]=vec4f(0,0,0,4);return;}
 let tile=512u*umBlockOf(cell>>vec3u(5u))+umTileLocal(vec3u(cell>>vec3u(2u)));
 if(!umKeyStored(tile)){result[i]=vec4f(0,0,0,1);return;}
 let owner=umOwnerAt(cell);if(owner.width!=1u&&owner.width!=4u){result[i]=vec4f(0,0,0,2);return;}
 let value=umSampleVertex(point);result[i]=vec4f(value,f32(owner.width),0,0);
}
`});
  const write=uniformDetailModule(d,{label:'Cloth paired local water commit',code:topology+/* wgsl */`
@group(1) @binding(0) var<storage,read> sites:array<vec4i>;
@group(1) @binding(1) var<storage,read> values:array<vec4f>;
@group(1) @binding(2) var<storage,read_write> budget:array<f32>;
@group(1) @binding(3) var<uniform> params:vec4f;
@group(1) @binding(4) var field:texture_storage_3d<r32float,write>;
@group(1) @binding(5) var output:texture_storage_3d<rgba32float,write>;
@compute @workgroup_size(64) fn commitWater(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=arrayLength(&sites)){return;}let site=sites[i];
 if(site.w==0){textureStore(field,site.xyz-32*umFramePage(),values[i]);}else{textureStore(output,site.xyz-32*umFramePage(),values[i]);}
 if(i==0u){budget[0]=params.x;budget[${NB_VOLUME_COMPENSATION_FLOAT}]=params.y;budget[3]=0.0;}
}`});
  {
   const layout=d.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}}]});
   this.budgetPipeline=await d.createComputePipelineAsync({layout:d.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{entryPoint:'addBudget',module:d.createShaderModule({label:'Cloth pore water budget delta',code:/* wgsl */`
@group(0) @binding(0) var<storage,read_write> budget:array<f32>;
@group(0) @binding(1) var<uniform> delta:vec4f;
@compute @workgroup_size(1) fn addBudget(){let old=budget[0];let next=max(0.0,old+delta.x);budget[0]=next;budget[${NB_VOLUME_COMPENSATION_FLOAT}]+=(old-next)+delta.x;budget[3]=0.0;}`})}});
   this.budgetParams=d.createBuffer({label:'Cloth pore water budget delta',size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
   this.budgetGroup=d.createBindGroup({layout,entries:[{binding:0,resource:{buffer:this.fields.waterBudget}},{binding:1,resource:{buffer:this.budgetParams}}]});
  }
  for(const module of [read,currentRead,write,phiQuery]){const errors=(await module.getCompilationInfo()).messages.filter(x=>x.type==='error');if(errors.length)throw new Error(errors.map(x=>x.message).join('\n'));}
  [this.readPipeline,this.currentReadPipeline,this.writePipeline,this.phiQueryPipeline]=await Promise.all([
   uniformDetailPipeline(d,this.fields.ownership,{layout:d.createPipelineLayout({bindGroupLayouts:[this.fields.ownership.bindLayout,this.readLayout,...(solid?[solid.tileLayout]:[])]}),compute:{module:read,entryPoint:'sampleWater'}}),
   uniformDetailPipeline(d,this.fields.ownership,{layout:d.createPipelineLayout({bindGroupLayouts:[this.fields.ownership.bindLayout,this.readLayout,...(solid?[solid.currentGeometryLayout]:[])]}),compute:{module:currentRead,entryPoint:'sampleWater'}}),
   uniformDetailPipeline(d,this.fields.ownership,{layout:d.createPipelineLayout({bindGroupLayouts:[this.fields.ownership.bindLayout,this.writeLayout]}),compute:{module:write,entryPoint:'commitWater'}}),
   uniformDetailPipeline(d,this.fields.ownership,{layout:d.createPipelineLayout({bindGroupLayouts:[this.fields.ownership.bindLayout,this.phiQueryLayout]}),compute:{module:phiQuery,entryPoint:'sampleAcceptedPhi'}}),
  ]);
 }

 /** Sample the authoritative hard collider aperture before the first native
  * free-volume census. No water ledger or synthetic geometry is involved. */
 async initializeCapacity(step:Pick<UniformClothStep,'state'>):Promise<void>{
  if(!this.readPipeline)throw new Error('Initialize the water adapter before capacity');
  const cells=new Map<number,Point>();for(let v=0;v<step.state.waterVolume_m3.length;v++)for(const cell of clothCellWeights(this.fields,step.state.positions_m.subarray(3*v,3*v+3),true))cells.set(key(cell.p),cell.p);
  const points=[...cells.values()],records=new Int32Array(4*points.length);points.forEach((p,i)=>records.set([...p,1],4*i));
  const d=this.device,sites=d.createBuffer({label:'Cloth initial hard-aperture sites',size:Math.max(16,records.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),result=d.createBuffer({label:'Cloth initial hard apertures',size:Math.max(16,records.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=d.createBuffer({label:'Cloth aperture readback',size:Math.max(16,records.byteLength),usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{
   writeBufferChunked(d.queue,sites,0,records);const group=uniformDetailGroup(d,{layout:this.readLayout,entries:[{binding:0,resource:{buffer:sites}},{binding:1,resource:{buffer:result}},{binding:2,resource:this.fields.phi},{binding:3,resource:this.fields.projectedVelocity}]});
   const e=d.createCommandEncoder(),pass=e.beginComputePass({label:'Cloth native initial hard apertures'});pass.setPipeline(this.readPipeline);pass.setBindGroup(0,this.fields.ownership.bindGroup);pass.setBindGroup(1,group.group);if(this.fields.solid)pass.setBindGroup(2,this.fields.solid.tileGroup);pass.dispatchWorkgroups(Math.ceil(points.length/64));pass.end();e.copyBufferToBuffer(result,0,read,0,records.byteLength);d.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);const values=new Float32Array(read.getMappedRange());
   const aperture=new Map<number,number>();points.forEach((p,i)=>{if(values[4*i+3]===3)throw new Error('Initial cloth capacity hard-geometry integration did not converge');if(values[4*i+3])throw new Error('Initial cloth capacity requires actual h support');aperture.set(key(p),values[4*i]!);});
   setClothHardApertures(this.fields,aperture);publishClothPorousCapacity(this.fields,step);
  }finally{if(read.mapState==='mapped')read.unmap();sites.destroy();result.destroy();read.destroy();}
 }

 /** Accepted physical free-water ledger, including this substep's native
  * sources/terminators. Deep grid-owned water does not require particles. */
 async readFreeWaterVolume():Promise<number>{
  if(this.disposed)throw new Error('Water adapter was disposed');
  const d=this.device,read=d.createBuffer({label:'Cloth free-water existence ledger',size:12,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{const e=d.createCommandEncoder();e.copyBufferToBuffer(this.fields.waterBudget,0,read,0,8);e.copyBufferToBuffer(this.fields.waterBudget,4*NB_VOLUME_COMPENSATION_FLOAT,read,8,4);d.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);const v=new Float32Array(read.getMappedRange());if(v[1]!==1)throw new Error('Free-water ledger is not initialized');return Math.max(0,v[0]!+v[2]!)*this.fields.cellSize_m.reduce((a,b)=>a*b,1);
  }finally{if(read.mapState==='mapped')read.unmap();read.destroy();}
 }

 /** Native free-liquid level set at physical positions, used for the
  * source cohesion bridge gate. This samples the actual accepted NB-FLIP
  * surface; bound saturation is never substituted for free-liquid evidence. */
 async sampleLiquidPhi(positions_m:ArrayLike<number>):Promise<Float64Array>{
  if(this.disposed||this.busy||!this.phiQueryPipeline)throw new Error('Liquid phi sampling requires an idle initialized adapter');
  if(positions_m.length%3||Array.from(positions_m).some(x=>!Number.isFinite(x)))throw new Error('Invalid liquid-phi sample points');
  if(!positions_m.length)return new Float64Array();
  const count=positions_m.length/3,records=new Float32Array(8*count),integer=new Int32Array(records.buffer);
  for(let v=0;v<count;v++)for(let a=0;a<3;a++){
   const q=(positions_m[3*v+a]!-this.fields.origin_m[a]!)/this.fields.cellSize_m[a]!,base=Math.floor(q);
   if(base< -2147483648||base>2147483647)throw new Error('Liquid query is outside the native coordinate range');
   integer[8*v+a]=base;records[8*v+4+a]=q-base;
  }
  const d=this.device,input=d.createBuffer({label:'Cloth mixed phi sites',size:records.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),output=d.createBuffer({label:'Cloth mixed phi values',size:16*count,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=d.createBuffer({label:'Cloth mixed phi readback',size:16*count,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  this.busy=true;
  try{
   writeBufferChunked(d.queue,input,0,records);const group=uniformDetailGroup(d,{layout:this.phiQueryLayout,entries:[{binding:0,resource:{buffer:input}},{binding:1,resource:{buffer:output}},{binding:2,resource:this.fields.phi}]});
   const e=d.createCommandEncoder(),pass=e.beginComputePass({label:'Cloth accepted mixed-resolution liquid samples'});pass.setPipeline(this.phiQueryPipeline);pass.setBindGroup(0,this.fields.ownership.bindGroup);pass.setBindGroup(1,group.group);pass.dispatchWorkgroups(Math.ceil(count/64));pass.end();e.copyBufferToBuffer(output,0,read,0,16*count);d.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);const values=new Float32Array(read.getMappedRange());
   return Float64Array.from({length:count},(_,i)=>{if(values[4*i+3]||!Number.isFinite(values[4*i]!))throw new Error(`Cohesion bridge has no accepted resident liquid owner at query ${i}`);return values[4*i]!;});
  }finally{if(read.mapState==='mapped')read.unmap();input.destroy();output.destroy();read.destroy();this.busy=false;if(this.disposed)releaseClothPorousCapacity(this.fields);}
 }

 /** Positive requests capture water, negative requests emit it. A transaction
  * uses one sign; simultaneous capture/release must not recapture its source. */
 async transfer(step:UniformClothStep,requestedBound_m3:ArrayLike<number>,capacityOnly=false,currentHardGeometry=false,captureFractions?:ArrayLike<number>):Promise<ClothWaterLedger>{
  if(this.disposed)throw new Error('Cloth water exchange was disposed');
  if(this.busy)throw new Error('Cloth water transactions must be serialized');
  if(!this.readPipeline||!this.writePipeline)throw new Error('Cloth water exchange is not initialized');
  const {state,data,materials,identity}=step,n=state.dryMass_kg.length;
  if(requestedBound_m3.length!==n||identity.generation!==state.generation||identity.topologyRevision!==state.topologyRevision)throw new Error('Invalid cloth water transaction identity');
  const requests=Float64Array.from(requestedBound_m3);if(requests.some(x=>!Number.isFinite(x)))throw new Error('Non-finite water request');
  const capture=requests.some(x=>x>0);if(capture&&requests.some(x=>x<0))throw new Error('Mixed-sign water transaction');
  for(let i=0;i<n;i++){
   if(capture)requests[i]=Math.min(requests[i]!,Math.max(0,state.controlVolume_m3[i]!-state.solidVolume_m3[i]!-state.waterVolume_m3[i]!));
   else requests[i]=-Math.min(-requests[i]!,state.waterVolume_m3[i]!);
  }
  const empty:ClothWaterLedger={generation:identity.generation,tick:identity.tick,freeDelta_m3:0,boundDelta_m3:0,externalDelta_m3:0};
  if(!capacityOnly&&!requests.some(x=>x!==0))return empty;
  this.busy=true;const resources:GPUBuffer[]=[];
  const allocate=(size:number,usage:GPUBufferUsageFlags,values?:ArrayBufferView)=>{
   const b=this.device.createBuffer({label:'Cloth local water transaction',size:Math.max(16,size),usage});resources.push(b);
   if(values)writeBufferChunked(this.device.queue,b,0,values);return b;
  };
  try{
   const h=this.fields.cellSize_m,cellVolume=h[0]*h[1]*h[2],vertices:Point[]=[],vertexMap=new Map<number,number>(),weights:number[]=[],samples:{indices:number[];weights:number[]}[]=Array.from({length:n},()=>({indices:[],weights:[]}));
   const vertex=(p:Point)=>{const k=key(p);let i=vertexMap.get(k);if(i===undefined){i=vertices.length;vertexMap.set(k,i);vertices.push(p);weights.push(0);}return i;};
   for(let v=0;v<n;v++)if(capacityOnly||requests[v]!==0){
    const sample=new Map<number,number>(),add=(p:Point,w:number)=>{if(!(w>0))return;const i=vertex(p);sample.set(i,(sample.get(i)??0)+w);};
    if(capture&&!capacityOnly){
     // Capture draws from the SAME staggered MAC reservoir as its source
     // diffusivity rate: 1/4*(2*centre+low+high), summed over three axes.
     // A nodal eight-corner phi kernel omits real nearby MAC donors and
     // incorrectly makes their conservative receiver matching impossible.
     // Install this footprint before enumerating ALL incident volume cells.
     for(const cell of clothMacCaptureReservoir(this.fields,state.positions_m.subarray(3*v,3*v+3)))for(let k=0;k<8;k++)add(cell.p.map((x,a)=>x+corner(k)[a]!) as Point,cell.weight/24);
    }else{
     const q=h.map((size,a)=>(state.positions_m[3*v+a]!-this.fields.origin_m[a]!)/size),base=q.map(Math.floor),fraction=q.map((x,a)=>x-base[a]!);
     for(let k=0;k<8;k++){const c=corner(k),p=base.map((x,a)=>x+c[a]!) as Point,w=c.reduce((s,b,a)=>s*(b?fraction[a]!:1-fraction[a]!),1);add(p,w);}
    }
    for(const [i,w] of sample){weights[i]!+=w*(capacityOnly?1:Math.abs(requests[v]!));samples[v]!.indices.push(i);samples[v]!.weights.push(w);}
   }
   if(capacityOnly){
    // Bound flow follows the real shell/rod graph. Its displaced free water
    // needs the same local corridor even when authored controls are several
    // cells apart; isolated vertex kernels cannot represent that flux.
    const edges=new Map<string,[number,number]>(),edge=(a:number,b:number)=>{if(a>b)[a,b]=[b,a];edges.set(`${a},${b}`,[a,b]);};
    for(let t=0;t<data.triangles.length;t+=3)for(let k=0;k<3;k++)edge(data.triangles[t+k]!,data.triangles[t+(k+1)%3]!);
    for(let r=0;r+1<data.rodOffsets.length;r++)for(let j=data.rodOffsets[r]!;j+1<data.rodOffsets[r+1]!;j++)edge(data.rodIndices[j]!,data.rodIndices[j+1]!);
    const addPath=(positions:ArrayLike<number>)=>{for(const [a,b] of edges.values()){
     const length=Math.hypot(...[0,1,2].map(k=>(positions[3*b+k]!-positions[3*a+k]!)/h[k]!)),segments=Math.max(1,Math.ceil(length));
     for(let j=0;j<=segments;j++){const t=j/segments,q=h.map((size,k)=>((1-t)*positions[3*a+k]!+t*positions[3*b+k]!-this.fields.origin_m[k]!)/size),base=q.map(Math.floor);
      for(let c=0;c<8;c++){const bits=corner(c),i=vertex(base.map((x,k)=>x+bits[k]!) as Point);weights[i]=1;}
     }
    }};
    addPath(state.positions_m);
    const previous=clothPorousCapacityOf(this.fields.porousCapacity);if(!previous)throw new Error('Cloth capacity was not initialized');
    if(previous.nodePositions_m.length===state.positions_m.length)addPath(previous.nodePositions_m);
    for(const cell of previous.cells)for(let c=0;c<8;c++){const bits=corner(c),i=vertex(cell.p.map((x,a)=>x+bits[a]!) as Point);weights[i]=1;}
   }
   const supportCount=vertices.length,affected=new Map<number,Point>();
   for(let i=0;i<supportCount;i++)for(let c=0;c<8;c++){const bit=corner(c),p=vertices[i]!.map((x,a)=>x-bit[a]!) as Point;affected.set(key(p),p);}
   // Neighbor cells give the actual masses of affected MAC dual faces.
   const allCells=new Map(affected);for(const p of affected.values())for(let a=0;a<3;a++)for(const sign of [-1,1]){const q=[...p] as Point;q[a]!+=sign;allCells.set(key(q),q);}
   const cells:Cell[]=[],cellMap=new Map<number,number>();
   for(const [k,p] of allCells){cellMap.set(k,cells.length);cells.push({p,vertices:Array.from({length:8},(_,i)=>vertex(p.map((x,a)=>x+corner(i)[a]!) as Point))});}
   const faceMap=new Map<number,Point>();for(const p of affected.values())for(let a=0;a<3;a++){faceMap.set(key(p),p);const q=[...p] as Point;q[a]!--;faceMap.set(key(q),q);}
   const faces=[...faceMap.values()],records=new Int32Array(4*(vertices.length+cells.length+faces.length));
   vertices.forEach((p,i)=>records.set([...p,0],4*i));cells.forEach((c,i)=>records.set([...c.p,1],4*(vertices.length+i)));faces.forEach((p,i)=>records.set([...p,2],4*(vertices.length+cells.length+i)));
   const siteBuffer=allocate(records.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,records),resultBuffer=allocate(records.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC),readback=allocate(records.byteLength+16,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
   const group=uniformDetailGroup(this.device,{layout:this.readLayout,entries:[{binding:0,resource:{buffer:siteBuffer}},{binding:1,resource:{buffer:resultBuffer}},{binding:2,resource:this.fields.phi},{binding:3,resource:this.fields.projectedVelocity}]});
   const encoder=this.device.createCommandEncoder({label:'Sample local cloth water authority'}),pass=encoder.beginComputePass();
   pass.setPipeline(currentHardGeometry?this.currentReadPipeline!:this.readPipeline);pass.setBindGroup(0,this.fields.ownership.bindGroup);pass.setBindGroup(1,group.group);if(this.fields.solid)pass.setBindGroup(2,currentHardGeometry?this.fields.solid.currentGeometryGroup:this.fields.solid.tileGroup);pass.dispatchWorkgroups(Math.ceil(records.length/256));pass.end();
   encoder.copyBufferToBuffer(resultBuffer,0,readback,0,records.byteLength);encoder.copyBufferToBuffer(this.fields.waterBudget,0,readback,records.byteLength,12);
   encoder.copyBufferToBuffer(this.fields.waterBudget,4*NB_VOLUME_COMPENSATION_FLOAT,readback,records.byteLength+12,4);this.device.queue.submit([encoder.finish()]);
   await readback.mapAsync(GPUMapMode.READ);const values=new Float32Array(readback.getMappedRange()).slice();readback.unmap();
   if(values[records.length+1]!==1)throw new Error('Free-water budget must be initialized before cloth absorption');
   const phi=Float32Array.from(vertices,(_,i)=>values[4*i]!);
   for(let v=0;v<n;v++)if(capture&&!captureFractions&&requests[v]!>0){const s=samples[v]!,value=s.indices.reduce((sum,i,j)=>sum+phi[i]!*s.weights[j]!,0);if(value>=0)requests[v]=0;}
   let requested=requests.reduce((a,b)=>a+b,0);if(!capacityOnly&&!requested)return empty;
   const previous=clothPorousCapacityOf(this.fields.porousCapacity);
   if(!previous)throw new Error('Cloth capacity must be installed before the initial free-water census');
   const hardOpen=cells.map((_,i)=>{if(values[4*(vertices.length+i)+3]===3)throw new Error('Local cloth water hard-geometry integration did not converge');const flag=values[4*(vertices.length+i)+3];if(flag)throw new Error(`Local cloth water exchange requires h ownership throughout its derivative support: cell ${cells[i]!.p} (world ${cells[i]!.p.map((x,a)=>(this.fields.origin_m[a]!+(x+.5)*h[a]!).toFixed(4))}) ${flag===1?'has a coarse owner':'has no stored tile'}`);return values[4*(vertices.length+i)]!;});
   setClothHardApertures(this.fields,new Map(cells.map((cell,i)=>[key(cell.p),hardOpen[i]!])));
   // A prescribed hard-body step changes the admissible interpolation basis.
   // Include that basis change in the native hard-domain transition, not as
   // apparent loss of fiber followed by an invented local water source. The
   // material-only remap compares old/new material in the SAME current hard
   // domain. Native head subsequently consumes the new hard geometry and the
   // published partition against its retained physical free-water budget.
   // In particular, an unchanged submerged cloth changes neither phi nor its
   // ledger merely because a clamp occludes part of its interpolation kernel.
   const oldPartition=currentHardGeometry?prepareClothPorousCapacity(this.fields,{state:{...state,
    positions_m:previous.nodePositions_m,waterVolume_m3:previous.nodeWaterVolume_m3,solidVolume_m3:previous.nodeSolidVolume_m3}}):previous;
   const oldCapacity=new Map(oldPartition.cells.map(c=>[key(c.p),1-c.solidFraction-c.boundFraction]));
   const model:LocalWaterCell[]=cells.map((c,i)=>({vertices:c.vertices,capacity_m3:0,previousCapacity_m3:hardOpen[i]!*cellVolume*(oldCapacity.get(key(c.p))??1)}));
   if(captureFractions){
    const free=cells.map((c,i)=>tetrahedralWaterFill(c.vertices.map(v=>phi[v]!))*model[i]!.previousCapacity_m3!);
    for(let v=0;v<n;v++)if(requests[v]!>0){let available=0;
     for(const cell of clothMacCaptureReservoir(this.fields,state.positions_m.subarray(3*v,3*v+3)))available+=cell.weight*(free[cellMap.get(key(cell.p))!]??0);
     requests[v]=Math.min(requests[v]!,Math.max(0,available*captureFractions[v]!));
    }
    requested=requests.reduce((a,b)=>a+b,0);if(!requested)return empty;
   }
   if(!capacityOnly)weights.fill(0);for(let v=0;v<n;v++)for(let j=0;j<samples[v]!.indices.length;j++)weights[samples[v]!.indices[j]!]!+=Math.abs(requests[v]!)*samples[v]!.weights[j]!;
   const maxWeight=weights.reduce((a,b)=>Math.max(a,b),0);for(let i=0;i<weights.length;i++)weights[i]!/=maxWeight;
   // Material geometry and actual f32 phi are closed against the same
   // physical free/bound ledger. Quantized phi does not participate in an
   // unbounded per-receiver fixed-point iteration.
   let plan:LocalLevelSetExchange,boundByVertex:Float64Array=new Float64Array(n),receiverShares:readonly (readonly {v:number;share:number}[])[]=Array.from({length:cells.length},()=>[]);
   if(capacityOnly){
    const prospective=prepareClothPorousCapacity(this.fields,{state}),capacity=new Map(prospective.cells.map(c=>[key(c.p),1-c.solidFraction-c.boundFraction]));
    cells.forEach((c,i)=>{model[i]={...model[i]!,capacity_m3:hardOpen[i]!*cellVolume*(capacity.get(key(c.p))??1)};});
    plan=planLocalCapacityRemap(phi,weights,model,Math.min(...h));
   }else{
    const receivers=Array.from(requests,(requestedBound_m3,v)=>({requestedBound_m3,
     capacityWeights:requestedBound_m3===0?[]:clothCellWeights(this.fields,state.positions_m.subarray(3*v,3*v+3)).map(cell=>{
      const at=cellMap.get(key(cell.p));if(at===undefined)throw new Error('Pore capacity escaped its sampled local support');return {cell:at,weight:cell.weight};
     }),phiWeights:samples[v]!.indices.map((vertex,i)=>({vertex,weight:samples[v]!.weights[i]!}))}));
    const closed=planLocalPoreExchange(phi,model,receivers,Math.min(...h));plan=closed.plan;boundByVertex=closed.boundByReceiver_m3;receiverShares=closed.receiverShares;
   }
   const bound=capacityOnly?0:-plan.freeDelta_m3;if(!capacityOnly&&Math.abs(bound)<1e-30)return empty;
   if(capture&&bound>(values[records.length]!+values[records.length+3]!)*cellVolume)throw new Error('Local cloth absorption would overdraw the free-water budget');
   const oldVolumes=model.map(c=>tetrahedralWaterFill(c.vertices.map(i=>phi[i]!))*(c.previousCapacity_m3??c.capacity_m3)),waterMomentumByVertex=new Float64Array(3*n);
   const rho=materials[0]!.waterDensity_kg_m3;
   const writesP:number[]=[],writesV:number[]=[];
   for(let i=0;i<vertices.length;i++)if(plan.phi_m[i]!==phi[i]){writesP.push(...vertices[i]!,0);writesV.push(plan.phi_m[i]!,0,0,0);}
   const capacityVelocities=capacityOnly?[0,1,2].map(a=>remapCapacityMomentum(faces.map((p,i)=>{
    const q=[...p] as Point;q[a]!++;const l=cellMap.get(key(p)),r=cellMap.get(key(q));
    return {p,oldVolume_m3:.5*((l===undefined?0:oldVolumes[l]!)+(r===undefined?0:oldVolumes[r]!)),
     deltaVolume_m3:.5*((l===undefined?0:plan.cellDelta_m3[l]!)+(r===undefined?0:plan.cellDelta_m3[r]!)),velocity_m_s:values[4*(vertices.length+cells.length+i)+a]!};
   }))):undefined;
   faces.forEach((p,i)=>{
    const velocity=Array.from(values.subarray(4*(vertices.length+cells.length+i),4*(vertices.length+cells.length+i)+4)),next=[...velocity];
    for(let a=0;a<3;a++){
     if(capacityVelocities){next[a]=Math.fround(capacityVelocities[a]![i]!);continue;}
     const q=[...p] as Point;q[a]!++;const l=cellMap.get(key(p)),r=cellMap.get(key(q));
     const old=.5*((l===undefined?0:oldVolumes[l]!)+(r===undefined?0:oldVolumes[r]!)),delta=.5*((l===undefined?0:plan.cellDelta_m3[l]!)+(r===undefined?0:plan.cellDelta_m3[r]!));
     if(!delta)continue;
     const donors=new Map<number,number>();for(const cell of [l,r])if(cell!==undefined)for(const {v,share} of receiverShares[cell]!){donors.set(v,(donors.get(v)??0)+.5*plan.cellDelta_m3[cell]!*share);}
     let emission=0;for(const [v,amount] of donors)emission+=amount*state.velocities_m_s[3*v+a]!;
     if(delta>0)next[a]=Math.fround((old*velocity[a]!+emission)/(old+delta));
     const momentum=rho*((old+delta)*next[a]!-old*velocity[a]!);
     for(const [v,amount] of donors)waterMomentumByVertex[3*v+a]!+=momentum*amount/delta;
    }
    if(next.some((v,a)=>v!==velocity[a])){writesP.push(...p,2);writesV.push(...next);}
   });
   if(!writesP.length){writesP.push(...vertices[0]!,0);writesV.push(phi[0]!,0,0,0);}
   const pArray=new Int32Array(writesP),vArray=new Float32Array(writesV),pBuffer=allocate(pArray.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,pArray),vBuffer=allocate(vArray.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,vArray);
   // CPU two-sum avoids shader fast-math reassociating the compensated low
   // word away. The serialized transaction sampled the current budget above.
   const targetBudget=values[records.length]!+values[records.length+3]!+(capacityOnly?0:plan.freeDelta_m3/cellVolume),high=Math.fround(targetBudget),low=Math.fround(targetBudget-high);
   const params=allocate(16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,new Float32Array([high,low,0,0]));
   const commit=uniformDetailGroup(this.device,{layout:this.writeLayout,entries:[{binding:0,resource:{buffer:pBuffer}},{binding:1,resource:{buffer:vBuffer}},{binding:2,resource:{buffer:this.fields.waterBudget}},{binding:3,resource:{buffer:params}},{binding:4,resource:this.fields.phi},{binding:5,resource:this.fields.projectedVelocity}]});
   const e=this.device.createCommandEncoder({label:'Commit paired cloth water exchange'}),p=e.beginComputePass();p.setPipeline(this.writePipeline);p.setBindGroup(0,this.fields.ownership.bindGroup);p.setBindGroup(1,commit.group);p.dispatchWorkgroups(Math.ceil(pArray.length/256));p.end();this.device.queue.submit([e.finish()]);await this.device.queue.onSubmittedWorkDone();
   for(let v=0;v<n;v++)if(boundByVertex[v]!==0){
    const oldMass=state.dryMass_kg[v]!+rho*state.waterVolume_m3[v]!,newMass=oldMass+rho*boundByVertex[v]!;
    if(step.physics?.solveSolid!==false&&!(((step.kinematicFixed??data.fixed)?.[v]??0)&1))for(let a=0;a<3;a++)state.velocities_m_s[3*v+a]=(oldMass*state.velocities_m_s[3*v+a]!-waterMomentumByVertex[3*v+a]!)/newMass;
    state.waterVolume_m3[v]!+=boundByVertex[v]!;
   }
   publishClothPorousCapacity(this.fields,step);
   return {generation:identity.generation,tick:identity.tick,freeDelta_m3:capacityOnly?0:plan.freeDelta_m3,boundDelta_m3:bound,externalDelta_m3:0};
  }finally{for(const r of resources)r.destroy();this.busy=false;if(this.disposed)releaseClothPorousCapacity(this.fields);}
 }

 /** Accepted material motion/transport changes the free partition. Preserve
  * each compact connected reservoir before performing any capture/release. */
 async reconcileCapacity(step:UniformClothStep,forceHardApertureRefresh=false,currentHardGeometry=false):Promise<ClothWaterLedger>{
  const previous=clothPorousCapacityOf(this.fields.porousCapacity),s=step.state;
  if(!forceHardApertureRefresh&&previous&&previous.nodePositions_m.length===s.positions_m.length&&previous.nodeWaterVolume_m3.length===s.waterVolume_m3.length
   &&previous.nodePositions_m.every((x,i)=>x===s.positions_m[i])&&previous.nodeWaterVolume_m3.every((x,i)=>x===s.waterVolume_m3[i])&&previous.nodeSolidVolume_m3.every((x,i)=>x===s.solidVolume_m3[i]))
   return {generation:step.identity.generation,tick:step.identity.tick,freeDelta_m3:0,boundDelta_m3:0,externalDelta_m3:0};
  return this.transfer(step,new Float64Array(step.state.waterVolume_m3.length),true,currentHardGeometry);
 }

 /** Darcy entry driven by the material suction. Uses current pore capacity;
  * accepted transfer is limited by the measured local free-water reservoir. */
 async absorb(step:UniformClothStep,vertexMaterial?:ArrayLike<number>):Promise<ClothWaterLedger>{
  if(step.physics?.absorption===false)return {generation:step.identity.generation,tick:step.identity.tick,freeDelta_m3:0,boundDelta_m3:0,externalDelta_m3:0};
  const s=step.state,captureRate=step.physics?.captureRate??1,h=Math.min(...this.fields.cellSize_m);
  if(!Number.isFinite(captureRate)||captureRate<0)throw new Error('Invalid cloth capture-rate multiplier');
  const fractions=new Float64Array(s.waterVolume_m3.length),requests=Float64Array.from(s.waterVolume_m3,(water,v)=>{
   const capacity=Math.max(0,s.controlVolume_m3[v]!-s.solidVolume_m3[v]!);if(water>=capacity||capacity===0)return 0;
   const material=step.materials[vertexMaterial?.[v]??0]!,fraction=s.solidFraction[v]!;if(!(fraction>0&&fraction<1))return 0;
   // Upstream capture uses the GRID-cell diffusion timescale and the actual
   // interpolated pure-free-liquid MAC reservoir, not shell thickness.
   fractions[v]=porousCaptureFraction(fraction,material,h,step.dt_s,captureRate);
   return capacity-water;
  });return this.transfer(step,requests,false,false,fractions);
 }
 async releaseOverflow(step:UniformClothStep):Promise<ClothWaterLedger>{
  return this.transfer(step,Float64Array.from(step.state.waterVolume_m3,(water,i)=>-Math.max(0,water-Math.max(0,step.state.controlVolume_m3[i]!-step.state.solidVolume_m3[i]!))));
 }
 /** One readback of the accepted native fields around every material control:
  * hard aperture (installed for the shared capacity basis), tetrahedral free
  * fill and MAC velocity, on each control's staggered capture reservoir. */
 async sampleSupport(positions_m:ArrayLike<number>,currentHardGeometry=false):Promise<ClothSupportSample>{
  if(this.disposed)throw new Error('Cloth water exchange was disposed');
  if(this.busy)throw new Error('Cloth water samples must be serialized');
  if(!this.readPipeline||!this.currentReadPipeline)throw new Error('Cloth water exchange is not initialized');
  const cells:Point[]=[],cellIndex=new Map<number,number>(),points:Point[]=[],pointIndex=new Map<number,number>();
  const point=(p:Point)=>{const k=ck(p);let i=pointIndex.get(k);if(i===undefined){i=points.length;pointIndex.set(k,i);points.push(p);}return i;};
  const corners:number[]=[];
  const cell=(p:Point)=>{const k=ck(p);if(cellIndex.has(k))return;cellIndex.set(k,cells.length);cells.push(p);for(let c=0;c<8;c++)corners.push(point(p.map((x,a)=>x+corner(c)[a]!) as Point));};
  for(let v=0;v<positions_m.length/3;v++)for(const raw of clothCellWeights(this.fields,[positions_m[3*v]!,positions_m[3*v+1]!,positions_m[3*v+2]!],true)){
   cell(raw.p);for(let a=0;a<3;a++)for(const s of [-1,1]){const q=[...raw.p] as Point;q[a]!+=s;cell(q);}
  }
  const records=new Int32Array(4*(points.length+2*cells.length));
  points.forEach((p,i)=>records.set([...p,0],4*i));cells.forEach((p,i)=>{records.set([...p,1],4*(points.length+i));records.set([...p,2],4*(points.length+cells.length+i));});
  const d=this.device,sites=d.createBuffer({label:'Cloth support sites',size:Math.max(16,records.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),
   result=d.createBuffer({label:'Cloth support samples',size:Math.max(16,records.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),
   read=d.createBuffer({label:'Cloth support readback',size:Math.max(16,records.byteLength),usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  this.busy=true;
  try{
   writeBufferChunked(d.queue,sites,0,records);
   const group=uniformDetailGroup(d,{layout:this.readLayout,entries:[{binding:0,resource:{buffer:sites}},{binding:1,resource:{buffer:result}},{binding:2,resource:this.fields.phi},{binding:3,resource:this.fields.projectedVelocity}]});
   const e=d.createCommandEncoder({label:'Sample cloth native support'}),pass=e.beginComputePass();
   pass.setPipeline(currentHardGeometry?this.currentReadPipeline:this.readPipeline);pass.setBindGroup(0,this.fields.ownership.bindGroup);pass.setBindGroup(1,group.group);
   if(this.fields.solid)pass.setBindGroup(2,currentHardGeometry?this.fields.solid.currentGeometryGroup:this.fields.solid.tileGroup);
   pass.dispatchWorkgroups(Math.ceil(records.length/256));pass.end();e.copyBufferToBuffer(result,0,read,0,records.byteLength);d.queue.submit([e.finish()]);
   await read.mapAsync(GPUMapMode.READ);const values=new Float32Array(read.getMappedRange()).slice();read.unmap();
   const open=new Float64Array(cells.length),fill=new Float64Array(cells.length),velocity=new Float64Array(3*cells.length),apertures=new Map<number,number>();
   cells.forEach((p,i)=>{
    const at=4*(points.length+i);if(values[at+3]===3)throw new Error('Cloth support hard-geometry integration did not converge');
    if(values[at+3])throw new Error(`Cloth support requires h ownership at cell ${p}`);
    open[i]=values[at]!;apertures.set(ck(p),open[i]!);
    fill[i]=tetrahedralWaterFill(Array.from({length:8},(_,c)=>values[4*corners[8*i+c]!]!));
    for(let a=0;a<3;a++)velocity[3*i+a]=values[4*(points.length+cells.length+i)+a]!;
   });
   setClothHardAperturesByKey(this.fields,apertures);
   return {cellIndex,cells,open,fill,velocity};
  }finally{if(read.mapState==='mapped')read.unmap();sites.destroy();result.destroy();read.destroy();this.busy=false;if(this.disposed)releaseClothPorousCapacity(this.fields);}
 }

 /** Add a committed free-water delta to the native NB budget target on the
  * GPU queue, with the same compensated low word as source emission. */
 private commitBudgetDelta(delta_m3:number):void{
  if(!this.budgetPipeline||!this.budgetParams||!this.budgetGroup)throw new Error('Cloth pore budget is not initialized');
  const cells=delta_m3/this.fields.cellSize_m.reduce((a,b)=>a*b,1);
  this.device.queue.writeBuffer(this.budgetParams!,0,new Float32Array([cells,0,0,0]));
  const e=this.device.createCommandEncoder({label:'Cloth pore water budget'}),pass=e.beginComputePass();pass.setPipeline(this.budgetPipeline);pass.setBindGroup(0,this.budgetGroup!);pass.dispatchWorkgroups(1);pass.end();this.device.queue.submit([e.finish()]);
 }
 private budgetPipeline?:GPUComputePipeline;
 private budgetParams?:GPUBuffer;
 private budgetGroup?:GPUBindGroup;

 /** Darcy capture from the measured local free reservoir. Native free water
  * is fill(phi)*open*(1-fiber-bound), so raising the bound partition removes
  * the captured water from the same cells without editing phi; the budget
  * target falls by exactly the accepted volume. Captured water brings its
  * fluid momentum into the material. Returns the accepted per-vertex volumes. */
 captureFromSample(step:UniformClothStep,sample:ClothSupportSample,vertexMaterial?:ArrayLike<number>):Float64Array{
  const s=step.state,n=s.waterVolume_m3.length,accepted=new Float64Array(n);
  if(step.physics?.absorption===false)return accepted;
  const captureRate=step.physics?.captureRate??1,h=Math.min(...this.fields.cellSize_m),V=this.fields.cellSize_m.reduce((a,b)=>a*b,1);
  if(!Number.isFinite(captureRate)||captureRate<0)throw new Error('Invalid cloth capture-rate multiplier');
  const table=clothPorousCapacityOf(this.fields.porousCapacity);if(!table)throw new Error('Cloth capacity must be installed before capture');
  const occupied=new Map(table.cells.map(c=>[ck(c.p),c.solidFraction+c.boundFraction]));
  const free=Float64Array.from(sample.cells,(p,i)=>sample.fill[i]!*sample.open[i]!*V*Math.max(0,1-(occupied.get(ck(p))??0)));
  const capacity=Float64Array.from(s.controlVolume_m3,(c,v)=>Math.max(0,c-s.solidVolume_m3[v]!));
  const requests:PoreWaterExchangeRequest[]=[];
  for(let v=0;v<n;v++){
   const room=capacity[v]!-s.waterVolume_m3[v]!,fraction=s.solidFraction[v]!;if(!(room>0)||!(fraction>0&&fraction<1))continue;
   const material=step.materials[vertexMaterial?.[v]??0]!,rate=porousCaptureFraction(fraction,material,h,step.dt_s,captureRate);if(!(rate>0))continue;
   const own:PoreWaterExchangeRequest[]=[];let total=0;
   for(const c of clothMacCaptureReservoir(this.fields,s.positions_m.subarray(3*v,3*v+3))){
    const i=sample.cellIndex.get(ck(c.p));if(i===undefined)throw new Error('Capture reservoir escaped its sampled support');
    const volume=rate*c.weight*free[i]!;if(volume>0){own.push({cell:i,vertex:v,volume_m3:volume});total+=volume;}
   }
   const scale=total>room?room/total:1;for(const r of own)requests.push({...r,volume_m3:r.volume_m3*scale});
  }
  if(!requests.length)return accepted;
  const plan=planPoreAbsorption(free,s.waterVolume_m3,capacity,requests);
  for(const t of plan.transfers)accepted[t.vertex]!+=t.volume_m3;
  const rho=step.materials[0]!.waterDensity_kg_m3;
  for(let v=0;v<n;v++)if(accepted[v]!>0){
   if(step.physics?.solveSolid!==false&&!(((step.kinematicFixed??step.data.fixed)?.[v]??0)&1)){
    const u=this.fluidVelocityAt(sample,s.positions_m.subarray(3*v,3*v+3),s.velocities_m_s.subarray(3*v,3*v+3)),mass=s.dryMass_kg[v]!+rho*s.waterVolume_m3[v]!,added=rho*accepted[v]!;
    for(let a=0;a<3;a++)s.velocities_m_s[3*v+a]=(mass*s.velocities_m_s[3*v+a]!+added*u[a]!)/(mass+added);
   }
   s.waterVolume_m3[v]!+=accepted[v]!;
  }
  return accepted;
 }

 /** Water above the compressed pore capacity seeps into adjoining free
  * liquid in proportion to the control's wetted kernel. Overflow in air is
  * retained as explicit unresolved storage (never deleted or duplicated). */
 releaseFromSample(step:UniformClothStep,sample:ClothSupportSample):Float64Array{
  const s=step.state,n=s.waterVolume_m3.length,released=new Float64Array(n),rho=step.materials[0]!.waterDensity_kg_m3;
  for(let v=0;v<n;v++){
   const overflow=s.waterVolume_m3[v]!-Math.max(0,s.controlVolume_m3[v]!-s.solidVolume_m3[v]!);if(!(overflow>0))continue;
   let wet=0;for(const c of clothCellWeights(this.fields,s.positions_m.subarray(3*v,3*v+3)))wet+=c.weight*sample.fill[sample.cellIndex.get(ck(c.p))!]!;
   const amount=overflow*Math.min(1,Math.max(0,wet));if(!(amount>0))continue;
   if(step.physics?.solveSolid!==false&&!(((step.kinematicFixed??step.data.fixed)?.[v]??0)&1)){
    const u=this.fluidVelocityAt(sample,s.positions_m.subarray(3*v,3*v+3),s.velocities_m_s.subarray(3*v,3*v+3)),mass=s.dryMass_kg[v]!+rho*s.waterVolume_m3[v]!,removed=rho*amount;
    for(let a=0;a<3;a++)s.velocities_m_s[3*v+a]=(mass*s.velocities_m_s[3*v+a]!-removed*u[a]!)/(mass-removed);
   }
   s.waterVolume_m3[v]!-=amount;released[v]=amount;
  }
  return released;
 }

 /** Fill-weighted cell-centred MAC velocity over the control's kernel. */
 private fluidVelocityAt(sample:ClothSupportSample,position:ArrayLike<number>,fallback:ArrayLike<number>):[number,number,number]{
  const u=[0,0,0];let weight=0;
  for(const c of clothCellWeights(this.fields,position)){
   const i=sample.cellIndex.get(ck(c.p))!,w=c.weight*sample.fill[i]!;if(!(w>0))continue;weight+=w;
   for(let a=0;a<3;a++){const q=[...c.p] as Point;q[a]!--;const low=sample.cellIndex.get(ck(q));u[a]!+=w*.5*(sample.velocity[3*i+a]!+(low===undefined?sample.velocity[3*i+a]!:sample.velocity[3*low+a]!));}
  }
  return weight>0?u.map(x=>x/weight) as [number,number,number]:[fallback[0]!,fallback[1]!,fallback[2]!];
 }

 /** Moving hard geometry changes the shared aperture basis: resample it at
  * the current collider poses and republish the material partition. */
 async refreshApertures(step:Pick<UniformClothStep,'state'>):Promise<void>{
  await this.sampleSupport(step.state.positions_m,true);publishClothPorousCapacity(this.fields,step);
 }
 /** Install the accepted capacity partition and commit the paired budget. */
 commitPoreWater(step:UniformClothStep,freeDelta_m3:number):ClothWaterLedger{
  if(!Number.isFinite(freeDelta_m3))throw new Error('Non-finite pore water exchange');
  publishClothPorousCapacity(this.fields,step);
  if(freeDelta_m3!==0)this.commitBudgetDelta(freeDelta_m3);
  return {generation:step.identity.generation,tick:step.identity.tick,freeDelta_m3,boundDelta_m3:-freeDelta_m3,externalDelta_m3:0};
 }
 destroy():void{this.disposed=true;this.budgetParams?.destroy();if(!this.busy)releaseClothPorousCapacity(this.fields);}
}
export interface ClothSupportSample {readonly cellIndex:ReadonlyMap<number,number>;readonly cells:readonly Point[];readonly open:Float64Array;readonly fill:Float64Array;readonly velocity:Float64Array}
