import type { PreparedUniformClothStencil, UniformClothGridFields } from './uniform-cloth-grid-coupling';
import { uniformDetailBindLayout,uniformDetailGroup,uniformDetailModule,uniformDetailPipeline,type UniformDetailGroup } from './uniform-detail-fields';
import { UniformClothDragMaterial } from './uniform-cloth-drag-material';
import { uniformMixedTopologyWGSL } from './uniform-mixed-topology.wgsl';
import { uniformMixedSolidWGSL } from './uniform-mixed-solid.wgsl';
import { CLOTH_DRAG_COLORS, type UniformClothGPUStencil } from './uniform-cloth-gpu-stencil';
import type { UniformClothGPUCells } from './uniform-cloth-gpu-cells';

/** One cloth topology's GPU-resident exchange tables (built on the GPU each step). */
export interface UniformClothTensorResources {readonly stencil:UniformClothGPUStencil;readonly cells:UniformClothGPUCells;readonly impulses:GPUBuffer;readonly status:GPUBuffer}
/** The step's scalars; no table crosses the host boundary. */
export interface UniformClothTensorStep {readonly dt_s:number;readonly density_kg_m3:number;readonly exponent:number;readonly solveSolid:boolean}
const ANCHORS='struct Anchor {p:vec3i,first:u32,counts:vec3u,unused:u32}; struct Anchors {count:vec4u,unused:vec4u,items:array<Anchor>};';
const VERTEX='struct Vertex {velocityMass:vec4f,linear0:vec4f,linear1:vec4f,linear2:vec4f,nonlinear0:vec4f,nonlinear1:vec4f,nonlinear2:vec4f,extra:vec4f};';

/** Conflict-free drag colors (the CPU oracle of the GPU Jones-Plassmann
 * coloring): sequential greedy in decreasing priority (clothColorPriorities),
 * a vertex taking the least color no anchor it touches holds. */
export function clothDragColors(p:Pick<PreparedUniformClothStencil,'vertices'|'ranges'|'gather'|'linkFaces'|'coordinates'>,priorities:Uint32Array):Uint32Array{
 const anchorsOf=(v:number)=>{const out=new Set<number>();for(let axis=0;axis<3;axis++)for(let j=0;j<p.ranges[6*v+2*axis+1]!;j++)out.add(Math.floor(p.linkFaces[p.gather[p.ranges[6*v+2*axis]!+j]!]!/3));return out;};
 const order=Array.from({length:p.vertices},(_,v)=>v).sort((a,b)=>priorities[b]!-priorities[a]!);
 const held:Set<number>[]=Array.from({length:p.coordinates.length},()=>new Set()),colors=new Uint32Array(p.vertices);
 for(const v of order){const anchors=anchorsOf(v);let color=0;for(;;color++){let free=true;for(const a of anchors)if(held[a]!.has(color)){free=false;break;}if(free)break;}colors[v]=color;for(const a of anchors)held[a]!.add(color);}
 return colors;
}
/** Full rotated SPD drag, including off-diagonal material terms. Each local
 * backward-Euler block couples one structural control to its actual staggered
 * fluid stencil. Conflict-free colors are advanced sequentially (Lie splitting
 * between controls); no float atomics or concurrent face writes are used.
 * Every block dissipates energy and publishes its equal/opposite impulse. */
export class UniformClothTensorDrag {
 private readonly device:GPUDevice;
 private readonly initLayout:GPUBindGroupLayout;
 private readonly blockLayout:GPUBindGroupLayout;
 private readonly commitLayout:GPUBindGroupLayout;
 private readonly vertexLayout:GPUBindGroupLayout;
 private initPipeline?:GPUComputePipeline;
 private blockPipeline?:GPUComputePipeline;
 private commitPipeline?:GPUComputePipeline;
 private vertexPipeline?:GPUComputePipeline;
 private initGroup?:UniformDetailGroup;
 private commitGroup?:UniformDetailGroup;
 private vertexGroup?:GPUBindGroup;
 private groups:GPUBindGroup[]=[];
 private buffers:GPUBuffer[]=[];
 private resources?:UniformClothTensorResources&{vertices:GPUBuffer;faces:GPUBuffer;physics:GPUBuffer;step:GPUBuffer};
 private material:UniformClothDragMaterial;
 constructor(private readonly fields:UniformClothGridFields){
  this.material=new UniformClothDragMaterial(fields);
  const d=this.device=fields.device,b=(binding:number,type:GPUBufferBindingType):GPUBindGroupLayoutEntry=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type}});
  const texture=(binding:number):GPUBindGroupLayoutEntry=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float',viewDimension:'3d'}});
  this.initLayout=uniformDetailBindLayout(d,{entries:[b(0,'read-only-storage'),b(2,'storage'),b(4,'uniform'),texture(5),texture(6)]});
  this.blockLayout=d.createBindGroupLayout({entries:[b(0,'storage'),b(1,'read-only-storage'),b(2,'storage'),b(3,'storage'),b(4,'storage'),b(5,'storage'),b(6,'uniform'),b(7,'uniform')]});
  this.commitLayout=uniformDetailBindLayout(d,{entries:[b(0,'read-only-storage'),b(1,'read-only-storage'),{binding:2,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:'write-only',format:'rgba32float',viewDimension:'3d'}}]});
  this.vertexLayout=d.createBindGroupLayout({entries:[b(0,'storage'),b(1,'read-only-storage'),b(2,'read-only-storage'),b(3,'uniform')]});
 }
 async initialize():Promise<void>{
  await this.material.initialize();
  const d=this.device,topology=uniformMixedTopologyWGSL(this.fields.ownership.capacity,0),anchor=ANCHORS,face='struct Face {velocity:vec4f,mass:vec4f};';
  const solid=this.fields.solid;
  const init=uniformDetailModule(d,{label:'Cloth tensor drag native MAC mass',code:topology+uniformMixedSolidWGSL(solid?2:undefined,solid?.coarse?.count)+anchor+face+/* wgsl */`
@group(1) @binding(0) var<storage,read> anchors:Anchors;
@group(1) @binding(2) var<storage,read_write> faces:array<Face>;
@group(1) @binding(4) var<uniform> physics:vec4f;
@group(1) @binding(5) var velocity:texture_3d<f32>;
@group(1) @binding(6) var volume:texture_3d<f32>;
fn supported(p:vec3i)->bool{let tile=512u*umBlockOf(p>>vec3u(5u))+umTileLocal(vec3u(p>>vec3u(2u)));return umKeyStored(tile)&&umTileWidth(tile)==1u;}
@compute @workgroup_size(64) fn initializeFaces(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=anchors.count.x){return;}var a=anchors.items[i];a.p-=32*umFramePage();var result=Face(vec4f(0.0),vec4f(0.0));
 if(!supported(a.p)){faces[i]=result;return;}result.velocity=/*c*/textureLoad(velocity,a.p,0);
 for(var axis=0u;axis<3u;axis++){
  var q=a.p;q[axis]+=1;if(!supported(q)){continue;}
  if(umPressureFaceV(a.p,axis)<=0.0){continue;}
  let wet=clamp(0.5*(/*c*/textureLoad(volume,a.p,0).x+/*c*/textureLoad(volume,q,0).x),0.0,1.0);
  result.mass[axis]=physics.x*physics.y*wet;
 }faces[i]=result;
}`});
  const vertexInit=d.createShaderModule({label:'Cloth tensor drag control state',code:VERTEX+/* wgsl */`
@group(0) @binding(0) var<storage,read_write> vertices:array<Vertex>;
@group(0) @binding(1) var<storage,read> predictor:array<vec4f>;
@group(0) @binding(2) var<storage,read> scalars:array<vec4f>;
@group(0) @binding(3) var<uniform> physics:vec4f;
// Same-step nonfluid predictor and wet mass (hard supports: zero inverse
// mass). The material kernels then fill all six tensor rows from actual
// deposited grid packing and native free-water volume.
@compute @workgroup_size(64) fn initializeVertices(@builtin(global_invocation_id) gid:vec3u){
 let v=gid.x;if(v>=arrayLength(&vertices)){return;}let s=scalars[v];let fixed=physics.z==0.0||s.w!=0.0;
 vertices[v].velocityMass=vec4f(predictor[v].xyz,select(s.x+physics.x*s.y,0.0,fixed));vertices[v].extra=vec4f(physics.w,0.0,0.0,0.0);
}`});
  const block=d.createShaderModule({label:'Cloth full tensor implicit block',code:face+VERTEX+/* wgsl */`
// dispatch: (color lists, color, gather, link faces); extra: (link weights, color ranges, 0, 0).
struct Parameters {dispatch:vec4u,extra:vec4u};
@group(0) @binding(0) var<storage,read_write> vertices:array<Vertex>;
@group(0) @binding(1) var<storage,read> graph:array<u32>;
@group(0) @binding(2) var<storage,read_write> faces:array<Face>;
@group(0) @binding(3) var<storage,read_write> links:array<vec4f>;
@group(0) @binding(4) var<storage,read_write> impulses:array<f32>;
@group(0) @binding(5) var<storage,read_write> status:array<atomic<u32>>;
@group(0) @binding(6) var<uniform> params:Parameters;
@group(0) @binding(7) var<uniform> stepTime:vec4f;
fn linkAt(v:u32,axis:u32,j:u32)->u32{return graph[params.dispatch.z+graph[6u*v+2u*axis]+j];}
fn faceAt(link:u32)->u32{return graph[params.dispatch.w+link];}
fn weightAt(link:u32)->f32{return bitcast<f32>(graph[params.extra.x+link]);}
fn inverse3(m:mat3x3f)->mat3x3f{let a=cross(m[1],m[2]);let b=cross(m[2],m[0]);let c=cross(m[0],m[1]);return transpose(mat3x3f(a,b,c))*(1.0/dot(m[0],a));}
@compute @workgroup_size(64) fn exchangeBlock(@builtin(global_invocation_id) gid:vec3u){
 let range=params.extra.y+2u*params.dispatch.y;if(gid.x>=graph[range+1u]){return;}let v=graph[params.dispatch.x+graph[range]+gid.x];let vertex=vertices[v];
 var weights=vec3f(0.0);var fluid=vec3f(0.0);var inverseMass=vec3f(0.0);
 for(var axis=0u;axis<3u;axis++){
  let count=graph[6u*v+2u*axis+1u];
  for(var j=0u;j<count;j++){let l=linkAt(v,axis,j);let f=faceAt(l);let mass=faces[f/3u].mass[f%3u];if(mass<=1e-30){continue;}let w=weightAt(l);weights[axis]+=w;fluid[axis]+=w*faces[f/3u].velocity[f%3u];inverseMass[axis]+=w*w/mass;}
 }
 // A vector block needs a physical water stencil for every component.
 if(any(weights<=vec3f(1e-12))){return;}
 fluid/=weights;inverseMass/=weights*weights;
 if(vertex.velocityMass.w>0.0){inverseMass+=vec3f(1.0/vertex.velocityMass.w);}
 let relative=fluid-vertex.velocityMass.xyz;let nonlinear=pow(length(relative),vertex.extra.x);
 let C=(mat3x3f(vertex.linear0.xyz,vertex.linear1.xyz,vertex.linear2.xyz)+nonlinear*mat3x3f(vertex.nonlinear0.xyz,vertex.nonlinear1.xyz,vertex.nonlinear2.xyz))*(dot(weights,vec3f(1.0))/3.0);
 let scale=max(max(C[0].x,C[1].y),C[2].z);if(scale<=0.0){return;}
 let normalized=C*(1.0/scale);let determinant=dot(normalized[0],cross(normalized[1],normalized[2]));
 if(determinant<=1e-20){atomicAdd(&status[0],1u);return;}
 let dt=stepTime.x;var A=inverse3(normalized)*(1.0/(scale*dt));
 A[0].x+=inverseMass.x;A[1].y+=inverseMass.y;A[2].z+=inverseMass.z;
 let aScale=max(max(A[0].x,A[1].y),A[2].z);let J=inverse3(A*(1.0/aScale))*relative/aScale;
 var nextSolid=vertex.velocityMass.xyz;if(vertex.velocityMass.w>0.0){nextSolid+=J/vertex.velocityMass.w;}vertices[v].velocityMass=vec4f(nextSolid,vertex.velocityMass.w);
 for(var axis=0u;axis<3u;axis++){
  let count=graph[6u*v+2u*axis+1u];
  for(var j=0u;j<count;j++){let l=linkAt(v,axis,j);let f=faceAt(l);let mass=faces[f/3u].mass[f%3u];links[l].w=nextSolid[axis];if(mass<=1e-30){continue;}let impulse=weightAt(l)/weights[axis]*J[axis];
   faces[f/3u].velocity[f%3u]-=impulse/mass;impulses[l]+=impulse;
  }
 }
}`});
  const commit=uniformDetailModule(d,{label:'Cloth tensor drag MAC commit',code:topology+anchor+face+/* wgsl */`
@group(1) @binding(0) var<storage,read> anchors:Anchors;
@group(1) @binding(1) var<storage,read> faces:array<Face>;
@group(1) @binding(2) var output:texture_storage_3d<rgba32float,write>;
@compute @workgroup_size(64) fn commitFaces(@builtin(global_invocation_id) gid:vec3u){let i=gid.x;if(i>=anchors.count.x){return;}let p=anchors.items[i].p-32*umFramePage();let tile=512u*umBlockOf(p>>vec3u(5u))+umTileLocal(vec3u(p>>vec3u(2u)));if(umKeyStored(tile)&&umTileWidth(tile)==1u){textureStore(output,p,faces[i].velocity);}}
`});
  for(const [name,module] of [['mass',init],['vertex',vertexInit],['block',block],['commit',commit]] as const){const errors=(await module.getCompilationInfo()).messages.filter(x=>x.type==='error');if(errors.length)throw new Error(errors.map(x=>`Cloth tensor ${name} ${x.lineNum}:${x.linePos}: ${x.message}`).join('\n'));}
  [this.initPipeline,this.blockPipeline,this.commitPipeline,this.vertexPipeline]=await Promise.all([
   uniformDetailPipeline(d,this.fields.ownership,{layout:d.createPipelineLayout({bindGroupLayouts:[this.fields.ownership.bindLayout,this.initLayout,...(solid?[solid.tileLayout]:[])]}),compute:{module:init,entryPoint:'initializeFaces'}}),
   d.createComputePipelineAsync({layout:d.createPipelineLayout({bindGroupLayouts:[this.blockLayout]}),compute:{module:block,entryPoint:'exchangeBlock'}}),
   uniformDetailPipeline(d,this.fields.ownership,{layout:d.createPipelineLayout({bindGroupLayouts:[this.fields.ownership.bindLayout,this.commitLayout]}),compute:{module:commit,entryPoint:'commitFaces'}}),
   d.createComputePipelineAsync({layout:d.createPipelineLayout({bindGroupLayouts:[this.vertexLayout]}),compute:{module:vertexInit,entryPoint:'initializeVertices'}}),
  ]);
 }
 /** Bind one topology's GPU tables. Persistent: per-color passes read their
  * range from the GPU coloring and launch indirectly (empty colors launch none). */
 attach(r:UniformClothTensorResources):void{
  this.release();const d=this.device,V=r.stencil.topology.vertices,graph=r.stencil.sections.graph;
  const create=(size:number,usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST)=>{const b=d.createBuffer({label:'Cloth tensor block data',size:Math.max(16,size),usage});this.buffers.push(b);return b;};
  const vertices=create(128*V),faces=create(32*r.stencil.anchorCapacity),physics=create(16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST),step=create(16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
  const colorParams=create(256*CLOTH_DRAG_COLORS,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST),words=new Uint32Array(64*CLOTH_DRAG_COLORS);
  for(let c=0;c<CLOTH_DRAG_COLORS;c++)words.set([graph.colors,c,graph.gather,graph.faces,graph.weights,graph.colorRanges,0,0],64*c);
  d.queue.writeBuffer(colorParams,0,words);
  for(let c=0;c<CLOTH_DRAG_COLORS;c++)this.groups.push(d.createBindGroup({layout:this.blockLayout,entries:[vertices,r.stencil.graph,faces,r.stencil.coefficients,r.impulses,r.status].map((buffer,binding):GPUBindGroupEntry=>({binding,resource:{buffer}}))
   .concat([{binding:6,resource:{buffer:colorParams,offset:256*c,size:32}},{binding:7,resource:{buffer:step,offset:0,size:16}}])}));
  this.vertexGroup=d.createBindGroup({layout:this.vertexLayout,entries:[vertices,r.stencil.predictor,r.stencil.scalars,physics].map((buffer,binding)=>({binding,resource:{buffer}}))});
  this.material.attach(r.cells,vertices,V);
  this.resources={...r,vertices,faces,physics,step};
 }
 /** The step's scalars, and the native-field groups at the fields' current textures. */
 prepare(step:UniformClothTensorStep):void{
  const r=this.resources;if(!r)throw new Error('Tensor drag has no attached cloth tables');const d=this.device;
  d.queue.writeBuffer(r.physics,0,new Float32Array([step.density_kg_m3,this.fields.cellSize_m.reduce((a,b)=>a*b,1),step.solveSolid?1:0,step.exponent]));
  d.queue.writeBuffer(r.step,0,new Float32Array([step.dt_s,0,0,0]));
  this.material.prepare();
  this.initGroup=uniformDetailGroup(d,{layout:this.initLayout,entries:[{binding:0,resource:{buffer:r.stencil.anchors}},{binding:2,resource:{buffer:r.faces}},{binding:4,resource:{buffer:r.physics}},{binding:5,resource:this.fields.velocity},{binding:6,resource:this.fields.volume}]});
  this.commitGroup=uniformDetailGroup(d,{layout:this.commitLayout,entries:[{binding:0,resource:{buffer:r.stencil.anchors}},{binding:1,resource:{buffer:r.faces}},{binding:2,resource:this.fields.velocity}]});
 }
 encode(encoder:GPUCommandEncoder):void{
  const r=this.resources;
  if(!r||!this.initPipeline||!this.blockPipeline||!this.commitPipeline||!this.vertexPipeline||!this.vertexGroup||!this.initGroup||!this.commitGroup)throw new Error('Tensor drag was not prepared');
  const V=r.stencil.topology.vertices,anchors=Math.ceil(r.stencil.anchorCapacity/64);
  const state=encoder.beginComputePass({label:'Cloth tensor drag control state'});state.setPipeline(this.vertexPipeline);state.setBindGroup(0,this.vertexGroup);state.dispatchWorkgroups(Math.ceil(V/64));state.end();
  this.material.encode(encoder);
  const first=encoder.beginComputePass({label:'Cloth native tensor support'});first.setPipeline(this.initPipeline);first.setBindGroup(0,this.fields.ownership.bindGroup);first.setBindGroup(1,this.initGroup.group);if(this.fields.solid)first.setBindGroup(2,this.fields.solid.tileGroup);first.dispatchWorkgroups(anchors);first.end();
  // Colors run in order (one dispatch each; dispatches are ordered).
  const pass=encoder.beginComputePass({label:'Cloth implicit SPD drag colors'});pass.setPipeline(this.blockPipeline);
  this.groups.forEach((group,c)=>{pass.setBindGroup(0,group);pass.dispatchWorkgroupsIndirect(r.stencil.colorDispatch,12*c);});pass.end();
  const last=encoder.beginComputePass({label:'Cloth tensor drag publication'});last.setPipeline(this.commitPipeline);last.setBindGroup(0,this.fields.ownership.bindGroup);last.setBindGroup(1,this.commitGroup.group);last.dispatchWorkgroups(anchors);last.end();
 }
 private release():void{for(const b of this.buffers)b.destroy();this.buffers=[];this.groups=[];this.resources=undefined;this.vertexGroup=undefined;this.initGroup=undefined;this.commitGroup=undefined;}
 destroy():void{this.material.destroy();this.release();}
}
