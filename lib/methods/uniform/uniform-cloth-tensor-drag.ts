import type { PreparedUniformClothStencil, UniformClothGridFields, UniformClothStep } from './uniform-cloth-grid-coupling';
import { uniformDetailBindLayout,uniformDetailGroup,uniformDetailModule,uniformDetailPipeline,type UniformDetailGroup } from './uniform-detail-fields';
import { UniformClothDragMaterial } from './uniform-cloth-drag-material';
import { writeBufferChunked } from './uniform-chunked-upload';
import { uniformMixedTopologyWGSL } from './uniform-mixed-topology.wgsl';
import { uniformMixedSolidWGSL } from './uniform-mixed-solid.wgsl';

interface Shared {anchors:GPUBuffer;links:GPUBuffer;occupied:GPUBuffer;impulses:GPUBuffer;status:GPUBuffer}
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
 private initPipeline?:GPUComputePipeline;
 private blockPipeline?:GPUComputePipeline;
 private commitPipeline?:GPUComputePipeline;
 private initGroup?:UniformDetailGroup;
 private commitGroup?:UniformDetailGroup;
 private groups:{group:GPUBindGroup;count:number}[]=[];
 private buffers:GPUBuffer[]=[];
 private count=0;
 private material:UniformClothDragMaterial;
 constructor(private readonly fields:UniformClothGridFields){
  this.material=new UniformClothDragMaterial(fields);
  const d=this.device=fields.device,b=(binding:number,type:GPUBufferBindingType):GPUBindGroupLayoutEntry=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type}});
  const texture=(binding:number):GPUBindGroupLayoutEntry=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float',viewDimension:'3d'}});
  this.initLayout=uniformDetailBindLayout(d,{entries:[b(0,'read-only-storage'),b(2,'storage'),b(4,'uniform'),texture(5),texture(6)]});
  this.blockLayout=d.createBindGroupLayout({entries:[b(0,'storage'),b(1,'read-only-storage'),b(2,'storage'),b(3,'storage'),b(4,'storage'),b(5,'storage'),b(6,'uniform')]});
  this.commitLayout=uniformDetailBindLayout(d,{entries:[b(0,'read-only-storage'),b(1,'read-only-storage'),{binding:2,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:'write-only',format:'rgba32float',viewDimension:'3d'}}]});
 }
 async initialize():Promise<void>{
  await this.material.initialize();
  const d=this.device,topology=uniformMixedTopologyWGSL(this.fields.ownership.capacity,0),anchor='struct Anchor {p:vec3i,first:u32,counts:vec3u,unused:u32};',face='struct Face {velocity:vec4f,mass:vec4f};';
  const solid=this.fields.solid;
  const init=uniformDetailModule(d,{label:'Cloth tensor drag native MAC mass',code:topology+uniformMixedSolidWGSL(solid?2:undefined,solid?.coarse?.count)+anchor+face+/* wgsl */`
@group(1) @binding(0) var<storage,read> anchors:array<Anchor>;
@group(1) @binding(2) var<storage,read_write> faces:array<Face>;
@group(1) @binding(4) var<uniform> physics:vec4f;
@group(1) @binding(5) var velocity:texture_3d<f32>;
@group(1) @binding(6) var volume:texture_3d<f32>;
fn supported(p:vec3i)->bool{let tile=512u*umBlockOf(p>>vec3u(5u))+umTileLocal(vec3u(p>>vec3u(2u)));return umKeyStored(tile)&&umTileWidth(tile)==1u;}
@compute @workgroup_size(64) fn initializeFaces(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=arrayLength(&anchors)){return;}var a=anchors[i];a.p-=32*umFramePage();var result=Face(vec4f(0.0),vec4f(0.0));
 if(!supported(a.p)){faces[i]=result;return;}result.velocity=/*c*/textureLoad(velocity,a.p,0);
 for(var axis=0u;axis<3u;axis++){
  var q=a.p;q[axis]+=1;if(!supported(q)){continue;}
  if(umPressureFaceV(a.p,axis)<=0.0){continue;}
  let wet=clamp(0.5*(/*c*/textureLoad(volume,a.p,0).x+/*c*/textureLoad(volume,q,0).x),0.0,1.0);
  result.mass[axis]=physics.x*physics.y*wet;
 }faces[i]=result;
}`});
  const block=d.createShaderModule({label:'Cloth full tensor implicit block',code:face+/* wgsl */`
struct Vertex {velocityMass:vec4f,linear0:vec4f,linear1:vec4f,linear2:vec4f,nonlinear0:vec4f,nonlinear1:vec4f,nonlinear2:vec4f,extra:vec4f};
struct Parameters {dispatch:vec4u,extra:vec4u};
@group(0) @binding(0) var<storage,read_write> vertices:array<Vertex>;
@group(0) @binding(1) var<storage,read> graph:array<u32>;
@group(0) @binding(2) var<storage,read_write> faces:array<Face>;
@group(0) @binding(3) var<storage,read_write> links:array<vec4f>;
@group(0) @binding(4) var<storage,read_write> impulses:array<f32>;
@group(0) @binding(5) var<storage,read_write> status:array<atomic<u32>>;
@group(0) @binding(6) var<uniform> params:Parameters;
fn linkAt(v:u32,axis:u32,j:u32)->u32{return graph[params.dispatch.z+graph[6u*v+2u*axis]+j];}
fn faceAt(link:u32)->u32{return graph[params.dispatch.w+link];}
fn weightAt(link:u32)->f32{return bitcast<f32>(graph[params.extra.x+link]);}
fn inverse3(m:mat3x3f)->mat3x3f{let a=cross(m[1],m[2]);let b=cross(m[2],m[0]);let c=cross(m[0],m[1]);return transpose(mat3x3f(a,b,c))*(1.0/dot(m[0],a));}
@compute @workgroup_size(64) fn exchangeBlock(@builtin(global_invocation_id) gid:vec3u){
 if(gid.x>=params.dispatch.y){return;}let v=graph[params.dispatch.x+gid.x];let vertex=vertices[v];
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
 let dt=bitcast<f32>(params.extra.y);var A=inverse3(normalized)*(1.0/(scale*dt));
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
@group(1) @binding(0) var<storage,read> anchors:array<Anchor>;
@group(1) @binding(1) var<storage,read> faces:array<Face>;
@group(1) @binding(2) var output:texture_storage_3d<rgba32float,write>;
@compute @workgroup_size(64) fn commitFaces(@builtin(global_invocation_id) gid:vec3u){let i=gid.x;if(i>=arrayLength(&anchors)){return;}let p=anchors[i].p-32*umFramePage();let tile=512u*umBlockOf(p>>vec3u(5u))+umTileLocal(vec3u(p>>vec3u(2u)));if(umKeyStored(tile)&&umTileWidth(tile)==1u){textureStore(output,p,faces[i].velocity);}}
`});
  for(const [name,module] of [['mass',init],['block',block],['commit',commit]] as const){const errors=(await module.getCompilationInfo()).messages.filter(x=>x.type==='error');if(errors.length)throw new Error(errors.map(x=>`Cloth tensor ${name} ${x.lineNum}:${x.linePos}: ${x.message}`).join('\n'));}
  [this.initPipeline,this.blockPipeline,this.commitPipeline]=await Promise.all([
   uniformDetailPipeline(d,this.fields.ownership,{layout:d.createPipelineLayout({bindGroupLayouts:[this.fields.ownership.bindLayout,this.initLayout,...(solid?[solid.tileLayout]:[])]}),compute:{module:init,entryPoint:'initializeFaces'}}),
   d.createComputePipelineAsync({layout:d.createPipelineLayout({bindGroupLayouts:[this.blockLayout]}),compute:{module:block,entryPoint:'exchangeBlock'}}),
   uniformDetailPipeline(d,this.fields.ownership,{layout:d.createPipelineLayout({bindGroupLayouts:[this.fields.ownership.bindLayout,this.commitLayout]}),compute:{module:commit,entryPoint:'commitFaces'}}),
  ]);
 }
 prepare(p:PreparedUniformClothStencil,step:UniformClothStep,shared:Shared):void{
  this.release();const d=this.device,create=(size:number,data?:ArrayBufferView,usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST)=>{const b=d.createBuffer({label:'Cloth tensor block data',size:Math.max(16,size),usage});this.buffers.push(b);if(data)writeBufferChunked(d.queue,b,0,data);return b;};
  const vertexData=new Float32Array(32*p.vertices),anchorColors:number[][]=Array.from({length:p.coordinates.length},()=>[]),colors:number[][]=[];
  // Greedy coloring: a vertex takes the least color no anchor it touches holds.
  const anchorStamp=new Int32Array(p.coordinates.length).fill(-1),usedStamp:number[]=[],anchors:number[]=[];
  for(let v=0;v<p.vertices;v++){
   anchors.length=0;for(let axis=0;axis<3;axis++)for(let j=0;j<p.ranges[6*v+2*axis+1]!;j++){const a=Math.floor(p.linkFaces[p.gather[p.ranges[6*v+2*axis]!+j]!]!/3);if(anchorStamp[a]!==v){anchorStamp[a]=v;anchors.push(a);}}
   for(const a of anchors)for(const used of anchorColors[a]!)usedStamp[used]=v;let color=0;while(usedStamp[color]===v)color++;(colors[color]??=[]).push(v);for(const a of anchors)anchorColors[a]!.push(color);
   const mass=(step.physics?.solveSolid===false||(((step.kinematicFixed??step.data.fixed)?.[v]??0)&1))?0:step.state.dryMass_kg[v]!+p.density_kg_m3*step.state.waterVolume_m3[v]!;
   const predictor=step.solidPredictorVelocity_m_s??step.state.velocities_m_s;
   vertexData.set([predictor[3*v]!,predictor[3*v+1]!,predictor[3*v+2]!,mass],32*v);
   // Material kernels fill all six tensor rows from actual deposited grid
   // packing and native free-water volume before any block is solved.
   vertexData[32*v+28]=p.exponent;
  }
  const indices=6*p.vertices,faceOffset=indices+p.gather.length,weightOffset=faceOffset+p.linkFaces.length,colorOffset=weightOffset+p.linkWeights.length,graph=new Uint32Array(colorOffset+p.vertices);
  graph.set(p.ranges);graph.set(p.gather,indices);graph.set(p.linkFaces,faceOffset);graph.set(new Uint32Array(p.linkWeights.buffer),weightOffset);
  let offset=colorOffset;for(const color of colors){graph.set(color,offset);offset+=color.length;}
  const vertices=create(vertexData.byteLength,vertexData),graphBuffer=create(graph.byteLength,graph),faces=create(32*p.coordinates.length),physics=create(16,new Float32Array([p.density_kg_m3,this.fields.cellSize_m.reduce((a,b)=>a*b,1),0,0]),GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
  this.material.prepare(p,step,vertices);
  this.initGroup=uniformDetailGroup(d,{layout:this.initLayout,entries:[{binding:0,resource:{buffer:shared.anchors}},{binding:2,resource:{buffer:faces}},{binding:4,resource:{buffer:physics}},{binding:5,resource:this.fields.velocity},{binding:6,resource:this.fields.volume}]});
  this.commitGroup=uniformDetailGroup(d,{layout:this.commitLayout,entries:[{binding:0,resource:{buffer:shared.anchors}},{binding:1,resource:{buffer:faces}},{binding:2,resource:this.fields.velocity}]});
  offset=colorOffset;
  for(const color of colors){const words=new Uint32Array([offset,color.length,indices,faceOffset,weightOffset,0,0,0]);new Float32Array(words.buffer)[5]=p.dt_s;const params=create(32,words,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
   this.groups.push({count:color.length,group:d.createBindGroup({layout:this.blockLayout,entries:[vertices,graphBuffer,faces,shared.links,shared.impulses,shared.status,params].map((buffer,binding)=>({binding,resource:{buffer}}))})});offset+=color.length;
  }this.count=p.coordinates.length;
 }
 encode(encoder:GPUCommandEncoder):void{
  if(!this.initPipeline||!this.blockPipeline||!this.commitPipeline||!this.initGroup||!this.commitGroup)throw new Error('Tensor drag was not prepared');
  this.material.encode(encoder);
  const first=encoder.beginComputePass({label:'Cloth native tensor support'});first.setPipeline(this.initPipeline);first.setBindGroup(0,this.fields.ownership.bindGroup);first.setBindGroup(1,this.initGroup.group);if(this.fields.solid)first.setBindGroup(2,this.fields.solid.tileGroup);first.dispatchWorkgroups(Math.ceil(this.count/64));first.end();
  for(const color of this.groups){const pass=encoder.beginComputePass({label:'Cloth implicit SPD drag color'});pass.setPipeline(this.blockPipeline);pass.setBindGroup(0,color.group);pass.dispatchWorkgroups(Math.ceil(color.count/64));pass.end();}
  const last=encoder.beginComputePass({label:'Cloth tensor drag publication'});last.setPipeline(this.commitPipeline);last.setBindGroup(0,this.fields.ownership.bindGroup);last.setBindGroup(1,this.commitGroup.group);last.dispatchWorkgroups(Math.ceil(this.count/64));last.end();
 }
 private release():void{for(const b of this.buffers)b.destroy();this.buffers=[];this.groups=[];}
 destroy():void{this.material.destroy();this.release();}
}
