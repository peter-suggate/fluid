import type {PreparedUniformClothStencil,UniformClothGridFields,UniformClothStep} from './uniform-cloth-grid-coupling';
import {uniformDetailBindLayout,uniformDetailGroup,uniformDetailModule,uniformDetailPipeline,type UniformDetailGroup} from './uniform-detail-fields';
import {clothCellWeights} from './uniform-porous-capacity';
import {uniformMixedTopologyWGSL} from './uniform-mixed-topology.wgsl';
import {uniformMixedSolidWGSL} from './uniform-mixed-solid.wgsl';

interface PoreCell {p:number[];solid:number;bound:number;weight:number;hardOpen:number;coefficient:number;tensor:Float64Array}
/** Actual Eulerian pore-pressure force (Fei 2018, Eq.9; upstream
 * computePorePressureGrads). Saturation includes BOTH native free water and
 * bound water. The full material tensor acts on the staggered pressure
 * gradient; submerged saturated material therefore has zero suction. */
export class UniformClothPorePressure {
 private readonly device:GPUDevice;
 private readonly layout:GPUBindGroupLayout;
 private readonly commitLayout:GPUBindGroupLayout;
 private pipeline?:GPUComputePipeline;
 private commitPipeline?:GPUComputePipeline;
 private group?:UniformDetailGroup;
 private commitGroup?:UniformDetailGroup;
 private texture?:GPUTexture;
 private buffers:GPUBuffer[]=[];
 private count=0;
 constructor(private readonly fields:UniformClothGridFields){
  const d=this.device=fields.device,b=(binding:number,type:GPUBufferBindingType):GPUBindGroupLayoutEntry=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type}});
  const texture=(binding:number,viewDimension:GPUTextureViewDimension):GPUBindGroupLayoutEntry=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float',viewDimension}});
  this.layout=uniformDetailBindLayout(d,{entries:[b(0,'read-only-storage'),texture(1,'2d'),texture(2,'3d'),texture(3,'3d'),b(4,'storage'),b(5,'uniform')]});
  this.commitLayout=uniformDetailBindLayout(d,{entries:[b(0,'read-only-storage'),b(1,'read-only-storage'),{binding:2,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:'write-only',format:'rgba32float',viewDimension:'3d'}}]});
 }
 async initialize():Promise<void>{
  const d=this.device,topology=uniformMixedTopologyWGSL(this.fields.ownership.capacity,0),anchor='struct Anchor {p:vec3i,first:u32,counts:vec3u,unused:u32};',solid=this.fields.solid;
  const force=uniformDetailModule(d,{label:'Cloth native liquid pore-pressure gradient',code:topology+uniformMixedSolidWGSL(solid?2:undefined,solid?.coarse?.count)+anchor+/* wgsl */`
@group(1) @binding(0) var<storage,read> anchors:array<Anchor>;
@group(1) @binding(1) var material:texture_2d<f32>;
@group(1) @binding(2) var volume:texture_3d<f32>;
@group(1) @binding(3) var velocity:texture_3d<f32>;
@group(1) @binding(4) var<storage,read_write> result:array<vec4f>;
@group(1) @binding(5) var<uniform> params:vec4f;
fn word(i:u32)->vec4f{let width=textureDimensions(material).x;return textureLoad(material,vec2i(i32(i%width),i32(i/width)),0);}
fn less(a:vec3i,b:vec3i)->bool{return a.z<b.z||(a.z==b.z&&(a.y<b.y||(a.y==b.y&&a.x<b.x)));}
fn lookup(local:vec3i)->u32{let p=local+32*umFramePage();let count=bitcast<u32>(word(0u).x);var low=0u;var high=count;
 loop{if(low>=high){break;}let mid=(low+high)/2u;if(less(bitcast<vec4i>(word(1u+5u*mid)).xyz,p)){low=mid+1u;}else{high=mid;}}
 if(low<count&&all(bitcast<vec4i>(word(1u+5u*low)).xyz==p)){return low;}return 0xffffffffu;
}
fn saturation(p:vec3i)->f32{let i=lookup(p);if(i==0xffffffffu){return clamp(/*c*/textureLoad(volume,p,0).x,0.0,1.0);}let m=word(2u+5u*i);return clamp((/*c*/textureLoad(volume,p,0).x/max(1e-20,m.w)+m.z)/max(1e-20,1.0-m.y),0.0,1.0);}
fn pressure(p:vec3i)->f32{let i=lookup(p);if(i==0xffffffffu){return 0.0;}let m=word(2u+5u*i);return m.x*m.y/max(1e-20,1.0-m.y)*(1.0-saturation(p));}
fn tensor(p:vec3i)->mat3x3f{let i=lookup(p);if(i==0xffffffffu){return mat3x3f(vec3f(0.0),vec3f(0.0),vec3f(0.0));}return mat3x3f(word(3u+5u*i).xyz,word(4u+5u*i).xyz,word(5u+5u*i).xyz);}
@compute @workgroup_size(64) fn poreForce(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=arrayLength(&anchors)){return;}let p=anchors[i].p-32*umFramePage();var value=/*c*/textureLoad(velocity,p,0);
 for(var axis=0u;axis<3u;axis++){
  if(anchors[i].counts[axis]==0u){continue;}var q=p;q[axis]+=1;
  let aperture=umPressureFaceV(p,axis);if(aperture<=0.0){continue;}
  let wet=.5*(/*c*/textureLoad(volume,p,0).x+/*c*/textureLoad(volume,q,0).x);if(wet<=0.0){continue;}
  // Discrete divergence of the tensor stress, including changing fiber
  // orientation. B*grad(p) alone drops the p*div(B) contribution.
  var force=(pressure(q)*tensor(q)[axis][axis]-pressure(p)*tensor(p)[axis][axis])/params[axis];
  for(var cross=0u;cross<3u;cross++){if(cross==axis){continue;}var delta=vec3i(0);delta[cross]=1;
   force+=(pressure(p+delta)*tensor(p+delta)[cross][axis]+pressure(q+delta)*tensor(q+delta)[cross][axis]
    -pressure(p-delta)*tensor(p-delta)[cross][axis]-pressure(q-delta)*tensor(q-delta)[cross][axis])/(4.0*params[cross]);}
  value[axis]+=params.w*aperture*.5*(saturation(p)+saturation(q))*force;
 }result[i]=value;
}`});
  const commit=uniformDetailModule(d,{label:'Cloth pore-pressure force commit',code:topology+anchor+/* wgsl */`
@group(1) @binding(0) var<storage,read> anchors:array<Anchor>;
@group(1) @binding(1) var<storage,read> result:array<vec4f>;
@group(1) @binding(2) var output:texture_storage_3d<rgba32float,write>;
@compute @workgroup_size(64) fn commit(@builtin(global_invocation_id) gid:vec3u){let i=gid.x;if(i<arrayLength(&anchors)){textureStore(output,anchors[i].p-32*umFramePage(),result[i]);}}
`});
  for(const module of [force,commit]){const errors=(await module.getCompilationInfo()).messages.filter(x=>x.type==='error');if(errors.length)throw new Error(errors.map(x=>`${x.lineNum}:${x.linePos} ${x.message}`).join('\n'));}
  [this.pipeline,this.commitPipeline]=await Promise.all([
   uniformDetailPipeline(d,this.fields.ownership,{layout:d.createPipelineLayout({bindGroupLayouts:[this.fields.ownership.bindLayout,this.layout,...(solid?[solid.tileLayout]:[])]}),compute:{module:force,entryPoint:'poreForce'}}),
   uniformDetailPipeline(d,this.fields.ownership,{layout:d.createPipelineLayout({bindGroupLayouts:[this.fields.ownership.bindLayout,this.commitLayout]}),compute:{module:commit,entryPoint:'commit'}}),
  ]);
 }
 prepare(p:PreparedUniformClothStencil,step:UniformClothStep,anchors:GPUBuffer):void{
  if(step.physics?.porePressure)throw new Error('Solid pore-pressure stress is not yet implemented; source water scenes disable it');
  this.release();this.count=p.coordinates.length;
  const h=this.fields.cellSize_m,V=h.reduce((a,b)=>a*b,1),cells=new Map<string,PoreCell>(),s=step.state;
  for(let v=0;v<s.waterVolume_m3.length;v++){
   for(const {p:at,weight,porosityWeight,hardOpen} of clothCellWeights(this.fields,s.positions_m.subarray(3*v,3*v+3))){
    const key=at.join(','),cell=cells.get(key)??{p:at,solid:0,bound:0,weight:0,hardOpen,coefficient:0,tensor:new Float64Array(9)},w=weight*s.controlVolume_m3[v]!;
    cell.solid+=porosityWeight*s.solidVolume_m3[v]!/V;cell.bound+=porosityWeight*s.waterVolume_m3[v]!/V;cell.weight+=w;cell.coefficient+=w*p.poreCoefficient_Pa[v]!;
    for(let k=0;k<9;k++)cell.tensor[k]!+=w*p.poreTensor[9*v+k]!;cells.set(key,cell);
   }
  }
  const ordered=[...cells.values()].sort((a,b)=>a.p[2]!-b.p[2]!||a.p[1]!-b.p[1]!||a.p[0]!-b.p[0]!),width=Math.min(1024,this.device.limits.maxTextureDimension2D),height=Math.ceil((1+5*ordered.length)/width);
  if(height>this.device.limits.maxTextureDimension2D)throw new Error('Pore-pressure support exceeds device capacity');
  const values=new Float32Array(4*width*height),bits=new Int32Array(values.buffer);bits[0]=ordered.length;
  ordered.forEach((cell,i)=>{bits.set([...cell.p,0],4*(1+5*i));values.set([cell.coefficient/cell.weight,cell.solid,cell.bound,cell.hardOpen],4*(2+5*i));for(let a=0;a<3;a++)values.set([cell.tensor[3*a]!/cell.weight,cell.tensor[3*a+1]!/cell.weight,cell.tensor[3*a+2]!/cell.weight,0],4*(3+a+5*i));});
  this.texture=this.device.createTexture({label:'Cloth pore-pressure material',size:[width,height],format:'rgba32float',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
  this.device.queue.writeTexture({texture:this.texture},values,{bytesPerRow:16*width,rowsPerImage:height},[width,height]);
  const result=this.device.createBuffer({label:'Cloth pore-pressure native velocities',size:16*this.count,usage:GPUBufferUsage.STORAGE}),params=this.device.createBuffer({label:'Cloth pore-pressure parameters',size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});this.buffers.push(result,params);
  this.device.queue.writeBuffer(params,0,new Float32Array([...h,step.dt_s/p.density_kg_m3]));
  this.group=uniformDetailGroup(this.device,{layout:this.layout,entries:[{binding:0,resource:{buffer:anchors}},{binding:1,resource:this.texture.createView()},{binding:2,resource:this.fields.volume},{binding:3,resource:this.fields.velocity},{binding:4,resource:{buffer:result}},{binding:5,resource:{buffer:params}}]});
  this.commitGroup=uniformDetailGroup(this.device,{layout:this.commitLayout,entries:[{binding:0,resource:{buffer:anchors}},{binding:1,resource:{buffer:result}},{binding:2,resource:this.fields.velocity}]});
 }
 encode(encoder:GPUCommandEncoder):void{
  if(!this.pipeline||!this.commitPipeline||!this.group||!this.commitGroup)throw new Error('Pore-pressure force was not prepared');
  for(const [pipeline,group] of [[this.pipeline,this.group],[this.commitPipeline,this.commitGroup]] as const){const pass=encoder.beginComputePass({label:'Cloth Eulerian capillary force'});pass.setPipeline(pipeline);pass.setBindGroup(0,this.fields.ownership.bindGroup);pass.setBindGroup(1,group.group);if(pipeline===this.pipeline&&this.fields.solid)pass.setBindGroup(2,this.fields.solid.tileGroup);pass.dispatchWorkgroups(Math.ceil(this.count/64));pass.end();}
 }
 private release():void{this.texture?.destroy();this.texture=undefined;for(const b of this.buffers)b.destroy();this.buffers=[];}
 destroy():void{this.release();}
}
