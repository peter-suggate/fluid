import type {PreparedUniformClothStencil,UniformClothGridFields,UniformClothStep} from './uniform-cloth-grid-coupling';
import {clothCellKey,clothCellWeights} from './uniform-porous-capacity';
import type {UniformClothGPUCells} from './uniform-cloth-gpu-cells';
import {porousDragCoefficients} from '../../cloth/porous-material';
import {uniformDetailBindLayout,uniformDetailGroup,uniformDetailModule,uniformDetailPipeline,type UniformDetailGroup} from './uniform-detail-fields';
import {uniformMixedTopologyWGSL} from './uniform-mixed-topology.wgsl';

export interface GridDragCell {p:readonly number[];fiberVolume_m3:number;boundVolume_m3:number;linear:Float64Array;nonlinear:Float64Array;members:readonly {vertex:number;share:number}[]}
/** Grid constitutive quadrature: psi is deposited fiber / native cell volume,
 * not rest textile packing. Partition C*Vm among controls by their deposited
 * fiber contribution, so a constant relative velocity integrates each grid
 * mixture exactly once. The SPD block solve still uses an adjoint stencil. */
export function prepareGridDragMaterial(fields:Pick<UniformClothGridFields,'origin_m'|'cellSize_m'>,p:PreparedUniformClothStencil,step:UniformClothStep):GridDragCell[]{
 const V=fields.cellSize_m.reduce((a,b)=>a*b,1),s=step.state;
 const cells=new Map<number,{p:readonly number[];fiber:number;bound:number;weight:number;diameter:number;viscosity:number;alpha:Float64Array;members:{vertex:number;volume:number}[]}>();
 for(let v=0;v<p.vertices;v++)for(const {p:at,weight} of clothCellWeights(fields,s.positions_m.subarray(3*v,3*v+3))){
  const key=clothCellKey(at[0],at[1],at[2]),cell=cells.get(key)??{p:at,fiber:0,bound:0,weight:0,diameter:0,viscosity:0,alpha:new Float64Array(9),members:[]},volume=weight*s.solidVolume_m3[v]!;
  cell.fiber+=volume;cell.bound+=weight*s.waterVolume_m3[v]!;cell.weight+=weight;cell.diameter+=weight*p.vertexMicro[2*v]!;cell.viscosity+=weight*p.vertexMicro[2*v+1]!;
  for(let k=0;k<9;k++)cell.alpha[k]!+=weight*((k%4===0?2:0)-p.poreTensor[9*v+k]!);
  cell.members.push({vertex:v,volume});cells.set(key,cell);
 }
 return [...cells.values()].filter(c=>c.fiber>0).map(c=>{
  const psi=c.fiber/V,material={...step.materials[0]!,fiberDiameter_m:c.diameter/c.weight,waterViscosity_Pa_s:c.viscosity/c.weight};
  if(!(psi>0&&psi<1))throw new Error('Grid drag fiber fraction is outside (0,1)');
  const a=porousDragCoefficients(psi,0,material,false,p.exponent),b=porousDragCoefficients(psi,1,material,(step.physics?.nonlinearDrag??step.nonlinearDrag)!==false,p.exponent);
  const tensor=(alpha:number,beta:number)=>Float64Array.from(c.alpha,(x,k)=>x/c.weight*(alpha-beta)+(k%4===0?beta:0));
  return {p:c.p,fiberVolume_m3:c.fiber,boundVolume_m3:c.bound,linear:tensor(a.alongFiber_kg_m3_s,a.acrossFiber_kg_m3_s),nonlinear:tensor(b.alongFiber_kg_m3_s-a.alongFiber_kg_m3_s,b.acrossFiber_kg_m3_s-a.acrossFiber_kg_m3_s),members:c.members.map(m=>({vertex:m.vertex,share:m.volume/c.fiber}))};
 });
}

export class UniformClothDragMaterial {
 private cellLayout:GPUBindGroupLayout;private vertexLayout:GPUBindGroupLayout;
 private cellPipeline?:GPUComputePipeline;private vertexPipeline?:GPUComputePipeline;
 private cellGroup?:UniformDetailGroup;private vertexGroup?:GPUBindGroup;
 private bound?:{cells:UniformClothGPUCells;result:GPUBuffer;params:GPUBuffer};
 private buffers:GPUBuffer[]=[];private cells=0;private vertices=0;
 constructor(private fields:UniformClothGridFields){const d=fields.device,b=(binding:number,type:GPUBufferBindingType):GPUBindGroupLayoutEntry=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type}});
  this.cellLayout=uniformDetailBindLayout(d,{entries:[b(0,'read-only-storage'),{binding:1,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float',viewDimension:'3d'}},b(2,'storage'),b(3,'uniform')]});
  this.vertexLayout=d.createBindGroupLayout({entries:[b(0,'read-only-storage'),b(2,'read-only-storage'),b(3,'storage')]});
 }
 async initialize():Promise<void>{const d=this.fields.device,material='struct Material {p:vec4i,volume:vec4f,linear0:vec4f,linear1:vec4f,linear2:vec4f,nonlinear0:vec4f,nonlinear1:vec4f,nonlinear2:vec4f};';
  const cells=uniformDetailModule(d,{label:'Cloth actual grid-mixture drag constitutive volume',code:uniformMixedTopologyWGSL(this.fields.ownership.capacity,0)+material+/* wgsl */`
@group(1) @binding(0) var<storage,read> source:array<Material>;
@group(1) @binding(1) var volume:texture_3d<f32>;
@group(1) @binding(2) var<storage,read_write> result:array<Material>;
@group(1) @binding(3) var<uniform> params:vec4f;
@compute @workgroup_size(64) fn mixture(@builtin(global_invocation_id) gid:vec3u){let i=gid.x;if(i>=arrayLength(&source)){return;}var m=source[i];
 // Fiber-free and unused records carry no drag; no share references them.
 if(m.p.w==0){return;}
 let free=max(0.0,/*c*/textureLoad(volume,m.p.xyz-32*umFramePage(),0).x)*params.x;
 let Vm=select(0.0,free+m.volume.x+m.volume.y,free>0.0);m.linear0*=Vm;m.linear1*=Vm;m.linear2*=Vm;m.nonlinear0*=Vm;m.nonlinear1*=Vm;m.nonlinear2*=Vm;result[i]=m;}
`});
  const vertices=d.createShaderModule({label:'Cloth conservative grid constitutive lumping',code:material+/* wgsl */`
struct Vertex {velocityMass:vec4f,linear0:vec4f,linear1:vec4f,linear2:vec4f,nonlinear0:vec4f,nonlinear1:vec4f,nonlinear2:vec4f,extra:vec4f};
@group(0) @binding(0) var<storage,read> cells:array<Material>;
// Eight corner shares a vertex, in corner order; an absent corner is 0xffffffff.
@group(0) @binding(2) var<storage,read> links:array<vec2u>;
@group(0) @binding(3) var<storage,read_write> vertices:array<Vertex>;
@compute @workgroup_size(64) fn lump(@builtin(global_invocation_id) gid:vec3u){let i=gid.x;if(i>=arrayLength(&vertices)){return;}var v=vertices[i];
 v.linear0=vec4f(0.0);v.linear1=vec4f(0.0);v.linear2=vec4f(0.0);v.nonlinear0=vec4f(0.0);v.nonlinear1=vec4f(0.0);v.nonlinear2=vec4f(0.0);
 for(var j=0u;j<8u;j++){let link=links[8u*i+j];if(link.x==0xffffffffu){continue;}let m=cells[link.x];let w=bitcast<f32>(link.y);v.linear0+=w*m.linear0;v.linear1+=w*m.linear1;v.linear2+=w*m.linear2;v.nonlinear0+=w*m.nonlinear0;v.nonlinear1+=w*m.nonlinear1;v.nonlinear2+=w*m.nonlinear2;}
 vertices[i]=v;}
`});
  for(const module of [cells,vertices]){const errors=(await module.getCompilationInfo()).messages.filter(x=>x.type==='error');if(errors.length)throw new Error(errors.map(x=>`${x.lineNum}:${x.linePos} ${x.message}`).join('\n'));}
  [this.cellPipeline,this.vertexPipeline]=await Promise.all([uniformDetailPipeline(d,this.fields.ownership,{layout:d.createPipelineLayout({bindGroupLayouts:[this.fields.ownership.bindLayout,this.cellLayout]}),compute:{module:cells,entryPoint:'mixture'}}),d.createComputePipelineAsync({layout:d.createPipelineLayout({bindGroupLayouts:[this.vertexLayout]}),compute:{module:vertices,entryPoint:'lump'}})]);
 }
 /** Bind one topology's GPU cell tables (built on the GPU each step). */
 attach(cells:UniformClothGPUCells,vertices:GPUBuffer,count:number):void{
  this.release();const d=this.fields.device;
  const result=d.createBuffer({label:'Cloth grid constitutive quadrature',size:cells.dragCells.size,usage:GPUBufferUsage.STORAGE}),params=d.createBuffer({label:'Cloth grid constitutive volume',size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  this.buffers.push(result,params);d.queue.writeBuffer(params,0,new Float32Array([this.fields.cellSize_m.reduce((a,b)=>a*b,1),0,0,0]));
  this.vertexGroup=d.createBindGroup({layout:this.vertexLayout,entries:[{binding:0,resource:{buffer:result}},{binding:2,resource:{buffer:cells.dragLinks}},{binding:3,resource:{buffer:vertices}}]});
  this.bound={cells,result,params};this.cells=cells.sections.entries;this.vertices=count;
 }
 /** The native-field group at the fields' current textures. */
 prepare():void{
  const b=this.bound;if(!b)throw new Error('Grid drag material has no attached cell tables');
  this.cellGroup=uniformDetailGroup(this.fields.device,{layout:this.cellLayout,entries:[{binding:0,resource:{buffer:b.cells.dragCells}},{binding:1,resource:this.fields.volume},{binding:2,resource:{buffer:b.result}},{binding:3,resource:{buffer:b.params}}]});
 }
 encode(encoder:GPUCommandEncoder):void{if(!this.cellPipeline||!this.vertexPipeline||!this.cellGroup||!this.vertexGroup)throw new Error('Grid drag material is not prepared');
  const first=encoder.beginComputePass({label:'Cloth physical grid mixture coefficients'});first.setPipeline(this.cellPipeline);first.setBindGroup(0,this.fields.ownership.bindGroup);first.setBindGroup(1,this.cellGroup.group);first.dispatchWorkgroups(Math.ceil(this.cells/64));first.end();
  const second=encoder.beginComputePass({label:'Cloth conservative constitutive assembly'});second.setPipeline(this.vertexPipeline);second.setBindGroup(0,this.vertexGroup);second.dispatchWorkgroups(Math.ceil(this.vertices/64));second.end();
 }
 private release():void{for(const b of this.buffers)b.destroy();this.buffers=[];this.bound=undefined;this.cellGroup=undefined;this.vertexGroup=undefined;}
 destroy():void{this.release();}
}
