import type {UniformMixedLayout} from './uniform-mixed-layout';
import type {UniformMixedBandBits} from './uniform-mixed-layout-builder';

/** Current h surface crossings, independent of bulk ownership. Encode after
 * phi advection/correction and before building pressure ownership. Each tile
 * examines every h vertex, including detail invisible at its 4h corners.
 * The mask is rebuilt once per pressure solve; it stores no connectivity. */
export class UniformPressureSurfaceBand {
 readonly band:UniformMixedBandBits;
 readonly allocatedBytes:number;
 private readonly resources:GPUBindGroupLayout;
 private readonly group:GPUBindGroup;
 private pipeline?:GPUComputePipeline;
 constructor(private readonly device:GPUDevice,private readonly layout:UniformMixedLayout,phi:GPUTexture){
  const d=layout.lattice.dimensions;
  if(phi.format!=='r32float'||[phi.width,phi.height,phi.depthOrArrayLayers].some((n,a)=>n!==d[a]!+1))throw new Error('Pressure surface band requires the full h vertex field');
  const buffer=device.createBuffer({label:'Uniform independent pressure surface band',size:Math.ceil(layout.tiles.length/32)*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  this.band={buffer,wordOffset:0};this.allocatedBytes=buffer.size;
  this.resources=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float',viewDimension:'3d'}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}}]});
  this.group=device.createBindGroup({layout:this.resources,entries:[{binding:0,resource:phi.createView()},{binding:1,resource:{buffer}}]});
 }
 async initialize():Promise<void>{
  const t=this.layout.tileDimensions;
  const module=this.device.createShaderModule({code:/* wgsl */`
@group(0) @binding(0) var phi:texture_3d<f32>;
@group(0) @binding(1) var<storage,read_write> band:array<atomic<u32>>;
const T=vec3u(${t.map(n=>`${n}u`).join(',')});
var<workgroup> signs:array<u32,64>;
@compute @workgroup_size(64) fn classify(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=group.x+${this.device.limits.maxComputeWorkgroupsPerDimension}u*group.y;if(tile>=${this.layout.tiles.length}u){return;}
 let base=vec3u(tile%T.x,(tile/T.x)%T.y,tile/(T.x*T.y))*4u;
 var flags=0u;
 for(var i=lane;i<125u;i+=64u){
  let p=base+vec3u(i%5u,(i/5u)%5u,i/25u);let v=textureLoad(phi,vec3i(p),0).x;
  // Zero belongs to the surface. Nonfinite data must never silently remove
  // pressure resolution; the solve's numerical checks report it separately.
  if(!(abs(v)<=3.0e38)||v==0.0){flags|=3u;}else{flags|=select(2u,1u,v<0.0);}
 }
 signs[lane]=flags;workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){if(lane<stride){signs[lane]|=signs[lane+stride];}workgroupBarrier();}
 if(lane==0u&&signs[0]==3u){atomicOr(&band[tile/32u],1u<<(tile%32u));}
}`});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');if(errors.length)throw new Error(errors.map(e=>e.message).join('\n'));
  this.pipeline=await this.device.createComputePipelineAsync({layout:this.device.createPipelineLayout({bindGroupLayouts:[this.resources]}),compute:{module,entryPoint:'classify'}});
 }
 encode(encoder:GPUCommandEncoder):void{
  if(!this.pipeline)throw new Error('Pressure surface band is not initialized');
  encoder.clearBuffer(this.band.buffer);
  const pass=encoder.beginComputePass({label:'Uniform independent h pressure surface census'});pass.setPipeline(this.pipeline);pass.setBindGroup(0,this.group);
  const n=this.layout.tiles.length,x=this.device.limits.maxComputeWorkgroupsPerDimension;pass.dispatchWorkgroups(Math.min(n,x),Math.ceil(n/x));pass.end();
 }
 destroy():void{this.band.buffer.destroy();}
}
