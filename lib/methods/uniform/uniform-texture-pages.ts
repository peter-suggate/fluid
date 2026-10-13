import {UNIFORM_FIELD_LOOP_BOUNDS,UNIFORM_FIELD_LOOP_METADATA,uniformFieldRuntimeLoops} from "./uniform-field-loop-bounds";
import {rewritePressureTextureCalls} from "./uniform-pressure-pages";
import {gpuCompilationManagerFor} from "../../core/gpu-compilation-manager";
import {UniformScratchArena,uniformScratchAccessWGSL} from "./uniform-scratch-arena";
import {uniformDetailField} from "./uniform-detail-fields";

type Dims=readonly [number,number,number];
type Field={dims:Dims;scratchOffset?:number;components?:number};
const BINDING=34;
/** Registry of the uniform fields' logical dimensions and scratch-arena
 * placement, and the shader adapter that addresses them. Logical dimensions
 * describe authored physical boundaries; they are never inferred from a texture's size. */
export class UniformTexturePages {
 private readonly fields=new Map<GPUTexture,Field>();
 private readonly views=new WeakMap<GPUTextureView,Field>();
 private readonly uniforms:GPUBuffer[]=[];
 private readonly copyPipelines=new Map<GPUTextureFormat,GPUComputePipeline>();
 private readonly copyGroups=new Map<GPUTexture,Map<GPUTexture,GPUBindGroup>>();
 private readonly snapshots=new Map<GPUTexture,GPUTexture>();
 constructor(private readonly device:GPUDevice, readonly scratch?:UniformScratchArena){}
 scratchMetadata(texture:GPUTexture):number {
  const offset=this.fields.get(texture)?.scratchOffset;
  return offset===undefined?0:(0x80000000|offset|(texture.format.startsWith('rgba')?3:0))>>>0;
 }
 private viewScratchMetadata(view:GPUTextureView):number {
  const f=this.views.get(view)!;return (0x80000000|f.scratchOffset!|((f.components??1)-1))>>>0;
 }
 createTexture(descriptor:GPUTextureDescriptor):GPUTexture{
  const size=descriptor.size as GPUExtent3DDict;
  const dims:Dims='width' in Object(size)?[size.width,size.height??1,size.depthOrArrayLayers??1]:[...descriptor.size as Iterable<number>] as unknown as Dims;
  const scratchOffset=this.scratch?.offset(descriptor.label??'');
  const texture=this.device.createTexture({...descriptor,size:scratchOffset===undefined?dims:[1,1,1],
   usage:descriptor.usage|(scratchOffset===undefined?0:GPUTextureUsage.TEXTURE_BINDING)});
  this.fields.set(texture,{dims,scratchOffset,components:descriptor.format.startsWith('rgba')?4:1});return texture;
 }
 /** Register a native texture made elsewhere (a detail-storage field) by its
  * logical dims: rewritten textureDimensions report these, not its size. */
 adopt(texture:GPUTexture,dims:Dims):GPUTexture{
  this.fields.set(texture,{dims,components:texture.format.startsWith('rgba')?4:1});return texture;
 }
 /** A packed detail field binds its 1³ stand-in: the dense kernels of these
  * groups never run on one (its texels are the mixed stages'). `physical`:
  * its texels, for a kernel compiled through uniformDetailDenseShader. */
 view(texture:GPUTexture,physical=false):GPUTextureView{
  const storage=uniformDetailField(texture)?.storage;
  const view=(storage?(physical?storage.physical(texture):storage.stub(texture)):texture).createView();this.views.set(view,this.fields.get(texture)??{dims:[texture.width,texture.height,texture.depthOrArrayLayers]});return view;
 }
 layout(entries:readonly GPUBindGroupLayoutEntry[],scratch=true):GPUBindGroupLayoutEntry[]{
  return [...entries,{binding:BINDING,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},
   ...(this.scratch&&scratch?[{binding:35,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage' as const}}]:[])];
 }
 createBindGroup(descriptor:GPUBindGroupDescriptor,scratch=true,scratchBytes?:number):GPUBindGroup{
  const entries=[...descriptor.entries],data=new Uint32Array(36*4);
  data.set(UNIFORM_FIELD_LOOP_BOUNDS,UNIFORM_FIELD_LOOP_METADATA*4);
  for(const entry of entries){const field=this.views.get(entry.resource as GPUTextureView);if(field)data.set([...field.dims,field.scratchOffset===undefined?0:this.viewScratchMetadata(entry.resource as GPUTextureView)],4*entry.binding);}
  const buffer=this.device.createBuffer({label:'Uniform field page layouts',size:data.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  this.device.queue.writeBuffer(buffer,0,data);this.uniforms.push(buffer);
  return this.device.createBindGroup({...descriptor,entries:[...entries,{binding:BINDING,resource:{buffer}},
   ...(this.scratch&&scratch?[{binding:35,resource:{buffer:this.scratch.buffer,...(scratchBytes===undefined?{}:{size:scratchBytes})}}]:[])]});
 }
 shader(source:string, fixedFields:ReadonlyMap<number,GPUTexture>=new Map(), literalLoops=false, nativeBindings:ReadonlySet<number>=new Set()):string{
  const scratch=!!this.scratch;
  const atomic=source.includes('var<storage,read_write> band:');
  const fields=new Map<string,{binding:number;type:string}>();
  for(const m of source.matchAll(/@group\(0\)\s*@binding\((\d+)\)\s*var\s+(\w+):\s*(texture(?:_storage)?_3d[^;]+);/g))fields.set(m[2]!,{binding:Number(m[1]),type:m[3]!});
  const fixed=(binding:number)=>{const texture=fixedFields.get(binding);return texture && this.fields.get(texture);};
  let code=source;
  for(const [name,{binding}] of fields)code=code.replace(new RegExp(`textureDimensions\\(${name}(?:,\\s*0)?\\)`,'g'),`uniformFieldPages[${binding}].xyz`);
  // Only a field that may sit in the scratch arena goes through its accessor.
  if(scratch)code=rewritePressureTextureCalls(code,new Map([...fields].filter(([,field])=>!nativeBindings.has(field.binding))));
  let helpers=`\n@group(0) @binding(${BINDING}) var<uniform> uniformFieldPages:array<vec4u,36>;\n`;
  if(scratch)helpers+=`\n@group(0) @binding(35) var<storage,read_write> uniformScratch:array<${atomic?'atomic<u32>':'u32'}>;\n`;
  for(const [name,{binding,type}] of fields){
   const kind=type.includes('u32')||type.includes('uint')?'u':'f';
   // Both level-set samplers clamp their taps before loading. Keeping that
   // invariant explicit avoids an extra branch inside iterative redistancing.
   const guard=name==='uvPhiIn'||name==='phi'?'':`if(any(at<vec3i(0))){return vec4${kind}(0);}`;
   const backing=scratch&&!nativeBindings.has(binding)?uniformScratchAccessWGSL(name,`uniformFieldPages[${binding}]`,type,atomic,fixed(binding)?.components):'';
   helpers+=type.startsWith('texture_storage')
    ? `fn ${name}Store(p:vec3i,value:vec4${kind}){${backing}let at=p; if(any(at<vec3i(0))){return;}textureStore(${name},at,value);}\n`
    : `fn ${name}Load(p:vec3i)->vec4${kind}{${backing}let at=p; ${guard}return textureLoad(${name},at,0);}\n`;
  }
  return (literalLoops?code:uniformFieldRuntimeLoops(code))+helpers;
 }
 upload(texture:GPUTexture,values:Float32Array):void{
  const field=this.fields.get(texture);if(!field)throw new Error('Upload requires a registered field');
  if(field.scratchOffset!==undefined){
   const offset=field.scratchOffset*4,buffer=this.scratch!.buffer;
   if(offset+values.byteLength>buffer.size)throw new Error(`Scratch upload ${texture.label}: ${offset}+${values.byteLength}>${buffer.size}`);
   // Keep initialization staging bounded even for a 256³ field.
   const chunkWords=2*1024*1024;
   for(let begin=0;begin<values.length;begin+=chunkWords)
    this.device.queue.writeBuffer(buffer,offset+begin*4,values.subarray(begin,begin+chunkWords) as Float32Array<ArrayBuffer>);
   return;
  }
  this.device.queue.writeTexture({texture},values as Float32Array<ArrayBuffer>,{bytesPerRow:texture.width*(texture.format==='rgba32float'?16:4),rowsPerImage:texture.height},[...field.dims]);
 }
 copy(encoder:GPUCommandEncoder,source:GPUTexture,destination:GPUTexture):void{
  const src=this.fields.get(source),dst=this.fields.get(destination);
  if(src?.scratchOffset!==undefined||dst?.scratchOffset!==undefined){
   const dims=dst?.dims??[destination.width,destination.height,destination.depthOrArrayLayers];
   const pipeline=this.copyPipelines.get(destination.format);if(!pipeline)throw new Error('Scratch copy pipeline is not initialized');
   let targets=this.copyGroups.get(source);if(!targets){targets=new Map();this.copyGroups.set(source,targets);}
   let group=targets.get(destination);if(!group){group=this.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[
    {binding:0,resource:this.view(source)},{binding:1,resource:this.view(destination)}]});targets.set(destination,group);}
   const pass=encoder.beginComputePass({label:'Uniform scratch field copy'});pass.setPipeline(pipeline);pass.setBindGroup(0,group);
   pass.dispatchWorkgroups(...dims.map(n=>Math.ceil(n/4)) as [number,number,number]);pass.end();return;
  }
  // Detail fields of one class share a physical layout: the whole texture copies.
  const storage=uniformDetailField(source)?.storage;
  // Two fields of one storage: their bases copy with them.
  if(storage&&storage===uniformDetailField(destination)?.storage){storage.copy(encoder,source,destination);return;}
  const s=storage?.physical(source)??source,d=uniformDetailField(destination)?.storage.physical(destination)??destination;
  if(s.width!==d.width||s.height!==d.height||s.depthOrArrayLayers!==d.depthOrArrayLayers)throw new Error('Incompatible uniform page copies');
  encoder.copyTextureToTexture({texture:s},{texture:d},[s.width,s.height,s.depthOrArrayLayers]);
 }
 /** Opt-in snapshots are captured at their owning stage. Normal simulation
  * need not retain intermediate fields after use. */
 snapshotTexture(field:GPUTexture):GPUTexture {
  if(this.fields.get(field)?.scratchOffset===undefined)return field;
  let texture=this.snapshots.get(field);if(texture)return texture;
  const dims=this.fields.get(field)!.dims;
  texture=this.device.createTexture({label:`${field.label} QA snapshot`,size:dims,dimension:'3d',format:field.format,
   usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});
  this.fields.set(texture,{dims});this.snapshots.set(field,texture);return texture;
 }
 encodeSnapshot(encoder:GPUCommandEncoder,field:GPUTexture):void{
  const target=this.snapshots.get(field);if(target)this.copy(encoder,field,target);
 }
 async initialize(signal?:AbortSignal):Promise<void>{
  const compiler=gpuCompilationManagerFor(this.device);
  if(this.scratch)for(const format of ['r32float','rgba32float'] as const){
   const source=`@group(0) @binding(0) var source:texture_3d<f32>;
    @group(0) @binding(1) var output:texture_storage_3d<${format},write>;
    @compute @workgroup_size(4,4,4) fn copyField(@builtin(global_invocation_id)g:vec3u){
      if(any(g>=textureDimensions(output))){return;}textureStore(output,vec3i(g),textureLoad(source,vec3i(g),0));}`;
   const module=compiler.createShaderModule({label:'Uniform scratch copy',code:this.shader(source)});
   const layout=this.device.createBindGroupLayout({entries:this.layout([
    {binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float',viewDimension:'3d'}},
    {binding:1,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:'write-only',format,viewDimension:'3d'}}])});
   this.copyPipelines.set(format,await compiler.compileComputePipeline({label:'Uniform scratch copy',layout:this.device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint:'copyField'}},{priority:'visible',signal}));
  }
 }
 get allocationOverheadBytes():number{
  let bytes=this.uniforms.reduce((sum,b)=>sum+b.size,0);
  for(const [texture,field] of this.fields){
   const components=texture.format.startsWith('rgba')?4:1;
   // A detail field's bytes are its storage's (UniformDetailStorage.allocatedBytes): none here.
   const texels=uniformDetailField(texture)?0:texture.width*texture.height*texture.depthOrArrayLayers;
   bytes+=(texels-field.dims[0]*field.dims[1]*field.dims[2])*components*4;
  }
  for(const texture of this.snapshots.values())bytes+=texture.width*texture.height*texture.depthOrArrayLayers*(texture.format.startsWith('rgba')?4:1)*4;
  return bytes+(this.scratch?.buffer.size??0);
 }
 destroy():void{for(const buffer of this.uniforms)buffer.destroy();for(const t of this.snapshots.values())t.destroy();}
}
