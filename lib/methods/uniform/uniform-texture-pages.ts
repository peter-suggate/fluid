import {UNIFORM_FIELD_LOOP_BOUNDS,UNIFORM_FIELD_LOOP_METADATA,uniformFieldRuntimeLoops} from "./uniform-field-loop-bounds";
import {uniformPressurePageExtent,uniformPressurePageAddressWGSL,rewritePressureTextureCalls} from "./uniform-pressure-pages";
import {gpuCompilationManagerFor} from "../../core/gpu-compilation-manager";
import {UniformScratchArena,uniformScratchAccessWGSL,uniformVolumeScratchShader} from "./uniform-scratch-arena";

type Dims=readonly [number,number,number];
type Field={dims:Dims;paged:boolean;scratchOffset?:number;components?:number};
const BINDING=34;
/** Shared physical page storage for uniform fields. Logical dimensions describe
 * authored physical boundaries; they are never inferred from the texture atlas. */
export class UniformTexturePages {
 private readonly fields=new Map<GPUTexture,Field>();
 private readonly views=new WeakMap<GPUTextureView,Field>();
 private readonly uniforms:GPUBuffer[]=[];
 private readonly publications=new Map<GPUTexture,{texture:GPUTexture;group?:GPUBindGroup}>();
 private initialized=false;
 private readonly publishPipelines=new Map<GPUTextureFormat,GPUComputePipeline>();
 private readonly copyPipelines=new Map<GPUTextureFormat,GPUComputePipeline>();
 private readonly copyGroups=new Map<GPUTexture,Map<GPUTexture,GPUBindGroup>>();
 private readonly snapshots=new Map<GPUTexture,GPUTexture>();
 constructor(private readonly device:GPUDevice, private readonly pagedStorage=true, readonly scratch?:UniformScratchArena){}
 /** All fields and hierarchy levels have native storage in this generation. */
 get nativeStorage():boolean{return !this.pagedStorage;}
 scratchMetadata(texture:GPUTexture):number {
  const offset=this.fields.get(texture)?.scratchOffset;
  return offset===undefined?0:(0x80000000|offset|(texture.format.startsWith('rgba')?3:0))>>>0;
 }
 private viewScratchMetadata(view:GPUTextureView):number {
  const f=this.views.get(view)!;return (0x80000000|f.scratchOffset!|((f.components??1)-1))>>>0;
 }
 createTexture(descriptor:GPUTextureDescriptor, native=false):GPUTexture{
  const size=descriptor.size as GPUExtent3DDict;
  const dims:Dims='width' in Object(size)?[size.width,size.height??1,size.depthOrArrayLayers??1]:[...descriptor.size as Iterable<number>] as unknown as Dims;
  const paged=this.pagedStorage && !native && descriptor.dimension==='3d';
  const extent=paged?uniformPressurePageExtent(dims):dims;
  if(extent.some(n=>n>this.device.limits.maxTextureDimension3D)&&paged)throw new RangeError(`Uniform page atlas exceeds texture limits: ${extent}`);
  const scratchOffset=this.scratch?.offset(descriptor.label??'');
  const texture=this.device.createTexture({...descriptor,size:scratchOffset===undefined?extent:[1,1,1],
   usage:descriptor.usage|(scratchOffset===undefined?0:GPUTextureUsage.TEXTURE_BINDING)});
  this.fields.set(texture,{dims,paged,scratchOffset,components:descriptor.format.startsWith('rgba')?4:1});return texture;
 }
 createTextureLike(reference:GPUTexture,descriptor:GPUTextureDescriptor):GPUTexture{
  const field=this.fields.get(reference);
  if(!field)throw new Error("Texture template must be registered");
  return this.createTexture(descriptor,!field.paged);
 }
 view(texture:GPUTexture):GPUTextureView{
  const view=texture.createView();this.views.set(view,this.fields.get(texture)??{dims:[texture.width,texture.height,texture.depthOrArrayLayers],paged:false});return view;
 }
 layout(entries:readonly GPUBindGroupLayoutEntry[],scratch=true):GPUBindGroupLayoutEntry[]{
  return [...entries,{binding:BINDING,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},
   ...(this.scratch&&scratch?[{binding:35,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage' as const}}]:[])];
 }
 createBindGroup(descriptor:GPUBindGroupDescriptor,scratch=true,scratchBytes?:number):GPUBindGroup{
  const entries=[...descriptor.entries],data=new Uint32Array(36*4);
  data.set(UNIFORM_FIELD_LOOP_BOUNDS,UNIFORM_FIELD_LOOP_METADATA*4);
  for(const entry of entries){const field=this.views.get(entry.resource as GPUTextureView);if(field)data.set([...field.dims,field.scratchOffset===undefined?Number(field.paged):this.viewScratchMetadata(entry.resource as GPUTextureView)],4*entry.binding);}
  const buffer=this.device.createBuffer({label:'Uniform field page layouts',size:data.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  this.device.queue.writeBuffer(buffer,0,data);this.uniforms.push(buffer);
  return this.device.createBindGroup({...descriptor,entries:[...entries,{binding:BINDING,resource:{buffer}},
   ...(this.scratch&&scratch?[{binding:35,resource:{buffer:this.scratch.buffer,...(scratchBytes===undefined?{}:{size:scratchBytes})}}]:[])]});
 }
 shader(source:string, fixedFields:ReadonlyMap<number,GPUTexture>=new Map(), auditPageReads=false, literalLoops=false, directNativeFields=false, nativeBindings:ReadonlySet<number>=new Set()):string{
  const scratch=!!this.scratch;
  const atomic=source.includes('var<storage,read_write> band:');
  if(scratch)source=source.replace('fn tileTableBase() -> u32 { let d = tileDims(); return u32(d.x * d.y * d.z); }','fn tileTableBase() -> u32 { return 0u; }');
  if(scratch&&source.includes('struct UVEdges'))source=uniformVolumeScratchShader(source,this.scratch!.sharpenBaseWords,this.scratch!.conditioningBytes);
  const fields=new Map<string,{binding:number;type:string}>();
  for(const m of source.matchAll(/@group\(0\)\s*@binding\((\d+)\)\s*var\s+(\w+):\s*(texture(?:_storage)?_3d[^;]+);/g))fields.set(m[2]!,{binding:Number(m[1]),type:m[3]!});
  const vector=(d:Dims)=>`vec3u(${d.map(n=>`${n}u`).join(',')})`;
  const fixed=(binding:number)=>{const texture=fixedFields.get(binding);return texture && this.fields.get(texture);};
  if(auditPageReads && (!source.includes("fn pageDomainContainsSample(") || !source.includes("var<storage,read_write> reductions:")))
   throw new Error("Page read auditing requires accepted membership and the root diagnostic buffer");
  let code=source;
  for(const [name,{binding}] of fields)code=code.replace(new RegExp(`textureDimensions\\(${name}(?:,\\s*0)?\\)`,'g'),`uniformFieldPages[${binding}].xyz`);
  const translatedFields=!scratch&&(directNativeFields || this.nativeStorage) && !auditPageReads
    ? new Map([...fields].filter(([,field])=>!this.nativeStorage && fixed(field.binding)?.paged !== false)) : new Map([...fields].filter(([,field])=>auditPageReads || !this.nativeStorage || !nativeBindings.has(field.binding)));
  code=rewritePressureTextureCalls(code,translatedFields);
  let helpers=`\n@group(0) @binding(${BINDING}) var<uniform> uniformFieldPages:array<vec4u,36>;\n${uniformPressurePageAddressWGSL.replaceAll('mgPageAddress','uniformFieldPageAddress')+uniformPressurePageAddressWGSL.replaceAll('mgPageAddress','uniformFieldPageAddressUnchecked').replace(' if(any(p<vec3i(0))||any(p>=vec3i(d))){return vec3i(-1);}','')}\n`;
  if(scratch)helpers+=`\n@group(0) @binding(35) var<storage,read_write> uniformScratch:array<${atomic?'atomic<u32>':'u32'}>;\n`;
  for(const [name,{binding,type}] of fields){
   const kind=type.includes('u32')||type.includes('uint')?'u':'f';
   const metadata=fixed(binding),texture=fixedFields.get(binding);
   const at=nativeBindings.has(binding) && this.nativeStorage ? `let at=p;` : metadata && texture
    ? (metadata.paged ? `let at=uniformFieldPageAddress(p,${vector(metadata.dims)},${vector(uniformPressurePageExtent(metadata.dims))});` : `let at=p;`)
    : `var at=p;if(uniformFieldPages[${binding}].w!=0u){at=uniformFieldPageAddress(p,uniformFieldPages[${binding}].xyz,textureDimensions(${name}));}`;
   // Both level-set samplers clamp their taps before loading. Keeping that
   // invariant explicit avoids an extra branch inside iterative redistancing.
   const clampedPhi=name==='uvPhiIn'||name==='phi';
   const address=clampedPhi?at.replace('uniformFieldPageAddress(', 'uniformFieldPageAddressUnchecked('):at;
   const guard=clampedPhi?'':`if(any(at<vec3i(0))){return vec4${kind}(0);}`;
   // Audit actual loads, including all interpolation taps. This records a
   // dependency fault; rollback/ambient substitution belongs to the operator
   // transaction, not the texture-address adapter.
   const audit=auditPageReads?`if(!pageDomainContainsSample(p,uniformFieldPages[${binding}].xyz)){atomicOr(&reductions[8],1u<<${binding}u);atomicAdd(&reductions[9],1u);}`:'';
   const backing=scratch&&!nativeBindings.has(binding)?uniformScratchAccessWGSL(name,`uniformFieldPages[${binding}]`,type,atomic,metadata?.components ?? (source.includes('var primaryIn:') ? (binding===1?1:4) : undefined)):'';
   helpers+=type.startsWith('texture_storage')
    ? `fn ${name}Store(p:vec3i,value:vec4${kind}){${backing}${at} if(any(at<vec3i(0))){return;}textureStore(${name},at,value);}\n`
    : `fn ${name}Load(p:vec3i)->vec4${kind}{${backing}${audit}${address} ${guard}return textureLoad(${name},at,0);}\n`;
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
  if(!field.paged){this.device.queue.writeTexture({texture},values as Float32Array<ArrayBuffer>,{bytesPerRow:texture.width*(texture.format==='rgba32float'?16:4),rowsPerImage:texture.height},[...field.dims]);return;}
  const [nx,ny,nz]=field.dims,components=texture.format==='rgba32float'?4:1;
  const row=Math.ceil(texture.width*components*4/256)*256/4;
  const data=new Float32Array(row*texture.height*texture.depthOrArrayLayers);
  const gx=Math.ceil(nx/16),gy=Math.ceil(ny/16),ax=texture.width/16,ay=texture.height/16;
  for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
   const page=Math.floor(x/16)+gx*(Math.floor(y/16)+gy*Math.floor(z/16));
   const px=x%16+16*(page%ax),py=y%16+16*(Math.floor(page/ax)%ay),pz=z%16+16*Math.floor(page/(ax*ay));
   const src=(x+nx*(y+ny*z))*components,dst=px*components+row*(py+texture.height*pz);
   for(let c=0;c<components;c++)data[dst+c]=values[src+c]!;
  }
  this.device.queue.writeTexture({texture},data,{bytesPerRow:row*4,rowsPerImage:texture.height},[texture.width,texture.height,texture.depthOrArrayLayers]);
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
  if(source.width!==destination.width||source.height!==destination.height||source.depthOrArrayLayers!==destination.depthOrArrayLayers)throw new Error('Incompatible uniform page copies');
  encoder.copyTextureToTexture({texture:source},{texture:destination},[source.width,source.height,source.depthOrArrayLayers]);
 }
 /** Opt-in snapshots are captured at their owning stage. Normal simulation
  * need not retain intermediate fields after use. */
 snapshotTexture(field:GPUTexture):GPUTexture {
  if(this.fields.get(field)?.scratchOffset===undefined)return this.publication(field);
  let texture=this.snapshots.get(field);if(texture)return texture;
  const dims=this.fields.get(field)!.dims;
  texture=this.device.createTexture({label:`${field.label} QA snapshot`,size:dims,dimension:'3d',format:field.format,
   usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});
  this.fields.set(texture,{dims,paged:false});this.snapshots.set(field,texture);return texture;
 }
 encodeSnapshot(encoder:GPUCommandEncoder,field:GPUTexture):void{
  const target=this.snapshots.get(field);if(target)this.copy(encoder,field,target);
 }
 publication(field:GPUTexture):GPUTexture{
  if(!this.fields.get(field)?.paged)return field;
  let result=this.publications.get(field);if(result)return result.texture;
  if(this.initialized)throw new Error("Page publications must be registered before initialization");
  const dims=this.fields.get(field)!.dims;
  const texture=this.device.createTexture({label:`${field.label} presentation`,size:[...dims],dimension:'3d',format:field.format,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});
  result={texture};this.publications.set(field,result);return texture;
 }
 async initialize(signal?:AbortSignal):Promise<void>{
  const compiler=gpuCompilationManagerFor(this.device);
  this.initialized=true;
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
  for(const format of new Set([...this.publications.keys()].map(t=>t.format))){
   const kind=format.includes('uint')?'u':'f';
   const module=compiler.createShaderModule({label:'Uniform page presentation',code:`
    @group(0) @binding(0) var source:texture_3d<${kind}32>;
    @group(0) @binding(1) var published:texture_storage_3d<${format},write>;
    ${uniformPressurePageAddressWGSL}
    @compute @workgroup_size(4,4,4) fn publish(@builtin(global_invocation_id)g:vec3u){
     let d=textureDimensions(published);if(any(g>=d)){return;}
     textureStore(published,vec3i(g),textureLoad(source,mgPageAddress(vec3i(g),d,textureDimensions(source)),0));
    }`});
   this.publishPipelines.set(format,await compiler.compileComputePipeline({label:'Uniform page presentation',layout:'auto',compute:{module,entryPoint:'publish'}},{priority:'visible',signal}));
  }
  for(const [field,publication] of this.publications){publication.group=this.device.createBindGroup({layout:this.publishPipelines.get(field.format)!.getBindGroupLayout(0),entries:[{binding:0,resource:field.createView()},{binding:1,resource:publication.texture.createView()}]});}
 }
 encodePublications(encoder:GPUCommandEncoder):void{
  for(const [field,publication] of this.publications){
   const pass=encoder.beginComputePass({label:'Publish uniform page field'});pass.setPipeline(this.publishPipelines.get(field.format)!);pass.setBindGroup(0,publication.group!);
   const d=this.fields.get(field)!.dims;pass.dispatchWorkgroups(...d.map(n=>Math.ceil(n/4)) as [number,number,number]);pass.end();
  }
 }
 get allocationOverheadBytes():number{
  let bytes=this.uniforms.reduce((sum,b)=>sum+b.size,0);
  for(const [texture,field] of this.fields){
   const components=texture.format.startsWith('rgba')?4:1;
   bytes+=(texture.width*texture.height*texture.depthOrArrayLayers-field.dims[0]*field.dims[1]*field.dims[2])*components*4;
  }
  for(const {texture} of this.publications.values())bytes+=texture.width*texture.height*texture.depthOrArrayLayers*(texture.format.startsWith('rgba')?4:1)*4;
  for(const texture of this.snapshots.values())bytes+=texture.width*texture.height*texture.depthOrArrayLayers*(texture.format.startsWith('rgba')?4:1)*4;
  return bytes+(this.scratch?.buffer.size??0);
 }
 destroy():void{for(const buffer of this.uniforms)buffer.destroy();for(const p of this.publications.values())p.texture.destroy();for(const t of this.snapshots.values())t.destroy();}
}
