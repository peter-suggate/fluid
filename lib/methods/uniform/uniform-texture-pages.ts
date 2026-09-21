import {uniformPressurePageExtent,uniformPressurePageAddressWGSL,rewritePressureTextureCalls} from "./uniform-pressure-pages";
import {gpuCompilationManagerFor} from "../../core/gpu-compilation-manager";

type Dims=readonly [number,number,number];
type Field={dims:Dims;paged:boolean};
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
 constructor(private readonly device:GPUDevice, private readonly pagedStorage=true){}
 createTexture(descriptor:GPUTextureDescriptor):GPUTexture{
  const size=descriptor.size as GPUExtent3DDict;
  const dims:Dims='width' in Object(size)?[size.width,size.height??1,size.depthOrArrayLayers??1]:[...descriptor.size as Iterable<number>] as unknown as Dims;
  const paged=this.pagedStorage && descriptor.dimension==='3d';
  const extent=paged?uniformPressurePageExtent(dims):dims;
  if(extent.some(n=>n>this.device.limits.maxTextureDimension3D)&&paged)throw new RangeError(`Uniform page atlas exceeds texture limits: ${extent}`);
  const texture=this.device.createTexture({...descriptor,size:extent});this.fields.set(texture,{dims,paged});return texture;
 }
 view(texture:GPUTexture):GPUTextureView{
  const view=texture.createView();this.views.set(view,this.fields.get(texture)??{dims:[texture.width,texture.height,texture.depthOrArrayLayers],paged:false});return view;
 }
 layout(entries:readonly GPUBindGroupLayoutEntry[]):GPUBindGroupLayoutEntry[]{
  return [...entries,{binding:BINDING,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}}];
 }
 createBindGroup(descriptor:GPUBindGroupDescriptor):GPUBindGroup{
  const entries=[...descriptor.entries],data=new Uint32Array(36*4);
  for(const entry of entries){const field=this.views.get(entry.resource as GPUTextureView);if(field)data.set([...field.dims,Number(field.paged)],4*entry.binding);}
  const buffer=this.device.createBuffer({label:'Uniform field page layouts',size:data.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  this.device.queue.writeBuffer(buffer,0,data);this.uniforms.push(buffer);
  return this.device.createBindGroup({...descriptor,entries:[...entries,{binding:BINDING,resource:{buffer}}]});
 }
 shader(source:string, fixedFields:ReadonlyMap<number,GPUTexture>=new Map()):string{
  const fields=new Map<string,{binding:number;type:string}>();
  for(const m of source.matchAll(/@group\(0\)\s*@binding\((\d+)\)\s*var\s+(\w+):\s*(texture(?:_storage)?_3d[^;]+);/g))fields.set(m[2]!,{binding:Number(m[1]),type:m[3]!});
  const vector=(d:Dims)=>`vec3u(${d.map(n=>`${n}u`).join(',')})`;
  const fixed=(binding:number)=>{const texture=fixedFields.get(binding);return texture && this.fields.get(texture);};
  let code=source;
  for(const [name,{binding}] of fields)code=code.replace(new RegExp(`textureDimensions\\(${name}(?:,\\s*0)?\\)`,'g'),`uniformFieldPages[${binding}].xyz`);
  code=rewritePressureTextureCalls(code,fields);
  let helpers=`\n@group(0) @binding(${BINDING}) var<uniform> uniformFieldPages:array<vec4u,36>;\n${uniformPressurePageAddressWGSL.replaceAll('mgPageAddress','uniformFieldPageAddress')+uniformPressurePageAddressWGSL.replaceAll('mgPageAddress','uniformFieldPageAddressUnchecked').replace(' if(any(p<vec3i(0))||any(p>=vec3i(d))){return vec3i(-1);}','')}\n`;
  for(const [name,{binding,type}] of fields){
   const kind=type.includes('u32')||type.includes('uint')?'u':'f';
   const metadata=fixed(binding),texture=fixedFields.get(binding);
   const at=metadata && texture
    ? `var at=p;if(uniformFieldPages[${binding}].w!=0u){at=uniformFieldPageAddress(p,${vector(metadata.dims)},${vector(uniformPressurePageExtent(metadata.dims))});}`
    : `var at=p;if(uniformFieldPages[${binding}].w!=0u){at=uniformFieldPageAddress(p,uniformFieldPages[${binding}].xyz,textureDimensions(${name}));}`;
   // Both level-set samplers clamp their taps before loading. Keeping that
   // invariant explicit avoids an extra branch inside iterative redistancing.
   const clampedPhi=name==='uvPhiIn'||name==='phi';
   const address=clampedPhi?at.replace('uniformFieldPageAddress(', 'uniformFieldPageAddressUnchecked('):at;
   const guard=clampedPhi?'':`if(any(at<vec3i(0))){return vec4${kind}(0);}`;
   helpers+=type.startsWith('texture_storage')
    ? `fn ${name}Store(p:vec3i,value:vec4${kind}){${at} if(any(at<vec3i(0))){return;}textureStore(${name},at,value);}\n`
    : `fn ${name}Load(p:vec3i)->vec4${kind}{${address} ${guard}return textureLoad(${name},at,0);}\n`;
  }
  return code+helpers;
 }
 upload(texture:GPUTexture,values:Float32Array):void{
  const field=this.fields.get(texture);if(!field)throw new Error('Upload requires a registered field');
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
  if(source.width!==destination.width||source.height!==destination.height||source.depthOrArrayLayers!==destination.depthOrArrayLayers)throw new Error('Incompatible uniform page copies');
  encoder.copyTextureToTexture({texture:source},{texture:destination},[source.width,source.height,source.depthOrArrayLayers]);
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
  return bytes;
 }
 destroy():void{for(const buffer of this.uniforms)buffer.destroy();for(const p of this.publications.values())p.texture.destroy();}
}
