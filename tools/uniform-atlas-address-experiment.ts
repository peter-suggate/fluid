/** Probe-only full-occupancy texture atlas. Not a sparse allocator or new solver.
 * Logical h fields are permuted in B^3 pages; vertices on outer domain planes
 * stay at their native addresses. Numerical operators and owner IDs are intact.
 * 'table' uses GPU page translations; 'affine' computes the same mirror layout;
 * 'dense' specializes to native addressing. All physical allocations stay dense.
 */
import assert from "node:assert/strict";
import {rewritePressureTextureCalls} from "../lib/methods/uniform/uniform-pressure-pages";

type Dims=readonly [number,number,number];
export type AtlasExperimentMode="dense"|"table"|"affine";
export class UniformAtlasAddressExperiment {
 readonly offsets:Int32Array<ArrayBuffer>;
 readonly counts={modules:0,fields:0,uploads:0,uploadBytes:0,bindGroups:0};
 constructor(readonly dimensions:Dims,readonly edge:16|32,readonly mode:AtlasExperimentMode){
  assert.ok(dimensions.every(n=>Number.isInteger(n)&&n>0&&n%edge===0),"Atlas domain must be a multiple of the page edge");
  const pages=dimensions.map(n=>n/edge);this.offsets=new Int32Array(pages[0]!*pages[1]!*pages[2]!*4);
  for(let z=0;z<pages[2]!;z++)for(let y=0;y<pages[1]!;y++)for(let x=0;x<pages[0]!;x++){
   this.offsets.set([(pages[0]!-1-2*x)*edge,(pages[1]!-1-2*y)*edge,(pages[2]!-1-2*z)*edge,0],4*(x+pages[0]!*(y+pages[1]!*z)));
  }
 }
 matches(d:readonly number[]):boolean{return d.length===3&&d.every((n,a)=>n===this.dimensions[a]||n===this.dimensions[a]!+1);}
 address(p:Dims,d:Dims):Dims{
  if(this.mode==="dense"||!this.matches(d)||p.some((n,a)=>n<0||n>=this.dimensions[a]!))return p;
  const q=p.map(n=>Math.floor(n/this.edge)),g=this.dimensions.map(n=>n/this.edge),i=4*(q[0]!+g[0]!*(q[1]!+g[1]!*q[2]!));
  return p.map((n,a)=>n+this.offsets[i+a]!) as unknown as Dims;
 }
 /** Mirror page placement is involutive, so this packs and unpacks fields. */
 reorder<T extends Uint8Array|Float32Array>(source:T,d:Dims,components:number):T{
  if(this.mode==="dense"||!this.matches(d))return source;
  const result=source.slice() as T;
  // A row inside a page stays contiguous. Copy spans, leaving the outer
  // vertex planes from slice() intact; do not allocate per-voxel vectors.
  for(let z=0;z<this.dimensions[2];z++)for(let y=0;y<this.dimensions[1];y++)for(let x=0;x<this.dimensions[0];x+=this.edge){
   const [a,b,c]=this.address([x,y,z],d),src=components*(x+d[0]*(y+d[1]*z)),dst=components*(a+d[0]*(b+d[1]*c));
   result.set(source.subarray(src,src+components*this.edge),dst);
  }
  return result;
 }
 shader(source:string):string{
  if(this.mode==="dense")return source;
  assert.ok(!source.includes("qaAtlas"),"Atlas source must be transformed once");
  const fields=new Map<string,{type:string}>();
  for(const m of source.matchAll(/var\s+(\w+)\s*:\s*(texture(?:_storage)?_3d\s*<[^;]+);/g))fields.set(m[1]!,{type:m[2]!});
  if(!fields.size)return source;
  // This fixture supports explicit texel loads only; filtered samplers need
  // separately certified page borders and must not silently bypass translation.
  assert.ok(!/textureSample\w*\s*\(/.test(source),"Atlas probe does not support filtered texture access");
  // Rename texture identifiers first so generated wrappers cannot collide
  // with existing phiLoad/volumeStore helpers in the production source.
  let code=source;
  const renamed=new Map<string,{type:string}>();
  for(const [name,value] of fields){
   const target=`qaAtlasBacking_${name}`;
   code=code.replace(new RegExp(`\\b${name}\\b`,"g"),target);renamed.set(target,value);
  }
  code=rewritePressureTextureCalls(code,renamed);
  const vector=(d:Dims)=>`vec3i(${d.join(",")})`;
  const pageAddress=this.mode==="table"
   ? `let page=vec3u(p)/${this.edge}u;let grid=vec3u(D)/${this.edge}u;return p+qaAtlasOffsets[page.x+grid.x*(page.y+grid.y*page.z)].xyz;`
   : `let page=p/${this.edge};return p+D-vec3i(${this.edge})-2*${this.edge}*page;`;
  let helpers=`\n${this.mode==="table"?`@group(0) @binding(63) var<uniform> qaAtlasOffsets:array<vec4i,${this.offsets.length/4}>;`:""}
fn qaAtlasAddress(p:vec3i,d:vec3u)->vec3i{
 const D=${vector(this.dimensions)};
 if(any(vec3i(d)<D)||any(vec3i(d)>D+vec3i(1))||any(p<vec3i(0))||any(p>=D)){return p;}
 ${pageAddress}
}\n`;
  for(const [name,{type}] of renamed){
   const kind=/u32|uint/.test(type)?"u":/i32|sint/.test(type)?"i":"f",storage=type.startsWith("texture_storage");
   if(!storage||/read/.test(type))helpers+=`fn ${name}Load(p:vec3i)->vec4${kind}{return textureLoad(${name},qaAtlasAddress(p,textureDimensions(${name}))${storage?"":",0"});}\n`;
   if(storage&&/write/.test(type))helpers+=`fn ${name}Store(p:vec3i,v:vec4${kind}){textureStore(${name},qaAtlasAddress(p,textureDimensions(${name})),v);}\n`;
  }
  this.counts.modules++;this.counts.fields+=fields.size;
  return code+helpers;
 }
 install(device:GPUDevice):{destroy:()=>void}{
  const patch=(object:object,key:string,value:unknown)=>Object.defineProperty(object,key,{value,configurable:true,writable:true});
  const shader=device.createShaderModule.bind(device),layout=device.createBindGroupLayout.bind(device),bind=device.createBindGroup.bind(device);
  const pipeline=device.createComputePipelineAsync.bind(device),syncPipeline=device.createComputePipeline.bind(device);
  const queue=device.queue,write=queue.writeTexture.bind(queue);
  const augmented=new WeakSet<GPUBindGroupLayout>();
  const metadata=this.mode==="table"?device.createBuffer({label:"Research atlas page translations",size:this.offsets.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}):undefined;
  if(metadata){assert.ok(metadata.size<=device.limits.maxUniformBufferBindingSize);queue.writeBuffer(metadata,0,this.offsets);}
  patch(device,"createShaderModule",(d:GPUShaderModuleDescriptor)=>shader({...d,code:this.shader(d.code)}));
  if(metadata){
   patch(device,"createBindGroupLayout",(d:GPUBindGroupLayoutDescriptor)=>{
    const entries=[...d.entries];assert.ok(!entries.some(e=>e.binding===63));
    const result=layout({...d,entries:[...entries,{binding:63,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]});augmented.add(result);return result;
   });
   patch(device,"createBindGroup",(d:GPUBindGroupDescriptor)=>{
    this.counts.bindGroups++;return bind(augmented.has(d.layout)?{...d,entries:[...d.entries,{binding:63,resource:{buffer:metadata}}]}:d);
   });
   const wrap=(p:GPUComputePipeline,auto:boolean)=>{
    if(auto){const get=p.getBindGroupLayout.bind(p);patch(p,"getBindGroupLayout",(index:number)=>{const l=get(index);if(index===0)augmented.add(l);return l;});}return p;
   };
   // Auto layouts without translated textures do not declare the atlas table.
   const modules=new WeakSet<GPUShaderModule>();
   patch(device,"createShaderModule",(d:GPUShaderModuleDescriptor)=>{const code=this.shader(d.code),m=shader({...d,code});if(code.includes("var<uniform> qaAtlasOffsets"))modules.add(m);return m;});
   patch(device,"createComputePipelineAsync",async(d:GPUComputePipelineDescriptor)=>wrap(await pipeline(d),d.layout==="auto"&&modules.has(d.compute.module)));
   patch(device,"createComputePipeline",(d:GPUComputePipelineDescriptor)=>wrap(syncPipeline(d),d.layout==="auto"&&modules.has(d.compute.module)));
  }
  patch(queue,"writeTexture",(destination:GPUTexelCopyTextureInfo,data:GPUAllowSharedBufferSource,packing:GPUTexelCopyBufferLayout,size:GPUExtent3D)=>{
   const t=destination.texture,d:Dims=[t.width,t.height,t.depthOrArrayLayers];
   if(this.mode==="dense"||!this.matches(d)||t.dimension!=="3d")return write(destination,data,packing,size);
   const extent=Array.isArray(size)?size:[(size as GPUExtent3DDict).width,(size as GPUExtent3DDict).height??1,(size as GPUExtent3DDict).depthOrArrayLayers??1];
   const origin=destination.origin??[0,0,0];assert.ok((Array.isArray(origin)?origin:Object.values(origin)).every(n=>n===0));
   assert.deepEqual(extent,d,"Atlas probe supports whole-field uploads only");assert.equal(destination.mipLevel??0,0);
   assert.ok(["r32float","rgba32float","r32uint","rgba32uint"].includes(t.format));
   const bytes=t.format.startsWith("rgba")?16:4,row=d[0]*bytes,pitch=packing.bytesPerRow??row,rows=packing.rowsPerImage??d[1],offset=packing.offset??0;
   const src=ArrayBuffer.isView(data)?new Uint8Array(data.buffer,data.byteOffset,data.byteLength):new Uint8Array(data);
   const packed=new Uint8Array(row*d[1]*d[2]);
   for(let z=0;z<d[2];z++)for(let y=0;y<d[1];y++)packed.set(src.subarray(offset+pitch*(y+rows*z),offset+pitch*(y+rows*z)+row),row*(y+d[1]*z));
   const permuted=this.reorder(packed,d,bytes);this.counts.uploads++;this.counts.uploadBytes+=packed.byteLength;
   return write(destination,permuted,{bytesPerRow:row,rowsPerImage:d[1]},size);
  });
  return {destroy:()=>metadata?.destroy()};
 }
}
