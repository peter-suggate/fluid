/** Read-only GPU status census over real primary receivers and the published
 * native FVX arena. Uses production projection, marcher and owner adapter. */
import { createSvoDrySceneFragmentWGSL, SVO_DRY_SCENE_ARENA_LAYOUT as ARENA } from "../lib/svo/features/shading/program";
import { svoPrimitiveWGSL, SVO_PRIMITIVE_KINDS } from "../lib/svo/contracts/svo-primitive-abi";
import { meshVolumeWGSL } from "../lib/core/imported-assets/mesh-volume.wgsl";
import { sampledSurfaceProjectionWGSL } from "../lib/svo/features/lighting-visibility/sampled-surface-projection.wgsl";
import { sampledAmbientVisibilityWGSL } from "../lib/svo/features/lighting-visibility/sampled-ambient-visibility.wgsl";
function declaration(source:string,prefix:string){const start=source.indexOf(prefix);if(start<0)throw new Error(`Missing production declaration ${prefix}`);let i=source.indexOf("{",start),depth=1;for(i++;depth&&i<source.length;i++){if(source[i]==="{")depth++;if(source[i]==="}")depth--;}return source.slice(start,i);}
export async function probeNativeAo(device:GPUDevice,pipeline:unknown,viewWords:Float32Array){
  const p=pipeline as {splitGeometry?:GPUTexture;splitOpaqueIdentity?:GPUTexture;sceneArenaBuffer:GPUBuffer;paramsBuffer:GPUBuffer};
  if(!p.splitGeometry||!p.splitOpaqueIdentity)throw new Error("Native AO probe requires primary geometry and identity");
  const geometry=p.splitGeometry,columns=160,rows=100,count=columns*rows;
  const view=device.createBuffer({size:64,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(view,0,new Float32Array(viewWords.subarray(0,16)));
  const output=device.createBuffer({size:count*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:output.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const source=createSvoDrySceneFragmentWGSL(1,"raster-primary","bounds","split",0,false,false,false,{surfaceMesh:true});
  const primitive=["svoPrimitiveCenter_m","svoPrimitiveKind","svoPrimitiveOwnerId","svoPrimitiveFieldProgramReference","svoQuaternionRotate","svoPrimitiveLocalPoint"].map(name=>declaration(svoPrimitiveWGSL,`fn ${name}(`)).join("\n");
  const adapter=["drySceneWords4","dryPrimitive","drySampledMaterialOwner","svoSculptCutCount","dryNativeAmbientVisibility"].map(name=>declaration(source,`fn ${name}(`)).join("\n");
  const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read> drySceneArena:array<u32>;
@group(0) @binding(1) var<uniform> params:array<vec4u,44>;
@group(0) @binding(2) var geometry:texture_2d<f32>;
@group(0) @binding(3) var identities:texture_2d<u32>;
@group(0) @binding(4) var<storage,read> view:array<vec4f>;
@group(0) @binding(5) var<storage,read_write> result:array<vec4f>;
struct Dry {metadata:vec4u}var<private> dry:Dry;
const DRY_OWNER_NONE=65535u;const SVO_KIND_FIELD_PROGRAM=${SVO_PRIMITIVE_KINDS.fieldProgram}u;
const DRY_SCENE_PRIMITIVE_WORD_OFFSET=${ARENA.primitiveOffsetBytes/4}u;
const DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET=${ARENA.fieldProgramOffsetBytes/4}u;
fn drySceneWord(at:u32)->u32{return drySceneArena[at];}
fn drySculptCutWord(at:u32)->u32{return drySceneArena[${ARENA.sculptCutOffsetBytes/4}u+at];}
${declaration(svoPrimitiveWGSL,"struct SvoPrimitiveRecord {")}
${primitive}
${meshVolumeWGSL("volume","drySceneWord")}
${sampledSurfaceProjectionWGSL("drySceneWord")}
${sampledAmbientVisibilityWGSL("drySceneWord")}
${adapter}
@compute @workgroup_size(32) fn main(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=${count}u){return;}dry.metadata=params[3];
 let dims=vec2f(textureDimensions(geometry));let coord=vec2i((vec2f(f32(id.x%${columns}u),f32(id.x/${columns}u))+.5)/vec2f(${columns},${rows})*dims);
 let g=textureLoad(geometry,coord,0);let identity=textureLoad(identities,coord,0).xy;
 let owner=select(identity.y&65535u,DRY_OWNER_NONE,(identity.y&0x80000000u)!=0u);
 var native=SampledAmbientVisibility(1.,SAMPLED_AO_NOT_APPLICABLE,0u,0u);
 if(g.w>0.0&&g.w<1e19){
  let ndc=(vec2f(coord)+.5)/dims*vec2f(2,-2)+vec2f(-1,1);let forward=normalize(view[2].xyz-view[1].xyz);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));
  let direction=normalize(forward+right*ndc.x*dims.x/dims.y*view[1].w+up*ndc.y*view[1].w);
  let cell=bitcast<vec3f>(params[1].xyz);let sceneScale=max(view[3].x,max(view[3].y,view[3].z));
  let radius=bitcast<vec4f>(params[22]).z*min(sceneScale*.06,max(max(cell.x,max(cell.y,cell.z))*6.,sceneScale*.01));
  native=dryNativeAmbientVisibility(view[1].xyz+direction*g.w,owner,radius);
 }
 result[id.x*2u]=vec4f(vec2f(coord),f32(owner),f32(native.status));
 result[id.x*2u+1u]=vec4f(native.visibility,f32(native.steps),f32(native.blocked),g.w);
}`});
  try{
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(JSON.stringify(errors));
    const compute=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
    const group=device.createBindGroup({layout:compute.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:p.sceneArenaBuffer}},{binding:1,resource:{buffer:p.paramsBuffer}},{binding:2,resource:geometry.createView()},{binding:3,resource:p.splitOpaqueIdentity.createView()},{binding:4,resource:{buffer:view}},{binding:5,resource:{buffer:output}}]});
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(compute);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(count/32));pass.end();encoder.copyBufferToBuffer(output,0,read,0,output.size);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
    const data=new Float32Array(read.getMappedRange().slice(0)),statusNames=["clear","occluded","exhausted","invalidProjectionOrField","notApplicable"],counts:Record<string,number>={},owners:Record<string,{count:number;status:Record<string,number>;visibilitySum:number;steps:number}>={};
    const samples=Array.from({length:count},(_,i)=>{const a=i*8,status=statusNames[data[a+3]!]??"unknown",owner=String(data[a+2]);counts[status]=(counts[status]??0)+1;
      if(status!=="notApplicable"){const row=owners[owner]??={count:0,status:{},visibilitySum:0,steps:0};row.count++;row.status[status]=(row.status[status]??0)+1;row.visibilitySum+=data[a+4]!;row.steps+=data[a+5]!;}
      return {pixel:[data[a],data[a+1]],owner:Number(owner),status,visibility:data[a+4],steps:data[a+5],blockedRays:data[a+6]};});
    return {columns,rows,width:geometry.width,height:geometry.height,counts,owners,description:"GPU census of actual primary receivers using production native FVX AO. Four cosine rays,512 steps each, radius=existing world contact radius controlled by AO reach. Unknown estimates retain existing AO. This probe does not include reflected receivers.",samples};
  }finally{if(read.mapState==="mapped")read.unmap();view.destroy();output.destroy();read.destroy();}
}
