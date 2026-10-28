/** Read-only native self-shadow comparison over actual crown receivers. */
import {rasterAoConsumerWGSL,rasterReceiverPlaneWGSL} from "../lib/svo/features/lighting-visibility/svo-raster-ao";
import {SVO_GBUFFER_NORMAL_OCT8_WGSL} from "../lib/svo/contracts/svo-gbuffer";
import { createSvoDrySceneFragmentWGSL, SVO_DRY_SCENE_ARENA_LAYOUT as ARENA } from "../lib/svo/features/shading/program";
import { svoPrimitiveWGSL, SVO_PRIMITIVE_KINDS } from "../lib/svo/contracts/svo-primitive-abi";
import { meshVolumeWGSL } from "../lib/core/imported-assets/mesh-volume.wgsl";
import { sampledSurfaceProjectionWGSL } from "../lib/svo/features/lighting-visibility/sampled-surface-projection.wgsl";
import { sampledAmbientVisibilityWGSL } from "../lib/svo/features/lighting-visibility/sampled-ambient-visibility.wgsl";
function declaration(source:string,prefix:string){const start=source.indexOf(prefix);if(start<0)throw new Error(`Missing production declaration ${prefix}`);let i=source.indexOf("{",start),depth=1;for(i++;depth&&i<source.length;i++){if(source[i]==="{")depth++;if(source[i]==="}")depth--;}return source.slice(start,i);}
export async function probeNativeSun(device:GPUDevice,pipeline:unknown,viewWords:Float32Array){
  const p=pipeline as {splitGeometry?:GPUTexture;splitOpaqueIdentity?:GPUTexture;sceneArenaBuffer:GPUBuffer;paramsBuffer:GPUBuffer;rasterAo?:{shadow:GPUTexture;params:GPUBuffer;guide:GPUTexture}};
  if(!p.splitGeometry||!p.splitOpaqueIdentity||!p.rasterAo)throw new Error("Native AO probe requires primary geometry and identity");
  const geometry=p.splitGeometry,columns=112,rows=48,count=columns*rows;
  const view=device.createBuffer({size:64,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(view,0,new Float32Array(viewWords.subarray(0,16)));
  const output=device.createBuffer({size:count*64,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:output.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const source=createSvoDrySceneFragmentWGSL(1,"raster-primary","bounds","split",0,false,false,false,{surfaceMesh:true});
  const primitive=["svoPrimitiveCenter_m","svoPrimitiveKind","svoPrimitiveOwnerId","svoPrimitiveFieldProgramReference","svoQuaternionRotate","svoPrimitiveLocalPoint"].map(name=>declaration(svoPrimitiveWGSL,`fn ${name}(`)).join("\n");
  const adapter=["drySceneWords4","dryPrimitive","drySampledMaterialOwner","svoSculptCutCount"].map(name=>declaration(source,`fn ${name}(`)).join("\n");
  const shadowSource=rasterAoConsumerWGSL(),sampling=shadowSource.slice(shadowSource.indexOf("fn rasterSunDirection"),shadowSource.indexOf("fn rasterOctEncode"));
  const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read> drySceneArena:array<u32>;
@group(0) @binding(1) var<uniform> params:array<vec4u,44>;
@group(0) @binding(2) var geometry:texture_2d<f32>;
@group(0) @binding(3) var identities:texture_2d<u32>;
@group(0) @binding(4) var<storage,read> view:array<vec4f>;
@group(0) @binding(5) var<storage,read_write> result:array<vec4f>;
struct Dry {metadata:vec4u,tuningRays0:vec4f,tuningRays1:vec4f}var<private> dry:Dry;
struct Uniforms {container:vec4f}var<private> uniforms:Uniforms;
struct RasterSpot {positionNear:vec4f,directionTan:vec4f,rangeRadiusIndex:vec4f}
struct Lighting {control:vec4u,farCenterExtent:vec4f,depthCell:vec4f,sun:vec4f,spots:array<RasterSpot,1>}
@group(0) @binding(6) var<uniform> rasterLighting:Lighting;
@group(0) @binding(7) var rasterSunDepth:texture_depth_2d_array;
@group(0) @binding(8) var visibilityGuide:texture_2d<u32>;
const DRY_OWNER_NONE=65535u;const SVO_KIND_FIELD_PROGRAM=${SVO_PRIMITIVE_KINDS.fieldProgram}u;
const DRY_SCENE_PRIMITIVE_WORD_OFFSET=${ARENA.primitiveOffsetBytes/4}u;
const DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET=${ARENA.fieldProgramOffsetBytes/4}u;
fn drySceneWord(at:u32)->u32{return drySceneArena[at];}
fn drySculptCutWord(at:u32)->u32{return drySceneArena[${ARENA.sculptCutOffsetBytes/4}u+at];}
${declaration(svoPrimitiveWGSL,"struct SvoPrimitiveRecord {")}
${primitive}
${meshVolumeWGSL("volume","drySceneWord")}
${sampledSurfaceProjectionWGSL("drySceneWord")}
${sampledAmbientVisibilityWGSL("drySceneWord",1024)}
${rasterReceiverPlaneWGSL}
${SVO_GBUFFER_NORMAL_OCT8_WGSL}
${sampling}
${adapter}
fn receiverNormal(n:vec3f,metadata:u32)->vec3f{
 if((metadata&0x80000000u)!=0u){if((metadata&0x40000000u)!=0u){return n;}return svoGBufferUnpackNormalOct8(metadata&65535u);}
 if((metadata&134217728u)==0u){return n;}let code=(metadata>>28u)&7u;var face=vec3f(0);face[code/2u]=select(-1.,1.,(code&1u)!=0u);return face;
}
@compute @workgroup_size(32) fn main(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=${count}u){return;}dry.metadata=params[3];dry.tuningRays0=bitcast<vec4f>(params[22]);dry.tuningRays1=bitcast<vec4f>(params[23]);uniforms.container=view[3];
 let dims=vec2f(textureDimensions(geometry));
 let coord=vec2i((vec2f(f32(id.x%${columns}u),f32(id.x/${columns}u))+.5)/vec2f(${columns},${rows})*vec2f(1.,.27)*dims+vec2f(0,.45)*dims);
 let g=textureLoad(geometry,coord,0);let identity=textureLoad(identities,coord,0).xy;
 let owner=select(identity.y&65535u,DRY_OWNER_NONE,(identity.y&0x80000000u)!=0u);
 var status=SAMPLED_AO_NOT_APPLICABLE;var raster=1.;var reference=1.;var blocked=0u;var unknown=0u;var steps=0u;var centerStatus=4u;var centerDistance=0.;var cell=0.;
 if(g.w>0.0&&g.w<1e19&&owner!=DRY_OWNER_NONE){
  let ndc=(vec2f(coord)+.5)/dims*vec2f(2,-2)+vec2f(-1,1);let forward=normalize(view[2].xyz-view[1].xyz);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));
  let direction=normalize(forward+right*ndc.x*dims.x/dims.y*view[1].w+up*ndc.y*view[1].w);let world=view[1].xyz+direction*g.w;
  raster=rasterSunVisibility(world,receiverNormal(normalize(g.xyz),identity.y));
  let index=drySampledMaterialOwner(owner);
  if(index!=0xffffffffu&&svoSculptCutCount()==0u){
   let record=dryPrimitive(index);let referenceAddress=svoPrimitiveFieldProgramReference(record);
   if(svoPrimitiveKind(record)==SVO_KIND_FIELD_PROGRAM&&referenceAddress>=DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET&&drySceneWord(referenceAddress)==0xfffffffeu){
    let base=DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET+drySceneWord(referenceAddress+1u);let local=svoPrimitiveLocalPoint(record,world);let surface=sampledSurfaceProjection(base,local);status=SAMPLED_AO_INVALID;
    if(surface.valid!=0u){
     cell=bitcast<f32>(drySceneWord(base+3u));let bias=max(cell*.02,2e-7*max(1.,max(abs(local.x),max(abs(local.y),abs(local.z)))));
     let origin=surface.position+surface.normal*bias;let invQ=vec4f(-record.orientation.xyz,record.orientation.w);
     let sun=svoQuaternionRotate(invQ,rasterSunDirection());let helper=select(vec3f(0,1,0),vec3f(1,0,0),abs(sun.y)>.9);let tangent=normalize(cross(helper,sun));let bitangent=cross(sun,tangent);
     let maximum=length(vec3f(vec3u(drySceneWord(base+4u),drySceneWord(base+5u),drySceneWord(base+6u))))*cell*2.;
     let center=sampledAoRay(base,origin,sun,maximum,1024u);centerStatus=center.status;centerDistance=center.distance;
     for(var ray=0u;ray<8u;ray++){
      let radius=sqrt((f32(ray)+.5)/8.)*tan(dry.tuningRays1.y*.5);let angle=(f32(ray)+.5)*2.39996322973;
      let towards=normalize(sun+radius*(tangent*cos(angle)+bitangent*sin(angle)));let hit=sampledAoRay(base,origin,towards,maximum,1024u);steps+=hit.steps;
      blocked+=select(0u,1u,hit.status==SAMPLED_AO_BLOCKED);unknown+=select(0u,1u,hit.status>=SAMPLED_AO_EXHAUSTED);
     }
     reference=1.-f32(blocked)/8.;status=select(select(SAMPLED_AO_CLEAR,SAMPLED_AO_BLOCKED,blocked>0u),SAMPLED_AO_EXHAUSTED,unknown>0u);
    }
   }
  }
 }
 result[id.x*4u]=vec4f(vec2f(coord),f32(owner),f32(status));
 result[id.x*4u+1u]=vec4f(raster,reference,f32(centerStatus),centerDistance);
 result[id.x*4u+2u]=vec4f(f32(blocked),f32(unknown),f32(steps),cell);
 let guide=unpack2x16float(textureLoad(visibilityGuide,coord,0).x);
 result[id.x*4u+3u]=vec4f(g.w,dry.tuningRays1.y,guide.y,guide.x);
}`});
  try{
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(JSON.stringify(errors.map(e=>({message:e.message,line:e.lineNum,column:e.linePos}))));
    const compute=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
    const group=device.createBindGroup({layout:compute.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:p.sceneArenaBuffer}},{binding:1,resource:{buffer:p.paramsBuffer}},{binding:2,resource:geometry.createView()},{binding:3,resource:p.splitOpaqueIdentity.createView()},{binding:4,resource:{buffer:view}},{binding:5,resource:{buffer:output}},{binding:6,resource:{buffer:p.rasterAo.params,offset:0,size:256}},{binding:7,resource:p.rasterAo.shadow.createView({dimension:"2d-array"})},{binding:8,resource:p.rasterAo.guide.createView()}]});
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(compute);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(count/32));pass.end();encoder.copyBufferToBuffer(output,0,read,0,output.size);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
    const data=new Float32Array(read.getMappedRange().slice(0)),statusNames=["clear","occluded","exhausted","invalidProjectionOrField","notApplicable"],counts:Record<string,number>={};
    const samples=Array.from({length:count},(_,i)=>{const a=i*16,status=statusNames[data[a+3]!]??"unknown";counts[status]=(counts[status]??0)+1;
      return {pixel:[data[a],data[a+1]],owner:data[a+2],status,rasterVisibility:data[a+4],nativeVisibility:data[a+5],centerStatus:statusNames[data[a+6]!]??"unknown",centerDistance_m:data[a+7],blockedRays:data[a+8],unknownRays:data[a+9],steps:data[a+10],cellSize_m:data[a+11],depth_m:data[a+12],aperture_rad:data[a+13],resolvedVisibility:data[a+14],resolvedAO:data[a+15]};});
    return {shadowResolution:p.rasterAo.shadow.width,columns,rows,width:geometry.width,height:geometry.height,counts,description:"Actual FVX crown receivers. Production PCF versus8 stratified finite-sun disk rays and1center ray through SAME OWNER native scalar field. Same-owner visibility is an upper bound on full-scene direct visibility. Projection and marcher use production algorithm, diagnostic-only1024step budget. Invalid/exhausted estimates are not visibility evidence.",samples};

  }finally{if(read.mapState==="mapped")read.unmap();view.destroy();output.destroy();read.destroy();}
}
