import { RASTER_LIGHTING_PARAM_BYTES, RASTER_MAX_LOCAL_SHADOW_VIEWS, rasterShadowAtlasWGSL } from "../lib/svo/features/lighting-visibility/raster-local-shadows";
/** Optional, read-only probe of production receiver and shadow publications. */
import { rasterAoConsumerWGSL, rasterReceiverPlaneWGSL } from "../lib/svo/features/lighting-visibility/svo-raster-ao";
import { SVO_GBUFFER_NORMAL_OCT8_WGSL } from "../lib/svo/contracts/svo-gbuffer";

export async function probeRasterShadow(device:GPUDevice,pipeline:unknown,viewWords:Float32Array,biasCells=.02,aperture=.065){
  const p=pipeline as {splitGeometry?:GPUTexture;splitOpaqueIdentity?:GPUTexture;rasterAo?:{shadow:GPUTexture;params:GPUBuffer}};
  if(!p.splitGeometry||!p.splitOpaqueIdentity||!p.rasterAo)throw new Error("Raster shadow diagnostic requires published geometry, identity and shadow maps");
  const geometry=p.splitGeometry,ao=p.rasterAo;
  const columns=112,rows=24,count=columns*rows;
  const words=new Float32Array(24);words.set(viewWords.subarray(0,16));words.set([columns,rows,biasCells,aperture],16);
  words.set([.02,.725,.4,.785],20);
  const input=device.createBuffer({size:words.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(input,0,words);
  const output=device.createBuffer({size:count*9*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const read=device.createBuffer({size:output.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const source=rasterAoConsumerWGSL(),sampling=source.slice(source.indexOf("fn rasterSunDirection"),source.indexOf("fn rasterOctEncode"));
  const module=device.createShaderModule({code:`
struct Uniforms {container:vec4f} struct Dry {tuningRays0:vec4f,tuningRays1:vec4f}
struct RasterSpot {positionNear:vec4f,directionTan:vec4f,rangeRadiusIndex:vec4f}
struct Lighting {control:vec4u,farCenterExtent:vec4f,depthCell:vec4f,sun:vec4f,spots:array<RasterSpot,${RASTER_MAX_LOCAL_SHADOW_VIEWS}>,lightSlots:array<vec4u,32>,localLayout:vec4u}
var<private> uniforms:Uniforms;var<private> dry:Dry;
@group(0) @binding(0) var geometry:texture_2d<f32>;
@group(0) @binding(1) var identities:texture_2d<u32>;
@group(0) @binding(2) var rasterSunDepth:texture_depth_2d_array;
@group(0) @binding(3) var<uniform> rasterLighting:Lighting;
@group(0) @binding(4) var<storage,read> view:array<vec4f>;
@group(0) @binding(5) var<storage,read_write> result:array<vec4f>;
${rasterReceiverPlaneWGSL}
${SVO_GBUFFER_NORMAL_OCT8_WGSL}
${rasterShadowAtlasWGSL}
${sampling}
fn worldAt(c:vec2i)->vec3f{
 let dimensions=vec2f(textureDimensions(geometry));let uv=(vec2f(c)+.5)/dimensions*vec2f(2,-2)+vec2f(-1,1);
 let forward=normalize(view[2].xyz-view[1].xyz);let right=normalize(view[26].xyz);let up=normalize(cross(right,forward));
 let ray=normalize(forward+right*uv.x*dimensions.x/dimensions.y*view[1].w+up*uv.y*view[1].w);
 return view[1].xyz+ray*textureLoad(geometry,c,0).w;
}
fn geometricNormal(normal:vec3f,metadata:u32)->vec3f{
 if((metadata&0x80000000u)!=0u){if((metadata&0x40000000u)!=0u){return normal;}return svoGBufferUnpackNormalOct8(metadata&0xffffu);}
 if((metadata&134217728u)==0u){return normal;}
 let code=(metadata>>28u)&7u;var n=vec3f(0);n[code/2u]=select(-1.,1.,(code&1u)!=0u);return n;
}
@compute @workgroup_size(32) fn main(@builtin(global_invocation_id) id:vec3u){
 let columns=u32(view[4].x);let count=columns*u32(view[4].y);if(id.x>=count){return;}
 uniforms.container=view[3];dry.tuningRays0=vec4f(view[4].z,0,0,0);dry.tuningRays1=vec4f(0,view[4].w,0,0);
 let c=vec2i(mix(view[5].xy,view[5].zw,vec2f(f32(id.x%columns)/f32(columns-1u),f32(id.x/columns)/f32(u32(view[4].y)-1u)))*vec2f(textureDimensions(geometry)));
 let g=textureLoad(geometry,c,0);let identity=textureLoad(identities,c,0).xy;let n=geometricNormal(normalize(g.xyz),identity.y);let world=worldAt(c);
 let right=worldAt(c+vec2i(1,0))-world;let left=world-worldAt(c-vec2i(1,0));
 let down=worldAt(c+vec2i(0,1))-world;let up=world-worldAt(c-vec2i(0,1));
 let dx=select(left,right,dot(right,right)<dot(left,left));let dy=select(up,down,dot(down,down)<dot(up,up));
 var plane=normalize(cross(dx,dy));if(dot(plane,n)<0.){plane=-plane;}
 let near=rasterShadowPosition(world,0u);let layer=select(0u,1u,max(abs(near.x),abs(near.y))>=.98);
 let projected=rasterShadowPosition(world,layer);let uv=projected.xy*vec2f(.5,-.5)+.5;
 let shadow=textureLoad(rasterSunDepth,clamp(vec2i(uv*vec2f(textureDimensions(rasterSunDepth))),vec2i(0),vec2i(textureDimensions(rasterSunDepth))-1),i32(layer),0);
 result[id.x*9u]=vec4f(vec2f(c),bitcast<f32>(identity.x),bitcast<f32>(identity.y));
 result[id.x*9u+1u]=vec4f(world,g.w);
 result[id.x*9u+2u]=vec4f(n,rasterSunVisibility(world,n));
 result[id.x*9u+3u]=vec4f(plane,rasterSunVisibility(world,plane));
 result[id.x*9u+4u]=vec4f(normalize(g.xyz),rasterSunVisibility(world,normalize(g.xyz)));
 result[id.x*9u+5u]=vec4f(uv,(projected.z-shadow)*rasterLighting.depthCell.x,f32(layer));
 result[id.x*9u+6u]=rasterLighting.depthCell;
 result[id.x*9u+7u]=vec4f(rasterSunDirection(),rasterMapExtent(layer));
 result[id.x*9u+8u]=vec4f(dot(n,plane),dot(n,rasterSunDirection()),dot(plane,rasterSunDirection()),0);
}`});
  try{
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(JSON.stringify(errors));
    const compute=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
    const group=device.createBindGroup({layout:compute.getBindGroupLayout(0),entries:[
      {binding:0,resource:geometry.createView()},{binding:1,resource:p.splitOpaqueIdentity.createView()},
      {binding:2,resource:ao.shadow.createView({dimension:"2d-array"})},{binding:3,resource:{buffer:ao.params,offset:0,size:RASTER_LIGHTING_PARAM_BYTES}},
      {binding:4,resource:{buffer:input}},{binding:5,resource:{buffer:output}},
    ]});
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(compute);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(count/32));pass.end();encoder.copyBufferToBuffer(output,0,read,0,output.size);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
    const data=new Float32Array(read.getMappedRange().slice(0)),u=new Uint32Array(data.buffer);
    return {shadowResolution:ao.shadow.width,width:geometry.width,height:geometry.height,columns,rows,biasCells,aperture,description:"Production G-buffer and shadow depth, production PCF. Plane normal estimated from closest adjacent screen-depth differences, not an authoritative triangle identity.",samples:Array.from({length:count},(_,i)=>{
      const a=i*36;return {pixel:[data[a],data[a+1]],material:u[a+2],metadata:u[a+3],world:[...data.subarray(a+4,a+7)],radialDepth:data[a+7],receiver:[...data.subarray(a+8,a+12)],depthPlane:[...data.subarray(a+12,a+16)],shading:[...data.subarray(a+16,a+20)],shadow:[...data.subarray(a+20,a+24)],depthCell:[...data.subarray(a+24,a+28)],sunExtent:[...data.subarray(a+28,a+32)],normalComparison:[...data.subarray(a+32,a+36)]};})};
  }finally{if(read.mapState==="mapped")read.unmap();input.destroy();output.destroy();read.destroy();}
}
