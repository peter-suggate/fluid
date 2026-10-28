import { worldVoxelRefinementWGSL, worldVoxelRefinementSceneWGSL } from "./world-voxel-refinement.wgsl";
import { createWaterWorldReflectionWGSL } from "../lib/svo/features/shading/water-world-reflections";
import { drySceneVertexShader, type SvoDryTraversalMode, type SvoBrickOccupancyMode } from "../lib/svo/features/shading/program";
import type { RasterWaterPipeline } from "../lib/core/webgpu-water-pipeline";
import type { SparseVoxelDrySceneRenderer } from "../lib/svo/pipeline/webgpu-svo-dry-scene";

/** Read-only diagnostic: production buffers and World traversal, with an experimental
 * refinement that is NOT applied by the production renderer. The diagnostic render is separate from all presentation targets. */
export async function captureWorldVoxelRefinementProbe(device:GPUDevice,scene:SparseVoxelDrySceneRenderer,water:RasterWaterPipeline){
 const dry=scene as unknown as {layout?:GPUBindGroupLayout;bindGroup?:GPUBindGroup;traversalMode:SvoDryTraversalMode;brickOccupancyMode:SvoBrickOccupancyMode};
 const optics=(water as unknown as {waterSceneOpticsBuffer?:GPUBuffer}).waterSceneOpticsBuffer;
 const positions=water.diagnosticCaptureTexture("interface-positions"),normals=water.diagnosticCaptureTexture("interfaces"),queries=water.diagnosticCaptureTexture("world-refractions");
 if(!dry.layout||!dry.bindGroup||!optics||!positions||!normals||!queries)throw new Error("World refinement probe needs the current bound dry scene, interfaces and World transmission");
 const columns=64,rows=48,count=columns*rows;
 const source=createWaterWorldReflectionWGSL(dry.traversalMode,dry.brickOccupancyMode)+worldVoxelRefinementWGSL+worldVoxelRefinementSceneWGSL+`
 @group(1) @binding(3) var<uniform> probeOptics:vec4f;
 struct ProbeOutput{@location(0) distances:vec4f,@location(1) normalMode:vec4f}
 @fragment fn refinementProbe(@builtin(position) pixel:vec4f)->ProbeOutput{
  let index=u32(pixel.x);let uv=(vec2f(f32(index%${columns}u),f32(index/${columns}u))+vec2f(.5))/vec2f(${columns},${rows});
  let at=clamp(vec2i(uv*vec2f(textureDimensions(waterWorldPosition))),vec2i(0),vec2i(textureDimensions(waterWorldPosition))-1);
  let existing=textureLoad(waterWorldScreen,at,0);let packed=textureLoad(waterWorldPosition,at,0);let packedNormal=textureLoad(waterWorldNormal,at,0);
  let inactive=ProbeOutput(vec4f(0),vec4f(0,0,0,dry.meshFilterNormals.w));
  if(existing.a<=0.0||packed.a<.5||dot(packedNormal.xyz,packedNormal.xyz)<1e-12){return inactive;}
  let position=packed.xyz/packed.a;let incoming=normalize(position-uniforms.cameraPosition.xyz);var n=normalize(packedNormal.xyz);if(dot(n,incoming)>0.0){n=-n;}
  let request=refract(incoming,n,1.0/max(probeOptics.w,1.0));let xy=unpack2x16float(pack2x16float(request.xy));let z=unpack2x16float(pack2x16float(vec2f(request.z,0))).x;let rd=normalize(vec3f(xy,z));
  let epsilon=max(.0001,.001*min(dry.mapping.cellSize.x,min(dry.mapping.cellSize.y,dry.mapping.cellSize.z)));let ro=position+rd*epsilon;
  waterWorldTraversalExhausted=false;dryDerivedPageFailure=0u;dryVisibilityFailure=0u;dryVisibilityIgnoredBody=DRY_OWNER_NONE;
  let raw=traceOpaqueScene(ro,rd);if(waterWorldTraversalExhausted){return ProbeOutput(vec4f(0,0,0,-2),inactive.normalMode);}
  if(raw.t>=DRY_MISS){return ProbeOutput(vec4f(0,0,0,-3),inactive.normalMode);}
  var status=3.0;var result=raw;
  if(waterWorldRefinementEligible(raw)){let proposal=worldVoxelRefine(ro,rd,raw.t,raw.materialId);status=select(1.0,2.0,proposal.valid!=0u);result=waterWorldRefineOpaque(raw,ro,rd);if(result.motionKind==DRY_GBUFFER_MOTION_RIGID){status=4.0;}}
  var diagnostic=vec4f(result.normal,dry.meshFilterNormals.w);
  if(status==1.0){
   let point=ro+rd*raw.t;let center=worldVoxelSample(point);var reason=1.0;
   if(center.valid!=0u){
    reason=7.0;let size=worldVoxelCellSize()*center.scale;let lower=worldVoxelOrigin()+(floor((point-worldVoxelOrigin())/size-vec3f(.5))+vec3f(.5))*size;var values:array<f32,8>;
    for(var k=0u;k<8u;k+=1u){let tap=worldVoxelSample(lower+worldVoxelCorner(k)*size);values[k]=tap.value;
     if(tap.valid==0u){reason=2.0;break;}if(tap.scale!=center.scale){reason=3.0;break;}if(!(abs(tap.value)<=4.0*length(size))){reason=4.0;break;}if(tap.material!=0u&&tap.material!=raw.materialId){reason=5.0;break;}
    }
    if(reason==7.0){for(var k=0u;k<8u;k+=1u){for(var axis=0u;axis<3u;axis+=1u){let bit=1u<<axis;if((k&bit)==0u&&abs(values[k]-values[k|bit])>1.25*size[axis]){reason=6.0;}}}}
   }
   diagnostic=vec4f(f32(center.valid),center.scale,reason,dry.meshFilterNormals.w);
  }
  return ProbeOutput(vec4f(raw.t+epsilon,result.t+epsilon,f32(raw.materialId),status),diagnostic);
 }`;
 const module=device.createShaderModule({label:"Read-only structural World receiver probe",code:source});
 const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>m.message).join("\n"));
 const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"unfilterable-float"}},...[1,2].map(binding=>({binding,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float" as const}})),{binding:3,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform",minBindingSize:16}}]});
 const pipeline=await device.createRenderPipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[dry.layout,layout]}),vertex:{module:device.createShaderModule({code:drySceneVertexShader}),entryPoint:"vertexMain"},fragment:{module,entryPoint:"refinementProbe",targets:[{format:"rgba32float"},{format:"rgba32float"}]},primitive:{topology:"triangle-list"}});
 const textures=Array.from({length:2},()=>device.createTexture({size:[count,1],format:"rgba32float",usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC}));
 const rowBytes=count*16,read=device.createBuffer({size:rowBytes*2,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 try{
  const group=device.createBindGroup({layout,entries:[{binding:0,resource:positions.texture.createView()},{binding:1,resource:normals.texture.createView()},{binding:2,resource:queries.texture.createView({dimension:"2d",baseArrayLayer:1,arrayLayerCount:1})},{binding:3,resource:{buffer:optics,size:16}}]});
  const e=device.createCommandEncoder(),p=e.beginRenderPass({colorAttachments:textures.map(t=>({view:t.createView(),loadOp:"clear" as const,storeOp:"store" as const,clearValue:{r:0,g:0,b:0,a:0}}))});p.setPipeline(pipeline);p.setBindGroup(0,dry.bindGroup);p.setBindGroup(1,group);p.draw(3);p.end();for(let i=0;i<2;i++)e.copyTextureToBuffer({texture:textures[i]!},{buffer:read,offset:i*rowBytes,bytesPerRow:rowBytes},[count,1]);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);const data=new Float32Array(read.getMappedRange()).slice();read.unmap();
  const counts:Record<string,number>={inactive:0,fallback:0,refined:0,excluded:0,nearerRigid:0,failed:0,miss:0};const names:Record<number,string>={0:"inactive",1:"fallback",2:"refined",3:"excluded",4:"nearerRigid",[-2]:"failed",[-3]:"miss"};const samples=[];
  for(let i=0;i<count;i++){const status=data[i*4+3]!,name=names[status]??"unexpected";counts[name]=(counts[name]??0)+1;if(status===0)continue;const n=4*(count+i);samples.push({pixel:[Math.floor(((i%columns)+.5)/columns*positions.texture.width),Math.floor((Math.floor(i/columns)+.5)/rows*positions.texture.height)],status:name,rawDistance_m:data[i*4],refinedDistance_m:data[i*4+1],material:data[i*4+2],...(status===1?{fallback:{centerValid:data[n],scale:data[n+1],reasonCode:data[n+2]}}:{normal:[data[n],data[n+1],data[n+2]]}),surfaceMode:data[n+3]});}
  return {measurement:"Read-only experimental refinement replay of actual accepted water receiver rays; production SVO/voxel scalar buffers and half-float request direction; refinement is not applied to the rendered image",grid:[columns,rows],fallbackReasons:{1:"center unavailable",2:"corner unavailable",3:"mixed scale",4:"distance not finite/local",5:"material seam",6:"distance discontinuity",7:"no acceptable entering root/gradient"},counts,samples};
 }finally{read.destroy();textures.forEach(t=>t.destroy());}
}
