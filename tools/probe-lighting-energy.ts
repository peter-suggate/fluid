/** Read-only term decomposition of the production split/raster lighting shader.
 * Reuses the actual scene, material, receiver and visibility bindings. This is
 * diagnostic radiance before grade/vignette, not a replacement shading path. */
import { createSvoDrySceneFragmentWGSL, drySceneVertexShader, type SvoDryOptimizationExperiments, type SvoDryTraversalMode, type SvoBrickOccupancyMode } from "../lib/svo/features/shading/program";

const columns=160,rows=100,terms=["direct","diffuseEnvironment","specularEnvironment","emissionAndIndirect","keyDependentSpecularEnvironment"] as const;
export function lightingEnergyProbeWGSL(source:string):string {
  const entry="@fragment fn dryLightingMain(input:VertexOut)->@location(0) vec4f{";
  const composition="var shaded=max(surface.emissive+diffuseEnvironment+specularEnvironment+direct*directScale+indirectDiffuse,vec3f(0.0));";
  if(!source.includes(entry)||!source.includes(composition))throw new Error("Production lighting diagnostic contract changed");
  return source.replace(entry,"fn energyShade(input:VertexOut)->vec4f{").replace(composition,`
  energyDirect=direct*directScale;energyDiffuse=diffuseEnvironment;energySpecular=specularEnvironment;
  energyOther=surface.emissive+indirectDiffuse;energyOwner=hit.ownerId;
  var noKey=dryLighting.environment;noKey.keyColorIntensity.w=0.0;
  energyKeySpecular=(dryEnvironment(reflected,surface.roughness)-svoEnvironmentPrefilteredSpecular(noKey,reflected,surface.roughness))*environmentBrdf*drySpecularVisibility(dot(contactVisibility,vec3f(1.0/3.0))*gi.visibility,max(dot(surface.normal,viewDirection),0.0),surface.roughness);
  ${composition}`)+/* wgsl */`
var<private> energyDirect:vec3f;var<private> energyDiffuse:vec3f;var<private> energySpecular:vec3f;
var<private> energyOther:vec3f;var<private> energyKeySpecular:vec3f;var<private> energyOwner:u32;
@fragment fn lightingEnergyMain(@builtin(position) pixel:vec4f)->@location(0) vec4f{
  let term=u32(pixel.x)/${columns}u;let col=u32(pixel.x)%${columns}u;
  let dimensions=vec2f(textureDimensions(drySplitGeometryRead));
  let coordinate=vec2i((vec2f(f32(col),pixel.y-.5)+.5)/vec2f(${columns},${rows})*dimensions);
  var input:VertexOut;input.position=vec4f(vec2f(coordinate)+.5,0,1);input.uv=(vec2f(coordinate)+.5)/dimensions*vec2f(1,-1)+vec2f(0,1);
  energyOwner=65535u;dryDerivedPageFailure=0u;dryVisibilityFailure=0u;
  let ignored=energyShade(input);
  var value=energyDirect;if(term==1u){value=energyDiffuse;}if(term==2u){value=energySpecular;}if(term==3u){value=energyOther;}if(term==4u){value=energyKeySpecular;}
  if(dryDerivedPageFailure!=0u||dryVisibilityFailure!=0u||any(value!=value)){return vec4f(0,0,0,-2);}
  return vec4f(value,f32(energyOwner));
}`;
}

export async function probeLightingEnergy(device:GPUDevice,pipeline:unknown){
  const p=pipeline as {
    layout?:GPUBindGroupLayout;bindGroup?:GPUBindGroup;splitLightingLayout?:GPUBindGroupLayout;splitLightingBindGroup?:GPUBindGroup;
    splitGeometry?:GPUTexture;rasterAo?:{consumerLayout:GPUBindGroupLayout;consumer:GPUBindGroup;visibilityEnabled:boolean};
    experiments:SvoDryOptimizationExperiments;splitPipelineScale?:number;splitPipelineBackdropTerrain:boolean;splitPipelineRigidBodies:boolean;
    traversalMode:SvoDryTraversalMode;brickOccupancyMode:SvoBrickOccupancyMode;screenSpaceTerminationPixels:number;
    lightingOptions:{globalIlluminationEnabled?:boolean};
  };
  if(!p.layout||!p.bindGroup||!p.splitLightingLayout||!p.splitLightingBindGroup||!p.splitGeometry||!p.rasterAo||p.splitPipelineScale!==1||p.lightingOptions.globalIlluminationEnabled)
    throw new Error("Energy diagnostic requires initialized full-rate split raster lighting with GI disabled");
  const source=createSvoDrySceneFragmentWGSL(1,p.traversalMode,p.brickOccupancyMode,"split",p.screenSpaceTerminationPixels,false,false,false,{
    ...p.experiments,halfPrecisionLighting:false,voxelLightCache:false,globalIlluminationAbsent:true,
    backdropTerrainAbsent:!p.splitPipelineBackdropTerrain,rigidBodiesAbsent:!p.splitPipelineRigidBodies,
    surfaceMeshSubgroups:p.experiments.surfaceMeshSubgroups!==false&&device.features.has("subgroups"),
    rasterAo:true,opaqueRasterLighting:true,precomputedRasterVisibility:p.rasterAo.visibilityEnabled,
  });
  const module=device.createShaderModule({label:"Production lighting energy diagnostic",code:lightingEnergyProbeWGSL(source)});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(JSON.stringify(errors));
  const render=await device.createRenderPipelineAsync({label:"Production lighting energy terms",layout:device.createPipelineLayout({bindGroupLayouts:[p.layout,p.splitLightingLayout,p.rasterAo.consumerLayout]}),
    vertex:{module:device.createShaderModule({code:drySceneVertexShader}),entryPoint:"vertexMain"},fragment:{module,entryPoint:"lightingEnergyMain",targets:[{format:"rgba32float"}]},primitive:{topology:"triangle-list"}});
  const width=columns*terms.length,height=rows,bytesPerRow=Math.ceil(width*16/256)*256;
  const texture=device.createTexture({size:[width,height],format:"rgba32float",usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});
  const read=device.createBuffer({size:bytesPerRow*height,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{
    const encoder=device.createCommandEncoder(),pass=encoder.beginRenderPass({colorAttachments:[{view:texture.createView(),clearValue:{r:0,g:0,b:0,a:65535},loadOp:"clear",storeOp:"store"}]});
    pass.setPipeline(render);pass.setBindGroup(0,p.bindGroup);pass.setBindGroup(1,p.splitLightingBindGroup);pass.setBindGroup(2,p.rasterAo.consumer);pass.draw(3);pass.end();
    encoder.copyTextureToBuffer({texture},{buffer:read,bytesPerRow},[width,height]);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
    const data=new Float32Array(read.getMappedRange()),owners:Record<string,{count:number;linearLuminanceSums:Record<string,number>}>={},samples=[];
    let invalid=0;
    for(let y=0;y<rows;y++)for(let x=0;x<columns;x++){
      const owner=data[y*bytesPerRow/4+x*4+3]!,radiance:Record<string,number[]>={};let valid=owner!==-2;
      for(let term=0;term<terms.length;term++){
        const at=y*bytesPerRow/4+(x+term*columns)*4;valid&&=data[at+3]===owner;
        radiance[terms[term]!]=Array.from(data.subarray(at,at+3));
      }
      if(!valid){invalid++;continue;}
      if(owner===65535)continue;
      const aggregate=owners[owner]??={count:0,linearLuminanceSums:Object.fromEntries(terms.map(t=>[t,0]))};aggregate.count++;
      for(const term of terms){const rgb=radiance[term]!;aggregate.linearLuminanceSums[term]!+=rgb[0]!*.2126+rgb[1]!*.7152+rgb[2]!*.0722;}
      samples.push({pixel:[Math.floor((x+.5)/columns*p.splitGeometry.width),Math.floor((y+.5)/rows*p.splitGeometry.height)],owner,radiance});
    }
    return {columns,rows,width:p.splitGeometry.width,height:p.splitGeometry.height,invalid,owners,samples,
      description:"Actual production receiver/material/visibility bindings, production BRDF and energy composition. Linear radiance before grade, vignette and haze. GI off, raster visibility. keyDependentSpecularEnvironment is a subset of specularEnvironment, not an additional term. No scene/source mutation."};
  }finally{if(read.mapState==="mapped")read.unmap();read.destroy();texture.destroy();}
}
