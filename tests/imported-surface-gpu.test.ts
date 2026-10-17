import test from 'node:test';
import { SVO_GBUFFER_RENDER_TARGET_CONTRACT as G } from '../lib/svo/features/primary-visibility/webgpu-svo-gbuffer-targets';
import assert from 'node:assert/strict';
import { withUniformDevice } from './helpers/uniform-geometric';
import { createSvoDrySceneFragmentWGSL } from '../lib/svo/features/shading/program';

test('native surface and camera-independent ranges compile in the production mesh shader', {skip:!process.env.WEBGPU_NODE_MODULE}, async()=>{
  await withUniformDevice('native surface shader',async device=>{
    const code=createSvoDrySceneFragmentWGSL(1,'raster-primary','off','split',0,false,false,false,
      {surfaceMesh:true,rasterAo:true,voxelLightCache:false,surfaceMeshSubgroups:device.features.has("subgroups")});
    const module=device.createShaderModule({code});
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==='error'),[]);
    await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'importedSurfaceSelect'}});
    await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'surfaceMeshCull'}});
    await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'surfaceMeshCullDispatch'}});
    await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'surfaceMeshPrimaryCandidates'}});
    await device.createRenderPipelineAsync({layout:'auto',vertex:{module,entryPoint:'importedSurfaceVertex',buffers:[{arrayStride:12,attributes:[{shaderLocation:0,offset:0,format:'float32x3'}]},{arrayStride:128,stepMode:'instance',attributes:[{shaderLocation:1,offset:0,format:'uint32'}]}]},
      fragment:{module,entryPoint:'importedSurfaceFragment',targets:[{format:G.packedSurfaceFormat},{format:G.identityMediaFormat},{format:'rgba32float'},{format:'rg32uint'}]},
      primitive:{topology:'triangle-list'},depthStencil:{format:'depth32float',depthWriteEnabled:true,depthCompare:'greater'}});
  },["subgroups"]);
});


test('rigid surfaces can publish the complete primary G-buffer directly', {skip:!process.env.WEBGPU_NODE_MODULE}, async()=>{
  const {svoRigidRasterShader}=await import('../lib/svo/features/primary-visibility/webgpu-svo-rigid-raster');
  await withUniformDevice('rigid direct surface',async device=>{
    const module=device.createShaderModule({code:svoRigidRasterShader});
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==='error'),[]);
    await device.createRenderPipelineAsync({layout:'auto',vertex:{module,entryPoint:'rigidRasterVertex'},
      fragment:{module,entryPoint:'rigidDirectFragment',targets:[{format:G.packedSurfaceFormat},{format:G.identityMediaFormat},{format:'rgba32float'},{format:'rg32uint'}]},
      primitive:{topology:'triangle-list'},depthStencil:{format:'depth32float',depthWriteEnabled:true,depthCompare:'greater'}});
  });
});
