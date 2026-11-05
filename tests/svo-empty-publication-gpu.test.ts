import assert from 'node:assert/strict';
import test from 'node:test';
import {createEmptyScene} from '../lib/core/empty-scene';
import {SPARSE_VOXEL_PUBLICATION_STATE,SPARSE_VOXEL_VALID_FIELDS} from '../lib/core/webgpu-voxel-debug';
import {WebGPULiveSvoScene} from '../lib/svo/features/scene-publication/webgpu-live-svo-scene';
import {withUniformDevice} from './helpers/uniform-geometric';

test('GPU publishes an authored empty static world as a complete valid generation',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice('empty static scene publication',async device=>{
  const scene=createEmptyScene({extents_m:{x:.32,y:.32,z:.32},finestCellSize_m:.01});
  scene.container={...scene.container,boundary:'free-space',vessel:'none'};
  scene.scenery={palettes:{},nodes:[{id:'empty',kind:'room-shell',materialModel:'room',faces:[],floor:{colorLinear:[1,1,1]},wall:{colorLinear:[1,1,1]},ceiling:{colorLinear:[1,1,1]}}]};
  scene.solidVoxels=[];scene.rigidBodies=[];delete scene.terrain;
  const world=await WebGPULiveSvoScene.create(device,scene,'balanced',()=>{},undefined,{derivedRadiance:false});
  const read=device.createBuffer({size:32,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{
   const structural=world.sparseVoxelSceneSource.structural!;
   const binding=structural.publication.state,encoder=device.createCommandEncoder();
   encoder.copyBufferToBuffer(binding.buffer,binding.offset??0,read,0,32);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
   const words=new Uint32Array(read.getMappedRange()),W=SPARSE_VOXEL_PUBLICATION_STATE,F=SPARSE_VOXEL_VALID_FIELDS;
   assert.ok(words[W.completeGeneration]!>0,'a fenced empty source must have a completed generation');
   const required=F.topology|F.sceneGeometry|F.materialOwner;
   assert.equal(words[W.validFields]!&required,required,'emptiness is valid geometry and owner data');
   assert.ok(words[W.publishedSceneRevision]!>0,'normal scene finalization owns the generation');
   assert.equal(structural.planarBoundaries.count,0,'publishing empty geometry adds no floor or wall');
  }finally{if(read.mapState==='mapped')read.unmap();read.destroy();world.destroy();}
 });
});
