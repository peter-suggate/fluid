import assert from "node:assert/strict";
import test from "node:test";
import { defaultScene } from "../lib/core/model";
import { buildSvoPrimitiveCandidates, packSvoPrimitiveCandidateArena } from "../lib/svo/features/scene-publication/svo-primitive-candidates";
import { SparseVoxelDrySceneRenderer, buildSparseVoxelDrySceneLightingMirrors } from "../lib/svo/pipeline/webgpu-svo-dry-scene";
import { SVO_MATERIAL_RECORD_STRIDE_BYTES } from "../lib/svo/contracts/svo-material-abi";
import { PLANAR_BOUNDARY_PATCH_BYTES } from "../lib/core/planar-boundary";
import { SVO_DRY_SCENE_ARENA_LAYOUT } from "../lib/svo/features/shading/program";

test("production atlas-only scene publication rebuilds scene and specialized bindings without arena growth",()=>{
 const primitiveRecords=new Uint32Array(),primitiveCandidates=buildSvoPrimitiveCandidates([]);
 const base={renderRevision:1,materialRevision:1,voxelGeometryRevision:1,primitiveRecords,primitiveCandidates,
  materialRecords:new Uint32Array(2*SVO_MATERIAL_RECORD_STRIDE_BYTES/4),...buildSparseVoxelDrySceneLightingMirrors(defaultScene,1)!};
 let groups=0;const atlasA={},atlasB={};let held:unknown;
 const surfaceTextures={view:{},publish(atlas:unknown){if(atlas!==held){held=atlas;this.view={};}}};surfaceTextures.publish(atlasA);
 const scene={...base,surfaceTextureAtlas:atlasA};
 const renderer=Object.assign(Object.create(SparseVoxelDrySceneRenderer.prototype),{
  source:{structural:{fields:{topology:{residency:"resident"},sceneGeometry:{residency:"resident"},materialOwner:{residency:"resident"}},planarBoundaries:{count:0,strideBytes:PLANAR_BOUNDARY_PATCH_BYTES,generation:1,records:{buffer:{size:0}}}}},
  scene,sceneArenaBuffer:{size:SVO_DRY_SCENE_ARENA_LAYOUT.sizeBytes},surfaceTextures,
  primitiveCandidateArena:packSvoPrimitiveCandidateArena(primitiveRecords,primitiveCandidates),
  device:{queue:{}},publishedLighting:{publish(){}},lightingOptions:{},shadingPath:"inline",bindGroup:{},
  backdropTerrainRequested(){return false;},latticeVisibilityRequested(){return false;},writeSurfaceMeshHostWords(){},publishImportedSurfaces(){},writeScenePrimitiveOverflowPublication(){},writeParams(){},
  rebuild(){groups++;this.bindGroup={view:surfaceTextures.view};},
 });
 assert.equal(renderer.publishScene({...scene,renderRevision:2,surfaceTextureAtlas:atlasB}),true);
 assert.equal(groups,1);assert.equal(renderer.bindGroup.view,surfaceTextures.view);
 assert.equal(renderer.publishScene({...renderer.scene,renderRevision:3}),true);
 assert.equal(groups,1,"same atlas retains compatible binding resources");
});
