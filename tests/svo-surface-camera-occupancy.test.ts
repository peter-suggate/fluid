import assert from "node:assert/strict";
import test from "node:test";
import { svoSurfaceMeshWGSL } from "../lib/svo/features/primary-visibility/svo-surface-mesh";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";

test("GPU camera occupancy distinguishes imported proxy air from real solids and missing evidence", {skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
  const source=svoSurfaceMeshWGSL(3,128);
  const point=source.slice(source.indexOf("fn meshCameraInsideSampledSource"),source.indexOf("// Whether the lattice-aligned cube"));
  await withUniformDevice("camera proxy occupancy",async device=>{
    const output=device.createBuffer({size:12*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    try{
      const module=device.createShaderModule({code:`
struct Mapping{worldOrigin:vec3f,cellSize:vec3f}
struct Dry{metadata:vec4u,mapping:Mapping}
struct Record{index:u32}
struct Field{distance_m:f32}
struct Region{identity:u32}
const SVO_KIND_FIELD_PROGRAM=9u;const DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET=16u;
var<private> dry:Dry;var<private> identity:u32;var<private> cuts:u32;
var<private> distance:f32;var<private> magic:u32;var<private> secondInside:bool;
fn meshRegionAt(p:vec3f)->Region{return Region(identity);}
fn sceneIdentitySolid(i:u32)->bool{return i!=0u;}
fn sceneIdentityMaterial(i:u32)->u32{return i;}
fn drySampledMaterial(m:u32)->bool{return m==2u;}
fn svoSculptCutCount()->u32{return cuts;}
fn svoSculptCutDistance(p:vec3f)->f32{return select(1e20,-.1,cuts==2u);}
fn dryPrimitive(i:u32)->Record{return Record(i);}
fn svoPrimitiveKind(r:Record)->u32{return SVO_KIND_FIELD_PROGRAM;}
fn svoPrimitiveMaterialId(r:Record)->u32{return 2u;}
fn svoPrimitiveFieldProgramReference(r:Record)->u32{return 16u+r.index;}
fn svoPrimitiveLocalPoint(r:Record,p:vec3f)->vec3f{return p;}
fn drySceneWord(i:u32)->u32{return magic;}
fn svoFieldProgramReferenceSample(r:u32,p:vec3f)->Field{return Field(select(distance,-.1,r==17u&&secondInside));}
${point}
@group(0) @binding(0) var<storage,read_write> output:array<f32>;
@compute @workgroup_size(1) fn main(){
  dry.metadata.x=1u;dry.mapping.cellSize=vec3f(1);distance=.1;magic=0xfffffffeu;
  identity=0u;output[0]=f32(meshPointSolid(vec3f(0)));
  identity=1u;output[1]=f32(meshPointSolid(vec3f(0)));
  identity=2u;output[2]=f32(meshPointSolid(vec3f(0)));
  distance=-.1;output[3]=f32(meshPointSolid(vec3f(0)));
  distance=0;output[4]=f32(meshPointSolid(vec3f(0)));
  distance=1e20;output[5]=f32(meshPointSolid(vec3f(0)));
  distance=.1;dry.metadata.x=0u;output[6]=f32(meshPointSolid(vec3f(0)));
  dry.metadata.x=1u;cuts=1u;output[7]=f32(meshPointSolid(vec3f(0)));
  cuts=0u;magic=0u;output[8]=f32(meshPointSolid(vec3f(0)));
  magic=0xfffffffeu;dry.metadata.x=2u;secondInside=true;output[9]=f32(meshPointSolid(vec3f(0)));
  cuts=2u;output[10]=f32(meshPointSolid(vec3f(0)));
  identity=8194u;output[11]=f32(meshPointSolid(vec3f(0)));
}`});
      assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
      const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
      const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:output}}]});
      const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();device.queue.submit([encoder.finish()]);
      assert.deepEqual([...await readMixedBuffer(device,output)],[0,1,0,1,1,1,1,0,1,1,0,0]);
    }finally{output.destroy();}
  });
});
