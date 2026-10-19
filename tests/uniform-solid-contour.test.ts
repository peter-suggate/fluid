import assert from "node:assert/strict";
import test from "node:test";
import {createUniformTroughScene} from "../lib/core/uniform-trough-scenes";
import {sampleSolidWorld,solidWorldForScene} from "../lib/core/solid-world";
import {writeFileSync} from "node:fs";
import {RasterWaterPipeline,surfaceExtractionShader} from "../lib/core/webgpu-water-pipeline";
import {advanceUniform,createUniformSolver,readUniformFields,withUniformDevice} from "./helpers/uniform-geometric";

// This measures the emitted mesh, not raw phi: pressure deliberately extends
// phi into closed portions of a 4h owner. That continuation is not liquid.
test("trough contact mesh survives coarsening and ball impact",{skip:!process.env.WEBGPU_NODE_MODULE},async t=>{
 await withUniformDevice("resolution-independent solid contact",async device=>{
  const h=.0125,scene=createUniformTroughScene("settled-tank",h);
  const level=Math.max(...scene.fluid.initialLiquidVolumes!.map(v=>v.shape==="box"?v.max_m.y:0));
  const uniform=device.createBuffer({size:416,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  const bodies=device.createBuffer({size:1536,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.UNIFORM});
  const words=new Float32Array(104);words.set([3.2,1.2,1.2,0],12);words.set([256,96,96,1],20);device.queue.writeBuffer(uniform,0,words);
  const water=new RasterWaterPipeline(device,"rgba8unorm",uniform,bodies);
  await water.initialize();
  const full=await device.createComputePipelineAsync({layout:water.denseExtractionLayoutForQA,compute:{module:device.createShaderModule({code:surfaceExtractionShader.replace("  classifyCube(vec3i(gid));","  classifyCube(vec3i(gid)+umWallLow());")}),entryPoint:"extractMain"}});
  const solver=await createUniformSolver(device,scene);
  try{
   for(let i=1;i<=120;i++)await advanceUniform(solver,i/60);
   let capture=0;
   const contacts=async(classify:"full"|"windows"="windows")=>{
    water.setVolume(solver.surfaceFieldTexture,solver.denseLevelSetVolumeSource);
    const e=device.createCommandEncoder();water.encodeDenseSurfaceExtractionForQA(e,256,96,96,classify,false,classify==="full"?full:undefined);device.queue.submit([e.finish()]);
    const mesh=await water.readDenseSurfaceExtractionForQA();assert.equal(mesh.vertexAllocator,mesh.vertexCount,"mesh must fit its allocation");
    if(process.env.UNIFORM_CONTACT_MESH_DIR)writeFileSync(`${process.env.UNIFORM_CONTACT_MESH_DIR}/mesh-${capture++}-${classify}.f32`,new Uint8Array(mesh.vertices.buffer));
    const points=new Map<string,number>();
    for(let i=0;i<mesh.vertices.length;i+=8){
     const x=mesh.vertices[i]!,y=mesh.vertices[i+1]!,z=mesh.vertices[i+2]!;
     for(let j=0;j<8;j++)assert.ok(Number.isFinite(mesh.vertices[i+j]),"finite positions and normals");
     // Edge crossings at the cell-centred x/y lattice on the submerged near
     // wall, safely below the impact's moving free surface.
     if(x< -1||x>1||y<.075||y>level-2*h||z<.2||z>.5)continue;
     const ix=(x+1.6)/h-.5,iy=y/h-.5;
     if(Math.abs(ix-Math.round(ix))>1e-4||Math.abs(iy-Math.round(iy))>1e-4)continue;
     const key=`${Math.round(ix)},${Math.round(iy)}`;points.set(key,Math.min(points.get(key)??Infinity,z));
    }
    assert.ok(points.size>1000,`cover the length and depth of the wall: ${classify}, vertices=${mesh.vertexCount}, contacts=${points.size}`);return points;
   };
   const initial=await contacts();let coarseContacts=0;
   const world=solidWorldForScene(scene);
   for(const [key,z] of initial){
    const [x,y]=key.split(",").map(Number);let face=NaN;
    for(let iz=Math.ceil(.8/h);iz<Math.floor(1.1/h);iz++){
     if(sampleSolidWorld(world,[x!,y!,iz],true).solidFraction>0){face=iz*h-.6;break;}
    }
    assert.ok(Math.abs(z-face)<1e-6,`contact ${key}: mesh ${z}, authored voxel face ${face}`);
   }
   solver.injectLiquidBall({centre_m:{x:-.5,y:level+.3,z:0},radius_m:.075});
   for(let step=1;step<=35;step++){
    await advanceUniform(solver,(120+step)/60);
    if(![2,5,20,35].includes(step))continue;
    const points=await contacts(),f=await readUniformFields(device,solver);let peak=0;
    for(const [key,z] of initial){
     assert.ok(points.has(key),`step ${step}: contact ${key} disappeared`);
     peak=Math.max(peak,Math.abs(points.get(key)!-z));
     const [x,y]=key.split(",").map(Number);const iz=Math.floor((z+.6)/h-1e-4);
     if(f.widthAt(x!,y!,iz)===4)coarseContacts++;
    }
    assert.ok(peak<1e-6,`step ${step}: fixed contact moved ${peak} m`);
    t.diagnostic(`step ${step}: ${points.size} contacts, maximum displacement ${peak} m`);
   }
   assert.ok(coarseContacts>100,"wall contacts must actually run at 4h");
   // The optimized window certificates must retain solid crossings even
   // where all 27 coarse liquid nodes are negative.
   assert.deepEqual(await contacts("full"),await contacts("windows"));
  }finally{solver.destroy();water.destroy();uniform.destroy();bodies.destroy();}
 });
});
