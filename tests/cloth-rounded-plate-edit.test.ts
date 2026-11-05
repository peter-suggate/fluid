import test from 'node:test';
import assert from 'node:assert/strict';
import {cloneScene,defaultScene,parseScene,serializeScene,validateScene} from '../lib/core/model';
import {createBodyDescription,initializeRigidBody} from '../lib/core/rigid-body';
import {sceneWithSolidStroke} from '../lib/core/solid-world';
import {visibleToolSurface} from '../lib/core/voxel-editor/surface';
import {publishSceneRigidSurfaces} from '../lib/core/rigid-surface';
import {svoPrimitiveForRigidBody,sampleSvoPrimitive} from '../lib/svo/contracts/svo-primitive-abi';
import {packSvoFieldProgramArena,unpackSvoFieldProgram,evaluateSvoFieldProgram} from '../lib/svo/features/scene-publication/svo-field-program';
import {clothCollidersForRigidBodies,sampleClothCollider} from '../lib/cloth/contact';
function fixture(){const scene=cloneScene(defaultScene);scene.container={width_m:.32,height_m:.32,depth_m:.32,fillFraction:0,top:'open',boundary:'free-space',vessel:'none',fluidWallMode:'free-slip'};scene.voxelDomain={finestCellSize_m:.01,brickSize_cells:8};scene.solidVoxels=[];delete scene.solidBoundary;delete scene.terrain;scene.scenery={palettes:{},nodes:[{id:'empty',kind:'room-shell',materialModel:'room',faces:[],floor:{colorLinear:[1,1,1]},wall:{colorLinear:[1,1,1]},ceiling:{colorLinear:[1,1,1]}}]};scene.rigidBodies=[{...createBodyDescription('rounded-plate',0,.32),position_m:{x:0,y:.2,z:0},orientation:{w:1,x:0,y:0,z:0},dimensions_m:{x:.08,y:.005,z:.1},motion:'static',kinematic:true}];return scene;}
test('rounded plate local cut opens picking, render and cloth collision, preserves detail and save/load/undo',()=>{
 const scene=fixture();assert.deepEqual(validateScene(scene),[]);const ray={origin:{x:0,y:.3,z:0},direction:{x:0,y:-1,z:0}},far={...ray,origin:{...ray.origin,x:.06}},before=visibleToolSurface(scene,ray),untouched=visibleToolSurface(scene,far);assert.ok(before?.rigidBodyId);assert.ok(Math.abs(before.point.y-.205)<1e-7);
 const cut=sceneWithSolidStroke(scene,[{operation:'clear',affectsScenery:true,minimum:[14,0,14],maximumExclusive:[18,32,18]}]);
 assert.equal(visibleToolSurface(cut,ray),undefined);assert.deepEqual(visibleToolSurface(cut,far),untouched);assert.equal(scene.rigidBodies[0]!.cuts,undefined);assert.equal(cut.rigidBodies[0]!.cuts?.length,1);
 const body=cut.rigidBodies[0]!,primitive=svoPrimitiveForRigidBody(body,0,0);assert.ok(sampleSvoPrimitive(primitive,{x:0,y:.2,z:0}).signedDistance_m>0);
 const collider=clothCollidersForRigidBodies([initializeRigidBody(body)])[0]!;assert.ok(sampleClothCollider(collider,[0,.2,0]).distance_m>0);assert.ok(sampleClothCollider(collider,[.06,.2,0]).distance_m<0);
 const saved=parseScene(serializeScene(cut));assert.equal(visibleToolSurface(saved,ray),undefined);assert.deepEqual(visibleToolSurface(saved,far),untouched);assert.deepEqual(visibleToolSurface(scene,ray),before);
 assert.equal(primitive.kind,'field-program');if(primitive.kind==='field-program'){const restored=unpackSvoFieldProgram(packSvoFieldProgramArena([primitive.program!]),0);assert.ok(evaluateSvoFieldProgram(restored,{x:0,y:0,z:0}).distance_m>0);}
});
test('shared cuts use the published moving pose and remain local after a later rotation',()=>{
 const scene=fixture(),id=scene.rigidBodies[0]!.id,q={w:Math.SQRT1_2,x:0,y:0,z:Math.SQRT1_2};publishSceneRigidSurfaces(scene,[{id,position_m:{x:.04,y:.16,z:0},orientation:q}]);
 const ray={origin:{x:.15,y:.16,z:0},direction:{x:-1,y:0,z:0}};assert.ok(Math.abs(visibleToolSurface(scene,ray)!.point.x-.045)<1e-6);
 const cut=sceneWithSolidStroke(scene,[{operation:'clear',affectsScenery:true,minimum:[18,14,14],maximumExclusive:[22,18,18]}]);assert.equal(visibleToolSurface(cut,ray),undefined);
 publishSceneRigidSurfaces(cut,[{id,position_m:{x:0,y:.2,z:0},orientation:{w:1,x:0,y:0,z:0}}]);assert.equal(visibleToolSurface(cut,{origin:{x:0,y:.3,z:0},direction:{x:0,y:-1,z:0}}),undefined);
});

test('native rigid apertures and water extraction use exact local cuts and undo',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const {withUniformDevice}=await import('./helpers/uniform-geometric'),{readMixedBuffer}=await import('./helpers/uniform-mixed-native-fields');
 const {UniformMixedSolid,uniformMixedSolidWGSL}=await import('../lib/methods/uniform/uniform-mixed-solid.wgsl');
 const {GPU_RIGID_BODY_CAPACITY}=await import('../lib/core/rigid-capacity');
 const {surfaceExtractionShader}=await import('../lib/core/webgpu-water-pipeline');
 await withUniformDevice('rounded plate native cut parity',async device=>{
  const buffer=(size:number,usage:number)=>device.createBuffer({size,usage}),params=buffer(272,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST),scratch=buffer(256,GPUBufferUsage.STORAGE),bodies=buffer(GPU_RIGID_BODY_CAPACITY*128,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC),terrain=device.createTexture({size:[1,1],format:'r32float',usage:GPUTextureUsage.TEXTURE_BINDING}),output=buffer(32,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC),scene=fixture(),description=scene.rigidBodies[0]!;
  const values=new Float32Array(68);values.set([32,32,32,.01]);values.set([.01,.01,.01,0],4);values[18]=1;device.queue.writeBuffer(params,0,values);
  const body=new Float32Array(GPU_RIGID_BODY_CAPACITY*32);body.set([0,.2,0,5,.08,.005,.1,.14,1,0,0,0]);device.queue.writeBuffer(bodies,0,body);
  const solid=new UniformMixedSolid(device,{params,scratch,bodies,terrain,coupledTiles:0});
  try{
   const module=device.createShaderModule({code:`const UM_D=vec3u(32);fn umFramePage()->vec3i{return vec3i(0);}fn umWorldSolid(page:vec3i)->bool{return false;}
${uniformMixedSolidWGSL(0,undefined,false,false,false)}
@group(1) @binding(0) var<storage,read_write> out:array<vec4f>;
@compute @workgroup_size(1) fn query(){out[0]=vec4f(umBodyDistance(0u,vec3f(0,.2,0)),umBodyDistance(0u,vec3f(.06,.2,0)),f32(umInsideBody(0u,vec3f(0,.2,0))),f32(umInsideBody(0u,vec3f(.06,.2,0))));out[1]=vec4f(umBodyFraction(0u,vec3i(0,19,0)),0,0,0);}`});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==='error').map(m=>m.message),[]);
   const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}}]}),pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[solid.bindLayout,layout]}),compute:{module,entryPoint:'query'}}),group=device.createBindGroup({layout,entries:[{binding:0,resource:{buffer:output}}]});
   const run=async()=>{const e=device.createCommandEncoder();solid.encodeBodies(e);const p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,solid.bindGroup);p.setBindGroup(1,group);p.dispatchWorkgroups(1);p.end();device.queue.submit([e.finish()]);return readMixedBuffer(device,output);};
   solid.setBodyCuts([description]);const before=await run();assert.ok(before[0]!<0);assert.equal(before[2],1);
   const cut=sceneWithSolidStroke(scene,[{operation:'clear',affectsScenery:true,minimum:[14,0,14],maximumExclusive:[18,32,18]}]);solid.setBodyCuts(cut.rigidBodies);const after=await run();assert.ok(after[0]!>0);assert.equal(after[2],0);assert.equal(after[1],before[1]);assert.equal(after[3],before[3]);assert.equal(after[4],0);assert.ok(before[4]!>0,'actual native cell aperture opens');
   solid.setBodyCuts([description]);assert.deepEqual(await run(),before);
   const extraction=device.createShaderModule({code:surfaceExtractionShader});assert.deepEqual((await extraction.getCompilationInfo()).messages.filter(m=>m.type==='error').map(m=>m.message),[]);
  }finally{solid.destroy();params.destroy();scratch.destroy();bodies.destroy();terrain.destroy();output.destroy();}
 });
});

test('GPU shared SVO field cuts retain original plate geometry outside a rotated cut',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const {withUniformDevice}=await import('./helpers/uniform-geometric'),{readMixedBuffer}=await import('./helpers/uniform-mixed-native-fields');
 const {svoFieldProgramWGSL}=await import('../lib/svo/features/scene-publication/svo-field-program'),{svoProceduralNoiseWGSL}=await import('../lib/svo/features/materials/svo-procedural-material');
 const scene=fixture(),id=scene.rigidBodies[0]!.id;publishSceneRigidSurfaces(scene,[{id,position_m:{x:.04,y:.16,z:0},orientation:{w:Math.SQRT1_2,x:0,y:0,z:Math.SQRT1_2}}]);
 const cut=sceneWithSolidStroke(scene,[{operation:'clear',affectsScenery:true,minimum:[18,14,14],maximumExclusive:[22,18,18]}]),primitive=svoPrimitiveForRigidBody(cut.rigidBodies[0]!,0,0);assert.equal(primitive.kind,'field-program');if(primitive.kind!=='field-program')return;
 const program=primitive.program!,arena=packSvoFieldProgramArena([program]);
 await withUniformDevice('SVO exact local rounded plate cut',async device=>{
  const input=device.createBuffer({size:arena.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),output=device.createBuffer({size:64*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});device.queue.writeBuffer(input,0,arena);
  try{
   const module=device.createShaderModule({code:`@group(0) @binding(0) var<storage,read> words:array<u32>;@group(0) @binding(1) var<storage,read_write> out:array<f32>;fn loadWord(i:u32)->u32{return words[i];}
${svoProceduralNoiseWGSL}
${svoFieldProgramWGSL({functionName:'cutField',loadWord:'loadWord',baseWordExpression:'0u',capacityExpression:'1u'})}
@compute @workgroup_size(64) fn query(@builtin(global_invocation_id) gid:vec3u){let p=vec3f((f32(gid.x)-32.0)*.003,0,0);out[gid.x]=cutField(0u,p).distance_m;}`});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==='error').map(m=>m.message),[]);
   const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'query'}}),group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:input}},{binding:1,resource:{buffer:output}}]}),encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();device.queue.submit([encoder.finish()]);
   const values=await readMixedBuffer(device,output);for(let i=0;i<64;i++)assert.ok(Math.abs(values[i]!-evaluateSvoFieldProgram(program,{x:(i-32)*.003,y:0,z:0}).distance_m)<3e-8);
   assert.ok(values[32]!>0);assert.ok(values[52]!<0);
  }finally{input.destroy();output.destroy();}
 });
});
