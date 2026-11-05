import assert from 'node:assert/strict';
import test from 'node:test';
import {wetClothSource} from '../lib/cloth/scenes/sources';
import {sourceMovingSolidBodies,sourceMovingSolidPoses} from '../lib/cloth/source-moving-solids';
import {SourceMotionRuntime,sourceFieldBounds} from '../lib/cloth/source-motion';
import {sourceFieldDistance} from '../lib/cloth/source-fields';
import {initializeRigidBody,quaternionRotate} from '../lib/core/rigid-body';
import {sceneShape} from '../lib/core/scene-shape';
import {svoPrimitiveForRigidBody,sampleSvoPrimitive} from '../lib/svo/contracts/svo-primitive-abi';
import type {WetClothSourceProgram} from '../lib/cloth/source-types';
import type {ClothSupportScript} from '../lib/cloth/types';
import {evaluateSvoFieldProgram,packSvoFieldProgramArena,unpackSvoFieldProgram,svoFieldProgramExtent_m} from '../lib/svo/features/scene-publication/svo-field-program';

test('released towel clamp shape and scripted pose agree with source and shared SVO distance',()=>{
 const source=wetClothSource('wetcloth-source-tighten-towel'),program:WetClothSourceProgram={schemaVersion:1,sourceId:source.id,sourceOffset_m:[0,0,0],timeStep_s:source.dt_s,fields:source.fields,scripts:source.scripts,requiredCapabilities:[]};
 const descriptions=sourceMovingSolidBodies(program),bodies=descriptions.map(initializeRigidBody);assert.equal(bodies.length,6);
 const scripts:ClothSupportScript[]=source.scripts.map(s=>({kind:s.kind,group:s.group,start_s:s.start_s,end_s:s.end_s,vector:s.kind==='translate'?s.displacement_m!:s.axis!,angle_rad:s.angle_rad,origin_m:s.origin_m,global:s.global,easing:s.easing,easeIn_s:s.easeStart_s,easeOut_s:s.easeEnd_s}));
 const motion=new SourceMotionRuntime(scripts,{restPositions_m:[],triangles:[],rodIndices:[],rodOffsets:[0]},program.fields);
 for(let i=0;i<3000;i++)motion.step(source.dt_s,i*source.dt_s);
 const moved=sourceMovingSolidPoses(program,motion.fields,bodies);
 for(let i=0;i<moved.length;i++){
  const body=moved[i]!,shape=sceneShape(body.description.shape),primitive=svoPrimitiveForRigidBody({...body.description,position_m:body.position_m,orientation:body.orientation},i,i);
  for(let j=0;j<100;j++){
   const p={x:(Math.sin(j*17)*1.4)*body.description.dimensions_m.x,y:Math.cos(j*11)*.002,z:Math.sin(j*7)*1.4*body.description.dimensions_m.z},rot=quaternionRotate(body.orientation,p),world={x:rot.x+body.position_m.x,y:rot.y+body.position_m.y,z:rot.z+body.position_m.z};
   const expected=sourceFieldDistance(motion.fields,i,[world.x,world.y,world.z]);
   assert.ok(Math.abs(shape.distance_m(body.description.dimensions_m,p)-expected)<1e-12);
   assert.ok(Math.abs(sampleSvoPrimitive(primitive,world).signedDistance_m-expected)<1e-10);
  }
 }
});

test('rounded-box field tape survives packing and keeps exact zero-thickness core',()=>{
 const program={result:0,ops:[{op:'rounded-box' as const,out:0,parameters:{a:.0045,b:0,c:.02974,d:.0005}}]};
 assert.ok(svoFieldProgramExtent_m(program).every((v,a)=>Math.abs(v-[.005,.0005,.03024][a]!)<1e-12));
 const restored=unpackSvoFieldProgram(packSvoFieldProgramArena([program]),0);
 for(const p of [{x:0,y:0,z:0},{x:.0045,y:.0005,z:.02},{x:.0048,y:.0004,z:.02974}]){
  const expected=Math.hypot(Math.max(Math.abs(p.x)-.0045,0),p.y,Math.max(Math.abs(p.z)-.02974,0))-.0005;
  assert.ok(Math.abs(evaluateSvoFieldProgram(program,p).distance_m-expected)<1e-12);
  assert.ok(Math.abs(evaluateSvoFieldProgram(restored,p).distance_m-expected)<1e-8);
 }
});

test('rounded plate GPU physical SDF and volume agree with CPU',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const {withUniformDevice}=await import('./helpers/uniform-geometric');
 const {readMixedBuffer}=await import('./helpers/uniform-mixed-native-fields');
 const {sceneShapeWgsl}=await import('../lib/core/scene-shape');
 await withUniformDevice('exact source rounded-plate shape',async device=>{
  const module=device.createShaderModule({code:sceneShapeWgsl()+`\n@group(0) @binding(0) var<storage,read_write> out:array<vec4f>;
  @compute @workgroup_size(64) fn run(@builtin(global_invocation_id) gid:vec3u){if(gid.x>=256u){return;}let j=f32(gid.x);let p=vec3f((j-128.0)*.00007,(j%7.0-3.0)*.0003,(j%13.0-6.0)*.006);let d=vec3f(.0045,.0005,.02974);out[gid.x]=vec4f(rigidShapeDistance(5,d,p),rigidShapeSupportRadius(5,d,normalize(vec3f(1,2,3))),rigidShapeBoundingRadius(5,d),rigidShapeVolume(5,d));}`});
  assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==='error'),[]);
  const buffer=device.createBuffer({size:256*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  try{
   const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'run'}}),group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer}}]}),encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(4);pass.end();device.queue.submit([encoder.finish()]);
   const data=await readMixedBuffer(device,buffer),shape=sceneShape('rounded-plate'),d={x:.0045,y:.0005,z:.02974},n={x:1/Math.sqrt(14),y:2/Math.sqrt(14),z:3/Math.sqrt(14)};
   for(let j=0;j<256;j++){const expected=[shape.distance_m(d,{x:(j-128)*.00007,y:(j%7-3)*.0003,z:(j%13-6)*.006}),shape.supportRadius_m(d,n),shape.boundingRadius_m(d),shape.volume_m3(d)];expected.forEach((v,a)=>assert.ok(Math.abs(v-data[4*j+a]!)<Math.max(a===0?5e-9:1e-16,Math.abs(v)*3e-6),`sample ${j} component ${a}: CPU ${v}, GPU ${data[4*j+a]}`));}
  }finally{buffer.destroy();}
 });
});

test('complete authored towel clamp sweep stays within the released source boundary',()=>{
 const source=wetClothSource('wetcloth-source-tighten-towel'),scripts:ClothSupportScript[]=source.scripts.map(s=>({kind:s.kind,group:s.group,start_s:s.start_s,end_s:s.end_s,vector:s.kind==='translate'?s.displacement_m!:s.axis!,angle_rad:s.angle_rad,origin_m:s.origin_m,global:s.global,easing:s.easing,easeIn_s:s.easeStart_s,easeOut_s:s.easeEnd_s}));
 const motion=new SourceMotionRuntime(scripts,{restPositions_m:[],triangles:[],rodOffsets:[0],rodIndices:[]},source.fields),lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
 for(let step=0;step<60000;step++){motion.step(source.dt_s,step*source.dt_s);for(let index=0;index<6;index++){const bounds=sourceFieldBounds(motion.fields,index);for(let a=0;a<3;a++){lo[a]=Math.min(lo[a]!,bounds.min[a]!);hi[a]=Math.max(hi[a]!,bounds.max[a]!);}}}
 const boundary=source.fields[7]!;for(let a=0;a<3;a++){assert.ok(lo[a]!>=boundary.center_m![a]!-boundary.halfExtents_m![a]!-boundary.roundingRadius_m!-1e-8);assert.ok(hi[a]!<=boundary.center_m![a]!+boundary.halfExtents_m![a]!+boundary.roundingRadius_m!+1e-8);}
 assert.ok(hi[1]!>.17,'full fold/lift/twist sweep exercised');
});
