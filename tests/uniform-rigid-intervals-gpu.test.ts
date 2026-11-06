import test from 'node:test';
import assert from 'node:assert/strict';
import {uniformRigidIntervalsWGSL} from '../lib/methods/uniform/uniform-rigid-intervals.wgsl';
import {rigidRayUnionLength,rigidRayIntervals} from '../lib/core/rigid-ray-intervals';
import {SCENE_SHAPE_NAMES,sceneShapeCode} from '../lib/core/scene-shape';
import {packRigidBodyCutAtlas,RIGID_BODY_CUT_ATLAS_WIDTH} from '../lib/core/solid-cuts';
import {withUniformDevice} from './helpers/uniform-geometric';

test('native analytic interval sweep matches authoritative CPU all shapes, rotations, cuts, inversions and unions',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice('native rigid interval oracle',async device=>{
  const orientation={w:Math.cos(.31),x:Math.sin(.31)/Math.sqrt(14),y:2*Math.sin(.31)/Math.sqrt(14),z:3*Math.sin(.31)/Math.sqrt(14)};
  const bodies:Parameters<typeof rigidRayIntervals>[0][]=SCENE_SHAPE_NAMES.flatMap(shape=>[false,true].map(inverted=>({shape,dimensions_m:{x:.13,y:shape==='rounded-plate'?.003:.14,z:shape==='cup'?.012:.11},position_m:{x:.02,y:-.01,z:.03},orientation,fluidBoundary:inverted?'interior' as const:undefined,cuts:[{center_m:{x:0,y:.01,z:0},halfExtents_m:{x:.017,y:.021,z:.08},orientation:{w:Math.cos(.17),x:0,y:0,z:Math.sin(.17)}}]})));
  const bodyValues=new Float32Array(bodies.length*12);bodies.forEach((b,i)=>bodyValues.set([b.position_m.x,b.position_m.y,b.position_m.z,b.fluidBoundary==='interior'?-sceneShapeCode(b.shape)-1:sceneShapeCode(b.shape),b.orientation.w,b.orientation.x,b.orientation.y,b.orientation.z,b.dimensions_m.x,b.dimensions_m.y,b.dimensions_m.z,1],i*12));
  const bodyBuffer=device.createBuffer({size:bodyValues.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(bodyBuffer,0,bodyValues);
  const cutValues=packRigidBodyCutAtlas(bodies),cuts=device.createTexture({size:[RIGID_BODY_CUT_ATLAS_WIDTH,1],format:'rgba32float',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});device.queue.writeTexture({texture:cuts},cutValues,{bytesPerRow:16*RIGID_BODY_CUT_ATLAS_WIDTH},[RIGID_BODY_CUT_ATLAS_WIDTH,1]);
  let seed=19;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
  const queries:{o:{x:number;y:number;z:number};d:{x:number;y:number;z:number};mask:number}[]=[];
  for(let body=0;body<bodies.length;body++)for(let k=0;k<64;k++)queries.push({o:{x:.4*(random()-.5),y:.4*(random()-.5),z:.4*(random()-.5)},d:{x:2*random()-1,y:2*random()-1,z:2*random()-1},mask:1<<body});
  for(let k=0;k<64;k++)queries.push({o:{x:.4*(random()-.5),y:.4*(random()-.5),z:.4*(random()-.5)},d:{x:2*random()-1,y:2*random()-1,z:2*random()-1},mask:0x555});
  const queryValues=new Float32Array(queries.length*8);queries.forEach((q,i)=>queryValues.set([q.o.x,q.o.y,q.o.z,q.mask,q.d.x,q.d.y,q.d.z,0],i*8));
  const input=device.createBuffer({size:queryValues.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(input,0,queryValues);
  const output=device.createBuffer({size:4*queries.length,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:4*queries.length,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  const module=device.createShaderModule({code:`struct Body {positionShape:vec4f,orientation:vec4f,dimensions:vec4f};
   @group(0) @binding(0) var<storage,read> umSolidBodies:array<Body>;
   @group(0) @binding(1) var umSolidBodyCuts:texture_2d<f32>;
   @group(0) @binding(2) var<storage,read> input:array<vec4f>;
   @group(0) @binding(3) var<storage,read_write> output:array<f32>;
   fn umQuatRotate(q:vec4f,v:vec3f)->vec3f{return v+2.0*(q.x*cross(q.yzw,v)+cross(q.yzw,cross(q.yzw,v)));}
   ${uniformRigidIntervalsWGSL}
   @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){if(id.x>=${queries.length}u){return;}let o=input[2u*id.x];let d=input[2u*id.x+1u];output[id.x]=umRayUnionLength(u32(o.w),o.xyz,d.xyz,vec2f(-.3,.3));}`});
  const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}}),group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:bodyBuffer}},{binding:1,resource:cuts.createView()},{binding:2,resource:{buffer:input}},{binding:3,resource:{buffer:output}}]});
  try{const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(queries.length/64));pass.end();encoder.copyBufferToBuffer(output,0,read,0,4*queries.length);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);const actual=new Float32Array(read.getMappedRange());
   queries.forEach((q,i)=>{const expected=rigidRayUnionLength(bodies.filter((_,b)=>(q.mask&(1<<b))!==0),q.o,q.d,[-.3,.3]);assert.ok(Math.abs(actual[i]!-expected)<2e-6,`ray ${i}: ${actual[i]} vs ${expected}`);});
  }finally{if(read.mapState==='mapped')read.unmap();for(const b of [bodyBuffer,input,output,read])b.destroy();cuts.destroy();}
 });
});

test('rounded-plate cell and face coverage is exact at the rim, without quadrature failure',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const {uniformRigidQuadratureWGSL}=await import('../lib/methods/uniform/uniform-rigid-intervals.wgsl');
 await withUniformDevice('exact rounded-plate coverage',async device=>{
  const quarter={w:Math.SQRT1_2,x:Math.SQRT1_2,y:0,z:0},identity={w:1,x:0,y:0,z:0};
  // The moving clamp, the released towel clamp's thin plate, a plate turned
  // onto z, and an inverted one; the cut lies far from every queried cell.
  const bodies:Parameters<typeof rigidRayIntervals>[0][]=[
   {shape:'rounded-plate',dimensions_m:{x:.09,y:.015,z:.07},position_m:{x:.001,y:.1353,z:-.002},orientation:identity},
   {shape:'rounded-plate',dimensions_m:{x:.0045,y:.0005,z:.02974},position_m:{x:.0012,y:.0007,z:.0003},orientation:identity},
   {shape:'rounded-plate',dimensions_m:{x:.03,y:.006,z:.02},position_m:{x:-.002,y:.003,z:.0041},orientation:quarter,cuts:[{center_m:{x:.025,y:0,z:0},halfExtents_m:{x:.004,y:.01,z:.01},orientation:identity}]},
   {shape:'rounded-plate',dimensions_m:{x:.03,y:.006,z:.02},position_m:{x:.0003,y:-.0011,z:.002},orientation:identity,fluidBoundary:'interior'},
   // Dyadic, so a face plane lies exactly on the flat top or bottom in f32.
   {shape:'rounded-plate',dimensions_m:{x:.0625,y:.015625,z:.0625},position_m:{x:.5,y:.125,z:0},orientation:identity}];
  const bodyValues=new Float32Array(bodies.length*12);bodies.forEach((b,i)=>bodyValues.set([b.position_m.x,b.position_m.y,b.position_m.z,b.fluidBoundary==='interior'?-sceneShapeCode(b.shape)-1:sceneShapeCode(b.shape),b.orientation.w,b.orientation.x,b.orientation.y,b.orientation.z,b.dimensions_m.x,b.dimensions_m.y,b.dimensions_m.z,1],i*12));
  const bodyBuffer=device.createBuffer({size:bodyValues.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(bodyBuffer,0,bodyValues);
  const cutValues=packRigidBodyCutAtlas(bodies),cuts=device.createTexture({size:[RIGID_BODY_CUT_ATLAS_WIDTH,1],format:'rgba32float',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});device.queue.writeTexture({texture:cuts},cutValues,{bytesPerRow:16*RIGID_BODY_CUT_ATLAS_WIDTH},[RIGID_BODY_CUT_ATLAS_WIDTH,1]);
  let seed=7;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
  type Query={body:number;c:[number,number,number];h:[number,number,number];ray:number;transverse:number;dims:1|2};const queries:Query[]=[];
  for(let body=0;body<bodies.length;body++){const b=bodies[body]!,d=b.dimensions_m,thin=d.y<.001,h:[number,number,number]=thin?[.002,.0015,.0025]:[.01,.0085,.012];
   const reach=b.orientation===quarter?[d.x+d.y,d.z+d.y,d.y]:[d.x+d.y,d.y,d.z+d.y];
   for(let k=0;k<16;k++){const c=[0,1,2].map(a=>[b.position_m.x,b.position_m.y,b.position_m.z][a]!+(2*random()-1)*(reach[a]!+.6*h[a]!)*(k<12?1:.5)) as [number,number,number];
    const ray=k%3,transverse=(ray+1+(k>>2)%2)%3;queries.push({body,c,h,ray,transverse,dims:k%2?1:2});}}
  for(const dims of [1,2] as const)for(const c of [[.011,.1353,-.012],[.2,.1353,0]] as [number,number,number][])queries.push({body:0,c,h:[.01,.0085,.012],ray:dims===2?1:2,transverse:0,dims});
  // The plate is a closed set: a face plane on its flat top (or bottom)
  // sections the whole rectangle, half of each of these faces.
  for(const y of [.140625,.109375])for(const ray of [0,2])queries.push({body:4,c:[.5625,y,.03125],h:[.0625,.0625,.0625],ray,transverse:2-ray,dims:1});
  const values=new Float32Array(queries.length*12);queries.forEach((q,i)=>values.set([...q.c,1<<q.body,...q.h,q.dims,q.ray,q.transverse,0,0],12*i));
  const input=device.createBuffer({size:values.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(input,0,values);
  const output=device.createBuffer({size:8*queries.length,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:8*queries.length,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  const module=device.createShaderModule({code:`struct Body {positionShape:vec4f,orientation:vec4f,dimensions:vec4f};
   @group(0) @binding(0) var<storage,read> umSolidBodies:array<Body>;
   @group(0) @binding(1) var umSolidBodyCuts:texture_2d<f32>;
   @group(0) @binding(2) var<storage,read> input:array<vec4f>;
   @group(0) @binding(3) var<storage,read_write> output:array<vec2f>;
   var<private> umRigidCoverageFailure:vec2u;
   fn umQuatRotate(q:vec4f,v:vec3f)->vec3f{return v+2.0*(q.x*cross(q.yzw,v)+cross(q.yzw,cross(q.yzw,v)));}
   ${uniformRigidIntervalsWGSL}
   ${uniformRigidQuadratureWGSL}
   @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){if(id.x>=${queries.length}u){return;}
    let a=input[3u*id.x];let b=input[3u*id.x+1u];let c=input[3u*id.x+2u];let ray=u32(c.x);let transverse=u32(c.y);let dims=u32(b.w);
    let axes=select(vec2u((ray+1u)%3u,(ray+2u)%3u),vec2u(transverse,3u-ray-transverse),dims==1u);
    let f=umRigidIntegratedFraction(u32(a.w),a.xyz,b.xyz,ray,axes,dims);output[id.x]=vec2f(f,f32(umRigidCoverageFailure.x));}`});
  const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}}),group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:bodyBuffer}},{binding:1,resource:cuts.createView()},{binding:2,resource:{buffer:input}},{binding:3,resource:{buffer:output}}]});
  try{const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(queries.length/64));pass.end();encoder.copyBufferToBuffer(output,0,read,0,8*queries.length);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);const actual=new Float32Array(read.getMappedRange().slice(0));
   let rim=0;
   queries.forEach((q,i)=>{
    // Independent reference: exact CPU ray intervals along the plate normal
    // (the chord is continuous there; a flat face across the rays would make
    // the midpoint rule step), midpoint rule across. Coverage is the plate's
    // volume (or face-plane area) in the box, whichever axis the rays use.
    const normal=bodies[q.body]!.orientation===quarter?2:1,fixed=3-q.ray-q.transverse;
    const ray=q.dims===2?normal:normal===fixed?q.ray:normal,span=q.dims===2?[0,1,2].filter(a=>a!==ray):[3-fixed-ray];
    const n=q.dims===2?256:normal===fixed?20000:4000,dir={x:0,y:0,z:0};(dir as Record<string,number>)['xyz'[ray]!]=1;let sum=0;
    for(let s=0;s<(q.dims===2?n*n:n);s++){const o=[...q.c];o[span[0]!]+=q.h[span[0]!]!*((s%n+.5)/n-.5);if(q.dims===2)o[span[1]!]+=q.h[span[1]!]!*((Math.floor(s/n)+.5)/n-.5);
     sum+=rigidRayUnionLength([bodies[q.body]!],{x:o[0]!,y:o[1]!,z:o[2]!},dir,[-.5*q.h[ray]!,.5*q.h[ray]!])/q.h[ray]!;}
    const expected=sum/(q.dims===2?n*n:n);if(expected>1e-3&&expected<1-1e-3)rim++;
    assert.equal(actual[2*i+1],0,`query ${i}: exact plate coverage must not raise the quadrature failure`);
    // A closed or clear box is exactly so: a rounded sliver would open an aperture.
    if(expected===0||expected===1)assert.equal(actual[2*i],expected,`query ${i}: a box wholly ${expected?'inside':'outside'} the plate`);
    assert.ok(Math.abs(actual[2*i]!-expected)<1.5e-4,`query ${i} (${JSON.stringify(q)}): ${actual[2*i]} vs ${expected}`);
   });
   assert.ok(rim>=25,`only ${rim} queries cut the plate surface`);
  }finally{if(read.mapState==='mapped')read.unmap();for(const b of [bodyBuffer,input,output,read])b.destroy();cuts.destroy();}
 });
});
