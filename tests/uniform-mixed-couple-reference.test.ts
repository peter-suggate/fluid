import assert from "node:assert/strict";
import test from "node:test";
import { sceneShapeWgsl, sceneShapeCode } from "../lib/core/scene-shape";
import { fluidRigidBoundaryWGSL } from "../lib/core/fluid-rigid-boundary.wgsl";
import { boundingRadius } from "../lib/core/rigid-body";
import { GPU_RIGID_BODY_CAPACITY } from "../lib/core/rigid-capacity";
import { uniformMixedSolidWGSL } from "../lib/methods/uniform/uniform-mixed-solid.wgsl";
import { uniformRigidCoverageSource } from "../lib/methods/uniform/uniform-rigid-coverage.wgsl";
import { uniformMixedCoupleWGSL } from "../lib/methods/uniform/uniform-mixed-bodies";
import { uniformMixedCertifiedEntriesWGSL } from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";

function fn(source:string,name:string):string {
 const start=source.indexOf(`fn ${name}(`);assert.ok(start>=0,name);let depth=0;
 for(let i=source.indexOf("{",start);i<source.length;i++){
  if(source[i]==="{")depth++;if(source[i]==="}"&&--depth===0)return source.slice(start,i+1);
 }
 throw new Error(name);
}

// The pre-optimization oracle: a lane an h cell, every body of the roster,
// one global atomic a word and contribution, zeros included.
const reference=/* wgsl */`
@compute @workgroup_size(64) fn reference(@builtin(global_invocation_id) gid:vec3u){
 if(gid.x>=64u*settings.z){return;}
 let o=umAllOwner(gid);if(o.width!=1u){return;}
 let id=vec3i(umOrigin(o));let h=umSolidParams.cellGravity.xyz;let world=umSolidWorldCell(id);
 for(var owner=0u;owner<umBodyCount();owner++){
  if(!umBodyNear(owner,world,length(h))){continue;}
  let pressure=umBodyPressure(id);let base=owner*12u;
  var area=vec3f(0.0);
  for(var axis=0u;axis<3u;axis++){
   var low=id;low[axis]-=1;
   area[axis]=(umBodyFaceFraction(owner,id,axis)-umBodyFaceFraction(owner,low,axis))*h[(axis+1u)%3u]*h[(axis+2u)%3u];
  }
  let reaction=pressure*area*umSolidParams.dimsDt.w;
  let torque=cross(world-umSolidBodies[owner].positionShape.xyz,reaction);
  atomicAdd(&rigidExchange[base],i32(round(reaction.x*1000000.0)));atomicAdd(&rigidExchange[base+1u],i32(round(reaction.y*1000000.0)));atomicAdd(&rigidExchange[base+2u],i32(round(reaction.z*1000000.0)));
  atomicAdd(&rigidExchange[base+3u],i32(round(torque.x*1000000.0)));atomicAdd(&rigidExchange[base+4u],i32(round(torque.y*1000000.0)));atomicAdd(&rigidExchange[base+5u],i32(round(torque.z*1000000.0)));
  atomicStore(&rigidExchange[base+10u],1);
  let fraction=umBodyFraction(owner,id);if(fraction==0.0){continue;}
  let displaced=umBodyOwnerWet(o)*fraction;
  atomicAdd(&rigidExchange[base+6u],i32(round(displaced*65536.0)));
  // No ambient velocity (words 7-9 and 11): dead under pressure coupling.
 }
}`;

test("rigid coupling over the tile's candidate bodies equals the all-body per-cell loop word for word",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("rigid coupling reference",async device=>{
  const solid=uniformMixedSolidWGSL(0);
  const functions=["umQuatRotate","umBodyLocal","umBodyNear","umInsideBody","umBodyDistance","umSolidFaceWorld"].map(n=>fn(solid,n)).join("\n");
  const coverage=fn(uniformRigidCoverageSource(),"umBodyFraction");
  // Tiles of a 10x5x10 block, listed in a scrambled order with gaps: the h list.
  const coords:number[]=[];for(let z=0;z<10;z++)for(let y=0;y<5;y++)for(let x=0;x<10;x++)if((x*7+y*3+z*5)%11!==4)coords.push(x,y,z,0);
  for(let i=coords.length/4-1;i>0;i--){const j=(i*7919)%(i+1);for(let k=0;k<4;k++){const t=coords[4*i+k]!;coords[4*i+k]=coords[4*j+k]!;coords[4*j+k]=t;}}
  const tiles=coords.length/4;
  const code=/* wgsl */`
${sceneShapeWgsl()}\n${fluidRigidBoundaryWGSL}
struct UMRigidBody{positionShape:vec4f,dimensions:vec4f,orientation:vec4f,linearVelocity:vec4f,angularVelocity:vec4f,inverseMassInertia:vec4f,angularMomentumRestitution:vec4f,material:vec4f}
@group(0) @binding(0) var<uniform> umSolidBodies:array<UMRigidBody,${GPU_RIGID_BODY_CAPACITY}>;
@group(0) @binding(1) var<uniform> settings:vec4u;
@group(0) @binding(2) var<storage,read_write> rigidExchange:array<atomic<i32>>;
@group(0) @binding(3) var<storage,read> tileCoords:array<vec4u>;
struct Params{cellGravity:vec4f,dimsDt:vec4f,frame:vec4f};
const umSolidParams=Params(vec4f(.025,.035,.02,0),vec4f(0,0,0,.0125),vec4f(-.4,-.3,-.5,0));
fn umSolidEnabled()->bool{return true;}
fn umBodyCount()->u32{return settings.x;}
fn umSolidWorldCell(p:vec3i)->vec3f{return umSolidParams.frame.xyz+(vec3f(p)+.5)*umSolidParams.cellGravity.xyz;}
struct UMOwner{tile:u32,lane:u32,width:u32,index:u32}
fn umCorner(k:u32,side:u32)->vec3u{return vec3u(k%side,(k/side)%side,k/(side*side));}
fn umAllOwner(gid:vec3u)->UMOwner{let slot=gid.x;if(slot>=64u*settings.z){return UMOwner();}return UMOwner(slot/64u,slot%64u,1u,slot);}
fn umTileCoord(t:u32)->vec3u{return tileCoords[t].xyz;}
fn umOrigin(o:UMOwner)->vec3u{return umTileCoord(o.tile)*4u+umCorner(o.lane,4u);}
fn hash(p:vec3i,k:u32)->u32{var x=(bitcast<u32>(p.x)*73856093u)^(bitcast<u32>(p.y)*19349663u)^(bitcast<u32>(p.z)*83492791u)^(k*2654435761u);x^=x>>13u;x*=0x5bd1e995u;x^=x>>15u;return x;}
fn unit(p:vec3i,k:u32)->f32{return f32(hash(p,k)&0xffffu)/65535.0;}
// Air rows and closed cells read zero pressure; a dry owner zero wet.
fn umBodyPressure(p:vec3i)->f32{if(hash(p,1u)%5u==0u){return 0.0;}return 2000.0*unit(p,2u)-300.0;}
fn umBodyOwnerWet(o:UMOwner)->f32{let p=vec3i(umOrigin(o));if(hash(p,3u)%4u==0u){return 0.0;}return unit(p,4u);}
${functions}\n${coverage}
${uniformMixedCertifiedEntriesWGSL(uniformMixedCoupleWGSL,["couple"],"settings.z")}
${reference}`;
  const module=device.createShaderModule({code});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");assert.deepEqual(errors.map(m=>`${m.lineNum}: ${m.message}`),[]);
  const [candidate,oracle]=await Promise.all(["couple","reference"].map(entryPoint=>device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint}})));
  const bodies=device.createBuffer({size:GPU_RIGID_BODY_CAPACITY*128,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  const settings=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  const exchange=device.createBuffer({size:12*4*GPU_RIGID_BODY_CAPACITY,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  const coordBuffer=device.createBuffer({size:coords.length*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  try{
   device.queue.writeBuffer(coordBuffer,0,new Uint32Array(coords));
   const group=(p:GPUComputePipeline)=>device.createBindGroup({layout:p.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:bodies}},{binding:1,resource:{buffer:settings}},{binding:2,resource:{buffer:exchange}},{binding:3,resource:{buffer:coordBuffer}}]});
   const groups=[group(candidate!),group(oracle!)];
   const shapes=["box","sphere","capsule","cylinder","cup"] as const;
   const run=async(pipeline:GPUComputePipeline,bind:GPUBindGroup,workgroups:number)=>{
    const e=device.createCommandEncoder();e.clearBuffer(exchange);const p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,bind);p.dispatchWorkgroups(workgroups);p.end();device.queue.submit([e.finish()]);
    return Array.from(new Int32Array((await readMixedBuffer(device,exchange)).buffer));
   };
   let nonzero=0;
   for(let trial=0;trial<6;trial++){
    const count=[1,3,24,18,12,24][trial]!;
    const words=new Float32Array(32*GPU_RIGID_BODY_CAPACITY);
    for(let i=0;i<count;i++){
     const shape=shapes[(i+trial)%shapes.length]!,s=(0.5+((i*37+trial*11)%23)/23)*(i%4===1?4:1);
     const d=shape==="box"?{x:.12*s,y:.05*s,z:.03*s}:{x:.04*s,y:.09*s,z:.02*s};
     const at=32*i,angle=.37*i+trial,tilt=.21*trial-.4*i;
     // Spread over the block (x 1 m, y .7 m, z .8 m from the frame); some overlap.
     words.set([-.4+((i*0.137+trial*0.071)%1)*1.0,-.3+((i*0.311+trial*0.13)%1)*.7,-.5+((i*0.529+trial*0.29)%1)*.8,sceneShapeCode(shape)],at);
     words.set([d.x,d.y,d.z,boundingRadius({shape,dimensions_m:d} as Parameters<typeof boundingRadius>[0])],at+4);
     const q=[Math.cos(angle)*Math.cos(tilt),Math.sin(tilt)*.6,Math.sin(angle)*Math.cos(tilt),Math.sin(tilt)*.8],n=Math.hypot(...q);
     words.set(q.map(v=>v/n),at+8);
    }
    // An exact duplicate (overlap) and, in one trial, an inverted shape.
    if(count>3)words.set(words.subarray(0,32),32*2);
    if(trial===4)words[32*5+3]=-sceneShapeCode("box")-1;
    device.queue.writeBuffer(bodies,0,words);device.queue.writeBuffer(settings,0,new Uint32Array([count,0,tiles,0]));
    const got=await run(candidate!,groups[0]!,Math.min(trial%2?7:tiles,tiles));
    const want=await run(oracle!,groups[1]!,tiles);
    assert.deepEqual(got,want,`trial ${trial}, ${count} bodies`);
    nonzero+=want.filter(v=>v!==0).length;
   }
   assert.ok(nonzero>100,`the trials exercise the exchange (${nonzero} nonzero words)`);
  }finally{bodies.destroy();settings.destroy();exchange.destroy();coordBuffer.destroy();}
 });
});
