import assert from "node:assert/strict";
import test from "node:test";
import {sceneShapeWgsl,sceneShapeCode} from "../lib/core/scene-shape";
import {fluidRigidBoundaryWGSL} from "../lib/core/fluid-rigid-boundary.wgsl";
import {boundingRadius} from "../lib/core/rigid-body";
import {uniformMixedSolidWGSL} from "../lib/methods/uniform/uniform-mixed-solid.wgsl";
import {narrowBandRigidContactSource} from "../lib/methods/uniform/uniform-particle-contact.wgsl";
import {narrowBandBodyWindowWGSL} from "../lib/methods/uniform/uniform-particle-body-window.wgsl";
import {withUniformDevice} from "./helpers/uniform-geometric";
import {readMixedBuffer} from "./helpers/uniform-mixed-native-fields";

function fn(source:string,name:string):string{
 const start=source.indexOf(`fn ${name}(`);assert.ok(start>=0,name);let depth=0;
 for(let i=source.indexOf("{",start);i<source.length;i++){if(source[i]==="{")depth++;if(source[i]==="}"&&--depth===0)return source.slice(start,i+1);}throw Error(name);
}

test("particle body windows preserve contact and containment, including contacts into excluded later bodies",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("particle body window equivalence",async device=>{
  const count=32768,solid=uniformMixedSolidWGSL(0);
  const functions=["umQuatRotate","umBodyLocal","umBodyNear","umInsideBody","umBodyDistance","umBodyVelocity","umBodyAt","umBodyAtCandidates","umBodyCandidates"].map(n=>fn(solid,n)).join("\n");
  const body=device.createBuffer({size:12*128,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  const settings=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  const output=device.createBuffer({size:count*64,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  try{
   const baseline=narrowBandRigidContactSource().replace("fn nbRigidContact(","fn referenceContact(");
   const candidate=narrowBandRigidContactSource("nbMoveBodyMask(point)").replace("struct NBRigidContact{position:vec3f,velocity:vec3f}","");
   const module=device.createShaderModule({code:`
${sceneShapeWgsl()}\n${fluidRigidBoundaryWGSL}
struct Body{positionShape:vec4f,dimensions:vec4f,orientation:vec4f,linearVelocity:vec4f,angularVelocity:vec4f,pad0:vec4f,pad1:vec4f,pad2:vec4f};
@group(0) @binding(0) var<uniform> umSolidBodies:array<Body,12>;
@group(0) @binding(1) var<uniform> settings:vec4u;
@group(0) @binding(2) var<storage,read_write> result:array<vec4u>;
struct Params{hDt:vec4f};const params=Params(vec4f(.025,.035,.02,1.0/60.0));
fn umBodyCount()->u32{return settings.x;}
fn umSolidTraceWorld(q:vec3f)->vec3f{return vec3f(-.13,.07,-.02)+q*params.hDt.xyz;}
${functions}
// Each test invocation owns an independently staged tile. Production
// stages the same masks in the existing workgroup window before its barrier.
var<private> nbWinFlags:array<u32,64>;var<private> homeTile:vec3i;
fn nbWinIndex(t:vec3i)->u32{return select(64u,0u,settings.w==0u&&all(t==homeTile));}
${narrowBandBodyWindowWGSL}
${baseline}\n${candidate}
@compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) g:vec3u){
 var q=vec3f(vec3i(i32(g.x&127u)-64,i32((g.x>>7u)&15u)-4,i32(g.x>>11u)-8))+vec3f(.37,.11,.83);
 if(settings.z==1u){q=(vec3f(.1,.2,0)-vec3f(-.13,.07,-.02))/params.hDt.xyz;}
 homeTile=vec3i(floor(q))>>vec3u(2u);let mask=nbBodyTileMask(homeTile);nbWinFlags[0]=mask<<8u;
 let v=vec3f(.3,-.2,.1);let inset=select(.25,.5,(g.x&1u)!=0u);
 let a=referenceContact(q,v,inset);let b=nbRigidContact(q,v,inset);
 result[4u*g.x]=vec4u(bitcast<vec3u>(a.position),bitcast<u32>(umBodyAt(umSolidTraceWorld(q))));
 result[4u*g.x+1u]=vec4u(bitcast<vec3u>(b.position),bitcast<u32>(nbMoveBodyAt(q)));
 result[4u*g.x+2u]=vec4u(bitcast<vec3u>(a.velocity),mask);
 result[4u*g.x+3u]=vec4u(bitcast<vec3u>(b.velocity),mask);
}`});
   const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
   assert.equal(errors.length,0,errors.map(m=>`${m.lineNum}:${m.linePos} ${m.message}`).join("\n"));
   const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"probe"}});
   const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:body}},{binding:1,resource:{buffer:settings}},{binding:2,resource:{buffer:output}}]});
   const words=new Float32Array(12*32),shapes=["box","sphere","capsule","cylinder","cup"] as const;
   for(let i=0;i<12;i++){
    const shape=shapes[i%5]!,d=shape==="box"?{x:.8,y:.23,z:.14}:{x:.13,y:.31,z:.025},at=32*i,angle=.19*i;
    words.set([i<2?0:(i-6)*.22,.15,i<2?0:(i%3-1)*.17,sceneShapeCode(shape)],at);
    words.set([d.x,d.y,d.z,boundingRadius({shape,dimensions_m:d} as Parameters<typeof boundingRadius>[0])],at+4);
    words.set([Math.cos(angle),0,Math.sin(angle),0],at+8);words.set([i*.13,-i*.07,.21-i*.02,0],at+12);words.set([.12*i,.23,.17,0],at+16);
   }
   words.set(words.subarray(0,12),32);
   for(const [bodies,interior,chain,outside] of [[0,0,0,0],[1,0,0,0],[2,0,0,0],[12,0,0,0],[12,1,0,0],[12,0,0,1],[2,0,1,0]]){
    words[3]=interior?-sceneShapeCode("box")-1:sceneShapeCode("box");
    if(chain){words.fill(0);for(let i=0;i<2;i++){const at=32*i,r=i===0?.6:.15;words.set([i*.7,.2,0,sceneShapeCode("sphere"),r,r,r,r,1,0,0,0],at);}}
    device.queue.writeBuffer(body,0,words);device.queue.writeBuffer(settings,0,new Uint32Array([bodies!,0,chain!,outside!]));
    const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(count/64);p.end();device.queue.submit([e.finish()]);
    const data=await readMixedBuffer(device,output),bits=new Uint32Array(data.buffer);
    for(let i=0;i<bits.length;i+=8)for(let j=0;j<4;j++)assert.equal(bits[i+j+4],bits[i+j],`roster ${bodies}, interior ${interior}, chain ${chain}, outside ${outside}, word ${i+j}`);
    if(chain){assert.equal(bits[11]!&2,0,"later body starts outside the cached candidate set");assert.ok(data[0]!*.025-.13<.6,"the later body must affect the corrected position");}
   }
  }finally{body.destroy();settings.destroy();output.destroy();}
 });
});
