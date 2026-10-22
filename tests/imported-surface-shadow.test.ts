import assert from "node:assert/strict";
import test from "node:test";
import { importedSurfaceShadowInstances } from "../lib/svo/features/lighting-visibility/imported-surface-shadow";
import { SvoRasterAo } from "../lib/svo/features/lighting-visibility/svo-raster-ao";

test("native shadows group the complete instance population independently of camera LOD", () => {
  const plan = importedSurfaceShadowInstances([
    {asset:"tree",index:8}, {asset:"rock",index:2}, {asset:"tree",index:0xffffffff},
    {asset:"fern",index:0}, {asset:"rock",index:51},
  ], asset => ({tree:12,rock:3,fern:.5})[asset]!);
  assert.deepEqual(plan.groups, [
    {asset:"tree",firstInstance:0,instanceCount:2},
    {asset:"rock",firstInstance:2,instanceCount:2},
    {asset:"fern",firstInstance:4,instanceCount:1},
  ]);
  assert.deepEqual([...plan.records].filter((_,i)=>i%2===0), [8,0xffffffff,2,51,0]);
  assert.deepEqual([...new Float32Array(plan.records.buffer)].filter((_,i)=>i%2===1), [12,12,3,3,.5]);
  assert.equal(plan.records.byteLength, 5*8);
  assert.deepEqual(importedSurfaceShadowInstances([], ()=>{throw Error("empty has no assets");}).groups, []);
});

test("native shadow records reject malformed identities and unbounded casters", () => {
  for (const index of [-1,.1,NaN,Infinity,0x100000000])
    assert.throws(()=>importedSurfaceShadowInstances([{asset:"a",index}],()=>1), /identity/);
  for (const radius of [0,-1,NaN,Infinity])
    assert.throws(()=>importedSurfaceShadowInstances([{asset:"a",index:0}],()=>radius), /radius/);
});

test("native sun and spot casters share the settled cache and invalidate on membership changes", () => {
  let nativeDraws=0, shadowPasses=0, writes=0, cacheClears=0;
  const pass={setPipeline(){},setBindGroup(){},dispatchWorkgroups(){},drawIndirect(){},end(){}};
  const encoder={beginComputePass:()=>pass,beginRenderPass:()=>{shadowPasses++;return pass;},clearBuffer:()=>cacheClears++};
  const state={};
  const ao=Object.assign(Object.create(SvoRasterAo.prototype), {
    device:{queue:{writeBuffer:()=>writes++}}, generation:7, shadowLayerCount:4,
    meshState:state, cacheGroup:{}, cache:{}, params:{}, ao:{width:8,height:8},
    shadowGroups:[{},{},{},{}], shadowViews:[{},{},{},{}],
  });
  const pipelines:{shadow:object;nativeShadow?:object;terrain:object;casters:object;ao:object}={shadow:{},nativeShadow:{},terrain:{},casters:{},ao:{}};
  const render=(key:string)=>ao.encode(encoder,pipelines,{},{},{},state,64,false,true,undefined,undefined,key,()=>nativeDraws++);
  render("initial"); assert.equal(nativeDraws,4,"both sun maps and both spots receive all native groups");
  render("initial"); assert.equal(nativeDraws,4); assert.equal(cacheClears,1);
  ao.invalidateNativeGeometry(); assert.equal(writes,4,"each light layer receives the new generation");
  render("initial"); assert.equal(nativeDraws,8,"same settled mesh key cannot hide changed native geometry");
  render("edited"); assert.equal(nativeDraws,12,"new mesh generation redraws native and edited voxel casters together");
  assert.equal(shadowPasses,12);
  delete pipelines.nativeShadow;
  render("reference"); assert.equal(nativeDraws,12,"reference proxy shadow path never double draws native geometry");
});

test("GPU native caster bounds retain sun/spot edge intersections and reject only disjoint geometry", {skip:!process.env.WEBGPU_NODE_MODULE}, async () => {
  const {withUniformDevice}=await import("./helpers/uniform-geometric");
  const {readMixedBuffer}=await import("./helpers/uniform-mixed-native-fields");
  const {importedSurfaceShadowWGSL}=await import("../lib/svo/features/lighting-visibility/imported-surface-shadow");
  // Simple analytic projections isolate the production caster decision from
  // camera state. Full production projections also compile in the scene smoke.
  const cases=[
    [0,0,5,1, 0,1,1], [10.5,0,5,1, 0,1,1], [12,0,5,1, 0,1,0],
    [0,0,12,1, 0,1,0], [0,0,5,1, 0,0,0],
    [0,0,5,1, 0,2,1], [5,0,5,1, 0,2,0],
    [0,0,5,1, 2,1,1], [0,0,-3,1, 2,1,0], [0,0,22,1, 2,1,0],
    [0,0,.3,1, 2,1,1], [5.5,0,5,1, 2,1,1], [12,0,5,1, 2,1,0],
  ];
  await withUniformDevice("Native shadow caster bounds",async device=>{
    const input=device.createBuffer({size:cases.length*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const output=device.createBuffer({size:cases.length*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    const packed=new Float32Array(cases.length*8);
    cases.forEach((c,i)=>packed.set(c.slice(0,6),i*8));device.queue.writeBuffer(input,0,packed);
    const module=device.createShaderModule({code:`
struct Spot {positionNear:vec4f,directionTan:vec4f,rangeRadiusIndex:vec4f}
struct Lighting {control:vec4u,depthCell:vec4f,spots:array<Spot,2>}
struct Redraw {minimum:vec4u}
struct Primitive {orientation:vec4f}
var<private> rasterLighting:Lighting;var<private> rasterShadowRedraw:Redraw;
fn dryPrimitive(i:u32)->Primitive{return Primitive(vec4f(0,0,0,1));}
fn svoPrimitiveCenter_m(p:Primitive)->vec3f{return vec3f(0);}
fn svoQuaternionRotate(q:vec4f,p:vec3f)->vec3f{return p;}
fn rasterShadowPosition(p:vec3f,l:u32)->vec3f{return p/10.;}
fn rasterSpotClip(p:vec3f,s:Spot)->vec4f{return vec4f(p,1);}
fn rasterShadowRedrawWhole()->bool{return rasterShadowRedraw.minimum.w==1u;}
fn rasterShadowRedrawCentre()->vec3f{return vec3f(0,0,5);}
fn rasterShadowRedrawHalf()->vec3f{return vec3f(1);}
fn rasterShadowRectangle(c:vec3f,h:vec3f,l:u32)->vec4f{
 if(l<2u){return vec4f(c.xy/10.,h.xy/10.);}
 let r=length(h);if(c.z-r<=.1){return vec4f(0,0,1,1);}
 let lo=min((c.xy-r)/(c.z-r),(c.xy-r)/(c.z+r));
 let hi=max((c.xy+r)/(c.z-r),(c.xy+r)/(c.z+r));return vec4f((lo+hi)*.5,(hi-lo)*.5);
}
${importedSurfaceShadowWGSL}
struct Case {centreRadius:vec4f,layerMode:vec4f}
@group(0) @binding(0) var<storage,read> cases:array<Case>;
@group(0) @binding(1) var<storage,read_write> result:array<u32>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){
 let c=cases[id.x];rasterLighting.depthCell=vec4f(10,0,0,0);
 rasterLighting.spots[0]=Spot(vec4f(0,0,0,.1),vec4f(0,0,1,1),vec4f(20,0,0,0));
 rasterShadowRedraw.minimum.w=u32(c.layerMode.y);
 result[id.x]=select(0u,1u,rasterNativeCasterVisible(c.centreRadius.xyz,c.centreRadius.w,u32(c.layerMode.x)));
}`});
    const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:input}},{binding:1,resource:{buffer:output}}]});
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();
    pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(cases.length);pass.end();device.queue.submit([encoder.finish()]);
    try {assert.deepEqual([...new Uint32Array((await readMixedBuffer(device,output)).buffer)],cases.map(c=>c[6]));}
    finally {input.destroy();output.destroy();}
  });
});

test("GPU shadow cache gate zeroes and restores every indirect group without changing geometry", {skip:!process.env.WEBGPU_NODE_MODULE}, async () => {
  const {withUniformDevice}=await import("./helpers/uniform-geometric");
  const {readMixedBuffer}=await import("./helpers/uniform-mixed-native-fields");
  const {importedSurfaceShadowGateWGSL}=await import("../lib/svo/features/lighting-visibility/imported-surface-shadow");
  await withUniformDevice("Native shadow cache gate",async device=>{
    const redraw=device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    const records=new Uint32Array([300,0,30,0,0,245, 90,0,330,0,0,1]);
    const args=device.createBuffer({size:records.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC|GPUBufferUsage.INDIRECT});
    device.queue.writeBuffer(args,0,records);
    const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module:device.createShaderModule({code:importedSurfaceShadowGateWGSL}),entryPoint:"main"}});
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:redraw}},{binding:1,resource:{buffer:args}}]});
    try {
      for(const mode of [1,0,2,3,0,1]){
        device.queue.writeBuffer(redraw,0,new Uint32Array([0,0,0,mode,0,0,0,0]));
        const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();device.queue.submit([encoder.finish()]);
        const expected=records.slice();expected[1]=mode?245:0;expected[7]=mode?1:0;
        assert.deepEqual(new Uint32Array((await readMixedBuffer(device,args)).buffer),expected);
      }
    } finally {redraw.destroy();args.destroy();}
  });
});

test("moving a light invalidates cached native shadows while identical light publications retain them", () => {
  let writes=0;
  const shadow=Object.assign(Object.create(SvoRasterAo.prototype),{
    device:{queue:{writeBuffer:()=>writes++}},generation:0,key:"",shadowLayerCount:2,
    shadow:{depthOrArrayLayers:2},params:{},coarseAoStrength:0,aoContrast:1,
  });
  const update=(direction:number[])=>shadow.update([-40,0,-48],[80,35,96],[.125,.125,.125],direction);
  update([0,1,0]);assert.equal(shadow.generation,1);assert.equal(writes,2);
  update([0,1,0]);assert.equal(shadow.generation,1);assert.equal(writes,2);
  update([1,1,0]);assert.equal(shadow.generation,2);assert.equal(writes,4);
});
