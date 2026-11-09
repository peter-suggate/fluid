import assert from "node:assert/strict";
import test from "node:test";
import { createTokyoRainScene } from "../lib/core/tokyo-rain-scene";
import { buildSvoSceneLights, packSvoLightRecords, SVO_LIGHT_RECORD_WORDS } from "../lib/svo/contracts/svo-light-abi";
import { rasterLocalShadowViews, rasterLocalMapsComplete, rasterShadowFar, rasterShadowTile, RASTER_LIGHTING_PARAM_BYTES, RASTER_LOCAL_LAYOUT_WORD, RASTER_POINT_SHADOW_SIZE } from "../lib/svo/features/lighting-visibility/raster-local-shadows";
import { SvoRasterAo, rasterAoConsumerWGSL, rasterReceiverPlaneWGSL } from "../lib/svo/features/lighting-visibility/svo-raster-ao";

const bounds = { origin: [-16, -2, -32], extent: [32, 16, 64] };
test("Tokyo's unlimited local lights get complete camera-independent shadow coverage without changing radiance", () => {
  const records = packSvoLightRecords(buildSvoSceneLights(createTokyoRainScene()).records), before = records.slice();
  const views = rasterLocalShadowViews(records, bounds);
  assert.deepEqual(records, before);
  assert.equal(views.filter(view => view.pointFace < 0).length, 2);
  assert.equal(views.filter(view => view.pointFace >= 0).length, 28 * 6);
  assert.equal(rasterLocalMapsComplete(records,views),true);
  for (let light = 1; light < records.length / SVO_LIGHT_RECORD_WORDS; light++) {
    const maps = views.filter(view => view.index === light);
    assert.ok(maps.length === 1 || maps.length === 6);
    for (const view of maps) {
      assert.ok(view.far > view.near);
      for (let corner = 0; corner < 8; corner++) {
        const point = bounds.origin.map((v, i) => v + ((corner >> i) & 1) * bounds.extent[i]!);
        assert.ok(Math.hypot(...point.map((p, i) => p - view.position[i]!)) < view.far);
      }
    }
    if (maps.length === 6) assert.deepEqual(maps.reduce((v, map) => v.map((x, i) => x + map.direction[i]!), [0, 0, 0]), [0, 0, 0]);
  }
  assert.equal(RASTER_LIGHTING_PARAM_BYTES % 256, 0);
  assert.ok(RASTER_LIGHTING_PARAM_BYTES >= (RASTER_LOCAL_LAYOUT_WORD + 4) * 4);
});

test("mapped visibility specialization rejects unsupported live light publications",()=>{
  const records=packSvoLightRecords(buildSvoSceneLights(createTokyoRainScene()).records);
  assert.equal(rasterLocalMapsComplete(undefined,[]),false);
  assert.equal(rasterLocalMapsComplete(new Uint32Array(3),[]),false);
  assert.equal(rasterLocalMapsComplete(records,[]),false);
  const unsupported=records.slice();unsupported[SVO_LIGHT_RECORD_WORDS+24]=99;
  assert.equal(rasterLocalMapsComplete(unsupported,rasterLocalShadowViews(unsupported,bounds)),false);
  const invalidSun=records.slice();new Float32Array(invalidSun.buffer).fill(0,4,7);
  assert.equal(rasterLocalMapsComplete(invalidSun,rasterLocalShadowViews(invalidSun,bounds)),false);
  const invalidPoint=records.slice();new Float32Array(invalidPoint.buffer)[SVO_LIGHT_RECORD_WORDS]=NaN;
  assert.equal(rasterLocalMapsComplete(invalidPoint,rasterLocalShadowViews(invalidPoint,bounds)),false);
  assert.equal(rasterLocalMapsComplete(records,rasterLocalShadowViews(records,bounds)),true,"restored lights regain the cached kernel");
});

test("visibility storage packs only fully mapped opaque samples and restores RGB for water shadows",()=>{
  const raster=Object.assign(Object.create(SvoRasterAo.prototype),{mapsComplete:true});
  for(const samples of [1,2,3,31,33,64]){
    raster.setVisibilityLayers(samples,false);assert.equal(raster.storedVisibilityLayers,Math.ceil(samples/3));
    raster.setVisibilityLayers(samples,true);assert.equal(raster.storedVisibilityLayers,samples);
    raster.setVisibilityLayers(samples,false);raster.mapsComplete=false;assert.equal(raster.storedVisibilityLayers,samples);
    raster.mapsComplete=true;
  }
});

test("monochrome visibility is withdrawn when live coverage changes without a projection change",()=>{
  const records=packSvoLightRecords(buildSvoSceneLights(createTokyoRainScene()).records);
  const controls:number[]=[];
  const raster=Object.assign(Object.create(SvoRasterAo.prototype),{
    device:{queue:{writeBuffer:(_buffer:unknown,offset:number,data:ArrayBuffer|Uint32Array)=>{
      if(data instanceof Uint32Array&&offset%RASTER_LIGHTING_PARAM_BYTES===12)controls.push(data[0]!);
    }}},params:{},generation:0,shadowLayerCount:2,shadowResolution:2048,visibilityEnabled:true,
    mapsComplete:false,key:"",updateShadowLayout(){},
  });
  const update=()=>raster.update(bounds.origin,bounds.extent,[.1,.1,.1],[0,1,0],records);
  update();assert.ok(controls.at(-1)!&4);
  const generation=raster.generation;
  // The sun is outside the local-view list. Invalidating its record leaves
  // all local projection parameters unchanged, but invalidates scalar proof.
  new Float32Array(records.buffer).fill(0,4,7);controls.length=0;update();
  assert.equal(raster.generation,generation);assert.ok(controls.length>0);
  assert.ok(controls.every(control=>(control&4)===0));
  new Float32Array(records.buffer).set([0,1,0],4);controls.length=0;update();
  assert.equal(raster.generation,generation);assert.ok(controls.every(control=>(control&4)!==0));
});

test("visibility selects cached-map kernels only with complete live coverage, including inline dispatch",()=>{
  for(const mapsComplete of [false,true])for(const visibilityQueueEnabled of [false,true]){
    const used:string[]=[],indirect:string[]=[];let selected="";
    const pass={setPipeline:(pipeline:string)=>{selected=pipeline;used.push(pipeline);},setBindGroup(){},dispatchWorkgroups(){},dispatchWorkgroupsIndirect(){indirect.push(selected);},end(){}};
    const encoder={beginComputePass:()=>pass,clearBuffer(){},copyBufferToBuffer(){}};
    const pipelines=Object.fromEntries(["aoTiled","classify","finalize","lights","mappedLights","lightEdges","inlineEdges","mappedInlineEdges","exactFinalize","exactEdges","mappedExactEdges"].map(k=>[k,k]));
    const state={};
    const raster=Object.assign(Object.create(SvoRasterAo.prototype),{
      meshState:state,cacheGroup:{},cache:{},generation:1,renderedSettledKey:"ready",renderedSettledGeneration:1,
      mapsComplete,visibilityQueueEnabled,visibilityEnabled:true,visibilityTilingEnabled:true,
      visibilityProducedFrames:0,fullWidth:16,fullHeight:16,ao:{width:8,height:8},aoPassDescriptor:{},
    });
    raster.encode(encoder,pipelines,{},{},{},state,0,false,true,undefined,undefined,"ready");
    assert.ok(used.includes(mapsComplete?"mappedLights":"lights"));
    assert.equal(used.includes(mapsComplete?"lights":"mappedLights"),false);
    if(visibilityQueueEnabled)assert.ok(used.includes(mapsComplete?"mappedExactEdges":"exactEdges"));
    else{
      const inline=mapsComplete?"mappedInlineEdges":"inlineEdges";
      assert.ok(used.includes(inline));assert.ok(!indirect.includes(inline),"inline resolves full-screen coordinates even when visibility tiles are enabled");
    }
  }
});

test("point faces occupy disjoint atlas tiles at both sun resolutions", () => {
  const views = rasterLocalShadowViews(packSvoLightRecords(buildSvoSceneLights(createTokyoRainScene()).records), bounds);
  const spots = views.filter(view => view.pointFace < 0).length;
  for (const size of [2048, 4096]) {
    const locations = new Set<string>();
    views.forEach((view, slot) => {
      const tile = rasterShadowTile(view, slot + 2, spots, size);
      assert.ok(tile.x >= 0 && tile.y >= 0 && tile.x + tile.size <= size && tile.y + tile.size <= size);
      if (view.pointFace >= 0) assert.equal(tile.size, RASTER_POINT_SHADOW_SIZE);
      else assert.deepEqual(tile, {layer: slot + 2, x: 0, y: 0, size});
      const key = `${tile.layer}/${tile.x}/${tile.y}`;
      assert.ok(!locations.has(key));locations.add(key);
    });
    assert.deepEqual(rasterShadowTile(undefined, 0, spots, size), {layer: 0, x: 0, y: 0, size});
  }
});

test("Tokyo batches shadow tiles by physical attachment and keeps camera-independent cache invalidation",()=>{
  const views=rasterLocalShadowViews(packSvoLightRecords(buildSvoSceneLights(createTokyoRainScene()).records),bounds);
  const count=2+views.length,spots=views.filter(v=>v.pointFace<0).length;
  const attachments:number[]=[],rectangles:number[][]=[];let native=0,writes=0;
  const pass={setViewport(){},setScissorRect:(...r:number[])=>rectangles.push(r),setPipeline(){},setBindGroup(){},dispatchWorkgroups(){},drawIndirect(){},end(){}};
  const encoder={beginComputePass:()=>pass,beginRenderPass:(descriptor:any)=>{attachments.push(descriptor.depthStencilAttachment.view);return pass;},clearBuffer(){}};
  const state={};
  const raster=Object.assign(Object.create(SvoRasterAo.prototype),{
    device:{queue:{writeBuffer:()=>writes++}},generation:1,meshState:state,cacheGroup:{},cache:{},params:{},ao:{width:8,height:8},
    shadowLayerCount:count,localViews:views,spotCount:spots,shadowResolution:2048,
    shadowGroups:Array(count).fill({}),shadowViews:Array.from({length:count},(_,i)=>rasterShadowTile(views[i-2],i,spots,2048).layer),
  });
  const pipelines={shadow:{},nativeShadow:{},terrain:{},casters:{},ao:{}};
  const draw=()=>raster.encode(encoder,pipelines,{},{},{},state,64,false,true,undefined,undefined,"settled",()=>native++);
  draw();assert.equal(native,count);assert.deepEqual(attachments,Array.from({length:15},(_,i)=>i));
  for(let i=0;i<count;i++){const t=rasterShadowTile(views[i-2],i,spots,2048);assert.deepEqual(rectangles[i],[t.x,t.y,t.size,t.size]);}
  draw();assert.equal(attachments.length,15,"unchanged world remains cached across camera draws");
  raster.invalidateNativeGeometry();assert.equal(writes,count);draw();assert.equal(attachments.length,30);assert.equal(native,count*2);
});

test("unsupported/malformed local lights stay outside the cache", () => {
  assert.deepEqual(rasterLocalShadowViews(undefined, bounds), []);
  assert.deepEqual(rasterLocalShadowViews(new Uint32Array(3), bounds), []);
  const records = packSvoLightRecords(buildSvoSceneLights(createTokyoRainScene()).records);
  for (const [word, value] of [[0, NaN], [20, -1], [3, -1]]) {
    const invalid = records.slice();new Float32Array(invalid.buffer)[SVO_LIGHT_RECORD_WORDS + word!] = value!;
    assert.ok(!rasterLocalShadowViews(invalid, bounds).some(view => view.index === 1));
  }
  assert.equal(rasterShadowFar([0, 0, 0], {origin: [-1, -1, -1], extent: [2, 2, 2]}), Math.sqrt(3) * 1.001);
});

test("GPU point shadows select all cube faces and clamp filtering inside their atlas tile after resizing", {skip:!process.env.WEBGPU_NODE_MODULE}, async () => {
  const {withUniformDevice}=await import("./helpers/uniform-geometric");
  const {readMixedBuffer}=await import("./helpers/uniform-mixed-native-fields");
  await withUniformDevice("local shadow atlas",async device=>{
    const records=new Uint32Array(3*SVO_LIGHT_RECORD_WORDS);
    const tokyo=packSvoLightRecords(buildSvoSceneLights(createTokyoRainScene()).records);
    for(let i=0;i<3;i++)records.set(tokyo.subarray(SVO_LIGHT_RECORD_WORDS,2*SVO_LIGHT_RECORD_WORDS),i*SVO_LIGHT_RECORD_WORDS);
    const floats=new Float32Array(records.buffer);
    for(let i=0;i<3;i++){floats.set([i*2,i*2,i*2],i*SVO_LIGHT_RECORD_WORDS);floats[i*SVO_LIGHT_RECORD_WORDS+20]=.1;}
    const views=rasterLocalShadowViews(records,bounds),raster=new SvoRasterAo(device);
    const output=device.createBuffer({size:18*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    try{
      raster.update(bounds.origin,bounds.extent,[.1,.1,.1],[0,1,0],records);
      const source=rasterAoConsumerWGSL();
      const declarations=source.slice(source.indexOf("struct RasterSpot"),source.indexOf("@group(2) @binding(21)"));
      const sampling=source.slice(source.indexOf("fn rasterSunDirection"),source.indexOf("fn rasterPcf"));
      const module=device.createShaderModule({code:(declarations+`
${rasterReceiverPlaneWGSL}
@group(2) @binding(1) var rasterSunDepth:texture_depth_2d_array;
@group(2) @binding(2) var<storage,read_write> output:array<vec4f>;
@group(2) @binding(3) var rasterSunSampler:sampler;
struct Uniforms{container:vec4f} var<private> uniforms:Uniforms;
struct Dry{tuningRays0:vec4f} var<private> dry:Dry;
${sampling}
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){
  let view=rasterLighting.spots[id.x];let position=view.positionNear.xyz+view.directionTan.xyz*5.0;
  let slot=rasterSpotSlot(id.x/6u,position);let layer=u32(slot)+2u;
  output[id.x]=vec4f(f32(slot),rasterSpotVisibility(position,-view.directionTan.xyz,u32(slot)),
    rasterLoadShadow(vec2i(-1,-1),layer),rasterLoadShadow(vec2i(512,512),layer));
}`).replaceAll("@group(2)","@group(0)")});
      const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
      const fillModule=device.createShaderModule({code:`
@vertex fn main(@builtin(vertex_index) i:u32)->@builtin(position) vec4f{
 let vertices=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));return vec4f(vertices[i],0,1);
}`});
      const fill=await device.createRenderPipelineAsync({layout:"auto",vertex:{module:fillModule,entryPoint:"main"},depthStencil:{format:"depth32float",depthWriteEnabled:true,depthCompare:"always"}});
      for(const size of [2048,4096]){
        raster.setShadowResolution(size);
        const encoder=device.createCommandEncoder();
        for(let layer=0;layer<raster.shadow.depthOrArrayLayers;layer++){
          const pass=encoder.beginRenderPass({colorAttachments:[],depthStencilAttachment:{view:raster.shadow.createView({dimension:"2d",baseArrayLayer:layer,arrayLayerCount:1}),depthLoadOp:"clear",depthClearValue:1,depthStoreOp:"store"}});
          pass.setPipeline(fill);
          views.forEach((view,slot)=>{const tile=rasterShadowTile(view,slot+2,0,size);if(tile.layer===layer&&slot%2===0){pass.setViewport(tile.x,tile.y,tile.size,tile.size,0,1);pass.setScissorRect(tile.x,tile.y,tile.size,tile.size);pass.draw(3);}});
          pass.end();
        }
        const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[
          {binding:0,resource:{buffer:raster.params,offset:0,size:RASTER_LIGHTING_PARAM_BYTES}},
          {binding:1,resource:raster.shadow.createView({dimension:"2d-array"})},{binding:2,resource:{buffer:output}},{binding:3,resource:raster.sampler}]});
        const pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(18);pass.end();device.queue.submit([encoder.finish()]);
        const values=await readMixedBuffer(device,output);
        for(let slot=0;slot<18;slot++)assert.deepEqual(Array.from(values.slice(slot*4,slot*4+4)),[slot,slot%2,slot%2,slot%2],`face ${slot} at ${size}`);
      }
    }finally{output.destroy();raster.destroy();}
  });
});

test("area emitters get one perspective map when the domain is in front of them, otherwise a soft cube",async()=>{
  const {getScenePreset}=await import("../lib/core/scenes");
  const {RASTER_MAX_SPOT_LIGHTS}=await import("../lib/svo/features/lighting-visibility/raster-spot-lights");
  for(const id of ["bridson-2002-rotating-sphere","bridson-2002-curtain","bridson-2002-tablecloth"]){
    const scene=getScenePreset(id).create(),box=scene.container;
    const domain={origin:[-box.width_m/2,0,-box.depth_m/2],extent:[box.width_m,box.height_m,box.depth_m]};
    const records=packSvoLightRecords(buildSvoSceneLights(scene).records),before=records.slice();
    const views=rasterLocalShadowViews(records,domain);
    assert.deepEqual(records,before,id);
    assert.equal(rasterLocalMapsComplete(records,views),true,`${id}: the softbox must not fall back to per-pixel SVO tracing`);
    const area=views.find(view=>records[view.index*SVO_LIGHT_RECORD_WORDS+24]===4);
    assert.ok(area&&area.pointFace<0&&area.radius>0,id);
    for(let corner=0;corner<8;corner++){
      const delta=domain.origin.map((v,i)=>v+((corner>>i)&1)*domain.extent[i]!-area.position[i]!);
      const axial=delta.reduce((s,v,i)=>s+v*area.direction[i]!,0);
      assert.ok(axial>area.near&&Math.hypot(...delta)<area.far,`${id}: corner ${corner} between the planes`);
      // The WGSL projection's square frustum basis.
      const f=area.direction,h=Math.abs(f[1]!)>.95?[1,0,0]:[0,1,0],x=(a:number[],b:number[])=>[a[1]!*b[2]!-a[2]!*b[1]!,a[2]!*b[0]!-a[0]!*b[2]!,a[0]!*b[1]!-a[1]!*b[0]!];
      const r=x(f,h).map((v,_,a)=>v/Math.hypot(...a)),u=x(r,f),dot=(a:number[])=>delta.reduce((s,v,i)=>s+v*a[i]!,0);
      assert.ok(Math.max(Math.abs(dot(r)),Math.abs(dot(u)))/axial<=area.tanHalfAngle,`${id}: corner ${corner} inside the frustum`);
    }
  }
  // The live Fig 2 studio: the softbox hangs inside the voxel domain, beside
  // walls no single frustum reaches, so it takes a soft cube and the cached-map
  // kernel still covers every light.
  const studio={origin:[-9.44,-.16,-9.44],extent:[18.88,12.32,18.88]};
  const fig2=packSvoLightRecords(buildSvoSceneLights(getScenePreset("bridson-2002-rotating-sphere").create()).records);
  const softbox=rasterLocalShadowViews(fig2,studio);
  assert.deepEqual(softbox.map(view=>[view.index,view.pointFace]),[0,1,2,3,4,5].map(face=>[1,face]));
  const disc=Math.sqrt(4*2.4192*2.4192/Math.PI);
  for(const face of softbox){
    assert.ok(Math.abs(face.penumbraRadius!-disc)<1e-4,"contact hardening uses the emitter's equivalent disc");
    assert.ok(face.radius<.2&&face.near<face.radius,"the caster clearance stays inside the 10 cm softbox proxy");
  }
  assert.equal(rasterLocalMapsComplete(fig2,softbox),true);
  const light=(kind:"rectangleArea"|"sphereArea",position:number[],id:number)=>({kind,position_m:position,direction:[0,-1,0],range_m:0,colorLinear:[1,1,1],intensity:1,
    axisU:[1,0,0],axisV:[0,0,1],halfWidth_m:.5,halfHeight_m:.5,radius_m:.3,lightId:id,ownerId:id,revision:0,sourceKey:`t${id}`});
  const pack=(lights:ReturnType<typeof light>[])=>packSvoLightRecords(lights as never);
  const inside=pack([light("sphereArea",[0,1,0],1)]),insideViews=rasterLocalShadowViews(inside,bounds);
  assert.deepEqual(insideViews.map(view=>view.pointFace),[0,1,2,3,4,5],"a sphere inside the domain takes a cube");
  assert.ok(insideViews.every(view=>Math.abs(view.radius-.303)<1e-6&&Math.abs(view.penumbraRadius!-.3)<1e-6),"its own surface is the caster clearance");
  assert.equal(rasterLocalMapsComplete(inside,insideViews),true);
  const many=pack(Array.from({length:RASTER_MAX_SPOT_LIGHTS+1},(_,i)=>light(i%2?"sphereArea":"rectangleArea",[i,60,0],i+1)));
  const capped=rasterLocalShadowViews(many,bounds);
  assert.equal(capped.filter(view=>view.pointFace<0).length,RASTER_MAX_SPOT_LIGHTS);
  assert.deepEqual(capped.filter(view=>view.pointFace>=0).map(view=>view.index),Array(6).fill(RASTER_MAX_SPOT_LIGHTS),"the emitter past the spot-layer budget takes a cube");
  assert.equal(rasterLocalMapsComplete(many,capped),true);
  const malformed=pack([light("rectangleArea",[0,60,0],1)]);new Float32Array(malformed.buffer)[15]=NaN;
  assert.deepEqual(rasterLocalShadowViews(malformed,bounds),[]);
  assert.equal(rasterLocalMapsComplete(malformed,[]),false,"a malformed emitter keeps exact tracing");
  const mixed=pack([{...light("rectangleArea",[0,60,0],1),kind:"point" as never},light("sphereArea",[0,60,0],2)]);
  const ordered=rasterLocalShadowViews(mixed,bounds);
  assert.deepEqual(ordered.map(view=>view.pointFace),[-1,0,1,2,3,4,5],"spot-layer views precede point faces in the atlas");
});
