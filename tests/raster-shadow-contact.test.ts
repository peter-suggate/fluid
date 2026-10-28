import assert from "node:assert/strict";
import test from "node:test";
import { rasterAoConsumerWGSL, rasterReceiverPlaneWGSL } from "../lib/svo/features/lighting-visibility/svo-raster-ao";

// Exercise the production projection, blocker search and filtered comparison,
// with a real rasterized depth attachment, independently of scene construction.
for(const size of [2048,4096]) test(`GPU ${size} raster shadows preserve leaf-scale contacts without self-shadowing tilted receivers`, {skip:!process.env.WEBGPU_NODE_MODULE}, async()=>{
  const {withUniformDevice}=await import("./helpers/uniform-geometric");
  const {readMixedBuffer}=await import("./helpers/uniform-mixed-native-fields");
  const consumer=rasterAoConsumerWGSL();
  const sampling=consumer.slice(consumer.indexOf("fn rasterSunDirection"),consumer.indexOf("fn rasterOctEncode"));
  // Half extent, global voxel width, light-space normal, blocker separation,
  // and blocker width. Normals are geometric plane normals, not shading detail.
  const fixtures=[
    [32,.25,0,0,1,0,.32], [32,.25,0,0,1,.08,.32],
    [64,.5,0,0,1,0,.32], [64,.5,0,0,1,.08,.32],
    [32,.25,.8,0,.6,0,.32], [32,.25,.8,0,.6,.08,.32],
    [64,.5,.8,0,.6,0,.32], [64,.5,.8,0,.6,.08,.32],
    [32,.25,0,0,1,.8,8], [64,.5,0,0,1,.08,.08], [64,.5,0,0,1,0,.08],
    [32,.5,0,0,1,0,.08,1], [32,.5,0,0,1,.08,.08,1],
    [32,.5,.8,0,.6,0,.08,1], [32,.5,.8,0,.6,.08,.08,1],
    [64,.5,0,0,1,.08,.04],
  ];
  await withUniformDevice("leaf-scale raster shadow contact",async device=>{
    const input=device.createBuffer({size:48,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const output=device.createBuffer({size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    const depth=device.createTexture({size:[size,size,3],format:"depth32float",usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING});
    const definitions=`
struct Fixture { projection:vec4f, normalGap:vec4f, width:vec4f }
@group(0) @binding(0) var<storage,read> fixture:Fixture;
`;
    const caster=device.createShaderModule({code:`${definitions}
@vertex fn vertex(@builtin(vertex_index) i:u32)->@builtin(position) vec4f{
 let points=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));return vec4f(points[i],0,1);
}
@fragment fn fragment(@builtin(position) p:vec4f)->@builtin(frag_depth) f32{
 let n=fixture.normalGap.xyz;
 if(fixture.projection.w==1.){
   let ray=vec3f((p.xy/f32(${size})-.5)*vec2f(2,-2)*.8,1.);
   let base=5.*n.z/dot(n,ray);
   let gap=select(0.,fixture.normalGap.w,all(abs(ray.xy*base)<vec2f(fixture.width.x*.5)));
   let z=(5.-gap)*n.z/dot(n,ray);
   return clamp(30.*(z-.1)/(29.9*z),0.,1.);
 }
 let xy=(p.xy/f32(${size})-.5)*vec2f(2,-2)*fixture.projection.x;
 let z=-dot(n.xy,xy)/n.z;
 let gap=select(0.,fixture.normalGap.w,all(abs(xy)<vec2f(fixture.width.x*.5)));
 return clamp(.5-(z+gap)/fixture.projection.z,0.,1.);
}`});
    const render=await device.createRenderPipelineAsync({layout:"auto",vertex:{module:caster,entryPoint:"vertex"},fragment:{module:caster,entryPoint:"fragment",targets:[]},primitive:{topology:"triangle-list"},depthStencil:{format:"depth32float",depthWriteEnabled:true,depthCompare:"always"}});
    const casterGroup=device.createBindGroup({layout:render.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:input}}]});
    const receiver=device.createShaderModule({code:`${definitions}
struct Uniforms {container:vec4f} struct Dry {tuningRays0:vec4f,tuningRays1:vec4f}
struct RasterSpot {positionNear:vec4f,directionTan:vec4f,rangeRadiusIndex:vec4f}
struct Lighting {control:vec4u,farCenterExtent:vec4f,depthCell:vec4f,sun:vec4f,spots:array<RasterSpot,1>}
var<private> uniforms:Uniforms;var<private> dry:Dry;var<private> rasterLighting:Lighting;
@group(0) @binding(1) var rasterSunDepth:texture_depth_2d_array;
@group(0) @binding(2) var rasterSunSampler:sampler_comparison;
@group(0) @binding(3) var<storage,read_write> result:vec4f;
${rasterReceiverPlaneWGSL}
${sampling}
@compute @workgroup_size(1) fn main(){
 uniforms.container=vec4f(fixture.projection.x/.7,0,0,0);
 rasterLighting.farCenterExtent=vec4f(0,0,0,fixture.projection.x);
 rasterLighting.depthCell=vec4f(fixture.projection.z,fixture.projection.y,0,0);
 rasterLighting.sun=vec4f(0,0,1,0);dry.tuningRays0=vec4f(fixture.width.y,0,0,0);dry.tuningRays1=vec4f(0,.0093,0,0);
 var visibility=rasterPcf(vec3f(0),fixture.normalGap.xyz,1u);
 if(fixture.projection.w==1.){
   rasterLighting.spots[0]=RasterSpot(vec4f(0,0,0,.1),vec4f(0,0,1,.8),vec4f(30,0,0,0));
   visibility=rasterSpotVisibility(vec3f(0,0,5),fixture.normalGap.xyz*vec3f(1,-1,-1),0u);
 }
 result=vec4f(visibility,0,0,0);
}`});
    const errors=(await receiver.getCompilationInfo()).messages.filter(m=>m.type==="error");assert.deepEqual(errors,[]);
    const layout=device.createBindGroupLayout({entries:[
      {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:1,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"depth",viewDimension:"2d-array"}},
      {binding:2,visibility:GPUShaderStage.COMPUTE,sampler:{type:"comparison"}},
      {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
    ]});
    const pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module:receiver,entryPoint:"main"}});
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer:input}},{binding:1,resource:depth.createView({dimension:"2d-array"})},
      {binding:2,resource:device.createSampler({compare:"less-equal",minFilter:"linear",magFilter:"linear"})},{binding:3,resource:{buffer:output}},
    ]});
    const measurements:number[]=[];
    try{
      for(const bias of [0,.02]) for(const [extent,cell,nx,ny,nz,gap,width,mode=0] of fixtures){
        device.queue.writeBuffer(input,0,new Float32Array([extent!,cell!,extent!*4,mode,nx!,ny!,nz!,gap!,width!,bias,0,0]));
        const encoder=device.createCommandEncoder();const pass=encoder.beginRenderPass({colorAttachments:[],depthStencilAttachment:{view:depth.createView({dimension:"2d",baseArrayLayer:mode===1?2:1,arrayLayerCount:1}),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}});
        pass.setPipeline(render);pass.setBindGroup(0,casterGroup);pass.draw(3);pass.end();
        const compute=encoder.beginComputePass();compute.setPipeline(pipeline);compute.setBindGroup(0,group);compute.dispatchWorkgroups(1);compute.end();device.queue.submit([encoder.finish()]);
        measurements.push((await readMixedBuffer(device,output))[0]!);
      }
      console.log("raster leaf-contact visibility",size,JSON.stringify(measurements));
      assert.ok(size===2048 ? measurements[15]!>.999 : measurements[15]!<.8, "4cm branch falls between 2048 texels but is represented at 4096");
      measurements.forEach((visibility,i)=>{
        const c=fixtures[i%fixtures.length]!;
        if(size===2048 && c[6]===.04) return;
        if(c[5]===0) assert.ok(visibility>.999,`unoccluded plane ${i} self-shadowed: ${measurements[i]}`);
        else assert.ok(visibility<.8,`8cm blocker ${i} erased: ${measurements[i]}`);
      });
    }finally{input.destroy();output.destroy();depth.destroy();}
  });
});

test("GPU near shadow projection covers the centered basin from every sun direction", {skip:!process.env.WEBGPU_NODE_MODULE}, async()=>{
  const {withUniformDevice}=await import("./helpers/uniform-geometric");
  const {readMixedBuffer}=await import("./helpers/uniform-mixed-native-fields");
  const source=rasterAoConsumerWGSL();
  const projection=source.slice(source.indexOf("fn rasterSunBasis"),source.indexOf("fn rasterSpotBasis"));
  const points:number[][]=[];
  for(const size of [[64,8,80],[8,24,12],[2,2,2]]) for(const sun of [[-.5080923,.2845317,.8129477],[1,.01,0],[0,1,0]]) {
    for(let corner=0;corner<8;corner++)points.push([...size,0,...sun,0,
      (corner&1?1:-1)*size[0]!*.5,corner&2?size[1]!:0,(corner&4?1:-1)*size[2]!*.5,0]);
    // Actual diagnostic bank point: previously selected the 0.294m far map.
    if(size[0]===64)points.push([...size,0,...sun,0,-29,4,-15,0]);
  }
  await withUniformDevice("centered near shadow coverage",async device=>{
    const input=device.createBuffer({size:points.length*48,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const output=device.createBuffer({size:points.length*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    device.queue.writeBuffer(input,0,new Float32Array(points.flat()));
    const module=device.createShaderModule({code:`
struct Case {container:vec4f,sun:vec4f,point:vec4f}
struct Uniforms {container:vec4f} struct Lighting {sun:vec4f,farCenterExtent:vec4f,depthCell:vec4f}
var<private> uniforms:Uniforms;var<private> rasterLighting:Lighting;
@group(0) @binding(0) var<storage,read> cases:array<Case>;
@group(0) @binding(1) var<storage,read_write> result:array<vec4f>;
fn rasterSunDirection()->vec3f{return normalize(rasterLighting.sun.xyz);}
${projection}
@compute @workgroup_size(32) fn main(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=arrayLength(&cases)){return;}let c=cases[id.x];uniforms.container=c.container;
 rasterLighting.sun=c.sun;rasterLighting.depthCell=vec4f(length(c.container.xyz)*4.,0,0,0);
 rasterLighting.farCenterExtent=vec4f(100,50,200,300);
 result[id.x*2u]=vec4f(rasterShadowPosition(c.point.xyz,0u),0);
 let centre=rasterMapCenter(0u);let basis=rasterSunBasis();let projected=result[id.x*2u].xyz;
 // Caster and receiver share this inverse: shadow texel ray starts at the
 // same centered world origin and recovers the actual world surface.
 let recovered=centre+(basis[0]*projected.x+basis[1]*projected.y)*rasterMapExtent(0u)+basis[2]*(.5-projected.z)*rasterLighting.depthCell.x;
 result[id.x*2u+1u]=vec4f(length(recovered-c.point.xyz),length(centre-vec3f(0,c.container.y*.5,0)),length(rasterMapCenter(1u)-vec3f(100,50,200)),0);
}`});
    try{
      const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
      const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:input}},{binding:1,resource:{buffer:output}}]});
      const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(points.length/32));pass.end();device.queue.submit([encoder.finish()]);
      const values=await readMixedBuffer(device,output);
      for(let i=0;i<points.length;i++){
        assert.ok(Math.max(Math.abs(values[i*8]!),Math.abs(values[i*8+1]!))<.85,`centered basin point ${i} should use the near map, got ${values[i*8]},${values[i*8+1]}`);
        assert.ok(values[i*8+2]!>0&&values[i*8+2]!<1);
        assert.ok(values[i*8+4]!<2e-5,"caster/receiver projection round-trip");
        assert.equal(values[i*8+5],0,"near map center respects centered X/Z");
        assert.equal(values[i*8+6],0,"far map retains authored structural center");
      }
    }finally{input.destroy();output.destroy();}
  });
});
