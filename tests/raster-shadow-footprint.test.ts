import assert from "node:assert/strict";
import test from "node:test";
import { rasterAoConsumerWGSL, rasterReceiverPlaneWGSL } from "../lib/svo/features/lighting-visibility/svo-raster-ao";

// A leaf covers a neighboring comparison texel, but not the integer texel
// selected by a point blocker search. Wide disk taps all miss this local leaf.
test("GPU PCSS blocker search preserves every weighted bilinear receiver texel",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
  const {withUniformDevice}=await import("./helpers/uniform-geometric");
  const {readMixedBuffer}=await import("./helpers/uniform-mixed-native-fields");
  const source=rasterAoConsumerWGSL();
  const sampling=source.slice(source.indexOf("fn rasterSunDirection"),source.indexOf("fn rasterOctEncode"));
  const sunDisk=[[.144338,0],[-.184342,.168873],[.028217,-.321513],[.232351,.303061],[-.426393,-.075423],[.403917,-.256939],[-.135102,.502574],[-.257655,-.496099],[.559008,.204149],[-.581555,.240057],[.280348,-.599087],[.207170,.660490],[-.624412,-.361860],[.732507,-.161040],[-.447038,.635865],[-.103276,-.796974],[.634013,.534347],[-.853183,.035282],[.622332,-.619303],[-.041636,.900426],[-.592151,-.709594],[.938032,.126211],[-.794793,.552996],[.217183,-.965401]];
  const spotDisk=[[0,0],[-.32,.23],[.25,-.39],[-.43,-.32],[.60,.15],[-.09,.67],[-.69,.12],[.09,-.76],[.55,.63],[-.57,.62],[-.64,-.64],[.79,-.48]];
  await withUniformDevice("PCSS receiver footprint",async device=>{
    const size=4096,input=device.createBuffer({size:32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const output=device.createBuffer({size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    const depth=device.createTexture({size:[size,size,3],format:"depth32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.RENDER_ATTACHMENT});
    const definitions=`struct Fixture { receiverBlocker:vec4f, options:vec4f } @group(0) @binding(0) var<storage,read> fixture:Fixture;`;
    try{
      const caster=device.createShaderModule({code:`${definitions}
@vertex fn vertex(@builtin(vertex_index) i:u32)->@builtin(position) vec4f{let p=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));return vec4f(p[i],0,1);}
@fragment fn fragment(@builtin(position) p:vec4f)->@builtin(frag_depth) f32{
 let blocked=all(vec2i(p.xy)==vec2i(fixture.receiverBlocker.zw))&&fixture.options.y>0.;
 if(fixture.options.x>0.){let z=select(5.,4.92,blocked);return 30.*(z-.1)/(29.9*z);}
 return .5-select(0.,.08/256.,blocked);
}`});
      const raster=await device.createRenderPipelineAsync({layout:"auto",vertex:{module:caster,entryPoint:"vertex"},fragment:{module:caster,entryPoint:"fragment",targets:[]},primitive:{topology:"triangle-list"},depthStencil:{format:"depth32float",depthWriteEnabled:true,depthCompare:"always"}});
      const casterGroup=device.createBindGroup({layout:raster.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:input}}]});
      const receiver=device.createShaderModule({code:`${definitions}
struct Uniforms{container:vec4f}struct Dry{tuningRays0:vec4f,tuningRays1:vec4f}
struct RasterSpot{positionNear:vec4f,directionTan:vec4f,rangeRadiusIndex:vec4f}
struct Lighting{control:vec4u,farCenterExtent:vec4f,depthCell:vec4f,sun:vec4f,spots:array<RasterSpot,1>}
var<private> uniforms:Uniforms;var<private> dry:Dry;var<private> rasterLighting:Lighting;
@group(0) @binding(1) var rasterSunDepth:texture_depth_2d_array;
@group(0) @binding(2) var<storage,read_write> result:vec4f;
@group(0) @binding(3) var rasterSunSampler:sampler;
fn rasterMapResolution(layer:u32)->f32{return f32(textureDimensions(rasterSunDepth).x);}
fn rasterAtlasTile(layer:u32)->vec3u{return vec3u(0,0,layer);}
${rasterReceiverPlaneWGSL}${sampling}
@compute @workgroup_size(1) fn main(){
 uniforms.container=vec4f(64,8,80,0);rasterLighting.farCenterExtent=vec4f(0,0,0,64);rasterLighting.depthCell=vec4f(256,.25,0,0);rasterLighting.sun=vec4f(0,0,1,0);
 dry.tuningRays0=vec4f(0);dry.tuningRays1=vec4f(0,.065,0,0);
 let uv=fixture.receiverBlocker.xy/4096.;let plane=(uv-.5)*vec2f(2,-2);
 var v=rasterPcf(vec3f(plane*64.,0),vec3f(0,0,1),1u);
 if(fixture.options.x>0.){
  // Small physical radius gives a one-texel reconstruction kernel; the
  // perspective blocker search is wider than the local comparison footprint.
  rasterLighting.spots[0]=RasterSpot(vec4f(0,0,0,.1),vec4f(0,0,1,.8),vec4f(30,.0001,0,0));
  v=rasterSpotVisibility(vec3f(-plane.x*.8*5.,plane.y*.8*5.,5),vec3f(0,0,-1),0u);
 }
 result=vec4f(v,0,0,0);
}`});
      const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module:receiver,entryPoint:"main"}});
      const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:input}},{binding:1,resource:depth.createView({dimension:"2d-array"})},{binding:2,resource:{buffer:output}},{binding:3,resource:device.createSampler({})}]});
      const measurements=[];
      for(const mode of [0,1])for(const blocked of [false,true])for(const offset of [[0,0],[1,0],[0,1],[1,1]]){
        const receiverXY=[1000.75,1000.75],blocker=[1000+offset[0]!,1000+offset[1]!];
        device.queue.writeBuffer(input,0,new Float32Array([...receiverXY,...blocker,mode,Number(blocked),0,0]));
        const encoder=device.createCommandEncoder(),pass=encoder.beginRenderPass({colorAttachments:[],depthStencilAttachment:{view:depth.createView({dimension:"2d",baseArrayLayer:mode===0?1:2,arrayLayerCount:1}),depthLoadOp:"clear",depthClearValue:1,depthStoreOp:"store"}});
        pass.setPipeline(raster);pass.setBindGroup(0,casterGroup);pass.draw(3);pass.end();
        const compute=encoder.beginComputePass();compute.setPipeline(pipeline);compute.setBindGroup(0,group);compute.dispatchWorkgroups(1);compute.end();device.queue.submit([encoder.finish()]);
        const actual=(await readMixedBuffer(device,output))[0]!;
        const disk=mode===0?sunDisk:spotDisk;
        const expected=blocked?1-disk.reduce((sum,[dx,dy])=>sum+Math.max(0,1-Math.abs(receiverXY[0]!+dx!-.5-blocker[0]!))*Math.max(0,1-Math.abs(receiverXY[1]!+dy!-.5-blocker[1]!)),0)/disk.length:1;
        measurements.push({mode,blocked,offset,actual,expected});
        assert.ok(Math.abs(actual-expected)<.002,`mode${mode} blocked${blocked} offset${offset}: ${actual} vs bilinear disk oracle${expected}`);
      }
      console.log("PCSS footprint measurements",JSON.stringify(measurements));
    }finally{input.destroy();output.destroy();depth.destroy();}
  });
});
