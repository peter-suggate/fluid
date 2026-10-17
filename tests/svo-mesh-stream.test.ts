import assert from "node:assert/strict";
import test from "node:test";
import { surfaceMeshStreamVertexWGSL, SVO_SURFACE_MESH_STATE as W, SVO_SURFACE_MESH_STATE_BYTES, svoSurfaceMeshWGSL } from "../lib/svo/features/primary-visibility/svo-surface-mesh";
import { withUniformDevice } from "./helpers/uniform-geometric";

test("mesh stream arguments retain the diagnostic header and support both culling paths", () => {
  assert.ok(W.packedDraw >= W.hostPrimitiveMaterialBase + 1);
  assert.equal(W.wordCount, W.packedDraw + 4);
  assert.equal(SVO_SURFACE_MESH_STATE_BYTES, W.wordCount * 4);
  for (const culling of [true,false]) {
    const shader=svoSurfaceMeshWGSL(3,128,culling,true);
    assert.match(shader,new RegExp(`atomicStore\\(&meshState\\[${W.packedDraw+1}\\],1u\\)`));
    assert.ok(shader.includes(culling
      ? `atomicAdd(&meshState[${W.packedDraw}],count*select(6u,3u,meshDmcEnabled()))`
      : `atomicStore(&meshState[${W.packedDraw}],quads*select(6u,3u,meshDmcEnabled()))`));
  }
});

test("GPU continuous mesh stream matches instanced strips for quads and DMC triangles", {skip:!process.env.WEBGPU_NODE_MODULE}, async () => {
  await withUniformDevice("mesh stream parity",async device=>{
    for(const trianglesOnly of [false,true]) {
      const module=device.createShaderModule({code:`
${surfaceMeshStreamVertexWGSL}
struct V {@builtin(position) p:vec4f,@location(0) uv:vec2f,@location(1) @interpolate(flat) id:u32}
fn vertex(corner:u32,record:u32)->V{
 let corners=array<vec2f,4>(vec2f(0,0),vec2f(1,0),vec2f(0,1),vec2f(1,1));
 var c=corner;if(${trianglesOnly}||(record&1u)!=0u){c=min(c,2u);}
 var uv=corners[c];if((record&2u)!=0u){uv=uv.yx;}
 let base=vec2f(f32(record%2u),f32(record/2u))-.95;
 return V(vec4f(base+uv*.85,.5,1),uv,record+1u);
}
@vertex fn old(@builtin(vertex_index) v:u32,@builtin(instance_index) i:u32)->V{return vertex(v,i);}
@vertex fn streamed(@builtin(vertex_index) v:u32)->V{let i=meshStreamVertex(v,${trianglesOnly});return vertex(i.x,i.y);}
@fragment fn frag(v:V,@builtin(front_facing) front:bool)->@location(0) vec4u{
 return vec4u(v.id,vec2u(round(v.uv*255.)),select(0u,1u,front));
}`});
      assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
      const pipelines=await Promise.all([false,true].map(stream=>device.createRenderPipelineAsync({
        layout:"auto",vertex:{module,entryPoint:stream?"streamed":"old"},
        fragment:{module,entryPoint:"frag",targets:[{format:"rgba32uint"}]},
        primitive:{topology:stream?"triangle-list":"triangle-strip"},
      })));
      const texture=device.createTexture({size:[64,64],format:"rgba32uint",usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});
      const read=device.createBuffer({size:64*64*16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      try {
        const images:Uint32Array[]=[];
        for(let stream=0;stream<2;stream++) {
          const encoder=device.createCommandEncoder(),pass=encoder.beginRenderPass({colorAttachments:[{view:texture.createView(),loadOp:"clear",storeOp:"store",clearValue:[0,0,0,0]}]});
          pass.setPipeline(pipelines[stream]!);pass.draw(stream?4*(trianglesOnly?3:6):4,stream?1:4);pass.end();
          encoder.copyTextureToBuffer({texture},{buffer:read,bytesPerRow:1024},[64,64]);device.queue.submit([encoder.finish()]);
          await read.mapAsync(GPUMapMode.READ);images.push(new Uint32Array(read.getMappedRange()).slice());read.unmap();
        }
        assert.deepEqual(images[1],images[0],`identical coverage, winding, identity and interpolants: trianglesOnly=${trianglesOnly}`);
        assert.ok(images[0]!.some(value=>value!==0));
      } finally {read.destroy();texture.destroy();}
    }
  });
});
