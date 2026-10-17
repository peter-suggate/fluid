import assert from "node:assert/strict";
import test from "node:test";
import { svoMeshSampleCoverageWGSL } from "../lib/svo/features/primary-visibility/svo-mesh-sample-coverage";
import { withUniformDevice } from "./helpers/uniform-geometric";

test("GPU sample culling retains every hardware-covered triangle, including subpixel and clipped geometry", {skip:!process.env.WEBGPU_NODE_MODULE}, async () => {
  await withUniformDevice("mesh sample coverage", async device => {
    for (const [width,height] of [[17,19],[64,64],[1000,720]] as const) {
      const vertices:number[]=[];
      const add=(points:number[][])=>{for(const [x,y,w=1] of points)vertices.push((2*x!/width-1)*w,(1-2*y!/height)*w,.1,w);};
      // Tiny triangles: only the first covers the centre, despite equal area.
      add([[.4,.4],[.6,.4],[.5,.6]]);
      add([[.05,.05],[.25,.05],[.15,.25]]);
      let state=12345;
      const random=()=>((state=(Math.imul(state,1664525)+1013904223)>>>0)/4294967296);
      for(let i=0;i<12000;i++) {
        const x=random()*Math.min(width+8,80)-4,y=random()*Math.min(height+8,80)-4;
        const span=i%7===0?12:random()*1.5;
        const points=Array.from({length:3},()=>[x+(random()-.5)*span,y+(random()-.5)*span,.2+random()*8]);
        if(i%17===0)points[0]![2]=.05; // near-plane clipping
        if(i%37===0)points[1]![2]=-.1; // crossing the eye
        if(i%41===0)points[2]=[...points[0]!]; // collapsed edge
        add(points);
      }
      // Sweep raster snapping boundaries in both windings. The conservative
      // guard must keep triangles that acquire a sample after vertex snapping.
      for(let i=-128;i<=128;i++) {
        const t=i/4096;
        add([[2.5+t,2.5],[3.5,2.5+t],[2.5,3.5]]);
        add([[2.5,3.5],[3.5,2.5+t],[2.5+t,2.5]]);
      }
      const count=vertices.length/12;
      const input=device.createBuffer({size:vertices.length*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
      const flags=device.createBuffer({size:count*8,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
      const read=device.createBuffer({size:count*8,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      const texture=device.createTexture({size:[width,height],format:"rgba8unorm",usage:GPUTextureUsage.RENDER_ATTACHMENT});
      device.queue.writeBuffer(input,0,new Float32Array(vertices));
      try {
        const module=device.createShaderModule({code:`
${svoMeshSampleCoverageWGSL}
@group(0) @binding(0) var<storage,read> vertices:array<vec4f>;
@group(0) @binding(1) var<storage,read_write> flags:array<atomic<u32>>;
@compute @workgroup_size(64) fn check(@builtin(global_invocation_id) id:vec3u){
 let i=id.x;if(i>=${count}u){return;}
 atomicStore(&flags[i*2u],u32(meshTriangleMayCoverSample(vertices[i*3u],vertices[i*3u+1u],vertices[i*3u+2u],vec2f(${width},${height}))));
}
struct V{@builtin(position) p:vec4f,@location(0) @interpolate(flat) id:u32}
@vertex fn vertex(@builtin(vertex_index) i:u32)->V{return V(vertices[i],i/3u);}
@fragment fn fragment(v:V)->@location(0) vec4f{atomicOr(&flags[v.id*2u+1u],1u);return vec4f(1);}
`});
        const compute=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"check"}});
        const raster=await device.createRenderPipelineAsync({layout:"auto",vertex:{module,entryPoint:"vertex"},fragment:{module,entryPoint:"fragment",targets:[{format:"rgba8unorm"}]},primitive:{topology:"triangle-list"}});
        const bind=(pipeline:GPUComputePipeline|GPURenderPipeline)=>device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[input,flags].map((buffer,binding)=>({binding,resource:{buffer}}))});
        const encoder=device.createCommandEncoder(),cp=encoder.beginComputePass();
        cp.setPipeline(compute);cp.setBindGroup(0,bind(compute));cp.dispatchWorkgroups(Math.ceil(count/64));cp.end();
        const rp=encoder.beginRenderPass({colorAttachments:[{view:texture.createView(),loadOp:"clear",storeOp:"discard"}]});
        rp.setPipeline(raster);rp.setBindGroup(0,bind(raster));rp.draw(count*3);rp.end();
        encoder.copyBufferToBuffer(flags,0,read,0,read.size);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
        const result=new Uint32Array(read.getMappedRange());let removed=0,covered=0;
        for(let i=0;i<count;i++) {
          if(result[2*i+1]){covered++;assert.equal(result[2*i],1,`hardware-covered triangle ${i} at ${width}×${height} must survive`);}
          if(!result[2*i])removed++;
        }
        assert.equal(result[0],1,"tiny triangle covering a sample stays");
        assert.equal(result[1],1,"hardware covers the tiny triangle");
        assert.equal(result[2],0,"tiny triangle between samples is culled");
        assert.ok(removed>count/2,`${removed}/${count} triangles culled`);
        assert.ok(covered>100);read.unmap();
      } finally {input.destroy();flags.destroy();read.destroy();texture.destroy();}
    }
  });
});
