import test from "node:test";
import assert from "node:assert/strict";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { RasterWaterPipeline, surfaceExtractionShader } from "../lib/core/webgpu-water-pipeline";

test("shared dense water samples preserve the reference triangle multiset", {skip:!process.env.WEBGPU_NODE_MODULE}, async()=>{
  await withUniformDevice("shared water classification",async device=>{
    const uniform=device.createBuffer({size:416,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    const bodies=device.createBuffer({size:768,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.UNIFORM});
    const water=new RasterWaterPipeline(device,"rgba8unorm",uniform,bodies);
    const textures:GPUTexture[]=[];
    try {
      await water.initialize();
      const module=device.createShaderModule({code:surfaceExtractionShader});
      const reference=await device.createComputePipelineAsync({layout:water.denseExtractionLayoutForQA,compute:{module,entryPoint:"extractMain"}});
      const n=17;
      const words=new Float32Array(104);words.set([1.7,1.7,1.7,0],12);words.set([n,n,n,1],20);
      device.queue.writeBuffer(uniform,0,words);
      const extract=async(legacy:boolean)=>{
        const encoder=device.createCommandEncoder();
        water.encodeDenseSurfaceExtractionForQA(encoder,n,n,n,"full",false,legacy?reference:undefined);
        device.queue.submit([encoder.finish()]);
        const mesh=await water.readDenseSurfaceExtractionForQA();
        assert.equal(mesh.vertexAllocator,mesh.vertexCount);
        const bits=new Uint32Array(mesh.vertices.buffer,mesh.vertices.byteOffset,mesh.vertices.length);
        const triangles:string[]=[];
        for(let i=0;i<bits.length;i+=24)triangles.push(Array.from(bits.subarray(i,i+24)).join(","));
        return {count:mesh.vertexCount,cubes:mesh.activeCubeCount,triangles:triangles.sort()};
      };
      for(const nodal of [false,true])for(const shape of ["sphere","sheet","walls"]){
        const size=n+(nodal?1:0),data=new Float32Array(size**3);
        for(let z=0;z<size;z++)for(let y=0;y<size;y++)for(let x=0;x<size;x++){
          const phi=shape==="sphere"?Math.hypot(x-8.3,y-8.1,z-7.8)-5.2
            :shape==="sheet"?Math.abs(y-8.3)-.7:y-4.4;
          data[x+size*(y+size*z)]=nodal?phi*.1:.5-phi;
        }
        const field=device.createTexture({dimension:"3d",size:[size,size,size],format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
        textures.push(field);device.queue.writeTexture({texture:field},data,{bytesPerRow:size*4,rowsPerImage:size},[size,size,size]);
        water.setVolume(field);
        const before=await extract(true),after=await extract(false);
        assert.ok(before.count>0);
        assert.deepEqual(after,before,`${shape}, ${nodal?"nodal phi":"cell density"}`);
      }
    } finally {water.destroy();uniform.destroy();bodies.destroy();textures.forEach(t=>t.destroy());}
  });
});
