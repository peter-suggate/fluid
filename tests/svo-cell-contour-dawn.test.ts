import assert from "node:assert/strict";
import test from "node:test";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createDawnRenderDevice } from "../tools/svo-dry-frame-harness";
import { svoCellContourWGSL } from "../lib/svo/features/construction/svo-cell-contour";
import { svoCellContourFitWGSL } from "../lib/svo/features/construction/svo-cell-contour-fit";
import { SVO_GBUFFER_NORMAL_OCT8_WGSL } from "../lib/svo/contracts/svo-gbuffer";
import { createWebgpuSolidWorldPageLayout } from "../lib/core/webgpu-solid-world-pages";
import { sparseSceneProxyVoxelizationShaderFor } from "../lib/core/webgpu-sparse-scene-proxies";

(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("Dawn contour producer bounds a source plane and polygonizer emits a closed clipped cube", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "svo-cell-contour");
  let device: GPUDevice | undefined;
  try {
    device = (await createDawnRenderDevice()).device;
    const canonical = createWebgpuSolidWorldPageLayout({ baseWords: 0, authoredPageCount: 1, includesMaterial: true });
    for (const mode of ["dense", "occupancy", "banded"] as const) {
      const module = device.createShaderModule({ code: sparseSceneProxyVoxelizationShaderFor("dry", "f16-unorm8", mode, canonical, undefined, true) });
      assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error"), [], `producer ${mode}`);
      await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "rebuildDirtyBrickPayload" } });
    }
    const module = device.createShaderModule({ code: `
      ${SVO_GBUFFER_NORMAL_OCT8_WGSL}
      ${svoCellContourWGSL}
      struct ScenePrimitive { centerType:vec4f, extentIdentity:vec4f, rotation:vec4f }
      @group(0) @binding(0) var<storage,read_write> maintenance:array<atomic<u32>>;
      @group(0) @binding(1) var<storage,read> primitives:array<ScenePrimitive>;
      @group(0) @binding(2) var<storage,read_write> output:array<vec4f>;
      fn candidateOffset()->u32{return 0u;}fn candidatesPerBrick()->u32{return 1u;}
      fn scenePrimitiveType(p:ScenePrimitive)->u32{return 1u;}
      fn inverseRotate(p:vec3f,r:vec4f)->vec3f{return p;}
      fn primitiveUsesThresholdOccupancy(p:ScenePrimitive)->bool{return p.centerType.w>0.5;}
      // Exact signed distance to an oblique halfspace through the cell centre.
      fn primitiveDistance(p:ScenePrimitive,w:vec3f)->f32{return dot(normalize(p.extentIdentity.xyz),w-vec3f(.5));}
      fn swCell()->vec3f{return vec3f(1.0);}
      fn swOrigin()->vec3f{return vec3f(0.0);}
      fn swFractionQ8(q:vec3i)->u32{return select(0u,255u,primitives[0].centerType.z>0.5);}
      ${svoCellContourFitWGSL(false, true)}
      @compute @workgroup_size(1) fn check(){
        let n=normalize(primitives[0].extentIdentity.xyz);
        let code=fitSceneContour(vec3f(.5),vec3f(1),n,.5,0u,1u);
        let baked=svoGBufferUnpackNormalOct8(svoGBufferPackNormalOct8(n));
        let contour=cellContour(baked,vec3f(1),code);
        output[0]=vec4f(contour.normal,contour.high);
        output[1]=vec4f(f32(code),f32(fitSceneContour(vec3f(.5),vec3f(1),n,1.,0u,1u)),0,0);
        var at=2u;
        for(var face=0u;face<7u;face+=1u){
          var poly=contourCap(contour);
          if(face<6u){poly=contourClipPolygon(contourCubeFace(face),contour);}
          for(var i=1u;i+1u<poly.count;i+=1u){
            output[at]=vec4f(contourUnpackPoint(contourPackPoint(poly.points[0])),f32(face));
            output[at+1u]=vec4f(contourUnpackPoint(contourPackPoint(poly.points[i])),f32(face));
            output[at+2u]=vec4f(contourUnpackPoint(contourPackPoint(poly.points[i+1u])),f32(face));at+=3u;
          }
        }
        output[1].z=f32(at);
      }` });
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "check" } });
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const maintenance = device.createBuffer({ size: 4, usage });
    const primitives = device.createBuffer({ size: 48, usage });
    const output = device.createBuffer({ size: 4096, usage });
    const read = device.createBuffer({ size: 4096, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: maintenance } }, { binding: 1, resource: { buffer: primitives } }, { binding: 2, resource: { buffer: output } }] });
    for (const [normal, voxelOnly, threshold] of [
      [[0,1,0],false,false], [[1,1,1],false,false], [[-1,2,-.5],false,false],
      [[1,0,0],false,false], [[0,-1,0],false,false], [[0,1,0],true,false], [[0,1,0],false,true],
    ] as const) {
      device.queue.writeBuffer(primitives, 0, new Float32Array([0,0,Number(voxelOnly),Number(threshold),...normal,0,0,0,0,1]));
      const encoder = device.createCommandEncoder(); const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
      encoder.copyBufferToBuffer(output, 0, read, 0, 4096); device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ); const data = new Float32Array(read.getMappedRange().slice(0)); read.unmap();
      if(voxelOnly||threshold){assert.equal(data[4],0,"voxel-only and threshold geometry retain cubes");continue;}
      assert.ok(data[4] > 0 && data[4] < 255, `a plane produces a nontrivial contour: ${normal}`);
      assert.equal(data[5], 0, "a full cell does not acquire a contour");
      const n = Array.from(data.slice(0,3)), high = data[3];
      // Exhaustively check source halfspace points against the stored bound.
      for (let z=0;z<=16;z++) for(let y=0;y<=16;y++) for(let x=0;x<=16;x++) {
        const p=[x/16-.5,y/16-.5,z/16-.5];
        if(p.reduce((s,v,i)=>s+v*normal[i],0)<=0) assert.ok(p.reduce((s,v,i)=>s+v*n[i],0)<=high+1e-6);
      }
      let volume = 0; const edges = new Map<string, number>();
      for(let at=2;at<data[6];at+=3){
        const pts=[0,1,2].map(i=>Array.from(data.slice((at+i)*4,(at+i)*4+3)));
        const [a,b,c]=pts;
        const cross=[b[1]*c[2]-b[2]*c[1],b[2]*c[0]-b[0]*c[2],b[0]*c[1]-b[1]*c[0]];
        volume+=a.reduce((s,v,i)=>s+v*cross[i],0)/6;
        for(let i=0;i<3;i++){
          const key=[pts[i].join(","),pts[(i+1)%3].join(",")].sort().join("|"); edges.set(key,(edges.get(key)??0)+1);
        }
      }
      assert.ok(volume>=.5-1e-3&&volume<1, `positive conservative clipped volume ${volume}`);
      assert.ok([...edges.values()].every(count=>count===2), "closed triangulation: every undirected edge has two owners");
    }
    maintenance.destroy(); primitives.destroy(); output.destroy(); read.destroy();
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
