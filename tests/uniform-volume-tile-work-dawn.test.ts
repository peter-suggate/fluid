import { UNIFORM_VOLUME_PHASE } from "../lib/methods/uniform/uniform-volume-stages";
import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { UNIFORM_VOLUME_SHARPEN_ENTRIES, UNIFORM_VOLUME_SHARPEN_TILE_COUNT_WORD } from "../lib/methods/uniform/uniform-volume.wgsl";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
type SharpenEntry = typeof UNIFORM_VOLUME_SHARPEN_ENTRIES[number];
interface Access {
  volumeA:GPUTexture;volumeB:GPUTexture;gammaA:GPUTexture;volumeEdges:GPUBuffer;conditioningScratch:GPUBuffer;
  sharpenComputeGroup:GPUBindGroup;sharpenResolveGroup:GPUBindGroup;
  tileClassifyPipeline?:GPUComputePipeline;volumePipelines:Record<string,GPUComputePipeline>;
  sharpenTileWork:boolean;sharpenTileCount:number;sharpenPipeline(entry:SharpenEntry):GPUComputePipeline;
  writeParams(dt:number,bodies:number,inflow:number):void;
  run(e:GPUCommandEncoder,label:string,p:GPUComputePipeline,g:GPUBindGroup):void;
  encodeGeometricVolume(e:GPUCommandEncoder,seam?:(phase:{label:string})=>void):void;
}
function write(device:GPUDevice,texture:GPUTexture,data:Float32Array) {
  device.queue.writeTexture({texture},data as Float32Array<ArrayBuffer>,{bytesPerRow:texture.width*4,rowsPerImage:texture.height},[texture.width,texture.height,texture.depthOrArrayLayers]);
}
async function read(device:GPUDevice,texture:GPUTexture) {
  const row=Math.ceil(texture.width*4/256)*256;
  const b=device.createBuffer({size:row*texture.height*texture.depthOrArrayLayers,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try {
    const e=device.createCommandEncoder();e.copyTextureToBuffer({texture},{buffer:b,bytesPerRow:row,rowsPerImage:texture.height},[texture.width,texture.height,texture.depthOrArrayLayers]);device.queue.submit([e.finish()]);await b.mapAsync(GPUMapMode.READ);
    const source=new Float32Array(b.getMappedRange()),out=new Float32Array(texture.width*texture.height*texture.depthOrArrayLayers);
    for(let i=0;i<texture.height*texture.depthOrArrayLayers;i++)out.set(source.subarray(i*row/4,i*row/4+texture.width),i*texture.width);
    return out;
  }finally{b.unmap();b.destroy();}
}
async function readWord(device:GPUDevice,buffer:GPUBuffer,byteOffset:number) {
  const b=device.createBuffer({size:4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try {
    const e=device.createCommandEncoder();e.copyBufferToBuffer(buffer,byteOffset,b,0,4);device.queue.submit([e.finish()]);
    await b.mapAsync(GPUMapMode.READ);return new Uint32Array(b.getMappedRange())[0]!;
  }finally{b.unmap();b.destroy();}
}
const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("4h sharpening work map matches the dense control across tile activation, stale scratch and live toggling",{timeout:240000},async()=>{
  await acquireWebGPUExclusiveLock("dawn-test","uniform geometric 4h scheduling parity");
  let device:GPUDevice|undefined;const solvers:WebGPUUniformReferenceSolver[]=[];
  try {
    const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
    const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]);const adapter=await gpu.requestAdapter();assert.ok(adapter);
    device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
    const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
    const scene=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
    // Partial workgroups exercise boundary tile indexing as well as interior seams.
    scene.voxelDomain.finestCellSize_m=scene.container.width_m/18;
    scene.container.height_m=scene.container.depth_m=scene.container.width_m*16/18;
    for(const geometricTileWork of [false,true])solvers.push(await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,{geometricVolume:true,geometricTileWork},()=>{}));
    const {nx:n,ny,nz}=solvers[0]!.info;assert.equal(n,18);assert.equal(ny,16);assert.equal(nz,16);
    const N=n*ny*nz;
    const tileTotal=Math.ceil(n/4)*Math.ceil(ny/4)*Math.ceil(nz/4);
    const countOffset=(2*N+UNIFORM_VOLUME_SHARPEN_TILE_COUNT_WORD)*4;
    assert.equal((solvers[0]! as unknown as Access).sharpenTileCount,tileTotal);
    assert.equal((solvers[0]! as unknown as Access).sharpenTileWork,false,"an explicit false stays dense");
    assert.equal((solvers[1]! as unknown as Access).sharpenTileWork,true);
    const poisonPipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module:device.createShaderModule({code:`@group(0) @binding(0) var<storage,read_write> data:array<f32>; @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){if(id.x<arrayLength(&data)){data[id.x]=17.0;}}`}),entryPoint:"main"}});
    const h=scene.container.width_m/n;
    // One synthetic sharpening stage: poison the stencil arena, classify if the
    // solver's map is live, then run the eight sweeps its own toggle selects.
    const encodeSharpening=(solver:WebGPUUniformReferenceSolver,fields:{phi:Float32Array,volume:Float32Array,target:Float32Array}) => {
      const a=solver as unknown as Access;a.writeParams(1/30,0,0);
      write(device!,solver.vertexPhiTexture!,fields.phi);write(device!,a.volumeB,fields.volume);
      write(device!,a.volumeA,new Float32Array(N).fill(-123));write(device!,a.gammaA,fields.target);
      const e=device!.createCommandEncoder();
      // Stale edge bytes deliberately nonzero; skipped neighbors may not read them.
      const poisonGroup=device!.createBindGroup({layout:poisonPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:a.volumeEdges}}]});
      const pass=e.beginComputePass();pass.setPipeline(poisonPipeline);pass.setBindGroup(0,poisonGroup);pass.dispatchWorkgroups(Math.ceil(a.volumeEdges.size/4/64));pass.end();
      if(a.sharpenTileWork){e.clearBuffer(a.conditioningScratch,countOffset,4);a.run(e,"classify",a.tileClassifyPipeline!,a.sharpenComputeGroup);}
      for(let round=0;round<8;round++)for(const entry of UNIFORM_VOLUME_SHARPEN_ENTRIES)a.run(e,entry,a.sharpenPipeline(entry),round%2===0?a.sharpenComputeGroup:a.sharpenResolveGroup);
      device!.queue.submit([e.finish()]);
      return device!.queue.onSubmittedWorkDone();
    };
    const sphereFields=(center:number) => {
      const phi=new Float32Array((n+1)*(ny+1)*(nz+1)),volume=new Float32Array(N),target=new Float32Array(N);
      for(let z=0;z<=nz;z++)for(let y=0;y<=ny;y++)for(let x=0;x<=n;x++)phi[x+(n+1)*(y+(ny+1)*z)]=(Math.hypot(x-center,y-8,z-8)-3.5)*h;
      for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<n;x++) {
        const i=x+n*(y+ny*z),distance=Math.hypot(x+0.5-center,y+0.5-8,z+0.5-8)-3.5;
        target[i]=Math.max(0,Math.min(1,0.5-distance));volume[i]=target[i]! * 0.8 + (Math.abs(distance)<1?0.23:0);
      }
      // Conserved volume far outside phi's surface must survive skipping.
      volume[0]=0.75;
      return {phi,volume,target};
    };
    for(const [center,band] of [[5.2,2.1],[12.1,0.1],[8,3.1],[40,2.1]] as const) {
      const fields=sphereFields(center);
      const outputs:Float32Array[]=[];
      for(const solver of solvers) {
        const a=solver as unknown as Access;
        solver.applyRuntimeValues({densitySharpening:"on",sharpeningDistance:band,sharpeningStrength:1});
        await encodeSharpening(solver,fields);
        outputs.push(await read(device,a.volumeB));assert.deepEqual(await read(device,solver.vertexPhiTexture!),fields.phi);
      }
      assert.deepEqual(outputs[1],outputs[0],`identical sharpening, center=${center}, band=${band}`);
      assert.equal(outputs[1]![0],0.75,"do not erase far-air conserved V");
      const sum=(v:Float32Array)=>v.reduce((a,b)=>a+b,0);
      assert.ok(Math.abs(sum(outputs[1]!)-sum(fields.volume))<1e-5*Math.max(1,sum(fields.volume)));
      if(center===5.2)assert.ok(outputs[1]!.some((v,i)=>Math.abs(v-fields.volume[i]!)>1e-5),"must exercise real flux across cells");
      const active=await readWord(device,(solvers[1]! as unknown as Access).conditioningScratch,countOffset);
      if(center===40)assert.equal(active,0,"a sphere outside the domain admits no tile");
      else assert.ok(active>0&&active<tileTotal,`center=${center} band=${band} must select a strict subset: ${active}/${tileTotal}`);
    }
    // A phi linear in z makes the admitted band exact: the cell-centre value is
    // (z+0.5-8)h, so band 2.1 admits z in {6,7,8,9}, i.e. the tz=1,2 tile rows.
    {
      const fields=sphereFields(8);
      for(let z=0;z<=nz;z++)for(let y=0;y<=ny;y++)for(let x=0;x<=n;x++)fields.phi[x+(n+1)*(y+(ny+1)*z)]=(z-8)*h;
      const tiled=solvers[1]!;tiled.applyRuntimeValues({densitySharpening:"on",sharpeningDistance:2.1,sharpeningStrength:1});
      await encodeSharpening(tiled,fields);
      assert.equal(await readWord(device,(tiled as unknown as Access).conditioningScratch,countOffset),
        Math.ceil(n/4)*Math.ceil(ny/4)*2,"published active-tile count must equal the CPU band count");
    }
    for(const s of solvers)s.destroy();solvers.length=0;
    // The map is the default, and flipping it live may not move a single bit.
    {
      const toggle=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,{geometricVolume:true},()=>{});
      solvers.push(toggle);const t=toggle as unknown as Access;
      assert.equal(t.sharpenTileWork,true,"geometricVolume enables the work map by default");
      assert.ok(toggle.advanceTo(1/30));await device.queue.onSubmittedWorkDone();
      const stats=await toggle.readStats();
      assert.equal(stats.uniformSharpenWorkMap,true);
      assert.equal(stats.uniformSharpenTilesTotal,tileTotal);
      assert.ok(stats.uniformSharpenTilesActive!>0&&stats.uniformSharpenTilesActive!<=tileTotal,
        `published active tiles ${stats.uniformSharpenTilesActive} of ${tileTotal}`);
      const fields=sphereFields(5.2);const outputs:Float32Array[]=[];
      for(const sharpeningWorkMap of ["on","off","on"] as const) {
        toggle.applyRuntimeValues({densitySharpening:"on",sharpeningDistance:2.1,sharpeningStrength:1,sharpeningWorkMap});
        assert.equal(t.sharpenTileWork,sharpeningWorkMap==="on");
        await encodeSharpening(toggle,fields);
        outputs.push(await read(device,t.volumeB));
      }
      assert.deepEqual(outputs[1],outputs[0],"toggling the map off may not change one bit");
      assert.deepEqual(outputs[2],outputs[0],"toggling the map back on may not change one bit");
      toggle.destroy();solvers.length=0;
    }
    // Replay each real, evolving dense frame into the tiled sharpening stage.
    // Both arms receive exactly the same transported V, phi and target capacity.
    for(const sceneId of ["minimal-power-dam-break-64","large-power-dam-break"]) {
      const liveScene=sceneDocument(getSceneDefinition(sceneId));
      const pair:WebGPUUniformReferenceSolver[]=[];
      for(const geometricTileWork of [false,true]) {
        const s=await WebGPUUniformReferenceSolver.createAsync(device,liveScene,"balanced",undefined,{
          geometricVolume:true,geometricTileWork,
          densitySharpening:true,velocityTransport:"semi-lagrangian",gammaDiffusionIterations:0,solidExcessCorrection:false,
        },()=>{});pair.push(s);solvers.push(s);
      }
      const dense=pair[0]!,tiled=pair[1]!,a=dense as unknown as Access,b=tiled as unknown as Access;
      const liveCountOffset=(2*dense.info.nx*dense.info.ny*dense.info.nz+UNIFORM_VOLUME_SHARPEN_TILE_COUNT_WORD)*4;
      const original=a.encodeGeometricVolume.bind(a);
      a.encodeGeometricVolume=(e,seam)=>original(e,phase=>{
        seam?.(phase);
        if(phase.label!==UNIFORM_VOLUME_PHASE.gather.label)return;
        for(const [from,to] of [[a.volumeB,b.volumeB],[a.gammaA,b.gammaA],[dense.vertexPhiTexture!,tiled.vertexPhiTexture!]]) {
          e.copyTextureToTexture({texture:from!},{texture:to!},[from!.width,from!.height,from!.depthOrArrayLayers]);
        }
        e.clearBuffer(b.conditioningScratch,liveCountOffset,4);
        b.run(e,"classify replay",b.tileClassifyPipeline!,b.sharpenComputeGroup);
        for(let round=0;round<8;round++)for(const entry of UNIFORM_VOLUME_SHARPEN_ENTRIES)b.run(e,entry,b.sharpenPipeline(entry),round%2===0?b.sharpenComputeGroup:b.sharpenResolveGroup);
      });
      b.writeParams(1/30,0,0);
      for(let frame=1;frame<=30;frame++) {
        assert.ok(dense.advanceTo(frame/30));await device.queue.onSubmittedWorkDone();
        const expected=await read(device,a.volumeB),actual=await read(device,b.volumeB);
        let error=0;for(let i=0;i<actual.length;i++)error=Math.max(error,Math.abs(actual[i]!-expected[i]!));
        assert.equal(error,0,`${sceneId} frame ${frame}: identical-input sharpening max error`);
      }
      const active=await readWord(device,b.conditioningScratch,liveCountOffset);
      assert.ok(active>0&&active<b.sharpenTileCount,`${sceneId} replay must skip tiles: ${active}/${b.sharpenTileCount}`);
      const denseStats=await dense.readStats();
      assert.equal(denseStats.uniformSharpenWorkMap,false,"the dense control must not claim a map");
      assert.equal(denseStats.uniformSharpenTilesActive,undefined);
      assert.equal(denseStats.uniformSharpenTilesTotal,undefined);
      for(const s of pair)s.destroy();solvers.length=0;
    }
    assert.deepEqual(errors,[]);
  }finally{for(const s of solvers)s.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});
