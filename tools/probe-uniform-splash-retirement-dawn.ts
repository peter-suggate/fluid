/** Splash retention census. Run sequentially under the repository GPU lease.
 * --frames=180 --every=30 --out=/tmp/retirement.json --values='{}' --verify
 * Baseline: FLUID_UNIFORM_AB_OFF=phiretire with --values='{"orphanDustThreshold":0}'.
 */
import assert from "node:assert/strict";
import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { uniformGeometricSolverOptions } from "../lib/methods/uniform/uniform-geometric-options";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";

const arg=(key:string,fallback:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3)??fallback;
const frames=Number(arg("frames","240")),every=Number(arg("every","30"));
const values=JSON.parse(arg("values","{}")),sceneId=arg("scene","cm12-figure-9");
assert.ok(Number.isInteger(frames)&&frames>0&&Number.isInteger(every)&&every>0);
async function readTexture(device:GPUDevice,texture:GPUTexture){
  const row=Math.ceil(texture.width*4/256)*256;
  const b=device.createBuffer({size:row*texture.height*texture.depthOrArrayLayers,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{
    const e=device.createCommandEncoder();e.copyTextureToBuffer({texture},{buffer:b,bytesPerRow:row,rowsPerImage:texture.height},[texture.width,texture.height,texture.depthOrArrayLayers]);
    device.queue.submit([e.finish()]);await b.mapAsync(GPUMapMode.READ);
    const raw=new Float32Array(b.getMappedRange()),out=new Float32Array(texture.width*texture.height*texture.depthOrArrayLayers);
    for(let z=0;z<texture.depthOrArrayLayers;z++)for(let y=0;y<texture.height;y++)out.set(raw.subarray((z*texture.height+y)*row/4,(z*texture.height+y)*row/4+texture.width),(z*texture.height+y)*texture.width);
    return out;
  }finally{b.unmap();b.destroy();}
}
await acquireWebGPUExclusiveLock("dawn-probe","Uniform splash retirement census");
let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined;
try{
  const dawn=await import(pathToFileURL(resolve(process.env.WEBGPU_NODE_MODULE??"node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const scene=sceneDocument(getSceneDefinition(sceneId)),options=uniformGeometricSolverOptions(values,scene);
  solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,options,()=>{});
  const {nx,ny,nz}=solver.info,h=scene.container.height_m/ny,dust=options.volumeDustThreshold!;
  const rows:unknown[]=[],timings:number[]=[];let initialMass=0;
  for(let frame=0;frame<=frames;frame++){
    if(frame){const start=performance.now();assert.ok(solver.advanceTo(frame/30));await solver.awaitFrameCompletion();timings.push(performance.now()-start);}
    if(frame%every!==0&&frame!==frames)continue;
    const v=await readTexture(device,solver.volumeTexture),phi=await readTexture(device,solver.vertexPhiTexture!);
    const counts={volumeOnly:0,phiOnly:0,both:0,empty:0,upperPhiOnly:0},bins=[0,0,0,0],massBins=[0,0,0,0];
    let mass=0,upperMass=0,positiveBandVertices=0;
    for(let i=0;i<v.length;i++){const a=Math.abs(v[i]!);mass+=v[i]!;const y=Math.floor(i/nx)%ny;if(y>=ny/2)upperMass+=v[i]!;
      if(a){const b=a<dust?0:a<=.05?1:a<.5?2:3;bins[b]!++;massBins[b]!+=v[i]!;}}
    for(let z=0;z<nz;z+=4)for(let y=0;y<ny;y+=4)for(let x=0;x<nx;x+=4){let vs=false,ps=false;
      for(let dz=0;dz<4&&z+dz<nz;dz++)for(let dy=0;dy<4&&y+dy<ny;dy++)for(let dx=0;dx<4&&x+dx<nx;dx++)vs ||= Math.abs(v[x+dx+nx*(y+dy+ny*(z+dz))]!)>=dust;
      for(let dz=0;dz<=4&&z+dz<=nz;dz++)for(let dy=0;dy<=4&&y+dy<=ny;dy++)for(let dx=0;dx<=4&&x+dx<=nx;dx++)ps ||= phi[x+dx+(nx+1)*(y+dy+(ny+1)*(z+dz))]!<4*h;
      counts[vs?(ps?"both":"volumeOnly"):(ps?"phiOnly":"empty")]++;if(!vs&&ps&&y>=ny/2)counts.upperPhiOnly++;
    }
    for(const p of phi)if(p>0&&p<4*h)positiveBandVertices++;
    const info=await solver.readStats();
    const work=Object.fromEntries(Object.entries(info).filter(([key])=>/^(uniformVolume|uniformDomain|uniformTwoLevel)/.test(key)));
    const sorted=timings.splice(0).sort((a,b)=>a-b);
    if(frame===0)initialMass=mass;
    const row={frame,time_s:frame/30,mass,massLostFraction:1-mass/initialMass,upperMass,counts,bins,massBins,positiveBandVertices,medianStep_ms:sorted[Math.floor(sorted.length/2)],work};
    rows.push(row);console.log(JSON.stringify(row));assert.deepEqual(errors,[]);
    writeFileSync(arg("out","/tmp/retirement.json"),JSON.stringify({sceneId,values,abOff:process.env.FLUID_UNIFORM_AB_OFF??"",dims:[nx,ny,nz],rows,errors},null,2));
    if(process.argv.includes("--verify")){
      assert.equal(sceneId,"cm12-figure-9","acceptance budgets are specific to figure 9");
      assert.equal(frames,180,"acceptance covers the six-second rebound");
      assert.ok(mass/initialMass>.995,`frame ${frame}: cumulative mass loss exceeds 0.5%`);
      if(frame===150||frame===180){
        assert.ok(counts.upperPhiOnly<1000,`frame ${frame}: unsupported upper phi still occupies ${counts.upperPhiOnly} tiles`);
        assert.ok(info.uniformTwoLevelFineTiles!<11500,`frame ${frame}: fine work failed to contract`);
      }
    }
  }
}finally{solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
