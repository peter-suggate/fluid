/** Scene-level differential probe: explicitly reduced GPU scene versus Rust 2D. */
import assert from "node:assert/strict";
import { spawnSync, type SpawnSyncReturns } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { pathToFileURL } from "node:url";
import { boxSolidVoxelShell, solidWorldForScene, sampleSolidWorld } from "../../lib/core/solid-world";
import { sceneDocument } from "../../lib/core/scene-definition";
import { getSceneDefinition } from "../../lib/core/scenes";
import { uniformGeometricSolverOptions } from "../../lib/methods/uniform/uniform-geometric-options";
import { resolveUniformGeometricValues, UNIFORM_GEOMETRIC_NATIVE_PARAMS } from "../../lib/methods/uniform/uniform-geometric-parameters";
import { WebGPUUniformReferenceSolver } from "../../lib/methods/uniform/webgpu-uniform-reference";
import { managedGPUDevice } from "../../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from "../../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../../lib/harness/webgpu-smoke-isolation";

async function read(device: GPUDevice, texture: GPUTexture) {
  const components=texture.format === "rgba32float" ? 4 : 1;
  const row=Math.ceil(texture.width*components*4/256)*256;
  const buffer=device.createBuffer({size:row*texture.height*texture.depthOrArrayLayers,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try {
    const encoder=device.createCommandEncoder();encoder.copyTextureToBuffer({texture},{buffer,bytesPerRow:row,rowsPerImage:texture.height},[texture.width,texture.height,texture.depthOrArrayLayers]);device.queue.submit([encoder.finish()]);await buffer.mapAsync(GPUMapMode.READ);
    const mapped=new Float32Array(buffer.getMappedRange());const result=new Float32Array(texture.width*texture.height*texture.depthOrArrayLayers*components);
    for(let z=0;z<texture.depthOrArrayLayers;z++)for(let y=0;y<texture.height;y++)result.set(mapped.subarray((z*texture.height+y)*row/4,(z*texture.height+y)*row/4+texture.width*components),(z*texture.height+y)*texture.width*components);
    return result;
  }finally{buffer.unmap();buffer.destroy();}
}
async function readBuffer(device: GPUDevice, source: GPUBuffer, size: number, offset = 0) {
  const buffer=device.createBuffer({size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(source,offset,buffer,0,size);device.queue.submit([encoder.finish()]);await buffer.mapAsync(GPUMapMode.READ);return [...new Float32Array(buffer.getMappedRange())];}finally{buffer.unmap();buffer.destroy();}
}
function slice(values:Float32Array,nx:number,ny:number,z:number,components=1){return [...values.subarray(z*nx*ny*components,(z+1)*nx*ny*components)];}
const build=spawnSync("cargo",["build","--manifest-path","rust/Cargo.toml","-p","fluid-core","--example","uniform_geometric_scene","--release"],{stdio:"inherit"});assert.equal(build.status,0);
const checkedFrames:number[]=(process.env.FLUID_UNIFORM_PARITY_FRAMES??"1,5,30").split(",").map(Number);
const wasmArtifacts=process.argv.includes("--wasm")?["scalar","simd"]:[];
const wasmRunners: {artifact:string;run:(request:string)=>string}[]=[];
for(const artifact of wasmArtifacts){
  const root=new URL(`../../public/wasm/fluid-wasm/${artifact}/`,import.meta.url);
  const wasmModule=await import(new URL("fluid_wasm.js",root).href);
  await wasmModule.default({module_or_path:readFileSync(new URL("fluid_wasm_bg.wasm",root))});
  assert.equal(typeof wasmModule.run_uniform_geometric_scene,"function",`Rebuild ${artifact} Wasm`);
  wasmRunners.push({artifact,run:wasmModule.run_uniform_geometric_scene});
}
const referenceFiles=["lib/methods/uniform/webgpu-uniform-reference.wgsl.ts","lib/methods/uniform/uniform-volume.wgsl.ts","lib/methods/uniform/webgpu-uniform-pressure-multigrid.wgsl.ts","lib/methods/uniform/webgpu-uniform-velocity-extrapolation.wgsl.ts","lib/methods/uniform/uniform-geometric-parameters.ts","lib/methods/uniform/parameters.ts","lib/methods/uniform/pressure-policy.ts","lib/methods/uniform/pressure-plan.ts","lib/methods/uniform/uniform-volume-initial.ts","lib/methods/uniform/webgpu-uniform-reference.ts","lib/methods/uniform/webgpu-uniform-pressure-multigrid.ts","lib/methods/uniform/uniform-volume-donor-sum.wgsl.ts","lib/core/geometric-plane-box.wgsl.ts","tools/wasm/uniform-geometric-parity.ts"];
const reference={commit:spawnSync("git",["rev-parse","HEAD"],{encoding:"utf8"}).stdout.trim(),files:Object.fromEntries(referenceFiles.map(file=>[file,createHash("sha256").update(readFileSync(file)).digest("hex")])),dimension:2,oracle:"WGSL with single physical Z cell and symmetry boundary",dt:1/30};
const source=process.env.WEBGPU_NODE_MODULE;assert.ok(source,"WEBGPU_NODE_MODULE is required");
await acquireWebGPUExclusiveLock("dawn-test","uniform-geometric Rust 2D scene parity");
let device:GPUDevice|undefined;
const report:unknown[]=[];
const parityFailures:string[]=[];
// Fixed one-step migration bounds; scene fixtures do not relax these by profile.
const oneStepBounds={volumeMaxError:2e-5,phiMaxError:2e-6,velocityMaxError:5e-5};
try {
  const dawn=await import(pathToFileURL(source).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn as NodeDawnProvider,[`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`]);
  const adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",event=>{event.preventDefault();errors.push(event.error.message);});
  for(const name of ["stationary-pool","hydrostatic-pool","dam-collapse","ceiling-release","open-top","tall-pool","empty","full","mirrored-dam","embedded-ceiling","odd-pool","wide-dam"]){
    if(process.env.FLUID_UNIFORM_PARITY_SCENE && name!==process.env.FLUID_UNIFORM_PARITY_SCENE)continue;
    // wide-dam spans 3x2 pages of 32, so its moving front runs the phi window.
    const expectedNx=name==="odd-pool"?15:name==="wide-dam"?96:16;
    const expectedNy=name==="tall-pool"?64:name==="wide-dam"?48:16;
    const scene=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
    scene.container.width_m=expectedNx*0.05;scene.container.height_m=expectedNy*0.05;scene.container.depth_m=0.05;scene.container.depthBoundary="symmetry";scene.container.top=name==="open-top"?"open":"closed";scene.container.fillFraction=name==="empty"?0:name==="full"?1:name==="tall-pool"?0.125:0.5;
    scene.voxelDomain.finestCellSize_m=0.05;scene.fluid.initialCondition="tank-fill";scene.fluid.initialLiquidVolumes=[];scene.fluid.gravity_m_s2={x:0,y:name==="stationary-pool"?0:-9.81,z:0};scene.fluid.dynamicViscosity_Pa_s=0;scene.fluid.surfaceTension_N_m=0;scene.rigidBodies=[];scene.fluid.inflow=undefined;
    scene.solidVoxels=[...boxSolidVoxelShell([expectedNx,expectedNy,1],{top:scene.container.top})];
    if(name==="embedded-ceiling")scene.solidVoxels.push({operation:"fill",minimum:[0,12,0],maximumExclusive:[expectedNx,13,1]});
    // The oracle is the 3D default algorithm; overrides are for diagnosis only.
    const values=resolveUniformGeometricValues(JSON.parse(process.env.FLUID_UNIFORM_PARITY_VALUES??"{}"));
    // pageSize is a WebGPU storage choice outside the native contract.
    const nativeValues=Object.fromEntries(UNIFORM_GEOMETRIC_NATIVE_PARAMS.map(p=>[p.key,values[p.key]]));
    const solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,{...uniformGeometricSolverOptions(values,scene),referenceDimension:2},()=>{});
    try {
      const {nx,ny,nz}=solver.info;assert.deepEqual([nx,ny,nz],[expectedNx,expectedNy,1]);
      if(["dam-collapse","mirrored-dam","ceiling-release","embedded-ceiling","wide-dam"].includes(name)){
        const v=new Float32Array(nx*ny*nz);const phi=new Float32Array((nx+1)*(ny+1)*(nz+1));
        for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++)v[x+nx*(y+ny*z)]=name==="wide-dam"?(x<36&&y<36?1:0):name==="ceiling-release"?(x>=4&&x<12&&y>=12?1:0):name==="embedded-ceiling"?(x>=4&&x<12&&y>=8&&y<12?1:0):(name==="mirrored-dam"?x>=10:x<6)&&y<12?1:0;
        for(let z=0;z<=nz;z++)for(let y=0;y<=ny;y++)for(let x=0;x<=nx;x++)phi[x+(nx+1)*(y+(ny+1)*z)]=(name==="wide-dam"?Math.max(x-36,y-36):name==="ceiling-release"?Math.max(4-x,x-12,12-y):name==="embedded-ceiling"?Math.max(4-x,x-12,8-y,y-12):Math.max(name==="mirrored-dam"?10-x:x-6,y-12))*0.05;
        device.queue.writeTexture({texture:solver.volumeTexture},v,{bytesPerRow:nx*4,rowsPerImage:ny},[nx,ny,nz]);
        device.queue.writeTexture({texture:solver.vertexPhiTexture!},phi,{bytesPerRow:(nx+1)*4,rowsPerImage:ny+1},[nx+1,ny+1,nz+1]);
      }
      const solidWorld=solidWorldForScene(scene);
      const initialCapacity=Array.from({length:nx*ny},(_,i)=>1-sampleSolidWorld(solidWorld,[i%nx,Math.floor(i/nx),0]).solidFraction);
      const initialVolume=slice(await read(device,solver.volumeTexture),nx,ny,Math.floor(nz/2));
      const initialPhi=slice(await read(device,solver.vertexPhiTexture!),nx+1,ny+1,Math.floor(nz/2));
      const samples=[];
      // Schedule state the matched Rust step must reproduce. The phi window is
      // census-driven and exists only with more than one page.
      const phiRegion=solver["phiRegion"] as GPUBuffer|undefined;
      const headerWords=async()=>phiRegion?[...new Uint32Array(new Float32Array(await readBuffer(device!,phiRegion,1024)).buffer)]:undefined;
      const rustCycleBudgets:number[]=[];
      for(let frame=1;frame<=30;frame++){
        const selected=checkedFrames.includes(frame);
        const before: {volume:number[];phi:number[];velocity:number[];boundary:number[];phiRegion?:number[]}|null=selected?{
          volume:slice(await read(device,solver.volumeTexture),nx,ny,0),
          phi:slice(await read(device,solver.vertexPhiTexture!),nx+1,ny+1,0),
          velocity:slice(await read(device,solver.velocityTexture),nx,ny,0,4),
          boundary:await readBuffer(device,solver.negativeBoundaryVelocityBuffer,solver.negativeBoundaryVelocityBytes),
          phiRegion:await headerWords(),
        }:null;
        assert.ok(solver.advanceTo(frame/30));
        // The lagged CM11a budget reads the previous step's executed cycles.
        // Stats every step pin that lag to exactly one, as Rust implements it.
        const cycleBudget=solver.info.uniformPressureCyclesEncoded!;
        rustCycleBudgets.push(cycleBudget);
        await solver.readStats();
        if(!checkedFrames.includes(frame))continue;
        await device.queue.onSubmittedWorkDone();
        const volume=slice(await read(device,solver.volumeTexture),nx,ny,Math.floor(nz/2));
        const phi=slice(await read(device,solver.vertexPhiTexture!),nx+1,ny+1,Math.floor(nz/2));
        const velocity4=slice(await read(device,solver.velocityTexture),nx,ny,Math.floor(nz/2),4);
        const velocity=velocity4.filter((_,i)=>i%4<2);
        const released=velocity4.filter((_,i)=>i%4===3).map(bits=>(bits&3)|((bits>>1)&12));
        const run:SpawnSyncReturns<string>=spawnSync("rust/target/release/examples/uniform_geometric_scene",[],{input:JSON.stringify({dimensions:[nx,ny],cellSize:[0.05,0.05],volume:initialVolume,capacity:initialCapacity,phi:initialPhi,gravity:[0,scene.fluid.gravity_m_s2.y],density:scene.fluid.density_kg_m3,options:nativeValues,openTop:scene.container.top==="open",frames:frame,gpuPressureCycleBudgets:rustCycleBudgets}),encoding:"utf8",maxBuffer:32*1024*1024});
        assert.equal(run.status,0,run.stderr);const rust=JSON.parse(run.stdout);
        const diff=(a:number[],b:number[])=>Math.max(...a.map((v,i)=>Math.abs(v-b[i]!)));
        const sample={frame,volumeMaxError:diff(volume,rust.volume),phiMaxError:diff(phi,rust.phi),velocityMaxError:diff(velocity,rust.velocity.flat()),gpuVolume:volume.reduce((a,b)=>a+b,0),rustVolume:rust.receipts.at(-1).volume,rustMaxSpeed:rust.receipts.at(-1).maxSpeed,rustPressure:rust.receipts.at(-1).pressure};
        const cellCount=nx*ny*nz;
        const balance=await readBuffer(device,solver["conditioningScratch"] as GPUBuffer,8,cellCount*12);
        const pressureTexture=solver.gridPressureTexture;
        const gpuSchedule={pressureCycleBudget:cycleBudget,pressureCyclesExecuted:solver.info.uniformPressureCyclesExecuted,
          pressureCyclesConverged:solver.info.uniformPressureCyclesConverged,balanceRate:balance[0],
          balanceRecords:new Uint32Array(new Float32Array([balance[1]!]).buffer)[0],phiRegionBefore:before!.phiRegion,phiRegionAfter:await headerWords(),
          pressure:{dims:[pressureTexture.width,pressureTexture.height,pressureTexture.depthOrArrayLayers],values:[...await read(device,pressureTexture)]},
          // Stage outputs for localizing a divergence: SL advection plus forces
          // before projection, and the gamma target the balance and sharpening read.
          preProjectionVelocity:slice(await read(device,solver.preProjectionVelocityTexture),nx,ny,0,4).filter((_,i)=>i%4<2),
          gamma:slice(await read(device,solver.physicsFieldsForQA.gamma),nx,ny,0),
          // CM11a pyramid as the last solve left it: haloed, 3 z layers.
          multigrid:await Promise.all((solver["pressureMultigrid"] as {levels:readonly {dimensions:readonly number[];phi:readonly GPUTexture[];rhs:readonly GPUTexture[];coefficients:GPUTexture}[]}).levels.map(async level=>({
            dims:level.dimensions,phi:[[...await read(device!,level.phi[0]!)],[...await read(device!,level.phi[1]!)]],
            rhs:[...await read(device!,level.rhs[0]!)],coefficients:[...await read(device!,level.coefficients)]})))};
        const matchedInput:Record<string,unknown>={auditSurface:true,pressureCycleBudget:cycleBudget,phiRegionHeader:gpuSchedule.phiRegionAfter,dimensions:[nx,ny],cellSize:[0.05,0.05],volume:before!.volume,capacity:initialCapacity,phi:before!.phi,velocity:Array.from({length:nx*ny},(_,i)=>before!.velocity.slice(4*i,4*i+2)),released:Array.from({length:nx*ny},(_,i)=>{const bits=before!.velocity[4*i+3]!;return (bits&3)|((bits>>1)&12);}),lowX:before!.boundary.slice(0,ny),lowY:before!.boundary.slice(ny,ny+nx),gravity:[0,scene.fluid.gravity_m_s2.y],density:scene.fluid.density_kg_m3,options:nativeValues,openTop:scene.container.top==="open",frames:1};
        const matched:SpawnSyncReturns<string>=spawnSync("rust/target/release/examples/uniform_geometric_scene",[],{input:JSON.stringify(matchedInput),encoding:"utf8",maxBuffer:32*1024*1024});
        assert.equal(matched.status,0,String(matched.stderr));const one=JSON.parse(String(matched.stdout));
        const advectedPhi3d:number[]=[...await read(device,solver.advectedVertexPhiTexture!)];
        const advectedPhi:number[]=advectedPhi3d.slice(0,(nx+1)*(ny+1));
        const isolated:SpawnSyncReturns<string>=spawnSync("rust/target/release/examples/uniform_geometric_scene",[],{input:JSON.stringify({...matchedInput,phi:advectedPhi,redistanceOnly:true}),encoding:"utf8"});
        assert.equal(isolated.status,0,isolated.stderr);
        const redistance:{phi:number[]}=JSON.parse(isolated.stdout);
        const tileSource=solver.tileClassSource?.records;
        const gpuTileClasses=tileSource?[...new Uint32Array(new Float32Array(await readBuffer(device,tileSource.buffer,tileSource.size!,tileSource.offset??0)).buffer)].filter((_,i)=>i%4===3):undefined;
        console.log(JSON.stringify({scene:name,frame,surfaceStages:{advectionMaxError:diff(advectedPhi,one.advectedPhi),redistanceMatchedInputMaxError:diff(phi,redistance.phi)}}));
        writeFileSync(`${process.env.FLUID_UNIFORM_PARITY_DUMP_DIR??"/tmp"}/fluid-uniform-${name}-${frame}.json`,JSON.stringify({input:matchedInput,gpu:{volume,phi,velocity,released,advectedPhi,advectedPhi3d,tileClasses:gpuTileClasses,...gpuSchedule},rust:one,redistance}));
        for(const wasm of wasmRunners){
          const actual=JSON.parse(wasm.run(JSON.stringify(matchedInput)));
          const expected={...one};delete expected.elapsedMs;
          assert.deepEqual(actual,expected,`${wasm.artifact} versus native: ${name}, frame ${frame}`);
        }
        let tileMismatches=0;
        if(gpuTileClasses) {
          tileMismatches=gpuTileClasses.filter((value,i)=>value!==one.tileClasses?.[i]).length;
          if(tileMismatches)parityFailures.push(`${name} frame ${frame}: ${tileMismatches} tile-class mismatches`);
        }
        const releaseMismatches=released.filter((value,i)=>value!==one.released[i]).length;
        if(releaseMismatches)parityFailures.push(`${name} frame ${frame}: ${releaseMismatches} release-mask mismatches`);
        const matchedStep={releaseMismatches,tileMismatches,volumeMaxError:diff(volume,one.volume),phiMaxError:diff(phi,one.phi),velocityMaxError:diff(velocity,one.velocity.flat())};
        for(const key of Object.keys(oneStepBounds) as (keyof typeof oneStepBounds)[]){
          if(!Number.isFinite(matchedStep[key])||matchedStep[key]>oneStepBounds[key])parityFailures.push(`${name} frame ${frame}: ${key} ${matchedStep[key]} > ${oneStepBounds[key]}`);
        }
        console.log(JSON.stringify({scene:name,frame,matchedStep}));
        samples.push({...sample,matchedStep});console.log(JSON.stringify({scene:name,...sample}));
      }
      report.push({scene:name,profile:values,samples});
    }finally{solver.destroy();}
  }
  assert.deepEqual(errors,[]);
  if(process.argv.includes("--verify"))assert.deepEqual(parityFailures,[],"Uniform 2D one-step parity is not accepted");
}catch(error){
  writeFileSync("/tmp/fluid-uniform-parity-error.txt", String(error instanceof Error ? error.stack : error));
  throw error;
}finally{
  device?.destroy();await releaseWebGPUExclusiveLock();
  writeFileSync(process.env.FLUID_UNIFORM_PARITY_REPORT??"/tmp/fluid-uniform-geometric-parity.json",JSON.stringify({reference,wasmArtifacts,oneStepBounds,parityFailures,scenes:report},null,2)+"\n");
}
