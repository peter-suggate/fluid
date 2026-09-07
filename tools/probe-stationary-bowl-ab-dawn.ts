/** Matched UI-scene A/B, with independent physics and published-geometry metrics. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import { sceneDocument } from '../lib/core/scene-definition';
import { getSceneDefinition } from '../lib/core/scenes';
import { resolveMethodValues } from '../lib/core/method-contract';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { adaptiveMassMethod, adaptiveMassSolverOptions } from '../lib/methods/adaptive-mass/method';
import { WebGPUAdaptiveMassSolver } from '../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver';
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from '../lib/harness/webgpu-smoke-isolation';
import { readPublishedCM12Field } from './sparse-cm12-published-field';
const arg=(name:string,fallback:string)=>process.argv.find(a=>a.startsWith(`--${name}=`))?.slice(name.length+3)??fallback;
const arm=arg('arm','adaptive'), steps=Number(arg('steps','120'));
const frozen=arg('frozen','false')==='true';
const conditioning=arg('conditioning','on');
const gravity=Number(arg('gravity','0'));
const curvatureScale=Number(arg('curvature-scale','1'));
assert.ok(Number.isFinite(curvatureScale)&&curvatureScale>=0);
const stepSize=arg('dt','scene');
const captureEvery=Number(arg('capture-every','30'));
assert.ok(Number.isFinite(gravity)&&gravity>=0);
assert.ok(Number.isInteger(captureEvery)&&captureEvery>0);
const output=arg('output',`artifacts/stationary-bowl-2x/baseline/${arm}`);
assert.ok(['adaptive','fixed1','fixed2','fixed4','fixed8'].includes(arm));
assert.ok(['on','off','gamma-only','sharpen-only'].includes(conditioning));
assert.ok(Number.isInteger(steps)&&steps>=0);
await acquireWebGPUExclusiveLock('dawn-probe',`stationary-bowl-2x:${arm}`);
let gpu:GPU|undefined, device:GPUDevice|undefined, solver:WebGPUAdaptiveMassSolver|undefined;
const live=new Set<GPU>();
try {
 const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE??`${process.cwd()}/node_modules/webgpu/index.js`).href);
 Object.assign(globalThis,dawn.globals); gpu=dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND??'metal'}`]);live.add(gpu!);
 const adapter=await gpu!.requestAdapter();assert.ok(adapter);
 device=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
 const errors:string[]=[], hashes:Record<string,string>={};
 device.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);});
 const compile=device.createShaderModule.bind(device);
 device.createShaderModule=d=>{hashes[d.label??String(Object.keys(hashes).length)]=createHash('sha256').update(d.code).digest('hex');return compile(d);};
 const definition=getSceneDefinition('stationary-bowl-2x'), scene=sceneDocument(definition);
 scene.fluid.gravity_m_s2={x:0,y:-gravity,z:0};
 if(scene.fluid.initialHeightField?.kind==='quadratic'){
  scene.fluid.initialHeightField={...scene.fluid.initialHeightField as import("../lib/core/initial-height-field").QuadraticLiquidHeightField,curvatureX_mInv:.12*curvatureScale,curvatureZ_mInv:.084*curvatureScale};
 }
 if(stepSize!=='scene'){
  const requestedDt=stepSize==='1/30'?1/30:Number(stepSize);
  assert.ok(Number.isFinite(requestedDt)&&requestedDt>0);
  scene.numerics.fixedDt_s=scene.numerics.maxDt_s=requestedDt;
 }
 const width=Number(arm.replace('fixed',''));
 if(arm!=='adaptive')scene.fluid.refinementRegions=[{id:'ab-resolution',rule:'minimum-cell-size',minimumCellSize_cells:width,maximumCellSize_cells:width,
 min_m:{x:-1.2,y:0,z:-1},max_m:{x:1.2,y:1.6,z:1}}];
 const values=resolveMethodValues(adaptiveMassMethod,'balanced',{...definition.methodProfile!.overrides,
 ...(arg('pressure-tolerance','')?{pressureRelativeTolerance:Number(arg('pressure-tolerance',''))}:{}),
 ...(arg('pressure-iterations','')?{pressureIterations:Number(arg('pressure-iterations',''))}:{}),
 gammaDiffusion:conditioning==='on'||conditioning==='gamma-only'?'on':'off',
 surfaceSharpening:conditioning==='on'||conditioning==='sharpen-only'?'on':'off'});
 solver=await WebGPUAdaptiveMassSolver.createAsync(device,scene,'balanced',undefined,adaptiveMassSolverOptions(values),()=>{});
 await solver.waitForSimulationReady();if(frozen)solver.setTopologyFrozen(true);
 const [nx,ny,nz]=[solver.info.nx,solver.info.ny,solver.info.nz],h=.05,dt=scene.numerics.fixedDt_s!;
 await mkdir(output,{recursive:true});
 await writeFile(`${output}/configuration.json`,JSON.stringify({arm,steps,frozen,conditioning,gravity,curvatureScale,captureEvery,reference:gravity===0?"exact stationary surface":"initial surface; not a dynamic analytic solution",scene,values,nx,ny,nz,h,dt},null,2));
 let initial:Float32Array|undefined, initialColumns:Float64Array|undefined;
 const trace:Record<string,unknown>[]=[];
 const rms=(v:number[])=>Math.sqrt(v.reduce((s,x)=>s+x*x,0)/v.length);
 for(let step=0;step<=steps;step++){
  let deferredPolls=0;const stepStart=performance.now();
  if(step){
      const captures = new Map<string, GPUBuffer>();
      const faceCaptures = new Map<string, GPUBuffer>();
      const pressureCaptures = new Map<string, GPUBuffer>();
      const audit = (process.env.BOWL_AUDIT_STEPS ?? "").split(",").includes(String(step));
      const source = solver.fieldSnapshotSourceForQA;
      const records = audit ? (await solver.readGPUActivityPolicy()).bricks : [];
      if (audit) solver.setStageCaptureForQA((stage, encoder) => {
        if(stage === "transport-velocity-extension" && source.effectiveTransportVelocity){
          const count=4*source.cellCapacity;
          const buffer=device!.createBuffer({size:4*count,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
          encoder.copyBufferToBuffer(source.effectiveTransportVelocity,0,buffer,0,4*count);
          pressureCaptures.set("effective",buffer);
        }
        if (stage === "pressure-solve") {
          for (const [name, base, count] of [
            ["liquid", source.layout.liquid, source.cellCapacity],
            ["rhs", source.layout.rhs, source.cellCapacity],
            ["diagonal", source.layout.diagonal, source.cellCapacity],
            ["theta", source.layout.theta, source.rowCapacity],
          ] as const) {
            const buffer = device!.createBuffer({ size: 4 * count,
              usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
            encoder.copyBufferToBuffer(source.state, 4 * base, buffer, 0, 4 * count);
            pressureCaptures.set(name, buffer);
          }
        }
        if (["transport-velocity-extension", "face-preparation", "body-forces", "pressure-solve", "velocity-projection"].includes(stage)) {
          const nr = source.rowCapacity;
          const buffer = device!.createBuffer({ size: 4 * (2 * nr + 1), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
          for (const [i, base] of [source.layout.faceA, source.layout.faceB].entries())
            encoder.copyBufferToBuffer(source.state, 4 * base, buffer, 4 * i * nr, 4 * nr);
          encoder.copyBufferToBuffer(source.topologyArena, 4 * (source.frameControlBaseWords + source.faceParityWord), buffer, 8 * nr, 4);
          faceCaptures.set(stage, buffer);
        }
        if (!["transport-velocity-extension", "conservative-transport", "gamma-diffusion", "surface-sharpening", "scalar-publication"].includes(stage)) return;
        const nc = source.cellCapacity;
        const buffer = device!.createBuffer({size: 4 * (6 * nc + 1), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
        for (const [i, base] of [source.layout.densityA, source.layout.densityB, source.layout.pressure,
          source.layout.gammaA, source.layout.gammaB, source.layout.rhs].entries())
          encoder.copyBufferToBuffer(source.state, 4 * base, buffer, 4 * i * nc, 4 * nc);
        encoder.copyBufferToBuffer(source.topologyArena, 4 * (source.frameControlBaseWords + source.scalarParityWord), buffer, 24 * nc, 4);
        captures.set(stage, buffer);
      });
      while (!solver.advanceTo(step * dt, [])) {
        deferredPolls++; assert.ok(performance.now()-stepStart<30_000, `step ${step} preparation timed out`);
        await new Promise(setImmediate);
      }
      await solver.waitForTopologyReady();
      if (audit) {
        solver.setStageCaptureForQA(undefined);
        await writeFile(`${output}/${step}-template.bin`, new Uint8Array(source.templateWords.buffer, source.templateWords.byteOffset, source.templateWords.byteLength));
        await writeFile(`${output}/${step}-audit.json`, JSON.stringify({ cells: source.cellCapacity, rows: source.rowCapacity, records }));
        for (const [name, buffer] of pressureCaptures) {
          await buffer.mapAsync(GPUMapMode.READ);
          await writeFile(`${output}/${step}-pressure-${name}.bin`, new Uint8Array(buffer.getMappedRange()));
          buffer.unmap(); buffer.destroy();
        }
        for (const [stage, buffer] of faceCaptures) {
          await buffer.mapAsync(GPUMapMode.READ);
          await writeFile(`${output}/${step}-${stage}-faces.bin`, new Uint8Array(buffer.getMappedRange()));
          buffer.unmap(); buffer.destroy();
        }
        for (const [stage, buffer] of captures) {
          await buffer.mapAsync(GPUMapMode.READ);
          const data = new Float32Array(buffer.getMappedRange()), nc = source.cellCapacity;
          const parity = new Uint32Array(data.buffer, 24 * nc, 1)[0]! ^ Number(stage !== "transport-velocity-extension");
          const offset = stage === "gamma-diffusion" ? 2 * nc : parity * nc;
          const dense = new Float32Array(nx * ny * nz);
          const gamma = new Float32Array(dense.length);
          const gammaOffset = stage === "gamma-diffusion" ? 5 * nc : (3 + parity) * nc;
          for (const b of records.filter(b => b.active)) {
            const r = b.acceptedResolution, width = 8 * b.spanBricks / r;
            const first = source.templateWords[source.templateWords[11]! + 2 * (4 * b.leafId + Math.log2(r))]!;
            for (let z = 0; z < 8 * b.spanBricks; z++) for (let y = 0; y < 8 * b.spanBricks; y++) for (let x = 0; x < 8 * b.spanBricks; x++) {
              const [qx,qy,qz] = [8*b.coordinate[0]+x,8*b.coordinate[1]+y,8*b.coordinate[2]+z];
              const cell = first+Math.floor(x/width)+r*(Math.floor(y/width)+r*Math.floor(z/width));
              const at = qx+nx*(qy+ny*qz);
              dense[at] = data[offset+cell]!;
              gamma[at] = data[gammaOffset+cell]!;
            }
          }
          console.log(JSON.stringify({ step, stage, density: Math.max(...dense.map((v,i)=>Math.abs(v-dense[(nx-1-i%nx)+nx*Math.floor(i/nx)]!))) }));
          await writeFile(`${output}/${step}-${stage}.bin`, new Uint8Array(dense.buffer));
          await writeFile(`${output}/${step}-${stage}-gamma.bin`, new Uint8Array(gamma.buffer));
          buffer.unmap(); buffer.destroy();
        }
      }

  }
  if(step>10&&step%captureEvery!==0&&step!==steps)continue;
  assert.equal(solver.info.encodedSteps,step);
  const fields=await solver.readDiagnosticFields(true),activity=await solver.readGPUActivityPolicy(),frame=await solver.readFrameControlQA();
  const phi=(await readPublishedCM12Field(device,solver)).values;
  initial??=fields.density.slice();
  const heights=new Float32Array(nx*nz).fill(NaN),columns=new Float64Array(nx*nz);
  let mass=0,maxSpeed=0,maxDensityChange=0,symmetry=0,kinetic=0,potential=0;
  for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
   const at=x+nx*(y+ny*z),rho=fields.density[at]!;
   assert.ok(Number.isFinite(rho)); mass+=rho*h**3;potential+=gravity*rho*h**3*(y+.5)*h;columns[x+nx*z]!+=rho*h;
   maxDensityChange=Math.max(maxDensityChange,Math.abs(rho-initial[at]!));
   symmetry=Math.max(symmetry,Math.abs(rho-fields.density[nx-1-x+nx*(y+ny*z)]!),Math.abs(rho-fields.density[x+nx*(y+ny*(nz-1-z))]!));
   const speed=Math.hypot(fields.velocity[4*at]!,fields.velocity[4*at+1]!,fields.velocity[4*at+2]!);
   assert.ok(Number.isFinite(speed));maxSpeed=Math.max(maxSpeed,speed);kinetic+=.5*rho*speed**2*h**3;
   if(y<ny-1&&phi[at]!<=0&&phi[at+nx]!>0)heights[x+nx*z]=(y+.5-phi[at]!/(phi[at+nx]!-phi[at]!))*h;
  }
  initialColumns??=columns.slice();
  const heightError:number[]=[],curvatureXError:number[]=[],curvatureZError:number[]=[],columnChange:number[]=[];
  for(let z=8;z<nz-8;z++)for(let x=8;x<nx-8;x++){
   const at=x+nx*z,exact=.865+curvatureScale*(.12*((x+.5-nx/2)*h)**2+.084*((z+.5-nz/2)*h)**2);
   assert.ok(Number.isFinite(heights[at]));
   heightError.push(heights[at]!-exact);
   curvatureXError.push((heights[at-1]!-2*heights[at]!+heights[at+1]!)/h**2-.24*curvatureScale);
   curvatureZError.push((heights[at-nx]!-2*heights[at]!+heights[at+nx]!)/h**2-.168*curvatureScale);
   columnChange.push(columns[at]!-initialColumns[at]!);
  }
  const fullHeightErrors:number[]=[];
  for(let z=0;z<nz;z++)for(let x=0;x<nx;x++){
   const exact=.865+curvatureScale*(.12*((x+.5-nx/2)*h)**2+.084*((z+.5-nz/2)*h)**2);
   assert.ok(Number.isFinite(heights[x+nx*z]));fullHeightErrors.push(heights[x+nx*z]!-exact);
  }
  const stats=await solver.readStats();
  const row={step,time:step*dt,mass,maxSpeed,kinetic,potential,mechanicalEnergy:kinetic+potential,maxDensityChange,symmetry,
   deferredPolls,stepWallMs:performance.now()-stepStart,committedFrames:frame?.committedFrames,
   frameAcceptedGeneration:frame?.acceptedGeneration,frameCandidateGeneration:frame?.candidateGeneration,
   pressureResidual:stats.pressureRelativeResidual,pressureIterations:stats.pressureIterationsExecuted,
   divergenceRMS:rms(Array.from(fields.divergence)),
   fullHeightRMSError_mm:rms(fullHeightErrors)*1000,
   fullHeightMaxError_mm:Math.max(...fullHeightErrors.map(Math.abs))*1000,
   columnChangeRMS_mm:rms(columnChange)*1000,heightRMSError_mm:rms(heightError)*1000,
   curvatureXRMSError:rms(curvatureXError),curvatureZRMSError:rms(curvatureZError),
   widths:Object.fromEntries([1,2,4,8].map(w=>[w,activity.bricks.filter(b=>b.active&&8*b.spanBricks/b.acceptedResolution===w).length])),
   generation:activity.acceptedTopologyGeneration,fault:frame?.fault,topologyFault:activity.faultFlags};
  trace.push(row);console.log(JSON.stringify(row));
  await writeFile(`${output}/trace.json`,JSON.stringify(trace,null,2));
  for(const[name,data]of Object.entries({density:fields.density,velocity:fields.velocity,pressure:fields.pressure,heights,columns}))
   await writeFile(`${output}/${step}-${name}.bin`,new Uint8Array(data.buffer,data.byteOffset,data.byteLength));
  await writeFile(`${output}/${step}-activity.json`,JSON.stringify(activity));
  await writeFile(`${output}/${step}-frame.json`,JSON.stringify(frame));
  await writeFile(`${output}/${step}-stats.json`,JSON.stringify(stats));
  assert.equal(activity.faultFlags,0);assert.equal(activity.commitFailed,false);assert.ok(frame);assert.equal(frame.fault,0);assert.equal(frame.committedFrames,step,"every requested frame must commit");assert.deepEqual(errors,[]);
 }
 await writeFile(`${output}/shader-hashes.json`,JSON.stringify(hashes,null,2));
}catch(error){
 await mkdir(output,{recursive:true});
 await writeFile(`${output}/failure.json`,JSON.stringify({message:String(error),encodedSteps:solver?.info.encodedSteps},null,2));
 throw error;
}finally{solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();if(gpu)live.delete(gpu);}
