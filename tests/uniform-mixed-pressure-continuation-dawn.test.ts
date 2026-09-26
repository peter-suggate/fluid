import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { WebGPUUniformPressureMultigrid } from "../lib/methods/uniform/webgpu-uniform-pressure-multigrid";
import { uniformGeometricSolverOptions } from "../lib/methods/uniform/uniform-geometric-options";
import { readMixedBuffer, readMixedTexture } from "./helpers/uniform-mixed-native-fields";
import { UniformMixedPressureContinuation } from "../lib/methods/uniform/uniform-mixed-pressure-continuation";
import { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { createUniformMixedLayout, uniformMixedPressureLevel } from "../lib/methods/uniform/uniform-mixed-layout";
import { uniformMixedPressureStorage } from "../lib/methods/uniform/uniform-mixed-pressure-boundary.wgsl";
import { refinementRegionLattice } from "../lib/core/refinement-regions";
import { UniformMixedPressureCycles } from "../lib/methods/uniform/uniform-mixed-pressure-cycles";
import { geometricSeamRows } from "./helpers/uniform-geometric-seam";
import { DEFAULT_UNIFORM_CM11A_SCHEDULE } from "../lib/methods/uniform/pressure-policy";
import { UniformMixedPressureAcceptance } from "../lib/methods/uniform/uniform-mixed-pressure-acceptance";
const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("mixed 4h pressure continuation uses the native lower hierarchy without touching finest fields",{timeout:240000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform mixed native pressure continuation");let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const scene=structuredClone(sceneDocument(getSceneDefinition("sparse-cm12-long-dam-break")));
  scene.container.width_m=scene.container.height_m=scene.container.depth_m=.8;scene.voxelDomain.finestCellSize_m=.025;scene.solidVoxels=[];
  scene.container.top="closed";
  scene.fluid.initialDamBreakDimensions_m={x:.2,y:.4,z:.4};
  solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,{
   ...uniformGeometricSolverOptions({},scene),volumePages:16,activeRegion:false,pressureWindow:false,pressureCycleDispatch:"direct",pressureCycleBudget:"fixed",
  },()=>{});
  console.log("native continuation host compiled");
  const internal=solver as unknown as {pressureMultigrid:WebGPUUniformPressureMultigrid;pressureMultigridGroup:GPUBindGroup;params:GPUBuffer};
  const mg=internal.pressureMultigrid;
  // Strict manufactured pressure problem, with unit dt/rho.
  device.queue.writeBuffer(internal.params,12,new Float32Array([1]));
  device.queue.writeBuffer(internal.params,48,new Float32Array([1]));
  mg.setResidualTolerance(0);
  // Disabling the outer early exit does not disable the inner relative gate.
  // The manufactured tolerance requires native strict coarse accuracy (the
  // existing 1e-4 absolute floor and 4096-sweep cap), not its live 10% budget.
  mg.setCoarseAccuracy(0);
  const continuation=mg.prepareMixedContinuation();
  assert.ok(mg.levelCount>3,"fixture must continue below 4h");
  const d=continuation.phi.dimensions,n=d.reduce((p,v)=>p*v,1),w=.1;
  const pressure=new Float32Array(n),rhs=new Float32Array(n),minimum=new Float32Array(n),phi=new Float32Array(n).fill(.5*w),topology=new Float32Array(n*4);
  for(let z=0;z<d[2];z++)for(let y=0;y<d[1];y++)for(let x=0;x<d[0];x++){
   const p=[x,y,z],i=x+d[0]*(y+d[1]*z),inside=p.every((v,a)=>v>0&&v<d[a]!-1);
   if(inside){phi[i]=-1;minimum[i]=-3.402823e38;topology[4*i]=1;
    for(let a=0;a<3;a++)topology[4*i+a+1]=p[a]===d[a]!-2?.5:1;
    if(y===d[1]-2)rhs[i]=.5/(w*w);
   }else{
    for(let a=0;a<3;a++)if(p[a]===0&&p.every((v,b)=>a===b||v>0&&v<d[b]!-1))topology[4*i+a+1]=.5;
    if(y===d[1]-1&&x>0&&x<d[0]-1&&z>0&&z<d[2]-1)rhs[i]=-1/(w*w);
   }
  }
  const write=(field:typeof continuation.phi,values:Float32Array<ArrayBuffer>)=>{
   if(field.buffer)device!.queue.writeBuffer(field.buffer.buffer,field.buffer.offset??0,values);
   else device!.queue.writeTexture({texture:field.texture},values,{bytesPerRow:d[0]*4*(values.length/n),rowsPerImage:d[1]},[...d]);
  };
  for (const kind of ["v", "full"] as const) {
  write(continuation.pressure,pressure);write(continuation.rhs,rhs);write(continuation.minimum,minimum);write(continuation.phi,phi);write(continuation.topology,topology);
  console.log("native continuation plan prepared");
  const scratch=continuation.pressure.buffer?.buffer;const before:Float32Array|undefined=scratch?await readMixedBuffer(device,scratch):undefined;
  // This entry point is one inner traversal, not an independently accepted
  // pressure solve. Compare the bridge with that native operation exactly;
  // convergence is checked on the complete mixed Full-/V-cycle path below.
  const encoder=device.createCommandEncoder();continuation.encode(encoder,internal.pressureMultigridGroup,kind);device.queue.submit([encoder.finish()]);
  let actual:Float32Array;
  if(scratch){const after=await readMixedBuffer(device,scratch),offset=(continuation.pressure.buffer!.offset??0)/4;actual=after.subarray(offset,offset+n);
   assert.deepEqual(after.subarray(0,offset),before!.subarray(0,offset),"native finest scratch was modified");
  }else actual=await readMixedTexture(device,continuation.pressure.texture);
  const layout=createUniformMixedLayout(refinementRegionLattice(scene),[], true, 4),ownership=new UniformMixedOwnership(device,layout),count=uniformMixedPressureStorage(layout).count;
  const borrowed:GPUBuffer[]=[];
  try{
   const buffer=(n:number)=>{const b=device!.createBuffer({size:4*n,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});borrowed.push(b);return {buffer:b};};
   const fields={pressure:buffer(count),rhs:buffer(count),minimum:buffer(count),phi:buffer(layout.cellCount)};
   const bridge=new UniformMixedPressureContinuation(device,ownership,continuation);await bridge.initialize();
   const mapping:[number,number][]=[];const size=d.map(v=>v-2),cells=layout.cellCount;
   for(let z=0;z<size[2]!;z++)for(let y=0;y<size[1]!;y++)for(let x=0;x<size[0]!;x++){
    const p=[x,y,z],i=x+size[0]!*(y+size[1]!*z),native=(x+1)+d[0]*((y+1)+d[1]*(z+1));mapping.push([i,native]);
    for(let a=0;a<3;a++)for(const side of [0,1])if(p[a]===(side?size[a]!-1:0)){
     const halo=cells+(a===0?side*size[1]!*size[2]!+y+size[1]!*z:a===1?2*size[1]!*size[2]!+side*size[0]!*size[2]!+x+size[0]!*z:2*(size[1]!*size[2]!+size[0]!*size[2]!)+side*size[0]!*size[1]!+x+size[0]!*y);
     const q=p.map(v=>v+1);q[a]=side?d[a]!-1:0;mapping.push([halo,q[0]!+d[0]*(q[1]!+d[1]*q[2]!)]);
    }
   }
   for(const [key,source] of [["rhs",rhs],["minimum",minimum],["phi",phi]] as const){
    const values=new Float32Array(key==="phi"?cells:count);for(const [i,j] of mapping)if(i<values.length)values[i]=source[j]!;
    device.queue.writeBuffer(fields[key].buffer,0,values);
   }
   const group=bridge.bind(fields),e=device.createCommandEncoder();bridge.encode(e,group,internal.pressureMultigridGroup,kind);device.queue.submit([e.finish()]);
   const mixed=await readMixedBuffer(device,fields.pressure.buffer);
   for(const [i,j] of mapping)assert.equal(mixed[i],actual[j],`bridged native continuation ${i}`);
   // A second correction in the same solve must reuse immutable topology,
   // coefficients and work lists without changing the numerical result.
   const repeated=device.createCommandEncoder();repeated.clearBuffer(fields.pressure.buffer);
   bridge.encode(repeated,group,internal.pressureMultigridGroup,kind,false);device.queue.submit([repeated.finish()]);
   const reused=await readMixedBuffer(device,fields.pressure.buffer);
   for(const [i,j] of mapping)assert.equal(reused[i],actual[j],`reused native ${kind} topology ${i}`);
  }finally{borrowed.forEach(b=>b.destroy());ownership.destroy();}
   assert.deepEqual(errors,[]);
   console.log(`native ${kind} continuation bridge matched exactly`);
  }
  // The manufactured solution must be a fixed point of the actual native
  // operator, including its separating-wall halo, before it is a valid oracle
  // for convergence of the mixed traversal.
  {
   const exact=new Float32Array(n);
   for(let z=0;z<d[2];z++)for(let y=0;y<d[1];y++)for(let x=0;x<d[0];x++){
    const inside=[x,y,z].map((v,a)=>v>0&&v<d[a]!-1);
    if(inside.filter(Boolean).length>=2&&y!==d[1]-1)exact[x+d[0]*(y+d[1]*z)]=1;
   }
   write(continuation.pressure,exact);write(continuation.rhs,rhs);write(continuation.minimum,minimum);write(continuation.phi,phi);write(continuation.topology,topology);
   const e=device.createCommandEncoder();continuation.encode(e,internal.pressureMultigridGroup,"v");device.queue.submit([e.finish()]);
   const values=continuation.pressure.buffer?await readMixedBuffer(device,continuation.pressure.buffer.buffer):await readMixedTexture(device,continuation.pressure.texture);
   const offset=(continuation.pressure.buffer?.offset??0)/4;
   let error=0;for(let i=0;i<n;i++)error=Math.max(error,Math.abs(values[offset+i]!-exact[i]!));
   assert.ok(error<2e-6,`native manufactured solution is not a fixed point: ${error}`);
  }
  if(process.env.FLUID_MIXED_PRESSURE_NATIVE_DIAGNOSTIC==="1"){
   // Independent outer Full/V traversal on the native 4h lattice. This
   // distinguishes a mixed operator regression from the baseline's finite
   // convergence under the manufactured fixture's fixed work budget.
   let p=new Float32Array(n);
   const readPressure=async()=>{const all=continuation.pressure.buffer?await readMixedBuffer(device!,continuation.pressure.buffer.buffer):await readMixedTexture(device!,continuation.pressure.texture);const offset=(continuation.pressure.buffer?.offset??0)/4;return all.slice(offset,offset+n);};
   const residual=(values:Float32Array)=>{
    const out=new Float32Array(n);
    for(let z=0;z<d[2];z++)for(let y=0;y<d[1];y++)for(let x=0;x<d[0];x++){
     const q=[x,y,z],at=x+d[0]*(y+d[1]*z),inside=q.map((v,a)=>v>0&&v<d[a]!-1);let applied=0;
     if(inside.every(Boolean)){
      for(let a=0;a<3;a++)for(const sign of [-1,1]){const next=[...q];next[a]!+=sign;const neighbor=next[0]!+d[0]*(next[1]!+d[1]*next[2]!);const coefficient=(next[a]===0||next[a]===d[a]!-1?.5:1)/(w*w);applied+=coefficient*(values[at]!-values[neighbor]!);}
     }else if(inside.filter(Boolean).length===2){
      const axis=inside.indexOf(false),next=[...q];next[axis]=q[axis]===0?1:d[axis]!-2;
      applied=.5/(w*w)*(values[at]!-values[next[0]!+d[0]*(next[1]!+d[1]*next[2]!)]!);
     }
     out[at]=rhs[at]!-applied;
    }
    return out;
   };
   for(const kind of ["full","full","full","v","v","v","v"] as const){
    const saved=p.slice(),b=kind==="full"?residual(p):rhs;
    const lower=kind==="full"?Float32Array.from(minimum,(v,i)=>v-p[i]!):minimum;
    write(continuation.pressure,kind==="full"?new Float32Array(n):p);write(continuation.rhs,b);write(continuation.minimum,lower);write(continuation.phi,phi);write(continuation.topology,topology);
    const e=device.createCommandEncoder();continuation.encode(e,internal.pressureMultigridGroup,kind);device.queue.submit([e.finish()]);p=await readPressure();
    if(kind==="full")p=Float32Array.from(p,(v,i)=>v+saved[i]!);
    let error=0;for(let z=1;z<d[2]-1;z++)for(let y=1;y<d[1]-1;y++)for(let x=1;x<d[0]-1;x++)error=Math.max(error,Math.abs(p[x+d[0]*(y+d[1]*z)]!-1));
    console.log({nativeOuterCycle:kind,error});
   }
  }
  // Exercise the complete h/2h/native-4h correction path, including the
  // separating-wall minima through Full-Cycles and V-cycles.
  const mixedLayout=createUniformMixedLayout(refinementRegionLattice(scene),process.env.FLUID_MIXED_PRESSURE_LAYOUT==="coarse"?[]:[{id:"manual-fine",rule:"minimum-cell-size",minimumCellSize_cells:1,maximumCellSize_cells:1,
   min_m:{x:-.1,y:.3,z:-.1},max_m:{x:0,y:.4,z:0}}], true, 4);
  const fieldsOwned:GPUBuffer[]=[],owners:UniformMixedOwnership[]=[];
  try{
   const buffer=(count:number)=>{const b=device!.createBuffer({size:count*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.UNIFORM|GPUBufferUsage.INDIRECT|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});fieldsOwned.push(b);return {buffer:b};};
   const levels=[mixedLayout,uniformMixedPressureLevel(mixedLayout,2),uniformMixedPressureLevel(mixedLayout,4)].map((layout,i)=>{
    const ownership=new UniformMixedOwnership(device!,layout);owners.push(ownership);const count=uniformMixedPressureStorage(layout).count;
    return {ownership,pressure:buffer(count),slopes:buffer(layout.cellCount*4),frozen:buffer(count),rhs:[buffer(count),buffer(count)] as const,
     residual:buffer(count),minimum:Array.from({length:i===0?2:1},()=>buffer(count)),phi:buffer(layout.cellCount)};
   });
   const root=levels[0]!,bottom=levels[2]!,storage=uniformMixedPressureStorage(mixedLayout),cells=geometricSeamRows(mixedLayout,()=>[0,0,0]).cells;
   const b=new Float32Array(storage.count),min=new Float32Array(storage.count).fill(-3.402823e38),dims=mixedLayout.lattice.dimensions.map(v=>v/storage.width);
   const validHalos:number[]=[];
   cells.forEach((cell,i)=>{for(let axis=0;axis<3;axis++)for(const side of [0,1]){
    if(side===0?cell.min[axis]!==0:cell.min[axis]!+cell.width!==mixedLayout.lattice.dimensions[axis])continue;
    const p=cell.min.map(v=>v/storage.width),at=cells.length+(axis===0?side*dims[1]!*dims[2]!+p[1]!+dims[1]!*p[2]!:axis===1?2*dims[1]!*dims[2]!+side*dims[0]!*dims[2]!+p[0]!+dims[0]!*p[2]!:2*(dims[1]!*dims[2]!+dims[0]!*dims[2]!)+side*dims[0]!*dims[1]!+p[0]!+dims[0]!*p[1]!);
    min[at]=0;validHalos.push(at);if(axis===1&&side===1){const coefficient=.5/(cell.width*.025)**2;b[i]=coefficient;b[at]=-2*coefficient;}
   }});
   device.queue.writeBuffer(root.rhs[0].buffer,0,b);device.queue.writeBuffer(root.minimum[0]!.buffer,0,min);device.queue.writeBuffer(root.phi.buffer,0,new Float32Array(cells.length).fill(-1));
   const bridge=new UniformMixedPressureContinuation(device,bottom.ownership,continuation);await bridge.initialize();
   const groups=new Map<GPUBuffer,GPUBindGroup>();for(const rhs of bottom.rhs)groups.set(rhs.buffer,bridge.bind({pressure:bottom.pressure,rhs,minimum:bottom.minimum[0]!,phi:bottom.phi}));
   const cycle=new UniformMixedPressureCycles(device,levels,buffer(storage.count),(encoder,rhs,kind)=>bridge.encode(encoder,groups.get(rhs.buffer)!,internal.pressureMultigridGroup,kind),[true,true],DEFAULT_UNIFORM_CM11A_SCHEDULE,{openTop:false});await cycle.initialize();
   const acceptance=new UniformMixedPressureAcceptance(device,root.ownership);await acceptance.initialize();
   const state=buffer(8).buffer,params=buffer(4).buffer;
   device.queue.writeBuffer(params,0,new Float32Array([1,0,0,0]));
   const ag=acceptance.bind({residual:root.residual,state,params});
   const encoder=device.createCommandEncoder();cycle.encodeSurfaceRestriction(encoder);
   const snapshots:{kind:string;pressure:GPUBuffer;state:GPUBuffer}[]=[];
   const checkpoint=(kind:"initial"|"cycle")=>{
    cycle.encodeMeasure(encoder);acceptance.encode(encoder,ag,state,kind);
    if(process.env.FLUID_MIXED_PRESSURE_DIAGNOSTICS==="1"){
     const p=buffer(storage.count).buffer,s=buffer(8).buffer;
     encoder.copyBufferToBuffer(root.pressure.buffer,0,p,0,p.size);encoder.copyBufferToBuffer(state,0,s,0,32);
     snapshots.push({kind,pressure:p,state:s});
    }
   };
   checkpoint("initial");
   const order=process.env.FLUID_MIXED_PRESSURE_ORDER==="v-first"?["v","full"] as const:["full","v"] as const;
   for(const kind of order)for(let i=0;i<(kind==="full"?DEFAULT_UNIFORM_CM11A_SCHEDULE.fullCycles:DEFAULT_UNIFORM_CM11A_SCHEDULE.vCycles);i++){
    if(kind==="full")cycle.encodeFullCycle(encoder);else cycle.encodeVCycle(encoder);checkpoint("cycle");
   }
   cycle.encodeMeasure(encoder);device.queue.submit([encoder.finish()]);
   const actual=await readMixedBuffer(device,root.pressure.buffer),residual=await readMixedBuffer(device,root.residual.buffer);
   for(const snapshot of snapshots){
    const p=await readMixedBuffer(device,snapshot.pressure),s=await readMixedBuffer(device,snapshot.state);
    console.log({checkpoint:snapshot.kind,error:Math.max(...p.subarray(0,cells.length).map(v=>Math.abs(v-1))),candidate:s[0],accepted:s[1],state:[...new Uint32Array(s.buffer)].slice(3)});
   }
   let error=0,norm=0;for(let i=0;i<cells.length;i++){error=Math.max(error,Math.abs(actual[i]!-1));norm=Math.max(norm,residual[i]!);}for(const at of validHalos)norm=Math.max(norm,residual[at]!);
   console.log(`mixed/native wall cycle: ${cells.length} owners, pressure error ${error}, projected residual ${norm}`);
   assert.ok(error<2e-4,`mixed/native pressure error ${error}`);assert.ok(norm<2e-3,`mixed/native wall residual ${norm}`);
  }finally{fieldsOwned.forEach(b=>b.destroy());owners.forEach(o=>o.destroy());}
  assert.deepEqual(errors,[]);
 }finally{solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});
