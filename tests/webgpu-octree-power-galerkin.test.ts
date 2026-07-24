import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";
import {
  OCTREE_POWER_GALERKIN_PIPELINES,
  WebGPUOctreePowerGalerkin,
  octreePowerGalerkinShader,
} from "../lib/webgpu-octree-power-galerkin";

test("native L2 Galerkin kernels use parent-owned gathers without probes or numeric atomics", () => {
  const refreshKernel = octreePowerGalerkinShader.match(
    /fn refreshGalerkinTarget[\s\S]*?(?=\n@compute @workgroup_size\(64\) fn clearSourceCorrection)/,
  )?.[0] ?? "";
  assert.match(refreshKernel,
    /workgroup_id[\s\S]*refreshRecordBase\(\)\+2u\*cursor[\s\S]*sum\+=sourceCoefficient\(fineEntry\)\*weight/);
  assert.match(refreshKernel,
    /cursor=first\+lane[\s\S]*cursor\+=64u[\s\S]*refreshPartial\[lane\][\s\S]*workgroupBarrier/);
  assert.doesNotMatch(refreshKernel, /pWeight|contributionIndex|left|right/,
    "numeric RAP refresh must consume baked combined weights without P-record indirections");
  assert.match(octreePowerGalerkinShader,
    /restrictSourceDefect[\s\S]*pWeight\(record\)\*\(sourceF\(RHS,fine\)-sourceApplied\(fine,A\)\)/);
  assert.match(octreePowerGalerkinShader,
    /importFineOperator[\s\S]*fixedRow\(other\.cell,other\.size\)[\s\S]*sourceColumn\(candidate\)==column[\s\S]*bitcast<u32>\(-entry\.coefficient\)/);
  assert.match(octreePowerGalerkinShader,
    /initializeFineOperator[\s\S]*bitcast<u32>\(0\.0\)/,
    "inactive fixed rows must be zero equations and must not contaminate P^T*A*P with identity mass");
  assert.match(octreePowerGalerkinShader, /smoothSource[\s\S]*diagonal==0\.0/);
  assert.match(octreePowerGalerkinShader,
    /fn stopped\(\)->bool[\s\S]*control\[1\][\s\S]*smoothSourceAtoB[\s\S]*!stopped\(\)/);
  for (const entryPoint of [
    "smoothSourceAtoB", "smoothSourceBtoA", "restrictSourceDefect",
    "prolongateTargetCorrection", "solveTargetCoarsest",
    "measureSourceResidualPartials", "measureSourceResidualReduce",
  ]) {
    assert.match(octreePowerGalerkinShader,
      new RegExp(`fn ${entryPoint}\\b[\\s\\S]{0,420}stopped\\(\\)`),
      `${entryPoint} must skip the statically encoded tail after convergence`);
  }
  assert.match(octreePowerGalerkinShader,
    /publishSourceResidual[\s\S]*control\[2\]\)\+1u[\s\S]*rr<=residualBudget[\s\S]*cycle>=u32\(params\.numerics\.z\)/);
  assert.doesNotMatch(octreePowerGalerkinShader, /atomicAdd|atomicCompareExchange|hash|probe|find\(/i);
  assert.match(WebGPUOctreePowerGalerkin.prototype.encode.toString(),
    /encodeFineOperatorImport[\s\S]*solve\.liveOperator\.leafHeaders/);
  assert.match(WebGPUOctreePowerGalerkin.prototype.encode.toString(),
    /refreshGalerkinTarget[\s\S]*true/,
    "refresh dispatch must assign one workgroup to each target entry");
  assert.match(WebGPUOctreePowerGalerkin.prototype.encode.toString(),
    /for\s*\(let cycle[\s\S]*measureSourceResidual[\s\S]*publishSourceResidual[\s\S]*\}\s*if\s*\(solve\.correction\)/);
  assert.equal("writeFineRhs" in WebGPUOctreePowerGalerkin.prototype, false,
    "the GPU solve must not retain a host-uploaded RHS fallback");
  assert.deepEqual([...OCTREE_POWER_GALERKIN_PIPELINES], [
    "initializeFineOperator", "importFineOperator", "refreshGalerkinTarget", "clearSourceCorrection",
    "smoothSourceAtoB", "smoothSourceBtoA",
    "restrictSourceDefect", "prolongateTargetCorrection", "solveTargetCoarsest",
    "measureSourceResidualPartials", "measureSourceResidualReduce",
    "publishSourceResidual", "exportLiveCorrection",
  ]);
});

test("Dawn accepts every native L2 Galerkin pipeline", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for WGSL validation",
}, async () => {
  // Isolate Metal teardown from node:test. Dawn's wrapper can release
  // auto-layout pipelines after the test worker has begun shutting down.
  const script = `
    import {pathToFileURL} from "node:url";
    import {octreePowerGalerkinShader,OCTREE_POWER_GALERKIN_PIPELINES} from "./lib/webgpu-octree-power-galerkin.ts";
    const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE).href);
    Object.assign(globalThis,dawn.globals);
    const gpu=dawn.create(["backend="+(process.env.WEBGPU_BACKEND??"metal")]);
    const adapter=await gpu.requestAdapter();
    if(!adapter)throw new Error("Dawn adapter unavailable");
    const device=await adapter.requestDevice();device.pushErrorScope("validation");
    const module=device.createShaderModule({code:octreePowerGalerkinShader});
    const info=await module.getCompilationInfo();
    const errors=info.messages.filter((message)=>message.type==="error");
    if(errors.length)throw new Error(errors.map((message)=>message.lineNum+":"+message.linePos+" "+message.message).join("\\n"));
    const pipelines=OCTREE_POWER_GALERKIN_PIPELINES.map((entryPoint)=>
      device.createComputePipeline({layout:"auto",compute:{module,entryPoint}}));
    if(pipelines.length!==OCTREE_POWER_GALERKIN_PIPELINES.length)throw new Error("pipeline count mismatch");
    const validation=await device.popErrorScope();if(validation)throw validation;device.destroy();
  `;
  const result = spawnSync(process.execPath, ["--import", "tsx", "--input-type=module", "-e", script], {
    cwd: process.cwd(), env: process.env, encoding: "utf8",
  });
  assert.equal(result.status, 0, `signal=${result.signal}\n${result.stdout}\n${result.stderr}`);
});

test("Dawn Galerkin residual gate accepts the step-13 absolute RMS floor", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for numerical GPU validation",
}, async () => {
  const script = `
    import assert from "node:assert/strict";
    import {pathToFileURL} from "node:url";
    import {octreePowerGalerkinShader} from "./lib/webgpu-octree-power-galerkin.ts";
    const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE).href);
    Object.assign(globalThis,dawn.globals);
    const gpu=dawn.create(["backend="+(process.env.WEBGPU_BACKEND??"metal")]);
    const adapter=await gpu.requestAdapter();
    if(!adapter)throw new Error("Dawn adapter unavailable");
    const device=await adapter.requestDevice();
    const module=device.createShaderModule({code:octreePowerGalerkinShader});
    const pipeline=device.createComputePipeline({
      layout:"auto",compute:{module,entryPoint:"publishSourceResidual"},
    });
    const parameterValues=new Float32Array([2/3,1e-4,2,1e-7]);
    const params=device.createBuffer({
      size:parameterValues.byteLength,
      usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(params,0,parameterValues);
    const run=async(residualSquared,publicationCount=1)=>{
      const words=new Uint32Array(16),floats=new Float32Array(words.buffer);
      words[3]=1452; // exact step-13 compact liquid rows
      floats[4]=residualSquared;
      floats[5]=1.0e-6;
      const control=device.createBuffer({
        size:words.byteLength,
        usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,
      });
      device.queue.writeBuffer(control,0,words);
      const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0,device.createBindGroup({
        layout:pipeline.getBindGroupLayout(0),
        entries:[
          {binding:5,resource:{buffer:control}},
          {binding:6,resource:{buffer:params}},
        ],
      }));
      for(let publication=0;publication<publicationCount;publication++){
        pass.dispatchWorkgroups(1);
      }
      pass.end();
      const readback=device.createBuffer({
        size:words.byteLength,
        usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ,
      });
      encoder.copyBufferToBuffer(control,0,readback,0,words.byteLength);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const resultWords=new Uint32Array(readback.getMappedRange().slice(0));
      const resultFloats=new Float32Array(resultWords.buffer);
      const result={
        flags:resultWords[0],converged:resultWords[1],cycles:resultWords[2],
        residualSquared:resultFloats[4],rhsSquared:resultFloats[5],
        relative:resultFloats[6],
      };
      readback.unmap();readback.destroy();control.destroy();
      return result;
    };
    const accepted=await run(1.666e-13);
    assert.equal(accepted.flags,0);
    assert.equal(accepted.converged,1);
    assert.equal(accepted.cycles,1);
    assert.ok(accepted.relative>4e-4&&accepted.relative<4.2e-4,
      "fixture must remain just outside the relative-only gate: "+accepted.relative);
    assert.ok(Math.sqrt(accepted.residualSquared/1452)<1e-7,
      "step-13 residual RMS must fit the absolute floor");
    const pending=await run(2.0e-11);
    assert.equal(pending.flags,0,
      "an above-tolerance intermediate checkpoint must leave later V-cycles runnable");
    assert.equal(pending.converged,0);
    assert.equal(pending.cycles,1);
    const rejected=await run(2.0e-11,2);
    assert.equal(rejected.flags,16);
    assert.equal(rejected.converged,0);
    assert.equal(rejected.cycles,2,
      "attempted cycles remain observable when convergence is rejected");
    assert.ok(Math.sqrt(rejected.residualSquared/1452)>1e-7);
    params.destroy();device.destroy();
  `;
  const result = spawnSync(process.execPath, ["--import", "tsx", "--input-type=module", "-e", script], {
    cwd: process.cwd(), env: process.env, encoding: "utf8",
  });
  assert.equal(result.status, 0, `signal=${result.signal}\n${result.stdout}\n${result.stderr}`);
});

test("Dawn adaptive Galerkin imports and solves adjacent live size-2 rows", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for numerical GPU validation",
}, async () => {
  const script = `
    import {pathToFileURL} from "node:url";
    import {
      buildFixedAdaptiveOctreePowerGalerkinHierarchy,
      refreshOctreePowerGalerkinOperators,
    } from "./lib/octree-power-galerkin.ts";
    import {WebGPUOctreePowerGalerkin} from "./lib/webgpu-octree-power-galerkin.ts";
    import {PassBroker} from "./lib/webgpu-pass-broker.ts";
    const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE).href);
    Object.assign(globalThis,dawn.globals);
    const hierarchy=buildFixedAdaptiveOctreePowerGalerkinHierarchy(
      [4,4,4],2,{transfer:"trilinear",coarsestNodeLimit:8},
    );
    const finest=hierarchy.levels[0];
    const operators=refreshOctreePowerGalerkinOperators(
      hierarchy,
      new Float64Array(finest.nodeCount).fill(1),
      new Float64Array(hierarchy.fineOffDiagonalEntries.length),
    );
    const gpu=dawn.create(["backend="+(process.env.WEBGPU_BACKEND??"metal")]);
    const adapter=await gpu.requestAdapter();
    if(!adapter)throw new Error("Dawn adapter unavailable");
    const device=await adapter.requestDevice();
    const solver=new WebGPUOctreePowerGalerkin(device,hierarchy,operators,{
      cycles:10,relativeTolerance:1e-4,
    });

    // Two compact adaptive rows occupy adjacent 2^3 leaves at origins
    // (0,0,0) and (2,0,0). Their SPD operator is [[2,-1],[-1,2]].
    const headerWords=new Uint32Array(2*12);
    const headerFloats=new Float32Array(headerWords.buffer);
    for(let row=0;row<2;row++){
      headerWords[row*12]=row===0?0:2;
      headerWords[row*12+1]=row;
      headerWords[row*12+2]=1;
      headerWords[row*12+3]=2;
      headerFloats[row*12+4]=2;
      headerFloats[row*12+5]=row===0?1:-1;
    }
    const entryWords=new Uint32Array(2*2);
    const entryFloats=new Float32Array(entryWords.buffer);
    entryWords[0]=1;entryFloats[1]=1;
    entryWords[2]=0;entryFloats[3]=1;
    const storage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;
    const upload=(words)=>{
      const buffer=device.createBuffer({size:words.byteLength,usage:storage});
      device.queue.writeBuffer(buffer,0,words);
      return buffer;
    };
    const liveHeaders=upload(headerWords);
    const liveEntries=upload(entryWords);
    const authority=device.createBuffer({size:64,usage:storage});
    device.queue.writeBuffer(authority,0,new Uint32Array([0x80000000,0,2,0,0,2]));
    const initialCorrection=device.createBuffer({size:2*4,usage:storage});
    const compactCorrection=device.createBuffer({size:2*4,usage:storage});
    // Destination contents belong to the previous ping-pong epoch and must
    // never be interpreted as a warm start for the current compact row order.
    device.queue.writeBuffer(compactCorrection,0,new Float32Array([NaN,NaN]));
    const readback=device.createBuffer({
      size:64,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ,
    });
    const encoder=device.createCommandEncoder();
    const broker=new PassBroker(encoder);
    solver.encode(broker,{liveOperator:{
      leafHeaders:liveHeaders,leafEntries:liveEntries,authorityControl:authority,
    },initialCorrection,correction:compactCorrection});
    broker.copyBufferToBuffer(solver.control,0,readback,0,32);
    broker.copyBufferToBuffer(compactCorrection,0,readback,32,8);
    device.queue.submit([broker.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const mapped=readback.getMappedRange();
    const control=Uint32Array.from(new Uint32Array(mapped,0,8));
    const correction=Float32Array.from(new Float32Array(mapped,32,2));
    readback.unmap();
    if(control[0]!==0)throw new Error("adaptive solver flags "+control[0]);
    if(control[1]!==1)throw new Error("adaptive solver did not converge: "+Array.from(control));
    if(!correction.every(Number.isFinite)){
      throw new Error("adaptive correction is nonfinite: "+Array.from(correction));
    }
    if(Math.abs(correction[0]+1/3)>2e-6||Math.abs(correction[1]-1/3)>2e-6){
      throw new Error("adaptive correction mismatch: "+Array.from(correction));
    }
    if(Math.abs(correction[0]+correction[1])>2e-6){
      throw new Error("adaptive corrections are not opposite: "+Array.from(correction));
    }
    solver.destroy();liveHeaders.destroy();liveEntries.destroy();authority.destroy();
    initialCorrection.destroy();compactCorrection.destroy();readback.destroy();device.destroy();
  `;
  const result = spawnSync(process.execPath, ["--import", "tsx", "--input-type=module", "-e", script], {
    cwd: process.cwd(), env: process.env, encoding: "utf8",
  });
  assert.equal(result.status, 0, `signal=${result.signal}\n${result.stdout}\n${result.stderr}`);
});

test("Dawn executes the packed native L2 V-cycle and publishes convergence", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for numerical GPU validation",
}, async () => {
  const script = `
    import {pathToFileURL} from "node:url";
    import {buildOctreePowerGalerkinHierarchy,refreshOctreePowerGalerkinOperators} from "./lib/octree-power-galerkin.ts";
    import {WebGPUOctreePowerGalerkin} from "./lib/webgpu-octree-power-galerkin.ts";
    import {PassBroker} from "./lib/webgpu-pass-broker.ts";
    const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE).href);
    Object.assign(globalThis,dawn.globals);
    const size=4,count=size**3,id=(x,y,z)=>x+size*(y+size*z);
    const geometry=Array.from({length:count},(_,row)=>({
      x:row%size,y:Math.floor(row/size)%size,z:Math.floor(row/(size*size)),span:1,
    }));
    const rows=Array.from({length:count},()=>[]);
    for(let z=0;z<size;z++)for(let y=0;y<size;y++)for(let x=0;x<size;x++){
      const row=id(x,y,z);
      for(const [dx,dy,dz] of [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]){
        const nx=x+dx,ny=y+dy,nz=z+dz;
        if(nx>=0&&nx<size&&ny>=0&&ny<size&&nz>=0&&nz<size)rows[row].push(id(nx,ny,nz));
      }
    }
    const rowOffsets=new Uint32Array(count+1);
    rows.forEach((row,index)=>{rowOffsets[index+1]=rowOffsets[index]+row.length;});
    const columns=Uint32Array.from(rows.flat());
    const diagonal=Float64Array.from(rows,(row)=>row.length+0.25);
    const offDiagonal=new Float64Array(columns.length).fill(1);
    const hierarchy=buildOctreePowerGalerkinHierarchy({
      dimensions:[size,size,size],geometry,rowOffsets,columns,coarsestNodeLimit:8,
    });
    // Seed deliberately stale numeric values. The solve must import the live
    // compact rows on device; constructor snapshots are not a fallback.
    const operators=refreshOctreePowerGalerkinOperators(
      hierarchy,
      Float64Array.from(diagonal,(value)=>value+1),
      new Float64Array(offDiagonal.length).fill(0.5),
    );
    // Dawn's Node binding is owned by the object returned from create(). Keep
    // it alive until after device destruction; a temporary here can be
    // collected underneath the adapter/device and cause a native SIGSEGV.
    const gpu=dawn.create(["backend="+(process.env.WEBGPU_BACKEND??"metal")]);
    const adapter=await gpu.requestAdapter();
    if(!adapter)throw new Error("Dawn adapter unavailable");
    const device=await adapter.requestDevice();
    const solver=new WebGPUOctreePowerGalerkin(device,hierarchy,operators,{
      cycles:10,relativeTolerance:0.02,
    });
    const rhs=Float32Array.from({length:count},(_,row)=>Math.sin(row*0.21)+0.2);
    const headerWords=new Uint32Array(count*12),headerFloats=new Float32Array(headerWords.buffer);
    for(let row=0;row<count;row++){
      headerWords[row*12]=row;headerWords[row*12+1]=rowOffsets[row];
      headerWords[row*12+2]=rowOffsets[row+1]-rowOffsets[row];headerWords[row*12+3]=1;
      headerFloats[row*12+4]=diagonal[row];headerFloats[row*12+5]=rhs[row];
    }
    const entryWords=new Uint32Array(columns.length*2),entryFloats=new Float32Array(entryWords.buffer);
    for(let entry=0;entry<columns.length;entry++){
      entryWords[entry*2]=columns[entry];entryFloats[entry*2+1]=offDiagonal[entry];
    }
    const liveHeaders=device.createBuffer({size:headerWords.byteLength,
      usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const liveEntries=device.createBuffer({size:entryWords.byteLength,
      usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const authority=device.createBuffer({size:64,
      usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(liveHeaders,0,headerWords);
    device.queue.writeBuffer(liveEntries,0,entryWords);
    device.queue.writeBuffer(authority,0,new Uint32Array([0x80000000,0,count]));
    const encoder=device.createCommandEncoder(),broker=new PassBroker(encoder);
    const compactCorrection=device.createBuffer({size:count*4,
      usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    solver.encode(broker,{liveOperator:{
      leafHeaders:liveHeaders,leafEntries:liveEntries,authorityControl:authority,
    },correction:compactCorrection});
    const correctionReadback=device.createBuffer({size:count*4,
      usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    const controlReadback=device.createBuffer({size:32,
      usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    broker.copyBufferToBuffer(compactCorrection,0,correctionReadback,0,count*4);
    broker.copyBufferToBuffer(solver.control,0,controlReadback,0,32);
    device.queue.submit([broker.finish()]);
    // mapAsync supplies the required queue completion fence.
    await correctionReadback.mapAsync(GPUMapMode.READ);
    await controlReadback.mapAsync(GPUMapMode.READ);
    const correction=new Float32Array(correctionReadback.getMappedRange().slice(0));
    const control=new Uint32Array(controlReadback.getMappedRange().slice(0));
    correctionReadback.unmap();controlReadback.unmap();
    if(control[0]!==0)throw new Error("solver control "+Array.from(control));
    if(control[1]!==1)throw new Error("solver did not publish convergence");
    if(!(control[2]>=1&&control[2]<solver.plan.cycles)){
      throw new Error("statically encoded tail did not stop after convergence: "+Array.from(control));
    }
    const relative=new Float32Array(Uint32Array.of(control[6]).buffer)[0];
    if(!(relative<0.02))throw new Error("relative residual "+relative);
    if(!correction.every(Number.isFinite)||!correction.some((value)=>Math.abs(value)>1e-5)){
      throw new Error("invalid correction");
    }
    solver.destroy();liveHeaders.destroy();liveEntries.destroy();authority.destroy();
    compactCorrection.destroy();correctionReadback.destroy();controlReadback.destroy();device.destroy();
  `;
  const result = spawnSync(process.execPath, ["--import", "tsx", "--input-type=module", "-e", script], {
    cwd: process.cwd(), env: process.env, encoding: "utf8",
  });
  assert.equal(result.status, 0, `signal=${result.signal}\n${result.stdout}\n${result.stderr}`);
});
