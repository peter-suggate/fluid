import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { octreePowerGalerkinPersistent3Shader } from "../lib/webgpu-octree-power-galerkin-persistent3";

test("persistent three-level Galerkin shader is one workgroup with explicit stable bindings", () => {
  assert.match(octreePowerGalerkinPersistent3Shader,
    /@binding\(0\)[\s\S]*@binding\(9\)[\s\S]*@compute @workgroup_size\(64\) fn solvePersistent3/);
  assert.match(octreePowerGalerkinPersistent3Shader,
    /for\(var row=lane;row<count\(level\);row\+=LANES\)/);
  assert.match(octreePowerGalerkinPersistent3Shader,
    /smoothLevel\(0u,lane\)[\s\S]*restrictDefect\(0u,lane\)[\s\S]*solveBottom\(lane\)[\s\S]*measureAndPublish\(lane\)/);
});

test("Dawn persistent three-level Galerkin matches the staged solver", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for numerical GPU validation",
}, () => {
  const script = `
    import assert from "node:assert/strict";
    import {pathToFileURL} from "node:url";
    import {
      buildOctreePowerGalerkinHierarchy,refreshOctreePowerGalerkinOperators,
    } from "./lib/octree-power-galerkin.ts";
    import {WebGPUOctreePowerGalerkin} from "./lib/webgpu-octree-power-galerkin.ts";
    import {PassBroker} from "./lib/webgpu-pass-broker.ts";
    const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE).href);
    Object.assign(globalThis,dawn.globals);
    const gpu=dawn.create(["backend="+(process.env.WEBGPU_BACKEND??"metal")]);
    const adapter=await gpu.requestAdapter();assert.ok(adapter);
    const device=await adapter.requestDevice({
      requiredLimits:{maxStorageBuffersPerShaderStage:10},
    });
    const size=8,count=size**3,id=(x,y,z)=>x+size*(y+size*z);
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
    const diagonal=Float64Array.from(rows,row=>row.length+0.5);
    const coefficients=new Float64Array(columns.length).fill(1);
    const hierarchy=buildOctreePowerGalerkinHierarchy({
      dimensions:[size,size,size],geometry,rowOffsets,columns,coarsestNodeLimit:8,
    });
    assert.deepEqual(hierarchy.levels.map(level=>level.nodeCount),[512,64,8]);
    const operators=refreshOctreePowerGalerkinOperators(
      hierarchy,diagonal,coefficients,
    );
    const persistent=new WebGPUOctreePowerGalerkin(device,hierarchy,operators,{
      cycles:8,relativeTolerance:0.02,persistentThreeLevel:true,
    });
    const staged=new WebGPUOctreePowerGalerkin(device,hierarchy,operators,{
      cycles:8,relativeTolerance:0.02,persistentThreeLevel:false,
    });
    const headerWords=new Uint32Array(count*12);
    const headerFloats=new Float32Array(headerWords.buffer);
    for(let row=0;row<count;row++){
      headerWords[row*12]=row;
      headerWords[row*12+1]=rowOffsets[row];
      headerWords[row*12+2]=rowOffsets[row+1]-rowOffsets[row];
      headerWords[row*12+3]=1;
      headerFloats[row*12+4]=diagonal[row];
      headerFloats[row*12+5]=Math.sin(row*0.071)+0.25*Math.cos(row*0.193);
    }
    const entryWords=new Uint32Array(columns.length*2);
    const entryFloats=new Float32Array(entryWords.buffer);
    for(let entry=0;entry<columns.length;entry++){
      entryWords[entry*2]=columns[entry];entryFloats[entry*2+1]=1;
    }
    const storage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;
    const upload=words=>{
      const buffer=device.createBuffer({size:words.byteLength,usage:storage});
      device.queue.writeBuffer(buffer,0,words);return buffer;
    };
    const headers=upload(headerWords),entries=upload(entryWords);
    const authority=device.createBuffer({size:64,usage:storage});
    device.queue.writeBuffer(authority,0,new Uint32Array([0x80000000,0,count]));
    const persistentCorrection=device.createBuffer({size:count*4,usage:storage});
    const stagedCorrection=device.createBuffer({size:count*4,usage:storage});
    const readback=device.createBuffer({
      size:count*8+64,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ,
    });
    const encoder=device.createCommandEncoder(),broker=new PassBroker(encoder);
    const liveOperator={leafHeaders:headers,leafEntries:entries,authorityControl:authority};
    persistent.encode(broker,{liveOperator,correction:persistentCorrection});
    staged.encode(broker,{liveOperator,correction:stagedCorrection});
    broker.copyBufferToBuffer(persistentCorrection,0,readback,0,count*4);
    broker.copyBufferToBuffer(stagedCorrection,0,readback,count*4,count*4);
    broker.copyBufferToBuffer(persistent.control,0,readback,count*8,32);
    broker.copyBufferToBuffer(staged.control,0,readback,count*8+32,32);
    device.queue.submit([broker.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const bytes=readback.getMappedRange();
    const a=Float32Array.from(new Float32Array(bytes,0,count));
    const b=Float32Array.from(new Float32Array(bytes,count*4,count));
    const ac=Uint32Array.from(new Uint32Array(bytes,count*8,8));
    const bc=Uint32Array.from(new Uint32Array(bytes,count*8+32,8));
    assert.equal(ac[0],0,"persistent flags "+Array.from(ac));
    assert.equal(bc[0],0,"staged flags "+Array.from(bc));
    assert.equal(ac[1],1,"persistent convergence "+Array.from(ac));
    assert.equal(bc[1],1,"staged convergence "+Array.from(bc));
    assert.equal(ac[2],bc[2],"persistent and staged cycle gates diverged");
    const maxError=a.reduce((error,value,index)=>Math.max(error,Math.abs(value-b[index])),0);
    assert.ok(maxError<2e-5,"persistent/staged max correction error "+maxError);
    const relativeA=new Float32Array(Uint32Array.of(ac[6]).buffer)[0];
    const relativeB=new Float32Array(Uint32Array.of(bc[6]).buffer)[0];
    assert.ok(Math.abs(relativeA-relativeB)<2e-5,
      "persistent/staged residual mismatch "+relativeA+" vs "+relativeB);
    readback.unmap();
    persistent.destroy();staged.destroy();headers.destroy();entries.destroy();authority.destroy();
    persistentCorrection.destroy();stagedCorrection.destroy();readback.destroy();device.destroy();
  `;
  const result = spawnSync(process.execPath, ["--import", "tsx", "--input-type=module", "-e", script], {
    cwd: process.cwd(), env: process.env, encoding: "utf8",
  });
  assert.equal(result.status, 0, `signal=${result.signal}\n${result.stdout}\n${result.stderr}`);
});
