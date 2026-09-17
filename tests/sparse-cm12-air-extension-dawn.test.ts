import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { airFixture, mixedAirFixture, runAirFixture } from "./support/air-extension-fixture";

const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("air projection preserves fixed faces and compatible flux on uniform, mixed, and enclosed 3D graphs",{timeout:120_000},async()=>{
  await acquireWebGPUExclusiveLock("dawn-test","air-extension");let device:GPUDevice|undefined;let gpu:GPU|undefined;
  try{
    const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
    gpu=dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`]);const adapter=await gpu!.requestAdapter();assert.ok(adapter);
    device=await adapter.requestDevice();
    for(const widths of [[1,1,1,1],[2,1,1],[1,1,2]])for(const enclosed of [false,true]){
      const fixture=airFixture(widths,enclosed);
      // A cut aperture and a prescribed inlet exercise both masks; neither
      // adds an extra aperture to the divergence's static dual weight.
      fixture.faces.find(f=>f.terms.length===2&&f.axis===0)!.open=.4;
      fixture.faces.find(f=>f.terms.length===2&&f.axis===2)!.prescribed=true;
      const result=await runAirFixture(device,fixture);
      assert.equal(result.receipt.ready,true,JSON.stringify(result.receipt));
      assert.equal(result.receipt.converged,true,JSON.stringify(result.receipt));
      assert.ok(result.receipt.initialMaxDivergence!>.05);
      assert.ok(result.receipt.finalMaxDivergenceError!<2e-4,JSON.stringify(result.receipt));
      for(const [r,face] of fixture.faces.entries()){
        if(face.terms.length===1||face.open<=1e-8||face.prescribed||face.terms.some(t=>fixture.cells[t.cell]!.phi<=0)){
          assert.equal(result.corrected[r],Math.fround(face.velocity),`fixed row ${r}`);
        }
      }
      for(const [c,cell] of fixture.cells.entries()){
        const at=4*(result.layout.cellBase+3*c);if(result.output[at]===0)continue;
        const flux=fixture.faces.reduce((sum,f,r)=>sum+f.terms.filter(t=>t.cell===c)
          .reduce((s,t)=>s+t.coefficient*f.weight*result.corrected[r]!,0),0);
        const compatible=result.output[at+3]!;
        assert.ok(Math.abs(flux/cell.volume-compatible)<2e-4,`cell ${c}: ${flux/cell.volume} vs ${compatible}`);
      }
      if(enclosed)assert.ok(result.receipt.compatibleMaxDivergence!>.01);
    }
    for(const reflected of [false,true]){
      const fixture=mixedAirFixture(reflected);
      assert.ok(fixture.faces.some(f=>f.terms.length===5),"real compiled coarse/fine seam");
      const result=await runAirFixture(device,fixture);
      assert.ok(result.receipt.ready&&result.receipt.converged,JSON.stringify(result.receipt));
      assert.ok(result.receipt.finalMaxDivergenceError!<2e-4,JSON.stringify(result.receipt));
      for(const [r,face] of fixture.faces.entries())if(face.terms.some(t=>fixture.cells[t.cell]!.phi<=0)){
        assert.equal(result.corrected[r],Math.fround(face.velocity));
      }
    }
    // Constant physical velocity survives extension across different apertures;
    // the legacy cell cache is deliberately unrelated to the face field.
    const constant=airFixture();const physical=[.7,-.2,.4];
    constant.cells.forEach(c=>c.velocity=[99,-27,51]);
    constant.faces.forEach((f,r)=>{f.open=r%3===0?.4:1;f.solid=.15;
      f.velocity=f.open*physical[f.axis]!+(1-f.open)*f.solid;});
    const constantResult=await runAirFixture(device,constant,{project:false});
    constant.faces.forEach((f,r)=>assert.ok(Math.abs(constantResult.corrected[r]!-f.velocity)<2e-6,
      `physical velocity extension must not apply aperture twice: row ${r}`));
    const original=mixedAirFixture();const originalResult=await runAirFixture(device,original);
    for(let axis=0;axis<3;axis++){
      const reflected=structuredClone(original);
      reflected.cells.forEach(c=>{c.center[axis]=reflected.dimensions[axis]!-c.center[axis]!;c.velocity[axis]*=-1;});
      reflected.faces.forEach(f=>{f.center[axis]=reflected.dimensions[axis]!-f.center[axis]!;
        if(f.axis===axis){f.velocity*=-1;f.solid*=-1;f.terms.forEach(t=>t.coefficient*=-1);}});
      const result=await runAirFixture(device,reflected);
      reflected.faces.forEach((f,r)=>assert.ok(Math.abs(result.corrected[r]!-(f.axis===axis?-1:1)*originalResult.corrected[r]!)<5e-5,
        `reflected seam axis ${axis}, row ${r}`));
    }
    const open=airFixture([1,1,1,1],true);
    open.faces.forEach(f=>{if(f.axis===1&&f.center[1]===4)f.open=1;});
    const openResult=await runAirFixture(device,open);
    assert.ok(openResult.receipt.ready&&openResult.receipt.converged);
    assert.equal(openResult.receipt.compatibleMaxDivergence,0,"an explicit open exterior anchors the component");
    assert.ok(openResult.receipt.finalMaxDivergenceError!<2e-4);
    const disconnected=airFixture([1,1,1,1],true);
    disconnected.faces.forEach(f=>{
      if(f.terms.length===1){f.velocity=f.axis===0?(f.center[0]===0?.1:.2):0;f.solid=f.velocity;}
      if(f.axis===0&&f.center[0]===2){f.open=0;f.velocity=f.solid=0;}
    });
    const disconnectedResult=await runAirFixture(device,disconnected);
    assert.ok(disconnectedResult.receipt.ready&&disconnectedResult.receipt.converged);
    disconnected.cells.forEach((c,i)=>assert.ok(Math.abs(disconnectedResult.output[4*(disconnectedResult.layout.cellBase+3*i)+3]!-(c.center[0]!<2?.05:-.1))<1e-6,
      "disconnected components retain their own compatible divergence"));
    const isolated=airFixture([1,1,1,1],true);
    isolated.faces.forEach(f=>{f.open=0;f.solid=.17;f.velocity=.17;});
    const isolatedResult=await runAirFixture(device,isolated);
    assert.equal(isolatedResult.receipt.isolatedCells,isolated.cells.length);
    assert.equal(isolatedResult.receipt.iterations,0);
    assert.ok(isolatedResult.receipt.ready);
    assert.ok(isolatedResult.corrected.every(v=>v===Math.fround(.17)));
    // A cross term is not affine: an unconstrained boundary MLS fit can
    // invent nonzero wall-normal velocity even with exact zero wall faces.
    const walls=airFixture();walls.cells.forEach(c=>c.phi=-1);
    walls.faces.forEach(f=>{const q=f.center;f.velocity=f.open===0?0:[q[0]!*q[1]!,q[1]!*q[2]!,q[2]!*q[0]!][f.axis]!;});
    const wallSamples=[[0,1.3,1.7],[1.3,0,1.7],[1.3,1.7,0],[0,0,0]];
    for(const snapshot of [false,true]){
      const result=await runAirFixture(device,walls,{samples:wallSamples,snapshot});
      for(let axis=0;axis<3;axis++)assert.equal(result.samples[3*axis+axis],0,"closed wall normal is authoritative");
      assert.deepEqual(result.samples.slice(9,12),[0,0,0],"three closed normals meet at a corner");
    }
    const affine=(q:number[])=>[.4+.2*q[0]!-.3*q[1]!+.1*q[2]!, -.2+.1*q[0]!+.15*q[1]!-.1*q[2]!, .3-.1*q[0]!+.1*q[1]!-.35*q[2]!];
    for(const [name,fixture] of [["uniform",airFixture()], ["rectilinear",airFixture([2,1,1])],
      ["five-term seam",mixedAirFixture()]] as const){
      fixture.cells.forEach(c=>{c.phi=-1;c.velocity=affine(c.center);});
      fixture.faces.forEach(f=>{f.open=1;f.velocity=affine(f.center)[f.axis]!;});
      const samples=[[1.25,1.3,1.7],[0,0,0],[...fixture.dimensions],[2,1.1,.7],[.1,3.8,1.3],[4,1.1,1.7]];
      const result=await runAirFixture(device,fixture,{samples});
      const immutable=await runAirFixture(device,fixture,{samples,snapshot:true});
      assert.deepEqual(immutable.samples,result.samples,"retiring the face bank and replacing live geometry/incidences must not change momentum samples");
      const shifted=structuredClone(fixture);shifted.origin=[-8,-4,-4];
      shifted.cells.forEach(c=>{c.center=c.center.map((v,a)=>v+shifted.origin![a]!);});
      shifted.faces.forEach(f=>{f.center=f.center.map((v,a)=>v+shifted.origin![a]!);f.velocity=affine(f.center)[f.axis]!;});
      const shiftedSamples=samples.map(q=>q.map((v,a)=>v+shifted.origin![a]!));
      const signed=await runAirFixture(device,shifted,{samples:shiftedSamples,snapshot:true});
      shiftedSamples.forEach((q,i)=>affine(q).forEach((expected,a)=>assert.ok(Math.abs(signed.samples[3*i+a]!-expected)<5e-5,
        `${name} signed sparse-world snapshot: ${q}, axis ${a}: ${signed.samples[3*i+a]} expected ${expected}`)));
      for(let i=0;i<samples.length;i++)for(let a=0;a<3;a++){
        assert.ok(Math.abs(result.samples[3*i+a]!-affine(samples[i]!)[a]!)<5e-5,
          `${name}: ${samples[i]} axis ${a}: ${result.samples[3*i+a]} expected ${affine(samples[i]!)[a]}`);
      }
    }
  }finally{device?.destroy();await releaseWebGPUExclusiveLock();assert.ok(gpu);}
});
