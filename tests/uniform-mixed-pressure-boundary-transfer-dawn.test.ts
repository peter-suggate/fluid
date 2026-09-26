import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { UniformMixedPressureBoundsStage, UniformMixedPressureTransferStage } from "../lib/methods/uniform/uniform-mixed-pressure-stage";
import { uniformMixedPressureStorage } from "../lib/methods/uniform/uniform-mixed-pressure-boundary.wgsl";
import { uniformMixedPressureLevel, type UniformMixedLayout } from "../lib/methods/uniform/uniform-mixed-layout";
import { mixedPressureLayouts } from "./helpers/uniform-mixed-pressure";
import { geometricSeamRows, seamLayout } from "./helpers/uniform-geometric-seam";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";

function geometry(layout:UniformMixedLayout){
 const cells=geometricSeamRows(layout,()=>[0,0,0]).cells;
 const storage=uniformMixedPressureStorage(layout),d=layout.lattice.dimensions.map(v=>v/storage.width);
 const halos:{index:number;owner:number;axis:number;side:number}[]=[];
 cells.forEach((c,owner)=>{for(let axis=0;axis<3;axis++)for(const side of [0,1]){
  if(side===0?c.min[axis]!==0:c.min[axis]!+c.width!==layout.lattice.dimensions[axis])continue;
  const p=c.min.map(v=>v/storage.width);
  const index=cells.length+(axis===0?side*d[1]!*d[2]!+p[1]!+d[1]!*p[2]!:axis===1?2*d[1]!*d[2]!+side*d[0]!*d[2]!+p[0]!+d[0]!*p[2]!:2*(d[1]!*d[2]!+d[0]!*d[2]!)+side*d[0]!*d[1]!+p[0]!+d[0]!*p[1]!);
  halos.push({index,owner,axis,side});
 }});
 return {cells,halos,storage};
}
const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("mixed pressure halo transfers preserve native restriction, bounds and renormalized prolongation",{timeout:180000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform mixed pressure halo transfers");let device:GPUDevice|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice();
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  for(const root of [seamLayout(0,"fine"),seamLayout(0,"coarse"),...mixedPressureLayouts()])for(const floor of [1,2]){
   const fine=floor===1?root:uniformMixedPressureLevel(root,2),coarse=uniformMixedPressureLevel(root,(floor*2) as 2|4);
   const f=geometry(fine),c=geometry(coarse),fo=new UniformMixedOwnership(device,fine),co=new UniformMixedOwnership(device,coarse),buffers:GPUBuffer[]=[];
   const buffer=(n:number)=>{const b=device!.createBuffer({size:n*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});buffers.push(b);return b;};
   try{
    const source=buffer(f.storage.count),pressure=buffer(f.storage.count),down=buffer(c.storage.count),up=buffer(f.storage.count);
    const values=Float32Array.from({length:f.storage.count},(_,i)=>Math.sin(i*.71)+2),p=Float32Array.from(values,(_,i)=>Math.cos(i*.31));
    device.queue.writeBuffer(source,0,values);device.queue.writeBuffer(pressure,0,p);
    const transfer=new UniformMixedPressureTransferStage(device,fo,co,false,true),bounds=new UniformMixedPressureBoundsStage(device,fo,co,true);
    await transfer.initialize();await bounds.initialize();
    const inside=(point:readonly number[],box:typeof f.cells[number])=>point.every((v,a)=>v>=box.min[a]!&&v<box.min[a]!+box.width);
    const children=c.halos.map(h=>f.halos.filter(child=>child.axis===h.axis&&child.side===h.side&&inside(f.cells[child.owner]!.min,c.cells[h.owner]!)));
    for(const entry of ["restrictValues","downsampleMinimum","downsampleSubtract"] as const){
     const encoder=device.createCommandEncoder();
     if(entry==="restrictValues")transfer.encode(encoder,entry,transfer.bind(entry,{buffer:source},{buffer:down}));
     else bounds.encode(encoder,entry,bounds.bind(entry,{buffer:source},{buffer:pressure},{buffer:down}));
     device.queue.submit([encoder.finish()]);const actual=await readMixedBuffer(device,down);
     c.halos.forEach((h,i)=>{
      assert.ok(children[i]!.length===1||children[i]!.length===4);
      const terms=children[i]!.map(child=>values[child.index]!-(entry==="downsampleSubtract"?p[child.index]!:0));
      const expected=entry==="restrictValues"?terms.reduce((s,v)=>s+v,0)/terms.length:Math.max(...terms);
      assert.ok(Math.abs(actual[h.index]!-expected)<1e-6,`${entry} halo ${h.index}`);
     });
    }
    {
     const encoder=device.createCommandEncoder();bounds.encode(encoder,"shiftMinimum",bounds.bind("shiftMinimum",{buffer:source},{buffer:pressure},{buffer:up}));device.queue.submit([encoder.finish()]);
     const actual=await readMixedBuffer(device,up);for(const h of f.halos)assert.ok(Math.abs(actual[h.index]!-(values[h.index]!-p[h.index]!))<1e-6);
    }
    const coarseValues=Float32Array.from({length:c.storage.count},(_,i)=>Math.sin(i*.27));device.queue.writeBuffer(down,0,coarseValues);
    for(const entry of ["prolongAssign","prolongAdd"] as const){
     device.queue.writeBuffer(up,0,new Float32Array(f.storage.count).fill(2));
     const encoder=device.createCommandEncoder();transfer.encode(encoder,entry,transfer.bind(entry,{buffer:down},{buffer:up}));device.queue.submit([encoder.finish()]);
     const actual=await readMixedBuffer(device,up);
     for(const h of f.halos){
      const cell=f.cells[h.owner]!,parent=c.cells.findIndex(box=>inside(cell.min,box)),box=c.cells[parent]!;
      let expected=0;
      if(cell.width===box.width){expected=coarseValues[c.halos.find(v=>v.owner===parent&&v.axis===h.axis&&v.side===h.side)!.index]!;}
      else{
       const center=cell.min.map(v=>v+cell.width/2);center[h.axis]=h.side?fine.lattice.dimensions[h.axis]!+cell.width/2:-cell.width/2;
       const q=center.map(v=>v/box.width-.5),base=q.map(Math.floor),fraction=q.map((v,a)=>v-base[a]!);let total=0;
       for(let k=0;k<8;k++){
        const bits=[k&1,(k>>1)&1,k>>2],point=bits.map((v,a)=>(base[a]!+v+.5)*box.width),donor=c.cells.findIndex(b=>inside(point,b));if(donor<0)continue;
        const weight=bits.reduce((w,v,a)=>w*(v?fraction[a]!:1-fraction[a]!),1);expected+=weight*coarseValues[donor]!;total+=weight;
       }
       expected/=total;
      }
      if(entry==="prolongAdd")expected+=2;
      assert.ok(Math.abs(actual[h.index]!-expected)<1e-6,`${entry} halo ${h.index}: ${actual[h.index]} != ${expected}`);
     }
    }
   }finally{buffers.forEach(b=>b.destroy());fo.destroy();co.destroy();}
  }
  assert.deepEqual(errors,[]);
 }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
