import { SparseCM12GenerationBudgetDeferred } from "../lib/methods/adaptive-mass/sparse-cm12-generation-budget";
import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createSparseAdaptiveMassAtlas } from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { buildSparseAtlasCompositeGrid } from "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import { transferSparseCM12GenerationFields, prepareSparseCM12GenerationTransfer,
 PreparedSparseCM12GenerationTransfer } from "../lib/methods/adaptive-mass/sparse-cm12-generation-transfer";
import { createCM12ResourceRecorder, realizeCM12ResourceRecipe } from "../lib/methods/adaptive-mass/sparse-cm12-resource-recipe";
const dawnModule = process.env.WEBGPU_NODE_MODULE;
(dawnModule ? test : test.skip)("GPU generation transfer conserves clipped liquid mass, gamma, momentum and boundary flux", async () => {
 await acquireWebGPUExclusiveLock("dawn-test", "sparse-cm12-generation-transfer-dawn");
 let device: GPUDevice | undefined;
 const buffers: GPUBuffer[] = [];
 try {
  const dawn = await import(pathToFileURL(dawnModule!).href);
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  device = await (await gpu.requestAdapter())!.requestDevice();
  assert.ok(device);
  device.pushErrorScope("validation");
  const grid = (r: 1 | 2 | 4 | 8) => buildSparseAtlasCompositeGrid(createSparseAdaptiveMassAtlas([13, 15, 11], [{
   key: 0, coordinate: [0, 0, 0], spanBricks: 2, resolution: r,
   density: new Float64Array(r ** 3).fill(0.5), gamma: new Float64Array(r ** 3).fill(1),
  }], 0, 8));
  const fields = (g: ReturnType<typeof grid>) => {
   const n = g.cells.length, k = g.gradientRows.length;
   const state = device!.createBuffer({ size: 4 * (13 * n + 2 * k),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
   buffers.push(state);
   return { state, densityOffset: 0, densityOtherOffset: n, gammaOffset: 2*n, gammaOtherOffset: 3*n,
    velocityOffset: 4*n, velocityOtherOffset: 8*n, pressureOffset: 12*n,
    faceOffset: 13*n, faceOtherOffset: 13*n+k,
    cellIds: Uint32Array.from(g.cells, c => c.id), rowIds: Uint32Array.from(g.gradientRows, r => r.id) };
  };
  const fine = grid(8), coarse = grid(2);
  const a = fields(fine), b = fields(coarse), c = fields(fine);
  const data = new Float32Array(a.state.size / 4);
  for (const cell of fine.cells) {
   data[a.densityOffset + cell.id] = 0.125 + (cell.id % 4) * 0.125;
   data[a.gammaOffset + cell.id] = 1 + (cell.id % 4) * 0.125;
   for (let axis = 0; axis < 3; axis++) data[a.velocityOffset + 4*cell.id+axis] = axis + (cell.id % 3) * 0.25;
  }
  for (const row of fine.gradientRows) data[a.faceOffset + row.id] = row.axis + 0.25;
  device.queue.writeBuffer(a.state, 0, data);
  await assert.rejects(transferSparseCM12GenerationFields(device, fine, coarse, a, b, 0),
   SparseCM12GenerationBudgetDeferred);
  await transferSparseCM12GenerationFields(device, fine, coarse, a, b);
  await transferSparseCM12GenerationFields(device, coarse, fine, b, c);
  const readback = device.createBuffer({ size: c.state.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  buffers.push(readback);
  const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(c.state, 0, readback, 0, c.state.size);
  device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
  const after = new Float32Array(readback.getMappedRange());
  const totals = (values: Float32Array, field: typeof a) => {
   const sums = [0, 0, 0, 0, 0];
   for (const cell of fine.cells) {
    const mass = values[field.densityOffset + cell.id]! * cell.volume;
    sums[0]! += mass; sums[1]! += values[field.gammaOffset + cell.id]! * cell.volume;
    for (let axis=0; axis<3; axis++) sums[axis+2]! += mass * values[field.velocityOffset + 4*cell.id+axis]!;
   }
   return sums;
  };
  const beforeTotals = totals(data, a), afterTotals = totals(after, c);
  for (let i=0; i<5; i++) assert.ok(Math.abs(beforeTotals[i]! - afterTotals[i]!) < 0.001,
   `conserved channel ${i}: ${beforeTotals[i]} -> ${afterTotals[i]}`);
  for (const row of fine.gradientRows) if (row.kind === "sparse-air") {
   assert.ok(Math.abs(after[c.faceOffset + row.id]! - data[a.faceOffset + row.id]!) < 1e-6);
  }
  assert.equal(await device.popErrorScope(), null);
 } finally { for (const buffer of buffers) buffer.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock(); }
});

for (const mode of ["direct", "recorded"] as const) (dawnModule ? test : test.skip)(`prepared ${mode} transfer reads live scalar and face banks at publication`, async () => {
 await acquireWebGPUExclusiveLock("dawn-test", "sparse-cm12-generation-transfer-live-bank-dawn");
 let device: GPUDevice | undefined, prepared: PreparedSparseCM12GenerationTransfer | undefined;
 const buffers: GPUBuffer[]=[];
 try {
  const dawn=await import(pathToFileURL(dawnModule!).href);
  Object.assign(globalThis,dawn.globals);
  const gpu=dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  device=await (await gpu.requestAdapter())!.requestDevice();
  assert.ok(device); device.pushErrorScope("validation");
  const grid=buildSparseAtlasCompositeGrid(createSparseAdaptiveMassAtlas([8,8,8],[{
   key:0,coordinate:[0,0,0],resolution:2,
   density:new Float64Array(8).fill(0.5),gamma:new Float64Array(8).fill(1),
  }],0,8));
  const n=grid.cells.length, k=grid.gradientRows.length;
  const buffer=(size:number,usage:number)=>{
   const value=device!.createBuffer({size,usage}); buffers.push(value); return value;
  };
  const fields=()=>({
   state:buffer(4*(13*n+2*k),GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST),
   densityOffset:0,densityOtherOffset:n,gammaOffset:2*n,gammaOtherOffset:3*n,
   velocityOffset:4*n,velocityOtherOffset:8*n,pressureOffset:12*n,
   faceOffset:13*n,faceOtherOffset:13*n+k,
   cellIds:Uint32Array.from(grid.cells,cell=>cell.id),rowIds:Uint32Array.from(grid.gradientRows,row=>row.id),
  });
  const source=fields(),target=fields();
  const control=buffer(8,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
  const readback=buffer(target.state.size,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
  const values=new Float32Array(source.state.size/4).fill(0.125);
  device.queue.writeBuffer(source.state,0,values);
  device.queue.writeBuffer(control,0,new Uint32Array([0,0]));
  const external = [source.state, control, target.state];
  const recorder = mode === "recorded" ? createCM12ResourceRecorder(device.limits,
   external.map(buffer => ({size:buffer.size,usage:buffer.usage}))) : undefined;
  const preparedDevice = recorder?.device ?? device;
  const recipeBuffers = recorder?.externalResources as GPUBuffer[] | undefined;
  prepared=await prepareSparseCM12GenerationTransfer(preparedDevice,grid,grid,{
   ...source,state:recipeBuffers?.[0] ?? source.state,liveControl:{buffer:recipeBuffers?.[1] ?? control,scalarParityWord:0,faceParityWord:1,
    densityOffsets:[source.densityOffset,source.densityOtherOffset],
    gammaOffsets:[source.gammaOffset,source.gammaOtherOffset],
    velocityOffsets:[source.velocityOffset,source.velocityOtherOffset],
    faceOffsets:[source.faceOffset,source.faceOtherOffset]},
  },{...target,state:recipeBuffers?.[2] ?? target.state});
  if (recorder) {
   const realized = await realizeCM12ResourceRecipe(device, structuredClone(recorder.finish(prepared)), external);
   prepared = Object.assign(Object.create(PreparedSparseCM12GenerationTransfer.prototype), realized.state, {device});
  }
  assert.ok(prepared);
  // The source advances after resource/pipeline preparation. Scalar and face
  // parity deliberately differ, exercising their independent GPU authorities.
  for(const scalarParity of [1,0] as const) {
   const faceParity=1-scalarParity;
   const density=scalarParity?source.densityOtherOffset:source.densityOffset;
   const gamma=scalarParity?source.gammaOtherOffset:source.gammaOffset;
   const velocity=scalarParity?source.velocityOtherOffset:source.velocityOffset;
   const face=faceParity?source.faceOtherOffset:source.faceOffset;
   for(const cell of grid.cells) {
    values[density+cell.id]=0.375+0.25*scalarParity+0.01*cell.id;
    values[gamma+cell.id]=1.25+0.25*scalarParity+0.01*cell.id;
    values[source.pressureOffset+cell.id]=4+scalarParity+cell.id;
    for(let axis=0;axis<3;axis++)
     values[velocity+4*cell.id+axis]=2*scalarParity-axis+0.05*cell.id;
   }
   for(const row of grid.gradientRows) values[face+row.id]=-0.125+faceParity+0.02*row.id;
   device.queue.writeBuffer(source.state,0,values);
   device.queue.writeBuffer(control,0,new Uint32Array([scalarParity,faceParity]));
   const encoder=device.createCommandEncoder();
   prepared.encode(encoder);
   encoder.copyBufferToBuffer(target.state,0,readback,0,target.state.size);
   device.queue.submit([encoder.finish()]); await prepared.validate();
   await readback.mapAsync(GPUMapMode.READ);
   try {
    const actual=new Float32Array(readback.getMappedRange());
    const close=(at:number,expected:number)=>assert.ok(Math.abs(actual[at]!-expected)<1e-5,
     `live parity ${scalarParity}/${faceParity}, target ${at}: ${actual[at]} != ${expected}`);
    for(const cell of grid.cells) {
     for(const offset of [target.densityOffset,target.densityOtherOffset]) close(offset+cell.id,values[density+cell.id]!);
     for(const offset of [target.gammaOffset,target.gammaOtherOffset]) close(offset+cell.id,values[gamma+cell.id]!);
     for(const offset of [target.velocityOffset,target.velocityOtherOffset])
      for(let axis=0;axis<3;axis++) close(offset+4*cell.id+axis,values[velocity+4*cell.id+axis]!);
     close(target.pressureOffset+cell.id,values[source.pressureOffset+cell.id]!);
    }
    for(const row of grid.gradientRows)
     for(const offset of [target.faceOffset,target.faceOtherOffset]) close(offset+row.id,values[face+row.id]!);
   } finally {readback.unmap();}
  }
  assert.equal(await device.popErrorScope(),null);
 } finally {
  prepared?.destroy(); for(const value of buffers)value.destroy(); device?.destroy();
  await releaseWebGPUExclusiveLock();
 }
});
