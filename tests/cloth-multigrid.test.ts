import test from 'node:test';
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
import { clothMGAggregates, withClothMultigrid } from '../lib/cloth/multigrid';
import { clothGridData } from '../lib/cloth/grid';
import { prepareClothTopology } from '../lib/cloth/topology';
import { buildClothSolverModel } from '../lib/cloth/oracle';
import { ClothGPUSolver, packClothGPUTables } from '../lib/cloth/gpu-solver';
import type { ClothMaterial } from '../lib/cloth/types';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
const material: ClothMaterial = { id:'cloth', arealDensity_kg_m2:0.2, stretchWarp_N_m:400, stretchWeft_N_m:250, shear_N_m:40, poisson:0.3, compressionRatio:0.05, bending_N_m:2e-5, damping_s:0.002, thickness_m:5e-4, strainLimit:0.02 };
const options = { origin_m:[0,0,0] as const, dt_s:1/60, gravity_m_s2:[0,-9.81,0] as const, integrator:'bdf2' as const,
  newton:{ iterations:12, pcgIterations:200, pcgTolerance:1e-4, residualTolerance:1e-5, contactLinearization:'off' as const } };
function model(n: number) {
  const data = clothGridData({ columns:n, rows:n, width_m:0.5, height_m:0.5, origin_m:[0,1,0], group:(i,j) => j===0 && (i===0 || i===n-1) ? 1 : 0 });
  return buildClothSolverModel(prepareClothTopology(data,[material]),[material],[]);
}
test('multigrid aggregation preserves disconnected pieces and covers every vertex', () => {
  const a = clothMGAggregates([[1],[0,2],[1],[],[5],[4]]);
  assert.deepEqual(a.groups, [[0,1,2],[3],[4,5]]);
  assert.deepEqual(Array.from(a.parent), [0,0,0,1,2,2]);
});
test('multigrid compiles a symmetric hierarchy and keeps control at the end of the work buffer', () => {
  const m = model(16), base = packClothGPUTables(m,options.origin_m,options), tables = withClothMultigrid(base,m), mg = tables.pcg.multigrid!;
  assert.ok(mg.levels.length > 1);
  assert.deepEqual(tables.topo.subarray(0,base.topo.length),base.topo);
  assert.equal(tables.params[55],tables.newtonRecords-16);
  for (const l of mg.levels) {
    const pairs = new Set<string>();
    for (let i=0;i<l.n;i++) for(let k=tables.topo[l.rows+i]!;k<tables.topo[l.rows+i+1]!;k++) pairs.add(`${i},${tables.topo[l.columns+k]}`);
    for(const pair of pairs) assert.ok(pairs.has(pair.split(',').reverse().join(',')),`transpose of ${pair}`);
  }
});
test('GPU multigrid-preconditioned Newton matches the diagonal-preconditioned converged step', { skip:!process.env.WEBGPU_NODE_MODULE }, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href); Object.assign(globalThis,dawn.globals);
  const adapter = await createProcessRetainedDawnGPU(dawn,[`backend=${process.env.FLUID_WEBGPU_BACKEND ?? 'metal'}`]).requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice({ requiredFeatures:adapter.features.has('subgroups')?['subgroups']:[],requiredLimits:requiredFluidDeviceLimits(adapter.limits) });
  const errors:string[]=[];device.addEventListener('uncapturederror',(e:Event)=>{e.preventDefault();errors.push((e as GPUUncapturedErrorEvent).error.message);});
  try {
    const m=model(8), positions=Float64Array.from(m.topology.data.restPositions_m);
    for(let v=0;v<64;v++) positions[3*v+1]+=0.004*Math.sin(Math.PI*(v%8)/7)*Math.sin(Math.PI*Math.floor(v/8)/7);
    const run=async(multigrid:boolean)=>{
      const solver=await ClothGPUSolver.create(device,m,{...options,positions_m:positions,multigrid});
      try {solver.step(1); const x=await solver.readPositions(),control=await solver.readNewtonControl(),status=await solver.readStatus();assert.equal(status.status,0);return {x,control:Array.from(control)};} finally{solver.destroy();}
    };
    const base=await run(false), mg=await run(true);let difference=0;
    for(let i=0;i<base.x.length;i++)difference=Math.max(difference,Math.abs(base.x[i]!-mg.x[i]!));
    console.log({difference,base:base.control,mg:mg.control});assert.ok(difference<2e-5,`position difference ${difference} m`);
    await device.queue.onSubmittedWorkDone();assert.deepEqual(errors,[]);
  } finally {device.destroy();}
});
