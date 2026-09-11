/** Hydrostatic patch test for a 2:1 face cut tangentially by the free surface.
 * This tests the production pressure gradient, independently of iteration
 * tolerance, time integration, density transport, and topology transfers.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { createCm12NumericsWGSL } from '../lib/core/cm12-numerics';
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from '../lib/harness/webgpu-smoke-isolation';
const source=readFileSync(new URL('../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts',import.meta.url),'utf8');
const gradient=source.match(/fn pressureRowGradient\([\s\S]*?\n}/)?.[0];assert.ok(gradient);
const factor=source.match(/fn mixedSurfacePressureFactor\([\s\S]*?\n}/)?.[0];assert.ok(factor);
const classify=source.match(/fn classifyPressureRow\([\s\S]*?\n}/)?.[0];assert.ok(classify);
const live=new Set<GPU>();Object.assign(globalThis,{surfaceSeamAffineGPUs:live});
(process.env.WEBGPU_NODE_MODULE?test:test.skip)('mixed free-surface pressure factors reproduce affine pressure gradients',async()=>{
 await acquireWebGPUExclusiveLock('dawn-test','surface-seam-affine');let device:GPUDevice|undefined;
 try{
  const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);
  const gpu:GPU=dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND??'metal'}`]);live.add(gpu);
  const adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice();
  // Coarse width 2, fine width 1; centre separation 3/2, and four
  // equal-area children. Coarse y=0, fine y=+/-1/2. Gravity is tangential
  // to the face. p=H-y inside liquid; g*d=1 only fixes the unit scale.
  const cases=[{nx:0,height:2},{nx:0,height:.25},{nx:.2,height:.25},{nx:.3,height:.5},{nx:-.3,height:0}];
  const centers=[[0,0],[1.5,-.5],[1.5,-.5],[1.5,.5],[1.5,.5]];
  const phi=cases.flatMap(c=>centers.map(([x,y])=>c.nx*x!+y!-c.height));
  const pressure=phi.map(v=>Math.max(0,-v));
  const shader=device.createShaderModule({code:`
@group(0)@binding(0)var<storage,read_write>state:array<f32>;
@group(0)@binding(1)var<storage,read_write>output:array<f32>;
@group(0)@binding(2)var<storage,read_write>activity:array<atomic<u32>>;
struct Params {stateOffsets3:vec4u, acceleration:vec4f}
const p=Params(vec4u(${2*phi.length}u,0u,0u,0u),vec4f(0.0));
const TEMPLATE_CELL_RESOLUTION_MASK=31u;const BRICK_FINE_RESOLUTION=8u;
const ACCEPTED_MIXED_ROW_COUNT=0u;const ACCEPTED_COARSE_ROW_COUNT=1u;const PRESSURE_ACTIVE_ROW_COUNT=2u;
fn rowAccepted(row:u32)->bool{_=row;return true;}
fn ta(at:u32)->u32{_=at;return 0u;}
fn rowRequirementOffset(row:u32)->u32{_=row;return 0u;}
fn rowKind(row:u32)->u32{_=row;return 2u;}
fn rowDualWeight(row:u32)->f32{_=row;return 1.0;}
fn rowAxis(row:u32)->u32{_=row;return 0u;}
fn cellWidths(cell:u32)->vec3f{return vec3f(select(1.0,2.0,cell%5u==0u));}
fn cellCenter(cell:u32)->vec3f{return vec3f(0.0,select(-0.5,0.5,cell%5u>=3u),0.0);}
fn pressureDensity(cell:u32)->f32{return 0.5-state[${phi.length}u+cell]/cellWidths(cell).x;}
fn pcmCellContains(cell:u32)->bool{return peiPressureCellMember(cell);}
fn rowExteriorPhi(row:u32)->f32{_=row;return 0.5;}
fn rowCenter(row:u32)->vec3f{_=row;return vec3f(0.0);}
fn rowDistance(row:u32)->f32{_=row;return 1.5;}
fn pressureHasPartialRefinementRegion()->bool{return false;}
fn pressurePlanarColumnHeight(row:u32)->vec2f{_=row;return vec2f(0.0);}
fn rowTermOffset(row:u32)->u32{return row*5u;}
fn rowTermCount(row:u32)->u32{_=row;return 5u;}
fn rowTermRange(row:u32)->vec2u{let first=rowTermOffset(row);return vec2u(first,first+rowTermCount(row));}
fn termCell(term:u32)->u32{return term;}
fn termCoefficient(term:u32)->f32{return select(1.0/6.0,-2.0/3.0,term%5u==0u);}
fn peiPressureCellMember(cell:u32)->bool{return state[cell]>0.0;}
${gradient}
${createCm12NumericsWGSL()}
${factor}
${classify}
@compute @workgroup_size(1)
fn main(){
  for(var row=0u;row<${cases.length}u;row++){
    let enabled=classifyPressureRow(row);let theta=state[p.stateOffsets3.x+row];
    var corrected=0.0;if(enabled&&theta>0.0){corrected=pressureRowGradient(row,0u)/theta;}
    output[row]=corrected;
  }
}
`});
  assert.deepEqual((await shader.getCompilationInfo()).messages.filter(m=>m.type==='error'),[]);
  const input=device.createBuffer({size:8*phi.length+4*cases.length,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(input,0,new Float32Array([...pressure,...phi]));
  const output=device.createBuffer({size:4*cases.length,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const read=device.createBuffer({size:4*cases.length,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const counts=device.createBuffer({size:12,usage:GPUBufferUsage.STORAGE});
  const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module:shader,entryPoint:'main'}});
  const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[input,output,counts].map((buffer,binding)=>({binding,resource:{buffer}}))});
  const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();
  encoder.copyBufferToBuffer(output,0,read,0,read.size);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
  const result=[...new Float32Array(read.getMappedRange())];read.unmap();read.destroy();input.destroy();output.destroy();counts.destroy();
  for(const [i,c] of cases.entries())assert.ok(Math.abs(result[i]!+c.nx)<1e-6,`case ${i}: expected gradient ${-c.nx}, got ${result[i]}`);
 }finally{device?.destroy();live.clear();await releaseWebGPUExclusiveLock();}
});
