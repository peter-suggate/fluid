import assert from 'node:assert/strict';
import test from 'node:test';
import { cloneScene, defaultScene, validateScene } from '../lib/core/model';
import { sceneFluidBoxPage, sceneFluidFrame } from '../lib/core/scene-lattice-dimensions';
import { solidVoxelShellForScene } from '../lib/core/scene-lattice';
import { sampleSolidWorld, solidWorldForFluidLattice } from '../lib/core/solid-world';
import { advanceRigidBodies, initializeRigidBody } from '../lib/core/rigid-body';
import { uniformPagePoolLayout } from '../lib/methods/uniform/uniform-page-pool';
import { uniformLatticeAddress } from '../lib/methods/uniform/uniform-lattice-address';
import { uniformWindowDerivedWords, uniformWindowWorldWGSL } from '../lib/methods/uniform/uniform-window-address';
import { withUniformDevice } from './helpers/uniform-geometric';

function scene() {
  const s = cloneScene(defaultScene);
  s.container = {width_m: .32, height_m: .32, depth_m: .32, top: 'open', fillFraction: 0, fluidWallMode: 'free-slip', boundary: 'free-space', vessel: 'none'};
  s.voxelDomain = {finestCellSize_m: .01, brickSize_cells: 8}; s.solidVoxels = []; s.rigidBodies = [];
  s.fluid = {...s.fluid, initialCondition: 'tank-fill', initialLiquidVolumes: [{shape: 'sphere', center_m: {x: 0, y: .1, z: 0}, radius_m: .02}]};
  return s;
}

test('free space has no generated floor/walls or padding, preserves cell spacing, and centers travel allowance', () => {
  const s = scene(); assert.deepEqual(validateScene(s), []);
  assert.deepEqual(solidVoxelShellForScene(s), []);
  const world = solidWorldForFluidLattice(s);
  for (const cell of [[-1, -1, -1], [16, -1, 16], [33, 16, 16], [16, 33, 16]] as const) assert.equal(sampleSolidWorld(world, cell).solidFraction, 0);
  const page = sceneFluidBoxPage(s); assert.ok(page.every(v => v > 0));
  assert.deepEqual(sceneFluidFrame(s).cellSize_m, [.01, .01, .01]);
  const address = uniformLatticeAddress([32, 32, 32], true, {blocks: 4}, {boxPage: page, freeSpace: true});
  assert.equal(uniformPagePoolLayout(address.pool).worldPolicy, 4); assert.equal(address.freeSpace, true);
  const bounds = new Int32Array(uniformWindowDerivedWords(4, page, [1, 1, 1]).buffer);
  assert.deepEqual([...bounds.slice(0, 6)], [-2147483648, -2147483648, -2147483648, 2147483647, 2147483647, 2147483647]);
  assert.ok(validateScene({...s, container: {...s.container, boundary: 'unknown' as never}}).some(e => e.includes('Unsupported container boundary')));
  assert.ok(validateScene({...s, container: {...s.container, walls: 'none'}}).length);
});

test('a rigid body may cross the allocation floor in free space while an ordinary tank still collides', () => {
  const s = scene(); s.fluid.gravity_m_s2 = {x: 0, y: 0, z: 0};
  const description = {...defaultScene.rigidBodies[0]!, shape: 'sphere' as const, dimensions_m: {x: .02, y: .02, z: .02},
    position_m: {x: 0, y: .01, z: 0}, linearVelocity_m_s: {x: 0, y: -1, z: 0}, angularVelocity_rad_s: {x: 0, y: 0, z: 0}};
  const free = initializeRigidBody(description); advanceRigidBodies([free], s, .1);
  assert.ok(free.position_m.y < -.08); assert.equal(free.collisionImpulse_N_s.y, 0);
  const tank = initializeRigidBody(description); advanceRigidBodies([tank], {...s, container: {...s.container, boundary: undefined}}, .1);
  assert.ok(tank.position_m.y >= 0);
});

test('GPU window policy 4 is air on every side, unlike grounded and tank policies', {skip: !process.env.WEBGPU_NODE_MODULE}, async () => {
  await withUniformDevice('free-space window classification', async device => {
    const header = new Uint32Array(4 * 32);
    for (let i = 0; i < 4; i++) { header[32 * i + 12] = i + 1; header.set([3, 4, 5], 32 * i + 13); header.set([2, 2, 2], 32 * i + 16); }
    const input = device.createBuffer({size: header.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
    const output = device.createBuffer({size: 4 * 6 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
    const read = device.createBuffer({size: output.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(input, 0, header);
    try {
      const module = device.createShaderModule({code: `
@group(0) @binding(0) var<storage,read> data:array<u32>;
@group(0) @binding(1) var<storage,read_write> result:array<u32>;
var<private> record:u32;
${uniformWindowWorldWGSL(i => `data[32u*record+${i}]`, 'probe')}
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){
 record=id.x/6u; let q=array<vec3i,6>(vec3i(3,4,5),vec3i(3,3,5),vec3i(2,4,5),vec3i(5,4,5),vec3i(3,6,5),vec3i(3,4,7))[id.x%6u];
 result[id.x]=select(0u,1u,probeWorldSolid(q));
}`});
      assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === 'error'), []);
      const pipeline = await device.createComputePipelineAsync({layout: 'auto', compute: {module, entryPoint: 'main'}});
      const group = device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: [{binding: 0, resource: {buffer: input}}, {binding: 1, resource: {buffer: output}}]});
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(24); pass.end();
      encoder.copyBufferToBuffer(output, 0, read, 0, output.size); device.queue.submit([encoder.finish()]); await read.mapAsync(GPUMapMode.READ);
      assert.deepEqual([...new Uint32Array(read.getMappedRange())], [0,1,1,1,1,1, 0,1,1,1,0,1, 0,1,0,0,0,0, 0,0,0,0,0,0]);
      read.unmap();
    } finally {input.destroy(); output.destroy(); read.destroy();}
  });
});
