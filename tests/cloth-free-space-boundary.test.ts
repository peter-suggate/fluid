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
import { createUniformSolver, readUniformFields, withUniformDevice } from './helpers/uniform-geometric';
import { bakeProceduralSolid, proceduralSolidField } from '../lib/core/procedural-solid';
import { fluidLatticeVertexPhi } from '../lib/core/initial-liquid-lattice';

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

// Free space places the box at a window page (sceneFluidBoxPage), and the GPU
// sampler keys procedural bricks by world cell: the bricks are uploaded at that
// page. Keyed at the box's own cells the sampler found no brick, every cut
// cell fell back to the half-solid voxel mask, and a wet-cloth source's seed
// lost 3% of its liquid (dam_break_nonlinear_drag_small: 18739 cells of 19255).
test('free space GPU seed reads its procedural walls at the box page: a cavity seeds its open volume', {skip: !process.env.WEBGPU_NODE_MODULE}, async () => {
  const s = cloneScene(defaultScene), n = 48, h = .01, r = 6;
  s.container = {width_m: n*h, height_m: n*h, depth_m: n*h, top: 'open', fillFraction: 0, fluidWallMode: 'free-slip', boundary: 'free-space', vessel: 'none'};
  s.voxelDomain = {finestCellSize_m: h, brickSize_cells: 8}; s.rigidBodies = []; s.scenery = undefined;
  Object.assign(s, bakeProceduralSolid([n, n, n], [h, h, h], ([x, y, z]) => r-Math.hypot(x-24, y-24, z-24)));
  s.fluid = {...s.fluid, initialCondition: 'tank-fill', initialLiquidVolumes: [{shape: 'sphere', center_m: {x: 0, y: 24*h, z: 0}, radius_m: r*h}]};
  assert.ok(sceneFluidBoxPage(s).some(v => v > 0));
  const field = proceduralSolidField(s)!, phi = fluidLatticeVertexPhi(s), K = 8;
  await withUniformDevice('free-space procedural seed', async device => {
    const solver = await createUniformSolver(device, s);
    try {
      // The lattice is whole pages; the box (and its field) its first n cells an axis.
      const f = await readUniformFields(device, solver), {nx, ny} = solver.info, X = nx+1;
      assert.equal(phi.length, (nx+1)*(ny+1)*(solver.info.nz+1));
      let gpu = 0, expected = 0, worst = 0;
      for (const v of f.density) gpu += v;
      for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
        const density = f.density[x+nx*(y+ny*z)]!, open = field.open([x, y, z]);
        if (open === undefined) continue;
        let wet = 0;
        for (let i = 0; i < K**3; i++) {
          const q = [(i%K+.5)/K, (Math.floor(i/K)%K+.5)/K, (Math.floor(i/K/K)+.5)/K]; let v = 0;
          for (let m = 0; m < 8; m++) { const e = [m&1, (m>>1)&1, m>>2]; v += phi[x+e[0]!+X*(y+e[1]!+(ny+1)*(z+e[2]!))]!*e.reduce((w, t, a) => w*(t ? q[a]! : 1-q[a]!), 1); }
          if (v < 0) wet++;
        }
        const cell = open*wet/K**3; expected += cell; worst = Math.max(worst, Math.abs(density-cell));
      }
      const exact = 4/3*Math.PI*r**3;
      assert.ok(worst < .05, `cut-cell seed density differs from open x fill by ${worst}`);
      assert.ok(Math.abs(gpu-expected) < .005*expected, `GPU seed ${gpu} cells, open x fill ${expected}`);
      assert.ok(Math.abs(gpu-exact) < .03*exact, `GPU seed ${gpu} cells of the cavity's ${exact}`);
    } finally { solver.destroy(); }
  });
});
