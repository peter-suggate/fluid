import test from 'node:test';
import assert from 'node:assert/strict';
import { withUniformDevice } from './helpers/uniform-geometric';
import { DEFAULT_CLOTH_MATERIAL } from '../lib/cloth/material';
import { createClothState, prepareClothTopology } from '../lib/cloth/topology';
import type { ClothMaterial, ClothTopologyData } from '../lib/cloth/types';
import { prepareUniformClothStencil, type UniformClothStep } from '../lib/methods/uniform/uniform-cloth-grid-coupling';
import { UniformGPUOrderingKernels } from '../lib/methods/uniform/uniform-gpu-radix-sort';
import { CLOTH_DRAG_COLORS, CLOTH_STENCIL_STATUS, UniformClothGPUStencil, clothColorPriorities, clothStencilTopology } from '../lib/methods/uniform/uniform-cloth-gpu-stencil';
import { clothDragColors } from '../lib/methods/uniform/uniform-cloth-tensor-drag';
import { UniformClothGPUCells } from '../lib/methods/uniform/uniform-cloth-gpu-cells';
import { prepareGridDragMaterial } from '../lib/methods/uniform/uniform-cloth-drag-material';
import { preparePorePressureCells } from '../lib/methods/uniform/uniform-cloth-pore-pressure';

async function read(device: GPUDevice, buffer: GPUBuffer, bytes: number, offset = 0): Promise<ArrayBuffer> {
  const out = device.createBuffer({ size: Math.max(16, bytes), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const e = device.createCommandEncoder(); if (bytes) e.copyBufferToBuffer(buffer, offset, out, 0, bytes); device.queue.submit([e.finish()]);
  await out.mapAsync(GPUMapMode.READ); const data = out.getMappedRange().slice(0, bytes); out.destroy(); return data;
}
function lcg(seed: number) { let s = seed >>> 0; return () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32; }

// An oblique two-material sheet beside free rods: triangles, rod-only
// vertices, fixed vertices, wet and dry vertices, oblique fibers. Positions
// are jittered off cell planes: there f32 rounds q to the plane, dropping the
// ~1e-7 weight faces f64 keeps (physically nil, but not layout parity).
function fixture() {
  const random = lcg(5), n = 36, positions: number[] = [], triangles: number[] = [], triangleMaterials: number[] = [];
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const u = .012 * i, v = .012 * j;
    positions.push(.11 + u + .3 * v + .002 * random(), .2 + .4 * u + .2 * v + .002 * random(), .13 + .9 * v - .2 * u + .002 * random());
  }
  for (let j = 0; j + 1 < n; j++) for (let i = 0; i + 1 < n; i++) {
    const a = i + n * j, m = (i + j) % 3 === 0 ? 1 : 0;
    triangles.push(a, a + 1, a + n + 1, a, a + n + 1, a + n); triangleMaterials.push(m, m);
  }
  const rodOffsets = [0], rodIndices: number[] = [];
  for (let rod = 0; rod < 3; rod++) {
    for (let k = 0; k < 40; k++) { rodIndices.push(positions.length / 3); positions.push(.15 + .01 * k + .001 * random(), .55 + .03 * rod + .004 * Math.sin(k) + .001 * random(), .2 + .007 * k * (rod + 1) / 3 + .001 * random()); }
    rodOffsets.push(rodIndices.length);
  }
  const fixed = Array.from({ length: positions.length / 3 }, (_, v) => v % 97 === 3 ? 1 : 0);
  const data: ClothTopologyData = { restPositions_m: positions.map(Math.fround), triangles, triangleMaterials, rodOffsets, rodIndices, rodMaterials: [0, 1, 1], fixed };
  const materials: ClothMaterial[] = [DEFAULT_CLOTH_MATERIAL, { ...DEFAULT_CLOTH_MATERIAL, id: 'wool', thickness_m: .0007, rodRadius_m: .0003, rodRadiusB_m: .0002, fiberDiameter_m: .00021, contactAngle_rad: 1.1, waterViscosity_Pa_s: .0011 }];
  const state = createClothState(prepareClothTopology(data, materials)), V = state.dryMass_kg.length;
  for (let v = 0; v < V; v++) if (v % 3) state.waterVolume_m3[v] = (.1 + .5 * random()) * (state.controlVolume_m3[v]! - state.solidVolume_m3[v]!);
  const predictor = Float64Array.from({ length: 3 * V }, () => Math.fround(random() - .5));
  const step: UniformClothStep = { state, data, materials, solidPredictorVelocity_m_s: predictor, identity: { generation: state.generation, topologyRevision: state.topologyRevision, tick: 0, time_s: 0 }, dt_s: .002 };
  return { step, V };
}

async function upload(device: GPUDevice, step: UniformClothStep, V: number, fields: { origin_m: readonly number[]; cellSize_m: readonly number[] }) {
  const s = step.state, stencil = await UniformClothGPUStencil.create(device, await UniformGPUOrderingKernels.create(device), clothStencilTopology(step.data, step.materials, V));
  const vec4 = (f: (v: number) => readonly number[]) => Float32Array.from({ length: 4 * V }, (_, i) => f(i >> 2)[i & 3]!);
  device.queue.writeBuffer(stencil.positions, 0, vec4(v => [...s.positions_m.subarray(3 * v, 3 * v + 3), s.controlVolume_m3[v]!]));
  device.queue.writeBuffer(stencil.predictor, 0, vec4(v => [0, 1, 2, 3].map(k => k < 3 ? step.solidPredictorVelocity_m_s![3 * v + k]! : 0)));
  device.queue.writeBuffer(stencil.scalars, 0, vec4(v => [s.dryMass_kg[v]!, s.waterVolume_m3[v]!, s.solidVolume_m3[v]!, step.data.fixed![v]! & 1]));
  device.queue.writeBuffer(stencil.cornerApertures, 0, new Float32Array(8 * V).fill(1));
  stencil.writeParameters({ origin_m: fields.origin_m, cellSize_m: fields.cellSize_m, density_kg_m3: 1000, pressureOnSolid: true, solveSolid: true });
  return stencil;
}

const close = (actual: number, expected: number, scale: number, label: string, bound = 2e-6) => assert.ok(Math.abs(actual - expected) <= bound * scale, `${label}: ${actual} vs ${expected}`);
// Cell averages weight by f32 trilinear weights, whose fractional coordinate
// carries the f32 ulp of the lattice coordinate (about 4e-6 at q = 50).
const CELL = 3e-5;
// Link mass and occupied volume are products of three trilinear weights; each
// carries the f32 rounding of its lattice coordinate (|q| < 32: 2^-19 a cell).
const LINK = 3 * 2 ** -19;

test('GPU cloth stencil reproduces the CPU stencil exactly in layout and to f32 in values', { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice('gpu cloth stencil', async device => {
   // The key frame follows the cloth: also where every cell index is negative.
   // Both frames keep |q| < 32, one f32 binade, so the same value bounds hold.
   for (const origin_m of [[0, .1, 0], [.7, .65, .55]] as const) {
    const { step, V } = fixture(), fields = { origin_m, cellSize_m: [.02, .02, .02] as const };
    const oracle = prepareUniformClothStencil(fields, step), s = step.state;
    const stencil = await upload(device, step, V, fields);
    // Twice: the second build must not see the first build's slots.
    for (let pass = 0; pass < 2; pass++) { const e = device.createCommandEncoder(); stencil.encode(e); device.queue.submit([e.finish()]); }
    const links = oracle.gather.length, anchors = oracle.anchors.length / 8, graph = stencil.sections.graph;
    const counts = new Uint32Array(await read(device, stencil.counts, 16));
    assert.deepEqual([...counts.slice(0, 3)], [links, anchors, 0]);
    const anchorWords = new Int32Array(await read(device, stencil.anchors, 32 * (1 + anchors)));
    assert.deepEqual([...anchorWords.subarray(0, 8)], [anchors, 0, 0, 0, 0, 0, 0, 0]); assert.deepEqual(anchorWords.subarray(8), oracle.anchors);
    assert.deepEqual(new Uint32Array(await read(device, stencil.graph, 4 * links, 4 * graph.faces)), oracle.linkFaces);
    assert.deepEqual(new Uint32Array(await read(device, stencil.graph, 24 * V, 4 * graph.ranges)), oracle.ranges);
    assert.deepEqual(new Uint32Array(await read(device, stencil.graph, 4 * links, 4 * graph.gather)), oracle.gather);
    const coefficients = new Float32Array(await read(device, stencil.coefficients, 16 * links)), occupied = new Float32Array(await read(device, stencil.occupied, 4 * links));
    const weights = new Float32Array(await read(device, stencil.graph, 4 * links, 4 * graph.weights)), material = new Float32Array(await read(device, stencil.vertexMaterial, 64 * V));
    const massScale = Math.max(...oracle.coefficients.filter((_, i) => i % 4 === 0)), volumeScale = Math.max(...oracle.occupiedVolumes_m3);
    for (let j = 0; j < links; j++) {
      close(coefficients[4 * j]!, oracle.coefficients[4 * j]!, massScale, `mass ${j}`, LINK);
      assert.equal(coefficients[4 * j + 3], oracle.coefficients[4 * j + 3], `predictor ${j}`);
      close(occupied[j]!, oracle.occupiedVolumes_m3[j]!, volumeScale, `occupied ${j}`, LINK); close(weights[j]!, oracle.linkWeights[j]!, 1, `weight ${j}`);
    }
    const coefficientScale = Math.max(...oracle.poreCoefficient_Pa);
    for (let v = 0; v < V; v++) {
      close(material[16 * v + 1]!, oracle.poreCoefficient_Pa[v]!, coefficientScale, `capillary ${v}`);
      close(material[16 * v + 2]!, oracle.vertexMicro[2 * v]!, 3e-4, `fiber ${v}`); close(material[16 * v + 3]!, oracle.vertexMicro[2 * v + 1]!, 2e-3, `viscosity ${v}`);
      for (let k = 0; k < 9; k++) close(material[16 * v + 4 + 4 * Math.floor(k / 3) + k % 3]!, oracle.poreTensor[9 * v + k]!, 2, `tensor ${v}.${k}`);
    }
    // Drag colors: the oracle's priority-order greedy, exactly; lists per color.
    const expectedColors = clothDragColors(oracle, clothColorPriorities(fields, s.positions_m)), work = stencil.sections.work, used = Math.max(...expectedColors) + 1;
    assert.deepEqual(new Uint32Array(await read(device, stencil.counts, 16 + 8 * CLOTH_DRAG_COLORS)).subarray(3, 4 + used), Uint32Array.from([used, ...Array.from({ length: used }, (_, c) => expectedColors.filter(x => x === c).length)]));
    const gpuColors = new Uint32Array(await read(device, stencil.work, 4 * V, 4 * work.colors));
    assert.deepEqual(gpuColors, expectedColors);
    const lists = new Uint32Array(await read(device, stencil.graph, 4 * V, 4 * graph.colors)), colorRanges = new Uint32Array(await read(device, stencil.graph, 8 * used, 4 * graph.colorRanges));
    for (let c = 0; c < used; c++) {
      const members = [...lists.subarray(colorRanges[2 * c]!, colorRanges[2 * c]! + colorRanges[2 * c + 1]!)].sort((a, b) => a - b);
      assert.deepEqual(members, Array.from({ length: V }, (_, v) => v).filter(v => expectedColors[v] === c), `color ${c}`);
    }
    // A cloth wider than 32-bit face keys address is a status error, never a partial stencil.
    const outside = Float32Array.from({ length: 4 * V }, (_, i) => i === 0 ? -1e6 : (i & 3) === 3 ? 0 : s.positions_m[3 * (i >> 2) + (i & 3)]!);
    device.queue.writeBuffer(stencil.positions, 0, outside);
    const e = device.createCommandEncoder(); stencil.encode(e); device.queue.submit([e.finish()]);
    assert.equal(new Uint32Array(await read(device, stencil.counts, 16))[2]! & CLOTH_STENCIL_STATUS.outsideLattice, CLOTH_STENCIL_STATUS.outsideLattice);
    stencil.destroy();
   }
  });
});

test('GPU cloth cell tables reproduce the CPU drag material and pore-pressure cells', { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice('gpu cloth cells', async device => {
    const { step, V } = fixture(), fields = { origin_m: [0, .1, 0] as const, cellSize_m: [.02, .02, .02] as const };
    const oracle = prepareUniformClothStencil(fields, step), drag = prepareGridDragMaterial(fields, oracle, step), pore = preparePorePressureCells(fields, oracle, step);
    const stencil = await upload(device, step, V, fields), cells = await UniformClothGPUCells.create(device, await UniformGPUOrderingKernels.create(device), stencil);
    cells.writeParameters({ origin_m: fields.origin_m, cellSize_m: fields.cellSize_m, density_kg_m3: 1000, nonlinearDrag: true, exponent: oracle.exponent });
    for (let pass = 0; pass < 2; pass++) { const e = device.createCommandEncoder(); stencil.encode(e); cells.encode(e); device.queue.submit([e.finish()]); }
    const counts = new Uint32Array(await read(device, cells.counts, 16)), n = pore.length;
    assert.deepEqual([counts[1], counts[2]], [n, 0]);
    // Pore texture: header, then five texels a cell in z, y, x order.
    const width = cells.poreTexture.width, rows = Math.ceil((1 + 5 * n) / width), out = device.createBuffer({ size: 16 * width * rows, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const e = device.createCommandEncoder(); e.copyTextureToBuffer({ texture: cells.poreTexture }, { buffer: out, bytesPerRow: 16 * width }, [width, rows]); device.queue.submit([e.finish()]);
    await out.mapAsync(GPUMapMode.READ); const texels = new Float32Array(out.getMappedRange().slice(0)), bits = new Int32Array(texels.buffer); out.destroy();
    assert.equal(bits[0], n);
    const coefficientScale = Math.max(...pore.map(c => Math.abs(c.coefficient / c.weight)));
    pore.forEach((c, i) => {
      assert.deepEqual([...bits.subarray(4 * (1 + 5 * i), 4 * (1 + 5 * i) + 3)], c.p, `pore cell ${i}`);
      const m = texels.subarray(4 * (2 + 5 * i), 4 * (2 + 5 * i) + 4);
      close(m[0]!, c.coefficient / c.weight, coefficientScale, `pore coefficient ${i}`, CELL); close(m[1]!, c.solid, 1, `pore solid ${i}`, CELL); close(m[2]!, c.bound, 1, `pore bound ${i}`, CELL); assert.equal(m[3], c.hardOpen);
      for (let a = 0; a < 3; a++) for (let k = 0; k < 3; k++) close(texels[4 * (3 + a + 5 * i) + k]!, c.tensor[3 * a + k]! / c.weight, 2, `pore tensor ${i}`, CELL);
    });
    // Drag cells: the fiber-carrying cells, keyed by cell; shares per vertex.
    const records = new Float32Array(await read(device, cells.dragCells, 128 * n)), ints = new Int32Array(records.buffer);
    const byCell = new Map<string, number>(); for (let i = 0; i < n; i++) if (ints[32 * i + 3]) byCell.set([...ints.subarray(32 * i, 32 * i + 3)].join(), i);
    assert.equal(byCell.size, drag.length);
    const scale = Math.max(...drag.map(c => Math.max(...c.linear.map(Math.abs)))), nonlinearScale = Math.max(...drag.map(c => Math.max(...c.nonlinear.map(Math.abs))));
    const expectedShares = new Map<string, number>(), fiberOf = new Map<number, number>(), fiberScale = Math.max(...drag.map(d => d.fiberVolume_m3));
    for (const c of drag) {
      const i = byCell.get(c.p.join()); assert.ok(i !== undefined, `drag cell ${c.p}`);
      close(records[32 * i + 4]!, c.fiberVolume_m3, Math.max(...drag.map(d => d.fiberVolume_m3)), 'fiber', CELL); close(records[32 * i + 5]!, c.boundVolume_m3, Math.max(...drag.map(d => d.boundVolume_m3)), 'bound', CELL);
      for (let a = 0; a < 3; a++) for (let k = 0; k < 3; k++) {
        close(records[32 * i + 8 + 4 * a + k]!, c.linear[3 * a + k]!, scale, `linear ${c.p}`, CELL); close(records[32 * i + 20 + 4 * a + k]!, c.nonlinear[3 * a + k]!, nonlinearScale, `nonlinear ${c.p}`, CELL);
      }
      fiberOf.set(i, c.fiberVolume_m3);
      for (const m of c.members) expectedShares.set(`${m.vertex}:${i}`, (expectedShares.get(`${m.vertex}:${i}`) ?? 0) + m.share);
    }
    const links = new Uint32Array(await read(device, cells.dragLinks, 64 * V)), shares = new Float32Array(links.buffer);let live = 0;
    for (let v = 0; v < V; v++) for (let c = 0; c < 8; c++) {
      const cell = links[2 * (8 * v + c)]!; if (cell === 0xffffffff) continue; live++;
      // A share's error is relative to its cell's fiber: compare deposited fiber volumes.
      const expected = expectedShares.get(`${v}:${cell}`); assert.ok(expected !== undefined, `share ${v}:${cell}`);
      close(shares[2 * (8 * v + c) + 1]! * records[32 * cell + 4]!, expected * fiberOf.get(cell)!, fiberScale, `share ${v}`, CELL);
    }
    assert.equal(live, expectedShares.size);
    cells.destroy(); stencil.destroy();
  });
});
