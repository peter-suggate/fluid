import test from 'node:test';
import assert from 'node:assert/strict';
import {clothContactStatic, gpuContactUnsupported, packClothColliders, clothStaticColliderRecords, KIND_BRICKS} from '../lib/cloth/gpu-contact';
import {clothLocalShellEdges, clothShellCache, clothVertexLocalFaces} from '../lib/cloth/contact-constraints';
import {DEFAULT_CLOTH_MATERIAL} from '../lib/cloth/material';
import {prepareClothTopology, type PreparedClothTopology} from '../lib/cloth/topology';
import type {ClothCollider} from '../lib/cloth/contact';
import type {ClothTopologyData} from '../lib/cloth/types';

/** A strip folded onto itself (rest flat, so the layers may contact), with a
 * fine pitch so each vertex's connected rest patch reaches past its one-ring. */
function foldedStrip(): PreparedClothTopology {
  const nx = 20, nz = 4, h = .0001, restPositions_m: number[] = [], triangles: number[] = [];
  for (let j = 0; j < nz; j++) for (let i = 0; i < nx; i++) restPositions_m.push(i*h, 0, j*h);
  for (let j = 0; j + 1 < nz; j++) for (let i = 0; i + 1 < nx; i++) {const p = j*nx + i, q = p + 1, r = p + nx, t = r + 1; triangles.push(p, r, q, q, r, t);}
  const data: ClothTopologyData = {restPositions_m, triangles, rodOffsets: [0], rodIndices: [], fixed: restPositions_m.filter((_, k) => k % 3 === 0).map(() => 0)};
  return prepareClothTopology(data, [DEFAULT_CLOTH_MATERIAL]);
}
const words = (statics: ReturnType<typeof clothContactStatic>, start: number, list: number, row: number) => Array.from(statics.words.subarray(statics.words[start + row]! + list, statics.words[start + row + 1]! + list));

test('GPU contact exclusions are the Float64 detection exclusions, pair for pair', () => {
  const topology = foldedStrip(), statics = clothContactStatic(topology), o = statics.offsets, n = topology.vertexCount;
  const ring = (i: number) => words(statics, o.sRingStart, o.sRingList, i), faceRes = (i: number) => words(statics, o.sFrStart, o.sFrList, i), vertexRes = (i: number) => words(statics, o.sLvStart, o.sLvList, i);
  // Vertex-face: one-ring of any face vertex, or the vertex's connected rest patch.
  let excludedByPatch = 0;
  for (let i = 0; i < n; i++) for (let f = 0; f < topology.triangles.length; f++) {
    const ids = topology.triangles[f]!.vertices;
    const shares = ids.includes(i) || ids.some(v => topology.incidentTriangles[i]!.some(g => topology.triangles[g]!.vertices.includes(v)));
    const cpu = shares || clothVertexLocalFaces(topology, i).has(f), gpu = ids.some(v => ring(i).includes(v)) || faceRes(i).includes(f);
    assert.equal(gpu, cpu, `vertex ${i} face ${f}`);
    if (cpu && !shares) excludedByPatch++;
  }
  assert.ok(excludedByPatch > 0, 'the rest patch reaches past the one-ring');
  // Edge-edge: shared triangle, either vertex's rest patch, or the edge's own rest patch.
  const edges = clothShellCache(topology).edges;
  assert.equal(statics.edges, edges.length);
  let excludedByEdgePatch = 0;
  for (let ai = 0; ai < edges.length; ai++) for (let bi = ai + 1; bi < edges.length; bi++) {
    const ea = edges[ai]!, eb = edges[bi]!;
    if (new Set([...ea.vertices, ...eb.vertices]).size < 4) continue;
    const sharesTriangle = ea.vertices.some(p => eb.vertices.some(q => topology.incidentTriangles[p]!.some(f => topology.triangles[f]!.vertices.includes(q))));
    const cpu = sharesTriangle || clothLocalShellEdges(topology, ea, eb);
    const cheap = ea.vertices.some(v => eb.vertices.some(w => ring(v).includes(w) || vertexRes(v).includes(w)));
    const gpu = cheap || words(statics, o.sE3Start, o.sE3List, ai).includes(bi);
    assert.equal(gpu, cpu, `edges ${ai} ${bi}`);
    if (cpu && !cheap) excludedByEdgePatch++;
  }
  // (On this pitch every edge patch lies inside its vertices' patches: excludedByEdgePatch is 0.)
  void excludedByEdgePatch;
  // Keys and weights follow the reference's edge orientation.
  edges.forEach((e, k) => assert.deepEqual([statics.words[o.sEdge + 4*k], statics.words[o.sEdge + 4*k + 1]], [...e.vertices]));
});

test('GPU contact detection names the collider sets it leaves to the CPU', () => {
  const topology = foldedStrip();
  assert.equal(gpuContactUnsupported(topology, [{kind: 'sphere', center_m: [0, 0, 0], radius_m: 1}, {kind: 'plane', normal: [0, 1, 0], offset_m: 0}]), undefined);
  assert.match(gpuContactUnsupported(topology, [{kind: 'sphere', center_m: [0, 0, 0], radius_m: 1, bodyId: 'ball'}])!, /reactions/);
  const moving: ClothCollider = {kind: 'sampled', maximumSpeed_m_s: 1, velocityAt: () => [0, 0, 0], sample: () => ({distance_m: 1, normal: [0, 1, 0]})};
  assert.match(gpuContactUnsupported(topology, [moving])!, /moving sampled/);
  clothStaticColliderRecords.set(moving, {h: .01, records: [], cuts: 1});
  assert.match(gpuContactUnsupported(topology, [moving])!, /cuts/);
  assert.throws(() => packClothColliders([moving], [0, 0, 0], 0), /unsupported sampled/);
  assert.throws(() => packClothColliders([{kind: 'capsule'} as unknown as ClothCollider], [0, 0, 0], 0), /unsupported collider capsule/);
});

test('static solid records pack into a bounded hierarchy with every record reachable once', () => {
  const values = Array.from({length: 729}, (_, k) => (k % 9) - 4);
  const records = Array.from({length: 37}, (_, k) => ({min: [k*.08, 0, 0] as const, max: [k*.08 + .08, .08, .08] as const, values: k % 3 ? values : undefined, lipschitz: 1.5}));
  const solid: ClothCollider = {kind: 'sampled', maximumSpeed_m_s: 0, velocityAt: () => [0, 0, 0], sample: () => ({distance_m: 1, normal: [0, 1, 0]})};
  clothStaticColliderRecords.set(solid, {h: .01, records, cuts: 0});
  const base = 100, packed = packClothColliders([solid], [.5, 0, 0], base), f = new Float32Array(packed.buffer);
  assert.equal(packed[0], KIND_BRICKS);
  const root = packed[9]!, count = packed[10]!, local = (word: number) => word - base;
  assert.equal(count, records.length);
  const seen = new Set<number>(), stack = [root];
  while (stack.length) {
    const node = local(stack.pop()!), first = packed[node + 6]!, n = packed[node + 7]!;
    if (!n) {stack.push(first, first + 8); continue;}
    for (let r = 0; r < n; r++) {
      const rec = local(first) + 12*r;
      assert.ok(!seen.has(rec), 'each record once');
      seen.add(rec);
      // Inside its leaf's padded bounds, relative to the frame.
      for (let a = 0; a < 3; a++) assert.ok(f[rec + a]! + f[rec + 3 + a]! >= f[node + a]! && f[rec + a]! + f[rec + 3 + a]! + f[rec + 6 + a]! <= f[node + 3 + a]! + 1e-6);
      const at = packed[rec + 10]!;
      if (at !== 0xffffffff) assert.deepEqual(Array.from(f.subarray(local(at), local(at) + 729)), values);
    }
  }
  assert.equal(seen.size, records.length);
});
