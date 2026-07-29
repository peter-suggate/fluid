import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  applyOctreePressureMGPageChebyshev,
  octreePressureMGInteriorHaloIndex,
  planOctreePressureMGPageChebyshev,
} from "../lib/webgpu-octree-brick-stencils";

const oracleSource = readFileSync(
  new URL("../lib/webgpu-octree-brick-stencils.ts", import.meta.url), "utf8");
const spgridSource = readFileSync(
  new URL("../lib/webgpu-octree-spgrid-vcycle.ts", import.meta.url), "utf8");
const topologySource = readFileSync(
  new URL("../lib/webgpu-octree-power-topology.ts", import.meta.url), "utf8");

test("page Chebyshev remains offline while production uses global synchronized phases", () => {
  assert.doesNotMatch(oracleSource,
    /@compute|createShaderModule|createComputePipeline|dispatchWorkgroups|GPUBuffer|GPUDevice/,
    "the page oracle must never become another executable pressure path");
  assert.match(spgridSource,
    /@compute @workgroup_size\(64\) fn smoothChebyshevAtoB0/);
  assert.match(spgridSource,
    /@compute @workgroup_size\(64\) fn smoothChebyshevBtoA3/);
  assert.doesNotMatch(spgridSource, /fn smoothPageChebyshev/);
  assert.match(spgridSource, /const PAGE_X=8u;const PAGE_Y=8u;const PAGE_Z=4u/);
  assert.match(topologySource, /Section 6\.3 coefficients/,
    "Section 6.3 channels must enter the accepted topology publication");
  assert.doesNotMatch(topologySource, /PressureMGPageChebyshev/,
    "Section 6.3 publication must not select a second page executor");
});

test("pressure/MG page staging retains the measured 8x8x4 shape", () => {
  assert.deepEqual(planOctreePressureMGPageChebyshev(), {
    pageShape: [8, 8, 4],
    haloShape: [10, 10, 6],
    pageElements: 256,
    haloElements: 600,
    workgroupSize: 128,
    localF32Channels: 4,
    localIndexChannels: 1,
    workgroupBytes: 12_000,
    haloAmplification: 2.34375,
    localChebyshevSweeps: 4,
  });
});

test("pressure/MG halo indexing covers the 8x8x4 interior exactly", () => {
  const indices = new Set<number>();
  for (let z = 0; z < 4; z += 1) for (let y = 0; y < 8; y += 1) {
    for (let x = 0; x < 8; x += 1) {
      indices.add(octreePressureMGInteriorHaloIndex([x, y, z]));
    }
  }
  assert.equal(indices.size, 256);
  assert.ok([...indices].every((index) => index >= 111 && index <= 488));
  assert.throws(() => octreePressureMGInteriorHaloIndex([-1, 0, 0]), /outside 8x8x4/);
  assert.throws(() => octreePressureMGInteriorHaloIndex([8, 0, 0]), /outside 8x8x4/);
});

test("four page-local Chebyshev sweeps preserve constant and affine null modes", () => {
  const halo = new Float32Array(600);
  const affine = new Float32Array(600);
  const rhs = new Float32Array(600);
  const diagonal = new Float32Array(600).fill(6);
  for (let z = 0; z < 6; z += 1) for (let y = 0; y < 10; y += 1) {
    for (let x = 0; x < 10; x += 1) {
      const index = x + 10 * (y + 10 * z);
      halo[index] = 3.25;
      affine[index] = x + 2 * y + 3 * z;
    }
  }
  const weights = [0.18, 0.22, 0.28, 0.36] as const;
  const constantOut = applyOctreePressureMGPageChebyshev(
    halo, rhs, diagonal, weights,
  );
  assert.ok([...constantOut].every((value) => Math.abs(value - 3.25) < 1e-6));
  const affineOut = applyOctreePressureMGPageChebyshev(
    affine, rhs, diagonal, weights,
  );
  let cursor = 0;
  for (let z = 0; z < 4; z += 1) for (let y = 0; y < 8; y += 1) {
    for (let x = 0; x < 8; x += 1) {
      const expected = affine[octreePressureMGInteriorHaloIndex([x, y, z])]!;
      assert.ok(Math.abs(affineOut[cursor] - expected) < 2e-5);
      cursor += 1;
    }
  }
});

test("page oracle rejects malformed halo fields and diagonals", () => {
  const halo = new Float32Array(600);
  const rhs = new Float32Array(600);
  const diagonal = new Float32Array(600).fill(6);
  const weights = [0.18, 0.22, 0.28, 0.36] as const;
  assert.throws(() => applyOctreePressureMGPageChebyshev(
    new Float32Array(599), rhs, diagonal, weights,
  ), /600-value halo fields/);
  diagonal[octreePressureMGInteriorHaloIndex([0, 0, 0])] = 0;
  assert.throws(() => applyOctreePressureMGPageChebyshev(
    halo, rhs, diagonal, weights,
  ), /positive finite diagonals/);
});
