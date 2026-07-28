import assert from "node:assert/strict";
import test from "node:test";
import { packOctreePowerRowTemplateSlot } from "../lib/octree-power-catalog";
import {
  buildResolvedPowerRowsPublication,
  planResolvedPowerRows,
} from "../lib/webgpu-octree-power-resolved-rows";
import {
  OCTREE_STRUCTURED_VELOCITY_INVALID_HANDLE,
  publishStructuredVelocityFamilies,
  reconstructStructuredRowVelocity,
  type StructuredVelocityCatalogSource,
} from "../lib/webgpu-octree-structured-velocity";
import {
  directStructuredVelocityPublicationWGSL,
  planStructuredVelocityGPU,
} from "../lib/webgpu-octree-structured-velocity-gpu";
import {
  applyOctreePressureMGPageChebyshev,
  octreePressureMGInteriorHaloIndex,
} from "../lib/webgpu-octree-brick-stencils";
import { traceFineLevelSetDeparture } from "../lib/octree-fine-levelset-transport";
import { structuredFineLevelSetTransportWGSL } from
  "../lib/webgpu-octree-fine-levelset-transport";

const INVALID = OCTREE_STRUCTURED_VELOCITY_INVALID_HANDLE;
const noFamilyHandles = [INVALID, INVALID, INVALID, INVALID, INVALID, INVALID];
const normals = [
  [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
] as const;

const dot = (a: readonly number[], b: readonly number[]) =>
  a.reduce((sum, value, axis) => sum + value * b[axis]!, 0);

function differentialCatalog(): StructuredVelocityCatalogSource {
  const rowTemplateHeaders = new Uint32Array([0, 6, 0, 6, 6, 6, 0, 6]);
  const rowTemplateSlots = new Uint32Array(12);
  const rowTemplateData = new Float32Array(36);
  const reconstructionData = new Float32Array(36);
  const faceData = new Float32Array(144);
  for (let entry = 0; entry < 2; entry += 1) {
    for (let local = 0; local < 6; local += 1) {
      const slot = entry * 6 + local;
      const normalIndex = entry === 0 ? local : local === 0 ? 1 : local === 1 ? 0 : local;
      const normal = normals[normalIndex]!;
      const axis = normal.findIndex((value) => value !== 0);
      rowTemplateSlots[slot] = packOctreePowerRowTemplateSlot({
        neighborSelector: local === 0 ? 0 : 0xff,
        family: axis,
        orientation: normal[axis]! > 0 ? 1 : 0,
        transferRelation: local === 0 ? 0 : 3,
        dynamicBoundarySlot: local,
        worldBoundary: local !== 0,
      });
      rowTemplateData.set([1, 1, 1], slot * 3);
      reconstructionData.set(normal.map((value) => 0.5 * value), slot * 3);
      faceData.set(normal, slot * 12 + 8);
    }
  }
  return { rowTemplateHeaders, rowTemplateSlots, rowTemplateData, reconstructionData, faceData };
}

function differentialRows() {
  return buildResolvedPowerRowsPublication(planResolvedPowerRows(4, 6), 11, [
    {
      row: 0, caseId: 1, catalogEntry: 0, physicalPage: 0, localCell: 0,
      volume: 1, scale: 1, neighborRows: [1], neighborCoefficients: [1],
      velocityFamilyHandles: noFamilyHandles, transition: false, physicalBoundary: true,
    },
    {
      row: 1, caseId: 1, catalogEntry: 1, physicalPage: 0, localCell: 1,
      volume: 1, scale: 1, neighborRows: [0], neighborCoefficients: [1],
      velocityFamilyHandles: noFamilyHandles, transition: false, physicalBoundary: true,
    },
  ]);
}

/** Emulates only the production reconstruction entry point over its packed SoA ABI. */
function reconstructPackedProductionRow(
  packed: Uint32Array,
  values: Float32Array,
  row: number,
  maximumCaseSlots: number,
  offsets: ReturnType<typeof planStructuredVelocityGPU>["offsets"],
  reconstruction: Float32Array,
): readonly [number, number, number] {
  const base = row * maximumCaseSlots;
  const canonical = new Float32Array(3);
  for (let local = 0; local < 6; local += 1) {
    const at = base + local;
    const handle = packed[offsets.rowSlotHandles + at]!;
    const sign = new Int32Array(packed.buffer)[offsets.rowSlotSigns + at]!;
    const catalogSlot = packed[offsets.rowCatalogSlots + at]!;
    for (let axis = 0; axis < 3; axis += 1) {
      canonical[axis] = Math.fround(canonical[axis]!
        + Math.fround(reconstruction[catalogSlot * 3 + axis]! * sign * values[handle]!));
    }
  }
  return [canonical[0]!, canonical[1]!, canonical[2]!];
}

test("packed production reconstruction is differential-exact with the named CPU oracle", () => {
  const catalog = differentialCatalog();
  const inputs = [
    { row: 0, neighborRows: [1, INVALID, INVALID, INVALID, INVALID, INVALID] },
    { row: 1, neighborRows: [0, INVALID, INVALID, INVALID, INVALID, INVALID] },
  ];
  const publication = publishStructuredVelocityFamilies(differentialRows(), catalog, inputs);
  for (let handle = 0; handle < publication.slotCount; handle += 1) {
    publication.values[handle] = Math.fround(0.1375 * (handle + 1) - 0.43);
  }

  const plan = planStructuredVelocityGPU(publication.rowCapacity, 30, 256);
  const packed = new Uint32Array(plan.authorityWords);
  const packedFloats = new Float32Array(packed.buffer);
  packedFloats.set(publication.values, plan.offsets.values);
  const packedSigns = new Int32Array(packed.buffer);
  for (let row = 0; row < 2; row += 1) for (let local = 0; local < 6; local += 1) {
    const oracleAt = row * publication.maximumCaseSlots + local;
    const productionAt = row * plan.maximumCaseSlots + local;
    packed[plan.offsets.rowSlotHandles + productionAt] = publication.rowSlotHandles[oracleAt]!;
    packedSigns[plan.offsets.rowSlotSigns + productionAt] = publication.rowSlotSigns[oracleAt]!;
    packed[plan.offsets.rowCatalogSlots + productionAt] = publication.rowCatalogSlots[oracleAt]!;
  }

  assert.match(directStructuredVelocityPublicationWGSL,
    /fn reconstructStructuredCellVelocity[\s\S]*rowHandleOffset[\s\S]*rowSignOffset[\s\S]*rowCatalogOffset[\s\S]*reconstructionOffset/,
    "the compared packed fields must remain the production reconstruction inputs");
  for (const row of [0, 1]) {
    const production = reconstructPackedProductionRow(packed, packedFloats.subarray(
      plan.offsets.values, plan.offsets.values + plan.slotCapacity), row,
    plan.maximumCaseSlots, plan.offsets, catalog.reconstructionData);
    const oracle = reconstructStructuredRowVelocity(publication, catalog, row);
    production.forEach((value, axis) => assert.ok(Math.abs(value - oracle[axis]!) < 2e-7,
      `row ${row} axis ${axis}: packed production ${value} != oracle ${oracle[axis]}`));
  }
});

type Vec3 = readonly [number, number, number];

function affineVelocity([x, y, z]: Vec3): Vec3 {
  return [0.4 + 0.12 * x - 0.07 * y + 0.03 * z,
    -0.2 + 0.05 * x + 0.09 * y - 0.02 * z,
    0.1 - 0.04 * x + 0.02 * y + 0.08 * z];
}

/** Exact CPU emulation of the production regularSample cube-trilinear gather. */
function productionRegularSample(point: Vec3): Vec3 {
  const low = point.map((value) => Math.floor(value - 0.5)) as [number, number, number];
  const t = point.map((value, axis) => value - (low[axis]! + 0.5)) as [number, number, number];
  const output = [0, 0, 0];
  for (let corner = 0; corner < 8; corner += 1) {
    const q = [low[0] + (corner & 1), low[1] + ((corner >> 1) & 1),
      low[2] + ((corner >> 2) & 1)] as Vec3;
    const velocity = affineVelocity(q.map((value) => value + 0.5) as [number, number, number]);
    const weight = ((corner & 1) ? t[0] : 1 - t[0])
      * ((corner & 2) ? t[1] : 1 - t[1])
      * ((corner & 4) ? t[2] : 1 - t[2]);
    for (let axis = 0; axis < 3; axis += 1) output[axis] += weight * velocity[axis]!;
  }
  return output as unknown as Vec3;
}

test("production regular structured transport matches affine interpolation and the fine trace oracle", () => {
  // Each corner now resolves its owner by position and checks that owner's
  // size and centre before contributing, instead of taking the anchor's tag on
  // faith. A neighbour at a different resolution is rejected rather than
  // interpolated, which is the case the oracle below cannot model -- so the
  // rejection is asserted here rather than left implicit.
  assert.match(structuredFineLevelSetTransportWGSL,
    /fn regularSampleExact[\s\S]*for\(var corner=0u;corner<8u;corner\+=1u\)[\s\S]*taggedVelocity\(owner\.tag\)[\s\S]*result\+=w\*value\.xyz/,
    "the numerical differential must remain attached to the production trilinear gather");
  assert.match(structuredFineLevelSetTransportWGSL,
    /if\(owner\.tag==INVALID\|\|owner\.size!=anchor\.size\|\|any\(abs\(ownerCenter-samplePoint\)>vec3f\(tolerance\)\)\)\{return RegularAttempt\(vec4f\(0,0,0,-1\),0u\);\}/,
    "a corner owner at another resolution must reject the sample, not be interpolated");
  assert.match(structuredFineLevelSetTransportWGSL,
    /fn regularCommonDeparture[\s\S]*x-=bitcast<f32>\(state\[2\]\)\*v\.xyz/,
    "the differential must remain attached to the production microstep update");

  const point: Vec3 = [1.18, 1.31, 1.43];
  const interpolated = productionRegularSample(point);
  const analytic = affineVelocity(point);
  interpolated.forEach((value, axis) => assert.ok(Math.abs(value - analytic[axis]!) < 2e-15));

  const dt = 0.24, segments = 8;
  const production = traceFineLevelSetDeparture(point, dt, segments, productionRegularSample);
  let reference = [...point] as [number, number, number];
  for (let stage = 0; stage < segments; stage += 1) {
    const velocity = affineVelocity(reference);
    reference = reference.map((value, axis) => value - dt / segments * velocity[axis]!) as
      [number, number, number];
  }
  production.departure.forEach((value, axis) =>
    assert.ok(Math.abs(value - reference[axis]!) < 2e-15));
  assert.equal(production.segments, 8);
  assert.equal(production.finite, true);
});

function independentChebyshevReference(
  input: Float32Array,
  rhs: Float32Array,
  diagonal: Float32Array,
  weights: readonly number[],
): Float32Array {
  let source = input.slice();
  for (const weight of weights) {
    const target = source.slice();
    for (let local = 0; local < 256; local += 1) {
      const x = local % 8, yz = Math.floor(local / 8);
      const y = yz % 8, z = Math.floor(yz / 8);
      const at = (x + 1) + 10 * ((y + 1) + 10 * (z + 1));
      const image = diagonal[at]! * source[at]!
        - [at - 1, at + 1, at - 10, at + 10, at - 100, at + 100]
          .reduce((sum, neighbor) => sum + source[neighbor]!, 0);
      target[at] = Math.fround(source[at]! + weight * (rhs[at]! - image) / diagonal[at]!);
    }
    source = target;
  }
  const output = new Float32Array(256);
  for (let local = 0; local < 256; local += 1) {
    const x = local % 8, yz = Math.floor(local / 8);
    output[local] = source[(x + 1) + 10 * ((yz % 8 + 1) + 10 * (Math.floor(yz / 8) + 1))]!;
  }
  return output;
}

test("page brick stencil matches an independent dense-halo Chebyshev reference", () => {
  const input = new Float32Array(600);
  const rhs = new Float32Array(600);
  const diagonal = new Float32Array(600);
  for (let index = 0; index < 600; index += 1) {
    input[index] = Math.fround(Math.sin(index * 0.37) + 0.002 * index);
    rhs[index] = Math.fround(Math.cos(index * 0.19) - 0.001 * index);
    diagonal[index] = Math.fround(6.5 + 0.25 * Math.sin(index * 0.11));
  }
  const weights = [0.13, 0.21, 0.29, 0.34] as const;
  const production = applyOctreePressureMGPageChebyshev(input, rhs, diagonal, weights);
  const reference = independentChebyshevReference(input, rhs, diagonal, weights);
  assert.deepEqual(production, reference,
    "production page indexing, ping/pong order, and four local sweeps must equal the independent reference");
  assert.equal(production.length, 256);
  assert.equal(octreePressureMGInteriorHaloIndex([7, 7, 3]), 488);
});
