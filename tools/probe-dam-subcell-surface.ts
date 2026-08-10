/**
 * Dawn-only audit of sub-cell authority in the production factor-one dam.
 *
 * At the requested accepted timesteps this compares:
 *  - zero crossings on the accepted adaptive graph's leaf edges;
 *  - zero crossings after exact trilinear reconstruction on the unit lattice;
 *  - vertices emitted by the production classify/scan/contour shaders.
 *
 * Usage:
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
 *     node --import tsx tools/run-webgpu-exclusive.ts \
 *     --import tsx tools/probe-dam-subcell-surface.ts
 */
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";

import { octreeMethod } from "../lib/methods/octree";
import type { GPUSolverInstance } from "../lib/methods/types";
import { getScenePreset } from "../lib/scenes";
import { requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";
import { globalFineSurfaceClassificationShader } from
  "../lib/webgpu-water-global-fine-classify";
import { globalFineClassifiedEmitShader, globalFineClassifiedIndirectScanShader } from
  "../lib/webgpu-water-global-fine-tetra";

type Point = readonly [number, number, number];
interface Snapshot {
  readonly graphControl: Uint32Array; readonly phiControl: Uint32Array;
  readonly leaves: Uint32Array; readonly nodalPhi: Uint32Array;
  readonly nodes: Uint32Array; readonly constraints: Uint32Array;
  readonly nodalVelocity: Uint32Array; readonly transportBandMask: Uint32Array;
  readonly redistanceDistanceA: Float32Array; readonly redistanceDistanceB: Float32Array;
  readonly phiReceipts: Uint32Array;
  readonly stencils: Uint32Array; readonly extendedFaceVelocity: Uint32Array;
  readonly renderer: Uint32Array; readonly dimensions: readonly [number, number, number];
}

const INVALID = 0xffff_ffff;
const GRID_EPSILON = 2e-5;
const EDGE_CORNERS = [
  [0, 1], [3, 2], [4, 5], [7, 6],
  [0, 3], [1, 2], [4, 7], [5, 6],
  [0, 4], [1, 5], [2, 6], [3, 7],
] as const;
const floatWord = new Uint32Array(1); const floatValue = new Float32Array(floatWord.buffer);
function fromBits(bits: number) { floatWord[0] = bits >>> 0; return floatValue[0]!; }

function constrainedValue(snapshot: Snapshot, slot: number, bank: number) {
  const count = snapshot.constraints[12 * slot + 1] ?? 0;
  if (count === 0) return fromBits(snapshot.nodalPhi[2 * slot + bank] ?? 0);
  const denominator = snapshot.constraints[12 * slot + 2] ?? 0;
  if ((count !== 2 && count !== 4) || denominator === 0) return Number.NaN;
  let value = 0;
  for (let term = 0; term < count; term += 1) {
    const master = snapshot.constraints[12 * slot + 4 + term] ?? INVALID;
    value += (snapshot.constraints[12 * slot + 8 + term] ?? 0)
      * fromBits(snapshot.nodalPhi[2 * master + bank] ?? 0);
  }
  return value / denominator;
}

function mixedCornerFreezeAudit(snapshot: Snapshot) {
  const leafCount = Math.min(snapshot.graphControl[1] ?? 0, snapshot.leaves.length / 16);
  const nodeCount = Math.min(snapshot.graphControl[2] ?? 0, snapshot.nodalPhi.length / 2);
  const bank = (snapshot.phiControl[6] ?? 0) & 1;
  const distanceA = new Uint32Array(snapshot.redistanceDistanceA.buffer,
    snapshot.redistanceDistanceA.byteOffset, snapshot.redistanceDistanceA.length);
  const distanceB = new Uint32Array(snapshot.redistanceDistanceB.buffer,
    snapshot.redistanceDistanceB.byteOffset, snapshot.redistanceDistanceB.length);
  const mixedCorners = new Set<number>();
  for (let leaf = 0; leaf < leafCount; leaf += 1) {
    const slots = Array.from({ length: 8 }, (_, corner) => snapshot.leaves[16 * leaf + 8 + corner]!);
    const values = slots.map(slot => slot < nodeCount
      ? fromBits(snapshot.nodalPhi[2 * slot + bank] ?? 0) : Number.NaN);
    if (values.every(Number.isFinite) && Math.min(...values) <= 0 && Math.max(...values) >= 0) {
      for (const slot of slots) mixedCorners.add(slot);
    }
  }
  let missingFreezeBits = 0, distanceMarkerMismatches = 0, masterClosureMismatches = 0;
  for (const slot of mixedCorners) {
    if (((snapshot.transportBandMask[slot] ?? 0) & 2) === 0) missingFreezeBits += 1;
    const phiMagnitude = (snapshot.nodalPhi[2 * slot + bank] ?? 0) & 0x7fff_ffff;
    const a = distanceA[slot] ?? 0, b = distanceB[slot] ?? 0;
    if ((a & 0x8000_0000) === 0 || (b & 0x8000_0000) === 0
      || (a & 0x7fff_ffff) !== phiMagnitude || (b & 0x7fff_ffff) !== phiMagnitude) {
      distanceMarkerMismatches += 1;
    }
    const count = snapshot.constraints[12 * slot + 1] ?? 0;
    for (let term = 0; term < count; term += 1) {
      const master = snapshot.constraints[12 * slot + 4 + term] ?? INVALID;
      if (master >= nodeCount || ((snapshot.transportBandMask[master] ?? 0) & 2) === 0) {
        masterClosureMismatches += 1;
      }
    }
  }
  return { mixedCornerCount: mixedCorners.size, missingFreezeBits,
    distanceMarkerMismatches, masterClosureMismatches };
}

function crossing(a: number, b: number) {
  if (!Number.isFinite(a) || !Number.isFinite(b) || a === b) return undefined;
  if (!((a < 0 && b >= 0) || (b < 0 && a >= 0))) return undefined;
  const fraction = -a / (b - a);
  return fraction >= 0 && fraction <= 1 ? fraction : undefined;
}

function fractionalPart(value: number) {
  const nearest = Math.round(value);
  return Math.abs(value - nearest) <= GRID_EPSILON ? 0 : value - Math.floor(value);
}

function pointDistribution(points: readonly Point[]) {
  const unique = new Map<string, Point>();
  for (const point of points) unique.set(point.map(value => value.toFixed(7)).join(","), point);
  const coordinateFractions = [...unique.values()].flatMap(point => point.map(fractionalPart));
  const offGrid = coordinateFractions.filter(value => value !== 0);
  const histogram = Array.from({ length: 10 }, () => 0);
  for (const fraction of offGrid) histogram[Math.min(9, Math.floor(fraction * 10))]! += 1;
  const offGridPoints = [...unique.values()].filter(point => point.some(value => fractionalPart(value) !== 0));
  return {
    pointCount: points.length,
    uniquePointCount: unique.size,
    allCoordinatesOnGridCount: unique.size - offGridPoints.length,
    anyCoordinateOffGridCount: offGridPoints.length,
    anyCoordinateOffGridFraction: offGridPoints.length / Math.max(1, unique.size),
    coordinateCount: coordinateFractions.length,
    offGridCoordinateCount: offGrid.length,
    offGridFractionHistogramTenths: histogram,
    offGridSamples: offGridPoints.slice(0, 8),
  };
}

function quantizedPointKey(point: Point) {
  return point.map(value => Math.round(value * 65_536)).join(",");
}

function acceptedScalarCrossings(snapshot: Snapshot) {
  const [nx, ny, nz] = snapshot.dimensions;
  const dx = nx + 1, dy = ny + 1, dz = nz + 1;
  const lattice = new Float32Array(dx * dy * dz); lattice.fill(Number.NaN);
  const at = (x: number, y: number, z: number) => x + dx * (y + dy * z);
  const leafCount = Math.min(snapshot.graphControl[1] ?? 0, snapshot.leaves.length / 16);
  const nodeCount = Math.min(snapshot.graphControl[2] ?? 0, snapshot.nodes.length / 4);
  const bank = (snapshot.phiControl[6] ?? 0) & 1;
  const adaptivePoints: Point[] = []; const adaptiveEdges = new Set<string>();
  const crossingLeafSpanHistogram: Record<string, number> = {};
  let maximumSharedLatticeMismatch = 0;
  for (let leaf = 0; leaf < leafCount; leaf += 1) {
    const base = 16 * leaf;
    const origin = [snapshot.leaves[base]!, snapshot.leaves[base + 1]!,
      snapshot.leaves[base + 2]!] as const;
    const span = snapshot.leaves[base + 3]!;
    if (!span) continue;
    const slots = Array.from({ length: 8 }, (_, corner) => snapshot.leaves[base + 8 + corner]!);
    const values = slots.map(slot => slot < nodeCount ? constrainedValue(snapshot, slot, bank) : Number.NaN);
    let leafHasCrossing = false;
    for (const [ca, cb] of EDGE_CORNERS) {
      const pa = [origin[0] + (ca & 1 ? span : 0), origin[1] + (ca & 2 ? span : 0),
        origin[2] + (ca & 4 ? span : 0)] as const;
      const pb = [origin[0] + (cb & 1 ? span : 0), origin[1] + (cb & 2 ? span : 0),
        origin[2] + (cb & 4 ? span : 0)] as const;
      const key = [pa.join(","), pb.join(",")].sort().join("|");
      if (adaptiveEdges.has(key)) continue;
      adaptiveEdges.add(key);
      const t = crossing(values[ca]!, values[cb]!);
      if (t !== undefined) {
        leafHasCrossing = true;
        adaptivePoints.push([pa[0] + t * (pb[0] - pa[0]),
          pa[1] + t * (pb[1] - pa[1]), pa[2] + t * (pb[2] - pa[2])]);
      }
    }
    if (leafHasCrossing) crossingLeafSpanHistogram[span] =
      (crossingLeafSpanHistogram[span] ?? 0) + 1;
    if (values.some(value => !Number.isFinite(value))) continue;
    for (let z = 0; z <= span; z += 1) for (let y = 0; y <= span; y += 1)
      for (let x = 0; x <= span; x += 1) {
        const tx = x / span, ty = y / span, tz = z / span;
        let value = 0;
        for (let corner = 0; corner < 8; corner += 1) value += values[corner]!
          * (corner & 1 ? tx : 1 - tx) * (corner & 2 ? ty : 1 - ty)
          * (corner & 4 ? tz : 1 - tz);
        const index = at(origin[0] + x, origin[1] + y, origin[2] + z);
        if (Number.isFinite(lattice[index])) maximumSharedLatticeMismatch = Math.max(
          maximumSharedLatticeMismatch, Math.abs(lattice[index]! - value));
        else lattice[index] = value;
      }
  }
  const finePoints: Point[] = []; const productionQuantizedFinePoints: Point[] = [];
  let maximumContourQuantizationCells = 0;
  for (let z = 0; z <= nz; z += 1) for (let y = 0; y <= ny; y += 1)
    for (let x = 0; x <= nx; x += 1) for (let axis = 0; axis < 3; axis += 1) {
      const q = [x, y, z]; if (q[axis]! >= snapshot.dimensions[axis]!) continue;
      q[axis]! += 1;
      const a = lattice[at(x, y, z)]!, b = lattice[at(q[0]!, q[1]!, q[2]!)]!;
      const t = crossing(a, b); if (t === undefined) continue;
      const point = [x, y, z]; point[axis]! += t; finePoints.push(point as [number, number, number]);
      const quantized = [x, y, z];
      const qt = Math.round(t * 65_536) / 65_536; quantized[axis]! += qt;
      maximumContourQuantizationCells = Math.max(maximumContourQuantizationCells, Math.abs(qt - t));
      productionQuantizedFinePoints.push(quantized as [number, number, number]);
    }
  return { maximumSharedLatticeMismatch,
    crossingLeafSpanHistogram,
    adaptiveLeafEdges: pointDistribution(adaptivePoints),
    reconstructedUnitEdges: pointDistribution(finePoints),
    productionQuantizedUnitEdges: pointDistribution(productionQuantizedFinePoints),
    maximumContourQuantizationCells,
    productionQuantizedFinePointKeys: new Set(productionQuantizedFinePoints.map(quantizedPointKey)) };
}

function nodePosition(snapshot: Snapshot, slot: number): Point {
  const [nx, ny] = snapshot.dimensions;
  const item = snapshot.nodes[4 * slot] ?? 0;
  const dx = nx + 1, dy = ny + 1;
  return [item % dx, Math.floor(item / dx) % dy, Math.floor(item / (dx * dy))];
}

function sampleLeafValues(snapshot: Snapshot, values: readonly number[], point: Point, leaf: number) {
  const base = 16 * leaf;
  const span = snapshot.leaves[base + 3] ?? 0;
  if (span === 0 || base + 15 >= snapshot.leaves.length) return Number.NaN;
  const origin = [snapshot.leaves[base]!, snapshot.leaves[base + 1]!,
    snapshot.leaves[base + 2]!] as const;
  const t = point.map((value, axis) => Math.min(1, Math.max(0,
    (value - origin[axis]!) / span))) as [number, number, number];
  let result = 0;
  for (let corner = 0; corner < 8; corner += 1) {
    const slot = snapshot.leaves[base + 8 + corner] ?? INVALID;
    const value = values[slot] ?? Number.NaN;
    if (!Number.isFinite(value)) return Number.NaN;
    result += value * (corner & 1 ? t[0] : 1 - t[0])
      * (corner & 2 ? t[1] : 1 - t[1]) * (corner & 4 ? t[2] : 1 - t[2]);
  }
  return result;
}

function projectConstraints(snapshot: Snapshot, values: number[]) {
  for (let slot = 0; slot < values.length; slot += 1) {
    const count = snapshot.constraints[12 * slot + 1] ?? 0;
    const denominator = snapshot.constraints[12 * slot + 2] ?? 0;
    if ((count !== 2 && count !== 4) || denominator === 0) continue;
    let value = 0;
    for (let term = 0; term < count; term += 1) {
      const master = snapshot.constraints[12 * slot + 4 + term] ?? INVALID;
      value += (snapshot.constraints[12 * slot + 8 + term] ?? 0)
        * (values[master] ?? Number.NaN);
    }
    values[slot] = value / denominator;
  }
}

function crossingGeometry(snapshot: Snapshot, values: readonly number[]) {
  const points = new Map<string, Point>();
  const leafCount = Math.min(snapshot.graphControl[1] ?? 0, snapshot.leaves.length / 16);
  for (let leaf = 0; leaf < leafCount; leaf += 1) {
    const base = 16 * leaf, span = snapshot.leaves[base + 3] ?? 0;
    if (!span) continue;
    const origin = [snapshot.leaves[base]!, snapshot.leaves[base + 1]!,
      snapshot.leaves[base + 2]!] as const;
    for (const [ca, cb] of EDGE_CORNERS) {
      const sa = snapshot.leaves[base + 8 + ca] ?? INVALID;
      const sb = snapshot.leaves[base + 8 + cb] ?? INVALID;
      const t = crossing(values[sa] ?? Number.NaN, values[sb] ?? Number.NaN);
      if (t === undefined) continue;
      const pa = [origin[0] + (ca & 1 ? span : 0), origin[1] + (ca & 2 ? span : 0),
        origin[2] + (ca & 4 ? span : 0)] as const;
      const pb = [origin[0] + (cb & 1 ? span : 0), origin[1] + (cb & 2 ? span : 0),
        origin[2] + (cb & 4 ? span : 0)] as const;
      const edge = [pa.join(","), pb.join(",")].sort().join("|");
      points.set(edge, [pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1]),
        pa[2] + t * (pb[2] - pa[2])]);
    }
  }
  return points;
}

function geometryDelta(from: ReadonlyMap<string, Point>, to: ReadonlyMap<string, Point>) {
  const shared = [...from.keys()].filter(key => to.has(key));
  let total = 0, maximum = 0, dx = 0, dy = 0, dz = 0;
  for (const key of shared) {
    const a = from.get(key)!, b = to.get(key)!;
    const distance = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    total += distance; maximum = Math.max(maximum, distance);
    dx += b[0] - a[0]; dy += b[1] - a[1]; dz += b[2] - a[2];
  }
  return { fromEdges: from.size, toEdges: to.size, sharedEdges: shared.length,
    lostEdges: from.size - shared.length, gainedEdges: to.size - shared.length,
    meanSharedDisplacementCells: total / Math.max(1, shared.length),
    maximumSharedDisplacementCells: maximum,
    meanSharedDeltaCells: [dx, dy, dz].map(value => value / Math.max(1, shared.length)) };
}

function adaptiveTransportAudit(snapshot: Snapshot, departures: Uint32Array, dt_s: number) {
  const nodeCount = Math.min(snapshot.graphControl[2] ?? 0, snapshot.nodes.length / 4);
  const currentBank = (snapshot.phiControl[6] ?? 0) & 1;
  const priorBank = 1 - currentBank;
  const prior = Array.from({ length: nodeCount }, (_, slot) =>
    constrainedValue(snapshot, slot, priorBank));
  const current = Array.from({ length: nodeCount }, (_, slot) =>
    constrainedValue(snapshot, slot, currentBank));
  const predictor = [...prior];
  const corrected = [...prior];
  const backwardLeaves = new Uint32Array(nodeCount); backwardLeaves.fill(INVALID);
  const reverseLeaves = new Uint32Array(nodeCount); reverseLeaves.fill(INVALID);
  const backwardPoints: Point[] = Array.from({ length: nodeCount }, () => [0, 0, 0]);
  const reversePoints: Point[] = Array.from({ length: nodeCount }, () => [0, 0, 0]);
  for (let slot = 0; slot < nodeCount; slot += 1) {
    if ((snapshot.constraints[12 * slot + 1] ?? 0) !== 0
      || (snapshot.transportBandMask[slot] ?? 0) === 0) continue;
    const base = 8 * slot;
    const leaf = departures[base + 3] ?? INVALID;
    const reverseLeaf = departures[base + 7] ?? INVALID;
    if (leaf >= (snapshot.graphControl[1] ?? 0)) continue;
    const point = [fromBits(departures[base] ?? 0), fromBits(departures[base + 1] ?? 0),
      fromBits(departures[base + 2] ?? 0)] as Point;
    backwardLeaves[slot] = leaf; backwardPoints[slot] = point;
    predictor[slot] = Math.round(sampleLeafValues(snapshot, prior, point, leaf) * 65_536) / 65_536;
    if (reverseLeaf < (snapshot.graphControl[1] ?? 0)) {
      reverseLeaves[slot] = reverseLeaf;
      reversePoints[slot] = [fromBits(departures[base + 4] ?? 0),
        fromBits(departures[base + 5] ?? 0), fromBits(departures[base + 6] ?? 0)];
    }
  }
  projectConstraints(snapshot, predictor);
  for (let slot = 0; slot < nodeCount; slot += 1) {
    const leaf = backwardLeaves[slot]!, reverseLeaf = reverseLeaves[slot]!;
    if (leaf === INVALID || reverseLeaf === INVALID) { corrected[slot] = predictor[slot]!; continue; }
    const reversed = sampleLeafValues(snapshot, predictor, reversePoints[slot]!, reverseLeaf);
    const leafBase = 16 * leaf;
    const bounds = Array.from({ length: 8 }, (_, corner) =>
      prior[snapshot.leaves[leafBase + 8 + corner] ?? INVALID] ?? Number.NaN);
    const correction = predictor[slot]! + .5 * (prior[slot]! - reversed);
    corrected[slot] = Math.round(Math.min(Math.max(correction, Math.min(...bounds)),
      Math.max(...bounds)) * 65_536) / 65_536;
  }
  projectConstraints(snapshot, corrected);

  const priorGeometry = crossingGeometry(snapshot, prior);
  const predictorGeometry = crossingGeometry(snapshot, predictor);
  const correctedGeometry = crossingGeometry(snapshot, corrected);
  const currentGeometry = crossingGeometry(snapshot, current);
  const interfaceNodes = new Set<number>();
  const frontNodes = new Set<number>();
  const leafCount = Math.min(snapshot.graphControl[1] ?? 0, snapshot.leaves.length / 16);
  for (let leaf = 0; leaf < leafCount; leaf += 1) {
    const base = 16 * leaf;
    const values = Array.from({ length: 8 }, (_, corner) =>
      prior[snapshot.leaves[base + 8 + corner] ?? INVALID] ?? Number.NaN);
    if (Math.min(...values) > 0 || Math.max(...values) < 0) continue;
    for (let corner = 0; corner < 8; corner += 1) interfaceNodes.add(
      snapshot.leaves[base + 8 + corner] ?? INVALID);
    for (const [ca, cb] of EDGE_CORNERS.slice(0, 4)) {
      if (crossing(values[ca]!, values[cb]!) === undefined) continue;
      frontNodes.add(snapshot.leaves[base + 8 + ca] ?? INVALID);
      frontNodes.add(snapshot.leaves[base + 8 + cb] ?? INVALID);
    }
  }
  const byDonorSpan = new Map<number, { count: number; speed: number; tracedSpeed: number;
    xVelocity: number; tracedXVelocity: number; predictorDelta: number;
    correctionDelta: number; redistanceDelta: number }>();
  let frontTransportedNodes = 0, frontXVelocity = 0, frontTracedXVelocity = 0;
  const frontXStencilSpan = new Map<number, { count: number; x: number; tracedX: number }>();
  let frontFaceMaximum = 0, frontPositiveFaceMean = 0, frontPositiveFaceNodes = 0,
    frontZeroWeight = 0, frontTotalWeight = 0;
  let coarseDonorNodes = 0, transportedInterfaceNodes = 0;
  for (const slot of interfaceNodes) {
    const leaf = backwardLeaves[slot] ?? INVALID;
    if (leaf === INVALID) continue;
    transportedInterfaceNodes += 1;
    const span = snapshot.leaves[16 * leaf + 3] ?? 0;
    if (span > 1) coarseDonorNodes += 1;
    const raw = 8 * slot;
    const speed = Math.hypot(fromBits(snapshot.nodalVelocity[raw] ?? 0),
      fromBits(snapshot.nodalVelocity[raw + 1] ?? 0), fromBits(snapshot.nodalVelocity[raw + 2] ?? 0));
    const q = nodePosition(snapshot, slot), departure = backwardPoints[slot]!;
    const tracedSpeed = Math.hypot(q[0] - departure[0], q[1] - departure[1],
      q[2] - departure[2]) * fromBits(snapshot.renderer[7] ?? 0) / dt_s;
    const row = byDonorSpan.get(span) ?? { count: 0, speed: 0, tracedSpeed: 0,
      xVelocity: 0, tracedXVelocity: 0, predictorDelta: 0, correctionDelta: 0,
      redistanceDelta: 0 };
    const xVelocity = fromBits(snapshot.nodalVelocity[raw] ?? 0);
    const tracedXVelocity = (q[0] - departure[0]) * fromBits(snapshot.renderer[7] ?? 0) / dt_s;
    row.count += 1; row.speed += speed; row.tracedSpeed += tracedSpeed;
    row.xVelocity += xVelocity; row.tracedXVelocity += tracedXVelocity;
    row.predictorDelta += Math.abs(predictor[slot]! - prior[slot]!);
    row.correctionDelta += Math.abs(corrected[slot]! - predictor[slot]!);
    row.redistanceDelta += Math.abs(current[slot]! - corrected[slot]!);
    byDonorSpan.set(span, row);
    if (frontNodes.has(slot)) {
      frontTransportedNodes += 1; frontXVelocity += xVelocity;
      frontTracedXVelocity += tracedXVelocity;
      const stencilSpan = snapshot.stencils[36 * (3 * slot) + 2] ?? 0;
      const stencilRow = frontXStencilSpan.get(stencilSpan) ?? { count: 0, x: 0, tracedX: 0 };
      stencilRow.count += 1; stencilRow.x += xVelocity; stencilRow.tracedX += tracedXVelocity;
      frontXStencilSpan.set(stencilSpan, stencilRow);
      const stencilBase = 36 * (3 * slot), faceCount = snapshot.stencils[stencilBase] ?? 0;
      let positiveSum = 0, positiveWeight = 0, nodeMaximum = -Infinity;
      for (let face = 0; face < Math.min(16, faceCount); face += 1) {
        const faceSlot = snapshot.stencils[stencilBase + 4 + face] ?? INVALID;
        const weight = snapshot.stencils[stencilBase + 20 + face] ?? 0;
        const value = fromBits(snapshot.extendedFaceVelocity[faceSlot] ?? 0);
        if (!Number.isFinite(value) || weight === 0) continue;
        frontTotalWeight += weight; if (value === 0) frontZeroWeight += weight;
        nodeMaximum = Math.max(nodeMaximum, value);
        if (value > 0) { positiveSum += weight * value; positiveWeight += weight; }
      }
      if (Number.isFinite(nodeMaximum)) frontFaceMaximum += nodeMaximum;
      if (positiveWeight > 0) {
        frontPositiveFaceMean += positiveSum / positiveWeight; frontPositiveFaceNodes += 1;
      }
    }
  }
  return { currentBank, interfaceNodeCount: interfaceNodes.size, transportedInterfaceNodes,
    frontNodeCount: frontNodes.size, frontTransportedNodes,
    meanFrontNodalXVelocity_m_s: frontXVelocity / Math.max(1, frontTransportedNodes),
    meanFrontRk2XVelocity_m_s: frontTracedXVelocity / Math.max(1, frontTransportedNodes),
    frontXStencilSpan: Object.fromEntries([...frontXStencilSpan].sort(([a], [b]) => a - b)
      .map(([span, row]) => [span, { count: row.count,
        meanNodalXVelocity_m_s: row.x / row.count,
        meanRk2XVelocity_m_s: row.tracedX / row.count }])),
    frontXFaceSupport: { meanMaximumFaceVelocity_m_s:
      frontFaceMaximum / Math.max(1, frontTransportedNodes),
    meanPositiveFaceVelocity_m_s: frontPositiveFaceMean / Math.max(1, frontPositiveFaceNodes),
    positiveFaceNodeCount: frontPositiveFaceNodes,
    zeroWeightFraction: frontZeroWeight / Math.max(1, frontTotalWeight) },
    coarseDonorNodes, coarseDonorFraction: coarseDonorNodes / Math.max(1, transportedInterfaceNodes),
    byDonorSpan: Object.fromEntries([...byDonorSpan].sort(([a], [b]) => a - b).map(([span, row]) =>
      [span, { count: row.count, meanNodalSpeed_m_s: row.speed / row.count,
        meanRk2TracedSpeed_m_s: row.tracedSpeed / row.count,
        tracedToNodalSpeedRatio: row.tracedSpeed / Math.max(Number.EPSILON, row.speed),
        meanNodalXVelocity_m_s: row.xVelocity / row.count,
        meanRk2TracedXVelocity_m_s: row.tracedXVelocity / row.count,
        meanAbsPredictorDelta_m: row.predictorDelta / row.count,
        meanAbsMacCormackDelta_m: row.correctionDelta / row.count,
        meanAbsRedistanceDelta_m: row.redistanceDelta / row.count }])),
    geometry: { priorToPredictor: geometryDelta(priorGeometry, predictorGeometry),
      predictorToMacCormack: geometryDelta(predictorGeometry, correctedGeometry),
      macCormackToAccepted: geometryDelta(correctedGeometry, currentGeometry),
      priorToAccepted: geometryDelta(priorGeometry, currentGeometry) } };
}

function makeBuffer(device: GPUDevice, label: string, size: number, usage: GPUBufferUsageFlags,
  contents?: ArrayBufferView) {
  const result = device.createBuffer({ label, size: Math.max(4, Math.ceil(size / 4) * 4),
    usage, mappedAtCreation: contents !== undefined });
  if (contents) {
    new Uint8Array(result.getMappedRange()).set(new Uint8Array(contents.buffer,
      contents.byteOffset, contents.byteLength)); result.unmap();
  }
  return result;
}

async function readBuffer(device: GPUDevice, source: GPUBuffer, bytes: number) {
  const target = device.createBuffer({ label: "dam subcell readback", size: Math.max(4, bytes),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, target, 0, Math.max(4, bytes));
  device.queue.submit([encoder.finish()]); await target.mapAsync(GPUMapMode.READ);
  const copy = target.getMappedRange().slice(0, bytes); target.unmap(); target.destroy(); return copy;
}

async function emittedMesh(device: GPUDevice, snapshot: Snapshot,
  scalarCrossingKeys: ReadonlySet<string>) {
  const dimensions = snapshot.dimensions;
  const volumeCubeCount = dimensions[0] * dimensions[1] * dimensions[2];
  // One free-surface record plus the exact closed-face owner bound: both x
  // walls, the floor, and both z walls. Production has a much larger
  // vertex-derived worklist; keep this focused probe independently sound.
  const cubeCapacity = volumeCubeCount + 2 * dimensions[1] * dimensions[2]
    + dimensions[0] * dimensions[2] + 2 * dimensions[0] * dimensions[1];
  const generation = snapshot.renderer[1]! & 0x3fff_ffff;
  const physicalCellSize = fromBits(snapshot.renderer[7]!);
  const uniformWords = new Uint32Array(28); const uniformFloats = new Float32Array(uniformWords.buffer);
  uniformFloats.set([1, 1, 1, 1], 12);
  const paramsWords = new Uint32Array(28); const paramsFloats = new Float32Array(paramsWords.buffer);
  paramsWords.set([dimensions[0], dimensions[1], dimensions[2], 4], 0);
  paramsWords.set([1, 1, 1, 64], 4); paramsWords.set([1, 6, 1, generation], 8);
  paramsFloats.set([0, 0, 0, physicalCellSize], 12); paramsFloats.set([1, 0, 0, 0], 16);
  const argsInitial = Uint32Array.from([0, 1, 0, 0, 0, INVALID, 0, INVALID]);
  const uniforms = makeBuffer(device, "dam subcell uniforms", uniformWords.byteLength,
    GPUBufferUsage.UNIFORM, uniformWords);
  const params = makeBuffer(device, "dam subcell params", paramsWords.byteLength,
    GPUBufferUsage.UNIFORM, paramsWords);
  const args = makeBuffer(device, "dam subcell args", argsInitial.byteLength,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, argsInitial);
  const cubes = makeBuffer(device, "dam subcell cubes", cubeCapacity * 8,
    GPUBufferUsage.STORAGE);
  const values = makeBuffer(device, "dam subcell values", cubeCapacity * 32,
    GPUBufferUsage.STORAGE);
  const offsets = makeBuffer(device, "dam subcell offsets", cubeCapacity * 24,
    GPUBufferUsage.STORAGE);
  const empty = makeBuffer(device, "dam subcell empty", 256,
    GPUBufferUsage.STORAGE, new Uint32Array(64));
  const directory = makeBuffer(device, "dam accepted adaptive renderer", snapshot.renderer.byteLength,
    GPUBufferUsage.STORAGE, snapshot.renderer);
  const classify = device.createComputePipeline({ label: "dam production classify", layout: "auto",
    compute: { module: device.createShaderModule({ code: globalFineSurfaceClassificationShader }),
      entryPoint: "extractGlobalCoarseMain" } });
  const scan = device.createComputePipeline({ label: "dam production scan", layout: "auto",
    compute: { module: device.createShaderModule({ code: globalFineClassifiedIndirectScanShader }),
      entryPoint: "scanGlobalFineTriangles" } });
  const emit = device.createComputePipeline({ label: "dam production emit", layout: "auto",
    compute: { module: device.createShaderModule({ code: globalFineClassifiedEmitShader }),
      entryPoint: "emitGlobalFineTetrahedra" } });
  const classifyGroup = device.createBindGroup({ layout: classify.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: uniforms } }, { binding: 4, resource: { buffer: args } },
    { binding: 5, resource: { buffer: cubes } }, { binding: 6, resource: { buffer: values } },
    { binding: 8, resource: { buffer: empty } }, { binding: 9, resource: { buffer: empty } },
    { binding: 10, resource: { buffer: params } }, { binding: 12, resource: { buffer: empty } },
    { binding: 16, resource: { buffer: directory } }, { binding: 17, resource: { buffer: empty } },
  ] });
  let vertices = makeBuffer(device, "dam subcell count placeholder", 32,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const scanGroup = (buffer: GPUBuffer) => device.createBindGroup({ layout: scan.getBindGroupLayout(0), entries: [
    { binding: 3, resource: { buffer } }, { binding: 4, resource: { buffer: args } },
    { binding: 5, resource: { buffer: cubes } }, { binding: 6, resource: { buffer: values } },
    { binding: 7, resource: { buffer: offsets } }, { binding: 10, resource: { buffer: params } },
    { binding: 11, resource: { buffer: empty } },
  ] });
  let encoder = device.createCommandEncoder(); let pass = encoder.beginComputePass();
  pass.setPipeline(classify); pass.setBindGroup(0, classifyGroup);
  pass.dispatchWorkgroups(Math.ceil(volumeCubeCount / 256));
  pass.setPipeline(scan); pass.setBindGroup(0, scanGroup(vertices)); pass.dispatchWorkgroups(1);
  pass.end(); device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
  const countArgs = new Uint32Array(await readBuffer(device, args, 32));
  const vertexCount = countArgs[5]!; assert.ok(vertexCount > 0 && vertexCount <= cubeCapacity * 36);
  vertices.destroy(); vertices = makeBuffer(device, "dam emitted surface", vertexCount * 32,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const emitGroup = device.createBindGroup({ layout: emit.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: uniforms } }, { binding: 3, resource: { buffer: vertices } },
    { binding: 4, resource: { buffer: args } }, { binding: 5, resource: { buffer: cubes } },
    { binding: 6, resource: { buffer: values } }, { binding: 7, resource: { buffer: offsets } },
    { binding: 8, resource: { buffer: empty } }, { binding: 9, resource: { buffer: empty } },
    { binding: 10, resource: { buffer: params } }, { binding: 12, resource: { buffer: empty } },
    { binding: 16, resource: { buffer: directory } },
  ] });
  encoder = device.createCommandEncoder(); pass = encoder.beginComputePass();
  pass.setPipeline(scan); pass.setBindGroup(0, scanGroup(vertices)); pass.dispatchWorkgroups(1);
  pass.setPipeline(emit); pass.setBindGroup(0, emitGroup);
  pass.dispatchWorkgroups(Math.ceil(countArgs[4]! / 64), 6, 1);
  pass.end(); device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
  const finalArgs = new Uint32Array(await readBuffer(device, args, 32));
  assert.equal(finalArgs[0], vertexCount); assert.equal(finalArgs[5], vertexCount);
  const mesh = new Float32Array(await readBuffer(device, vertices, vertexCount * 32));
  const points: Point[] = []; const normals: Point[] = [];
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    points.push([
    // `clipped` adds 0.5 for the native nodal descriptor before `world`.
    // Remove that render-space shift to report accepted scalar-lattice coordinates.
    mesh[8 * vertex]! * dimensions[0] + .5 * dimensions[0],
    mesh[8 * vertex + 1]! * dimensions[1],
    mesh[8 * vertex + 2]! * dimensions[2] + .5 * dimensions[2],
    ]);
    normals.push([mesh[8 * vertex + 4]!, mesh[8 * vertex + 5]!, mesh[8 * vertex + 6]!]);
  }
  for (const resource of [uniforms, params, args, cubes, values, offsets, empty, directory, vertices]) {
    resource.destroy();
  }
  const unique = new Map(points.map(point => [quantizedPointKey(point), point]));
  const scalarCrossingVertexCount = [...unique.keys()].filter(key => scalarCrossingKeys.has(key)).length;
  const normalKey = (normal: Point) => normal.map(value => value.toFixed(7)).join(",");
  let flatNormalTriangles = 0;
  for (let vertex = 0; vertex < vertexCount; vertex += 3) {
    const a = normalKey(normals[vertex]!);
    if (a === normalKey(normals[vertex + 1]!) && a === normalKey(normals[vertex + 2]!)) {
      flatNormalTriangles += 1;
    }
  }
  return { activeCubeCount: finalArgs[4], vertexCount, vertices: pointDistribution(points),
    uniqueScalarCrossingVertexCount: scalarCrossingVertexCount,
    uniqueFanCentreVertexCount: unique.size - scalarCrossingVertexCount,
    normalFiltering: {
      uniqueNormalCount: new Set(normals.map(normalKey)).size,
      flatNormalTriangles,
      varyingNormalTriangles: vertexCount / 3 - flatNormalTriangles,
    } };
}

const modulePath = process.env.WEBGPU_NODE_MODULE
  ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
const { create, globals } = await import(pathToFileURL(modulePath).href) as {
  create(options: string[]): GPU; globals: Record<string, unknown>;
};
Object.assign(globalThis, globals);
const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" }); assert.ok(adapter);
const device = await adapter.requestDevice({ requiredFeatures: adapter.features.has("subgroups")
  ? ["subgroups"] : [], requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
const validationErrors: string[] = [];
device.addEventListener("uncapturederror", event => validationErrors.push(event.error.message));

const scene = getScenePreset("water-box-dam-break").create();
const fixedDt_s = Number(process.env.FLUID_MAX_DT ?? .008);
assert.ok(Number.isFinite(fixedDt_s) && fixedDt_s > 0);
scene.numerics.fixedDt_s = fixedDt_s; scene.numerics.maxDt_s = fixedDt_s;
const solver = await octreeMethod.createSolverAsync!(device, scene, "balanced", {
  ...octreeMethod.presetFor("balanced"), coarseBackend: "losasso",
  losassoVelocityExtension: "causal-front",
  maximumLeafSize: process.env.FLUID_MAXIMUM_LEAF_SIZE ?? "16",
  interfaceRefinementBandCells: 4, globalFineLevelSetFactor: "1", secondaryParticles: "off",
}, undefined, () => {}) as GPUSolverInstance;
const projection = (solver as unknown as { octreeProjection?: {
  readAdaptiveSurfacePublicationDiagnostics(): Promise<Snapshot | undefined>;
  losassoBackend?: { adaptivePhiSource?: { transportDepartures: GPUBuffer } };
} }).octreeProjection; assert.ok(projection);
const checkpoints = (process.env.FLUID_SAMPLE_TIMES_S ?? "0.128,0.248,0.496")
  .split(",").map(value => Number(value.trim()))
  .filter(value => Number.isFinite(value) && value > 0)
  .sort((a, b) => a - b);
assert.ok(checkpoints.length > 0);
const samples: unknown[] = [];
let acceptedStep = 0;
for (const time_s of checkpoints) {
  const wantedStep = Math.round(time_s / fixedDt_s);
  while (acceptedStep < wantedStep) {
    acceptedStep += 1;
    while (!solver.advanceTo(acceptedStep * fixedDt_s, [])) {
      await new Promise(resolve => setImmediate(resolve));
    }
  }
  await device.queue.onSubmittedWorkDone(); const snapshot = await projection.readAdaptiveSurfacePublicationDiagnostics();
  assert.ok(snapshot); const scalar = acceptedScalarCrossings(snapshot);
  const departureSource = projection.losassoBackend?.adaptivePhiSource?.transportDepartures;
  assert.ok(departureSource);
  const departureWords = new Uint32Array(await readBuffer(device, departureSource,
    departureSource.size));
  const transport = adaptiveTransportAudit(snapshot, departureWords, scene.numerics.fixedDt_s);
  const freezeAudit = mixedCornerFreezeAudit(snapshot);
  const mesh = await emittedMesh(device, snapshot, scalar.productionQuantizedFinePointKeys);
  const allCrossingLeavesUnit = Object.keys(scalar.crossingLeafSpanHistogram)
    .every(span => span === "1");
  assert.ok(scalar.reconstructedUnitEdges.anyCoordinateOffGridCount > 0,
    `accepted scalar must retain sub-cell crossings at ${time_s}s`);
  assert.ok(mesh.vertices.anyCoordinateOffGridCount > 0,
    `production contour must retain sub-cell vertices at ${time_s}s`);
  assert.ok(mesh.normalFiltering.varyingNormalTriangles > 0,
    `adaptive renderer must filter the accepted SDF normal across cells at ${time_s}s`);
  assert.deepEqual([freezeAudit.missingFreezeBits, freezeAudit.distanceMarkerMismatches,
    freezeAudit.masterClosureMismatches], [0, 0, 0],
  `mixed-corner freeze closure must be exact at ${time_s}s`);
  const { productionQuantizedFinePointKeys: _keys, ...scalarReport } = scalar;
  const phiReceipt = { transportedNodes: snapshot.phiReceipts[4] ?? 0,
    scheduledIndependentNodes: snapshot.phiReceipts[32] ?? 0,
    retainedIndependentNodes: snapshot.phiReceipts[33] ?? 0,
    scheduledConstrainedNodes: snapshot.phiReceipts[34] ?? 0,
    retainedConstrainedNodes: snapshot.phiReceipts[35] ?? 0,
    redistanceActiveIndependentNodes: snapshot.phiReceipts[52] ?? 0,
    redistanceActiveConstrainedNodes: snapshot.phiReceipts[53] ?? 0 };
  samples.push({ time_s, topologyEpoch: snapshot.graphControl[0], generation: snapshot.graphControl[5],
    graphNodeCount: snapshot.graphControl[2], allCrossingLeavesUnit, phiReceipt,
    scalar: scalarReport, freezeAudit, transport, emittedMesh: mesh });
}
solver.destroy(); await device.queue.onSubmittedWorkDone(); device.destroy();
console.log(JSON.stringify({ phase: "dam-subcell-surface", dimensions: [solver.info.nx, solver.info.ny,
  solver.info.nz], validationErrors, samples }, null, 2));
assert.deepEqual(validationErrors, []);
