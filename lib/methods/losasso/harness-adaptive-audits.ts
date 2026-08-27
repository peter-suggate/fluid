import type { HarnessRasterReceipt } from "../../core/method-harness-contract";
import {
  OCTREE_LOSASSO_ADAPTIVE_VELOCITY_DIAGNOSTIC_BANK_WORDS,
  OCTREE_LOSASSO_ADAPTIVE_VELOCITY_DIAGNOSTIC_HEADER_WORDS,
  OCTREE_LOSASSO_ADAPTIVE_VELOCITY_DIAGNOSTIC_RECORDS,
  OCTREE_LOSASSO_ADAPTIVE_VELOCITY_DIAGNOSTIC_WORDS,
} from "./webgpu-octree-losasso-adaptive-velocity";

/**
 * The Losasso lane's own audits of its adaptive surface publication.
 *
 * These lived in the Dawn smoke executor, which made a generic run loop the
 * owner of one method's D4 equivariance oracles, its redistance-unresolved
 * forensics and its velocity-stencil reconstruction — 772 lines that read
 * buffers no other method builds, against ABI constants imported straight out
 * of this package. Nothing here is reachable from the browser: the harness
 * plugin that calls them is itself loaded on demand.
 */

export function losassoRasterCutoverFailures(
  raster: HarnessRasterReceipt,
  fineGeneration: number,
): readonly string[] {
  const failures: string[] = [];
  const reverse = raster.reverseView;
  if (raster.frontInterfacePixels === 0 || raster.backInterfacePixels === 0
    || raster.pairedInterfacePixels === 0 || raster.backOnlyInterfacePixels !== 0) {
    failures.push("forward raster lacks closed paired front/back interface coverage");
  }
  if (!reverse || reverse.frontInterfacePixels === 0 || reverse.backInterfacePixels === 0
    || reverse.pairedInterfacePixels === 0 || reverse.backOnlyInterfacePixels !== 0) {
    failures.push("reverse raster lacks closed paired front/back interface coverage");
  }
  const holeCount = raster.enclosedSurfaceHoles.front.count
    + raster.enclosedSurfaceHoles.back.count
    + (reverse?.enclosedSurfaceHoles.front.count ?? 0)
    + (reverse?.enclosedSurfaceHoles.back.count ?? 0);
  const slitCount = raster.narrowVerticalSlits.count
    + (reverse?.narrowVerticalSlits.count ?? 0);
  if (holeCount !== 0) failures.push(`raster contains ${holeCount} enclosed surface holes`);
  if (slitCount !== 0) failures.push(`raster contains ${slitCount} narrow vertical slits`);
  if (raster.rendererValidationErrorCount !== 0 || raster.rendererUncapturedErrorCount !== 0) {
    failures.push("renderer reported validation or uncaptured GPU errors");
  }
  if (raster.surfaceGeometrySource !== "global-fine-coarse"
    || raster.globalFineCrossingPublished !== true
    || raster.presentationFallbackActive !== false
    || (raster.globalFineAuthorityLatch ?? 0) === 0) {
    failures.push("renderer did not consume the current production global-fine/coarse geometry");
  }
  if ((raster.vertexCount ?? 0) === 0 || (raster.activeCubeCount ?? 0) === 0
    || raster.vertexAllocator !== raster.vertexCount
    || (raster.vertexCount ?? Infinity) > (raster.vertexCapacity ?? -1)
    || (raster.activeCubeCount ?? Infinity) > (raster.activeCubeCapacity ?? -1)) {
    failures.push("production surface geometry is empty or its allocation receipt is invalid");
  }
  const retained = raster.globalFineAuthorityTransition;
  if (!retained || retained.cleanFineCoarseRequired !== true
    || retained.validGeneration !== fineGeneration
    || retained.unpublishedGeneration !== fineGeneration + 1
    || retained.retainedGeometrySource !== "retained-previous"
    || retained.retainedFrontInterfacePixels !== raster.frontInterfacePixels
    || retained.retainedBackInterfacePixels !== raster.backInterfacePixels
    || retained.retainedFrontInterfaceHash !== raster.frontInterfaceHash
    || retained.retainedBackInterfaceHash !== raster.backInterfaceHash) {
    failures.push("unpublished-generation probe did not coherently retain the accepted raster");
  }
  return Object.freeze(failures);
}

export function losassoAdaptiveRasterCutoverFailures(
  raster: HarnessRasterReceipt,
  surfaceGeneration: number,
): readonly string[] {
  const failures: string[] = [];
  const reverse = raster.reverseView;
  if (raster.frontInterfacePixels === 0 || raster.backInterfacePixels === 0
    || raster.pairedInterfacePixels === 0 || raster.backOnlyInterfacePixels !== 0) {
    failures.push("forward adaptive raster lacks closed paired front/back interface coverage");
  }
  if (!reverse || reverse.frontInterfacePixels === 0 || reverse.backInterfacePixels === 0
    || reverse.pairedInterfacePixels === 0 || reverse.backOnlyInterfacePixels !== 0) {
    failures.push("reverse adaptive raster lacks closed paired front/back interface coverage");
  }
  const holeCount = raster.enclosedSurfaceHoles.front.count
    + raster.enclosedSurfaceHoles.back.count
    + (reverse?.enclosedSurfaceHoles.front.count ?? 0)
    + (reverse?.enclosedSurfaceHoles.back.count ?? 0);
  const slitCount = raster.narrowVerticalSlits.count
    + (reverse?.narrowVerticalSlits.count ?? 0);
  if (holeCount !== 0) failures.push(`adaptive raster contains ${holeCount} enclosed surface holes`);
  if (slitCount !== 0) failures.push(`adaptive raster contains ${slitCount} narrow vertical slits`);
  if (raster.rendererValidationErrorCount !== 0 || raster.rendererUncapturedErrorCount !== 0) {
    failures.push("adaptive renderer reported validation or uncaptured GPU errors");
  }
  if (raster.surfaceGeometrySource !== "compact-coarse"
    || raster.presentationFallbackActive !== false
    || raster.meshPublicationGeneration !== surfaceGeneration) {
    failures.push("renderer did not consume the current adaptive scalar publication");
  }
  if ((raster.vertexCount ?? 0) === 0 || (raster.activeCubeCount ?? 0) === 0
    || raster.vertexAllocator !== raster.vertexCount
    || (raster.vertexCount ?? Infinity) > (raster.vertexCapacity ?? -1)
    || (raster.activeCubeCount ?? Infinity) > (raster.activeCubeCapacity ?? -1)) {
    failures.push("adaptive production surface geometry is empty or its allocation receipt is invalid");
  }
  if (process.env.FLUID_RASTER_MESH_SYMMETRY === "1") {
    const mesh = raster.surfaceMeshSymmetry;
    const cubes = raster.activeCubeSymmetry;
    const sharp = raster.sharpPatchRaster;
    if (!mesh || mesh.exactPositionMismatchCount !== 0 || mesh.nonFiniteCount !== 0) {
      failures.push("adaptive production surface mesh is not exactly D4-symmetric in position");
    }
    if (!cubes || Object.values(cubes.transforms).some(
      (transform) => transform.exactIdentityMismatchCount !== 0)) {
      failures.push("adaptive production classified cubes are not exactly D4-symmetric");
    }
    if (!sharp || sharp.invalidPatchCount !== 0) {
      failures.push("adaptive production surface has invalid sharp patches");
    }
  }
  return Object.freeze(failures);
}

export type AdaptiveD4Transform = "reflect-x" | "reflect-z" | "swap-xz";
export interface AdaptiveSurfacePublicationSnapshot {
  readonly authorityControl: Uint32Array;
  readonly faceGeometry: Uint32Array;
  readonly pressureFaces: Uint32Array;
  readonly pressureRowToGraphLeaf: Uint32Array;
  readonly projectedFaceVelocity: Uint32Array;
  readonly predictedFaceVelocity: Uint32Array;
  readonly advectedFaceVelocity: Uint32Array;
  readonly extendedFaceVelocity: Uint32Array;
  readonly stencilControl: Uint32Array;
  readonly stencils: Uint32Array;
  readonly candidateAuthorityControl: Uint32Array;
  readonly candidateFaceGeometry: Uint32Array;
  readonly candidateExtendedFaceVelocity: Uint32Array;
  readonly candidateNodalVelocity: Uint32Array;
  readonly graphControl: Uint32Array;
  readonly phiControl: Uint32Array;
  readonly leaves: Uint32Array;
  readonly nodalPhi: Uint32Array;
  readonly nodes: Uint32Array;
  readonly constraints: Uint32Array;
  readonly adjacency: Uint32Array;
  readonly nodalVelocity: Uint32Array;
  readonly nodeValidity: Uint32Array;
  readonly renderer: Uint32Array;
  readonly transportBandMask: Uint32Array;
  readonly redistanceDistanceA: Float32Array;
  readonly redistanceDistanceB: Float32Array;
  readonly dimensions: readonly [number, number, number];
}

export function adaptivePhiRedistanceUnresolvedAudit(snapshot: AdaptiveSurfacePublicationSnapshot) {
  const { graphControl, phiControl, nodalPhi, nodes, constraints, adjacency,
    transportBandMask, redistanceDistanceA, redistanceDistanceB, dimensions } = snapshot;
  const nodeCount = Math.min(graphControl[2] ?? 0, transportBandMask.length,
    redistanceDistanceA.length, redistanceDistanceB.length);
  const nextBank = 1 - ((phiControl[6] ?? 0) & 1);
  const dx = dimensions[0] + 1, dy = dimensions[1] + 1;
  const position = (item: number) => [item % dx, Math.floor(item / dx) % dy,
    Math.floor(item / (dx * dy))] as const;
  const unresolved = [];
  for (let slot = 0; slot < nodeCount && unresolved.length < 16; slot += 1) {
    const distance = redistanceDistanceA[slot]!;
    if (transportBandMask[slot] === 0 || (Number.isFinite(distance) && distance < 3.402823e38)) continue;
    const item = nodes[4 * slot] ?? 0xffff_ffff;
    unresolved.push(Object.freeze({ slot, item, position: position(item),
      constraintCount: constraints[12 * slot + 1] ?? 0,
      phi: new Float32Array(Uint32Array.of(nodalPhi[2 * slot + nextBank] ?? 0).buffer)[0],
      distanceA: redistanceDistanceA[slot], distanceB: redistanceDistanceB[slot],
      neighbors: Object.freeze(Array.from({ length: 6 }, (_none, direction) => {
        const neighbor = adjacency[12 * slot + direction] ?? 0xffff_ffff;
        if (neighbor === 0xffff_ffff || neighbor >= nodeCount) return Object.freeze({ direction, slot: neighbor });
        const neighborItem = nodes[4 * neighbor] ?? 0xffff_ffff;
        return Object.freeze({ direction, slot: neighbor, item: neighborItem,
          position: position(neighborItem), mask: transportBandMask[neighbor] ?? 0,
          constraintCount: constraints[12 * neighbor + 1] ?? 0,
          phi: new Float32Array(Uint32Array.of(nodalPhi[2 * neighbor + nextBank] ?? 0).buffer)[0],
          distanceA: redistanceDistanceA[neighbor], distanceB: redistanceDistanceB[neighbor] });
      })),
    }));
  }
  return Object.freeze({ nodeCount, nextBank, unresolved: Object.freeze(unresolved) });
}

export function decodeAdaptiveVelocityGPUFailureDiagnostics(words: readonly number[],
  dimensions: readonly [number, number, number]) {
  const bits = new Uint32Array(1);
  const float = (word: number) => {
    bits[0] = word >>> 0;
    return new Float32Array(bits.buffer)[0]!;
  };
  const position = (item: number) => {
    if (item === 0xffff_ffff) return undefined;
    const dx = dimensions[0] + 1, dy = dimensions[1] + 1;
    return [item % dx, Math.floor(item / dx) % dy,
      Math.floor(item / (dx * dy))] as const;
  };
  const names = ["accepted", "predictor", "candidate", "candidatePredictor"] as const;
  return Object.freeze(names.map((name, bank) => {
    const base = bank * OCTREE_LOSASSO_ADAPTIVE_VELOCITY_DIAGNOSTIC_BANK_WORDS;
    const captured = Math.min(words[base + 6] ?? 0,
      OCTREE_LOSASSO_ADAPTIVE_VELOCITY_DIAGNOSTIC_RECORDS);
    const records = Array.from({ length: captured }, (_unused, record) => {
      const at = base + OCTREE_LOSASSO_ADAPTIVE_VELOCITY_DIAGNOSTIC_HEADER_WORDS
        + record * OCTREE_LOSASSO_ADAPTIVE_VELOCITY_DIAGNOSTIC_WORDS;
      const causeBits = words[at + 5] ?? 0;
      const item = words[at + 1] ?? 0xffff_ffff;
      return Object.freeze({ node: words[at] ?? 0xffff_ffff, item,
        position: position(item), missingComponents: words[at + 2] ?? 0,
        transportBandMask: words[at + 3] ?? 0,
        phi: float(words[at + 4] ?? 0), causeBits,
        causes: Object.freeze([
          ...(causeBits & 1 ? ["no-adjacency"] : []),
          ...(causeBits & 2 ? ["no-nonfar-adjacency"] : []),
          ...(causeBits & 4 ? ["nonfar-neighbors-miss-components"] : []),
          ...(causeBits & 8 ? ["constrained"] : []),
          ...(causeBits & 16 ? ["invalid-masters"] : []),
        ]),
        neighbors: Object.freeze(Array.from({ length: 6 }, (_none, direction) => ({
          direction, slot: words[at + 6 + direction] ?? 0xffff_ffff,
          phi: (words[at + 12 + direction] ?? 0xffff_ffff) === 0xffff_ffff
            ? undefined : float(words[at + 12 + direction]!),
          mask: (words[at + 18 + direction] ?? 0) & 7,
          transportBand: ((words[at + 18 + direction] ?? 0) & 0x100) !== 0,
        }))),
      });
    });
    const invalidOutsideReach = words[base + 9] ?? 0;
    const invalidOutsideIndependent = words[base + 12] ?? 0;
    return Object.freeze({ name, header: Object.freeze({
      unresolved: words[base] ?? 0,
      noAdjacency: words[base + 1] ?? 0,
      noNonfarAdjacency: words[base + 2] ?? 0,
      nonfarNeighborsMissComponents: words[base + 3] ?? 0,
      constrained: words[base + 4] ?? 0,
      invalidMasters: words[base + 5] ?? 0,
      captured, demandedMasters: words[base + 7] ?? 0,
      dilatedSupport: words[base + 8] ?? 0,
      invalidOutsideReach,
      minimumInvalidOutsideAbsPhi: (words[base + 10] ?? 0xffff_ffff) === 0xffff_ffff
        ? undefined : float(words[base + 10]!),
      maximumInvalidOutsideAbsPhi: float(words[base + 11] ?? 0),
      invalidOutsideIndependent,
      invalidOutsideConstrained: invalidOutsideReach - invalidOutsideIndependent,
    }), records: Object.freeze(records.slice(0, 12)) });
  }));
}

export function adaptiveVelocityUnresolvedNodeAudit(snapshot: AdaptiveSurfacePublicationSnapshot,
  velocityReceipts: readonly number[]) {
  const { graphControl, phiControl, nodalPhi, nodes, constraints, adjacency,
    nodalVelocity, nodeValidity, dimensions } = snapshot;
  const nodeCount = Math.min(graphControl[2] ?? 0, Math.floor(nodes.length / 4),
    Math.floor(constraints.length / 12), Math.floor(adjacency.length / 12),
    Math.floor(nodalPhi.length / 2), Math.floor(nodalVelocity.length / 8),
    nodeValidity.length);
  const bits = new Uint32Array(1);
  const float = (word: number) => {
    bits[0] = word >>> 0;
    return new Float32Array(bits.buffer)[0]!;
  };
  const reach_m = float(velocityReceipts[1] ?? 0);
  const phiBank = (phiControl[6] ?? 0) & 1;
  const nodePosition = (node: number) => {
    const item = nodes[4 * node] ?? 0;
    const dx = dimensions[0] + 1, dy = dimensions[1] + 1;
    return [item % dx, Math.floor(item / dx) % dy, Math.floor(item / (dx * dy))] as const;
  };
  const phi = (node: number) => float(nodalPhi[2 * node + phiBank] ?? 0);
  const recordMask = (node: number, bank: number) => nodalVelocity[8 * node + 4 * bank + 3]! & 7;
  const validityMask = (node: number, bank: number) => nodeValidity[node]! >>> (4 * bank) & 7;
  const result = [] as Array<Readonly<Record<string, unknown>>>;
  for (const bank of [0, 1] as const) {
    let unresolvedWithinReach = 0, unresolvedIndependent = 0, unresolvedConstrained = 0;
    let noComponentDonorEdges = 0, onlyIncreasingDonorEdges = 0;
    let monotoneDonorEdgesButUnresolved = 0, maskReceiptMismatches = 0;
    const first: Array<Readonly<Record<string, unknown>>> = [];
    for (let node = 0; node < nodeCount; node += 1) {
      const mask = recordMask(node, bank), validity = validityMask(node, bank);
      if (mask !== validity) maskReceiptMismatches += 1;
      const nodePhi = phi(node);
      if (!(Math.abs(nodePhi) <= reach_m) || mask === 7) continue;
      unresolvedWithinReach += 1;
      const constraintBase = 12 * node;
      const constraintCount = constraints[constraintBase + 1] ?? 0;
      if (constraintCount === 0) unresolvedIndependent += 1;
      else unresolvedConstrained += 1;
      const missingComponents = [] as Array<Readonly<Record<string, unknown>>>;
      for (let component = 0; component < 3; component += 1) {
        if ((mask & 1 << component) !== 0) continue;
        let componentDonors = 0, monotoneDonors = 0, plateauDonors = 0;
        for (let direction = 0; direction < 6; direction += 1) {
          const neighbor = adjacency[constraintBase + direction] ?? 0xffff_ffff;
          if (neighbor >= nodeCount || (recordMask(neighbor, bank) & 1 << component) === 0) continue;
          componentDonors += 1;
          const neighborPhi = phi(neighbor);
          if (Math.abs(neighborPhi) <= Math.abs(nodePhi)) {
            monotoneDonors += 1;
            if (Math.abs(neighborPhi) === Math.abs(nodePhi)) plateauDonors += 1;
          }
        }
        if (componentDonors === 0) noComponentDonorEdges += 1;
        else if (monotoneDonors === 0) onlyIncreasingDonorEdges += 1;
        else monotoneDonorEdgesButUnresolved += 1;
        missingComponents.push(Object.freeze({ component, componentDonors,
          monotoneDonors, plateauDonors }));
      }
      if (first.length < 12) {
        const neighbors = Array.from({ length: 6 }, (_unused, direction) => {
          const neighbor = adjacency[constraintBase + direction] ?? 0xffff_ffff;
          if (neighbor >= nodeCount) return Object.freeze({ direction, slot: neighbor });
          return Object.freeze({ direction, slot: neighbor, position: nodePosition(neighbor),
            distance_m: float(adjacency[constraintBase + 6 + direction] ?? 0),
            phi: phi(neighbor), recordMask: recordMask(neighbor, bank),
            validityMask: validityMask(neighbor, bank) });
        });
        first.push(Object.freeze({ slot: node, position: nodePosition(node), phi: nodePhi,
          recordMask: mask, validityMask: validity,
          constraint: {
            kind: constraints[constraintBase] ?? 0, count: constraintCount,
            denominator: constraints[constraintBase + 2] ?? 0,
            masters: Array.from(constraints.slice(constraintBase + 4,
              constraintBase + 4 + Math.min(4, constraintCount))),
            numerators: Array.from(constraints.slice(constraintBase + 8,
              constraintBase + 8 + Math.min(4, constraintCount))),
          }, missingComponents, neighbors }));
      }
    }
    result.push(Object.freeze({ bank, generation: velocityReceipts[12 * bank] ?? 0,
      receiptErrors: velocityReceipts[12 * bank + 2] ?? 0,
      receiptUnresolved: velocityReceipts[12 * bank + 5] ?? 0,
      receiptValid: velocityReceipts[12 * bank + 7] ?? 0,
      reach_m, unresolvedWithinReach, unresolvedIndependent, unresolvedConstrained,
      noComponentDonorEdges, onlyIncreasingDonorEdges, monotoneDonorEdgesButUnresolved,
      maskReceiptMismatches, first: Object.freeze(first) }));
  }
  return Object.freeze({ nodeCount, phiBank, fields: Object.freeze(result) });
}

export function adaptiveSurfacePublicationD4Audit(snapshot: AdaptiveSurfacePublicationSnapshot) {
  const { graphControl, phiControl, leaves, nodalPhi, renderer, dimensions } = snapshot;
  const leafCount = Math.min(graphControl[1] ?? 0, Math.floor(leaves.length / 16));
  const nodeCount = Math.min(graphControl[2] ?? 0, Math.floor(nodalPhi.length / 2));
  const rendererRows = Math.min(renderer[2] ?? 0,
    Math.floor(Math.max(0, renderer.length - 8) / 16));
  const bank = (phiControl[6] ?? 0) & 1;
  const float = (word: number) => {
    const bits = new Uint32Array(1); bits[0] = word;
    return new Float32Array(bits.buffer)[0]!;
  };
  const identity = (x: number, y: number, z: number, span: number) =>
    `${x}:${y}:${z}:${span}`;
  const transforms: Readonly<Record<AdaptiveD4Transform, {
    origin(x: number, y: number, z: number, span: number): readonly [number, number, number];
    corner(corner: number): number;
  }>> = {
    "reflect-x": {
      origin: (x, y, z, span) => [dimensions[0] - x - span, y, z],
      corner: (corner) => corner ^ 1,
    },
    "reflect-z": {
      origin: (x, y, z, span) => [x, y, dimensions[2] - z - span],
      corner: (corner) => corner ^ 4,
    },
    "swap-xz": {
      origin: (x, y, z) => [z, y, x],
      corner: (corner) => (corner & 2) | ((corner & 1) << 2) | ((corner & 4) >> 2),
    },
  };
  const graphByIdentity = new Map<string, number>();
  for (let leaf = 0; leaf < leafCount; leaf += 1) {
    const base = 16 * leaf;
    graphByIdentity.set(identity(leaves[base]!, leaves[base + 1]!, leaves[base + 2]!,
      leaves[base + 3]!), leaf);
  }
  const rendererByIdentity = new Map<string, number>();
  for (let row = 0; row < rendererRows; row += 1) {
    const base = 8 + 8 * row, cell = (renderer[base] ?? 1) - 1;
    const x = cell % dimensions[0];
    const y = Math.floor(cell / dimensions[0]) % dimensions[1];
    const z = Math.floor(cell / (dimensions[0] * dimensions[1]));
    rendererByIdentity.set(identity(x, y, z, renderer[base + 1] ?? 0), row);
  }
  const result = {} as Record<AdaptiveD4Transform, {
    graphLeafIdentityMismatches: number;
    nodalCornerMismatches: number;
    nodalCornerMaximumAbsoluteError: number;
    rendererLeafIdentityMismatches: number;
    rendererPrimaryMismatches: number;
    rendererPrimaryMaximumAbsoluteError: number;
    rendererAuxCornerMismatches: number;
    rendererAuxMaximumAbsoluteError: number;
    firstNodalMismatch?: Record<string, unknown>;
    firstRendererPrimaryMismatch?: Record<string, unknown>;
    firstRendererAuxMismatch?: Record<string, unknown>;
  }>;
  let graphLeafIdentityMismatches = 0, nodalCornerMismatches = 0;
  let rendererLeafIdentityMismatches = 0, rendererPrimaryMismatches = 0;
  let rendererAuxCornerMismatches = 0, publicationCornerMismatches = 0;
  let nodalCornerMaximumAbsoluteError = 0, rendererPrimaryMaximumAbsoluteError = 0;
  let rendererAuxMaximumAbsoluteError = 0, publicationCornerMaximumAbsoluteError = 0;
  for (const [name, transform] of Object.entries(transforms) as [AdaptiveD4Transform,
    (typeof transforms)[AdaptiveD4Transform]][]) {
    const metrics = {
      graphLeafIdentityMismatches: 0, nodalCornerMismatches: 0,
      nodalCornerMaximumAbsoluteError: 0, rendererLeafIdentityMismatches: 0,
      rendererPrimaryMismatches: 0, rendererPrimaryMaximumAbsoluteError: 0,
      rendererAuxCornerMismatches: 0, rendererAuxMaximumAbsoluteError: 0,
    } as (typeof result)[AdaptiveD4Transform];
    for (let leaf = 0; leaf < leafCount; leaf += 1) {
      const base = 16 * leaf;
      const x = leaves[base]!, y = leaves[base + 1]!, z = leaves[base + 2]!;
      const span = leaves[base + 3]!;
      const targetOrigin = transform.origin(x, y, z, span);
      const targetLeaf = graphByIdentity.get(identity(...targetOrigin, span));
      if (targetLeaf === undefined) {
        metrics.graphLeafIdentityMismatches += 1;
        continue;
      }
      const targetBase = 16 * targetLeaf;
      for (let corner = 0; corner < 8; corner += 1) {
        const sourceSlot = leaves[base + 8 + corner] ?? 0xffff_ffff;
        const targetCorner = transform.corner(corner);
        const targetSlot = leaves[targetBase + 8 + targetCorner] ?? 0xffff_ffff;
        const sourceWord = sourceSlot < nodeCount ? nodalPhi[2 * sourceSlot + bank] : undefined;
        const targetWord = targetSlot < nodeCount ? nodalPhi[2 * targetSlot + bank] : undefined;
        if (sourceWord === undefined || targetWord === undefined || sourceWord !== targetWord) {
          metrics.nodalCornerMismatches += 1;
          const error = sourceWord === undefined || targetWord === undefined
            ? Number.NaN : Math.abs(float(sourceWord) - float(targetWord));
          if (Number.isFinite(error)) {
            metrics.nodalCornerMaximumAbsoluteError = Math.max(
              metrics.nodalCornerMaximumAbsoluteError, error);
          }
          metrics.firstNodalMismatch ??= { source: [x, y, z, span, corner, sourceSlot],
            target: [...targetOrigin, span, targetCorner, targetSlot],
            sourceValue: sourceWord === undefined ? undefined : float(sourceWord),
            targetValue: targetWord === undefined ? undefined : float(targetWord), error };
        }
      }
    }
    for (let row = 0; row < rendererRows; row += 1) {
      const base = 8 + 8 * row, cell = (renderer[base] ?? 1) - 1;
      const x = cell % dimensions[0];
      const y = Math.floor(cell / dimensions[0]) % dimensions[1];
      const z = Math.floor(cell / (dimensions[0] * dimensions[1]));
      const span = renderer[base + 1] ?? 0;
      const targetOrigin = transform.origin(x, y, z, span);
      const targetRow = rendererByIdentity.get(identity(...targetOrigin, span));
      if (targetRow === undefined) {
        metrics.rendererLeafIdentityMismatches += 1;
        continue;
      }
      const targetBase = 8 + 8 * targetRow;
      let primaryMismatch = false;
      for (const field of [2, 3, 4, 5, 7] as const) {
        const sourceWord = renderer[base + field]!;
        const targetWord = renderer[targetBase + field]!;
        if (sourceWord !== targetWord) {
          primaryMismatch = true;
          metrics.firstRendererPrimaryMismatch ??= { source: [x, y, z, span, field],
            target: [...targetOrigin, span, field], sourceWord, targetWord,
            sourceValue: field === 5 ? undefined : float(sourceWord),
            targetValue: field === 5 ? undefined : float(targetWord),
            category: field === 5 ? "flags"
              : ((sourceWord === 0x80000000 && targetWord === 0)
                || (sourceWord === 0 && targetWord === 0x80000000)) ? "signed-zero" : "float" };
        }
        if (field !== 5) {
          const error = Math.abs(float(sourceWord) - float(targetWord));
          if (Number.isFinite(error)) metrics.rendererPrimaryMaximumAbsoluteError = Math.max(
            metrics.rendererPrimaryMaximumAbsoluteError, error);
        }
      }
      if (primaryMismatch) metrics.rendererPrimaryMismatches += 1;
      for (let corner = 0; corner < 8; corner += 1) {
        const sourceAt = 8 + 8 * rendererRows + 8 * row + corner;
        const targetCorner = transform.corner(corner);
        const targetAt = 8 + 8 * rendererRows + 8 * targetRow + targetCorner;
        const sourceWord = renderer[sourceAt], targetWord = renderer[targetAt];
        if (sourceWord === undefined || targetWord === undefined || sourceWord !== targetWord) {
          metrics.rendererAuxCornerMismatches += 1;
          const error = sourceWord === undefined || targetWord === undefined
            ? Number.NaN : Math.abs(float(sourceWord) - float(targetWord));
          if (Number.isFinite(error)) metrics.rendererAuxMaximumAbsoluteError = Math.max(
            metrics.rendererAuxMaximumAbsoluteError, error);
          metrics.firstRendererAuxMismatch ??= { source: [x, y, z, span, corner],
            target: [...targetOrigin, span, targetCorner],
            sourceValue: sourceWord === undefined ? undefined : float(sourceWord),
            targetValue: targetWord === undefined ? undefined : float(targetWord), error };
        }
      }
    }
    result[name] = metrics;
    graphLeafIdentityMismatches += metrics.graphLeafIdentityMismatches;
    nodalCornerMismatches += metrics.nodalCornerMismatches;
    rendererLeafIdentityMismatches += metrics.rendererLeafIdentityMismatches;
    rendererPrimaryMismatches += metrics.rendererPrimaryMismatches;
    rendererAuxCornerMismatches += metrics.rendererAuxCornerMismatches;
    nodalCornerMaximumAbsoluteError = Math.max(nodalCornerMaximumAbsoluteError,
      metrics.nodalCornerMaximumAbsoluteError);
    rendererPrimaryMaximumAbsoluteError = Math.max(rendererPrimaryMaximumAbsoluteError,
      metrics.rendererPrimaryMaximumAbsoluteError);
    rendererAuxMaximumAbsoluteError = Math.max(rendererAuxMaximumAbsoluteError,
      metrics.rendererAuxMaximumAbsoluteError);
  }
  const rendererCapacity = Math.floor(Math.max(0, renderer.length - 8) / 16);
  const parity = (sourceBank: number, auxiliaryRows: number) => {
    let mismatches = 0, maximumAbsoluteError = 0;
    let first: Record<string, unknown> | undefined;
    for (let row = 0; row < rendererRows; row += 1) {
      const base = 8 + 8 * row, cell = (renderer[base] ?? 1) - 1;
      const x = cell % dimensions[0];
      const y = Math.floor(cell / dimensions[0]) % dimensions[1];
      const z = Math.floor(cell / (dimensions[0] * dimensions[1]));
      const span = renderer[base + 1] ?? 0;
      const leaf = graphByIdentity.get(identity(x, y, z, span));
      if (leaf === undefined) continue;
      for (let corner = 0; corner < 8; corner += 1) {
        const slot = leaves[16 * leaf + 8 + corner] ?? 0xffff_ffff;
        const sourceWord = slot < nodeCount ? nodalPhi[2 * slot + sourceBank] : undefined;
        const rendererAt = 8 + 8 * auxiliaryRows + 8 * row + corner;
        const rendererWord = renderer[rendererAt];
        if (sourceWord === undefined || rendererWord === undefined || sourceWord !== rendererWord) {
          mismatches += 1;
          const error = sourceWord === undefined || rendererWord === undefined
            ? Number.NaN : Math.abs(float(sourceWord) - float(rendererWord));
          if (Number.isFinite(error)) maximumAbsoluteError = Math.max(maximumAbsoluteError, error);
          first ??= { row, leaf, origin: [x, y, z, span], corner, slot, sourceBank,
            auxiliaryRows, rendererAt,
            sourceValue: sourceWord === undefined ? undefined : float(sourceWord),
            rendererValue: rendererWord === undefined ? undefined : float(rendererWord), error };
        }
      }
    }
    return Object.freeze({ mismatches, maximumAbsoluteError, first });
  };
  const publicationLayouts = Object.freeze({
    bank0Live: parity(0, rendererRows), bank1Live: parity(1, rendererRows),
    bank0Capacity: parity(0, rendererCapacity), bank1Capacity: parity(1, rendererCapacity),
  });
  publicationCornerMismatches = bank === 0
    ? publicationLayouts.bank0Live.mismatches : publicationLayouts.bank1Live.mismatches;
  publicationCornerMaximumAbsoluteError = bank === 0
    ? publicationLayouts.bank0Live.maximumAbsoluteError
    : publicationLayouts.bank1Live.maximumAbsoluteError;
  return Object.freeze({ leafCount, nodeCount, rendererRows, bank,
    graphLeafIdentityMismatches, nodalCornerMismatches, nodalCornerMaximumAbsoluteError,
    rendererLeafIdentityMismatches, rendererPrimaryMismatches,
    rendererPrimaryMaximumAbsoluteError, rendererAuxCornerMismatches,
    rendererAuxMaximumAbsoluteError, publicationCornerMismatches,
    publicationCornerMaximumAbsoluteError, rendererCapacity, publicationLayouts,
    transforms: Object.freeze(result) });
}

export function adaptiveVelocityD4Audit(snapshot: AdaptiveSurfacePublicationSnapshot) {
  const { graphControl, nodes, nodalVelocity, dimensions } = snapshot;
  const nodeCount = Math.min(graphControl[2] ?? 0, Math.floor(nodes.length / 4),
    Math.floor(nodalVelocity.length / 8));
  const dx = dimensions[0] + 1, dy = dimensions[1] + 1;
  const position = (node: number) => {
    const item = nodes[4 * node] ?? 0;
    return [item % dx, Math.floor(item / dx) % dy,
      Math.floor(item / (dx * dy))] as const;
  };
  const byPosition = new Map<string, number>();
  for (let node = 0; node < nodeCount; node += 1) {
    byPosition.set(position(node).join(":"), node);
  }
  const bits = new Uint32Array(1);
  const float = (word: number) => {
    bits[0] = word >>> 0;
    return new Float32Array(bits.buffer)[0]!;
  };
  const transforms = {
    "reflect-x": {
      position: (q: readonly [number, number, number]) =>
        [dimensions[0] - q[0], q[1], q[2]] as const,
      value: (v: readonly [number, number, number]) => [-v[0], v[1], v[2]] as const,
    },
    "reflect-z": {
      position: (q: readonly [number, number, number]) =>
        [q[0], q[1], dimensions[2] - q[2]] as const,
      value: (v: readonly [number, number, number]) => [v[0], v[1], -v[2]] as const,
    },
    "swap-xz": {
      position: (q: readonly [number, number, number]) => [q[2], q[1], q[0]] as const,
      value: (v: readonly [number, number, number]) => [v[2], v[1], v[0]] as const,
    },
  } as const;
  return Object.freeze(Object.fromEntries(Object.entries(transforms).map(([name, transform]) => {
    const banks = [0, 1].map((bank) => {
      let missingNodes = 0, maskMismatches = 0, valueMismatches = 0, maximumAbsoluteError = 0;
      let first: Readonly<Record<string, unknown>> | undefined;
      for (let node = 0; node < nodeCount; node += 1) {
        const q = position(node), targetQ = transform.position(q);
        const target = byPosition.get(targetQ.join(":"));
        if (target === undefined) { missingNodes += 1; continue; }
        const sourceAt = 8 * node + 4 * bank, targetAt = 8 * target + 4 * bank;
        const sourceMask = nodalVelocity[sourceAt + 3]! & 7;
        const targetMask = nodalVelocity[targetAt + 3]! & 7;
        if (sourceMask !== targetMask) maskMismatches += 1;
        const source = [float(nodalVelocity[sourceAt]!), float(nodalVelocity[sourceAt + 1]!),
          float(nodalVelocity[sourceAt + 2]!)] as const;
        const expected = transform.value(source);
        const actual = [float(nodalVelocity[targetAt]!), float(nodalVelocity[targetAt + 1]!),
          float(nodalVelocity[targetAt + 2]!)] as const;
        for (let component = 0; component < 3; component += 1) {
          if (Object.is(expected[component], actual[component])) continue;
          valueMismatches += 1;
          const error = Math.abs(expected[component]! - actual[component]!);
          maximumAbsoluteError = Math.max(maximumAbsoluteError, error);
          first ??= Object.freeze({ sourceNode: node, sourcePosition: q, targetNode: target,
            targetPosition: targetQ, component, source: source[component],
            expected: expected[component], actual: actual[component], error });
        }
      }
      return Object.freeze({ bank, missingNodes, maskMismatches, valueMismatches,
        maximumAbsoluteError, first });
    });
    return [name, Object.freeze(banks)] as const;
  })));
}

export function adaptiveVelocityInputD4Audit(snapshot: AdaptiveSurfacePublicationSnapshot) {
  const { authorityControl, faceGeometry, projectedFaceVelocity, predictedFaceVelocity,
    advectedFaceVelocity,
    extendedFaceVelocity, dimensions } = snapshot;
  const faceCount = Math.min(authorityControl[2] ?? 0, Math.floor(faceGeometry.length / 4),
    projectedFaceVelocity.length, predictedFaceVelocity.length, advectedFaceVelocity.length,
    extendedFaceVelocity.length);
  const geometry = (face: number) => {
    const at = 4 * face, packed = faceGeometry[at] ?? 0;
    return { axis: packed & 3, span: 1 << (packed >>> 2),
      q: [faceGeometry[at + 1] ?? 0, faceGeometry[at + 2] ?? 0,
        faceGeometry[at + 3] ?? 0] as [number, number, number] };
  };
  const key = (g: ReturnType<typeof geometry>) =>
    `${g.axis}:${g.span}:${g.q[0]}:${g.q[1]}:${g.q[2]}`;
  const byGeometry = new Map<string, number>();
  for (let face = 0; face < faceCount; face += 1) byGeometry.set(key(geometry(face)), face);
  const bits = new Uint32Array(1);
  const float = (word: number) => {
    bits[0] = word >>> 0; return new Float32Array(bits.buffer)[0]!;
  };
  const transforms = {
    "reflect-x": {
      geometry: (g: ReturnType<typeof geometry>) => ({ ...g,
        q: [dimensions[0] - g.q[0] - (g.axis === 0 ? 0 : g.span),
          g.q[1], g.q[2]] as [number, number, number] }),
      sign: (axis: number) => axis === 0 ? -1 : 1,
    },
    "reflect-z": {
      geometry: (g: ReturnType<typeof geometry>) => ({ ...g,
        q: [g.q[0], g.q[1], dimensions[2] - g.q[2] - (g.axis === 2 ? 0 : g.span)] as
          [number, number, number] }),
      sign: (axis: number) => axis === 2 ? -1 : 1,
    },
    "swap-xz": {
      geometry: (g: ReturnType<typeof geometry>) => ({ axis: g.axis === 0 ? 2
        : g.axis === 2 ? 0 : 1, span: g.span,
      q: [g.q[2], g.q[1], g.q[0]] as [number, number, number] }),
      sign: () => 1,
    },
  } as const;
  const fields = { predicted: predictedFaceVelocity, projected: projectedFaceVelocity,
    advected: advectedFaceVelocity,
    extended: extendedFaceVelocity } as const;
  return Object.freeze(Object.fromEntries(Object.entries(transforms).map(([name, transform]) => {
    let missingFaces = 0;
    const fieldResult = Object.fromEntries(Object.entries(fields).map(([fieldName, values]) => {
      let mismatches = 0, maximumAbsoluteError = 0;
      let first: Readonly<Record<string, unknown>> | undefined;
      for (let face = 0; face < faceCount; face += 1) {
        const sourceGeometry = geometry(face);
        const target = byGeometry.get(key(transform.geometry(sourceGeometry)));
        if (target === undefined) { if (fieldName === "projected") missingFaces += 1; continue; }
        const source = float(values[face]!), expected = transform.sign(sourceGeometry.axis) * source;
        const actual = float(values[target]!);
        if (Object.is(expected, actual)) continue;
        mismatches += 1;
        const error = Math.abs(expected - actual);
        maximumAbsoluteError = Math.max(maximumAbsoluteError, error);
        first ??= Object.freeze({ face, geometry: sourceGeometry, target,
          targetGeometry: geometry(target), source, expected, actual, error });
      }
      return [fieldName, Object.freeze({ mismatches, maximumAbsoluteError, first })];
    }));
    return [name, Object.freeze({ missingFaces, fields: Object.freeze(fieldResult) })];
  })));
}

/** Physical coefficient equivariance of the accepted pressure faces. */
export function adaptivePressureFaceSwapAudit(snapshot: AdaptiveSurfacePublicationSnapshot) {
  const { authorityControl, faceGeometry, pressureFaces, pressureRowToGraphLeaf,
    leaves, acceptedMass, dimensions } = snapshot;
  const faceCount = Math.min(authorityControl[2] ?? 0, Math.floor(faceGeometry.length / 4),
    Math.floor(pressureFaces.length / 8));
  const geometry = (face: number) => {
    const at = 4 * face, packed = faceGeometry[at] ?? 0;
    return { axis: packed & 3, span: 1 << (packed >>> 2),
      q: [faceGeometry[at + 1] ?? 0, faceGeometry[at + 2] ?? 0,
        faceGeometry[at + 3] ?? 0] as const };
  };
  const key = (g: ReturnType<typeof geometry>) =>
    `${g.axis}:${g.span}:${g.q[0]}:${g.q[1]}:${g.q[2]}`;
  const byGeometry = new Map<string, number>();
  for (let face = 0; face < faceCount; face += 1) byGeometry.set(key(geometry(face)), face);
  const bits = new Uint32Array(1);
  const float = (word: number) => {
    bits[0] = word >>> 0; return new Float32Array(bits.buffer)[0]!;
  };
  const fields = ["area", "inverseDistance", "openFraction", "normalVelocity",
    "coefficient"] as const;
  const rowEvidence = (row: number) => {
    if (row >= pressureRowToGraphLeaf.length) return Object.freeze({ row });
    const leaf = pressureRowToGraphLeaf[row] ?? 0xffff_ffff;
    if (leaf >= Math.floor(leaves.length / 16)) return Object.freeze({ row, leaf });
    const at = 16 * leaf;
    return Object.freeze({ row, leaf,
      originSpan: [leaves[at], leaves[at + 1], leaves[at + 2], leaves[at + 3]],
      surfaceMass_m3: acceptedMass[leaf],
    });
  };
  const cellOwners = new Uint32Array(dimensions[0] * dimensions[1] * dimensions[2]);
  cellOwners.fill(0xffff_ffff);
  for (let leaf = 0; leaf < Math.floor(leaves.length / 16); leaf += 1) {
    const at = 16 * leaf, ox = leaves[at]!, oy = leaves[at + 1]!, oz = leaves[at + 2]!;
    const span = leaves[at + 3]!;
    for (let z = oz; z < oz + span; z += 1) for (let y = oy; y < oy + span; y += 1) {
      for (let x = ox; x < ox + span; x += 1) {
        cellOwners[x + dimensions[0] * (y + dimensions[1] * z)] = leaf;
      }
    }
  }
  const leafEvidence = (leaf: number) => {
    if (leaf >= Math.floor(leaves.length / 16)) return undefined;
    const at = 16 * leaf;
    return Object.freeze({ leaf,
      originSpan: [leaves[at], leaves[at + 1], leaves[at + 2], leaves[at + 3]],
      surfaceMass_m3: acceptedMass[leaf],
    });
  };
  const ghostEvidence = (faceAt: number) => {
    const negativeRow = pressureFaces[faceAt]!, positiveRow = pressureFaces[faceAt + 1]!;
    if (negativeRow >= pressureRowToGraphLeaf.length
      || positiveRow < pressureRowToGraphLeaf.length) return undefined;
    const wetLeaf = pressureRowToGraphLeaf[negativeRow]!;
    if (wetLeaf >= Math.floor(leaves.length / 16)) return undefined;
    const leafAt = 16 * wetLeaf, axis = pressureFaces[faceAt + 2]!;
    const wetCentre = [leaves[leafAt]! + 0.5 * leaves[leafAt + 3]!,
      leaves[leafAt + 1]! + 0.5 * leaves[leafAt + 3]!,
      leaves[leafAt + 2]! + 0.5 * leaves[leafAt + 3]!] as [number, number, number];
    const direction = (pressureFaces[faceAt + 3]! & 0x8000_0000) !== 0 ? -1 : 1;
    const probe = [...wetCentre] as [number, number, number];
    probe[axis] += direction * (0.5 * leaves[leafAt + 3]! + 0.0001);
    const cell = probe.map(Math.floor) as [number, number, number];
    const airLeaf = cell.some((q, a) => q < 0 || q >= dimensions[a]!)
      ? 0xffff_ffff
      : cellOwners[cell[0] + dimensions[0] * (cell[1] + dimensions[1] * cell[2])]!;
    return Object.freeze({ axis, direction, wetCentre, probe, cell,
      wet: leafEvidence(wetLeaf), air: leafEvidence(airLeaf) });
  };
  const leafByOrigin = new Map<string, number>();
  for (let leaf = 0; leaf < Math.floor(leaves.length / 16); leaf += 1) {
    const at = 16 * leaf;
    leafByOrigin.set(`${leaves[at]}:${leaves[at + 1]}:${leaves[at + 2]}:${leaves[at + 3]}`,
      leaf);
  }
  let surfaceMassMismatches = 0, surfaceMassMaximumAbsoluteError_m3 = 0;
  let firstSurfaceMassMismatch: Readonly<Record<string, unknown>> | undefined;
  for (let leaf = 0; leaf < Math.floor(leaves.length / 16); leaf += 1) {
    const at = 16 * leaf, target = leafByOrigin.get(
      `${leaves[at + 2]}:${leaves[at + 1]}:${leaves[at]}:${leaves[at + 3]}`);
    if (target === undefined || Object.is(acceptedMass[leaf], acceptedMass[target])) continue;
    surfaceMassMismatches += 1;
    const error = Math.abs((acceptedMass[leaf] ?? Number.NaN)
      - (acceptedMass[target] ?? Number.NaN));
    if (Number.isFinite(error)) surfaceMassMaximumAbsoluteError_m3 = Math.max(
      surfaceMassMaximumAbsoluteError_m3, error);
    firstSurfaceMassMismatch ??= Object.freeze({ source: leafEvidence(leaf),
      target: leafEvidence(target), absoluteError_m3: error });
  }
  const metrics = Object.fromEntries(fields.map((field) => [field,
    { mismatches: 0, maximumAbsoluteError: 0,
      first: undefined as Readonly<Record<string, unknown>> | undefined }])) as Record<
        typeof fields[number], { mismatches: number; maximumAbsoluteError: number;
          first?: Readonly<Record<string, unknown>> }>;
  let missingFaces = 0;
  for (let face = 0; face < faceCount; face += 1) {
    const sourceGeometry = geometry(face);
    const targetGeometry = { axis: sourceGeometry.axis === 0 ? 2
      : sourceGeometry.axis === 2 ? 0 : 1, span: sourceGeometry.span,
    q: [sourceGeometry.q[2], sourceGeometry.q[1], sourceGeometry.q[0]] as const };
    const target = byGeometry.get(key(targetGeometry));
    if (target === undefined) { missingFaces += 1; continue; }
    const sourceAt = 8 * face, targetAt = 8 * target;
    const sourceValues = {
      area: float(pressureFaces[sourceAt + 4]!),
      inverseDistance: float(pressureFaces[sourceAt + 5]!),
      openFraction: float(pressureFaces[sourceAt + 6]!),
      normalVelocity: float(pressureFaces[sourceAt + 7]!),
      coefficient: (float(pressureFaces[sourceAt + 6]!)
        * float(pressureFaces[sourceAt + 4]!)) * float(pressureFaces[sourceAt + 5]!),
    };
    const targetValues = {
      area: float(pressureFaces[targetAt + 4]!),
      inverseDistance: float(pressureFaces[targetAt + 5]!),
      openFraction: float(pressureFaces[targetAt + 6]!),
      normalVelocity: float(pressureFaces[targetAt + 7]!),
      coefficient: (float(pressureFaces[targetAt + 6]!)
        * float(pressureFaces[targetAt + 4]!)) * float(pressureFaces[targetAt + 5]!),
    };
    for (const field of fields) {
      if (Object.is(sourceValues[field], targetValues[field])) continue;
      const metric = metrics[field], error = Math.abs(sourceValues[field] - targetValues[field]);
      metric.mismatches += 1;
      if (error > metric.maximumAbsoluteError) metric.maximumAbsoluteError = error;
      metric.first ??= Object.freeze({ face, geometry: sourceGeometry, target, targetGeometry,
        source: sourceValues[field], actual: targetValues[field], error,
        sourceRows: [rowEvidence(pressureFaces[sourceAt]!),
          rowEvidence(pressureFaces[sourceAt + 1]!)],
        targetRows: [rowEvidence(pressureFaces[targetAt]!),
          rowEvidence(pressureFaces[targetAt + 1]!)],
        sourceGhost: ghostEvidence(sourceAt), targetGhost: ghostEvidence(targetAt),
      });
    }
  }
  return Object.freeze({ faceCount, missingFaces,
    surfaceMass: Object.freeze({ mismatches: surfaceMassMismatches,
      maximumAbsoluteError_m3: surfaceMassMaximumAbsoluteError_m3,
      first: firstSurfaceMassMismatch }),
    fields: Object.freeze(Object.fromEntries(fields.map((field) =>
      [field, Object.freeze(metrics[field])]))),
  });
}

export function adaptiveVelocityStencilAudit(snapshot: AdaptiveSurfacePublicationSnapshot) {
  const { graphControl, nodes, constraints, nodalVelocity, faceGeometry, stencils,
    projectedFaceVelocity, advectedFaceVelocity, dimensions } = snapshot;
  const nodeCount = Math.min(graphControl[2] ?? 0, Math.floor(nodes.length / 4),
    Math.floor(constraints.length / 12), Math.floor(nodalVelocity.length / 8),
    Math.floor(stencils.length / (3 * 36)));
  const dx = dimensions[0] + 1, dy = dimensions[1] + 1;
  const bits = new Uint32Array(1);
  const float = (word: number) => {
    bits[0] = word >>> 0; return new Float32Array(bits.buffer)[0]!;
  };
  let seededComponents = 0, reconstructionMismatches = 0, maximumReconstructionError = 0;
  let firstReconstruction: Readonly<Record<string, unknown>> | undefined;
  for (let node = 0; node < nodeCount; node += 1) {
    if ((constraints[12 * node + 1] ?? 0) !== 0) continue;
    for (let component = 0; component < 3; component += 1) {
      const at = 36 * (3 * node + component), covered = stencils[at + 1] ?? 0;
      if ((stencils[at + 3] ?? 0) === 0 || covered === 0) continue;
      seededComponents += 1;
      // Both graph banks use the same geometric stencil; only their face field differs.
      for (const [bank, values] of [projectedFaceVelocity, advectedFaceVelocity].entries()) {
        let exact = 0;
        for (let entry = 0; entry < 16; entry += 1) {
          const slot = stencils[at + 4 + entry] ?? 0xffff_ffff;
          const weight = stencils[at + 20 + entry] ?? 0;
          if (weight !== 0 && slot < values.length) exact += Math.fround(weight * float(values[slot]!));
        }
        const expected = Math.fround(exact / covered);
        const actual = float(nodalVelocity[8 * node + 4 * bank + component]!);
        if (Object.is(expected, actual)) continue;
        reconstructionMismatches += 1;
        const error = Math.abs(expected - actual);
        maximumReconstructionError = Math.max(maximumReconstructionError, error);
        firstReconstruction ??= Object.freeze({ node, component, bank, covered,
          expected, actual, error });
      }
    }
  }
  const position = (node: number) => {
    const item = nodes[4 * node] ?? 0;
    return [item % dx, Math.floor(item / dx) % dy,
      Math.floor(item / (dx * dy))] as const;
  };
  const byPosition = new Map<string, number>();
  for (let node = 0; node < nodeCount; node += 1) byPosition.set(position(node).join(":"), node);
  const faceKey = (slot: number) => {
    if (4 * slot + 3 >= faceGeometry.length) return `invalid:${slot}`;
    const packed = faceGeometry[4 * slot] ?? 0, axis = packed & 3;
    return { axis, span: 1 << (packed >>> 2), q: [faceGeometry[4 * slot + 1] ?? 0,
      faceGeometry[4 * slot + 2] ?? 0, faceGeometry[4 * slot + 3] ?? 0] as const };
  };
  const transforms = {
    "reflect-x": { node: (q: readonly number[]) => [dimensions[0] - q[0]!, q[1]!, q[2]!],
      component: (c: number) => c,
      face: (g: Exclude<ReturnType<typeof faceKey>, string>) => [g.axis, g.span,
        dimensions[0] - g.q[0] - (g.axis === 0 ? 0 : g.span), g.q[1], g.q[2]] },
    "reflect-z": { node: (q: readonly number[]) => [q[0]!, q[1]!, dimensions[2] - q[2]!],
      component: (c: number) => c,
      face: (g: Exclude<ReturnType<typeof faceKey>, string>) => [g.axis, g.span,
        g.q[0], g.q[1], dimensions[2] - g.q[2] - (g.axis === 2 ? 0 : g.span)] },
    "swap-xz": { node: (q: readonly number[]) => [q[2]!, q[1]!, q[0]!],
      component: (c: number) => c === 0 ? 2 : c === 2 ? 0 : 1,
      face: (g: Exclude<ReturnType<typeof faceKey>, string>) => [g.axis === 0 ? 2
        : g.axis === 2 ? 0 : 1, g.span, g.q[2], g.q[1], g.q[0]] },
  } as const;
  const d4 = Object.fromEntries(Object.entries(transforms).map(([name, transform]) => {
    let missingNodes = 0, mismatches = 0;
    let first: Readonly<Record<string, unknown>> | undefined;
    for (let node = 0; node < nodeCount; node += 1) {
      const target = byPosition.get(transform.node(position(node)).join(":"));
      if (target === undefined) { missingNodes += 1; continue; }
      for (let component = 0; component < 3; component += 1) {
        const targetComponent = transform.component(component);
        const record = (slot: number, c: number, mapFace: boolean) => {
          const at = 36 * (3 * slot + c), header = Array.from(stencils.slice(at, at + 4));
          const entries: string[] = [];
          for (let entry = 0; entry < 16; entry += 1) {
            const weight = stencils[at + 20 + entry] ?? 0;
            if (weight === 0) continue;
            const g = faceKey(stencils[at + 4 + entry] ?? 0xffff_ffff);
            entries.push(`${mapFace && typeof g !== "string" ? transform.face(g).join(":")
              : typeof g === "string" ? g : [g.axis, g.span, ...g.q].join(":")}:${weight}`);
          }
          entries.sort(); return { header, entries };
        };
        const source = record(node, component, true), actual = record(target, targetComponent, false);
        if (JSON.stringify(source) === JSON.stringify(actual)) continue;
        mismatches += 1;
        first ??= Object.freeze({ node, position: position(node), component, target,
          targetPosition: position(target), targetComponent, source, actual });
      }
    }
    return [name, Object.freeze({ missingNodes, mismatches, first })];
  }));
  return Object.freeze({ seededComponents, reconstructionMismatches,
    maximumReconstructionError, firstReconstruction, d4: Object.freeze(d4) });
}
