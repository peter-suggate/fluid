import { FINE_LEVELSET_INVALID, FINE_LEVELSET_SAMPLE_FLAGS, type FineLevelSetBrickPlan } from "../lib/octree-fine-levelset-bricks";
import { OCTREE_COARSE_PHI_FLAG } from "../lib/octree-coarse-levelset";
import { globalFineCoarseGenerationPairIsValid } from "../lib/octree-consumer-sampling";
import { OCTREE_POWER_COARSE_LEVELSET_VALID } from "../lib/webgpu-octree-power-coarse-levelset";

export interface CompactOctreeFieldSnapshot {
  readonly plan: FineLevelSetBrickPlan;
  readonly generation: number;
  readonly hash: Uint32Array;
  readonly metadata: Uint32Array;
  readonly flags: Uint32Array;
  readonly phi: Float32Array;
  readonly worklist: Uint32Array;
  /** Header plus entries in the PowerCoarseSampleDirectory WGSL ABI. */
  readonly coarseDirectory: Uint32Array;
  /** Sixteen-word WebGPUOctreePowerCoarseLevelSet control ABI. */
  readonly coarseControl?: Uint32Array;
  readonly fineRestrictionControl?: Uint32Array;
  readonly topologyControl?: Uint32Array;
  readonly transportControl?: Uint32Array;
  readonly redistanceControl?: Uint32Array;
  readonly volumeControl?: Uint32Array;
  readonly faceBandControl?: Uint32Array;
  readonly faceBandTransitionControl?: Uint32Array;
  readonly faceBandTransientPowerControl?: Uint32Array;
  readonly faceBandPointFieldControl?: Uint32Array;
  readonly faceBandPowerPublicationControl?: Uint32Array;
  readonly powerVelocityControl?: Uint32Array;
  readonly powerProjectionControl?: Uint32Array;
  readonly powerVelocitySampleControl?: Uint32Array;
  readonly mgpcgControl?: Uint32Array;
}

export interface CompactOctreeFieldEvidence {
  readonly fineSamples: number;
  readonly coarseSamples: number;
  readonly positiveAirSamples: number;
  readonly generation: number;
  readonly activePages: number;
  readonly malformedActivePages: number;
  readonly validSamples: number;
  readonly finiteValidSamples: number;
  readonly negativeValidSamples: number;
  readonly positiveValidSamples: number;
  readonly publicationValid: boolean;
  readonly topologyFlags?: number;
  readonly topologyPublished?: boolean;
  readonly topologyRolledBack?: boolean;
  /** `FineLevelSetTopologyControl.control[7]`: rejected downstream stage bits. */
  readonly downstreamFinalizeReason?: number;
  /** Complete raw downstream transaction controls, retained for rejection telemetry. */
  readonly transportControl?: readonly number[];
  readonly redistanceControl?: readonly number[];
  readonly volumeControl?: readonly number[];
  readonly faceBandControl?: readonly number[];
  readonly faceBandTransitionControl?: readonly number[];
  readonly faceBandTransientPowerControl?: readonly number[];
  readonly faceBandPointFieldControl?: readonly number[];
  readonly faceBandPowerPublicationControl?: readonly number[];
  readonly powerVelocityControl?: readonly number[];
  readonly powerProjectionControl?: readonly number[];
  readonly powerVelocitySampleControl?: readonly number[];
  readonly mgpcgControl?: readonly number[];
}

export interface CompactOctreePublicationHeaderEvidence {
  readonly fineGeneration: number;
  readonly worklistActivePages?: number;
  readonly worklistGeneration?: number;
  readonly worklistInitialized?: number;
  readonly worklistPublished?: number;
  readonly coarseState?: number;
  readonly coarseGeneration?: number;
  readonly coarseHashCapacity?: number;
  readonly coarseMaximumLeafSize?: number;
  readonly coarseControlFlags?: number;
  readonly coarseControlFirstErrorRow?: number;
  readonly coarseControlRowCount?: number;
  readonly coarseControlAdvectedRows?: number;
  readonly coarseControlCorrectedRows?: number;
  readonly coarseControlInterfaceRows?: number;
  readonly coarseControlContributionCount?: number;
  readonly coarseControlGeneration?: number;
  readonly coarseControlValid?: number;
  readonly fineRestrictionCount?: number;
  readonly fineRestrictionMaximumPerRow?: number;
  readonly fineRestrictionFlags?: number;
  readonly fineRestrictionUnowned?: number;
  readonly fineRestrictionRows?: number;
  readonly fineRestrictionValid?: number;
  readonly fineRestrictionFirstUnownedLiquidLogical?: number;
  readonly fineRestrictionMaximumUnownedLiquidMagnitude?: number;
  readonly topologyFlags?: number;
  readonly topologyInterfaceBricks?: number;
  readonly topologyDesiredBricks?: number;
  readonly topologyActivatedBricks?: number;
  readonly topologyPublished?: number;
  readonly topologyRolledBack?: number;
  /** Dilation rings on success; required desired pages on capacity failure. */
  readonly topologyCapacityOrDilation?: number;
  readonly downstreamFinalizeReason?: number;
  /** Complete raw downstream transaction controls, retained for rejection telemetry. */
  readonly transportControl?: readonly number[];
  readonly redistanceControl?: readonly number[];
  readonly volumeControl?: readonly number[];
  readonly faceBandControl?: readonly number[];
  readonly faceBandTransitionControl?: readonly number[];
  readonly faceBandTransientPowerControl?: readonly number[];
  readonly faceBandPointFieldControl?: readonly number[];
  readonly faceBandPowerPublicationControl?: readonly number[];
  readonly powerVelocityControl?: readonly number[];
  readonly powerProjectionControl?: readonly number[];
  readonly powerVelocitySampleControl?: readonly number[];
  readonly mgpcgControl?: readonly number[];
}

export interface CompactOctreeFieldReconstruction extends CompactOctreeFieldEvidence {
  readonly field: Float32Array;
}

/** Required factor-4 acceptance proof; a plausible coarse-only field is insufficient. */
export function compactOctreeFieldEvidenceIsAcceptable(evidence: CompactOctreeFieldEvidence): boolean {
  return evidence.publicationValid
    && evidence.activePages > 0 && evidence.malformedActivePages === 0
    && evidence.validSamples > 0 && evidence.finiteValidSamples === evidence.validSamples
    && evidence.negativeValidSamples > 0 && evidence.positiveValidSamples > 0
    && evidence.fineSamples > 0 && evidence.coarseSamples > 0
    && evidence.topologyFlags === 0 && evidence.topologyPublished === true
    && evidence.topologyRolledBack === false && evidence.downstreamFinalizeReason === 0;
}

/** Header-only evidence is safe to report even when a publication is invalid. */
export function compactOctreePublicationHeaderEvidence(
  snapshot: Pick<CompactOctreeFieldSnapshot, "generation" | "worklist" | "coarseDirectory" | "coarseControl"
    | "fineRestrictionControl" | "topologyControl" | "transportControl" | "redistanceControl" | "volumeControl"
    | "faceBandControl" | "faceBandTransitionControl" | "faceBandTransientPowerControl"
    | "faceBandPointFieldControl"
    | "faceBandPowerPublicationControl"
    | "powerVelocityControl" | "powerProjectionControl" | "powerVelocitySampleControl" | "mgpcgControl">,
): CompactOctreePublicationHeaderEvidence {
  const worklist = snapshot.worklist, coarse = snapshot.coarseDirectory;
  const coarseControl = snapshot.coarseControl, restriction = snapshot.fineRestrictionControl,
    topology = snapshot.topologyControl;
  return {
    fineGeneration: snapshot.generation,
    ...(worklist.length >= 5 ? {
      worklistActivePages: worklist[0], worklistGeneration: worklist[1],
      worklistInitialized: worklist[3], worklistPublished: worklist[4],
    } : {}),
    ...(coarse.length >= 4 ? {
      coarseState: coarse[0], coarseGeneration: coarse[1], coarseHashCapacity: coarse[2],
      coarseMaximumLeafSize: coarse[3],
    } : {}),
    ...(coarseControl && coarseControl.length >= 13 ? {
      coarseControlFlags: coarseControl[0], coarseControlFirstErrorRow: coarseControl[1],
      coarseControlRowCount: coarseControl[2], coarseControlAdvectedRows: coarseControl[3],
      coarseControlCorrectedRows: coarseControl[8], coarseControlInterfaceRows: coarseControl[9],
      coarseControlContributionCount: coarseControl[10],
      coarseControlGeneration: coarseControl[11], coarseControlValid: coarseControl[12],
    } : {}),
    ...(topology && topology.length >= 8 ? {
      topologyFlags: topology[0], topologyInterfaceBricks: topology[1],
      topologyDesiredBricks: topology[2], topologyActivatedBricks: topology[3],
      topologyPublished: topology[4], topologyRolledBack: topology[5],
      topologyCapacityOrDilation: topology[6],
      downstreamFinalizeReason: topology[7],
    } : {}),
    ...(restriction && restriction.length >= 6 ? {
      fineRestrictionCount: restriction[0], fineRestrictionMaximumPerRow: restriction[1],
      fineRestrictionFlags: restriction[2], fineRestrictionUnowned: restriction[3],
      fineRestrictionRows: restriction[4], fineRestrictionValid: restriction[5],
      ...(restriction.length >= 8 ? {
        fineRestrictionFirstUnownedLiquidLogical: restriction[6],
        fineRestrictionMaximumUnownedLiquidMagnitude: finiteFloat(restriction, 7),
      } : {}),
    } : {}),
    ...(snapshot.transportControl ? { transportControl: Array.from(snapshot.transportControl) } : {}),
    ...(snapshot.redistanceControl ? { redistanceControl: Array.from(snapshot.redistanceControl) } : {}),
    ...(snapshot.volumeControl ? { volumeControl: Array.from(snapshot.volumeControl) } : {}),
    ...(snapshot.faceBandControl ? { faceBandControl: Array.from(snapshot.faceBandControl) } : {}),
    ...(snapshot.faceBandTransitionControl
      ? { faceBandTransitionControl: Array.from(snapshot.faceBandTransitionControl) } : {}),
    ...(snapshot.faceBandTransientPowerControl
      ? { faceBandTransientPowerControl: Array.from(snapshot.faceBandTransientPowerControl) } : {}),
    ...(snapshot.faceBandPointFieldControl
      ? { faceBandPointFieldControl: Array.from(snapshot.faceBandPointFieldControl) } : {}),
    ...(snapshot.faceBandPowerPublicationControl
      ? { faceBandPowerPublicationControl: Array.from(snapshot.faceBandPowerPublicationControl) } : {}),
    ...(snapshot.powerVelocityControl ? { powerVelocityControl: Array.from(snapshot.powerVelocityControl) } : {}),
    ...(snapshot.powerProjectionControl ? { powerProjectionControl: Array.from(snapshot.powerProjectionControl) } : {}),
    ...(snapshot.powerVelocitySampleControl
      ? { powerVelocitySampleControl: Array.from(snapshot.powerVelocitySampleControl) } : {}),
    ...(snapshot.mgpcgControl ? { mgpcgControl: Array.from(snapshot.mgpcgControl) } : {}),
  };
}

function hashFineKey(key: number, capacity: number): number {
  return (Math.imul((key ^ (key >>> 16)) >>> 0, 0x9e37_79b1) >>> 0) & (capacity - 1);
}

function hashCoarseSite(cell: number, size: number): number {
  let value = (cell ^ Math.imul(size, 0x9e37_79b9)) >>> 0;
  value = Math.imul(value ^ (value >>> 16), 0x7feb_352d) >>> 0;
  value = Math.imul(value ^ (value >>> 15), 0x846c_a68b) >>> 0;
  return (value ^ (value >>> 16)) >>> 0;
}

function finiteFloat(words: Uint32Array, index: number): number {
  return new Float32Array(words.buffer, words.byteOffset + index * 4, 1)[0];
}

function validateSnapshot(snapshot: CompactOctreeFieldSnapshot, dimensions: readonly [number, number, number]): void {
  const { plan, generation, hash, metadata, flags, phi, worklist, coarseDirectory, coarseControl, topologyControl } = snapshot;
  if (!Number.isSafeInteger(generation) || generation < 1) throw new Error("Compact octree QA field requires a positive fine generation");
  if (hash.length !== plan.hashCapacity * 2) throw new Error("Compact octree QA fine hash has the wrong length");
  if (metadata.length !== plan.maximumResidentBricks * 10) throw new Error("Compact octree QA fine metadata has the wrong length");
  const sampleCapacity = plan.maximumResidentBricks * plan.samplesPerBrick;
  if (flags.length !== sampleCapacity || phi.length !== sampleCapacity) throw new Error("Compact octree QA fine payload has the wrong length");
  if (worklist.length !== 5 + plan.maximumResidentBricks) throw new Error("Compact octree QA worklist has the wrong length");
  if (coarseControl !== undefined && coarseControl.length < 16) throw new Error("Compact octree QA coarse control has the wrong length");
  if (topologyControl !== undefined && topologyControl.length < 8) throw new Error("Compact octree QA topology control has the wrong length");
  if (coarseDirectory.length < 8) throw new Error("Compact octree QA coarse directory is missing its header");
  const capacity = coarseDirectory[2];
  const publication = compactOctreePublicationHeaderEvidence(snapshot);
  if (coarseDirectory[0] !== OCTREE_POWER_COARSE_LEVELSET_VALID) {
    throw new Error(`Compact octree QA coarse publication is not valid: ${JSON.stringify(publication)}`);
  }
  if (!globalFineCoarseGenerationPairIsValid(generation, coarseDirectory[1], topologyControl)) {
    throw new Error(`Compact octree QA coarse/fine generation mismatch: ${JSON.stringify(publication)}`);
  }
  const worklistClaimsPublication = worklist[0] !== 0 || worklist[1] !== 0
    || worklist[3] !== 0 || worklist[4] !== 0;
  if (worklistClaimsPublication
    && (worklist[1] !== generation || worklist[3] !== 1 || worklist[4] !== 1)) {
    throw new Error(`Compact octree QA fine publication is not valid/current: ${JSON.stringify(publication)}`);
  }
  if (capacity < 1 || (capacity & (capacity - 1)) !== 0 || coarseDirectory.length !== 8 + capacity * 8) {
    throw new Error("Compact octree QA coarse directory has an invalid hash capacity");
  }
  if (dimensions.some((value, axis) => value !== coarseDirectory[4 + axis]
    || value !== plan.finestCellDimensions[axis])) {
    throw new Error("Compact octree QA publication dimensions differ from the requested field");
  }
  if (plan.domainOrigin.some((value) => value !== 0)) {
    throw new Error("Compact octree QA coarse fallback requires the production zero-origin frame");
  }
  const physicalCellSize = finiteFloat(coarseDirectory, 7);
  if (!(physicalCellSize > 0) || Math.abs(physicalCellSize - plan.finestCellWidth)
    > 1e-5 * Math.max(physicalCellSize, plan.finestCellWidth)) {
    throw new Error("Compact octree QA coarse and fine publications use different cell widths");
  }
  const maximumLeafSize = coarseDirectory[3];
  if (maximumLeafSize < 1 || (maximumLeafSize & (maximumLeafSize - 1)) !== 0) {
    throw new Error("Compact octree QA coarse directory has an invalid maximum leaf size");
  }
}

function finePhiAt(snapshot: CompactOctreeFieldSnapshot, position: readonly [number, number, number]): number | undefined {
  const { plan, generation, hash, metadata, flags, phi } = snapshot;
  const q = position.map((value, axis) => Math.floor((value - plan.domainOrigin[axis]) / plan.fineCellWidth));
  if (q.some((value, axis) => value < 0 || value >= plan.sampleDimensions[axis])) return undefined;
  const brick = q.map((value) => Math.floor(value / plan.brickResolution));
  const local = q.map((value, axis) => value - brick[axis] * plan.brickResolution);
  const key = brick[0] + plan.brickDimensions[0] * (brick[1] + plan.brickDimensions[1] * brick[2]);
  let slot = hashFineKey(key, plan.hashCapacity);
  for (let probe = 0; probe < Math.min(plan.maximumHashProbes, plan.hashCapacity); probe += 1) {
    const stored = hash[slot * 2];
    if (stored === FINE_LEVELSET_INVALID) return undefined;
    if (stored === key) {
      const physicalId = hash[slot * 2 + 1];
      const base = physicalId * 10;
      if (physicalId >= plan.maximumResidentBricks || metadata[base] !== physicalId
        || metadata[base + 1] !== key || metadata[base + 2] !== generation) return undefined;
      const localIndex = local[0] + plan.brickResolution * (local[1] + plan.brickResolution * local[2]);
      const sampleIndex = physicalId * plan.samplesPerBrick + localIndex;
      const value = phi[sampleIndex];
      return (flags[sampleIndex] & FINE_LEVELSET_SAMPLE_FLAGS.valid) !== 0 && Number.isFinite(value)
        ? value : undefined;
    }
    slot = (slot + 1) & (plan.hashCapacity - 1);
  }
  return undefined;
}

function coarsePhiAt(snapshot: CompactOctreeFieldSnapshot, position: readonly [number, number, number]): { phi: number; positiveAir: boolean } {
  const words = snapshot.coarseDirectory;
  const capacity = words[2], maximumLeafSize = words[3];
  const dimensions = [words[4], words[5], words[6]] as const;
  const physicalCellSize = finiteFloat(words, 7);
  const q = position.map((value) => Math.floor(value / physicalCellSize));
  if (q.some((value, axis) => value < 0 || value >= dimensions[axis])) {
    throw new Error("Compact octree QA sample lies outside the coarse publication");
  }
  for (let size = 1; size <= maximumLeafSize; size *= 2) {
    const origin = q.map((value) => Math.floor(value / size) * size);
    const cell = origin[0] + dimensions[0] * (origin[1] + dimensions[1] * origin[2]);
    let slot = hashCoarseSite(cell, size) & (capacity - 1);
    for (let probe = 0; probe < Math.min(32, capacity); probe += 1) {
      const base = 8 + slot * 8, cellPlusOne = words[base];
      if (cellPlusOne === 0) break;
      if (cellPlusOne === cell + 1 && words[base + 1] === size) {
        const value = finiteFloat(words, base + 2);
        if ((words[base + 5] & OCTREE_COARSE_PHI_FLAG.valid) === 0 || !Number.isFinite(value)) {
          throw new Error("Compact octree QA encountered an invalid containing coarse leaf");
        }
        return { phi: value, positiveAir: false };
      }
      slot = (slot + 1) & (capacity - 1);
    }
  }
  // The production sampler defines an absent containing leaf as the compact
  // directory's positive-air complement, not as zero or an aggregate value.
  return { phi: physicalCellSize * maximumLeafSize, positiveAir: true };
}

/**
 * Reconstruct a QA-only cubic occupancy field from the same current sparse
 * fine publication and compact-coarse fallback used by production consumers.
 */
export function reconstructCompactOctreeOccupancyField(
  snapshot: CompactOctreeFieldSnapshot,
  dimensions: readonly [number, number, number],
): CompactOctreeFieldReconstruction {
  validateSnapshot(snapshot, dimensions);
  const field = new Float32Array(dimensions[0] * dimensions[1] * dimensions[2]);
  const h = snapshot.plan.finestCellWidth, factor = snapshot.plan.fineFactor;
  const activePages = Math.min(snapshot.worklist[0], snapshot.plan.maximumResidentBricks);
  let malformedActivePages = 0, validSamples = 0, finiteValidSamples = 0;
  let negativeValidSamples = 0, positiveValidSamples = 0;
  for (let work = 0; work < activePages; work += 1) {
    const id = snapshot.worklist[5 + work];
    if (id >= snapshot.plan.maximumResidentBricks || snapshot.metadata[id * 10] !== id
      || snapshot.metadata[id * 10 + 2] !== snapshot.generation) {
      malformedActivePages += 1; continue;
    }
    for (let local = 0; local < snapshot.plan.samplesPerBrick; local += 1) {
      const index = id * snapshot.plan.samplesPerBrick + local;
      if ((snapshot.flags[index] & FINE_LEVELSET_SAMPLE_FLAGS.valid) === 0) continue;
      validSamples += 1;
      const value = snapshot.phi[index];
      if (!Number.isFinite(value)) continue;
      finiteValidSamples += 1;
      if (value < 0) negativeValidSamples += 1; else positiveValidSamples += 1;
    }
  }
  let fineSamples = 0, coarseSamples = 0, positiveAirSamples = 0;
  for (let z = 0; z < dimensions[2]; z += 1) for (let y = 0; y < dimensions[1]; y += 1) for (let x = 0; x < dimensions[0]; x += 1) {
    let occupancy = 0;
    // Average the factor^3 fine samples owned by this finest octree cell. The
    // phi-to-occupancy scaling matches renderer/tall QA sampling and yields a
    // comparable cubic field. Missing fine samples use compact-coarse authority.
    for (let fz = 0; fz < factor; fz += 1) for (let fy = 0; fy < factor; fy += 1) for (let fx = 0; fx < factor; fx += 1) {
      const position = [
        snapshot.plan.domainOrigin[0] + (x * factor + fx + 0.5) * snapshot.plan.fineCellWidth,
        snapshot.plan.domainOrigin[1] + (y * factor + fy + 0.5) * snapshot.plan.fineCellWidth,
        snapshot.plan.domainOrigin[2] + (z * factor + fz + 0.5) * snapshot.plan.fineCellWidth,
      ] as const;
      let value = finePhiAt(snapshot, position);
      if (value === undefined) {
        const coarse = coarsePhiAt(snapshot, position);
        value = coarse.phi; coarseSamples += 1;
        if (coarse.positiveAir) positiveAirSamples += 1;
      } else fineSamples += 1;
      occupancy += Math.min(1, Math.max(0, 0.5 - value / h));
    }
    field[x + dimensions[0] * (y + dimensions[1] * z)] = occupancy / (factor ** 3);
  }
  const topology = snapshot.topologyControl;
  return { field, fineSamples, coarseSamples, positiveAirSamples, generation: snapshot.generation,
    activePages, malformedActivePages, validSamples, finiteValidSamples,
    negativeValidSamples, positiveValidSamples,
    publicationValid: snapshot.worklist[1] === snapshot.generation
      && snapshot.worklist[3] === 1 && snapshot.worklist[4] === 1
      && activePages > 0 && malformedActivePages === 0
      && validSamples > 0 && finiteValidSamples === validSamples,
    ...(topology ? { topologyFlags: topology[0], topologyPublished: topology[4] !== 0,
      topologyRolledBack: topology[5] !== 0, downstreamFinalizeReason: topology[7] } : {}),
  };
}
