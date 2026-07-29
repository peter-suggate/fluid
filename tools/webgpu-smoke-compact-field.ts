import { FINE_LEVELSET_SAMPLE_FLAGS, type FineLevelSetBrickPlan } from "../lib/octree-fine-levelset-bricks";
import { OCTREE_COARSE_PHI_FLAG } from "../lib/octree-coarse-levelset";
import { globalFineCoarseGenerationPairIsValid } from "../lib/octree-consumer-sampling";
import { OCTREE_POWER_COARSE_LEVELSET_VALID } from "../lib/webgpu-octree-power-coarse-levelset";
import { unpackFineLevelSetGPURedistanceControl } from "../lib/webgpu-octree-fine-levelset-redistance";

export interface CompactOctreeFieldSnapshot {
  readonly plan: FineLevelSetBrickPlan;
  readonly generation: number;
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
  readonly coarseGeneration: number;
  /** True when the current fine SPGrid intentionally uses the last committed
   * background octree after a rejected adaptation transaction. */
  readonly retainedCoarseAuthority: boolean;
  readonly topologyFlags?: number;
  readonly topologyPublished?: boolean;
  readonly topologyRolledBack?: boolean;
  /** `FineLevelSetTopologyControl.control[7]`: rejected downstream stage bits. */
  readonly downstreamFinalizeReason?: number;
  readonly latchedFinalizeReason?: number;
  readonly latchedFirstRejectedGeneration?: number;
  readonly latchedRejectionCount?: number;
  /** Complete raw downstream transaction controls, retained for rejection telemetry. */
  readonly transportControl?: readonly number[];
  readonly redistanceControl?: readonly number[];
  readonly redistanceUnresolvedCells?: number;
  readonly redistanceMaximumResidualScaled?: number;
  readonly redistanceSeedCount?: number;
  readonly redistanceCommitted?: boolean;
  readonly redistanceFlags?: number;
  readonly redistanceFirstError?: number;
  readonly redistanceAcceptedCells?: number;
  readonly redistanceInitialPages?: number;
  readonly redistanceFinalPages?: number;
  readonly volumeControl?: readonly number[];
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
  readonly coarseRowCount?: number;
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
  readonly latchedFinalizeReason?: number;
  readonly latchedFirstRejectedGeneration?: number;
  readonly latchedRejectionCount?: number;
  /** Complete raw downstream transaction controls, retained for rejection telemetry. */
  readonly transportControl?: readonly number[];
  readonly redistanceControl?: readonly number[];
  readonly redistanceUnresolvedCells?: number;
  readonly redistanceMaximumResidualScaled?: number;
  readonly redistanceSeedCount?: number;
  readonly redistanceCommitted?: boolean;
  readonly redistanceFlags?: number;
  readonly redistanceFirstError?: number;
  readonly redistanceAcceptedCells?: number;
  readonly redistanceInitialPages?: number;
  readonly redistanceFinalPages?: number;
  readonly volumeControl?: readonly number[];
  readonly mgpcgControl?: readonly number[];
}

export interface CompactOctreeFieldReconstruction extends CompactOctreeFieldEvidence {
  readonly field: Float32Array;
}

/** Required factor-4 acceptance proof; a plausible coarse-only field is insufficient. */
export function compactOctreeFieldEvidenceIsAcceptable(evidence: CompactOctreeFieldEvidence): boolean {
  const cleanCurrentTopology = evidence.topologyFlags === 0
    && evidence.topologyPublished === true && evidence.topologyRolledBack === false
    && evidence.downstreamFinalizeReason === 0;
  return evidence.publicationValid
    && evidence.activePages > 0 && evidence.malformedActivePages === 0
    && evidence.validSamples > 0 && evidence.finiteValidSamples === evidence.validSamples
    && evidence.negativeValidSamples > 0 && evidence.positiveValidSamples > 0
    && evidence.fineSamples > 0 && evidence.coarseSamples > 0
    && (cleanCurrentTopology || evidence.retainedCoarseAuthority);
}

/** Header-only evidence is safe to report even when a publication is invalid. */
export function compactOctreePublicationHeaderEvidence(
  snapshot: Pick<CompactOctreeFieldSnapshot, "generation" | "worklist" | "coarseDirectory" | "coarseControl"
    | "fineRestrictionControl" | "topologyControl" | "transportControl" | "redistanceControl" | "volumeControl"
    | "mgpcgControl">,
): CompactOctreePublicationHeaderEvidence {
  const worklist = snapshot.worklist, coarse = snapshot.coarseDirectory;
  const coarseControl = snapshot.coarseControl, restriction = snapshot.fineRestrictionControl,
    topology = snapshot.topologyControl;
  const redistance = snapshot.redistanceControl
    ? unpackFineLevelSetGPURedistanceControl(snapshot.redistanceControl)
    : undefined;
  return {
    fineGeneration: snapshot.generation,
    ...(worklist.length >= 7 ? {
      worklistActivePages: worklist[1], worklistGeneration: worklist[0],
      worklistInitialized: worklist[3] & 1, worklistPublished: (worklist[3] >>> 1) & 1,
    } : {}),
    ...(coarse.length >= 4 ? {
      coarseState: coarse[0], coarseGeneration: coarse[1], coarseRowCount: coarse[2],
      coarseMaximumLeafSize: coarse[3],
    } : {}),
    ...(coarseControl && coarseControl.length >= 13 ? {
      coarseControlFlags: coarseControl[0], coarseControlFirstErrorRow: coarseControl[1],
      coarseControlRowCount: coarseControl[2], coarseControlAdvectedRows: coarseControl[3],
      coarseControlCorrectedRows: coarseControl[7], coarseControlInterfaceRows: coarseControl[8],
      coarseControlContributionCount: coarseControl[9],
      coarseControlGeneration: coarseControl[10], coarseControlValid: coarseControl[11],
    } : {}),
    ...(topology && topology.length >= 8 ? {
      topologyFlags: topology[0], topologyInterfaceBricks: topology[1],
      topologyDesiredBricks: topology[2], topologyActivatedBricks: topology[3],
      topologyPublished: topology[4], topologyRolledBack: topology[5],
      topologyCapacityOrDilation: topology[6],
      downstreamFinalizeReason: topology[7],
      ...(topology.length >= 12 ? {
        // Sticky across generations; words 0..8 are cleared per generation.
        latchedFinalizeReason: topology[10],
        latchedFirstRejectedGeneration: topology[11] >>> 16,
        latchedRejectionCount: topology[11] & 0xffff,
      } : {}),
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
    ...(snapshot.redistanceControl && redistance ? {
      redistanceControl: Array.from(snapshot.redistanceControl),
      redistanceUnresolvedCells: redistance.unresolvedCells,
      redistanceMaximumResidualScaled: redistance.maximumResidualScaled,
      redistanceSeedCount: redistance.seedCount,
      redistanceCommitted: redistance.committed,
      redistanceFlags: redistance.flags,
      redistanceFirstError: redistance.firstError,
      redistanceAcceptedCells: redistance.acceptedCells,
      redistanceInitialPages: redistance.initialPages,
      redistanceFinalPages: redistance.finalPages,
    } : {}),
    ...(snapshot.volumeControl ? { volumeControl: Array.from(snapshot.volumeControl) } : {}),
    ...(snapshot.mgpcgControl ? { mgpcgControl: Array.from(snapshot.mgpcgControl) } : {}),
  };
}

function mortonPart10(value: number): number {
  let x = value & 1023;
  x = (x | (x << 16)) & 0x030000ff; x = (x | (x << 8)) & 0x0300f00f;
  x = (x | (x << 4)) & 0x030c30c3; x = (x | (x << 2)) & 0x09249249;
  return x >>> 0;
}

function coarseMorton(cell: number, dimensions: readonly number[]): number {
  const x = cell % dimensions[0], y = Math.floor(cell / dimensions[0]) % dimensions[1];
  const z = Math.floor(cell / (dimensions[0] * dimensions[1]));
  return (mortonPart10(x) | (mortonPart10(y) << 1) | (mortonPart10(z) << 2)) >>> 0;
}

function finiteFloat(words: Uint32Array, index: number): number {
  return new Float32Array(words.buffer, words.byteOffset + index * 4, 1)[0];
}

function validateSnapshot(snapshot: CompactOctreeFieldSnapshot, dimensions: readonly [number, number, number]): void {
  const { plan, generation, metadata, flags, phi, worklist, coarseDirectory, coarseControl, topologyControl } = snapshot;
  if (!Number.isSafeInteger(generation) || generation < 1) throw new Error("Compact octree QA field requires a positive fine generation");
  if (metadata.length !== plan.maximumResidentBricks * 10) throw new Error("Compact octree QA fine metadata has the wrong length");
  const sampleCapacity = plan.maximumResidentBricks * plan.samplesPerBrick;
  if (flags.length !== sampleCapacity || phi.length !== sampleCapacity) throw new Error("Compact octree QA fine payload has the wrong length");
  const expectedWorklistWords = 7 + plan.maximumResidentBricks + plan.logicalBrickCount
    + (plan.includeHalo27 ? 27 * plan.maximumResidentBricks : 0);
  if (worklist.length !== expectedWorklistWords) throw new Error("Compact octree QA worklist has the wrong length");
  if (coarseControl !== undefined && coarseControl.length < 16) throw new Error("Compact octree QA coarse control has the wrong length");
  if (topologyControl !== undefined && topologyControl.length < 8) throw new Error("Compact octree QA topology control has the wrong length");
  if (coarseDirectory.length < 8) throw new Error("Compact octree QA coarse directory is missing its header");
  const rowCount = coarseDirectory[2];
  const publication = compactOctreePublicationHeaderEvidence(snapshot);
  if (coarseDirectory[0] !== OCTREE_POWER_COARSE_LEVELSET_VALID) {
    throw new Error(`Compact octree QA coarse publication is not valid: ${JSON.stringify(publication)}`);
  }
  if (!globalFineCoarseGenerationPairIsValid(generation, coarseDirectory[1], topologyControl)) {
    const malformedCoarseRows: unknown[] = [];
    const requiredFlags = OCTREE_COARSE_PHI_FLAG.valid | OCTREE_COARSE_PHI_FLAG.finite;
    for (let slot = 0; slot < Math.min(rowCount, (coarseDirectory.length - 8) / 8); slot += 1) {
      const base = 8 + slot * 8;
      const floats = new Float32Array(coarseDirectory.buffer,
        coarseDirectory.byteOffset + base * 4, 8);
      const phi = floats[2], minimumPhi = floats[3], maximumPhi = floats[4];
      const flags = coarseDirectory[base + 5], row = coarseDirectory[base + 6];
      const physicalVolume = floats[7];
      if (coarseDirectory[base] === 0 || coarseDirectory[base + 1] === 0
        || row >= rowCount || (flags & requiredFlags) !== requiredFlags
        || !Number.isFinite(phi) || !Number.isFinite(minimumPhi)
        || !Number.isFinite(maximumPhi) || minimumPhi > phi || phi > maximumPhi
        || !Number.isFinite(physicalVolume) || physicalVolume <= 0) {
        malformedCoarseRows.push({ slot, cellPlusOne: coarseDirectory[base],
          size: coarseDirectory[base + 1], phi, minimumPhi, maximumPhi,
          flags, row, physicalVolume });
        if (malformedCoarseRows.length >= 8) break;
      }
    }
    throw new Error(`Compact octree QA coarse/fine generation mismatch: ${JSON.stringify({
      ...publication, malformedCoarseRows,
    })}`);
  }
  const worklistClaimsPublication = worklist[0] !== 0 || worklist[1] !== 0 || worklist[3] !== 0;
  if (worklistClaimsPublication
    && (worklist[0] !== generation || worklist[2] !== plan.maximumResidentBricks
      || (worklist[3] & 3) !== 3 || worklist[5] !== 1 || worklist[6] !== 1)) {
    throw new Error(`Compact octree QA fine publication is not valid/current: ${JSON.stringify(publication)}`);
  }
  const active = Math.min(worklist[1], plan.maximumResidentBricks);
  if (7 + active > worklist.length) throw new Error("Compact octree QA fine directory exceeds its worklist");
  for (let index = 0; index < active; index += 1) {
    const physicalId = worklist[7 + index], base = physicalId * 10;
    if (physicalId >= plan.maximumResidentBricks || base + 2 >= metadata.length
      || metadata[base] !== physicalId || metadata[base + 2] !== generation
      || physicalId !== index) {
      throw new Error("Compact octree QA fine directory is malformed or not compact-ranked");
    }
    const key = metadata[base + 1], directoryBase = 7 + plan.maximumResidentBricks;
    if (key >= plan.logicalBrickCount || worklist[directoryBase + key] !== physicalId) {
      throw new Error("Compact octree QA fine direct directory is malformed");
    }
  }
  const rowCapacity = (coarseDirectory.length - 8) / 8;
  if (!Number.isSafeInteger(rowCapacity) || rowCount < 1 || rowCount > rowCapacity) {
    throw new Error("Compact octree QA coarse directory has an invalid row count");
  }
  if (dimensions.some((value, axis) => value !== coarseDirectory[4 + axis]
    || value !== plan.finestCellDimensions[axis])) {
    throw new Error("Compact octree QA publication dimensions differ from the requested field");
  }
  let priorLevel = -1, priorMorton = -1;
  for (let slot = 0; slot < rowCount; slot += 1) {
    const base = 8 + slot * 8, cellPlusOne = coarseDirectory[base], size = coarseDirectory[base + 1];
    if (cellPlusOne === 0 || size === 0 || (size & (size - 1)) !== 0) {
      throw new Error("Compact octree QA coarse directory contains a malformed row");
    }
    const level = 31 - Math.clz32(size), morton = coarseMorton(cellPlusOne - 1, dimensions);
    if (level < priorLevel || (level === priorLevel && morton <= priorMorton)) {
      throw new Error("Compact octree QA coarse directory is not strictly sorted");
    }
    priorLevel = level; priorMorton = morton;
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
  const { plan, generation, metadata, flags, phi, worklist } = snapshot;
  const q = position.map((value, axis) => Math.floor((value - plan.domainOrigin[axis]) / plan.fineCellWidth));
  if (q.some((value, axis) => value < 0 || value >= plan.sampleDimensions[axis])) return undefined;
  const brick = q.map((value) => Math.floor(value / plan.brickResolution));
  const local = q.map((value, axis) => value - brick[axis] * plan.brickResolution);
  const key = brick[0] + plan.brickDimensions[0] * (brick[1] + plan.brickDimensions[1] * brick[2]);
  if (worklist.length < 7 || worklist[0] !== generation || worklist[2] !== plan.maximumResidentBricks
    || (worklist[3] & 3) !== 3 || worklist[5] !== 1 || worklist[6] !== 1) return undefined;
  const directoryBase = 7 + plan.maximumResidentBricks;
  if (key >= plan.logicalBrickCount || directoryBase + key >= worklist.length) return undefined;
  const physicalId = worklist[directoryBase + key], base = physicalId * 10;
  if (physicalId >= plan.maximumResidentBricks || base + 2 >= metadata.length
    || metadata[base] !== physicalId || metadata[base + 1] !== key
    || metadata[base + 2] !== generation) return undefined;
  const localIndex = local[0] + plan.brickResolution * (local[1] + plan.brickResolution * local[2]);
  const sampleIndex = physicalId * plan.samplesPerBrick + localIndex;
  const value = phi[sampleIndex];
  return (flags[sampleIndex] & FINE_LEVELSET_SAMPLE_FLAGS.valid) !== 0 && Number.isFinite(value)
    ? value : undefined;
}

function coarsePhiAt(snapshot: CompactOctreeFieldSnapshot, position: readonly [number, number, number]): { phi: number; positiveAir: boolean } {
  const words = snapshot.coarseDirectory;
  const rowCount = words[2], maximumLeafSize = words[3];
  const dimensions = [words[4], words[5], words[6]] as const;
  const physicalCellSize = finiteFloat(words, 7);
  const q = position.map((value) => Math.floor(value / physicalCellSize));
  if (q.some((value, axis) => value < 0 || value >= dimensions[axis])) {
    throw new Error("Compact octree QA sample lies outside the coarse publication");
  }
  for (let size = 1; size <= maximumLeafSize; size *= 2) {
    const origin = q.map((value) => Math.floor(value / size) * size);
    const cell = origin[0] + dimensions[0] * (origin[1] + dimensions[1] * origin[2]);
    const wantedLevel = 31 - Math.clz32(size), wantedMorton = coarseMorton(cell, dimensions);
    let low = 0, high = rowCount;
    while (low < high) {
      const middle = low + Math.floor((high - low) / 2), base = 8 + middle * 8;
      const entryLevel = 31 - Math.clz32(words[base + 1]);
      const entryMorton = coarseMorton(words[base] - 1, dimensions);
      if (entryLevel < wantedLevel || (entryLevel === wantedLevel && entryMorton < wantedMorton)) low = middle + 1;
      else high = middle;
    }
    if (low < rowCount) {
      const base = 8 + low * 8;
      if (words[base] === cell + 1 && words[base + 1] === size) {
        const value = finiteFloat(words, base + 2);
        if ((words[base + 5] & OCTREE_COARSE_PHI_FLAG.valid) === 0 || !Number.isFinite(value)) {
          throw new Error("Compact octree QA encountered an invalid containing coarse leaf");
        }
        return { phi: value, positiveAir: false };
      }
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
  const activePages = Math.min(snapshot.worklist[1], snapshot.plan.maximumResidentBricks);
  let malformedActivePages = 0, validSamples = 0, finiteValidSamples = 0;
  let negativeValidSamples = 0, positiveValidSamples = 0;
  for (let work = 0; work < activePages; work += 1) {
    const id = snapshot.worklist[7 + work];
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
    publicationValid: snapshot.worklist[0] === snapshot.generation
      && (snapshot.worklist[3] & 3) === 3 && snapshot.worklist[5] === 1 && snapshot.worklist[6] === 1
      && activePages > 0 && malformedActivePages === 0
      && validSamples > 0 && finiteValidSamples === validSamples,
    coarseGeneration: snapshot.coarseDirectory[1],
    retainedCoarseAuthority: topology?.[4] === 1 && topology?.[5] === 1,
    ...(topology ? { topologyFlags: topology[0], topologyPublished: topology[4] !== 0,
      topologyRolledBack: topology[5] !== 0, downstreamFinalizeReason: topology[7] } : {}),
  };
}
