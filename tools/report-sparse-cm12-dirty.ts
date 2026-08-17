#!/usr/bin/env node
/**
 * Strict reporter for the Sparse CM12 temporal-dirty receipt.
 *
 * This tool deliberately owns no simulator imports. A captured benchmark can
 * feed it a JSON receipt and get the same census in CI, an artifact, or a local
 * SVG without constructing WebGPU. Missing provenance is a gate failure, not a
 * zero and not an omitted bar.
 */
import { readFileSync } from "node:fs";

export const SPARSE_CM12_DIRTY_RECEIPT_VERSION = 1 as const;
export const SPARSE_CM12_DIRTY_PACKING_RECEIPT_VERSION = 1 as const;
export const SPARSE_CM12_DIRTY_PACKING_MAX_PACKETS = 32;

export const SPARSE_CM12_DIRTY_STAGES = [
  "face-preparation",
  "mass-transport",
  "gamma-transport",
  "surface-conditioning",
  "pressure-coefficients",
  "presentation",
] as const;
export type SparseCM12DirtyStageId = typeof SPARSE_CM12_DIRTY_STAGES[number];

export const SPARSE_CM12_DIRTY_CAUSES = [
  "topology-created",
  "topology-retired",
  "phase-crossing",
  "density-changed",
  "gamma-changed",
  "velocity-characteristic",
  "cfl-growth",
  "moving-solid-sweep",
  "boundary-source",
  "coefficient-changed",
  "dependency-closure",
  "generation-mismatch",
  "capacity-or-provenance",
] as const;
export type SparseCM12DirtyCause = typeof SPARSE_CM12_DIRTY_CAUSES[number];

export const SPARSE_CM12_DIRTY_ORIGIN_STAGES = [
  "external",
  ...SPARSE_CM12_DIRTY_STAGES,
] as const;
export type SparseCM12DirtyOriginStage = typeof SPARSE_CM12_DIRTY_ORIGIN_STAGES[number];

export interface SparseCM12DirtyStageReceipt {
  readonly inputGenerations: Readonly<Record<string, number>>;
  /** Generation of the dirty producer this stage consumed. */
  readonly producerGeneration: number;
  /** Generation of this stage's output/work publication. */
  readonly consumerGeneration: number;
  readonly eligibleTiles: number;
  readonly reusedTiles: number;
  readonly directDirtyTiles: number;
  readonly closureDirtyTiles: number;
  readonly unknownTiles: number;
  readonly processedTiles: number;
  /** Stage-native rows/edges/subtiles in the equivalent complete schedule. */
  readonly eligibleWorkItems: number;
  /** Stage-native rows/edges/subtiles actually dispatched. */
  readonly executedWorkItems: number;
  readonly skippedWorkItems: number;
  /** Writes observed outside this stage's declared dirty closure. */
  readonly uncoveredWriteFaults: number;
  readonly stage_ms: number;
  /** Overlapping direct-cause attribution: these counts are not a partition. */
  readonly originCauses: Partial<Readonly<Record<SparseCM12DirtyCause, number>>>;
  /** Causes carried through closure or a downstream stage handoff. */
  readonly inheritedCauses: Partial<Readonly<Record<SparseCM12DirtyCause, number>>>;
  /** Where dirt originated before being derived into this stage. */
  readonly derivedFromStages: Partial<Readonly<Record<SparseCM12DirtyOriginStage, number>>>;
  /** Depth zero is direct dirt; positive depths are dependency closure. */
  readonly closureDepth: Readonly<Record<string, number>>;
  readonly blast: {
    readonly rootTiles: number;
    readonly uniqueTouchedTiles: number;
    readonly forwardedTiles: number;
    readonly maximumDepth: number;
    readonly maximumManhattanFineCells: number;
    readonly boundsFine: null | {
      readonly min: readonly [number, number, number];
      /** Exclusive upper bound. */
      readonly max: readonly [number, number, number];
    };
  };
}

export interface SparseCM12DirtyFrameReceipt {
  readonly sequence: number;
  readonly simulationTime_s: number;
  readonly acceptedGeneration: number;
  readonly candidateGeneration: number;
  readonly provenanceGeneration: number;
  readonly publication: "accepted" | "rejected";
  readonly rejectionReasons: readonly string[];
  /** Unique unknown tile identities across stages, not a stage-count sum. */
  readonly unknownTiles: number;
  readonly stages: Readonly<Record<SparseCM12DirtyStageId, SparseCM12DirtyStageReceipt>>;
  /** Optional v1-compatible census of physical packets coalescing logical stages. */
  readonly packing?: SparseCM12DirtyPassPackingReceipt;
}

export interface SparseCM12DirtyPassPacketReceipt {
  /** Dense 0..31 index used by the PKT1 per-tile bitmasks. */
  readonly packet: number;
  /** Stable producer-chosen identity; unlike `packet`, it may survive recompaction. */
  readonly identity: number;
  readonly label: string;
  readonly epoch: number;
  readonly generation: number;
  readonly logicalStages: readonly SparseCM12DirtyStageId[];
  readonly eligibleTiles: number;
  readonly executedTiles: number;
  readonly skippedTiles: number;
  /** Logical stage-tile executions inside this physical packet; counts overlap. */
  readonly stageTileCounts: Partial<Readonly<Record<SparseCM12DirtyStageId, number>>>;
  readonly packet_ms: number;
}

export interface SparseCM12DirtyPassEpochReceipt {
  readonly epoch: number;
  readonly generation: number;
  readonly packetIds: readonly number[];
  readonly dependsOnEpochs: readonly number[];
}

export interface SparseCM12DirtyPassPackingReceipt {
  readonly version: typeof SPARSE_CM12_DIRTY_PACKING_RECEIPT_VERSION;
  readonly generation: number;
  readonly eligibleTiles: number;
  readonly reusedTiles: number;
  /** Exactly one active logical stage and one physical packet. */
  readonly singleStageTiles: number;
  /** Two or more active stages, all assigned to the same physical packet. */
  readonly coalescedTiles: number;
  /** Active logical stages assigned to more than one physical packet. */
  readonly splitTiles: number;
  readonly unknownTiles: number;
  readonly uncoveredPackingFaults: number;
  readonly epochs: readonly SparseCM12DirtyPassEpochReceipt[];
  readonly packets: readonly SparseCM12DirtyPassPacketReceipt[];
}

export interface SparseCM12DirtyReceipt {
  readonly kind: "sparse-cm12-dirty-provenance";
  readonly version: typeof SPARSE_CM12_DIRTY_RECEIPT_VERSION;
  readonly tileEdge: 4;
  readonly frames: readonly SparseCM12DirtyFrameReceipt[];
}

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

function objectAt(parent: Record<string, unknown>, key: string, at: string): Record<string, unknown> {
  const value = parent[key];
  if (!isObject(value)) throw new Error(`${at}.${key} must be an object`);
  return value;
}

function arrayAt(parent: Record<string, unknown>, key: string, at: string): readonly unknown[] {
  const value = parent[key];
  if (!Array.isArray(value)) throw new Error(`${at}.${key} must be an array`);
  return value;
}

function countAt(parent: Record<string, unknown>, key: string, at: string): number {
  const value = parent[key];
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new Error(`${at}.${key} must be a non-negative safe integer`);
  }
  return value as number;
}

function scalarAt(parent: Record<string, unknown>, key: string, at: string): number {
  const value = parent[key];
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new Error(`${at}.${key} must be finite and non-negative`);
  }
  return value;
}

function stringAt(parent: Record<string, unknown>, key: string, at: string): string {
  const value = parent[key];
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${at}.${key} must be a non-empty string`);
  }
  return value;
}

function tuple3(value: unknown, at: string): readonly [number, number, number] {
  if (!Array.isArray(value) || value.length !== 3
    || value.some((item) => !Number.isSafeInteger(item) || item < 0)) {
    throw new Error(`${at} must be a non-negative integer triple`);
  }
  return value as unknown as readonly [number, number, number];
}

function countRecord(value: unknown, at: string, allowedKeys?: readonly string[]): Record<string, number> {
  if (!isObject(value)) throw new Error(`${at} must be an object`);
  const result: Record<string, number> = {};
  for (const [key, item] of Object.entries(value)) {
    if (allowedKeys && !allowedKeys.includes(key)) throw new Error(`${at}.${key} is not a known key`);
    if (!Number.isSafeInteger(item) || (item as number) < 0) {
      throw new Error(`${at}.${key} must be a non-negative safe integer`);
    }
    result[key] = item as number;
  }
  return result;
}

function parseStage(value: unknown, at: string): SparseCM12DirtyStageReceipt {
  if (!isObject(value)) throw new Error(`${at} must be an object`);
  const inputGenerations = countRecord(value.inputGenerations, `${at}.inputGenerations`);
  if (Object.keys(inputGenerations).length === 0) {
    throw new Error(`${at}.inputGenerations must name at least one authority`);
  }
  const eligibleTiles = countAt(value, "eligibleTiles", at);
  const reusedTiles = countAt(value, "reusedTiles", at);
  const directDirtyTiles = countAt(value, "directDirtyTiles", at);
  const closureDirtyTiles = countAt(value, "closureDirtyTiles", at);
  const unknownTiles = countAt(value, "unknownTiles", at);
  const processedTiles = countAt(value, "processedTiles", at);
  const eligibleWorkItems = countAt(value, "eligibleWorkItems", at);
  const executedWorkItems = countAt(value, "executedWorkItems", at);
  const skippedWorkItems = countAt(value, "skippedWorkItems", at);
  const uncoveredWriteFaults = countAt(value, "uncoveredWriteFaults", at);
  const partition = reusedTiles + directDirtyTiles + closureDirtyTiles + unknownTiles;
  if (partition !== eligibleTiles) {
    throw new Error(`${at} tile census ${partition} does not equal eligibleTiles ${eligibleTiles}`);
  }
  if (processedTiles !== directDirtyTiles + closureDirtyTiles) {
    throw new Error(`${at}.processedTiles must equal directDirtyTiles + closureDirtyTiles`);
  }
  if (executedWorkItems + skippedWorkItems !== eligibleWorkItems) {
    throw new Error(`${at} executedWorkItems + skippedWorkItems must equal eligibleWorkItems`);
  }
  const originCauses = countRecord(
    value.originCauses, `${at}.originCauses`, SPARSE_CM12_DIRTY_CAUSES);
  const inheritedCauses = countRecord(
    value.inheritedCauses, `${at}.inheritedCauses`, SPARSE_CM12_DIRTY_CAUSES);
  const derivedFromStages = countRecord(
    value.derivedFromStages, `${at}.derivedFromStages`, SPARSE_CM12_DIRTY_ORIGIN_STAGES);
  const closureDepth = countRecord(value.closureDepth, `${at}.closureDepth`);
  for (const depth of Object.keys(closureDepth)) {
    if (!/^\d+$/.test(depth)) throw new Error(`${at}.closureDepth.${depth} is not an integer depth`);
  }
  if ((closureDepth["0"] ?? 0) !== directDirtyTiles) {
    throw new Error(`${at}.closureDepth[0] must equal directDirtyTiles`);
  }
  const positiveDepth = Object.entries(closureDepth)
    .filter(([depth]) => Number(depth) > 0)
    .reduce((sum, [, count]) => sum + count, 0);
  if (positiveDepth !== closureDirtyTiles) {
    throw new Error(`${at} positive closure-depth bins must sum to closureDirtyTiles`);
  }
  const blastObject = objectAt(value, "blast", at);
  const boundsValue = blastObject.boundsFine;
  let boundsFine: SparseCM12DirtyStageReceipt["blast"]["boundsFine"] = null;
  if (boundsValue !== null) {
    if (!isObject(boundsValue)) throw new Error(`${at}.blast.boundsFine must be null or an object`);
    const min = tuple3(boundsValue.min, `${at}.blast.boundsFine.min`);
    const max = tuple3(boundsValue.max, `${at}.blast.boundsFine.max`);
    if (max.some((item, axis) => item <= min[axis])) {
      throw new Error(`${at}.blast.boundsFine max must be greater than min`);
    }
    boundsFine = { min, max };
  }
  const blast = {
    rootTiles: countAt(blastObject, "rootTiles", `${at}.blast`),
    uniqueTouchedTiles: countAt(blastObject, "uniqueTouchedTiles", `${at}.blast`),
    forwardedTiles: countAt(blastObject, "forwardedTiles", `${at}.blast`),
    maximumDepth: countAt(blastObject, "maximumDepth", `${at}.blast`),
    maximumManhattanFineCells: countAt(blastObject, "maximumManhattanFineCells", `${at}.blast`),
    boundsFine,
  };
  if (blast.rootTiles !== directDirtyTiles) {
    throw new Error(`${at}.blast.rootTiles must equal directDirtyTiles`);
  }
  if (blast.uniqueTouchedTiles !== processedTiles) {
    throw new Error(`${at}.blast.uniqueTouchedTiles must equal processedTiles`);
  }
  const observedMaximumDepth = Math.max(0, ...Object.keys(closureDepth).map(Number));
  if (blast.maximumDepth !== observedMaximumDepth) {
    throw new Error(`${at}.blast.maximumDepth does not match closureDepth`);
  }
  if (processedTiles === 0 && boundsFine !== null) {
    throw new Error(`${at}.blast.boundsFine must be null for an empty blast`);
  }
  if (processedTiles > 0 && boundsFine === null) {
    throw new Error(`${at}.blast.boundsFine is required for a non-empty blast`);
  }
  return {
    inputGenerations,
    producerGeneration: countAt(value, "producerGeneration", at),
    consumerGeneration: countAt(value, "consumerGeneration", at),
    eligibleTiles, reusedTiles, directDirtyTiles, closureDirtyTiles,
    unknownTiles, processedTiles, eligibleWorkItems, executedWorkItems,
    skippedWorkItems, uncoveredWriteFaults,
    stage_ms: scalarAt(value, "stage_ms", at), originCauses, inheritedCauses,
    derivedFromStages,
    closureDepth, blast,
  };
}

function integerArray(value: unknown, at: string): readonly number[] {
  if (!Array.isArray(value)) throw new Error(`${at} must be an array`);
  return value.map((item, index) => {
    if (!Number.isSafeInteger(item) || (item as number) < 0) {
      throw new Error(`${at}[${index}] must be a non-negative safe integer`);
    }
    return item as number;
  });
}

function stageArray(value: unknown, at: string): readonly SparseCM12DirtyStageId[] {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error(`${at} must be a non-empty stage array`);
  }
  const stages = value.map((item, index) => {
    if (typeof item !== "string"
      || !(SPARSE_CM12_DIRTY_STAGES as readonly string[]).includes(item)) {
      throw new Error(`${at}[${index}] is not a known logical stage`);
    }
    return item as SparseCM12DirtyStageId;
  });
  if (new Set(stages).size !== stages.length) throw new Error(`${at} contains a duplicate stage`);
  return stages;
}

function parsePassPacking(
  value: unknown,
  at: string,
  candidateGeneration: number,
  publication: "accepted" | "rejected",
): SparseCM12DirtyPassPackingReceipt {
  if (!isObject(value)) throw new Error(`${at} must be an object`);
  if (value.version !== SPARSE_CM12_DIRTY_PACKING_RECEIPT_VERSION) {
    throw new Error(`${at}.version must be ${SPARSE_CM12_DIRTY_PACKING_RECEIPT_VERSION}`);
  }
  const generation = countAt(value, "generation", at);
  const eligibleTiles = countAt(value, "eligibleTiles", at);
  const reusedTiles = countAt(value, "reusedTiles", at);
  const singleStageTiles = countAt(value, "singleStageTiles", at);
  const coalescedTiles = countAt(value, "coalescedTiles", at);
  const splitTiles = countAt(value, "splitTiles", at);
  const unknownTiles = countAt(value, "unknownTiles", at);
  const uncoveredPackingFaults = countAt(value, "uncoveredPackingFaults", at);
  const partition = reusedTiles + singleStageTiles + coalescedTiles + splitTiles + unknownTiles;
  if (partition !== eligibleTiles) {
    throw new Error(`${at} packing tile census ${partition} does not equal eligibleTiles ${eligibleTiles}`);
  }

  const rawPackets = arrayAt(value, "packets", at);
  if (rawPackets.length > SPARSE_CM12_DIRTY_PACKING_MAX_PACKETS) {
    throw new Error(`${at}.packets exceeds the PKT1 32-packet mask`);
  }
  const packetIds = new Set<number>();
  const packets = rawPackets.map((rawPacket, index): SparseCM12DirtyPassPacketReceipt => {
    const packetAt = `${at}.packets[${index}]`;
    if (!isObject(rawPacket)) throw new Error(`${packetAt} must be an object`);
    const packet = countAt(rawPacket, "packet", packetAt);
    if (packet >= SPARSE_CM12_DIRTY_PACKING_MAX_PACKETS) {
      throw new Error(`${packetAt}.packet must fit the PKT1 32-bit packet mask`);
    }
    if (packetIds.has(packet)) throw new Error(`${packetAt}.packet is duplicated`);
    packetIds.add(packet);
    const logicalStages = stageArray(rawPacket.logicalStages, `${packetAt}.logicalStages`);
    const stageTileCounts = countRecord(rawPacket.stageTileCounts,
      `${packetAt}.stageTileCounts`, SPARSE_CM12_DIRTY_STAGES);
    for (const stage of logicalStages) {
      if (stageTileCounts[stage] === undefined) {
        throw new Error(`${packetAt}.stageTileCounts must include ${stage}`);
      }
    }
    for (const stage of Object.keys(stageTileCounts)) {
      if (!logicalStages.includes(stage as SparseCM12DirtyStageId)) {
        throw new Error(`${packetAt}.stageTileCounts.${stage} is outside logicalStages`);
      }
    }
    const eligible = countAt(rawPacket, "eligibleTiles", packetAt);
    const executed = countAt(rawPacket, "executedTiles", packetAt);
    const skipped = countAt(rawPacket, "skippedTiles", packetAt);
    if (executed + skipped !== eligible) {
      throw new Error(`${packetAt} executedTiles + skippedTiles must equal eligibleTiles`);
    }
    const stageExecutions = Object.values(stageTileCounts).reduce((sum, count) => sum + count, 0);
    const maximumStageExecutions = Math.max(0, ...Object.values(stageTileCounts));
    if (maximumStageExecutions > executed || (executed > 0 && stageExecutions < executed)) {
      throw new Error(`${packetAt}.stageTileCounts cannot cover executedTiles`);
    }
    return {
      packet, identity: countAt(rawPacket, "identity", packetAt),
      label: stringAt(rawPacket, "label", packetAt),
      epoch: countAt(rawPacket, "epoch", packetAt),
      generation: countAt(rawPacket, "generation", packetAt), logicalStages,
      eligibleTiles: eligible, executedTiles: executed, skippedTiles: skipped,
      stageTileCounts, packet_ms: scalarAt(rawPacket, "packet_ms", packetAt),
    };
  });
  const sortedPacketIds = [...packetIds].sort((a, b) => a - b);
  if (sortedPacketIds.some((packet, index) => packet !== index)) {
    throw new Error(`${at}.packets must use dense indices 0..packetCount-1`);
  }

  const rawEpochs = arrayAt(value, "epochs", at);
  const epochIds = new Set<number>();
  const epochs = rawEpochs.map((rawEpoch, index): SparseCM12DirtyPassEpochReceipt => {
    const epochAt = `${at}.epochs[${index}]`;
    if (!isObject(rawEpoch)) throw new Error(`${epochAt} must be an object`);
    const epoch = countAt(rawEpoch, "epoch", epochAt);
    if (epochIds.has(epoch)) throw new Error(`${epochAt}.epoch is duplicated`);
    epochIds.add(epoch);
    const ids = integerArray(rawEpoch.packetIds, `${epochAt}.packetIds`);
    if (new Set(ids).size !== ids.length) throw new Error(`${epochAt}.packetIds contains duplicates`);
    const dependencies = integerArray(rawEpoch.dependsOnEpochs, `${epochAt}.dependsOnEpochs`);
    if (new Set(dependencies).size !== dependencies.length) {
      throw new Error(`${epochAt}.dependsOnEpochs contains duplicates`);
    }
    if (dependencies.includes(epoch)) throw new Error(`${epochAt} cannot depend on itself`);
    return { epoch, generation: countAt(rawEpoch, "generation", epochAt),
      packetIds: ids, dependsOnEpochs: dependencies };
  });
  if (packets.length > 0 && epochs.length === 0) throw new Error(`${at}.epochs is empty`);
  const epochPackets = new Map<number, Set<number>>();
  for (const epoch of epochs) {
    if (epoch.generation !== generation) throw new Error(`${at}.epochs generation mismatch`);
    for (const dependency of epoch.dependsOnEpochs) {
      if (!epochIds.has(dependency)) throw new Error(`${at}.epochs references missing dependency ${dependency}`);
    }
    const ids = new Set(epoch.packetIds);
    epochPackets.set(epoch.epoch, ids);
    for (const packet of ids) {
      if (!packetIds.has(packet)) throw new Error(`${at}.epochs references missing packet ${packet}`);
    }
  }
  const dependenciesByEpoch = new Map(epochs.map((epoch) => [epoch.epoch, epoch.dependsOnEpochs]));
  const visiting = new Set<number>();
  const visited = new Set<number>();
  const visitEpoch = (epoch: number): void => {
    if (visited.has(epoch)) return;
    if (visiting.has(epoch)) throw new Error(`${at}.epochs dependency graph contains a cycle`);
    visiting.add(epoch);
    for (const dependency of dependenciesByEpoch.get(epoch) ?? []) visitEpoch(dependency);
    visiting.delete(epoch);
    visited.add(epoch);
  };
  for (const epoch of epochIds) visitEpoch(epoch);
  const packetMembershipCount = new Map<number, number>();
  for (const epoch of epochs) for (const packet of epoch.packetIds) {
    packetMembershipCount.set(packet, (packetMembershipCount.get(packet) ?? 0) + 1);
  }
  for (const packet of packets) {
    if (packet.generation !== generation) throw new Error(`${at}.packets generation mismatch`);
    if (!epochPackets.get(packet.epoch)?.has(packet.packet)) {
      throw new Error(`${at}.packets[${packet.packet}] is absent from epoch ${packet.epoch}`);
    }
    if (packetMembershipCount.get(packet.packet) !== 1) {
      throw new Error(`${at}.packets[${packet.packet}] must belong to exactly one epoch`);
    }
  }
  if (publication === "accepted") {
    if (generation !== candidateGeneration) throw new Error(`${at} accepted stale packing generation`);
    if (unknownTiles !== 0 || uncoveredPackingFaults !== 0) {
      throw new Error(`${at} accepted incomplete pass-packing provenance`);
    }
  }
  return {
    version: value.version, generation, eligibleTiles, reusedTiles, singleStageTiles,
    coalescedTiles, splitTiles, unknownTiles, uncoveredPackingFaults, epochs, packets,
  };
}

/** Parse and cross-check a dirty receipt. Accepted incomplete provenance throws. */
export function parseSparseCM12DirtyReceipt(value: unknown): SparseCM12DirtyReceipt {
  if (!isObject(value)) throw new Error("receipt must be an object");
  if (value.kind !== "sparse-cm12-dirty-provenance") throw new Error("unexpected receipt kind");
  if (value.version !== SPARSE_CM12_DIRTY_RECEIPT_VERSION) {
    throw new Error(`unsupported dirty receipt version ${String(value.version)}`);
  }
  if (value.tileEdge !== 4) throw new Error("dirty receipt tileEdge must be 4");
  const rawFrames = arrayAt(value, "frames", "receipt");
  if (rawFrames.length === 0) throw new Error("receipt.frames must not be empty");
  const sequences = new Set<number>();
  const frames = rawFrames.map((rawFrame, frameIndex): SparseCM12DirtyFrameReceipt => {
    const at = `receipt.frames[${frameIndex}]`;
    if (!isObject(rawFrame)) throw new Error(`${at} must be an object`);
    const sequence = countAt(rawFrame, "sequence", at);
    if (sequences.has(sequence)) throw new Error(`${at}.sequence is duplicated`);
    sequences.add(sequence);
    const acceptedGeneration = countAt(rawFrame, "acceptedGeneration", at);
    const candidateGeneration = countAt(rawFrame, "candidateGeneration", at);
    const provenanceGeneration = countAt(rawFrame, "provenanceGeneration", at);
    const publication = stringAt(rawFrame, "publication", at);
    if (publication !== "accepted" && publication !== "rejected") {
      throw new Error(`${at}.publication must be accepted or rejected`);
    }
    const rejectionReasons = arrayAt(rawFrame, "rejectionReasons", at).map((reason, index) => {
      if (typeof reason !== "string" || reason.length === 0) {
        throw new Error(`${at}.rejectionReasons[${index}] must be a non-empty string`);
      }
      return reason;
    });
    const unknownTiles = countAt(rawFrame, "unknownTiles", at);
    const rawStages = objectAt(rawFrame, "stages", at);
    for (const key of Object.keys(rawStages)) {
      if (!(SPARSE_CM12_DIRTY_STAGES as readonly string[]).includes(key)) {
        throw new Error(`${at}.stages.${key} is not a known stage`);
      }
    }
    const stages = Object.fromEntries(SPARSE_CM12_DIRTY_STAGES.map((stage) => [
      stage, parseStage(rawStages[stage], `${at}.stages.${stage}`),
    ])) as unknown as SparseCM12DirtyFrameReceipt["stages"];
    const stageUnknownMaximum = Math.max(...SPARSE_CM12_DIRTY_STAGES.map(
      (stage) => stages[stage].unknownTiles));
    if (unknownTiles < stageUnknownMaximum) {
      throw new Error(`${at}.unknownTiles cannot be below a stage's unknown count`);
    }
    if (publication === "accepted") {
      if (unknownTiles !== 0 || stageUnknownMaximum !== 0) {
        throw new Error(`${at} accepted incomplete dirty provenance`);
      }
      if (provenanceGeneration !== candidateGeneration) {
        throw new Error(`${at} accepted a stale provenance generation`);
      }
      if (rejectionReasons.length !== 0) {
        throw new Error(`${at} accepted publication has rejection reasons`);
      }
      for (const stage of SPARSE_CM12_DIRTY_STAGES) {
        if (stages[stage].consumerGeneration !== candidateGeneration) {
          throw new Error(`${at}.stages.${stage} accepted a stale consumer generation`);
        }
        if (stages[stage].uncoveredWriteFaults !== 0) {
          throw new Error(`${at}.stages.${stage} accepted an uncovered write`);
        }
      }
    } else if (rejectionReasons.length === 0) {
      throw new Error(`${at} rejected publication must explain why`);
    }
    const packing = rawFrame.packing === undefined ? undefined
      : parsePassPacking(rawFrame.packing, `${at}.packing`, candidateGeneration, publication);
    if (packing) {
      for (const stage of SPARSE_CM12_DIRTY_STAGES) {
        if (stages[stage].eligibleTiles !== packing.eligibleTiles) {
          throw new Error(`${at}.packing.eligibleTiles disagrees with stages.${stage}`);
        }
        const packedStageTiles = packing.packets.reduce((sum, packet) =>
          sum + (packet.stageTileCounts[stage] ?? 0), 0);
        if (packedStageTiles !== stages[stage].processedTiles) {
          throw new Error(`${at}.packing packet coverage disagrees with stages.${stage}.processedTiles`);
        }
      }
    }
    return {
      sequence, simulationTime_s: scalarAt(rawFrame, "simulationTime_s", at),
      acceptedGeneration, candidateGeneration, provenanceGeneration,
      publication, rejectionReasons, unknownTiles, stages, packing,
    };
  });
  frames.sort((a, b) => a.sequence - b.sequence);
  return { kind: value.kind, version: value.version, tileEdge: value.tileEdge, frames };
}

const count = (value: number): string => value.toLocaleString("en-NZ");
const percent = (part: number, whole: number): string =>
  whole > 0 ? `${(100 * part / whole).toFixed(1)}%` : "—";
const ratio = (numerator: number, denominator: number): string =>
  denominator > 0 ? (numerator / denominator).toFixed(2) : numerator > 0 ? "∞" : "0.00";
const label = (value: string): string => value.replaceAll("-", " ");

export function sparseCM12DirtyMarkdown(receipt: SparseCM12DirtyReceipt): string {
  const lines = [
    "# Sparse CM12 dirty-provenance report", "",
    `Frames: ${count(receipt.frames.length)} · work tile: ${receipt.tileEdge}³ fine cells`, "",
    "## Timeline", "",
    "| frame | publication | generations accepted/candidate/provenance | unknown | "
      + SPARSE_CM12_DIRTY_STAGES.map(label).join(" | ") + " |",
    "|---:|:---|:---|---:|" + SPARSE_CM12_DIRTY_STAGES.map(() => "---:").join("|") + "|",
  ];
  for (const frame of receipt.frames) {
    lines.push(`| ${frame.sequence} | ${frame.publication.toUpperCase()} | `
      + `${frame.acceptedGeneration}/${frame.candidateGeneration}/${frame.provenanceGeneration} | `
      + `${count(frame.unknownTiles)} | `
      + SPARSE_CM12_DIRTY_STAGES.map((stage) => {
        const value = frame.stages[stage];
        return percent(value.processedTiles + value.unknownTiles, value.eligibleTiles);
      }).join(" | ") + " |");
  }
  lines.push("", "Cells show dirty-or-unknown share of eligible 4³ work tiles.", "", "## Stage census", "",
    "| stage | eligible | reused | direct | closure | unknown | dirty | closure/direct | work exec/skip | uncovered writes | ms | max blast |",
    "|:---|---:|---:|---:|---:|---:|---:|---:|:---|---:|---:|:---|");
  for (const stage of SPARSE_CM12_DIRTY_STAGES) {
    const values = receipt.frames.map((frame) => frame.stages[stage]);
    const sum = (pick: (value: SparseCM12DirtyStageReceipt) => number) =>
      values.reduce((total, value) => total + pick(value), 0);
    const eligible = sum((value) => value.eligibleTiles);
    const direct = sum((value) => value.directDirtyTiles);
    const closure = sum((value) => value.closureDirtyTiles);
    const maximum = values.reduce((best, value) =>
      value.blast.maximumManhattanFineCells > best.blast.maximumManhattanFineCells ? value : best,
    values[0]);
    const bounds = maximum.blast.boundsFine;
    const blast = bounds
      ? `${maximum.blast.maximumManhattanFineCells} cells; [${bounds.min.join(",")})–[${bounds.max.join(",")})`
      : "empty";
    lines.push(`| ${label(stage)} | ${count(eligible)} | ${count(sum((value) => value.reusedTiles))} | `
      + `${count(direct)} | ${count(closure)} | ${count(sum((value) => value.unknownTiles))} | `
      + `${percent(direct + closure, eligible)} | ${ratio(closure, direct)}× | `
      + `${count(sum((value) => value.executedWorkItems))}/${count(sum((value) => value.skippedWorkItems))} | `
      + `${count(sum((value) => value.uncoveredWriteFaults))} | `
      + `${sum((value) => value.stage_ms).toFixed(3)} | ${blast} |`);
  }
  lines.push("", "## Physical pass packing", "",
    "PKT1 is an optional CMD1 v1 tail. `unavailable` means the older v1 producer remains readable; it is not evidence that stages shared a pass.", "",
    "| frame | generation | tile classes reused/single/coalesced/split/unknown | epochs | packets | logical stage-tile executions | physical packet-tile executions | coalescing savings | faults |",
    "|---:|---:|:---|---:|---:|---:|---:|---:|---:|");
  for (const frame of receipt.frames) {
    const packing = frame.packing;
    if (!packing) {
      lines.push(`| ${frame.sequence} | — | unavailable | — | — | — | — | — | — |`);
      continue;
    }
    const logicalExecutions = packing.packets.reduce((sum, packet) => sum
      + Object.values(packet.stageTileCounts).reduce((packetSum, value) => packetSum + value, 0), 0);
    const physicalExecutions = packing.packets.reduce((sum, packet) => sum + packet.executedTiles, 0);
    lines.push(`| ${frame.sequence} | ${packing.generation} | `
      + `${count(packing.reusedTiles)}/${count(packing.singleStageTiles)}/${count(packing.coalescedTiles)}/${count(packing.splitTiles)}/${count(packing.unknownTiles)} | `
      + `${count(packing.epochs.length)} | ${count(packing.packets.length)} | ${count(logicalExecutions)} | `
      + `${count(physicalExecutions)} | ${count(Math.max(0, logicalExecutions - physicalExecutions))} | `
      + `${count(packing.uncoveredPackingFaults)} |`);
  }
  for (const frame of receipt.frames) {
    if (!frame.packing) continue;
    lines.push("", `### Frame ${frame.sequence} packet epochs`, "",
      "| epoch | depends on | physical packets |", "|---:|:---|:---|");
    for (const epoch of [...frame.packing.epochs].sort((a, b) => a.epoch - b.epoch)) {
      lines.push(`| ${epoch.epoch} | ${epoch.dependsOnEpochs.length ? epoch.dependsOnEpochs.join(", ") : "root"} | ${epoch.packetIds.join(", ") || "none"} |`);
    }
    lines.push("", "| packet | identity | epoch | logical stages | tiles executed/skipped | logical stage-tiles | coalesced stage-tiles | ms |",
      "|---:|---:|---:|:---|:---|---:|---:|---:|");
    for (const packet of [...frame.packing.packets].sort((a, b) => a.packet - b.packet)) {
      const logical = Object.values(packet.stageTileCounts).reduce((sum, value) => sum + value, 0);
      lines.push(`| ${packet.packet} ${label(packet.label)} | ${packet.identity} | ${packet.epoch} | `
        + `${packet.logicalStages.map(label).join(" + ")} | ${count(packet.executedTiles)}/${count(packet.skippedTiles)} | `
        + `${count(logical)} | ${count(Math.max(0, logical - packet.executedTiles))} | ${packet.packet_ms.toFixed(3)} |`);
    }
  }
  const causeTotals = new Map<string, number>();
  const inheritedCauseTotals = new Map<string, number>();
  const originStageTotals = new Map<string, number>();
  const depthTotals = new Map<number, number>();
  for (const frame of receipt.frames) for (const stage of SPARSE_CM12_DIRTY_STAGES) {
    for (const [cause, value] of Object.entries(frame.stages[stage].originCauses)) {
      causeTotals.set(cause, (causeTotals.get(cause) ?? 0) + value);
    }
    for (const [cause, value] of Object.entries(frame.stages[stage].inheritedCauses)) {
      inheritedCauseTotals.set(cause, (inheritedCauseTotals.get(cause) ?? 0) + value);
    }
    for (const [origin, value] of Object.entries(frame.stages[stage].derivedFromStages)) {
      originStageTotals.set(origin, (originStageTotals.get(origin) ?? 0) + value);
    }
    for (const [depth, value] of Object.entries(frame.stages[stage].closureDepth)) {
      depthTotals.set(Number(depth), (depthTotals.get(Number(depth)) ?? 0) + value);
    }
  }
  lines.push("", "## Origin-cause census", "",
    "Cause counts overlap when one tile has multiple reasons.", "",
    "| origin cause | direct stage-tile attributions |", "|:---|---:|",
    ...[...causeTotals.entries()].sort((a, b) => b[1] - a[1])
      .map(([cause, value]) => `| ${label(cause)} | ${count(value)} |`),
    "", "## Inherited-cause census", "",
    "| inherited cause | closure/downstream stage-tile attributions |", "|:---|---:|",
    ...[...inheritedCauseTotals.entries()].sort((a, b) => b[1] - a[1])
      .map(([cause, value]) => `| ${label(cause)} | ${count(value)} |`),
    "", "## Derived-stage census", "",
    "| origin stage | derived stage-tile attributions |", "|:---|---:|",
    ...[...originStageTotals.entries()].sort((a, b) => b[1] - a[1])
      .map(([origin, value]) => `| ${label(origin)} | ${count(value)} |`),
    "", "## Closure-depth census", "", "| dependency depth | tiles |", "|---:|---:|",
    ...[...depthTotals.entries()].sort((a, b) => a[0] - b[0])
      .map(([depth, value]) => `| ${depth} | ${count(value)} |`));
  const rejected = receipt.frames.filter((frame) => frame.publication === "rejected");
  if (rejected.length > 0) {
    lines.push("", "## Fail-closed publications", "",
      ...rejected.map((frame) => `- Frame ${frame.sequence}: ${frame.rejectionReasons.join("; ")}`));
  }
  return `${lines.join("\n")}\n`;
}

const escapeXml = (value: string): string => value.replace(/[&<>"']/g, (character) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&apos;",
})[character]!);

/** A dependency-free SVG dashboard: temporal heatmap plus per-stage census. */
export function sparseCM12DirtySvg(receipt: SparseCM12DirtyReceipt): string {
  const marginLeft = 190;
  const cellWidth = Math.max(10, Math.min(30, 900 / receipt.frames.length));
  const timelineWidth = receipt.frames.length * cellWidth;
  const width = Math.max(920, marginLeft + timelineWidth + 40);
  const rowHeight = 30;
  const timelineTop = 108;
  const censusTop = timelineTop + SPARSE_CM12_DIRTY_STAGES.length * rowHeight + 90;
  const height = censusTop + SPARSE_CM12_DIRTY_STAGES.length * 42 + 100;
  const packingFrames = receipt.frames.filter((frame) => frame.packing !== undefined);
  const coalescedTiles = packingFrames.reduce((sum, frame) => sum + frame.packing!.coalescedTiles, 0);
  const parts = [`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
    `<rect width="100%" height="100%" fill="#08111f"/>`,
    `<style>text{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;fill:#dbeafe}.muted{fill:#8192aa}.bad{fill:#ff2fa4}.title{font-size:18px;font-weight:700}</style>`,
    `<text x="24" y="34" class="title">Sparse CM12 dirty provenance</text>`,
    `<text x="24" y="58" class="muted">4³ work tiles · gold direct · blue closure · green reused · magenta unknown/rejected</text>`,
    `<text x="24" y="78" class="muted">PKT1 pass packing: ${packingFrames.length}/${receipt.frames.length} frames · ${coalescedTiles} fully coalesced tiles · missing extension remains unavailable</text>`];
  for (let stageIndex = 0; stageIndex < SPARSE_CM12_DIRTY_STAGES.length; stageIndex += 1) {
    const stage = SPARSE_CM12_DIRTY_STAGES[stageIndex];
    const y = timelineTop + stageIndex * rowHeight;
    parts.push(`<text x="24" y="${y + 19}" font-size="12">${escapeXml(label(stage))}</text>`);
    receipt.frames.forEach((frame, frameIndex) => {
      const value = frame.stages[stage];
      const x = marginLeft + frameIndex * cellWidth;
      const total = Math.max(1, value.eligibleTiles);
      const shares: readonly [number, string][] = [
        [value.reusedTiles / total, "#174f3b"],
        [value.directDirtyTiles / total, "#f5a524"],
        [value.closureDirtyTiles / total, "#2979ff"],
        [value.unknownTiles / total, "#ff2fa4"],
      ];
      let offset = 0;
      for (const [share, color] of shares) {
        const segment = Math.max(0, share * (cellWidth - 1));
        if (segment > 0) parts.push(`<rect x="${(x + offset).toFixed(2)}" y="${y}" width="${segment.toFixed(2)}" height="22" fill="${color}"/>`);
        offset += segment;
      }
      const title = `frame ${frame.sequence}; ${stage}; direct ${value.directDirtyTiles}; closure ${value.closureDirtyTiles}; unknown ${value.unknownTiles}; max depth ${value.blast.maximumDepth}; blast ${value.blast.maximumManhattanFineCells} fine cells`;
      parts.push(`<rect x="${x}" y="${y}" width="${cellWidth - 1}" height="22" fill="none" stroke="${frame.publication === "rejected" ? "#ff2fa4" : "#2b3a50"}" stroke-width="1"><title>${escapeXml(title)}</title></rect>`);
    });
  }
  const tickStride = Math.max(1, Math.ceil(receipt.frames.length / 12));
  receipt.frames.forEach((frame, index) => {
    if (index % tickStride !== 0 && index !== receipt.frames.length - 1) return;
    parts.push(`<text x="${marginLeft + index * cellWidth}" y="${timelineTop - 12}" font-size="10" class="muted">${frame.sequence}</text>`);
  });
  parts.push(`<text x="24" y="${censusTop - 25}" class="title">Census across captured frames</text>`);
  const barLeft = marginLeft;
  const barWidth = width - barLeft - 170;
  for (let stageIndex = 0; stageIndex < SPARSE_CM12_DIRTY_STAGES.length; stageIndex += 1) {
    const stage = SPARSE_CM12_DIRTY_STAGES[stageIndex];
    const y = censusTop + stageIndex * 42;
    const values = receipt.frames.map((frame) => frame.stages[stage]);
    const sum = (pick: (value: SparseCM12DirtyStageReceipt) => number) => values.reduce((total, value) => total + pick(value), 0);
    const eligible = Math.max(1, sum((value) => value.eligibleTiles));
    const direct = sum((value) => value.directDirtyTiles);
    const closure = sum((value) => value.closureDirtyTiles);
    const unknown = sum((value) => value.unknownTiles);
    const reused = sum((value) => value.reusedTiles);
    parts.push(`<text x="24" y="${y + 17}" font-size="12">${escapeXml(label(stage))}</text>`);
    let offset = 0;
    for (const [amount, color] of [[reused, "#174f3b"], [direct, "#f5a524"], [closure, "#2979ff"], [unknown, "#ff2fa4"]] as const) {
      const segment = amount / eligible * barWidth;
      if (segment > 0) parts.push(`<rect x="${(barLeft + offset).toFixed(2)}" y="${y}" width="${segment.toFixed(2)}" height="22" fill="${color}"/>`);
      offset += segment;
    }
    parts.push(`<text x="${barLeft + barWidth + 12}" y="${y + 17}" font-size="11" class="muted">${percent(direct + closure, eligible)} dirty · ${ratio(closure, direct)}× closure</text>`);
  }
  const rejected = receipt.frames.filter((frame) => frame.publication === "rejected").length;
  parts.push(`<text x="24" y="${height - 35}" class="${rejected ? "bad" : "muted"}" font-size="12">${rejected ? `${rejected} fail-closed publication(s); inspect the Markdown report for reasons` : "all captured publications carry complete provenance"}</text>`, "</svg>");
  return `${parts.join("\n")}\n`;
}

function exampleReceipt(): SparseCM12DirtyReceipt {
  const stage = (direct: number, closure: number, stage_ms: number): SparseCM12DirtyStageReceipt => ({
    inputGenerations: { topology: 7, state: 41 },
    producerGeneration: 41, consumerGeneration: 41, eligibleTiles: 64,
    reusedTiles: 64 - direct - closure, directDirtyTiles: direct,
    closureDirtyTiles: closure, unknownTiles: 0, processedTiles: direct + closure,
    eligibleWorkItems: 256, executedWorkItems: (direct + closure) * 4,
    skippedWorkItems: 256 - (direct + closure) * 4, uncoveredWriteFaults: 0, stage_ms,
    originCauses: direct ? { "density-changed": direct } : {},
    inheritedCauses: closure ? { "dependency-closure": closure, "density-changed": closure } : {},
    derivedFromStages: direct + closure ? { external: direct, "mass-transport": closure } : {},
    closureDepth: { "0": direct, ...(closure ? { "1": closure } : {}) },
    blast: {
      rootTiles: direct, uniqueTouchedTiles: direct + closure, forwardedTiles: closure,
      maximumDepth: closure ? 1 : 0, maximumManhattanFineCells: closure ? 8 : direct ? 4 : 0,
      boundsFine: direct + closure ? { min: [8, 0, 8], max: [24, 16, 24] } : null,
    },
  });
  return {
    kind: "sparse-cm12-dirty-provenance", version: 1, tileEdge: 4,
    frames: [{
      sequence: 41, simulationTime_s: 0.164, acceptedGeneration: 40,
      candidateGeneration: 41, provenanceGeneration: 41, publication: "accepted",
      rejectionReasons: [], unknownTiles: 0,
      stages: {
        "face-preparation": stage(3, 7, 0.12), "mass-transport": stage(4, 8, 0.18),
        "gamma-transport": stage(2, 6, 0.09), "surface-conditioning": stage(2, 3, 0.07),
        "pressure-coefficients": stage(3, 11, 0.15), "presentation": stage(2, 4, 0.06),
      },
      packing: {
        version: 1, generation: 41, eligibleTiles: 64, reusedTiles: 50,
        singleStageTiles: 2, coalescedTiles: 8, splitTiles: 4, unknownTiles: 0,
        uncoveredPackingFaults: 0,
        epochs: [
          { epoch: 0, generation: 41, packetIds: [0], dependsOnEpochs: [] },
          { epoch: 1, generation: 41, packetIds: [1], dependsOnEpochs: [0] },
          { epoch: 2, generation: 41, packetIds: [2], dependsOnEpochs: [1] },
        ],
        packets: [
          { packet: 0, identity: 100, label: "transport", epoch: 0, generation: 41,
            logicalStages: ["face-preparation", "mass-transport"],
            eligibleTiles: 64, executedTiles: 12, skippedTiles: 52,
            stageTileCounts: { "face-preparation": 10, "mass-transport": 12 }, packet_ms: 0.24 },
          { packet: 1, identity: 101, label: "conditioning", epoch: 1, generation: 41,
            logicalStages: ["gamma-transport", "surface-conditioning"],
            eligibleTiles: 64, executedTiles: 8, skippedTiles: 56,
            stageTileCounts: { "gamma-transport": 8, "surface-conditioning": 5 }, packet_ms: 0.13 },
          { packet: 2, identity: 102, label: "coefficients-publication", epoch: 2, generation: 41,
            logicalStages: ["pressure-coefficients", "presentation"],
            eligibleTiles: 64, executedTiles: 14, skippedTiles: 50,
            stageTileCounts: { "pressure-coefficients": 14, presentation: 6 }, packet_ms: 0.21 },
        ],
      },
    }],
  };
}

function usage(): string {
  return `Usage: node --import tsx tools/report-sparse-cm12-dirty.ts --input=<receipt.json> [--format=markdown|svg|json]\n`
    + `       node --import tsx tools/report-sparse-cm12-dirty.ts --example\n`;
}

function main(): void {
  const args = process.argv.slice(2);
  if (args.includes("--help") || args.includes("-h")) {
    process.stdout.write(usage());
    return;
  }
  if (args.includes("--example")) {
    process.stdout.write(`${JSON.stringify(exampleReceipt(), null, 2)}\n`);
    return;
  }
  const inputArgument = args.find((argument) => argument.startsWith("--input="));
  if (!inputArgument) throw new Error(`--input is required\n${usage()}`);
  const format = args.find((argument) => argument.startsWith("--format="))?.slice(9) ?? "markdown";
  if (!(["markdown", "svg", "json"] as const).includes(format as "markdown")) {
    throw new Error(`unsupported format ${format}`);
  }
  const inputPath = inputArgument.slice(8);
  const source = inputPath === "-" ? readFileSync(0, "utf8") : readFileSync(inputPath, "utf8");
  const receipt = parseSparseCM12DirtyReceipt(JSON.parse(source) as unknown);
  const output = format === "svg" ? sparseCM12DirtySvg(receipt)
    : format === "json" ? `${JSON.stringify(receipt, null, 2)}\n`
      : sparseCM12DirtyMarkdown(receipt);
  process.stdout.write(output);
  if (receipt.frames.some((frame) => frame.publication === "rejected" || frame.unknownTiles > 0)) {
    process.exitCode = 2;
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    main();
  } catch (error) {
    process.stderr.write(`[sparse-cm12-dirty] FAIL: ${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  }
}
