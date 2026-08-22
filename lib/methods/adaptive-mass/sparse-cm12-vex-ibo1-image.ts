import {
  SPARSE_CM12_FACTORED_AEI_INVALID,
  type SparseCM12FactoredAEICanonicalDescriptor,
} from "./sparse-cm12-factored-aei-topology";
import {
  type SparseCM12VexIBO1Operation,
  type SparseCM12VexIBO1PacketProgram,
  type SparseCM12VexIBO1Program,
  type SparseCM12VexIBO1TemplateProgram,
} from "./sparse-cm12-vex-ibo1-consumer";

export const SPARSE_CM12_VEX_IBO1_IMAGE_MAGIC = 0x5658_4931; // VXI1
export const SPARSE_CM12_VEX_IBO1_IMAGE_VERSION = 1;
export const SPARSE_CM12_VEX_IBO1_IMAGE_HEADER_WORDS = 32;
export const SPARSE_CM12_VEX_IBO1_PACKETS_PER_LEAF = 64;
export const SPARSE_CM12_VEX_IBO1_OPERATION_WORDS = 3;
export const SPARSE_CM12_VEX_IBO1_INVALID = 0xffff_ffff;

export const SPARSE_CM12_VEX_IBO1_IMAGE_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, totalWords: 3,
  canonicalCapacity: 4, coreTemplateCount: 5, faceTemplateCount: 6,
  templateCount: 7, canonicalTemplateBase: 8, directoryBase: 9,
  operationBase: 10, operationCount: 11, packetsPerLeaf: 12,
  operationWords: 13,
} as const);

export interface SparseCM12VexIBO1ImageLayout {
  readonly canonicalCapacity: number;
  readonly coreTemplateCount: number;
  readonly faceTemplateCount: number;
  readonly templateCount: number;
  readonly canonicalTemplateBaseWords: number;
  readonly directoryBaseWords: number;
  readonly operationBaseWords: number;
  readonly operationCount: number;
  readonly totalWords: number;
  readonly totalBytes: number;
}

export interface SparseCM12VexIBO1Image {
  readonly program: SparseCM12VexIBO1Program;
  readonly layout: SparseCM12VexIBO1ImageLayout;
  readonly words: Uint32Array;
  readonly coreTemplates: readonly SparseCM12VexIBO1TemplateProgram[];
  /** Maps physical IBO1 template id to this image's global template id. */
  readonly faceTemplateIds: Uint32Array;
}

const align = (value: number, words = 64): number => Math.ceil(value / words) * words;
const setLane = (mask: [number, number], lane: number) => {
  mask[lane >>> 5] = (mask[lane >>> 5]! | (1 << (lane & 31))) >>> 0;
};
const localAddress = (x: number, y: number, z: number, resolution: number) => {
  const axis = Math.max(1, Math.ceil(resolution / 4));
  return { packet: Math.floor(x / 4) + axis * (Math.floor(y / 4)
    + axis * Math.floor(z / 4)), lane: (x & 3) + 4 * (y & 3) + 16 * (z & 3) };
};

interface MutableOperation {
  targetDomain: 0 | 1;
  targetLocalPacket: number;
  laneDelta: number;
  source: [number, number];
}

/** Compile exact self plus six bounded same-leaf endpoints for one descriptor. */
const compileCore = (descriptor: SparseCM12FactoredAEICanonicalDescriptor):
SparseCM12VexIBO1TemplateProgram => {
  const packets = new Map<number, Map<string, MutableOperation>>();
  const add = (sourcePacket: number, sourceLane: number, targetPacket: number,
    targetLane: number) => {
    let byOperation = packets.get(sourcePacket);
    if (!byOperation) packets.set(sourcePacket, byOperation = new Map());
    const delta = targetLane - sourceLane, key = `${targetPacket}/${delta}`;
    let operation = byOperation.get(key);
    if (!operation) byOperation.set(key, operation = { targetDomain: 0,
      targetLocalPacket: targetPacket, laneDelta: delta, source: [0, 0] });
    setLane(operation.source, sourceLane);
  };
  const [nx, ny, nz] = descriptor.validDimensions;
  let singletonEdgeCount = 0;
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
    for (let x = 0; x < nx; x += 1) {
      const source = localAddress(x, y, z, descriptor.resolution);
      for (const [tx, ty, tz] of [[x, y, z], [x - 1, y, z], [x + 1, y, z],
        [x, y - 1, z], [x, y + 1, z], [x, y, z - 1], [x, y, z + 1]]) {
        if (tx! < 0 || ty! < 0 || tz! < 0 || tx! >= nx || ty! >= ny || tz! >= nz) {
          continue;
        }
        const target = localAddress(tx!, ty!, tz!, descriptor.resolution);
        add(source.packet, source.lane, target.packet, target.lane);
        singletonEdgeCount += 1;
      }
    }
  }
  let operationCount = 0;
  const packetPrograms = [...packets].sort((a, b) => a[0] - b[0]).map(
    ([sourceLocalPacket, values]) => {
      const operations = [...values.values()].map((value) => Object.freeze({
        targetDomain: value.targetDomain, targetLocalPacket: value.targetLocalPacket,
        laneDelta: value.laneDelta, sourceLow: value.source[0] >>> 0,
        sourceHigh: value.source[1] >>> 0,
      })).sort((a, b) => a.targetLocalPacket - b.targetLocalPacket
        || a.laneDelta - b.laneDelta);
      operationCount += operations.length;
      return Object.freeze({ sourceLocalPacket,
        operations: Object.freeze(operations) });
    });
  return Object.freeze({ templateId: 0,
    packetPrograms: new Map(packetPrograms.map((value) =>
      [value.sourceLocalPacket, value] as const)), operationCount, singletonEdgeCount });
};

const templateSignature = (template: SparseCM12VexIBO1TemplateProgram) =>
  JSON.stringify([...template.packetPrograms].map(([packet, value]) => [packet,
    value.operations.map((operation) => [operation.targetDomain,
      operation.targetLocalPacket, operation.laneDelta,
      operation.sourceLow, operation.sourceHigh]) ]));

const checkedOperation = (operation: SparseCM12VexIBO1Operation) => {
  if (operation.targetLocalPacket < 0
    || operation.targetLocalPacket >= SPARSE_CM12_VEX_IBO1_PACKETS_PER_LEAF) {
    throw new RangeError("VXI1 target local packet exceeds the stable packet ABI");
  }
  if (operation.laneDelta < -63 || operation.laneDelta > 63) {
    throw new RangeError("VXI1 lane delta exceeds one packet ballot");
  }
};

/**
 * Pack the exact canonical core recipes and interned face recipes as an
 * immutable supplement to IBO1. There are no stable ids in the supplement;
 * the selected IBO1 slot supplies source/target leaves and template refs.
 */
export function compileSparseCM12VexIBO1Image(
  program: SparseCM12VexIBO1Program,
): SparseCM12VexIBO1Image {
  const canonicalCapacity = program.catalog.layout.canonicalCapacity;
  const canonicalTemplate = new Uint32Array(canonicalCapacity)
    .fill(SPARSE_CM12_VEX_IBO1_INVALID);
  const coreTemplates: SparseCM12VexIBO1TemplateProgram[] = [];
  const coreBySignature = new Map<string, number>();
  for (const descriptor of program.catalog.canonical) {
    if (!descriptor.certified || descriptor.cellFirst === SPARSE_CM12_FACTORED_AEI_INVALID) {
      continue;
    }
    const compiled = compileCore(descriptor), signature = templateSignature(compiled);
    let id = coreBySignature.get(signature);
    if (id === undefined) {
      id = coreTemplates.length; coreBySignature.set(signature, id);
      coreTemplates.push(Object.freeze({ ...compiled, templateId: id }));
    }
    canonicalTemplate[descriptor.id] = id;
  }
  const faceTemplateIds = new Uint32Array(program.templates.length);
  for (const template of program.templates) {
    faceTemplateIds[template.templateId] = coreTemplates.length + template.templateId;
  }
  const templates = [...coreTemplates, ...program.templates];
  const directoryEntries = templates.length * SPARSE_CM12_VEX_IBO1_PACKETS_PER_LEAF;
  const canonicalTemplateBaseWords = SPARSE_CM12_VEX_IBO1_IMAGE_HEADER_WORDS;
  const directoryBaseWords = align(canonicalTemplateBaseWords + canonicalCapacity);
  const operationBaseWords = align(directoryBaseWords + directoryEntries);
  let operationCount = 0;
  for (const template of templates) operationCount += template.operationCount;
  if (operationCount >= (1 << 20)) throw new RangeError("VXI1 operation offset exceeds 20 bits");
  // The immutable supplement may be inserted immediately before slot0; keep
  // its end at the same 256-byte arena alignment as IBO1 slot bases.
  const totalWords = align(operationBaseWords
    + SPARSE_CM12_VEX_IBO1_OPERATION_WORDS * operationCount);
  const words = new Uint32Array(totalWords).fill(0);
  const h = SPARSE_CM12_VEX_IBO1_IMAGE_HEADER;
  words[h.magic] = SPARSE_CM12_VEX_IBO1_IMAGE_MAGIC;
  words[h.version] = SPARSE_CM12_VEX_IBO1_IMAGE_VERSION;
  words[h.headerWords] = SPARSE_CM12_VEX_IBO1_IMAGE_HEADER_WORDS;
  words[h.totalWords] = totalWords; words[h.canonicalCapacity] = canonicalCapacity;
  words[h.coreTemplateCount] = coreTemplates.length;
  words[h.faceTemplateCount] = program.templates.length;
  words[h.templateCount] = templates.length;
  words[h.canonicalTemplateBase] = canonicalTemplateBaseWords;
  words[h.directoryBase] = directoryBaseWords;
  words[h.operationBase] = operationBaseWords; words[h.operationCount] = operationCount;
  words[h.packetsPerLeaf] = SPARSE_CM12_VEX_IBO1_PACKETS_PER_LEAF;
  words[h.operationWords] = SPARSE_CM12_VEX_IBO1_OPERATION_WORDS;
  words.set(canonicalTemplate, canonicalTemplateBaseWords);
  let nextOperation = 0;
  for (let templateId = 0; templateId < templates.length; templateId += 1) {
    const template = templates[templateId]!;
    for (let sourcePacket = 0; sourcePacket < 64; sourcePacket += 1) {
      const packet = template.packetPrograms.get(sourcePacket);
      const count = packet?.operations.length ?? 0;
      if (count >= (1 << 12)) throw new RangeError("VXI1 packet recipe exceeds 12 bits");
      words[directoryBaseWords + templateId * 64 + sourcePacket]
        = (nextOperation | (count << 20)) >>> 0;
      for (const operation of packet?.operations ?? []) {
        checkedOperation(operation);
        const at = operationBaseWords
          + SPARSE_CM12_VEX_IBO1_OPERATION_WORDS * nextOperation++;
        words[at] = ((operation.targetDomain << 31)
          | ((operation.laneDelta + 63) << 6) | operation.targetLocalPacket) >>> 0;
        words[at + 1] = operation.sourceLow; words[at + 2] = operation.sourceHigh;
      }
    }
  }
  if (nextOperation !== operationCount) throw new Error("VXI1 operation packing drifted");
  const layout = Object.freeze({ canonicalCapacity, coreTemplateCount: coreTemplates.length,
    faceTemplateCount: program.templates.length, templateCount: templates.length,
    canonicalTemplateBaseWords, directoryBaseWords, operationBaseWords,
    operationCount, totalWords, totalBytes: 4 * totalWords });
  return Object.freeze({ program, layout, words,
    coreTemplates: Object.freeze(coreTemplates), faceTemplateIds });
}

const mask64 = (low: number, high: number) => BigInt(low >>> 0)
  | (BigInt(high >>> 0) << 32n);
const unpackMask = (value: bigint): readonly [number, number] => Object.freeze([
  Number(value & 0xffff_ffffn) >>> 0,
  Number((value >> 32n) & 0xffff_ffffn) >>> 0,
]);
const MASK64 = (1n << 64n) - 1n;

/** Execute one packed recipe exactly as the WGSL bridge does. */
export function applySparseCM12VexIBO1ImageTemplate(options: Readonly<{
  image: SparseCM12VexIBO1Image;
  templateId: number;
  sourceLeaf: number;
  targetLeaf: number;
  sourceStablePacket: number;
  sourceLow: number;
  sourceHigh: number;
}>): ReadonlyMap<number, readonly [number, number]> {
  const { image } = options, localPacket = options.sourceStablePacket & 63;
  if (Math.floor(options.sourceStablePacket / 64) !== options.sourceLeaf
    || options.templateId < 0 || options.templateId >= image.layout.templateCount) {
    return new Map();
  }
  const packedDirectory = image.words[image.layout.directoryBaseWords
    + 64 * options.templateId + localPacket]!;
  const first = packedDirectory & 0x000f_ffff, count = packedDirectory >>> 20;
  const source = mask64(options.sourceLow, options.sourceHigh);
  const result = new Map<number, bigint>();
  for (let local = 0; local < count; local += 1) {
    const at = image.layout.operationBaseWords + 3 * (first + local);
    const encoded = image.words[at]!, targetDomain = encoded >>> 31;
    const targetPacket = encoded & 63, delta = ((encoded >>> 6) & 127) - 63;
    const selected = source & mask64(image.words[at + 1]!, image.words[at + 2]!);
    const shifted = delta < 0 ? selected >> BigInt(-delta)
      : (selected << BigInt(delta)) & MASK64;
    if (shifted === 0n) continue;
    const leaf = targetDomain === 0 ? options.sourceLeaf : options.targetLeaf;
    if (leaf === SPARSE_CM12_VEX_IBO1_INVALID) {
      throw new Error("VXI1 selected a missing target leaf");
    }
    const stablePacket = leaf * 64 + targetPacket;
    result.set(stablePacket, (result.get(stablePacket) ?? 0n) | shifted);
  }
  return new Map([...result].map(([packet, mask]) => [packet, unpackMask(mask)]));
}

const mergeMasks = (target: Map<number, [number, number]>,
  source: ReadonlyMap<number, readonly [number, number]>) => {
  for (const [packet, mask] of source) {
    const prior = target.get(packet) ?? [0, 0] as [number, number];
    prior[0] = (prior[0] | mask[0]) >>> 0; prior[1] = (prior[1] | mask[1]) >>> 0;
    target.set(packet, prior);
  }
};

/** Apply selected canonical core plus all selected IBO1 face refs for a leaf. */
export function applySparseCM12VexIBO1ImageSelected(options: Readonly<{
  image: SparseCM12VexIBO1Image;
  sourceStablePacket: number;
  sourceLow: number;
  sourceHigh: number;
}>): ReadonlyMap<number, readonly [number, number]> {
  const { image } = options, sourceLeaf = Math.floor(options.sourceStablePacket / 64);
  const canonicalId = image.program.ibo.catalog.descriptorIdByLeaf[sourceLeaf];
  if (canonicalId === undefined) return new Map();
  const coreTemplateId = image.words[image.layout.canonicalTemplateBaseWords + canonicalId]!;
  if (coreTemplateId === SPARSE_CM12_VEX_IBO1_INVALID) return new Map();
  const output = new Map<number, [number, number]>();
  mergeMasks(output, applySparseCM12VexIBO1ImageTemplate({ ...options, image,
    templateId: coreTemplateId, sourceLeaf, targetLeaf: sourceLeaf }));
  for (const instance of image.program.ibo.instances) {
    if (instance.sourceLeaf !== sourceLeaf) continue;
    mergeMasks(output, applySparseCM12VexIBO1ImageTemplate({ ...options, image,
      templateId: image.faceTemplateIds[instance.templateId]!, sourceLeaf,
      targetLeaf: instance.targetLeaf }));
  }
  return new Map([...output].map(([packet, mask]) =>
    [packet, Object.freeze(mask)] as const));
}
