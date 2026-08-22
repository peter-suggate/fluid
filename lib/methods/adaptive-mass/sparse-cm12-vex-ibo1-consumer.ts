import {
  SPARSE_CM12_INTERNED_BOUNDARY_REFS_PER_LEAF,
  SPARSE_CM12_INTERNED_BOUNDARY_LOGICAL_REF_WORDS,
  SPARSE_CM12_INTERNED_BOUNDARY_TEMPLATE_HEADER_WORDS,
  SPARSE_CM12_INTERNED_BOUNDARY_ROW_WORDS,
  SPARSE_CM12_INTERNED_BOUNDARY_TERM_WORDS,
  type SparseCM12InternedBoundaryCompilation,
  type SparseCM12InternedBoundaryTemplate,
} from "./sparse-cm12-interned-boundary-operators";
import {
  SPARSE_CM12_FACTORED_AEI_INVALID,
  type SparseCM12FactoredAEICanonicalDescriptor,
  type SparseCM12FactoredAEICatalog,
} from "./sparse-cm12-factored-aei-topology";

/** Exact VEX packet-mask consumer compiled directly from IBO1 local operators. */
export interface SparseCM12VexIBO1Operation {
  readonly targetDomain: 0 | 1;
  readonly targetLocalPacket: number;
  readonly laneDelta: number;
  readonly sourceLow: number;
  readonly sourceHigh: number;
}

export interface SparseCM12VexIBO1PacketProgram {
  readonly sourceLocalPacket: number;
  readonly operations: readonly SparseCM12VexIBO1Operation[];
}

export interface SparseCM12VexIBO1TemplateProgram {
  readonly templateId: number;
  readonly packetPrograms: ReadonlyMap<number, SparseCM12VexIBO1PacketProgram>;
  readonly operationCount: number;
  readonly singletonEdgeCount: number;
}

export interface SparseCM12VexIBO1Instance {
  readonly patchId: number;
  readonly sourceLeaf: number;
  readonly sourceCanonicalId: number;
  readonly targetLeaf: number;
  readonly targetCanonicalId: number;
  readonly side: number;
  readonly templateId: number;
  readonly authoritativeRowBase: number;
  /** Exact slot ABI tuple `[templateId,targetLeaf,rowBase]`. */
  readonly slotRef: readonly [number, number, number];
}

export interface SparseCM12VexIBO1Program {
  readonly catalog: SparseCM12FactoredAEICatalog;
  readonly ibo: SparseCM12InternedBoundaryCompilation;
  readonly templates: readonly SparseCM12VexIBO1TemplateProgram[];
  readonly instances: readonly SparseCM12VexIBO1Instance[];
  readonly selectedSlotRefCount: number;
  readonly selectedSlotRefsExact: boolean;
  readonly immutableProgramBytes: number;
}

export interface SparseCM12VexIBO1DepthZeroRoot {
  readonly cell: number;
  readonly cause: number;
  readonly depth: 0;
}

interface LocalAddress { readonly packet: number; readonly lane: number }
interface MutableOperation {
  targetDomain: 0 | 1;
  targetLocalPacket: number;
  laneDelta: number;
  source: [number, number];
}

const setLane = (mask: [number, number], lane: number) => {
  mask[lane >>> 5] = (mask[lane >>> 5]! | (1 << (lane & 31))) >>> 0;
};
const toMask = (low: number, high: number): bigint =>
  BigInt(low >>> 0) | (BigInt(high >>> 0) << 32n);
const MASK64 = (1n << 64n) - 1n;

const localAddress = (ordinal: number, dimensions: readonly number[], resolution: number):
LocalAddress => {
  const xy = dimensions[0]! * dimensions[1]!;
  const z = Math.floor(ordinal / xy), remain = ordinal - z * xy;
  const y = Math.floor(remain / dimensions[0]!), x = remain - y * dimensions[0]!;
  if (x < 0 || y < 0 || z < 0 || x >= dimensions[0]!
    || y >= dimensions[1]! || z >= dimensions[2]!) {
    throw new Error(`IBO1 local ordinal ${ordinal} is outside template dimensions`);
  }
  const axis = Math.max(1, Math.ceil(resolution / 4));
  const packet = Math.floor(x / 4) + axis
    * (Math.floor(y / 4) + axis * Math.floor(z / 4));
  return { packet, lane: (x & 3) + 4 * (y & 3) + 16 * (z & 3) };
};

const templateTerms = (template: SparseCM12InternedBoundaryTemplate):
readonly (readonly number[])[] => {
  const result: number[][] = [];
  const termBase = SPARSE_CM12_INTERNED_BOUNDARY_TEMPLATE_HEADER_WORDS
    + SPARSE_CM12_INTERNED_BOUNDARY_ROW_WORDS * template.rowCount;
  for (let row = 0; row < template.rowCount; row += 1) {
    const rowAt = SPARSE_CM12_INTERNED_BOUNDARY_TEMPLATE_HEADER_WORDS
      + SPARSE_CM12_INTERNED_BOUNDARY_ROW_WORDS * row;
    const packed = template.words[rowAt + 1]!, first = packed & 0x007f_ffff;
    const count = packed >>> 23, terms: number[] = [];
    for (let local = 0; local < count; local += 1) {
      terms.push(template.words[termBase
        + SPARSE_CM12_INTERNED_BOUNDARY_TERM_WORDS * (first + local)]!);
    }
    result.push(terms);
  }
  return result;
};

const compileTemplate = (
  template: SparseCM12InternedBoundaryTemplate,
): SparseCM12VexIBO1TemplateProgram => {
  // packet -> target-domain/local-packet/delta -> source mask
  const packets = new Map<number, Map<string, MutableOperation>>();
  const operation = (source: LocalAddress, targetDomain: 0 | 1,
    target: LocalAddress): MutableOperation => {
    let operations = packets.get(source.packet);
    if (!operations) packets.set(source.packet, operations = new Map());
    const delta = target.lane - source.lane;
    const key = `${targetDomain}/${target.packet}/${delta}`;
    let value = operations.get(key);
    if (!value) operations.set(key, value = { targetDomain,
      targetLocalPacket: target.packet, laneDelta: delta, source: [0, 0] });
    setLane(value.source, source.lane); return value;
  };
  let singletonEdgeCount = 0;
  for (const terms of templateTerms(template)) {
    const sources = terms.filter((value) => (value & 0x8000_0000) === 0);
    for (const sourceValue of sources) {
      const source = localAddress(sourceValue, template.sourceDimensions,
        template.sourceResolution);
      for (const targetValue of terms) {
        const targetDomain = (targetValue >>> 31) as 0 | 1;
        const ordinal = targetValue & 0x7fff_ffff;
        const target = localAddress(ordinal, targetDomain === 0
          ? template.sourceDimensions : template.targetDimensions,
        targetDomain === 0 ? template.sourceResolution : template.targetResolution);
        operation(source, targetDomain, target); singletonEdgeCount += 1;
      }
    }
  }
  const packetPrograms: SparseCM12VexIBO1PacketProgram[] = [];
  let operationCount = 0;
  for (const [sourceLocalPacket, mutable] of [...packets].sort((a, b) => a[0] - b[0])) {
    const operations = [...mutable.values()].map((value) => Object.freeze({
      targetDomain: value.targetDomain, targetLocalPacket: value.targetLocalPacket,
      laneDelta: value.laneDelta, sourceLow: value.source[0] >>> 0,
      sourceHigh: value.source[1] >>> 0,
    })).sort((a, b) => a.targetDomain - b.targetDomain
      || a.targetLocalPacket - b.targetLocalPacket || a.laneDelta - b.laneDelta);
    operationCount += operations.length;
    packetPrograms.push(Object.freeze({ sourceLocalPacket,
      operations: Object.freeze(operations) }));
  }
  return Object.freeze({ templateId: template.id,
    packetPrograms: new Map(packetPrograms.map((value) =>
      [value.sourceLocalPacket, value] as const)), operationCount, singletonEdgeCount });
};

const patchRows = (catalog: SparseCM12FactoredAEICatalog, patchId: number) => {
  const patch = catalog.patches[patchId]!;
  return patch.exceptionCount === 0
    ? Array.from({ length: patch.rowCount }, (_, local) => patch.rowFirst + local)
    : [...catalog.exceptionRows.subarray(patch.exceptionFirst,
      patch.exceptionFirst + patch.exceptionCount)];
};

/** Compile all certified packed IBO1 templates and synthetic all-rung slot refs. */
export function compileSparseCM12VexIBO1Program(options: Readonly<{
  catalog: SparseCM12FactoredAEICatalog;
  ibo: SparseCM12InternedBoundaryCompilation;
}>): SparseCM12VexIBO1Program {
  const { catalog, ibo } = options;
  const templates = ibo.templates.map(compileTemplate);
  const instances = catalog.patches.map((patch) => {
    const rows = patchRows(catalog, patch.id);
    const rowBase = Math.min(...rows), templateId = ibo.templateIdByPatch[patch.id]!;
    const template = ibo.templates[templateId]!;
    for (let local = 0; local < template.rowCount; local += 1) {
      const at = SPARSE_CM12_INTERNED_BOUNDARY_TEMPLATE_HEADER_WORDS
        + SPARSE_CM12_INTERNED_BOUNDARY_ROW_WORDS * local;
      if (rowBase + template.words[at]! !== rows[local]) {
        throw new Error(`IBO1 patch ${patch.id} row-base translation differs`);
      }
    }
    return Object.freeze({ patchId: patch.id, sourceLeaf: patch.sourceLeaf,
      sourceCanonicalId: patch.sourceCanonicalId, targetLeaf: patch.targetLeaf,
      targetCanonicalId: patch.targetCanonicalId, side: patch.sourceSide,
      templateId, authoritativeRowBase: rowBase,
      slotRef: Object.freeze([templateId, patch.targetLeaf, rowBase] as const) });
  });
  let selectedSlotRefCount = 0, selectedSlotRefsExact = true;
  for (const instance of ibo.instances) {
    const at = (instance.sourceLeaf * SPARSE_CM12_INTERNED_BOUNDARY_REFS_PER_LEAF
      + instance.side * 4 + instance.localRef)
      * SPARSE_CM12_INTERNED_BOUNDARY_LOGICAL_REF_WORDS;
    const actual = [ibo.slotRefs[at]!, ibo.slotRefs[at + 1]!, ibo.slotRefs[at + 2]!];
    const expected = [instance.templateId, instance.targetLeaf,
      instance.authoritativeRowBase];
    selectedSlotRefsExact &&= actual.every((value, index) => value === expected[index]);
    selectedSlotRefCount += 1;
  }
  // Proposed immutable execution records: 16-byte template header plus one
  // 16-byte masked-shift operation. Physical leaves remain in slot refs.
  const immutableProgramBytes = templates.reduce((sum, template) =>
    sum + 16 + 16 * template.operationCount, 0);
  return Object.freeze({ catalog, ibo, templates: Object.freeze(templates),
    instances: Object.freeze(instances), selectedSlotRefCount, selectedSlotRefsExact,
    immutableProgramBytes });
}

const descriptorCell = (descriptor: SparseCM12FactoredAEICanonicalDescriptor,
  localPacket: number, lane: number): number | undefined => {
  const axis = Math.max(1, Math.ceil(descriptor.resolution / 4));
  const pz = Math.floor(localPacket / (axis * axis));
  const remain = localPacket - pz * axis * axis;
  const py = Math.floor(remain / axis), px = remain - py * axis;
  const x = 4 * px + (lane & 3), y = 4 * py + ((lane >>> 2) & 3);
  const z = 4 * pz + (lane >>> 4), [nx, ny, nz] = descriptor.validDimensions;
  if (x >= nx || y >= ny || z >= nz) return undefined;
  return descriptor.cellFirst + x + nx * (y + ny * z);
};

/** Execute one all-rung IBO1 instance directly into stable packet masks. */
export function applySparseCM12VexIBO1Instance(
  program: SparseCM12VexIBO1Program,
  instance: SparseCM12VexIBO1Instance,
  sourceStablePacket: number,
  sourceLow: number,
  sourceHigh: number,
): ReadonlyMap<number, readonly [number, number]> {
  if (Math.floor(sourceStablePacket / 64) !== instance.sourceLeaf) return new Map();
  const localPacket = sourceStablePacket & 63;
  const packetProgram = program.templates[instance.templateId]!
    .packetPrograms.get(localPacket);
  if (!packetProgram) return new Map();
  const ballot = toMask(sourceLow, sourceHigh), output = new Map<number, bigint>();
  for (const op of packetProgram.operations) {
    const selected = ballot & toMask(op.sourceLow, op.sourceHigh);
    const shifted = op.laneDelta < 0 ? selected >> BigInt(-op.laneDelta)
      : (selected << BigInt(op.laneDelta)) & MASK64;
    if (shifted === 0n) continue;
    const leaf = op.targetDomain === 0 ? instance.sourceLeaf : instance.targetLeaf;
    if (leaf === SPARSE_CM12_FACTORED_AEI_INVALID) {
      throw new Error(`IBO1 template ${instance.templateId} selected absent target`);
    }
    const packet = leaf * 64 + op.targetLocalPacket;
    output.set(packet, (output.get(packet) ?? 0n) | shifted);
  }
  return new Map([...output].map(([packet, mask]) => [packet, Object.freeze([
    Number(mask & 0xffff_ffffn) >>> 0,
    Number((mask >> 32n) & 0xffff_ffffn) >>> 0,
  ])] as const));
}

/** Resolve exact stable masks as cell roots, preserving cause and depth zero. */
export function materializeSparseCM12VexIBO1DepthZero(
  program: SparseCM12VexIBO1Program,
  instance: SparseCM12VexIBO1Instance,
  sourceStablePacket: number,
  sourceLow: number,
  sourceHigh: number,
  cause: number,
): readonly SparseCM12VexIBO1DepthZeroRoot[] {
  if (!Number.isSafeInteger(cause) || cause <= 0 || cause > 0xffff_ffff) {
    throw new RangeError("IBO1 VEX cause must be a nonzero u32");
  }
  const source = program.catalog.canonical[instance.sourceCanonicalId]!;
  const target = instance.targetCanonicalId === SPARSE_CM12_FACTORED_AEI_INVALID
    ? undefined : program.catalog.canonical[instance.targetCanonicalId];
  const roots: SparseCM12VexIBO1DepthZeroRoot[] = [];
  for (const [packet, mask] of applySparseCM12VexIBO1Instance(program, instance,
    sourceStablePacket, sourceLow, sourceHigh)) {
    const leaf = Math.floor(packet / 64), descriptor = leaf === instance.sourceLeaf
      ? source : leaf === instance.targetLeaf ? target : undefined;
    if (!descriptor) throw new Error(`IBO1 output packet ${packet} has no descriptor`);
    for (let lane = 0; lane < 64; lane += 1) {
      if ((((lane < 32 ? mask[0] : mask[1]) >>> (lane & 31)) & 1) === 0) continue;
      const cell = descriptorCell(descriptor, packet & 63, lane);
      if (cell === undefined) throw new Error(`IBO1 output selects invalid ${packet}/${lane}`);
      roots.push(Object.freeze({ cell, cause: cause >>> 0, depth: 0 as const }));
    }
  }
  return Object.freeze(roots.sort((a, b) => a.cell - b.cell));
}
