import {
  SPARSE_CM12_FINAL_SCALAR_MASK_HEADER,
  SPARSE_CM12_FINAL_SCALAR_MASK_INVALID,
  SPARSE_CM12_FINAL_SCALAR_MASK_PHASE,
  createSparseCM12FinalScalarPacketMaskInitialWords,
  createSparseCM12FinalScalarPacketMaskLayout,
  type SparseCM12FinalScalarPacketMaskLayout,
} from "../sparse-cm12-final-scalar-packet-masks";
import { slicePressureMembershipPredicate,
  type SliceNumericalFields, type SliceNumericalTopology } from "./slice-stage-numerics";
import type { SliceRuntimeAuthority } from "./slice-runtime-authority";
import type { SliceTopology } from "./slice-topology";

const floatBits = (value: number): number =>
  new Uint32Array(new Float32Array([value]).buffer)[0]!;

export interface SliceScalarPublicationReceipt {
  readonly generation: number;
  readonly topologyGeneration: number;
  readonly topologySlot: 0 | 1;
  readonly changedCellCount: number;
  readonly nonexactCellCount: number;
  readonly bulkCellCount: number;
  readonly flipCellCount: number;
  readonly fault: number;
  readonly firstFaultPacket: number;
}

/** Persistent FSM1 packet masks in the literal production word layout. */
export interface SliceScalarAuthority {
  readonly layout: SparseCM12FinalScalarPacketMaskLayout;
  readonly words: Uint32Array;
  readonly sourceDensity: Float32Array;
  readonly sourceGamma: Float32Array;
  readonly receipt: SliceScalarPublicationReceipt;
}

export function createSliceScalarAuthority(runtime: SliceRuntimeAuthority,
  cellCount: number): SliceScalarAuthority {
  const layout = createSparseCM12FinalScalarPacketMaskLayout({ packetCapacity:
    Math.max(1, runtime.leafCapacity * 64), brickFineResolution: 8 });
  const words = createSparseCM12FinalScalarPacketMaskInitialWords(layout);
  return { layout, words, sourceDensity: new Float32Array(cellCount),
    sourceGamma: new Float32Array(cellCount), receipt: { generation: 0,
      topologyGeneration: runtime.accepted.topology.generation,
      topologySlot: runtime.acceptedSlot, changedCellCount: 0, nonexactCellCount: 0,
      bulkCellCount: 0, flipCellCount: 0, fault: 0,
      firstFaultPacket: SPARSE_CM12_FINAL_SCALAR_MASK_INVALID } };
}

function packetAddress(topology: SliceTopology, cellId: number): readonly [number, number] {
  const cell = topology.cells[cellId]!, brick = topology.brickByKey.get(cell.brickKey)!;
  const packetAxis = Math.max(1, Math.floor((brick.resolution + 3) / 4));
  const packetLocal = (cell.local[0] >> 2) + packetAxis * (cell.local[1] >> 2);
  const lane = (cell.local[0] & 3) + 4 * (cell.local[1] & 3);
  return [64 * brick.id + packetLocal, lane];
}

function writeMask(words: Uint32Array, base: number, packet: number,
  lane: number): void {
  words[base + packet] = (words[base + packet]! | (1 << (lane & 31))) >>> 0;
}

/** CPU execution of begin/publish/sealSparseCM12FinalScalarMasks. */
export function publishSliceScalarAuthority(previous: SliceScalarAuthority,
  topology: SliceTopology, numerical: SliceNumericalTopology,
  fields: SliceNumericalFields, runtime: SliceRuntimeAuthority,
  sourceDensityInput: ArrayLike<number>, sourceGammaInput: ArrayLike<number>,
  generation: number, hasRigidBodies: boolean): SliceScalarAuthority {
  if (sourceDensityInput.length !== topology.cells.length
    || sourceGammaInput.length !== topology.cells.length) {
    throw new RangeError("slice FSM1 source banks do not match accepted cells");
  }
  const layout = previous.layout;
  if (layout.packetCapacity !== runtime.leafCapacity * 64) {
    throw new RangeError("slice FSM1 packet arena changed after allocation");
  }
  const words = createSparseCM12FinalScalarPacketMaskInitialWords(layout),
    h = SPARSE_CM12_FINAL_SCALAR_MASK_HEADER;
  words[h.generation] = Math.max(1, generation);
  words[h.topologyGeneration] = topology.generation;
  words[h.topologySlot] = runtime.acceptedSlot;
  words[h.phase] = SPARSE_CM12_FINAL_SCALAR_MASK_PHASE.collecting;
  const sourceDensity = Float32Array.from(sourceDensityInput),
    sourceGamma = Float32Array.from(sourceGammaInput);
  let changedCellCount = 0, nonexactCellCount = 0, bulkCellCount = 0, flipCellCount = 0;
  for (const cell of topology.cells) {
    const id = cell.id, [packet, lane] = packetAddress(topology, id);
    if (packet >= layout.packetCapacity) {
      words[h.fault] = 1; words[h.firstFaultPacket] = packet; continue;
    }
    const changed = floatBits(fields.density[id]!) !== floatBits(sourceDensity[id]!)
      || floatBits(fields.gamma[id]!) !== floatBits(sourceGamma[id]!);
    const exact = floatBits(fields.capacity[id]!) === 0x3f80_0000
      && floatBits(sourceGamma[id]!) === 0x3f80_0000
      && floatBits(fields.gamma[id]!) === 0x3f80_0000
      && floatBits(sourceDensity[id]!) === floatBits(fields.density[id]!)
      && (floatBits(fields.density[id]!) === 0
        || floatBits(fields.density[id]!) === 0x3f80_0000);
    const bulk = !hasRigidBodies && fields.capacity[id]! >= 1 - 1e-6
      && (fields.characteristicClearance?.[id] ?? 0) > 0
      && Math.abs(fields.density[id]! - 1) <= 0.005
      && Math.abs(fields.gamma[id]! - 1) <= 0.005;
    const flip = slicePressureMembershipPredicate(numerical, fields, id)
      !== (fields.pressureMember[id]! !== 0);
    if (changed) { writeMask(words, lane < 32 ? layout.changedLowBaseWords
      : layout.changedHighBaseWords, packet, lane); changedCellCount++; }
    if (!exact) { writeMask(words, lane < 32 ? layout.nonexactLowBaseWords
      : layout.nonexactHighBaseWords, packet, lane); nonexactCellCount++; }
    if (bulk) { writeMask(words, lane < 32 ? layout.bulkLowBaseWords
      : layout.bulkHighBaseWords, packet, lane); bulkCellCount++;
      // Exact resident value-bearing dead-bank mirror.
      sourceDensity[id] = fields.density[id]!; sourceGamma[id] = fields.gamma[id]!; }
    if (flip) { writeMask(words, lane < 32 ? layout.flipLowBaseWords
      : layout.flipHighBaseWords, packet, lane); flipCellCount++; }
  }
  words[h.changedCellCount] = changedCellCount;
  words[h.nonexactCellCount] = nonexactCellCount;
  words[h.bulkCellCount] = bulkCellCount;
  words[h.flipCellCount] = flipCellCount;
  words[h.phase] = words[h.fault] === 0
    ? SPARSE_CM12_FINAL_SCALAR_MASK_PHASE.published : SPARSE_CM12_FINAL_SCALAR_MASK_PHASE.fault;
  const receipt = { generation: words[h.generation]!, topologyGeneration: topology.generation,
    topologySlot: runtime.acceptedSlot, changedCellCount, nonexactCellCount,
    bulkCellCount, flipCellCount, fault: words[h.fault]!,
    firstFaultPacket: words[h.firstFaultPacket]! };
  return { layout, words, sourceDensity, sourceGamma, receipt };
}

