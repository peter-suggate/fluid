import { LEVELSET_VOLUME_GLOBAL_HEADER as GLOBAL,
  LEVELSET_VOLUME_SLOT_HEADER as SLOT } from
  "../lib/methods/adaptive-volume/levelset-volume-layout";
import type { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";

const STAGES = new Set(["transport-velocity-extension", "velocity-projection",
  "conservative-transport", "candidate-transfer", "presentation-publication"]);

/** Read-only copies of the adaptive phi authority at existing production seams. */
export function sparseGeometricLsvStageAudit(device: GPUDevice,
  solver: WebGPUAdaptiveMassSolver, dimensions: readonly [number, number, number]) {
  const source = solver.fieldSnapshotSourceForQA;
  const layout = source.levelSetVolumeLayout;
  const capacity = layout.vertexCapacity;
  const cellCapacity = layout.activeCellCapacity;
  const captures = new Map<string, GPUBuffer>();
  const wordsPerSlot = 32 + 8 * capacity + 8 * cellCapacity;
  const originalPhiAt = 32 + 2 * wordsPerSlot;
  const originalSupportAt = originalPhiAt + capacity;
  const totalWords = originalSupportAt + capacity;
  const encode = (name: string, encoder: GPUCommandEncoder) => {
    if (captures.has(name)) return;
    const buffer = device.createBuffer({ label: `LSV stage ${name}`, size: 4 * totalWords,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder.copyBufferToBuffer(source.topologyArena, 4 * layout.headerBaseWords,
      buffer, 0, 128);
    let destination = 32;
    for (const slot of layout.slots) {
      for (const [at, count] of [[slot.headerBaseWords, 32],
        [slot.vertexRecordsBaseWords, 4 * capacity], [slot.phi0BaseWords, capacity],
        [slot.phi1BaseWords, capacity], [slot.support0BaseWords, capacity],
        [slot.support1BaseWords, capacity], [slot.cellRecordsBaseWords, 8 * cellCapacity]] as const) {
        encoder.copyBufferToBuffer(source.topologyArena, 4 * at, buffer,
          4 * destination, 4 * count); destination += count;
      }
    }
    encoder.copyBufferToBuffer(source.topologyArena,
      4 * layout.redistanceOriginalPhiBaseWords, buffer, 4 * originalPhiAt, 4 * capacity);
    encoder.copyBufferToBuffer(source.topologyArena,
      4 * layout.redistanceOriginalSupportBaseWords, buffer, 4 * originalSupportAt, 4 * capacity);
    captures.set(name, buffer);
  };
  const initial = device.createCommandEncoder({ label: "LSV initial stage audit" });
  encode("initial", initial); device.queue.submit([initial.finish()]);
  solver.setStageCaptureForQA((stage, encoder) => { if (STAGES.has(stage)) encode(stage, encoder); });

  const read = async () => {
    solver.setStageCaptureForQA(undefined);
    const result: Record<string, unknown> = {};
    for (const [name, buffer] of captures) try {
      await buffer.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(buffer.getMappedRange());
      const floats = new Float32Array(words.buffer, words.byteOffset, words.length);
      const acceptedSlot = words[GLOBAL.acceptedSlot]!;
      if (acceptedSlot > 1) { result[name] = { acceptedSlot, available: false }; continue; }
      const base = 32 + acceptedSlot * wordsPerSlot;
      const count = Math.min(capacity, words[base + SLOT.vertexCount]!);
      const bank = words[base + SLOT.sourceBank]! & 1;
      const records = base + 32;
      const phi = records + 4 * capacity + bank * capacity;
      const support = records + 6 * capacity + bank * capacity;
      const cellRecords = records + 8 * capacity;
      const cellCount = Math.min(cellCapacity, words[base + SLOT.activeCellCount]!);
      const cellWidthHistogram: Record<string, number> = {};
      const cellBoundsFine = { minimum: [Infinity, Infinity, Infinity],
        maximum: [-Infinity, -Infinity, -Infinity] };
      for (let cell = 0; cell < cellCount; cell++) {
        const at = cellRecords + 8 * cell;
        const widths = [floats[at + 3]!, floats[at + 4]!, floats[at + 5]!];
        const key = widths.join("x");
        cellWidthHistogram[key] = (cellWidthHistogram[key] ?? 0) + 1;
        for (let axis = 0; axis < 3; axis++) {
          const lower = words[at + axis]! | 0;
          cellBoundsFine.minimum[axis] = Math.min(cellBoundsFine.minimum[axis]!, lower);
          cellBoundsFine.maximum[axis] = Math.max(cellBoundsFine.maximum[axis]!, lower + widths[axis]!);
        }
      }
      const byPosition = new Map<string, number>();
      const vertices: Array<{ positionFine: [number, number, number]; phiFine: number;
        support: number }> = [];
      let liquidVertices = 0, negativeOutsideAuthoredBox = 0, nonFinite = 0;
      let originalLiquidVertices = 0, maximumRedistanceChangeFine = 0;
      const originalByPosition = new Map<string, number>();
      const outside: typeof vertices = [];
      for (let vertex = 0; vertex < count; vertex++) {
        const position = [words[records + 4 * vertex]!, words[records + 4 * vertex + 1]!,
          words[records + 4 * vertex + 2]!]!.map(value => value | 0) as [number, number, number];
        const value = floats[phi + vertex]!;
        const item = { positionFine: position, phiFine: value, support: words[support + vertex]! };
        vertices.push(item); byPosition.set(position.join(","), value);
        const original = floats[originalPhiAt + vertex]!;
        originalByPosition.set(position.join(","), original);
        if (original <= 0) originalLiquidVertices++;
        maximumRedistanceChangeFine = Math.max(maximumRedistanceChangeFine,
          Math.abs(value - original));
        if (!Number.isFinite(value)) nonFinite++;
        if (value <= 0) liquidVertices++;
        if (value < 0 && (position[0] < 8 || position[0] > 24 || position[1] < 0
          || position[1] > 8 || position[2] < 8 || position[2] > 24)) {
          negativeOutsideAuthoredBox++;
          if (outside.length < 32) outside.push(item);
        }
      }
      let reflectX = 0, reflectZ = 0, transposeXZ = 0, missingD4Partners = 0;
      let originalReflectX = 0, originalReflectZ = 0, originalTransposeXZ = 0;
      for (const vertex of vertices) for (const [kind, position] of [
        ["reflectX", [dimensions[0] - vertex.positionFine[0], vertex.positionFine[1],
          vertex.positionFine[2]]],
        ["reflectZ", [vertex.positionFine[0], vertex.positionFine[1],
          dimensions[2] - vertex.positionFine[2]]],
        ["transposeXZ", [vertex.positionFine[2], vertex.positionFine[1],
          vertex.positionFine[0]]],
      ] as const) {
        const partner = byPosition.get(position.join(","));
        if (partner === undefined) { missingD4Partners++; continue; }
        const error = Math.abs(partner - vertex.phiFine);
        const original = originalByPosition.get(vertex.positionFine.join(","))!;
        const originalPartner = originalByPosition.get(position.join(","))!;
        const originalError = Math.abs(originalPartner - original);
        if (kind === "reflectX") reflectX = Math.max(reflectX, error);
        else if (kind === "reflectZ") reflectZ = Math.max(reflectZ, error);
        else transposeXZ = Math.max(transposeXZ, error);
        if (kind === "reflectX") originalReflectX = Math.max(originalReflectX, originalError);
        else if (kind === "reflectZ") originalReflectZ = Math.max(originalReflectZ, originalError);
        else originalTransposeXZ = Math.max(originalTransposeXZ, originalError);
      }
      result[name] = { available: true, acceptedSlot,
        generation: words[base + SLOT.generation]!, fault: words[base + SLOT.fault]!,
        activeVertices: count, activeCells: cellCount, sourceBank: bank,
        cellWidthHistogram, cellBoundsFine: cellCount ? cellBoundsFine : null,
        liquidVertices, nonFinite,
        advectedBeforeRedistance: name === "conservative-transport" ? {
          available: true, identity: "same accepted generation and vertex ordinals",
          liquidVertices: originalLiquidVertices,
          supportNonzero: Array.from({ length: count }, (_, vertex) =>
            words[originalSupportAt + vertex]!).filter(Boolean).length,
          symmetryFine: { reflectX: originalReflectX, reflectZ: originalReflectZ,
            transposeXZ: originalTransposeXZ }, maximumRedistanceChangeFine,
        } : { available: false,
          reason: "redistance scratch is only ordinal-compatible at conservative-transport" },
        negativeOutsideAuthoredBox, firstNegativeOutsideAuthoredBox: outside,
        symmetryFine: { reflectX, reflectZ, transposeXZ, missingD4Partners } };
    } finally { if (buffer.mapState === "mapped") buffer.unmap(); buffer.destroy(); }
    return result;
  };
  return { read };
}

/** Rolling accepted LSV snapshot immediately before conservative transport.
 * Copies only in QA command encoders and overwrites one buffer every frame. */
export function sparseGeometricLsvFaultContextAudit(device: GPUDevice,
  solver: WebGPUAdaptiveMassSolver) {
  const source = solver.fieldSnapshotSourceForQA;
  const layout = source.levelSetVolumeLayout;
  const vertexCapacity = layout.vertexCapacity, cellCapacity = layout.activeCellCapacity;
  const wordsPerSlot = 32 + 8 * vertexCapacity + 16 * cellCapacity;
  const buffer = device.createBuffer({ label: "LSV rolling prephysics fault context",
    size: 4 * (32 + 2 * wordsPerSlot),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  let captured = false;
  const capture = (stage: string, encoder: GPUCommandEncoder) => {
    if (stage !== "velocity-projection") return;
    captured = true;
    encoder.copyBufferToBuffer(source.topologyArena, 4 * layout.headerBaseWords, buffer, 0, 128);
    let destination = 32;
    for (const slot of layout.slots) {
      for (const [at, count] of [[slot.headerBaseWords, 32],
        [slot.vertexRecordsBaseWords, 4 * vertexCapacity],
        [slot.phi0BaseWords, vertexCapacity], [slot.phi1BaseWords, vertexCapacity],
        [slot.support0BaseWords, vertexCapacity], [slot.support1BaseWords, vertexCapacity],
        [slot.cellRecordsBaseWords, 8 * cellCapacity],
        [slot.cornerRefsBaseWords, 8 * cellCapacity]] as const) {
        encoder.copyBufferToBuffer(source.topologyArena, 4 * at, buffer,
          4 * destination, 4 * count); destination += count;
      }
    }
  };
  solver.setStageCaptureForQA(capture);
  const read = async (ownerVertexId: number, samplePositionFine?: readonly number[]) => {
    if (!captured) return { available: false, reason: "failed before velocity-projection seam" };
    await buffer.mapAsync(GPUMapMode.READ);
    try {
      const words = new Uint32Array(buffer.getMappedRange());
      const floats = new Float32Array(words.buffer, words.byteOffset, words.length);
      const acceptedSlot = words[GLOBAL.acceptedSlot]!;
      if (acceptedSlot > 1) return { available: false, acceptedSlot };
      const base = 32 + acceptedSlot * wordsPerSlot;
      const vertexCount = Math.min(vertexCapacity, words[base + SLOT.vertexCount]!);
      const cellCount = Math.min(cellCapacity, words[base + SLOT.activeCellCount]!);
      const bank = words[base + SLOT.sourceBank]! & 1;
      const records = base + 32;
      const phi = records + 4 * vertexCapacity + bank * vertexCapacity;
      const support = records + 6 * vertexCapacity + bank * vertexCapacity;
      const cells = records + 8 * vertexCapacity;
      const corners = cells + 8 * cellCapacity;
      const vertexAt = (id: number) => id >= vertexCount ? null : ({ id,
        positionFine: [words[records + 4 * id]! | 0, words[records + 4 * id + 1]! | 0,
          words[records + 4 * id + 2]! | 0],
        phiFine: floats[phi + id]!, support: words[support + id]! & 3,
        constrained: (words[records + 4 * id + 3]! & 7) !== 0,
      });
      const cellAt = (ordinal: number) => {
        const at = cells + 8 * ordinal;
        const lower = [words[at]! | 0, words[at + 1]! | 0, words[at + 2]! | 0];
        const widths = [floats[at + 3]!, floats[at + 4]!, floats[at + 5]!];
        const cornerIds = Array.from(words.slice(corners + 8 * ordinal,
          corners + 8 * ordinal + 8));
        return { ordinal, lowerFine: lower, widthsFine: widths,
          spanFine: words[at + 6]!, stableCellId: words[at + 7]!, cornerIds,
          corners: cornerIds.map(vertexAt) };
      };
      const incidentCells = [], departureContainingCells = [];
      for (let ordinal = 0; ordinal < cellCount; ordinal++) {
        const cornerIds = words.subarray(corners + 8 * ordinal, corners + 8 * ordinal + 8);
        if (cornerIds.includes(ownerVertexId)) incidentCells.push(cellAt(ordinal));
        if (samplePositionFine?.length === 3) {
          const at = cells + 8 * ordinal;
          const inside = [0, 1, 2].every(axis => samplePositionFine[axis]! >= (words[at + axis]! | 0)
            && samplePositionFine[axis]! <= (words[at + axis]! | 0) + floats[at + 3 + axis]!);
          if (inside) departureContainingCells.push(cellAt(ordinal));
        }
      }
      return { available: true, acceptedSlot,
        generation: words[base + SLOT.generation]!, sourceBank: bank,
        vertexCount, cellCount, ownerVertex: vertexAt(ownerVertexId),
        incidentCells, departureContainingCells };
    } finally { buffer.unmap(); }
  };
  const destroy = () => { solver.setStageCaptureForQA(undefined); buffer.destroy(); };
  return { read, destroy };
}
