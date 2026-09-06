import assert from "node:assert/strict";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

/** Read-only stage copies on a fixed, authored topology; no tracer kernels. */
export async function createPoolEnergyBudget(device: GPUDevice, solver: WebGPUAdaptiveMassSolver,
  cellWidth_m: number, gravity = 9.81) {
  const source = solver.fieldSnapshotSourceForQA;
  const records = (await solver.readGPUActivityPolicy()).bricks;
  assert.ok(records.every(r => r.leafId < source.templateWords[13]!), "budget requires authored leaves");
  const words = source.templateWords, floats = new Float32Array(words.buffer, words.byteOffset, words.length);
  const nc = source.cellCapacity, nr = source.rowCapacity;
  const staticRows = words[3]!;
  const rowBase = words[7]!, termBase = words[8]!, incidenceBase = words[9]!, incidenceRecords = words[10]!;
  const rowEnabled = new Uint8Array(nr);
  for (let row = 0; row < staticRows; row++) {
    const requirements = words[rowBase + staticRows + row]! & 0x0fffffff;
    const owners = words[requirements]!;
    assert.ok(owners <= 32, `valid row ownership count ${owners} for ${row}`);
    if (!owners) continue;
    rowEnabled[row] = Number(Array.from({ length: owners }, (_, i) => {
      const meta = words[requirements + 1 + i]!;
      const record = records[meta >>> 5];
      return record?.active && record.acceptedResolution === (meta & 31);
    }).every(Boolean));
  }
  const cells: { id: number; y: number; volume: number; width: number;
    faces: { row: number; axis: number; weight: number }[] }[] = [];
  const levels = 4;
  for (const record of records) {
    if (!record.active) continue;
    const range = words[11]! + 2 * (levels * record.leafId + Math.log2(record.acceptedResolution));
    const first = words[range]!, count = words[range + 1]!;
    for (let id = first; id < first + count; id++) {
      const base = words[6]! + 8 * id;
      const faces = [];
      for (let at = words[incidenceBase + id]!; at < words[incidenceBase + id + 1]!; at++) {
        const row = words[incidenceRecords + 2 * at]!, term = words[incidenceRecords + 2 * at + 1]!;
        if (!rowEnabled[row]) continue;
        faces.push({ row, axis: words[rowBase + staticRows + row]! >>> 30,
          weight: Math.abs(floats[termBase + 2 * term + 1]!) * floats[rowBase + 2 * staticRows + row]! });
      }
      assert.ok(faces.length > 0, `nonempty incidence for cell ${id}`);
      cells.push({ id, y: floats[base + 1]!, volume: floats[base + 3]!, width: floats[base + 4]!, faces });
    }
  }
  const captures = new Map<string, GPUBuffer>();
  const stages = ["transport-velocity-extension", "face-preparation", "conservative-transport",
    "gamma-diffusion", "surface-sharpening", "body-forces", "velocity-projection"];
  return {
    cells: cells.length,
    arm() {
      assert.equal(captures.size, 0);
      solver.setStageCaptureForQA((stage, encoder) => {
        if (!stages.includes(stage)) return;
        const bytes = 4 * (2 * nc + 2 * nr + 2);
        const copy = device.createBuffer({ label: `pool energy ${stage}`, size: bytes,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        let offset = 0;
        for (const [base, count] of [[source.layout.densityA, nc], [source.layout.densityB, nc],
          [source.layout.faceA, nr], [source.layout.faceB, nr]]) {
          encoder.copyBufferToBuffer(source.state, 4 * base!, copy, offset, 4 * count!);
          offset += 4 * count!;
        }
        for (const word of [source.scalarParityWord, source.faceParityWord]) {
          encoder.copyBufferToBuffer(source.topologyArena, 4 * (source.frameControlBaseWords + word), copy, offset, 4);
          offset += 4;
        }
        captures.set(stage, copy);
      });
    },
    async read() {
      solver.setStageCaptureForQA(undefined);
      const result = [];
      for (const [stage, buffer] of captures) {
        try {
          await buffer.mapAsync(GPUMapMode.READ);
          const state = new Float32Array(buffer.getMappedRange());
          const parity = new Uint32Array(state.buffer, 4 * (2 * nc + 2 * nr), 2);
          const scalarParity = parity[0]! ^ Number(stages.indexOf(stage) >= 2);
          const faceParity = parity[1]! ^ Number(stages.indexOf(stage) >= 1);
          const densityOffset = scalarParity * nc, faceOffset = 2 * nc + faceParity * nr;
          let mass = 0, potential = 0, kinetic = 0, collocatedKinetic = 0;
          const byWidth: Record<string, { mass: number; kinetic: number; potential: number }> = {};
          for (const cell of cells) {
            const m = state[densityOffset + cell.id]! * cell.volume * cellWidth_m ** 3;
            const linear = [0, 0, 0], square = [0, 0, 0], weight = [0, 0, 0];
            for (const face of cell.faces) {
              const v = state[faceOffset + face.row]! * cellWidth_m;
              linear[face.axis]! += face.weight * v;
              square[face.axis]! += face.weight * v * v;
              weight[face.axis]! += face.weight;
            }
            let v2 = 0, meanV2 = 0;
            for (let axis = 0; axis < 3; axis++) if (weight[axis]! > 0) {
              v2 += square[axis]! / weight[axis]!;
              meanV2 += (linear[axis]! / weight[axis]!) ** 2;
            }
            const ke = .5 * m * v2, pe = m * gravity * cell.y * cellWidth_m;
            mass += m; potential += pe; kinetic += ke; collocatedKinetic += .5 * m * meanV2;
            const bin = byWidth[cell.width] ??= { mass: 0, kinetic: 0, potential: 0 };
            bin.mass += m; bin.kinetic += ke; bin.potential += pe;
          }
          assert.ok([mass, potential, kinetic, collocatedKinetic].every(Number.isFinite));
          result.push({ stage, mass, potential, kinetic, collocatedKinetic, total: potential + kinetic, byWidth });
        } finally { if (buffer.mapState === "mapped") buffer.unmap(); buffer.destroy(); }
      }
      captures.clear();
      return result;
    },
  };
}
