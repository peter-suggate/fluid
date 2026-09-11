import assert from "node:assert/strict";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";

/** Optional native-stage observation. Uses the same pre-transport liquid
 * volumes at every tap, so advection and projection comparisons do not mix
 * different mass epochs. These face quadratures are diagnostics, not exact
 * PLIC mechanical energy. No additional production kernels are dispatched. */
export async function createGeometricDamStageEnergy(device: GPUDevice,
  solver: WebGPUAdaptiveMassSolver, h: number, waterDensity: number,
  expectedVelocity?: { x: number; y: number; z: number }) {
  const source = solver.fieldSnapshotSourceForQA;
  const activity = await solver.readGPUActivityPolicy();
  const w = source.templateWords;
  const f = new Float32Array(w.buffer, w.byteOffset, w.length);
  const records = new Map(activity.bricks.map(b => [b.leafId, b]));
  const covered = activity.bricks.every(b => !b.active || b.leafId < w[13]!);
  const nr = w[3]!, nc = source.cellCapacity, rowCapacity = source.rowCapacity;
  const enabled = new Uint8Array(nr);
  if (covered) for (let row = 0; row < nr; row++) {
    const at = w[w[7]! + nr + row]! & 0x0fffffff;
    enabled[row] = Number(w[at]! > 0 && Array.from({ length: w[at]! }, (_, i) => {
      const meta = w[at + 1 + i]!, brick = records.get(meta >>> 5);
      return brick?.active && brick.acceptedResolution === (meta & 31);
    }).every(Boolean));
  }
  const cells = covered ? activity.bricks.filter(b => b.active).flatMap(b => {
    const range = w[11]! + 2 * (4 * b.leafId + Math.log2(b.acceptedResolution));
    return Array.from({ length: w[range + 1]! }, (_, i) => {
      const id = w[range]! + i, at = w[6]! + 8 * id;
      const faces: { row: number; axis: number; weight: number }[] = [];
      for (let j = w[w[9]! + id]!; j < w[w[9]! + id + 1]!; j++) {
        const row = w[w[10]! + 2 * j]!, term = w[w[10]! + 2 * j + 1]!;
        if (!enabled[row]) continue;
        faces.push({ row, axis: w[w[7]! + nr + row]! >>> 30,
          weight: Math.abs(f[w[8]! + 2 * term + 1]!) * f[w[7]! + 2 * nr + row]! });
      }
      return { id, volume: f[at + 3]! * h ** 3, faces };
    });
  }) : [];
  const captures = new Map<string, GPUBuffer>();
  const stages = ["transport-velocity-extension", "face-preparation", "body-forces", "velocity-projection"];
  const floatCount = 2 * nc + 2 * rowCapacity;
  const dispose = () => {
    solver.setStageCaptureForQA(undefined);
    for (const buffer of captures.values()) { if (buffer.mapState === "mapped") buffer.unmap(); buffer.destroy(); }
    captures.clear();
  };
  return {
    arm() {
      assert.equal(captures.size, 0);
      if (!covered) return;
      solver.setStageCaptureForQA((stage, encoder) => {
        if (!stages.includes(stage)) return;
        // Admission must finish before arming: never read a different generation.
        assert.equal(solver.fieldSnapshotSourceForQA.state, source.state,
          "stage energy observer generation changed before frame encoding");
        const buffer = device.createBuffer({ label: `Geometric dam energy ${stage}`,
          size: 4 * (floatCount + 2), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        let offset = 0;
        for (const [base, count] of [[source.layout.densityA, nc], [source.layout.densityB, nc],
          [source.layout.faceA, rowCapacity], [source.layout.faceB, rowCapacity]]) {
          encoder.copyBufferToBuffer(source.state, 4 * base!, buffer, offset, 4 * count!);
          offset += 4 * count!;
        }
        for (const word of [source.scalarParityWord, source.faceParityWord]) {
          encoder.copyBufferToBuffer(source.topologyArena, 4 * (source.frameControlBaseWords + word), buffer, offset, 4);
          offset += 4;
        }
        captures.set(stage, buffer);
      });
    },
    async read() {
      solver.setStageCaptureForQA(undefined);
      if (!covered) return { scope: "unavailable: active GPU-grown leaves lack host template metadata", stages: [] };
      const result = [];
      try {
        for (const [stage, buffer] of captures) {
          await buffer.mapAsync(GPUMapMode.READ);
          const data = new Float32Array(buffer.getMappedRange());
          const parity = new Uint32Array(data.buffer, 4 * floatCount, 2);
          // All taps precede geometric transport and FCA commit. Scalar source
          // is unchanged; face preparation and later taps use the destination.
          const rhoAt = parity[0]! * nc;
          const faceAt = 2 * nc + (parity[1]! ^ Number(stage !== stages[0])) * rowCapacity;
          let mass_kg = 0, faceKinetic_J = 0, collocatedKinetic_J = 0;
          const momentum = [0, 0, 0];
          let velocityErrorSquared = 0;
          const expected = expectedVelocity ? [expectedVelocity.x, expectedVelocity.y, expectedVelocity.z] : undefined;
          for (const cell of cells) {
            const mass = waterDensity * data[rhoAt + cell.id]! * cell.volume;
            const sum = [0, 0, 0], square = [0, 0, 0], weight = [0, 0, 0];
            for (const face of cell.faces) {
              const velocity = data[faceAt + face.row]! * h;
              sum[face.axis]! += face.weight * velocity;
              square[face.axis]! += face.weight * velocity ** 2;
              weight[face.axis]! += face.weight;
            }
            mass_kg += mass;
            for (let axis = 0; axis < 3; axis++) if (weight[axis]! > 0) {
              const mean = sum[axis]! / weight[axis]!;
              momentum[axis]! += mass * mean;
              if (expected) velocityErrorSquared += mass * (mean - expected[axis]!) ** 2;
              faceKinetic_J += .5 * mass * square[axis]! / weight[axis]!;
              collocatedKinetic_J += .5 * mass * mean ** 2;
            }
          }
          assert.ok([mass_kg, faceKinetic_J, collocatedKinetic_J].every(Number.isFinite));
          result.push({ stage, mass_kg, faceKinetic_J, collocatedKinetic_J,
            meanVelocity_m_s: mass_kg > 0 ? momentum.map(value => value / mass_kg) : null,
            velocityRmsError_m_s: expected && mass_kg > 0 ? Math.sqrt(Math.max(0, velocityErrorSquared / mass_kg)) : null });
        }
        return { scope: "accepted authored cells; fixed pre-transport V weights; face-square quadrature; open static flume only", stages: result };
      } finally { dispose(); }
    },
    dispose,
  };
}
