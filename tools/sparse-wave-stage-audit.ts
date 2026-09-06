import assert from "node:assert/strict";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

/** Native accepted-cell/face measurements, captured after real production stages. */
export async function createSparseWaveStageAudit(device: GPUDevice, solver: WebGPUAdaptiveMassSolver,
  h: number, kx: number, kz: number, horizontalArea: number, gammaDiffusion: boolean) {
  const source = solver.fieldSnapshotSourceForQA;
  const activity = await solver.readGPUActivityPolicy();
  const w = source.templateWords, f = new Float32Array(w.buffer, w.byteOffset, w.length);
  const nr = w[3]!, nc = source.cellCapacity, capacityRows = source.rowCapacity;
  assert.ok(activity.bricks.every(b => b.leafId < w[13]!), "audit uses authored accepted leaves");
  const buildCells = (snapshot: typeof activity) => {
  const enabled = new Uint8Array(nr);
  for (let row = 0; row < nr; row++) {
    const at = w[w[7]! + nr + row]! & 0x0fffffff;
    enabled[row] = Number(w[at]! > 0 && Array.from({ length: w[at]! }, (_, i) => {
      const meta = w[at + 1 + i]!, b = snapshot.bricks[meta >>> 5];
      return b?.active && b.acceptedResolution === (meta & 31);
    }).every(Boolean));
  }
  const sinc = (v: number) => v === 0 ? 1 : Math.sin(v) / v;
  const cells = snapshot.bricks.filter(b => b.active).flatMap(b => {
    const range = w[11]! + 2 * (4 * b.leafId + Math.log2(b.acceptedResolution));
    return Array.from({ length: w[range + 1]! }, (_, i) => {
      const id = w[range]! + i, at = w[6]! + 8 * id;
      const x = f[at]! * h, y = f[at + 1]! * h, z = f[at + 2]! * h;
      const wx = f[at + 4]! * h, wz = f[at + 6]! * h;
      const factor = sinc(kx * wx / 2) * sinc(kz * wz / 2);
      const faces = [];
      for (let j = w[w[9]! + id]!; j < w[w[9]! + id + 1]!; j++) {
        const row = w[w[10]! + 2 * j]!, term = w[w[10]! + 2 * j + 1]!;
        if (!enabled[row]) continue;
        faces.push({ row, axis: w[w[7]! + nr + row]! >>> 30,
          orientation: Math.sign(f[w[8]! + 2 * term + 1]!),
          weight: Math.abs(f[w[8]! + 2 * term + 1]!) * f[w[7]! + 2 * nr + row]! });
      }
      return { id, y, volume: f[at + 3]! * h ** 3, width: wx / h, faces,
        phi: Math.cos(kx * x) * Math.cos(kz * z) * factor,
        dx: -kx * Math.sin(kx * x) * Math.cos(kz * z) * factor,
        dz: -kz * Math.cos(kx * x) * Math.sin(kz * z) * factor };
    });
  });
  return cells;
  };
  const cells = buildCells(activity);
  const norm = horizontalArea * (kx ? .5 : 1) * (kz ? .5 : 1);
  const stages = ["transport-velocity-extension", "face-preparation", "conservative-transport",
    "gamma-diffusion", "surface-sharpening", "scalar-publication", "body-forces", "velocity-projection", "candidate-transfer"];
  const captures = new Map<string, GPUBuffer>();
  // Gamma's intermediate output reuses the pressure scratch, not densityA/B.
  const segments = [[source.layout.densityA, nc], [source.layout.densityB, nc], [source.layout.pressure, nc],
    [source.layout.faceA, capacityRows], [source.layout.faceB, capacityRows],
    [source.layout.faceVelocitySupport, 4 * nc], [source.layout.liquid, nc]];
  const floatCount = 8 * nc + 2 * capacityRows;
  return {
    arm() {
      assert.equal(captures.size, 0);
      solver.setStageCaptureForQA((stage, encoder) => {
        if (!stages.includes(stage)) return;
        const target = device.createBuffer({ label: `wave stage ${stage}`, size: 4 * (floatCount + 2),
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        let offset = 0;
        for (const [base, count] of segments) {
          encoder.copyBufferToBuffer(source.state, 4 * base!, target, offset, 4 * count!); offset += 4 * count!;
        }
        for (const parity of [source.scalarParityWord, source.faceParityWord]) {
          encoder.copyBufferToBuffer(source.topologyArena, 4 * (source.frameControlBaseWords + parity), target, offset, 4); offset += 4;
        }
        assert.equal(offset, target.size);
        captures.set(stage, target);
      });
    },
    async read() {
      solver.setStageCaptureForQA(undefined);
      const result = [];
      for (const [stage, buffer] of captures) {
        try {
          await buffer.mapAsync(GPUMapMode.READ);
          const data = new Float32Array(buffer.getMappedRange());
          const parity = new Uint32Array(data.buffer, 4 * floatCount, 2);
          const index = stages.indexOf(stage);
          const densityOffset = stage === "gamma-diffusion" && gammaDiffusion ? 2 * nc : (parity[0]! ^ Number(index >= 2)) * nc;
          const faceOffset = 3 * nc + (parity[1]! ^ Number(index >= 1)) * capacityRows;
          const supportOffset = 3 * nc + 2 * capacityRows;
          let mass = 0, potential = 0, faceKinetic = 0, cellKinetic = 0, transportKinetic = 0;
          let mode = 0, modeRate = 0, transportModeRate = 0, pressureMembers = 0;
          let maxNativeDivergence = 0;
          const byWidth: Record<string, { mass: number; faceKinetic: number }> = {};
          const stageCells = stage === "candidate-transfer" ? buildCells(await solver.readGPUActivityPolicy()) : cells;
          for (const cell of stageCells) {
            const m = data[densityOffset + cell.id]! * cell.volume;
            const sum = [0, 0, 0], square = [0, 0, 0], weight = [0, 0, 0];
            let divergence = 0;
            for (const face of cell.faces) {
              const v = data[faceOffset + face.row]! * h;
              sum[face.axis]! += face.weight * v; square[face.axis]! += face.weight * v * v; weight[face.axis]! += face.weight;
              divergence += face.orientation * face.weight * v * h * h / cell.volume;
            }
            const v = sum.map((s, a) => weight[a]! ? s / weight[a]! : 0);
            const ke = .5 * m * square.reduce((s, x, a) => s + (weight[a]! ? x / weight[a]! : 0), 0);
            const transport = [0, 1, 2].map(a => data[supportOffset + 4 * cell.id + a]! * h);
            mass += m; potential += m * 9.81 * cell.y; faceKinetic += ke;
            cellKinetic += .5 * m * v.reduce((s, x) => s + x * x, 0);
            transportKinetic += .5 * m * transport.reduce((s, x) => s + x * x, 0);
            mode += m * cell.phi / norm;
            modeRate += m * (v[0]! * cell.dx + v[2]! * cell.dz) / norm;
            transportModeRate += m * (transport[0]! * cell.dx + transport[2]! * cell.dz) / norm;
            pressureMembers += Number(data[supportOffset + 4 * nc + cell.id]! > .5);
            if (data[supportOffset + 4 * nc + cell.id]! > .5) maxNativeDivergence = Math.max(maxNativeDivergence, Math.abs(divergence));
            const bin = byWidth[cell.width] ??= { mass: 0, faceKinetic: 0 };
            bin.mass += m; bin.faceKinetic += ke;
          }
          assert.ok([mass, potential, faceKinetic, mode, modeRate].every(Number.isFinite));
          result.push({ stage, mass, potential, faceKinetic, cellKinetic, transportKinetic,
            mode, modeRate, transportModeRate, pressureMembers, maxNativeDivergence, byWidth });
        } finally { if (buffer.mapState === "mapped") buffer.unmap(); buffer.destroy(); }
      }
      captures.clear();
      return result;
    },
  };
}
