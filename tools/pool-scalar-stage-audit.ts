import assert from "node:assert/strict";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";

/** Observe scalar stage budgets without changing a dispatch or accepted field. */
export async function createPoolScalarStageAudit(device: GPUDevice, solver: WebGPUAdaptiveMassSolver,
  cellWidth_m: number, gammaDiffusionEnabled = true) {
  const source = solver.fieldSnapshotSourceForQA;
  const records = (await solver.readGPUActivityPolicy()).bricks;
  const words = source.templateWords, floats = new Float32Array(words.buffer, words.byteOffset, words.length);
  assert.ok(records.every(r => r.leafId < words[13]!), "scalar budget requires authored leaves");
  const cells: { id: number; volume: number }[] = [];
  for (const record of records) {
    if (!record.active) continue;
    const range = words[11]! + 2 * (4 * record.leafId + Math.log2(record.acceptedResolution));
    const first = words[range]!, count = words[range + 1]!;
    for (let id = first; id < first + count; id++)
      cells.push({ id, volume: floats[words[6]! + 8 * id + 3]! });
  }
  const nc = source.cellCapacity;
  const stages = ["transport-velocity-extension", "conservative-transport", "gamma-diffusion",
    "surface-sharpening", "scalar-publication"];
  const captures = new Map<string, GPUBuffer>();
  solver.setStageCaptureForQA((stage, encoder) => {
    if (!stages.includes(stage)) return;
    const copy = device.createBuffer({ label: `pool scalar ${stage}`, size: 4 * (6 * nc + 1),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const bases = [source.layout.densityA, source.layout.densityB, source.layout.gammaA,
      source.layout.gammaB, source.layout.pressure, source.layout.rhs];
    for (let i = 0; i < bases.length; i++)
      encoder.copyBufferToBuffer(source.state, 4 * bases[i]!, copy, 4 * nc * i, 4 * nc);
    encoder.copyBufferToBuffer(source.topologyArena,
      4 * (source.frameControlBaseWords + source.scalarParityWord), copy, 4 * nc * 6, 4);
    captures.set(stage, copy);
  });
  return async () => {
    solver.setStageCaptureForQA(undefined);
    const results = [];
    for (const [stage, buffer] of captures) {
      try {
        await buffer.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(buffer.getMappedRange());
        const sourceParity = new Uint32Array(data.buffer, 4 * nc * 6, 1)[0]!;
        const parity = sourceParity ^ Number(stage !== stages[0]);
        const gammaScratch = stage === "gamma-diffusion" && gammaDiffusionEnabled;
        const density = gammaScratch ? 4 * nc : parity * nc;
        const gamma = gammaScratch ? 5 * nc : (2 + parity) * nc;
        let mass = 0, gammaVolume = 0, oldRoundedMass = 0;
        for (const cell of cells) {
          const rho = data[density + cell.id]!;
          mass += rho * cell.volume;
          gammaVolume += data[gamma + cell.id]! * cell.volume;
          // On an equal-volume D4 orbit, averaging after rounding preserves
          // this quantized sum; it does not preserve the original float sum.
          const scaled = rho * 65536, floor = Math.floor(scaled);
          const rounded = scaled - floor === .5 ? floor + (floor % 2) : Math.round(scaled);
          oldRoundedMass += rounded / 65536 * cell.volume;
        }
        results.push({ stage, mass_m3: mass * cellWidth_m ** 3,
          gammaVolume_m3: gammaVolume * cellWidth_m ** 3,
          legacyD4RoundedMass_m3: oldRoundedMass * cellWidth_m ** 3 });
      } finally { if (buffer.mapState === "mapped") buffer.unmap(); buffer.destroy(); }
    }
    return results;
  };
}
