import { writeFile } from "node:fs/promises";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

/** Read-only scalar snapshots at existing production stage boundaries. */
export async function axisArtifactStageAudit(device: GPUDevice, solver: WebGPUAdaptiveMassSolver,
  output: string, step: number, gammaDiffusion = true) {
  const source = solver.fieldSnapshotSourceForQA;
  const records = (await solver.readGPUActivityPolicy()).bricks.filter(b => b.active);
  const nc = source.cellCapacity;
  const captures = new Map<string, GPUBuffer>();
  const stages = ["transport-velocity-extension", "conservative-transport", "gamma-diffusion",
    "surface-sharpening", "density-capacity-repair", "scalar-publication"];
  solver.setStageCaptureForQA((stage, encoder) => {
    if (!stages.includes(stage)) return;
    const buffer = device.createBuffer({ size: 4 * (6 * nc + 1),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const bases = [source.layout.densityA, source.layout.densityB, source.layout.gammaA,
      source.layout.gammaB, source.layout.pressure, source.layout.rhs];
    bases.forEach((base,i) => encoder.copyBufferToBuffer(source.state, 4 * base, buffer, 4 * nc * i, 4 * nc));
    encoder.copyBufferToBuffer(source.topologyArena,
      4 * (source.frameControlBaseWords + source.scalarParityWord), buffer, 24 * nc, 4);
    captures.set(stage, buffer);
  });
  return async () => {
    solver.setStageCaptureForQA(undefined);
    const { nx, ny, nz } = solver.info;
    for (const [stage, buffer] of captures) {
      try {
        await buffer.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(buffer.getMappedRange());
        const parity = new Uint32Array(data.buffer, 24 * nc, 1)[0]! ^ Number(stage !== stages[0]);
        const scratch = stage === "gamma-diffusion" && gammaDiffusion;
        for (const [name, offset] of [["density", scratch ? 4 * nc : parity * nc],
          ["gamma", scratch ? 5 * nc : (2 + parity) * nc]] as const) {
          const dense = new Float32Array(nx * ny * nz);
          for (const b of records) {
            const r = b.acceptedResolution, width = 8 * b.spanBricks / r;
            const first = source.templateWords[source.templateWords[11]! + 2 * (4 * b.leafId + Math.log2(r))]!;
            for (let z=0; z<8*b.spanBricks; z++) for(let y=0; y<8*b.spanBricks; y++) for(let x=0; x<8*b.spanBricks; x++) {
              const qx=8*b.coordinate[0]+x, qy=8*b.coordinate[1]+y, qz=8*b.coordinate[2]+z;
              if(qx<0||qy<0||qz<0||qx>=nx||qy>=ny||qz>=nz) continue;
              const cell=first+Math.floor(x/width)+r*(Math.floor(y/width)+r*Math.floor(z/width));
              dense[qx+nx*(qy+ny*qz)]=data[offset+cell]!;
            }
          }
          await writeFile(`${output}/${step}-${stage}-${name}.bin`, new Uint8Array(dense.buffer));
        }
      } finally { buffer.unmap(); buffer.destroy(); }
    }
  };
}
