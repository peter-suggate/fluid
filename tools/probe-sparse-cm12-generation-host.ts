/** Node-only preload for locating generation preparation heap peaks. */
import { writeFileSync } from "node:fs";
import { WebGPUSparseCM12Resident } from "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

const startedAt = performance.now();
const memory = (phase: string, extra?: unknown) => console.error(JSON.stringify({
  probe: "cm12-generation-host", phase, elapsedMilliseconds: performance.now() - startedAt,
  ...process.memoryUsage(), extra,
}));
const resident = WebGPUSparseCM12Resident as unknown as Record<string, any>;
const prototype = resident.prototype as Record<string, any>;
for (const name of ["captureGenerationTransferSourceWhileLeased", "prepareGenerationReplacement", "createReplacement"]) {
  const original = prototype[name];
  prototype[name] = async function (...args: any[]) {
    memory(`${name}:begin`);
    if (name === "createReplacement") {
      const atlas = args[0];
      writeFileSync("/tmp/fluid-cm12-generation-atlas.json", JSON.stringify({
        dimensions: atlas.dimensions, generation: atlas.generation,
        brickFineResolution: atlas.brickFineResolution, signedCoordinates: atlas.signedCoordinates,
        active: [...args[2]], bricks: atlas.bricks.map((brick: any) => ({
          key: brick.key, coordinate: brick.coordinate, resolution: brick.resolution,
          spanBricks: brick.spanBricks, unclipped: brick.unclipped,
          density: brick.density[0], gamma: brick.gamma[0],
        })),
      }));
    }
    try { return await original.apply(this, args); }
    finally { memory(`${name}:end`); }
  };
}
const configured = resident.createConfigured;
resident.createConfigured = async function (...args: any[]) {
  const report = args[14];
  args[14] = (phase: string) => {
    memory(`createConfigured:${phase}`, {cells: args[2].cells.length,
      rows: args[2].gradientRows.length, bricks: args[1].bricks.length});
    report?.(phase);
  };
  return configured.apply(this, args);
};
