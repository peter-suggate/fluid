/**
 * How many of a Sparse CM12 advance's accepted cells actually carry work?
 *
 * Every non-pressure physics kernel is dispatched over the accepted cell
 * worklist, while the pressure solve runs on a compacted liquid worklist. This
 * reports the four populations that matter for narrowing that gap: accepted,
 * liquid, interface (a cell with a strictly-mixed neighbourhood), and the
 * one-ring dilation of the interface.
 *
 * Run:
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
 *   node --import tsx tools/probe-sparse-cm12-cell-population.ts --scene=long-dam
 */
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
import { resolveMethodValues } from "../lib/core/method-contract";
import {
  createMinimalPowerDamBreak32Scene,
  createSparseCM12LongDamBreakScene,
  createSymmetricExpansionScene,
} from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const argument = (name: string, fallback: string): string =>
  process.argv.slice(2).find((value) => value.startsWith(`--${name}=`))
    ?.slice(name.length + 3) ?? fallback;
const sceneName = argument("scene", "long-dam");
const frames = Number(argument("frames", "40"));
const buildScene = sceneName === "mini32" ? createMinimalPowerDamBreak32Scene
  : sceneName === "long-dam" ? createSparseCM12LongDamBreakScene
  : createSymmetricExpansionScene;

await acquireWebGPUExclusiveLock("dawn-probe", "tools/probe-sparse-cm12-cell-population.ts");
try {
  const modulePath = process.env.WEBGPU_NODE_MODULE
    ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
  const { create, globals } = await import(pathToFileURL(modulePath).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter, "WebGPU did not expose an adapter");
  const device = await adapter.requestDevice({
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const scene = buildScene();
  const values = resolveMethodValues(adaptiveMassMethod, "balanced", { timeStep: "scene" });
  const solver = await adaptiveMassMethod.createSolverAsync!(
    device, scene, "balanced", values, undefined, () => {}) as WebGPUAdaptiveMassSolver;
  const dt_s = scene.numerics.fixedDt_s ?? scene.numerics.maxDt_s;
  const [nx, ny, nz] = sceneName === "long-dam" ? [192, 96, 32]
    : sceneName === "mini32" ? [32, 32, 32] : [32, 16, 32];

  const rows: unknown[] = [];
  for (let frame = 1; frame <= frames; frame += 1) {
    while (!solver.advanceTo(frame * dt_s, [])) await new Promise(setImmediate);
    if (frame % 10 !== 0) continue;
    const diagnostics = await solver.readStats();
    const fields = await solver.readDiagnosticFields();
    const density = fields.density;
    const wet = (index: number) => density[index]! >= 0.5;
    const occupied = (index: number) => density[index]! > 1e-4;
    let liquid = 0; let anyMass = 0; let interfaceCells = 0; let dilated = 0;
    const isInterface = new Uint8Array(nx * ny * nz);
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1)
      for (let x = 0; x < nx; x += 1) {
        const index = x + nx * (y + ny * z);
        if (wet(index)) liquid += 1;
        if (occupied(index)) anyMass += 1;
        let mixed = false;
        const self = wet(index);
        for (let axis = 0; axis < 3 && !mixed; axis += 1)
          for (const step of [-1, 1]) {
            const q = [x, y, z];
            q[axis]! += step;
            if (q[0]! < 0 || q[1]! < 0 || q[2]! < 0
              || q[0]! >= nx || q[1]! >= ny || q[2]! >= nz) continue;
            if (wet(q[0]! + nx * (q[1]! + ny * q[2]!)) !== self) { mixed = true; break; }
          }
        if (mixed) { interfaceCells += 1; isInterface[index] = 1; }
      }
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1)
      for (let x = 0; x < nx; x += 1) {
        const index = x + nx * (y + ny * z);
        if (isInterface[index]) { dilated += 1; continue; }
        let near = false;
        for (let axis = 0; axis < 3 && !near; axis += 1)
          for (const step of [-1, 1]) {
            const q = [x, y, z];
            q[axis]! += step;
            if (q[0]! < 0 || q[1]! < 0 || q[2]! < 0
              || q[0]! >= nx || q[1]! >= ny || q[2]! >= nz) continue;
            if (isInterface[q[0]! + nx * (q[1]! + ny * q[2]!)]) { near = true; break; }
          }
        if (near) dilated += 1;
      }
    rows.push({
      frame,
      acceptedCells: diagnostics.adaptiveAcceptedCellCount,
      acceptedRows: diagnostics.adaptiveAcceptedRowCount,
      pressureCells: diagnostics.adaptivePressureCellCount,
      pressureRows: diagnostics.adaptivePressureActiveRowCount,
      residentBricks: diagnostics.fluidBrickResidentCount,
      fineBricks: diagnostics.adaptiveFineBrickCount,
      coarseBricks: diagnostics.adaptiveCoarseBrickCount,
      surfaceBricks: diagnostics.adaptiveActivitySurfaceBrickCount,
      quietBricks: diagnostics.adaptiveActivityQuietBrickCount,
      hotBricks: diagnostics.adaptiveActivityHotBrickCount,
      topologyCommitted: diagnostics.adaptiveTopologyCommittedBrickCount,
      topologyPrepared: diagnostics.adaptiveTopologyPreparedBrickCount,
      denseLiquidVoxels: liquid,
      denseOccupiedVoxels: anyMass,
      denseInterfaceVoxels: interfaceCells,
      denseInterfaceDilatedVoxels: dilated,
    });
  }
  console.log(JSON.stringify({
    probe: "sparse-cm12-cell-population", scene: sceneName,
    denseLattice: nx * ny * nz, rows,
  }, null, 2));
  solver.destroy();
} finally {
  releaseWebGPUExclusiveLock();
}
