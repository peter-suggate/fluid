// Rigid->fluid coupling disruption probe: a kinematic sphere is dragged
// sinusoidally through a settled tank and the surface disturbance is compared
// against a stationary-sphere control. Strong coupling => the dragged run's
// surface deviation and fluid speed dwarf the control's.
// Usage: WEBGPU_NODE_MODULE=... PROBE_METHOD=uniform|quadtree-tall-cell PROBE_DRAG=0|1 npx tsx tmp/tall-cell-audit/probe-drag-sphere.ts
import { pathToFileURL } from "node:url";
import { uniformMethod } from "../../lib/methods/uniform";
import { quadtreeTallCellMethod } from "../../lib/methods/quadtree-tall-cell";
import { cloneScene, defaultScene } from "../../lib/model";
import { initializeRigidBodies, type RigidBodyState } from "../../lib/rigid-body";

const modulePath = process.env.WEBGPU_NODE_MODULE!;
const { create, globals } = await import(pathToFileURL(modulePath).href) as { create(o: string[]): GPU; globals: Record<string, unknown> };
Object.assign(globalThis, globals);
const gpu = create(["backend=metal"]);
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
const device = await adapter!.requestDevice();
device.addEventListener("uncapturederror", (event: unknown) => console.error("UNCAPTURED:", (event as { error: { message: string } }).error.message));

const drag = process.env.PROBE_DRAG !== "0";
const methodId = process.env.PROBE_METHOD === "quadtree-tall-cell" ? "quadtree-tall-cell" : "uniform";
const method = methodId === "quadtree-tall-cell" ? quadtreeTallCellMethod : uniformMethod;

const scene = cloneScene(defaultScene);
scene.sceneId = "probe-drag-sphere";
scene.fluid.initialCondition = "tank-fill";
scene.container.fillFraction = 0.55;
scene.container.top = "open";
scene.fluid.surfaceTension_N_m = 0;
delete scene.fluid.inflow;
scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 120;
const waterHeight = scene.container.height_m * scene.container.fillFraction;
const sphereY = waterHeight * 0.55;
scene.rigidBodies = [{
  id: "probe-sphere", name: "Probe sphere", shape: "sphere",
  dimensions_m: { x: 0.10, y: 0.10, z: 0.10 }, density_kg_m3: 5000,
  position_m: { x: 0, y: sphereY, z: 0 }, orientation: { w: 1, x: 0, y: 0, z: 0 },
  linearVelocity_m_s: { x: 0, y: 0, z: 0 }, angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
  restitution: 0.05, friction: 0.8, motion: "static"
}];

const values = Object.fromEntries(method.params.map((p) => [p.key, "default" in p ? p.default : 0])) as Record<string, string | number | boolean>;
const solver: unknown = method.createSolver!(device, scene, "balanced", values, undefined);
const s = solver as {
  info: { nx: number; ny: number; nz: number; storedNy?: number; maxSpeed_m_s?: number };
  advanceTo(t: number, bodies: RigidBodyState[]): boolean;
  readStats(): Promise<unknown>;
  volumeTexture: GPUTexture;
};
const dt = scene.numerics.maxDt_s;
const bodies = initializeRigidBodies(scene.rigidBodies);

// Sinusoidal sweep: amplitude 0.28 m, period 2 s, peak speed ~0.88 m/s.
const amplitude = 0.28, omega = Math.PI;
function driveBody(t: number) {
  const body = bodies[0];
  if (drag) {
    body.position_m.x = amplitude * Math.sin(omega * t);
    body.linearVelocity_m_s.x = amplitude * omega * Math.cos(omega * t);
  }
}

async function readVolume() {
  const { nx, nz } = s.info; const ny = s.info.storedNy ?? s.info.ny;
  const bytesPerRow = Math.ceil(nx * 4 / 256) * 256;
  const buffer = device.createBuffer({ size: bytesPerRow * ny * nz, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture: s.volumeTexture }, { buffer, bytesPerRow, rowsPerImage: ny }, [nx, ny, nz]);
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const raw = new Float32Array(buffer.getMappedRange().slice(0));
  buffer.destroy();
  return { raw, rowFloats: bytesPerRow / 4, ny };
}

async function surfaceStats() {
  const { nx, nz } = s.info;
  const { raw, rowFloats, ny } = await readVolume();
  const heights: number[] = [];
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    let column = 0;
    for (let y = 0; y < ny; y += 1) column += Math.max(0, Math.min(1, raw[x + rowFloats * (y + ny * z)]));
    heights.push(column);
  }
  const mean = heights.reduce((a, b) => a + b, 0) / heights.length;
  const variance = heights.reduce((a, b) => a + (b - mean) * (b - mean), 0) / heights.length;
  return { mean, std: Math.sqrt(variance), max: Math.max(...heights), min: Math.min(...heights) };
}

console.log(JSON.stringify({ method: methodId, drag, grid: [s.info.nx, s.info.ny, s.info.nz], dt, sphereY: Number(sphereY.toFixed(3)), waterHeight: Number(waterHeight.toFixed(3)) }));
let t = 0;
const initial = await surfaceStats();
console.log(JSON.stringify({ t: 0, ...initial }));
for (let target = 0.5; target <= 3.0001; target += 0.5) {
  while (t < target - 1e-9) {
    const next = Math.min(target, t + dt);
    driveBody(next);
    if (!s.advanceTo(next, bodies)) {
      // Quadtree topology rebuilds resolve on the event loop.
      await new Promise((resolve) => setTimeout(resolve, 0));
      continue;
    }
    t = next;
  }
  await s.readStats();
  await device.queue.onSubmittedWorkDone();
  const stats = await surfaceStats();
  console.log(JSON.stringify({ t: Number(t.toFixed(2)), heightMean: Number(stats.mean.toFixed(3)), heightStd: Number(stats.std.toFixed(4)), heightMin: Number(stats.min.toFixed(3)), heightMax: Number(stats.max.toFixed(3)), maxSpeed_m_s: Number((s.info.maxSpeed_m_s ?? 0).toFixed(3)), sphereX: Number(bodies[0].position_m.x.toFixed(3)), sphereVx: Number(bodies[0].linearVelocity_m_s.x.toFixed(3)) }));
}
process.exit(0);
