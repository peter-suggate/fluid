// Dam-break collapse audit: track the free-surface height profile, vertical
// velocity profile through the dam interior, and pressure vs hydrostatic to
// find why the column top rounds into a slow-falling dome.
// Usage: WEBGPU_NODE_MODULE=... FLUID_METHOD=tall-cell|uniform npx tsx tmp/tall-cell-audit/probe-collapse.ts
import { pathToFileURL } from "node:url";
import { tallCellMethod } from "../../lib/methods/tall-cell";
import { uniformMethod } from "../../lib/methods/uniform";
import { createSmokeScenario } from "../../tools/webgpu-smoke-scenarios";
import { initializeRigidBodies } from "../../lib/rigid-body";
import { damBreakFractions } from "../../lib/initial-fluid";

const modulePath = process.env.WEBGPU_NODE_MODULE!;
const { create, globals } = await import(pathToFileURL(modulePath).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
Object.assign(globalThis, globals);
const gpu = create(["backend=metal"]);
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });

const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
const device = await adapter!.requestDevice();
device.addEventListener("uncapturederror", (event: any) => console.error("UNCAPTURED:", (event as any).error.message));

const methodId = process.env.FLUID_METHOD ?? "tall-cell";
const method = methodId === "uniform" ? uniformMethod : tallCellMethod;
const scenario = createSmokeScenario("dam-break-ui");
const scene = scenario.scene;
if (process.env.PROBE_TANK === "1") scene.fluid.initialCondition = "tank-fill";
if (process.env.PROBE_FILL) scene.container.fillFraction = Number(process.env.PROBE_FILL);
const bodies = initializeRigidBodies(scene.rigidBodies);
const values = Object.fromEntries(method.params.map((p) => [p.key, p.default])) as Record<string, string | number>;
if (process.env.PROBE_PRESSURE_CYCLES) values.pressureCycles = Number(process.env.PROBE_PRESSURE_CYCLES);
const solver: any = method.createSolver!(device, scene, "balanced", values, () => {});
const info = solver.info;
const dt = scene.numerics.maxDt_s;
console.log(`method=${methodId} grid=${info.nx}x${info.ny}x${info.nz} storedNy=${info.storedNy} kind=${info.gridKind} dt=${dt}`);

const dam = damBreakFractions(scene.container.fillFraction);
const damXCells = dam.width * info.nx, damZCells = dam.depth * info.nz, damHCells = dam.height * info.ny;
console.log(`dam extent: x<${damXCells.toFixed(1)} z<${damZCells.toFixed(1)} height=${damHCells.toFixed(1)} cells; cell h=${info.cellSize_m.toFixed?.(4) ?? info.cellSize_m}`);
const cx = Math.floor(damXCells / 2), cz = Math.floor(damZCells / 2);

async function readTexture(texture: GPUTexture, w: number, h: number, d: number, components: number) {
  const bytesPerRow = Math.ceil(w * 4 * components / 256) * 256;
  const buffer = device.createBuffer({ size: bytesPerRow * h * d, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow, rowsPerImage: h }, [w, h, d]);
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const raw = new Float32Array(buffer.getMappedRange().slice(0));
  buffer.destroy();
  return { raw, rowFloats: bytesPerRow / 4 };
}

const isTall = info.gridKind !== "uniform";

async function snapshot() {
  const { nx, ny, nz, storedNy } = info;
  const vol = await readTexture(solver.volumeTexture, nx, storedNy, nz, 1);
  const vel = await readTexture(solver.velocityTexture, nx, storedNy, nz, 4);
  const prs = await readTexture((solver as any).pressureA, nx, storedNy, nz, 1);
  let bases: Float32Array | null = null;
  if (isTall) {
    const b = await readTexture(solver.columnBaseTexture, nx, nz, 1, 1);
    bases = new Float32Array(nx * nz);
    for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) bases[x + nx * z] = b.raw[x + b.rowFloats * z];
  }
  const packedAlpha = (x: number, py: number, z: number) => vol.raw[x + vol.rowFloats * (py + storedNy * z)];
  const packedPressure = (x: number, py: number, z: number) => prs.raw[x + prs.rowFloats * (py + storedNy * z)];
  const pressureAt = (x: number, y: number, z: number): number => {
    if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) return 0;
    if (!isTall) return packedPressure(x, y, z);
    const base = Math.round(bases![x + nx * z]);
    if (y < base && base > 0) {
      const t = Math.min(1, Math.max(0, y / Math.max(base - 1, 1)));
      return packedPressure(x, 0, z) * (1 - t) + packedPressure(x, 1, z) * t;
    }
    const py = 2 + y - base;
    if (py < 2 || py >= storedNy) return 0;
    return packedPressure(x, py, z);
  };
  const packedVel = (x: number, py: number, z: number, c: number) => vel.raw[4 * x + vel.rowFloats * (py + storedNy * z) + c];
  const alphaAt = (x: number, y: number, z: number): number => {
    if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) return 0;
    if (!isTall) return packedAlpha(x, y, z);
    const base = Math.round(bases![x + nx * z]);
    if (y < base && base > 0) return Math.min(1, Math.max(0, packedAlpha(x, 0, z)));
    const py = 2 + y - base;
    if (py < 2 || py >= storedNy) return 0;
    return packedAlpha(x, py, z);
  };
  const velAt = (x: number, y: number, z: number, c: number): number => {
    if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) return 0;
    if (!isTall) return packedVel(x, y, z, c);
    const base = Math.round(bases![x + nx * z]);
    if (y < base && base > 0) {
      const t = Math.min(1, Math.max(0, y / Math.max(base - 1, 1)));
      return packedVel(x, 0, z, c) * (1 - t) + packedVel(x, 1, z, c) * t;
    }
    const py = 2 + y - base;
    if (py < 2 || py >= storedNy) return 0;
    return packedVel(x, py, z, c);
  };
  return { alphaAt, velAt, pressureAt, bases };
}

function surfaceHeight(alphaAt: (x: number, y: number, z: number) => number, x: number, z: number) {
  for (let y = info.ny - 1; y >= 0; y -= 1) if (alphaAt(x, y, z) >= 0.5) return y + 1;
  return 0;
}

const targets = (process.env.PROBE_TIMES ?? "0.05,0.1,0.15,0.2,0.3,0.4,0.5").split(",").map(Number);
let t = 0;
for (const target of targets) {
  while (t < target - 1e-9) { t = Math.min(target, t + dt); solver.advanceTo(t, bodies); }
  await solver.readStats();
  await device.queue.onSubmittedWorkDone();
  const { alphaAt, velAt, pressureAt } = await snapshot();
  const h = scene.container.height_m / info.ny;
  const rho = scene.fluid.density_kg_m3, g = 9.80665;
  const divAt = (x: number, y: number, z: number) => {
    const vxm = x > 0 ? velAt(x - 1, y, z, 0) : 0, vym = y > 0 ? velAt(x, y - 1, z, 1) : 0, vzm = z > 0 ? velAt(x, y, z - 1, 2) : 0;
    return (velAt(x, y, z, 0) - vxm + velAt(x, y, z, 1) - vym + velAt(x, y, z, 2) - vzm) / h;
  };
  // Height profile along x at dam center z
  const profile: number[] = [];
  for (let x = 0; x < info.nx; x += 1) profile.push(surfaceHeight(alphaAt, x, cz));
  // Column water content (sum of alpha) along x at dam center z — immune to
  // the wet-threshold and shows the real mass distribution.
  const water: number[] = [];
  for (let x = 0; x < info.nx; x += 1) { let s = 0; for (let y = 0; y < info.ny; y += 1) s += Math.min(1, Math.max(0, alphaAt(x, y, cz))); water.push(s); }
  // KE and mass
  let ke = 0, mass = 0, maxDown = 0, maxUp = 0;
  for (let z = 0; z < info.nz; z += 1) for (let y = 0; y < info.ny; y += 1) for (let x = 0; x < info.nx; x += 1) {
    const a = Math.min(1, Math.max(0, alphaAt(x, y, z)));
    if (a <= 0) continue;
    mass += a;
    const vx = velAt(x, y, z, 0), vy = velAt(x, y, z, 1), vz = velAt(x, y, z, 2);
    ke += a * (vx * vx + vy * vy + vz * vz);
    if (a >= 0.5) { maxDown = Math.min(maxDown, vy); maxUp = Math.max(maxUp, vy); }
  }
  const loc = (l?: { x: number; y: number; z: number }) => l ? `${l.x},${l.y},${l.z}` : "-";
  console.log(`\n=== t=${t.toFixed(3)}s mass=${mass.toFixed(0)} KE=${ke.toFixed(1)} vy[min,max]=[${maxDown.toFixed(2)},${maxUp.toFixed(2)}] maxDivAfter=${info.maxDivergenceAfter_s?.toExponential(2)}@${loc(info.maxDivergenceAfterLocation)} relRes=${info.pressureRelativeResidual?.toExponential(2)} res@${loc(info.maxPressureResidualLocation)} maxP=${info.maxPressure_Pa?.toExponential(2)}@${loc(info.maxPressureLocation)} maxSpd@${loc(info.maxSpeedLocation)} cfl=${info.maxComponentCfl?.toFixed(2)} flags=${info.stabilityFlags?.join("|") || "-"}`);
  console.log(`surface@z=${cz}: ${profile.map((v) => String(v).padStart(3)).join("")}`);
  console.log(`water  @z=${cz}: ${water.map((v) => String(Math.round(v)).padStart(3)).join("")}`);
  // Vertical profile at the dam interior column (cx, cz): alpha and vy
  const rows: string[] = [];
  for (let y = info.ny - 1; y >= 0; y -= 1) {
    const a = alphaAt(cx, y, cz);
    if (a < 0.01 && y > surfaceHeight(alphaAt, cx, cz) + 2) continue;
    const top = surfaceHeight(alphaAt, cx, cz);
    const hydro = rho * g * Math.max(0, top - y - 0.5) * h;
    rows.push(`  y=${String(y).padStart(2)} a=${a.toFixed(2)} vy=${velAt(cx, y, cz, 1).toFixed(3)} vx=${velAt(cx, y, cz, 0).toFixed(3)} p=${pressureAt(cx, y, cz).toFixed(0)} pHydro=${hydro.toFixed(0)} div=${divAt(cx, y, cz).toFixed(2)}`);
  }
  console.log(`column (${cx},${cz}):`);
  console.log(rows.join("\n"));
  if (process.env.PROBE_CORNER === "1") {
    console.log("corner neighborhood (world cells):");
    for (const [x, y, z] of [[0,0,1],[1,0,1],[2,0,1],[1,0,0],[1,0,2],[1,1,1],[1,0,3],[0,0,3],[2,0,3],[1,1,3],[1,0,4],[3,0,1],[1,2,1],[0,1,1]] as const) {
      console.log(`  (${x},${y},${z}) a=${alphaAt(x,y,z).toFixed(3)} p=${pressureAt(x,y,z).toFixed(1)} v=(${velAt(x,y,z,0).toFixed(4)},${velAt(x,y,z,1).toFixed(4)},${velAt(x,y,z,2).toFixed(4)}) div=${divAt(x,y,z).toFixed(3)}`);
    }
  }
}
solver.destroy(); device.destroy();
Reflect.deleteProperty(globalThis, "navigator");
