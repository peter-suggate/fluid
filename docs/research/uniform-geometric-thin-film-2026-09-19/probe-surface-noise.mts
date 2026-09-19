// Surface-noise ledger for Uniform Geometric's V-aware pressure phi, on a scene that comes to rest.
// As the slosh decays, the 3x3 residual of a height field is what is left over: noise.
const ROOT = new URL("../../..", import.meta.url).pathname.replace(/\/$/, "");
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import { writeFileSync } from "node:fs";
const lib = (p: string) => import(pathToFileURL(`${ROOT}/${p}`).href);
const { createProcessRetainedDawnGPU } = await lib("lib/harness/node-dawn-provider.ts");
const { managedGPUDevice } = await lib("lib/core/gpu-compilation-manager.ts");
const { requiredFluidDeviceLimits } = await lib("lib/core/webgpu-device-limits.ts");
const { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } = await lib("lib/harness/webgpu-smoke-isolation.ts");
const { sceneDocument } = await lib("lib/core/scene-definition.ts");
const { getSceneDefinition } = await lib("lib/core/scenes.ts");
const { uniformVolumeMethod } = await lib("lib/methods/uniform/uniform-volume-method.ts");
const { resolveMethodValues } = await lib("lib/core/method-contract.ts");
const arg = (n: string, f: string) => process.argv.find(a => a.startsWith(`--${n}=`))?.slice(n.length + 3) ?? f;
// Arms are the live toggle's values. The attribution arms (grant, noair, noliquid, rescue) were shader rewrites of the
// first landed expression, `return min(phi,h*(0.5-volume(q)/open));`; their ledgers are surface-noise-dam-break*.json.
const TOGGLE: Record<string, string> = { off: "off", abandoned: "abandoned", all: "all" };
const armPhi: Record<string, (phi: number, v: number, h: number, nearLiquid?: boolean) => number> = {
  abandoned: (phi, v, h, near) => { const p = h * (0.5 - v); return phi < 0 || p >= 0 || near ? phi : Math.max(p, -0.5 * h); },
  off: phi => phi, all: (phi, v, h) => Math.min(phi, h * (0.5 - v)),
  grant: (phi, v, h) => { const p = h * (0.5 - v); return phi < 0 || p >= 0 ? phi : Math.max(p, -0.5 * h); },
  noair: (phi, v, h) => { const p = h * (0.5 - v); return p >= 0 ? phi : Math.min(phi, p); },
  noliquid: (phi, v, h) => phi < 0 ? phi : Math.min(phi, h * (0.5 - v)),
};
const arms = arg("arms", "off,abandoned,all").split(","); const frames = Number(arg("frames", "90")); const every = Number(arg("every", "10"));
const gravity = arg("gravity", "1.5,-9.81,0.7").split(",").map(Number);
await acquireWebGPUExclusiveLock("dawn-probe", "uniform-volume surface noise");
let device: any, solver: any; const report: any = { scene: arg("scene", "water-box-dam-break"), gravity, arms: {} };
try {
  const dawn = await import(pathToFileURL(`${ROOT}/node_modules/webgpu/index.js`).href);
  Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const rawDevice = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  device = managedGPUDevice(rawDevice, { requireWorkerRealm: false });
  const errors: string[] = []; device.addEventListener("uncapturederror", (e: any) => { e.preventDefault(); errors.push(e.error.message); });
  const read = async (tex: GPUTexture, channels = 1) => {
    const row = Math.ceil(tex.width * 4 * channels / 256) * 256;
    const raw = device.createBuffer({ size: row * tex.height * tex.depthOrArrayLayers, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const e = device.createCommandEncoder(); e.copyTextureToBuffer({ texture: tex }, { buffer: raw, bytesPerRow: row, rowsPerImage: tex.height }, [tex.width, tex.height, tex.depthOrArrayLayers]);
    device.queue.submit([e.finish()]); await raw.mapAsync(GPUMapMode.READ);
    const src = new Float32Array(raw.getMappedRange()); const out = new Float32Array(tex.width * tex.height * tex.depthOrArrayLayers * channels);
    for (let z = 0; z < tex.depthOrArrayLayers; z++) for (let y = 0; y < tex.height; y++)
      out.set(src.subarray((z * tex.height + y) * row / 4, (z * tex.height + y) * row / 4 + tex.width * channels), (z * tex.height + y) * tex.width * channels);
    raw.unmap(); raw.destroy(); return out;
  };
  for (const arm of arms) {
    const scene = sceneDocument(getSceneDefinition(report.scene)); // The solver only reads gravity.y, so the disturbance is a brick of water dropped into the pool off-centre.
    // (--drop adds one. NB an additive seed over tank-fill gets V but no phi: with rows off it hangs in the air, frozen.)
    if (process.argv.includes("--drop")) { scene.fluid.initialBrickSeeds_m = [{ x: -0.4, y: 0.98, z: 0 }]; scene.fluid.initialBrickSeedsAdditive = true; } scene.duration_s = frames / 30 + 1;
    const values = resolveMethodValues(uniformVolumeMethod, "balanced", { volumePressureRows: TOGGLE[arm]! });
    solver = await uniformVolumeMethod.createSolverAsync!(device, scene, "balanced", values, undefined, () => {}, new AbortController().signal);
    await solver.waitForSimulationReady?.();
    const nx = solver.info.nx, ny = solver.info.ny, nz = solver.info.nz; const h = scene.container.height_m / ny; const ledger: any[] = [];
    for (let frame = 0; frame <= frames; frame++) {
      if (frame > 0) { while (!solver.advanceTo(frame / 30, [])) await new Promise(setImmediate); await solver.awaitFrameCompletion?.(); await device.queue.onSubmittedWorkDone(); }
      if (frame % every || frame < Number(arg("from", "0"))) continue;
      const V = await read(solver.volumeTexture), phi = await read(solver.vertexPhiTexture), U = await read(solver.velocityTexture, 4);
      const ux = solver.velocityTexture.width, uy = solver.velocityTexture.height;
      const C = new Float32Array(nx * ny * nz); const at = (x: number, y: number, z: number) => x + nx * (y + ny * z);
      for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) { let s = 0;
        for (let k = 0; k < 8; k++) s += phi[x + (k & 1) + (nx + 1) * (y + ((k >> 1) & 1) + (ny + 1) * (z + (k >> 2)))]!; C[at(x, y, z)] = s / 8; }
      // Height field of the rendered surface (phi only), and columns that cross it more than once.
      const H = new Float32Array(nx * nz); let multiCross = 0;
      for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) { let top = -1, crossings = 0;
        for (let y = 0; y + 1 < ny; y++) if ((C[at(x, y, z)]! < 0) !== (C[at(x, y + 1, z)]! < 0)) { crossings++; if (C[at(x, y, z)]! < 0) top = y; }
        if (crossings > 1) multiCross++;
        if (top < 0) { H[x + nx * z] = C[at(x, 0, z)]! < 0 ? ny : 0; continue; } const l = -C[at(x, top, z)]!, a = C[at(x, top + 1, z)]!; H[x + nx * z] = top + 0.5 + l / (l + a); }
      let rough = 0, roughN = 0, roughMax = 0;
      for (let z = 1; z + 1 < nz; z++) for (let x = 1; x + 1 < nx; x++) { let m = 0; for (let dz = -1; dz <= 1; dz++) for (let dx = -1; dx <= 1; dx++) m += H[x + dx + nx * (z + dz)]!;
        const r = H[x + nx * z]! - m / 9; rough += r * r; roughN++; roughMax = Math.max(roughMax, Math.abs(r)); }
      // Velocity in the surface band: speed, and the vertical component's 3x3 lateral residual (grid-scale jitter).
      let bandN = 0, speed2 = 0, jitter2 = 0, maxSpeed = 0; const uyAt = (x: number, y: number, z: number) => U[4 * (x + ux * (y + uy * z)) + 1]!;
      for (let z = 1; z + 1 < nz; z++) for (let y = 0; y < ny; y++) for (let x = 1; x + 1 < nx; x++) { if (Math.abs(C[at(x, y, z)]!) > 1.5 * h) continue;
        const o = 4 * (x + ux * (y + uy * z)); const s = Math.hypot(U[o]!, U[o + 1]!, U[o + 2]!); speed2 += s * s; maxSpeed = Math.max(maxSpeed, s); bandN++;
        let m = 0; for (let dz = -1; dz <= 1; dz++) for (let dx = -1; dx <= 1; dx++) m += uyAt(x + dx, y, z + dz); const j = uyAt(x, y, z) - m / 9; jitter2 += j * j; }
      // Ghost-fraction audit: what this arm's pressurePhi does to each liquid-air face, against pure phi.
      const near = new Uint8Array(nx * ny * nz); for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) { if (C[at(x, y, z)]! >= 0) continue;
        if (x > 0) near[at(x - 1, y, z)] = 1; if (x + 1 < nx) near[at(x + 1, y, z)] = 1; if (y > 0) near[at(x, y - 1, z)] = 1; if (y + 1 < ny) near[at(x, y + 1, z)] = 1; if (z > 0) near[at(x, y, z - 1)] = 1; if (z + 1 < nz) near[at(x, y, z + 1)] = 1; }
      const f = armPhi[arm]!; const P = (i: number) => f(C[i]!, V[i]!, h, near[i] === 1); const theta = (l: number, a: number) => Math.min(1, Math.max(0.05, -l / (-l + a)));
      let faces = 0, moved = 0, dTheta = 0, dThetaMax = 0, airClamped = 0, liquidOverridden = 0, granted = 0, grantedIsolated = 0, relP = 0;
      for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) { const i = at(x, y, z); if (P(i) >= 0) continue;
        if (C[i]! >= 0) { granted++; let near = false;
          for (const [dx, dy, dz] of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]] as const) { const X = x + dx, Y = y + dy, Z = z + dz;
            if (X >= 0 && Y >= 0 && Z >= 0 && X < nx && Y < ny && Z < nz && C[at(X, Y, Z)]! < 0) near = true; } if (!near) grantedIsolated++; }
        for (const [dx, dy, dz] of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]] as const) { const X = x + dx, Y = y + dy, Z = z + dz;
          if (X < 0 || Y < 0 || Z < 0 || X >= nx || Y >= ny || Z >= nz) continue; const n = at(X, Y, Z); if (P(n) < 0) continue; faces++;
          if (C[i]! >= 0 || C[n]! < 0) { moved++; continue; } const t0 = theta(C[i]!, C[n]!), t1 = theta(P(i), P(n)); const d = Math.abs(t1 - t0);
          if (d > 1e-4) { moved++; dTheta += d; dThetaMax = Math.max(dThetaMax, d); relP += Math.abs(t1 / t0 - 1); if (P(n) < C[n]! - 1e-6 * h) airClamped++; if (P(i) < C[i]! - 1e-6 * h) liquidOverridden++; } } }
      // V's own height field (column sum; single-valued pool) against the phi height field it is supposed to match.
      const HV = new Float32Array(nx * nz); for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) HV[x + nx * z] += V[at(x, y, z)]!;
      let vRough = 0, mismatch = 0;
      for (let z = 1; z + 1 < nz; z++) for (let x = 1; x + 1 < nx; x++) { let m = 0; for (let dz = -1; dz <= 1; dz++) for (let dx = -1; dx <= 1; dx++) m += HV[x + dx + nx * (z + dz)]!;
        const r = HV[x + nx * z]! - m / 9; vRough += r * r; const d = HV[x + nx * z]! - H[x + nx * z]!; mismatch += d * d; }
      // The free surface as the PRESSURE SOLVE sees it under a given pressurePhi: top liquid row plus its ghost fraction.
      const pressureHeightRough = (g: (phi: number, v: number, h: number) => number) => { const HP = new Float32Array(nx * nz);
        for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) { let top = -1; for (let y = 0; y < ny; y++) if (g(C[at(x, y, z)]!, V[at(x, y, z)]!, h) < 0) top = y;
          if (top < 0) continue; const l = -g(C[at(x, top, z)]!, V[at(x, top, z)]!, h); const a = top + 1 < ny ? Math.max(0, g(C[at(x, top + 1, z)]!, V[at(x, top + 1, z)]!, h)) : l; HP[x + nx * z] = top + 0.5 + Math.min(1, Math.max(0.05, l / (l + a))); }
        let r2 = 0; for (let z = 1; z + 1 < nz; z++) for (let x = 1; x + 1 < nx; x++) { let m = 0; for (let dz = -1; dz <= 1; dz++) for (let dx = -1; dx <= 1; dx++) m += HP[x + dx + nx * (z + dz)]!; r2 += (HP[x + nx * z]! - m / 9) ** 2; }
        return +Math.sqrt(r2 / roughN).toFixed(4); };
      // Where the granted rows sit: on top of a phi-liquid cell (the pool's own surface) or anywhere else, and how far V is from phi's fill there.
      let grantedOnSurface = 0, grantedV = 0, grantedPhiFill = 0;
      for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) { const i = at(x, y, z); if (C[i]! < 0 || V[i]! < 0.5) continue;
        grantedV += V[i]!; grantedPhiFill += Math.max(0, 0.5 - C[i]! / h); if (y > 0 && C[at(x, y - 1, z)]! < 0) grantedOnSurface++; }
      let mass = 0; for (const v of V) mass += v;
      const entry = { frame, mass: +mass.toFixed(3), roughRms: +Math.sqrt(rough / roughN).toFixed(4), roughMax: +roughMax.toFixed(3), multiCross, vHeightRough: +Math.sqrt(vRough / roughN).toFixed(4), vPhiHeightMismatch: +Math.sqrt(mismatch / roughN).toFixed(4),
        bandRmsSpeed: +Math.sqrt(speed2 / Math.max(1, bandN)).toFixed(4), bandJitterUy: +Math.sqrt(jitter2 / Math.max(1, bandN)).toFixed(4), maxSpeed: +maxSpeed.toFixed(3),
        pRoughPhi: pressureHeightRough(armPhi.off!), pRoughAll: pressureHeightRough(armPhi.all!), grantedOnSurface, grantedMeanV: +(grantedV / Math.max(1, granted)).toFixed(3), grantedMeanPhiFill: +(grantedPhiFill / Math.max(1, granted)).toFixed(3),
        faces, moved, meanDTheta: +(dTheta / Math.max(1, moved)).toFixed(3), dThetaMax: +dThetaMax.toFixed(3), meanRelPressure: +(relP / Math.max(1, moved)).toFixed(3), airClamped, liquidOverridden, granted, grantedIsolated };
      ledger.push(entry); console.log(arm.padEnd(9) + JSON.stringify(entry));
    }
    report.arms[arm] = { ledger }; solver.destroy(); solver = undefined;
  }
  report.errors = errors; if (errors.length) console.log("VALIDATION ERRORS", errors.slice(0, 3));
  writeFileSync(arg("output", `${import.meta.dirname}/surface-noise.json`), JSON.stringify(report, null, 1));
} finally { solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock(); }
