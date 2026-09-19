// Energy ledger for Uniform Geometric's phi/V agreement toggles: is a stage dissipative, and which one?
// Per frame: V centre of mass (the slosh), kinetic energy over phi-liquid cells, potential energy of V and of phi,
// and the speed of cells that turned phi-liquid since the previous frame against the liquid they joined.
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
const ARMS: Record<string, Record<string, string>> = { off: {}, compact: { volumeCompaction: "on" }, seed: { phiSeedFromVolume: "on" },
  shift: { volumeCompaction: "on", phiAgreement: "on" }, "shift-only": { phiAgreement: "on" }, "all-rows": { volumeCompaction: "on", phiSeedFromVolume: "on", phiAgreement: "on", volumePressureRows: "all" }, "compact-rows": { volumeCompaction: "on", volumePressureRows: "all" }, rows: { volumePressureRows: "all" }, all: { volumeCompaction: "on", phiSeedFromVolume: "on", phiAgreement: "on" } };
const SET = Object.fromEntries(arg("set", "").split(",").filter(Boolean).map(kv => kv.split(":") as [string, string]));
const arms = arg("arms", "off,compact,seed,shift,all").split(","); const frames = Number(arg("frames", "240"));
await acquireWebGPUExclusiveLock("dawn-probe", "uniform-volume energy");
let device: any, solver: any; const report: any = { scene: arg("scene", "water-box-dam-break"), set: SET, arms: {} };
try {
  const dawn = await import(pathToFileURL(`${ROOT}/node_modules/webgpu/index.js`).href);
  Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const rawDevice = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  // --patch=keep|keepab: does V keeping its OWN face velocity stop the dissipation? project zeroes every face between two
  // rowless cells and the extension then overwrites it from the nearest phi-liquid, so mass that leaves phi loses its momentum.
  // The rewrite makes a cell holding V >= 1/2 (keepab: and with no phi-liquid face neighbour) an extension SOURCE, and stops
  // project zeroing its faces: they stay advected + gravity, i.e. ballistic. Hook the RAW device; the managed proxy recurses.
  const AUTH = "textureStore(volumeOut,id,vec4f(0.5-pressurePhi(id)/min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z))));";
  const KEEP_FN = `fn uvKeepsVelocity(q:vec3i)->bool{if(!valid(q)||volume(q)<0.5||cellOpenFraction(q)<0.99999){return false;}
    if(KEEP_ABANDONED){for(var axis=0;axis<3;axis+=1){for(var side=-1;side<=1;side+=2){var n=q;n[axis]+=side;if(pressureLiquid(n)){return false;}}}}return true;}
`;
  const mode = arg("patch", ""); const patch: [string, string][] = mode ? [
    ["fn storeExtrapolationAuthority(id:vec3i){", KEEP_FN.replace("KEEP_ABANDONED", mode === "keepab" ? "true" : "false") + "fn storeExtrapolationAuthority(id:vec3i){"],
    [AUTH, AUTH.replace("vec4f(0.5-", "vec4f(select(0.0,1.0,uvKeepsVelocity(id))+0.5-")],
    ["}else{v[axis]=0.0;}", "}else{v[axis]=select(0.0,v[axis],uvKeepsVelocity(id)||uvKeepsVelocity(neighbor));}"]] : [];
  const patchHits: number[] = []; report.patch = mode; report.patchHits = patchHits;
  { const original = rawDevice.createShaderModule.bind(rawDevice); rawDevice.createShaderModule = (d: any) => { let code = d.code as string;
      patch.forEach(([from, to], k) => { if (code.includes(from)) { code = code.split(from).join(to); patchHits[k] = (patchHits[k] ?? 0) + 1; } }); return original({ ...d, code }); }; }
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
    const scene = sceneDocument(getSceneDefinition(report.scene)); scene.duration_s = frames / 30 + 1;
    const values = resolveMethodValues(uniformVolumeMethod, "balanced", { ...ARMS[arm]!, ...SET });
    solver = await uniformVolumeMethod.createSolverAsync!(device, scene, "balanced", values, undefined, () => {}, new AbortController().signal);
    await solver.waitForSimulationReady?.();
    const nx = solver.info.nx, ny = solver.info.ny, nz = solver.info.nz; const h = scene.container.height_m / ny; const g = 9.81; const ledger: any[] = [];
    let wasLiquid: Uint8Array | undefined, prevSpeed: Float32Array | undefined;
    for (let frame = 0; frame <= frames; frame++) {
      if (frame > 0) { while (!solver.advanceTo(frame / 30, [])) await new Promise(setImmediate); await solver.awaitFrameCompletion?.(); await device.queue.onSubmittedWorkDone(); }
      const V = await read(solver.volumeTexture), phi = await read(solver.vertexPhiTexture), U = await read(solver.velocityTexture, 4);
      const ux = solver.velocityTexture.width, uy = solver.velocityTexture.height;
      const at = (x: number, y: number, z: number) => x + nx * (y + ny * z);
      const liquid = new Uint8Array(nx * ny * nz);
      const speedNow = new Float32Array(nx * ny * nz);
      let mass = 0, mx = 0, my = 0, ke = 0, peV = 0, pePhi = 0, cells = 0, fresh = 0, freshSpeed = 0, seeded = 0, seededSpeed = 0, seededV = 0, oldSpeed = 0, old = 0;
      for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) { let s = 0;
        for (let k = 0; k < 8; k++) s += phi[x + (k & 1) + (nx + 1) * (y + ((k >> 1) & 1) + (ny + 1) * (z + (k >> 2)))]!;
        const v = V[at(x, y, z)]!; mass += v; mx += v * (x + 0.5); my += v * (y + 0.5); peV += v * g * (y + 0.5) * h;
        { const o = 4 * (x + ux * (y + uy * z)); speedNow[at(x, y, z)] = Math.hypot(U[o]!, U[o + 1]!, U[o + 2]!); }
        if (s >= 0) continue; liquid[at(x, y, z)] = 1; cells++; pePhi += g * (y + 0.5) * h;
        const o = 4 * (x + ux * (y + uy * z)); const speed2 = U[o]! ** 2 + U[o + 1]! ** 2 + U[o + 2]! ** 2; ke += 0.5 * speed2;
        if (wasLiquid && !wasLiquid[at(x, y, z)]) { fresh++; freshSpeed += Math.sqrt(speed2);
          // Seeded rather than advected: no cell within two of it was liquid a frame ago. Its V has been riding some velocity; compare.
          let near = false; for (let dz = -2; dz <= 2 && !near; dz++) for (let dy = -2; dy <= 2 && !near; dy++) for (let dx = -2; dx <= 2; dx++) {
            const X = x + dx, Y = y + dy, Z = z + dz; if (X >= 0 && Y >= 0 && Z >= 0 && X < nx && Y < ny && Z < nz && wasLiquid[at(X, Y, Z)]) { near = true; break; } }
          if (!near) { seeded++; seededSpeed += Math.sqrt(speed2); seededV += v; } } else { old++; oldSpeed += Math.sqrt(speed2); } }
      // Row loss with mass aboard: a cell that was phi-liquid a frame ago, is not now, and still holds V >= 0.5. project zeroes
      // its air-air faces and the extension overwrites the rest, so whatever it was carrying is gone: sum V*|u_prev|.
      let dropped = 0, droppedMomentum = 0, liquidMomentum = 0;
      for (let i = 0; i < liquid.length; i++) { if (liquid[i]) liquidMomentum += V[i]! * speedNow[i]!;
        if (wasLiquid && wasLiquid[i] && !liquid[i] && V[i]! >= 0.5) { dropped++; droppedMomentum += V[i]! * prevSpeed![i]!; } }
      wasLiquid = liquid; prevSpeed = speedNow;
      ledger.push({ frame, mass: +mass.toFixed(2), comX: +(mx / mass).toFixed(4), comY: +(my / mass).toFixed(4), ke: +ke.toFixed(2), peV: +peV.toFixed(2), pePhi: +pePhi.toFixed(2),
        liquidCells: cells, dropped, droppedMomentum: +droppedMomentum.toFixed(2), liquidMomentum: +liquidMomentum.toFixed(2), seeded, seededMeanSpeed: +(seededSpeed / Math.max(1, seeded)).toFixed(3), seededMeanV: +(seededV / Math.max(1, seeded)).toFixed(2), fresh, freshMeanSpeed: +(freshSpeed / Math.max(1, fresh)).toFixed(3), oldMeanSpeed: +(oldSpeed / Math.max(1, old)).toFixed(3) });
    }
    report.arms[arm] = { ledger }; solver.destroy(); solver = undefined; console.log(arm, "done");
  }
  report.errors = errors; if (errors.length) console.log("VALIDATION ERRORS", errors.slice(0, 3));
  writeFileSync(arg("output", `${import.meta.dirname}/energy.json`), JSON.stringify(report));
} finally { solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock(); }
