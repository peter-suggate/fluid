// Per-frame V vs phi ledger for Uniform Geometric on a thin-film scene.
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
const overrides: Record<string, string | number> = {};
for (const a of process.argv) if (a.startsWith("--set=")) { const [k, v] = a.slice(6).split(":"); overrides[k!] = Number.isNaN(Number(v)) ? v! : Number(v); }
await acquireWebGPUExclusiveLock("dawn-probe", "uniform-volume thin film");
let device: any, solver: any;
try {
  const dawn = await import(pathToFileURL(`${ROOT}/node_modules/webgpu/index.js`).href);
  Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const rawDevice = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  const shaderHooks: ((code: string) => string)[] = [];
  { const original = rawDevice.createShaderModule.bind(rawDevice); rawDevice.createShaderModule = (d: any) => original({ ...d, code: shaderHooks.reduce((c, f) => f(c), d.code as string) }); }
  device = managedGPUDevice(rawDevice, { requireWorkerRealm: false });
  const errors: string[] = []; device.addEventListener("uncapturederror", (e: any) => { e.preventDefault(); errors.push(e.error.message); });
  const patches: Record<string, [string, string][]> = {
    // V-aware pressure phi: a cell holding V >= half its capacity owns a row, with a V-derived ghost distance.
    vphi: [["return uvPhi(vec3f(clampCell(p))+vec3f(0.5));", "let q=clampCell(p);let hh=min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));return min(uvPhi(vec3f(q)+vec3f(0.5)),hh*(0.5-volume(q)/max(cellOpenFraction(q),1e-6)));"]],
    // Landed volumePressureRows narrowed to cells the projection abandons (no phi-liquid face neighbour). Needs the toggle on.
    rescue: [["return min(phi,h*(0.5-volume(q)/open));", "let vphi=h*(0.5-volume(q)/open);if(phi<0.0||vphi>=0.0){return phi;}for(var rescueAxis=0;rescueAxis<3;rescueAxis+=1){for(var rescueSide=-1;rescueSide<=1;rescueSide+=2){var n=q;n[rescueAxis]+=rescueSide;if(valid(n)&&uvPhi(vec3f(n)+vec3f(0.5))<0.0){return phi;}}}return max(vphi,-0.5*h);"]],
    // Wall continuation probes each incident closed plane on its own as well as the diagonal.
    wall: [["  if(!contact||uvOpen(clampCell(vec3i(floor(interior))))<=1e-5){return advected;}\n  let continued=uvPhi(uvTrace(interior,params.dimsDt.w));\n  return select(advected,min(advected,continued),continued<0.0);",
      "  if(!contact){return advected;}var best=advected;\n  for(var axis=0u;axis<3u;axis++){if(interior[axis]==p[axis]){continue;}var single=p;single[axis]=interior[axis];\n    if(uvOpen(clampCell(vec3i(floor(single))))<=1e-5){continue;}let c=uvPhi(uvTrace(single,params.dimsDt.w));if(c<0.0){best=min(best,c);}}\n  if(uvOpen(clampCell(vec3i(floor(interior))))>1e-5){let c=uvPhi(uvTrace(interior,params.dimsDt.w));if(c<0.0){best=min(best,c);}}\n  return best;"]],
  };
  const active = arg("patch", "").split(",").filter(Boolean); const hits: Record<string, number> = {};
  if (active.length) shaderHooks.push(input => { let code = input;
      for (const name of active) for (const [from, to] of patches[name]!) if (code.includes(from)) { code = code.split(from).join(to); hits[name] = (hits[name] ?? 0) + 1; }
      return code; });
  const scene = sceneDocument(getSceneDefinition(arg("scene", "corner-brick-drop")));
  const values = resolveMethodValues(uniformVolumeMethod, "balanced", overrides);
  solver = await uniformVolumeMethod.createSolverAsync!(device, scene, "balanced", values, undefined, () => {}, new AbortController().signal);
  await solver.waitForSimulationReady?.();
  if (active.length) console.log(JSON.stringify({ patchHits: hits }));
  const nx = solver.info.nx, ny = solver.info.ny, nz = solver.info.nz;
  const h = scene.container.height_m / ny;
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
  if (arg("method", "uniform-volume") === "uniform") {
    const { uniformMethod } = await lib("lib/methods/uniform/method.ts"); solver.destroy();
    solver = await uniformMethod.createSolverAsync!(device, scene, "balanced", resolveMethodValues(uniformMethod, "balanced", overrides), undefined, () => {}, new AbortController().signal);
    await solver.waitForSimulationReady?.();
    for (let frame = 0; frame <= Number(arg("frames", "30")); frame++) {
      if (frame > 0) { while (!solver.advanceTo(frame / 30, [])) await new Promise(setImmediate); await solver.awaitFrameCompletion?.(); await device.queue.onSubmittedWorkDone(); }
      if (frame % 3) continue; const V = await read(solver.volumeTexture); let mass = 0, visible = 0, floorVisible = 0, maxV = 0;
      for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) { const v = V[x + nx * (y + ny * z)]!; mass += v; maxV = Math.max(maxV, v); if (v > 0.5) { visible++; if (y === 0) floorVisible++; } }
      console.log(JSON.stringify({ paper: true, frame, mass: +mass.toFixed(2), cellsAboveHalf: visible, floorCellsAboveHalf: floorVisible, maxV: +maxV.toFixed(2) })); }
    process.exitCode = 0; throw new Error("paper control done");
  }
  const frames = Number(arg("frames", "30")); const lineFrames = arg("line-frames", "").split(",").filter(Boolean).map(Number); const S = 4; const ledger: any[] = []; let last: any;
  for (let frame = 0; frame <= frames; frame++) {
    if (frame > 0) { while (!solver.advanceTo(frame / 30, [])) await new Promise(setImmediate); await solver.awaitFrameCompletion?.(); await device.queue.onSubmittedWorkDone(); }
    const V = await read(solver.volumeTexture); const phi = await read(solver.vertexPhiTexture);
    const P = (x: number, y: number, z: number) => phi[x + (nx + 1) * (y + (ny + 1) * z)]!;
    let mass = 0, phiVolume = 0, negCentres = 0, airSideV = 0, orphanV = 0, orphanCells = 0, maxV = 0, minPhi = Infinity, wetCells = 0, overfull = 0;
    const rowV = new Array(ny).fill(0), rowPhi = new Array(ny).fill(0);
    const floorV: number[] = [], floorPhiFrac: number[] = [];
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      const v = V[x + nx * (y + ny * z)]!; mass += v; rowV[y] += v; maxV = Math.max(maxV, v); if (v > 1e-6) wetCells++; if (v > 1.05) overfull += v - 1;
      const c: number[] = []; for (let k = 0; k < 8; k++) c.push(P(x + (k & 1), y + ((k >> 1) & 1), z + (k >> 2)));
      const lo = Math.min(...c), centre = c.reduce((a, b) => a + b, 0) / 8; minPhi = Math.min(minPhi, lo);
      let frac = 0;
      if (lo < 0) { if (Math.max(...c) <= 0) frac = 1; else { let n = 0;
        for (let k = 0; k < S; k++) for (let j = 0; j < S; j++) for (let i = 0; i < S; i++) { const fx = (i + .5) / S, fy = (j + .5) / S, fz = (k + .5) / S; let s = 0;
          for (let q = 0; q < 8; q++) s += c[q]! * ((q & 1) ? fx : 1 - fx) * (((q >> 1) & 1) ? fy : 1 - fy) * ((q >> 2) ? fz : 1 - fz);
          if (s < 0) n++; } frac = n / S ** 3; } }
      phiVolume += frac; rowPhi[y] += frac; if (centre < 0) negCentres++;
      if (centre > 0) airSideV += v; if (lo > 0) { orphanV += v; if (v > 1e-6) orphanCells++; }
      if (y === 0) { floorV.push(v); floorPhiFrac.push(frac); }
    }
    const entry = { frame, t: frame / 30, mass, phiVolume, negCentres, airSideV, orphanV, orphanCells, wetCells, maxV, overfull, minPhiCells: minPhi / h,
      rowV: rowV.slice(0, 4).map(v => +v.toFixed(1)), rowPhi: rowPhi.slice(0, 4).map(v => +v.toFixed(1)) };
    // Overfull V split by whether the holding cell owns a pressure row, and an x-line through the film.
    let overfullDry = 0, overfullWet = 0, maxSpeed = 0;
    const U = await read(solver.velocityTexture, 4); const ux = solver.velocityTexture.width, uy = solver.velocityTexture.height;
    const centrePhi = (x: number, y: number, z: number) => { let s = 0; for (let k = 0; k < 8; k++) s += P(x + (k & 1), y + ((k >> 1) & 1), z + (k >> 2)); return s / 8; };
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      const v = V[x + nx * (y + ny * z)]!; const e = Math.max(0, v - 1); if (centrePhi(x, y, z) < 0) overfullWet += e; else overfullDry += e;
      const o = 4 * (x + ux * (y + uy * z)); maxSpeed = Math.max(maxSpeed, Math.hypot(U[o]!, U[o + 1]!, U[o + 2]!)); }
    Object.assign(entry, { overfullDry, overfullWet, maxSpeed });
    // Stage attribution: the scratch texture still holds this step's advected, not yet redistanced, phi.
    if (frame > 0 && solver.vertexPhiScratch && overrides.redistance !== "off") { const pre = await read(solver.vertexPhiScratch);
      const Q = (x: number, y: number, z: number) => pre[x + (nx + 1) * (y + (ny + 1) * z)]!; let preVolume = 0, preNeg = 0;
      for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) { const c: number[] = []; for (let k = 0; k < 8; k++) c.push(Q(x + (k & 1), y + ((k >> 1) & 1), z + (k >> 2)));
        const lo = Math.min(...c); if (c.reduce((a, b) => a + b, 0) < 0) preNeg++; if (lo >= 0) continue; if (Math.max(...c) <= 0) { preVolume++; continue; } let n = 0;
        for (let k = 0; k < S; k++) for (let j = 0; j < S; j++) for (let i = 0; i < S; i++) { const fx = (i + .5) / S, fy = (j + .5) / S, fz = (k + .5) / S; let t = 0;
          for (let q = 0; q < 8; q++) t += c[q]! * ((q & 1) ? fx : 1 - fx) * (((q >> 1) & 1) ? fy : 1 - fy) * ((q >> 2) ? fz : 1 - fz); if (t < 0) n++; } preVolume += n / S ** 3; }
      Object.assign(entry, { preRedistanceVolume: preVolume, preRedistanceNegCentres: preNeg });
      if (lineFrames.includes(frame)) { const z = Number(arg("line-z", "12")); const f = (a: number[]) => a.map(v => v.toFixed(2).padStart(6)).join("");
        console.log(`--- frame ${frame} PRE-redistance x-line at z=${z}`);
        console.log("pre y=0 vtx " + f([...Array(nx + 1).keys()].map(x => Q(x, 0, z) / h)));
        console.log("pre y=1 vtx " + f([...Array(nx + 1).keys()].map(x => Q(x, 1, z) / h))); } }
    if (lineFrames.includes(frame)) { const z = Number(arg("line-z", "12")); const f = (a: number[]) => a.map(v => v.toFixed(2).padStart(6)).join("");
      const xs = [...Array(nx).keys()];
      console.log(`--- frame ${frame} x-line at z=${z}`);
      console.log("phi y=0 vtx " + f([...Array(nx + 1).keys()].map(x => P(x, 0, z) / h)));
      console.log("phi y=1 vtx " + f([...Array(nx + 1).keys()].map(x => P(x, 1, z) / h)));
      console.log("phi centre  " + f(xs.map(x => centrePhi(x, 0, z) / h)));
      console.log("V y=0       " + f(xs.map(x => V[x + nx * (0 + ny * z)]!)));
      console.log("V y=1       " + f(xs.map(x => V[x + nx * (1 + ny * z)]!)));
      console.log("u.x y=0     " + f(xs.map(x => U[4 * (x + ux * (0 + uy * z))]!)));
      console.log("u.y y=0     " + f(xs.map(x => U[4 * (x + ux * (0 + uy * z)) + 1]!))); }
    if (process.argv.includes("--stats") && frame % 30 === 0) Object.assign(entry, { representedVolumeDrift: (await solver.readStats()).representedVolumeDrift });
    ledger.push(entry); last = { floorV, floorPhiFrac };
    if (!process.argv.includes("--quiet")) console.log(JSON.stringify(entry));
  }
  const fmt = (a: number[]) => { let s = ""; for (let z = 0; z < nz; z++) { s += a.slice(z * nx, (z + 1) * nx).map(v => v < 0.005 ? "  . " : v.toFixed(2).padStart(4)).join("") + "\n"; } return s; };
  console.log("floor-layer V (y=0), final frame:\n" + fmt(last.floorV));
  console.log("floor-layer phi fraction (y=0), final frame:\n" + fmt(last.floorPhiFrac));
  writeFileSync(arg("output", `${import.meta.dirname}/thin-ledger.json`), JSON.stringify({ overrides, ledger, errors }, null, 1));
  if (errors.length) console.log("VALIDATION ERRORS", errors.slice(0, 3));
} finally { solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock(); }
