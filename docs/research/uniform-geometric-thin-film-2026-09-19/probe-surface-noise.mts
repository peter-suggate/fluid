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
const TOGGLE: Record<string, string> = { off: "off", abandoned: "abandoned", all: "all", compact: "off", shift: "off" };
// WP1 of docs/uniform-geometric-phi-volume-agreement-handoff.md as a shader rewrite, so the plan can be judged on the scene
// before production code exists. Sharpening admits every phi-liquid cell, and a liquid cell may pour ALL of its V into a
// face neighbour with smaller phi (deeper); toward anything else it still offers only its surplus over phi's fill.
const PATCHES: Record<string, [string, string][]> = { compact: [
  ["let admitted=uvOpen(id)>0.99999&&abs(phi)<params.tuning.y*h;", "let admitted=uvOpen(id)>0.99999&&phi<params.tuning.y*h;"],
  ["uvEdges[i].weight[3]=select(0.0,dose*max(own-desired,0.0),admitted);", "uvEdges[i].weight[3]=select(0.0,dose*select(max(own-desired,0.0),own,phi<0.0),admitted);"],
  ["let ab=select(0.0,min(uvEdges[i].weight[3],uvEdges[j].weight[4]),(middle<=epsilon&&!relayB)||inwardA);\n    let ba=select(0.0,min(uvEdges[j].weight[3],uvEdges[i].weight[4]),(middle<=epsilon&&!relayA)||inwardB);",
   "let pourDose=clamp(params.tuning.x,0.0,1.0);let legacyA=min(uvEdges[i].weight[3],pourDose*max(volume(id)-textureLoad(gammaIn,id,0).x,0.0));let legacyB=min(uvEdges[j].weight[3],pourDose*max(volume(q)-textureLoad(gammaIn,q,0).x,0.0));\n    let capA=select(legacyA,uvEdges[i].weight[3],phiA<0.0&&phiB<phiA-epsilon);let capB=select(legacyB,uvEdges[j].weight[3],phiB<0.0&&phiA<phiB-epsilon);\n    let ab=select(0.0,min(capA,uvEdges[j].weight[4]),(middle<=epsilon&&!relayB)||inwardA);\n    let ba=select(0.0,min(capB,uvEdges[i].weight[4]),(middle<=epsilon&&!relayA)||inwardB);"],
] };
// WP2 as a rewrite of uvAdvectPhi: a band vertex gathers, over the 8^3 cells around it with tent weights, the residual
// R = sum w (V - uvTarget) and the cut-cell weight A, from start-of-step V and gamma (both consistent with start-of-step phi),
// and moves along its own normal by gain*R/A cells, clamped to 0.1 cell, with a deadband. Brute force: prototype only.
const SHIFT: [string, string][] = [
  ["@compute @workgroup_size(4,4,4)\nfn uvAdvectPhi(", `fn uvAgreementShift(p:vec3f)->f32{
  let h=min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));var R=0.0;var A=0.0;let base=vec3i(p);
  for(var dz=-4;dz<4;dz++){for(var dy=-4;dy<4;dy++){for(var dx=-4;dx<4;dx++){
    let c=base+vec3i(dx,dy,dz);if(!valid(c)||uvOpen(c)<0.99999){continue;}
    let g=textureLoad(gammaIn,c,0).x;let v=volume(c);if(g<=0.0&&v<=0.0){continue;}
    if(abs(uvPhi(vec3f(c)+vec3f(0.5)))>=1.5*h){continue;}
    let o=abs(vec3f(f32(dx),f32(dy),f32(dz))+vec3f(0.5))/4.5;let w=(1.0-o.x)*(1.0-o.y)*(1.0-o.z);
    R+=w*(v-g);if(g>0.0&&g<1.0){A+=w;}}}}
  if(A<1.0){return 0.0;}let s=R/A;if(abs(s)<0.02){return 0.0;}
  return h*clamp(SHIFT_GAIN*s,-SHIFT_CLAMP,SHIFT_CLAMP);
}
@compute @workgroup_size(4,4,4)
fn uvAdvectPhi(`],
  ["textureStore(uvPhiOut,vec3i(gid),vec4f(uvSourcePhi(p,uvReleasedWalls(p,uvClosedWallPhi(p,uvPhi(uvTrace(p,params.dimsDt.w)))))));",
   "let advected=uvSourcePhi(p,uvReleasedWalls(p,uvClosedWallPhi(p,uvPhi(uvTrace(p,params.dimsDt.w)))));let hh=min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));\n  textureStore(uvPhiOut,vec3i(gid),vec4f(select(advected,advected-uvAgreementShift(p),abs(advected)<2.0*hh)));"],
];
// WP3 (seed half) on top of the shift: a vertex whose 4^3 cell neighbourhood has NO phi-liquid centre, but whose 8 adjacent
// cells average more than a quarter full, takes phi = min(phi, h(0.5 - Vbar)). V is read as geometry only where phi offers none.
const SEED_FN = `fn uvSeedPhi(p:vec3f,phi:f32)->f32{
  let h=min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));let base=vec3i(p);var sum=0.0;var n=0.0;
  for(var k=0u;k<8u;k++){let c=base-vec3i(1)+uvCorner(k);if(valid(c)&&uvOpen(c)>=0.99999){sum+=volume(c);n+=1.0;}}
  if(n<1.0||sum/n<=0.25){return phi;}
  for(var dz=-2;dz<2;dz++){for(var dy=-2;dy<2;dy++){for(var dx=-2;dx<2;dx++){let c=base+vec3i(dx,dy,dz);
    if(valid(c)&&uvPhi(vec3f(c)+vec3f(0.5))<0.0){return phi;}}}}
  return min(phi,h*(0.5-sum/n));
}
`;
// The 4h sharpening work map only admits band tiles; the rewrite needs the dense schedule.
const EXTRA: Record<string, Record<string, string>> = { compact: { sharpeningWorkMap: "off" }, shift: { sharpeningWorkMap: "off" } };
PATCHES.seed = [...PATCHES.compact!, [SHIFT[0]![0], SEED_FN + SHIFT[0]![1].replace("SHIFT_GAIN", arg("gain", "0.25")).split("SHIFT_CLAMP").join(arg("clamp", "0.1"))],
  [SHIFT[1]![0], SHIFT[1]![1].replace("select(advected,advected-uvAgreementShift(p),abs(advected)<2.0*hh)", "uvSeedPhi(p,select(advected,advected-uvAgreementShift(p),abs(advected)<2.0*hh))")]];
TOGGLE.seed = "off"; EXTRA.seed = { sharpeningWorkMap: "off" };
PATCHES.shift = [...PATCHES.compact!, ...SHIFT.map(([a, b]) => [a, b.replace("SHIFT_GAIN", arg("gain", "0.25")).split("SHIFT_CLAMP").join(arg("clamp", "0.1"))] as [string, string])];
// The production toggles (the compact/shift/seed rewrites above targeted the pre-toggle shader text and no longer hit).
// t-compact-dense is t-compact on the dense sharpening schedule: the work map's extra admission must not change the result.
const GAIN = { phiAgreement: "on", phiAgreementGain: arg("gain", "0.05"), phiAgreementClamp: arg("clamp", "0.02") };
for (const [name, extra] of Object.entries({ "t-compact": { volumeCompaction: "on" }, "t-compact-dense": { volumeCompaction: "on", sharpeningWorkMap: "off" },
  "t-seed": { volumeCompaction: "on", phiSeedFromVolume: "on" }, "t-seed-only": { phiSeedFromVolume: "on" }, "t-shift": { volumeCompaction: "on", ...GAIN },
  "t-all": { volumeCompaction: "on", phiSeedFromVolume: "on", ...GAIN } })) { TOGGLE[name] = "off"; EXTRA[name] = extra as unknown as Record<string, string>; }
// --set=key:value,key:value overrides method values on every arm.
const SET = Object.fromEntries(arg("set", "").split(",").filter(Boolean).map(kv => kv.split(":") as [string, string]));
const armPhi: Record<string, (phi: number, v: number, h: number, nearLiquid?: boolean) => number> = {
  abandoned: (phi, v, h, near) => { const p = h * (0.5 - v); return phi < 0 || p >= 0 || near ? phi : Math.max(p, -0.5 * h); },
  off: phi => phi, "t-compact": phi => phi, "t-compact-dense": phi => phi, "t-seed": phi => phi, "t-seed-only": phi => phi, "t-shift": phi => phi, "t-all": phi => phi, compact: phi => phi, shift: phi => phi, seed: phi => phi, all: (phi, v, h) => Math.min(phi, h * (0.5 - v)),
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
  let patch: [string, string][] = []; const patchHits: number[] = [];
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
    patch = PATCHES[arm] ?? []; patchHits.length = 0;
    const scene = sceneDocument(getSceneDefinition(report.scene)); // The solver only reads gravity.y, so the disturbance is a brick of water dropped into the pool off-centre.
    // (--drop adds one. NB an additive seed over tank-fill gets V but no phi: with rows off it hangs in the air, frozen.)
    if (process.argv.includes("--drop")) { scene.fluid.initialBrickSeeds_m = [{ x: -0.4, y: 0.98, z: 0 }]; scene.fluid.initialBrickSeedsAdditive = true; } scene.duration_s = frames / 30 + 1;
    const values = resolveMethodValues(uniformVolumeMethod, "balanced", { volumePressureRows: TOGGLE[arm]!, ...(EXTRA[arm] ?? {}), ...SET });
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
      // WP0 of docs/uniform-geometric-phi-volume-agreement-handoff.md: the 4h-tile residual R = sum(V - phi fill) and cut-cell
      // count A over the interface band, and the normal shift R/A they imply (cells). Is that field smooth, and does the
      // band hold the deficit? Phi fill is a 4^3 subsample of the trilinear cell, as in probe-thin-film.mts.
      const fillOf = (x: number, y: number, z: number) => { const c: number[] = []; for (let k = 0; k < 8; k++) c.push(phi[x + (k & 1) + (nx + 1) * (y + ((k >> 1) & 1) + (ny + 1) * (z + (k >> 2)))]!);
        const lo = Math.min(...c), hi = Math.max(...c); if (lo >= 0) return 0; if (hi <= 0) return 1; let n = 0;
        for (let k = 0; k < 4; k++) for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) { const fx = (i + .5) / 4, fy = (j + .5) / 4, fz = (k + .5) / 4; let t = 0;
          for (let q = 0; q < 8; q++) t += c[q]! * ((q & 1) ? fx : 1 - fx) * (((q >> 1) & 1) ? fy : 1 - fy) * ((q >> 2) ? fz : 1 - fz); if (t < 0) n++; } return n / 64; };
      const tx = Math.ceil(nx / 4), ty = Math.ceil(ny / 4), tz = Math.ceil(nz / 4); const R = new Float64Array(tx * ty * tz), A = new Float64Array(tx * ty * tz);
      let deficitAll = 0, deficitBand = 0, absBand = 0;
      for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) { const i = at(x, y, z); const fill = fillOf(x, y, z); const r = V[i]! - fill; deficitAll += r;
        if (Math.abs(C[i]!) >= 1.5 * h) continue; const t = (x >> 2) + tx * ((y >> 2) + ty * (z >> 2)); R[t] += r; deficitBand += r; absBand += Math.abs(r); if (fill > 0 && fill < 1) A[t] += 1; }
      // Where the residual lives, by depth from phi's surface: [cells, sum V, sum phi fill] per shell, plus a V histogram of the deep interior.
      const shells: Record<string, number[]> = { "air>1.5h": [0, 0, 0], "air band": [0, 0, 0], "liquid band": [0, 0, 0], "liquid 1.5-4h": [0, 0, 0], "liquid >4h": [0, 0, 0] }; const deepHist = [0, 0, 0, 0, 0];
      for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) { const i = at(x, y, z); const c = C[i]! / h;
        const k = c >= 1.5 ? "air>1.5h" : c >= 0 ? "air band" : c > -1.5 ? "liquid band" : c > -4 ? "liquid 1.5-4h" : "liquid >4h"; const e = shells[k]!; e[0]++; e[1] += V[i]!; e[2] += fillOf(x, y, z);
        if (c <= -4) deepHist[V[i]! < 0.25 ? 0 : V[i]! < 0.75 ? 1 : V[i]! < 0.98 ? 2 : V[i]! <= 1.02 ? 3 : 4]++; }
      for (const k in shells) shells[k] = shells[k]!.map(v => +v.toFixed(1));
      let tiles = 0, shiftSum = 0, shift2 = 0, pairs = 0, step2 = 0, shiftMax = 0;
      for (let z = 0; z < tz; z++) for (let y = 0; y < ty; y++) for (let x = 0; x < tx; x++) { const t = x + tx * (y + ty * z); if (A[t]! < 4) continue; const sh = R[t]! / A[t]!; tiles++; shiftSum += sh; shift2 += sh * sh; shiftMax = Math.max(shiftMax, Math.abs(sh));
        for (const [dx, dz] of [[1, 0], [0, 1]] as const) { const X = x + dx, Z = z + dz; if (X >= tx || Z >= tz) continue; const n = X + tx * (y + ty * Z); if (A[n]! < 4) continue; pairs++; step2 += (R[n]! / A[n]! - sh) ** 2; } }
      // WP2 validation: disjoint 4h tiles split R from A wherever the surface runs near a tile face. Overlapping patches do not:
      // (a) [1,2,1]^3 blur of the tile sums, (b) cell-lattice box filters of radius k applied twice (a tent). For each, the shift
      // R/A sampled at every column's top liquid cell, and its 3x3 lateral residual -- the roughness a unit-gain shift would inject.
      const rCell = new Float64Array(nx * ny * nz), aCell = new Float64Array(nx * ny * nz);
      for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) { const i = at(x, y, z); if (Math.abs(C[i]!) >= 1.5 * h) continue; const fill = fillOf(x, y, z); rCell[i] = V[i]! - fill; if (fill > 0 && fill < 1) aCell[i] = 1; }
      const box = (src: Float64Array, k: number) => { let a = src; for (let axis = 0; axis < 3; axis++) { const b = new Float64Array(a.length); const n = [nx, ny, nz][axis]!; const stride = [1, nx, nx * ny][axis]!;
          for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) { const c = [x, y, z][axis]!; let sum = 0; for (let d = -k; d <= k; d++) if (c + d >= 0 && c + d < n) sum += a[at(x, y, z) + d * stride]!; b[at(x, y, z)] = sum; } a = b; } return a; };
      const surfaceStats = (shiftAt: (x: number, y: number, z: number) => number) => { const S = new Float32Array(nx * nz); const ok = new Uint8Array(nx * nz);
        for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) { let top = -1; for (let y = 0; y + 1 < ny; y++) if (C[at(x, y, z)]! < 0 && C[at(x, y + 1, z)]! >= 0) top = y; if (top < 0) continue; const v = shiftAt(x, top, z); if (Number.isFinite(v)) { S[x + nx * z] = v; ok[x + nx * z] = 1; } }
        let n = 0, sum = 0, sum2 = 0, res2 = 0, resN = 0; for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) { if (!ok[x + nx * z]) continue; n++; sum += S[x + nx * z]!; sum2 += S[x + nx * z]! ** 2;
          let m = 0, c = 0; for (let dz = -1; dz <= 1; dz++) for (let dx = -1; dx <= 1; dx++) { const X = x + dx, Z = z + dz; if (X < 0 || Z < 0 || X >= nx || Z >= nz || !ok[X + nx * Z]) continue; m += S[X + nx * Z]!; c++; } if (c === 9) { res2 += (S[x + nx * z]! - m / 9) ** 2; resN++; } }
        return { columns: n, mean: +(sum / Math.max(1, n)).toFixed(3), rms: +Math.sqrt(sum2 / Math.max(1, n)).toFixed(3), lateralResidualRms: +Math.sqrt(res2 / Math.max(1, resN)).toFixed(4) }; };
      const shiftFields: Record<string, unknown> = {};
      { const blur = (src: Float64Array) => { const out = new Float64Array(src.length); for (let z = 0; z < tz; z++) for (let y = 0; y < ty; y++) for (let x = 0; x < tx; x++) { let sum = 0;
            for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const X = x + dx, Y = y + dy, Z = z + dz; if (X < 0 || Y < 0 || Z < 0 || X >= tx || Y >= ty || Z >= tz) continue; sum += src[X + tx * (Y + ty * Z)]! * (2 - Math.abs(dx)) * (2 - Math.abs(dy)) * (2 - Math.abs(dz)); } out[x + tx * (y + ty * z)] = sum; } return out; };
        const Rb = blur(R), Ab = blur(A); shiftFields.disjointTiles = surfaceStats((x, y, z) => { const t = (x >> 2) + tx * ((y >> 2) + ty * (z >> 2)); return A[t]! >= 4 ? R[t]! / A[t]! : NaN; });
        shiftFields.blurredTiles = surfaceStats((x, y, z) => { const t = (x >> 2) + tx * ((y >> 2) + ty * (z >> 2)); return Ab[t]! >= 32 ? Rb[t]! / Ab[t]! : NaN; }); }
      for (const k of [2, 4]) { const Rk = box(box(rCell, k), k), Ak = box(box(aCell, k), k); shiftFields[`cellTent_r${k}`] = surfaceStats((x, y, z) => Ak[at(x, y, z)]! > 1 ? Rk[at(x, y, z)]! / Ak[at(x, y, z)]! : NaN); }
      const tileShift = { shiftFields, tiles, meanShift: +(shiftSum / Math.max(1, tiles)).toFixed(3), rmsShift: +Math.sqrt(shift2 / Math.max(1, tiles)).toFixed(3), maxShift: +shiftMax.toFixed(3),
        neighbourStepRms: +Math.sqrt(step2 / Math.max(1, pairs)).toFixed(3), deficitAll: +deficitAll.toFixed(1), deficitInBand: +deficitBand.toFixed(1), cellAbsResidualInBand: +absBand.toFixed(1), shells, deepInteriorVHist_lt25_lt75_lt98_full_over: deepHist };
      let mass = 0, maxV = 0; for (const v of V) { mass += v; maxV = Math.max(maxV, v); }
      const entry = { frame, mass: +mass.toFixed(3), maxV: +maxV.toFixed(2), roughRms: +Math.sqrt(rough / roughN).toFixed(4), roughMax: +roughMax.toFixed(3), multiCross, vHeightRough: +Math.sqrt(vRough / roughN).toFixed(4), vPhiHeightMismatch: +Math.sqrt(mismatch / roughN).toFixed(4),
        bandRmsSpeed: +Math.sqrt(speed2 / Math.max(1, bandN)).toFixed(4), bandJitterUy: +Math.sqrt(jitter2 / Math.max(1, bandN)).toFixed(4), maxSpeed: +maxSpeed.toFixed(3),
        pRoughPhi: pressureHeightRough(armPhi.off!), pRoughAll: pressureHeightRough(armPhi.all!), grantedOnSurface, grantedMeanV: +(grantedV / Math.max(1, granted)).toFixed(3), grantedMeanPhiFill: +(grantedPhiFill / Math.max(1, granted)).toFixed(3),
        tileShift,
        faces, moved, meanDTheta: +(dTheta / Math.max(1, moved)).toFixed(3), dThetaMax: +dThetaMax.toFixed(3), meanRelPressure: +(relP / Math.max(1, moved)).toFixed(3), airClamped, liquidOverridden, granted, grantedIsolated };
      ledger.push(entry); console.log(arm.padEnd(9) + JSON.stringify(entry));
    }
    report.arms[arm] = { patchHits: [...patchHits], ledger }; solver.destroy(); solver = undefined;
  }
  report.errors = errors; if (errors.length) console.log("VALIDATION ERRORS", errors.slice(0, 3));
  writeFileSync(arg("output", `${import.meta.dirname}/surface-noise.json`), JSON.stringify(report, null, 1));
} finally { solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock(); }
