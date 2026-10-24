/** Fast-front reach probe (tiered-adaptivity plan §3.3, Phase 4): runs the
 * fast-jet fixture (tools/fast-jet-scene.ts) at one cell size under
 * reservoir-seep's NB-FLIP profile and reports, every `every` steps, the
 * page pool, the h tiles, the liquid and its drift, the catch chamber's
 * volume (discharge), the jet's mean x in two slabs below the throat
 * (trajectory) and the largest speed in h a step. Stops at the first pool or
 * solver fatal and prints it with its step. Browser GPU must be off.
 * node --import tsx tools/probe-nbflip-fast-jet.ts <cell_mm> <steps> <every> [--blocks=N] [--param=key=value ...] [--arms=<json>] [--lock=<file>] [--hold=170] [--depth=m] [--reservoir=m]
 * --param overrides an NB-FLIP method value over the reservoir-seep profile;
 * --arms runs each of a JSON list of such overrides in turn on one device
 * (later arms reuse the first's compiled pipelines).
 * --lock: build each arm's solver (shader compilation, CPU) without the GPU
 * lock, then take the lock (lockf on that file) for the arm's steps, at most
 * --hold seconds: an arm that runs out of time stops and says so. Without
 * it, run the whole probe under lockf. */
import { resolve } from "node:path";
import { spawn } from "node:child_process";
import { existsSync, unlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { pathToFileURL } from "node:url";
import assert from "node:assert/strict";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { resolveMethodValues } from "../lib/core/method-contract";
import { uniformNarrowBandMethod } from "../lib/methods/uniform/uniform-narrow-band-method";
import { UNIFORM_PAGE_POOL_STATUS as S, uniformPagePoolFatalMessage, type UniformPagePool } from "../lib/methods/uniform/uniform-page-pool";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { createNbflipVolumeProbe } from "./probe-nbflip-volume";
import { createFastJetScene, FAST_JET_METHOD_OVERRIDES } from "./fast-jet-scene";

const positional = process.argv.slice(2).filter(a => !a.startsWith("--"));
const option = (name: string) => process.argv.find(a => a.startsWith(`--${name}=`))?.slice(name.length + 3);
const cell_mm = Number(positional[0]), steps = Number(positional[1]), every = Number(positional[2] ?? 30);
assert.ok(cell_mm > 0 && Number.isInteger(steps) && steps >= 1 && Number.isInteger(every) && every >= 1,
  "usage: probe-nbflip-fast-jet.ts <cell_mm> <steps> <every> [--blocks=N] [--param=key=value]");
const params: Record<string, string | number | boolean> = {};
for (const a of process.argv.filter(a => a.startsWith("--param="))) {
  const [k, v] = a.slice(8).split("=");
  params[k!] = v === "true" ? true : v === "false" ? false : Number.isFinite(Number(v)) ? Number(v) : v!;
}
const arms = (JSON.parse(option("arms") ?? "[{}]") as Record<string, string | number | boolean>[]).map(arm => ({ ...params, ...arm }));
const lockFile = option("lock"), hold_s = Number(option("hold") ?? 170);
/** Takes the GPU lock: a lockf child that holds it until the returned
 * release, or until this process exits. */
async function takeLock(file: string): Promise<() => void> {
  const flag = resolve(tmpdir(), `fast-jet-lock-${process.pid}`);
  spawn("lockf", [file, "sh", "-c", `touch '${flag}'; while [ -e '${flag}' ] && kill -0 ${process.pid} 2>/dev/null; do sleep 0.2; done; rm -f '${flag}'`], { stdio: "ignore", detached: false });
  while (!existsSync(flag)) await new Promise(r => setTimeout(r, 200));
  return () => { try { unlinkSync(flag); } catch { /* released */ } };
}
const h = cell_mm / 1000, dt = 1 / 60;
const jet = createFastJetScene({ cellSize_m: h, blocks: option("blocks") ? Number(option("blocks")) : undefined,
  depth_m: option("depth") ? Number(option("depth")) : undefined, reservoirWidth_m: option("reservoir") ? Number(option("reservoir")) : undefined });
const { scene, cells: [nx, ny, nz], wall: [low, high], throat: [holeY], throatCells: n } = jet;
const method = uniformNarrowBandMethod;
let device: GPUDevice | undefined, solver: WebGPUUniformReferenceSolver | undefined, read: GPUBuffer | undefined;
const errors: string[] = [];
try {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve("node_modules/webgpu/index.js")).href); Object.assign(globalThis, dawn.globals);
  const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter(); assert.ok(adapter);
  device = managedGPUDevice(await adapter.requestDevice({ requiredFeatures: (["float32-filterable", "subgroups"] as GPUFeatureName[]).filter(f => adapter.features.has(f)), requiredLimits: requiredFluidDeviceLimits(adapter.limits) }), { requireWorkerRealm: false });
  device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); console.error(e.error.message); });
  for (const arm of arms) {
  const values = resolveMethodValues(method, "balanced", { ...method.appDefaults, timeStep: "scene", ...FAST_JET_METHOD_OVERRIDES, ...arm });
  const built = performance.now();
  solver = await method.createSolverAsync!(device, structuredClone(scene), "balanced", values, undefined, () => {}) as WebGPUUniformReferenceSolver;
  const release = lockFile ? await takeLock(lockFile) : () => {};
  const locked = performance.now();
  const frame = (solver as unknown as { mixedFrame: { pagePool: UniformPagePool; ownership: { capacity: { fineTiles: number } } } }).mixedFrame;
  const pool = frame.pagePool, l = pool.layout, status = new Uint32Array(S.words);
  read = device.createBuffer({ label: "Fast jet pool status", size: 4 * S.words, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const probe = await createNbflipVolumeProbe(device, solver);
  const { nx: px, ny: py, nz: pz } = solver.info;
  // The solver's lattice is the container's rounded up to whole pages.
  assert.ok(px >= nx && py >= ny && pz >= nz, "the probe's lattice holds the fixture's");
  console.log(`fast-jet ${cell_mm} mm: lattice ${nx}x${ny}x${nz}, wall x [${jet.wall.join(", ")}), throat y ${holeY}; ideal exit ${jet.exitSpeed_m_s.toFixed(2)} m/s = ${jet.travelPerStep_h.toFixed(1)} h a step; pool ${l.blocks} blocks (${l.slots} pages), closureCap ${l.closureCap}; h capacity ${frame.ownership.capacity.fineTiles} tiles; allocated ${(solver.info.allocatedBytes / 1048576).toFixed(0)} MiB; built in ${((locked - built) / 1000).toFixed(0)} s (lock wait included); params ${JSON.stringify(arm)}`);
  const slab = (y_m: number) => [Math.floor(y_m / h - 0.005 / h), Math.ceil(y_m / h + 0.005 / h)] as const;
  const y0 = (holeY + n / 2) * h, slabs = [y0 - 0.02, y0 - 0.04].map(slab);
  let residentPeak = 0, finePeak = 0, speedPeak = 0, last = 0;
  const fines: number[] = [], residents: number[] = [];
  for (let step = 1; step <= steps; step++) {
    if (lockFile && performance.now() - locked > 1000 * hold_s) { console.log(`stopped before step ${step}: the ${hold_s} s lock budget is spent`); break; }
    let thrown: string | undefined;
    try { solver.advanceTo(step * dt, []); await solver.awaitFrameCompletion(); } catch (e) { const cause = (e as Error).cause; thrown = `${(e as Error).message}${cause instanceof Error ? `: ${cause.message}` : ""}`; }
    const encoder = device.createCommandEncoder(); pool.encodeReceipt(encoder, read, 0); device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ); status.set(new Uint32Array(read.getMappedRange())); read.unmap();
    const fatal = uniformPagePoolFatalMessage(l, status), failure = thrown ?? solver.info.simulationPipelineError;
    const resident = status[S.resident]!, fine = solver.info.uniformMixedFineTiles ?? 0;
    residents.push(resident); fines.push(fine); residentPeak = Math.max(residentPeak, resident); finePeak = Math.max(finePeak, fine);
    if (step % every === 0 || step <= 3 || fatal || failure || step === steps) {
      const stats = failure ? undefined : await solver.readStats();
      const speed = stats?.maxSpeed_m_s ?? NaN; speedPeak = Math.max(speedPeak, speed || 0);
      let row = "";
      if (!failure) {
        const v = await probe.read();
        let caught = 0, held = 0; const jetX = slabs.map(() => [0, 0]);
        for (let z = 0; z < pz; z++) for (let y = 0; y < py; y++) for (let x = 0; x < low; x++) held += v[x + px * (y + py * z)]!;
        for (let z = 0; z < pz; z++) for (let y = 0; y < py; y++) for (let x = high; x < px; x++) {
          const c = v[x + px * (y + py * z)]!; if (!(c > 0)) continue; caught += c;
          slabs.forEach(([a, b], k) => { if (y >= a && y < b) { jetX[k]![0] += c * (x + 0.5 - high) * h; jetX[k]![1] += c; } });
        }
        const litres = caught * h * h * h * 1000, rate = (litres - last) / ((step % every === 0 ? every : 1) * dt); last = litres;
        // Head over the throat's centre from the reservoir's liquid, and the
        // discharge coefficient against Torricelli at that head.
        const head = held * h * h * h / jet.reservoirArea_m2 - y0, ideal = n * n * h * h * Math.sqrt(2 * 9.80665 * Math.max(head, 0));
        row = ` head ${head.toFixed(3)} m caught ${litres.toFixed(4)} L (${(rate * 1000).toFixed(2)} mL/s, Cd ${(rate / 1000 / ideal).toFixed(3)}) jet x ${jetX.map(([s, w]) => w! > 0.05 ? (s! / w!).toFixed(3) : "-").join(" ")} m`;
      }
      console.log(`step ${step} t ${(step * dt).toFixed(2)} resident ${resident} free ${status[S.free]} requests ${status[S.requests]} h tiles ${fine} particles ${solver.narrowBandFlipInfo?.particles ?? 0} liquid ${Math.round(stats?.volumeCellSum ?? NaN)} drift ${(stats?.volumeDrift ?? NaN).toFixed(4)} speed ${speed.toFixed(2)} m/s ${(speed * dt / h).toFixed(1)} h/step${row} wall ${((performance.now() - locked) / 1000).toFixed(0)} s fatal ${fatal ?? "none"}`);
    }
    if (failure || fatal) { if (failure) console.log(`solver at step ${step}: ${failure}`); process.exitCode = 1; break; }
  }
  release();
  console.log(`peak resident ${residentPeak} of ${l.slots}; peak h tiles ${finePeak}; peak speed ${speedPeak.toFixed(2)} m/s (${(speedPeak * dt / h).toFixed(1)} h/step)`);
  console.log(`h tiles mean ${(fines.reduce((a, b) => a + b, 0) / fines.length).toFixed(0)}, resident mean ${(residents.reduce((a, b) => a + b, 0) / residents.length).toFixed(1)}`);
  console.log(`errors ${JSON.stringify(errors)}`);
  probe.destroy(); read.destroy(); read = undefined; solver.destroy(); solver = undefined;
  }
} finally { read?.destroy(); solver?.destroy(); await device?.queue.onSubmittedWorkDone(); device?.destroy(); }
