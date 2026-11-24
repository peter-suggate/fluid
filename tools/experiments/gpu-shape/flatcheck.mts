// The FLAT guard on a scene (brick-abi.md §3; docs/plans/gpu-shape/m1-distinct-once.md): the directory's brick -> F
// slot table against the walk page -> child after every step of the gate run, by the debug pass setFlatCheck (on the
// GPU: FAIL.UNSUPPORTED, info 0x464c....) and by BrickLayout.flatMismatches on the read-back dir (on the CPU). Then a
// save, STEPS2 more steps and a restore: FLAT again. Then a live solid edit (a fill over the 3^3 bricks around a
// resident liquid brick, brick-solid-edit.test.ts's applySceneUniforms path), checked at once and after each of
// STEPS2 steps, and its undo, the same. Census settings as stepab.mts (brick engine, fixed dt 1/60,
// presentation off, moored bodies on paddleboats).
//   SCENE=seep|dam|letters|paddleboats|<id> STEPS=100 [STEPS2=10] [TREE=tree]
//   tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/experiments/gpu-shape/flatcheck.mts
// Exit 0: 0 mismatches at every check; 1 otherwise (or a GPU error).
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const root = resolve(process.env.TREE ?? resolve(import.meta.dirname, "../../.."));
const SCENES: Record<string, string> = { seep: "reservoir-seep", dam: "high-resolution-dam-break", letters: "nbflip-figure-8-letters", paddleboats: "paddleboats" };
const sceneKey = process.env.SCENE ?? "dam", sceneId = SCENES[sceneKey] ?? sceneKey;
const N = Number(process.env.STEPS ?? 100), N2 = Number(process.env.STEPS2 ?? 10);

await import(`${root}/lib/methods/index.ts`);
const { getMethod } = await import(`${root}/lib/core/method-registry.ts`);
const { resolveMethodValues } = await import(`${root}/lib/core/method-contract.ts`);
const { sceneDocument } = await import(`${root}/lib/core/scene-definition.ts`);
const { getSceneDefinition } = await import(`${root}/lib/core/scenes.ts`);
const { managedGPUDevice } = await import(`${root}/lib/core/gpu-compilation-manager.ts`);
const { requiredFluidDeviceLimits } = await import(`${root}/lib/core/webgpu-device-limits.ts`);
const { createProcessRetainedDawnGPU, paceDawnEventPolling } = await import(`${root}/lib/harness/node-dawn-provider.ts`);
const { mooredMotorBodies } = await import(`${root}/tools/moored-motor-bodies.ts`);
const { FLAT_CHECK_GROUPS, brickSetKernels } = await import(`${root}/lib/methods/narrow-band/brick-set.wgsl.ts`);
const { H, F_CLASS, REC, brickKeyCoord } = await import(`${root}/lib/methods/narrow-band/brick-abi.ts`);
const { sceneWithSolidStroke } = await import(`${root}/lib/core/solid-world.ts`);
const { snapshot } = await import(`${root}/lib/methods/narrow-band/brick-readback.ts`);
const methodId = "uniform-narrow-band-flip", method = getMethod(methodId);
const definition = getSceneDefinition(sceneId), profile = definition.methodProfile, base = sceneDocument(definition);
const scene = { ...base, numerics: { ...base.numerics, fixedDt_s: 1 / 60, maxDt_s: 1 / 60 } };
const quality = profile?.methodId === methodId ? profile.quality : "balanced";
const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === methodId ? profile.overrides : {}), engine: "brick" });
paceDawnEventPolling();
const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve(root, "node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter();
const raw: GPUDevice = await adapter.requestDevice({ requiredFeatures: ["subgroups", "float32-filterable"].filter((f: string) => adapter.features.has(f)), requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
let gpuErrors = 0;
raw.addEventListener?.("uncapturederror", (ev: any) => { gpuErrors++; console.log("GPU ERROR", ev.error?.message); });
const device = managedGPUDevice(raw, { requireWorkerRealm: false });
const s: any = await method.createSolverAsync!(device, scene, quality, values, undefined, () => {});
await s.waitForSimulationReady?.();
s.setPresentationEnabled?.(false);
const e = s.brickEngine, L = e.layout, check = brickSetKernels(L).setFlatCheck;
const dt = 1 / 60;
const bodies: (t: number) => unknown[] = sceneKey === "paddleboats" || (scene.rigidBodies?.length ?? 0) > 0 ? mooredMotorBodies(scene, dt) : () => [];

let bad = 0, checks = 0, failSeen = 0, lastBricks = -1, churn = 0;
const verify = async (at: string) => {
  await e.fence();
  const before = new Uint32Array(await e.read("set", 0, 2));
  await e.run([{ kernel: check, dispatch: { direct: [FLAT_CHECK_GROUPS, 1, 1] } }]);
  const after = new Uint32Array(await e.read("set", 0, 2)), dir: Uint32Array = await e.read("dir");
  const cpu = L.flatMismatches(dir);
  let bricks = 0; for (let i = L.dir.flat; i < L.dir.frec; i++) if (dir[i] !== 0xffffffff) bricks++;
  if (lastBricks >= 0 && bricks !== lastBricks) churn++;
  lastBricks = bricks; checks++;
  // setFlatCheck latches FAIL.UNSUPPORTED; a FAIL bit already set before it is the step's own (not FLAT's).
  const gpu = (after[H.FAIL]! & ~before[H.FAIL]!) !== 0 || (after[H.FAIL_INFO]! >>> 16) === 0x464c && before[H.FAIL_INFO] !== after[H.FAIL_INFO];
  if (before[H.FAIL]) failSeen++;
  if (cpu.count || gpu) { bad++; console.log(`${sceneKey} ${at}: FLAT mismatches ${cpu.count} (${cpu.first ?? "-"}), GPU check ${gpu ? `latched 0x${after[H.FAIL_INFO]!.toString(16)}` : "clean"}`); }
  return bricks;
};

await verify("boot");
for (let k = 1; k <= N; k++) {
  while (!s.advanceTo(k * dt + 1e-12, bodies(k * dt))) { if (s.info.simulationFailure) throw new Error(s.info.simulationFailure.message); await new Promise<void>(r => setImmediate(r)); }
  const bricks = await verify(`step ${k}`);
  if (k === N) console.log(`${sceneKey} step ${N}: ${bricks} bricks in FLAT`);
}
// Save, more steps, restore: the restored FLAT is the saved walk's.
const S0 = await e.save();
for (let k = N + 1; k <= N + N2; k++) { while (!s.advanceTo(k * dt + 1e-12, bodies(k * dt))) await new Promise<void>(r => setImmediate(r)); await verify(`step ${k}`); }
await e.restore(S0); await verify("restore");
// The edit: around a resident liquid F brick (the median one by key), clipped to the lattice.
let t = N + N2;
const advance = async (k: number, at: string) => { for (let j = 0; j < k; j++) { t++; while (!s.advanceTo(t * dt + 1e-12, bodies(t * dt))) { if (s.info.simulationFailure) throw new Error(s.info.simulationFailure.message); await new Promise<void>(r => setImmediate(r)); } await verify(`${at} step ${t}`); } };
const snap = await snapshot(e), liquid = [...snap.bricks.keys()].filter(k => (snap.dir[L.dir.frec + 32 * snap.bricks.get(k)! + REC.CLASS]! & F_CLASS.LIQUID) !== 0).sort((a, b) => a - b);
const key = liquid[liquid.length >> 1]!, b = brickKeyCoord(key), lat = L.lattice;
const lo = b.map((v: number) => Math.max(0, 8 * v - 8)), hi = b.map((v: number, a: number) => Math.min(lat[a]!, 8 * v + 16));
const filled = sceneWithSolidStroke(scene, [{ operation: "fill", minimum: lo, maximumExclusive: hi, materialId: 1 }]);
let edits = 0;
try { s.validateLiveSolidEdit?.(filled); } catch (err) { console.log(`${sceneKey}: the fill [${lo}]..[${hi}] is refused live (${(err as Error).message}): no edit check`); }
if (s.applySceneUniforms(filled) !== false) {
  edits++; await verify("fill"); await advance(N2, "fill");
  if (s.applySceneUniforms(scene) !== false) { edits++; await verify("undo"); await advance(N2, "undo"); }
  else console.log(`${sceneKey}: the undo is not taken live`);
} else console.log(`${sceneKey}: the fill is not taken live`);
console.log(`${sceneKey}: edit [${lo}]..[${hi}] around brick ${b}: ${edits} live edits taken (fill, undo)`);
console.log(`${sceneKey}: ${checks} checks (boot, ${t} steps, restore, edits), ${churn} with a changed brick count; ${bad} with a FLAT mismatch; steps with a FAIL word set: ${failSeen}; GPU errors ${gpuErrors}`);
s.destroy();
process.exit(bad || gpuErrors ? 1 : 0);
