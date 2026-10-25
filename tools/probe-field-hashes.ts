/** Field hashes of an NB-FLIP scene at chosen steps: the gate of a storage
 * change that must leave the physics bit-identical. Each field is read
 * logically (readMixedTexture through the live address), so a change of
 * where texels are stored hashes the same while the values do not change.
 *   node --import tsx tools/probe-field-hashes.ts SCENE [--steps=N] [--at=10,20] [--set=key=value] [--blocks=N] [--audit]
 * Prints a JSON line per checkpoint: sha256 (first 16 hex) of velocity,
 * pre-projection velocity, vertex phi, volume and centre phi, and the
 * particle count. --blocks sets the page pool's block count; --audit prints
 * the requested GPU bytes by resource at build (two runs a few blocks apart
 * give the bytes a block). --ring adds, at each checkpoint, the detail ring's
 * tiles and the blocks holding any of them (what block-granular h storage
 * keeps resident). --mask hashes each field with its non-canonical texels
 * outside the detail ring zeroed (probe-ring-mask.ts): the gate of a storage
 * that keeps h texels only near the ring. --fine-blocks=off|auto|N sets
 * fluid.tiers.fineBlocks (off: unset). --gate="RUNNER MODE" builds the solver (mostly CPU:
 * pipeline compiles) first and holds the runner's GPU slot only over the
 * frames and readbacks (as benchmark-region-tier-paired.ts). */
import assert from "node:assert/strict";
import { spawn, type ChildProcess } from "node:child_process";
import { createHash } from "node:crypto";
import { resolveMethodValues } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { uniformNarrowBandMethod } from "../lib/methods/uniform/uniform-narrow-band-method";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { readUniformLiveAddress, withUniformDevice } from "../tests/helpers/uniform-geometric";
import { readMixedTexture } from "../tests/helpers/uniform-mixed-native-fields";
import { auditUniformGPUAllocations } from "./uniform-gpu-allocation-audit";
import { maskOutsideRing, readRingBits } from "./probe-ring-mask";

const option = (key: string) => process.argv.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
const id = process.argv.slice(2).find(a => !a.startsWith("--")); assert.ok(id, "a scene id");
const steps = Number(option("steps") ?? 24);
const at = new Set((option("at") ?? String(steps)).split(",").map(Number));
const sets = Object.fromEntries(process.argv.filter(a => a.startsWith("--set=")).map(a => {
  const [key, ...value] = a.slice(6).split("="); const text = value.join("=");
  return [key!, text !== "" && Number.isFinite(Number(text)) ? Number(text) : text === "true" ? true : text === "false" ? false : text];
}));
const definition = getSceneDefinition(id), scene = structuredClone(sceneDocument(definition));
const profile = definition.methodProfile;
if (option("blocks") !== undefined) scene.fluid = { ...scene.fluid!, pagePool: { blocks: Number(option("blocks")) } };
if (option("fine-blocks") !== undefined) scene.fluid = { ...scene.fluid!, tiers: { ...scene.fluid?.tiers, ...(option("fine-blocks") === "off" ? {} : { fineBlocks: option("fine-blocks") === "auto" ? "auto" as const : Number(option("fine-blocks")) }) } };
assert.equal(profile?.methodId, uniformNarrowBandMethod.id);
const values = resolveMethodValues(uniformNarrowBandMethod, profile!.quality, { ...uniformNarrowBandMethod.appDefaults, ...profile!.overrides, ...sets });
const dt = scene.numerics.fixedDt_s || scene.numerics.maxDt_s;
const hash = (a: Float32Array) => createHash("sha256").update(new Uint8Array(a.buffer, a.byteOffset, a.byteLength)).digest("hex").slice(0, 16);

/** The ring bit (second stencil word, bit 0) of every tile key: ring tiles and the blocks holding one. */
async function ringCensus(device: GPUDevice, solver: WebGPUUniformReferenceSolver): Promise<{ ringTiles: number; ringBlocks: number; blocks: number }> {
  const ownership = (solver as unknown as { mixedFrame: { ownership: { presentation: { buffer: GPUBuffer }; capacity: { tiles: number } } } }).mixedFrame.ownership;
  const n = ownership.capacity.tiles, staging = device.createBuffer({ size: 8 * n, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(ownership.presentation.buffer, 8 * n, staging, 0, 8 * n); device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ); const words = new Uint32Array(staging.getMappedRange().slice(0)); staging.destroy();
  const held = new Set<number>(); let ringTiles = 0;
  for (let t = 0; t < n; t++) if (words[2 * t + 1]! & 1) { ringTiles++; held.add(t >>> 9); }
  return { ringTiles, ringBlocks: held.size, blocks: n / 512 };
}

/** Hold the gate's slot until the returned process's stdin closes. */
async function acquireGate(gate: string): Promise<ChildProcess> {
  const [command, ...args] = gate.split(" ").filter(Boolean);
  const child = spawn(command!, [...args, "cat"], { stdio: ["pipe", "pipe", "inherit"] });
  await new Promise<void>((done, fail) => { child.stdout!.once("data", () => done()); child.once("exit", code => fail(new Error(`gate exited ${code}`))); child.stdin!.write("held\n"); });
  return child;
}
let lock: ChildProcess | undefined;

await withUniformDevice(`hashes ${id}`, async gpu => {
  const audit = process.argv.includes("--audit") ? auditUniformGPUAllocations(gpu) : undefined, device = audit?.device ?? gpu;
  const solver = await uniformNarrowBandMethod.createSolverAsync!(device, scene, profile!.quality, values, undefined, () => {}) as WebGPUUniformReferenceSolver;
  try {
    console.log(JSON.stringify({ scene: id, dt, allocatedBytes: solver.info.allocatedBytes, particleCapacity: solver.narrowBandFlipInfo?.capacity }));
    if (audit) { const s = audit.snapshot(); console.log(JSON.stringify({ liveBytes: s.liveBytes, resources: s.resources.slice(0, 40) })); }
    const gate = option("gate");
    if (gate) { await device.queue.onSubmittedWorkDone(); const t = performance.now(); lock = await acquireGate(gate); console.log(JSON.stringify({ phase: "locked", wait_ms: performance.now() - t })); }
    for (let step = 1; step <= steps; step++) {
      solver.advanceTo(step * dt, []); await solver.awaitFrameCompletion();
      assert.equal(solver.info.simulationPipelineError, undefined);
      if (!at.has(step)) continue;
      const address = await readUniformLiveAddress(device, solver);
      const ringBits = process.argv.includes("--mask") ? await readRingBits(device, solver) : undefined;
      const read = async (t: GPUTexture) => { const v = await readMixedTexture(device, t, address); return ringBits ? maskOutsideRing(v, address, ringBits, address.cells) : v; };
      const [velocity, scratch, phi, volume, centre] = await Promise.all([read(solver.velocityTexture), read(solver.preProjectionVelocityTexture),
        read(solver.vertexPhiTexture!), read(solver.volumeTexture), read(solver.surfaceFieldTexture)]);
      const flip = solver.narrowBandFlipInfo!;
      const ring = process.argv.includes("--ring") ? await ringCensus(device, solver) : undefined;
      console.log(JSON.stringify({ step, velocity: hash(velocity), scratch: hash(scratch), phi: hash(phi), volume: hash(volume), centre: hash(centre),
        particles: flip.particles, fineTiles: solver.info.uniformMixedFineTiles, allocatedBytes: solver.info.allocatedBytes, ...ring }));
    }
  } finally { solver.destroy(); lock?.stdin?.end(); }
}, ["timestamp-query", "subgroups"]);
