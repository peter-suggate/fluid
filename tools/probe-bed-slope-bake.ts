/** Dawn proof of the bed-slope bake's kernels (lib/methods/uniform/uniform-bed-slope-bake.ts).
 *
 * The rule is settled on the host emulation (tools/bed-slope-bake-reference.ts,
 * run over every shape by tests/uniform-bed-slope-bake-rule.test.ts); this
 * holds the kernels to it. The bake alone is built: a buffer of the solid
 * scratch's shape for each lattice (header, mask bits, lane, then canary
 * words) and UniformBedSlopeBake over it. The drum's and the long slide's
 * masks are built on the host and uploaded; no solver is made.
 *
 * For every shape the lane is poisoned, the bake runs and the buffer is read
 * back whole: the mask and the canaries are untouched, the lane is the
 * emulation's word for word, and it meets the shape's own expectation. For
 * every edit run a region bake after each step leaves the lane the
 * emulation's full bake of the edited mask, and a full GPU bake's: on a small
 * ramp, on a lattice whose window is partial along both axes (a short
 * spread, two edits to one submit) and on the long slide. Last, GPU time
 * (submit to work done, the least of five) of a full bake and of a small
 * edit's region bake on the 128x32x32 ramp, the drum and the long slide.
 *
 * node --import tsx tools/probe-bed-slope-bake.ts [--only=text]
 *   --only   the shapes and edit runs whose names hold the text ("timing" for the timings alone)
 */
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { UniformBedSlopeBake, uniformBedSlopeBakeRegion, type UniformBedSlopeBakeLayout } from "../lib/methods/uniform/uniform-bed-slope-bake";
import { BED_SLOPE_RAMP, bedMask, bedSlopeEdits, bedSlopeSceneMask, bedSlopeShapes, box, edited, imageLayoutOf, imageOf, layoutOf, rampBed, reference, sameLane, type Mask, type Reference, type Size } from "./bed-slope-bake-reference";

const started = performance.now(), only = process.argv.find(a => a.startsWith("--only="))?.slice(7);
const wanted = (name: string) => !only || name.includes(only);
const shapes = bedSlopeShapes(wanted), edits = bedSlopeEdits(wanted);
const cache = new Map<Mask, Reference>(), refOf = (mask: Mask, l: UniformBedSlopeBakeLayout) => { let r = cache.get(mask); if (!r) cache.set(mask, r = reference(mask, l)); return r; };

const errors: string[] = [];
const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve("node_modules/webgpu/index.js")).href); Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter(); assert.ok(adapter);
const device = managedGPUDevice(await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 10, maxStorageTexturesPerShaderStage: 8 } }), { requireWorkerRealm: false });
device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); console.error(e.error.message); });

const CANARY = 64, DEAD = 0xdeadbeef, POISON = 0xa5a55a5a;
/** A solid scratch of one lattice and the bake over it. */
class Rig {
  readonly buffer: GPUBuffer; readonly bake: UniformBedSlopeBake; private readonly read: GPUBuffer; private readonly words: number; private image = new Uint32Array(0);
  private constructor(readonly l: UniformBedSlopeBakeLayout) {
    // The canaries are the words after the lane: the image's cut map.
    this.words = imageLayoutOf(l.size, CANARY).words;
    this.buffer = device.createBuffer({ label: `solid scratch ${l.size.join("x")}`, size: 4 * this.words, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.read = device.createBuffer({ size: 4 * this.words, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.buffer, 4 * (l.laneBaseWord + l.laneWords), new Uint32Array(CANARY).fill(DEAD));
    this.bake = new UniformBedSlopeBake(device, this.buffer, { size: l.size, laneBaseWord: l.laneBaseWord, passes: l.passes, layers: l.layers });
  }
  static async of(l: UniformBedSlopeBakeLayout): Promise<Rig> {
    const rig = new Rig(l);
    assert.throws(() => rig.bake.encode(device.createCommandEncoder()), /not initialized/);
    await rig.bake.initialize(); return rig;
  }
  /** The mask as its image, around the lane and the canaries: the directory, then the page slots. */
  upload(mask: Mask): void {
    const image = this.image = imageOf(mask, CANARY), slots = imageLayoutOf(this.l.size, CANARY).firstSlot << 10;
    device.queue.writeBuffer(this.buffer, 0, image, 0, this.l.laneBaseWord); device.queue.writeBuffer(this.buffer, 4 * slots, image, slots);
  }
  poison(): void { device.queue.writeBuffer(this.buffer, 4 * this.l.laneBaseWord, new Uint32Array(this.l.laneWords).fill(POISON)); }
  /** Run `encode`, then the lane; the mask and the canaries must be as they were. */
  async lane(name: string, mask: Mask, encode: (encoder: GPUCommandEncoder) => void): Promise<Uint32Array> {
    const encoder = device.createCommandEncoder(); encode(encoder);
    encoder.copyBufferToBuffer(this.buffer, 0, this.read, 0, 4 * this.words); device.queue.submit([encoder.finish()]); this.bake.submitted();
    await this.read.mapAsync(GPUMapMode.READ); const all = new Uint32Array(this.read.getMappedRange().slice(0)); this.read.unmap();
    const end = this.l.laneBaseWord + this.l.laneWords, image = this.image; assert.equal(mask.size.join(), this.l.size.join());
    for (let w = 0; w < this.words; w++) if ((w < this.l.laneBaseWord || w >= end + CANARY) && all[w] !== image[w]) assert.fail(`${name}: the bake changed mask word ${w}`);
    for (let w = end; w < end + CANARY; w++) if (all[w] !== DEAD) assert.fail(`${name}: the bake wrote past the lane, word ${w}`);
    return all.subarray(this.l.laneBaseWord, this.l.laneBaseWord + this.l.laneWords);
  }
  /** Submit to work done, in ms: the least and the median of `rounds`. */
  async time(encode: (encoder: GPUCommandEncoder) => void, rounds = 5): Promise<[least: number, median: number]> {
    const times: number[] = [];
    for (let i = 0; i < rounds; i++) {
      const encoder = device.createCommandEncoder(); encode(encoder); const buffer = encoder.finish();
      await device.queue.onSubmittedWorkDone(); const t0 = performance.now(); device.queue.submit([buffer]); this.bake.submitted(); await device.queue.onSubmittedWorkDone(); times.push(performance.now() - t0);
    }
    times.sort((p, q) => p - q); return [times[0]!, times[rounds >> 1]!];
  }
}
const rigs = new Map<string, Rig>();
const rigOf = async (size: Size, passes?: number) => { const key = `${size.join("x")}/${passes ?? ""}`; let rig = rigs.get(key); if (!rig) rigs.set(key, rig = await Rig.of(layoutOf(size, passes))); return rig; };

assert.throws(() => layoutOf([128, 32, 32], 7), /even count/);
for (const shape of shapes) {
  const rig = await rigOf(shape.mask.size, shape.passes), ref = refOf(shape.mask, rig.l);
  rig.upload(shape.mask); rig.poison();
  const lane = await rig.lane(shape.name, shape.mask, encoder => rig.bake.encode(encoder));
  sameLane(shape.name, lane, ref.lane, shape.mask.size);
  console.log(`${shape.name} [${shape.mask.size.join("x")}]: the lane is the emulation's; ${shape.check(lane, ref)}`);
}
for (const edit of edits) {
  const rig = await rigOf(edit.base.size, edit.passes), l = rig.l, region = new Uint32Array(8); let mask = edit.base, columns = 0;
  rig.upload(mask); rig.poison();
  sameLane(`${edit.name} (before)`, await rig.lane(edit.name, mask, encoder => rig.bake.encode(encoder)), refOf(mask, l).lane, mask.size);
  for (const step of edit.steps) {
    for (const [b, solid] of step) { mask = edited(mask, b, solid); uniformBedSlopeBakeRegion(l, b, region); columns += region[6]! * region[7]!; }
    const now = mask; rig.upload(now);
    // Every region bake of the step in one encoder: each holds a params record of its own.
    const lane = (await rig.lane(edit.name, now, encoder => { for (const [b] of step) rig.bake.encodeRegion(encoder, b); })).slice();
    sameLane(`${edit.name} (region bake against the emulation's full bake)`, lane, reference(now, l).lane, now.size);
    rig.poison();
    sameLane(`${edit.name} (region bake against a full bake)`, lane, await rig.lane(edit.name, now, encoder => rig.bake.encode(encoder)), now.size);
  }
  console.log(`${edit.name}: ${edit.steps.length} steps, each region bake (${columns} columns written in all, of ${l.columns}) left the lane a full bake does`);
}
if (wanted("timing")) {
  const bed = rampBed(20), ramp = bedMask(BED_SLOPE_RAMP, x => bed(x));
  for (const [name, mask, small] of [["128x32x32 ramp", ramp, box(44, bed(44), 12, 2, 2, 2)], ["160x96x160 drum", bedSlopeSceneMask("drum"), box(80, 40, 80, 2, 2, 2)], ["512x96x128 long slide", bedSlopeSceneMask("long"), box(60, 40, 60, 2, 2, 2)]] as const) {
    const rig = await rigOf(mask.size), l = rig.l, region = uniformBedSlopeBakeRegion(l, small, new Uint32Array(8));
    rig.upload(mask);
    const full = await rig.time(encoder => rig.bake.encode(encoder)), part = await rig.time(encoder => rig.bake.encodeRegion(encoder, small));
    console.log(`timing ${name}: full bake ${full[0].toFixed(2)} ms (median ${full[1].toFixed(2)}), ${l.launches} launches over ${l.columns} columns; region bake of a 2x2x2 edit ${part[0].toFixed(2)} ms (median ${part[1].toFixed(2)}), window ${region[2]}x${region[3]} columns, ${region[6]}x${region[7]} written; scratch ${l.scratchBytes} bytes (${(l.scratchBytes / 2 ** 20).toFixed(2)} MiB)`);
  }
}
for (const rig of rigs.values()) { rig.bake.destroy(); rig.buffer.destroy(); }
await device.queue.onSubmittedWorkDone();
assert.deepEqual(errors, []);
console.log(`bed-slope bake: ${shapes.length} shapes, ${edits.length} edit runs, ${rigs.size} lattices, ${((performance.now() - started) / 1000).toFixed(1)} s`);
process.exit(0);
