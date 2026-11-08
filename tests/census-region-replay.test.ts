import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneDocumentAtLattice } from "../lib/core/scene-definition";
import { createBulkRestTankScene, BULK_REST_TANK_ID } from "../lib/core/bulk-rest-tank-scene";
import type { SceneDescription } from "../lib/core/model";
import type { UniformMixedDynamicPolicy } from "../lib/methods/uniform/uniform-mixed-dynamic";
import { advanceUniform, createUniformSolver, withUniformDevice } from "./helpers/uniform-geometric";

// The census's deep COLD cache (regionList, replay): a replayed region must
// publish exactly what classifying it publishes. On the drop frame of the
// pure tank (a 2-cell cube released over a still pool: a few HOT and WARM
// regions, the rest deep COLD), the census runs twice from the frame's
// saved words. A: every region classified (the region stamps zeroed), which
// writes every deep region's records and stamps. B: the same census words
// again, the bound cubes and records A left; the deep regions now replay
// A's records. Every census word and every bounds word but the modes and
// lists must be A's, bit for bit (B starts from A's velocity keys, the
// ones its replay checks). C: the frame's own census again, its records and
// keys as the frame left them. The deep epoch certifies phi, not the
// extended velocity: replay takes the keys afresh and demotes a region
// whose keys moved, so C's census words are A's too.
//
// The layout builder's repeated build (UNIFORM_MIXED_BUILD_SAME_WORD)
// skips the adoption: on the still tank, after each frame whose build
// repeated, every word the adoption would copy must already be the live
// ownership's.
const gpu = { skip: !process.env.WEBGPU_NODE_MODULE };
const HEADER = 20;

type Census = {
  work: GPUBuffer; bounds: GPUBuffer; words: number; pages: number; regionOffset: number;
  ownership: { capacity: { tiles: number } };
  encode(encoder: GPUCommandEncoder, policy: UniformMixedDynamicPolicy, readback?: boolean): void;
};

async function words(device: GPUDevice, buffer: GPUBuffer, offset = 0, size = buffer.size - offset): Promise<Uint32Array> {
  const read = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  try {
    const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(buffer, offset, read, 0, size); device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ); return new Uint32Array(read.getMappedRange().slice(0));
  } finally { if (read.mapState === "mapped") read.unmap(); read.destroy(); }
}

function tank(drop: boolean, steps: number): SceneDescription {
  const cell = 0.01, base = getSceneDefinition(BULK_REST_TANK_ID);
  const scene = sceneDocumentAtLattice({ ...base, buildAt: (l: { cellSize_m: number }) => createBulkRestTankScene(l.cellSize_m, { pagesX: 8, pagesY: 8, pagesZ: 4 }) }, { cellSize_m: cell }).scene;
  if (drop) {
    const box = scene.fluid.initialLiquidVolumes![0]! as { min_m: { x: number; y: number; z: number }; max_m: { x: number; y: number; z: number } };
    const x = box.min_m.x + 16 * cell, z = box.min_m.z + 16 * cell, y = box.max_m.y + 8 * cell;
    scene.fluid.initialLiquidVolumes!.push({ shape: "box", min_m: { x, y, z }, max_m: { x: x + 2 * cell, y: y + 2 * cell, z: z + 2 * cell } });
  }
  scene.duration_s = Math.max(scene.duration_s, steps * scene.numerics.fixedDt_s);
  return scene;
}

test("a replayed deep COLD region publishes its classification bit for bit", gpu, async () => {
  await withUniformDevice("census region replay", async device => {
    const steps = 32, scene = tank(true, steps);
    const solver = await createUniformSolver(device, scene, getSceneDefinition(BULK_REST_TANK_ID).methodProfile?.overrides ?? {});
    try {
      for (let step = 1; step <= steps; step++) await advanceUniform(solver, step * scene.numerics.fixedDt_s);
      const s = solver as unknown as { mixedDynamic: Census; mixedCensusPolicy: UniformMixedDynamicPolicy };
      const c = s.mixedDynamic, policy = s.mixedCensusPolicy, tiles = c.ownership.capacity.tiles, pages = c.pages;
      const lists = c.regionOffset + 3 * pages, keys = { offset: 4 * (HEADER + c.words), size: 4 * 6 * tiles };
      const copy = (from: GPUBuffer) => device.createBuffer({ size: from.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      const saved = [copy(c.work), copy(c.bounds)] as const, a = copy(c.work);
      try {
        const frame = await words(device, c.bounds, 4 * lists, 8);
        { const e = device.createCommandEncoder(); e.copyBufferToBuffer(c.work, 0, saved[0], 0, c.work.size); e.copyBufferToBuffer(c.bounds, 0, saved[1], 0, c.bounds.size); device.queue.submit([e.finish()]); }
        // A: every region classified.
        { const e = device.createCommandEncoder();
          e.copyBufferToBuffer(saved[0], 0, c.work, 0, c.work.size); e.copyBufferToBuffer(saved[1], 0, c.bounds, 0, c.bounds.size);
          e.clearBuffer(c.bounds, 4 * c.regionOffset, 4 * 2 * pages);
          c.encode(e, policy, false); e.copyBufferToBuffer(c.work, 0, a, 0, c.work.size); device.queue.submit([e.finish()]); }
        const workA = await words(device, c.work), boundsA = await words(device, c.bounds);
        // B: the deep regions replay A's records.
        { const e = device.createCommandEncoder();
          e.copyBufferToBuffer(saved[0], 0, c.work, 0, c.work.size); e.copyBufferToBuffer(a, keys.offset, c.work, keys.offset, keys.size);
          c.encode(e, policy, false); device.queue.submit([e.finish()]); }
        const workB = await words(device, c.work), boundsB = await words(device, c.bounds);
        // C: the frame's own census again, its records and keys as the frame left them.
        { const e = device.createCommandEncoder();
          e.copyBufferToBuffer(saved[0], 0, c.work, 0, c.work.size); e.copyBufferToBuffer(saved[1], 0, c.bounds, 0, c.bounds.size);
          c.encode(e, policy, false); device.queue.submit([e.finish()]); }
        const workC = await words(device, c.work), boundsC = await words(device, c.bounds);
        const replayedC = boundsC.subarray(lists + 4 + pages, lists + 4 + pages + boundsC[lists + 1]!);
        console.log(`C: ${replayedC.filter(region => boundsC[c.regionOffset + 2 * pages + region] === 2).length} of ${replayedC.length} replayed regions demoted (keys moved)`);
        const resident = boundsA[lists]! + boundsA[lists + 1]!;
        console.log(`frame census: ${frame[0]} regions classified, ${frame[1]} replayed; A ${boundsA[lists]}/${boundsA[lists + 1]}, B ${boundsB[lists]}/${boundsB[lists + 1]}`);
        assert.equal(boundsA[lists + 1], 0, "A replays no region");
        assert.ok(boundsB[lists + 1]! > 0 && boundsB[lists]! + boundsB[lists + 1]! === resident, "B replays the deep regions A stamped");
        const sections = (i: number) => i < HEADER ? "header" : i < HEADER + c.words ? "band" : i < HEADER + c.words + 6 * tiles ? "keys" : i < HEADER + c.words + 7 * tiles ? "gap" : i < HEADER + c.words + 8 * tiles ? "travel" : "later";
        const differ = (x: Uint32Array, y: Uint32Array, skip?: (i: number) => boolean) => {
          const by = new Map<string, number>(); let first = -1;
          for (let i = 0; i < x.length; i++) if (x[i] !== y[i] && !skip?.(i)) { if (first < 0) first = i; by.set(sections(i), (by.get(sections(i)) ?? 0) + 1); }
          return first < 0 ? "" : `first at ${first} (${x[first]} vs ${y[first]}): ${[...by].map(([k, n]) => `${k} ${n}`).join(", ")}`;
        };
        assert.equal(differ(workA, workB), "", "census words");
        assert.equal(differ(workA, workC), "", "the frame's own census words");
        assert.equal(differ(boundsA, boundsB, i => i >= c.regionOffset + 2 * pages), "", "bounds words outside the modes and lists");
      } finally { for (const b of [...saved, a]) b.destroy(); }
    } finally { solver.destroy(); }
  });
});

type Builder = { level: { work: GPUBuffer; status: GPUBuffer; topology: GPUBuffer; support: GPUBuffer; slots: GPUBuffer } };
type Ownership = { counts: GPUBuffer; adoptionTargets: { topology: GPUBuffer; support: GPUBuffer; hanging?: GPUBuffer } };

test("a repeated build leaves the live ownership the generation it would adopt", gpu, async () => {
  await withUniformDevice("census repeated build", async device => {
    const steps = 24, scene = tank(false, steps);
    const solver = await createUniformSolver(device, scene, getSceneDefinition(BULK_REST_TANK_ID).methodProfile?.overrides ?? {});
    try {
      const s = solver as unknown as { mixedBuilder: Builder; mixedFrame: { ownership: Ownership }; mixedDynamic: Census };
      const b = s.mixedBuilder.level, live = s.mixedFrame.ownership, N = s.mixedDynamic.ownership.capacity.tiles;
      const { UNIFORM_MIXED_BUILD_SAME_WORD: SAME, UNIFORM_MIXED_RELAYOUT_RECEIPT: R } = await import("../lib/methods/uniform/uniform-mixed-layout-builder");
      // A frame builds only while the layout is due: the receipt counts the builds.
      let repeated = 0, built = 0, builds = 0;
      for (let step = 1; step <= steps; step++) {
        await advanceUniform(solver, step * scene.numerics.fixedDt_s);
        const count = (await words(device, b.status, 4 * R.builds, 4))[0]!;
        if (count === builds) continue;
        builds = count; built++;
        if ((await words(device, b.work, 4 * SAME, 4))[0] === 0) continue;
        repeated++;
        if (repeated > 3) continue;
        // The adoption's words (uniformMixedAdoptWGSL), source against live.
        const counts = await words(device, b.work, 4 * 64, 16), liveCounts = await words(device, live.counts, 0, 16);
        assert.deepEqual([...liveCounts], [...counts], `counts, step ${step}`);
        const [srcTop, top, srcSup, sup] = [await words(device, b.topology), await words(device, live.adoptionTargets.topology), await words(device, b.support), await words(device, live.adoptionTargets.support)];
        const hanging = live.adoptionTargets.hanging, srcSlots = hanging ? await words(device, b.slots) : undefined, slots = hanging ? await words(device, hanging) : undefined;
        const bad: string[] = [];
        const key = (t: number) => {
          for (const i of [t, 2 * N + 2 * t, 2 * N + 2 * t + 1, 4 * N + t]) if (top[i] !== srcTop[i]) bad.push(`tile word ${i} of key ${t}`);
          if (slots && slots[t] !== srcSlots![t]) bad.push(`slot of key ${t}`);
        };
        const listed = counts[0]! + counts[1]!;
        for (let j = 0; j < listed; j++) { const t = srcTop[N + j]!; if (top[N + j] !== t) bad.push(`list ${j}`); key(t); }
        for (let k = 0; k < 1024; k++) key(k < 512 ? k : N - 1024 + k);
        const FINE = 6 * N + 16, SEAMS = 7 * N + 16, REGULAR = 8 * N + 20, seams = srcSup[SEAMS]! + srcSup[SEAMS + 1]!;
        for (const [from, n] of [[FINE, counts[0]!], [SEAMS, 4 + seams], [REGULAR, 4 + srcSup[REGULAR]!]] as const)
          for (let w = from; w < from + n; w++) if (sup[w] !== srcSup[w]) bad.push(`support ${w}`);
        if (slots) for (let w = N; w < N + seams && w < slots.length; w++) if (slots[w] !== srcSlots![w]) bad.push(`slot table ${w}`);
        assert.equal(bad.slice(0, 8).join("; "), "", `${bad.length} adopted words differ, step ${step}`);
      }
      console.log(`repeated builds: ${repeated} of ${built} built in ${steps} steps`);
      assert.ok(repeated > 0, "the still tank repeats a build");
    } finally { solver.destroy(); }
  });
});
