import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import type { CameraState, SceneDescription } from "../lib/core/model";
import type { SparseVoxelDrySceneRenderer } from "../lib/svo/pipeline/webgpu-svo-dry-scene";

/** Opt-in comparison over one world build. Never changes shipping tuning. */
export async function compareSvoSunlight(options: {
  directory: string;
  renderer: SparseVoxelDrySceneRenderer;
  camera: CameraState;
  scene: SceneDescription;
  configure: (scale: 1 | 0.5) => void;
  view: (camera: CameraState) => void;
  publish: (scene: SceneDescription, geometry: boolean) => Promise<void>;
  frame: () => Promise<number>;
  capture: (name: string) => Promise<Uint32Array>;
  save: (name: string, rows: Uint32Array) => void;
  log: (message: string) => void;
  fluid?: (filled: boolean | undefined) => Promise<void>;
  cacheCounters?: () => Promise<number[]>;
}): Promise<void> {
  const { renderer, directory, camera, scene, log } = options;
  mkdirSync(directory, { recursive: true });
  const arms = [
    { name: "full", scale: 1, cache: false },
    { name: "half", scale: 0.5, cache: false },
    { name: "cached", scale: 1, cache: true },
    { name: "half-cached", scale: 0.5, cache: true },
  ] as const;
  const views = [
    { name: "hero", camera },
    { name: "low-orbit", camera: { ...camera, elevation_rad: 0.15 } },
    { name: "wide", camera: { ...camera, elevation_rad: 0.3, distance_m: 4.5 } },
  ];
  const results: unknown[] = [];
  const median = (values: number[]) => {
    const sorted = [...values].sort((a, b) => a - b);
    return (sorted[Math.floor((sorted.length - 1) / 2)] + sorted[Math.floor(sorted.length / 2)]) / 2;
  };
  async function configure(arm: typeof arms[number]) {
    renderer.setVoxelLightCacheEnabled(arm.cache);
    options.configure(arm.scale);
    if (arm.scale !== 1) await renderer.ensureConeLightingPrepass();
    const deadline = performance.now() + 120_000;
    while (renderer.presentationBundleStatus.state === "compiling") {
      assert.ok(performance.now() < deadline, "comparison pipeline compilation timed out");
      await new Promise((resolve) => setTimeout(resolve, 25));
    }
    assert.equal(renderer.presentationBundleStatus.state, "ready");
    if (arm.cache) assert.ok(renderer.voxelLightCacheAllocatedBytes > 0, "cache arm unavailable on this device");
  }
  async function warm(count = 32) {
    for (let i = 0; i < count; i += 1) await options.frame();
  }
  async function capture(name: string) {
    const rows = await options.capture(name);
    options.save(path.join(directory, `${name}.png`), rows);
    writeFileSync(path.join(directory, `${name}.rgba16f`), Buffer.from(rows.buffer, rows.byteOffset, rows.byteLength));
    return rows;
  }
  // Reverse arm order on repetition two to expose drift; cache warmup is
  // explicit and excluded, and every arm retains identical geometry and GI.
  for (let repeat = 0; repeat < 2; repeat += 1) {
    for (const view of views) {
      options.view(view.camera);
      for (const arm of repeat === 0 ? arms : [...arms].reverse()) {
        await configure(arm);
        await warm();
        const samples: number[] = [];
        for (let i = 0; i < 8; i += 1) samples.push(await options.frame());
        if (repeat === 0) await capture(`${view.name}-${arm.name}`);
        const row = { test: "settled", view: view.name, arm: arm.name, repeat,
          median_ms: median(samples), samples_ms: samples, cacheBytes: renderer.voxelLightCacheAllocatedBytes };
        results.push(row);
        log(`Sunlight ${JSON.stringify(row)}`);
      }
    }
  }
  for (const arm of arms) {
    options.view(camera);
    await configure(arm);
    await warm();
    const samples: number[] = [];
    for (let i = 0; i < 12; i += 1) {
      options.view({ ...camera, azimuth_rad: camera.azimuth_rad + (i + 1) * 0.015 });
      samples.push(await options.frame());
      if ([0, 5, 11].includes(i)) await capture(`motion-${i}-${arm.name}`);
    }
    results.push({ test: "motion", arm: arm.name, median_ms: median(samples), samples_ms: samples });
  }
  options.view(camera);
  // A sun edit must invalidate cached visibility for distant receivers too.
  const movedSun: SceneDescription = { ...scene, lighting: { ...scene.lighting,
    directional: { ...scene.lighting?.directional, direction: [-0.7, 0.6, 0.35] } } };
  for (const arm of arms) {
    await options.publish(scene, false);
    await configure(arm);
    await warm();
    await options.publish(movedSun, false);
    const first_ms = await options.frame();
    await capture(`sun-edit-first-${arm.name}`);
    await warm();
    await capture(`sun-edit-settled-${arm.name}`);
    results.push({ test: "sun-edit", arm: arm.name, first_ms });
  }
  await options.publish(scene, false);
  if (options.fluid) {
    // Keep the cache alive through changing coverage. An empty-water frame
    // must return to the same image after an intervening filled-water frame.
    for (const arm of [arms[0], arms[2]]) {
      await configure(arm);
      await options.fluid(false);
      await warm();
      const empty = await capture(`fluid-empty-${arm.name}`);
      const counters = await options.cacheCounters?.();
      if (arm.cache && counters) assert.ok(counters[2] > 0, "fluid-attached cache has no hits");
      await options.fluid(true);
      await options.frame();
      const filled = await capture(`fluid-filled-${arm.name}`);
      assert.notDeepEqual(filled, empty, "fluid coverage did not affect lighting");
      await options.fluid(false);
      await warm();
      const restored = await capture(`fluid-restored-${arm.name}`);
      assert.deepEqual(restored, empty, "moving fluid left stale cached lighting");
      results.push({ test: "fluid-reuse", arm: arm.name, counters, restoredExact: true });
    }
    await options.fluid(undefined);
  }
  // Remove and restore the tree through the live scene publication path.
  // This exercises visibility invalidation well beyond the edited leaves.
  await options.publish(scene, false);
  if (scene.scenery?.nodes.some((node) => node.id === "tree")) {
    const removed: SceneDescription = { ...scene, scenery: { ...scene.scenery,
      nodes: scene.scenery.nodes.filter((node) => node.id !== "tree") } };
    // Warm the cached arm BEFORE the edit, then compare all arms over the
    // SAME resulting publication. Repeated removal/restoration can repack the
    // live voxel arena and is not an identical-geometry lighting comparison.
    await configure(arms[2]);
    await warm();
    await options.publish(removed, true);
    for (const arm of [arms[2], arms[0], arms[1], arms[3]]) {
      await configure(arm);
      await options.frame();
      await capture(`tree-removed-first-${arm.name}`);
      await warm();
      await capture(`tree-removed-settled-${arm.name}`);
    }
    await options.publish(scene, true);
  }
  await options.publish(scene, false);
  options.view(camera);
  renderer.setVoxelLightCacheEnabled(false);
  options.configure(1);
  writeFileSync(path.join(directory, "results.json"), JSON.stringify({
    metric: "serialized submit-to-fence milliseconds; excludes CPU encode, compilation and warmup",
    warmupFrames: 32, timedFrames: 8, timedFramesHaveFluidCoverage: false,
    fluidCoverageFixture: Boolean(options.fluid), results,
  }, null, 2));
}
