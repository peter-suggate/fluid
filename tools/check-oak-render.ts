/** Serial production renders; waits for the repository-wide browser/Dawn lease. */
import { spawn } from "node:child_process";
import assert from "node:assert/strict";
import { mkdirSync, openSync, closeSync, readFileSync } from "node:fs";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
let acquired = false;
for (let attempt = 0; attempt < 240; attempt++) {
  try {
    await acquireWebGPUExclusiveLock("oak-v2-render", "oak v2 voxel depth comparison");
    acquired = true;
    break;
  }
  catch (error) {
    if ((error as {
      cause?: {
        code?: string;
      };
    }).cause?.code !== "EEXIST")
      throw error;
    if (!attempt)
      console.log("Waiting for the active browser/Dawn GPU lease; it will not be removed or interrupted.");
    await new Promise(resolve => setTimeout(resolve, 5000));
  }
}
if (!acquired)
  throw new Error("Oak render could not acquire the GPU lease within twenty minutes");
try {
  const depths = (process.env.OAK_RENDER_DEPTHS ?? "0,1,2,3").split(",").map(Number);
  if (depths.some(depth => !Number.isInteger(depth) || depth < 0 || depth > 3)) throw new RangeError("Oak render depths must be in 0..3");
  const prefix = process.env.OAK_RENDER_OUT_PREFIX ?? "artifacts/oak-v2/production";
  for (const depth of depths) {
    const directory = `${prefix}-depth${depth}`;
    mkdirSync(directory, { recursive: true });
    const fd = openSync(`${directory}/render.log`, "w");
    console.log(`Rendering depth ${depth}`);
    try {
      const code = await new Promise<number | null>((resolve, reject) => {
        const child = spawn(process.execPath, ["--import", "tsx", "tools/benchmark-svo-dry-frame-gpu.ts"], {
          env: { ...process.env, WEBGPU_NODE_MODULE: `${process.cwd()}/node_modules/webgpu/index.js`, FLUID_WEBGPU_BACKEND: "metal",
            FLUID_SVO_DRY_FRAME_SCENE_MODULE: "tools/preview/oak-v2.ts", FLUID_SVO_DRY_FRAME_ENVIRONMENT_REFINEMENT: String(depth),
            FLUID_SVO_DRY_FRAME_WIDTH: "800", FLUID_SVO_DRY_FRAME_HEIGHT: "650", FLUID_SVO_DRY_FRAME_WARMUPS: "2",
            FLUID_SVO_DRY_FRAME_CYCLES: "5", FLUID_SVO_DRY_FRAME_CONE_SCALE: "0.5", FLUID_SVO_DRY_FRAME_TIMING: "gpu",
            FLUID_SVO_DRY_FRAME_OUT: `${directory}/frame.json` }, stdio: ["ignore", fd, fd],
        });
        child.once("error", reject);
        child.once("exit", resolve);
      });
      if (code !== 0)
        throw new Error(`Oak render depth ${depth} exited ${code}; see ${directory}/render.log`);
      const report = JSON.parse(readFileSync(`${directory}/frame.json`, "utf8"));
      assert.ok(Object.values(report.coneLighting.failureTintPixels).every(count => count === 0), "Render contains failure-tint pixels");
      assert.equal(report.scene.tetrahedralRadiance.blackPages, 0, "Render contains unlit radiance pages");
      const baseDepth = Math.ceil(Math.log2(Math.max(...Object.values(report.scene.grid) as number[]) / report.scene.brickSize));
      assert.equal(report.scene.maximumDepth, baseDepth + depth, "Render did not build the requested refinement depth");
      assert.ok(Number.isFinite(report.timing.median_ms) && report.timing.median_ms > 0);
      console.log(`Ready: ${directory}/reference.png`);
    }
    finally {
      closeSync(fd);
    }
  }
}
finally {
  await releaseWebGPUExclusiveLock();
}
