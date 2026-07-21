import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";

interface TransitionRecord {
  phase: "topology-transition-audit";
  step: number;
  generation: number;
  deepCell: number;
  frontierContainsDeepCell: boolean;
  frontierCount: number;
  activeTopologyTiles: number;
  retiredTopologyTiles: number;
  topologyGeneration: number;
  tileCapacity: number;
}

test("Dawn preserves a deep dam-break pressure row across bounded recurring topology publications", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for the production topology-transition gate",
  timeout: 90_000,
}, () => {
  const child = spawnSync(process.execPath, ["--import", "tsx", "tools/run-webgpu-smoke.ts"], {
    cwd: process.cwd(), encoding: "utf8", timeout: 75_000, killSignal: "SIGKILL", maxBuffer: 32 * 1024 * 1024,
    env: { ...process.env, FLUID_SCENE: "dam-break-ui", FLUID_METHOD: "octree",
      FLUID_TARGET_S: "0.008", FLUID_ORACLE_STEPS: "2", FLUID_VOXEL_CELL_SIZE: "0.02",
      FLUID_PRESSURE_CYCLES: "32", FLUID_CPU_ORACLE: "0", FLUID_FIELD_STATS: "0",
      FLUID_DISABLE_TIMESTAMPS: "1", FLUID_OCTREE_FACE_TRANSPORT: "1",
      FLUID_OCTREE_POWER_PROJECTION: "authoritative", FLUID_OCTREE_GLOBAL_FINE_FACTOR: "4",
      FLUID_POWER_GENERATION_AUDIT: "1", FLUID_POWER_GENERATION_AUDIT_LOG: "0",
      FLUID_TOPOLOGY_TRANSITION_AUDIT: "1" },
  });
  assert.equal(child.error, undefined, `transition process failed: ${child.error?.message ?? "unknown"}`);
  assert.equal(child.status, 0, `transition smoke failed:\n${child.stderr}\n${child.stdout.slice(-12_000)}`);
  const records = child.stdout.split("\n").flatMap((line) => {
    try { const value = JSON.parse(line) as TransitionRecord; return value.phase === "topology-transition-audit" ? [value] : []; }
    catch { return []; }
  });
  assert.equal(records.length, 2, "cold-to-recurring gate did not audit both publications");
  assert.equal(records[1].deepCell, records[0].deepCell, "the sampled deep-liquid row identity changed");
  for (const record of records) {
    assert.equal(record.frontierContainsDeepCell, true, `step ${record.step} lost the deep pressure row`);
    assert.ok(record.frontierCount > 0, `step ${record.step} published an empty pressure frontier`);
    assert.ok(record.activeTopologyTiles + record.retiredTopologyTiles < record.tileCapacity,
      `step ${record.step} recurring topology expanded to the full ${record.tileCapacity}-tile domain`);
    assert.ok(record.topologyGeneration > 0, `step ${record.step} topology generation is unpublished`);
  }
  assert.ok(records[1].generation > records[0].generation, "the next power generation did not publish");
  assert.ok(records[1].topologyGeneration >= records[0].topologyGeneration,
    "the recurring topology generation regressed");
});
