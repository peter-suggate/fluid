import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const packageJson = JSON.parse(readFileSync(
  new URL("../package.json", import.meta.url), "utf8")) as { scripts: Record<string, string> };

const laneNames = [
  "test:webgpu:octree-cutover-moving-solid-open-top",
  "test:webgpu:octree-cutover-fine-factor4",
  "test:webgpu:octree-cutover-fine-factor8",
] as const;

test("cutover evidence lanes are isolated and the aggregate runs them serially", () => {
  for (const name of laneNames.slice(1)) {
    const command = packageJson.scripts[name];
    assert.ok(command, `${name} must be an explicit package lane`);
    assert.match(command, /node --import tsx tools\/run-webgpu-smoke-isolated\.ts$/,
      `${name} must acquire the process-wide lock before Dawn is imported`);
    assert.doesNotMatch(command, /tools\/run-webgpu-smoke\.ts(?:\s|$)/,
      `${name} must not bypass the isolated lock-owning runner`);
    assert.match(command, /FLUID_METHOD=octree/);
    assert.match(command, /FLUID_EXPECT_EXACT_STEPS=1/);
    assert.match(command, /FLUID_POWER_GENERATION_AUDIT=1/);
  }
  assert.match(packageJson.scripts[laneNames[0]]!,
    /tools\/run-webgpu-exclusive\.ts --import tsx tools\/run-octree-lagged-rigid-smoke\.ts$/,
    "the moving-solid lane must hold the exclusive lock for its complete Dawn lifetime");
  const aggregate = packageJson.scripts["acceptance:octree-cutover-lifecycle"];
  assert.equal(aggregate, laneNames.map((name) => `npm run ${name}`).join(" && "),
    "the aggregate must serialize its three lock-owning child lanes");
});

test("the moving-solid/open-top lane combines both lifecycle conditions", () => {
  const command = packageJson.scripts[laneNames[0]]!;
  assert.match(command, /FLUID_CONTAINER_TOP=open/);
  assert.match(command, /run-octree-lagged-rigid-smoke\.ts/);
  const runner = readFileSync(new URL("../tools/run-octree-lagged-rigid-smoke.ts", import.meta.url), "utf8");
  assert.match(runner, /createPaperScenario\("dam-break-boxes"\)/,
    "the lane must retain the authored dynamic rigid stack");
  assert.match(runner, /scene\.container\.top = containerTop/,
    "the open-top override must reach the solver scene");
  assert.match(runner, /Math\.max\(\.\.\.bodyMotion_m\) > 1e-4/,
    "the lane must reject a nominal moving-solid case with no actual body motion");
});

test("factor-4 and factor-8 lanes differ only at the requested fine cutover", () => {
  const factor4 = packageJson.scripts[laneNames[1]]!;
  const factor8 = packageJson.scripts[laneNames[2]]!;
  for (const command of [factor4, factor8]) {
    assert.match(command, /FLUID_SCENE=minimal-power-dam-break/);
    assert.match(command, /FLUID_TARGET_S=0\.004/);
    assert.match(command, /FLUID_MAX_DT=0\.004/);
    assert.match(command, /FLUID_ORACLE_STEPS=1/);
    assert.match(command, /FLUID_EXPECT_GRID=16,16,16/);
    assert.match(command, /FLUID_GLOBAL_FINE_GENERATION_TRANSITION=1/);
  }
  assert.match(factor4, /FLUID_OCTREE_GLOBAL_FINE_FACTOR=4/);
  assert.match(factor8, /FLUID_OCTREE_GLOBAL_FINE_FACTOR=8/);
  assert.equal(factor4.replace("FACTOR=4", "FACTOR=8"), factor8,
    "fine-factor evidence must not drift into two incomparable smoke contracts");
});
