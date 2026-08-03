import assert from "node:assert/strict";
import test from "node:test";

import { minimalRoomSceneryGraph } from "../lib/empty-scene";
import { environmentIds } from "../lib/environments";
import { cloneScene, defaultScene } from "../lib/model";
import type { SceneryGraph } from "../lib/scenery-graph";
import { SCENERY_GRAPHS } from "../lib/scenery-presets";

function assertPureWhiteRoomWall(graph: SceneryGraph, label: string): void {
  const shell = graph.nodes.find(({ kind }) => kind === "room-shell");
  if (shell?.kind !== "room-shell") return;
  assert.deepEqual(shell.wall, { colorLinear: [1, 1, 1] }, `${label} wall albedo`);
}

test("every authored room uses pure-white walls", () => {
  const scene = cloneScene(defaultScene);
  for (const environmentId of environmentIds) {
    assertPureWhiteRoomWall(SCENERY_GRAPHS[environmentId](scene), environmentId);
  }
  assertPureWhiteRoomWall(minimalRoomSceneryGraph, "empty scene");
});
