import assert from "node:assert/strict";
import test from "node:test";
import { terrainFieldStamp } from "../lib/core/live-terrain-overlay";
import { cloneScene, defaultScene } from "../lib/core/model";
import { sceneCellSizes_m } from "../lib/core/scene-lattice";
import { sceneWithSolidStroke, solidWorldForScene } from "../lib/core/solid-world";
import { OctreeSparseBrickWorld } from "../lib/svo/features/construction/webgpu-svo-sparse-bricks";

// Exercise the actual host preflight without allocating a GPU world. All state
// read by these methods is supplied explicitly; no GPU method may be invoked.
function fixture() {
  const scene = cloneScene(defaultScene); scene.solidVoxels = []; scene.terrain = undefined;
  const cell = sceneCellSizes_m(scene);
  const state = Object.assign(Object.create(OctreeSparseBrickWorld.prototype), {
    destroyed: false, terrainFieldStamp: terrainFieldStamp(scene),
    sampledTerrainNodes: new Map<string, { minimum: [number, number, number]; maximum: [number, number, number] }>(),
    solidWorld: solidWorldForScene(scene),
    proxyVoxelizer: { validateSolidWorld() {} },
    sceneWorldOrigin: [-scene.container.width_m / 2, 0, -scene.container.depth_m / 2],
    sceneBrickDimensions: [1, 1, 1], brickSize: 8, cellSize: cell,
    finestLevel: 2, topologyMutationCapacity: 16, remainingTopologyLeafReserve: 15,
    reservedTopologyCoordinates: new Set<string>(), pendingTopologyCoordinates: new Map(),
    planarSceneBrickNodes: new Set(["0:0,0,0"]), reservedTopologySplits: new Set<string>(),
    sceneBrickCovered: () => false,
  });
  const host = state as OctreeSparseBrickWorld;
  const edited = (x: number) => sceneWithSolidStroke(scene, [{ operation: "fill", minimum: [x, 0, 0], maximumExclusive: [x + 1, 1, 1], materialId: 2 }]);
  return { scene, state, host, edited };
}

test("sampled terrain splitting resamples every ancestor sibling and rejects excessive work before publication", () => {
  const { state, host } = fixture();
  state.sceneWorldOrigin = [0, 0, 0];
  const ancestor = { minimum: [0, 0, 0] as [number, number, number], maximum: [.1, .1, .1] as [number, number, number] };
  state.sampledTerrainNodes.set("0:0,0,0", ancestor);
  const resample = (host as unknown as { terrainResamplingBounds(bounds: typeof ancestor[]): typeof ancestor[] }).terrainResamplingBounds.bind(host);
  const edit = { minimum: [.01, .01, .01] as [number, number, number], maximum: [.02, .02, .02] as [number, number, number] };
  assert.deepEqual(resample([edit]), [ancestor], "all sibling payloads must refresh, not merely the tiny edited region");
  state.reservedTopologySplits.add("0:0,0,0");
  assert.deepEqual(resample([edit]), [ancestor], "later edits can split a remaining coarse descendant and still need ancestor coverage");
  state.sampledTerrainNodes.set("0:0,0,0", { ...ancestor, maximum: [100, 100, 100] });
  assert.throws(() => resample([edit]), /4096-brick resampling budget/);
  assert.equal(state.pendingTopologyCoordinates.size, 0);
});
