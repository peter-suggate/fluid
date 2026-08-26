import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene, validateScene } from "../lib/core/model";
import { sceneLatticeDimensions, solidVoxelShellForScene } from "../lib/core/scene-lattice";
import {
  buildVesselOutlineGeometry,
  sceneVesselPresentation,
} from "../lib/core/vessel-outline";
import { optionalRendererPipelineRequests } from "../lib/core/webgpu-renderer";

test("missing vessel presentation defaults to the canonical shell-volume wireframe", () => {
  const scene = cloneScene(defaultScene);
  scene.environment = "stage";
  scene.container.vessel = undefined;
  const outline = buildVesselOutlineGeometry(scene);

  assert.equal(sceneVesselPresentation(scene), "outline");
  assert.equal(outline?.geometry.segmentCount, 60,
    "six slab boxes have 72 edges, with twelve shared edges emitted once");
  assert.match(outline!.key, /^vessel-voxel-volume:/);
  const points: number[][] = [];
  for (let segment = 0; segment < outline!.geometry.segmentCount; segment += 1) {
    const base = segment * 16;
    points.push([...outline!.geometry.segments.slice(base, base + 3)]);
    points.push([...outline!.geometry.segments.slice(base + 4, base + 7)]);
  }
  const extrema = (axis: number) => [
    Math.min(...points.map((point) => point[axis]!)),
    Math.max(...points.map((point) => point[axis]!)),
  ];
  const assertExtrema = (axis: number, expected: readonly [number, number],
    message: string) => extrema(axis).forEach((value, side) =>
      assert.ok(Math.abs(value - expected[side]!) < 1e-6, message));
  assertExtrema(0, [-0.65, 0.65],
    "side rails include the actual one-cell wall thickness");
  assertExtrema(1, [-0.05, 0.85],
    "floor and closed lid rails include their actual one-cell thickness");
  assertExtrema(2, [-0.45, 0.45],
    "front and back rails include the actual one-cell wall thickness");
  assert.ok(optionalRendererPipelineRequests(undefined, false, false,
    false, false, false, true, true).includes("decoration-overlay"));
});

test("an edited shell face is not falsely represented by the volume wireframe", () => {
  const scene = cloneScene(defaultScene);
  scene.environment = "stage";
  scene.container.vessel = "outline";
  scene.solidVoxels.push({ operation: "clear", minimum: [4, -1, 4],
    maximumExclusive: [5, 0, 5] });

  const outline = buildVesselOutlineGeometry(scene);
  assert.ok(outline);
  assert.doesNotMatch(outline.key, /yLow:/,
    "the cut floor remains residual voxel geometry instead of receiving a false full-slab cue");
  assert.equal(outline.geometry.segmentCount, 52);
});

test("glass and hidden vessel modes publish no outline geometry", () => {
  for (const vessel of ["glass", "none"] as const) {
    const scene = cloneScene(defaultScene);
    scene.environment = "stage";
    scene.container.vessel = vessel;
    assert.equal(sceneVesselPresentation(scene), vessel);
    assert.equal(buildVesselOutlineGeometry(scene), undefined);
    assert.deepEqual(validateScene(scene), []);
  }
});

test("spherical outline is three staircase sections of the canonical voxel cavity", () => {
  const scene = cloneScene(defaultScene);
  scene.environment = "stage";
  scene.container.shape = "sphere";
  scene.container.top = "closed";
  scene.container.vessel = "outline";
  scene.solidVoxels = [...solidVoxelShellForScene(scene)];
  const outline = buildVesselOutlineGeometry(scene);

  assert.ok(outline);
  assert.equal(outline.geometry.segmentCount, 192);
  const dimensions = sceneLatticeDimensions(scene);
  const cell = [scene.container.width_m / dimensions[0],
    scene.container.height_m / dimensions[1],
    scene.container.depth_m / dimensions[2]];
  const origin = [-0.5 * scene.container.width_m, 0,
    -0.5 * scene.container.depth_m];
  for (let segment = 0; segment < outline.geometry.segmentCount; segment += 1) {
    for (const endpointOffset of [0, 4]) for (let axis = 0; axis < 3; axis += 1) {
      const world = outline.geometry.segments[segment * 16 + endpointOffset + axis]!;
      const lattice = (world - origin[axis]!) / cell[axis]!;
      assert.ok(Math.abs(lattice - Math.round(lattice)) < 1e-5,
        "every sphere cue vertex lies on an actual voxel face");
    }
  }
});

test("garden terrain suppresses every rectangular vessel presentation", () => {
  const scene = cloneScene(defaultScene);
  scene.environment = "garden";
  scene.container.vessel = "outline";
  assert.equal(sceneVesselPresentation(scene), "none");
  assert.equal(buildVesselOutlineGeometry(scene), undefined);
});
