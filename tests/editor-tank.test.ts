import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { validateScene, type SceneDescription } from "../lib/model";
import { getScenePreset } from "../lib/scenes";
import { sceneLatticeDimensions } from "../lib/scene-lattice";
import { fluidBodyBox } from "../lib/editor-fluid-body";
import { boxHandles } from "../lib/editor-entity";
import { sides } from "./helpers/editor-entities";
import {
  dragTankExtents,
  tankBox,
  tankBoxForExtents,
  tankHandleIsGrabbable,
  tankLatticeForExtents,
  tankResizeIsStructural,
  tankResizePatch,
  TANK_MINIMUM_CELLS,
} from "../lib/editor-tank";
import { EDITOR_TOOLS, getEditorTool } from "../lib/editor-tools";

function preset(id: string): SceneDescription {
  return getScenePreset(id).create();
}

test("there is no separate mode for editing the tank", () => {
  // BOUNDS was a mode that showed the tank and the water permanently, because
  // neither could be clicked. Both are ordinary selectable entities now, so the
  // mode has nothing left to do and SELECT is the only editing tool.
  assert.equal(EDITOR_TOOLS.some((tool) => tool.id === "select" && tool.status === "active"), true);
  assert.equal(EDITOR_TOOLS.some((tool) => (tool.id as string) === "bounds"), false);
  // Its key went to BALL, which is a placement and not a second editing mode:
  // what must never come back is a tool that only exists to make the tank and
  // the water visible, whatever letter it is spelled with.
  assert.equal(EDITOR_TOOLS.find((tool) => tool.shortcut === "b")?.id, "fluid-ball");
  assert.match(getEditorTool("select").hint, /click anything to select/);
});

test("the tank box is the container interior, resting on the floor", () => {
  const scene = preset("water-box-dam-break");
  const box = tankBox(scene);
  assert.equal(box.min.y, 0);
  assert.equal(box.max.y, scene.container.height_m);
  assert.equal(box.max.x - box.min.x, scene.container.width_m);
  assert.equal(box.min.x, -box.max.x, "the container is centred on x");
  assert.equal(box.min.z, -box.max.z, "the container is centred on z");
});

test("the floor has no handle, because nothing in the schema can move it", () => {
  const box = tankBox(preset("water-box-dam-break"));
  const grabbable = boxHandles(box, { grabbable: tankHandleIsGrabbable, drag: () => ({}) });
  assert.equal(grabbable.length, 26 - 9, "the nine handles on the -y face are not offered");
  assert.ok(grabbable.every((handle) => handle.position_m.y > box.min.y));
});

test("a horizontal drag resizes the tank about its centre, moving both walls", () => {
  const scene = preset("water-box-dam-break");
  const box = tankBox(scene);
  const cell = scene.voxelDomain.finestCellSize_m;
  const extents = dragTankExtents(scene, sides("+00"), { x: box.max.x + 4 * cell, y: 0, z: 0 });
  assert.ok(extents.width_m > scene.container.width_m);
  assert.equal(extents.height_m, scene.container.height_m, "an x handle must not change the height");
  assert.equal(extents.depth_m, scene.container.depth_m);
  const dragged = tankBoxForExtents(extents);
  assert.equal(dragged.min.x, -dragged.max.x, "both walls move, keeping the centre");
});

test("the ceiling handle sets the height from the floor", () => {
  const scene = preset("water-box-dam-break");
  const extents = dragTankExtents(scene, sides("0+0"), { x: 0, y: scene.container.height_m * 1.5, z: 0 });
  assert.ok(extents.height_m > scene.container.height_m);
  assert.equal(extents.width_m, scene.container.width_m);
  assert.equal(tankBoxForExtents(extents).min.y, 0);
});

test("tank extents snap to whole cells, so a drag lands on an exact lattice", () => {
  const scene = preset("water-box-dam-break");
  const cell = scene.voxelDomain.finestCellSize_m;
  for (const target of [0.37, 1.02, 2.61]) {
    const extents = dragTankExtents(scene, sides("+00"), { x: target, y: 0, z: 0 });
    const cells = extents.width_m / cell;
    assert.ok(Math.abs(cells - Math.round(cells)) < 1e-9, `${extents.width_m} m is not a whole number of cells`);
    assert.deepEqual(tankLatticeForExtents(scene, extents)[0], Math.round(cells));
  }
});

test("a tank cannot be dragged below the lattice floor", () => {
  const scene = preset("water-box-dam-break");
  const cell = scene.voxelDomain.finestCellSize_m;
  const extents = dragTankExtents(scene, sides("+00"), { x: 0, y: 0, z: 0 });
  assert.ok(Math.abs(extents.width_m - TANK_MINIMUM_CELLS * cell) < 1e-9);
});

test("resizing the tank repairs what the moved walls invalidated", () => {
  const scene = preset("water-box-dam-break");
  const shrunk = dragTankExtents(scene, sides("+++"),
    { x: 0.2, y: scene.container.height_m * 0.3, z: 0.2 });
  const next = { ...scene, ...tankResizePatch(scene, shrunk) };
  assert.deepEqual(validateScene(next), [], "a resized tank must still be a valid scene");
  const body = fluidBodyBox(next);
  assert.ok(body);
  assert.ok(body.max.x <= next.container.width_m / 2 + 1e-9, "the water must have been brought inside");
  assert.ok(body.max.y <= next.container.height_m + 1e-9);
});

test("a resize that leaves the lattice alone is not announced as structural", () => {
  const scene = preset("water-box-dam-break");
  const before = sceneLatticeDimensions(scene);
  const unchanged = {
    width_m: scene.container.width_m,
    height_m: scene.container.height_m,
    depth_m: scene.container.depth_m,
  };
  assert.equal(tankResizeIsStructural(scene, unchanged), false);
  const grown = { ...unchanged, width_m: unchanged.width_m + 4 * scene.voxelDomain.finestCellSize_m };
  assert.equal(tankResizeIsStructural(scene, grown), true);
  assert.deepEqual(tankLatticeForExtents(scene, grown)[0], before[0] + 4);
});

test("every direct-manipulation drag previews without touching the document", () => {
  // The cost of these gestures is entirely in when the document is written: the
  // water body, the tank, the fill level, the hose, and the terrain all live in
  // the solver's rebuild key, so a per-move write asks the renderer to re-seed
  // at pointer rate. Each drag writes the draft store and spends one document
  // write on release. Every handle on every entity now runs through the one
  // `entity-handle` branch, which is why the list is as short as it is.
  const viewport = readFileSync(new URL("../components/WebGPUViewport.tsx", import.meta.url), "utf8");
  const branches = ["entity-handle", "fill-level", "terrain-handle"];
  const move = viewport.slice(viewport.indexOf("const pointerMove ="), viewport.indexOf("const pointerUp ="));
  for (const branch of branches) {
    const start = move.indexOf(`if (active.action === "${branch}") {`);
    assert.ok(start >= 0, `${branch} must have a pointer-move branch`);
    let body = move.slice(start, move.indexOf("\n    }", start));
    // A branch may delegate: the handle drag is resolved by a helper the axis
    // lock also calls, so it can re-resolve without a pointer move. Follow one
    // hop and hold the helper to the same contract.
    const delegate = /\n\s+(\w+)\(active, ray\);/.exec(body)?.[1];
    if (delegate) {
      const helper = viewport.indexOf(`const ${delegate} = (`);
      assert.ok(helper >= 0, `${branch} delegates to ${delegate}, which must exist`);
      body += viewport.slice(helper, viewport.indexOf("\n  };", helper));
    }
    assert.match(body, /updateDraft\(/, `${branch} must propose through the draft store`);
    assert.doesNotMatch(body, /patchScene|patchContainer|patchFluid|commitEdit|commitDraft/,
      `${branch} must not write the scene document while the pointer is down`);
  }

  const controller = readFileSync(new URL("../lib/simulation/controller.ts", import.meta.url), "utf8");
  assert.match(controller, /shapeFluidBody\(box: FluidBodyBox\)/,
    "there must be no per-move variant that a caller could reach for");
  assert.match(controller, /commitDraft\(options: \{ announceRebuild\?: string \} = \{\}\)/);
});

test("only geometry-preserving drafts are presented to the renderer", () => {
  // The renderer draws the fluid from solver-owned textures. Presenting a
  // container the solver did not allocate for would tear, so the render loop
  // pins the solver to the committed scene and presents a draft only for
  // terrain, which cannot move the lattice.
  const viewport = readFileSync(new URL("../components/WebGPUViewport.tsx", import.meta.url), "utf8");
  assert.match(viewport, /PRESENTED_DRAFT_SUBJECTS\.has\(draft\?\.subject/,
    "the presented subjects must be a declared list, not an ad-hoc condition");
  const declared = /const PRESENTED_DRAFT_SUBJECTS[\s\S]*?new Set(?:<[^>]*>)?\(\[([^\]]*)\]/.exec(viewport)?.[1] ?? "";
  // Terrain cannot move the lattice, and scenery is render-only and outside the
  // solver's keys entirely. Anything else — the tank, the water, a body — owns
  // geometry the solver allocated for, and presenting it early would tear.
  assert.deepEqual(declared.match(/"[a-z-]+"/g)?.sort(), ['"scenery"', '"terrain"'],
    "only a draft that cannot outrun the solver may reach the renderer");
  assert.match(viewport, /renderer\.setSimulationScene\(presentationScene === scene \? undefined : scene\)/,
    "presenting a draft must pin the solver to the committed scene");

  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  assert.match(renderer, /this\.currentGPUFluid\(this\.simulationScene \?\? scene, sceneConfig/,
    "the rebuild key must read the pinned scene, never the presented one");
});
