import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { canonicalScene, validateScene, type SceneDescription, type Vec3 } from "../lib/model";
import { createEmptyScene, SCENE_STARTERS } from "../lib/empty-scene";
import { getScenePreset } from "../lib/scenes";
import {
  editorFluidLattice,
  fluidBrickCenter,
  fluidBrickIndexAt,
  fluidBrickKey,
  fluidBrushSample,
} from "../lib/editor-fluid";
import { applySceneDraft, useSceneDraftStore } from "../lib/stores/scene-draft-store";
import { planSceneRuntime } from "../lib/scene-runtime";

/**
 * Drive a stroke the way `WebGPUViewport.paintFluidAt` does: sample against the
 * scene as the pointer currently proposes it, revise the draft, and never touch
 * the committed document until the pointer comes up.
 */
function stroke(committed: SceneDescription, points: readonly Vec3[], erase = false) {
  const draftStore = useSceneDraftStore.getState();
  draftStore.beginDraft("fluid-body", erase ? "Erased water" : "Painted water");
  let lastBrickKey: string | undefined;
  let revisions = 0;
  for (const point of points) {
    const proposed = applySceneDraft(committed, useSceneDraftStore.getState().draft);
    const sample = fluidBrushSample(committed, proposed, point, erase);
    if (!sample || sample.brickKey === lastBrickKey) continue;
    lastBrickKey = sample.brickKey;
    if (!sample.patch) continue;
    useSceneDraftStore.getState().updateDraft(sample.patch);
    revisions += 1;
  }
  const draft = useSceneDraftStore.getState().draft;
  useSceneDraftStore.getState().clearDraft();
  return { revisions, patch: draft?.patch ?? {} };
}

/** Centres of `count` distinct paintable bricks, walking the tank floor first. */
function brickRun(scene: SceneDescription, count: number): Vec3[] {
  const lattice = editorFluidLattice(scene);
  const points: Vec3[] = [];
  for (let y = 0; y < lattice.bricks.y; y += 1) {
    for (let z = 0; z < lattice.bricks.z; z += 1) {
      for (let x = 0; x < lattice.bricks.x; x += 1) {
        const centre = fluidBrickCenter(lattice, { x, y, z });
        // The lattice rounds its brick count up, so the last row on an axis can
        // have its centre outside the tank. Those are not paintable.
        if (centre.y < scene.container.height_m && points.length < count) points.push(centre);
      }
    }
  }
  assert.equal(points.length, count, "the tank must actually hold this many paintable bricks");
  return points;
}

test("a twenty-brick stroke leaves the committed document untouched until release", () => {
  const committed = SCENE_STARTERS.find(({ id }) => id === "hall")!.create();
  const before = canonicalScene(committed);
  const points = brickRun(committed, 20);

  const { revisions, patch } = stroke(committed, points);
  assert.equal(canonicalScene(committed), before,
    "the scene the solver keys off must not move while a stroke is open");
  // Twenty proposals are free; what the plan bounds is document writes, and
  // there is exactly one — the caller's single `commitDraft`.
  assert.equal(revisions, 20);
  assert.equal(patch.fluid?.initialBrickSeeds_m?.length, 20);
  assert.deepEqual(validateScene({ ...committed, ...patch }), []);
});

test("painted water adds to the authored initial condition", () => {
  const committed = getScenePreset("water-box-dam-break").create();
  const { patch } = stroke(committed, brickRun(committed, 3));
  assert.equal(patch.fluid?.initialCondition, "dam-break", "the dam is still there");
  assert.equal(patch.fluid?.initialBrickSeedsAdditive, true,
    "a brush stroke adds to the authored dam rather than replacing it");
});

test("re-sampling inside one brick proposes nothing", () => {
  const committed = getScenePreset("water-box-dam-break").create();
  const lattice = editorFluidLattice(committed);
  const centre = fluidBrickCenter(lattice, { x: 1, y: 0, z: 0 });
  const jitter = lattice.brickSize_m.x * 0.2;
  const { revisions } = stroke(committed, [
    centre,
    { ...centre, x: centre.x + jitter },
    { ...centre, x: centre.x - jitter },
  ]);
  assert.equal(revisions, 1, "a slow drag inside one brick is one proposal, not three");
});

test("each brick unions with the stroke so far, not with the committed scene", () => {
  // Sampling against the committed document would make every brick overwrite
  // the last, and a five-brick stroke would land one brick of water.
  const committed = getScenePreset("water-box-dam-break").create();
  const points = brickRun(committed, 5);
  const { patch } = stroke(committed, points);
  const lattice = editorFluidLattice(committed);
  const key = (point: Vec3) => fluidBrickKey(fluidBrickIndexAt(lattice, point)!);
  const painted = new Set((patch.fluid?.initialBrickSeeds_m ?? []).map(key));
  for (const point of points) {
    assert.ok(painted.has(key(point)), `brick ${key(point)} must survive the bricks painted after it`);
  }
});

test("a brick whose centre falls outside the tank is not paintable", () => {
  // The lattice rounds up, so the top brick row of a 24 x 18 x 16 cell tank has
  // its centre above the lid. Painting there authored a seed `validateScene`
  // rejects, which showed up later as a document that would not save.
  const committed = getScenePreset("water-box-dam-break").create();
  const lattice = editorFluidLattice(committed);
  const outside = fluidBrickCenter(lattice, { x: 0, y: lattice.bricks.y - 1, z: 0 });
  assert.ok(outside.y >= committed.container.height_m, "this fixture must actually be the overhanging row");
  const sample = fluidBrushSample(committed, committed, outside, false);
  assert.ok(sample, "it is still a brick, so a drag through it must not restart the stroke");
  assert.equal(sample?.patch, undefined);
});

test("the first painted brick asks a fluid-free scene for a solver", () => {
  const empty = createEmptyScene();
  assert.equal(planSceneRuntime(empty).fluidSolver, false);
  const { patch } = stroke(empty, brickRun(empty, 4));
  // The one expensive transition in the editor, and the plan requires it to be
  // exactly this: user-initiated, named, and once per scene.
  assert.equal(patch.systems?.fluid, true);
  const painted = { ...empty, ...patch };
  assert.deepEqual(validateScene(painted), []);
  assert.equal(planSceneRuntime(painted).fluidSolver, true);
  // Every sample has to carry the flip, because `updateDraft` replaces the
  // draft's patch outright — a later sample that omitted it would retract it.
  assert.equal(stroke(empty, brickRun(empty, 1)).patch.systems?.fluid, true);
});

test("erasing never takes the solver back off a scene that has had water", () => {
  const empty = createEmptyScene();
  const painted = { ...empty, ...stroke(empty, brickRun(empty, 3)).patch };
  const { patch } = stroke(painted, brickRun(painted, 3), true);
  assert.equal(patch.fluid?.initialBrickSeeds_m, undefined, "an emptied list drops the field");
  assert.notEqual(patch.systems?.fluid, false, "erasing is not a way to unrequest the fluid lane");
  assert.deepEqual(validateScene({ ...painted, ...patch }), []);
});

test("the brush writes through the draft rather than the document", () => {
  // The source-level guard exists because the failure is silent: a stroke that
  // patched the scene per brick would still paint correctly, and only the
  // re-seed storm behind it would show, on a device this suite cannot reach.
  const viewport = readFileSync(new URL("../components/WebGPUViewport.tsx", import.meta.url), "utf8");
  const paint = viewport.slice(viewport.indexOf("const paintFluidAt"), viewport.indexOf("const placeBodyAt"));
  assert.ok(paint.length > 0, "paintFluidAt must still exist for this guard to mean anything");
  assert.equal(/patchScene/.test(paint), false, "a stroke must not write the document per brick");
  assert.match(paint, /updateDraft/);
  assert.match(viewport, /simulation\.beginDraft\("fluid-body"/);
});
