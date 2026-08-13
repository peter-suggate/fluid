import assert from "node:assert/strict";
import test from "node:test";
import { SCENE_STARTERS } from "../lib/empty-scene";
import { sceneDamBreakBox } from "../lib/initial-fluid";
import { cloneScene, defaultScene, type SceneDescription } from "../lib/model";
import { sceneIsoGlyph, sceneIsoGlyphLabel, type IsoBox, type SceneIsoGlyph } from "../lib/scene-glyph-iso";
import { getScenePreset, scenePresets } from "../lib/scenes";

function tankFill(fillFraction: number): SceneDescription {
  const scene = cloneScene(defaultScene);
  scene.fluid.initialCondition = "tank-fill";
  scene.container.fillFraction = fillFraction;
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.initialBrickSeedsAdditive;
  delete scene.fluid.inflow;
  return scene;
}

function volumes(glyph: SceneIsoGlyph): readonly IsoBox[] {
  return [...(glyph.water ?? []), ...glyph.bodies];
}

test("the mark is a pure function of the document", () => {
  for (const id of ["garden-pond", "ocean-seiche", "hose-tank"]) {
    const scene = getScenePreset(id).create();
    assert.deepEqual(sceneIsoGlyph(scene), sceneIsoGlyph(scene), id);
  }
});

test("every emitted coordinate is finite and inside the room", () => {
  for (const preset of scenePresets) {
    const glyph = sceneIsoGlyph(preset.create());
    const { extent } = glyph;
    // The longest axis is exactly one, which is what lets two scenes be drawn
    // against each other without either being stretched to fill its own card.
    assert.ok([extent.x, extent.y, extent.z].every((axis) => axis > 0 && axis <= 1), `${preset.id} extent`);
    assert.equal(Math.max(extent.x, extent.y, extent.z), 1, `${preset.id} longest axis`);
    assert.ok(Number.isFinite(glyph.size_m) && glyph.size_m > 0, `${preset.id} size`);

    for (const { min, max } of volumes(glyph)) {
      for (const axis of ["x", "y", "z"] as const) {
        assert.ok(Number.isFinite(min[axis]) && Number.isFinite(max[axis]), `${preset.id} finite ${axis}`);
        assert.ok(min[axis] >= 0 && max[axis] <= extent[axis] + 1e-12, `${preset.id} inside ${axis}`);
        assert.ok(max[axis] > min[axis], `${preset.id} non-degenerate ${axis}`);
      }
    }
    for (const row of glyph.terrain?.rows ?? []) {
      assert.ok(row.z >= 0 && row.z <= extent.z, `${preset.id} row depth`);
      assert.ok(row.heights.every((height) => height >= 0 && height <= extent.y), `${preset.id} row heights`);
    }
    if (glyph.inflow) {
      const { origin, direction } = glyph.inflow;
      assert.ok(origin.x >= 0 && origin.x <= extent.x && origin.y >= 0 && origin.y <= extent.y, `${preset.id} inflow origin`);
      assert.ok(Math.abs(Math.hypot(direction.x, direction.y, direction.z) - 1) < 1e-12, `${preset.id} inflow direction`);
    }
    assert.ok(glyph.bodies.length <= 12, `${preset.id} respects the renderer's body cap`);
  }
});

test("a tank fill puts the waterline at the fill fraction", () => {
  const scene = tankFill(0.75);
  const glyph = sceneIsoGlyph(scene);
  const [water] = glyph.water ?? [];
  assert.ok(water);
  assert.deepEqual(water.min, { x: 0, y: 0, z: 0 });
  assert.equal(water.max.x, glyph.extent.x);
  assert.equal(water.max.z, glyph.extent.z);
  assert.ok(Math.abs(water.max.y - 0.75 * glyph.extent.y) < 1e-12);
  assert.equal(sceneIsoGlyphLabel(glyph).includes("water at 75%"), true, sceneIsoGlyphLabel(glyph));

  // An empty tank is not "water at 0%" — it has no water at all.
  assert.equal(sceneIsoGlyph(tankFill(0)).water, undefined);
});

test("a dam break draws the reservoir the seeder uses", () => {
  const scene = getScenePreset("minimal-power-dam-break").create();
  const reservoir = sceneDamBreakBox(scene);
  const glyph = sceneIsoGlyph(scene);
  const [water] = glyph.water ?? [];
  assert.ok(water);
  const { width_m, height_m, depth_m } = scene.container;
  const scale = 1 / glyph.size_m;
  assert.ok(Math.abs(water.max.x - reservoir.max.x * width_m * scale) < 1e-12);
  assert.ok(Math.abs(water.max.y - reservoir.max.y * height_m * scale) < 1e-12);
  assert.ok(Math.abs(water.max.z - reservoir.max.z * depth_m * scale) < 1e-12);
});

test("spherical paper scenes advertise a spherical vessel", () => {
  for (const id of ["cm12-figure-8", "cm12-figure-12"]) {
    const glyph = sceneIsoGlyph(getScenePreset(id).create());
    assert.equal(glyph.tank.shape, "sphere");
    assert.match(sceneIsoGlyphLabel(glyph), /^Glass sphere, closed vessel,/);
  }
});

test("a fluid-disabled scene has no water and no hose", () => {
  const scene = getScenePreset("garden-svo-lighting").create();
  assert.equal(scene.systems?.fluid, false);
  const glyph = sceneIsoGlyph(scene);
  assert.equal(glyph.water, undefined);
  assert.equal(glyph.inflow, undefined);
  assert.ok(glyph.terrain, "the ground is still part of the document");
  assert.equal(sceneIsoGlyphLabel(glyph).includes("no water"), true, sceneIsoGlyphLabel(glyph));
});

test("a slab seeded once per brick along depth merges into one volume", () => {
  // Twenty translucent boxes stacked face to face is a grid of seams, not a
  // body of water; the mark has to show the slab as the one volume it is.
  const scene = getScenePreset("ocean-seiche").create();
  assert.equal(scene.fluid.initialBrickSeeds_m?.length, 20, "the ocean slab is authored as twenty seeds");
  const glyph = sceneIsoGlyph(scene);
  const seeded = (glyph.water ?? []).slice(1);
  assert.ok(seeded.length > 0 && seeded.length <= 2, `merged to ${seeded.length} volumes`);
  assert.ok(seeded.every((volume) => volume.max.z - volume.min.z > 0.5 * glyph.extent.z),
    "the merged slab spans the depth its seeds do");
});

test("non-additive seeds own the water outright", () => {
  const glyph = sceneIsoGlyph(getScenePreset("twin-dam-collision").create());
  // Two reservoirs; the base dam-break box is not drawn beneath them.
  assert.equal(glyph.water?.length, 2);
  assert.ok(glyph.water!.every((volume) => volume.max.y < glyph.extent.y), "seeds are brick-sized, not tank-sized");
});

test("an inflow becomes an origin and a unit direction", () => {
  const scene = getScenePreset("hose-tank").create();
  const inflow = scene.fluid.inflow;
  assert.ok(inflow);
  const glyph = sceneIsoGlyph(scene);
  assert.ok(glyph.inflow);
  const scale = 1 / glyph.size_m;
  assert.equal(glyph.inflow.origin.x, (inflow.center_m.x + 0.5 * scene.container.width_m) * scale);
  assert.equal(glyph.inflow.origin.y, inflow.center_m.y * scale);
  // The jet runs +x, and the mark keeps world axes: nothing is projected until
  // the component draws it.
  assert.equal(glyph.inflow.direction.x, 1);
});

test("a body authored outside the room is trimmed to the room", () => {
  const scene = tankFill(0.3);
  scene.rigidBodies = [{
    id: "runaway", name: "Runaway", shape: "box",
    dimensions_m: { x: scene.container.width_m * 4, y: 0.1, z: 0.1 },
    density_kg_m3: 900, position_m: { x: scene.container.width_m, y: 0.5 * scene.container.height_m, z: 0 },
    orientation: { w: 1, x: 0, y: 0, z: 0 },
    linearVelocity_m_s: { x: 0, y: 0, z: 0 }, angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
    restitution: 0.2, friction: 0.4,
  }];
  const glyph = sceneIsoGlyph(scene);
  const [body] = glyph.bodies;
  assert.ok(body);
  assert.equal(body.max.x, glyph.extent.x);
  assert.ok(body.min.x >= 0);
  assert.equal(body.round, false, "a box is not drawn as a round mass");
});

test("the presets a card must tell apart are pairwise distinct", () => {
  const ids = ["garden-pond", "ocean-seiche", "minimal-power-dam-break", "twin-dam-collision"];
  const marks = ids.map((id) => JSON.stringify(sceneIsoGlyph(getScenePreset(id).create())));
  assert.equal(new Set(marks).size, ids.length, marks.join("\n"));
});

test("the starters differ by size, which is the choice they offer", () => {
  // The row draws them against one another, so the mark has to carry true size
  // as well as proportion — three identically sized boxes would be a lie.
  const glyphs = SCENE_STARTERS.map((starter) => sceneIsoGlyph(starter.create()));
  const sizes = glyphs.map(({ size_m }) => size_m);
  assert.equal(new Set(sizes).size, sizes.length, `starter sizes ${sizes.join(", ")}`);
  const [room, small, hall] = glyphs;
  assert.ok(small.size_m < room.size_m && room.size_m < hall.size_m, "small room, room, hall");
  // A cube is a cube under any scale; a hall is longer than it is tall.
  assert.deepEqual(small.extent, { x: 1, y: 1, z: 1 });
  assert.ok(hall.extent.y < hall.extent.x);
  for (const glyph of glyphs) {
    assert.equal(glyph.water, undefined, "a fresh room has no water");
    assert.deepEqual(glyph.bodies, []);
    assert.equal(glyph.tank.glass, false, "a starter hands over a room, not an aquarium");
  }
});
