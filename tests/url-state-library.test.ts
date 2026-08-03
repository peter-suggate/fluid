import assert from "node:assert/strict";
import test from "node:test";
import { getScenePreset } from "../lib/scenes";
import { parseQueryState, serializeQueryState, shellSessionFromQuery, shellViewFromQuery } from "../lib/url-state";

function roundTrip(search: string) {
  const parsed = parseQueryState(search);
  const query = serializeQueryState(search, {
    presetId: parsed.presetId,
    scene: parsed.scene,
  }, {
    methodId: parsed.methodId,
    quality: parsed.quality,
    overrides: parsed.overrides,
  }, parsed.ui, { view: parsed.view });
  return { parsed, params: new URLSearchParams(query), query };
}

test("a link to the library round-trips as the open layer", () => {
  const { parsed, params, query } = roundTrip("?view=library");
  assert.equal(parsed.view, "library");
  assert.equal(params.get("view"), "library");
  assert.equal(parseQueryState(query).view, "library");
});

/**
 * The studio is the absence of the key. Emitting `view=studio` would put a
 * redundant parameter into every link the app has ever produced, and the shell
 * already treats anything that is not the library as the studio.
 */
test("the studio round-trips without growing a redundant key", () => {
  const { parsed, params, query } = roundTrip("?scene=water-box-dam-break");
  assert.equal(parsed.view, "studio");
  assert.equal(params.has("view"), false);
  assert.equal(parseQueryState(query).view, "studio");
});

test("an absent view is the studio", () => {
  assert.equal(parseQueryState("").view, "studio");
  assert.equal(shellViewFromQuery(""), "studio");
});

test("a named scene restores the studio after a page or module reload", () => {
  assert.deepEqual(shellSessionFromQuery("?scene=garden-pond", {
    view: "library",
    studioEntered: false,
  }), {
    view: "studio",
    studioEntered: true,
  });
});

test("a bare first visit remains the library front door", () => {
  assert.deepEqual(shellSessionFromQuery("", {
    view: "library",
    studioEntered: false,
  }), {
    view: "library",
    studioEntered: false,
  });
});

test("an explicitly opened library retains the current session's scene", () => {
  assert.deepEqual(shellSessionFromQuery("?scene=garden-pond&view=library", {
    view: "library",
    studioEntered: true,
  }), {
    view: "library",
    studioEntered: true,
  });
});

test("a canonicalized fresh library URL does not invent an entered studio", () => {
  assert.deepEqual(shellSessionFromQuery("?scene=garden-pond&view=library", {
    view: "library",
    studioEntered: false,
  }), {
    view: "library",
    studioEntered: false,
  });
});

test("an unreadable view resolves to the studio rather than throwing", () => {
  for (const value of ["presentation", "STUDIO", "1", ""]) {
    const { parsed, params } = roundTrip(`?view=${value}`);
    assert.equal(parsed.view, "studio");
    assert.equal(params.has("view"), false);
  }
});

/**
 * The shell view is one more key beside the scene state, not a replacement for
 * it: a library link that has been edited is still an edited scene.
 */
test("the scene and preset keys are untouched by the view", () => {
  const scene = getScenePreset("dam-break-boxes").create();
  scene.container.width_m = 1.75;
  const method = { methodId: "tall-cell", quality: "balanced" as const, overrides: {} };

  const query = serializeQueryState("", { presetId: "dam-break-boxes", scene }, method, undefined, { view: "library" });
  const params = new URLSearchParams(query);
  assert.equal(params.get("scene"), "dam-break-boxes");
  assert.equal(params.get("method"), "tall-cell");
  assert.equal(params.get("quality"), "balanced");
  assert.equal(params.get("scene.container.width_m"), "1.75");
  assert.equal(params.get("view"), "library");

  const parsed = parseQueryState(query);
  assert.equal(parsed.presetId, "dam-break-boxes");
  assert.equal(parsed.scene.container.width_m, 1.75);
  assert.equal(parsed.view, "library");

  // Drop the one key and the link is byte-identical to the studio's, so nothing
  // about the library layer can drift into what a link means for the simulation.
  params.delete("view");
  assert.equal(params.toString(), serializeQueryState("", { presetId: "dam-break-boxes", scene }, method));
});
