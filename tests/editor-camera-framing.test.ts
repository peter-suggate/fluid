import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { defaultCamera } from "../lib/model";
import { CAMERA_FRAMINGS, cameraForFraming, cameraFramingForKey } from "../lib/editor-camera-framing";
import { EDITOR_TOOLS } from "../lib/editor-tools";

test("camera framing keys do not collide with tool shortcuts", () => {
  const toolKeys = new Set(EDITOR_TOOLS.map((tool) => tool.shortcut));
  for (const framing of CAMERA_FRAMINGS) {
    assert.equal(toolKeys.has(framing.key), false, `${framing.key} is already a tool shortcut`);
  }
  assert.equal(new Set(CAMERA_FRAMINGS.map((framing) => framing.key)).size, CAMERA_FRAMINGS.length);
});

test("reset survives the removal of the camera toolbar", () => {
  // The buttons are gone; without this the only way back from a camera orbited
  // into empty space would be reloading the page.
  assert.deepEqual(cameraForFraming("reset"), defaultCamera);
  assert.equal(cameraFramingForKey("0"), "reset");
});

test("each framing looks somewhere different", () => {
  const seen = new Set(CAMERA_FRAMINGS.map((framing) => {
    const camera = cameraForFraming(framing.id);
    return `${camera.azimuth_rad}:${camera.elevation_rad}:${camera.distance_m}`;
  }));
  assert.equal(seen.size, CAMERA_FRAMINGS.length);
});

test("an unbound key is not a framing", () => {
  for (const key of ["q", "s", "9", "Escape"]) assert.equal(cameraFramingForKey(key), undefined);
});

test("the viewport frame carries no permanently disabled or duplicated controls", () => {
  const toolbar = readFileSync(new URL("../components/EditorToolbar.tsx", import.meta.url), "utf8");
  assert.match(toolbar, /EDITOR_TOOLS\.filter\(\(tool\) => tool\.status === "active"\)/,
    "a tool that cannot be clicked must not occupy a row");
  assert.doesNotMatch(toolbar, /editor-tool-hint/,
    "the armed tool speaks in the viewport, not in a paragraph pinned under the strip");

  const transport = readFileSync(new URL("../components/TransportBar.tsx", import.meta.url), "utf8");
  assert.doesNotMatch(transport, /FIXED STEP|GPU STEP/,
    "the timestep knobs live in CONFIGURE -> Numerics; two homes is one too many");

  const shell = readFileSync(new URL("../components/FluidLab.tsx", import.meta.url), "utf8");
  assert.doesNotMatch(shell, /camera-toolbar/,
    "camera framing is on the keyboard now");
});

test("choosing a scene does not silently kill every keyboard shortcut", () => {
  // A focused <select> reads as text editing to the shortcut chassis, so a
  // preset dropdown that keeps focus after its choice disables tool arming,
  // camera framing, undo, and delete — with nothing on screen to explain why.
  const shortcuts = readFileSync(new URL("../lib/use-editor-shortcuts.ts", import.meta.url), "utf8");
  assert.match(shortcuts, /\["INPUT", "TEXTAREA", "SELECT"\]\.includes\(target\.tagName\)/,
    "this is the guard the overlay below has to stay out from under");

  // The dropdown is gone: scenes are chosen in the library, and the overlay is
  // a button that opens it. A button cannot hold the focus state that caused
  // this, which is a stronger guarantee than remembering to blur.
  const overlay = readFileSync(new URL("../components/SceneOverlay.tsx", import.meta.url), "utf8");
  assert.doesNotMatch(overlay, /<select/, "the scene chip must not reintroduce a focusable form control");
  assert.match(overlay, /openLibrary/, "the chip's job is to open the library");

  // The library's search box is a text field and is therefore inside the guard
  // above by construction; its own Escape handling must not fight it.
  const library = readFileSync(new URL("../components/SceneLibrary.tsx", import.meta.url), "utf8");
  assert.match(library, /\["INPUT", "TEXTAREA", "SELECT"\]\.includes\(target\.tagName\)/,
    "the library reuses the same editing test rather than inventing a second one");
});
