import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";

const source = (path: string) => readFileSync(new URL(`../${path}`, import.meta.url), "utf8");

test("the library and viewed scenes have separate route pages", () => {
  assert.match(source("app/page.tsx"), /<SceneLibrary\s*\/>/);
  assert.doesNotMatch(source("app/page.tsx"), /<FluidLab\s*\/>/);
  assert.match(source("app/scene/page.tsx"), /return null/);
  assert.doesNotMatch(source("components/FluidLab.tsx"), /<SceneLibrary\s*\/>/);
});

test("the shared app shell retains one lazy studio mount across those pages", () => {
  const shell = source("components/AppShell.tsx");
  assert.match(shell, /studioEntered \|\| sceneVisible/);
  assert.match(shell, /hidden=\{!sceneVisible\}/);
  assert.match(shell, /<FluidLab\s*\/>/);
  assert.match(source("app/layout.tsx"), /<AppShell>\{children\}<\/AppShell>/);
});

test("opening a card establishes the scene before navigating to its page", () => {
  const library = source("components/SceneLibrary.tsx");
  assert.match(library, /simulation\.openSceneCard\(card\)/);
  assert.match(library, /router\.push\(currentScenePageUrl\(\)\)/);
});

test("the retained studio leaves the library URL and document untouched", () => {
  const urlState = source("lib/url-state.ts");
  assert.match(urlState, /!scenePageActive\(\)/);
  assert.match(urlState, /if \(scenePageActive\(\)\) hydrate\(\)/);
});
