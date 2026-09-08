import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { EditorHighlightLayer } from "../components/EditorHighlightLayer";
import type { EditorTarget } from "../lib/core/editor-target";
import { findEntity } from "../lib/core/editor-entity-catalog";
import { EDITOR_PROBES } from "../lib/core/editor-probe-catalog";
import { defaultCamera, cloneScene, defaultScene } from "../lib/core/model";

const bounds = { kind: "box" as const, box: { min: { x: -.1, y: .1, z: -.1 }, max: { x: .1, y: .3, z: .1 } } };
const target: EditorTarget = { probeId: "entity", kind: "entity", id: "object", label: "Object", tone: "prop",
  distance_m: 1, point_m: { x: 0, y: .2, z: 0 }, normal: { x: 0, y: 1, z: 0 },
  highlight: { kind: "instance-range", first: 4, last: 12 }, hoverHighlight: bounds };
const render = (props: Partial<Parameters<typeof EditorHighlightLayer>[0]> = {}) => renderToStaticMarkup(createElement(EditorHighlightLayer,
  { target, camera: defaultCamera, width: 1000, height: 800, ...props }));

test("instanced objects expose their selection bounds as a quiet hover without a GPU readback", () => {
  const html = render();
  assert.match(html, /data-hover="true"/);
  assert.match(html, /data-highlight-kind="box"/);
  assert.equal((html.match(/class="highlight-edge"/g) ?? []).length, 12);
  assert.doesNotMatch(html, /data-live="true"/);
  assert.equal(render({ target: undefined }), "");
});

test("idle shape ghosts retain faint object bounds but an active gesture suppresses them", () => {
  const held = { highlight: { kind: "paths" as const, paths: [[{ x: -.1, y: .2, z: 0 }, { x: .1, y: .2, z: 0 }]] }, tone: "fluid" as const };
  const idle = render({ held, showHoverWithHeld: true });
  assert.match(idle, /data-hover="true"/);
  assert.match(idle, /data-live="true"/);
  assert.match(idle, /data-highlight-kind="paths"/);
  const dragging = render({ held });
  assert.doesNotMatch(dragging, /data-hover="true"/);
  assert.doesNotMatch(dragging, /data-highlight-kind="box"/);
  assert.match(dragging, /data-highlight-kind="paths"/);
  const ground = render({ target: { ...target, kind: "solid-voxel", hoverHighlight: undefined, highlight: bounds }, held, showHoverWithHeld: true });
  assert.doesNotMatch(ground, /data-highlight-kind="box"/);
});

test("the entity probe contributes the same generic bounds its selection handles use", () => {
  const scene = cloneScene(defaultScene);
  const probe = EDITOR_PROBES.find(probe => probe.id === "entity")!;
  const result = probe.probe({ scene, bodies: [], pickingAvailable: true },
    { origin: { x: 0, y: 10, z: 0 }, direction: { x: 0, y: -1, z: 0 } });
  assert.ok(result?.selection);
  assert.equal(result.hoverHighlight?.kind, "box");
  const entity = findEntity({ scene, bodies: [], pickingAvailable: true }, result.selection!);
  assert.deepEqual(result.hoverHighlight, { kind: "box", box: entity!.box, frame: entity!.frame });
});
