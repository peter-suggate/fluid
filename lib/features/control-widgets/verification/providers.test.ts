import assert from "node:assert/strict";
import test from "node:test";
import { createPaddleboatsScene, paddleboatsWidgets } from "../../../core/paddleboats-scene";
import { findSceneDefinition } from "../../../core/scenes";
import { collectWidgets, resolveWidgetLayout } from "../../../framework/widgets";
import { interactionCatalog, type InteractionView } from "../../../interaction";
import { ViewportInputRouter } from "../../interactive/input-router";
import type { InteractionPane } from "../../interactive/pane";
import type { WidgetContext } from "../context";
import { interactionEntityWidgets } from "../interaction-entity";
import { rigidMotorWidgets } from "../rigid-motor";

const identity = { x: 0, y: 0, z: 0, w: 1 };

test("one throttle per motor-carrying hull, placed on its own side by the scene", () => {
  const scene = createPaddleboatsScene();
  scene.rigidBodies[0] = { ...scene.rigidBodies[0]!, motorStrength: -.35 };
  const widgets = collectWidgets([rigidMotorWidgets], { paneId: "a", scene } satisfies WidgetContext);
  assert.deepEqual(widgets.map(widget => [widget.id, widget.label]), [["motor:paddleboat-0", "Ember"], ["motor:paddleboat-1", "Jade"]]);
  assert.deepEqual(widgets[0]!.channels[0], { kind: "axis1", id: "strength", label: "Motor", min: -100, max: 100, step: 5, unit: "%", value: -35, zeroLabel: "Stop" });
  // The boats open at one third of the wheels' rated speed.
  assert.equal(widgets[1]!.channels[0]!.kind === "axis1" && widgets[1]!.channels[0]!.value, 33);
  assert.equal(findSceneDefinition("paddleboats")?.widgets, paddleboatsWidgets);
  const groups = resolveWidgetLayout(widgets, paddleboatsWidgets);
  assert.deepEqual(groups.map(group => [group.anchor, group.widgets[0]!.id]), [["bottom-left", "motor:paddleboat-0"], ["bottom-right", "motor:paddleboat-1"]]);
});

function robotView(controlled: boolean): InteractionView {
  const entity = interactionCatalog.createEntity({ id: "bub", kind: "fluid.robot", name: "Bub", transform: { position_m: { x: 0, y: 0, z: 0 }, orientation: identity } });
  return {
    generation: 1, documentRevision: 1, document: { schemaVersion: 1, entities: [entity], connections: [] },
    readiness: { state: "ready", reasons: [] }, running: controlled, context: controlled ? "play" : "edit",
    controlledEntityId: controlled ? "bub" : undefined, controlCandidates: [{ entityId: "bub", availability: { available: true } }],
  };
}

test("a controllable entity offers take-control, then a pad built from its behaviours' actions", () => {
  const router = new ViewportInputRouter(interactionCatalog);
  let pumped = 0, taken: string | undefined;
  const pane = { catalog: interactionCatalog, router, pumpInput: () => { pumped += 1; }, control: async (id: string) => { taken = id; }, release: async () => {} } as unknown as InteractionPane;
  const idle = { paneId: "a" as const, scene: createPaddleboatsScene(), interaction: { pane, view: robotView(false), busy: false } };
  const [offer] = collectWidgets([interactionEntityWidgets], idle);
  assert.deepEqual(offer!.channels.map(channel => channel.id), ["take-control"]);
  interactionEntityWidgets.input(idle, offer!.id, "take-control", true);
  assert.equal(taken, "bub");

  const view = robotView(true);
  router.update(view);
  const playing = { ...idle, interaction: { pane, view, busy: false } };
  const [pad] = collectWidgets([interactionEntityWidgets], playing);
  const move = pad!.channels.find(channel => channel.kind === "axis2")!;
  assert.ok(pad!.channels.some(channel => channel.kind === "button" && channel.label === "Jump" && channel.shortcut === "Space"));
  assert.equal(pad!.channels.at(-1)!.id, "release-control");
  interactionEntityWidgets.input(playing, pad!.id, move.id, [0, -1]);
  assert.equal(pumped, 1);
  const [sample] = router.sample();
  assert.deepEqual(sample?.value, [0, -1]);
  // The stick and a key sum and clamp like any two held inputs; clear lets go of both.
  router.key("KeyD", true);
  assert.deepEqual(router.sample()[0]?.value, [1, -1]);
  assert.deepEqual(router.clear().map(release => release.value), [[0, 0]]);
  assert.equal(router.holding(), false);
});
