import assert from "node:assert/strict";
import test from "node:test";
import {
  initialResourceReadiness,
  reduceGPUResourceStatus,
  resourceActivities,
  resourceActivitiesFor,
  resourceActivityPresentation,
  type ResourceActivityPresentation,
  type ResourcePluginDefinition,
} from "../lib/resource-readiness";
import { RESOURCE_PLUGIN_CATALOG } from "../lib/resource-plugin-catalog";

const PRESENTATIONS: readonly ResourceActivityPresentation[] = ["card", "transport-inline", "pill"];

const preparing = (plugin: ResourcePluginDefinition, startedAt_ms: number) => ({
  state: "initializing" as const,
  label: `Preparing ${plugin.label}`,
  phase: "allocation",
  completed: 1,
  total: 4,
  startedAt_ms,
  resource: plugin,
});

test("every declared plugin claims exactly one presentation", () => {
  assert.ok(RESOURCE_PLUGIN_CATALOG.length > 0);
  for (const plugin of RESOURCE_PLUGIN_CATALOG) {
    const claimed = PRESENTATIONS.filter((presentation) => resourceActivityPresentation(plugin) === presentation);
    assert.equal(claimed.length, 1, `${plugin.id} must claim exactly one presentation, not ${claimed.length}`);
  }
});

test("what a resource blocks decides how loud it is allowed to be", () => {
  const expected = { viewport: "card", transport: "transport-inline", nothing: "pill" } as const;
  for (const plugin of RESOURCE_PLUGIN_CATALOG) {
    assert.equal(resourceActivityPresentation(plugin), expected[plugin.blocks], plugin.id);
  }
  const byId = Object.fromEntries(
    RESOURCE_PLUGIN_CATALOG.map((plugin) => [plugin.id, resourceActivityPresentation(plugin)]));
  assert.equal(byId["platform.webgpu-renderer"], "card");
  // The declaration decides, not the lane: the complete sparse presentation
  // blocks the viewport because scene interaction waits on its first fenced
  // frame, so it keeps a card without being platform startup.
  assert.equal(byId["presentation.svo-global"], "card");
  assert.equal(byId["fluid.power-octree"], "transport-inline");
  assert.equal(byId["scene.live-svo-source"], "pill");
});

test("an unrecognised declaration takes the quietest presentation rather than throwing", () => {
  assert.equal(resourceActivityPresentation(undefined), "pill");
  assert.equal(resourceActivityPresentation({} as ResourcePluginDefinition), "pill");
  assert.equal(
    resourceActivityPresentation({ blocks: "everything" as ResourcePluginDefinition["blocks"] }),
    "pill");
});

test("concurrent activities partition across the three presentations", () => {
  let readiness = initialResourceReadiness();
  RESOURCE_PLUGIN_CATALOG.forEach((plugin, index) => {
    readiness = reduceGPUResourceStatus(readiness, preparing(plugin, index + 1));
  });

  const all = resourceActivities(readiness);
  assert.equal(all.length, RESOURCE_PLUGIN_CATALOG.length);
  assert.deepEqual(
    PRESENTATIONS.flatMap((presentation) => [...resourceActivitiesFor(readiness, presentation)])
      .map((activity) => activity.pluginId).sort(),
    all.map((activity) => activity.pluginId).sort(),
    "every activity appears in exactly one group and none is dropped");
  assert.deepEqual(
    resourceActivitiesFor(readiness, "transport-inline").map((activity) => activity.pluginId),
    ["fluid.power-octree"]);
  assert.deepEqual(
    resourceActivitiesFor(readiness, "pill").map((activity) => activity.pluginId),
    ["scene.live-svo-source"]);
});

test("an activity from an undeclared plugin never promotes itself to a card", () => {
  const stranger: ResourcePluginDefinition = {
    id: "unknown.resource", lane: "optional", label: "Unknown resource",
    provides: ["optional-tooling"], blocks: "nowhere" as ResourcePluginDefinition["blocks"],
  };
  const readiness = reduceGPUResourceStatus(initialResourceReadiness(), preparing(stranger, 1));

  assert.deepEqual(resourceActivitiesFor(readiness, "card"), []);
  assert.deepEqual(resourceActivitiesFor(readiness, "transport-inline"), []);
  assert.deepEqual(
    resourceActivitiesFor(readiness, "pill").map((activity) => activity.pluginId),
    ["unknown.resource"]);
});
