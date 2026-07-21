import assert from "node:assert/strict";
import test from "node:test";
import { finePublicationGateDiagnostics } from "../lib/octree-fine-publication-diagnostics";

test("fine publication diagnostics preserve the paper's stage order", () => {
  const stages = finePublicationGateDiagnostics({
    generation: 3, topologyFlags: 0, downstreamReason: 0, published: true, rolledBack: false,
    transportCommitted: true, redistanceCommitted: true, volumeFlags: 0x8000_0000,
  });
  assert.deepEqual(stages.map(({ id }) => id),
    ["section5", "transport", "topology", "redistance", "volume", "publication"]);
  assert.ok(stages.every(({ state }) => state === "ready"));
});

test("fine publication diagnostics localize a Section 5 transport rejection", () => {
  const stages = finePublicationGateDiagnostics({
    generation: 3, topologyFlags: 16, downstreamReason: 8, published: true, rolledBack: true,
    redistanceCommitted: true, volumeFlags: 0x8000_0000, transportCommitted: false,
    transportUnavailable: 11, transportFaceBandUnavailable: 11,
    faceBandTransitionFlags: 4, faceBandPointFlags: 1,
  });
  assert.equal(stages.find(({ id }) => id === "topology")?.state, "ready",
    "downstream bit 16 must not be mislabeled as a topology-construction fault");
  assert.equal(stages.find(({ id }) => id === "transport")?.state, "failed");
  assert.match(stages.find(({ id }) => id === "section5")?.detail ?? "", /transition adjacency/);
  assert.match(stages.find(({ id }) => id === "publication")?.detail ?? "", /transport/);
});

test("bootstrap names transport and the Section 5 band as not required", () => {
  const stages = finePublicationGateDiagnostics({
    generation: 2, topologyFlags: 0, downstreamReason: 0, published: true, rolledBack: false,
    redistanceCommitted: true, volumeFlags: 0x8000_0000,
  });
  assert.equal(stages.find(({ id }) => id === "transport")?.state, "not-required");
  assert.equal(stages.find(({ id }) => id === "section5")?.state, "not-required");
});
