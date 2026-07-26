import assert from "node:assert/strict";
import test from "node:test";
import { finePublicationGateDiagnostics } from "../lib/octree-fine-publication-diagnostics";

test("fine publication diagnostics preserve the paper's stage order", () => {
  const stages = finePublicationGateDiagnostics({
    generation: 3, topologyFlags: 0, downstreamReason: 0, published: true, rolledBack: false,
    transportCommitted: true, redistanceCommitted: true, volumeFlags: 0x8000_0000,
  });
  assert.deepEqual(stages.map(({ id }) => id),
    ["transport", "topology", "redistance", "volume", "publication"]);
  assert.ok(stages.every(({ state }) => state === "ready"));
});

test("fine publication diagnostics localize a structured transport rejection", () => {
  const stages = finePublicationGateDiagnostics({
    generation: 3, topologyFlags: 16, downstreamReason: 8, published: true, rolledBack: true,
    redistanceCommitted: true, volumeFlags: 0x8000_0000, transportCommitted: false,
    transportUnavailable: 11, transportStructuredAuthorityUnavailable: 11,
  });
  assert.equal(stages.find(({ id }) => id === "topology")?.state, "ready",
    "downstream bit 16 must not be mislabeled as a topology-construction fault");
  assert.equal(stages.find(({ id }) => id === "transport")?.state, "failed");
  assert.match(stages.find(({ id }) => id === "transport")?.detail ?? "", /structured authority unavailable/);
  assert.match(stages.find(({ id }) => id === "publication")?.detail ?? "", /transport/);
});

test("bootstrap names transport as not required", () => {
  const stages = finePublicationGateDiagnostics({
    generation: 2, topologyFlags: 0, downstreamReason: 0, published: true, rolledBack: false,
    redistanceCommitted: true, volumeFlags: 0x8000_0000,
  });
  assert.equal(stages.find(({ id }) => id === "transport")?.state, "not-required");
});
