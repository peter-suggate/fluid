import assert from "node:assert/strict";
import test from "node:test";

import {
  auditOctreeConnectedFlow,
  type OctreeFlowFaceFragment,
} from "../lib/octree-connected-flow-contract";

test("canonical fragments connect ocean through a refined channel to the garden and spillway", () => {
  const regionIds = ["ocean", "channel-west", "channel-east", "garden", "spillway"];
  const regions = regionIds.map((id) => ({ id }));
  const worldX = 1_500_000;
  const coarse = (x: number, negative: string, positive: string): OctreeFlowFaceFragment => ({
    negative, positive, origin: [x, 2_000_000, 3_000_000], axis: 0, span: 4, area: 16, normalVelocity: 0.5,
  });
  const fragments: OctreeFlowFaceFragment[] = [
    coarse(worldX, "ocean", "channel-west"),
    // Four equal-area fine fragments cross the adaptive 4 -> 2 transition.
    ...[[0, 0], [0, 2], [2, 0], [2, 2]].map(([dy, dz]) => ({
      negative: "channel-west", positive: "channel-east",
      origin: [worldX + 8, 2_000_000 + dy, 3_000_000 + dz] as const,
      axis: 0 as const, span: 2, area: 4, normalVelocity: 0.5,
    })),
    coarse(worldX + 16, "channel-east", "garden"),
    coarse(worldX + 24, "garden", "spillway"),
  ];

  const audit = auditOctreeConnectedFlow(regions, fragments, "ocean", "spillway");
  assert.equal(audit.connected, true);
  assert.deepEqual([...audit.visited].sort(), [...regionIds].sort());
  assert.equal(audit.faceKeys.length, 7);
  assert.ok(audit.faceKeys.every((key) => Number(key.split(":")[0]) > 1023), "wide origins remain exact");
  assert.equal(audit.netOutwardFlux.get("ocean"), 8);
  assert.equal(audit.netOutwardFlux.get("channel-west"), 0);
  assert.equal(audit.netOutwardFlux.get("channel-east"), 0);
  assert.equal(audit.netOutwardFlux.get("garden"), 0);
  assert.equal(audit.netOutwardFlux.get("spillway"), -8);
  assert.equal(audit.totalBoundaryFlux, 0, "internal canonical fragments conserve global volume");
});

test("connected-flow audit fails closed on duplicate or disconnected canonical topology", () => {
  const regions = [{ id: "ocean" }, { id: "garden" }];
  const face: OctreeFlowFaceFragment = {
    negative: "ocean", positive: "garden", origin: [65_536, 7, 9], axis: 0, span: 1, area: 1, normalVelocity: 2,
  };
  assert.throws(() => auditOctreeConnectedFlow(regions, [face, face], "ocean", "garden"), /duplicate canonical/);
  assert.equal(auditOctreeConnectedFlow(regions, [], "ocean", "garden").connected, false);
});
