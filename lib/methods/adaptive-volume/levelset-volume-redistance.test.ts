import assert from "node:assert/strict";
import test from "node:test";

import { incidentCrossingRedistanceBandReference } from
  "./levelset-volume-redistance.wgsl";

test("coarse crossing cells raise only incident vertex metric bands", () => {
  const planeAcrossH16 = [-8, 8, -8, 8, -8, 8, -8, 8];
  const local = incidentCrossingRedistanceBandReference(4, [{
    widths: [16, 16, 16], cornerPhi: planeAcrossH16,
  }]);
  assert.equal(local, 28, "ceil of the H16 cell diagonal covers every crossing corner");
  assert.ok(local > 8, "the old four-fine-cell band would drop the crossing corners");
  assert.equal(incidentCrossingRedistanceBandReference(4, [{
    widths: [16, 16, 16], cornerPhi: Array(8).fill(8),
  }]), 4, "an unrelated coarse air cell does not inflate the narrow band");
  assert.equal(incidentCrossingRedistanceBandReference(4, [{
    widths: [16, 16, 16], cornerPhi: planeAcrossH16, metric: false,
  }]), 4, "unsupported corners cannot certify a coarse contour crossing");
});
