import assert from "node:assert/strict";
import test from "node:test";

import { geometricInterfaceResidentWGSL } from
  "../lib/methods/adaptive-volume/geometric-interface-resident.wgsl";

const tolerance = 9.5367431640625e-7;
const certifiedFill = (fill: number): number | undefined =>
  !(fill >= -tolerance && fill <= 1 + tolerance)
    ? undefined : Math.min(1, Math.max(0, fill));

test("interface certification accepts and locally clamps transport roundoff", () => {
  assert.equal(certifiedFill(-1.862645149230957e-9), 0);
  assert.equal(certifiedFill(1 + 2 ** -22), 1);
  assert.equal(certifiedFill(-2 * tolerance), undefined);
  assert.equal(certifiedFill(1 + 2 * tolerance), undefined);
  assert.equal(certifiedFill(Number.NaN), undefined);
  assert.equal(certifiedFill(Number.POSITIVE_INFINITY), undefined);
  assert.equal(certifiedFill(Number.NEGATIVE_INFINITY), undefined);
});

test("uniform 2-D and 3-D reconstruction share the certified fill observation", () => {
  assert.match(geometricInterfaceResidentWGSL,
    /fn geometricResidentCertifiedFill[\s\S]*!\(fill >= -tolerance && fill <= 1\.0\+tolerance\)[\s\S]*clamp\(fill,0\.0,1\.0\)/);
  assert.match(geometricInterfaceResidentWGSL,
    /fn geometricResidentFitUniformExtrusion[\s\S]*geometricResidentCertifiedFill\(other,densityOffset\)/);
  assert.match(geometricInterfaceResidentWGSL,
    /fn geometricResidentUniformSample[\s\S]*return geometricResidentCertifiedFill\(cell,densityOffset\)/);
});

test("ELVIRA candidates require orientation evidence on their integration axis", () => {
  assert.match(geometricInterfaceResidentWGSL,
    /fn geometricResidentIntegrationSupported[\s\S]*normal\[axis\]!=0\.0/);
  assert.match(geometricInterfaceResidentWGSL,
    /for\(var direction=0u;direction<2u[\s\S]*geometricResidentIntegrationSupported\(normal,integration\)/);
  assert.match(geometricInterfaceResidentWGSL,
    /for\(var integration=0u;integration<3u[\s\S]*geometricResidentIntegrationSupported\(fallback\.normal,integration\)/);
});
