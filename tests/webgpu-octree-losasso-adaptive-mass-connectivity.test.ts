import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { octreeLosassoAdaptiveMassWGSL } from
  "../lib/webgpu-octree-losasso-adaptive-mass.wgsl";
import { adaptiveMassUnitsPerFinestCell } from
  "../lib/webgpu-octree-losasso-adaptive-mass";
test("sharpening only scatters mass after reaching the rho=.5 contour", () => {
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /stagedRho\(current\)>=SURFACE_RHO_ISO/);
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /2\.1\*sourceSpan-distance/);
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /f32\(requested\)\*weights\[corner\]\/totalWeight/);
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /plan\.recipientUnits\[slot\]\+=remainder/);
});

test("rho=.5 remains the unmasked simulation surface", () => {
  assert.doesNotMatch(octreeLosassoAdaptiveMassWGSL, /priorWetReach|copyNodalPseudoPhi/);
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /let value=clamp\(\(SURFACE_RHO_ISO-rho\)\*localH,-2\.\*localH,2\.\*localH\)/);
});

test("transport admission is independent of donor scheduling order", () => {
  assert.doesNotMatch(octreeLosassoAdaptiveMassWGSL,
    /atomicCompareExchangeWeak\(&transportAdmission\[transfer\.recipient\]/);
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /let tentative=min\(units,u32\(floor\(f32\(units\)\/max\(1\.,predicted\)\)\)\)/);
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /atomicStore\(&transportAdmission\[dst\],remoteUnits\)/);
  assert.doesNotMatch(octreeLosassoAdaptiveMassWGSL, /priorFaceSurfaceReach/);
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /let remainder=units-accepted/);
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /fn returnDonorRemainders/);
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /atomicStore\(&nextMass\[donor\],incoming\+remainderUnits\)/);
});

test("new liquid is not rejected by a prior-frame connectivity veto", () => {
  assert.doesNotMatch(octreeLosassoAdaptiveMassWGSL, /tentativeWet|priorFaceSurfaceReach/);
  const host = readFileSync(new URL(
    "../lib/webgpu-octree-losasso-adaptive-mass.ts", import.meta.url), "utf8");
  const scatter = host.indexOf('this.run(broker, "gatherTentativeTransport"');
  const finalize = host.indexOf('this.run(broker, "finalizeDestinationTransport"');
  assert.ok(scatter >= 0 && scatter < finalize,
    "destination finalization must observe complete tentative recipient aggregates");
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /fn surfaceThresholdUnits\(r:Leaf\)->u32\{return leafCells\(r\)\*\(unitsPerFineCell\(\)\/2u\);\}/);
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /let accepted=acceptedTransferUnits\[arc\]/,
    "all beta\/gamma-admitted mass reaches its traced destination");
});

test("cumulative gamma uses volume integrals across an unequal 2:1 permutation", () => {
  // One span-2 donor splits over eight span-1 recipients while eight span-1
  // donors merge into one span-2 recipient. This is a volume-preserving row
  // permutation: gamma=1 must remain exactly one despite unequal leaf volumes.
  const donorVolumes = [8, ...new Array<number>(8).fill(1)];
  const recipientVolumes = [8, ...new Array<number>(8).fill(1)];
  const transfers = [
    ...new Array(8).fill(0).map((_value, fine) =>
      ({ donor: 0, recipient: fine + 1, fraction: 1 / 8 })),
    ...new Array(8).fill(0).map((_value, fine) =>
      ({ donor: fine + 1, recipient: 0, fraction: 1 })),
  ];
  const integrals = new Array<number>(9).fill(0);
  for (const transfer of transfers) {
    integrals[transfer.recipient] += transfer.fraction * donorVolumes[transfer.donor]!;
  }
  assert.deepEqual(integrals.map((integral, recipient) =>
    integral / recipientVolumes[recipient]!), new Array<number>(9).fill(1));
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /compression\[donor\]\*volume\(leaves\[donor\]\)/);
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /compression\[dst\]=unitsToMass\(atomicLoad\(&nextCompressionMass\[dst\]\)\)\/max\(volume\(leaves\[dst\]\)/);
  assert.doesNotMatch(octreeLosassoAdaptiveMassWGSL,
    /finishTransportLeaves[^\n]*compression\[i\]=1/);
});

test("gamma limits row compression without imposing a rho<=1 mass cap", () => {
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /f32\(units\)\/max\(1\.,predicted\)/);
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /let remainder=units-accepted/);
  assert.doesNotMatch(octreeLosassoAdaptiveMassWGSL,
    /min\(transfer\.weightBits,massToUnits\(volume\(leaves\[transfer\.recipient\]\)\)\)/);
});

test("closed box boundaries retain outward donor mass", () => {
  // Section 3.6 no-flux walls: no tensor weight may leave the axis-aligned
  // domain, independently of the local gamma/compression state.
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /let outwardLow=displacement\[axis\]<0\.&&r\.originSpan\[axis\]==0u/);
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /let outwardHigh=displacement\[axis\]>0\.&&r\.originSpan\[axis\]\+r\.originSpan\.w>=p\.dimsMax\[axis\]/);
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /if\(outwardLow\|\|outwardHigh\)\{crossing\[axis\]=0\.;\}/);
});

test("fixed-point density scale supports 64-cubed and larger domains", () => {
  assert.equal(adaptiveMassUnitsPerFinestCell([32, 32, 32]), 65_536);
  assert.equal(adaptiveMassUnitsPerFinestCell([64, 64, 64]), 8_192);
  assert.equal(adaptiveMassUnitsPerFinestCell([128, 128, 128]), 1_024);
  assert.ok(64 ** 3 * adaptiveMassUnitsPerFinestCell([64, 64, 64]) <= 0xffff_ff00);
});
