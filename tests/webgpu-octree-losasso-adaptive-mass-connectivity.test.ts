import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { octreeLosassoAdaptiveMassWGSL } from
  "../lib/webgpu-octree-losasso-adaptive-mass.wgsl";
test("sharpening only scatters mass after reaching the rho=.5 contour", () => {
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /stagedRho\(neighbor\)<SHARPEN_RHO_THRESHOLD/);
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /D=1\.1h is within the paper's stated 1\.1h\.\.3\.1h range/);
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /if\(count==0u\)\{transfers\[first\]=Transfer\(donor,donor,units,1u\);return;\}/);
});

test("rho=.5 remains the unmasked simulation surface", () => {
  assert.doesNotMatch(octreeLosassoAdaptiveMassWGSL, /priorWetReach|copyNodalPseudoPhi/);
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /let value=clamp\(\(threshold-rho\)\*localH,-2\.\*localH,2\.\*localH\)/);
});

test("transport admission is independent of donor scheduling order", () => {
  assert.doesNotMatch(octreeLosassoAdaptiveMassWGSL,
    /atomicCompareExchangeWeak\(&transportAdmission\[transfer\.recipient\]/);
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /let accepted=min\(transfer\.weightBits,u32\(floor\(f32\(transfer\.weightBits\)\/max\(1\.,predicted\)\)\)\)/);
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /atomicAdd\(&transportAdmission\[transfer\.recipient\],accepted\)/);
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /factor=f32\(room\)\/f32\(max\(remoteTotal,1u\)\)/);
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /let accepted=min\(tentative,u32\(floor\(f32\(tentative\)\*factor\)\)\)/);
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /transferRemainders\[i\]=transfer\.weightBits-accepted/);
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /fn returnTransferRemainders/);
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /atomicAdd\(&nextMass\[transfer\.donor\],remainder\)/);
});

test("new liquid requires a face-reachable bridge which stays wet", () => {
  const admissible = (recipientWasWet: boolean,
    neighbours: readonly { wasWet: boolean; predictedRho: number }[]) =>
    recipientWasWet || neighbours.some((neighbor) =>
      neighbor.wasWet && neighbor.predictedRho >= 0.5);
  assert.equal(admissible(false, [{ wasWet: true, predictedRho: 0.49 }]), false,
    "a wet->dry / dry->wet threshold swap must not open a gap");
  assert.equal(admissible(false, [{ wasWet: true, predictedRho: 0.5 }]), true,
    "a persistent wet bridge admits CFL-local surface advance");
  assert.equal(admissible(true, []), true,
    "existing detached sheets remain valid liquid authority");
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /rhoOf\(neighbor\)>=SURFACE_RHO_ISO&&tentativeWet\(neighbor\)/);
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /atomicOr\(&transportAdmission\[i\],0x80000000u\)/);
  const host = readFileSync(new URL(
    "../lib/webgpu-octree-losasso-adaptive-mass.ts", import.meta.url), "utf8");
  const scatter = host.indexOf('this.run(broker, "scatterTransfers"');
  const mark = host.indexOf('this.run(broker, "markTransportSurfaceReach"');
  const finalize = host.indexOf('this.run(broker, "finalizeTransfers"');
  assert.ok(scatter >= 0 && scatter < mark && mark < finalize,
    "persistent reach must observe complete tentative recipient aggregates");
  for (const span of [1, 2, 4]) {
    const covered = span ** 3;
    const threshold = covered * 32_768;
    const cap = threshold - covered;
    assert.equal(threshold / covered, 32_768,
      `span-${span} rho=.5 threshold stays exact in integer space`);
    assert.equal(Math.floor(cap / covered), 32_767,
      `span-${span} cap remains one unit air-side after uniform refine`);
    const massQuantum = Math.fround(0.05 ** 3 / 65_536);
    const decodedRho = Math.fround(threshold * massQuantum) / (span * 0.05) ** 3;
    assert.ok(Math.abs(decodedRho - 0.5) <= 1 / 65_536,
      `span-${span} integer threshold decodes to rho=.5 within one unit`);
  }
  const threshold = 32_768, cap = threshold - 1;
  const selfUnits = 30_000, remoteUnits = 5_000;
  const admitted = selfUnits + Math.min(remoteUnits, Math.max(0, cap - selfUnits));
  assert.equal(admitted, threshold - 1,
    "an unreachable tail must remain strictly air-side, not seed equality next step");
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /fn surfaceThresholdUnits\(r:Leaf\)->u32\{return leafCells\(r\)\*32768u;\}/);
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /let cap=select\(0u,threshold-covered,threshold>=covered\)/);
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /rhoOf\(i\)>=SURFACE_RHO_ISO/,
    "authored exact-contour cells remain admissible");
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
    /f32\(transfer\.weightBits\)\/max\(1\.,predicted\)/);
  assert.match(octreeLosassoAdaptiveMassWGSL,
    /transferRemainders\[i\]=transfer\.weightBits-accepted/);
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
