import assert from "node:assert/strict";
import test from "node:test";
import { assertUniformMixedOptions, type UniformMixedOptionalFeatures } from "../lib/methods/uniform/uniform-mixed-options";
import { uniformGeometricSolverOptions } from "../lib/methods/uniform/uniform-geometric-options";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";

function defaults(): UniformMixedOptionalFeatures {
  const o = uniformGeometricSolverOptions();
  return {
    velocityTransport: o.velocityTransport!, airborneMomentum: o.airborneMomentum!,
    liquidOnlyVelocityAdvection: o.liquidOnlyVelocityAdvection!, volumePressureRows: 0,
    volumeCompaction: o.volumeCompaction!, phiSeedFromVolume: o.phiSeedFromVolume!,
    phiAgreementGain: o.phiAgreementGain!, redistanceSurface: o.redistanceSurface!,
    orphanVolume: o.orphanVolume!, orphanVolumeRender: o.orphanVolumeRender!,
    isolatedBodyVolume: o.isolatedBodyVolume!, phiSeedCells: o.phiSeedCells!,
  };
}

test("mixed optional-feature gate accepts actual Geometric defaults", () => {
  assert.doesNotThrow(() => assertUniformMixedOptions(defaults()));
  assert.doesNotThrow(() => assertUniformMixedOptions({...defaults(), redistanceSurface: "rebuild"}));
});

test("mixed rejects unsupported optional features explicitly", () => {
  const experiments: Partial<UniformMixedOptionalFeatures>[] = [
    {velocityTransport: "maccormack"}, {airborneMomentum: true}, {liquidOnlyVelocityAdvection: true},
    {volumePressureRows: 1}, {volumePressureRows: 2}, {volumeCompaction: true}, {phiSeedFromVolume: true},
    {phiAgreementGain: .05}, {redistanceSurface: "preserve"}, {redistanceSurface: "sparse"},
    {orphanVolume: "local"}, {orphanVolume: "compact"}, {orphanVolumeRender: "density"},
    {isolatedBodyVolume: true}, {phiSeedCells: true},
  ];
  for (const options of experiments) assert.throws(() => assertUniformMixedOptions({...defaults(), ...options}), /Mixed Uniform currently requires/);
  assert.throws(() => assertUniformMixedOptions({...defaults(), velocityTransport: "maccormack", airborneMomentum: true}), /MacCormack momentum, airborne momentum/);
});

test("preparation and capture reject before touching GPU or consuming state", async () => {
  // No device: accessing GPU resources before validation would fail this check.
  const host = Object.assign(Object.create(WebGPUUniformReferenceSolver.prototype), defaults(), {airborneMomentum: true});
  await assert.rejects(host.prepareMixedTransportForQA([]), /airborne momentum/);
  assert.throws(() => host.captureMixedTransportForQA(1 / 30), /airborne momentum/);
  assert.equal(host.mixedCaptureConsumed, undefined);
  assert.equal(host.mixedTransportForQA, undefined);
});
