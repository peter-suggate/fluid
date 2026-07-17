import assert from "node:assert/strict";
import test from "node:test";
import {
  DEFAULT_SECONDARY_PARTICLE_CAPACITY,
  SECONDARY_PARTICLE_STRIDE_BYTES,
  secondaryParticleCapacity,
  secondaryParticleComputeShader,
  secondaryParticleOpticalShader,
  secondaryParticleRenderShader
} from "../lib/webgpu-secondary-particles";

test("secondary particle budgets are bounded and allocation-aligned", () => {
  assert.equal(secondaryParticleCapacity(undefined), DEFAULT_SECONDARY_PARTICLE_CAPACITY);
  assert.equal(secondaryParticleCapacity(1), 1_024);
  assert.equal(secondaryParticleCapacity(4_500), 4_096);
  assert.equal(secondaryParticleCapacity(100_000), 65_536);
  assert.equal(SECONDARY_PARTICLE_STRIDE_BYTES, 48);
});

test("secondary liquid particles use the paper escape and phase rules", () => {
  assert.match(secondaryParticleComputeShader, /samplePhi\(escaped\) <= 2\.0 \* radius/);
  assert.match(secondaryParticleComputeShader, /let generationDt = max\(dt\(\), 1\.0 \/ 30\.0\)/);
  assert.match(secondaryParticleComputeShader, /let radius = clamp\(-trialPhi/);
  assert.match(secondaryParticleComputeShader, /let clusterCount = 1u \+ min\(3u/);
  assert.match(secondaryParticleComputeShader, /velocity \+ params\.gravityAndSeed\.xyz \* dt\(\)/);
  assert.match(secondaryParticleComputeShader, /let drag = select\(0\.18, 2\.8/);
  assert.match(secondaryParticleComputeShader, /kind = 2\.0/);
  assert.match(secondaryParticleComputeShader, /velocity = sampleVelocity\(position\)/);
  assert.doesNotMatch(secondaryParticleComputeShader, /pressure/);
});

test("secondary particle sampling abstracts dense and restricted fields", () => {
  assert.match(secondaryParticleComputeShader, /fn restrictedLayout\(\) -> bool/);
  assert.match(secondaryParticleComputeShader, /fn packedY\(cell: vec3i\)/);
  assert.match(secondaryParticleComputeShader, /fn occupancySurface\(\) -> bool/);
  assert.match(secondaryParticleComputeShader, /here\.x \+ velocityRaw\(q - vec3i\(1, 0, 0\)\)\.x/);
});

test("secondary particle renderer draws soft typed billboards", () => {
  assert.match(secondaryParticleRenderShader, /@builtin\(instance_index\)/);
  assert.match(secondaryParticleRenderShader, /input\.kind > 1\.5/);
  assert.match(secondaryParticleRenderShader, /input\.kind > 0\.5/);
  assert.match(secondaryParticleRenderShader, /smoothstep\(0\.34, 1\.0, radius2\)/);
});

test("spray contributes sphere interfaces to the water optical composite", () => {
  assert.match(secondaryParticleOpticalShader, /fn sphereInterface/);
  assert.match(secondaryParticleOpticalShader, /@builtin\(frag_depth\)/);
  assert.match(secondaryParticleOpticalShader, /@fragment fn sphereFront/);
  assert.match(secondaryParticleOpticalShader, /@fragment fn sphereBack/);
  assert.match(secondaryParticleOpticalShader, /input\.kind >= 0\.5/);
});
