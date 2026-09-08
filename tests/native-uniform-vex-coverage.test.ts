import assert from "node:assert/strict";
import test from "node:test";
import { assertSweptNativeCoverage, NATIVE_UNIFORM_VEX_FIXTURE as config,
  requiredSweptSphereNativeCenters } from "../tools/implicit-density/native-uniform-vex-capture";

const geometry = { dimensions: config.dimensions, origin: config.origin_m, h: config.h_m,
  center: config.sphere.center_m, wetRadius: Math.sqrt(.25 ** 2 + .25 * .05), displacement: [.025, 0, 0] as const };
test("swept wet-cube proof contains every intermediate off-center trilinear donor", () => {
  const proof = requiredSweptSphereNativeCenters(geometry), required = new Set(proof.requiredCenterIds);
  assert.ok(proof.wetSupportIds.length > 0 && proof.requiredCenterIds.length > proof.wetSupportIds.length);
  for (const cell of proof.wetSupportIds) {
    const q = [cell % 32, Math.floor(cell / 32) % 32, Math.floor(cell / 1024)];
    for (const t of [0, .13, .47, .91, 1]) for (const local of [[0, 0, 0], [1, 1, 1], [.17, .83, .41]]) {
      const position = q.map((value, axis) => value + local[axis]! + t * geometry.displacement[axis]! / geometry.h - .5);
      const base = position.map(Math.floor);
      for (let corner = 0; corner < 8; corner++) {
        const donor = base.map((value, axis) => value + ((corner >> axis) & 1));
        const weight = position.reduce((product, value, axis) => product * (((corner >> axis) & 1)
          ? value - base[axis]! : 1 - (value - base[axis]!)), 1);
        if (weight > 0) assert.ok(required.has(donor[0]! + 32 * (donor[1]! + 32 * donor[2]!)), "uncovered interpolated donor");
      }
    }
  }
});
test("missing noncenter velocity halo rejects before a map comparison", () => {
  const proof = requiredSweptSphereNativeCenters(geometry), wet = new Set(proof.wetSupportIds);
  const halo = proof.requiredCenterIds.find(id => !wet.has(id)); assert.notEqual(halo, undefined);
  const mask = new Uint32Array(32 ** 3); for (const id of proof.requiredCenterIds) mask[id] = id + 7;
  assert.doesNotThrow(() => assertSweptNativeCoverage(mask, proof));
  mask[halo!] = 0; assert.throws(() => assertSweptNativeCoverage(mask, proof), /Missing actual VEX/);
});
test("the fixture coverage proof never clips a trajectory to the world boundary", () => {
  assert.throws(() => requiredSweptSphereNativeCenters({ ...geometry, displacement: [2, 0, 0] }), /outside the physical grid/);
});
