import assert from "node:assert/strict";
import test from "node:test";

import type { Vec3 } from "../lib/model";
import {
  sampleSvoPrimitive,
  SVO_SMOOTH_UNION_CLUSTER_NEIGHBOURHOOD,
  type SvoSmoothUnionClusterPacking,
  type SvoSmoothUnionClusterPrimitive,
} from "../lib/svo-primitive-abi";

/**
 * The two properties the procedural field family is allowed to have, checked
 * numerically because neither is visible from any other lane.
 *
 * A field that stops being a Lipschitz-1 lower bound does not throw, log or
 * fail a render — the march simply steps through the surface, and the symptom
 * is a hole in a frame nobody happened to look at. A field that stops
 * reproducing the lattice it grew out of is worse, because everything renders:
 * every aggregate already in the tree quietly re-packs.
 */

const CLUSTER_ORIGIN: Vec3 = { x: 0, y: 0, z: 0 };

function clusterAt(lobeRadii_m: Vec3, packing: SvoSmoothUnionClusterPacking): SvoSmoothUnionClusterPrimitive {
  // Centred and unrotated, so the world point the sampler takes is the local
  // point the field sees and the two implementations compare directly.
  return {
    kind: "smooth-union-cluster",
    primitiveId: 1,
    materialId: 3,
    center_m: CLUSTER_ORIGIN,
    lobeRadii_m,
    clusterReference: 1,
    packing,
  };
}

function clusterDistance_m(
  lobeRadii_m: Vec3,
  packing: SvoSmoothUnionClusterPacking,
  point_m: Vec3,
): number {
  return sampleSvoPrimitive(clusterAt(lobeRadii_m, packing), point_m).signedDistance_m;
}

/** Deterministic and self-contained, so a failure is reproducible from the seed alone. */
function randomStream(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b_79f5) >>> 0;
    let mixed = Math.imul(state ^ (state >>> 15), 1 | state);
    mixed = (mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)) ^ mixed;
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4_294_967_296;
  };
}

// ---------------------------------------------------------------------------
// The single-octave identity.
// ---------------------------------------------------------------------------

const REFERENCE_ENVELOPE: Vec3 = { x: 0.15, y: 0.09, z: 0.12 };

/**
 * The regime `docs/HERO_GARDEN_AGGREGATE_SDF_ASSESSMENT.md` §3 measured at zero
 * missed rays: 10 mm lattice lobes on a 16 mm period.
 */
const REFERENCE_LATTICE = Object.freeze({
  latticeLobeRadius_m: 0.010,
  latticePeriod_m: 0.016,
  jitter: 0.3,
  smoothRadius_m: 0.006,
  seed: 0x5701_e5,
});

/**
 * The lattice as it stood before the field family existed, transcribed rather
 * than imported.
 *
 * A reference the implementation cannot drift into: comparing the octave loop
 * against itself would pass however wrong it was. The operation order is the
 * historical one exactly — hash, jitter, the eight inner corners folded from
 * corner zero, then the outer shell rejected by its unjittered centre — which
 * is why the assertion below can be exact equality rather than a tolerance.
 */
function referenceLatticeDistance_m(point: Vec3, lobeRadii_m: Vec3): number {
  const { latticeLobeRadius_m, latticePeriod_m, jitter, smoothRadius_m, seed } = REFERENCE_LATTICE;
  const cellJitter = (cellX: number, cellY: number, cellZ: number): Vec3 => {
    let hash = seed >>> 0;
    hash = (hash ^ Math.imul(cellX | 0, 0x9e37_79b1)) >>> 0;
    hash = (hash ^ Math.imul(cellY | 0, 0x85eb_ca77)) >>> 0;
    hash = (hash ^ Math.imul(cellZ | 0, 0xc2b2_ae3d)) >>> 0;
    hash = (hash ^ (hash >>> 16)) >>> 0;
    hash = Math.imul(hash, 0x7feb_352d) >>> 0;
    hash = (hash ^ (hash >>> 15)) >>> 0;
    hash = Math.imul(hash, 0x846c_a68b) >>> 0;
    hash = (hash ^ (hash >>> 16)) >>> 0;
    return {
      x: ((hash & 0x3ff) / 1023) * 2 - 1,
      y: (((hash >>> 10) & 0x3ff) / 1023) * 2 - 1,
      z: (((hash >>> 20) & 0x3ff) / 1023) * 2 - 1,
    };
  };
  const smoothMinimum = (a: number, b: number, k: number): number => {
    if (!(k > 0)) return Math.min(a, b);
    const h = Math.min(1, Math.max(0, 0.5 + (0.5 * (b - a)) / k));
    return b + (a - b) * h - k * h * (1 - h);
  };
  const sphere = (cellX: number, cellY: number, cellZ: number): number => {
    const offset = cellJitter(cellX, cellY, cellZ);
    return Math.hypot(
      point.x - (cellX + 0.5 + jitter * offset.x) * latticePeriod_m,
      point.y - (cellY + 0.5 + jitter * offset.y) * latticePeriod_m,
      point.z - (cellZ + 0.5 + jitter * offset.z) * latticePeriod_m,
    ) - latticeLobeRadius_m;
  };

  const span = SVO_SMOOTH_UNION_CLUSTER_NEIGHBOURHOOD;
  const ring = span / 2 - 1;
  const baseX = Math.floor(point.x / latticePeriod_m - 0.5) - ring;
  const baseY = Math.floor(point.y / latticePeriod_m - 0.5) - ring;
  const baseZ = Math.floor(point.z / latticePeriod_m - 0.5) - ring;
  let distance = 0;
  for (let corner = 0; corner < 8; corner += 1) {
    const sample = sphere(baseX + ring + (corner & 1), baseY + ring + ((corner >> 1) & 1), baseZ + ring + ((corner >> 2) & 1));
    distance = corner === 0 ? sample : smoothMinimum(distance, sample, smoothRadius_m);
  }
  const reach = jitter * latticePeriod_m * Math.sqrt(3) + latticeLobeRadius_m + smoothRadius_m;
  for (let index = 0; index < span * span * span; index += 1) {
    const stepX = index % span;
    const stepY = Math.floor(index / span) % span;
    const stepZ = Math.floor(index / (span * span)) % span;
    if (stepX >= ring && stepX <= ring + 1 && stepY >= ring && stepY <= ring + 1 && stepZ >= ring && stepZ <= ring + 1) continue;
    const cellX = baseX + stepX, cellY = baseY + stepY, cellZ = baseZ + stepZ;
    const nominal = Math.hypot(
      point.x - (cellX + 0.5) * latticePeriod_m,
      point.y - (cellY + 0.5) * latticePeriod_m,
      point.z - (cellZ + 0.5) * latticePeriod_m,
    );
    if (nominal - reach > distance) continue;
    distance = smoothMinimum(distance, sphere(cellX, cellY, cellZ), smoothRadius_m);
  }
  const scaled = Math.hypot(point.x / lobeRadii_m.x, point.y / lobeRadii_m.y, point.z / lobeRadii_m.z);
  return Math.max(distance, (scaled - 1) * Math.min(lobeRadii_m.x, lobeRadii_m.y, lobeRadii_m.z));
}

test("one octave of the lattice field is bit-for-bit the field this kind has always published", () => {
  const random = randomStream(0x1234_5678);
  const reach = 1.25 * Math.max(REFERENCE_ENVELOPE.x, REFERENCE_ENVELOPE.y, REFERENCE_ENVELOPE.z);
  for (let sample = 0; sample < 3_000; sample += 1) {
    const point = {
      x: (random() * 2 - 1) * reach,
      y: (random() * 2 - 1) * reach,
      z: (random() * 2 - 1) * reach,
    };
    const expected_m = referenceLatticeDistance_m(point, REFERENCE_ENVELOPE);
    // Exact, not approximate. An octave count of one must scale every length by
    // an exact 1 and leave the seed unmixed, and a tolerance here would let a
    // 2^-0 that is not quite one — or a seed mix that is not quite the identity
    // — through while silently re-packing every aggregate in the tree.
    assert.equal(
      clusterDistance_m(REFERENCE_ENVELOPE, { ...REFERENCE_LATTICE, octaves: 1 }, point),
      expected_m,
      `octaves: 1 diverged at (${point.x}, ${point.y}, ${point.z})`,
    );
    assert.equal(
      clusterDistance_m(REFERENCE_ENVELOPE, REFERENCE_LATTICE, point),
      expected_m,
      `an absent octave count diverged at (${point.x}, ${point.y}, ${point.z})`,
    );
  }

  // And a second octave has to actually do something, or the assertion above is
  // satisfied by a loop that never runs past the first.
  const denser = clusterDistance_m(REFERENCE_ENVELOPE, { ...REFERENCE_LATTICE, octaves: 3 }, { x: 0.02, y: 0.01, z: 0.03 });
  assert.notEqual(denser, referenceLatticeDistance_m({ x: 0.02, y: 0.01, z: 0.03 }, REFERENCE_ENVELOPE));
});

// ---------------------------------------------------------------------------
// The Lipschitz bound. This is the safety gate.
// ---------------------------------------------------------------------------

interface FieldCase {
  readonly name: string;
  readonly lobeRadii_m: Vec3;
  readonly packing: SvoSmoothUnionClusterPacking;
}

const FIELD_CASES: readonly FieldCase[] = [
  {
    name: "lattice, three octaves",
    lobeRadii_m: REFERENCE_ENVELOPE,
    packing: { ...REFERENCE_LATTICE, octaves: 3 },
  },
  {
    name: "seeded lobes",
    lobeRadii_m: { x: 0.18, y: 0.12, z: 0.15 },
    packing: { field: "seeded-lobes", lobeCount: 7, anisotropy: 2.6, smoothRadius_m: 0.02, seed: 0x9a11 },
  },
  {
    name: "tapered sweep",
    lobeRadii_m: { x: 0.4, y: 0.25, z: 0.25 },
    packing: {
      field: "tapered-sweep",
      smoothRadius_m: 0.008,
      seed: 0x51b3,
      points: [
        { position_m: { x: -0.25, y: -0.10, z: 0.00 }, radius_m: 0.035 },
        { position_m: { x: -0.05, y: 0.05, z: 0.02 }, radius_m: 0.030 },
        { position_m: { x: 0.15, y: 0.02, z: -0.03 }, radius_m: 0.025 },
        { position_m: { x: 0.30, y: -0.05, z: 0.00 }, radius_m: 0.020 },
      ],
    },
  },
];

test("every field of the family steps by at most the distance it moved", () => {
  for (const { name, lobeRadii_m, packing } of FIELD_CASES) {
    const random = randomStream(0x0f1e_2d3c);
    const reach = 1.3 * Math.max(lobeRadii_m.x, lobeRadii_m.y, lobeRadii_m.z);
    let worst = 0;
    let worstAt = CLUSTER_ORIGIN;
    for (let pair = 0; pair < 2_000; pair += 1) {
      const from = {
        x: (random() * 2 - 1) * reach,
        y: (random() * 2 - 1) * reach,
        z: (random() * 2 - 1) * reach,
      };
      // Separations spanning three decades below the smallest feature. Smaller
      // than this and f64 cancellation in the difference dominates the ratio;
      // larger and a genuine local overshoot is averaged away by the smooth
      // stretches around it, which is exactly the overshoot the march dies on.
      const separation_m = 1e-4 * 10 ** (2 * random());
      const direction = { x: random() * 2 - 1, y: random() * 2 - 1, z: random() * 2 - 1 };
      const length = Math.hypot(direction.x, direction.y, direction.z) || 1;
      const to = {
        x: from.x + (direction.x / length) * separation_m,
        y: from.y + (direction.y / length) * separation_m,
        z: from.z + (direction.z / length) * separation_m,
      };
      const slope = Math.abs(clusterDistance_m(lobeRadii_m, packing, to) - clusterDistance_m(lobeRadii_m, packing, from))
        / separation_m;
      if (slope > worst) { worst = slope; worstAt = from; }
    }
    // The march advances by |d| and the acceptance band is a ten-thousandth of
    // t, so a field whose gradient exceeds one overshoots and the ray leaves
    // through the surface it was looking for. Two per cent of headroom absorbs
    // the finite-difference error at these separations and nothing else: the
    // construction this family rejects measured 60 on the same instrument.
    assert.ok(
      worst <= 1.02,
      `${name} has Lipschitz constant ${worst.toFixed(4)}, measured near (${worstAt.x}, ${worstAt.y}, ${worstAt.z})`,
    );
  }
});
