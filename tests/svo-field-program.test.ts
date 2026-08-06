import assert from "node:assert/strict";
import { test } from "node:test";

import type { Vec3 } from "../lib/model";
import {
  evaluateSvoFieldProgram,
  packSvoFieldProgramArena,
  SVO_FIELD_NOISE_JACOBIAN_BOUND,
  SVO_FIELD_OP_TABLE,
  SVO_FIELD_PROGRAM_BLOCK_WORDS,
  SVO_FIELD_PROGRAM_MAXIMUM_OPS,
  SVO_FIELD_RECOMMENDED_OCTAVE_GAIN,
  SVO_FIELD_WORLEY_JACOBIAN_BOUND,
  sampleSvoFieldWorley_m,
  sampleSvoProceduralNoise,
  svoFieldBandLimitFade,
  svoFieldProgramExtent_m,
  svoFieldProgramWGSL,
  svoFieldStep_m,
  unpackSvoFieldProgram,
  validateSvoFieldProgram,
  type SvoFieldOpDescriptor,
  type SvoFieldOpName,
  type SvoFieldOpParameters,
  type SvoFieldProgram,
} from "../lib/svo-field-program";

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

/**
 * Parameters that make each op do something, keyed by name.
 *
 * A table rather than one set of numbers for everything, because "the evaluator
 * accepts this op" is only worth asserting if the op was handed parameters it
 * can act on. A `scatter` given a sphere's radius as its cell size is a tape
 * that refuses for reasons that have nothing to do with the arm being present.
 */
const OP_FIXTURE: Record<SvoFieldOpName, Partial<SvoFieldOpParameters>> = {
  sphere: { a: 0.1 },
  ellipsoid: { a: 0.12, b: 0.09, c: 0.1 },
  box: { a: 0.1, b: 0.06, c: 0.08 },
  "domain-warp": { a: 0.02, b: 1 / 0.06 },
  fracture: { a: 0.3, b: 0.9, c: 0.2, d: 0.06, e: 0.004 },
  "smooth-union": { a: 0.5, b: 0.002, c: 0.02 },
  "smooth-subtract": { a: 0.4, b: 0.002, c: 0.02 },
  "smooth-intersect": { a: 0.4, b: 0.002, c: 0.02 },
  "anisotropic-frame": { a: 1, b: 0.7, c: 0.9, d: 0.4 },
  "worley-subtract": { a: 0.006, b: 0.0022, c: 0.0008 },
  "ridged-worley-add": { a: 0.012, b: 0.004, c: 0.001, d: 0.004 },
  "erosion-mask": { a: 0.8, b: 0.6, c: 0.004, d: 0.15 },
  scatter: { a: 0.05, b: 1, c: 0.02, d: 0.014 },
};

/**
 * A stone with every op in the language on it, in the order §2.2 puts them.
 *
 * This is the tape the Lipschitz and extent assertions run over, and it is
 * deliberately one tape rather than twelve: the failure §2.4 warns about is a
 * *composition* failure — a bound that is right for each op alone and wrong
 * once a warp sits under a frame under a scatter — and the only way to catch
 * that is to compose them.
 *
 * Fourteen ops, inside the sixteen-op machine limit, which is the evidence for
 * that limit still being the right one.
 */
const EVERY_OP: SvoFieldProgram = {
  ops: [
    // The mass: lying the way it fell, then undulating at the 40-80 mm form
    // band the plate measures over a boulder's cap.
    { op: "anisotropic-frame", out: 1, point: 0, seed: 0x51a7_0e31, parameters: { a: 1, b: 0.72, c: 0.9, d: 0.35 } },
    { op: "domain-warp", out: 2, point: 1, seed: 0x2f19_7c05, parameters: { a: 0.02, b: 1 / 0.06 } },
    // A finite cluster of gravel, folded into 5 cm cells with 2 cm of jitter.
    // The occupant radius is 12 mm against a 10 mm sphere, which leaves 3 mm of
    // clearance at the cell wall — and the union below blends at 1.25 mm, so
    // the clearance has to stay above that or the extent bound refuses the tape.
    { op: "scatter", out: 3, point: 2, seed: 0x7b3d_1e92, parameters: { a: 0.05, b: 1, c: 0.02, d: 0.012 } },
    { op: "sphere", out: 0, point: 3, parameters: { a: 0.01 } },
    { op: "ellipsoid", out: 1, point: 1, parameters: { a: 0.16, b: 0.11, c: 0.13 } },
    // The variable-k union: a 1 cm pebble against an 11 cm mass blends at the
    // pebble's radius, which is the whole point of op 4.
    { op: "smooth-union", out: 1, point: 0, fieldA: 1, fieldB: 0, parameters: { a: 0.5, b: 0.002, c: 0.02 } },
    { op: "box", out: 2, point: 1, parameters: { a: 0.15, b: 0.15, c: 0.15 } },
    { op: "smooth-intersect", out: 1, point: 0, fieldA: 1, fieldB: 2, parameters: { a: 0.4, b: 0.002, c: 0.02 } },
    { op: "sphere", out: 2, point: 2, parameters: { a: 0.05 } },
    { op: "smooth-subtract", out: 1, point: 0, fieldA: 1, fieldB: 2, parameters: { a: 0.3, b: 0.002, c: 0.01 } },
    // The fracture facet, with a tight arris.
    { op: "fracture", out: 1, point: 1, fieldA: 1, parameters: { a: 0.3, b: 0.9, c: 0.2, d: 0.07, e: 0.003 } },
    // The grain band, subtractive: 6 mm cells, 2.2 mm pits.
    { op: "worley-subtract", out: 2, point: 1, fieldA: 1, seed: 0x1c07_5aa3, parameters: { a: 0.006, b: 0.0022, c: 0.0008 } },
    // Masked, so the pitting collects on the up-facing and already-recessed
    // parts instead of reading as fabric.
    { op: "erosion-mask", out: 3, point: 1, fieldA: 1, fieldB: 2, parameters: { a: 0.8, b: 0.6, c: 0.004, d: 0.15 } },
    // And a sparser set of proud inclusions in a 4 mm shell of the surface.
    { op: "ridged-worley-add", out: 3, point: 1, fieldA: 3, seed: 0x60d2_913f, parameters: { a: 0.012, b: 0.004, c: 0.001, d: 0.004 } },
  ],
  result: 3,
};

// ---------------------------------------------------------------------------
// The op table
// ---------------------------------------------------------------------------

test("op codes, names and WGSL constants are each unique", () => {
  const codes = new Set<number>();
  const names = new Set<string>();
  const constants = new Set<string>();
  for (const op of SVO_FIELD_OP_TABLE) {
    assert.ok(!codes.has(op.code), `duplicate op code ${op.code}`);
    assert.ok(!names.has(op.name), `duplicate op name ${op.name}`);
    assert.ok(!constants.has(op.wgslConstant), `duplicate WGSL constant ${op.wgslConstant}`);
    codes.add(op.code);
    names.add(op.name);
    constants.add(op.wgslConstant);
    assert.ok(op.code > 0, "op code 0 is reserved for an unwritten word");
    assert.ok(op.buys.length > 10, `${op.name} needs a line saying what it buys`);
  }
});

/**
 * The check `tests/svo-primitive-kind-table.test.ts` makes for primitive kinds,
 * for the same reason it makes it there: a constant that is *declared and never
 * read* is the shape of a silent fallthrough, and a silent fallthrough in this
 * evaluator renders a shape nobody authored.
 */
test("every op's WGSL constant is both declared and read in the generated shader", () => {
  const wgsl = svoFieldProgramWGSL({
    functionName: "testEvaluate",
    loadWord: "loadWord",
    baseWordExpression: "0u",
    capacityExpression: "16u",
  });
  for (const op of SVO_FIELD_OP_TABLE) {
    assert.ok(wgsl.includes(`const ${op.wgslConstant}:u32=${op.code}u;`), `${op.wgslConstant} is not declared`);
    assert.ok(
      wgsl.includes(`code==${op.wgslConstant}`),
      `${op.wgslConstant} is declared but never compared against — a silent fallthrough`,
    );
  }
});

/** A minimal tape that exercises one op, over a source the op can act on. */
function singleOpProgram(name: SvoFieldOpName): SvoFieldProgram {
  const op = SVO_FIELD_OP_TABLE.find((entry) => entry.name === name)!;
  const parameters = OP_FIXTURE[name];
  if (op.result === "point") {
    return {
      ops: [
        { op: name, out: 1, point: 0, seed: 0x51a7, parameters },
        // Small enough to fit inside a scatter cell's declared occupant.
        { op: "sphere", out: 0, point: 1, parameters: { a: 0.01 } },
      ],
      result: 0,
    };
  }
  if (op.readsFields === 0) return { ops: [{ op: name, out: 0, point: 0, parameters }], result: 0 };
  if (op.readsFields === 1) {
    return {
      ops: [
        { op: "ellipsoid", out: 0, point: 0, parameters: { a: 0.12, b: 0.09, c: 0.1 } },
        { op: name, out: 1, point: 0, fieldA: 0, seed: 0x2f19, parameters },
      ],
      result: 1,
    };
  }
  return {
    ops: [
      { op: "ellipsoid", out: 0, point: 0, parameters: { a: 0.12, b: 0.09, c: 0.1 } },
      { op: "sphere", out: 1, point: 0, parameters: { a: 0.05 } },
      { op: name, out: 2, point: 0, fieldA: 0, fieldB: 1, seed: 0x2f19, parameters },
    ],
    result: 2,
  };
}

test("every op the table declares has a CPU evaluator arm", () => {
  for (const op of SVO_FIELD_OP_TABLE) {
    assert.doesNotThrow(
      () => evaluateSvoFieldProgram(singleOpProgram(op.name), { x: 0.2, y: 0.05, z: -0.1 }),
      `op ${op.name} is in the table but the evaluator refuses it`,
    );
    assert.doesNotThrow(
      () => svoFieldProgramExtent_m(singleOpProgram(op.name)),
      `op ${op.name} is in the table but the extent bound refuses it`,
    );
  }
});

test("every op the table declares has a fixture, so none is silently untested", () => {
  for (const op of SVO_FIELD_OP_TABLE) {
    assert.ok(OP_FIXTURE[op.name], `op ${op.name} has no parameter fixture`);
  }
  assert.equal(Object.keys(OP_FIXTURE).length, SVO_FIELD_OP_TABLE.length);
});

test("the worked stone uses every op in the language", () => {
  const used = new Set(EVERY_OP.ops.map((descriptor) => descriptor.op));
  for (const op of SVO_FIELD_OP_TABLE) {
    assert.ok(used.has(op.name), `the worked stone tape does not exercise ${op.name}`);
  }
  assert.ok(
    EVERY_OP.ops.length <= SVO_FIELD_PROGRAM_MAXIMUM_OPS,
    `the worked stone is ${EVERY_OP.ops.length} ops, over the machine limit`,
  );
  assert.doesNotThrow(() => validateSvoFieldProgram(EVERY_OP));
});

// ---------------------------------------------------------------------------
// Validation refuses rather than degrades
// ---------------------------------------------------------------------------

test("validateSvoFieldProgram refuses every malformed tape", () => {
  assert.throws(() => validateSvoFieldProgram({ ops: [], result: 0 }), /at least one op/);

  assert.throws(
    () =>
      validateSvoFieldProgram({
        ops: new Array(SVO_FIELD_PROGRAM_MAXIMUM_OPS + 1).fill({ op: "sphere", out: 0, parameters: { a: 0.1 } }),
        result: 0,
      }),
    /machine limit/,
  );

  assert.throws(
    () => validateSvoFieldProgram({ ops: [{ op: "sphere", out: 9, parameters: { a: 0.1 } }], result: 9 }),
    /outside 0\.\./,
  );

  assert.throws(
    () => validateSvoFieldProgram({ ops: [{ op: "sphere", out: 0, point: 2, parameters: { a: 0.1 } }], result: 0 }),
    /before anything wrote it/,
  );

  assert.throws(
    () => validateSvoFieldProgram({ ops: [{ op: "sphere", out: 0, parameters: { a: 0.1 } }], result: 3 }),
    /never written/,
  );
});

// ---------------------------------------------------------------------------
// Sources
// ---------------------------------------------------------------------------

test("the sphere source is the exact SDF", () => {
  const program: SvoFieldProgram = { ops: [{ op: "sphere", out: 0, parameters: { a: 0.25 } }], result: 0 };
  assert.ok(Math.abs(evaluateSvoFieldProgram(program, { x: 0.75, y: 0, z: 0 }).distance_m - 0.5) < 1e-6);
  assert.ok(Math.abs(evaluateSvoFieldProgram(program, { x: 0, y: 0, z: 0 }).distance_m + 0.25) < 1e-6);
});

test("the box source is the exact SDF outside and the Chebyshev depth inside", () => {
  const program: SvoFieldProgram = {
    ops: [{ op: "box", out: 0, parameters: { a: 0.2, b: 0.1, c: 0.3 } }],
    result: 0,
  };
  // Straight out of one face.
  assert.ok(Math.abs(evaluateSvoFieldProgram(program, { x: 0.5, y: 0, z: 0 }).distance_m - 0.3) < 1e-6);
  // Diagonally off a corner: the true distance is the corner offset's length.
  const corner = evaluateSvoFieldProgram(program, { x: 0.5, y: 0.5, z: 0.6 }).distance_m;
  assert.ok(Math.abs(corner - Math.hypot(0.3, 0.4, 0.3)) < 1e-6, `corner distance was ${corner}`);
  assert.ok(evaluateSvoFieldProgram(program, { x: 0, y: 0, z: 0 }).distance_m < 0);
});

test("a sphere-shaped ellipsoid reproduces the sphere exactly", () => {
  // The bound is k0(k0-1)/k1, which collapses to |p| - r when the radii agree —
  // the cheapest available check that the bound is the one intended.
  const ellipsoid: SvoFieldProgram = {
    ops: [{ op: "ellipsoid", out: 0, parameters: { a: 0.2, b: 0.2, c: 0.2 } }],
    result: 0,
  };
  for (const point of [
    { x: 0.5, y: 0, z: 0 },
    { x: 0.1, y: 0.2, z: 0.3 },
    { x: -0.4, y: 0.4, z: 0.1 },
  ]) {
    const expected = Math.hypot(point.x, point.y, point.z) - 0.2;
    const actual = evaluateSvoFieldProgram(ellipsoid, point).distance_m;
    assert.ok(Math.abs(actual - expected) < 1e-5, `at ${JSON.stringify(point)}: ${actual} vs ${expected}`);
  }
});

// ---------------------------------------------------------------------------
// The noise Jacobian bound
// ---------------------------------------------------------------------------

/**
 * The tracer divides by a constant derived from this bound. A bound that is
 * merely asserted in a comment is a decoration; if the true Jacobian exceeds
 * it, every rock in the scene grows holes at its silhouette.
 */
test("the measured noise Jacobian stays under its declared bound, and the bound is not absurd", () => {
  const step = 1e-4;
  let worst = 0;
  // A dense-enough grid at an irrational stride, so samples do not all land on
  // lattice corners where the derivative is smallest.
  for (let index = 0; index < 40000; index += 1) {
    const t = index * 0.0007;
    const point: Vec3 = {
      x: (t * 1.0) % 3.7 - 1.85,
      y: (t * 1.618) % 3.1 - 1.55,
      z: (t * 2.414) % 4.3 - 2.15,
    };
    // Frobenius norm of the 3x3 Jacobian of the signed displacement, per unit
    // of scaled domain: the quantity SVO_FIELD_NOISE_JACOBIAN_BOUND bounds.
    let frobeniusSquared = 0;
    for (const seed of [0x51ed_2701, 0x9c1e_7f3b, 0x2f6d_c5a9]) {
      for (const axis of ["x", "y", "z"] as const) {
        const forward = { ...point, [axis]: point[axis] + step };
        const backward = { ...point, [axis]: point[axis] - step };
        const derivative =
          (2 * sampleSvoProceduralNoise(forward, [1, 1, 1], seed) -
            2 * sampleSvoProceduralNoise(backward, [1, 1, 1], seed)) /
          (2 * step);
        frobeniusSquared += derivative * derivative;
      }
    }
    worst = Math.max(worst, Math.sqrt(frobeniusSquared));
  }
  assert.ok(
    worst <= SVO_FIELD_NOISE_JACOBIAN_BOUND,
    `measured Jacobian ${worst.toFixed(4)} exceeds the declared bound ${SVO_FIELD_NOISE_JACOBIAN_BOUND.toFixed(4)}`,
  );
  assert.ok(
    worst > 0.1 * SVO_FIELD_NOISE_JACOBIAN_BOUND,
    `measured Jacobian ${worst.toFixed(4)} is so far under the bound ${SVO_FIELD_NOISE_JACOBIAN_BOUND.toFixed(4)} ` +
      "that the bound would throttle the tracer for nothing",
  );
});

// ---------------------------------------------------------------------------
// Lipschitz through the graph — §2.4's debug mode, as a test
// ---------------------------------------------------------------------------

function finiteDifferenceGradient(program: SvoFieldProgram, point: Vec3, step: number): number {
  const at = (offset: Vec3) => evaluateSvoFieldProgram(program, offset).distance_m;
  const dx = (at({ ...point, x: point.x + step }) - at({ ...point, x: point.x - step })) / (2 * step);
  const dy = (at({ ...point, y: point.y + step }) - at({ ...point, y: point.y - step })) / (2 * step);
  const dz = (at({ ...point, z: point.z + step }) - at({ ...point, z: point.z - step })) / (2 * step);
  return Math.hypot(dx, dy, dz);
}

/**
 * The claim the sphere trace depends on: `|∇d| ≤ lipschitz` everywhere the
 * tracer can step. Checked by finite difference over a sample grid, which is
 * exactly the debug mode §2.4 specifies.
 *
 * Two warps at different scales, over an ellipsoid — the handoff's own worked
 * example of what turns a sphere into a rock ("≈15 cm for the mass, ≈2 cm for
 * the lobing").
 */
test("a two-level warped ellipsoid never exceeds its declared Lipschitz constant", () => {
  const program: SvoFieldProgram = {
    ops: [
      { op: "domain-warp", out: 1, point: 0, seed: 0x51a7, parameters: { a: 0.045, b: 1 / 0.15 } },
      { op: "domain-warp", out: 2, point: 1, seed: 0x2f19, parameters: { a: 0.008, b: 1 / 0.02 } },
      { op: "ellipsoid", out: 0, point: 2, parameters: { a: 0.16, b: 0.11, c: 0.13 } },
    ],
    result: 0,
  };
  const declared = evaluateSvoFieldProgram(program, { x: 0.3, y: 0.2, z: 0.1 }).lipschitz;
  assert.ok(declared > 1, `a warped field must declare a Lipschitz above 1, got ${declared}`);

  let worst = 0;
  let worstAt: Vec3 = { x: 0, y: 0, z: 0 };
  const step = 2e-4;
  for (let ix = 0; ix < 26; ix += 1) {
    for (let iy = 0; iy < 26; iy += 1) {
      for (let iz = 0; iz < 26; iz += 1) {
        const point: Vec3 = {
          x: -0.36 + (ix * 0.72) / 25,
          y: -0.36 + (iy * 0.72) / 25,
          z: -0.36 + (iz * 0.72) / 25,
        };
        // The ellipsoid bound is undefined at the centre (k1 -> 0), which is
        // deep inside the solid and unreachable by a trace from outside.
        if (Math.hypot(point.x, point.y, point.z) < 0.05) continue;
        const gradient = finiteDifferenceGradient(program, point, step);
        if (gradient > worst) {
          worst = gradient;
          worstAt = point;
        }
      }
    }
  }
  assert.ok(
    worst <= declared,
    `|grad d| reached ${worst.toFixed(4)} against a declared Lipschitz of ${declared.toFixed(4)} ` +
      `at ${JSON.stringify(worstAt)} — the tracer would overshoot and punch holes here`,
  );
});

// ---------------------------------------------------------------------------
// Lipschitz, op by op — the assertion the whole op set stands on
// ---------------------------------------------------------------------------

interface GradientSurvey {
  readonly worst: number;
  readonly worstAt: Vec3;
  readonly declared: number;
}

/**
 * The worst finite-difference gradient over a grid, against the tape's own
 * declared constant.
 *
 * Sampled at an irrational stride within the box so that the samples do not all
 * land on the lattice corners of the noise, the Worley cells or the scatter's
 * cell walls — those are exactly the places a bound fails, and a grid that
 * lands on them symmetrically would step over the failure.
 */
function surveyGradient(
  program: SvoFieldProgram,
  reach_m: number,
  options: { readonly steps?: number; readonly step_m?: number; readonly skipRadius_m?: number } = {},
): GradientSurvey {
  const steps = options.steps ?? 22;
  const step_m = options.step_m ?? 2e-4;
  const skipRadius_m = options.skipRadius_m ?? 0.02;
  const declared = evaluateSvoFieldProgram(program, { x: reach_m * 0.61, y: reach_m * 0.37, z: reach_m * 0.23 })
    .lipschitz;
  const at = (point: Vec3) => evaluateSvoFieldProgram(program, point).distance_m;
  let worst = 0;
  let worstAt: Vec3 = { x: 0, y: 0, z: 0 };
  for (let ix = 0; ix < steps; ix += 1) {
    for (let iy = 0; iy < steps; iy += 1) {
      for (let iz = 0; iz < steps; iz += 1) {
        const point: Vec3 = {
          x: -reach_m + (2 * reach_m * (ix + 0.317)) / steps,
          y: -reach_m + (2 * reach_m * (iy + 0.618)) / steps,
          z: -reach_m + (2 * reach_m * (iz + 0.141)) / steps,
        };
        // Every source's bound is undefined at its own centre — the ellipsoid's
        // gradient normalisation divides by zero there — and that point is deep
        // inside the solid where no trace from outside can reach.
        if (Math.hypot(point.x, point.y, point.z) < skipRadius_m) continue;
        // And the claim is about the region a sphere trace steps through, which
        // is the exterior and the surface. Inside the solid only the *sign* is
        // read, by the occupancy test; the ellipsoid's gradient-normalised
        // bound is steeper than one in there and always has been, and holding
        // it to a constant it does not need would either fail honestly or force
        // every ellipsoid in the scene to march at that interior rate.
        if (at(point) < 0) continue;
        const dx = (at({ ...point, x: point.x + step_m }) - at({ ...point, x: point.x - step_m })) / (2 * step_m);
        const dy = (at({ ...point, y: point.y + step_m }) - at({ ...point, y: point.y - step_m })) / (2 * step_m);
        const dz = (at({ ...point, z: point.z + step_m }) - at({ ...point, z: point.z - step_m })) / (2 * step_m);
        const gradient = Math.hypot(dx, dy, dz);
        if (gradient > worst) {
          worst = gradient;
          worstAt = point;
        }
      }
    }
  }
  return { worst, worstAt, declared };
}

/**
 * Every op, alone, against its own declared bound.
 *
 * §2.4 names this as the constraint that kills naive versions of the design,
 * and the reason is worth restating at the assertion: a tape that overstates
 * its clearance makes the tracer step past the surface, and the result is not a
 * subtle shading error but a hole, worst at the silhouette. So the bound is
 * measured rather than asserted, for each op on its own — a composed tape can
 * pass on a bound that is generous somewhere else.
 *
 * The `looseness` ceiling is the other half. A bound that is a hundred times
 * the truth is technically safe and practically a refusal to march: the
 * voxelizer's step is `distance / lipschitz`, so an idle factor of ten is an
 * order of magnitude more samples per stone.
 */
const LIPSCHITZ_CASES: readonly {
  readonly op: SvoFieldOpName;
  readonly reach_m: number;
  readonly looseness: number;
  readonly step_m?: number;
}[] = [
  { op: "sphere", reach_m: 0.2, looseness: 1.05 },
  { op: "ellipsoid", reach_m: 0.2, looseness: 1.05 },
  { op: "box", reach_m: 0.2, looseness: 1.05 },
  { op: "domain-warp", reach_m: 0.12, looseness: 40 },
  { op: "fracture", reach_m: 0.2, looseness: 1.05 },
  { op: "smooth-union", reach_m: 0.2, looseness: 1.05 },
  { op: "smooth-subtract", reach_m: 0.2, looseness: 1.05 },
  { op: "smooth-intersect", reach_m: 0.2, looseness: 1.05 },
  { op: "anisotropic-frame", reach_m: 0.1, looseness: 1.6 },
  { op: "worley-subtract", reach_m: 0.2, looseness: 1.05, step_m: 5e-5 },
  { op: "ridged-worley-add", reach_m: 0.2, looseness: 1.05, step_m: 5e-5 },
  { op: "erosion-mask", reach_m: 0.2, looseness: 30 },
  { op: "scatter", reach_m: 0.12, looseness: 1.05, step_m: 5e-5 },
];

for (const testCase of LIPSCHITZ_CASES) {
  test(`${testCase.op} never exceeds the Lipschitz constant it declares`, () => {
    const program = singleOpProgram(testCase.op);
    const survey = surveyGradient(program, testCase.reach_m, {
      step_m: testCase.step_m,
      // A scatter's occupant is a 1 cm sphere, so the usual 2 cm skip would
      // swallow the instance at the origin along with the singularity.
      skipRadius_m: testCase.op === "scatter" ? 0.002 : 0.02,
    });
    assert.ok(
      // The relative slack is finite-difference truncation, not tolerance for a
      // wrong bound: a central difference across a kink in a piecewise-linear
      // field lands a few parts per million either side of the true one-sided
      // slope. Anything a bound is actually short by shows up orders of
      // magnitude above this.
      survey.worst <= survey.declared * (1 + 1e-4),
      `${testCase.op}: |grad d| reached ${survey.worst.toFixed(4)} against a declared ` +
        `${survey.declared.toFixed(4)} at ${JSON.stringify(survey.worstAt)} — the tracer overshoots and punches ` +
        "holes here",
    );
    assert.ok(
      survey.declared <= testCase.looseness * Math.max(survey.worst, 1),
      `${testCase.op}: declares ${survey.declared.toFixed(4)} but never exceeds ${survey.worst.toFixed(4)}. A bound ` +
        "that idle throttles every march step through this op for detail that is not there",
    );
  });
}

/**
 * And the composition, which is where a per-op bound that is individually right
 * goes wrong: a warp under a frame under a scatter multiplies three factors
 * that were each derived on their own.
 */
test("the worked stone — every op at once — never exceeds its declared Lipschitz constant", () => {
  const extent = svoFieldProgramExtent_m(EVERY_OP);
  const survey = surveyGradient(EVERY_OP, Math.max(...extent) * 1.1, { steps: 20, step_m: 5e-5, skipRadius_m: 0.01 });
  assert.ok(survey.declared > 1, `a dressed stone must declare a Lipschitz above 1, got ${survey.declared}`);
  assert.ok(Number.isFinite(survey.declared), "the declared Lipschitz must be finite");
  assert.ok(
    survey.worst <= survey.declared * (1 + 1e-4),
    `|grad d| reached ${survey.worst.toFixed(4)} against a declared ${survey.declared.toFixed(4)} at ` +
      `${JSON.stringify(survey.worstAt)} — a composition failure, which is the one this test exists for`,
  );
});

// ---------------------------------------------------------------------------
// The Worley bound, measured — the tracer divides by it
// ---------------------------------------------------------------------------

/**
 * The cellular distance is claimed to be exactly 1-Lipschitz, which is a
 * stronger claim than it looks: the truncated eight-cell search is only
 * continuous because of the half-cell clamp, and a version without that clamp
 * would jump at lattice boundaries. A jump has no bound at all.
 *
 * Measured with a step far finer than the cell so that a sample straddling a
 * boundary reports the jump divided by the step — which is to say, enormous —
 * rather than averaging it away.
 */
test("the measured Worley Jacobian stays under its declared bound, including across cell walls", () => {
  const cell_m = 0.02;
  const step_m = 1e-5;
  const seed = 0x1c07_5aa3;
  let worst = 0;
  let worstAt: Vec3 = { x: 0, y: 0, z: 0 };
  for (let index = 0; index < 30000; index += 1) {
    const t = index * 0.00037;
    const point: Vec3 = {
      x: (t % 0.31) - 0.155,
      y: ((t * 1.618) % 0.29) - 0.145,
      z: ((t * 2.414) % 0.37) - 0.185,
    };
    const at = (offset: Vec3) => sampleSvoFieldWorley_m(offset, cell_m, seed);
    const dx = (at({ ...point, x: point.x + step_m }) - at({ ...point, x: point.x - step_m })) / (2 * step_m);
    const dy = (at({ ...point, y: point.y + step_m }) - at({ ...point, y: point.y - step_m })) / (2 * step_m);
    const dz = (at({ ...point, z: point.z + step_m }) - at({ ...point, z: point.z - step_m })) / (2 * step_m);
    const gradient = Math.hypot(dx, dy, dz);
    if (gradient > worst) {
      worst = gradient;
      worstAt = point;
    }
  }
  assert.ok(
    worst <= SVO_FIELD_WORLEY_JACOBIAN_BOUND + 1e-3,
    `measured Worley Jacobian ${worst.toFixed(4)} exceeds the declared bound ${SVO_FIELD_WORLEY_JACOBIAN_BOUND} at ` +
      `${JSON.stringify(worstAt)}. If this is far above one, the eight-cell search has a discontinuity and the ` +
      "half-cell clamp is not doing its job",
  );
  assert.ok(
    worst > 0.5,
    `measured Worley Jacobian ${worst.toFixed(4)} is far under one, which means the field is flat and the ` +
      "sampler is not wired up",
  );
});

test("the Worley distance is clamped at half a cell, which is what makes the truncated search exact", () => {
  const cell_m = 0.02;
  let maximum = 0;
  for (let index = 0; index < 4000; index += 1) {
    const t = index * 0.0013;
    const point: Vec3 = { x: (t % 0.5) - 0.25, y: ((t * 1.618) % 0.5) - 0.25, z: ((t * 2.414) % 0.5) - 0.25 };
    maximum = Math.max(maximum, sampleSvoFieldWorley_m(point, cell_m, 0x99));
  }
  assert.ok(
    maximum <= 0.5 * cell_m + 1e-9,
    `the cellular distance reached ${maximum} m, past the ${0.5 * cell_m} m clamp the continuity argument needs`,
  );
  assert.ok(maximum > 0.3 * cell_m, "the clamp should be reached, or the lattice is far denser than authored");
});

// ---------------------------------------------------------------------------
// The band limit
// ---------------------------------------------------------------------------

test("the band-limit fade is one above four samples per period, zero at two, and monotone between", () => {
  const sampling_m = 0.025;
  assert.equal(svoFieldBandLimitFade(0.2, sampling_m), 1, "a 200 mm form on a 25 mm grid is fully resolved");
  assert.equal(svoFieldBandLimitFade(0.003, sampling_m), 0, "a 3 mm grain on a 25 mm grid is pure alias");
  assert.equal(svoFieldBandLimitFade(0.05, sampling_m), 0, "exactly at Nyquist the fade has reached zero");
  assert.equal(svoFieldBandLimitFade(0.1, sampling_m), 1, "one octave above Nyquist the fade is complete");
  let previous = -1;
  for (let index = 0; index <= 40; index += 1) {
    const period_m = 0.05 + (index * 0.05) / 40;
    const fade = svoFieldBandLimitFade(period_m, sampling_m);
    assert.ok(fade >= previous - 1e-9, "the fade must not go backwards as the feature gets coarser");
    previous = fade;
  }
  // Zero means "do not band-limit", which is what the extent bound and the CPU
  // oracles need: they must see the finest form of the shape.
  assert.equal(svoFieldBandLimitFade(0.003, 0), 1);
});

/**
 * The half that makes the approach affordable rather than merely correct.
 *
 * A faded-out octave that still paid its worst-case bound would make coarse
 * voxelization march at fine-detail cost for detail it cannot show. So the fade
 * has to reach the Lipschitz constant, not only the amplitude.
 */
test("a band-limited octave costs nothing in amplitude AND nothing in the Lipschitz bound", () => {
  // A 3 mm grain over a 12 cm mass: resolvable on a 1 mm grid, pure alias at 25 mm.
  const program: SvoFieldProgram = {
    ops: [
      { op: "domain-warp", out: 1, point: 0, seed: 0x51a7, parameters: { a: 0.0012, b: 1 / 0.003 } },
      { op: "sphere", out: 0, point: 1, parameters: { a: 0.12 } },
      { op: "worley-subtract", out: 1, point: 0, fieldA: 0, seed: 0x77, parameters: { a: 0.003, b: 0.0011, c: 0.0004 } },
    ],
    result: 1,
  };
  const probe: Vec3 = { x: 0.1, y: 0.04, z: -0.03 };
  const fine = evaluateSvoFieldProgram(program, probe, 0.0005);
  const coarse = evaluateSvoFieldProgram(program, probe, 0.025);
  const unbanded = evaluateSvoFieldProgram(program, probe, 0);

  assert.ok(fine.lipschitz > 1.5, `a resolved 3 mm grain should cost real march steps, got ${fine.lipschitz}`);
  assert.ok(
    Math.abs(coarse.lipschitz - 1) < 1e-6,
    `a faded-out grain must return the bound to that of the bare mass, got ${coarse.lipschitz}`,
  );
  assert.ok(
    Math.abs(fine.lipschitz - unbanded.lipschitz) < 1e-6,
    "a sampling length below the feature must be identical to no band limit at all",
  );

  // And the shape: the coarse evaluation is the smooth sphere, exactly.
  const bare: SvoFieldProgram = { ops: [{ op: "sphere", out: 0, parameters: { a: 0.12 } }], result: 0 };
  assert.ok(
    Math.abs(coarse.distance_m - evaluateSvoFieldProgram(bare, probe).distance_m) < 1e-6,
    "a fully faded tape must evaluate to its unmodified source",
  );
  assert.ok(
    Math.abs(fine.distance_m - evaluateSvoFieldProgram(bare, probe).distance_m) > 1e-5,
    "a fully resolved tape must not evaluate to its unmodified source, or nothing is wired up",
  );
});

/**
 * With lacunarity 2, the product of octave factors converges exactly when the
 * gain is under a half. This is not a style note: at a half, march cost through
 * the stone grows without limit in octave count.
 */
test("the recommended octave gain makes each further octave cost less, and one half does not", () => {
  /** A warp chain of `octaves` levels at lacunarity 2, over a sphere. */
  const chain = (gain: number, octaves: number): SvoFieldProgram => {
    const ops: SvoFieldOpDescriptor[] = [];
    for (let index = 0; index < octaves; index += 1) {
      // Three point registers are enough for a chain of any depth: after the
      // third, each octave overwrites the register it reads.
      ops.push({
        op: "domain-warp",
        out: Math.min(index + 1, 3),
        point: Math.min(index, 3),
        seed: 0x51a7 + index,
        parameters: { a: 0.04 * gain ** index, b: (1 / 0.12) * 2 ** index },
      });
    }
    ops.push({ op: "sphere", out: 0, point: Math.min(octaves, 3), parameters: { a: 0.1 } });
    return { ops, result: 0 };
  };
  const probe: Vec3 = { x: 0.2, y: 0.1, z: 0.05 };
  const bound = (gain: number, octaves: number) => evaluateSvoFieldProgram(chain(gain, octaves), probe).lipschitz;
  /** What the `n`th octave multiplies the bound by. */
  const octaveFactor = (gain: number, octave: number) => bound(gain, octave + 1) / bound(gain, octave);

  assert.ok(SVO_FIELD_RECOMMENDED_OCTAVE_GAIN < 0.5, "the recommended gain must be under the convergence threshold");

  // At exactly one half, amplitude halves as frequency doubles, so `A·f` — and
  // therefore the factor each octave contributes — is the *same every time*.
  // The bound, and every march step through the stone, then grows geometrically
  // in octave count for detail that is by construction getting less visible.
  const flatEarly = octaveFactor(0.5, 4);
  const flatLate = octaveFactor(0.5, 11);
  assert.ok(
    Math.abs(flatLate - flatEarly) < 1e-3,
    `at gain 0.5 the eleventh octave costs ${flatLate.toFixed(4)} against the fourth's ${flatEarly.toFixed(4)}; ` +
      "they should be identical, which is exactly the problem",
  );
  assert.ok(flatEarly > 2, "the fixture's warp should be strong enough for the difference to be visible");

  // Under one half the factors decay geometrically, so the infinite product is
  // bounded: a chain of any depth costs within a constant of a shallow one.
  const early = octaveFactor(SVO_FIELD_RECOMMENDED_OCTAVE_GAIN, 4);
  const late = octaveFactor(SVO_FIELD_RECOMMENDED_OCTAVE_GAIN, 11);
  assert.ok(
    late < early,
    `at gain ${SVO_FIELD_RECOMMENDED_OCTAVE_GAIN} the eleventh octave costs ${late.toFixed(4)} against the ` +
      `fourth's ${early.toFixed(4)} — the series is not decaying`,
  );
  // Geometric decay, stated as such: the *excess* over one is what sums, and at
  // lacunarity 2 it falls by `2·gain` per octave, so seven octaves later it
  // should be down by `0.84⁷ ≈ 0.29`. Asserting the excess rather than the
  // factor is what makes this a statement about convergence rather than about
  // whether this particular fixture's warp happened to be gentle.
  assert.ok(
    late - 1 < 0.4 * (early - 1),
    `at gain ${SVO_FIELD_RECOMMENDED_OCTAVE_GAIN} the eleventh octave's excess is ${(late - 1).toFixed(4)} against ` +
      `the fourth's ${(early - 1).toFixed(4)}; seven octaves of decay at 2·gain should have cut it to under 30%`,
  );
  // And the tail: four more octaves past eight cost far less here than at a half.
  const convergingTail = bound(SVO_FIELD_RECOMMENDED_OCTAVE_GAIN, 12) / bound(SVO_FIELD_RECOMMENDED_OCTAVE_GAIN, 8);
  const divergingTail = bound(0.5, 12) / bound(0.5, 8);
  assert.ok(
    divergingTail > 4 * convergingTail,
    `the tail cost of four further octaves is ${convergingTail.toFixed(2)}x at the recommended gain and ` +
      `${divergingTail.toFixed(2)}x at one half; the gap is the whole reason for the recommendation`,
  );
});

test("svoFieldStep_m never returns more than the distance itself", () => {
  assert.equal(svoFieldStep_m({ distance_m: 0.5, lipschitz: 1 }), 0.5);
  assert.equal(svoFieldStep_m({ distance_m: 0.5, lipschitz: 2 }), 0.25);
  // A degenerate sub-unit Lipschitz must never *grow* the step.
  assert.equal(svoFieldStep_m({ distance_m: 0.5, lipschitz: 0.25 }), 0.5);
});

// ---------------------------------------------------------------------------
// The warp does something
// ---------------------------------------------------------------------------

test("a domain warp measurably moves the surface it wraps", () => {
  const radii = { a: 0.16, b: 0.11, c: 0.13 };
  const plain: SvoFieldProgram = { ops: [{ op: "ellipsoid", out: 0, parameters: radii }], result: 0 };
  const warped: SvoFieldProgram = {
    ops: [
      { op: "domain-warp", out: 1, point: 0, seed: 0x51a7, parameters: { a: 0.045, b: 1 / 0.15 } },
      { op: "ellipsoid", out: 0, point: 1, parameters: radii },
    ],
    result: 0,
  };
  let maximumShift = 0;
  let samples = 0;
  for (let index = 0; index < 4000; index += 1) {
    const t = index * 0.0011;
    const point: Vec3 = { x: (t % 0.6) - 0.3, y: ((t * 1.618) % 0.6) - 0.3, z: ((t * 2.414) % 0.6) - 0.3 };
    const difference = Math.abs(
      evaluateSvoFieldProgram(warped, point).distance_m - evaluateSvoFieldProgram(plain, point).distance_m,
    );
    maximumShift = Math.max(maximumShift, difference);
    samples += 1;
  }
  assert.ok(samples > 0);
  // A warp of 45 mm amplitude that moved the field by under a centimetre
  // anywhere would be a warp that is not wired up.
  assert.ok(maximumShift > 0.01, `the warp shifted the field by at most ${maximumShift.toFixed(5)} m`);
});

// ---------------------------------------------------------------------------
// Extent
// ---------------------------------------------------------------------------

/**
 * The proxy box drives brick binning and the march bracket. A box that clipped
 * the warped surface would delete exactly the silhouette detail the warp exists
 * to create — silently.
 */
test("the conservative extent contains every point the program calls solid", () => {
  const program: SvoFieldProgram = {
    ops: [
      { op: "domain-warp", out: 1, point: 0, seed: 0x9c3, parameters: { a: 0.05, b: 1 / 0.12 } },
      { op: "ellipsoid", out: 0, point: 1, parameters: { a: 0.16, b: 0.11, c: 0.13 } },
    ],
    result: 0,
  };
  const extent = svoFieldProgramExtent_m(program);
  assert.ok(extent[0] >= 0.16 && extent[1] >= 0.11 && extent[2] >= 0.13, "extent must cover the unwarped solid");

  const reach = Math.max(...extent) * 1.6;
  let escapes = 0;
  for (let ix = 0; ix < 40; ix += 1) {
    for (let iy = 0; iy < 40; iy += 1) {
      for (let iz = 0; iz < 40; iz += 1) {
        const point: Vec3 = {
          x: -reach + (ix * 2 * reach) / 39,
          y: -reach + (iy * 2 * reach) / 39,
          z: -reach + (iz * 2 * reach) / 39,
        };
        if (evaluateSvoFieldProgram(program, point).distance_m > 0) continue;
        if (
          Math.abs(point.x) > extent[0] + 1e-6 ||
          Math.abs(point.y) > extent[1] + 1e-6 ||
          Math.abs(point.z) > extent[2] + 1e-6
        ) {
          escapes += 1;
        }
      }
    }
  }
  assert.equal(escapes, 0, `${escapes} solid samples fell outside the conservative extent`);
});

/**
 * The same claim over the tape that uses every op, which is where it gets hard:
 * a scatter replicates the solid, a frame rotates it so per-axis accounting
 * stops meaning anything, and `ridged-worley-add` is the one op in the language
 * that puts material *outside* what it was given.
 */
test("the conservative extent contains the worked stone, every op included", () => {
  const extent = svoFieldProgramExtent_m(EVERY_OP);
  assert.ok(extent.every((half) => half > 0.16), `the extent must cover the 160 mm mass, got ${JSON.stringify(extent)}`);
  assert.ok(extent.every((half) => half < 1), `the extent must not have run away, got ${JSON.stringify(extent)}`);

  const reach = Math.max(...extent) * 1.5;
  const steps = 34;
  let escapes = 0;
  let worstEscape: Vec3 = { x: 0, y: 0, z: 0 };
  for (let ix = 0; ix < steps; ix += 1) {
    for (let iy = 0; iy < steps; iy += 1) {
      for (let iz = 0; iz < steps; iz += 1) {
        const point: Vec3 = {
          x: -reach + (2 * reach * (ix + 0.317)) / steps,
          y: -reach + (2 * reach * (iy + 0.618)) / steps,
          z: -reach + (2 * reach * (iz + 0.141)) / steps,
        };
        if (evaluateSvoFieldProgram(EVERY_OP, point).distance_m > 0) continue;
        if (
          Math.abs(point.x) > extent[0] + 1e-6 ||
          Math.abs(point.y) > extent[1] + 1e-6 ||
          Math.abs(point.z) > extent[2] + 1e-6
        ) {
          escapes += 1;
          worstEscape = point;
        }
      }
    }
  }
  assert.equal(
    escapes,
    0,
    `${escapes} solid samples fell outside the conservative extent, e.g. ${JSON.stringify(worstEscape)} against ` +
      `${JSON.stringify(extent)}`,
  );
});

test("the extent is the unbanded shape, so a coarser voxelization can never outgrow it", () => {
  // A band limit shrinks the field. The proxy box is authored once and has to
  // hold at every resolution the record will be sampled at, so it must be the
  // largest of them — which is the one with nothing faded out.
  const extent = svoFieldProgramExtent_m(EVERY_OP);
  const reach = Math.max(...extent);
  for (const samplingLength_m of [0, 0.0005, 0.006, 0.025, 0.2]) {
    for (let index = 0; index < 900; index += 1) {
      const t = index * 0.0071;
      const point: Vec3 = {
        x: ((t % (2 * reach)) - reach) * 1.4,
        y: (((t * 1.618) % (2 * reach)) - reach) * 1.4,
        z: (((t * 2.414) % (2 * reach)) - reach) * 1.4,
      };
      if (evaluateSvoFieldProgram(EVERY_OP, point, samplingLength_m).distance_m > 0) continue;
      assert.ok(
        Math.abs(point.x) <= extent[0] + 1e-6 &&
          Math.abs(point.y) <= extent[1] + 1e-6 &&
          Math.abs(point.z) <= extent[2] + 1e-6,
        `at a ${samplingLength_m} m sampling length a solid sample escaped the extent at ${JSON.stringify(point)}`,
      );
    }
  }
});

// ---------------------------------------------------------------------------
// What each op actually does
// ---------------------------------------------------------------------------

test("fracture cuts a flat face with a tight arris", () => {
  const offset_m = 0.06;
  const program: SvoFieldProgram = {
    ops: [
      { op: "sphere", out: 0, parameters: { a: 0.1 } },
      // A halfspace on +y, cutting the top off at 60 mm with a 2 mm arris.
      { op: "fracture", out: 1, point: 0, fieldA: 0, parameters: { a: 0, b: 1, c: 0, d: offset_m, e: 0.002 } },
    ],
    result: 1,
  };
  // Straight up, past the plane but inside the sphere: removed.
  assert.ok(evaluateSvoFieldProgram(program, { x: 0, y: 0.08, z: 0 }).distance_m > 0, "the cut must remove material");
  // Below the plane and inside the sphere: kept.
  assert.ok(evaluateSvoFieldProgram(program, { x: 0, y: 0.02, z: 0 }).distance_m < 0, "the cut must keep the rest");

  // The face is flat: the surface height over the cut is the plane's offset,
  // everywhere it is the plane and not the sphere that is nearest.
  for (const [x, z] of [
    [0, 0],
    [0.03, 0.01],
    [-0.02, 0.04],
  ] as const) {
    const above = evaluateSvoFieldProgram(program, { x, y: offset_m + 0.01, z }).distance_m;
    assert.ok(
      Math.abs(above - 0.01) < 2e-3,
      `over the facet at (${x}, ${z}) the field read ${above.toFixed(5)} where a flat face at ${offset_m} m gives 0.01`,
    );
  }
  // And the arris is tight: a 2 mm blend rounds the edge by no more than 2 mm.
  const plain: SvoFieldProgram = { ops: [{ op: "sphere", out: 0, parameters: { a: 0.1 } }], result: 0 };
  assert.ok(
    svoFieldProgramExtent_m(program)[1] <= svoFieldProgramExtent_m(plain)[1],
    "a cut can only shrink the solid, so it can only shrink the proxy box",
  );
});

test("smooth-union blends at the smaller operand's radius, not at one radius for the whole aggregate", () => {
  // The comparison §2.2 op 4 is about: the same authored blend *fraction*
  // against a 20 cm feature and a 2 cm one. A fixed radius fillets both
  // identically, which dissolves the small one and barely marks the large one.
  //
  // A sphere against a slab, both with a face at x = 200 mm, so the two
  // operands are exactly equal at the probe and the blend shows as the dip
  // below the hard minimum — which the smooth-min form makes exactly k/4.
  const aggregate = (minorHalf_m: number): SvoFieldProgram => ({
    ops: [
      { op: "sphere", out: 0, parameters: { a: 0.2 } },
      { op: "box", out: 1, parameters: { a: 0.2, b: minorHalf_m, c: 0.2 } },
      { op: "smooth-union", out: 2, fieldA: 0, fieldB: 1, parameters: { a: 0.5, b: 0, c: 1 } },
    ],
    result: 2,
  });
  const probe: Vec3 = { x: 0.2, y: 0, z: 0 };
  const bigLobe = -evaluateSvoFieldProgram(aggregate(0.2), probe).distance_m;
  const smallLobe = -evaluateSvoFieldProgram(aggregate(0.02), probe).distance_m;
  assert.ok(
    Math.abs(bigLobe - (0.5 * 0.2) / 4) < 1e-6,
    `a 200 mm feature should blend at k = 100 mm and dip 25 mm, dipped ${bigLobe}`,
  );
  assert.ok(
    Math.abs(smallLobe - (0.5 * 0.02) / 4) < 1e-6,
    `a 20 mm feature should blend at k = 10 mm and dip 2.5 mm, dipped ${smallLobe}`,
  );
  assert.ok(bigLobe > 9 * smallLobe, "the two must not blend identically — that identity is the blobbiness");
});

test("smooth-subtract and smooth-intersect only ever raise the field, so the solid only shrinks", () => {
  const base: SvoFieldProgram = { ops: [{ op: "ellipsoid", out: 0, parameters: { a: 0.16, b: 0.11, c: 0.13 } }], result: 0 };
  const subtracted: SvoFieldProgram = {
    ops: [
      { op: "ellipsoid", out: 0, parameters: { a: 0.16, b: 0.11, c: 0.13 } },
      { op: "sphere", out: 1, parameters: { a: 0.06 } },
      { op: "smooth-subtract", out: 2, fieldA: 0, fieldB: 1, parameters: { a: 0.3, b: 0.002, c: 0.02 } },
    ],
    result: 2,
  };
  const intersected: SvoFieldProgram = {
    ops: [
      { op: "ellipsoid", out: 0, parameters: { a: 0.16, b: 0.11, c: 0.13 } },
      { op: "box", out: 1, parameters: { a: 0.1, b: 0.1, c: 0.1 } },
      { op: "smooth-intersect", out: 2, fieldA: 0, fieldB: 1, parameters: { a: 0.3, b: 0.002, c: 0.02 } },
    ],
    result: 2,
  };
  let subtractedSomething = false;
  let intersectedSomething = false;
  for (let index = 0; index < 3000; index += 1) {
    const t = index * 0.0017;
    const point: Vec3 = { x: (t % 0.4) - 0.2, y: ((t * 1.618) % 0.4) - 0.2, z: ((t * 2.414) % 0.4) - 0.2 };
    const plain = evaluateSvoFieldProgram(base, point).distance_m;
    const minus = evaluateSvoFieldProgram(subtracted, point).distance_m;
    const meet = evaluateSvoFieldProgram(intersected, point).distance_m;
    assert.ok(minus >= plain - 1e-9, `subtraction lowered the field at ${JSON.stringify(point)}`);
    assert.ok(meet >= plain - 1e-9, `intersection lowered the field at ${JSON.stringify(point)}`);
    if (minus > plain + 1e-4) subtractedSomething = true;
    if (meet > plain + 1e-4) intersectedSomething = true;
  }
  assert.ok(subtractedSomething, "the subtraction removed nothing at all");
  assert.ok(intersectedSomething, "the intersection removed nothing at all");
});

test("anisotropic-frame flattens a sphere into a bedded slab and declares 1/min(scale)", () => {
  const flatten = 0.4;
  const program: SvoFieldProgram = {
    ops: [
      // No rotation, so the axes are the authored ones and the shape is checkable.
      { op: "anisotropic-frame", out: 1, point: 0, seed: 0x51, parameters: { a: 1, b: flatten, c: 1, d: 0 } },
      { op: "sphere", out: 0, point: 1, parameters: { a: 0.1 } },
    ],
    result: 0,
  };
  // The solid reaches its full radius across and `flatten` of it up.
  assert.ok(evaluateSvoFieldProgram(program, { x: 0.095, y: 0, z: 0 }).distance_m < 0, "the long axis must survive");
  assert.ok(evaluateSvoFieldProgram(program, { x: 0, y: 0.035, z: 0 }).distance_m < 0, "the flattened axis is shorter");
  assert.ok(evaluateSvoFieldProgram(program, { x: 0, y: 0.045, z: 0 }).distance_m > 0, "and stops where it should");
  assert.ok(
    Math.abs(evaluateSvoFieldProgram(program, { x: 0.2, y: 0, z: 0 }).lipschitz - 1 / flatten) < 1e-3,
    "the declared constant is 1/min(scale) — a squashed axis makes every distance downstream read too far",
  );
  const extent = svoFieldProgramExtent_m(program);
  assert.ok(extent[0] >= 0.1 && extent[2] >= 0.1, "the unrotated frame keeps its per-axis extent");

  // With rotation the same tape must still contain its solid, which is what the
  // isotropic fallback in the extent bound is for.
  const turned: SvoFieldProgram = {
    ops: [
      { op: "anisotropic-frame", out: 1, point: 0, seed: 0x93, parameters: { a: 1, b: flatten, c: 0.7, d: 1 } },
      { op: "sphere", out: 0, point: 1, parameters: { a: 0.1 } },
    ],
    result: 0,
  };
  const turnedExtent = svoFieldProgramExtent_m(turned);
  for (let index = 0; index < 6000; index += 1) {
    const t = index * 0.0031;
    const point: Vec3 = { x: (t % 0.5) - 0.25, y: ((t * 1.618) % 0.5) - 0.25, z: ((t * 2.414) % 0.5) - 0.25 };
    if (evaluateSvoFieldProgram(turned, point).distance_m > 0) continue;
    assert.ok(
      Math.abs(point.x) <= turnedExtent[0] + 1e-6 &&
        Math.abs(point.y) <= turnedExtent[1] + 1e-6 &&
        Math.abs(point.z) <= turnedExtent[2] + 1e-6,
      `a rotated frame put solid material at ${JSON.stringify(point)} outside ${JSON.stringify(turnedExtent)}`,
    );
  }
});

test("worley-subtract pits the surface — it only removes, and it removes measurably", () => {
  const base: SvoFieldProgram = { ops: [{ op: "sphere", out: 0, parameters: { a: 0.12 } }], result: 0 };
  const pitted: SvoFieldProgram = {
    ops: [
      { op: "sphere", out: 0, parameters: { a: 0.12 } },
      { op: "worley-subtract", out: 1, point: 0, fieldA: 0, seed: 0x77, parameters: { a: 0.006, b: 0.0022, c: 0.0008 } },
    ],
    result: 1,
  };
  let deepest = 0;
  for (let index = 0; index < 6000; index += 1) {
    const t = index * 0.00021;
    const direction: Vec3 = { x: Math.cos(t * 7.3), y: Math.sin(t * 4.1), z: Math.cos(t * 2.9) };
    const length = Math.hypot(direction.x, direction.y, direction.z);
    const point: Vec3 = {
      x: (direction.x / length) * 0.1195,
      y: (direction.y / length) * 0.1195,
      z: (direction.z / length) * 0.1195,
    };
    const plain = evaluateSvoFieldProgram(base, point).distance_m;
    const carved = evaluateSvoFieldProgram(pitted, point).distance_m;
    assert.ok(carved >= plain - 1e-9, `a subtractive pit added material at ${JSON.stringify(point)}`);
    deepest = Math.max(deepest, carved - plain);
  }
  assert.ok(deepest > 1e-3, `the pitting moved the surface by at most ${(deepest * 1000).toFixed(3)} mm`);
});

test("ridged-worley-add only adds inside the shell it declares", () => {
  const shell_m = 0.004;
  const base: SvoFieldProgram = { ops: [{ op: "sphere", out: 0, parameters: { a: 0.12 } }], result: 0 };
  const proud: SvoFieldProgram = {
    ops: [
      { op: "sphere", out: 0, parameters: { a: 0.12 } },
      {
        op: "ridged-worley-add",
        out: 1,
        point: 0,
        fieldA: 0,
        seed: 0x60d2,
        parameters: { a: 0.012, b: 0.004, c: 0.001, d: shell_m },
      },
    ],
    result: 1,
  };
  const blend_m = 0.001;
  let raised = 0;
  let farthestSolid = 0;
  for (let index = 0; index < 8000; index += 1) {
    const t = index * 0.00019;
    const radius = 0.1 + (index % 400) * 0.0002;
    const direction: Vec3 = { x: Math.cos(t * 7.3), y: Math.sin(t * 4.1), z: Math.cos(t * 2.9) };
    const length = Math.hypot(direction.x, direction.y, direction.z);
    const point: Vec3 = {
      x: (direction.x / length) * radius,
      y: (direction.y / length) * radius,
      z: (direction.z / length) * radius,
    };
    const plain = evaluateSvoFieldProgram(base, point).distance_m;
    const grown = evaluateSvoFieldProgram(proud, point).distance_m;
    // The op is a union, so it can only lower the field — never raise it, which
    // would be removing material an "add" has no business removing.
    assert.ok(grown <= plain + 1e-9, `the additive op raised the field at ${JSON.stringify(point)}`);
    // And the *solid* it adds stays inside the declared shell. This is the
    // claim the extent bound rests on: a union of lattice balls with no gate is
    // solid everywhere in space and has no proxy box at all.
    if (grown <= 0) {
      farthestSolid = Math.max(farthestSolid, plain);
      assert.ok(
        plain <= shell_m + blend_m / 4 + 1e-9,
        `an inclusion was solid ${plain.toFixed(4)} m off the surface, outside the ${shell_m} m shell`,
      );
    }
    raised = Math.max(raised, plain - grown);
  }
  assert.ok(raised > 5e-4, `the inclusions raised the surface by at most ${(raised * 1000).toFixed(3)} mm`);
  assert.ok(farthestSolid > 1e-4, "no inclusion stood proud of the surface at all");
});

/**
 * The cost of that gate, stated where it can be measured rather than left to be
 * discovered.
 *
 * Restricting the inclusions with `max(ball, base − shell)` is exact for the
 * *solid* and cheap in Lipschitz terms — a max of two fields is no steeper than
 * the steeper of them. What it is not is tight far from the surface: out there
 * the gate is what the union sees, so the field reads a constant `shell` short
 * of the truth. That is an under-estimate, which is the safe direction for a
 * sphere trace, and it costs a few millimetres of step on a march that the
 * proxy box has already bracketed.
 *
 * The alternative — fading the whole op out on a smoothstep of the base field —
 * removes the offset and costs about 4.6x in the declared Lipschitz constant,
 * because the fade's own gradient scales with 1/shell. That is a much worse
 * trade, and this test is here so the offset is a known quantity rather than a
 * surprise in a march profile.
 */
test("the inclusion gate under-estimates far from the surface by exactly its shell depth", () => {
  const shell_m = 0.004;
  const base: SvoFieldProgram = { ops: [{ op: "sphere", out: 0, parameters: { a: 0.12 } }], result: 0 };
  const proud: SvoFieldProgram = {
    ops: [
      { op: "sphere", out: 0, parameters: { a: 0.12 } },
      {
        op: "ridged-worley-add",
        out: 1,
        point: 0,
        fieldA: 0,
        seed: 0x60d2,
        parameters: { a: 0.012, b: 0.004, c: 0.001, d: shell_m },
      },
    ],
    result: 1,
  };
  const far: Vec3 = { x: 0.4, y: 0.1, z: -0.2 };
  const offset_m =
    evaluateSvoFieldProgram(base, far).distance_m - evaluateSvoFieldProgram(proud, far).distance_m;
  assert.ok(
    Math.abs(offset_m - shell_m) < 1e-6,
    `the far-field offset is ${(offset_m * 1000).toFixed(3)} mm against the ${shell_m * 1000} mm shell; if this ` +
      "has grown, the gate has stopped being a plain max and the trade above no longer holds",
  );
});

test("erosion-mask weathers the up-facing side harder, and never adds material", () => {
  const shared = { a: 0.006, b: 0.0022, c: 0.0008 } as const;
  const unmasked: SvoFieldProgram = {
    ops: [
      { op: "sphere", out: 0, parameters: { a: 0.12 } },
      { op: "worley-subtract", out: 1, point: 0, fieldA: 0, seed: 0x77, parameters: shared },
    ],
    result: 1,
  };
  const masked: SvoFieldProgram = {
    ops: [
      { op: "sphere", out: 0, parameters: { a: 0.12 } },
      { op: "worley-subtract", out: 1, point: 0, fieldA: 0, seed: 0x77, parameters: shared },
      // Purely orientation-driven, so the comparison isolates the up term.
      { op: "erosion-mask", out: 2, point: 0, fieldA: 0, fieldB: 1, parameters: { a: 1, b: 0, c: 0.004, d: 0.12 } },
    ],
    result: 2,
  };
  let topRemoval = 0;
  let bottomRemoval = 0;
  for (let index = 0; index < 4000; index += 1) {
    const angle = index * 0.0016;
    const radius = 0.1185 + (index % 7) * 0.0002;
    const upward: Vec3 = { x: Math.cos(angle) * radius * 0.3, y: radius * 0.95, z: Math.sin(angle) * radius * 0.3 };
    const downward: Vec3 = { x: upward.x, y: -upward.y, z: upward.z };
    const at = (point: Vec3) =>
      evaluateSvoFieldProgram(masked, point).distance_m - evaluateSvoFieldProgram(unmasked, point).distance_m;
    // The masked field sits between the untouched solid and the fully eroded
    // one, so its difference from the eroded one is what the mask held back.
    topRemoval += Math.abs(at(upward));
    bottomRemoval += Math.abs(at(downward));

    for (const point of [upward, downward]) {
      const bare = evaluateSvoFieldProgram({ ops: [{ op: "sphere", out: 0, parameters: { a: 0.12 } }], result: 0 }, point)
        .distance_m;
      assert.ok(
        evaluateSvoFieldProgram(masked, point).distance_m >= bare - 1e-9,
        "an erosion mask must never put material back",
      );
    }
  }
  assert.ok(
    bottomRemoval > 1.5 * topRemoval,
    `the mask held back ${bottomRemoval.toFixed(4)} on the underside against ${topRemoval.toFixed(4)} on top; ` +
      "weathering that does not collect where water collects reads as fabric, not stone",
  );
});

// ---------------------------------------------------------------------------
// Scatter — the op whose naive form breaks the distance bound
// ---------------------------------------------------------------------------

test("scatter replicates its occupant into a finite cluster with a bounded extent", () => {
  const cell_m = 0.05;
  const program: SvoFieldProgram = {
    ops: [
      { op: "scatter", out: 1, point: 0, seed: 0x7b3d, parameters: { a: cell_m, b: 1, c: 0.0, d: 0.014 } },
      { op: "sphere", out: 0, point: 1, parameters: { a: 0.012 } },
    ],
    result: 0,
  };
  // An instance at the origin and one a whole cell away, with no jitter.
  assert.ok(evaluateSvoFieldProgram(program, { x: 0, y: 0, z: 0 }).distance_m < 0, "the origin instance is solid");
  assert.ok(
    evaluateSvoFieldProgram(program, { x: cell_m, y: 0, z: 0 }).distance_m < 0,
    "the neighbouring instance is solid",
  );
  assert.ok(
    evaluateSvoFieldProgram(program, { x: 0.5 * cell_m, y: 0, z: 0 }).distance_m > 0,
    "the gap between them is not",
  );
  // And the repetition stops: the index is clamped, so two cells out is empty.
  assert.ok(
    evaluateSvoFieldProgram(program, { x: 2 * cell_m, y: 0, z: 0 }).distance_m > 0,
    "a scatter is a finite cluster; an infinite lattice has no extent to cull against",
  );
  const extent = svoFieldProgramExtent_m(program);
  assert.ok(
    extent.every((half) => Math.abs(half - (cell_m + 0.012)) < 1e-9),
    `one repetition of a 12 mm occupant on a 50 mm lattice reaches 62 mm, got ${JSON.stringify(extent)}`,
  );
});

/**
 * The failure the brief flags as the dangerous one, isolated.
 *
 * Naive domain repetition folds the point by a translation that jumps at each
 * cell wall, and with per-cell jitter the occupant on the two sides of a wall
 * is in a different place, so the folded field jumps too. A jump is not a large
 * Lipschitz constant, it is the absence of one.
 *
 * The construction here saturates the field at the wall clearance, which makes
 * the value on the wall identical from both sides. This test walks a line
 * straight through several walls at a step far finer than the jitter and asserts
 * the field never moves faster than it declares.
 */
test("a jittered scatter stays continuous across its cell walls", () => {
  const cell_m = 0.04;
  const program: SvoFieldProgram = {
    ops: [
      { op: "scatter", out: 1, point: 0, seed: 0x7b3d_1e92, parameters: { a: cell_m, b: 2, c: 0.016, d: 0.011 } },
      { op: "sphere", out: 0, point: 1, parameters: { a: 0.01 } },
    ],
    result: 0,
  };
  const declared = evaluateSvoFieldProgram(program, { x: 0.03, y: 0.017, z: 0.009 }).lipschitz;
  const step_m = 2e-6;
  let worst = 0;
  let worstAt = 0;
  // Straight through five cell walls, off-axis so the walls are crossed on all
  // three axes rather than squarely on one.
  for (let index = 0; index < 60000; index += 1) {
    const t = -0.1 + index * 4e-6;
    const along = (offset: number): Vec3 => ({ x: t + offset, y: 0.37 * (t + offset) + 0.013, z: 0.11 * (t + offset) - 0.007 });
    const forward = evaluateSvoFieldProgram(program, along(step_m)).distance_m;
    const backward = evaluateSvoFieldProgram(program, along(-step_m)).distance_m;
    // The line's own length per unit of `t`, so the measured slope is a true
    // directional derivative rather than one scaled by the parameterisation.
    const speed = Math.hypot(1, 0.37, 0.11);
    const slope = Math.abs(forward - backward) / (2 * step_m * speed);
    if (slope > worst) {
      worst = slope;
      worstAt = t;
    }
  }
  assert.ok(
    worst <= declared * (1 + 1e-3),
    `the folded field moved at ${worst.toFixed(4)} against a declared ${declared.toFixed(4)} at t=${worstAt}. A ` +
      "jittered domain repetition that jumps at a cell wall tears every instance along that wall",
  );
  assert.ok(worst > 0.5, "the folded field is flat, so the walk missed the cluster entirely");
});

test("the fold clearance is what a scatter saturates at, and it grows outside the cluster", () => {
  const program: SvoFieldProgram = {
    ops: [
      { op: "scatter", out: 1, point: 0, seed: 0x7b3d, parameters: { a: 0.04, b: 1, c: 0.012, d: 0.011 } },
      { op: "sphere", out: 0, point: 1, parameters: { a: 0.01 } },
    ],
    result: 0,
  };
  // clearance = cell/2 - jitter/2 - occupant = 20 - 6 - 11 = 3 mm.
  const inside = evaluateSvoFieldProgram(program, { x: 0.02, y: 0.02, z: 0.02 }).distance_m;
  assert.ok(inside <= 0.003 + 1e-9, `inside the cluster the field must saturate at 3 mm, read ${inside}`);
  // Well outside the array box the clearance grows with the distance to it, or
  // a ray still metres away would crawl at three millimetres a step.
  const far = evaluateSvoFieldProgram(program, { x: 0.5, y: 0, z: 0 }).distance_m;
  assert.ok(far > 0.3, `outside the cluster the field must open back up, read ${far}`);
});

// ---------------------------------------------------------------------------
// Refusals — the rules that keep the two constructions above true
// ---------------------------------------------------------------------------

test("validateSvoFieldProgram refuses a scatter whose occupant does not fit its cell", () => {
  // No clearance at all: a 30 mm occupant in a 40 mm cell.
  assert.throws(
    () =>
      validateSvoFieldProgram({
        ops: [
          { op: "scatter", out: 1, point: 0, parameters: { a: 0.04, b: 1, c: 0.01, d: 0.03 } },
          { op: "sphere", out: 0, point: 1, parameters: { a: 0.01 } },
        ],
        result: 0,
      }),
    /no clearance at the cell wall/,
  );
  // Declared occupant smaller than the source actually under it.
  assert.throws(
    () =>
      validateSvoFieldProgram({
        ops: [
          { op: "scatter", out: 1, point: 0, parameters: { a: 0.05, b: 1, c: 0.01, d: 0.008 } },
          { op: "sphere", out: 0, point: 1, parameters: { a: 0.02 } },
        ],
        result: 0,
      }),
    /declared an occupant of at most/,
  );
  // A warp under a scatter counts against the same budget, because it moves the
  // occupant's surface outward by its own amplitude.
  assert.throws(
    () =>
      validateSvoFieldProgram({
        ops: [
          { op: "scatter", out: 1, point: 0, parameters: { a: 0.05, b: 1, c: 0.01, d: 0.014 } },
          { op: "domain-warp", out: 2, point: 1, parameters: { a: 0.01, b: 40 } },
          { op: "sphere", out: 0, point: 2, parameters: { a: 0.012 } },
        ],
        result: 0,
      }),
    /declared an occupant of at most/,
  );
  // And the same tape with the warp accounted for is accepted.
  assert.doesNotThrow(() =>
    validateSvoFieldProgram({
      ops: [
        { op: "scatter", out: 1, point: 0, parameters: { a: 0.08, b: 1, c: 0.01, d: 0.03 } },
        { op: "domain-warp", out: 2, point: 1, parameters: { a: 0.01, b: 40 } },
        { op: "sphere", out: 0, point: 2, parameters: { a: 0.012 } },
      ],
      result: 0,
    }),
  );
});

test("validateSvoFieldProgram refuses the two ops that cannot be continuous on a folded point", () => {
  const folded = (op: SvoFieldOpName): SvoFieldProgram => ({
    ops: [
      { op: "scatter", out: 1, point: 0, parameters: { a: 0.05, b: 1, c: 0.01, d: 0.014 } },
      { op: "sphere", out: 0, point: 1, parameters: { a: 0.012 } },
      { op: "sphere", out: 1, point: 1, parameters: { a: 0.008 } },
      { op, out: 2, point: 1, fieldA: 0, fieldB: 1, parameters: OP_FIXTURE[op] },
    ],
    result: 2,
  });
  assert.throws(() => validateSvoFieldProgram(folded("ridged-worley-add")), /which a scatter has folded/);
  assert.throws(() => validateSvoFieldProgram(folded("erosion-mask")), /which a scatter has folded/);
  // Fracture and the subtractive carve are squeezed back to the wall value by
  // the fold clamp, so they are allowed — and the worked stone relies on it.
  assert.doesNotThrow(() => validateSvoFieldProgram(folded("fracture")));
  assert.doesNotThrow(() => validateSvoFieldProgram(folded("worley-subtract")));
});

test("the extent bound refuses a dilation past a folded field's saturation rather than guessing", () => {
  assert.throws(
    () =>
      svoFieldProgramExtent_m({
        ops: [
          // 2 mm of clearance at the cell wall...
          { op: "scatter", out: 1, point: 0, parameters: { a: 0.05, b: 1, c: 0.01, d: 0.018 } },
          { op: "sphere", out: 0, point: 1, parameters: { a: 0.01 } },
          { op: "ellipsoid", out: 1, point: 0, parameters: { a: 0.2, b: 0.2, c: 0.2 } },
          // ...against a union asking for 25 mm of blend, which dips 6 mm.
          { op: "smooth-union", out: 2, fieldA: 1, fieldB: 0, parameters: { a: 1, b: 0.025, c: 0.025 } },
        ],
        result: 2,
      }),
    /saturates at .* but an op asks the extent bound to dilate it/,
  );
});

// ---------------------------------------------------------------------------
// Arena packing
// ---------------------------------------------------------------------------

test("a packed field program round-trips through the arena", () => {
  const program: SvoFieldProgram = {
    ops: [
      { op: "domain-warp", out: 1, point: 0, seed: 0xabcd_1234, parameters: { a: 0.045, b: 6.6667 } },
      { op: "domain-warp", out: 2, point: 1, seed: 0x0001_0203, parameters: { a: 0.008, b: 50 } },
      { op: "ellipsoid", out: 3, point: 2, parameters: { a: 0.16, b: 0.11, c: 0.13 } },
    ],
    result: 3,
  };
  const arena = packSvoFieldProgramArena([program], 4);
  assert.equal(arena.length, 4 * SVO_FIELD_PROGRAM_BLOCK_WORDS);
  const restored = unpackSvoFieldProgram(arena, 0);
  assert.equal(restored.result, program.result);
  assert.equal(restored.ops.length, program.ops.length);
  restored.ops.forEach((op, index) => {
    const original = program.ops[index];
    assert.equal(op.op, original.op);
    assert.equal(op.out, original.out);
    assert.equal(op.point, original.point ?? 0);
    assert.equal(op.seed, original.seed ?? 0);
    assert.ok(Math.abs((op.parameters?.a ?? 0) - (original.parameters?.a ?? 0)) < 1e-6);
    assert.ok(Math.abs((op.parameters?.b ?? 0) - (original.parameters?.b ?? 0)) < 1e-6);
  });
  // The round-tripped tape must also *evaluate* the same, which the field
  // comparison above does not by itself prove. The tolerance is f32: the arena
  // stores single precision, so a tape authored with `1 / 0.15` evaluates
  // through the rounded parameter once it has been packed — which is what the
  // GPU will run, and therefore the answer that matters.
  const point: Vec3 = { x: 0.11, y: -0.07, z: 0.19 };
  const before = evaluateSvoFieldProgram(program, point).distance_m;
  const after = evaluateSvoFieldProgram(restored, point).distance_m;
  assert.ok(Math.abs(before - after) < 1e-6, `packing moved the field by ${Math.abs(before - after)}`);

  // The exact property, which the f32 rounding does not weaken: re-packing what
  // was unpacked reproduces the bytes. Anything less means a tape that survives
  // one publication and drifts on the next.
  assert.deepEqual(Array.from(packSvoFieldProgramArena([restored], 4)), Array.from(arena));
});

test("the worked stone round-trips through the arena, fifth parameter included", () => {
  const arena = packSvoFieldProgramArena([EVERY_OP], 2);
  const restored = unpackSvoFieldProgram(arena, 0);
  assert.equal(restored.ops.length, EVERY_OP.ops.length);
  restored.ops.forEach((op, index) => {
    const original = EVERY_OP.ops[index];
    assert.equal(op.op, original.op);
    assert.equal(op.out, original.out);
    assert.equal(op.point, original.point ?? 0);
    assert.equal(op.fieldA, original.fieldA);
    assert.equal(op.fieldB, original.fieldB);
    assert.equal(op.seed, original.seed ?? 0);
    for (const parameter of ["a", "b", "c", "d", "e"] as const) {
      assert.equal(
        op.parameters?.[parameter] ?? 0,
        Math.fround(original.parameters?.[parameter] ?? 0),
        `op ${index} (${op.op}) lost parameter ${parameter} through the arena`,
      );
    }
  });
  // The fifth parameter is the fracture's blend radius, and it lives in the
  // word that used to be reserved. A tape that lost it would render every
  // facet with a razor arris and no error anywhere.
  const fracture = restored.ops.find((op) => op.op === "fracture")!;
  assert.ok((fracture.parameters?.e ?? 0) > 0, "the fracture's blend radius did not survive packing");

  const probe: Vec3 = { x: 0.11, y: -0.07, z: 0.19 };
  assert.ok(
    Math.abs(
      evaluateSvoFieldProgram(EVERY_OP, probe).distance_m - evaluateSvoFieldProgram(restored, probe).distance_m,
    ) < 1e-6,
    "packing moved the field",
  );
  assert.deepEqual(Array.from(packSvoFieldProgramArena([restored], 2)), Array.from(arena));
});

test("the generated WGSL exposes a banded entry point and an unbanded one that forwards to it", () => {
  const wgsl = svoFieldProgramWGSL({
    functionName: "testEvaluate",
    loadWord: "loadWord",
    baseWordExpression: "0u",
    capacityExpression: "16u",
  });
  assert.ok(
    wgsl.includes("fn testEvaluateBanded(blockIndex:u32,localPoint:vec3f,samplingLength_m:f32)->SvoFieldValue"),
    "the voxelizer's entry point, which takes the voxel cell size, is missing",
  );
  assert.ok(
    wgsl.includes("fn testEvaluate(blockIndex:u32,localPoint:vec3f)->SvoFieldValue"),
    "the unbanded entry point every existing call site spells is missing",
  );
  assert.ok(
    wgsl.includes("return testEvaluateBanded(blockIndex,localPoint,0.0);"),
    "the unbanded entry point must forward, or the two will drift",
  );
  // The struct is the ABI two other modules spell out by hand; widening it
  // there is a shader compile failure a long way from here.
  assert.ok(wgsl.includes("struct SvoFieldValue{distance_m:f32,lipschitz:f32}"));
  // Every scalar parameter has to be decoded, including the fifth, or an op
  // silently runs on a zero it never read.
  for (const word of ["at+3u", "at+4u", "at+5u", "at+6u", "at+7u"]) {
    assert.ok(wgsl.includes(`bitcast<f32>(loadWord(${word}))`), `parameter word ${word} is never decoded`);
  }
});

test("the arena refuses a capacity that cannot hold every program", () => {
  const program: SvoFieldProgram = { ops: [{ op: "sphere", out: 0, parameters: { a: 0.1 } }], result: 0 };
  assert.throws(() => packSvoFieldProgramArena([program, program], 1), /capacity must hold every program/);
});

test("unused arena slots stay zero, so an unwritten block cannot decode as a shape", () => {
  const program: SvoFieldProgram = { ops: [{ op: "sphere", out: 0, parameters: { a: 0.1 } }], result: 0 };
  const arena = packSvoFieldProgramArena([program], 3);
  for (let index = SVO_FIELD_PROGRAM_BLOCK_WORDS; index < arena.length; index += 1) {
    assert.equal(arena[index], 0, `arena word ${index} is not zero`);
  }
  // Op count zero means the result register is never written, which the
  // evaluator reports as the empty field rather than as a solid.
  assert.equal(unpackSvoFieldProgram(arena, 1).ops.length, 0);
});
