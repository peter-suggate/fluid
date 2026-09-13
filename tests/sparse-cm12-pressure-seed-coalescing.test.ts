import assert from "node:assert/strict";
import test from "node:test";

const WORKGROUP_SIZE = 64;
const f32 = Math.fround;

type SeedInput = Readonly<{
  rhs: number;
  image: number;
  diagonal: number;
}>;

type SeedReceipt = readonly [gamma: number, rhs2: number,
  residual2: number, maximumResidual: number];

function reduceWorkgroup(values: readonly number[], maximum = false): number {
  const lanes = Array.from({ length: WORKGROUP_SIZE }, (_, lane) =>
    f32(values[lane] ?? 0));
  for (let width = 32; width >= 1; width /= 2) {
    for (let lane = 0; lane < width; lane += 1) {
      lanes[lane] = maximum
        ? Math.max(lanes[lane]!, lanes[lane + width]!)
        : f32(lanes[lane]! + lanes[lane + width]!);
    }
  }
  return lanes[0]!;
}

function reducePartials(partials: readonly number[], maximum = false): number {
  const lanes = Array.from({ length: WORKGROUP_SIZE }, (_, lane) => {
    let value = 0;
    for (let at = lane; at < partials.length; at += WORKGROUP_SIZE) {
      value = maximum ? Math.max(value, partials[at]!)
        : f32(value + partials[at]!);
    }
    return value;
  });
  return reduceWorkgroup(lanes, maximum);
}

function oldSeed(inputs: readonly SeedInput[]): Readonly<{
  residual: readonly number[];
  z: readonly number[];
  direction: readonly number[];
  receipt: SeedReceipt;
}> {
  const residual = inputs.map(({ rhs, image }) => f32(rhs - image));
  // initializePCG writes z, then initializeJacobiDirection recomputes it.
  const initialZ = residual.map((value, index) => inputs[index]!.diagonal > 0
    ? f32(value / inputs[index]!.diagonal) : 0);
  const z = residual.map((value, index) => inputs[index]!.diagonal > 0
    ? f32(value / inputs[index]!.diagonal) : 0);
  assert.deepEqual(initialZ, z);
  const groups = Math.ceil(inputs.length / WORKGROUP_SIZE);
  const gammaPartials: number[] = [], rhs2Partials: number[] = [];
  const residual2Partials: number[] = [], maximumPartials: number[] = [];
  for (let group = 0; group < groups; group += 1) {
    const begin = group * WORKGROUP_SIZE;
    const gamma = residual.slice(begin, begin + WORKGROUP_SIZE)
      .map((value, lane) => f32(value * z[begin + lane]!));
    const rhs2 = inputs.slice(begin, begin + WORKGROUP_SIZE)
      .map(value => f32(value.rhs * value.rhs));
    gammaPartials.push(reduceWorkgroup(gamma));
    rhs2Partials.push(reduceWorkgroup(rhs2));

    // The former true-residual pass recomputed the same b-Ap before any update.
    const measured = inputs.slice(begin, begin + WORKGROUP_SIZE)
      .map(value => f32(value.rhs - value.image));
    residual2Partials.push(reduceWorkgroup(measured.map(value => f32(value * value))));
    maximumPartials.push(reduceWorkgroup(measured.map(Math.abs), true));
  }
  return { residual, z, direction: z,
    receipt: [reducePartials(gammaPartials), reducePartials(rhs2Partials),
      reducePartials(residual2Partials), reducePartials(maximumPartials, true)] };
}

function coalescedSeed(inputs: readonly SeedInput[]): ReturnType<typeof oldSeed> {
  const residual: number[] = [], z: number[] = [];
  const groupReceipts: SeedReceipt[] = [];
  const groups = Math.ceil(inputs.length / WORKGROUP_SIZE);
  for (let group = 0; group < groups; group += 1) {
    const lanes: SeedReceipt[] = [];
    for (let lane = 0; lane < WORKGROUP_SIZE; lane += 1) {
      const input = inputs[group * WORKGROUP_SIZE + lane];
      if (!input) { lanes.push([0, 0, 0, 0]); continue; }
      const r = f32(input.rhs - input.image);
      const preconditioned = input.diagonal > 0 ? f32(r / input.diagonal) : 0;
      residual.push(r); z.push(preconditioned);
      lanes.push([f32(r * preconditioned), f32(input.rhs * input.rhs),
        f32(r * r), Math.abs(r)]);
    }
    groupReceipts.push([
      reduceWorkgroup(lanes.map(value => value[0])),
      reduceWorkgroup(lanes.map(value => value[1])),
      reduceWorkgroup(lanes.map(value => value[2])),
      reduceWorkgroup(lanes.map(value => value[3]), true),
    ]);
  }
  return { residual, z, direction: z,
    receipt: [
      reducePartials(groupReceipts.map(value => value[0])),
      reducePartials(groupReceipts.map(value => value[1])),
      reducePartials(groupReceipts.map(value => value[2])),
      reducePartials(groupReceipts.map(value => value[3]), true),
    ] };
}

function deterministicInputs(count: number): SeedInput[] {
  let state = 0x1234_5678;
  const next = () => {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
    return state / 0x1_0000_0000;
  };
  return Array.from({ length: count }, (_, index) => {
    const rhs = f32((next() - 0.5) * (1 + index % 19));
    const image = f32((next() - 0.5) * (1 + index % 7));
    const diagonal = index % 31 === 0 ? 0 : f32(0.125 + 8 * next());
    return { rhs, image, diagonal };
  });
}

test("coalesced pressure seed preserves vectors and reduction trees", () => {
  // Cross partial workgroups and the final reducer's 64-way striding boundary.
  for (const count of [0, 1, 63, 64, 65, 4_095, 4_096, 4_097, 11_003]) {
    const input = deterministicInputs(count);
    assert.deepEqual(coalescedSeed(input), oldSeed(input), `cell count ${count}`);
  }
});

test("coalesced pressure seed preserves the initial convergence decision", () => {
  const input = deterministicInputs(1_337);
  const old = oldSeed(input).receipt;
  const coalesced = coalescedSeed(input).receipt;
  for (const tolerance of [0, 1e-7, 1e-4, 0.1, 1]) {
    const oldConverged = tolerance > 0 && old[2] <= tolerance * tolerance * old[1];
    const coalescedConverged = tolerance > 0
      && coalesced[2] <= tolerance * tolerance * coalesced[1];
    assert.equal(coalescedConverged, oldConverged, `tolerance ${tolerance}`);
  }
});
