/**
 * Exact, partition-independent summation of finite f32 values for WebGPU.
 *
 * Each value is decomposed into signed radix-256 digits and deposited into
 * atomic i32 limbs. Integer addition is exact, associative, and commutative,
 * so the total is invariant to row order, workgroup partitioning, and dispatch
 * scheduling. The singleton decoder propagates carries and rounds only once
 * when it returns to f32.
 */

/** Least-significant represented bit. Covers every finite f32 subnormal. */
export const SIGNED_RADIX_256_F32_MIN_EXPONENT = -152;
/** 35 value limbs plus one signed carry limb. */
export const SIGNED_RADIX_256_F32_LIMBS = 36;
/** Largest arbitrary term count whose byte limbs plus propagated carry fit i32. */
export const SIGNED_RADIX_256_F32_MAX_TERMS = 8_388_607;

export interface SignedRadix256F32ReductionLayout {
  readonly scalarCount: number;
  readonly partialCapacity: number;
  readonly wordsPerPartial: number;
  readonly bytesPerPartial: number;
  readonly byteLength: number;
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`${label} must be a positive integer`);
  }
  return value;
}

/** Plan the atomic-limb buffer shared by the deposit and singleton finish passes. */
export function planSignedRadix256F32Reduction(
  scalarCountValue: number,
  partialCapacityValue: number,
): SignedRadix256F32ReductionLayout {
  const scalarCount = positiveInteger(
    scalarCountValue,
    "Signed radix-256 reduction scalar count",
  );
  const partialCapacity = positiveInteger(
    partialCapacityValue,
    "Signed radix-256 reduction partial capacity",
  );
  const wordsPerPartial = scalarCount * SIGNED_RADIX_256_F32_LIMBS;
  const bytesPerPartial = wordsPerPartial * 4;
  const byteLength = partialCapacity * bytesPerPartial;
  if (!Number.isSafeInteger(byteLength)) {
    throw new RangeError("Signed radix-256 reduction buffer size exceeds host integer precision");
  }
  return Object.freeze({
    scalarCount,
    partialCapacity,
    wordsPerPartial,
    bytesPerPartial,
    byteLength,
  });
}

/**
 * Prove that neither deposited byte limbs nor canonical carry propagation can
 * overflow i32. Each input contributes at most 255 plus one incoming carry to
 * any limb, yielding the conservative 256-times-term-count bound above.
 */
export function assertSignedRadix256F32TermCapacity(maximumTermCountValue: number): number {
  const maximumTermCount = positiveInteger(
    maximumTermCountValue,
    "Signed radix-256 reduction maximum term count",
  );
  if (maximumTermCount > SIGNED_RADIX_256_F32_MAX_TERMS) {
    throw new RangeError(
      `Signed radix-256 reduction supports at most ${SIGNED_RADIX_256_F32_MAX_TERMS.toLocaleString("en-US")} terms`,
    );
  }
  return maximumTermCount;
}

export interface SignedRadix256F32ReductionWGSLOptions {
  /** Bind group containing the atomic limb buffer. */
  readonly group: number;
  /** Binding occupied by the atomic limb buffer. */
  readonly binding: number;
  /** Independent scalar totals stored in each partial. */
  readonly scalarCount: number;
  /** Number of invocations participating in clearFixedPartial. */
  readonly reductionLanes: number;
  /** WGSL expression returning the number of live partials to merge. */
  readonly livePartialCountExpression: string;
  /** Host-attested upper bound on deposited terms across all live partials. */
  readonly maximumTermCount: number;
}

function nonNegativeInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${label} must be a non-negative integer`);
  }
  return value;
}

function wgslExpression(value: string): string {
  const expression = value.trim();
  if (expression.length === 0 || expression.includes(";")) {
    throw new TypeError("Signed radix-256 live partial count must be a WGSL expression");
  }
  return expression;
}

/**
 * Lanes co-operating on one limb column of the partial fold.
 *
 * The fold is a `livePartialCount x FIXED_LIMBS` integer matrix summed down its
 * columns. Assigning `limb = lane % FIXED_LIMBS` makes each group of 36 lanes
 * read one partial record as a single contiguous 144-byte span, so the strided
 * partial walk stays coalesced; `group = lane / FIXED_LIMBS` then splits the
 * partial axis across as many such groups as the workgroup can host.
 */
export function signedRadix256F32LimbGroups(reductionLanes: number): number {
  return Math.floor(
    positiveInteger(reductionLanes, "Signed radix-256 reduction lane count")
      / SIGNED_RADIX_256_F32_LIMBS,
  );
}

/**
 * Emit the complete WGSL storage/helper ABI for one exact-reduction instance.
 *
 * The emitted public helpers are `clearFixedPartial`, `addFixedF32`, and
 * `fixedScalarValue`. A consumer clears one partial cooperatively, deposits
 * finite row values, then folds with `fixedScalarValue` from a finish kernel.
 * Non-finite inputs must be rejected by the consumer before deposit.
 */
export function createSignedRadix256F32ReductionWGSL(
  options: SignedRadix256F32ReductionWGSLOptions,
): string {
  const group = nonNegativeInteger(options.group, "Signed radix-256 bind group");
  const binding = nonNegativeInteger(options.binding, "Signed radix-256 binding");
  const scalarCount = positiveInteger(
    options.scalarCount,
    "Signed radix-256 reduction scalar count",
  );
  const reductionLanes = positiveInteger(
    options.reductionLanes,
    "Signed radix-256 reduction lane count",
  );
  const limbGroups = signedRadix256F32LimbGroups(reductionLanes);
  if (limbGroups < 1) {
    throw new RangeError(
      `Signed radix-256 cooperative fold needs at least ${SIGNED_RADIX_256_F32_LIMBS} reduction lanes`,
    );
  }
  const livePartialCount = wgslExpression(options.livePartialCountExpression);
  assertSignedRadix256F32TermCapacity(options.maximumTermCount);

  return /* wgsl */ `
// Exact signed radix-256 superaccumulator storage. Integer limbs make the
// result independent of scheduling, partitioning, and input permutation.
@group(${group}) @binding(${binding}) var<storage, read_write> partials: array<atomic<i32>>;

const FIXED_LIMBS = ${SIGNED_RADIX_256_F32_LIMBS}u;
const FIXED_SCALARS = ${scalarCount}u;
const FIXED_WORDS_PER_PARTIAL = FIXED_LIMBS * FIXED_SCALARS;
const FIXED_MIN_EXPONENT = ${SIGNED_RADIX_256_F32_MIN_EXPONENT};
// Independent partial-axis slices the fold splits across; see
// signedRadix256F32LimbGroups for why the mapping is limb-major.
const FIXED_LIMB_GROUPS = ${limbGroups}u;
const FIXED_FOLD_LANES = FIXED_LIMB_GROUPS * FIXED_LIMBS;

/** Live partials the consumer deposited into; the fold's default upper bound. */
fn fixedLivePartialCount() -> u32 { return ${livePartialCount}; }

var<workgroup> fixedFoldShare: array<i32, ${reductionLanes}>;

fn fixedAt(partial: u32, scalar: u32, limb: u32) -> u32 {
  return partial * FIXED_WORDS_PER_PARTIAL + scalar * FIXED_LIMBS + limb;
}

fn clearFixedPartial(partial: u32, lane: u32) {
  for (var word = lane; word < FIXED_WORDS_PER_PARTIAL; word += ${reductionLanes}u) {
    atomicStore(&partials[partial * FIXED_WORDS_PER_PARTIAL + word], 0);
  }
  workgroupBarrier();
}

// Deposit the exact finite f32 bit pattern into four signed radix-256 limbs.
// Normal values have value=M*2^(rawExponent-150); subnormals have M*2^-149.
// FIXED_MIN_EXPONENT leaves a non-negative shift for both forms.
fn addFixedF32(partial: u32, scalar: u32, value: f32) {
  let bits = bitcast<u32>(value);
  let magnitude = bits & 0x7fffffffu;
  if (magnitude == 0u) { return; }
  let rawExponent = (magnitude >> 23u) & 0xffu;
  let fraction = magnitude & 0x7fffffu;
  let significand = select(fraction, 0x800000u | fraction, rawExponent != 0u);
  let shift = select(3u, rawExponent + 2u, rawExponent != 0u);
  let firstLimb = shift >> 3u;
  let shifted = significand << (shift & 7u);
  let sign = select(1, -1, (bits & 0x80000000u) != 0u);
  for (var digit = 0u; digit < 4u; digit += 1u) {
    let limb = firstLimb + digit;
    let byte = i32((shifted >> (digit * 8u)) & 0xffu);
    if (byte != 0 && limb < FIXED_LIMBS) {
      atomicAdd(&partials[fixedAt(partial, scalar, limb)], sign * byte);
    }
  }
}

fn floorDiv256(value: i32) -> vec2i {
  var carry = value / 256;
  var digit = value - carry * 256;
  if (digit < 0) { digit += 256; carry -= 1; }
  return vec2i(carry, digit);
}

// Integer-only merge followed by one f32 rounding. Decode scaled limbs
// directly: materializing the unscaled integer in f32 can overflow even when
// its represented physical value is ordinary.
fn fixedDecodeLimbs(totals: array<i32, ${SIGNED_RADIX_256_F32_LIMBS}>) -> f32 {
  var limbs = totals;
  for (var limb = 0u; limb + 1u < FIXED_LIMBS; limb += 1u) {
    let normalized = floorDiv256(limbs[limb]);
    limbs[limb] = normalized.y;
    limbs[limb + 1u] += normalized.x;
  }

  let negative = limbs[FIXED_LIMBS - 1u] < 0;
  if (negative) {
    for (var limb = 0u; limb < FIXED_LIMBS; limb += 1u) {
      limbs[limb] = -limbs[limb];
    }
    for (var limb = 0u; limb + 1u < FIXED_LIMBS; limb += 1u) {
      let normalized = floorDiv256(limbs[limb]);
      limbs[limb] = normalized.y;
      limbs[limb + 1u] += normalized.x;
    }
  }
  var magnitude = 0.0;
  for (var limb = 0u; limb < FIXED_LIMBS; limb += 1u) {
    magnitude += ldexp(f32(limbs[limb]), FIXED_MIN_EXPONENT + i32(limb * 8u));
  }
  return select(magnitude, -magnitude, negative);
}

// Cooperative partial fold. Bit-identical to a serial walk: the limb totals are
// integer sums, so associativity and commutativity make them invariant under
// ANY partition of the (partial, limb) grid, and only lane 0's single carry
// propagation and f32 rounding follow.
//
// EVERY lane of the workgroup must reach this call -- it contains workgroup
// barriers, so it may only be invoked from uniform control flow. Callers select
// work by passing partialCount, never by branching around the call: a
// partialCount of 0 folds nothing and returns exactly +0.0, which is what a
// cleared (never-deposited) scalar folds to anyway.
//
// Only lane 0's return value is meaningful.
fn fixedScalarValue(scalar: u32, lane: u32, partialCount: u32) -> f32 {
  let limb = lane % FIXED_LIMBS;
  let group = lane / FIXED_LIMBS;
  var slice = 0;
  if (group < FIXED_LIMB_GROUPS) {
    for (var partial = group; partial < partialCount; partial += FIXED_LIMB_GROUPS) {
      slice += atomicLoad(&partials[fixedAt(partial, scalar, limb)]);
    }
  }
  fixedFoldShare[lane] = slice;
  workgroupBarrier();
  var value = 0.0;
  if (lane == 0u) {
    var totals: array<i32, ${SIGNED_RADIX_256_F32_LIMBS}>;
    for (var index = 0u; index < FIXED_LIMBS; index += 1u) {
      var total = 0;
      for (var slot = 0u; slot < FIXED_LIMB_GROUPS; slot += 1u) {
        total += fixedFoldShare[slot * FIXED_LIMBS + index];
      }
      totals[index] = total;
    }
    value = fixedDecodeLimbs(totals);
  }
  // Release the share before a caller folds its next scalar into it.
  workgroupBarrier();
  return value;
}
`;
}
