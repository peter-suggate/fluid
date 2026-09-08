import type { SimulationFailure } from "../../core/simulation-failure";

// Dedicated tail of topologyArena. Never cleared by frame/scratch initialization.
export const CM12_FAILURE_WORDS = 16;
export const CM12_FAILURE_BYTES = CM12_FAILURE_WORDS * 4;
const kernels = new Map<number, string>();
export function cm12FailureKernelId(name: string): number {
  let hash = 0x811c9dc5;
  for (const char of name) hash = Math.imul(hash ^ char.charCodeAt(0), 0x01000193) >>> 0;
  const prior = kernels.get(hash);
  if (prior && prior !== name) throw new Error(`CM12 failure kernel hash collision: ${prior}/${name}`);
  kernels.set(hash, name);
  return hash;
}

const reasons: Readonly<Record<number, readonly [string, string]>> = {
  1: ["INCIDENCE_RANGE", "Corrupt incidence range would have been replaced by an empty range"],
  2: ["EMPTY_DEFICIT_STENCIL", "Forward transport has no recipient support; donor self-return refused"],
  3: ["EMPTY_SHARPENING_STENCIL", "Sharpening has no recipient support; donor self-return refused"],
  5: ["TRANSPORT_STENCIL_GEOMETRY", "Could not locate a positive geometric transport stencil"],
  4: ["INVALID_CONSERVED_VALUE", "Nonfinite or negative transported density/gamma would have been clamped"],
  6: ["RETAINED_DENSITY_INTEGRAL", "Retained density support could not represent the accepted native amount"],
};
const currentFieldReasons: Readonly<Record<number, readonly [string, string]>> = {
  101: ["CURRENT_FIELD_NONFINITE", "Current spatial field contains a nonfinite value"],
  102: ["CURRENT_FIELD_TRACE_BUDGET", "Current spatial field trajectory exceeds the supported trace budget"],
  103: ["CURRENT_FIELD_ORIENTATION", "Current spatial field could not certify positive orientation over every map cell"],
  104: ["CURRENT_FIELD_COLLAR", "Current spatial field lost its identity exterior boundary"],
  105: ["CURRENT_FIELD_CHAIN_CAPACITY", "Current spatial field exhausted its certified increment storage"],
  108: ["CURRENT_FIELD_NATIVE_COVERAGE", "Current spatial field contains liquid outside accepted native support"],
  120: ["CURRENT_FIELD_INVALID_MEASURE", "Current spatial field integration produced an invalid measure"],
  121: ["CURRENT_FIELD_QUADRATURE", "Current spatial field integration did not meet its accuracy target"],
  124: ["CURRENT_FIELD_INCOMPLETE_DISPATCH", "Current spatial field did not complete every required GPU invocation"],
  123: ["CURRENT_FIELD_MATERIAL_COVERAGE", "Current spatial field could not bound omitted seed material within its coverage budget"],
};
function retainedFailureOperands(words: Uint32Array) {
  const stage = words[6];
  const floats = new Float32Array(words.slice(7, 10).buffer);
  switch (stage) {
    case 121: return { names: ["stage", "integrationError", "tolerance", "measureComponent"],
      values: [stage, ...floats] };
    case 124: return { names: ["stage", "receiptSlot", "completedInvocations", "expectedInvocations"], values: [...words.subarray(6, 10)] };
    case 123: return { names: ["stage", "omittedAmountBound", "seedAmount", "reserved"],
      values: [stage, floats[0], floats[1], words[9]] };
    case 2: return { names: ["stage", "targetDensity", "openFraction", "reserved"],
      values: [stage, floats[0], floats[1], words[9]] };
    case 3: return { names: ["stage", "density", "targetDensity", "previousDensity"],
      values: [stage, ...floats] };
    case 4: return { names: ["stage", "density", "openFraction", "hopDistance"],
      values: [stage, floats[0], floats[1], words[9]] };
    case 5: return { names: ["stage", "packetAmount", "hopDistance", "reserved"],
      values: [stage, floats[0], words[8], words[9]] };
    default: return { names: ["stage", "operand1", "operand2", "operand3"],
      values: [...words.slice(6, 10)] };
  }
}
export function decodeCM12SimulationFailure(words: Uint32Array, kernelNames: readonly string[] = []): SimulationFailure | undefined {
  if (words.length !== CM12_FAILURE_WORDS) throw new Error("Incomplete CM12 failure receipt");
  if (words[0] === 0) return undefined;
  for (const name of kernelNames) cm12FailureKernelId(name);
  const [code, message] = (words[1] === 6 ? currentFieldReasons[words[6]] : undefined)
    ?? reasons[words[1]] ?? ["UNKNOWN_GPU_FAULT", `Unknown GPU failure code ${words[1]}`];
  const retained = words[1] === 6 ? retainedFailureOperands(words) : undefined;
  return {
    method: "adaptive-mass", code, message,
    kernel: kernels.get(words[2]) ?? `0x${words[2].toString(16)}`,
    frame: words[3], generation: words[4], ownerId: words[5],
    operandNames: retained?.names ?? ({
      1: ["begin", "end", "maximumCount", "reserved"],
      2: ["visibleWeight", "deficit", "donorDensity", "reserved"],
      3: ["recipientWeight", "removedFixed", "reserved", "reserved"],
      5: ["positionX", "positionY", "positionZ", "reserved"],
      4: ["rawDensity", "rawGamma", "reserved", "reserved"],
    } as Record<number, string[]>)[words[1]],
    // Sharpening records its recipient weight as f32 and its mass receipt as
    // i32. Reinterpreting the latter as a float disguises two mass quanta as
    // 2.8e-45, making a real missing-recipient failure look like underflow.
    operands: retained?.values ?? (words[1] === 3
      ? [new Float32Array(words.slice(6, 7).buffer)[0], words[7] | 0, words[8], words[9]]
      : words[1] >= 2 && words[1] <= 5
      ? [...new Float32Array(words.slice(6, 10).buffer)] : [...words.slice(6, 10)]), rawWords: [...words],
  };
}
