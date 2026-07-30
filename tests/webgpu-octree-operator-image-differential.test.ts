import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  OCTREE_SPGRID_OPERATOR_IMAGE,
  SPGRID_MAXIMUM_ROW_CAPACITY,
  WebGPUOctreeSPGridVCycle,
  octreeSPGridAccurateOperatorShader,
  planOctreeSPGridVCycle,
} from "../lib/webgpu-octree-spgrid-vcycle";

/**
 * Differential harness for the per-epoch compiled Section 6.3 operator image
 * (POWER_LIQUIDS_ULTIMATE_M1MAX, Part F1 stage 2).
 *
 * The claim under test is that compiling the operator's addressing once per
 * accepted topology epoch is restructuring-only: the same coefficient, the same
 * operand, the same product, the same summation order, and the same fail-closed
 * reports — only the address arrives from a table instead of from a five-to-
 * seven-deep dependent chase.
 *
 * Three independent arms:
 *
 *  1. The addressing cannot diverge, because there is exactly one definition of
 *     it. `resolveDirectChannel` is the only place a stencil direction becomes a
 *     destination row, and both the inline walk and the image builder call it.
 *  2. The codec is lossless and report-faithful. Every outcome the resolution
 *     can produce is encoded by the builder and decoded by the consumer, and the
 *     decoded (report sequence, staged term) must equal the inline form's,
 *     including the exact f32 of the term.
 *  3. On Dawn (skipped without WEBGPU_NODE_MODULE) both addressings are applied
 *     to the same vectors on the same published topology and the staged terms
 *     must be bit-identical. `stageMergedBandTermsByChase` is the reference arm:
 *     it is the pre-image body, retained in the shader for exactly this.
 */

const IMAGE = OCTREE_SPGRID_OPERATOR_IMAGE;

/** The WGSL body of one function, brace-matched. */
function wgslFunctionBody(shader: string, name: string): string {
  const signature = new RegExp(`\\bfn\\s+${name}\\s*\\(`);
  const start = shader.search(signature);
  assert.notEqual(start, -1, `missing WGSL function ${name}`);
  const open = shader.indexOf("{", start);
  let depth = 1, cursor = open + 1;
  while (cursor < shader.length && depth > 0) {
    if (shader[cursor] === "{") depth += 1;
    else if (shader[cursor] === "}") depth -= 1;
    cursor += 1;
  }
  assert.equal(depth, 0, `unterminated WGSL function ${name}`);
  return shader.slice(open + 1, cursor - 1);
}

test("one definition resolves a stencil direction, and both operators call it", () => {
  const shader = octreeSPGridAccurateOperatorShader;
  assert.equal(shader.match(/\bfn resolveDirectChannel\s*\(/g)?.length, 1,
    "the destination resolution must have exactly one definition");
  assert.equal(shader.match(/\bfn pageSlotCoded\s*\(/g)?.length, 1,
    "the page/brick/rank walk must have exactly one definition");

  // The dependent part of the walk - page neighbour, origin compare, brick
  // masks, ranked slot - has ONE definition, and the epoch builder and the
  // inline walk both reach it. That is what stops the compiled address from
  // drifting away from the address the walk would have produced.
  assert.match(wgslFunctionBody(shader, "resolveDirectChannel"), /pageSlotCoded\(l,page,q,vec3u\(targetQ\)\)/);
  assert.match(wgslFunctionBody(shader, "pageSlot"), /pageSlotCoded\(l,page,origin,q\)/);
  assert.match(wgslFunctionBody(shader, "buildOperatorRow"), /resolveDirectChannel\(/,
    "the epoch builder must resolve through the shared definition");
  // The chase arm is deliberately independent: it is the reference the
  // differential compares against, so it must keep walking pageSlot itself.
  assert.match(wgslFunctionBody(shader, "stageDirectTermByChase"), /pageSlot\(l,page,q,vec3u\(targetQ\),row\)/,
    "the reference arm must retain the chase it is the reference for");
  // The remaining guards must be the same guards, in the same order, on both
  // sides. Compare the two decision sequences symbol for symbol.
  const guards = (body: string) => [
    /targetQ=vec3i\(q\)\+worldDirection\(canonicalDirection\(channel\)/,
    /any\(targetQ<vec3i\(0\)\)\|\|any\(targetQ>=/,
    /pageSlotCoded|pageSlot\(/,
    /FLAGS,slotBase,/, /MG_ONLY/, /OWNER,slotBase,/,
    /encoded==0u\|\|encoded>capacity\(\)/,
  ].map((pattern) => body.search(pattern));
  const resolverGuards = guards(wgslFunctionBody(shader, "resolveDirectChannel"));
  const chaseGuards = guards(wgslFunctionBody(shader, "stageDirectTermByChase"));
  assert.ok(resolverGuards.every((at) => at >= 0), "the resolver lost a guard");
  assert.ok(chaseGuards.every((at) => at >= 0), "the chase reference lost a guard");
  for (const sequence of [resolverGuards, chaseGuards]) {
    assert.deepEqual([...sequence].sort((a, b) => a - b), sequence,
      "both forms must apply the guards in the same order");
  }

  // The production direct-term stage reaches neither the resolution nor any
  // part of the page directory: that is the whole point of the image.
  const staged = wgslFunctionBody(shader, "stageDirectTerm");
  assert.doesNotMatch(staged, /resolveDirectChannel|pageSlot|pageFor\(|brickRecord|rankedSlotsBase/,
    "the compiled direct-term stage must retain no addressing chase");
  assert.match(staged, /let code=operatorRows\[image\+1u\+channel\];/,
    "the compiled direct-term stage must gather one linear image word");

  // Values and evaluation order are untouched: same coefficient load, same
  // operand, same single product expression.
  assert.match(staged, /term=c\*\(inputVector\[row\]-inputVector\[code\]\);/);
  assert.match(wgslFunctionBody(shader, "stageDirectTermByChase"),
    /term=c\*\(inputVector\[row\]-inputVector\[encoded-1u\]\);/);

  // u32 only. A float stored to the image and reloaded would end the term
  // expression and cost the fused multiply-add; POWER_LIQUIDS_ULTIMATE_M1MAX
  // refuted lever 10 measured that as a physics change, not a restructuring.
  assert.match(shader, /@binding\(13\) var<storage,read_write> operatorRows:array<u32>;/);
  assert.doesNotMatch(shader, /bitcast<f32>\(operatorRows|operatorRows\[[^\]]*\]\s*=\s*bitcast/,
    "no float may round-trip through the compiled operator image");
});

test("the compiled image ABI matches the shader's own constants", () => {
  const shader = octreeSPGridAccurateOperatorShader;
  assert.match(shader, new RegExp(`const OPERATOR_ROW_WORDS=${IMAGE.rowWords}u;`));
  assert.match(shader, /const CHANNEL_CODE_BASE=0xffff0000u;/);
  assert.match(shader, /const CHANNEL_SKIP=0xffff0000u;/);
  assert.match(shader,
    /fn channelCode\(primary:u32,secondary:u32\)->u32\{return CHANNEL_CODE_BASE\|\(secondary<<8u\)\|primary;\}/);
  assert.match(shader,
    /fn operatorRowBase\(row:u32\)->u32\{return\(operatorImageBank\(\)\*capacity\(\)\+row\)\*OPERATOR_ROW_WORDS;\}/);
  assert.match(shader,
    /fn persistentImagePredecessor[\s\S]*ROW_DELTA_VALID[\s\S]*encoded&ROW_DELTA_STRUCTURAL[\s\S]*value-1u/,
    "clean rows must carry the prior generation's exact compiled image by stable identity");
  assert.equal(IMAGE.codeBase, 0xffff_0000);
  assert.equal(IMAGE.skip, 0);
  // A row index can never be mistaken for a code. This is what lets the
  // consumer discriminate with a single compare.
  assert.ok(SPGRID_MAXIMUM_ROW_CAPACITY < IMAGE.codeBase,
    "every representable row index must sit below the code space");
  for (const stage of [IMAGE.outOfDomain, IMAGE.unresolvedSlot, IMAGE.invalidOwner,
    IMAGE.pageUnresolved]) {
    assert.ok(stage > 0 && stage < 256, "a report stage must fit the code's low byte");
  }
  // The stages the builder encodes are the stages the inline walk raises.
  const inline = wgslFunctionBody(octreeSPGridAccurateOperatorShader, "stageDirectTermByChase");
  for (const [name, stage] of [["outOfDomain", IMAGE.outOfDomain],
    ["unresolvedSlot", IMAGE.unresolvedSlot], ["invalidOwner", IMAGE.invalidOwner],
    ["pageUnresolved", IMAGE.pageUnresolved]] as const) {
    assert.match(inline, new RegExp(`reportAt\\(2u,${stage}u,row\\)`),
      `the chase arm must raise ${name} as stage ${stage}`);
  }
});

/** Every outcome `resolveDirectChannel` can produce for one channel. */
type Outcome =
  | { readonly kind: "page-unresolved" }
  | { readonly kind: "out-of-domain" }
  | { readonly kind: "unresolved-slot"; readonly pageStage: 0 | 21 | 22 | 23 }
  | { readonly kind: "multigrid-only" }
  | { readonly kind: "invalid-owner" }
  | { readonly kind: "destination"; readonly row: number };

type Staged = { readonly reports: readonly number[]; readonly term: number | "unwritten" };

/**
 * The inline form: stageDirectTermByChase's decision tree, transcribed from the
 * shader body this test also pins textually above.
 */
function stagedByChase(outcome: Outcome, c: number, x: number, vector: Float32Array): Staged {
  if (c === 0) return { reports: [], term: 0 };
  if (outcome.kind === "page-unresolved") {
    return { reports: [IMAGE.pageUnresolved], term: "unwritten" };
  }
  if (outcome.kind === "out-of-domain") return { reports: [IMAGE.outOfDomain], term: 0 };
  if (outcome.kind === "unresolved-slot") {
    // pageSlot raises its own stage first, then the caller raises 28.
    const reports = outcome.pageStage === 0
      ? [IMAGE.unresolvedSlot] : [outcome.pageStage, IMAGE.unresolvedSlot];
    return { reports, term: 0 };
  }
  if (outcome.kind === "multigrid-only") return { reports: [], term: 0 };
  if (outcome.kind === "invalid-owner") return { reports: [IMAGE.invalidOwner], term: 0 };
  return { reports: [], term: Math.fround(c * Math.fround(x - vector[outcome.row]!)) };
}

/** buildOperatorRow's encoding: (row status word, channel word). */
function compileImage(outcome: Outcome): { status: number; code: number } {
  const code = (primary: number, secondary: number) =>
    (IMAGE.codeBase | (secondary << 8) | primary) >>> 0;
  switch (outcome.kind) {
    case "page-unresolved":
      return { status: IMAGE.pageUnresolved, code: (IMAGE.codeBase | IMAGE.skip) >>> 0 };
    case "out-of-domain": return { status: 0, code: code(IMAGE.outOfDomain, 0) };
    case "unresolved-slot":
      return { status: 0, code: code(IMAGE.unresolvedSlot, outcome.pageStage) };
    case "multigrid-only": return { status: 0, code: (IMAGE.codeBase | IMAGE.skip) >>> 0 };
    case "invalid-owner": return { status: 0, code: code(IMAGE.invalidOwner, 0) };
    default: return { status: 0, code: outcome.row };
  }
}

/** stageDirectTerm's decoding, including reportChannelCode's replay order. */
function stagedByImage(image: { status: number; code: number }, c: number, x: number,
  vector: Float32Array, capacity: number): Staged {
  if (c === 0) return { reports: [], term: 0 };
  if (image.status !== 0) return { reports: [IMAGE.pageUnresolved], term: "unwritten" };
  if (image.code >>> 0 >= IMAGE.codeBase) {
    const secondary = (image.code >>> 8) & 0xff, primary = image.code & 0xff;
    const reports: number[] = [];
    if (secondary !== 0) reports.push(secondary);
    if (primary !== 0) reports.push(primary);
    return { reports, term: 0 };
  }
  assert.ok(image.code < capacity);
  return { reports: [], term: Math.fround(c * Math.fround(x - vector[image.code]!)) };
}

test("compiling and reading back the operator image reproduces the chase exactly", () => {
  const capacity = 512;
  const vector = new Float32Array(capacity);
  // Values chosen to make cancellation, rounding and sign all reachable rather
  // than to be pretty: the term equality below is a bit-for-bit f32 claim.
  let seed = 0x9e3779b1;
  const next = () => {
    seed = (Math.imul(seed, 1103515245) + 12345) >>> 0;
    return seed / 0x1_0000_0000;
  };
  for (let index = 0; index < capacity; index += 1) {
    vector[index] = Math.fround((next() - 0.5) * 2 ** (Math.floor(next() * 24) - 8));
  }
  const outcomes: Outcome[] = [
    { kind: "page-unresolved" }, { kind: "out-of-domain" },
    { kind: "unresolved-slot", pageStage: 0 }, { kind: "unresolved-slot", pageStage: 21 },
    { kind: "unresolved-slot", pageStage: 22 }, { kind: "unresolved-slot", pageStage: 23 },
    { kind: "multigrid-only" }, { kind: "invalid-owner" },
    ...Array.from({ length: capacity }, (_, row) => ({ kind: "destination", row }) as const),
  ];
  let compared = 0;
  for (const outcome of outcomes) {
    for (const c of [0, 1, -1, 0.1, 1e-30, 3.4e38, Math.fround(next() * 4 - 2)]) {
      for (const x of [0, 1, -2.5, Math.fround(next() * 4 - 2)]) {
        const chase = stagedByChase(outcome, Math.fround(c), Math.fround(x), vector);
        const image = stagedByImage(compileImage(outcome), Math.fround(c), Math.fround(x),
          vector, capacity);
        assert.deepEqual(image.reports, chase.reports,
          `report sequence differs for ${outcome.kind}`);
        if (chase.term === "unwritten" || image.term === "unwritten") {
          assert.equal(image.term, chase.term);
        } else {
          // Bit-for-bit, not approximately: the image changes addressing only.
          assert.equal(Object.is(image.term, chase.term)
            || (Number.isNaN(image.term) && Number.isNaN(chase.term)), true,
          `staged term differs for ${outcome.kind}: ${image.term} vs ${chase.term}`);
        }
        compared += 1;
      }
    }
  }
  assert.ok(compared >= (8 + capacity) * 7 * 4);
});

/** Every terminal outcome of resolveAdjointCandidate. Page-resolution reports
 * are carried as the secondary byte because pageSlot raised them before the
 * adjoint caller returned. */
type AdjointOutcome =
  | { readonly kind: "skip" }
  | { readonly kind: "page-overflow" }
  | { readonly kind: "unresolved"; readonly pageStage: 21 | 22 | 23 }
  | { readonly kind: "invalid-owner" }
  | { readonly kind: "edge"; readonly row: number; readonly channel: number };

function compileAdjointImage(outcome: AdjointOutcome): number {
  const code = (primary: number, secondary: number) =>
    (IMAGE.codeBase | (secondary << 8) | primary) >>> 0;
  switch (outcome.kind) {
    case "skip": return IMAGE.codeBase;
    case "page-overflow": return code(31, 0);
    case "unresolved": return code(0, outcome.pageStage);
    case "invalid-owner": return code(24, 0);
    case "edge": return (outcome.row | (outcome.channel << IMAGE.adjointChannelShift)) >>> 0;
  }
}

function stagedAdjointByChase(outcome: AdjointOutcome, c: number, x: number,
  vector: Float32Array): Staged {
  switch (outcome.kind) {
    case "skip": return { reports: [], term: 0 };
    case "page-overflow": return { reports: [31], term: 0 };
    case "unresolved": return { reports: [outcome.pageStage], term: 0 };
    case "invalid-owner": return { reports: [24], term: 0 };
    case "edge": return { reports: [], term: c > 0
      ? Math.fround(c * Math.fround(x - vector[outcome.row]!)) : 0 };
  }
}

function stagedAdjointByImage(code: number, coefficients: Float32Array, x: number,
  vector: Float32Array): Staged {
  if (code >= IMAGE.codeBase) {
    const secondary = (code >>> 8) & 0xff, primary = code & 0xff;
    return { reports: [...(secondary === 0 ? [] : [secondary]),
      ...(primary === 0 ? [] : [primary])], term: 0 };
  }
  const other = code & IMAGE.adjointRowMask;
  const channel = code >>> IMAGE.adjointChannelShift;
  const c = coefficients[channel]!;
  return { reports: [], term: c > 0
    ? Math.fround(c * Math.fround(x - vector[other]!)) : 0 };
}

test("compiling and reading the fine-adjoint image reproduces its chase exactly", () => {
  const shader = octreeSPGridAccurateOperatorShader;
  assert.equal(shader.match(/\bfn resolveAdjointCandidate\s*\(/g)?.length, 1);
  assert.match(wgslFunctionBody(shader, "resolveAdjointCandidate"),
    /pageSlotCoded\(fine,ghostPage,ghostQ,ghostQ\)[\s\S]*pageSlotCoded\(fine,ghostPage,ghostQ,vec3u\(activeQ\)\)/,
    "both adjoint address chases must share the direct image's single resolver");
  assert.doesNotMatch(wgslFunctionBody(shader, "stageAdjointCandidate"),
    /pageSlot|pageFor\(|brickRecord|rankedSlotsBase/,
    "the production adjoint stage must retain no topology chase");
  assert.match(wgslFunctionBody(shader, "stageAdjointCandidateByChase"),
    /pageSlot\(fine,ghostPage,ghostQ,ghostQ,row\)[\s\S]*pageSlot\(fine,ghostPage,ghostQ,vec3u\(activeQ\),row\)/,
    "the differential reference must retain both replaced chases");
  assert.doesNotMatch(shader,
    /bitcast<f32>\(adjointRows|adjointRows\[[^\]]*\]\s*=\s*bitcast/,
    "the adjoint image must remain an index/channel table, never a rounded term table");

  const capacity = 257;
  const vector = new Float32Array(capacity);
  for (let row = 0; row < capacity; row += 1) {
    vector[row] = Math.fround(Math.sin(row * 1.61803398875) * 2 ** ((row % 17) - 8));
  }
  const coefficients = new Float32Array(18);
  for (let channel = 0; channel < coefficients.length; channel += 1) {
    coefficients[channel] = Math.fround(channel % 5 === 0 ? 0 : (channel + 1) / 37);
  }
  const fixed: AdjointOutcome[] = [
    { kind: "skip" }, { kind: "page-overflow" },
    { kind: "unresolved", pageStage: 21 }, { kind: "unresolved", pageStage: 22 },
    { kind: "unresolved", pageStage: 23 }, { kind: "invalid-owner" },
  ];
  const outcomes: AdjointOutcome[] = [...fixed];
  for (let row = 0; row < capacity; row += 1) for (let channel = 0; channel < 18; channel += 1) {
    outcomes.push({ kind: "edge", row, channel });
  }
  const x = Math.fround(-13.375);
  for (const outcome of outcomes) {
    const code = compileAdjointImage(outcome);
    const image = stagedAdjointByImage(code, coefficients, x, vector);
    const c = outcome.kind === "edge" ? coefficients[outcome.channel]! : 0;
    const chase = stagedAdjointByChase(outcome, c, x, vector);
    assert.deepEqual(image.reports, chase.reports, `adjoint reports differ for ${outcome.kind}`);
    assert.equal(Object.is(image.term, chase.term), true,
      `adjoint term differs for ${outcome.kind}: ${image.term} vs ${chase.term}`);
  }
  assert.equal(outcomes.length, fixed.length + capacity * 18);
});

test("the image is compiled once per epoch, in the commit that publishes it", () => {
  const commit = WebGPUOctreeSPGridVCycle.prototype.encodeReadySetupCommit.toString();
  assert.match(commit, /accurateOperatorRowsPipeline/,
    "the compile must ride the topology commit, which is the only writer of its inputs");
  assert.match(commit, /commitCandidateLevels/);
  assert.ok(commit.indexOf("commitCandidateLevels") < commit.indexOf("accurateOperatorRowsPipeline"),
    "the compile must follow the level commit inside the same pass");
  // Nothing per-apply may rebuild it; that would put the chase back on the
  // solve's critical path once per SpMV, which is the cost this change exists
  // to delete (~100 SpMV-equivalents x 11 iterations per frame).
  type Encoders = Record<string, (...args: never[]) => void>;
  const prototype = WebGPUOctreeSPGridVCycle.prototype as unknown as Encoders;
  for (const encode of ["encodeAccurateWorksets", "encodeAccurateMergedBandWorkset",
    "encodeCorrection"]) {
    assert.equal(typeof prototype[encode], "function", `missing ${encode}`);
    assert.doesNotMatch(prototype[encode]!.toString(), /accurateOperatorRowsPipeline/,
      `${encode} must consume the compiled image, never rebuild it`);
  }
  assert.match(WebGPUOctreeSPGridVCycle.prototype.constructor.toString(),
    /buildAccurateOperatorRows/,
    "the compile pipeline must be constructed from the accurate operator module");
  // The reference arm is reachable only through an explicit measurement flag;
  // the default constructor path remains the compiled image. Keeping the arm in
  // the same build is what makes an interleaved performance differential valid.
  const source = [prototype.encodeAccurateWorksets, prototype.encodeAccurateMergedBandWorkset,
    WebGPUOctreeSPGridVCycle.prototype.constructor].map((fn) => fn.toString()).join("\n");
  assert.match(source,
    /this\.directByChase\?"stageMergedBandTermsByChase":"stageMergedBandTerms"/,
    "the direct chase must remain an explicit opt-in measurement arm");
  assert.match(wgslFunctionBody(octreeSPGridAccurateOperatorShader,
    "stageAcceptedUnionTermsByChase"), /stageUnionItemByChase/,
    "the class apply and merged-band apply must price the same chase");
});

test("Dawn applies both addressings to the same vectors and stages identical terms", {
  skip: !process.env.WEBGPU_NODE_MODULE
    && "set WEBGPU_NODE_MODULE for the operator-image differential",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const nativeGpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await nativeGpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice({
    requiredLimits: { maxStorageBuffersPerShaderStage: 10 },
  });

  // One 8x8x4 page, three level-0 rows at (1,1,1), (2,1,1) and (1,2,1). Every
  // one of the eighteen canonical directions from those cells stays inside the
  // page AND inside brick (0,0,0), so the whole stencil resolves and neither
  // arm may raise a report. That is what makes the comparison deterministic:
  // `stopped()` short-circuits both stages, so a reporting arena would make the
  // written set schedule-dependent.
  const dimensions = [8, 8, 4] as const;
  const rowCapacity = 64, rowCount = 3, generation = 0x51a7;
  const plan = planOctreeSPGridVCycle({ dimensions, rowCapacity });

  // The constructor's memoized sixteen-slot level tables, re-derived here with
  // the identical closed form so this harness addresses the arena exactly as
  // the shader's helpers do.
  const slots = 16;
  const maximumSparseSlots = (() => { let r = 1; while (r < rowCapacity * 16) r *= 2; return r; })();
  const levelCapacityAt = (level: number) => {
    if (level < plan.levelCount) return plan.levelCapacities[level]!;
    const scale = 2 ** level;
    const cells = dimensions.map((value) => Math.ceil(value / scale))
      .reduce((product, value) => product * value, 1);
    let r = 1; const bound = Math.min(maximumSparseSlots, 2 * cells); while (r < bound) r *= 2;
    return r;
  };
  const levelCaps: number[] = [], levelBases: number[] = [], brickOffsets: number[] = [];
  const pageOffsets: number[] = [], transferOffsets: number[] = [];
  let slotBase = 0, brickBase = 0, pageBase = 0, transferBase = 0;
  for (let level = 0; level < slots; level += 1) {
    const capacity = levelCapacityAt(level);
    const levelDimensions = dimensions.map((value) => Math.ceil(value / 2 ** level));
    levelCaps.push(capacity);
    levelBases.push(slotBase); slotBase += capacity;
    brickOffsets.push(brickBase);
    brickBase += levelDimensions.map((value) => Math.ceil(value / 4))
      .reduce((product, value) => product * value, 1);
    pageOffsets.push(pageBase);
    pageBase += Math.ceil(levelDimensions[0]! / 8) * Math.ceil(levelDimensions[1]! / 8)
      * Math.ceil(levelDimensions[2]! / 4);
    transferOffsets.push(transferBase);
    transferBase += Math.min(plan.transferStride, capacity * 8) * 4 + 4 * capacity;
  }

  const rowMapBase = 16;
  const workBase = rowMapBase + plan.levelCount * rowCapacity;
  const pageWorkBase = workBase + plan.totalLevelSlots;
  const pageDirectoryBase = pageWorkBase + 28 * plan.totalLevelSlots;
  const transferArenaBase = pageDirectoryBase + pageOffsets[plan.levelCount]!;
  const directoryBase = transferArenaBase + transferOffsets[plan.levelCount - 1]!;
  const rankedSlotsBase = directoryBase + 16 + 4 * brickOffsets[plan.levelCount]!;
  // Self-check on the whole address chain: the shader lays the eighteen
  // published stencil columns immediately after the ranked slot vector, and the
  // plan sizes the arena for exactly that. If any base above were wrong this
  // total would not land.
  assert.equal((rankedSlotsBase + plan.totalLevelSlots + 18 * plan.totalLevelSlots) * 4,
    plan.topologyBytes, "the harness address model disagrees with the planned arena");

  const topology = new Uint32Array(plan.topologyBytes / 4);
  topology[pageDirectoryBase + pageOffsets[0]!] = 0;      // logical page 0 -> physical 0
  topology[pageWorkBase] = 1;                              // physical page 0 key = coord (0,0,0)
  for (let ordinal = 0; ordinal < 27; ordinal += 1) topology[pageWorkBase + 1 + ordinal] = 0;
  const brick = directoryBase + 16;                        // level 0, brick (0,0,0)
  topology[brick + 1] = 0xffff_ffff; topology[brick + 2] = 0xffff_ffff; topology[brick + 3] = 0;
  for (let bit = 0; bit < 64; bit += 1) topology[rankedSlotsBase + levelBases[0]! + bit] = bit;

  const state = new Uint32Array(plan.stateBytes / 4);
  const span = plan.totalLevelSlots;
  for (let slot = 0; slot < 64; slot += 1) {
    state[1 * span + levelBases[0]! + slot] = 1;                       // FLAGS = ACTIVE
    state[24 * span + levelBases[0]! + slot] = (slot % rowCount) + 1;  // OWNER = row + 1
  }

  const geometry = new Uint32Array(rowCapacity * 4);
  const cells = [[1, 1, 1], [2, 1, 1], [1, 2, 1]] as const;
  for (let row = 0; row < rowCount; row += 1) {
    const [x, y, z] = cells[row]!;
    geometry[row * 4] = x + dimensions[0] * (y + dimensions[1] * z);
    geometry[row * 4 + 1] = 1;
  }
  const metrics = new Uint32Array(rowCapacity * 4);
  const metricFloats = new Float32Array(metrics.buffer);
  for (let row = 0; row < rowCapacity; row += 1) {
    metrics[row * 4 + 1] = 0x8000_0000; metricFloats[row * 4 + 2] = 1;
  }

  let seed = 0x2545_f491;
  const next = () => { seed = (Math.imul(seed, 1103515245) + 12345) >>> 0; return seed / 2 ** 32; };
  const coefficients = new Float32Array(2 * rowCapacity * 19);
  for (let row = 0; row < rowCount; row += 1) {
    coefficients[row * 19] = Math.fround(1 + next());
    for (let channel = 0; channel < 18; channel += 1) {
      // A third of the channels are exactly zero, which is the branch the
      // staged form skips without touching the image at all.
      coefficients[row * 19 + 1 + channel] = channel % 3 === 0 ? 0 : Math.fround(next() * 2 - 1);
    }
  }
  const inputVector = new Float32Array(rowCapacity);
  for (let row = 0; row < rowCapacity; row += 1) inputVector[row] = Math.fround(next() * 8 - 4);

  const worksetStride = 7 + rowCapacity, worksetBankStride = 5 * worksetStride;
  const worksets = new Uint32Array(2 * worksetBankStride);
  const worksetBase = 4 * worksetStride;                   // class 4, bank 0
  worksets[worksetBase] = generation; worksets[worksetBase + 1] = rowCount;
  worksets[worksetBase + 2] = rowCapacity;
  for (let row = 0; row < rowCount; row += 1) worksets[worksetBase + 7 + row] = row;
  const accepted = new Uint32Array(16);
  accepted.set([0, 0xffff_ffff, rowCount, generation, 0, 64], 0);

  const layout = new Uint32Array(24 + 5 * slots);
  layout.set([worksetStride, worksetBankStride, 0, 0,
    dimensions[0], dimensions[1], dimensions[2], rowCapacity,
    plan.levelCount, plan.levelStride, plan.totalLevelSlots, rowCapacity * 19], 0);
  layout.set(levelCaps, 16); layout.set(levelBases, 16 + slots);
  layout.set(brickOffsets, 16 + 2 * slots); layout.set(pageOffsets, 16 + 3 * slots);
  layout.set(transferOffsets, 16 + 4 * slots);

  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const upload = (label: string, data: Uint32Array | Float32Array, usage = storage) => {
    const buffer = device.createBuffer({ label, size: Math.max(16, data.byteLength), usage });
    device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
    return buffer;
  };
  const buffers: Record<number, GPUBuffer> = {
    0: upload("inputVector", inputVector),
    2: upload("solverControl", new Uint32Array(16)),
    3: upload("accepted", accepted),
    4: upload("worksets", worksets),
    5: upload("worksetLayout", new Uint32Array([worksetStride, worksetBankStride, 0, 0]),
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
    6: upload("topology", topology),
    7: upload("state", state),
    8: upload("geometry", geometry),
    9: upload("metrics", metrics),
    10: upload("coefficients", coefficients),
    11: upload("layout", layout, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
    12: upload("accurateTerms", new Float32Array(rowCapacity * 163 + 1)),
    13: upload("operatorRows", new Uint32Array(rowCapacity * 19)),
  };

  const module = device.createShaderModule({ code: octreeSPGridAccurateOperatorShader });
  const pipeline = (entryPoint: string) => device.createComputePipeline({
    layout: "auto", compute: { module, entryPoint },
  });
  const group = (compute: GPUComputePipeline, bindings: readonly number[]) =>
    device.createBindGroup({ layout: compute.getBindGroupLayout(0),
      entries: bindings.map((binding) => ({ binding, resource: { buffer: buffers[binding]! } })) });
  const build = pipeline("buildAccurateOperatorRows");
  const image = pipeline("stageMergedBandTerms");
  const chase = pipeline("stageMergedBandTermsByChase");
  const groups = {
    build: group(build, [6, 7, 8, 9, 11, 13]),
    image: group(image, [0, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13]),
    chase: group(chase, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
  };

  const termWords = rowCapacity * 163 + 1;
  const readback = device.createBuffer({ size: termWords * 4 + 64,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const run = async (arm: "image" | "chase") => {
    device.queue.writeBuffer(buffers[2]!, 0, new Uint32Array(16));
    device.queue.writeBuffer(buffers[12]!, 0, new Float32Array(termWords));
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    if (arm === "image") {
      pass.setPipeline(build); pass.setBindGroup(0, groups.build);
      pass.dispatchWorkgroups(Math.ceil(rowCapacity * 19 / 64), 1, 1);
    }
    pass.setPipeline(arm === "image" ? image : chase);
    pass.setBindGroup(0, arm === "image" ? groups.image : groups.chase);
    pass.dispatchWorkgroups(Math.ceil(rowCount * 18 / 64), 1, 1);
    pass.end();
    encoder.copyBufferToBuffer(buffers[12]!, 0, readback, 0, termWords * 4);
    encoder.copyBufferToBuffer(buffers[2]!, 0, readback, termWords * 4, 64);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = new Uint8Array(readback.getMappedRange().slice(0));
    readback.unmap();
    return { terms: copy.slice(0, termWords * 4),
      control: new Uint32Array(copy.buffer, termWords * 4, 16) };
  };

  const chased = await run("chase");
  const compiled = await run("image");
  assert.equal(chased.control[0], 0,
    "the reference arm must resolve every direction; a report makes the comparison schedule-dependent");
  assert.equal(compiled.control[0], 0, "the compiled arm must raise no report the chase did not");
  assert.deepEqual([...compiled.terms], [...chased.terms],
    "the compiled operator image must stage bit-identical terms");
  device.destroy();
});
