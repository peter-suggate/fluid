import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";
import {
  OCTREE_RUNTIME_DIALS,
  OCTREE_RUNTIME_DIAL_BUILT_BOTTOM_SWEEPS,
  OCTREE_RUNTIME_DIAL_DEFAULTS,
  OCTREE_RUNTIME_DIAL_KEYS,
  octreeDialledIterationCap,
  octreeDialledRelativeTolerance,
  octreeRuntimeDialsAreDefault,
  resolveOctreeRuntimeDials,
} from "../lib/octree-runtime-dials";
import { octreeMethod } from "../lib/methods/octree";
import { residentLosassoMGPCGWGSL } from "../lib/webgpu-octree-losasso-resident-mgpcg";

// Offsets and capacities only: the WGSL generator reads the layout, never the
// arena buffer, so a parser/shape gate needs no GPU device.
const fusedLayout = {
  arena: undefined as unknown as GPUBuffer,
  rowCapacity: 1_152,
  faceCapacity: 4_096,
  transitionStrideWords: 65_536,
  controlOffsetWords: 0,
  rowOffsetsOffsetWords: 16,
  rowFacesOffsetWords: 4_096,
  facesOffsetWords: 12_288,
  parentsOffsetWords: 45_056,
  childOffsetsOffsetWords: 49_152,
  childListOffsetWords: 53_248,
  levelRowCapacities: [1_152, 288],
} as const;

const residentShader = (sharedBottom: boolean) => residentLosassoMGPCGWGSL({
  rowCapacity: 1_152,
  maximumIterations: 33,
  relativeTolerance: 1e-8,
  absoluteTolerance: 0,
  levelCount: 2,
  fused: fusedLayout,
  lanes: 1_024,
  sharedBottom,
});

test("an untouched dial bag reproduces the construction-time solver", () => {
  const dials = resolveOctreeRuntimeDials({});
  assert.ok(octreeRuntimeDialsAreDefault(dials));
  // The two "AUTO" dials must be transparent, not merely small: a fresh panel
  // may not quietly retune the authored tolerance or narrow the envelope.
  assert.equal(octreeDialledRelativeTolerance(1e-8, dials), 1e-8);
  assert.equal(octreeDialledIterationCap(33, dials), 33);
  assert.equal(dials.vcycleBottomSweeps, OCTREE_RUNTIME_DIAL_BUILT_BOTTOM_SWEEPS);
  assert.equal(dials.topologyRebuildCadence, 0);
});

test("dial values are clamped and snapped before they reach the GPU", () => {
  // An odd bottom-sweep count would leave the smoother's result in the wrong
  // ping-pong bank; the step snap is what makes that unreachable from a URL,
  // a stored profile, or a typed value, not just from the slider.
  assert.equal(resolveOctreeRuntimeDials({ vcycleBottomSweeps: 5 }).vcycleBottomSweeps, 6);
  assert.equal(resolveOctreeRuntimeDials({ vcycleBottomSweeps: 3 }).vcycleBottomSweeps, 4);
  assert.equal(resolveOctreeRuntimeDials({ vcycleBottomSweeps: 0 }).vcycleBottomSweeps, 2);
  assert.equal(resolveOctreeRuntimeDials({ vcycleBottomSweeps: 99 }).vcycleBottomSweeps, 8);
  assert.equal(resolveOctreeRuntimeDials({ velocityExtensionSweeps: 1 }).velocityExtensionSweeps, 2);
  assert.equal(resolveOctreeRuntimeDials({ topologyRebuildCadence: 40 }).topologyRebuildCadence, 8);
  // Non-numeric or absent values fall back rather than producing NaN uniforms.
  assert.deepEqual(resolveOctreeRuntimeDials({
    pressureToleranceDecades: Number.NaN,
    pressureIterationCap: "twelve",
  }), OCTREE_RUNTIME_DIAL_DEFAULTS);
});

test("the tolerance dial is decades on the authored scene value", () => {
  const looser = resolveOctreeRuntimeDials({ pressureToleranceDecades: 3 });
  assert.equal(octreeDialledRelativeTolerance(1e-8, looser), 1e-5);
  const stricter = resolveOctreeRuntimeDials({ pressureToleranceDecades: -2 });
  assert.ok(Math.abs(octreeDialledRelativeTolerance(1e-8, stricter) - 1e-10) < 1e-24);
  // Two scenes authoring different tolerances must both move by the same
  // ratio, which is the whole reason this is a multiplier and not an absolute.
  assert.ok(Math.abs(octreeDialledRelativeTolerance(1e-4, looser) - 1e-1) < 1e-12);
});

test("the iteration dial may only lower the compiled envelope", () => {
  // Widening it would let the loop outrun the control-word accounting and the
  // fail-closed exhaustion gate, both of which were sized at construction.
  assert.equal(octreeDialledIterationCap(33, resolveOctreeRuntimeDials({
    pressureIterationCap: 48,
  })), 33);
  assert.equal(octreeDialledIterationCap(33, resolveOctreeRuntimeDials({
    pressureIterationCap: 8,
  })), 8);
  assert.equal(octreeDialledIterationCap(33, resolveOctreeRuntimeDials({
    pressureIterationCap: 0,
  })), 33);
});

test("every dial is runtime-applied, so a drag cannot reset the simulation", () => {
  assert.deepEqual([...octreeMethod.runtimeParamKeys ?? []], [...OCTREE_RUNTIME_DIAL_KEYS]);
  for (const dial of OCTREE_RUNTIME_DIALS) {
    const spec = octreeMethod.params.find((candidate) => candidate.key === dial.key);
    assert.ok(spec, `${dial.key} must be a declared method parameter`);
    assert.equal(spec.update, "runtime",
      `${dial.key} must be runtime-applied or the controller resets t=0 on every drag`);
    assert.equal(spec.kind, "number");
    if (spec.kind !== "number") continue;
    assert.equal(spec.min, dial.min);
    assert.equal(spec.max, dial.max);
    assert.equal(spec.step, dial.step);
    assert.equal(spec.default, dial.default);
  }
  // A quality preset that pinned a dial would make the panel's value a lie.
  const preset = octreeMethod.presetFor("balanced");
  for (const key of OCTREE_RUNTIME_DIAL_KEYS) assert.ok(!(key in preset));
});

test("the resident kernel reads its dials through staged control words", () => {
  const shader = residentShader(true);
  // The envelope, tolerance and sweep count are staged, not baked. A literal
  // MAX_ITER/REL_TOL would silently make the panel inert.
  assert.match(shader, /const TUNING_STAGE = 40u;/);
  assert.match(shader, /fn maxIterations\(\)[\s\S]*?min\(requested, MAX_ITER_ENVELOPE\)/,
    "the dial must clamp to the compiled envelope inside the kernel too");
  assert.match(shader, /fn bottomSweeps\(\)[\s\S]*?tuned\(1u\) & 0xfffffffeu/,
    "an odd sweep request must round down inside the kernel");
  assert.match(shader, /fn relativeTolerance\(\)[\s\S]*?bitcast<f32>\(bits\)/);
  // Barrier-bearing loops must take workgroup-uniform bounds: a storage read
  // is not provably uniform and Tint rejects the kernel outright.
  assert.match(shader, /var<workgroup> wgMaxIterations: u32;/);
  assert.match(shader, /var<workgroup> wgBottomSweeps: u32;/);
  assert.match(shader,
    /let iterationCeiling = workgroupUniformLoad\(&wgMaxIterations\);/);
  assert.match(shader, /let sweeps = workgroupUniformLoad\(&wgBottomSweeps\);/);
  assert.match(shader, /for \(var iteration = 0u; iteration < iterationCeiling;/);
  assert.doesNotMatch(shader, /iteration < MAX_ITER;/);
  for (const variant of [true, false]) {
    assert.match(residentShader(variant), /for \(var sweep = 0u; sweep < sweeps;/,
      "both bottom smoothers must honour the sweep dial");
  }
});

test("the smoother absorbs an odd sweep count in its starting bank", () => {
  const shader = residentShader(true);
  // The whole reason odd counts are reachable: pre-smoothing starts from xB
  // when the count is odd, so the chain still ends in xA — which is where the
  // residual reads and where the coarse correction is accumulated.
  assert.match(shader,
    /var source = select\(L0_SRC_XA, L0_SRC_XB, \(smoothing & 1u\) == 1u\);/);
  assert.match(shader, /fn fusedSmoothToXA[\s\S]*?var fromB = \(sweeps & 1u\) == 1u;/);
  // Both banks must be seeded, or an odd chain's first read sees the previous
  // application's leftovers instead of a zero correction.
  assert.match(shader, /arena\[V_XA_BASE \+ row\] = 0\.;\s*\/\/[\s\S]*?arena\[V_XB_BASE \+ row\] = 0\.;/);
  assert.match(shader,
    /arena\[V_XA_BASE \+fusedVectorIndex\(coarseLevel, row\)\] = 0\.;\s*arena\[V_XB_BASE \+fusedVectorIndex\(coarseLevel, row\)\] = 0\.;/);
  // Prolongation has to be told which bank the coarse post-smoothing landed in.
  assert.match(shader, /fn fusedProlongInto\(fineLevel: u32, lane: u32, enabled: bool, coarseFromB: bool\)/);
  assert.match(shader, /fusedCorrectionValue\(coarseFromB, coarseLevel, parent\)/);
  // Publication can no longer test the source bank: an odd post-chain ends in
  // the other one, and Z is the only consumer either way.
  assert.match(shader, /if \(publish\) \{ arena\[Z_BASE \+ row\] = value; \}/);
  assert.doesNotMatch(shader, /publish && source == L0_SRC_XB/);
});

test("the standalone V-cycle sizes its dispatch count from both sweep dials", async () => {
  const { WebGPUOctreeLosassoVCycle } = await import("../lib/webgpu-octree-losasso-vcycle-gpu");
  const cycle = Object.create(WebGPUOctreeLosassoVCycle.prototype) as
    InstanceType<typeof WebGPUOctreeLosassoVCycle>;
  const shape = (levels: number, fused: boolean, smoothing: number, bottom: number) => {
    Object.assign(cycle, {
      hierarchy: { levels: Array.from({ length: levels }, () => ({})) },
      useFusedSubL0: fused, smoothingSweeps: smoothing, bottomSweeps: bottom,
    });
    return cycle.encodedCorrectionDispatchCount;
  };
  // The fused form runs its sub-L0 sweeps inside one workgroup, so only its
  // level-0 smoothing is encoded and the bottom dial cannot move the count.
  // Nine at the authored counts: the dial must not tax the default graph.
  assert.equal(shape(2, true, 2, 8), 9);
  assert.equal(shape(2, true, 2, 2), 9);
  // An odd count buys one extra clear — the xB seed — and still comes out
  // ahead of the even count above it.
  assert.equal(shape(2, true, 1, 8), 8);
  assert.equal(shape(2, true, 3, 8), 12);
  // The per-level path — what a domain too large for the fused walk runs —
  // spends a dispatch per sweep at EVERY level, which is what makes this the
  // biggest lever left on a big scene once iterations are already few.
  assert.equal(shape(4, false, 2, 8), 2 * 4 + 3 * 7 + 2 + 8);
  assert.equal(shape(4, false, 1, 8), 3 * 4 + 3 * 5 + 2 + 8);
  assert.ok(shape(4, false, 1, 8) < shape(4, false, 2, 8));
});

test("the resident kernel is accepted by naga at every sweep setting", () => {
  const naga = process.env.NAGA ?? "naga";
  if (spawnSync(naga, ["--version"], { encoding: "utf8" }).error) {
    console.log("skipped: naga is not on PATH");
    return;
  }
  const directory = mkdtempSync(join(tmpdir(), "fluid-resident-dials-wgsl-"));
  try {
    for (const sharedBottom of [true, false]) {
      const path = join(directory, `resident-${sharedBottom}.wgsl`);
      writeFileSync(path, residentShader(sharedBottom));
      const result = spawnSync(naga, [path], { encoding: "utf8" });
      assert.equal(result.status, 0,
        `sharedBottom=${sharedBottom}:\n${result.stderr || result.stdout}`);
    }
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
