import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const read = (path: string) => readFileSync(new URL(path, import.meta.url), "utf8");

test("Losasso ghost conditioning reconstructs its dual from geometry", () => {
  const source = read("../lib/webgpu-octree-losasso-coarse-phi.wgsl.ts");
  const publish = source.slice(source.indexOf("fn publishLosassoGhostDistances"));
  assert.doesNotMatch(publish, /distance=1\.\/face\.inverseDistance/,
    "a conditioning pass must never use its own prior conditioned coefficient as geometry");
  assert.match(publish, /let dual=f32\(rowSize\)\*p\.cellSize/,
    "the staged missing neighbor has the same size, so the dual is one row width");
  assert.match(publish, /let faceToAir=\.5\*dual/);
  assert.match(publish, /theta=clamp\(-liquid\/\(airPhi-liquid\),1e-4,1\.\)/,
    "idempotence, not a raised theta floor, bounds repeated conditioning");
});

test("Losasso V-cycle transfers preserve aggregate constants", () => {
  const source = read("../lib/webgpu-octree-losasso-vcycle-gpu.ts");
  const transfers = source.slice(source.indexOf("function transferWGSL"));
  assert.match(transfers, /addFixedF32\(0u,parent,input\[row\]\)/,
    "restriction is the plain adjoint sum");
  assert.match(transfers, /output\[row\]\+=input\[parent\]/,
    "a constant coarse correction prolongs unchanged to every child");
  assert.doesNotMatch(transfers,
    /\(weights\[row\]\*coarseInverseVolumes\[parent\]\)\*input/);
});

test("Losasso face publication masks T-junction quadrants and stages walls closed", () => {
  const source = read("../lib/webgpu-octree-losasso-backend.wgsl.ts");
  assert.match(source, /struct FacePatchPlan \{ mask: u32, subdivision: u32 \}/);
  assert.match(source, /mask \|= 1u << \(a \+ 2u \* b\)/,
    "each uncovered negative subface gets its own publication bit");
  assert.match(source, /countOneBits\(positiveFacePlan\(header, axis\)\.mask\)/);
  assert.match(source, /if \(\(negativePlan\.mask & patchBit\) != 0u\)/);
  assert.equal((source.match(/select\(1\.0, 0\.0, closedBoundary\)/g) ?? []).length, 2,
    "both wall orientations must be staged with zero aperture");
  assert.match(source,
    /face\.openFraction=select\(openSum\/f32\(span\*span\),0\.,closedBoundary\)/,
    "solid conditioning must not reopen a staged container wall");
});

test("Losasso diagnostics and tripwires are no longer backend-gated", () => {
  const uniform = read("../lib/webgpu-uniform-eulerian.ts");
  const stats = uniform.slice(uniform.indexOf("async readStats()"), uniform.indexOf("\n  destroy()"));
  assert.match(stats,
    /globalFineDiagnosticsPromise = compactFineExpected[\s\S]*readGlobalFineLevelSetDiagnostics/);
  assert.doesNotMatch(stats,
    /globalFineDiagnosticsPromise = this\.octreeProjection\?\.coarseBackend === "power2017"/);
  assert.match(stats, /if \(failures\.length > 0\) \{[\s\S]*\[losasso-step-receipt\]/,
    "every bad receipt is emitted even after the first sequence fault");

  const smoke = read("../tools/webgpu-smoke-executor.ts");
  assert.match(smoke, /const tripwireCapacity = !tripwiresDisabled && method\.id === "octree"/);
  assert.doesNotMatch(smoke,
    /const tripwireCapacity =[^\n]*!losassoCutoverLane/);
  assert.match(smoke,
    /losassoCutoverLane\s*\? \["topology", "mgpcg", "fineWorklist"\]/);
});

test("Losasso returns to the shared sixteen-iteration hard ceiling", () => {
  const source = read("../lib/webgpu-octree.ts");
  const construction = source.slice(source.indexOf("this.losassoBackend ="),
    source.indexOf("this.pressureSolverControl", source.indexOf("this.losassoBackend =")));
  assert.match(construction,
    /maximumIterations: this\.solveTailPolicy\.hardOuterIterationCeiling/);
  assert.match(construction,
    /hardIterationCeiling: this\.solveTailPolicy\.hardOuterIterationCeiling/);
  assert.doesNotMatch(construction, /maximumIterations: 32|hardIterationCeiling: 32/);
});

test("Losasso resolves ambiguous redistance seeds without changing Power", () => {
  const redistance = read("../lib/webgpu-octree-fine-levelset-redistance.ts");
  assert.match(redistance,
    /axisPermutationInvariantSeeds \? "if\(tied\)\{return 6u<<24u;\}" : ""/,
    "an equal axis-distance crossing uses the centre sentinel on the symmetric lane");
  assert.match(redistance,
    /axisPermutationInvariantSeeds = false/,
    "shared and Power construction retains the historical axis tie rule by default");

  const octree = read("../lib/webgpu-octree.ts");
  const losasso = octree.slice(octree.indexOf("const redistanceOptions ="),
    octree.indexOf("this.globalFineRedistanceA", octree.indexOf("const redistanceOptions =")));
  assert.match(losasso, /axisPermutationInvariantSeeds: true/);
});

test("Losasso coarse advection averages exact tangential sampling ties", () => {
  const source = readFileSync(new URL(
    "../lib/webgpu-octree-losasso-dynamics.wgsl.ts", import.meta.url,
  ), "utf8");
  assert.match(source, /fn velocityAtGrid\(gridValue: vec3f\)/);
  assert.match(source, /grid\[component\] == floor\(grid\[component\]\)/);
  assert.match(source, /splitMask \|= 1u << tangent/);
  assert.match(source, /result\[axis\] = exactValue\(&exact\) \/ f32\(sampleCount\)/);
});

test("Losasso diagnostics do not interpret absent Power controls as failures", () => {
  const source = read("../lib/webgpu-uniform-eulerian.ts");
  assert.match(source,
    /const powerStructuredAuthority = this\.octreeProjection\?\.coarseBackend === "power2017"/);
  assert.match(source,
    /const liveFailure = powerStructuredAuthority && !!support/);
  assert.match(source,
    /const latchedFailure = powerStructuredAuthority &&/);
});
