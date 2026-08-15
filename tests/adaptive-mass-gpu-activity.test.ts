import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { webgpuSparseCM12ResidentWGSL } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl";
import {
  evaluateSparseCM12Performance,
  SPARSE_CM12_MINI_DAM_32_PERFORMANCE_ACCEPTANCE,
  type SparseCM12BenchmarkArm,
} from "../lib/methods/adaptive-mass/adaptive-mass-performance";

test("resident activity measurement is GPU-owned and disjoint from accepted fields", () => {
  assert.match(webgpuSparseCM12ResidentWGSL,
    /@group\(0\)@binding\(12\)var<storage,read_write>activity:array<atomic<u32>>/);
  assert.match(webgpuSparseCM12ResidentWGSL, /fn advanceActivityClock\(\)/);
  const begin = webgpuSparseCM12ResidentWGSL.indexOf("fn measureBrickActivity");
  const end = webgpuSparseCM12ResidentWGSL.indexOf(
    "fn classifyPresentationBricks", begin,
  );
  assert.ok(begin >= 0 && end > begin, "activity kernel must be independently inspectable");
  const kernel = webgpuSparseCM12ResidentWGSL.slice(begin, end);
  assert.match(kernel, /atomicStore\(&activity\[/);
  assert.doesNotMatch(kernel, /state\[[^\]]+\]\s*=(?!=)/,
    "measurement must not mutate accepted physics state");
  assert.doesNotMatch(kernel, /topology\[[^\]]+\]\s*=(?!=)/,
    "measurement must not mutate accepted topology");
});

test("frame scheduling never waits for or consumes activity readback", () => {
  const source = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver.ts",
    import.meta.url,
  ), "utf8");
  const advanceBegin = source.indexOf("  advanceTo(");
  const advanceEnd = source.indexOf("  private finishFrameCapture(", advanceBegin);
  const advance = source.slice(advanceBegin, advanceEnd);
  assert.doesNotMatch(advance, /readGPUActivityPolicy|readActivitySnapshot|mapAsync/);
  assert.match(source, /Explicit acceptance\/debug readback; never consulted by advanceTo/);
});

test("mini32 performance accepts a Rung-A fine set without freezing the old one-seed count", () => {
  const arm = (methodId: "uniform" | "adaptive-mass"): SparseCM12BenchmarkArm => ({
    methodId,
    sceneId: "minimal-power-dam-break-32",
    finestDimensions: [32, 32, 32],
    dt_s: 0.004,
    constructionExcluded: true,
    endToEndFrame_ms: new Array(30).fill(methodId === "uniform" ? 10 : 9),
    cpuTraces: [],
    gpuTraces: [],
    initialTopology: methodId === "adaptive-mass" ? {
      fineBricks: 28,
      coarseBricks: 36,
      fineCoarseFaceConnectedPairs: 24,
      mixedSeamRows: 384,
    } : undefined,
    evolvedTopology: methodId === "adaptive-mass" ? [{
      fineBricks: 28,
      coarseBricks: 36,
      fineCoarseFaceConnectedPairs: 24,
      mixedSeamRows: 384,
    }] : undefined,
  });
  const verdict = evaluateSparseCM12Performance(
    arm("uniform"), arm("adaptive-mass"),
    SPARSE_CM12_MINI_DAM_32_PERFORMANCE_ACCEPTANCE,
  );
  assert.equal(verdict.passed, true, verdict.failures.join("\n"));
});
