import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { emptyPerformanceReport } from "../lib/stores/diagnostics-store";

test("empty performance report claims no lane sample", () => {
  assert.deepEqual(emptyPerformanceReport, {
    methodId: "",
    context: "",
    capturedAt_ms: 0,
  });
});

test("controller accepts only exact independently typed lane observations", () => {
  const controller = readFileSync(new URL("../lib/simulation/controller.ts", import.meta.url), "utf8");
  assert.match(controller, /performanceTraceMatchesLane\(metrics\.cpu, "cpu", "main-thread"\)/);
  assert.match(controller, /performanceTraceMatchesLane\(physicsTrace, "gpu", "physics"\)/);
  assert.match(controller, /performanceTraceMatchesLane\(metrics\.presentation, "gpu", "presentation"\)/);
  assert.doesNotMatch(controller, /cpu[^;\n]*\+[^;\n]*(?:physics|presentation)|physics[^;\n]*\+[^;\n]*presentation/);
});

test("performance publication is throttled independently of per-frame diagnostics", () => {
  const controller = readFileSync(new URL("../lib/simulation/controller.ts", import.meta.url), "utf8");
  const store = readFileSync(new URL("../lib/stores/diagnostics-store.ts", import.meta.url), "utf8");

  assert.match(controller, /capturedAt_ms - this\.lastPerformanceReportAt_ms >= 100/);
  assert.match(controller, /if \(!reportDue\) return/);
  assert.match(store, /performanceReports\.slice\(-239\)/);
});
