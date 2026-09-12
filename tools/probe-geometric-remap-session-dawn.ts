/** Runs the exact UI session through consecutive accepted steps and resets. */
import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { GeometricRemapSession, type RemapSettings, type RemapFrame } from "../lib/core/geometric-remap/session";
import { fingerprintSparseCM12RepositorySources } from "./sparse-cm12-source-content-fingerprint";

const output = resolve("artifacts/analytic-motion/large-remap-ui-session.json");
const report: Record<string, unknown> & { cases: Record<string, unknown>[] } = {
  probe: "geometric-remap-ui-session", passed: false, cases: [],
  sourceFingerprint: await fingerprintSparseCM12RepositorySources(process.cwd()),
};
let device: GPUDevice | undefined, locked = false;
const validationErrors: string[] = [];
try {
  await acquireWebGPUExclusiveLock("dawn-probe", "geometric-remap-ui-session"); locked = true;
  const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href) as NodeDawnProvider;
  Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  device = await adapter.requestDevice();
  device.addEventListener("uncapturederror", e => validationErrors.push(e.error.message));
  const session = await GeometricRemapSession.create(device);
  const cases: Array<{ id: string; settings: RemapSettings; steps: number }> = [
    { id: "fractional-translation", settings: { shape: "slab", travel: .5, deformation: 0 }, steps: 24 },
    { id: "large-translation", settings: { shape: "slab", travel: 25.5, deformation: 0 }, steps: 24 },
    { id: "nonlinear-slab", settings: { shape: "slab", travel: .5, deformation: 1 }, steps: 24 },
    { id: "nonlinear-oblique", settings: { shape: "oblique", travel: 25.5, deformation: 1 }, steps: 24 },
    { id: "full-domain", settings: { shape: "full", travel: 2.5, deformation: 1 }, steps: 16 },
  ];
  for (const c of cases) {
    session.reset(c.settings.shape);
    let frame: RemapFrame | undefined, maximumDrift = 0, maximumCellError = 0;
    const frames: Array<Record<string, unknown>> = [];
    const entry = { id: c.id, settings: c.settings, steps: c.steps, passed: false, frames,
      maximumDrift, maximumCellError };
    report.cases.push(entry);
    for (let step = 1; step <= c.steps; step++) {
      frame = await session.advance(c.settings);
      assert.equal(frame.step, step);
      maximumDrift = Math.max(maximumDrift, Math.abs(frame.relativeDrift));
      if (c.settings.deformation === 0) for (let cell = 0; cell < 512; cell++) {
        const left = (cell % 8) - c.settings.travel * step;
        let exact = 0;
        for (let k = Math.floor(left / 8) - 1; k <= Math.floor((left + 1) / 8) + 1; k++) {
          exact += Math.max(0, Math.min(left + 1, 4.75 + 8 * k) - Math.max(left, 2.25 + 8 * k));
        }
        maximumCellError = Math.max(maximumCellError, Math.abs(frame.volumes[cell]! - exact));
      }
      const { volumes: _volumes, ...receipt } = frame; frames.push(receipt);
    }
    assert.ok(maximumCellError <= 2e-5, `${c.id} translation shape mismatch ${maximumCellError}`);
    entry.maximumCellError = maximumCellError; entry.maximumDrift = maximumDrift; entry.passed = true;
    console.error(JSON.stringify({ id: c.id, step: frame!.step, maximumDrift, maximumCellError }));
  }
  session.reset("slab");
  const pending = session.advance({ shape: "slab", travel: .5, deformation: 1 });
  const reset = session.reset("full");
  await assert.rejects(pending, /cancelled by reset/);
  assert.equal(reset.step, 0); assert.equal(reset.volume, 512);
  const next = await session.advance({ shape: "full", travel: .5, deformation: 0 });
  assert.equal(next.step, 1); assert.ok(Math.abs(next.volume - 512) < .001);
  report.resetDuringStep = "passed";
  assert.deepEqual(validationErrors, []); report.passed = true;
} catch (e) { report.failure = e instanceof Error ? e.stack : String(e); }
finally {
  try { if (device) await device.queue.onSubmittedWorkDone(); }
  finally { device?.destroy(); if (locked) await releaseWebGPUExclusiveLock(); }
  report.validationErrors = validationErrors;
  report.sourceFingerprintAfter = await fingerprintSparseCM12RepositorySources(process.cwd());
  report.sourceUnchanged = (report.sourceFingerprint as { sha256: string }).sha256
    === (report.sourceFingerprintAfter as { sha256: string }).sha256;
  if (!report.sourceUnchanged) report.passed = false;
  await mkdir(dirname(output), { recursive: true }); await writeFile(output, JSON.stringify(report, null, 2));
}
console.log(JSON.stringify({ passed: report.passed, failure: report.failure, receipt: output }));
if (!report.passed) process.exitCode = 1;
