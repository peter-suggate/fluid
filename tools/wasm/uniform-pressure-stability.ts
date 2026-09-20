/** Real authored Figure 9 scene: native and shipped Wasm must survive the old failure. */
import assert from "node:assert/strict";
import { readFileSync, writeFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { spawnSync } from "node:child_process";
import { createMassConservingFigure9DamBreak } from "../../lib/core/paper-scenarios";
import { uniformLabSeed, UNIFORM_LAB_VALUES } from "../../lib/physics-wasm/uniform-controller";
const request = { ...uniformLabSeed(createMassConservingFigure9DamBreak()), options: UNIFORM_LAB_VALUES, dt: 1 / 30, frames: 180 };
const build = spawnSync("cargo", ["build", "--release", "--manifest-path", "rust/Cargo.toml", "-p", "fluid-core", "--example", "uniform_geometric_scene", "--example", "uniform_geometric_pressure"], { stdio: "inherit" });
assert.equal(build.status, 0);
const frozen = spawnSync("rust/target/release/examples/uniform_geometric_pressure", [], {
  input: gunzipSync(readFileSync(new URL("../../docs/research/uniform-geometric-fig9-instability-2026-09-20/frame-104-pressure.json.gz", import.meta.url))), encoding: "utf8", maxBuffer: 8 * 1024 * 1024,
});
assert.equal(frozen.status, 0, frozen.stderr);
const replay = JSON.parse(frozen.stdout);
assert.ok(replay.receipt.residual < 1, "original failing matrix must converge with two Full-Cycles");
assert.equal(replay.receipt.rejectedCycles, 0, "full-depth bounds repair the matrix without fallback");
assert.ok(replay.pressure.every((p: number) => Number.isFinite(p) && Math.abs(p) < 1e6));
const native = spawnSync("rust/target/release/examples/uniform_geometric_scene", [], { input: JSON.stringify(request), encoding: "utf8", maxBuffer: 32 * 1024 * 1024 });
assert.equal(native.status, 0, native.stderr);
const expected = JSON.parse(native.stdout);
delete expected.elapsedMs;
function verify(output: typeof expected) {
  assert.equal(output.receipts.length, 180);
  for (const r of output.receipts) {
    assert.ok(Number.isFinite(r.maxSpeed) && r.maxSpeed < 100, `frame ${r.frame}: speed ${r.maxSpeed}`);
    assert.ok(r.phiArea > 0.75 * output.initialVolume && r.phiArea < 1.25 * output.initialVolume, `frame ${r.frame}: surface survives`);
    assert.ok(r.pressure.residual <= r.pressure.initialResidual, `frame ${r.frame}: rejected pressure must not be published`);
    assert.ok(r.pressure.recoverySweeps <= 64);
  }
  for (const field of [output.volume, output.phi, output.pressure, output.velocity.flat()]) assert.ok(field.every(Number.isFinite));
  const last = output.receipts.at(-1);
  assert.ok(Math.abs(last.volume / output.initialVolume - 1) < 1e-5);
}
verify(expected);
const artifacts = ["native"];
if (!process.argv.includes("--native-only")) for (const artifact of ["scalar", "simd"]) {
  const root = new URL(`../../public/wasm/fluid-wasm/${artifact}/`, import.meta.url);
  const wasm = await import(new URL("fluid_wasm.js", root).href);
  await wasm.default({ module_or_path: readFileSync(new URL("fluid_wasm_bg.wasm", root)) });
  const output = JSON.parse(wasm.run_uniform_geometric_scene(JSON.stringify(request)));
  verify(output);
  assert.deepEqual(output, expected, `${artifact}: exact native parity`);
  artifacts.push(artifact);
}
const report = { frozenPressure: replay.receipt, artifacts, frames: request.frames, peakSpeed: Math.max(...expected.receipts.map((r: { maxSpeed: number }) => r.maxSpeed)), receipts: expected.receipts };
if (process.env.FLUID_UNIFORM_STABILITY_REPORT) writeFileSync(process.env.FLUID_UNIFORM_STABILITY_REPORT, JSON.stringify(report, null, 2) + "\n");
console.log(JSON.stringify({ ...report, receipts: undefined, last: expected.receipts.at(-1) }));
