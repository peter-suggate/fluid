import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const host = readFileSync(`${root}/lib/webgpu-uniform-pressure-multigrid.ts`, "utf8");
const shader = readFileSync(`${root}/lib/webgpu-uniform-pressure-multigrid.wgsl.ts`, "utf8");

test("CM11a bakes immutable per-level coefficients before the fixed cycle schedule", () => {
  assert.match(host, /"mgBakeCoefficients"/);
  assert.match(host, /coefficients: texture\(`Uniform CM11a L\$\{index\} coefficients`, "rgba32float", size\)/);
  assert.match(host, /emit\("mgExtrapolatePhiOneCell"[^]*emit\("mgBakeCoefficients"[^]*planStage = "full-cycle"/);
  assert.match(host, /mgResidual: \[0, 1, 3, 10, 14\]/);
  assert.match(host, /mgSmoothColour: \[0, 1, 2, 3, 11, 13, 14\]/);
});

test("smoother and residual operator read the bake instead of rebuilding face geometry", () => {
  assert.match(shader, /fn mgBakeCoefficients[^]*mgCoefficientRaw/);
  assert.match(shader, /fn mgCoefficient\([^]*textureLoad\(mgCoefficientsIn/);
  assert.match(shader, /fn mgApply[^]*mgBakedLiquid/);
  assert.match(shader, /fn mgSmoothColour[^]*mgBakedLiquid/);
  assert.doesNotMatch(shader, /fn mgSmoothColour[^]*mgCoefficientRaw/);
});

test("ghost-fluid face coefficients are symmetric in liquid orientation", () => {
  assert.match(shader,
    /let idLiquid=mgLiquid\(id\);let qLiquid=mgLiquid\(q\);var theta=1\.0;/);
  assert.match(shader, /if\(idLiquid&&!qLiquid\)\{theta=mgTheta\(id,q\);\}/);
  assert.match(shader, /if\(!idLiquid&&qLiquid\)\{theta=mgTheta\(q,id\);\}/,
    "an air-owned face with liquid on its positive side needs the same theta as projection");
  assert.doesNotMatch(shader, /select\(mgTheta\(id,q\),1\.0,mgLiquid\(q\)\)/);
});

test("CM11a excludes RHS and residual forcing on inactive air rows", () => {
  assert.match(shader,
    /if\(mgInterior\(id,mg\.levelDims\.xyz\)\)\{\s*minimum=select\([^;]+;\s*if\(pressureLiquid\(simulation\)\)\{/,
    "the finest RHS must exist only on pressure unknown rows");
  assert.match(shader,
    /let residual=select\(0\.0,textureLoad\(mgRhsIn,id,0\)\.x-mgApply\(id\),mgBakedLiquid\(id\)\)/,
    "inactive coarse rows must contribute zero residual before restriction");
  assert.doesNotMatch(shader,
    /textureStore\(mgResidualOut,id,vec4f\(textureLoad\(mgRhsIn,id,0\)\.x-mgApply\(id\)\)\)/);
});
