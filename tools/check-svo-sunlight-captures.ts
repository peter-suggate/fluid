import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { compareSvoScreenSpaceImages } from "../lib/svo/features/lighting-visibility/svo-screen-space-termination";

const directory = process.argv[2];
assert.ok(directory, "usage: check-svo-sunlight-captures <comparison-directory> [width height]");
const width = Number(process.argv[3] ?? 800), height = Number(process.argv[4] ?? 460);
function decode(file: string): Float32Array {
  const bytes = readFileSync(path.join(directory, file));
  assert.equal(bytes.length, width * height * 8);
  const half = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.length / 2);
  return Float32Array.from(half, (word) => {
    const sign = word & 0x8000 ? -1 : 1;
    const exponent = (word >> 10) & 31, fraction = word & 1023;
    return exponent === 0 ? sign * fraction * 2 ** -24
      : exponent === 31 ? fraction ? NaN : sign * Infinity
      : sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
  });
}
const reports = [];
for (const file of readdirSync(directory).filter((name) => name.endsWith(".rgba16f") && !name.endsWith("-full.rgba16f"))) {
  const reference = file.replace(/-(half-cached|cached|half)\.rgba16f$/, "-full.rgba16f");
  assert.ok(reference !== file && existsSync(path.join(directory, reference)), `missing full-rate reference for ${file}`);
  const report = compareSvoScreenSpaceImages(decode(reference), decode(file), { width, height });
  reports.push({ file, reference, ...report });
}
assert.ok(reports.length >= 9, "expected at least the three static views and three comparison arms");
writeFileSync(path.join(directory, "linear-image-errors.json"), JSON.stringify(reports, null, 2));
// A coarse broad-coverage regression gate, not a claim of pixel equality or
// visual acceptance. The old container-clipped worker fails this luminance
// bound; the 2x2 reconstruction still needs inspection at fine silhouettes.
for (const report of reports) {
  assert.equal(report.absoluteDepthError.maximum, 0, `${report.file}: lighting changed geometry depth`);
  assert.ok(report.absoluteLuminanceError.mean <= 0.025, `${report.file}: mean linear luminance error ${report.absoluteLuminanceError.mean} > 0.025`);
  console.log(`${report.file}: mean luminance error ${report.absoluteLuminanceError.mean.toFixed(5)}, depth unchanged`);
}
