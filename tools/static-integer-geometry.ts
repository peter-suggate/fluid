/** Diagnostic equivalent of keeping integer scene roots specialized while
 * leaving work counts and floating spacing in the ownership uniform. Derived
 * geometry helpers deliberately remain functions, matching the production
 * runtime-constant transform's structure. */
import assert from "node:assert/strict";

/** Recover the previous runtime geometry functions from a specialized module,
 * so diagnostic input frames still use the original shader compilation path. */
export function dynamicIntegerGeometryWGSL(code: string): string {
  code = code.replace(/fn (\w*)UM_D\(\)->vec3u\{return vec3u\(\d+u,\d+u,\d+u\);\}/g,
    (_, prefix: string) => `fn ${prefix}UM_D()->vec3u{return ${prefix}umCounts.dimensions.xyz;}`);
  return code.replace(/fn (\w*)UM_TILES\(\)->u32\{return \d+u;\}/g,
    (_, prefix: string) => `fn ${prefix}UM_TILES()->u32{return ${prefix}umCounts.dimensions.w;}`);
}

export function staticIntegerGeometryWGSL(code: string, dimensions: readonly number[]): string {
  assert.equal(dimensions.length, 3);
  assert.ok(dimensions.every(n => Number.isInteger(n) && n > 0 && n % 4 === 0));
  const dims = `vec3u(${dimensions.map(n => `${n}u`).join(",")})`;
  const tiles = `${dimensions.reduce((a, b) => a * b, 1) / 64}u`;
  let replacements = 0;
  code = code.replace(/fn (\w*UM_D)\(\)->vec3u\{return \w*umCounts\.dimensions\.xyz;\}/g, (_, name: string) => {
    replacements++; return `fn ${name}()->vec3u{return ${dims};}`;
  });
  code = code.replace(/fn (\w*UM_TILES)\(\)->u32\{return \w*umCounts\.dimensions\.w;\}/g, (_, name: string) => {
    replacements++; return `fn ${name}()->u32{return ${tiles};}`;
  });
  assert.ok(replacements >= 2, "Expected runtime integer geometry roots");
  return code;
}
