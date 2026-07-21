import assert from "node:assert/strict";
import test from "node:test";
import {
  octreeDiagnosticShader,
  octreePressureCouplingShader,
  octreeProjectionShader,
} from "../lib/webgpu-octree";
import { octreeSurfacePageShader } from "../lib/webgpu-octree-surface-pages";

function wgslFunction(source: string, name: string): string {
  const start = source.indexOf(`fn ${name}`);
  assert.notEqual(start, -1, `missing WGSL function ${name}`);
  const body = source.indexOf("{", start);
  assert.notEqual(body, -1, `missing WGSL body for ${name}`);
  let depth = 0;
  for (let cursor = body; cursor < source.length; cursor += 1) {
    if (source[cursor] === "{") depth += 1;
    if (source[cursor] === "}") depth -= 1;
    if (depth === 0) return source.slice(start, cursor + 1);
  }
  assert.fail(`unterminated WGSL function ${name}`);
}

function occurrences(source: string, token: string): number {
  return source.split(token).length - 1;
}

function firstLoop(fn: string): string {
  const loop = fn.indexOf("loop");
  assert.notEqual(loop, -1, "missing WGSL loop");
  const body = fn.indexOf("{", loop);
  let depth = 0;
  for (let cursor = body; cursor < fn.length; cursor += 1) {
    if (fn[cursor] === "{") depth += 1;
    if (fn[cursor] === "}") depth -= 1;
    if (depth === 0) return fn.slice(loop, cursor + 1);
  }
  assert.fail("unterminated WGSL loop");
}

test("octree owner fallback loops do not append unreachable tail returns", () => {
  for (const [label, source] of [
    ["projection", octreeProjectionShader],
    ["pressure coupling", octreePressureCouplingShader],
    ["diagnostic overlay", octreeDiagnosticShader],
  ] as const) {
    const fn = wgslFunction(source, "canonicalOwner");
    assert.match(fn, /loop\s*\{/);
    assert.doesNotMatch(firstLoop(fn), /\breturn\b/, `${label} canonicalOwner loop must exit with break`);
    assert.equal(occurrences(fn, "return Owner("), 1, `${label} canonicalOwner must have one reachable tail return`);
  }
});

test("octree lock-free free-list loops do not append unreachable tail returns", () => {
  const ownerPop = wgslFunction(octreeProjectionShader, "popOwnerPage");
  assert.match(ownerPop, /loop\s*\{/);
  assert.doesNotMatch(firstLoop(ownerPop), /\breturn\b/);
  assert.match(ownerPop, /var physical = 0xffffffffu;/);
  assert.equal(occurrences(ownerPop, "return physical;"), 1);

  const surfacePop = wgslFunction(octreeSurfacePageShader, "popFree");
  assert.match(surfacePop, /loop\s*\{/);
  assert.doesNotMatch(firstLoop(surfacePop), /\breturn\b/);
  assert.equal(occurrences(surfacePop, "return slot;"), 1);
});
