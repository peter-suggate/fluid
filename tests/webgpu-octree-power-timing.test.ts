import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const uniformSource = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");
const octreeSource = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");

test("power timing subdivisions share one query boundary with their aggregate", () => {
  const splitStart = uniformSource.indexOf("private splitTiming");
  const split = uniformSource.slice(
    splitStart,
    uniformSource.indexOf("private statsReadback()", splitStart),
  );
  assert.match(split, /this\.queryCount \+ 3/);
  assert.match(split, /\{ name: totalName, start, end \}/);
  assert.match(split, /\{ name: firstName, start, end: boundary \}/);
  assert.match(split, /\{ name: secondName, start: boundary, end \}/);
});

test("power timing is encoded only for power projection and reaches the async decoder", () => {
  assert.match(uniformSource, /powerDiagramProjection === "off" \? undefined/);
  assert.match(uniformSource, /splitTiming\("pressure_ms", "powerAssembly_ms", "pressureSolve_ms"\)/);
  assert.match(uniformSource, /splitTiming\("projection_ms", "powerProjection_ms", "velocityProjection_ms"\)/);
  assert.match(uniformSource, /powerAssembly_ms: "powerAssembly"/);
  assert.match(uniformSource, /velocityProjection_ms: "velocityProjection"/);
});

test("power boundaries are written by existing solve and compatibility passes", () => {
  const encode = octreeSource.slice(
    octreeSource.indexOf("  encode(\n    encoder: GPUCommandEncoder"),
    octreeSource.indexOf("  /** Publish lazily allocated diagnostic textures", octreeSource.indexOf("  encode(\n    encoder: GPUCommandEncoder")),
  );
  const assembly = encode.indexOf("this.encodePowerAssemblyMirror(encoder)");
  const solve = encode.indexOf('beginComputePass({ label: "Octree leaf pressure solve"', assembly);
  assert.ok(assembly >= 0 && solve > assembly);
  assert.match(encode.slice(solve, solve + 220), /timestampWrites: pressureBoundary/);

  const powerProjection = encode.indexOf("this.encodePowerProjectionMirror(encoder");
  const compatibilityProjection = encode.indexOf('beginComputePass({ label: "Octree finite-volume velocity projection"', powerProjection);
  assert.ok(powerProjection >= 0 && compatibilityProjection > powerProjection);
  assert.match(encode.slice(compatibilityProjection, compatibilityProjection + 260), /timestampWrites: projectionBoundary/);
});
