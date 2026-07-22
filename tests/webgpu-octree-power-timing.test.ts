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
  assert.match(split, /\{ name: totalName, start, end, requiredBoundaries \}/);
  assert.match(split, /\{ name: firstName, start, end: boundary, requiredBoundaries:/);
  assert.match(split, /\{ name: secondName, start: boundary, end, requiredBoundaries:/);
  assert.match(uniformSource, /decodeGPUPhysicsTimestampSegments\(times, querySegments\)/,
    "shared-boundary ranges must use the guarded decoder");
});

test("power timing is encoded only for power projection and reaches the async decoder", () => {
  assert.match(uniformSource, /powerDiagramProjection === "off" \? undefined/);
  assert.match(uniformSource, /splitTiming\("pressure_ms", "powerAssembly_ms", "pressureSolve_ms"\)/);
  assert.match(uniformSource, /splitTimingWithFirst3\("projection_ms", "powerProjection_ms", "faceBand_ms",\s*"faceMarch_ms", "powerPublication_ms", "velocityProjection_ms"\)/);
  assert.match(uniformSource, /powerAssembly_ms: "powerAssembly"/);
  assert.match(uniformSource, /velocityProjection_ms: "velocityProjection"/);
});

test("Section 5 timing ranges share boundaries and reuse the existing async query readback", () => {
  const helpers = uniformSource.slice(
    uniformSource.indexOf("private splitTiming"),
    uniformSource.indexOf("private statsReadback()"),
  );
  assert.match(helpers, /private splitTiming3/);
  assert.match(helpers, /private splitTimingWithFirst3/);
  assert.match(helpers, /this\.queryCount \+ 4/);
  assert.match(helpers, /this\.queryCount \+ 5/);
  assert.doesNotMatch(helpers, /mapAsync|createBuffer|copyBufferToBuffer|queue\.submit/,
    "timestamp subdivisions must not add a GPU readback or submission");
  for (const field of ["fineTopology_ms", "fineTransport_ms", "fineRedistance_ms",
    "faceBand_ms", "faceMarch_ms", "powerPublication_ms"]) {
    assert.match(uniformSource, new RegExp(`${field}:`), `${field} must reach the asynchronous timing decoder`);
  }
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

test("Section 5 boundaries are written at the fine and face phase transitions", () => {
  const faceBand = octreeSource.slice(
    octreeSource.indexOf("private encodeGlobalFineFaceBand("),
    octreeSource.indexOf("/** Encode one independently fenceable Section 5 face-band checkpoint", octreeSource.indexOf("private encodeGlobalFineFaceBand(")),
  );
  assert.match(faceBand, /phase === "transition-adjacency"[^]*faceMarchStartWriteIndex/);
  assert.match(faceBand, /phase === "fast-march"[^]*powerPublicationStartWriteIndex/);

  const surface = octreeSource.slice(
    octreeSource.indexOf("  encodeSurface("),
    octreeSource.indexOf("  encodeSurfaceBand(", octreeSource.indexOf("  encodeSurface(")),
  );
  const transport = surface.indexOf("transport.encode(encoder");
  const topologyBoundary = surface.indexOf("beginFineTopologyTiming()", transport);
  const topology = surface.indexOf("publicationTopology.encode(encoder", topologyBoundary);
  const redistanceBoundary = surface.indexOf("beginFineRedistanceTiming()", topology);
  const redistance = surface.indexOf("publicationRedistance.encode(encoder", redistanceBoundary);
  assert.ok(transport >= 0 && topologyBoundary > transport && topology > topologyBoundary
    && redistanceBoundary > topology && redistance > redistanceBoundary);
});
