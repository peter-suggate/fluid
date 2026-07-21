import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { publishedGlobalFineVolumeCells, sparseSurfaceVolumeCells } from "../lib/webgpu-uniform-eulerian";

const source = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");

test("uniform and adaptive rigid coupling stays resident while telemetry remains pooled", () => {
  const statsHelper = source.slice(source.indexOf("private statsReadback()"), source.indexOf("private retireQuadtreeProjection"));
  const stats = source.slice(source.indexOf("async readStats()"), source.indexOf("\n  destroy()"));
  const advance = source.slice(source.indexOf("advanceTo(time_s"), source.indexOf("async readStats()"));

  assert.match(statsHelper, /this\.statsReadbackBuffer \?\?=/, "statistics lazily allocate one staging buffer");
  assert.doesNotMatch(stats, /createBuffer\(/, "each statistics poll must not allocate a GPU buffer");
  assert.doesNotMatch(advance, /createBuffer\(/, "each rigid exchange must not allocate a GPU buffer");
  assert.doesNotMatch(advance, /mapAsync|getMappedRange|rigidReadbackPool|encodeBodyImpulseReadback/, "the physics path must not map rigid feedback");
  assert.match(advance, /encodeBodyImpulseExchange\(encoder, this\.rigidExchangeBuffer\)/, "quadtree pressure impulses remain in GPU storage");
  assert.match(advance, /this\.rigidSystem\.encode\(encoder, delta/, "the resident rigid solver consumes the exchange in the same command stream");
  assert.match(stats, /await mapPromise\.catch/, "a rejected diagnostic channel must wait for the pooled mapping to settle");
  assert.match(stats, /finally \{[\s\S]*this\.readbackPending = false/, "failed quadtree diagnostics release the pooled readback slot");
});

test("compact octree velocity telemetry bypasses dense velocity reduction", () => {
  const stats = source.slice(source.indexOf("async readStats()"), source.indexOf("\n  destroy()"));
  const advance = source.slice(source.indexOf("advanceTo(time_s"), source.indexOf("async readStats()"));
  assert.match(advance, /if \(!this\.adaptiveFaceVelocityCutover\)[\s\S]*this\.reductionPipeline/, "dense velocity reduction is disabled after compact-face cutover");
  assert.match(stats, /readAdaptiveFaceVelocityDiagnostics\(\)/, "telemetry reads the compact face reduction");
  assert.match(stats, /this\.info\.maxSpeed_m_s = faceVelocityDiagnostics\.maxSpeed_m_s/);
  assert.match(stats, /this\.info\.maxComponentCfl = faceVelocityDiagnostics\.maxComponentCfl/);
  assert.match(stats, /this\.info\.nonFiniteCount = faceVelocityDiagnostics\.nonFiniteCount/);
});

test("compact octree volume telemetry accepts only the current committed publication", () => {
  const bytes = new ArrayBuffer(64);
  const words = new Uint32Array(bytes);
  const floats = new Float32Array(bytes);
  words[0] = 0x8000_0000;
  words[1] = 1;
  words[2] = 8;
  floats[3] = 3;
  floats[4] = 2.7;
  words[7] = 1;
  words[11] = 12;
  words[13] = 7;
  const diagnostics = { published: true, rolledBack: false, downstreamFinalizeReason: 0,
    generation: 7, volumeControl: Array.from(words) };

  const accepted = publishedGlobalFineVolumeCells(diagnostics, 0.001);
  assert.ok(accepted);
  assert.equal(accepted.referenceVolumeCells, 3000);
  assert.ok(Math.abs(accepted.volumeCells - 2700) < 1e-3);
  assert.ok(Math.abs(accepted.drift + 0.1) < 1e-6);
  assert.equal(publishedGlobalFineVolumeCells({ ...diagnostics, rolledBack: true }, 0.001), undefined,
    "the shared control describes a rejected candidate after rollback");
  assert.equal(publishedGlobalFineVolumeCells({ ...diagnostics, generation: 8 }, 0.001), undefined,
    "a stale controller generation is not the current compact field");
  words[14] = 1;
  assert.equal(publishedGlobalFineVolumeCells({ ...diagnostics, volumeControl: Array.from(words) }, 0.001), undefined,
    "owner lookup failures invalidate the compact volume publication");
});

test("compact analytic surface telemetry restores the physical t=0 reference", () => {
  assert.deepEqual(sparseSurfaceVolumeCells({ referenceVolumeCells: 0, volumeCells: 0 }, 864), {
    referenceVolumeCells: 864,
    volumeCells: 864,
  });
  assert.deepEqual(sparseSurfaceVolumeCells({ referenceVolumeCells: 0, volumeCells: -2.5 }, 864), {
    referenceVolumeCells: 864,
    volumeCells: 861.5,
  });
  assert.deepEqual(sparseSurfaceVolumeCells({ referenceVolumeCells: 40, volumeCells: 39 }, 864), {
    referenceVolumeCells: 40,
    volumeCells: 39,
  });
});

test("compact octree readback never reports the cleared dense volume reduction", () => {
  const stats = source.slice(source.indexOf("async readStats()"), source.indexOf("\n  destroy()"));
  assert.match(stats, /compactFineExpected/);
  assert.match(stats, /publishedGlobalFineVolumeCells\(globalFineDiagnostics/);
  assert.match(stats, /this\.info\.volumeCellSum=compactVolume\?\.volumeCells/);
  assert.match(stats, /if\(!this\.adaptiveFaceVelocityCutover\)\{this\.info\.front_m/,
    "compact transport must not publish the cleared dense front reduction");
});
