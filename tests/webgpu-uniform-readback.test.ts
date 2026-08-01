import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { publishedGlobalFineVolumeCells, sparseSurfaceVolumeCells } from "../lib/webgpu-uniform-eulerian";

const source = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");

test("compact octree never constructs or dispatches dense velocity telemetry", () => {
  const stats = source.slice(source.indexOf("async readStats()"), source.indexOf("\n  destroy()"));
  const advance = source.slice(source.indexOf("advanceTo(time_s"), source.indexOf("async readStats()"));
  assert.match(advance, /if \(this\.hostAllocation\) \{[\s\S]*this\.reductionPipeline/,
    "only a real dense host may dispatch the dense reduction");
  assert.match(stats,
    /this\.info\.maxSpeed_m_s = this\.octreeProjection\s*\?\s*undefined\s*:/,
    "cleared dense scratch must not masquerade as octree velocity telemetry");
  assert.doesNotMatch(`${advance}\n${stats}`, /adaptiveFaceVelocityCutover/);
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
  words[7] = 0;
  assert.ok(publishedGlobalFineVolumeCells({ ...diagnostics, volumeControl: Array.from(words) }, 0.001),
    "paper-path volume measurement is valid telemetry without claiming a global correction");
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
  assert.match(stats,
    /surfaceDiagnosticsPromise\s*=\s*compactFineExpected\s*\?\s*undefined\s*:/,
    "committed compact-fine telemetry must not also submit the obsolete adaptive-surface readback");
  assert.match(stats, /compactVolumeTopology=compactFineExpected/);
  assert.match(stats, /published:compactVolumeTopology\.published/);
  assert.match(stats, /volumeControl:globalFineDiagnostics\.fineVolumeControl/,
    "compact volume telemetry must decode the actual shared GPU publication controls");
  assert.doesNotMatch(stats, /as unknown as GlobalFineVolumePublicationDiagnostics/,
    "a type assertion cannot synthesize the publication fields at runtime");
  assert.match(stats, /this\.info\.volumeCellSum=compactVolume\?\.volumeCells/);
  assert.match(stats, /if\(!this\.octreeProjection\)\{this\.info\.front_m/,
    "compact transport must not publish the cleared dense front reduction");
});

test("power generation telemetry comes from the accepted GPU epoch receipt", () => {
  const applyFine = source.slice(source.indexOf("private applyGlobalFineDiagnostics"),
    source.indexOf("private async validateInitialSparseAuthority"));
  const applyOctreeStart = source.indexOf("private applyOctreeInfo");
  const applyOctree = source.slice(applyOctreeStart,
    source.indexOf("private statsReadback", applyOctreeStart));
  assert.match(applyFine,
    /powerDiagramGeneration = this\.info\.structuredVelocityValid[\s\S]*velocity\[3\]/,
    "the accepted structured generation, not the host attempt stamp, owns telemetry");
  assert.match(applyOctree,
    /\.\.\.\(projection\.powerPublicationGeneration === undefined \? \{\}/,
    "an unavailable host generation must preserve the last observed GPU receipt");
});
