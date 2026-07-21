import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { octreeProjectionShader, WebGPUOctreeProjection } from "../lib/webgpu-octree";
import { octreeSurfaceAdapterShader, octreeSurfaceCandidateShader } from "../lib/webgpu-octree-surface-adapter";

const projectionSource = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");

test("cold octree topology evaluates the authored analytic SDF without dense-phi authority", () => {
  assert.match(octreeProjectionShader,
    /fn analyticInitialPhi\(point: vec3f\)[\s\S]*heightFraction = max\(0\.92, fill\)[\s\S]*footprintFraction = sqrt/,
    "dam bootstrap must reproduce the authored fill-preserving analytic block");
  assert.match(octreeProjectionShader,
    /fn legacyPhi[\s\S]*analyticInitialPhiEnabled\(\)[\s\S]*frontierGeneration\(\) == 0u \|\| !pagedSurfaceBindings\(\) \|\| !pagedSurfaceAuthority\(\)[\s\S]*analyticInitialPhi/,
    "cold topology, non-page sizing groups, and an unpublished sparse-page generation must retain analytic bootstrap phi");
  assert.match(octreeProjectionShader,
    /fn pagedSurfaceAuthority\(\)[\s\S]*atomicLoad\(&solidOrSurface\[3\]\) == 0u[\s\S]*atomicLoad\(&solidOrSurface\[6\]\) > 0u[\s\S]*atomicLoad\(&solidOrSurface\[7\]\) > 0u/,
    "paged phi authority requires a fault-free, non-empty GPU-published generation");
  assert.match(octreeProjectionShader,
    /fn solidAt[\s\S]*if \(pagedSurfaceBindings\(\)\)[\s\S]*fn rasterizeSolidsAt[\s\S]*if \(pagedSurfaceBindings\(\)\)/,
    "the sparse page ABI must never be decoded as legacy solid-cell storage while authority is pending");
  assert.match(projectionSource,
    /hasImportedInitialSeeds[\s\S]*initialCondition === "dam-break"[\s\S]*-20 - this\.surfaceDetailStrength/,
    "dam/tank scenes select analytic bootstrap while imported seeded shapes retain compatibility staging");
  assert.match(projectionSource,
    /hasImportedInitialSeeds \|\| !this\.analyticSparseBootstrap[\s\S]*this\.surfaceDetailStrength/,
    "dense compatibility scenes must transport their level-set texture instead of retaining the analytic sparse sentinel");
  assert.doesNotMatch(octreeProjectionShader,
    /analyticInitialPhi\(point: vec3f\)[\s\S]{0,800}textureLoad/,
    "analytic initial classification must not sample the dense level-set texture");
});

test("first SurfaceLeaf and indexed fine-page generation share analytic bootstrap phi", () => {
  assert.match(octreeSurfaceAdapterShader,
    /fn analyticInitialPhi\(point:vec3f\)[\s\S]*heightFraction=max\(0\.92,fill\)[\s\S]*footprintFraction=sqrt/);
  assert.match(octreeSurfaceAdapterShader,
    /if\(!pagedPhiAvailable\(\)\)\{if\(params\.selection\.z!=0u\)\{return analyticInitialPhi/);
  assert.match(projectionSource,
    /analyticSparseBootstrap[\s\S]*new Float32Array\(\[Math\.max\(cell\.x, cell\.y, cell\.z\) \* this\.maxLeafSize\]\)[\s\S]*analyticInitialCondition/,
    "eligible analytic scenes must bind only one format texel while pages are bootstrapped");
});

test("eligible analytic scenes use the GPU-authored resident cold worklist", () => {
  assert.match(projectionSource,
    /if \(this\.analyticBootstrapWorklist\)[\s\S]*\.encode\(encoder\)[\s\S]*this\.topologyWorklistReady = true[\s\S]*this\.encodeInlineRebuild\(encoder\)/,
    "analytic bootstrap must enter the resident rebuild without a finest-domain cold dispatch");
  assert.match(projectionSource,
    /if \(analyticSparseBootstrap\)[\s\S]*planOctreeAnalyticBootstrapBounds[\s\S]*new WebGPUOctreeAnalyticBootstrapWorklist/);
  assert.match(projectionSource,
    /const topologyWorklistReady = this\.topologyWorklistReady[\s\S]*this\.topologyWorklistReady = false/,
    "non-analytic compatibility bootstrap must retain the existing full-domain fallback");
});

test("global fine bootstrap accepts interface seeds from coarse octree leaves", () => {
  const construction = WebGPUOctreeProjection.toString();
  assert.match(construction, /finestLeafSize:\s*this\.maxLeafSize/,
    "fine brick identity is global, so coarse interface leaves must remain eligible seeds");
  assert.match(octreeSurfaceCandidateShader,
    /fn selectSurfaceCandidates[\s\S]*leaf\.size>params\.selection\.x[\s\S]*candidateFlags==0u/);
  assert.match(octreeSurfaceAdapterShader,
    /radius=0\.5\*f32\(size\)\*length\(params\.cellHalo\.xyz\)[\s\S]*centrePhi-radius,centrePhi\+radius/,
    "coarse interface leaves require a conservative physical phi interval");
});

test("runtime detail refinement decodes independently from the analytic bootstrap selector", () => {
  assert.match(octreeProjectionShader, /fn surfaceDetailStrengthValue\(\) -> f32/);
  assert.doesNotMatch(octreeProjectionShader, /detailActivity = params\.physical\.w/);
  assert.match(octreeProjectionShader, /detailActivity = surfaceDetailStrengthValue\(\)/);
});
