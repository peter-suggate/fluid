import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { octreeProjectionShader, WebGPUOctreeProjection } from "../lib/webgpu-octree";
import { octreeFineSeedAdapterShader, octreeFineSeedCandidateShader } from "../lib/webgpu-octree-fine-seed-adapter";

const projectionSource = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");

test("authored t=0 topology keeps one analytic SDF authority through its first solve", () => {
  assert.match(octreeProjectionShader,
    /fn analyticInitialPhi\(point: vec3f\)[\s\S]*heightFraction = max\(0\.92, fill\)[\s\S]*footprintFraction = sqrt/,
    "dam bootstrap must reproduce the authored fill-preserving analytic block");
  assert.match(octreeProjectionShader,
    /let exposedMaximum = vec3f[\s\S]*let q = world - exposedMaximum/,
    "closed tank-contact planes must extend negative phi and leave only the three exposed dam faces");
  assert.match(octreeProjectionShader,
    /fn phi\(p: vec3i\)[\s\S]*if\(analyticInitialPhiEnabled\(\)\)\{[\s\S]*return analyticInitialPhi[\s\S]*if\(coarse\.authority\)\{return coarseClassificationPhi\(coarse\);\}/,
    "the first advancing frontier cannot mix corrected-coarse rows with analytic Power-face samples");
  assert.match(octreeProjectionShader,
    /fn currentPressureOwnerWet\(owner: Owner\)[\s\S]*var wet=liquidOwner\(owner\);[\s\S]*if\(analyticInitialPhiEnabled\(\)\)\{return wet;\}/,
    "fine summaries cannot override frontier membership while the face builder still samples t=0");
  assert.doesNotMatch(octreeProjectionShader, /legacyPhi|pagedSurface|surfacePagePhi/,
    "the deleted page authority must not remain as a bootstrap branch");
  assert.match(projectionSource,
    /const analyticBootstrapSelector = !this\.analyticSparseBootstrap[\s\S]*initialCondition === "dam-break" \? -20 : -10/,
    "dam/tank scenes select their analytic bootstrap sentinel while other scenes publish zero");
  assert.match(projectionSource,
    /case "structured-authority":[\s\S]*analyticBootstrapRetirementByEncoder\.add\(encoder\)[\s\S]*retireSubmittedEncoder\(encoder[\s\S]*analyticBootstrapRetirementByEncoder\.delete\(encoder\)[\s\S]*queue\.writeBuffer\(this\.params, 124/,
    "the analytic selector must retire only after submission of the first structured solve");
  assert.match(projectionSource,
    /analyticBootstrapRetirementByEncoder\.delete\(encoder\)[\s\S]*structuredBoundary\?\.retireAnalyticBootstrap\(\)/,
    "the boundary authored t=0 selector must retire with the invocation-stable bootstrap encoder");
  assert.doesNotMatch(octreeProjectionShader,
    /analyticInitialPhi\(point: vec3f\)[\s\S]{0,800}textureLoad/,
    "analytic initial classification must not sample the dense level-set texture");
});

test("first fine-seed leaves and indexed global-fine generation share analytic bootstrap phi", () => {
  assert.match(octreeFineSeedAdapterShader,
    /fn analyticInitialPhi\(point:vec3f\)[\s\S]*heightFraction=max\(0\.92,fill\)[\s\S]*footprintFraction=sqrt/);
  assert.match(octreeFineSeedAdapterShader, /let exposedMaximum=vec3f[\s\S]*let q=world-exposedMaximum/,
    "fine seed phi must share the coarse closed-wall extension");
  assert.match(octreeFineSeedAdapterShader,
    /if\(!coarse&&params\.selection\.z==0u\)\{return;\}[\s\S]*else\{[\s\S]*centrePhi=analyticInitialPhi/);
  assert.match(projectionSource,
    /analyticSparseBootstrap[\s\S]*new Float32Array\(\[Math\.max\(cell\.x, cell\.y, cell\.z\) \* this\.maxLeafSize\]\)[\s\S]*analyticInitialCondition/,
    "eligible analytic scenes must bind only one format texel while global fine is bootstrapped");
});

test("eligible analytic scenes use the GPU-authored resident cold worklist", () => {
  assert.match(projectionSource,
    /if \(this\.analyticBootstrapWorklist\)[\s\S]*\.encode\(encoder\)[\s\S]*this\.topologyWorklistReady = true[\s\S]*this\.encodeInactiveTopologyCandidate\(encoder/,
    "analytic bootstrap must enter the resident rebuild without a finest-domain cold dispatch");
  assert.match(projectionSource,
    /const analyticBootstrapPlan = analyticSparseBootstrap \? planOctreeAnalyticBootstrapBounds/);
  assert.match(projectionSource,
    /if \(analyticBootstrapPlan\)[\s\S]*new WebGPUOctreeAnalyticBootstrapWorklist/);
  assert.match(projectionSource,
    /this\.topologyResidency\.encode\(encoder, this\.surfaceState\.texture\)[\s\S]*this\.topologyWorklistReady = true[\s\S]*this\.encodeInactiveTopologyCandidate\(encoder, false, true\)/,
    "non-analytic t=0 must explicitly publish dense-SDF tile residency before its cold topology and owner-page transaction");
});

test("global fine bootstrap accepts interface seeds from coarse octree leaves", () => {
  const construction = WebGPUOctreeProjection.toString();
  assert.match(construction, /finestLeafSize:\s*this\.maxLeafSize/,
    "fine brick identity is global, so coarse interface leaves must remain eligible seeds");
  assert.match(octreeFineSeedCandidateShader,
    /fn selectedCandidate[\s\S]*leaf\.size>params\.selection\.x[\s\S]*\(!delta&&selectedFlags==0u\)/);
  assert.match(octreeFineSeedAdapterShader,
    /radius=0\.5\*f32\(header\.size\)\*length\(params\.cellHalo\.xyz\)[\s\S]*minimumPhi=centrePhi-radius;maximumPhi=centrePhi\+radius/,
    "coarse interface leaves require a conservative physical phi interval");
});

test("analytic bootstrap selector no longer carries retired surface-detail policy", () => {
  assert.doesNotMatch(projectionSource, /surfaceDetailStrength/);
  assert.doesNotMatch(octreeProjectionShader, /surfaceDetailStrengthValue|detailActivity|maximumCurvatureProxy|strainActivity/);
});
