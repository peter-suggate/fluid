import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { gridOverlayShader } from "../lib/webgpu-grid-overlay";

const rendererSource = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
const octreeSource = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
const gridSource = readFileSync(new URL("../lib/webgpu-grid-overlay.ts", import.meta.url), "utf8");

test("grid overlay is an independent alpha-composited presentation layer", () => {
  assert.match(gridOverlayShader, /@group\(0\) @binding\(2\) var fluidField: texture_3d<f32>/);
  assert.match(gridOverlayShader, /@group\(0\) @binding\(3\) var tallCellBases: texture_2d<f32>/);
  assert.match(gridOverlayShader, /@group\(0\) @binding\(4\) var adaptiveCells: texture_3d<u32>/);
  assert.match(gridOverlayShader, /@group\(0\) @binding\(6\) var pressureSamples: texture_3d<u32>/);
  assert.match(gridOverlayShader, /@group\(0\) @binding\(7\) var divergenceField: texture_3d<f32>/);
  assert.match(gridOverlayShader, /@group\(0\) @binding\(8\) var mappedPressureField: texture_3d<f32>/);
  assert.match(gridOverlayShader, /let axis = i32\(round\(u\.debug\.x\)\)/);
  assert.match(gridOverlayShader, /@fragment fn fragmentMain/);
  assert.match(gridOverlayShader, /return vec4f\(displayColor\(overlay\.color\), overlay\.alpha\)/);
});

test("grid optional-source fallback satisfies the canonical 64-byte surface-leaf ABI", () => {
  assert.match(gridOverlayShader,
    /struct OctreeSurfaceLeaf \{ originX:u32,originY:u32,originZ:u32,size:u32,flags:u32,pad0:u32,pad1:u32,pad2:u32,phiGradient:vec4f,motion:vec4f \}/);
  assert.match(gridSource,
    /label: "Grid sparse-surface control fallback",\s*\n\s*size: OCTREE_SURFACE_LEAF_RECORD_BYTES/);
  assert.doesNotMatch(gridSource, /Grid sparse-surface control fallback"[^\n]*size:\s*48/);
});

test("grid overlay supports a horizontal Y slice with X-Z adaptive boundaries", () => {
  assert.match(gridOverlayShader, /else if \(axis == 3\) \{\n    samplePosition = local3\.xz/);
  assert.match(gridOverlayShader, /secondPlaneAxis = 2/);
  assert.match(gridOverlayShader, /planeCoordinate = boundsMin\.y \+ \(layer \+ 0\.5\) \* size\.y \/ dims\.y/);
  assert.match(gridOverlayShader, /denominator = direction\.y/);
  assert.match(gridOverlayShader, /axis == 3\) \* 0\.8/);
});

test("adaptive diagnostic modes expose coverage, level set, divergence, and pressure", () => {
  assert.match(gridOverlayShader, /fieldMode == 3/);
  assert.match(gridOverlayShader, /fieldMode == 4/);
  assert.match(gridOverlayShader, /fieldMode == 5/);
  assert.match(gridOverlayShader, /fieldMode == 6/);
  assert.match(gridOverlayShader, /let unrepresented = adaptiveGrid && wet && !hasLiquidPressureDof\(cell\)/);
  assert.match(gridOverlayShader, /divergence \* max\(u\.environment\.y, 1e-6\)/);
  assert.match(gridOverlayShader, /textureLoad\(mappedPressureField, cell, 0\)\.x/);
  assert.match(gridOverlayShader, /fieldMode == 8/);
  assert.match(gridOverlayShader, /bitcast<f32>\(textureLoad\(pressureSamples, cell, 0\)\.y\)/,
    "the octree projection-update alarm must consume the resident packed diagnostic");
  assert.match(rendererSource, /gridOverlay\?\.mode === "projection" && gpuInfo\?\.gridKind === "octree" \? 8 : gridOverlay\?\.mode === "resolution"/);
});

test("adaptive cell-scale mode audits the compact octree leaf level", () => {
  assert.match(gridOverlayShader, /fieldMode == 9 && adaptiveGrid/);
  assert.match(gridOverlayShader, /var representedSize = max\(horizontalSize, verticalSize\)/);
  assert.match(gridOverlayShader, /representedSize=i32\(octreeSurfaceLeaves\[row\]\.size\)/);
  assert.match(gridOverlayShader, /missingCompactOwner=row==SPARSE_SURFACE_INVALID/);
  assert.match(gridOverlayShader, /let maximumRepresentedSize = max\(2\.0, u\.options\.w\)/);
  assert.match(gridOverlayShader, /log2\(f32\(representedSize\)\) \/ log2\(maximumRepresentedSize\)/);
  assert.match(gridOverlayShader, /let fineColor = vec3f\(0\.22, 0\.68, 0\.74\)/,
    "pressure leaves retain their own cyan-to-blue hierarchy");
  assert.match(gridOverlayShader, /fn sparseSurfaceCoreSample/);
  assert.match(gridOverlayShader, /abs\(sparseSurfacePhi\[payload\]\) <= 1\.5\*fineH/);
  assert.match(gridOverlayShader, /let fineColor=vec3f\(1\.0,0\.08,0\.55\)/,
    "pink is reserved for the fine phi=0 shell");
  assert.match(gridOverlayShader, /fill = select\(mix\(middleColor, coarseColor/);
  assert.match(rendererSource, /gridOverlay\?\.mode === "resolution" && \(gpuInfo\?\.gridKind === "quadtree-tall-cell" \|\| gpuInfo\?\.gridKind === "octree"\) \? 9 : gridOverlay\?\.mode === "surface"/);
  assert.match(rendererSource, /gpuInfo\?\.quadtreeMaximumFluidScale \?\? 1/);
});

test("default adaptive structure mode marks live sparse-surface cells pink", () => {
  assert.match(gridOverlayShader, /fieldMode == 0 \|\| fieldMode == 9/);
  assert.match(gridOverlayShader, /sparseSurfaceAvailable\(\)/);
  assert.match(gridOverlayShader, /fine3=local3\*factor/);
  assert.match(gridOverlayShader, /sparseSurfaceCoreSample\(fineCell\)/);
  assert.match(gridOverlayShader, /gridLineColor=fineColor/);
});

test("surface-band audit mode separates interface core, page support, halo, and fallback", () => {
  assert.match(gridOverlayShader, /fn sparseSurfacePageState/);
  assert.match(gridOverlayShader, /fieldMode == 10 && adaptiveGrid/);
  assert.match(gridOverlayShader, /resident && halo/);
  assert.match(gridOverlayShader, /resident && core/);
  assert.match(gridOverlayShader, /sparseSurfaceCoreSample\(fineCell\)/);
  assert.match(gridOverlayShader, /fill = vec3f\(0\.05, 0\.12, 0\.35\)/, "blue is coarse fallback");
  assert.match(gridOverlayShader, /vec3f\(0\.12, 0\.72, 0\.82\)/, "cyan is transport halo");
  assert.match(gridOverlayShader, /vec3f\(0\.42, 0\.19, 0\.62\)/, "violet is core-page support");
  assert.match(gridOverlayShader, /vec3f\(1\.0, 0\.03, 0\.52\)/, "pink is the core interface shell");
  assert.match(rendererSource, /gridOverlay\?\.mode === "surface" && gpuInfo\?\.gridKind === "octree" \? 10 : gridOverlay\?\.mode === "faces"/);
});

test("optical-layer mode distinguishes retained cubes from the merged tall interior", () => {
  assert.match(gridOverlayShader, /fn adaptiveCellVerticalShape/);
  assert.match(gridOverlayShader, /fn isOpticalCube/);
  assert.match(gridOverlayShader, /shape\.y - shape\.x == 1/);
  assert.match(gridOverlayShader, /fieldMode == 7 && u\.environment\.w > 0\.5/);
  assert.match(gridOverlayShader, /belowIsTall = !isOpticalCube/);
  assert.match(gridOverlayShader, /color = mix\(color, opticalBoundaryColor, opticalBoundary\)/);
  assert.match(gridOverlayShader, /u\.environment\.w > 1\.5/, "adaptive and fixed boundaries remain visually distinguishable");
  assert.match(rendererSource, /gridOverlay\?\.mode === "optical" \? 7 : gridOverlay\?\.mode === "projection" && gpuInfo\?\.gridKind === "octree" \? 8 : gridOverlay\?\.mode === "resolution"/);
  assert.match(rendererSource, /quadtreeOpticalLayerMode === "adaptive-motion" \? 2 : 1/);
});

test("solver option switches atomically rebind the overlay before retiring GPU textures", () => {
  const begin = rendererSource.indexOf("private beginGPUFluidInitialization");
  const end = rendererSource.indexOf("private currentGPUFluid", begin);
  const replacement = rendererSource.slice(begin, end);
  const candidateRebind = replacement.indexOf("this.updateRenderSources(solver.surfaceFieldTexture??solver.volumeTexture");
  const retire = replacement.indexOf("this.retireGPUFluid(previous)");
  assert.ok(begin >= 0 && end > begin);
  assert.ok(candidateRebind >= 0 && retire > candidateRebind,
    "candidate presentation bind groups must replace old solver textures before retirement");
  const debugDetach = replacement.indexOf("this.voxelDebugPipeline?.setSource(undefined)");
  assert.ok(debugDetach >= 0 && debugDetach < retire);
  const svoAttach = replacement.indexOf("this.svoDryScenePipeline?.setSource(sparseSceneSource,drySceneData)");
  assert.ok(svoAttach >= 0 && svoAttach < retire);
});

test("compact octree overlay samples leaf pages and never requests dense phi", () => {
  assert.match(gridOverlayShader, /fn octreeSurfaceRow/);
  assert.match(gridOverlayShader, /fn octreeSurfacePhi/);
  assert.match(gridOverlayShader, /r==2u\|\|r==4u/);
  assert.match(gridOverlayShader, /fn octreeSurfaceLoad/);
  assert.match(gridOverlayShader, /grid=clamp\(\(point-origin\)\/f32\(leaf\.size\)\*f32\(r\)/);
  assert.match(gridOverlayShader, /octreeSurfaceLoad\(base,b\)/);
  assert.match(gridOverlayShader, /fieldMode == 10[\s\S]*?octreeSurfaceBound\(\)/);
  assert.match(rendererSource, /adaptiveWaterReady \? this\.scalarFallbackTexture : readyGPUFluid\.surfaceFieldTexture/);
  assert.match(rendererSource, /compactSurface\?this\.scalarFallbackTexture:this\.gpuFluid\.surfaceFieldTexture\?\?this\.gpuFluid\.volumeTexture/);
});

test("compact velocity diagnostics reuse owner rows and bounded face incidence", () => {
  assert.match(gridOverlayShader, /let hinted=textureLoad\(pressureSamples,cell,0\)\.x/);
  assert.match(gridOverlayShader, /if\(octreeLeafContains\(hinted,point\)\)\{return hinted;\}/);
  assert.match(gridOverlayShader, /struct OctreeFaceAudit/);
  assert.match(gridOverlayShader, /let count=min\(publishedCount,OCTREE_FACE_INCIDENCE_PER_ROW\)/);
  assert.match(gridOverlayShader, /weighted\[axis\]\+=weight\*face\.normalVelocity/);
  assert.match(gridOverlayShader, /fieldMode == 11 && octreeGrid/);
  assert.match(gridOverlayShader, /compactVelocityFault = row == SPARSE_SURFACE_INVALID \|\| audit\.fault != 0u \|\| audit\.axisMask != 7u/);
  assert.match(rendererSource, /gridOverlay\?\.mode === "faces" && gpuInfo\?\.gridKind === "octree" \? 11 : 0/);
  assert.match(octreeSource, /vec4u\(row, bitcast<u32>\(pressureUpdate\), vertical, horizontal\)/,
    "the existing overlay materialization lane publishes the compact row without another pass");
});

test("grid overlay suppresses dense backing-grid lines inside adaptive cells", () => {
  assert.match(gridOverlayShader, /let adaptiveGrid = u\.debug\.z > 0\.5/);
  assert.match(gridOverlayShader, /any\(adaptiveCellKey\(lowerFirst, dims\) != own\)/);
  assert.match(gridOverlayShader, /any\(adaptiveCellKey\(lowerSecond, dims\) != own\)/);
  assert.match(gridOverlayShader, /any\(adaptiveCellKey\(below, dims\) != own\)/);
  assert.match(gridOverlayShader, /let leafSize = i32\(\(key\.x >> 20u\) & 1023u\)/);
});

test("octree structure view keeps the refined air band outline-only", () => {
  assert.match(gridOverlayShader, /let octreeGrid = u\.gridInfo\.w > 2\.5/);
  assert.match(gridOverlayShader, /let dryAlpha = select\(select\(0\.08, 0\.03, isTall\), 0\.0, octreeGrid\)/);
  assert.match(gridOverlayShader, /alpha = select\(dryAlpha, wetAlpha, wet\)/);
});

test("grid overlay preserves rigid-body occlusion independently of the water renderer", () => {
  assert.match(gridOverlayShader, /fn nearestBodyDistance/);
  assert.match(gridOverlayShader, /distance >= nearestBodyDistance\(origin, direction\) && !overlay\.solid/);
});

test("grid overlay renders rigid bodies as complete represented cells", () => {
  assert.match(gridOverlayShader, /fn bodySignedDistance/);
  assert.match(gridOverlayShader, /let sphereDistance = length\(closest - bodies\[index\]\.positionRadius\.xyz\) - bodies\[index\]\.positionRadius\.w/);
  assert.match(gridOverlayShader, /gridBodySample\(representedCell\(cell, dims, boundsMin, size, adaptiveGrid, tallGrid\)\)/);
  assert.match(gridOverlayShader, /return GridSample\(color, alpha, gridBody\.occupied\)/);
});

test("grid overlay field modes sample live GPU velocity without readback", () => {
  assert.match(gridOverlayShader, /@group\(0\) @binding\(5\) var velocityField: texture_3d<f32>/);
  assert.match(gridOverlayShader, /let fieldMode = i32\(round\(u\.debug\.w\)\)/);
  // CFL mode uses the solver's substep dt carried in the shared uniform.
  assert.match(gridOverlayShader, /let dt = max\(u\.environment\.y, 1e-6\)/);
  assert.match(gridOverlayShader, /abs\(velocity\.x\) \* dt \/ h\.x/);
  // Speed mode normalizes by the last reported liquid maximum.
  assert.match(gridOverlayShader, /max\(u\.environment\.z, 1e-4\)/);
});

test("grid overlay velocity sampling honours the packed tall-cell layout", () => {
  // Piecewise tall-cell reconstruction: top world cell = top endpoint dof,
  // every other interior row = bottom dof — the field projection controls.
  assert.match(gridOverlayShader, /let row = select\(0, 1, q\.y == base - 1\)/);
  assert.match(gridOverlayShader, /let packedY = 2 \+ q\.y - base;\n  let stored = vec3i\(textureDimensions\(velocityField\)\)/);
});
