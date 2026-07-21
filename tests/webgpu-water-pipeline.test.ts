import assert from "node:assert/strict";
import test from "node:test";
import {
  activeCubeCapacity,
  compositeShader,
  CONTACT_RESOLVE_BAND_CELLS,
  EXTRACTION_POLYGONISE_WORKGROUP,
  extractionPrepareShader,
  shouldResolveRigidContact,
  shouldUpdateWaterSurface,
  surfaceExtractionDispatchPlan,
  surfaceExtractionRepresentation,
  surfaceExtractionShader,
  surfaceVertexCapacity,
  WATER_INTERFACE_CULL_MODES
} from "../lib/webgpu-water-pipeline";

test("rigid contact resolution is confined to a narrow body/surface band", () => {
  assert.equal(CONTACT_RESOLVE_BAND_CELLS, 1.5);
  assert.equal(shouldResolveRigidContact(4, 4.149, 0.1, 1), true);
  assert.equal(shouldResolveRigidContact(4, 4.151, 0.1, 1), false);
  assert.equal(shouldResolveRigidContact(4, 4, 0.1, 0), false, "empty scenes must not pay for contact refinement");
  assert.equal(shouldResolveRigidContact(4, Number.POSITIVE_INFINITY, 0.1, 1), false);
});

test("the optical composite locally refines rigid contacts and terminates water at opaque bodies", () => {
  assert.match(compositeShader, /fn refineContactSurface/);
  assert.match(compositeShader, /abs\(rigidFront\.t-frontDepth\)<=contactBand/, "implicit sampling stays behind the analytic contact gate");
  assert.match(compositeShader, /rigidFront\.t<=frontDepth/, "exact rigid depth owns pixels in front of the refined liquid surface");
  assert.match(compositeShader, /opaqueSolidExit=true/, "refracted water rays terminate on submerged rigid bodies");
  assert.match(compositeShader, /var liquidField:texture_3d<f32>/);
  assert.match(compositeShader, /var<storage,read> bodies:array<BodyGPU,12>/);
  assert.match(compositeShader, /if\(!adaptiveSurface&&u\.gridInfo\.w>/,
    "adaptive presentation must not read the dense contact field");
});

test("outward water triangles preserve camera-entry interfaces after framebuffer Y inversion", () => {
  assert.deepEqual(WATER_INTERFACE_CULL_MODES, { front: "back", back: "front" });
});

test("surface extraction follows the fixed presentation cadence", () => {
  assert.equal(shouldUpdateWaterSurface(-1, 0, -Infinity, 0), true, "the first mesh is immediate");
  assert.equal(shouldUpdateWaterSurface(4, 5, 100, 115), false);
  assert.equal(shouldUpdateWaterSurface(4, 9, 100, 116.2), true, "60 Hz is the smooth raster cadence");
  assert.equal(shouldUpdateWaterSurface(9, 9, 100, 1000), false, "an unchanged solver field is never rebuilt");
});

test("restricted tall-cell extraction follows the surface band and preserves full-height walls", () => {
  const plan = surfaceExtractionDispatchPlan(80, 160, 60, 26, true, 4);
  assert.equal(plan.mode, "restricted-band");
  assert.equal(plan.bandCubeRows, 33, "24 packed regular layers plus an eight-cell diagonal base allowance and the crossing row");
  assert.deepEqual(plan.band, [20, 9, 15]);
  assert.deepEqual(plan.tallSides, [10, 8, 1]);
  assert.deepEqual(plan.walls, [705, 1, 1]);

  const full = surfaceExtractionDispatchPlan(80, 160, 60, 160, false, 0);
  assert.deepEqual(full.full, [21, 41, 16]);
});

test("surface normals reuse the classified cube instead of resampling the volume", () => {
  assert.match(surfaceExtractionShader, /Analytic gradient of the cube's trilinear reconstruction/);
  assert.doesNotMatch(surfaceExtractionShader, /fn fieldLinear/);
});

test("restricted extraction has separate surface-band and tank-wall entry points", () => {
  assert.match(surfaceExtractionShader, /fn extractBandMain/);
  assert.match(surfaceExtractionShader, /fn extractTallSidesMain/);
  assert.match(surfaceExtractionShader, /fn extractWallMain/);
  assert.match(surfaceExtractionShader, /let minimumBase/);
});

test("extraction is split into a lean classify sweep and a compacted polygonise pass", () => {
  assert.match(surfaceExtractionShader, /fn classifyCube/, "sweep kernels classify and append to the worklist");
  assert.match(surfaceExtractionShader, /fn polygoniseMain/, "triangle emission runs over compacted surface cubes only");
  const polygonise = surfaceExtractionShader.match(/fn polygoniseMain[\s\S]*?\n\}/)?.[0] ?? "";
  assert.doesNotMatch(polygonise, /atomicCompareExchangeWeak/, "the per-triangle global compare-exchange loop must not return");
  assert.match(surfaceExtractionShader, /atomicAdd\(&drawArgs\.activeCubeCount, 1u\)/, "classification appends with a single atomic per surface cube");
  assert.match(surfaceExtractionShader, /var<workgroup> workgroupVertexTotal/, "vertex blocks are reserved per workgroup, not per triangle");
  assert.match(surfaceExtractionShader, new RegExp(`@compute @workgroup_size\\(${EXTRACTION_POLYGONISE_WORKGROUP}\\)\\s*\\nfn polygoniseMain`));
});

test("adaptive octree pages drive the production mesh without a dense publication sweep", () => {
  assert.match(surfaceExtractionShader, /fn extractAdaptiveMain/);
  assert.match(surfaceExtractionShader, /fn extractAdaptiveLeafMain/);
  assert.match(surfaceExtractionShader, /adaptiveArena\[adaptiveParams\.offsets1\.x\+4u\+item\]/);
  assert.match(surfaceExtractionShader, /adaptiveOrigin\(leaf\.packedOrigin\)\*resolution\+local/);
  assert.match(surfaceExtractionShader, /classifyCubeScaled\(vec3i\(adaptiveOrigin\(leaf\.packedOrigin\)\*resolution\),max\(1u,leaf\.size\*resolution\)\)/);
  const leafExtraction = surfaceExtractionShader.match(/fn extractAdaptiveLeafMain[\s\S]*?\n\}/)?.[0] ?? "";
  assert.match(leafExtraction, /\(leaf\.flags&32u\)==0u/,
    "all live affine leaves are presentation-authoritative even without a CORE residency hint");
  assert.doesNotMatch(leafExtraction, /leaf\.flags&2u/);
  assert.equal(surfaceExtractionRepresentation(true, true), "adaptive-octree");
  assert.equal(surfaceExtractionRepresentation(false, true), "sparse-band");
  assert.equal(surfaceExtractionRepresentation(false, false), "dense-texture");
  assert.doesNotMatch(surfaceExtractionShader.match(/fn extractAdaptiveMain[\s\S]*?\n\}/)?.[0] ?? "", /textureLoad\(volume/);
  assert.match(surfaceExtractionShader, /activeCubes: array<vec2u>/, "adaptive owner propagation must retain the 8-byte worklist record");
  assert.match(surfaceExtractionShader, /adaptiveOwnerRow = \(packedCube\.x >> 26u\)/, "polygonisation must restore the classifier's owner row from the existing record");
});

test("the prepare kernel sizes the indirect polygonise dispatch from the worklist", () => {
  assert.match(extractionPrepareShader, /fn prepareMain/);
  assert.match(extractionPrepareShader, /@workgroup_size\(1\)/);
  assert.match(extractionPrepareShader, new RegExp(`\\+ ${EXTRACTION_POLYGONISE_WORKGROUP - 1}u\\) / ${EXTRACTION_POLYGONISE_WORKGROUP}u`), "ceiling division must match the polygonise workgroup size");
  assert.match(extractionPrepareShader, /min\(drawArgs\.activeCubeCount, arrayLength\(&activeCubes\)\)/, "an overflowing worklist must clamp instead of dispatching past the buffer");
});

test("buffer capacities keep the worklist aligned with the vertex allocation", () => {
  assert.equal(surfaceVertexCapacity(16, 16, 16), 262_144, "small grids use the floor allocation");
  assert.equal(surfaceVertexCapacity(512, 512, 512), 2_097_152, "large grids hit the 64 MiB ceiling");
  assert.equal(surfaceVertexCapacity(80, 160, 60), (80 * 160 + 80 * 60 + 160 * 60) * 64, "mid-size grids scale with transported surface area");
  assert.equal(activeCubeCapacity(262_144), 87_382, "every appended cube emits at least one triangle, so capacity/3 entries suffice");
  assert.equal(activeCubeCapacity(2_097_152), 699_051);
});
