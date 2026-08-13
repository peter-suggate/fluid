import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  OCTREE_POWER_COARSE_LEVELSET_SAMPLE_ENTRY_BYTES,
  OCTREE_POWER_COARSE_LEVELSET_SAMPLE_HEADER_BYTES,
} from "../lib/webgpu-octree-power-coarse-levelset";
import {
  activeCubeCapacity,
  causticReceiverContentKey,
  compositeShader,
  CONTACT_RESOLVE_BAND_CELLS,
  EXTRACTION_POLYGONISE_WORKGROUP,
  extractionPrepareShader,
  resolvePremultipliedCausticSample,
  shouldResolveRigidContact,
  shouldUpdateWaterSurface,
  surfaceExtractionDispatchPlan,
  surfaceExtractionShader,
  surfaceRasterShader,
  surfaceVertexCapacity,
  WATER_DISABLED_STORAGE_BYTES,
  WATER_INTERFACE_CULL_MODES
} from "../lib/webgpu-water-pipeline";
import { terrainContentStamp, type TerrainDescription } from "../lib/terrain";

test("caustic receiver invalidation follows sculpted content at a fixed shape", () => {
  const terrain: TerrainDescription = {
    baseHeight_m: 0.2,
    features: [],
    grid: {
      kind: "grid", origin_m: { x: -1, z: -1 }, spacing_m: 1,
      size: { nx: 2, nz: 2 }, heights_m: [0.2, 0.2, 0.2, 0.2],
    },
  };
  const sculpted = {
    ...terrain,
    grid: { ...terrain.grid!, heights_m: [0.2, 0.2, 0.3, 0.2] },
  };
  assert.notEqual(
    causticReceiverContentKey(terrain, 3, 2, terrainContentStamp(terrain)),
    causticReceiverContentKey(sculpted, 3, 2, terrainContentStamp(sculpted)),
    "a brush stroke must rebake the receiver without resizing its grid or container",
  );
});

test("filtered caustic coverage blends back to the neutral floor without erasing overlaps", () => {
  assert.deepEqual(resolvePremultipliedCausticSample([0, 0, 0, 0]), [1, 1, 1]);
  assert.deepEqual(resolvePremultipliedCausticSample([0.1, 0.08, 0.06, 0.1]), [1, 0.98, 0.96],
    "ten-percent filtered coverage must retain the ninety-percent unchanged floor share");
  assert.deepEqual(resolvePremultipliedCausticSample([0.6, 0.8, 1.2, 1]), [0.6, 0.8, 1.2],
    "complete coverage keeps the deposited modulation verbatim");
  assert.deepEqual(resolvePremultipliedCausticSample([1.8, 2.2, 2.6, 2]), [1.8, 2.2, 2.6],
    "alpha above one is a real overlapping bundle and must not be normalized away");
  assert.deepEqual(resolvePremultipliedCausticSample([0.6, 0.8, 1.2, 1], 0.5), [0.8, 0.9, 1.1]);
  assert.match(compositeShader, /let coverage=clamp\(sampled\.a,0\.0,1\.0\)/);
  assert.match(compositeShader, /let modulation=mix\(vec3f\(1\.0\),deposited,coverage\)/);
});

test("rigid contact resolution is confined to a narrow body/surface band", () => {
  assert.equal(CONTACT_RESOLVE_BAND_CELLS, 1.5);
  assert.equal(shouldResolveRigidContact(4, 4.149, 0.1, 1), true);
  assert.equal(shouldResolveRigidContact(4, 4.151, 0.1, 1), false);
  assert.equal(shouldResolveRigidContact(4, 4, 0.1, 0), false, "empty scenes must not pay for contact refinement");
  assert.equal(shouldResolveRigidContact(4, Number.POSITIVE_INFINITY, 0.1, 1), false);
});

test("the optical composite locally refines rigid contacts and terminates water at opaque bodies", () => {
  assert.match(compositeShader, /fn safeInterfaceSample/);
  assert.match(compositeShader, /sampled\.rgb\/sampled\.a/,
    "filtered interface positions and normals must be resolved out of their fractional validity coverage");
  assert.doesNotMatch(compositeShader, /front=safeSample\((?:rearFrontPosition|frontPosition)/);
  assert.doesNotMatch(compositeShader, /normalize\(safeSample\((?:rearFrontNormal|frontNormal|rearBackNormal|backNormal)/);
  assert.match(compositeShader, /fn refineContactSurface/);
  assert.match(compositeShader, /abs\(rigidFront\.t-frontDepth\)<=contactBand/, "implicit sampling stays behind the analytic contact gate");
  assert.match(compositeShader, /rigidFront\.t<=frontDepth/, "exact rigid depth owns pixels in front of the refined liquid surface");
  assert.match(compositeShader, /opaqueSolidExit=true/, "refracted water rays terminate on submerged rigid bodies");
  assert.match(compositeShader, /var liquidField:texture_3d<f32>/);
  assert.match(compositeShader, /var<storage,read> bodies:array<BodyGPU,12>/);
  assert.match(compositeShader, /if\(u\.gridInfo\.w>/,
    "octree presentation must gate dense contact refinement");
  assert.doesNotMatch(compositeShader, /adaptiveSurface/,
    "the deleted page renderer must not leave a presentation switch behind");
});

test("outward water triangles preserve camera-entry interfaces after framebuffer Y inversion", () => {
  assert.deepEqual(WATER_INTERFACE_CULL_MODES, { front: "back", back: "front" });
});

test("every water projection consumes the authored camera aperture", () => {
  assert.match(surfaceRasterShader, /let aperture=cameraTanHalfFov\(\)/);
  assert.match(compositeShader, /fn project\([^}]+let aperture=cameraTanHalfFov\(\)/);
  assert.match(compositeShader, /fn cameraRay\([^}]+let aperture=cameraTanHalfFov\(\)/);
  assert.doesNotMatch(surfaceRasterShader, /depth\*aspect\*0?\.72|depth\*0?\.72/);
  assert.doesNotMatch(compositeShader, /ndc\.[xy][^;]*\*0?\.72|depth[^;]*\*0?\.72/);
});

test("front interface rasterization conservatively covers shared liquid-wall silhouettes", () => {
  assert.match(surfaceRasterShader, /override interfaceCoverageExpansionPixels:f32=0\.0/);
  assert.match(surfaceRasterShader, /first=index-index%3u/,
    "coverage expansion must use each triangle centroid, not move the physical mesh");
  const source = readFileSync(new URL("../lib/webgpu-water-pipeline.ts", import.meta.url), "utf8");
  assert.match(source, /surfaceFrontPipeline\s*=.*WATER_INTERFACE_CULL_MODES\.front,1\.0/);
  assert.match(source, /surfaceBackPipeline\s*=.*WATER_INTERFACE_CULL_MODES\.back\)\)/,
    "back coverage remains exact so the back-without-front smoke gate still detects holes");
});

test("interface rasterization depth-peels fluid hidden behind a foreground sheet", () => {
  assert.match(surfaceRasterShader, /override peelBehindFirstExit:f32=0\.0/);
  assert.match(surfaceRasterShader, /firstBack=textureLoad\(firstBackPosition/);
  assert.match(surfaceRasterShader, /candidateDepth<=firstExitDepth/);
  assert.match(compositeShader, /fn compositeRearWater/);
  assert.match(compositeShader,
    /transmittedScene=compositeRearWater\(backgroundUV/,
    "the foreground interval must consume radiance after the rear interval is shaded");

  const source = readFileSync(new URL("../lib/webgpu-water-pipeline.ts", import.meta.url), "utf8")
    .replace(/\s+/g, "");
  assert.match(source, /surfaceRearFrontPipeline=.*surfaceDescriptor\("Rasterwaterrearfrontinterfaces",WATER_INTERFACE_CULL_MODES\.front,0,true\)/);
  assert.match(source, /surfaceRearBackPipeline=.*surfaceDescriptor\("Rasterwaterrearbackinterfaces",WATER_INTERFACE_CULL_MODES\.back,0,true\)/);
  assert.match(source, /interfacePass\("Waterrearfrontinterfaces"[^;]+false,true\)[\s\S]*interfacePass\("Waterrearbackinterfaces"[^;]+false,true\)/,
    "the rear pair must be rasterized after the first back interface establishes the peel depth");
  assert.match(source, /surfaceUnpeeledBindGroup=.*resource:this\.sceneTextureView/,
    "ordinary interface passes need a harmless binding because Dawn requires every pipeline-layout group");
  assert.match(source, /pass\.setBindGroup\(1,peel\?this\.surfacePeelBindGroup!:this\.surfaceUnpeeledBindGroup!\)/,
    "only rear passes may bind the first back texture; binding it while rendering the first back interface is a WebGPU usage conflict");
});

test("missing dry-scene publication does not suppress authoritative water", () => {
  const source = readFileSync(new URL("../lib/webgpu-water-pipeline.ts", import.meta.url), "utf8")
    .replace(/\s+/g, "");
  // The second condition is the render panel's ablation switch, which is the
  // one thing allowed to withhold these passes — and it takes the fluid-less
  // clears when it does, so nothing composites geometry this frame never drew.
  assert.match(source,
    /if\(this\.sceneHasFluid&&!interfacesWithheld\)\{interfacePass\("Water\+sprayfrontinterfaces"/,
    "fluid interface passes must depend on fluid authority, not dry-scene publication");
  assert.doesNotMatch(source, /if\(sparseSceneResult&&this\.sceneHasFluid\)/,
    "the visible fail-closed background must not clear otherwise valid water geometry");
  assert.match(source,
    /clearValue:\{r:\.18,g:0,b:\.045,a:65504\}/,
    "the missing dry scene remains unmistakably failed closed behind the water");
});

test("fluid-only presentation retains one clear background behind unchanged raster water", () => {
  const source = readFileSync(new URL("../lib/webgpu-water-pipeline.ts", import.meta.url), "utf8")
    .replace(/\s+/g, "");
  assert.match(source, /backgroundMode==="clear"/);
  assert.match(source, /if\(!this\.clearBackgroundEncoded\)\{/,
    "the empty background must clear once rather than write every pixel every frame");
  assert.match(source, /label:"Fluid-onlyclearbackground"/);
  assert.match(source, /this\.sceneHasFluid&&!interfacesWithheld\)\{interfacePass\("Water\+sprayfrontinterfaces"/,
    "fluid-only mode must retain the exact raster interface passes");
  assert.match(source, /composite\.setPipeline\(this\.compositePipeline\);composite\.setBindGroup/,
    "fluid-only mode must retain the pretty optical composite");
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

test("sub-cell tank-wall mass is emitted as a conservative thin shell", () => {
  assert.match(surfaceExtractionShader, /fn emitWallFilmMain/,
    "wall-supported density below the bulk isovalue needs its own two-sided geometry");
  assert.match(surfaceExtractionShader,
    /return wallFilmDensity\(face,q\)\*wallFilmAxisCellWidths\(face\)\.x/,
    "film thickness must be the represented volume fraction times the wall-normal cell width");
  assert.match(surfaceExtractionShader, /return rho\/wallContactCount\(c\)/,
    "edge and corner mass must be divided rather than rendered once on every wall");
  assert.doesNotMatch(surfaceExtractionShader, /fn wallFilmGhost/,
    "one ghost edge cannot represent both interfaces of a thinner-than-cell slab");
  assert.match(surfaceRasterShader, /interfaceCoverageExpansionPixels>0\.0&&v\.normal\.w<0\.5/,
    "per-triangle silhouette expansion must not inflate the regular wall-film tiles");
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

test("the deleted row-attached page renderer has no shader or class fallback", () => {
  const source = readFileSync(new URL("../lib/webgpu-water-pipeline.ts", import.meta.url), "utf8");
  assert.doesNotMatch(surfaceExtractionShader,
    /extractAdaptive|adaptiveArena|adaptiveParams|adaptiveLeaf|adaptiveOwnerRow/);
  assert.doesNotMatch(source, /setAdaptiveOctree|adaptiveSurface|octreeSurfacePages/);
});

test("disabled water storage satisfies only the compact coarse-directory ABI", () => {
  assert.equal(
    OCTREE_POWER_COARSE_LEVELSET_SAMPLE_HEADER_BYTES
      + OCTREE_POWER_COARSE_LEVELSET_SAMPLE_ENTRY_BYTES,
    64,
    "the compact coarse directory requires its header and at least one runtime-sized entry",
  );
  assert.equal(WATER_DISABLED_STORAGE_BYTES,
    Math.max(
      64,
      OCTREE_POWER_COARSE_LEVELSET_SAMPLE_HEADER_BYTES
        + OCTREE_POWER_COARSE_LEVELSET_SAMPLE_ENTRY_BYTES,
    ));
  const source = readFileSync(new URL("../lib/webgpu-water-pipeline.ts", import.meta.url), "utf8");
  assert.match(source,
    /Water disabled storage binding"\s*,\s*size:\s*WATER_DISABLED_STORAGE_BYTES/,
    "the optional coarse directory must have a valid disabled binding");
  assert.doesNotMatch(source, /OCTREE_FINE_SEED_LEAF_RECORD_BYTES|adaptive\?\.leaves/);
});

test("the prepare kernel sizes the indirect polygonise dispatch from the worklist", () => {
  assert.match(extractionPrepareShader, /fn prepareMain/);
  assert.match(extractionPrepareShader, /@workgroup_size\(1\)/);
  assert.match(extractionPrepareShader, new RegExp(`\\+ ${EXTRACTION_POLYGONISE_WORKGROUP - 1}u\\) / ${EXTRACTION_POLYGONISE_WORKGROUP}u`), "ceiling division must match the polygonise workgroup size");
  assert.match(extractionPrepareShader, /min\(drawArgs\.activeCubeCount, arrayLength\(&activeCubes\)\)/, "an overflowing worklist must clamp instead of dispatching past the buffer");
  assert.doesNotMatch(extractionPrepareShader, /globalFineAuthorityLatch/,
    "the generic prepare kernel must not retain a zero-work global-fine switch");
  const source = readFileSync(new URL("../lib/webgpu-water-pipeline.ts", import.meta.url), "utf8")
    .replace(/\s+/g, "");
  const scopeBoundary = source.indexOf("constprepareAndPolygonise=");
  assert.ok(scopeBoundary >= 0);
  const helper = source.slice(scopeBoundary, source.indexOf("constglobalFine=", scopeBoundary));
  assert.match(helper,
    /compute\.setPipeline\(this\.preparePipeline!\)[\s\S]*compute\.dispatchWorkgroups\(1\);compute\.end\(\);compute=encoder\.beginComputePass/,
    "writable polygonise preparation must end its usage scope before indirect consumption");
  assert.match(helper,
    /compute\.dispatchWorkgroupsIndirect\(this\.polygoniseDispatchBuffer!,0\)/);
});

test("global-fine presentation compiles and encodes only the combined mesh path", () => {
  const source = readFileSync(new URL("../lib/webgpu-water-pipeline.ts", import.meta.url), "utf8")
    .replace(/\s+/g, "");
  assert.doesNotMatch(source, /globalFineClassifiedEmitShaders|polygoniseGlobalFineEmitPipelines/,
    "the six unused per-tetrahedron modules and pipelines must stay deleted");
  assert.match(source, /consttotal=17/,
    "pipeline progress must include the dedicated wall-film pipeline");

  const globalStart = source.indexOf("if(globalFine){");
  const volumeStart = source.indexOf("}else{", globalStart);
  const extractionEnd = source.indexOf("this.encodeSurfaceDiagnostics(encoder,forceSurfaceDiagnostics)", volumeStart);
  assert.ok(globalStart >= 0 && volumeStart > globalStart && extractionEnd > volumeStart);
  const globalBranch = source.slice(globalStart, volumeStart);
  assert.match(globalBranch, /polygoniseGlobalFineScanPipeline/);
  assert.match(globalBranch, /polygoniseGlobalFineEmitPipeline/);
  assert.doesNotMatch(globalBranch, /prepareAndPolygonise|preparePipeline|dispatchWorkgroupsIndirect/,
    "global-fine authority must end after its direct scan/emit path");
  assert.match(source.slice(volumeStart, extractionEnd), /prepareAndPolygonise\(this\.polygonisePipeline,this\.extractBindGroup\)/,
    "the generic volume representation must retain its indirect polygonise path");
});

test("buffer capacities keep the worklist aligned with the vertex allocation", () => {
  assert.equal(surfaceVertexCapacity(16, 16, 16), 262_144, "small grids use the floor allocation");
  assert.equal(surfaceVertexCapacity(512, 512, 512), 2_097_152, "large grids hit the 64 MiB ceiling");
  assert.equal(surfaceVertexCapacity(64, 64, 64), 2_097_152,
    "the mini dam allocation covers its bulk mesh plus the worst-case two-sided wall shell");
  assert.equal(surfaceVertexCapacity(60, 120, 50), 2_097_152,
    "mid-size grids remain bounded by the absolute 64 MiB vertex ceiling");
  assert.equal(activeCubeCapacity(262_144), 87_382, "every appended cube emits at least one triangle, so capacity/3 entries suffice");
  assert.equal(activeCubeCapacity(2_097_152), 699_051);
});
