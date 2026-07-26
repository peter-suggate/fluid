import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import test from "node:test";

const source = (path: string) => readFileSync(new URL(path, import.meta.url), "utf8");
const runtimeSource = (path: string) => source(path)
  .replace(/\/\*[\s\S]*?\*\//g, "")
  .replace(/\/\/[^\n\r]*/g, "");

const retiredOctreeModules = [
  "webgpu-octree-power-faces.ts",
  "webgpu-octree-power-operator.ts",
  "webgpu-octree-power-velocity.ts",
  "webgpu-octree-power-velocity-prepass.ts",
  "webgpu-octree-power-face-advection.ts",
  "webgpu-octree-power-face-seed.ts",
  "webgpu-octree-power-solid-faces.ts",
  "webgpu-octree-face-closest-point.ts",
  "webgpu-octree-page-pool.ts",
] as const;

test("retired octree executable modules are physically absent", () => {
  for (const retiredModule of retiredOctreeModules) {
    assert.equal(existsSync(new URL(`../lib/${retiredModule}`, import.meta.url)), false, retiredModule);
  }
  assert.equal(existsSync(new URL("../lib/webgpu-octree-mgpcg.ts", import.meta.url)), false);
  assert.equal(existsSync(new URL("../lib/webgpu-octree-power-galerkin.ts", import.meta.url)), false);
  assert.equal(existsSync(new URL("../lib/webgpu-octree-power-galerkin-persistent3.ts", import.meta.url)), false);
});

test("production imports only the direct structured velocity graph", () => {
  const projection = source("../lib/webgpu-octree.ts");
  const fineSeed = source("../lib/webgpu-octree-fine-seed-adapter.ts");
  assert.match(projection, /WebGPUDirectStructuredVelocityAuthority/);
  assert.match(projection, /WebGPUStructuredBoundaryCoefficients/);
  assert.match(projection, /WebGPUStructuredVelocityDynamics/);
  assert.doesNotMatch(projection + fineSeed,
    /OctreePowerFaceSource|OctreePowerVelocitySource|PowerFaceRecord|PowerIncidence|faceControl|incidenceRows/);
  assert.match(fineSeed, /DirectStructuredVelocitySource/);
});

test("CPU differential oracles contain no shadow GPU executors", () => {
  const structuredOracle = source("../lib/webgpu-octree-structured-velocity.ts");
  const pageOracle = source("../lib/webgpu-octree-brick-stencils.ts");
  const structuredProduction = source("../lib/webgpu-octree-structured-velocity-gpu.ts");
  const spgridProduction = source("../lib/webgpu-octree-spgrid-vcycle.ts");
  assert.match(structuredOracle, /publishStructuredVelocityFamilies/);
  assert.match(structuredOracle, /reconstructStructuredRowVelocity/);
  assert.doesNotMatch(structuredOracle,
    /@compute|createShaderModule|createComputePipeline|dispatchWorkgroups|GPUBuffer|GPUDevice/);
  assert.match(structuredProduction, /class WebGPUDirectStructuredVelocityAuthority/);
  assert.match(pageOracle, /offline CPU numerical oracle/);
  assert.doesNotMatch(pageOracle,
    /@compute|createShaderModule|createComputePipeline|dispatchWorkgroups|GPUBuffer|GPUDevice/);
  assert.match(spgridProduction, /fn smoothPageChebyshevForward/);
  assert.match(spgridProduction, /fn smoothPageChebyshevReverse/);
});

test("live owners cover the deleted duplicate page-pool responsibilities", () => {
  const ownerPages = source("../lib/webgpu-octree-owner-pages.ts");
  const finePages = source("../lib/webgpu-octree-fine-levelset-bricks.ts");
  const spgrid = source("../lib/webgpu-octree-spgrid-vcycle.ts");
  assert.match(ownerPages, /assigns stable physical IDs/);
  assert.match(ownerPages, /directoryBase\(inactiveTable\(\)\)/);
  assert.match(finePages, /direct page directory/);
  assert.match(finePages, /data\.haloIds/);
  assert.match(spgrid, /buildSPGridPhysicalPageAdjacency/);
  assert.match(spgrid, /pageShape = \[8, 8, 4\] as const/);
  assert.match(spgrid, /Twenty-seven physical IDs/);
});

test("direct structured authority has one fail-closed A\/B publication", () => {
  const authority = source("../lib/webgpu-octree-structured-velocity-gpu.ts");
  const boundary = source("../lib/webgpu-octree-structured-boundary.ts");
  const dynamics = source("../lib/webgpu-octree-structured-dynamics.ts");
  assert.match(authority, /Packed A\/B structured velocity authority/);
  assert.match(authority, /finalizeStructuredPublication/);
  assert.match(authority, /acceptStructuredPublication/);
  assert.match(authority, /readonly authorityBankStrideWords/);
  assert.match(authority, /readonly rowBankStrideWords/);
  assert.match(authority, /readonly worksetBankStrideWords/);
  assert.doesNotMatch(authority + boundary + dynamics,
    /PowerFaceRecord|PowerIncidence|incidenceRows|general(?:ized)?Faces?/i);
  assert.doesNotMatch(runtimeSource("../lib/webgpu-octree-structured-velocity-gpu.ts")
    + runtimeSource("../lib/webgpu-octree-structured-boundary.ts")
    + runtimeSource("../lib/webgpu-octree-structured-dynamics.ts"),
    /fallback|legacy|compatibility/i);
});

test("structured dynamics owns transport and projection while the face producer owns air completion", () => {
  const dynamics = source("../lib/webgpu-octree-structured-dynamics.ts");
  assert.match(dynamics, /fn regularSample\(/);
  assert.match(dynamics, /fn transitionSample\(/);
  assert.match(dynamics, /fn divergenceRow\(/);
  assert.match(dynamics, /fn projectFamily\(/);
  assert.doesNotMatch(dynamics, /rowCpt|neighborAverage|fn extendFrom[AB]\(/);
  const support = source("../lib/webgpu-octree-air-velocity-support-gpu.ts");
  assert.match(support, /seedAirSupportFaces/);
  assert.match(support, /extendAirSupportFacesAtoB/);
  assert.match(support, /commitAirSupportPublication/);
  assert.match(dynamics, /encodeForcesAndDivergence/);
  assert.match(dynamics, /encodeProjection/);
});

test("production has no runtime switch back to a retired authority", () => {
  const production = [
    source("../lib/webgpu-octree.ts"),
    source("../lib/webgpu-uniform-eulerian.ts"),
    source("../lib/methods/octree.ts"),
    source("../lib/webgpu-octree-pipelined-mgpcg.ts"),
    source("../lib/webgpu-octree-spgrid-vcycle.ts"),
    source("../package.json"),
  ].join("\n");
  assert.doesNotMatch(production,
    /powerDiagramProjection|leafSolver|faceVelocityTransport|FLUID_OCTREE_FACE_TRANSFER/);
  assert.doesNotMatch(production,
    /WebGPUOctreePowerFaces|WebGPUOctreePowerOperator|WebGPUOctreePowerVelocity|WebGPUOctreePowerFace/);
  assert.doesNotMatch(runtimeSource("../lib/webgpu-octree.ts"),
    /options\.pressureIterations|private readonly iterations|galerkin|fallback|legacy/i);
});

test("pressure is the sole subgroup pipelined M1 authority", () => {
  const pressure = runtimeSource("../lib/webgpu-octree-pipelined-mgpcg.ts");
  const preconditioner = runtimeSource("../lib/webgpu-octree-spgrid-vcycle.ts");
  assert.match(pressure, /subgroups/);
  assert.doesNotMatch(pressure, /WebGPUOctreeMGPCG|stagedExecution|restartPCG|fallback|legacy/i);
  assert.doesNotMatch(preconditioner, /weightedJacobi|galerkin|fallback|legacy/i);
});

test("unsupported octree devices fail before the first GPU allocation", () => {
  const host = source("../lib/webgpu-uniform-eulerian.ts");
  const constructor = host.slice(host.indexOf("constructor("), host.indexOf("this.hostAllocation ="));
  assert.match(constructor, /options\.octree[\s\S]*device\.features\.has\("subgroups"\)/);
  assert.match(constructor, /supportsFluidM1MaxReduction\(device\.limits\)/);
  assert.doesNotMatch(constructor, /createBuffer|createTexture|createShaderModule/);
});
