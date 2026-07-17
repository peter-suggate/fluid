import { pathToFileURL } from "node:url";
import { quadtreeConstructionShader, quadtreeSurfaceShader } from "../lib/webgpu-quadtree-builder";
import { quadtreeDispatchShader, quadtreeDivergenceShader, quadtreeMultigridShader, quadtreeSurfaceTransportShader, quadtreeTallCellProjectionShader, quadtreeVelocityClampShader, quadtreeVelocityExtrapolationShader } from "../lib/webgpu-quadtree-tall-cell";
import { quadtreeCsrPackShader, quadtreeFacePackShader, quadtreePackAuxShader, quadtreePackCopyShader, quadtreePackFinalizeShader, quadtreePackScanIOShader, quadtreePackScanShader, quadtreePackTextureShader, quadtreeSegmentationPackShader } from "../lib/webgpu-quadtree-pack-builder";
import { gridOverlayShader } from "../lib/webgpu-grid-overlay";
import { secondaryParticleComputeShader } from "../lib/webgpu-secondary-particles";

const modulePath = process.env.WEBGPU_NODE_MODULE;
if (!modulePath) throw new Error("Set WEBGPU_NODE_MODULE to the installed webgpu package index.js");
const { create, globals } = await import(pathToFileURL(modulePath).href) as {
  create(flags: string[]): GPU;
  globals: Record<string, unknown>;
};
Object.assign(globalThis, globals);
const gpu = create(["backend=metal"]), adapter = await gpu.requestAdapter();
if (!adapter) throw new Error("No WebGPU adapter");
const device = await adapter.requestDevice();
let failures = 0;

async function validate(label: string, code: string, entryPoints: string[]) {
  device.pushErrorScope("validation");
  const shaderModule = device.createShaderModule({ label, code }), compilation = await shaderModule.getCompilationInfo();
  for (const message of compilation.messages) if (message.type === "error") {
    failures += 1; console.error(`${label} ${message.lineNum}:${message.linePos} ${message.message}`);
  }
  const moduleError = await device.popErrorScope();
  if (moduleError) { failures += 1; console.error(`${label}: ${moduleError.message}`); return; }
  for (const entryPoint of entryPoints) {
    device.pushErrorScope("validation");
    device.createComputePipeline({ layout: "auto", compute: { module: shaderModule, entryPoint } });
    const error = await device.popErrorScope();
    if (error) { failures += 1; console.error(`${label}.${entryPoint}: ${error.message}`); }
  }
}

async function validateOverlay() {
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ label: "grid-overlay", code: gridOverlayShader });
  const compilation = await module.getCompilationInfo();
  for (const message of compilation.messages) if (message.type === "error") { failures += 1; console.error(`grid-overlay ${message.lineNum}:${message.linePos} ${message.message}`); }
  device.createRenderPipeline({ layout: "auto", vertex: { module, entryPoint: "vertexMain" }, fragment: { module, entryPoint: "fragmentMain", targets: [{ format: "bgra8unorm" }] } });
  const error = await device.popErrorScope(); if (error) { failures += 1; console.error(`grid-overlay: ${error.message}`); }
}

await validate("surface", quadtreeSurfaceShader, ["advectLevelSet", "seedDistance", "jumpFlood", "finalizeDistance", "cullDebris"]);
await validate("construction", quadtreeConstructionShader, ["advectLevelSet", "evaluateSizing", "evaluateOpticalLayer", "dilateOpticalLayerX", "dilateOpticalLayerZ", "smoothOpticalLayer", "refine", "smoothTopology", "sampleLeafProfiles"]);
await validate("projection", quadtreeTallCellProjectionShader, [
  "refreshFaces", "refreshRows", "initialize", "initializeJacobiDirection", "initializePolynomialStart", "precondition", "preconditionBlockIC", "preconditionJacobi", "preconditionLine",
  "preconditionPolynomialStart", "preconditionPolynomialMultiply", "preconditionPolynomialUpdate", "preconditionPolynomialUpdateDirection",
  "startDirection", "reduceInitial", "solveMegakernel", "multiply",
  "applyStep", "applyStepPartial", "applyStepFinalize", "applyStepUpdate", "applyStepUpdateJacobi", "applyStepUpdatePolynomialStart",
  "finishIteration", "finishIterationPartial", "preconditionPolynomialUpdateFinishPartial", "finishIterationFinalize", "finishIterationUpdate",
  "mapPressure", "refreshFaceMls", "project", "coupleReduce", "coupleApply", "coupleImpulse"
]);
await validate("dispatch", quadtreeDispatchShader, ["updateDispatch"]);
await validate("geometric-multigrid", quadtreeMultigridShader, [
  "clearCoarseMatrix", "assembleGalerkin", "copyFineResidual", "lineSmoothInitial", "computeDefect",
  "restrictDefectGather", "solveCoarsest", "prolongateCorrection", "lineSmoothFinal", "copyFineCorrection",
]);
await validate("velocity-extrapolation", quadtreeVelocityExtrapolationShader, ["extrapolateVelocity"]);
await validate("divergence-diagnostic", quadtreeDivergenceShader, ["computeDivergence"]);
await validate("velocity-clamp", quadtreeVelocityClampShader, ["clampVelocity"]);
await validate("surface-transport", quadtreeSurfaceTransportShader, ["buildSurfaceTransport"]);
await validate("secondary-liquid-particles", secondaryParticleComputeShader, ["updateParticles", "spawnParticles"]);
await validate("resident-segmentation-pack", quadtreeSegmentationPackShader, ["classifySegments", "emitSegments"]);
await validateOverlay();
await validate("resident-face-pack", quadtreeFacePackShader, ["countFaces", "emitFaces"]);
await validate("resident-csr-pack", quadtreeCsrPackShader, ["emitCsr"]);
await validate("resident-pack-scan-io", quadtreePackScanIOShader, ["prepareSegmentScan", "finishSegmentScan", "prepareFaceScan", "finishFaceScan", "prepareRowScan", "finishRowScan"]);
await validate("resident-pack-scan", quadtreePackScanShader, ["scanValueBlocks", "scanBlockTotals"]);
await validate("resident-texture-pack", quadtreePackTextureShader, ["unpackCellFields"]);
await validate("resident-finalize-control", quadtreePackFinalizeShader, ["finalizeControl"]);
await validate("resident-pack-copies", quadtreePackCopyShader, ["copyFaces", "copyRowOffsets", "copyRowEntries", "copyMatrix"]);
await validate("resident-pack-aux", quadtreePackAuxShader, ["writeAux"]);
device.destroy();
if (failures > 0) throw new Error(`${failures} quadtree shader validation failure(s)`);
console.log("All quadtree shader pipelines are valid");
