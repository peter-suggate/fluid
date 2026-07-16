import { pathToFileURL } from "node:url";
import { quadtreeConstructionShader, quadtreeSurfaceShader } from "../lib/webgpu-quadtree-builder";
import { quadtreeDispatchShader, quadtreeTallCellProjectionShader } from "../lib/webgpu-quadtree-tall-cell";

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

await validate("surface", quadtreeSurfaceShader, ["advectLevelSet", "seedDistance", "jumpFlood", "finalizeDistance"]);
await validate("construction", quadtreeConstructionShader, ["advectLevelSet", "evaluateSizing", "refine", "smoothTopology", "sampleLeafProfiles"]);
await validate("projection", quadtreeTallCellProjectionShader, [
  "refreshFaces", "refreshRows", "initialize", "precondition", "preconditionBlockIC", "preconditionJacobi", "preconditionLine",
  "preconditionPolynomialStart", "preconditionPolynomialMultiply", "preconditionPolynomialUpdate",
  "startDirection", "reduceInitial", "multiply",
  "applyStep", "applyStepPartial", "applyStepFinalize", "applyStepUpdate",
  "finishIteration", "finishIterationPartial", "finishIterationFinalize", "finishIterationUpdate",
  "project", "coupleReduce", "coupleApply", "coupleImpulse"
]);
await validate("dispatch", quadtreeDispatchShader, ["updateDispatch"]);
device.destroy();
if (failures > 0) throw new Error(`${failures} quadtree shader validation failure(s)`);
console.log("All quadtree shader pipelines are valid");
