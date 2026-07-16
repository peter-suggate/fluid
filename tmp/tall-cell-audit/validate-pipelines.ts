import { pathToFileURL } from "node:url";
import { tallCellComputeShader } from "../../lib/tall-cell-kernels";
import { legacyUniformComputeShader } from "../../lib/webgpu-eulerian";
import { tallCellExtrapolationShader } from "../../lib/tall-cell-extrapolation";
const { create, globals } = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as any;
Object.assign(globalThis, globals);
const gpu = create(["backend=metal"]);
const adapter = await gpu.requestAdapter();
const device = await adapter.requestDevice();
let failures = 0;
async function check(label: string, code: string, entryPoints: string[]) {
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ code });
  const info = await module.getCompilationInfo();
  for (const m of info.messages) if (m.type === "error") { failures += 1; console.log(`${label} WGSL ${m.lineNum}:${m.linePos} ${m.message.slice(0, 160)}`); }
  const moduleError = await device.popErrorScope();
  if (moduleError) { failures += 1; console.log(`${label} module: ${moduleError.message.slice(0, 200)}`); return; }
  for (const entryPoint of entryPoints) {
    device.pushErrorScope("validation");
    device.createComputePipeline({ layout: "auto", compute: { module, entryPoint } });
    const error = await device.popErrorScope();
    if (error) { failures += 1; console.log(`${label}.${entryPoint}: ${error.message.slice(0, 200)}`); }
  }
  console.log(`${label}: ${entryPoints.length} entry points checked`);
}
await check("tall", tallCellComputeShader, ["extrapolateVelocity","predictVelocity","reverseVelocity","finishAdvection","finishSemiLagrangianAdvection","buildPressureRhs","jacobi","project","coupleRigid","reduceBeforeProjection","reduceDiagnostics","planRemesh","smoothRemesh","remap","sharpenCompute","sharpenScatter","sharpenResolve"]);
await check("uniform", legacyUniformComputeShader, ["advect","semiLagrangianAdvection","reverseAdvection","correctAdvection","jacobi","project","coupleRigid","reduceDiagnostics","buildOccupancy","buildTransport","buildFluxScales","sharpenCompute","sharpenScatter","sharpenResolve"]);
await check("hierarchy", tallCellExtrapolationShader, ["downsampleExtrapolationBase","downsampleVelocity","fillUnknownVelocity"]);
console.log(failures === 0 ? "ALL PIPELINES VALID" : `${failures} FAILURES`);
device.destroy();
